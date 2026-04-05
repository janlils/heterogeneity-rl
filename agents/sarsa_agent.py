"""
agents/sarsa_agent.py
─────────────────────
Implementacja SARSA (on-policy TD(0)) dla benchmarku HTM.

Kluczowe założenia projektowe:
  • Jeden obiekt SARSAAgent = jeden agent rynkowy
  • Każdy agent ma WŁASNĄ tablicę Q (indywidualizm)
  • Każdy agent ma WŁASNE gamma z AgentParams (heterogeniczność)
  • Dyskretyzacja obserwacji → indeks stanu (prosta, deterministyczna)
  • SARSAMultiAgent zarządza całą populacją N agentów

Różnica vs PPO globalny (dla artykułu):
  PPO globalny używa JEDNEJ sieci i JEDNEGO gamma dla wszystkich.
  SARSA używa N tablic Q i N różnych gamma → respektuje heterogeniczność.
"""

import numpy as np
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("htm.sarsa")


# ─────────────────────────────────────────────────────────────
# Konfiguracja hiperparametrów SARSA
# ─────────────────────────────────────────────────────────────

@dataclass
class SARSAConfig:
    """Hiperparametry wspólne dla wszystkich agentów SARSA."""

    alpha: float = 0.1          # learning rate — jak szybko agent aktualizuje Q
    epsilon_start: float = 0.3  # eksploracja na początku treningu
    epsilon_end: float = 0.05   # minimalna eksploracja po decay
    epsilon_decay: float = 0.995 # mnożnik epsilon co epizod

    n_bins: int = 6             # liczba binów na każdą cechę obserwacji
    n_actions: int = 20         # liczba dyskretnych poziomów ceny (z EnvConfig)

    # Tablicowy SARSA operuje na zredukowanej, 6-wymiarowej obserwacji.
    # Gdy środowisko zwraca pełny wektor 12D, wybieramy z niego:
    # [private_value, last_price, spread, frac_traded, wealth_norm, gamma]
    n_obs_features: int = 6


# ─────────────────────────────────────────────────────────────
# Pojedynczy agent SARSA
# ─────────────────────────────────────────────────────────────

class SARSAAgent:
    """
    On-policy TD(0) z tablicą Q. Jeden agent = jedna instancja.

    Tablica Q ma wymiary: [n_states × n_actions]
    gdzie n_states = n_bins ^ n_obs_features

    Dla n_bins=6, n_obs_features=6: 6^6 = 46 656 stanów
    Przy 20 akcjach: tablica 46 656 × 20 = ~3.7M wartości float32 ≈ 15 MB
    Dla 20 agentów: ~300 MB — akceptowalne.
    """

    def __init__(
        self,
        agent_id: str,
        gamma: float,           # indywidualny discount factor z AgentParams
        cfg: SARSAConfig = None,
        seed: int = 42,
    ):
        self.agent_id = agent_id
        self.gamma = gamma      # ← kluczowe: każdy agent ma swoje gamma
        self.cfg = cfg or SARSAConfig()
        self.rng = np.random.default_rng(seed)

        # Rozmiar przestrzeni stanów
        self.n_states = self.cfg.n_bins ** self.cfg.n_obs_features

        # Tablica Q: inicjalizowana małymi losowymi wartościami
        # (nie zerami — żeby uniknąć ties na starcie)
        self.Q = self.rng.uniform(0, 0.01, size=(self.n_states, self.cfg.n_actions))

        # Stan i akcja z poprzedniego kroku (wymagane przez SARSA on-policy)
        self._prev_state: Optional[int] = None
        self._prev_action: Optional[int] = None
        self._prev_obs: Optional[np.ndarray] = None

        # Epsilon do epsilon-greedy (maleje w czasie)
        self.epsilon = self.cfg.epsilon_start

        # Statystyki do logowania
        self.total_updates = 0
        self.total_reward = 0.0
        self.episode_td_errors: List[float] = []

        logger.debug(
            f"[{agent_id}] SARSA init | gamma={gamma:.3f} | "
            f"n_states={self.n_states} | epsilon={self.epsilon:.3f}"
        )

    # ── Dyskretyzacja obserwacji ──────────────────────────────

    def _discretize(self, obs: np.ndarray) -> int:
        """
        Zamienia ciągły wektor obserwacji na indeks stanu (int).

        Strategia: każda cecha → n_bins równomiernych binów w [0, 1]
        Kombinacja binów → single int przez mixed-radix encoding.

        Przykład (n_bins=6, n_features=6):
          obs = [0.75, 0.50, 0.10, 0.60, 0.80, 0.90]
          bins = [4,    3,    0,    3,    4,    5   ]
          state = 4 + 6*(3 + 6*(0 + 6*(3 + 6*(4 + 6*5)))) = jakiś int
        """
        obs = np.asarray(obs, dtype=np.float32)

        # DoubleAuction zwraca pełną obserwację 12D. Dla tablicowego SARSA
        # redukujemy ją do 6 cech o stabilnym zakresie [0, 1].
        if obs.shape[0] == 12:
            obs = obs[[0, 1, 4, 5, 7, 6]]
        elif obs.shape[0] != self.cfg.n_obs_features:
            raise ValueError(
                f"[{self.agent_id}] Nieobsługiwany wymiar obserwacji: "
                f"{obs.shape[0]} (oczekiwano 12 lub {self.cfg.n_obs_features})"
            )

        obs_clipped = np.clip(obs, 0.0, 1.0 - 1e-9)
        bins = (obs_clipped * self.cfg.n_bins).astype(int)  # [0, n_bins-1]

        # Mixed-radix encoding
        state = 0
        for i, b in enumerate(bins):
            state += b * (self.cfg.n_bins ** i)

        return int(state % self.n_states)  # safety modulo

    # ── Wybór akcji ───────────────────────────────────────────

    def act(self, obs: np.ndarray, explore: bool = True) -> Tuple[int, int]:
        """
        Epsilon-greedy wybór akcji.

        Returns:
            (action, state) — akcja to indeks ceny [0, n_actions-1]
            Rzeczywista cena = action / (n_actions - 1) ∈ [0, 1]
        """
        state = self._discretize(obs)

        if explore and self.rng.random() < self.epsilon:
            action = int(self.rng.integers(self.cfg.n_actions))
            logger.debug(f"[{self.agent_id}] EXPLORE action={action} state={state}")
        else:
            action = int(np.argmax(self.Q[state]))
            logger.debug(f"[{self.agent_id}] EXPLOIT action={action} Q={self.Q[state, action]:.4f}")

        return action, state

    # ── Aktualizacja SARSA ────────────────────────────────────

    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> float:
        """
        Aktualizacja SARSA (on-policy TD(0)):

            Q(s, a) ← Q(s, a) + α * [r + γ * Q(s', a') - Q(s, a)]
                                       └── td_target ──┘   └─ td_error ─┘

        Różnica SARSA vs Q-learning:
          SARSA:      a' = π(s') — akcja wg AKTUALNEJ polityki (on-policy)
          Q-learning: a' = argmax Q(s') — akcja optymalna (off-policy)

        On-policy jest bezpieczniejsze w środowiskach multi-agent
        bo uwzględnia że inni agenci też się uczą.
        """
        state = self._discretize(obs)
        next_state = self._discretize(next_obs)

        if done:
            td_target = reward
        else:
            # SARSA: wybierz a' wg aktualnej polityki (nie argmax!)
            next_action, _ = self.act(next_obs, explore=True)
            td_target = reward + self.gamma * self.Q[next_state, next_action]

        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.cfg.alpha * td_error

        # Statystyki
        self.total_updates += 1
        self.total_reward += reward
        self.episode_td_errors.append(abs(td_error))

        logger.debug(
            f"[{self.agent_id}] UPDATE s={state} a={action} "
            f"r={reward:.4f} td_err={td_error:.4f}"
        )

        return abs(td_error)

    def decay_epsilon(self):
        """Zmniejsza epsilon po każdym epizodzie (wywołaj raz na koniec epizodu)."""
        self.epsilon = max(
            self.cfg.epsilon_end,
            self.epsilon * self.cfg.epsilon_decay
        )

    def episode_stats(self) -> dict:
        """Zwraca statystyki bieżącego epizodu i resetuje bufor."""
        stats = {
            "mean_td_error": float(np.mean(self.episode_td_errors)) if self.episode_td_errors else 0.0,
            "max_td_error": float(np.max(self.episode_td_errors)) if self.episode_td_errors else 0.0,
            "epsilon": self.epsilon,
            "gamma": self.gamma,
        }
        self.episode_td_errors = []
        return stats

    def reset_episode(self):
        """Reset stanu między epizodami (Q-tablica zostaje — to pamięć agenta)."""
        self._prev_state = None
        self._prev_action = None
        self._prev_obs = None
        self.episode_td_errors = []


# ─────────────────────────────────────────────────────────────
# Zarządca populacji N agentów SARSA
# ─────────────────────────────────────────────────────────────

class SARSAMultiAgent:
    """
    Zarządca N niezależnych agentów SARSA.

    Każdy agent:
      • ma własną tablicę Q (nie dzielą wiedzy)
      • ma własne gamma z populacji (heterogeniczność)
      • uczy się tylko ze swoich własnych doświadczeń

    To jest reprezentant "indywidualistycznego" podejścia w benchmarku.
    """

    def __init__(
        self,
        agent_ids: List[str],
        agent_gammas: np.ndarray,  # gamma per agent z AgentPopulation
        cfg: SARSAConfig = None,
        seed: int = 42,
    ):
        self.cfg = cfg or SARSAConfig()
        self.agents: Dict[str, SARSAAgent] = {}

        for i, aid in enumerate(agent_ids):
            gamma = float(agent_gammas[i])
            self.agents[aid] = SARSAAgent(
                agent_id=aid,
                gamma=gamma,      # ← każdy agent ma swoje gamma
                cfg=self.cfg,
                seed=seed + i,    # różne seedy = różna inicjalizacja Q
            )

        logger.info(
            f"SARSAMultiAgent init | {len(agent_ids)} agentów | "
            f"gamma range: [{agent_gammas.min():.3f}, {agent_gammas.max():.3f}]"
        )

    def act(self, observations: Dict[str, np.ndarray], explore: bool = True) -> Dict[str, int]:
        """Zwraca akcje dla wszystkich agentów którzy mają obserwacje."""
        return {
            aid: self.agents[aid].act(obs, explore=explore)[0]
            for aid, obs in observations.items()
            if aid in self.agents
        }

    def update_all(
        self,
        obs: Dict[str, np.ndarray],
        actions: Dict[str, int],
        rewards: Dict[str, float],
        next_obs: Dict[str, np.ndarray],
        dones: Dict[str, bool],
    ) -> Dict[str, float]:
        """Aktualizuje Q-tablice wszystkich agentów. Zwraca TD errors do logowania."""
        td_errors = {}
        for aid in obs:
            if aid in self.agents and aid in actions:
                td_errors[aid] = self.agents[aid].update(
                    obs[aid], actions[aid], rewards[aid],
                    next_obs[aid], dones[aid]
                )
        return td_errors

    def end_episode(self):
        """Wywołaj po każdym epizodzie — decay epsilon, reset bufory."""
        for agent in self.agents.values():
            agent.decay_epsilon()
            agent.reset_episode()

    def population_stats(self) -> dict:
        """Zagregowane statystyki populacji dla W&B / logów."""
        epsilons = [a.epsilon for a in self.agents.values()]
        gammas = [a.gamma for a in self.agents.values()]
        td_errors = []
        for a in self.agents.values():
            if a.episode_td_errors:
                td_errors.extend(a.episode_td_errors)

        return {
            "mean_epsilon": float(np.mean(epsilons)),
            "mean_gamma": float(np.mean(gammas)),
            "mean_td_error": float(np.mean(td_errors)) if td_errors else 0.0,
        }
