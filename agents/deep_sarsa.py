"""
agents/deep_sarsa.py — Deep SARSA z siecią neuronową (numpy)
=============================================================
Zastępuje tablicę Q siecią neuronową 2-warstwową.
Zaimplementowane w czystym numpy — brak zależności GPU/PyTorch.

Dlaczego sieć zamiast tablicy Q:
  Tablica Q przy N=20 agentach × 40 kroków × 6 cech × 6 binów = 46 656 stanów.
  Każdy agent dostaje ~2 tury / epizod. Po 1000 epizodach = 2000 aktualizacji.
  Większość stanów nigdy nie odwiedzona → tablica Q nie uogólnia.
  Sieć NN interpoluje między nieodwiedzonymi stanami → uczy się szybciej.

Architektura per agent:
  Input(12) → Dense(64, ReLU) → Dense(64, ReLU) → Output(n_actions)
  ~4 500 parametrów per agent. Dla N=20: ~90 000 parametrów łącznie.

Kluczowe właściwości zachowane z SARSA:
  • Per-agent: każdy agent ma WŁASNĄ sieć (indywidualizm)
  • On-policy: aktualizacja używa a' z bieżącej polityki (nie argmax)
  • Indywidualne gamma (heterogeniczność)
  • Epsilon-greedy eksplor→eksploit decay

Użycie:
    sarsa = DeepSARSAMultiAgent(agent_ids, agent_gammas, cfg)
    actions = sarsa.act(obs_dict)               # {aid: action_idx}
    sarsa.update_all(obs, actions, rew, next_obs, dones)
    sarsa.end_episode()
"""

import numpy as np
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("htm.deep_sarsa")


# ===========================================================================
# Konfiguracja
# ===========================================================================

@dataclass
class DeepSARSAConfig:
    """Hiperparametry sieci neuronowej."""
    hidden_size:   int   = 64
    lr:            float = 3e-3    # SGD learning rate (wyższy niż Adam)
    epsilon_start: float = 0.35
    epsilon_end:   float = 0.05
    epsilon_decay: float = 0.993
    grad_clip:     float = 1.0     # gradient clipping (stabilność)


# ===========================================================================
# Sieć neuronowa w numpy
# ===========================================================================

class NumpyMLP:
    """
    2-warstwowa sieć MLP zaimplementowana w numpy.

    Architektura: Input → Dense(hidden, ReLU) → Dense(hidden, ReLU) → Output
    Optymalizator: SGD z gradient clipping.
    Inicjalizacja: He (dla ReLU).
    """

    def __init__(
        self,
        n_input:  int,
        n_hidden: int,
        n_output: int,
        lr:       float = 3e-3,
        grad_clip:float = 1.0,
        seed:     int   = 42,
    ):
        rng = np.random.default_rng(seed)
        self.lr        = lr
        self.grad_clip = grad_clip

        # Inicjalizacja He (dobra dla ReLU)
        scale1 = np.sqrt(2.0 / n_input)
        scale2 = np.sqrt(2.0 / n_hidden)

        # Wagi i biasy warstwy 1
        self.W1 = rng.normal(0, scale1, (n_input,  n_hidden)).astype(np.float32)
        self.b1 = np.zeros(n_hidden, dtype=np.float32)

        # Wagi i biasy warstwy 2
        self.W2 = rng.normal(0, scale2, (n_hidden, n_hidden)).astype(np.float32)
        self.b2 = np.zeros(n_hidden, dtype=np.float32)

        # Wagi i biasy warstwy wyjściowej
        self.W3 = rng.normal(0, 0.01, (n_hidden, n_output)).astype(np.float32)
        self.b3 = np.zeros(n_output, dtype=np.float32)

        # Cache do backpropagation
        self._cache: dict = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass. x ma kształt (n_input,) lub (batch, n_input).
        Zwraca Q-wartości kształtu (n_output,) lub (batch, n_output).
        """
        if x.ndim == 1:
            x = x[np.newaxis, :]   # dodaj wymiar batch

        # Warstwa 1: ReLU
        z1 = x @ self.W1 + self.b1
        a1 = np.maximum(0, z1)     # ReLU

        # Warstwa 2: ReLU
        z2 = a1 @ self.W2 + self.b2
        a2 = np.maximum(0, z2)     # ReLU

        # Warstwa wyjściowa: liniowa (Q-wartości mogą być ujemne)
        out = a2 @ self.W3 + self.b3

        self._cache = {"x": x, "z1": z1, "a1": a1, "z2": z2, "a2": a2}
        return out.squeeze()       # usuń batch dim jeśli był 1

    def backward(self, action: int, td_error: float) -> float:
        """
        Backward pass dla jednej akcji.
        Gradient: ∂L/∂Q(s,a) = -td_error (dla MSE loss)
        Zwraca normę gradientu (do monitorowania).
        """
        x, z1, a1, z2, a2 = (
            self._cache["x"],  self._cache["z1"],
            self._cache["a1"], self._cache["z2"], self._cache["a2"]
        )

        # Gradient wyjścia — tylko dla wybranej akcji
        dout = np.zeros((1, self.W3.shape[1]), dtype=np.float32)
        dout[0, action] = -td_error   # -td_error bo chcemy minimalizować błąd

        # Warstwa 3 (liniowa)
        dW3 = a2.T @ dout
        db3 = dout.sum(axis=0)
        da2 = dout @ self.W3.T

        # Warstwa 2 (ReLU)
        dz2 = da2 * (z2 > 0)   # ReLU gradient
        dW2 = a1.T @ dz2
        db2 = dz2.sum(axis=0)
        da1 = dz2 @ self.W2.T

        # Warstwa 1 (ReLU)
        dz1 = da1 * (z1 > 0)
        dW1 = x.T @ dz1
        db1 = dz1.sum(axis=0)

        # Gradient clipping (zapobiega eksplozji gradientów)
        grads = [dW1, db1, dW2, db2, dW3, db3]
        total_norm = np.sqrt(sum(np.sum(g**2) for g in grads))
        if total_norm > self.grad_clip:
            clip_coef = self.grad_clip / (total_norm + 1e-8)
            grads = [g * clip_coef for g in grads]

        dW1, db1, dW2, db2, dW3, db3 = grads

        # SGD update
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3

        return float(total_norm)


# ===========================================================================
# DeepSARSAAgent — jeden agent
# ===========================================================================

class DeepSARSAAgent:
    """
    On-policy TD(0) z siecią neuronową. Jeden agent = jedna sieć.

    Aktualizacja SARSA (on-policy):
        Q(s,a) ← Q(s,a) + α · [r + γ·Q(s',a') - Q(s,a)]
                                 └─── td_target ────┘  └─ td_error ─┘
        gdzie a' = π(s') — akcja z AKTUALNEJ polityki (epsilon-greedy)

    Różnica vs Q-learning (off-policy):
        Q-learning: a' = argmax Q(s') — zawsze najlepsza akcja
        SARSA:      a' = π(s')         — może być eksploracyjna

    On-policy jest bezpieczniejszy w MARL bo uwzględnia że inni agenci
    też explorują i środowisko jest przez to niestacjonarne.
    """

    def __init__(
        self,
        agent_id: str,
        gamma:    float,
        n_obs:    int,
        n_actions:int,
        cfg:      DeepSARSAConfig = None,
        seed:     int = 42,
    ):
        self.agent_id  = agent_id
        self.gamma     = gamma     # indywidualny! heterogeniczność
        self.n_actions = n_actions
        self.cfg       = cfg or DeepSARSAConfig()
        self.rng       = np.random.default_rng(seed)

        self.net = NumpyMLP(
            n_input  = n_obs,
            n_hidden = self.cfg.hidden_size,
            n_output = n_actions,
            lr       = self.cfg.lr,
            grad_clip= self.cfg.grad_clip,
            seed     = seed,
        )

        self.epsilon = self.cfg.epsilon_start

        # Statystyki epizodu
        self.episode_td_errors:   List[float] = []
        self.episode_grad_norms:  List[float] = []
        self.episode_rewards:     List[float] = []
        self.total_updates:       int         = 0

        logger.debug(
            f"[{agent_id}] DeepSARSA init | γ={gamma:.3f} | "
            f"net={n_obs}→{self.cfg.hidden_size}→{self.cfg.hidden_size}→{n_actions}"
        )

    def act(self, obs: np.ndarray, explore: bool = True) -> int:
        """Epsilon-greedy wybór akcji."""
        if explore and self.rng.random() < self.epsilon:
            action = int(self.rng.integers(self.n_actions))
            logger.debug(f"[{self.agent_id}] EXPLORE → {action}")
        else:
            q_vals = self.net.forward(obs.astype(np.float32))
            action = int(np.argmax(q_vals))
            logger.debug(f"[{self.agent_id}] EXPLOIT → {action} (Q={q_vals[action]:.4f})")
        return action

    def update(
        self,
        obs:      np.ndarray,
        action:   int,
        reward:   float,
        next_obs: np.ndarray,
        done:     bool,
    ) -> Tuple[float, float]:
        """
        SARSA update:
          1. Oblicz Q(s, a) — forward przez sieć
          2. Oblicz Q(s', a') — forward przez sieć, a' z aktualnej polityki
          3. td_error = r + γ·Q(s', a') - Q(s, a)
          4. Backward pass — update wag

        Returns: (|td_error|, gradient_norm)
        """
        obs_f      = obs.astype(np.float32)
        next_obs_f = next_obs.astype(np.float32)

        # Q(s, a)
        q_vals    = self.net.forward(obs_f)
        q_current = float(q_vals[action])

        # TD target
        if done:
            td_target = reward
        else:
            # SARSA: a' z aktualnej polityki (on-policy)
            next_action = self.act(next_obs, explore=True)
            q_next_vals = self.net.forward(next_obs_f)
            q_next      = float(q_next_vals[next_action])
            td_target   = reward + self.gamma * q_next

        td_error = td_target - q_current

        # Backward pass przez NumpyMLP
        grad_norm = self.net.backward(action, td_error)

        # Statystyki
        self.total_updates += 1
        self.episode_td_errors.append(abs(td_error))
        self.episode_grad_norms.append(grad_norm)
        self.episode_rewards.append(reward)

        return abs(td_error), grad_norm

    def decay_epsilon(self):
        self.epsilon = max(
            self.cfg.epsilon_end,
            self.epsilon * self.cfg.epsilon_decay
        )

    def reset_episode(self):
        """Reset statystyk epizodu (sieć zostaje — to pamięć agenta)."""
        self.episode_td_errors  = []
        self.episode_grad_norms = []
        self.episode_rewards    = []

    def episode_stats(self) -> dict:
        return {
            "mean_td_error":  float(np.mean(self.episode_td_errors))  if self.episode_td_errors  else 0.0,
            "mean_grad_norm": float(np.mean(self.episode_grad_norms)) if self.episode_grad_norms else 0.0,
            "mean_reward":    float(np.mean(self.episode_rewards))    if self.episode_rewards    else 0.0,
            "epsilon":        self.epsilon,
            "gamma":          self.gamma,
        }

    def q_values_for_obs(self, obs: np.ndarray) -> np.ndarray:
        """Q-wartości dla danej obserwacji — do wizualizacji."""
        return self.net.forward(obs.astype(np.float32))


# ===========================================================================
# DeepSARSAMultiAgent — zarządca N agentów
# ===========================================================================

class DeepSARSAMultiAgent:
    """
    N niezależnych agentów DeepSARSA.

    Każdy agent:
      • Ma WŁASNĄ sieć neuronową (nie dzielą parametrów)
      • Uczy się tylko ze swoich doświadczeń (nie współdzieli danych)
      • Ma WŁASNE gamma (heterogeniczność)

    To jest reprezentant "indywidualistycznego" podejścia.
    Porównanie z PPO globalnym (jeden model dla wszystkich) jest
    główną osią badawczą artykułu.
    """

    def __init__(
        self,
        agent_ids:    List[str],
        agent_gammas: np.ndarray,
        n_obs:        int,
        n_actions:    int,
        cfg:          DeepSARSAConfig = None,
        seed:         int = 42,
    ):
        self.cfg      = cfg or DeepSARSAConfig()
        self.n_obs    = n_obs
        self.n_actions= n_actions
        self.agents: Dict[str, DeepSARSAAgent] = {}

        for i, aid in enumerate(agent_ids):
            gamma = float(agent_gammas[i])
            self.agents[aid] = DeepSARSAAgent(
                agent_id  = aid,
                gamma     = gamma,
                n_obs     = n_obs,
                n_actions = n_actions,
                cfg       = self.cfg,
                seed      = seed + i,
            )

        logger.info(
            f"DeepSARSAMultiAgent | N={len(agent_ids)} agentów | "
            f"γ=[{agent_gammas.min():.2f}, {agent_gammas.max():.2f}] | "
            f"net: {n_obs}→{self.cfg.hidden_size}→{self.cfg.hidden_size}→{n_actions}"
        )

    def act(
        self,
        observations: Dict[str, np.ndarray],
        explore: bool = True,
    ) -> Dict[str, int]:
        return {
            aid: self.agents[aid].act(obs, explore=explore)
            for aid, obs in observations.items()
            if aid in self.agents
        }

    def update_all(
        self,
        obs:      Dict[str, np.ndarray],
        actions:  Dict[str, int],
        rewards:  Dict[str, float],
        next_obs: Dict[str, np.ndarray],
        dones:    Dict[str, bool],
    ) -> Dict[str, float]:
        """Aktualizuje wszystkich agentów. Zwraca TD errors."""
        td_errors = {}
        for aid in obs:
            if aid in self.agents and aid in actions:
                td_e, _ = self.agents[aid].update(
                    obs[aid], actions[aid], rewards[aid],
                    next_obs[aid], dones[aid]
                )
                td_errors[aid] = td_e
        return td_errors

    def end_episode(self):
        """Koniec epizodu: decay epsilon, reset statystyki."""
        for agent in self.agents.values():
            agent.decay_epsilon()
            agent.reset_episode()

    def population_stats(self) -> dict:
        """Zagregowane statystyki populacji."""
        epsilons   = [a.epsilon for a in self.agents.values()]
        gammas     = [a.gamma   for a in self.agents.values()]
        td_errors  = [np.mean(a.episode_td_errors)  if a.episode_td_errors  else 0.0
                      for a in self.agents.values()]
        grad_norms = [np.mean(a.episode_grad_norms) if a.episode_grad_norms else 0.0
                      for a in self.agents.values()]
        return {
            "mean_epsilon":   float(np.mean(epsilons)),
            "mean_gamma":     float(np.mean(gammas)),
            "mean_td_error":  float(np.mean(td_errors)),
            "mean_grad_norm": float(np.mean(grad_norms)),
        }

    def update_gammas(self, agent_ids: List[str], new_gammas: np.ndarray):
        """Aktualizuje gamma agentów po resecie populacji."""
        for i, aid in enumerate(agent_ids):
            if aid in self.agents:
                self.agents[aid].gamma = float(new_gammas[i])