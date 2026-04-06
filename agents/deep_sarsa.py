"""
agents/deep_sarsa.py — Deep SARSA z PyTorch
============================================
Przepisanie z czystego numpy na PyTorch.

Zyski:
  - Autograd zamiast ręcznego backpropu
  - Adam optimizer (adaptacyjny lr, lepsza zbieżność niż SGD)
  - MPS acceleration na MacBooku M-series
  - Bezpieczne torch.no_grad() dla td_target
  - save/load wag (torch.save/load)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

logger = logging.getLogger("htm.deep_sarsa")


def _get_device() -> torch.device:
    """
    Dla małych sieci (< ~50k parametrów) CPU jest szybszy niż MPS/CUDA.

    Transfer tensor CPU<->GPU kosztuje ~200μs per wywołanie.
    Przy batch_size=1 i sieci 1701 parametrów obliczenia trwają <1μs —
    overhead transferu dominuje 200:1 nad faktyczną pracą.

    MPS/CUDA opłaca się gdy batch_size >= 512 LUB sieć >= 100k parametrów.
    Dla SARSA z batch=1 per krok -> zawsze CPU.
    """
    return torch.device("cpu")

DEVICE = _get_device()

# Krytyczne przy multiprocessing: ogranicz wątki PyTorch per proces.
# Domyślnie PyTorch używa wszystkich rdzeni → 8 workerów × 8 wątków = 64 wątki
# na 10 rdzeniach → context switching dominuje.
# Z num_threads=1: 8 workerów × 1 wątek = 8 wątków na 10 rdzeniach → OK.
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


# ===========================================================================
# Konfiguracja
# ===========================================================================

@dataclass
class DeepSARSAConfig:
    hidden_size:   int   = 32
    lr:            float = 3e-3    # Adam lr
    epsilon_start: float = 0.35
    epsilon_end:   float = 0.05
    epsilon_decay: float = 0.993
    grad_clip:     float = 1.0


# ===========================================================================
# Sieć Q
# ===========================================================================

class QNetwork(nn.Module):
    def __init__(self, n_input: int, n_hidden: int, n_output: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_input,  n_hidden), nn.ReLU(),
            nn.Linear(n_hidden, n_hidden), nn.ReLU(),
            nn.Linear(n_hidden, n_output),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ===========================================================================
# Jeden agent
# ===========================================================================

class DeepSARSAAgent:
    """
    On-policy TD(0) — jeden agent, jedna sieć.

    SARSA update:
        Q(s,a) ← Q(s,a) + α [r + γ Q(s',a') - Q(s,a)]
        a' = pi(s') — on-policy (nie argmax)

    Maskowanie akcji:
        Brak sygnału (|value_signal - 0.5| <= 1/6) → zawsze PASS
        Jest sygnał  → MARKET/LIMIT, PASS zablokowany przez -inf
    """

    def __init__(
        self,
        agent_id:  str,
        gamma:     float,
        n_obs:     int,
        n_actions: int,
        cfg:       DeepSARSAConfig = None,
        seed:      int = 42,
    ):
        self.agent_id  = agent_id
        self.gamma     = gamma
        self.n_actions = n_actions
        self.cfg       = cfg or DeepSARSAConfig()

        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)

        self.net = QNetwork(n_obs, self.cfg.hidden_size, n_actions).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr)

        self.epsilon = self.cfg.epsilon_start
        self.episode_td_errors: List[float] = []
        self.episode_losses:    List[float] = []
        self.total_updates:     int         = 0

    def _t(self, obs: np.ndarray) -> torch.Tensor:
        # torch.from_numpy: zero-copy gdy array jest float32 i C-contiguous
        # Znacznie szybsze niż torch.FloatTensor() który zawsze alokuje i kopiuje
        arr = obs if obs.dtype == np.float32 else obs.astype(np.float32)
        return torch.from_numpy(arr)

    def act(self, obs: np.ndarray, explore: bool = True) -> int:
        value_signal = float(obs[2])
        has_signal   = abs(value_signal - 0.5) > (1.0 / 6.0)

        if not has_signal:
            return 0  # PASS — bez forward pass, szybciej

        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(1, self.n_actions))

        # net.eval()/net.train() usunięte — brak Dropout/BatchNorm w sieci,
        # więc mode switch jest czystym kosztem bez żadnego efektu.
        # torch.no_grad() wystarczy żeby zablokować gradient.
        with torch.no_grad():
            q = self.net(self._t(obs))
            q[0] = float("-inf")  # mask PASS
            return int(q.argmax().item())

    def update(
        self,
        obs:      np.ndarray,
        action:   int,
        reward:   float,
        next_obs: np.ndarray,
        done:     bool,
    ) -> Tuple[float, float]:
        """
        SARSA update. Jeden forward pass na obs (z gradientem),
        jeden forward pass na next_obs (bez gradientu).
        """
        # Q(s, a) — jeden forward pass z gradientem
        q_current = self.net(self._t(obs))[action]

        # TD target — bez gradientu, jeden forward pass na next_obs
        with torch.no_grad():
            if done:
                td_target_val = float(reward)
            else:
                # Wyznacz next_action inline (bez osobnego wywołania act())
                vs_next = float(next_obs[2])
                if abs(vs_next - 0.5) <= (1.0 / 6.0):
                    # Brak sygnału → PASS (nie trzeba forward pass)
                    next_q_val = 0.0
                    next_action = 0
                else:
                    # Jeden forward pass — reuse dla argmax I dla Q[next_action]
                    q_next = self.net(self._t(next_obs))  # shape: (n_actions,)
                    if self.rng.random() < self.epsilon:
                        next_action = int(self.rng.integers(1, self.n_actions))
                    else:
                        q_next[0] = float("-inf")  # mask PASS
                        next_action = int(q_next.argmax().item())
                    next_q_val = q_next[next_action].item()
                td_target_val = reward + self.gamma * next_q_val
            td_target = torch.tensor(td_target_val, dtype=torch.float32)

        loss = F.mse_loss(q_current, td_target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.grad_clip)
        self.optimizer.step()

        td_error = abs((td_target - q_current).item())
        self.total_updates += 1
        self.episode_td_errors.append(td_error)
        self.episode_losses.append(loss.item())
        return td_error, loss.item()

    def decay_epsilon(self):
        self.epsilon = max(self.cfg.epsilon_end,
                          self.epsilon * self.cfg.epsilon_decay)

    def reset_episode(self):
        self.episode_td_errors = []
        self.episode_losses    = []

    def episode_stats(self) -> dict:
        return {
            "mean_td_error": float(np.mean(self.episode_td_errors)) if self.episode_td_errors else 0.0,
            "mean_loss":     float(np.mean(self.episode_losses))    if self.episode_losses    else 0.0,
            "epsilon":       self.epsilon,
            "gamma":         self.gamma,
        }

    def q_values(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self.net(self._t(obs)).numpy()

    def state_dict(self):
        return self.net.state_dict()

    def load_state_dict(self, sd):
        self.net.load_state_dict(sd)


# ===========================================================================
# Zarządca N agentów
# ===========================================================================

class DeepSARSAMultiAgent:
    """N niezależnych agentów — identyczny interfejs jak wersja numpy."""

    def __init__(
        self,
        agent_ids:    List[str],
        agent_gammas: np.ndarray,
        n_obs:        int,
        n_actions:    int,
        cfg:          DeepSARSAConfig = None,
        seed:         int = 42,
    ):
        self.cfg    = cfg or DeepSARSAConfig()
        self.agents: Dict[str, DeepSARSAAgent] = {}

        for i, aid in enumerate(agent_ids):
            self.agents[aid] = DeepSARSAAgent(
                agent_id=aid, gamma=float(agent_gammas[i]),
                n_obs=n_obs, n_actions=n_actions,
                cfg=self.cfg, seed=seed + i,
            )

        n_params = sum(p.numel() for p in list(self.agents.values())[0].net.parameters())
        logger.info(
            f"DeepSARSAMultiAgent | N={len(agent_ids)} | device={DEVICE} | "
            f"params/agent={n_params} | "
            f"net: {n_obs}->  {self.cfg.hidden_size}->{n_actions}"
        )

    def act(self, observations: Dict[str, np.ndarray], explore: bool = True) -> Dict[str, int]:
        return {aid: self.agents[aid].act(obs, explore=explore)
                for aid, obs in observations.items() if aid in self.agents}

    def update_all(
        self,
        obs: Dict[str, np.ndarray], actions: Dict[str, int],
        rewards: Dict[str, float], next_obs: Dict[str, np.ndarray],
        dones: Dict[str, bool],
    ) -> Dict[str, float]:
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
        for agent in self.agents.values():
            agent.decay_epsilon()
            agent.reset_episode()

    def population_stats(self) -> dict:
        epsilons = [a.epsilon for a in self.agents.values()]
        gammas   = [a.gamma   for a in self.agents.values()]
        tds      = [np.mean(a.episode_td_errors) if a.episode_td_errors else 0.0
                    for a in self.agents.values()]
        losses   = [np.mean(a.episode_losses)    if a.episode_losses    else 0.0
                    for a in self.agents.values()]
        return {
            "mean_epsilon":  float(np.mean(epsilons)),
            "mean_gamma":    float(np.mean(gammas)),
            "mean_td_error": float(np.mean(tds)),
            "mean_loss":     float(np.mean(losses)),
        }

    def update_gammas(self, agent_ids: List[str], new_gammas: np.ndarray):
        for i, aid in enumerate(agent_ids):
            if aid in self.agents:
                self.agents[aid].gamma = float(new_gammas[i])

    def save(self, path: str):
        torch.save({aid: a.state_dict() for aid, a in self.agents.items()}, path)
        logger.info(f"Saved -> {path}")

    def load(self, path: str):
        state = torch.load(path, map_location=DEVICE)
        for aid, sd in state.items():
            if aid in self.agents:
                self.agents[aid].load_state_dict(sd)
        logger.info(f"Loaded <- {path}")
