"""
codes/deep_sarsa.py — Deep SARSA w czystym numpy
==================================================
Powrót do numpy po testach z PyTorch.

Dlaczego numpy, nie PyTorch:
  Przy batch_size=1 PyTorch jest 6-10x wolniejszy niż numpy.
  Overhead dispatcha i alokacji tensorów (~150μs/call) dominuje
  nad faktycznymi obliczeniami (<1μs dla sieci 1700 param).
  Numpy forward+backward = ~27μs. PyTorch = ~150-200μs.
  Pełny trening (300ep × 200 kroków × 20 agentów) = ~5 min numpy vs ~30 min torch.

Kiedy warto wrócić do PyTorch:
  Batch updates (batch >= 32) lub sieć >= 100k parametrów.
  Dla PPO z trajectory buffer (T=200) batch jest duży → wtedy torch.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from codes.config import DeepSARSAConfig

logger = logging.getLogger("htm.deep_sarsa")


class NumpyMLP:
    """
    2-ukryta MLP w czystym numpy.
    Input → Dense(hidden, ReLU) → Dense(hidden, ReLU) → Output
    """

    def __init__(self, n_in: int, n_hidden: int, n_out: int, rng: np.random.Generator):
        # He initialization
        scale1 = np.sqrt(2.0 / n_in)
        scale2 = np.sqrt(2.0 / n_hidden)
        self.W1 = rng.standard_normal((n_hidden, n_in)).astype(np.float32)  * scale1
        self.b1 = np.zeros(n_hidden, np.float32)
        self.W2 = rng.standard_normal((n_hidden, n_hidden)).astype(np.float32) * scale2
        self.b2 = np.zeros(n_hidden, np.float32)
        self.W3 = rng.standard_normal((n_out, n_hidden)).astype(np.float32) * scale2
        self.b3 = np.zeros(n_out, np.float32)

        # Adam moments
        self.mW1 = np.zeros_like(self.W1); self.vW1 = np.zeros_like(self.W1)
        self.mb1 = np.zeros_like(self.b1); self.vb1 = np.zeros_like(self.b1)
        self.mW2 = np.zeros_like(self.W2); self.vW2 = np.zeros_like(self.W2)
        self.mb2 = np.zeros_like(self.b2); self.vb2 = np.zeros_like(self.b2)
        self.mW3 = np.zeros_like(self.W3); self.vW3 = np.zeros_like(self.W3)
        self.mb3 = np.zeros_like(self.b3); self.vb3 = np.zeros_like(self.b3)
        self.t = 0  # Adam step counter
        self.beta1 = 0.9; self.beta2 = 0.999; self.eps_adam = 1e-8

    def forward(self, x: np.ndarray):
        """Returns (q_values, cache) — cache potrzebny do backward."""
        z1 = self.W1 @ x + self.b1; h1 = np.maximum(0.0, z1)
        z2 = self.W2 @ h1 + self.b2; h2 = np.maximum(0.0, z2)
        q  = self.W3 @ h2 + self.b3
        return q, (x, z1, h1, z2, h2)

    def backward(self, action: int, td_error: float, cache, lr: float, grad_clip: float) -> float:
        """Adam update dla wybranej akcji. Zwraca normę gradientu."""
        x, z1, h1, z2, h2 = cache
        self.t += 1

        # Gradient wyjścia — tylko dla wybranej akcji
        dq = np.zeros(len(self.b3), np.float32)
        dq[action] = np.float32(td_error)

        # Backprop
        dh2 = self.W3.T @ dq
        dz2 = dh2 * (z2 > 0)
        dh1 = self.W2.T @ dz2
        dz1 = dh1 * (z1 > 0)

        # Gradienty wag
        gW3 = np.outer(dq, h2); gb3 = dq
        gW2 = np.outer(dz2, h1); gb2 = dz2
        gW1 = np.outer(dz1, x);  gb1 = dz1

        # Gradient clipping
        grad_norm = float(np.sqrt(
            np.sum(gW3**2) + np.sum(gW2**2) + np.sum(gW1**2) +
            np.sum(gb3**2) + np.sum(gb2**2) + np.sum(gb1**2)
        ))
        if grad_norm > grad_clip:
            scale = grad_clip / (grad_norm + 1e-8)
            gW3 *= scale; gb3 *= scale
            gW2 *= scale; gb2 *= scale
            gW1 *= scale; gb1 *= scale

        # Adam update
        b1t = self.beta1 ** self.t; b2t = self.beta2 ** self.t

        def adam_step(W, gW, mW, vW):
            mW[:] = self.beta1 * mW + (1 - self.beta1) * gW
            vW[:] = self.beta2 * vW + (1 - self.beta2) * gW * gW
            m_hat = mW / (1 - b1t)
            v_hat = vW / (1 - b2t)
            W += lr * m_hat / (np.sqrt(v_hat) + self.eps_adam)

        adam_step(self.W3, gW3, self.mW3, self.vW3)
        adam_step(self.b3, gb3, self.mb3, self.vb3)
        adam_step(self.W2, gW2, self.mW2, self.vW2)
        adam_step(self.b2, gb2, self.mb2, self.vb2)
        adam_step(self.W1, gW1, self.mW1, self.vW1)
        adam_step(self.b1, gb1, self.mb1, self.vb1)

        return grad_norm

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Forward bez cache (szybszy — do act())."""
        h1 = np.maximum(0.0, self.W1 @ x + self.b1)
        h2 = np.maximum(0.0, self.W2 @ h1 + self.b2)
        return self.W3 @ h2 + self.b3

    def state_dict(self) -> dict:
        return {k: v.copy() for k, v in self.__dict__.items()
                if isinstance(v, np.ndarray)}

    def load_state_dict(self, sd: dict):
        for k, v in sd.items():
            if hasattr(self, k):
                getattr(self, k)[:] = v


class DeepSARSAAgent:
    """
    On-policy SARSA z numpy MLP + Adam.

    Maskowanie akcji (CT — Continuous Trading):
      obs[1]  = position_norm ∈ [-1,+1]
      can_buy  = position_norm < 0.99   (nie max long)
      can_sell = position_norm > -0.99  (nie max short, symetrycznie)
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
        self.rng       = np.random.default_rng(seed)

        self.net = NumpyMLP(n_obs, self.cfg.hidden_size, n_actions, self.rng)
        self.epsilon = self.cfg.epsilon_start

        self.episode_td_errors:  List[float] = []
        self.episode_grad_norms: List[float] = []
        self.total_updates:      int         = 0

    def _mask(self, obs: np.ndarray) -> Tuple[bool, bool]:
        # obs[1] = position_norm = position / max_position ∈ [-1, +1]
        pos_norm = float(obs[1])
        can_buy  = pos_norm < 0.99   # nie na maksimum long
        can_sell = pos_norm > -0.99  # nie na maksimum short
        return can_buy, can_sell

    def act(self, obs: np.ndarray, explore: bool = True) -> int:
        can_buy, can_sell = self._mask(obs)

        if not can_buy and not can_sell:
            return 0  # HOLD (na granicy pozycji w obu kierunkach)

        if explore and self.rng.random() < self.epsilon:
            valid = [0]
            if can_buy:  valid += [1]   # BUY
            if can_sell: valid += [2]   # SELL
            return int(self.rng.choice(valid))

        q = self.net.predict(obs)
        if not can_buy:  q[1] = -np.inf
        if not can_sell: q[2] = -np.inf
        return int(np.argmax(q))

    def update(
        self,
        obs:      np.ndarray,
        action:   int,
        reward:   float,
        next_obs: np.ndarray,
        done:     bool,
    ) -> Tuple[float, float]:
        # Forward z cache dla backprop
        q_vals, cache = self.net.forward(obs)
        q_current = float(q_vals[action])

        # TD target
        if done:
            td_target = float(reward)
        else:
            can_b, can_s = self._mask(next_obs)
            if not can_b and not can_s:
                next_q = 0.0
            else:
                q_next = self.net.predict(next_obs)
                if not can_b: q_next[1] = -np.inf
                if not can_s: q_next[2] = -np.inf
                # on-policy: epsilon-greedy next action
                if self.rng.random() < self.epsilon:
                    valid = [0]
                    if can_b: valid += [1]
                    if can_s: valid += [2]
                    next_a = int(self.rng.choice(valid))
                else:
                    next_a = int(np.argmax(q_next))
                next_q = float(q_next[next_a])
            td_target = float(reward) + self.gamma * next_q

        td_error  = td_target - q_current
        grad_norm = self.net.backward(
            action, td_error, cache, self.cfg.lr, self.cfg.grad_clip
        )

        self.total_updates += 1
        self.episode_td_errors.append(abs(td_error))
        self.episode_grad_norms.append(grad_norm)

        return abs(td_error), grad_norm

    def decay_epsilon(self):
        self.epsilon = max(self.cfg.epsilon_end,
                          self.epsilon * self.cfg.epsilon_decay)

    def reset_episode(self):
        self.episode_td_errors  = []
        self.episode_grad_norms = []

    def episode_stats(self) -> dict:
        return {
            "mean_td_error":  float(np.mean(self.episode_td_errors))  if self.episode_td_errors  else 0.0,
            "mean_grad_norm": float(np.mean(self.episode_grad_norms)) if self.episode_grad_norms else 0.0,
            "epsilon":        self.epsilon,
            "gamma":          self.gamma,
        }

    def q_values(self, obs: np.ndarray) -> np.ndarray:
        return self.net.predict(obs)

    def state_dict(self) -> dict:
        return {"net": self.net.state_dict(), "epsilon": self.epsilon}

    def load_state_dict(self, sd: dict):
        self.net.load_state_dict(sd["net"])
        self.epsilon = sd["epsilon"]


class DeepSARSAMultiAgent:
    """N niezależnych agentów SARSA — identyczny interfejs jak wersja PyTorch."""

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

        n_params = (n_obs * self.cfg.hidden_size + self.cfg.hidden_size +
                    self.cfg.hidden_size**2 + self.cfg.hidden_size +
                    self.cfg.hidden_size * n_actions + n_actions)
        logger.info(
            f"DeepSARSAMultiAgent (numpy) | N={len(agent_ids)} | "
            f"params/agent={n_params} | "
            f"net: {n_obs}->{self.cfg.hidden_size}->{n_actions}"
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
        epsilons  = [a.epsilon for a in self.agents.values()]
        tds       = [np.mean(a.episode_td_errors)  if a.episode_td_errors  else 0.0
                     for a in self.agents.values()]
        gnorms    = [np.mean(a.episode_grad_norms) if a.episode_grad_norms else 0.0
                     for a in self.agents.values()]
        return {
            "mean_epsilon":  float(np.mean(epsilons)),
            "mean_gamma":    float(np.mean([a.gamma for a in self.agents.values()])),
            "mean_td_error": float(np.mean(tds)),
            "mean_grad_norm":float(np.mean(gnorms)),
        }

    def update_gammas(self, agent_ids: List[str], new_gammas: np.ndarray):
        for i, aid in enumerate(agent_ids):
            if aid in self.agents:
                self.agents[aid].gamma = float(new_gammas[i])

    def save(self, path: str):
        import pickle
        with open(path, "wb") as f:
            pickle.dump({aid: a.state_dict() for aid, a in self.agents.items()}, f)
        logger.info(f"Saved -> {path}")

    def load(self, path: str):
        import pickle
        with open(path, "rb") as f:
            state = pickle.load(f)
        for aid, sd in state.items():
            if aid in self.agents:
                self.agents[aid].load_state_dict(sd)
        logger.info(f"Loaded <- {path}")
