"""
Skonsolidowane implementacje algorytmów i helperów decyzyjnych HTM.
"""

from __future__ import annotations

import logging
import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from codes.config import HTMConfig, DeepSARSAConfig

logger = logging.getLogger("htm.deep_sarsa")

DECISION_FEATURE_NAMES = [
    "signal_i",
    "pos_norm",
    "unrealized_pnl",
    "time_remaining",
    "price_vs_start",
    "trend_short",
]


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def action_mask_from_obs(obs: np.ndarray) -> np.ndarray:
    pos_norm = float(obs[1])
    can_buy = pos_norm < 0.99
    can_sell = pos_norm > -0.99
    return np.array([True, can_buy, can_sell], dtype=bool)


def action_masks_from_obs_batch(obs_batch: np.ndarray) -> np.ndarray:
    if obs_batch.size == 0:
        return np.zeros((0, 3), dtype=bool)
    pos_norm = obs_batch[:, 1].astype(np.float32, copy=False)
    can_buy = pos_norm < 0.99
    can_sell = pos_norm > -0.99
    return np.stack(
        [
            np.ones(len(obs_batch), dtype=bool),
            can_buy.astype(bool, copy=False),
            can_sell.astype(bool, copy=False),
        ],
        axis=1,
    )


def append_agent_id_feature(
    obs: np.ndarray,
    agent_id: str,
    agent_to_idx: Optional[Dict[str, int]],
) -> np.ndarray:
    if not agent_to_idx:
        return obs.astype(np.float32, copy=False)
    one_hot = np.zeros(len(agent_to_idx), dtype=np.float32)
    one_hot[agent_to_idx[agent_id]] = 1.0
    return np.concatenate([obs.astype(np.float32, copy=False), one_hot])


def build_episode_record(
    episode: int,
    diversity_score: float,
    seed: int,
    algorithm: str,
    cfg: HTMConfig,
    metrics: dict,
    extra: Optional[dict] = None,
    agent_gammas: Optional[Sequence[float]] = None,
) -> dict:
    gamma_std = float(np.std(agent_gammas)) if agent_gammas is not None and len(agent_gammas) > 0 else 0.0
    record = {
        "episode": episode,
        "diversity_score": diversity_score,
        "seed": seed,
        "algorithm": algorithm,
        "n_agents": cfg.env.n_agents,
        "gamma_std": gamma_std,
        "eq_price": metrics.get("eq_price", 0.5),
        "eq_price_start": metrics.get("eq_price_start", 0.5),
        "ref_price_final": metrics.get("ref_price_final", 0.5),
        "trade_accuracy": metrics.get("trade_accuracy", 0.0),
        "mean_pnl": metrics.get("mean_pnl", 0.0),
        "mean_total_pnl": metrics.get("mean_total_pnl", 0.0),
        "mean_total_pnl_gross": metrics.get("mean_total_pnl_gross", metrics.get("mean_total_pnl", 0.0)),
        "mean_transaction_cost": metrics.get("mean_transaction_cost", 0.0),
        "mean_terminal_pnl": metrics.get("mean_terminal_pnl", 0.0),
        "positive_pnl_frac": metrics.get("positive_pnl_frac", 0.0),
        "terminal_positive_frac": metrics.get("terminal_positive_frac", 0.0),
        "n_trades": metrics.get("n_trades", 0),
        "n_trades_closed": metrics.get("n_trades_closed", 0),
        "n_position_closes": metrics.get("n_position_closes", 0),
        "price_volatility": metrics.get("price_volatility", 0.0),
        "price_range": metrics.get("price_range", 0.0),
        "mean_abs_position": metrics.get("mean_abs_position", 0.0),
        "mean_value_gap": metrics.get("mean_value_gap", 0.0),
        "pct_chartists": metrics.get("pct_chartists", 0.0),
        "corr_type_pnl": metrics.get("corr_type_pnl", 0.0),
        "action_buy_frac": metrics.get("action_buy_frac", 0.0),
        "action_sell_frac": metrics.get("action_sell_frac", 0.0),
        "action_hold_frac": metrics.get("action_hold_frac", 0.0),
        "gini": metrics.get("gini_pnl", metrics.get("gini", 0.0)),
        "primary_metric": "trade_accuracy",
        "transaction_cost_per_fill": cfg.env.transaction_cost_per_fill,
    }
    if extra:
        record.update(extra)
    return record


def build_agent_sample_row(
    *,
    algorithm: str,
    phase: str,
    diversity_score: float,
    seed: int,
    episode: int,
    step: int,
    agent_id: str,
    trader_type: float,
    agent_type: str,
    action: int,
    action_name: str,
    executed: bool,
    obs: np.ndarray,
    public_gap_before: float,
    eq_price_before: float,
    ref_price_before: float,
    public_gap_after: float,
    eq_price_after: float,
    ref_price_after: float,
    position_before: int,
    position: int,
    entry_price_after: float,
    reward_this_step: float,
    gross_realized_pnl_this_step: float,
    realized_pnl_this_step: float,
    transaction_cost_this_step: float,
    gross_realized_pnl_cum: float,
    realized_pnl_cum: float,
    transaction_cost_cum: float,
    n_trades_closed: int,
    sigma_i: float,
) -> dict:
    reward_this_step = float(reward_this_step)
    gross_realized_pnl_this_step = float(gross_realized_pnl_this_step)
    realized_pnl_this_step = float(realized_pnl_this_step)
    transaction_cost_this_step = float(transaction_cost_this_step)
    return {
        "algorithm": algorithm,
        "phase": phase,
        "diversity_score": diversity_score,
        "seed": seed,
        "episode": episode,
        "step": step,
        "agent_id": agent_id,
        "trader_type": float(trader_type),
        "agent_type": agent_type,
        "action": int(action),
        "action_name": action_name,
        "executed": bool(executed),
        "signal_i": float(obs[0]),
        "pos_norm": float(obs[1]),
        "unrealized_pnl": float(obs[2]),
        "time_remaining": float(obs[3]),
        "price_vs_start": float(obs[4]),
        "trend_short": float(obs[5]),
        "public_gap_before": float(public_gap_before),
        "eq_price_before": float(eq_price_before),
        "ref_price_before": float(ref_price_before),
        "public_gap_after": float(public_gap_after),
        "eq_price_after": float(eq_price_after),
        "ref_price_after": float(ref_price_after),
        "position_before": int(position_before),
        "position": int(position),
        "entry_price_after": float(entry_price_after),
        "reward_this_step": reward_this_step,
        "gross_realized_pnl_this_step": gross_realized_pnl_this_step,
        "realized_pnl_this_step": realized_pnl_this_step,
        "transaction_cost_this_step": transaction_cost_this_step,
        "mtm_this_step": reward_this_step - realized_pnl_this_step,
        "gross_realized_pnl_cum": float(gross_realized_pnl_cum),
        "realized_pnl_cum": float(realized_pnl_cum),
        "transaction_cost_cum": float(transaction_cost_cum),
        "n_trades_closed": int(n_trades_closed),
        "sigma_i": float(sigma_i),
    }


def build_env_step_row(
    *,
    algorithm: str,
    phase: str,
    diversity_score: float,
    seed: int,
    episode: int,
    step: int,
    eq_price_before: float,
    ref_price_before: float,
    public_gap_before: float,
    eq_price_after: float,
    ref_price_after: float,
    exec_price: float,
    public_gap_after: float,
    price_delta_step: float,
    sigma_step: float,
    crisis_step: bool,
    mean_signal: float,
    std_signal: float,
    mean_sigma: float,
    mean_position_before: float,
    mean_position_after: float,
    n_buy: int,
    n_sell: int,
    n_hold: int,
    net_flow: int,
    mean_reward: float,
    mean_gross_realized_pnl: float,
    mean_realized_pnl: float,
    mean_transaction_cost: float,
    mean_mtm: float,
    n_executed: int,
    n_trades_closed_cum: int,
) -> dict:
    return {
        "algorithm": algorithm,
        "phase": phase,
        "diversity_score": diversity_score,
        "seed": seed,
        "episode": episode,
        "step": step,
        "eq_price_before": float(eq_price_before),
        "ref_price_before": float(ref_price_before),
        "public_gap_before": float(public_gap_before),
        "eq_price_after": float(eq_price_after),
        "ref_price_after": float(ref_price_after),
        "exec_price": float(exec_price),
        "public_gap_after": float(public_gap_after),
        "price_delta_step": float(price_delta_step),
        "sigma_step": float(sigma_step),
        "crisis_step": bool(crisis_step),
        "mean_signal": float(mean_signal),
        "std_signal": float(std_signal),
        "mean_sigma": float(mean_sigma),
        "mean_position_before": float(mean_position_before),
        "mean_position_after": float(mean_position_after),
        "n_buy": int(n_buy),
        "n_sell": int(n_sell),
        "n_hold": int(n_hold),
        "net_flow": int(net_flow),
        "mean_reward": float(mean_reward),
        "mean_gross_realized_pnl": float(mean_gross_realized_pnl),
        "mean_realized_pnl": float(mean_realized_pnl),
        "mean_transaction_cost": float(mean_transaction_cost),
        "mean_mtm": float(mean_mtm),
        "n_executed": int(n_executed),
        "n_trades_closed_cum": int(n_trades_closed_cum),
    }


def aggregate_agent_eval_episode_rows(rows: Sequence[dict]) -> list[dict]:
    grouped: Dict[tuple, dict] = {}
    for row in rows:
        key = (
            row.get("algorithm"),
            row.get("phase"),
            float(row.get("diversity_score", 0.0)),
            int(row.get("seed", 0)),
            row.get("agent_id"),
        )
        bucket = grouped.get(key)
        if bucket is None:
            bucket = {
                "algorithm": row.get("algorithm"),
                "phase": row.get("phase"),
                "diversity_score": float(row.get("diversity_score", 0.0)),
                "seed": int(row.get("seed", 0)),
                "agent_id": row.get("agent_id"),
                "sigma_i": float(row.get("sigma_i", 0.0)),
                "trader_type": float(row.get("trader_type", 0.0)),
                "sum_realized_pnl": 0.0,
                "sum_trade_accuracy_agent": 0.0,
                "sum_n_trades_closed": 0.0,
                "sum_n_trades_won": 0.0,
                "sum_position_end": 0.0,
                "sum_buy_frac": 0.0,
                "sum_sell_frac": 0.0,
                "sum_hold_frac": 0.0,
                "sum_signal_alignment_rate": 0.0,
                "sum_directional_action_rate": 0.0,
                "n_eval_episodes": 0,
            }
            grouped[key] = bucket
        bucket["sum_realized_pnl"] += float(row.get("realized_pnl", 0.0))
        bucket["sum_trade_accuracy_agent"] += float(row.get("trade_accuracy_agent", 0.0))
        bucket["sum_n_trades_closed"] += float(row.get("n_trades_closed", 0.0))
        bucket["sum_n_trades_won"] += float(row.get("n_trades_won", 0.0))
        bucket["sum_position_end"] += float(row.get("position_end", 0.0))
        bucket["sum_buy_frac"] += float(row.get("buy_frac", 0.0))
        bucket["sum_sell_frac"] += float(row.get("sell_frac", 0.0))
        bucket["sum_hold_frac"] += float(row.get("hold_frac", 0.0))
        bucket["sum_signal_alignment_rate"] += float(row.get("signal_alignment_rate", 0.0))
        bucket["sum_directional_action_rate"] += float(row.get("directional_action_rate", 0.0))
        bucket["n_eval_episodes"] += 1

    out: list[dict] = []
    for bucket in grouped.values():
        n = max(int(bucket["n_eval_episodes"]), 1)
        out.append({
            "algorithm": bucket["algorithm"],
            "phase": bucket["phase"],
            "diversity_score": bucket["diversity_score"],
            "seed": bucket["seed"],
            "agent_id": bucket["agent_id"],
            "sigma_i": bucket["sigma_i"],
            "trader_type": bucket["trader_type"],
            "mean_realized_pnl": bucket["sum_realized_pnl"] / n,
            "mean_trade_accuracy_agent": bucket["sum_trade_accuracy_agent"] / n,
            "mean_n_trades_closed": bucket["sum_n_trades_closed"] / n,
            "mean_n_trades_won": bucket["sum_n_trades_won"] / n,
            "mean_position_end": bucket["sum_position_end"] / n,
            "buy_frac": bucket["sum_buy_frac"] / n,
            "sell_frac": bucket["sum_sell_frac"] / n,
            "hold_frac": bucket["sum_hold_frac"] / n,
            "signal_alignment_rate": bucket["sum_signal_alignment_rate"] / n,
            "directional_action_rate": bucket["sum_directional_action_rate"] / n,
            "n_eval_episodes": bucket["n_eval_episodes"],
        })
    return out


def init_decision_feature_stats() -> dict:
    feature_stats = {
        name: {"sum_x": 0.0, "sum_x2": 0.0, "sum_xy": 0.0}
        for name in DECISION_FEATURE_NAMES
    }
    return {
        "n": 0,
        "sum_y": 0.0,
        "sum_y2": 0.0,
        "buy": 0,
        "sell": 0,
        "hold": 0,
        "features": feature_stats,
    }


def update_decision_feature_stats(stats: dict, obs: np.ndarray, action: int) -> None:
    y = 1.0 if int(action) == 1 else (-1.0 if int(action) == 2 else 0.0)
    stats["n"] += 1
    stats["sum_y"] += y
    stats["sum_y2"] += y * y
    if int(action) == 1:
        stats["buy"] += 1
    elif int(action) == 2:
        stats["sell"] += 1
    else:
        stats["hold"] += 1
    for idx, name in enumerate(DECISION_FEATURE_NAMES):
        x = float(obs[idx])
        f = stats["features"][name]
        f["sum_x"] += x
        f["sum_x2"] += x * x
        f["sum_xy"] += x * y


def _corr_from_sums(n: int, sum_x: float, sum_x2: float, sum_y: float, sum_y2: float, sum_xy: float) -> float:
    if n < 2:
        return 0.0
    cov = sum_xy - (sum_x * sum_y / n)
    var_x = sum_x2 - (sum_x * sum_x / n)
    var_y = sum_y2 - (sum_y * sum_y / n)
    if var_x <= 1e-12 or var_y <= 1e-12:
        return 0.0
    return float(cov / np.sqrt(var_x * var_y))


def finalize_decision_feature_summary(
    stats: dict,
    *,
    algorithm: str,
    phase: str,
    diversity_score: float,
    seed: int,
) -> dict:
    n = max(int(stats["n"]), 1)
    out = {
        "algorithm": algorithm,
        "phase": phase,
        "diversity_score": float(diversity_score),
        "seed": int(seed),
        "n_obs_actions": int(stats["n"]),
        "buy_frac": float(stats["buy"] / n),
        "sell_frac": float(stats["sell"] / n),
        "hold_frac": float(stats["hold"] / n),
    }
    for name in DECISION_FEATURE_NAMES:
        f = stats["features"][name]
        out[f"corr_{name}_action_dir"] = _corr_from_sums(
            int(stats["n"]),
            float(f["sum_x"]),
            float(f["sum_x2"]),
            float(stats["sum_y"]),
            float(stats["sum_y2"]),
            float(f["sum_xy"]),
        )
    return out


class NumpyMLP:
    """
    2-ukryta MLP w czystym numpy.
    Input -> Dense(hidden, ReLU) -> Dense(hidden, ReLU) -> Output
    """

    def __init__(self, n_in: int, n_hidden: int, n_out: int, rng: np.random.Generator):
        scale1 = np.sqrt(2.0 / n_in)
        scale2 = np.sqrt(2.0 / n_hidden)
        self.W1 = rng.standard_normal((n_hidden, n_in)).astype(np.float32) * scale1
        self.b1 = np.zeros(n_hidden, np.float32)
        self.W2 = rng.standard_normal((n_hidden, n_hidden)).astype(np.float32) * scale2
        self.b2 = np.zeros(n_hidden, np.float32)
        self.W3 = rng.standard_normal((n_out, n_hidden)).astype(np.float32) * 1e-2
        self.b3 = np.zeros(n_out, np.float32)

        self.mW1 = np.zeros_like(self.W1)
        self.vW1 = np.zeros_like(self.W1)
        self.mb1 = np.zeros_like(self.b1)
        self.vb1 = np.zeros_like(self.b1)
        self.mW2 = np.zeros_like(self.W2)
        self.vW2 = np.zeros_like(self.W2)
        self.mb2 = np.zeros_like(self.b2)
        self.vb2 = np.zeros_like(self.b2)
        self.mW3 = np.zeros_like(self.W3)
        self.vW3 = np.zeros_like(self.W3)
        self.mb3 = np.zeros_like(self.b3)
        self.vb3 = np.zeros_like(self.b3)
        self.t = 0
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps_adam = 1e-8

    def forward(self, x: np.ndarray):
        z1 = self.W1 @ x + self.b1
        h1 = np.maximum(0.0, z1)
        z2 = self.W2 @ h1 + self.b2
        h2 = np.maximum(0.0, z2)
        q = self.W3 @ h2 + self.b3
        return q, (x, z1, h1, z2, h2)

    def backward(self, action: int, td_error: float, cache, lr: float, grad_clip: float) -> float:
        x, z1, h1, z2, h2 = cache
        self.t += 1

        dq = np.zeros(len(self.b3), np.float32)
        dq[action] = np.float32(td_error)

        dh2 = self.W3.T @ dq
        dz2 = dh2 * (z2 > 0)
        dh1 = self.W2.T @ dz2
        dz1 = dh1 * (z1 > 0)

        gW3 = np.outer(dq, h2)
        gb3 = dq
        gW2 = np.outer(dz2, h1)
        gb2 = dz2
        gW1 = np.outer(dz1, x)
        gb1 = dz1

        grad_norm = float(np.sqrt(
            np.sum(gW3 ** 2) + np.sum(gW2 ** 2) + np.sum(gW1 ** 2) +
            np.sum(gb3 ** 2) + np.sum(gb2 ** 2) + np.sum(gb1 ** 2)
        ))
        if grad_norm > grad_clip:
            scale = grad_clip / (grad_norm + 1e-8)
            gW3 *= scale
            gb3 *= scale
            gW2 *= scale
            gb2 *= scale
            gW1 *= scale
            gb1 *= scale

        b1t = self.beta1 ** self.t
        b2t = self.beta2 ** self.t

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
        h1 = np.maximum(0.0, self.W1 @ x + self.b1)
        h2 = np.maximum(0.0, self.W2 @ h1 + self.b2)
        return self.W3 @ h2 + self.b3

    def state_dict(self) -> dict:
        return {k: v.copy() for k, v in self.__dict__.items() if isinstance(v, np.ndarray)}

    def load_state_dict(self, sd: dict):
        for k, v in sd.items():
            if hasattr(self, k):
                getattr(self, k)[:] = v


class DeepSARSAAgent:
    def __init__(
        self,
        agent_id: str,
        gamma: float,
        n_obs: int,
        n_actions: int,
        cfg: DeepSARSAConfig = None,
        seed: int = 42,
    ):
        self.agent_id = agent_id
        self.gamma = gamma
        self.n_actions = n_actions
        self.cfg = cfg or DeepSARSAConfig()
        self.rng = np.random.default_rng(seed)

        self.net = NumpyMLP(n_obs, self.cfg.hidden_size, n_actions, self.rng)
        self.epsilon = self.cfg.epsilon_start
        self.episode_td_errors: List[float] = []
        self.episode_grad_norms: List[float] = []
        self.total_updates = 0

    def _mask(self, obs: np.ndarray) -> Tuple[bool, bool]:
        pos_norm = float(obs[1])
        can_buy = pos_norm < 0.99
        can_sell = pos_norm > -0.99
        return can_buy, can_sell

    def act(self, obs: np.ndarray, explore: bool = True) -> int:
        can_buy, can_sell = self._mask(obs)
        if not can_buy and not can_sell:
            return 0
        if explore and self.rng.random() < self.epsilon:
            valid = [0]
            if can_buy:
                valid.append(1)
            if can_sell:
                valid.append(2)
            return int(self.rng.choice(valid))
        q = self.net.predict(obs)
        if not can_buy:
            q[1] = -np.inf
        if not can_sell:
            q[2] = -np.inf
        return int(np.argmax(q))

    def expected_next_q(self, obs: np.ndarray) -> float:
        can_buy, can_sell = self._mask(obs)
        valid = [0]
        if can_buy:
            valid.append(1)
        if can_sell:
            valid.append(2)
        if len(valid) == 1 and valid[0] == 0 and not can_buy and not can_sell:
            return 0.0

        q_next = self.net.predict(obs)
        if not can_buy:
            q_next[1] = -np.inf
        if not can_sell:
            q_next[2] = -np.inf
        greedy = int(np.argmax(q_next))
        eps_share = self.epsilon / len(valid)
        expected = 0.0
        for action in valid:
            prob = eps_share + (1.0 - self.epsilon if action == greedy else 0.0)
            expected += prob * float(q_next[action])
        return expected

    def _scaled_reward(self, reward: float) -> float:
        return float(reward) * float(self.cfg.reward_scale)

    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> Tuple[float, float]:
        q_vals, cache = self.net.forward(obs)
        q_current = float(q_vals[action])
        if done:
            td_target = self._scaled_reward(reward)
        else:
            next_q = self.expected_next_q(next_obs)
            td_target = self._scaled_reward(reward) + self.gamma * next_q
        td_error = td_target - q_current
        grad_norm = self.net.backward(action, td_error, cache, self.cfg.lr, self.cfg.grad_clip)
        self.total_updates += 1
        self.episode_td_errors.append(abs(td_error))
        self.episode_grad_norms.append(grad_norm)
        return abs(td_error), grad_norm

    def update_with_target(self, obs: np.ndarray, action: int, target: float) -> Tuple[float, float]:
        q_vals, cache = self.net.forward(obs)
        td_error = float(target) - float(q_vals[action])
        grad_norm = self.net.backward(action, td_error, cache, self.cfg.lr, self.cfg.grad_clip)
        self.total_updates += 1
        self.episode_td_errors.append(abs(td_error))
        self.episode_grad_norms.append(grad_norm)
        return abs(td_error), grad_norm

    def decay_epsilon(self):
        self.epsilon = max(self.cfg.epsilon_end, self.epsilon * self.cfg.epsilon_decay)

    def reset_episode(self):
        self.episode_td_errors = []
        self.episode_grad_norms = []

    def episode_stats(self) -> dict:
        return {
            "mean_td_error": float(np.mean(self.episode_td_errors)) if self.episode_td_errors else 0.0,
            "mean_grad_norm": float(np.mean(self.episode_grad_norms)) if self.episode_grad_norms else 0.0,
            "epsilon": self.epsilon,
            "gamma": self.gamma,
        }

    def q_values(self, obs: np.ndarray) -> np.ndarray:
        return self.net.predict(obs)

    def state_dict(self) -> dict:
        return {"net": self.net.state_dict(), "epsilon": self.epsilon}

    def load_state_dict(self, sd: dict):
        self.net.load_state_dict(sd["net"])
        self.epsilon = sd["epsilon"]


class DeepSARSAMultiAgent:
    def __init__(
        self,
        agent_ids: List[str],
        agent_gammas: np.ndarray,
        n_obs: int,
        n_actions: int,
        cfg: DeepSARSAConfig = None,
        seed: int = 42,
    ):
        self.cfg = cfg or DeepSARSAConfig()
        self.agents: Dict[str, DeepSARSAAgent] = {}
        for i, aid in enumerate(agent_ids):
            self.agents[aid] = DeepSARSAAgent(
                agent_id=aid,
                gamma=float(agent_gammas[i]),
                n_obs=n_obs,
                n_actions=n_actions,
                cfg=self.cfg,
                seed=seed + i,
            )

        n_params = (
            n_obs * self.cfg.hidden_size + self.cfg.hidden_size +
            self.cfg.hidden_size ** 2 + self.cfg.hidden_size +
            self.cfg.hidden_size * n_actions + n_actions
        )
        logger.info(
            f"DeepSARSAMultiAgent (numpy) | N={len(agent_ids)} | "
            f"params/agent={n_params} | "
            f"net: {n_obs}->{self.cfg.hidden_size}->{n_actions}"
        )

    def act(self, observations: Dict[str, np.ndarray], explore: bool = True) -> Dict[str, int]:
        return {aid: self.agents[aid].act(obs, explore=explore) for aid, obs in observations.items() if aid in self.agents}

    def update_all(
        self,
        obs: Dict[str, np.ndarray],
        actions: Dict[str, int],
        rewards: Dict[str, float],
        next_obs: Dict[str, np.ndarray],
        dones: Dict[str, bool],
    ) -> Dict[str, float]:
        td_errors = {}
        for aid in obs:
            if aid in self.agents and aid in actions:
                td_e, _ = self.agents[aid].update(obs[aid], actions[aid], rewards[aid], next_obs[aid], dones[aid])
                td_errors[aid] = td_e
        return td_errors

    def end_episode(self):
        for agent in self.agents.values():
            agent.decay_epsilon()
            agent.reset_episode()

    def population_stats(self) -> dict:
        epsilons = [a.epsilon for a in self.agents.values()]
        tds = [np.mean(a.episode_td_errors) if a.episode_td_errors else 0.0 for a in self.agents.values()]
        gnorms = [np.mean(a.episode_grad_norms) if a.episode_grad_norms else 0.0 for a in self.agents.values()]
        return {
            "mean_epsilon": float(np.mean(epsilons)),
            "mean_gamma": float(np.mean([a.gamma for a in self.agents.values()])),
            "mean_td_error": float(np.mean(tds)),
            "mean_grad_norm": float(np.mean(gnorms)),
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

class SignalRulePolicy:
    """
    Prosta polityka regułowa oparta na prywatnym sygnale.

    Dla max_position=1 działa jak target inventory:
      - BUY jeśli sygnał jest dodatni i agent nie jest już long
      - SELL jeśli sygnał jest ujemny i agent nie jest już short
      - HOLD w przeciwnym razie
    """

    def __init__(self, threshold: float = 0.0):
        self.threshold = float(threshold)

    def act(self, obs: np.ndarray) -> int:
        signal_i = float(obs[0])
        pos_norm = float(obs[1])
        if signal_i > self.threshold and pos_norm < 0.99:
            return 1
        if signal_i < -self.threshold and pos_norm > -0.99:
            return 2
        return 0
