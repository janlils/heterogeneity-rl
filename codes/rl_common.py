"""
Wspólne helpery dla algorytmów RL w środowisku HTM.
"""

from __future__ import annotations

import random
from typing import Dict, Optional, Sequence

import numpy as np

from codes.config import HTMConfig


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
    pos_norm = float(obs[1])  # obs[1] = position_norm (indeks stały)
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
    realized_pnl_this_step: float,
    realized_pnl_cum: float,
    n_trades_closed: int,
    sigma_i: float,
) -> dict:
    reward_this_step = float(reward_this_step)
    realized_pnl_this_step = float(realized_pnl_this_step)
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
        "realized_pnl_this_step": realized_pnl_this_step,
        "mtm_this_step": reward_this_step - realized_pnl_this_step,
        "realized_pnl_cum": float(realized_pnl_cum),
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
    mean_realized_pnl: float,
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
        "mean_realized_pnl": float(mean_realized_pnl),
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
        name: {
            "sum_x": 0.0,
            "sum_x2": 0.0,
            "sum_xy": 0.0,
        }
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
