"""
Wspólne helpery dla algorytmów RL w środowisku HTM.
"""

from __future__ import annotations

import random
from typing import Dict, Optional, Sequence

import numpy as np

from codes.config import HTMConfig


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
    sentiment: float,
    sigma_i: float,
    threshold: float,
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
        "gamma_obs": float(obs[4]),
        "price_vs_start": float(obs[5]),
        "trend_short": float(obs[6]),
        "sigma_norm": float(obs[7]),
        "sentiment": float(sentiment),
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
        "threshold": float(threshold),
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
    public_gap_after: float,
    price_delta_step: float,
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
        "public_gap_after": float(public_gap_after),
        "price_delta_step": float(price_delta_step),
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
