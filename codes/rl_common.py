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
        "v_perceived_std": metrics.get("v_perceived_std", 0.0),
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
