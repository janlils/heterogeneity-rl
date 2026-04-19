"""
Wspólny evaluator polityk HTM.

Każdy algorytm jest oceniany tym samym protokołem równoległym:
wszyscy agenci obserwują ten sam stan rynku, akcje są wykonywane wspólnie,
reward liczony jest po pełnym kroku.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from codes.config import HTMConfig
from codes.double_auction import DoubleAuction, ZeroIntelligenceAgent
from codes.rl_common import build_episode_record, set_global_seeds


def _action_for_policy(algorithm_name: str, policy, obs, aid: str) -> int:
    name = algorithm_name.lower()
    if name in {"zi", "zero_intelligence", "zerointelligence"}:
        return int(policy[aid].act(obs))
    if "sarsa" in name:
        return int(policy.agents[aid].act(obs, explore=True))
    if "ppo" in name:
        deterministic = "deterministic" in name or "argmax" in name
        action, _, _, _ = policy.act_np(obs, aid, deterministic=deterministic)
        return int(action)
    if hasattr(policy, "act_np"):
        action, _, _, _ = policy.act_np(obs, aid, deterministic=True)
        return int(action)
    if hasattr(policy, "agents"):
        return int(policy.agents[aid].act(obs, explore=False))
    raise ValueError(f"Nieznany typ polityki dla evaluatora: {algorithm_name}")


def evaluate_policy(
    algorithm_name: str,
    policy,
    cfg: HTMConfig,
    diversity_score: float,
    n_episodes: int,
    seed: int,
    zi_baseline_trade_accuracy: Optional[float] = None,
    zi_baseline_positive_pnl_frac: Optional[float] = None,
) -> List[dict]:
    set_global_seeds(seed)
    da = DoubleAuction(cfg, seed=seed)
    da.reset(diversity_score=diversity_score, seed=seed)
    agent_ids = list(da.population.agents.keys())

    if algorithm_name.lower() in {"zi", "zero_intelligence", "zerointelligence"}:
        policy = {
            aid: ZeroIntelligenceAgent(p, cfg.env, seed=seed + i)
            for i, (aid, p) in enumerate(da.population.agents.items())
        }

    records: List[dict] = []

    for episode in range(n_episodes):
        da.reset_episode()

        while not da.done:
            actions = {
                aid: _action_for_policy(
                    algorithm_name,
                    policy,
                    da.get_observation(aid),
                    aid,
                )
                for aid in agent_ids
            }
            da.execute_parallel_actions(actions)
            da.compute_step_rewards()

        metrics = da.episode_metrics()
        extra = {}
        if zi_baseline_trade_accuracy is not None:
            extra["zi_baseline_trade_accuracy"] = zi_baseline_trade_accuracy
            extra["zi_baseline"] = zi_baseline_trade_accuracy
            extra["beats_zi"] = metrics.get("trade_accuracy", 0.0) > zi_baseline_trade_accuracy
        if zi_baseline_positive_pnl_frac is not None:
            extra["zi_baseline_positive_pnl_frac"] = zi_baseline_positive_pnl_frac
        records.append(build_episode_record(
            episode=episode,
            diversity_score=diversity_score,
            seed=seed,
            algorithm=algorithm_name,
            cfg=cfg,
            metrics=metrics,
            extra=extra,
        ))

    return records


def evaluate_zi(
    cfg: HTMConfig,
    diversity_score: float,
    n_episodes: int,
    seed: int,
) -> List[dict]:
    return evaluate_policy("ZI", None, cfg, diversity_score, n_episodes, seed)


def evaluate_sarsa(
    policy,
    cfg: HTMConfig,
    diversity_score: float,
    n_episodes: int,
    seed: int,
    zi_baseline_trade_accuracy: Optional[float] = None,
    zi_baseline_positive_pnl_frac: Optional[float] = None,
) -> List[dict]:
    return evaluate_policy(
        "DeepSARSA_EVAL",
        policy,
        cfg,
        diversity_score,
        n_episodes,
        seed,
        zi_baseline_trade_accuracy=zi_baseline_trade_accuracy,
        zi_baseline_positive_pnl_frac=zi_baseline_positive_pnl_frac,
    )


def evaluate_ppo(
    policy,
    cfg: HTMConfig,
    diversity_score: float,
    n_episodes: int,
    seed: int,
    zi_baseline_trade_accuracy: Optional[float] = None,
    zi_baseline_positive_pnl_frac: Optional[float] = None,
) -> List[dict]:
    return evaluate_policy(
        "PPO_EVAL",
        policy,
        cfg,
        diversity_score,
        n_episodes,
        seed,
        zi_baseline_trade_accuracy=zi_baseline_trade_accuracy,
        zi_baseline_positive_pnl_frac=zi_baseline_positive_pnl_frac,
    )
