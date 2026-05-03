"""
Wspólny evaluator polityk HTM.

Każdy algorytm jest oceniany tym samym protokołem równoległym:
wszyscy agenci obserwują ten sam stan rynku, akcje są wykonywane wspólnie,
reward liczony jest po pełnym kroku.
"""

from __future__ import annotations

import dataclasses
from typing import List, Optional

import numpy as np

from codes.config import EnvConfig, HTMConfig
from codes.double_auction import DoubleAuction, ZeroIntelligenceAgent
from codes.evaluation import coordination_stats, evaluate_same_population
from codes.rl_common import set_global_seeds


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
    if hasattr(policy, "act"):
        return int(policy.act(obs))
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
) -> tuple[List[dict], List[dict], List[dict]]:
    set_global_seeds(seed)
    da = DoubleAuction(cfg, seed=seed)
    da.reset(diversity_score=diversity_score, seed=seed)

    if algorithm_name.lower() in {"zi", "zero_intelligence", "zerointelligence"}:
        policy = {
            aid: ZeroIntelligenceAgent(p, cfg.env, seed=seed + i)
            for i, (aid, p) in enumerate(da.population.agents.items())
        }

    def action_selector(aid: str, obs: np.ndarray) -> int:
        if hasattr(policy, "set_global_state"):
            policy.set_global_state(da.get_global_state())
        return _action_for_policy(algorithm_name, policy, obs, aid)

    def extra_builder(metrics: dict, step_actions: List[np.ndarray]) -> dict:
        same_action_frac, effective_n = coordination_stats(step_actions, cfg.env.n_actions)
        extra = {
            "same_action_frac": same_action_frac,
            "effective_N": effective_n,
        }
        if zi_baseline_trade_accuracy is not None:
            extra["zi_baseline_trade_accuracy"] = zi_baseline_trade_accuracy
            extra["zi_baseline"] = zi_baseline_trade_accuracy
            extra["beats_zi"] = metrics.get("trade_accuracy", 0.0) > zi_baseline_trade_accuracy
        if zi_baseline_positive_pnl_frac is not None:
            extra["zi_baseline_positive_pnl_frac"] = zi_baseline_positive_pnl_frac
        return extra

    return evaluate_same_population(
        da=da,
        cfg=cfg,
        diversity_score=diversity_score,
        seed=seed,
        n_eval_episodes=n_episodes,
        algorithm_name=algorithm_name,
        phase="eval",
        action_selector=action_selector,
        extra_builder=extra_builder,
        collect_coordination=True,
    )


def evaluate_zi(
    cfg: HTMConfig,
    diversity_score: float,
    n_episodes: int,
    seed: int,
) -> tuple[List[dict], List[dict], List[dict]]:
    return evaluate_policy("ZI", None, cfg, diversity_score, n_episodes, seed)


def evaluate_sarsa(
    policy,
    cfg: HTMConfig,
    diversity_score: float,
    n_episodes: int,
    seed: int,
    zi_baseline_trade_accuracy: Optional[float] = None,
    zi_baseline_positive_pnl_frac: Optional[float] = None,
 ) -> tuple[List[dict], List[dict], List[dict]]:
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
 ) -> tuple[List[dict], List[dict], List[dict]]:
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


def evaluate_ppo_no_impact(
    policy,
    cfg: HTMConfig,
    diversity_score: float,
    n_episodes: int,
    seed: int,
    zi_baseline_trade_accuracy: Optional[float] = None,
    zi_baseline_positive_pnl_frac: Optional[float] = None,
) -> tuple[List[dict], List[dict], List[dict]]:
    env_no_impact = dataclasses.replace(
        EnvConfig.no_impact(),
        n_agents=cfg.env.n_agents,
        episode_steps=cfg.env.episode_steps,
        max_position=cfg.env.max_position,
        use_market_maker=cfg.env.use_market_maker,
        temp_impact=cfg.env.temp_impact,
        p_min=cfg.env.p_min,
        p_max=cfg.env.p_max,
        auto_liquidate_end=cfg.env.auto_liquidate_end,
    )
    cfg_no_impact = dataclasses.replace(cfg, env=env_no_impact)
    return evaluate_policy(
        "PPO_EVAL_NO_IMPACT",
        policy,
        cfg_no_impact,
        diversity_score,
        n_episodes,
        seed,
        zi_baseline_trade_accuracy=zi_baseline_trade_accuracy,
        zi_baseline_positive_pnl_frac=zi_baseline_positive_pnl_frac,
    )
