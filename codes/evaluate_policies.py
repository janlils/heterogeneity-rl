"""
Wspólny evaluator polityk HTM.

Każdy algorytm jest oceniany tym samym protokołem równoległym:
wszyscy agenci obserwują ten sam stan rynku, akcje są wykonywane wspólnie,
reward liczony jest po pełnym kroku.
"""

from __future__ import annotations

import csv
import dataclasses
from pathlib import Path
from typing import List, Optional

import numpy as np

from codes.config import EnvConfig, HTMConfig, RESULTS_DIR
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


def _coordination_stats(step_actions: List[np.ndarray], n_actions: int) -> tuple[float, float]:
    if not step_actions:
        return 0.0, 0.0

    same_action_steps = 0
    entropies = []
    for actions in step_actions:
        if actions.size == 0:
            continue
        if np.all(actions == actions[0]):
            same_action_steps += 1
        counts = np.bincount(actions.astype(int), minlength=n_actions).astype(np.float64)
        probs = counts / max(np.sum(counts), 1.0)
        probs = probs[probs > 0]
        entropy = float(-np.sum(probs * np.log(probs))) if probs.size > 0 else 0.0
        entropies.append(entropy)

    same_action_frac = same_action_steps / len(step_actions)
    mean_entropy = float(np.mean(entropies)) if entropies else 0.0
    effective_n = float(np.exp(mean_entropy))
    return float(same_action_frac), effective_n


def evaluate_policy(
    algorithm_name: str,
    policy,
    cfg: HTMConfig,
    diversity_score: float,
    n_episodes: int,
    seed: int,
    zi_baseline_trade_accuracy: Optional[float] = None,
    zi_baseline_positive_pnl_frac: Optional[float] = None,
    log_trajectories: bool = False,
    trajectory_path: Optional[Path] = None,
) -> List[dict]:
    log_episode_stride = 10
    log_step_stride = 10
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
    agent_gammas = [da.population.agents[aid].gamma for aid in agent_ids]
    trajectory_rows: List[dict] = []
    agent_step_rows: List[dict] = []
    trajectory_path = trajectory_path or (RESULTS_DIR / "trajectories_eval.csv")
    agent_step_path = RESULTS_DIR / "agent_step_log.csv"
    should_log_agent_steps = log_trajectories and abs(diversity_score - 1.0) < 1e-9

    for episode in range(n_episodes):
        da.reset_episode()
        step_actions: List[np.ndarray] = []
        should_log_episode = log_trajectories and (episode % log_episode_stride == 0)

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
            e = cfg.env
            buy_count = sum(
                1 for aid, action in actions.items()
                if action == e.ACTION_BUY_MARKET
                and da.population.agents[aid].position < da.population.agents[aid].max_position
            )
            sell_count = sum(
                1 for aid, action in actions.items()
                if action == e.ACTION_SELL_MARKET
                and da.population.agents[aid].position > -da.population.agents[aid].max_position
            )
            net_flow = buy_count - sell_count
            step_actions.append(np.array([actions[aid] for aid in agent_ids], dtype=np.int32))
            da.execute_parallel_actions(actions)
            rewards, _ = da.compute_step_rewards()
            should_log_step = should_log_episode and (da._step % log_step_stride == 0)
            if should_log_step:
                sentiments = [da.population.agents[aid].sentiment for aid in agent_ids]
                trajectory_rows.append({
                    "algorithm": algorithm_name,
                    "diversity_score": diversity_score,
                    "seed": seed,
                    "episode": episode,
                    "step": da._step,
                    "ref_price": da.ref_price,
                    "eq_price": da.eq_price,
                    "mean_sentiment": float(np.mean(sentiments)),
                    "std_sentiment": float(np.std(sentiments)),
                    "net_flow": net_flow,
                })
            if should_log_agent_steps and should_log_step:
                for aid in agent_ids:
                    agent = da.population.agents[aid]
                    trader_type = agent.alpha_i / max(agent.alpha_i + agent.beta_i, 1e-9)
                    value_gap = float(np.tanh((agent.V_perceived - da.ref_price) / 0.05))
                    agent_step_rows.append({
                        "algorithm": algorithm_name,
                        "diversity_score": diversity_score,
                        "episode": episode,
                        "step": da._step,
                        "agent_id": aid,
                        "trader_type": trader_type,
                        "action": actions[aid],
                        "sentiment": float(agent.sentiment),
                        "value_gap": value_gap,
                        "position": int(agent.position),
                        "realized_pnl_this_step": float(rewards.get(aid, 0.0)),
                    })

        metrics = da.episode_metrics()
        same_action_frac, effective_n = _coordination_stats(step_actions, cfg.env.n_actions)
        extra = {}
        if zi_baseline_trade_accuracy is not None:
            extra["zi_baseline_trade_accuracy"] = zi_baseline_trade_accuracy
            extra["zi_baseline"] = zi_baseline_trade_accuracy
            extra["beats_zi"] = metrics.get("trade_accuracy", 0.0) > zi_baseline_trade_accuracy
        if zi_baseline_positive_pnl_frac is not None:
            extra["zi_baseline_positive_pnl_frac"] = zi_baseline_positive_pnl_frac
        extra["same_action_frac"] = same_action_frac
        extra["effective_N"] = effective_n
        records.append(build_episode_record(
            episode=episode,
            diversity_score=diversity_score,
            seed=seed,
            algorithm=algorithm_name,
            cfg=cfg,
            metrics=metrics,
            extra=extra,
            agent_gammas=agent_gammas,
        ))

    if log_trajectories and trajectory_rows:
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(trajectory_rows[0].keys())
        file_exists = trajectory_path.exists()
        with trajectory_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(trajectory_rows)
    if should_log_agent_steps and agent_step_rows:
        agent_step_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(agent_step_rows[0].keys())
        file_exists = agent_step_path.exists()
        with agent_step_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(agent_step_rows)

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
    log_trajectories: bool = False,
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
        log_trajectories=log_trajectories,
    )


def evaluate_ppo(
    policy,
    cfg: HTMConfig,
    diversity_score: float,
    n_episodes: int,
    seed: int,
    zi_baseline_trade_accuracy: Optional[float] = None,
    zi_baseline_positive_pnl_frac: Optional[float] = None,
    log_trajectories: bool = False,
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
        log_trajectories=log_trajectories,
    )


def evaluate_ppo_no_impact(
    policy,
    cfg: HTMConfig,
    diversity_score: float,
    n_episodes: int,
    seed: int,
    zi_baseline_trade_accuracy: Optional[float] = None,
    zi_baseline_positive_pnl_frac: Optional[float] = None,
) -> List[dict]:
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
