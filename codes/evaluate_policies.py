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
from codes.rl_common import build_agent_sample_row, build_env_step_row, build_episode_record, set_global_seeds


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
) -> tuple[List[dict], List[dict], List[dict]]:
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
    sample_rows: List[dict] = []
    env_step_rows: List[dict] = []
    sample_episodes = {0}
    trader_meta = []
    for aid in agent_ids:
        agent = da.population.agents[aid]
        trader_type = agent.sigma_i / max(cfg.sentiment.sigma_chart, 1e-9)
        trader_meta.append((aid, trader_type))
    trader_meta.sort(key=lambda item: item[1])
    fundamentalist_id, fundamentalist_type = trader_meta[0]
    chartist_id, chartist_type = trader_meta[-1]
    mixed_id, mixed_type = min(trader_meta, key=lambda item: abs(item[1] - 0.5))
    sampled_agents = {
        fundamentalist_id: ("fundamentalista", fundamentalist_type),
        mixed_id: ("mieszany", mixed_type),
        chartist_id: ("chartista", chartist_type),
    }

    for episode in range(n_episodes):
        da.reset_episode()
        step_actions: List[np.ndarray] = []
        prev_positions = {aid: da.population.agents[aid].position for aid in agent_ids}
        sample_this_episode = episode in sample_episodes and seed == 0 and abs(diversity_score - 1.0) < 1e-9

        while not da.done:
            obs_by_agent = {}
            actions = {}
            positions_before = dict(prev_positions)
            eq_price_before = float(da.eq_price)
            ref_price_before = float(da.ref_price)
            public_gap = float(np.clip(
                (eq_price_before - ref_price_before) / max(cfg.sentiment.signal_scale, 1e-9),
                -1.0,
                1.0,
            ))
            realized_before = {
                aid: da.population.agents[aid].realized_pnl
                for aid in agent_ids
            }
            for aid in agent_ids:
                obs = da.get_observation(aid)
                obs_by_agent[aid] = obs
                actions[aid] = _action_for_policy(
                    algorithm_name,
                    policy,
                    obs,
                    aid,
                )
            step_actions.append(np.array([actions[aid] for aid in agent_ids], dtype=np.int32))
            da.execute_parallel_actions(actions)
            rewards, _ = da.compute_step_rewards()
            if sample_this_episode:
                public_gap_after = float(np.clip(
                    (da.eq_price - da.ref_price) / max(cfg.sentiment.signal_scale, 1e-9),
                    -1.0,
                    1.0,
                ))
                for aid in agent_ids:
                    agent = da.population.agents[aid]
                    obs = obs_by_agent[aid]
                    trader_type = agent.sigma_i / max(cfg.sentiment.sigma_chart, 1e-9)
                    if trader_type <= 0.33:
                        agent_type = "fundamentalista"
                    elif trader_type >= 0.67:
                        agent_type = "chartista"
                    else:
                        agent_type = "mieszany"
                    executed = agent.position != positions_before.get(aid, agent.position)
                    realized_pnl_this_step = float(agent.realized_pnl - realized_before[aid])
                    sample_rows.append(build_agent_sample_row(
                        algorithm=algorithm_name,
                        phase="eval",
                        diversity_score=diversity_score,
                        seed=seed,
                        episode=episode,
                        step=da._step,
                        agent_id=aid,
                        trader_type=trader_type,
                        agent_type=agent_type,
                        action=actions[aid],
                        action_name=cfg.env.action_name(actions[aid]),
                        executed=executed,
                        obs=obs,
                        public_gap_before=public_gap,
                        eq_price_before=eq_price_before,
                        ref_price_before=ref_price_before,
                        public_gap_after=public_gap_after,
                        eq_price_after=da.eq_price,
                        ref_price_after=da.ref_price,
                        position_before=positions_before.get(aid, agent.position),
                        position=agent.position,
                        entry_price_after=agent.entry_price,
                        reward_this_step=float(rewards.get(aid, 0.0)),
                        realized_pnl_this_step=realized_pnl_this_step,
                        realized_pnl_cum=agent.realized_pnl,
                        n_trades_closed=agent.n_trades_closed,
                        sentiment=agent.sentiment,
                        sigma_i=agent.sigma_i,
                        threshold=agent.threshold,
                    ))
                    prev_positions[aid] = agent.position
                mean_signal = float(np.mean([float(obs_by_agent[aid][0]) for aid in agent_ids]))
                std_signal = float(np.std([float(obs_by_agent[aid][0]) for aid in agent_ids]))
                mean_sigma = float(np.mean([float(da.population.agents[aid].sigma_i) for aid in agent_ids]))
                mean_position_before = float(np.mean([positions_before.get(aid, 0) for aid in agent_ids]))
                mean_position_after = float(np.mean([da.population.agents[aid].position for aid in agent_ids]))
                n_buy = sum(1 for aid in agent_ids if actions[aid] == cfg.env.ACTION_BUY_MARKET)
                n_sell = sum(1 for aid in agent_ids if actions[aid] == cfg.env.ACTION_SELL_MARKET)
                n_hold = sum(1 for aid in agent_ids if actions[aid] == cfg.env.ACTION_HOLD)
                realized_vals = [
                    float(da.population.agents[aid].realized_pnl - realized_before[aid])
                    for aid in agent_ids
                ]
                reward_vals = [float(rewards.get(aid, 0.0)) for aid in agent_ids]
                env_step_rows.append(build_env_step_row(
                    algorithm=algorithm_name,
                    phase="eval",
                    diversity_score=diversity_score,
                    seed=seed,
                    episode=episode,
                    step=da._step,
                    eq_price_before=eq_price_before,
                    ref_price_before=ref_price_before,
                    public_gap_before=public_gap,
                    eq_price_after=da.eq_price,
                    ref_price_after=da.ref_price,
                    public_gap_after=public_gap_after,
                    price_delta_step=da.ref_price - ref_price_before,
                    mean_signal=mean_signal,
                    std_signal=std_signal,
                    mean_sigma=mean_sigma,
                    mean_position_before=mean_position_before,
                    mean_position_after=mean_position_after,
                    n_buy=n_buy,
                    n_sell=n_sell,
                    n_hold=n_hold,
                    net_flow=n_buy - n_sell,
                    mean_reward=float(np.mean(reward_vals)),
                    mean_realized_pnl=float(np.mean(realized_vals)),
                    mean_mtm=float(np.mean([r - x for r, x in zip(reward_vals, realized_vals)])),
                    n_executed=sum(
                        1 for aid in agent_ids if da.population.agents[aid].position != positions_before.get(aid, 0)
                    ),
                    n_trades_closed_cum=sum(da.population.agents[aid].n_trades_closed for aid in agent_ids),
                ))

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

    return records, sample_rows, env_step_rows


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
