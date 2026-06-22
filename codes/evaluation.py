"""
Wspólne narzędzia ewaluacji polityk.

Ten moduł nie zna szczegółów treningu SARSA/PPO. Dostaje:
  - środowisko,
  - funkcję wybierającą akcję,
  - nazwę algorytmu/fazy,
  - opcjonalny builder dodatkowych pól rekordu.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np

from codes.config import HTMConfig
from codes.double_auction import DoubleAuction
from codes.rl_common import build_agent_sample_row, build_env_step_row, build_episode_record


ActionSelector = Callable[[str, np.ndarray], int]
ExtraBuilder = Callable[[dict, List[np.ndarray]], dict]
EpisodeEndCallback = Callable[[int, DoubleAuction], None]
StepCallback = Callable[[int, DoubleAuction, Dict[str, np.ndarray], Dict[str, int], Dict[str, float], Dict[str, int]], None]


def coordination_stats(step_actions: List[np.ndarray], n_actions: int) -> tuple[float, float]:
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


def evaluate_same_population(
    da: DoubleAuction,
    cfg: HTMConfig,
    diversity_score: float,
    seed: int,
    n_eval_episodes: int,
    algorithm_name: str,
    phase: str,
    action_selector: ActionSelector,
    extra_builder: Optional[ExtraBuilder] = None,
    collect_coordination: bool = False,
    episode_end_callback: Optional[EpisodeEndCallback] = None,
    step_callback: Optional[StepCallback] = None,
    debug_diversity_score: Optional[float] = 1.0,
) -> tuple[List[dict], List[dict], List[dict]]:
    agent_ids = list(da.population.agents.keys())
    agent_gammas = [da.population.agents[aid].gamma for aid in agent_ids]
    records: List[dict] = []
    sample_rows: List[dict] = []
    env_step_rows: List[dict] = []

    for episode in range(n_eval_episodes):
        da.reset_episode()
        prev_positions = {aid: da.population.agents[aid].position for aid in agent_ids}
        sample_this_episode = (
            debug_diversity_score is not None
            and seed == 0
            and episode == (n_eval_episodes - 1)
            and abs(float(diversity_score) - float(debug_diversity_score)) < 1e-9
        )
        step_actions: List[np.ndarray] = []

        while not da.done:
            obs_by_agent: Dict[str, np.ndarray] = {}
            actions: Dict[str, int] = {}
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
                actions[aid] = int(action_selector(aid, obs))

            if collect_coordination:
                step_actions.append(np.array([actions[aid] for aid in agent_ids], dtype=np.int32))

            da.execute_parallel_actions(actions)
            rewards, _ = da.compute_step_rewards()

            if step_callback is not None:
                step_callback(episode, da, obs_by_agent, actions, rewards, positions_before)

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
                    executed = agent.position != positions_before[aid]
                    realized_pnl_this_step = float(agent.realized_pnl - realized_before[aid])
                    sample_rows.append(build_agent_sample_row(
                        algorithm=algorithm_name,
                        phase=phase,
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
                        position_before=positions_before[aid],
                        position=agent.position,
                        entry_price_after=agent.entry_price,
                        reward_this_step=float(rewards.get(aid, 0.0)),
                        realized_pnl_this_step=realized_pnl_this_step,
                        realized_pnl_cum=agent.realized_pnl,
                        n_trades_closed=agent.n_trades_closed,
                        sigma_i=agent.sigma_i,
                    ))
                    prev_positions[aid] = agent.position

                mean_signal = float(np.mean([float(obs_by_agent[aid][0]) for aid in agent_ids]))
                std_signal = float(np.std([float(obs_by_agent[aid][0]) for aid in agent_ids]))
                mean_sigma = float(np.mean([float(da.population.agents[aid].sigma_i) for aid in agent_ids]))
                mean_position_before = float(np.mean([positions_before[aid] for aid in agent_ids]))
                mean_position_after = float(np.mean([da.population.agents[aid].position for aid in agent_ids]))
                position_changes = {
                    aid: da.population.agents[aid].position - positions_before[aid]
                    for aid in agent_ids
                }
                n_buy = sum(1 for aid, delta in position_changes.items() if delta > 0)
                n_sell = sum(1 for aid, delta in position_changes.items() if delta < 0)
                n_hold = sum(1 for aid in agent_ids if actions[aid] == cfg.env.ACTION_HOLD)
                net_flow_actual = int(sum(position_changes.values()))
                realized_vals = [float(da.population.agents[aid].realized_pnl - realized_before[aid]) for aid in agent_ids]
                reward_vals = [float(rewards.get(aid, 0.0)) for aid in agent_ids]
                env_step_rows.append(build_env_step_row(
                    algorithm=algorithm_name,
                    phase=phase,
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
                    net_flow=net_flow_actual,
                    mean_reward=float(np.mean(reward_vals)),
                    mean_realized_pnl=float(np.mean(realized_vals)),
                    mean_mtm=float(np.mean([r - x for r, x in zip(reward_vals, realized_vals)])),
                    n_executed=sum(1 for delta in position_changes.values() if delta != 0),
                    n_trades_closed_cum=sum(da.population.agents[aid].n_trades_closed for aid in agent_ids),
                ))

        metrics = da.episode_metrics()
        if episode_end_callback is not None:
            episode_end_callback(episode, da)
        extra = extra_builder(metrics, step_actions) if extra_builder is not None else {}
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
