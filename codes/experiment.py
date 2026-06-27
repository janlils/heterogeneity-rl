"""
Skonsolidowany pipeline eksperymentów HTM.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import logging
import os
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, TypeVar

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODULE_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(MODULE_ROOT / ".matplotlib_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(MODULE_ROOT / ".cache"))
(MODULE_ROOT / ".matplotlib_cache").mkdir(exist_ok=True)
(MODULE_ROOT / ".cache").mkdir(exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from codes.algorithms import (
    DeepSARSAMultiAgent,
    SignalRulePolicy,
    aggregate_agent_eval_episode_rows,
    build_agent_sample_row,
    build_env_step_row,
    build_episode_record,
    finalize_decision_feature_summary,
    init_decision_feature_stats,
    set_global_seeds,
    update_decision_feature_stats,
)
from codes.config import (
    HTMConfig,
    EnvConfig,
    LogConfig,
    build_ippo_benchmark_settings,
    build_ppo_benchmark_settings,
    build_sarsa_benchmark_settings,
    build_signal_rule_benchmark_settings,
)
from codes.market_env import DoubleAuction, ZeroIntelligenceAgent
from codes.results import (
    AGENT_EVAL_SUMMARY_FIELDS,
    AGENT_SAMPLE_FIELDS,
    DECISION_FEATURE_SUMMARY_FIELDS,
    ENV_STEP_FIELDS,
    EPISODE_FIELDS,
    append_rows,
    prepare_run_dir,
    write_run_config,
)
from codes.reporting import (
    plot_agent_eval_distribution as reporting_plot_agent_eval_distribution,
    plot_policy_learning_curves,
    plot_sarsa_final_comparison,
    plot_sarsa_learning_curves,
)

T = TypeVar("T")
R = TypeVar("R")
log = logging.getLogger("htm.experiment")

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

def _build_record(
    episode: int,
    diversity_score: float,
    seed: int,
    cfg: HTMConfig,
    metrics: dict,
    algorithm: str,
    zi_baseline_trade_accuracy: float,
    zi_baseline_positive_pnl_frac: float,
    extra: Optional[dict] = None,
    agent_gammas: Optional[Sequence[float]] = None,
) -> dict:
    trade_acc = metrics.get("trade_accuracy", 0.0)
    positive_pnl_frac = metrics.get("positive_pnl_frac", 0.0)
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
        "mean_pnl": metrics.get("mean_pnl", 0.0),
        "pnl_positive_frac": positive_pnl_frac,
        "trade_accuracy": trade_acc,
        "n_trades": metrics.get("n_trades", 0),
        "n_trades_closed": metrics.get("n_trades_closed", 0),
        "n_position_closes": metrics.get("n_position_closes", 0),
        "price_volatility": metrics.get("price_volatility", 0.0),
        "price_range": metrics.get("price_range", 0.0),
        "open_positions": metrics.get("open_positions_end", 0),
        "mean_abs_position": metrics.get("mean_abs_position", 0.0),
        "mean_value_gap": metrics.get("mean_value_gap", 0.0),
        "pct_chartists": metrics.get("pct_chartists", 0.0),
        "corr_type_pnl": metrics.get("corr_type_pnl", 0.0),
        "action_buy_frac": metrics.get("action_buy_frac", 0.0),
        "action_sell_frac": metrics.get("action_sell_frac", 0.0),
        "action_hold_frac": metrics.get("action_hold_frac", 0.0),
        "zi_baseline_trade_accuracy": zi_baseline_trade_accuracy,
        "zi_baseline_positive_pnl_frac": zi_baseline_positive_pnl_frac,
        "primary_metric": "trade_accuracy",
        "beats_zi": trade_acc > zi_baseline_trade_accuracy,
        "positive_pnl_frac": positive_pnl_frac,
        "zi_baseline": zi_baseline_trade_accuracy,
        "gini": metrics.get("gini_pnl", 0.0),
        "mean_terminal_pnl": metrics.get("mean_terminal_pnl", 0.0),
        "terminal_positive_frac": metrics.get("terminal_positive_frac", 0.0),
        "mean_total_pnl": metrics.get("mean_total_pnl", 0.0),
    }
    if extra:
        record.update(extra)
    return record

def _agent_diagnostic_rows(
    da: DoubleAuction,
    episode: int,
    diversity_score: float,
    seed: int,
) -> List[dict]:
    rows: List[dict] = []
    agent_metrics = da.agent_metrics()
    for aid, meta in agent_metrics.items():
        sigma_i = float(meta.get("sigma_i", 0.0))
        trader_type = sigma_i / max(da.cfg.sentiment.sigma_chart, 1e-9)
        rows.append({
            "episode": episode,
            "diversity_score": diversity_score,
            "seed": seed,
            "agent_id": aid,
            "trader_type": trader_type,
            "sigma_i": sigma_i,
            "gamma": float(meta.get("gamma", 0.0)),
            "realized_pnl": float(meta.get("realized_pnl", 0.0)),
            "n_trades_closed": int(meta.get("n_trades_closed", 0)),
            "trade_accuracy_agent": float(meta.get("trade_accuracy", 0.0)),
        })
    return rows

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

def evaluate_zi(
    cfg: HTMConfig,
    diversity_score: float,
    n_episodes: int,
    seed: int,
) -> tuple[List[dict], List[dict], List[dict]]:
    return evaluate_policy("ZI", None, cfg, diversity_score, n_episodes, seed)

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

def build_standard_eval_extra_builder(
    cfg: HTMConfig,
    *,
    zi_baseline_trade_accuracy: float,
    zi_baseline_positive_pnl_frac: float,
    eval_mode: Optional[str] = None,
    collect_coordination: bool = False,
) -> ExtraBuilder:
    def extra_builder(metrics: dict, step_actions: List[np.ndarray]) -> dict:
        extra = {
            "zi_baseline_trade_accuracy": zi_baseline_trade_accuracy,
            "zi_baseline_positive_pnl_frac": zi_baseline_positive_pnl_frac,
            "zi_baseline": zi_baseline_trade_accuracy,
            "beats_zi": metrics.get("trade_accuracy", 0.0) > zi_baseline_trade_accuracy,
        }
        if collect_coordination:
            same_action_frac, effective_n = coordination_stats(step_actions, cfg.env.n_actions)
            extra.update({
                "same_action_frac": same_action_frac,
                "effective_N": effective_n,
            })
        if eval_mode is not None:
            extra.update({
                "eval_mode": eval_mode,
                "eval_trade_accuracy": metrics.get("trade_accuracy", 0.0),
                "eval_mean_total_pnl": metrics.get("mean_total_pnl", 0.0),
                "eval_n_trades": metrics.get("n_trades", 0),
                "eval_mean_terminal_pnl": metrics.get("mean_terminal_pnl", 0.0),
            })
        return extra

    return extra_builder

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
    debug_diversity_score: Optional[float] = None,
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
            seed == 0
            and episode == (n_eval_episodes - 1)
            and (
                debug_diversity_score is None
                or abs(float(diversity_score) - float(debug_diversity_score)) < 1e-9
            )
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
                    exec_price=da._last_exec_price,
                    public_gap_after=public_gap_after,
                    price_delta_step=da.ref_price - ref_price_before,
                    sigma_step=da._last_step_sigma,
                    crisis_step=da._last_step_crisis,
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

def evaluate_same_population_with_diagnostics(
    *,
    da: DoubleAuction,
    cfg: HTMConfig,
    diversity_score: float,
    seed: int,
    n_eval_episodes: int,
    algorithm_name: str,
    action_selector: ActionSelector,
    extra_builder: Optional[ExtraBuilder] = None,
    collect_coordination: bool = False,
    debug_diversity_score: Optional[float] = None,
) -> tuple[List[dict], List[dict], List[dict], List[dict], List[dict]]:
    feature_stats = init_decision_feature_stats()
    alignment_counts = {aid: {"aligned": 0, "directional": 0, "total": 0} for aid in da.agent_ids}
    agent_eval_rows: List[dict] = []

    def step_callback(
        _episode: int,
        _da_eval: DoubleAuction,
        obs_by_agent: Dict[str, np.ndarray],
        actions: Dict[str, int],
        _rewards: Dict[str, float],
        _positions_before: Dict[str, int],
    ) -> None:
        for aid, obs in obs_by_agent.items():
            action = int(actions[aid])
            update_decision_feature_stats(feature_stats, obs, action)
            alignment_counts[aid]["total"] += 1
            if action in (cfg.env.ACTION_BUY_MARKET, cfg.env.ACTION_SELL_MARKET):
                alignment_counts[aid]["directional"] += 1
                signal = float(obs[0])
                if (signal > 0 and action == cfg.env.ACTION_BUY_MARKET) or (
                    signal < 0 and action == cfg.env.ACTION_SELL_MARKET
                ):
                    alignment_counts[aid]["aligned"] += 1

    def episode_end_callback(episode: int, da_eval: DoubleAuction) -> None:
        action_counts = {aid: [0, 0, 0] for aid in da_eval.agent_ids}
        for entry in da_eval._actions_log:
            aid = entry.get("agent_id")
            action = int(entry.get("action", 0))
            if aid in action_counts and 0 <= action <= 2:
                action_counts[aid][action] += 1
        for aid, meta in da_eval.agent_metrics().items():
            sigma_i = float(meta.get("sigma_i", 0.0))
            counts = action_counts.get(aid, [0, 0, 0])
            total_actions = max(sum(counts), 1)
            directional = max(alignment_counts[aid]["directional"], 1)
            agent_eval_rows.append({
                "algorithm": algorithm_name,
                "phase": "eval_same_population",
                "diversity_score": diversity_score,
                "seed": seed,
                "episode": episode,
                "agent_id": aid,
                "sigma_i": sigma_i,
                "trader_type": sigma_i / max(cfg.sentiment.sigma_chart, 1e-9),
                "realized_pnl": float(meta.get("realized_pnl", 0.0)),
                "trade_accuracy_agent": float(meta.get("trade_accuracy", 0.0)),
                "n_trades_closed": int(meta.get("n_trades_closed", 0)),
                "n_trades_won": int(meta.get("n_trades_won", 0)),
                "position_end": int(meta.get("position", 0)),
                "buy_frac": counts[cfg.env.ACTION_BUY_MARKET] / total_actions,
                "sell_frac": counts[cfg.env.ACTION_SELL_MARKET] / total_actions,
                "hold_frac": counts[cfg.env.ACTION_HOLD] / total_actions,
                "signal_alignment_rate": alignment_counts[aid]["aligned"] / directional,
                "directional_action_rate": alignment_counts[aid]["directional"] / max(alignment_counts[aid]["total"], 1),
            })

    records, sample_rows, env_step_rows = evaluate_same_population(
        da=da,
        cfg=cfg,
        diversity_score=diversity_score,
        seed=seed,
        n_eval_episodes=n_eval_episodes,
        algorithm_name=algorithm_name,
        phase="eval_same_population",
        action_selector=action_selector,
        extra_builder=extra_builder,
        collect_coordination=collect_coordination,
        episode_end_callback=episode_end_callback,
        step_callback=step_callback,
        debug_diversity_score=debug_diversity_score,
    )
    feature_rows = [finalize_decision_feature_summary(
        feature_stats,
        algorithm=algorithm_name,
        phase="eval_same_population",
        diversity_score=diversity_score,
        seed=seed,
    )]
    return records, sample_rows, aggregate_agent_eval_episode_rows(agent_eval_rows), feature_rows, env_step_rows

@dataclass
class RunArtifacts:
    run_id: str
    run_dir: Path
    episodes_csv: Path
    agents_sample_csv: Path
    env_steps_csv: Path
    agent_eval_summary_csv: Path
    decision_feature_summary_csv: Path
    run_config_path: Path

def setup_script_logger(name: str, log_path: Path, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger

def stamp_episode_rows(rows: List[dict], run_id: str, phase: str) -> List[dict]:
    return [{**row, "run_id": run_id, "phase": phase} for row in rows]

def stamp_rows(rows: List[dict], run_id: str) -> List[dict]:
    return [{**row, "run_id": run_id} for row in rows]

def configure_experiment_logger(artifacts: RunArtifacts, filename: str = "experiment.log") -> logging.Logger:
    return setup_script_logger("htm.experiment", artifacts.run_dir / filename)

def configure_worker_logger(filename: str) -> logging.Logger:
    return setup_script_logger("htm.experiment", PROJECT_ROOT / "logs" / filename)

def init_run_artifacts(run_tag: str, run_id: Optional[str], run_dir: Optional[str]) -> RunArtifacts:
    resolved_run_id, resolved_run_dir = prepare_run_dir(run_tag, run_id, run_dir)
    return RunArtifacts(
        run_id=resolved_run_id,
        run_dir=resolved_run_dir,
        episodes_csv=resolved_run_dir / "episodes.csv",
        agents_sample_csv=resolved_run_dir / "agents_sample.csv",
        env_steps_csv=resolved_run_dir / "env_steps.csv",
        agent_eval_summary_csv=resolved_run_dir / "agent_eval_summary.csv",
        decision_feature_summary_csv=resolved_run_dir / "decision_feature_summary.csv",
        run_config_path=resolved_run_dir / "run_config.json",
    )

def ensure_run_config(
    artifacts: RunArtifacts,
    run_tag: str,
    algorithm: str,
    settings: dict,
    cfg: HTMConfig,
    eval_new_population: bool,
) -> None:
    if artifacts.run_config_path.exists():
        return
    write_run_config(artifacts.run_config_path, {
        "run_id": artifacts.run_id,
        "run_tag": run_tag,
        "timestamp": (
            artifacts.run_id.split("_", 1)[1]
            if artifacts.run_id.startswith("run_")
            else artifacts.run_id
        ),
        "algorithm": algorithm,
        "diversity_scores": settings["diversity_scores"],
        "n_seeds": settings["n_seeds"],
        "n_episodes": settings["n_episodes"],
        "n_agents": cfg.env.n_agents,
        "market_condition": {
            "init_value": cfg.market.init_value,
            "alpha": cfg.market.alpha,
            "beta": cfg.market.beta,
            "impact_stress_gain": cfg.market.impact_stress_gain,
            "nu": cfg.market.nu,
            "garch_w": cfg.market.garch_w,
            "garch_a": cfg.market.garch_a,
            "garch_b": cfg.market.garch_b,
            "crisis_prob": cfg.market.crisis_prob,
        },
        "eval_new_population": eval_new_population,
    })

def compute_shared_zi_baseline(
    cfg: HTMConfig,
    diversity_scores: Sequence[float],
    zi_episodes: int,
    run_id: str,
    episodes_csv: Path,
    stamp_episode_rows: Callable[[List[dict], str, str], List[dict]],
    seed: int = 42,
) -> Tuple[List[dict], Dict[float, float], Dict[float, float]]:
    baseline_d = float(diversity_scores[0]) if diversity_scores else 0.0
    zi_records, _, _ = evaluate_zi(
        cfg,
        diversity_score=baseline_d,
        n_episodes=zi_episodes,
        seed=seed,
    )
    append_rows(episodes_csv, EPISODE_FIELDS, stamp_episode_rows(zi_records, run_id, "zi_baseline"))
    shared_zi_acc = float(np.mean([r["trade_accuracy"] for r in zi_records])) if zi_records else 0.0
    shared_zi_pos = float(np.mean([r["positive_pnl_frac"] for r in zi_records])) if zi_records else 0.0
    zi_acc = {float(d): shared_zi_acc for d in diversity_scores}
    zi_pos = {float(d): shared_zi_pos for d in diversity_scores}
    return zi_records, zi_acc, zi_pos

def worker_results(
    tasks: Sequence[T],
    worker_fn: Callable[[T], R],
    n_workers: int,
) -> Iterator[R]:
    if n_workers == 1:
        yield from map(worker_fn, tasks)
        return

    pool = Pool(processes=n_workers)
    try:
        yield from pool.imap_unordered(worker_fn, tasks)
    finally:
        pool.close()
        pool.join()

def append_experiment_outputs(
    *,
    artifacts: RunArtifacts,
    run_id: str,
    stamp_episode_rows: Callable[[List[dict], str, str], List[dict]],
    stamp_sample_rows: Callable[[List[dict], str], List[dict]],
    train_records: Optional[List[dict]] = None,
    eval_same_population_records: Optional[List[dict]] = None,
    eval_new_population_records: Optional[List[dict]] = None,
    sample_rows: Optional[List[dict]] = None,
    agent_eval_rows: Optional[List[dict]] = None,
    decision_feature_rows: Optional[List[dict]] = None,
    env_step_rows: Optional[List[dict]] = None,
) -> None:
    append_rows(
        artifacts.episodes_csv,
        EPISODE_FIELDS,
        stamp_episode_rows(train_records or [], run_id, "train"),
    )
    append_rows(
        artifacts.episodes_csv,
        EPISODE_FIELDS,
        stamp_episode_rows(eval_same_population_records or [], run_id, "eval_same_population"),
    )
    append_rows(
        artifacts.episodes_csv,
        EPISODE_FIELDS,
        stamp_episode_rows(eval_new_population_records or [], run_id, "eval_new_population"),
    )
    append_rows(artifacts.agents_sample_csv, AGENT_SAMPLE_FIELDS, stamp_sample_rows(sample_rows or [], run_id))
    append_rows(artifacts.agent_eval_summary_csv, AGENT_EVAL_SUMMARY_FIELDS, stamp_sample_rows(agent_eval_rows or [], run_id))
    append_rows(artifacts.decision_feature_summary_csv, DECISION_FEATURE_SUMMARY_FIELDS, stamp_sample_rows(decision_feature_rows or [], run_id))
    append_rows(artifacts.env_steps_csv, ENV_STEP_FIELDS, stamp_sample_rows(env_step_rows or [], run_id))

def run_eval_only_experiment(
    *,
    log,
    settings: dict,
    artifacts: RunArtifacts,
    worker_fn: Callable[[tuple], tuple],
    tasks: Sequence[tuple],
    stamp_episode_rows: Callable[[List[dict], str, str], List[dict]],
    stamp_sample_rows: Callable[[List[dict], str], List[dict]],
    algorithm_label: str,
) -> tuple[List[dict], List[dict]]:
    t0 = time.time()
    all_eval_same_population_records: List[dict] = []
    all_eval_new_population_records: List[dict] = []

    for i, worker_result in enumerate(worker_results(tasks, worker_fn, settings["n_workers"]), start=1):
        eval_records, eval_new_records, sample_rows, agent_eval_rows, decision_feature_rows, env_step_rows = worker_result
        all_eval_same_population_records.extend(eval_records)
        all_eval_new_population_records.extend(eval_new_records)
        append_experiment_outputs(
            artifacts=artifacts,
            run_id=artifacts.run_id,
            stamp_episode_rows=stamp_episode_rows,
            stamp_sample_rows=stamp_sample_rows,
            eval_same_population_records=eval_records,
            eval_new_population_records=eval_new_records,
            sample_rows=sample_rows,
            agent_eval_rows=agent_eval_rows,
            decision_feature_rows=decision_feature_rows,
            env_step_rows=env_step_rows,
        )
        d_done = eval_records[0]["diversity_score"] if eval_records else "?"
        seed_done = eval_records[0]["seed"] if eval_records else "?"
        log.info(
            f"  Zakończono {algorithm_label} D={d_done} seed={seed_done} "
            f"({i}/{len(tasks)}) | eval_same={len(all_eval_same_population_records)}"
        )

    log.info(f"Agent eval summary: {artifacts.agent_eval_summary_csv}")
    log.info(f"Decision feature summary: {artifacts.decision_feature_summary_csv}")
    log.info(f"Próbka agentów: {artifacts.agents_sample_csv}")
    log.info(f"Agregaty środowiska: {artifacts.env_steps_csv}")
    log.info(f"Czas: {time.time() - t0:.0f}s")
    return all_eval_same_population_records, all_eval_new_population_records

def run_sarsa_experiment(
    *,
    log,
    project_root: Path,
    args,
    settings: dict,
    cfg: HTMConfig,
    artifacts: RunArtifacts,
    worker_fn: Callable[[tuple], tuple],
    stamp_episode_rows: Callable[[List[dict], str, str], List[dict]],
    stamp_sample_rows: Callable[[List[dict], str], List[dict]],
    plot_learning_curves: Callable[..., None],
    plot_final_comparison: Callable[..., None],
    plot_agent_eval_distribution: Callable[..., None],
) -> None:
    import pandas as pd
    import numpy as np
    import time

    run_id = artifacts.run_id
    episodes_csv = artifacts.episodes_csv
    agents_sample_csv = artifacts.agents_sample_csv
    env_steps_csv = artifacts.env_steps_csv
    agent_eval_summary_csv = artifacts.agent_eval_summary_csv
    decision_feature_summary_csv = artifacts.decision_feature_summary_csv

    diversity_scores = settings["diversity_scores"]
    n_agents = settings["n_agents"]
    n_episodes = settings["n_episodes"]
    episode_steps = settings["episode_steps"]
    n_seeds = settings["n_seeds"]
    zi_episodes = settings["zi_episodes"]
    n_eval_episodes = settings["eval_episodes"]
    n_workers = settings["n_workers"]
    sarsa_cfg = settings["sarsa_cfg"]
    run_name = settings["run_name"]
    output_stem = "deep_sarsa" if run_name == "full" else f"deep_sarsa_{run_name}"

    log.info("=" * 65)
    log.info("HTM Benchmark — Deep SARSA (model spekulacyjny)")
    log.info(f"Tryb: {run_name}")
    log.info(f"N={n_agents} | D={diversity_scores} | ep={n_episodes} | steps/ep={episode_steps} | seeds={n_seeds}")
    log.info(f"Łączne kroki per agent per D: {n_episodes}×{episode_steps}={n_episodes*episode_steps}")
    log.info(
        f"SARSA epsilon: start={sarsa_cfg.epsilon_start:.3f} | "
        f"end={sarsa_cfg.epsilon_end:.3f} | decay={sarsa_cfg.epsilon_decay:.3f}"
    )
    log.info(
        f"Market: v2 | alpha={cfg.market.alpha:.3f} | beta={cfg.market.beta:.3f} | "
        f"impact_gain={cfg.market.impact_stress_gain:.2f} | crisis_prob={cfg.market.crisis_prob:.3f}"
    )
    log.info("=" * 65)

    for d in ["logs", "plots", "results", "experiments"]:
        (project_root / d).mkdir(exist_ok=True)

    ensure_run_config(
        artifacts=artifacts,
        run_tag=args.run_tag,
        algorithm="DeepSARSA",
        settings=settings,
        cfg=cfg,
        eval_new_population=args.eval_new_population,
    )

    log.info(cfg.summary())
    log.info(f"episode_steps={cfg.env.episode_steps} | n_actions={cfg.env.n_actions}")
    log.info("\n--- Liczę wspólny ZI baseline ---")
    baseline_d = float(diversity_scores[0]) if diversity_scores else 0.0
    zi_records, zi_baselines, zi_positive_baselines = compute_shared_zi_baseline(
        cfg=cfg,
        diversity_scores=diversity_scores,
        zi_episodes=zi_episodes,
        run_id=run_id,
        episodes_csv=episodes_csv,
        stamp_episode_rows=stamp_episode_rows,
        seed=42,
    )
    shared_zi_acc = zi_baselines[baseline_d] if diversity_scores else 0.0
    shared_zi_positive = zi_positive_baselines[baseline_d] if diversity_scores else 0.0
    zi_pnl = float(np.mean([r["mean_pnl"] for r in zi_records])) if zi_records else 0.0
    zi_term = float(np.mean([r["mean_terminal_pnl"] for r in zi_records])) if zi_records else 0.0
    log.info(
        f"  ZI | D_ref={baseline_d:.1f} | eff={shared_zi_positive:.3f} | "
        f"acc={shared_zi_acc:.3f} | pnl={zi_pnl:.4f} | term={zi_term:.4f}"
    )

    log.info(f"\n--- Start treningu ({n_workers} równoległych procesów) ---")
    t_total = time.time()
    tasks = [
        (
            d, seed, cfg, zi_baselines[d], zi_positive_baselines[d],
            n_episodes, n_eval_episodes, sarsa_cfg, settings["log_every"], args.eval_new_population,
        )
        for d in diversity_scores
        for seed in range(n_seeds)
    ]
    n_tasks = len(tasks)
    log.info(f"Łącznie zadań: {n_tasks} ({len(diversity_scores)} D × {n_seeds} seeds)")

    all_records: List[dict] = []
    all_eval_same_population_records: List[dict] = []
    all_eval_new_population_records: List[dict] = []
    all_agent_eval_rows: List[dict] = []

    for i, worker_result in enumerate(worker_results(tasks, worker_fn, n_workers)):
        task_records, eval_same_population_records, eval_new_population_records, sample_rows, agent_eval_rows, decision_feature_rows, env_step_rows = worker_result
        all_records.extend(task_records)
        all_eval_same_population_records.extend(eval_same_population_records)
        all_eval_new_population_records.extend(eval_new_population_records)
        all_agent_eval_rows.extend(agent_eval_rows)
        append_experiment_outputs(
            artifacts=artifacts,
            run_id=run_id,
            stamp_episode_rows=stamp_episode_rows,
            stamp_sample_rows=stamp_sample_rows,
            train_records=task_records,
            eval_same_population_records=eval_same_population_records,
            eval_new_population_records=eval_new_population_records,
            sample_rows=sample_rows,
            agent_eval_rows=agent_eval_rows,
            decision_feature_rows=decision_feature_rows,
            env_step_rows=env_step_rows,
        )
        d_done = task_records[0]["diversity_score"] if task_records else "?"
        s_done = task_records[0]["seed"] if task_records else "?"
        log.info(
            f"  Zakończono: D={d_done} seed={s_done} "
            f"({i+1}/{n_tasks}) | train: {len(all_records)} | eval_same: {len(all_eval_same_population_records)}"
        )

    log.info(f"Trening zakończony — {len(all_records)} rekordów")
    df = pd.DataFrame(all_records)
    eval_df = pd.DataFrame(all_eval_same_population_records)
    eval_new_df = pd.DataFrame(all_eval_new_population_records)
    total_rows = len(all_records) + len(all_eval_same_population_records) + len(all_eval_new_population_records) + len(zi_records)
    log.info(f"\nWyniki: {episodes_csv} ({total_rows} wierszy łącznie z fazami)")

    try:
        plot_learning_curves(
            all_records, zi_baselines,
            project_root / "plots" / f"{output_stem}_learning_curves.png",
            diversity_scores=diversity_scores,
            rolling_window=settings["rolling_window"],
            n_agents=n_agents,
        )
    except Exception as exc:
        log.exception(f"Plot learning curves failed: {exc}")
    try:
        plot_final_comparison(
            all_records, zi_baselines,
            project_root / "plots" / f"{output_stem}_final_comparison.png",
            diversity_scores=diversity_scores,
            n_episodes=n_episodes,
        )
    except Exception as exc:
        log.exception(f"Plot final comparison failed: {exc}")
    try:
        plot_agent_eval_distribution(
            all_agent_eval_rows,
            project_root / "plots" / f"{output_stem}_agent_eval_distribution.png",
            diversity_scores=diversity_scores,
        )
    except Exception as exc:
        log.exception(f"Plot agent eval distribution failed: {exc}")

    final_window = min(50, max(1, n_episodes // 3))
    log.info("\n" + "=" * 82)
    log.info(f"PODSUMOWANIE — ostatnie {final_window} epizodów, uśrednione po seedach")
    log.info(
        f"{'D':>5} | {'acc':>7} | {'ZI acc':>7} | {'pnl_tot':>9} | "
        f"{'term':>8} | {'Trades':>7} | {'Closed':>7}"
    )
    log.info("-" * 82)
    final = df[df["episode"] >= n_episodes - final_window]
    for d in diversity_scores:
        sub = final[final["diversity_score"] == d]
        if sub.empty:
            continue
        s_pnl = sub["mean_total_pnl"].mean()
        term_pnl = sub["mean_terminal_pnl"].mean()
        tacc = sub["trade_accuracy"].mean()
        zi = zi_baselines.get(d, 0.0)
        trades = sub["n_trades"].mean()
        closed = sub["n_trades_closed"].mean()
        sign = "↑" if tacc > zi else "↓"
        log.info(
            f"{d:>5.1f} | {tacc:>7.3f}{sign} | {zi:>7.3f} | "
            f"{s_pnl:>9.4f} | {term_pnl:>8.4f} | {trades:>7.1f} | {closed:>7.1f}"
        )

    if not eval_df.empty:
        log.info("")
        log.info("EWALUACJA SARSA — ta sama populacja treningowa, reset_episode(), explore=False")
        log.info(
            f"{'D':>5} | {'eval acc':>8} | {'ZI acc':>7} | {'eval pnl':>9} | "
            f"{'term':>8} | {'Trades':>7} | {'Closed':>7}"
        )
        log.info("-" * 76)
        for d in diversity_scores:
            sub = eval_df[eval_df["diversity_score"] == d]
            if sub.empty:
                continue
            eval_acc = sub["trade_accuracy"].mean()
            eval_pnl = sub["mean_total_pnl"].mean()
            eval_term = sub["mean_terminal_pnl"].mean()
            eval_trades = sub["n_trades"].mean()
            eval_closed = sub["n_trades_closed"].mean()
            zi = zi_baselines.get(d, 0.0)
            sign = "↑" if eval_acc > zi else "↓"
            log.info(
                f"{d:>5.1f} | {eval_acc:>7.3f}{sign} | {zi:>7.3f} | "
                f"{eval_pnl:>9.4f} | {eval_term:>8.4f} | "
                f"{eval_trades:>7.1f} | {eval_closed:>7.1f}"
            )

    if not eval_new_df.empty:
        log.info("")
        log.info("EWALUACJA SARSA — nowa populacja, seed+1000, explore=False")
        log.info(
            f"{'D':>5} | {'eval acc':>8} | {'ZI acc':>7} | {'eval pnl':>9} | "
            f"{'term':>8} | {'Trades':>7} | {'Closed':>7}"
        )
        log.info("-" * 76)
        for d in diversity_scores:
            sub = eval_new_df[eval_new_df["diversity_score"] == d]
            if sub.empty:
                continue
            eval_acc = sub["trade_accuracy"].mean()
            eval_pnl = sub["mean_total_pnl"].mean()
            eval_term = sub["mean_terminal_pnl"].mean()
            eval_trades = sub["n_trades"].mean()
            eval_closed = sub["n_trades_closed"].mean()
            zi = zi_baselines.get(d, 0.0)
            sign = "↑" if eval_acc > zi else "↓"
            log.info(
                f"{d:>5.1f} | {eval_acc:>7.3f}{sign} | {zi:>7.3f} | "
                f"{eval_pnl:>9.4f} | {eval_term:>8.4f} | "
                f"{eval_trades:>7.1f} | {eval_closed:>7.1f}"
            )

    log.info(f"\nCałkowity czas: {time.time()-t_total:.0f}s")
    log.info(f"Wykresy: plots/{output_stem}_learning_curves.png")
    log.info(f"         plots/{output_stem}_final_comparison.png")
    log.info(f"         plots/{output_stem}_agent_eval_distribution.png")
    log.info(f"Dane:    {episodes_csv.relative_to(project_root)}")
    log.info(f"Agenty:  {agent_eval_summary_csv.relative_to(project_root)}")
    log.info(f"Decyzje: {decision_feature_summary_csv.relative_to(project_root)}")
    log.info(f"Próbka:  {agents_sample_csv.relative_to(project_root)}")
    log.info(f"Środow.: {env_steps_csv.relative_to(project_root)}")

def run_ppo_experiment(
    *,
    log,
    project_root: Path,
    args,
    settings: dict,
    cfg: HTMConfig,
    artifacts: RunArtifacts,
    worker_fn: Callable[[tuple], tuple],
    stamp_episode_rows: Callable[[List[dict], str, str], List[dict]],
    stamp_sample_rows: Callable[[List[dict], str], List[dict]],
    plot_learning_curves: Callable[..., None],
    log_final_summary: Callable[..., None],
    algorithm_label: str = "PPO",
    artifact_stem: Optional[str] = None,
) -> None:
    import time

    run_id = artifacts.run_id
    episodes_csv = artifacts.episodes_csv
    agents_sample_csv = artifacts.agents_sample_csv
    env_steps_csv = artifacts.env_steps_csv
    agent_eval_summary_csv = artifacts.agent_eval_summary_csv
    decision_feature_summary_csv = artifacts.decision_feature_summary_csv

    run_name = artifact_stem or ("ppo_quick" if args.quick else "ppo")
    log.info("=" * 70)
    log.info(
        f"{algorithm_label} | {run_name} | N={cfg.env.n_agents} | D={settings['diversity_scores']} | "
        f"ep={settings['n_episodes']} | steps={cfg.env.episode_steps} | "
        f"seeds={settings['n_seeds']} | workers={settings['n_workers']} | "
        f"agent_id_features={cfg.ppo.use_agent_id_features}"
    )
    log.info("=" * 70)

    results_dir = project_root / "results"
    plots_dir = project_root / "plots"
    checkpoint_dir = results_dir / "checkpoints" / algorithm_label.lower() / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    ensure_run_config(
        artifacts=artifacts,
        run_tag=args.run_tag,
        algorithm=algorithm_label,
        settings=settings,
        cfg=cfg,
        eval_new_population=args.eval_new_population,
    )

    zi_records, zi_acc, zi_pos = compute_shared_zi_baseline(
        cfg=cfg,
        diversity_scores=settings["diversity_scores"],
        zi_episodes=settings["zi_episodes"],
        run_id=run_id,
        episodes_csv=episodes_csv,
        stamp_episode_rows=stamp_episode_rows,
        seed=42,
    )

    all_records: List[dict] = []
    all_eval_same_population_records: List[dict] = []
    all_eval_new_population_records: List[dict] = []
    checkpoints: List[Path] = []
    t0 = time.time()

    tasks = [
        (
            d,
            seed,
            cfg,
            zi_acc[d],
            zi_pos[d],
            settings["n_episodes"],
            checkpoint_dir,
            settings["log_every"],
            args.eval_new_population,
        )
        for d in settings["diversity_scores"]
        for seed in range(settings["n_seeds"])
    ]
    log.info(
        f"Start {algorithm_label}: {len(tasks)} zadań "
        f"({len(settings['diversity_scores'])} D × {settings['n_seeds']} seeds)"
    )

    for i, worker_result in enumerate(worker_results(tasks, worker_fn, settings["n_workers"]), start=1):
        train_records, _learning_curve_records, _agent_diagnostics, eval_records, eval_new_population_records, sample_rows, agent_eval_rows, decision_feature_rows, env_step_rows, ckpt = worker_result
        all_records.extend(train_records)
        all_eval_same_population_records.extend(eval_records)
        all_eval_new_population_records.extend(eval_new_population_records)
        checkpoints.append(ckpt)
        append_experiment_outputs(
            artifacts=artifacts,
            run_id=run_id,
            stamp_episode_rows=stamp_episode_rows,
            stamp_sample_rows=stamp_sample_rows,
            train_records=train_records,
            eval_same_population_records=eval_records,
            eval_new_population_records=eval_new_population_records,
            sample_rows=sample_rows,
            agent_eval_rows=agent_eval_rows,
            decision_feature_rows=decision_feature_rows,
            env_step_rows=env_step_rows,
        )

        d_done = train_records[0]["diversity_score"] if train_records else "?"
        seed_done = train_records[0]["seed"] if train_records else "?"
        log.info(
            f"  Zakończono D={d_done} seed={seed_done} "
            f"({i}/{len(tasks)}) | train={len(all_records)} "
            f"eval_same={len(all_eval_same_population_records)} ckpt={ckpt.name}"
        )

    try:
        plot_learning_curves(
            all_records,
            zi_acc,
            plots_dir / f"{run_name}_learning_curves.png",
            settings["diversity_scores"],
            settings["rolling_window"],
        )
    except Exception as exc:
        log.exception(f"Plot learning curves failed: {exc}")
    try:
        log_final_summary(
            all_records,
            all_eval_same_population_records,
            all_eval_new_population_records,
            zi_acc,
            settings["diversity_scores"],
            settings["n_episodes"],
        )
    except Exception as exc:
        log.exception(f"Final summary/plotting failed: {exc}")

    total_rows = len(all_records) + len(all_eval_same_population_records) + len(all_eval_new_population_records) + len(zi_records)
    log.info(f"Wyniki: {episodes_csv} ({total_rows} wierszy)")
    log.info(f"Agent eval summary: {agent_eval_summary_csv}")
    log.info(f"Decision feature summary: {decision_feature_summary_csv}")
    log.info(f"Próbka agentów: {agents_sample_csv}")
    log.info(f"Agregaty środowiska: {env_steps_csv}")
    log.info(f"Checkpointy: {checkpoint_dir}")
    log.info(f"Czas: {time.time() - t0:.0f}s")

def evaluate_trained_sarsa_same_population(
    da: DoubleAuction,
    sarsa: DeepSARSAMultiAgent,
    diversity_score: float,
    seed: int,
    cfg: HTMConfig,
    n_eval_episodes: int,
    zi_baseline_trade_accuracy: float,
    zi_baseline_positive_pnl_frac: float,
) -> tuple[List[dict], List[dict], List[dict], List[dict], List[dict]]:
    def action_selector(aid: str, obs: np.ndarray) -> int:
        return int(sarsa.agents[aid].act(obs, explore=False))

    extra_builder = build_standard_eval_extra_builder(
        cfg,
        zi_baseline_trade_accuracy=zi_baseline_trade_accuracy,
        zi_baseline_positive_pnl_frac=zi_baseline_positive_pnl_frac,
    )

    records, sample_rows, agent_eval_rows, feature_rows, env_step_rows = evaluate_same_population_with_diagnostics(
        da=da,
        cfg=cfg,
        diversity_score=diversity_score,
        seed=seed,
        n_eval_episodes=n_eval_episodes,
        algorithm_name="DeepSARSA_EVAL_SAME_POPULATION",
        action_selector=action_selector,
        extra_builder=extra_builder,
        collect_coordination=False,
    )
    return records, sample_rows, agent_eval_rows, feature_rows, env_step_rows

def run_sarsa_training(
    diversity_score: float,
    n_episodes:      int,
    seed:            int,
    cfg:             HTMConfig,
    zi_baseline_trade_accuracy: float,
    zi_baseline_positive_pnl_frac: float,
    sarsa_cfg:       DeepSARSAConfig,
    log_every:       int,
 ) -> tuple[List[dict], List[dict], List[dict], DeepSARSAMultiAgent, DoubleAuction]:
    """
    Trenuje Deep SARSA — Continuous Trading.

    Jeden epizod = T kroków z cfg.env.episode_steps
    (wszyscy agenci aktywni przez cały czas).
    Między epizodami: portfele resetowane, wyceny dryfują (pamięć rynku).
    Gamma jest istotna w każdym kroku (done=True dopiero po T krokach).
    """
    da = DoubleAuction(cfg, seed=seed)
    da.reset(diversity_score=diversity_score, seed=seed)

    agent_ids    = list(da.population.agents.keys())
    agent_gammas = np.array(
        [da.population.agents[aid].gamma for aid in agent_ids],
        dtype=np.float64,
    )

    gammas_pop = [da.population.agents[a].gamma for a in agent_ids]
    log.info(
        f"  Populacja | N={len(agent_ids)} | "
        f"gamma=[{min(gammas_pop):.2f}, {max(gammas_pop):.2f}] mean={np.mean(gammas_pop):.3f}"
    )

    sarsa = DeepSARSAMultiAgent(
        agent_ids=agent_ids, agent_gammas=agent_gammas,
        n_obs=cfg.env.n_obs, n_actions=cfg.env.n_actions,
        cfg=sarsa_cfg, seed=seed,
    )

    # Parametry populacji — stałe przez cały run, zapisywane do każdego rekordu
    _ap = da.population.agents
    pop_meta = {
        "pop_mean_gamma":         float(np.mean([_ap[a].gamma         for a in agent_ids])),
        "pop_std_gamma":          float(np.std( [_ap[a].gamma         for a in agent_ids])),
        "pop_mean_sigma":         float(np.mean([_ap[a].sigma_i       for a in agent_ids])),
        "pop_std_sigma":          float(np.std( [_ap[a].sigma_i       for a in agent_ids])),
        "pop_mean_max_position":  float(np.mean([_ap[a].max_position  for a in agent_ids])),
    }

    records  = []
    learning_curve_records: List[dict] = []
    agent_diagnostics: List[dict] = []
    t_start  = time.time()

    n_step = sarsa_cfg.n_step
    traj_buffers: Dict[str, deque] = {aid: deque() for aid in agent_ids}

    for episode in range(n_episodes):

        # Nowy epizod: reset portfeli, wyceny dryfują, cena zostaje
        da.reset_episode()
        ep_rewards = {aid: 0.0 for aid in agent_ids}
        for aid in agent_ids:
            traj_buffers[aid].clear()

        while not da.done:
            # Parallel execution: wszyscy obserwują ten sam P_t.
            obs_at_action: Dict[str, np.ndarray] = {}
            actions_taken: Dict[str, int]        = {}
            obs_batch = da.get_observations(agent_ids)

            for i, aid in enumerate(agent_ids):
                obs = obs_batch[i]
                obs_at_action[aid] = obs
                actions_taken[aid] = sarsa.agents[aid].act(obs, explore=True)

            da.execute_parallel_actions(actions_taken)

            # Nagrody na końcu kroku (po wszystkich agentach)
            rewards, dones = da.compute_step_rewards()
            episode_done = any(dones.values())
            next_obs_batch = None if episode_done else da.get_observations(agent_ids)
            next_obs_map = None if next_obs_batch is None else {
                aid: next_obs_batch[i] for i, aid in enumerate(agent_ids)
            }

            for aid in agent_ids:
                r = rewards.get(aid, 0.0)
                ep_rewards[aid] += r

                traj_buffers[aid].append(
                    (obs_at_action[aid], actions_taken[aid], r)
                )

                if len(traj_buffers[aid]) >= n_step:
                    agent = sarsa.agents[aid]
                    obs_0, action_0, _ = traj_buffers[aid][0]

                    G = sum(
                        agent.gamma ** k * agent._scaled_reward(traj_buffers[aid][k][2])
                        for k in range(n_step)
                    )
                    if not episode_done:
                        next_obs = next_obs_map[aid]
                        G += agent.gamma ** n_step * agent.expected_next_q(next_obs)

                    agent.update_with_target(obs_0, action_0, G)
                    traj_buffers[aid].popleft()

        for aid in agent_ids:
            buf = list(traj_buffers[aid])
            agent = sarsa.agents[aid]
            for i in range(len(buf)):
                obs_i, action_i, _ = buf[i]
                G = sum(
                    agent.gamma ** k * agent._scaled_reward(buf[i + k][2])
                    for k in range(len(buf) - i)
                )
                agent.update_with_target(obs_i, action_i, G)
            traj_buffers[aid].clear()

        # Metryki epizodu
        m     = da.episode_metrics()
        pop_s = sarsa.population_stats()
        sarsa.end_episode()

        trade_acc      = m.get("trade_accuracy", 0.0)

        record = _build_record(
            episode=episode,
            diversity_score=diversity_score,
            seed=seed,
            cfg=cfg,
            metrics=m,
            algorithm="DeepSARSA_CT",
            zi_baseline_trade_accuracy=zi_baseline_trade_accuracy,
            zi_baseline_positive_pnl_frac=zi_baseline_positive_pnl_frac,
            extra={
            "mean_reward":       float(np.mean(list(ep_rewards.values()))),
            "mean_epsilon":      pop_s["mean_epsilon"],
            "mean_td_error":     pop_s["mean_td_error"],
            "mean_grad_norm":    pop_s.get("mean_grad_norm", 0.0),
            **pop_meta,
            },
            agent_gammas=agent_gammas,
        )
        records.append(record)

        if (episode + 1) % 50 == 0:
            agent_diagnostics.extend(
                _agent_diagnostic_rows(da, episode + 1, diversity_score, seed)
            )
            short_eval_records, _, _, _, _ = evaluate_trained_sarsa(
                da=da,
                sarsa=sarsa,
                diversity_score=diversity_score,
                seed=seed + 1000,
                cfg=cfg,
                n_eval_episodes=5,
                zi_baseline_trade_accuracy=zi_baseline_trade_accuracy,
                zi_baseline_positive_pnl_frac=zi_baseline_positive_pnl_frac,
            )
            learning_curve_records.append({
                "episode": episode + 1,
                "diversity_score": diversity_score,
                "seed": seed,
                "train_trade_accuracy": m.get("trade_accuracy", 0.0),
                "eval_trade_accuracy": float(np.mean([r["trade_accuracy"] for r in short_eval_records])),
                "mean_td_error": pop_s["mean_td_error"],
                "gamma_std": float(np.std(agent_gammas)),
            })

        if (episode + 1) % log_every == 0:
            recent     = records[-log_every:]
            r_tacc  = np.mean([r["trade_accuracy"]    for r in recent])
            r_pnl   = np.mean([r["mean_pnl"]          for r in recent])
            r_td    = np.mean([r["mean_td_error"]      for r in recent])
            elapsed = time.time() - t_start
            r_term = np.mean([r["mean_terminal_pnl"] for r in recent])
            log.info(
                f"  [D={diversity_score:.1f} s={seed}] "
                f"ep={episode+1:4d}/{n_episodes} | "
                f"acc={r_tacc:.3f} | "
                f"pnl={r_pnl:.4f} | "
                f"term={r_term:.4f} | "
                f"eps={pop_s['mean_epsilon']:.3f} | "
                f"td={r_td:.5f} | "
                f"t={elapsed:.0f}s"
            )

    return records, learning_curve_records, agent_diagnostics, sarsa, da

def evaluate_trained_sarsa(
    da: DoubleAuction,
    sarsa: DeepSARSAMultiAgent,
    diversity_score: float,
    seed: int,
    cfg: HTMConfig,
    n_eval_episodes: int,
    zi_baseline_trade_accuracy: float,
    zi_baseline_positive_pnl_frac: float,
    same_population: bool = True,
) -> tuple[List[dict], List[dict], List[dict], List[dict], List[dict]]:
    """
    Ewaluacja wytrenowanej polityki z końcowym epsilonem i bez update'ów sieci.
    Używa tego samego równoległego protokołu kroku co trening, ale na
    osobnej populacji ewaluacyjnej z seedem przesuniętym względem treningu.
    """
    if same_population:
        return evaluate_trained_sarsa_same_population(
            da=da,
            sarsa=sarsa,
            diversity_score=diversity_score,
            seed=seed,
            cfg=cfg,
            n_eval_episodes=n_eval_episodes,
            zi_baseline_trade_accuracy=zi_baseline_trade_accuracy,
            zi_baseline_positive_pnl_frac=zi_baseline_positive_pnl_frac,
        )

    eval_seed = seed if seed >= 1000 else seed + 1000
    records, sample_rows, env_step_rows = evaluate_sarsa(
        sarsa,
        cfg,
        diversity_score,
        n_eval_episodes,
        eval_seed,
        zi_baseline_trade_accuracy=zi_baseline_trade_accuracy,
        zi_baseline_positive_pnl_frac=zi_baseline_positive_pnl_frac,
    )
    return records, sample_rows, [], [], env_step_rows

def _sarsa_train_worker(args: tuple) -> list:
    """
    Uruchamia run_training dla jednej kombinacji (D, seed).
    Musi być funkcją modułu (nie lambda) żeby pickle działał.
    Każdy worker ma własny proces — brak konfliktów między sieciami.
    """
    # numpy-only — brak zależności PyTorch, nic do konfigurowania
    (
        diversity_score, seed, cfg, zi_baseline_trade_accuracy,
        zi_baseline_positive_pnl_frac, n_episodes, n_eval_episodes,
        sarsa_cfg, log_every, eval_new_population,
    ) = args
    global log
    log = configure_worker_logger(f"sarsa_worker_D{diversity_score:.1f}_seed{seed}.log")
    train_records, _learning_curve_records, _agent_diagnostics, sarsa, da = run_sarsa_training(
        diversity_score = diversity_score,
        n_episodes      = n_episodes,
        seed            = seed,
        cfg             = cfg,
        zi_baseline_trade_accuracy = zi_baseline_trade_accuracy,
        zi_baseline_positive_pnl_frac = zi_baseline_positive_pnl_frac,
        sarsa_cfg       = sarsa_cfg,
        log_every       = log_every,
    )
    eval_same_population_records, sample_rows, agent_eval_rows, decision_feature_rows, env_step_rows = evaluate_trained_sarsa(
        da=da,
        sarsa=sarsa,
        diversity_score=diversity_score,
        seed=seed,
        cfg=cfg,
        n_eval_episodes=n_eval_episodes,
        zi_baseline_trade_accuracy=zi_baseline_trade_accuracy,
        zi_baseline_positive_pnl_frac=zi_baseline_positive_pnl_frac,
        same_population=True,
    )
    eval_new_population_records: List[dict] = []
    if eval_new_population:
        eval_new_population_records, _, _, _, _ = evaluate_trained_sarsa(
            da=da,
            sarsa=sarsa,
            diversity_score=diversity_score,
            seed=seed + 1000,
            cfg=cfg,
            n_eval_episodes=n_eval_episodes,
            zi_baseline_trade_accuracy=zi_baseline_trade_accuracy,
            zi_baseline_positive_pnl_frac=zi_baseline_positive_pnl_frac,
            same_population=False,
        )
    return train_records, eval_same_population_records, eval_new_population_records, sample_rows, agent_eval_rows, decision_feature_rows, env_step_rows

def parse_sarsa_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trening Deep SARSA dla HTM.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Szybki smoke trening: mniej D, seedów, epizodów i kroków.",
    )
    parser.add_argument("--episodes", type=int, help="Nadpisz liczbę epizodów treningu.")
    parser.add_argument("--steps", type=int, help="Nadpisz liczbę kroków w epizodzie.")
    parser.add_argument("--seeds", type=int, help="Nadpisz liczbę seedów.")
    parser.add_argument("--agents", type=int, help="Nadpisz liczbę agentów.")
    parser.add_argument("--zi-episodes", type=int, help="Nadpisz liczbę epizodów ZI baseline.")
    parser.add_argument("--eval-episodes", type=int, help="Nadpisz liczbę epizodów ewaluacji bez eksploracji.")
    parser.add_argument("--workers", type=int, help="Nadpisz liczbę workerów.")
    parser.add_argument(
        "--eval-new-population",
        action="store_true",
        help="Dodatkowo uruchom osobny eval na nowej populacji z seedem przesuniętym o 1000.",
    )
    parser.add_argument("--run-tag", type=str, default="run", help="Krótki tag do nazwy folderu run.")
    parser.add_argument("--run-id", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=str, help=argparse.SUPPRESS)
    return parser.parse_args(argv)

def build_sarsa_run_settings(args: argparse.Namespace) -> dict:
    settings = build_sarsa_benchmark_settings(args.quick, default_workers=cpu_count())

    if args.episodes is not None:
        settings["n_episodes"] = args.episodes
    if args.steps is not None:
        settings["episode_steps"] = args.steps
    if args.seeds is not None:
        settings["n_seeds"] = args.seeds
    if args.agents is not None:
        settings["n_agents"] = args.agents
    if args.zi_episodes is not None:
        settings["zi_episodes"] = args.zi_episodes
    if args.eval_episodes is not None:
        settings["eval_episodes"] = args.eval_episodes
    if args.workers is not None:
        settings["n_workers"] = args.workers

    settings["n_episodes"] = max(1, settings["n_episodes"])
    settings["episode_steps"] = max(1, settings["episode_steps"])
    settings["n_seeds"] = max(1, settings["n_seeds"])
    settings["n_agents"] = max(1, settings["n_agents"])
    settings["zi_episodes"] = max(1, settings["zi_episodes"])
    settings["eval_episodes"] = max(1, settings["eval_episodes"])
    max_workers = settings["n_seeds"] * len(settings["diversity_scores"])
    settings["n_workers"] = max(1, min(settings["n_workers"], max_workers))
    settings["log_every"] = max(1, min(settings["log_every"], settings["n_episodes"]))
    settings["rolling_window"] = max(1, min(settings["rolling_window"], settings["n_episodes"]))
    return settings

def plot_learning_curves(
    all_records:  List[dict],
    zi_baselines: Dict[float, float],
    save_path:    Path,
    diversity_scores: List[float],
    rolling_window: int,
    n_agents: int,
) -> None:
    plot_sarsa_learning_curves(
        all_records=all_records,
        zi_baselines=zi_baselines,
        save_path=save_path,
        diversity_scores=diversity_scores,
        rolling_window=rolling_window,
        n_agents=n_agents,
        logger=log,
    )

def plot_final_comparison(
    all_records:  List[dict],
    zi_baselines: Dict[float, float],
    save_path:    Path,
    diversity_scores: List[float],
    n_episodes: int,
) -> None:
    plot_sarsa_final_comparison(
        all_records=all_records,
        zi_baselines=zi_baselines,
        save_path=save_path,
        diversity_scores=diversity_scores,
        n_episodes=n_episodes,
        logger=log,
    )

def plot_agent_eval_distribution(
    agent_eval_rows: List[dict],
    save_path: Path,
    diversity_scores: List[float],
) -> None:
    reporting_plot_agent_eval_distribution(
        agent_eval_rows=agent_eval_rows,
        save_path=save_path,
        diversity_scores=diversity_scores,
        title="SARSA eval same-population — rozkład wyników per agent",
        logger=log,
    )

def run_sarsa_cli(argv: Optional[List[str]] = None):
    global log
    args = parse_sarsa_args(argv)
    settings = build_sarsa_run_settings(args)
    artifacts = init_run_artifacts(args.run_tag, args.run_id, args.run_dir)
    log = configure_experiment_logger(artifacts, "deep_sarsa.log")
    cfg = HTMConfig(
        env    = EnvConfig(n_agents=settings["n_agents"], episode_steps=settings["episode_steps"]),
        market = settings["market"],
        log    = LogConfig(level="INFO", save_to_file=True, save_plots=True),
        sarsa  = settings["sarsa_cfg"],
    )
    run_sarsa_experiment(
        log=log,
        project_root=PROJECT_ROOT,
        args=args,
        settings=settings,
        cfg=cfg,
        artifacts=artifacts,
        worker_fn=_sarsa_train_worker,
        stamp_episode_rows=stamp_episode_rows,
        stamp_sample_rows=stamp_rows,
        plot_learning_curves=plot_learning_curves,
        plot_final_comparison=plot_final_comparison,
        plot_agent_eval_distribution=plot_agent_eval_distribution,
    )

def evaluate_trained_ppo_same_population(
    da: DoubleAuction,
    trainer: SharedPPOTrainer,
    diversity_score: float,
    seed: int,
    cfg: HTMConfig,
    n_eval_episodes: int,
    zi_baseline_trade_accuracy: float,
    zi_baseline_positive_pnl_frac: float,
    deterministic: bool = False,
    algorithm_name: Optional[str] = None,
) -> tuple[List[dict], List[dict], List[dict], List[dict]]:
    algorithm_name = algorithm_name or (
        "PPO_EVAL_SAME_POPULATION_DETERMINISTIC"
        if deterministic else
        "PPO_EVAL_SAME_POPULATION"
    )

    def action_selector(aid: str, obs: np.ndarray) -> int:
        action, _, _, _ = trainer.act_np(obs, aid, deterministic=deterministic)
        return int(action)

    extra_builder = build_standard_eval_extra_builder(
        cfg,
        zi_baseline_trade_accuracy=zi_baseline_trade_accuracy,
        zi_baseline_positive_pnl_frac=zi_baseline_positive_pnl_frac,
        eval_mode="deterministic_argmax" if deterministic else "stochastic_sample",
        collect_coordination=True,
    )

    records, sample_rows, agent_eval_rows, feature_rows, env_step_rows = evaluate_same_population_with_diagnostics(
        da=da,
        cfg=cfg,
        diversity_score=diversity_score,
        seed=seed,
        n_eval_episodes=n_eval_episodes,
        algorithm_name=algorithm_name,
        action_selector=action_selector,
        extra_builder=extra_builder,
        collect_coordination=True,
    )
    return records, sample_rows, agent_eval_rows, feature_rows, env_step_rows

def build_ppo_run_settings(quick: bool, args) -> dict:
    settings = build_ppo_benchmark_settings(
        quick=quick,
        use_agent_id_features=args.agent_id_features,
        default_workers=cpu_count(),
    )

    if args.episodes is not None:
        settings["n_episodes"] = args.episodes
    if args.steps is not None:
        settings["episode_steps"] = args.steps
    if args.seeds is not None:
        settings["n_seeds"] = args.seeds
    if args.agents is not None:
        settings["n_agents"] = args.agents
    if args.zi_episodes is not None:
        settings["zi_episodes"] = args.zi_episodes
    if args.eval_episodes is not None:
        settings["eval_episodes"] = args.eval_episodes
    if args.workers is not None:
        settings["n_workers"] = args.workers

    max_workers = settings["n_seeds"] * len(settings["diversity_scores"])
    settings["n_workers"] = max(1, min(settings["n_workers"], max_workers))
    settings["log_every"] = max(1, min(settings["log_every"], settings["n_episodes"]))
    settings["rolling_window"] = max(1, min(settings["rolling_window"], settings["n_episodes"]))
    return settings

def make_ppo_cfg(settings: dict) -> HTMConfig:
    return HTMConfig(
        env=EnvConfig(
            n_agents=settings["n_agents"],
            episode_steps=settings["episode_steps"],
        ),
        market=settings["market"],
        ppo=settings["ppo_cfg"],
        log=LogConfig(level="INFO", save_to_file=True, save_plots=True),
    )

def run_ppo_training(
    diversity_score: float,
    n_episodes: int,
    seed: int,
    cfg: HTMConfig,
    zi_baseline_trade_accuracy: float,
    zi_baseline_positive_pnl_frac: float,
    checkpoint_dir: Path,
    log_every: int,
    eval_new_population: bool,
) -> tuple[List[dict], List[dict], List[dict], List[dict], List[dict], List[dict], List[dict], SharedPPOTrainer, Path]:
    from codes.ppo_core import SharedPPOTrainer

    set_global_seeds(seed)
    da = DoubleAuction(cfg, seed=seed)
    da.reset(diversity_score=diversity_score, seed=seed)
    agent_ids = list(da.population.agents.keys())
    agent_gammas = [da.population.agents[aid].gamma for aid in agent_ids]
    ppo_cfg = dataclasses.replace(cfg.ppo)
    cfg.ppo = ppo_cfg

    obs_dim = cfg.env.n_obs + (len(agent_ids) if ppo_cfg.use_agent_id_features else 0)
    trainer = SharedPPOTrainer(
        obs_dim=obs_dim,
        n_actions=cfg.env.n_actions,
        cfg=ppo_cfg,
        seed=seed,
        agent_ids=agent_ids,
    )
    rng = np.random.default_rng(seed + 10_000)
    records: List[dict] = []
    learning_curve_records: List[dict] = []
    agent_diagnostics: List[dict] = []
    t0 = time.time()
    episode = 0

    while episode < n_episodes:
        rollout_episodes = min(cfg.ppo.rollout_episodes, n_episodes - episode)
        metrics_list, agent_metrics_list = trainer.collect_rollout(
            da,
            agent_ids,
            rng,
            deterministic=False,
            rollout_episodes=rollout_episodes,
        )
        update_stats = trainer.update()

        for metrics, agent_metrics in zip(metrics_list, agent_metrics_list):
            record = build_episode_record(
                episode=episode,
                diversity_score=diversity_score,
                seed=seed,
                algorithm="PPO_shared_plus_agent_id" if cfg.ppo.use_agent_id_features else "PPO_shared_plain",
                cfg=cfg,
                metrics=metrics,
                extra={
                    "zi_baseline_trade_accuracy": zi_baseline_trade_accuracy,
                    "zi_baseline_positive_pnl_frac": zi_baseline_positive_pnl_frac,
                    "zi_baseline": zi_baseline_trade_accuracy,
                    "beats_zi": metrics.get("trade_accuracy", 0.0) > zi_baseline_trade_accuracy,
                    "policy_loss": update_stats.get("policy_loss", 0.0),
                    "value_loss": update_stats.get("value_loss", 0.0),
                    "entropy": update_stats.get("entropy", 0.0),
                    "approx_kl": update_stats.get("approx_kl", 0.0),
                    "clip_fraction": update_stats.get("clip_fraction", 0.0),
                    "mean_advantage": update_stats.get("mean_advantage", 0.0),
                    "mean_return": update_stats.get("mean_return", 0.0),
                },
                agent_gammas=agent_gammas,
            )
            records.append(record)
            episode += 1

            if episode % 50 == 0:
                for aid, meta in agent_metrics.items():
                    sigma_i = float(meta.get("sigma_i", 0.0))
                    agent_diagnostics.append({
                        "episode": episode,
                        "diversity_score": diversity_score,
                        "seed": seed,
                        "agent_id": aid,
                        "trader_type": sigma_i / max(cfg.sentiment.sigma_chart, 1e-9),
                        "sigma_i": sigma_i,
                        "gamma": float(meta.get("gamma", 0.0)),
                        "realized_pnl": float(meta.get("realized_pnl", 0.0)),
                        "n_trades_closed": int(meta.get("n_trades_closed", 0)),
                        "trade_accuracy_agent": float(meta.get("trade_accuracy", 0.0)),
                    })
                short_eval_records, _, _, _, _ = evaluate_trained_ppo(
                    da,
                    trainer,
                    diversity_score,
                    seed,
                    cfg,
                    5,
                    zi_baseline_trade_accuracy,
                    zi_baseline_positive_pnl_frac,
                    deterministic=True,
                    same_population=True,
                )
                learning_curve_records.append({
                    "episode": episode,
                    "diversity_score": diversity_score,
                    "seed": seed,
                    "train_trade_accuracy": metrics.get("trade_accuracy", 0.0),
                    "eval_trade_accuracy": float(np.mean([r["trade_accuracy"] for r in short_eval_records])),
                    "entropy": update_stats.get("entropy", 0.0),
                    "policy_loss": update_stats.get("policy_loss", 0.0),
                    "gamma_std": float(np.std(agent_gammas)),
                })

        if episode % log_every == 0 or episode == n_episodes:
            recent = records[-min(log_every, len(records)):]
            elapsed = time.time() - t0
            log.info(
                f"  [D={diversity_score:.1f} s={seed}] "
                f"ep={episode:4d}/{n_episodes} | "
                f"acc={np.mean([r['trade_accuracy'] for r in recent]):.3f} | "
                f"pnl={np.mean([r['mean_total_pnl'] for r in recent]):.4f} | "
                f"trades={np.mean([r['n_trades'] for r in recent]):.1f} | "
                f"pl={update_stats.get('policy_loss', 0.0):.4f} | "
                f"vl={update_stats.get('value_loss', 0.0):.5f} | "
                f"ent={update_stats.get('entropy', 0.0):.3f} | "
                f"kl={update_stats.get('approx_kl', 0.0):.5f} | "
                f"clip={update_stats.get('clip_fraction', 0.0):.3f} | "
                f"t={elapsed:.0f}s"
            )

    checkpoint_path = checkpoint_dir / f"ppo_D{diversity_score:.1f}_seed{seed}.pt"
    trainer.save(checkpoint_path, episode=n_episodes)

    eval_records, sample_rows, agent_eval_rows, decision_feature_rows, env_step_rows = evaluate_trained_ppo(
        da,
        trainer,
        diversity_score,
        seed,
        cfg,
        cfg.exp.n_eval_episodes,
        zi_baseline_trade_accuracy,
        zi_baseline_positive_pnl_frac,
        deterministic=True,
        algorithm_name="PPO_EVAL_SAME_POPULATION_DETERMINISTIC",
        same_population=True,
    )
    new_population_eval_records: List[dict] = []
    if eval_new_population:
        new_population_eval_records, _, _, _, _ = evaluate_trained_ppo(
            da,
            trainer,
            diversity_score,
            seed + 1000,
            cfg,
            cfg.exp.n_eval_episodes,
            zi_baseline_trade_accuracy,
            zi_baseline_positive_pnl_frac,
            deterministic=True,
            algorithm_name="PPO_EVAL_NEW_POPULATION_DETERMINISTIC",
            same_population=False,
        )

    return (
        records,
        learning_curve_records,
        agent_diagnostics,
        eval_records,
        new_population_eval_records,
        sample_rows,
        agent_eval_rows,
        decision_feature_rows,
        env_step_rows,
        trainer,
        checkpoint_path,
    )

def evaluate_trained_ppo(
    da: DoubleAuction,
    trainer: SharedPPOTrainer,
    diversity_score: float,
    seed: int,
    cfg: HTMConfig,
    n_eval_episodes: int,
    zi_baseline_trade_accuracy: float,
    zi_baseline_positive_pnl_frac: float,
    deterministic: bool = True,
    algorithm_name: Optional[str] = None,
    same_population: bool = True,
) -> tuple[List[dict], List[dict], List[dict], List[dict], List[dict]]:
    """
    Ewaluacja PPO. Domyślnie używa tej samej populacji co trening
    (`reset_episode()` na tej samej instancji `da`). Opcjonalnie można
    przełączyć na nową populację z seedem przesuniętym względem treningu.
    """
    if same_population:
        return evaluate_trained_ppo_same_population(
            da=da,
            trainer=trainer,
            diversity_score=diversity_score,
            seed=seed,
            cfg=cfg,
            n_eval_episodes=n_eval_episodes,
            zi_baseline_trade_accuracy=zi_baseline_trade_accuracy,
            zi_baseline_positive_pnl_frac=zi_baseline_positive_pnl_frac,
            deterministic=deterministic,
            algorithm_name=algorithm_name,
        )

    eval_seed = seed if seed >= 1000 else seed + 1000
    algorithm_name = algorithm_name or (
        "PPO_EVAL_DETERMINISTIC" if deterministic else "PPO_EVAL_STOCHASTIC"
    )
    records, sample_rows, env_step_rows = evaluate_policy(
        algorithm_name,
        trainer,
        cfg,
        diversity_score,
        n_eval_episodes,
        eval_seed,
        zi_baseline_trade_accuracy=zi_baseline_trade_accuracy,
        zi_baseline_positive_pnl_frac=zi_baseline_positive_pnl_frac,
    )
    for row in records:
        row["algorithm"] = algorithm_name
        row["eval_mode"] = "deterministic_argmax" if deterministic else "stochastic_sample"
        row["eval_trade_accuracy"] = row.get("trade_accuracy", 0.0)
        row["eval_mean_total_pnl"] = row.get("mean_total_pnl", 0.0)
        row["eval_n_trades"] = row.get("n_trades", 0)
        row["eval_mean_terminal_pnl"] = row.get("mean_terminal_pnl", 0.0)
    for row in sample_rows:
        row["algorithm"] = algorithm_name
    return records, sample_rows, [], [], env_step_rows

def plot_ppo_learning_curves(
    records: List[dict],
    zi_baselines: Dict[float, float],
    save_path: Path,
    diversity_scores: List[float],
    rolling_window: int,
) -> None:
    plot_policy_learning_curves(
        records=records,
        zi_baselines=zi_baselines,
        save_path=save_path,
        diversity_scores=diversity_scores,
        rolling_window=rolling_window,
        algorithm_label="PPO",
        logger=log,
    )

def log_ppo_final_summary(
    records: List[dict],
    eval_same_population_records: List[dict],
    eval_new_population_records: List[dict],
    zi_baselines: Dict[float, float],
    diversity_scores: List[float],
    n_episodes: int,
) -> None:
    if not records:
        return

    df = pd.DataFrame(records)
    eval_df = pd.DataFrame(eval_same_population_records) if eval_same_population_records else pd.DataFrame()
    eval_new_df = pd.DataFrame(eval_new_population_records) if eval_new_population_records else pd.DataFrame()
    final_window = min(50, max(1, n_episodes // 3))
    final = df[df["episode"] >= n_episodes - final_window]

    log.info("")
    log.info("=" * 96)
    log.info(f"PODSUMOWANIE PPO — ostatnie {final_window} epizodów, uśrednione po seedach")
    log.info(
        f"{'D':>5} | {'acc':>7} | {'ZI acc':>7} | {'pnl_tot':>9} | "
        f"{'term':>8} | {'Trades':>7} | {'Closed':>7} | {'ent':>6} | {'KL':>8}"
    )
    log.info("-" * 96)

    for d in diversity_scores:
        d_final = final[final["diversity_score"] == d]
        if d_final.empty:
            continue

        acc = float(d_final["trade_accuracy"].mean())
        zi = float(zi_baselines.get(d, 0.0))
        sign = "↑" if acc > zi else "↓"
        pnl = float(d_final["mean_total_pnl"].mean())
        term = float(d_final["mean_terminal_pnl"].mean())
        trades = float(d_final["n_trades"].mean())
        closed = float(d_final["n_trades_closed"].mean())
        entropy = float(d_final["entropy"].mean()) if "entropy" in d_final else 0.0
        kl = float(d_final["approx_kl"].mean()) if "approx_kl" in d_final else 0.0

        log.info(
            f"{d:5.1f} | {acc:6.3f}{sign} | {zi:7.3f} | "
            f"{pnl:9.4f} | {term:8.4f} | {trades:7.1f} | "
            f"{closed:7.1f} | {entropy:6.3f} | {kl:8.5f}"
        )

    if not eval_df.empty:
        log.info("")
        log.info("EWALUACJA PPO — ta sama populacja, stochastic sample z masked policy")
        log.info(
            f"{'D':>5} | {'eval acc':>8} | {'ZI acc':>7} | {'eval pnl':>9} | "
            f"{'term':>8} | {'Trades':>7} | {'Closed':>7}"
        )
        log.info("-" * 76)
        for d in diversity_scores:
            d_eval = eval_df[eval_df["diversity_score"] == d]
            if d_eval.empty:
                continue
            eval_acc = float(d_eval["trade_accuracy"].mean())
            zi = float(zi_baselines.get(d, 0.0))
            sign = "↑" if eval_acc > zi else "↓"
            eval_pnl = float(d_eval["mean_total_pnl"].mean())
            eval_term = float(d_eval["mean_terminal_pnl"].mean())
            eval_trades = float(d_eval["n_trades"].mean())
            eval_closed = float(d_eval["n_trades_closed"].mean())
            log.info(
                f"{d:5.1f} | {eval_acc:7.3f}{sign} | {zi:7.3f} | "
                f"{eval_pnl:9.4f} | {eval_term:8.4f} | "
                f"{eval_trades:7.1f} | {eval_closed:7.1f}"
            )

    if not eval_new_df.empty:
        log.info("")
        log.info("EWALUACJA PPO — nowa populacja, seed+1000, stochastic sample z masked policy")
        log.info(
            f"{'D':>5} | {'eval acc':>8} | {'ZI acc':>7} | {'eval pnl':>9} | "
            f"{'term':>8} | {'Trades':>7} | {'Closed':>7}"
        )
        log.info("-" * 76)
        for d in diversity_scores:
            d_eval = eval_new_df[eval_new_df["diversity_score"] == d]
            if d_eval.empty:
                continue
            eval_acc = float(d_eval["trade_accuracy"].mean())
            zi = float(zi_baselines.get(d, 0.0))
            sign = "↑" if eval_acc > zi else "↓"
            eval_pnl = float(d_eval["mean_total_pnl"].mean())
            eval_term = float(d_eval["mean_terminal_pnl"].mean())
            eval_trades = float(d_eval["n_trades"].mean())
            eval_closed = float(d_eval["n_trades_closed"].mean())
            log.info(
                f"{d:5.1f} | {eval_acc:7.3f}{sign} | {zi:7.3f} | "
                f"{eval_pnl:9.4f} | {eval_term:8.4f} | "
                f"{eval_trades:7.1f} | {eval_closed:7.1f}"
            )

def _ppo_train_worker(args: tuple) -> tuple[List[dict], List[dict], List[dict], List[dict], List[dict], List[dict], List[dict], Path]:
    """
    Jeden niezależny trening PPO dla kombinacji (D, seed).

    Funkcja jest na poziomie modułu, żeby multiprocessing mógł ją picklować.
    """
    (
        diversity_score,
        seed,
        cfg,
        zi_baseline_trade_accuracy,
        zi_baseline_positive_pnl_frac,
        n_episodes,
        checkpoint_dir,
        log_every,
        eval_new_population,
    ) = args
    global log
    log = configure_worker_logger(f"ppo_worker_D{diversity_score:.1f}_seed{seed}.log")

    try:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    cfg = copy.deepcopy(cfg)
    train_records, learning_curve_records, agent_diagnostics, eval_records, eval_new_population_records, sample_rows, agent_eval_rows, decision_feature_rows, env_step_rows, _, checkpoint_path = run_ppo_training(
        diversity_score=diversity_score,
        n_episodes=n_episodes,
        seed=seed,
        cfg=cfg,
        zi_baseline_trade_accuracy=zi_baseline_trade_accuracy,
        zi_baseline_positive_pnl_frac=zi_baseline_positive_pnl_frac,
        checkpoint_dir=checkpoint_dir,
        log_every=log_every,
        eval_new_population=eval_new_population,
    )
    return (
        train_records,
        learning_curve_records,
        agent_diagnostics,
        eval_records,
        eval_new_population_records,
        sample_rows,
        agent_eval_rows,
        decision_feature_rows,
        env_step_rows,
        checkpoint_path,
    )

def run_ppo_cli(argv: List[str] | None = None) -> None:
    global log
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Szybki smoke test PPO")
    parser.add_argument("--episodes", type=int, help="Override liczby epizodów.")
    parser.add_argument("--steps", type=int, help="Override liczby kroków w epizodzie.")
    parser.add_argument("--seeds", type=int, help="Override liczby seedów.")
    parser.add_argument("--agents", type=int, help="Override liczby agentów.")
    parser.add_argument("--zi-episodes", type=int, help="Override liczby epizodów ZI baseline.")
    parser.add_argument("--eval-episodes", type=int, help="Override liczby epizodów eval.")
    parser.add_argument(
        "--agent-id-features",
        action="store_true",
        help="Doklej one-hot agent_id do obserwacji PPO",
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Liczba równoległych procesów dla niezależnych zadań (D, seed).",
    )
    parser.add_argument(
        "--eval-new-population",
        action="store_true",
        help="Dodatkowo uruchom osobny eval PPO na nowej populacji z seedem przesuniętym o 1000.",
    )
    parser.add_argument("--run-tag", type=str, default="run", help="Krótki tag do nazwy folderu run.")
    parser.add_argument("--run-id", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=str, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    settings = build_ppo_run_settings(args.quick, args)
    cfg = make_ppo_cfg(settings)
    cfg.exp.n_eval_episodes = settings["eval_episodes"]
    cfg.exp.diversity_scores = list(settings["diversity_scores"])
    artifacts = init_run_artifacts(args.run_tag, args.run_id, args.run_dir)
    log = configure_experiment_logger(artifacts, "ppo.log")
    run_ppo_experiment(
        log=log,
        project_root=PROJECT_ROOT,
        args=args,
        settings=settings,
        cfg=cfg,
        artifacts=artifacts,
        worker_fn=_ppo_train_worker,
        stamp_episode_rows=stamp_episode_rows,
        stamp_sample_rows=stamp_rows,
        plot_learning_curves=plot_ppo_learning_curves,
        log_final_summary=log_ppo_final_summary,
    )

def evaluate_trained_ippo_same_population(
    da: DoubleAuction,
    trainer: IndependentPPOTrainer,
    diversity_score: float,
    seed: int,
    cfg: HTMConfig,
    n_eval_episodes: int,
    zi_baseline_trade_accuracy: float,
    zi_baseline_positive_pnl_frac: float,
    deterministic: bool = False,
    algorithm_name: Optional[str] = None,
) -> tuple[List[dict], List[dict], List[dict], List[dict]]:
    algorithm_name = algorithm_name or (
        "IPPO_EVAL_SAME_POPULATION_DETERMINISTIC"
        if deterministic else
        "IPPO_EVAL_SAME_POPULATION"
    )

    def action_selector(aid: str, obs: np.ndarray) -> int:
        action, _, _, _ = trainer.act_np(obs, aid, deterministic=deterministic)
        return int(action)

    extra_builder = build_standard_eval_extra_builder(
        cfg,
        zi_baseline_trade_accuracy=zi_baseline_trade_accuracy,
        zi_baseline_positive_pnl_frac=zi_baseline_positive_pnl_frac,
        eval_mode="deterministic_argmax" if deterministic else "stochastic_sample",
        collect_coordination=True,
    )

    records, sample_rows, agent_eval_rows, feature_rows, env_step_rows = evaluate_same_population_with_diagnostics(
        da=da,
        cfg=cfg,
        diversity_score=diversity_score,
        seed=seed,
        n_eval_episodes=n_eval_episodes,
        algorithm_name=algorithm_name,
        action_selector=action_selector,
        extra_builder=extra_builder,
        collect_coordination=True,
    )
    return records, sample_rows, agent_eval_rows, feature_rows, env_step_rows

def build_ippo_run_settings(quick: bool, args) -> dict:
    settings = build_ippo_benchmark_settings(quick=quick, default_workers=cpu_count())
    if args.episodes is not None:
        settings["n_episodes"] = args.episodes
    if args.steps is not None:
        settings["episode_steps"] = args.steps
    if args.seeds is not None:
        settings["n_seeds"] = args.seeds
    if args.agents is not None:
        settings["n_agents"] = args.agents
    if args.zi_episodes is not None:
        settings["zi_episodes"] = args.zi_episodes
    if args.eval_episodes is not None:
        settings["eval_episodes"] = args.eval_episodes
    if args.workers is not None:
        settings["n_workers"] = args.workers
    max_workers = settings["n_seeds"] * len(settings["diversity_scores"])
    settings["n_workers"] = max(1, min(settings["n_workers"], max_workers))
    settings["log_every"] = max(1, min(settings["log_every"], settings["n_episodes"]))
    settings["rolling_window"] = max(1, min(settings["rolling_window"], settings["n_episodes"]))
    return settings

def make_ippo_cfg(settings: dict) -> HTMConfig:
    return HTMConfig(
        env=EnvConfig(
            n_agents=settings["n_agents"],
            episode_steps=settings["episode_steps"],
        ),
        market=settings["market"],
        ppo=settings["ppo_cfg"],
        log=LogConfig(level="INFO", save_to_file=True, save_plots=True),
    )

def run_ippo_training(
    diversity_score: float,
    n_episodes: int,
    seed: int,
    cfg: HTMConfig,
    zi_baseline_trade_accuracy: float,
    zi_baseline_positive_pnl_frac: float,
    checkpoint_dir: Path,
    log_every: int,
    eval_new_population: bool,
) -> tuple[List[dict], List[dict], List[dict], List[dict], List[dict], List[dict], List[dict], IndependentPPOTrainer, Path]:
    from codes.ppo_core import IndependentPPOTrainer

    set_global_seeds(seed)
    da = DoubleAuction(cfg, seed=seed)
    da.reset(diversity_score=diversity_score, seed=seed)
    agent_ids = list(da.population.agents.keys())
    agent_gammas = [da.population.agents[aid].gamma for aid in agent_ids]
    ppo_cfg = dataclasses.replace(cfg.ppo, use_agent_id_features=False)
    cfg.ppo = ppo_cfg

    obs_dim = cfg.env.n_obs
    trainer = IndependentPPOTrainer(
        obs_dim=obs_dim,
        n_actions=cfg.env.n_actions,
        cfg=ppo_cfg,
        seed=seed,
        agent_ids=agent_ids,
    )
    rng = np.random.default_rng(seed + 10_000)
    records: List[dict] = []
    learning_curve_records: List[dict] = []
    agent_diagnostics: List[dict] = []
    t0 = time.time()
    episode = 0

    while episode < n_episodes:
        rollout_episodes = min(cfg.ppo.rollout_episodes, n_episodes - episode)
        metrics_list, agent_metrics_list = trainer.collect_rollout(
            da,
            agent_ids,
            rng,
            deterministic=False,
            rollout_episodes=rollout_episodes,
        )
        update_stats = trainer.update()

        for metrics, agent_metrics in zip(metrics_list, agent_metrics_list):
            record = build_episode_record(
                episode=episode,
                diversity_score=diversity_score,
                seed=seed,
                algorithm="IPPO_light",
                cfg=cfg,
                metrics=metrics,
                extra={
                    "zi_baseline_trade_accuracy": zi_baseline_trade_accuracy,
                    "zi_baseline_positive_pnl_frac": zi_baseline_positive_pnl_frac,
                    "zi_baseline": zi_baseline_trade_accuracy,
                    "beats_zi": metrics.get("trade_accuracy", 0.0) > zi_baseline_trade_accuracy,
                    "policy_loss": update_stats.get("policy_loss", 0.0),
                    "value_loss": update_stats.get("value_loss", 0.0),
                    "entropy": update_stats.get("entropy", 0.0),
                    "approx_kl": update_stats.get("approx_kl", 0.0),
                    "clip_fraction": update_stats.get("clip_fraction", 0.0),
                    "mean_advantage": update_stats.get("mean_advantage", 0.0),
                    "mean_return": update_stats.get("mean_return", 0.0),
                },
                agent_gammas=agent_gammas,
            )
            records.append(record)
            episode += 1

            if episode % 50 == 0:
                for aid, meta in agent_metrics.items():
                    sigma_i = float(meta.get("sigma_i", 0.0))
                    agent_diagnostics.append({
                        "episode": episode,
                        "diversity_score": diversity_score,
                        "seed": seed,
                        "agent_id": aid,
                        "trader_type": sigma_i / max(cfg.sentiment.sigma_chart, 1e-9),
                        "sigma_i": sigma_i,
                        "gamma": float(meta.get("gamma", 0.0)),
                        "realized_pnl": float(meta.get("realized_pnl", 0.0)),
                        "n_trades_closed": int(meta.get("n_trades_closed", 0)),
                        "trade_accuracy_agent": float(meta.get("trade_accuracy", 0.0)),
                    })
                short_eval_records, _, _, _, _ = evaluate_trained_ippo(
                    da,
                    trainer,
                    diversity_score,
                    seed,
                    cfg,
                    5,
                    zi_baseline_trade_accuracy,
                    zi_baseline_positive_pnl_frac,
                    deterministic=True,
                    same_population=True,
                )
                learning_curve_records.append({
                    "episode": episode,
                    "diversity_score": diversity_score,
                    "seed": seed,
                    "train_trade_accuracy": metrics.get("trade_accuracy", 0.0),
                    "eval_trade_accuracy": float(np.mean([r["trade_accuracy"] for r in short_eval_records])),
                    "entropy": update_stats.get("entropy", 0.0),
                    "policy_loss": update_stats.get("policy_loss", 0.0),
                    "gamma_std": float(np.std(agent_gammas)),
                })

        if episode % log_every == 0 or episode == n_episodes:
            recent = records[-min(log_every, len(records)):]
            elapsed = time.time() - t0
            log.info(
                f"  [D={diversity_score:.1f} s={seed}] "
                f"ep={episode:4d}/{n_episodes} | "
                f"acc={np.mean([r['trade_accuracy'] for r in recent]):.3f} | "
                f"pnl={np.mean([r['mean_total_pnl'] for r in recent]):.4f} | "
                f"trades={np.mean([r['n_trades'] for r in recent]):.1f} | "
                f"pl={update_stats.get('policy_loss', 0.0):.4f} | "
                f"vl={update_stats.get('value_loss', 0.0):.5f} | "
                f"ent={update_stats.get('entropy', 0.0):.3f} | "
                f"kl={update_stats.get('approx_kl', 0.0):.5f} | "
                f"clip={update_stats.get('clip_fraction', 0.0):.3f} | "
                f"t={elapsed:.0f}s"
            )

    checkpoint_path = checkpoint_dir / f"ippo_D{diversity_score:.1f}_seed{seed}.pt"
    trainer.save(checkpoint_path, episode=n_episodes)

    eval_records, sample_rows, agent_eval_rows, decision_feature_rows, env_step_rows = evaluate_trained_ippo(
        da,
        trainer,
        diversity_score,
        seed,
        cfg,
        cfg.exp.n_eval_episodes,
        zi_baseline_trade_accuracy,
        zi_baseline_positive_pnl_frac,
        deterministic=True,
        algorithm_name="IPPO_EVAL_SAME_POPULATION_DETERMINISTIC",
        same_population=True,
    )
    new_population_eval_records: List[dict] = []
    if eval_new_population:
        new_population_eval_records, _, _, _, _ = evaluate_trained_ippo(
            da,
            trainer,
            diversity_score,
            seed + 1000,
            cfg,
            cfg.exp.n_eval_episodes,
            zi_baseline_trade_accuracy,
            zi_baseline_positive_pnl_frac,
            deterministic=True,
            algorithm_name="IPPO_EVAL_NEW_POPULATION_DETERMINISTIC",
            same_population=False,
        )

    return (
        records,
        learning_curve_records,
        agent_diagnostics,
        eval_records,
        new_population_eval_records,
        sample_rows,
        agent_eval_rows,
        decision_feature_rows,
        env_step_rows,
        trainer,
        checkpoint_path,
    )

def evaluate_trained_ippo(
    da: DoubleAuction,
    trainer: IndependentPPOTrainer,
    diversity_score: float,
    seed: int,
    cfg: HTMConfig,
    n_eval_episodes: int,
    zi_baseline_trade_accuracy: float,
    zi_baseline_positive_pnl_frac: float,
    deterministic: bool = True,
    algorithm_name: Optional[str] = None,
    same_population: bool = True,
) -> tuple[List[dict], List[dict], List[dict], List[dict], List[dict]]:
    if same_population:
        return evaluate_trained_ippo_same_population(
            da=da,
            trainer=trainer,
            diversity_score=diversity_score,
            seed=seed,
            cfg=cfg,
            n_eval_episodes=n_eval_episodes,
            zi_baseline_trade_accuracy=zi_baseline_trade_accuracy,
            zi_baseline_positive_pnl_frac=zi_baseline_positive_pnl_frac,
            deterministic=deterministic,
            algorithm_name=algorithm_name,
        )

    eval_seed = seed if seed >= 1000 else seed + 1000
    algorithm_name = algorithm_name or (
        "IPPO_EVAL_DETERMINISTIC" if deterministic else "IPPO_EVAL_STOCHASTIC"
    )
    records, sample_rows, env_step_rows = evaluate_policy(
        algorithm_name,
        trainer,
        cfg,
        diversity_score,
        n_eval_episodes,
        eval_seed,
        zi_baseline_trade_accuracy=zi_baseline_trade_accuracy,
        zi_baseline_positive_pnl_frac=zi_baseline_positive_pnl_frac,
    )
    for row in records:
        row["algorithm"] = algorithm_name
        row["eval_mode"] = "deterministic_argmax" if deterministic else "stochastic_sample"
        row["eval_trade_accuracy"] = row.get("trade_accuracy", 0.0)
        row["eval_mean_total_pnl"] = row.get("mean_total_pnl", 0.0)
        row["eval_n_trades"] = row.get("n_trades", 0)
        row["eval_mean_terminal_pnl"] = row.get("mean_terminal_pnl", 0.0)
    for row in sample_rows:
        row["algorithm"] = algorithm_name
    return records, sample_rows, [], [], env_step_rows

def plot_ippo_learning_curves(
    records: List[dict],
    zi_baselines: Dict[float, float],
    save_path: Path,
    diversity_scores: List[float],
    rolling_window: int,
) -> None:
    plot_policy_learning_curves(
        records=records,
        zi_baselines=zi_baselines,
        save_path=save_path,
        diversity_scores=diversity_scores,
        rolling_window=rolling_window,
        algorithm_label="IPPO",
        logger=log,
    )

def log_ippo_final_summary(
    records: List[dict],
    eval_same_population_records: List[dict],
    eval_new_population_records: List[dict],
    zi_baselines: Dict[float, float],
    diversity_scores: List[float],
    n_episodes: int,
) -> None:
    if not records:
        return
    df = pd.DataFrame(records)
    eval_df = pd.DataFrame(eval_same_population_records) if eval_same_population_records else pd.DataFrame()
    eval_new_df = pd.DataFrame(eval_new_population_records) if eval_new_population_records else pd.DataFrame()
    final_window = min(50, max(1, n_episodes // 3))
    final = df[df["episode"] >= n_episodes - final_window]

    log.info("")
    log.info("=" * 96)
    log.info(f"PODSUMOWANIE IPPO — ostatnie {final_window} epizodów, uśrednione po seedach")
    log.info(
        f"{'D':>5} | {'acc':>7} | {'ZI acc':>7} | {'pnl_tot':>9} | "
        f"{'term':>8} | {'Trades':>7} | {'Closed':>7} | {'ent':>6} | {'KL':>8}"
    )
    log.info("-" * 96)

    for d in diversity_scores:
        d_final = final[final["diversity_score"] == d]
        if d_final.empty:
            continue
        acc = float(d_final["trade_accuracy"].mean())
        zi = float(zi_baselines.get(d, 0.0))
        sign = "↑" if acc > zi else "↓"
        pnl = float(d_final["mean_total_pnl"].mean())
        term = float(d_final["mean_terminal_pnl"].mean())
        trades = float(d_final["n_trades"].mean())
        closed = float(d_final["n_trades_closed"].mean())
        entropy = float(d_final["entropy"].mean()) if "entropy" in d_final else 0.0
        kl = float(d_final["approx_kl"].mean()) if "approx_kl" in d_final else 0.0
        log.info(
            f"{d:5.1f} | {acc:6.3f}{sign} | {zi:7.3f} | "
            f"{pnl:9.4f} | {term:8.4f} | {trades:7.1f} | "
            f"{closed:7.1f} | {entropy:6.3f} | {kl:8.5f}"
        )

    def _log_eval_block(title: str, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        log.info("")
        log.info(title)
        log.info(
            f"{'D':>5} | {'eval acc':>8} | {'ZI acc':>7} | {'eval pnl':>9} | "
            f"{'term':>8} | {'Trades':>7} | {'Closed':>7}"
        )
        log.info("-" * 76)
        for d in diversity_scores:
            d_eval = frame[frame["diversity_score"] == d]
            if d_eval.empty:
                continue
            eval_acc = float(d_eval["trade_accuracy"].mean())
            zi = float(zi_baselines.get(d, 0.0))
            sign = "↑" if eval_acc > zi else "↓"
            eval_pnl = float(d_eval["mean_total_pnl"].mean())
            eval_term = float(d_eval["mean_terminal_pnl"].mean())
            eval_trades = float(d_eval["n_trades"].mean())
            eval_closed = float(d_eval["n_trades_closed"].mean())
            log.info(
                f"{d:5.1f} | {eval_acc:7.3f}{sign} | {zi:7.3f} | "
                f"{eval_pnl:9.4f} | {eval_term:8.4f} | "
                f"{eval_trades:7.1f} | {eval_closed:7.1f}"
            )

    _log_eval_block("EWALUACJA IPPO — ta sama populacja, deterministic argmax", eval_df)
    _log_eval_block("EWALUACJA IPPO — nowa populacja, seed+1000, deterministic argmax", eval_new_df)

def _ippo_train_worker(args: tuple) -> tuple[List[dict], List[dict], List[dict], List[dict], List[dict], List[dict], List[dict], Path]:
    (
        diversity_score,
        seed,
        cfg,
        zi_baseline_trade_accuracy,
        zi_baseline_positive_pnl_frac,
        n_episodes,
        checkpoint_dir,
        log_every,
        eval_new_population,
    ) = args
    global log
    log = configure_worker_logger(f"ippo_worker_D{diversity_score:.1f}_seed{seed}.log")

    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    cfg = copy.deepcopy(cfg)
    train_records, learning_curve_records, agent_diagnostics, eval_records, eval_new_population_records, sample_rows, agent_eval_rows, decision_feature_rows, env_step_rows, _, checkpoint_path = run_ippo_training(
        diversity_score=diversity_score,
        n_episodes=n_episodes,
        seed=seed,
        cfg=cfg,
        zi_baseline_trade_accuracy=zi_baseline_trade_accuracy,
        zi_baseline_positive_pnl_frac=zi_baseline_positive_pnl_frac,
        checkpoint_dir=checkpoint_dir,
        log_every=log_every,
        eval_new_population=eval_new_population,
    )
    return (
        train_records,
        learning_curve_records,
        agent_diagnostics,
        eval_records,
        eval_new_population_records,
        sample_rows,
        agent_eval_rows,
        decision_feature_rows,
        env_step_rows,
        checkpoint_path,
    )

def run_ippo_cli(argv: List[str] | None = None) -> None:
    global log
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Szybki smoke test IPPO")
    parser.add_argument("--episodes", type=int, help="Override liczby epizodów.")
    parser.add_argument("--steps", type=int, help="Override liczby kroków w epizodzie.")
    parser.add_argument("--seeds", type=int, help="Override liczby seedów.")
    parser.add_argument("--agents", type=int, help="Override liczby agentów.")
    parser.add_argument("--zi-episodes", type=int, help="Override liczby epizodów ZI baseline.")
    parser.add_argument("--eval-episodes", type=int, help="Override liczby epizodów eval.")
    parser.add_argument("--workers", type=int, help="Liczba równoległych procesów dla niezależnych zadań (D, seed).")
    parser.add_argument(
        "--eval-new-population",
        action="store_true",
        help="Dodatkowo uruchom osobny eval IPPO na nowej populacji z seedem przesuniętym o 1000.",
    )
    parser.add_argument("--run-tag", type=str, default="run", help="Krótki tag do nazwy folderu run.")
    parser.add_argument("--run-id", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=str, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    settings = build_ippo_run_settings(args.quick, args)
    cfg = make_ippo_cfg(settings)
    cfg.exp.n_eval_episodes = settings["eval_episodes"]
    cfg.exp.diversity_scores = list(settings["diversity_scores"])
    artifacts = init_run_artifacts(args.run_tag, args.run_id, args.run_dir)
    log = configure_experiment_logger(artifacts, "ippo.log")
    run_ppo_experiment(
        log=log,
        project_root=PROJECT_ROOT,
        args=args,
        settings=settings,
        cfg=cfg,
        artifacts=artifacts,
        worker_fn=_ippo_train_worker,
        stamp_episode_rows=stamp_episode_rows,
        stamp_sample_rows=stamp_rows,
        plot_learning_curves=plot_ippo_learning_curves,
        log_final_summary=log_ippo_final_summary,
        algorithm_label="IPPO",
        artifact_stem="ippo_quick" if args.quick else "ippo",
    )

def build_signal_rule_cli_settings(quick: bool, args) -> dict:
    settings = build_signal_rule_benchmark_settings(quick=quick, default_workers=cpu_count())
    if args.steps is not None:
        settings["episode_steps"] = args.steps
    if args.seeds is not None:
        settings["n_seeds"] = args.seeds
    if args.agents is not None:
        settings["n_agents"] = args.agents
    if args.zi_episodes is not None:
        settings["zi_episodes"] = args.zi_episodes
    if args.eval_episodes is not None:
        settings["eval_episodes"] = args.eval_episodes
    if args.workers is not None:
        settings["n_workers"] = args.workers
    max_workers = settings["n_seeds"] * len(settings["diversity_scores"])
    settings["n_workers"] = max(1, min(settings["n_workers"], max_workers))
    return settings

def make_signal_rule_cfg(settings: dict) -> HTMConfig:
    return HTMConfig(
        env=EnvConfig(
            n_agents=settings["n_agents"],
            episode_steps=settings["episode_steps"],
        ),
        market=settings["market"],
        log=LogConfig(level="INFO", save_to_file=True, save_plots=False),
    )

def evaluate_signal_rule_same_population(
    da: DoubleAuction,
    policy: SignalRulePolicy,
    diversity_score: float,
    seed: int,
    cfg: HTMConfig,
    n_eval_episodes: int,
    zi_baseline_trade_accuracy: float,
    zi_baseline_positive_pnl_frac: float,
    algorithm_name: str = "SIGNAL_RULE_EVAL_SAME_POPULATION",
) -> tuple[List[dict], List[dict], List[dict], List[dict], List[dict]]:
    def action_selector(_aid: str, obs: np.ndarray) -> int:
        return int(policy.act(obs))

    extra_builder = build_standard_eval_extra_builder(
        cfg,
        zi_baseline_trade_accuracy=zi_baseline_trade_accuracy,
        zi_baseline_positive_pnl_frac=zi_baseline_positive_pnl_frac,
        eval_mode="rule_based",
        collect_coordination=True,
    )

    records, sample_rows, agent_eval_rows, feature_rows, env_step_rows = evaluate_same_population_with_diagnostics(
        da=da,
        cfg=cfg,
        diversity_score=diversity_score,
        seed=seed,
        n_eval_episodes=n_eval_episodes,
        algorithm_name=algorithm_name,
        action_selector=action_selector,
        extra_builder=extra_builder,
        collect_coordination=True,
    )
    return records, sample_rows, agent_eval_rows, feature_rows, env_step_rows

def _signal_rule_worker(task: tuple) -> tuple:
    (
        diversity_score,
        seed,
        cfg,
        zi_baseline_trade_accuracy,
        zi_baseline_positive_pnl_frac,
        n_eval_episodes,
        rule_threshold,
        eval_new_population,
    ) = task
    set_global_seeds(seed)
    policy = SignalRulePolicy(threshold=rule_threshold)

    da = DoubleAuction(cfg, seed=seed)
    da.reset(diversity_score=diversity_score, seed=seed)
    eval_records, sample_rows, agent_eval_rows, decision_feature_rows, env_step_rows = evaluate_signal_rule_same_population(
        da=da,
        policy=policy,
        diversity_score=diversity_score,
        seed=seed,
        cfg=cfg,
        n_eval_episodes=n_eval_episodes,
        zi_baseline_trade_accuracy=zi_baseline_trade_accuracy,
        zi_baseline_positive_pnl_frac=zi_baseline_positive_pnl_frac,
    )

    eval_new_population_records: List[dict] = []
    if eval_new_population:
        da_new = DoubleAuction(cfg, seed=seed + 1000)
        da_new.reset(diversity_score=diversity_score, seed=seed + 1000)
        eval_new_population_records, _, _, _, _ = evaluate_signal_rule_same_population(
            da=da_new,
            policy=policy,
            diversity_score=diversity_score,
            seed=seed,
            cfg=cfg,
            n_eval_episodes=n_eval_episodes,
            zi_baseline_trade_accuracy=zi_baseline_trade_accuracy,
            zi_baseline_positive_pnl_frac=zi_baseline_positive_pnl_frac,
            algorithm_name="SIGNAL_RULE_EVAL_NEW_POPULATION",
        )
    return eval_records, eval_new_population_records, sample_rows, agent_eval_rows, decision_feature_rows, env_step_rows

def parse_signal_rule_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Uruchom benchmark SignalRule w trybie quick.")
    parser.add_argument("--steps", type=int, help="Override liczby kroków w epizodzie.")
    parser.add_argument("--seeds", type=int, help="Override liczby seedów.")
    parser.add_argument("--agents", type=int, help="Override liczby agentów.")
    parser.add_argument("--zi-episodes", type=int, help="Override liczby epizodów ZI baseline.")
    parser.add_argument("--eval-episodes", type=int, help="Override liczby epizodów eval.")
    parser.add_argument("--workers", type=int, help="Override workerów dla benchmarku.")
    parser.add_argument("--run-tag", type=str, default="signal_rule", help="Krótki tag do nazwy folderu run.")
    parser.add_argument("--eval-new-population", action="store_true", help="Dodatkowo oceń politykę na nowo zsample’owanej populacji.")
    parser.add_argument("--run-id", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=str, help=argparse.SUPPRESS)
    return parser.parse_args(argv)

def run_signal_rule_cli(argv: Optional[List[str]] = None) -> None:
    global log
    args = parse_signal_rule_args(argv)
    settings = build_signal_rule_cli_settings(args.quick, args)
    cfg = make_signal_rule_cfg(settings)
    artifacts = init_run_artifacts(args.run_tag, args.run_id, args.run_dir)
    log = configure_experiment_logger(artifacts, "signal_rule.log")

    ensure_run_config(
        artifacts=artifacts,
        run_tag=args.run_tag,
        algorithm="SignalRule",
        settings=settings,
        cfg=cfg,
        eval_new_population=args.eval_new_population,
    )

    log.info("=" * 70)
    log.info(
        f"SignalRule | {settings['run_name']} | N={cfg.env.n_agents} | "
        f"D={settings['diversity_scores']} | steps={cfg.env.episode_steps} | "
        f"seeds={settings['n_seeds']} | workers={settings['n_workers']} | "
        f"eval_ep={settings['eval_episodes']} | thr={settings['rule_threshold']:.3f}"
    )
    log.info("=" * 70)

    zi_records, zi_acc, zi_pos = compute_shared_zi_baseline(
        cfg=cfg,
        diversity_scores=settings["diversity_scores"],
        zi_episodes=settings["zi_episodes"],
        run_id=artifacts.run_id,
        episodes_csv=artifacts.episodes_csv,
        stamp_episode_rows=stamp_episode_rows,
        seed=42,
    )

    tasks = [
        (
            d,
            seed,
            cfg,
            zi_acc[d],
            zi_pos[d],
            settings["eval_episodes"],
            settings["rule_threshold"],
            args.eval_new_population,
        )
        for d in settings["diversity_scores"]
        for seed in range(settings["n_seeds"])
    ]

    all_eval_same_population_records, all_eval_new_population_records = run_eval_only_experiment(
        log=log,
        settings=settings,
        artifacts=artifacts,
        worker_fn=_signal_rule_worker,
        tasks=tasks,
        stamp_episode_rows=stamp_episode_rows,
        stamp_sample_rows=stamp_rows,
        algorithm_label="SignalRule",
    )

    total_rows = len(all_eval_same_population_records) + len(all_eval_new_population_records) + len(zi_records)
    log.info(f"Wyniki: {artifacts.episodes_csv} ({total_rows} wierszy)")
