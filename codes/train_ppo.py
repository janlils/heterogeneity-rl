"""
Runner treningowy shared-policy PPO dla HTM.

Uruchomienie:
    python -m codes.train_ppo
    python -m codes.train_ppo --quick
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import logging
import os
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
(PROJECT_ROOT / ".matplotlib_cache").mkdir(exist_ok=True)
(PROJECT_ROOT / ".cache").mkdir(exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from codes.config import HTMConfig, EnvConfig, LogConfig, MarketDynamics, PPOConfig
from codes.double_auction import DoubleAuction
from codes.evaluate_policies import evaluate_policy, evaluate_zi
from codes.ppo import SharedPPOTrainer
from codes.rl_common import build_agent_sample_row, build_env_step_row, build_episode_record, set_global_seeds
from codes.results_store import (
    AGENT_SAMPLE_FIELDS,
    EPISODE_FIELDS,
    ENV_STEP_FIELDS,
    append_rows,
    prepare_run_dir,
    write_run_config,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "logs" / "ppo.log", mode="w"),
    ],
)
log = logging.getLogger("htm.train_ppo")


DIVERSITY_SCORES = [0.0, 0.3, 0.5, 0.7, 1.0]
N_AGENTS = 50
N_EPISODES = 500
N_SEEDS = 10
EPISODE_STEPS = 500
ZI_EPISODES = 30
EVAL_EPISODES = 30
LOG_EVERY = 25
ROLLING_WINDOW = 30
MARKET = MarketDynamics.stable()
PPO_CFG = PPOConfig()


def _stamp_episode_rows(rows: List[dict], run_id: str, phase: str) -> List[dict]:
    stamped: List[dict] = []
    for row in rows:
        item = dict(row)
        item["run_id"] = run_id
        item["phase"] = phase
        stamped.append(item)
    return stamped


def _stamp_sample_rows(rows: List[dict], run_id: str) -> List[dict]:
    stamped: List[dict] = []
    for row in rows:
        item = dict(row)
        item["run_id"] = run_id
        stamped.append(item)
    return stamped


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


def _sampled_agent_types(da: DoubleAuction) -> dict[str, tuple[str, float]]:
    trader_meta = []
    for aid, agent in da.population.agents.items():
        trader_type = agent.sigma_i / max(da.cfg.sentiment.sigma_chart, 1e-9)
        trader_meta.append((aid, trader_type))
    trader_meta.sort(key=lambda item: item[1])
    fundamentalist_id, fundamentalist_type = trader_meta[0]
    chartist_id, chartist_type = trader_meta[-1]
    mixed_id, mixed_type = min(trader_meta, key=lambda item: abs(item[1] - 0.5))
    return {
        fundamentalist_id: ("fundamentalista", fundamentalist_type),
        mixed_id: ("mieszany", mixed_type),
        chartist_id: ("chartista", chartist_type),
    }


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
) -> tuple[List[dict], List[dict], List[dict]]:
    agent_ids = list(da.population.agents.keys())
    agent_gammas = [da.population.agents[aid].gamma for aid in agent_ids]
    records: List[dict] = []
    sample_rows: List[dict] = []
    env_step_rows: List[dict] = []
    algorithm_name = algorithm_name or (
        "PPO_EVAL_SAME_POPULATION_DETERMINISTIC"
        if deterministic else
        "PPO_EVAL_SAME_POPULATION"
    )

    for episode in range(n_eval_episodes):
        da.reset_episode()
        prev_positions = {aid: da.population.agents[aid].position for aid in agent_ids}
        sample_this_episode = seed == 0 and abs(diversity_score - 1.0) < 1e-9 and episode == 0
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
                action, _, _, _ = trainer.act_np(obs, aid, deterministic=deterministic)
                actions[aid] = int(action)

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
                    executed = agent.position != positions_before[aid]
                    trader_type = agent.sigma_i / max(cfg.sentiment.sigma_chart, 1e-9)
                    if trader_type <= 0.33:
                        agent_type = "fundamentalista"
                    elif trader_type >= 0.67:
                        agent_type = "chartista"
                    else:
                        agent_type = "mieszany"
                    realized_pnl_this_step = float(agent.realized_pnl - realized_before[aid])
                    sample_rows.append(build_agent_sample_row(
                        algorithm=algorithm_name,
                        phase="eval_same_population",
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
                        sentiment=agent.sentiment,
                        sigma_i=agent.sigma_i,
                        threshold=agent.threshold,
                    ))
                    prev_positions[aid] = agent.position
                mean_signal = float(np.mean([float(obs_by_agent[aid][0]) for aid in agent_ids]))
                std_signal = float(np.std([float(obs_by_agent[aid][0]) for aid in agent_ids]))
                mean_sigma = float(np.mean([float(da.population.agents[aid].sigma_i) for aid in agent_ids]))
                mean_position_before = float(np.mean([positions_before[aid] for aid in agent_ids]))
                mean_position_after = float(np.mean([da.population.agents[aid].position for aid in agent_ids]))
                n_buy = sum(1 for aid in agent_ids if actions[aid] == cfg.env.ACTION_BUY_MARKET)
                n_sell = sum(1 for aid in agent_ids if actions[aid] == cfg.env.ACTION_SELL_MARKET)
                n_hold = sum(1 for aid in agent_ids if actions[aid] == cfg.env.ACTION_HOLD)
                realized_vals = [float(da.population.agents[aid].realized_pnl - realized_before[aid]) for aid in agent_ids]
                reward_vals = [float(rewards.get(aid, 0.0)) for aid in agent_ids]
                env_step_rows.append(build_env_step_row(
                    algorithm=algorithm_name,
                    phase="eval_same_population",
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
                    n_executed=sum(1 for aid in agent_ids if da.population.agents[aid].position != positions_before[aid]),
                    n_trades_closed_cum=sum(da.population.agents[aid].n_trades_closed for aid in agent_ids),
                ))

        metrics = da.episode_metrics()
        same_action_frac, effective_n = _coordination_stats(step_actions, cfg.env.n_actions)
        record = build_episode_record(
            episode=episode,
            diversity_score=diversity_score,
            seed=seed,
            algorithm=algorithm_name,
            cfg=cfg,
            metrics=metrics,
            extra={
                "zi_baseline_trade_accuracy": zi_baseline_trade_accuracy,
                "zi_baseline_positive_pnl_frac": zi_baseline_positive_pnl_frac,
                "zi_baseline": zi_baseline_trade_accuracy,
                "beats_zi": metrics.get("trade_accuracy", 0.0) > zi_baseline_trade_accuracy,
                "same_action_frac": same_action_frac,
                "effective_N": effective_n,
                "eval_mode": "deterministic_argmax" if deterministic else "stochastic_sample",
                "eval_trade_accuracy": metrics.get("trade_accuracy", 0.0),
                "eval_mean_total_pnl": metrics.get("mean_total_pnl", 0.0),
                "eval_n_trades": metrics.get("n_trades", 0),
                "eval_mean_terminal_pnl": metrics.get("mean_terminal_pnl", 0.0),
            },
            agent_gammas=agent_gammas,
        )
        records.append(record)

    return records, sample_rows, env_step_rows


def build_run_settings(quick: bool, args) -> dict:
    if quick:
        ppo_cfg = PPOConfig(
            hidden_size=32,
            update_epochs=4,
            minibatch_size=128,
            rollout_episodes=5,
            use_agent_id_features=args.agent_id_features,
        )
        settings = {
            "run_name": "quick",
            "diversity_scores": [0.0, 0.5, 1.0],
            "n_agents": 50,
            "n_episodes": 50,
            "episode_steps": 150,
            "n_seeds": 1,
            "zi_episodes": 10,
            "eval_episodes": 50,
            "log_every": 1,
            "rolling_window": 2,
            "n_workers": 1,
            "market": MarketDynamics.stable(),
            "ppo_cfg": ppo_cfg,
        }
    else:
        ppo_cfg = PPOConfig(use_agent_id_features=args.agent_id_features)
        settings = {
            "run_name": "full",
            "diversity_scores": DIVERSITY_SCORES,
            "n_agents": N_AGENTS,
            "n_episodes": N_EPISODES,
            "episode_steps": EPISODE_STEPS,
            "n_seeds": N_SEEDS,
            "zi_episodes": ZI_EPISODES,
            "eval_episodes": EVAL_EPISODES,
            "log_every": LOG_EVERY,
            "rolling_window": ROLLING_WINDOW,
            "n_workers": cpu_count(),
            "market": MARKET,
            "ppo_cfg": ppo_cfg,
        }

    if args.workers is not None:
        settings["n_workers"] = args.workers

    max_workers = settings["n_seeds"] * len(settings["diversity_scores"])
    settings["n_workers"] = max(1, min(settings["n_workers"], max_workers))
    settings["log_every"] = max(1, min(settings["log_every"], settings["n_episodes"]))
    settings["rolling_window"] = max(1, min(settings["rolling_window"], settings["n_episodes"]))
    return settings


def make_cfg(settings: dict) -> HTMConfig:
    return HTMConfig(
        env=EnvConfig(
            n_agents=settings["n_agents"],
            episode_steps=settings["episode_steps"],
        ),
        market=settings["market"],
        ppo=settings["ppo_cfg"],
        log=LogConfig(level="INFO", save_to_file=True, save_plots=True),
    )


def run_training(
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
    set_global_seeds(seed)
    da = DoubleAuction(cfg, seed=seed)
    da.reset(diversity_score=diversity_score, seed=seed)
    agent_ids = list(da.population.agents.keys())
    agent_gammas = [da.population.agents[aid].gamma for aid in agent_ids]
    mean_gamma_pop = float(np.mean([da.population.agents[aid].gamma for aid in agent_ids]))
    ppo_cfg = dataclasses.replace(cfg.ppo, gamma=mean_gamma_pop)
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
                        "threshold": float(meta.get("threshold", 0.0)),
                        "gamma": float(meta.get("gamma", 0.0)),
                        "realized_pnl": float(meta.get("realized_pnl", 0.0)),
                        "n_trades_closed": int(meta.get("n_trades_closed", 0)),
                        "trade_accuracy_agent": float(meta.get("trade_accuracy", 0.0)),
                    })
                short_eval_records, _, _ = evaluate_trained_ppo(
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

    eval_records, sample_rows, env_step_rows = evaluate_trained_ppo(
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
        new_population_eval_records, _, _ = evaluate_trained_ppo(
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
) -> tuple[List[dict], List[dict], List[dict]]:
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
    return records, sample_rows, env_step_rows


def plot_learning_curves(
    records: List[dict],
    zi_baselines: Dict[float, float],
    save_path: Path,
    diversity_scores: List[float],
    rolling_window: int,
) -> None:
    if not records:
        return
    df = pd.DataFrame(records)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for d in diversity_scores:
        df_d = df[df["diversity_score"] == d]
        if df_d.empty:
            continue
        by_ep = df_d.groupby("episode")
        acc = by_ep["trade_accuracy"].mean().rolling(rolling_window, min_periods=1).mean()
        pnl = by_ep["mean_total_pnl"].mean().rolling(rolling_window, min_periods=1).mean()
        axes[0].plot(acc.index, acc.values, lw=2, label=f"PPO D={d:.1f}")
        axes[1].plot(pnl.index, pnl.values, lw=2, label=f"D={d:.1f}")
        if d in zi_baselines:
            axes[0].axhline(zi_baselines[d], color="gray", ls="--", lw=1, alpha=0.5)

    axes[0].set_title("PPO trade_accuracy")
    axes[0].set_xlabel("Epizod")
    axes[0].set_ylabel("trade_accuracy")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].set_title("PPO mean_total_pnl")
    axes[1].set_xlabel("Epizod")
    axes[1].set_ylabel("mean_total_pnl")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Wykres: {save_path}")


def log_final_summary(
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


def _train_worker(args: tuple) -> tuple[List[dict], List[dict], List[dict], List[dict], List[dict], List[dict], List[dict], Path]:
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

    try:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    cfg = copy.deepcopy(cfg)
    train_records, learning_curve_records, agent_diagnostics, eval_records, eval_new_population_records, sample_rows, env_step_rows, _, checkpoint_path = run_training(
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
        env_step_rows,
        checkpoint_path,
    )


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Szybki smoke test PPO")
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

    settings = build_run_settings(args.quick, args)
    cfg = make_cfg(settings)
    cfg.exp.n_eval_episodes = settings["eval_episodes"]
    cfg.exp.diversity_scores = list(settings["diversity_scores"])
    run_id, run_dir = prepare_run_dir(args.run_tag, args.run_id, args.run_dir)
    episodes_csv = run_dir / "episodes.csv"
    agents_sample_csv = run_dir / "agents_sample.csv"
    env_steps_csv = run_dir / "env_steps.csv"
    run_config_path = run_dir / "run_config.json"

    run_name = "ppo_quick" if args.quick else "ppo"
    log.info("=" * 70)
    log.info(
        f"{run_name} | N={cfg.env.n_agents} | D={settings['diversity_scores']} | "
        f"ep={settings['n_episodes']} | steps={cfg.env.episode_steps} | "
        f"seeds={settings['n_seeds']} | workers={settings['n_workers']} | "
        f"agent_id_features={cfg.ppo.use_agent_id_features}"
    )
    log.info("=" * 70)

    results_dir = PROJECT_ROOT / "results"
    plots_dir = PROJECT_ROOT / "plots"
    checkpoint_dir = results_dir / "checkpoints" / "ppo" / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if not run_config_path.exists():
        write_run_config(run_config_path, {
            "run_id": run_id,
            "run_tag": args.run_tag,
            "timestamp": run_id.split("_", 1)[1] if run_id.startswith("run_") else run_id,
            "algorithm": "PPO",
            "diversity_scores": settings["diversity_scores"],
            "n_seeds": settings["n_seeds"],
            "n_episodes": settings["n_episodes"],
            "n_agents": cfg.env.n_agents,
            "market_condition": {
                "eq_center": cfg.market.eq_center,
                "eq_spread": cfg.market.eq_spread,
                "drift_enabled": cfg.market.drift_enabled,
            },
            "eval_new_population": args.eval_new_population,
        })

    baseline_d = float(settings["diversity_scores"][0]) if settings["diversity_scores"] else 0.0
    zi_records, _, _ = evaluate_zi(cfg, diversity_score=baseline_d, n_episodes=settings["zi_episodes"], seed=42)
    append_rows(episodes_csv, EPISODE_FIELDS, _stamp_episode_rows(zi_records, run_id, "zi_baseline"))
    shared_zi_acc = float(np.mean([r["trade_accuracy"] for r in zi_records])) if zi_records else 0.0
    shared_zi_pos = float(np.mean([r["positive_pnl_frac"] for r in zi_records])) if zi_records else 0.0
    zi_acc = {d: shared_zi_acc for d in settings["diversity_scores"]}
    zi_pos = {d: shared_zi_pos for d in settings["diversity_scores"]}

    all_records: List[dict] = []
    all_eval_same_population_records: List[dict] = []
    all_eval_new_population_records: List[dict] = []
    all_env_step_rows: List[dict] = []
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
        f"Start PPO: {len(tasks)} zadań "
        f"({len(settings['diversity_scores'])} D × {settings['n_seeds']} seeds)"
    )

    if settings["n_workers"] == 1:
        iterator = map(_train_worker, tasks)
        pool = None
    else:
        pool = Pool(processes=settings["n_workers"])
        iterator = pool.imap_unordered(_train_worker, tasks)

    try:
        for i, worker_result in enumerate(iterator, start=1):
            train_records, _learning_curve_records, _agent_diagnostics, eval_records, eval_new_population_records, sample_rows, env_step_rows, ckpt = worker_result
            all_records.extend(train_records)
            all_eval_same_population_records.extend(eval_records)
            all_eval_new_population_records.extend(eval_new_population_records)
            all_env_step_rows.extend(env_step_rows)
            checkpoints.append(ckpt)
            append_rows(episodes_csv, EPISODE_FIELDS, _stamp_episode_rows(train_records, run_id, "train"))
            append_rows(episodes_csv, EPISODE_FIELDS, _stamp_episode_rows(eval_records, run_id, "eval_same_population"))
            append_rows(episodes_csv, EPISODE_FIELDS, _stamp_episode_rows(eval_new_population_records, run_id, "eval_new_population"))
            append_rows(agents_sample_csv, AGENT_SAMPLE_FIELDS, _stamp_sample_rows(sample_rows, run_id))
            append_rows(env_steps_csv, ENV_STEP_FIELDS, _stamp_sample_rows(env_step_rows, run_id))

            d_done = train_records[0]["diversity_score"] if train_records else "?"
            seed_done = train_records[0]["seed"] if train_records else "?"
            log.info(
                f"  Zakończono D={d_done} seed={seed_done} "
                f"({i}/{len(tasks)}) | train={len(all_records)} "
                f"eval_same={len(all_eval_same_population_records)} ckpt={ckpt.name}"
            )
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    plot_learning_curves(
        all_records,
        zi_acc,
        plots_dir / f"{run_name}_learning_curves.png",
        settings["diversity_scores"],
        settings["rolling_window"],
    )
    log_final_summary(
        all_records,
        all_eval_same_population_records,
        all_eval_new_population_records,
        zi_acc,
        settings["diversity_scores"],
        settings["n_episodes"],
    )

    total_rows = len(all_records) + len(all_eval_same_population_records) + len(all_eval_new_population_records) + len(zi_records)
    log.info(f"Wyniki: {episodes_csv} ({total_rows} wierszy)")
    log.info(f"Próbka agentów: {agents_sample_csv}")
    log.info(f"Agregaty środowiska: {env_steps_csv}")
    log.info(f"Checkpointy: {checkpoint_dir}")
    log.info(f"Czas: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
