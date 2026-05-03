"""
Runner treningowy MAPPO dla HTM.

MAPPO w tej wersji:
  - decentralized actor na lokalnej obserwacji
  - centralized critic na lokalnej obserwacji + agregatach rynku
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import logging
import os
import sys
import time
from multiprocessing import cpu_count
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

from codes.config import HTMConfig, EnvConfig, LogConfig, PPOConfig
from codes.double_auction import DoubleAuction
from codes.evaluation import coordination_stats, evaluate_same_population
from codes.experiment_runner import init_run_artifacts, run_ppo_experiment
from codes.experiment_settings import (
    DEFAULT_DIVERSITY_SCORES,
    DEFAULT_EPISODE_STEPS,
    DEFAULT_EVAL_EPISODES,
    DEFAULT_LOG_EVERY,
    DEFAULT_MARKET,
    DEFAULT_N_AGENTS,
    DEFAULT_N_EPISODES,
    DEFAULT_N_SEEDS,
    DEFAULT_ROLLING_WINDOW,
    DEFAULT_ZI_EPISODES,
    build_mappo_settings,
)
from codes.evaluate_policies import evaluate_policy
from codes.ppo import MAPPOTrainer
from codes.rl_common import (
    aggregate_agent_eval_episode_rows,
    build_episode_record,
    finalize_decision_feature_summary,
    init_decision_feature_stats,
    set_global_seeds,
    update_decision_feature_stats,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "logs" / "mappo.log", mode="w"),
    ],
)
log = logging.getLogger("htm.train_mappo")


DIVERSITY_SCORES = DEFAULT_DIVERSITY_SCORES
N_AGENTS = DEFAULT_N_AGENTS
N_EPISODES = DEFAULT_N_EPISODES
N_SEEDS = DEFAULT_N_SEEDS
EPISODE_STEPS = DEFAULT_EPISODE_STEPS
ZI_EPISODES = DEFAULT_ZI_EPISODES
EVAL_EPISODES = DEFAULT_EVAL_EPISODES
LOG_EVERY = DEFAULT_LOG_EVERY
ROLLING_WINDOW = DEFAULT_ROLLING_WINDOW
MARKET = DEFAULT_MARKET
PPO_CFG = PPOConfig()


def _stamp_episode_rows(rows: List[dict], run_id: str, phase: str) -> List[dict]:
    return [{**row, "run_id": run_id, "phase": phase} for row in rows]


def _stamp_sample_rows(rows: List[dict], run_id: str) -> List[dict]:
    return [{**row, "run_id": run_id} for row in rows]


def evaluate_trained_mappo_same_population(
    da: DoubleAuction,
    trainer: MAPPOTrainer,
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
        "MAPPO_EVAL_SAME_POPULATION_DETERMINISTIC"
        if deterministic else
        "MAPPO_EVAL_SAME_POPULATION"
    )

    def action_selector(aid: str, obs: np.ndarray) -> int:
        trainer.set_global_state(da.get_global_state())
        action, _, _, _ = trainer.act_np(obs, aid, deterministic=deterministic)
        return int(action)

    def extra_builder(metrics: dict, step_actions: List[np.ndarray]) -> dict:
        same_action_frac, effective_n = coordination_stats(step_actions, cfg.env.n_actions)
        return {
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
        }

    agent_eval_rows: List[dict] = []
    feature_stats = init_decision_feature_stats()
    alignment_counts = {aid: {"aligned": 0, "directional": 0, "total": 0} for aid in da.agent_ids}

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
            if action in (1, 2):
                alignment_counts[aid]["directional"] += 1
                signal = float(obs[0])
                if (signal > 0 and action == 1) or (signal < 0 and action == 2):
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
                "buy_frac": counts[1] / total_actions,
                "sell_frac": counts[2] / total_actions,
                "hold_frac": counts[0] / total_actions,
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
        collect_coordination=True,
        episode_end_callback=episode_end_callback,
        step_callback=step_callback,
    )
    feature_rows = [finalize_decision_feature_summary(
        feature_stats,
        algorithm=algorithm_name,
        phase="eval_same_population",
        diversity_score=diversity_score,
        seed=seed,
    )]
    return records, sample_rows, aggregate_agent_eval_episode_rows(agent_eval_rows), feature_rows, env_step_rows


def build_run_settings(quick: bool, args) -> dict:
    settings = build_mappo_settings(quick=quick, default_workers=cpu_count())
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
) -> tuple[List[dict], List[dict], List[dict], List[dict], List[dict], List[dict], List[dict], MAPPOTrainer, Path]:
    set_global_seeds(seed)
    da = DoubleAuction(cfg, seed=seed)
    da.reset(diversity_score=diversity_score, seed=seed)
    agent_ids = list(da.population.agents.keys())
    agent_gammas = [da.population.agents[aid].gamma for aid in agent_ids]
    ppo_cfg = dataclasses.replace(cfg.ppo, use_agent_id_features=False)
    cfg.ppo = ppo_cfg

    actor_obs_dim = cfg.env.n_obs
    critic_obs_dim = cfg.env.n_obs + len(da.get_global_state())
    trainer = MAPPOTrainer(
        actor_obs_dim=actor_obs_dim,
        critic_obs_dim=critic_obs_dim,
        n_actions=cfg.env.n_actions,
        cfg=ppo_cfg,
        seed=seed,
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
                algorithm="MAPPO",
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
                short_eval_records, _, _, _, _ = evaluate_trained_mappo(
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

    checkpoint_path = checkpoint_dir / f"mappo_D{diversity_score:.1f}_seed{seed}.pt"
    trainer.save(checkpoint_path, episode=n_episodes)

    eval_records, sample_rows, agent_eval_rows, decision_feature_rows, env_step_rows = evaluate_trained_mappo(
        da,
        trainer,
        diversity_score,
        seed,
        cfg,
        cfg.exp.n_eval_episodes,
        zi_baseline_trade_accuracy,
        zi_baseline_positive_pnl_frac,
        deterministic=True,
        algorithm_name="MAPPO_EVAL_SAME_POPULATION_DETERMINISTIC",
        same_population=True,
    )
    new_population_eval_records: List[dict] = []
    if eval_new_population:
        new_population_eval_records, _, _, _, _ = evaluate_trained_mappo(
            da,
            trainer,
            diversity_score,
            seed + 1000,
            cfg,
            cfg.exp.n_eval_episodes,
            zi_baseline_trade_accuracy,
            zi_baseline_positive_pnl_frac,
            deterministic=True,
            algorithm_name="MAPPO_EVAL_NEW_POPULATION_DETERMINISTIC",
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


def evaluate_trained_mappo(
    da: DoubleAuction,
    trainer: MAPPOTrainer,
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
        return evaluate_trained_mappo_same_population(
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
        "MAPPO_EVAL_DETERMINISTIC" if deterministic else "MAPPO_EVAL_STOCHASTIC"
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
        axes[0].plot(acc.index, acc.values, lw=2, label=f"MAPPO D={d:.1f}")
        axes[1].plot(pnl.index, pnl.values, lw=2, label=f"D={d:.1f}")
        if d in zi_baselines:
            axes[0].axhline(zi_baselines[d], color="gray", ls="--", lw=1, alpha=0.5)

    axes[0].set_title("MAPPO trade_accuracy")
    axes[0].set_xlabel("Epizod")
    axes[0].set_ylabel("trade_accuracy")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].set_title("MAPPO mean_total_pnl")
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
    log.info(f"PODSUMOWANIE MAPPO — ostatnie {final_window} epizodów, uśrednione po seedach")
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

    _log_eval_block("EWALUACJA MAPPO — ta sama populacja, deterministic argmax", eval_df)
    _log_eval_block("EWALUACJA MAPPO — nowa populacja, seed+1000, deterministic argmax", eval_new_df)


def _train_worker(args: tuple) -> tuple[List[dict], List[dict], List[dict], List[dict], List[dict], List[dict], List[dict], Path]:
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
    train_records, learning_curve_records, agent_diagnostics, eval_records, eval_new_population_records, sample_rows, agent_eval_rows, decision_feature_rows, env_step_rows, _, checkpoint_path = run_training(
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


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Szybki smoke test MAPPO")
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
        help="Dodatkowo uruchom osobny eval MAPPO na nowej populacji z seedem przesuniętym o 1000.",
    )
    parser.add_argument("--run-tag", type=str, default="run", help="Krótki tag do nazwy folderu run.")
    parser.add_argument("--run-id", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=str, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    settings = build_run_settings(args.quick, args)
    cfg = make_cfg(settings)
    cfg.exp.n_eval_episodes = settings["eval_episodes"]
    cfg.exp.diversity_scores = list(settings["diversity_scores"])
    artifacts = init_run_artifacts(args.run_tag, args.run_id, args.run_dir)
    run_ppo_experiment(
        log=log,
        project_root=PROJECT_ROOT,
        args=args,
        settings=settings,
        cfg=cfg,
        artifacts=artifacts,
        worker_fn=_train_worker,
        stamp_episode_rows=_stamp_episode_rows,
        stamp_sample_rows=_stamp_sample_rows,
        plot_learning_curves=plot_learning_curves,
        log_final_summary=log_final_summary,
        algorithm_label="MAPPO",
        artifact_stem="mappo_quick" if args.quick else "mappo",
    )


if __name__ == "__main__":
    main()
