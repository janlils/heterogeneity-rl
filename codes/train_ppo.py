"""
Runner treningowy shared-policy PPO dla HTM.

Uruchomienie:
    python -m codes.train_ppo
    python -m codes.train_ppo --quick
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
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
from codes.double_auction import DoubleAuction, run_zi_baseline
from codes.ppo import SharedPPOTrainer
from codes.rl_common import build_episode_record, set_global_seeds


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
N_AGENTS = 20
N_EPISODES = 500
N_SEEDS = 3
EPISODE_STEPS = 200
ZI_EPISODES = 30
EVAL_EPISODES = 30
LOG_EVERY = 25
ROLLING_WINDOW = 30
MARKET = MarketDynamics.random_eq()
PPO_CFG = PPOConfig()


def build_run_settings(quick: bool, args) -> dict:
    if quick:
        ppo_cfg = PPOConfig(
            hidden_size=32,
            update_epochs=4,
            minibatch_size=128,
            rollout_episodes=5,
            use_agent_id_features=args.agent_id_features,
        )
        return {
            "run_name": "quick",
            "diversity_scores": [0.0, 0.3, 0.5, 0.7, 0.9, 1.0],
            "n_agents": 20,
            "n_episodes": 50,
            "episode_steps": 150,
            "n_seeds": 1,
            "zi_episodes": 10,
            "eval_episodes": 50,
            "log_every": 1,
            "rolling_window": 2,
            "market": MarketDynamics.random_eq(),
            "ppo_cfg": ppo_cfg,
        }

    ppo_cfg = PPOConfig(use_agent_id_features=args.agent_id_features)
    return {
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
        "market": MARKET,
        "ppo_cfg": ppo_cfg,
    }


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


def compute_zi_baselines(
    cfg: HTMConfig,
    diversity_scores: List[float],
    n_episodes: int,
) -> tuple[Dict[float, float], Dict[float, float]]:
    zi_acc: Dict[float, float] = {}
    zi_pos: Dict[float, float] = {}
    log.info("ZI baseline:")
    for d in diversity_scores:
        r = run_zi_baseline(cfg, diversity_score=d, n_episodes=n_episodes, seed=42)
        zi_acc[d] = r["trade_accuracy"]["mean"]
        zi_pos[d] = r["positive_pnl_frac"]["mean"]
        log.info(
            f"  D={d:.1f} acc={zi_acc[d]:.3f} "
            f"pnl={r['mean_pnl']['mean']:.4f} term={r['mean_terminal_pnl']['mean']:.4f}"
        )
    return zi_acc, zi_pos


def run_training(
    diversity_score: float,
    n_episodes: int,
    seed: int,
    cfg: HTMConfig,
    zi_baseline_trade_accuracy: float,
    zi_baseline_positive_pnl_frac: float,
    checkpoint_dir: Path,
    log_every: int,
) -> tuple[List[dict], List[dict], SharedPPOTrainer, Path]:
    set_global_seeds(seed)
    da = DoubleAuction(cfg, seed=seed)
    da.reset(diversity_score=diversity_score, seed=seed)
    agent_ids = list(da.population.agents.keys())

    obs_dim = cfg.env.n_obs + (len(agent_ids) if cfg.ppo.use_agent_id_features else 0)
    trainer = SharedPPOTrainer(
        obs_dim=obs_dim,
        n_actions=cfg.env.n_actions,
        cfg=cfg.ppo,
        seed=seed,
        agent_ids=agent_ids,
    )
    rng = np.random.default_rng(seed + 10_000)
    records: List[dict] = []
    t0 = time.time()
    episode = 0

    while episode < n_episodes:
        rollout_episodes = min(cfg.ppo.rollout_episodes, n_episodes - episode)
        metrics_list = trainer.collect_rollout(
            da,
            agent_ids,
            rng,
            deterministic=False,
            rollout_episodes=rollout_episodes,
        )
        update_stats = trainer.update()

        for metrics in metrics_list:
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
            )
            records.append(record)
            episode += 1

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

    eval_records = evaluate_trained_ppo(
        da,
        trainer,
        diversity_score,
        seed,
        cfg,
        cfg.exp.n_eval_episodes,
        zi_baseline_trade_accuracy,
        zi_baseline_positive_pnl_frac,
    )

    return records, eval_records, trainer, checkpoint_path


def evaluate_trained_ppo(
    da: DoubleAuction,
    trainer: SharedPPOTrainer,
    diversity_score: float,
    seed: int,
    cfg: HTMConfig,
    n_eval_episodes: int,
    zi_baseline_trade_accuracy: float,
    zi_baseline_positive_pnl_frac: float,
    deterministic: bool = False,
) -> List[dict]:
    """
    Ewaluacja PPO na tej samej populacji i stanie środowiska po treningu.
    To jest odpowiednik evaluate_trained_sarsa(): resetuje tylko epizod,
    nie tworzy nowej populacji z innym seedem.

    Domyślnie używa stochastycznego sample z wyuczonej masked policy, bo PPO
    uczy rozkład akcji. deterministic=True zostaje jako diagnostyka argmax.
    """
    eval_records: List[dict] = []
    agent_ids = list(da.population.agents.keys())
    eval_rng = np.random.default_rng(seed + 70_000)

    for episode in range(n_eval_episodes):
        da.reset_episode()

        while not da.done:
            for aid in eval_rng.permutation(agent_ids):
                obs = da.get_observation(aid)
                action, _, _, mask = trainer.act_np(obs, aid, deterministic=deterministic)
                if not mask[action]:
                    raise RuntimeError(f"PPO eval selected illegal action {action} for {aid}")
                da.execute_single_action(aid, action)
            da.compute_step_rewards()

        m = da.episode_metrics()
        eval_records.append(build_episode_record(
            episode=episode,
            diversity_score=diversity_score,
            seed=seed,
            algorithm="PPO_EVAL_DETERMINISTIC" if deterministic else "PPO_EVAL_STOCHASTIC",
            cfg=cfg,
            metrics=m,
            extra={
                "eval_mode": "deterministic_argmax" if deterministic else "stochastic_sample",
                "zi_baseline_trade_accuracy": zi_baseline_trade_accuracy,
                "zi_baseline_positive_pnl_frac": zi_baseline_positive_pnl_frac,
                "zi_baseline": zi_baseline_trade_accuracy,
                "beats_zi": m.get("trade_accuracy", 0.0) > zi_baseline_trade_accuracy,
                "eval_trade_accuracy": m.get("trade_accuracy", 0.0),
                "eval_mean_total_pnl": m.get("mean_total_pnl", 0.0),
                "eval_n_trades": m.get("n_trades", 0),
                "eval_mean_terminal_pnl": m.get("mean_terminal_pnl", 0.0),
            },
        ))

    return eval_records


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
    eval_records: List[dict],
    zi_baselines: Dict[float, float],
    diversity_scores: List[float],
    n_episodes: int,
) -> None:
    if not records:
        return

    df = pd.DataFrame(records)
    eval_df = pd.DataFrame(eval_records) if eval_records else pd.DataFrame()
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


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Szybki smoke test PPO")
    parser.add_argument(
        "--agent-id-features",
        action="store_true",
        help="Doklej one-hot agent_id do obserwacji PPO",
    )
    args = parser.parse_args(argv)

    settings = build_run_settings(args.quick, args)
    cfg = make_cfg(settings)
    cfg.exp.n_eval_episodes = settings["eval_episodes"]

    run_name = "ppo_quick" if args.quick else "ppo"
    log.info("=" * 70)
    log.info(
        f"{run_name} | N={cfg.env.n_agents} | D={settings['diversity_scores']} | "
        f"ep={settings['n_episodes']} | steps={cfg.env.episode_steps} | "
        f"agent_id_features={cfg.ppo.use_agent_id_features}"
    )
    log.info("=" * 70)

    results_dir = PROJECT_ROOT / "results"
    plots_dir = PROJECT_ROOT / "plots"
    checkpoint_dir = results_dir / "checkpoints" / "ppo" / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    zi_acc, zi_pos = compute_zi_baselines(
        cfg,
        settings["diversity_scores"],
        settings["zi_episodes"],
    )

    all_records: List[dict] = []
    all_eval_records: List[dict] = []
    checkpoints: List[Path] = []
    t0 = time.time()

    for d in settings["diversity_scores"]:
        for seed in range(settings["n_seeds"]):
            train_records, eval_records, _, ckpt = run_training(
                diversity_score=d,
                n_episodes=settings["n_episodes"],
                seed=seed,
                cfg=cfg,
                zi_baseline_trade_accuracy=zi_acc[d],
                zi_baseline_positive_pnl_frac=zi_pos[d],
                checkpoint_dir=checkpoint_dir,
                log_every=settings["log_every"],
            )
            all_records.extend(train_records)
            all_eval_records.extend(eval_records)
            checkpoints.append(ckpt)
            log.info(
                f"  Zakończono D={d:.1f} seed={seed} | "
                f"train={len(all_records)} eval={len(all_eval_records)} ckpt={ckpt.name}"
            )

    train_csv = results_dir / f"{run_name}_results.csv"
    eval_csv = results_dir / f"{run_name}_eval_results.csv"
    pd.DataFrame(all_records).to_csv(train_csv, index=False)
    pd.DataFrame(all_eval_records).to_csv(eval_csv, index=False)
    plot_learning_curves(
        all_records,
        zi_acc,
        plots_dir / f"{run_name}_learning_curves.png",
        settings["diversity_scores"],
        settings["rolling_window"],
    )
    log_final_summary(
        all_records,
        all_eval_records,
        zi_acc,
        settings["diversity_scores"],
        settings["n_episodes"],
    )

    log.info(f"Wyniki: {train_csv} ({len(all_records)} wierszy)")
    log.info(f"Ewaluacja: {eval_csv} ({len(all_eval_records)} wierszy)")
    log.info(f"Checkpointy: {checkpoint_dir}")
    log.info(f"Czas: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
