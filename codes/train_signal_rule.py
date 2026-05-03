"""
Runner benchmarku SignalRule dla HTM.

To nie jest algorytm uczący się. To deterministyczna polityka regułowa:
  - BUY gdy prywatny sygnał jest dodatni i agent nie jest już long
  - SELL gdy prywatny sygnał jest ujemny i agent nie jest już short
  - HOLD w przeciwnym razie
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from multiprocessing import cpu_count
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
(PROJECT_ROOT / ".matplotlib_cache").mkdir(exist_ok=True)
(PROJECT_ROOT / ".cache").mkdir(exist_ok=True)

import numpy as np

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from codes.config import HTMConfig, EnvConfig, LogConfig
from codes.double_auction import DoubleAuction
from codes.evaluation import evaluate_same_population
from codes.experiment_runner import (
    compute_shared_zi_baseline,
    ensure_run_config,
    init_run_artifacts,
    worker_results,
)
from codes.experiment_settings import (
    DEFAULT_DIVERSITY_SCORES,
    DEFAULT_EPISODE_STEPS,
    DEFAULT_EVAL_EPISODES,
    DEFAULT_MARKET,
    DEFAULT_N_AGENTS,
    DEFAULT_N_EPISODES,
    DEFAULT_N_SEEDS,
    DEFAULT_ZI_EPISODES,
    build_signal_rule_settings,
)
from codes.results_store import (
    AGENT_EVAL_SUMMARY_FIELDS,
    AGENT_SAMPLE_FIELDS,
    DECISION_FEATURE_SUMMARY_FIELDS,
    ENV_STEP_FIELDS,
    EPISODE_FIELDS,
    append_rows,
)
from codes.rule_policies import SignalRulePolicy
from codes.rl_common import (
    aggregate_agent_eval_episode_rows,
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
        logging.FileHandler(PROJECT_ROOT / "logs" / "signal_rule.log", mode="w"),
    ],
)
log = logging.getLogger("htm.train_signal_rule")


DIVERSITY_SCORES = DEFAULT_DIVERSITY_SCORES
N_AGENTS = DEFAULT_N_AGENTS
N_EPISODES = DEFAULT_N_EPISODES
N_SEEDS = DEFAULT_N_SEEDS
EPISODE_STEPS = DEFAULT_EPISODE_STEPS
ZI_EPISODES = DEFAULT_ZI_EPISODES
EVAL_EPISODES = DEFAULT_EVAL_EPISODES
MARKET = DEFAULT_MARKET


def _stamp_episode_rows(rows: List[dict], run_id: str, phase: str) -> List[dict]:
    return [{**row, "run_id": run_id, "phase": phase} for row in rows]


def _stamp_sample_rows(rows: List[dict], run_id: str) -> List[dict]:
    return [{**row, "run_id": run_id} for row in rows]


def build_run_settings(quick: bool, args) -> dict:
    settings = build_signal_rule_settings(quick=quick, default_workers=cpu_count())
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


def make_cfg(settings: dict) -> HTMConfig:
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

    def extra_builder(metrics: dict, step_actions: List[np.ndarray]) -> dict:
        from codes.evaluation import coordination_stats

        same_action_frac, effective_n = coordination_stats(step_actions, cfg.env.n_actions)
        return {
            "zi_baseline_trade_accuracy": zi_baseline_trade_accuracy,
            "zi_baseline_positive_pnl_frac": zi_baseline_positive_pnl_frac,
            "zi_baseline": zi_baseline_trade_accuracy,
            "beats_zi": metrics.get("trade_accuracy", 0.0) > zi_baseline_trade_accuracy,
            "same_action_frac": same_action_frac,
            "effective_N": effective_n,
            "eval_mode": "rule_based",
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
        obs_by_agent: dict,
        actions: dict,
        _rewards: dict,
        _positions_before: dict,
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


def _worker(task: tuple) -> tuple:
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


def parse_args() -> argparse.Namespace:
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = build_run_settings(args.quick, args)
    cfg = make_cfg(settings)
    artifacts = init_run_artifacts(args.run_tag, args.run_id, args.run_dir)

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
        stamp_episode_rows=_stamp_episode_rows,
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

    t0 = time.time()
    all_eval_same_population_records: List[dict] = []
    all_eval_new_population_records: List[dict] = []

    for i, worker_result in enumerate(worker_results(tasks, _worker, settings["n_workers"]), start=1):
        eval_records, eval_new_records, sample_rows, agent_eval_rows, decision_feature_rows, env_step_rows = worker_result
        all_eval_same_population_records.extend(eval_records)
        all_eval_new_population_records.extend(eval_new_records)
        append_rows(artifacts.episodes_csv, EPISODE_FIELDS, _stamp_episode_rows(eval_records, artifacts.run_id, "eval_same_population"))
        append_rows(artifacts.episodes_csv, EPISODE_FIELDS, _stamp_episode_rows(eval_new_records, artifacts.run_id, "eval_new_population"))
        append_rows(artifacts.agents_sample_csv, AGENT_SAMPLE_FIELDS, _stamp_sample_rows(sample_rows, artifacts.run_id))
        append_rows(artifacts.agent_eval_summary_csv, AGENT_EVAL_SUMMARY_FIELDS, _stamp_sample_rows(agent_eval_rows, artifacts.run_id))
        append_rows(artifacts.decision_feature_summary_csv, DECISION_FEATURE_SUMMARY_FIELDS, _stamp_sample_rows(decision_feature_rows, artifacts.run_id))
        append_rows(artifacts.env_steps_csv, ENV_STEP_FIELDS, _stamp_sample_rows(env_step_rows, artifacts.run_id))
        d_done = eval_records[0]["diversity_score"] if eval_records else "?"
        seed_done = eval_records[0]["seed"] if eval_records else "?"
        log.info(
            f"  Zakończono D={d_done} seed={seed_done} "
            f"({i}/{len(tasks)}) | eval_same={len(all_eval_same_population_records)}"
        )

    total_rows = len(all_eval_same_population_records) + len(all_eval_new_population_records) + len(zi_records)
    log.info(f"Wyniki: {artifacts.episodes_csv} ({total_rows} wierszy)")
    log.info(f"Agent eval summary: {artifacts.agent_eval_summary_csv}")
    log.info(f"Decision feature summary: {artifacts.decision_feature_summary_csv}")
    log.info(f"Próbka agentów: {artifacts.agents_sample_csv}")
    log.info(f"Agregaty środowiska: {artifacts.env_steps_csv}")
    log.info(f"Czas: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
