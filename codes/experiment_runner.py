"""
Łagodna warstwa wspólnej orkiestracji eksperymentów.

Nie ukrywa treningu SARSA/PPO pod jednym interfejsem. Daje tylko wspólne
helpery dla:
  - przygotowania folderu run,
  - zapisu run_config,
  - liczenia wspólnego ZI baseline,
  - obsługi puli workerów.
"""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, TypeVar

import numpy as np

from codes.config import HTMConfig
from codes.evaluate_policies import evaluate_zi
from codes.results_store import AGENT_EVAL_SUMMARY_FIELDS, AGENT_SAMPLE_FIELDS, DECISION_FEATURE_SUMMARY_FIELDS, ENV_STEP_FIELDS, EPISODE_FIELDS, append_rows, prepare_run_dir, write_run_config


T = TypeVar("T")
R = TypeVar("R")


@dataclass(frozen=True)
class RunArtifacts:
    run_id: str
    run_dir: Path
    episodes_csv: Path
    agents_sample_csv: Path
    env_steps_csv: Path
    agent_eval_summary_csv: Path
    decision_feature_summary_csv: Path
    run_config_path: Path


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
            "psi": cfg.market.psi,
            "kappa": cfg.market.kappa,
            "nu": cfg.market.nu,
            "stress_low": cfg.market.stress_low,
            "crisis_prob": cfg.market.crisis_prob,
            "k_impact": cfg.env.k_impact,
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
        f"Market: v1 | psi={cfg.market.psi:.2f} | kappa={cfg.market.kappa:.2f} | "
        f"stress_low={cfg.market.stress_low:.3f} | k_impact={cfg.env.k_impact:.3f}"
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
        append_rows(episodes_csv, EPISODE_FIELDS, stamp_episode_rows(task_records, run_id, "train"))
        append_rows(episodes_csv, EPISODE_FIELDS, stamp_episode_rows(eval_same_population_records, run_id, "eval_same_population"))
        append_rows(episodes_csv, EPISODE_FIELDS, stamp_episode_rows(eval_new_population_records, run_id, "eval_new_population"))
        append_rows(agents_sample_csv, AGENT_SAMPLE_FIELDS, stamp_sample_rows(sample_rows, run_id))
        append_rows(agent_eval_summary_csv, AGENT_EVAL_SUMMARY_FIELDS, stamp_sample_rows(agent_eval_rows, run_id))
        append_rows(decision_feature_summary_csv, DECISION_FEATURE_SUMMARY_FIELDS, stamp_sample_rows(decision_feature_rows, run_id))
        append_rows(env_steps_csv, ENV_STEP_FIELDS, stamp_sample_rows(env_step_rows, run_id))
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
    checkpoint_dir = results_dir / "checkpoints" / "ppo" / run_name
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
        append_rows(episodes_csv, EPISODE_FIELDS, stamp_episode_rows(train_records, run_id, "train"))
        append_rows(episodes_csv, EPISODE_FIELDS, stamp_episode_rows(eval_records, run_id, "eval_same_population"))
        append_rows(episodes_csv, EPISODE_FIELDS, stamp_episode_rows(eval_new_population_records, run_id, "eval_new_population"))
        append_rows(agents_sample_csv, AGENT_SAMPLE_FIELDS, stamp_sample_rows(sample_rows, run_id))
        append_rows(agent_eval_summary_csv, AGENT_EVAL_SUMMARY_FIELDS, stamp_sample_rows(agent_eval_rows, run_id))
        append_rows(decision_feature_summary_csv, DECISION_FEATURE_SUMMARY_FIELDS, stamp_sample_rows(decision_feature_rows, run_id))
        append_rows(env_steps_csv, ENV_STEP_FIELDS, stamp_sample_rows(env_step_rows, run_id))

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
