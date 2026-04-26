"""
Uruchamia benchmarki Deep SARSA i PPO jednym poleceniem.

Przykłady:
    python -m codes.train_all --quick
    python -m codes.train_all --quick --episodes 20 --steps 200 --eval-episodes 10
    python -m codes.train_all --quick --agent-id-features
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

from codes.results_store import latest_run_dir, prepare_run_dir, write_run_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Uruchom oba algorytmy w trybie quick.")
    parser.add_argument(
        "--only",
        choices=["all", "sarsa", "ppo"],
        default="all",
        help="Który benchmark uruchomić.",
    )
    parser.add_argument("--episodes", type=int, help="Override epizodów dla SARSA.")
    parser.add_argument("--steps", type=int, help="Override kroków w epizodzie dla SARSA.")
    parser.add_argument("--seeds", type=int, help="Override seedów dla SARSA.")
    parser.add_argument("--agents", type=int, help="Override agentów dla SARSA.")
    parser.add_argument("--zi-episodes", type=int, help="Override epizodów ZI baseline dla SARSA.")
    parser.add_argument("--eval-episodes", type=int, help="Override epizodów eval dla SARSA.")
    parser.add_argument("--workers", type=int, help="Override workerów dla SARSA i PPO.")
    parser.add_argument(
        "--agent-id-features",
        action="store_true",
        help="Uruchom PPO z one-hot agent_id doklejonym do obserwacji.",
    )
    parser.add_argument("--run-tag", type=str, default="run", help="Krótki tag do nazwy folderu run.")
    return parser.parse_args()


def _add_optional(cmd: List[str], flag: str, value) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def build_sarsa_cmd(args: argparse.Namespace, run_id: str, run_dir: Path) -> List[str]:
    cmd = [sys.executable, "-m", "codes.train_deep_sarsa"]
    if args.quick:
        cmd.append("--quick")
    _add_optional(cmd, "--episodes", args.episodes)
    _add_optional(cmd, "--steps", args.steps)
    _add_optional(cmd, "--seeds", args.seeds)
    _add_optional(cmd, "--agents", args.agents)
    _add_optional(cmd, "--zi-episodes", args.zi_episodes)
    _add_optional(cmd, "--eval-episodes", args.eval_episodes)
    _add_optional(cmd, "--workers", args.workers)
    cmd.extend(["--run-tag", args.run_tag, "--run-id", run_id, "--run-dir", str(run_dir)])
    return cmd


def build_ppo_cmd(args: argparse.Namespace, run_id: str, run_dir: Path) -> List[str]:
    cmd = [sys.executable, "-m", "codes.train_ppo"]
    if args.quick:
        cmd.append("--quick")
    if args.agent_id_features:
        cmd.append("--agent-id-features")
    _add_optional(cmd, "--workers", args.workers)
    cmd.extend(["--run-tag", args.run_tag, "--run-id", run_id, "--run-dir", str(run_dir)])
    return cmd


def run_command(label: str, cmd: List[str]) -> None:
    print()
    print("=" * 78, flush=True)
    print(f"START {label}: {' '.join(cmd)}", flush=True)
    print("=" * 78, flush=True)
    t0 = time.time()
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    print("=" * 78, flush=True)
    print(f"KONIEC {label}: {time.time() - t0:.0f}s", flush=True)
    print("=" * 78, flush=True)


def _mean_or_none(df, column: str) -> Optional[float]:
    if column not in df.columns or df.empty:
        return None
    return float(df[column].mean())


def _fmt(value: Optional[float], width: int, decimals: int = 3) -> str:
    if value is None:
        return f"{'n/a':>{width}}"
    return f"{value:>{width}.{decimals}f}"


def print_eval_comparison(quick: bool) -> None:
    run_dir = latest_run_dir()
    episodes_csv = None if run_dir is None else run_dir / "episodes.csv"
    if episodes_csv is None or not episodes_csv.exists():
        print()
        print("=" * 78, flush=True)
        print("PORÓWNANIE EVAL POMINIĘTE", flush=True)
        print("Brak results/run_*/episodes.csv", flush=True)
        print("=" * 78, flush=True)
        return

    import pandas as pd

    episodes = pd.read_csv(episodes_csv)
    eval_df = episodes[episodes["phase"].astype(str).str.startswith("eval")].copy()
    sarsa = eval_df[eval_df["algorithm"].astype(str).str.contains("SARSA", case=False, na=False)]
    ppo = eval_df[eval_df["algorithm"].astype(str).str.contains("PPO", case=False, na=False)]
    ppo = ppo[~ppo["algorithm"].astype(str).str.contains("NO_IMPACT", case=False, na=False)]
    if sarsa.empty or ppo.empty:
        print()
        print("=" * 78, flush=True)
        print("PORÓWNANIE EVAL POMINIĘTE", flush=True)
        print(f"Brak odpowiednich rekordów eval w {episodes_csv}", flush=True)
        print("=" * 78, flush=True)
        return
    d_vals = sorted(set(sarsa["diversity_score"].unique()) | set(ppo["diversity_score"].unique()))

    print()
    print("=" * 118, flush=True)
    print("PORÓWNANIE EVAL — SARSA vs PPO", flush=True)
    print(
        f"{'D':>5} | {'ZI':>6} | {'SARSA acc':>9} | {'PPO acc':>7} | {'Δ acc':>7} | "
        f"{'SARSA pnl':>9} | {'PPO pnl':>8} | {'SARSA term':>10} | {'PPO term':>8} | "
        f"{'SARSA Closed':>12} | {'PPO Closed':>10}",
        flush=True,
    )
    print("-" * 118, flush=True)

    for d in d_vals:
        s_d = sarsa[sarsa["diversity_score"] == d]
        p_d = ppo[ppo["diversity_score"] == d]

        s_acc = _mean_or_none(s_d, "trade_accuracy")
        p_acc = _mean_or_none(p_d, "trade_accuracy")
        delta = None if s_acc is None or p_acc is None else p_acc - s_acc
        zi = _mean_or_none(s_d, "zi_baseline_trade_accuracy")
        if zi is None:
            zi = _mean_or_none(p_d, "zi_baseline_trade_accuracy")

        s_pnl = _mean_or_none(s_d, "mean_total_pnl")
        p_pnl = _mean_or_none(p_d, "mean_total_pnl")
        s_term = _mean_or_none(s_d, "mean_terminal_pnl")
        p_term = _mean_or_none(p_d, "mean_terminal_pnl")
        s_closed = _mean_or_none(s_d, "n_trades_closed")
        p_closed = _mean_or_none(p_d, "n_trades_closed")

        print(
            f"{d:5.1f} | {_fmt(zi, 6)} | {_fmt(s_acc, 9)} | {_fmt(p_acc, 7)} | "
            f"{_fmt(delta, 7)} | {_fmt(s_pnl, 9, 4)} | {_fmt(p_pnl, 8, 4)} | "
            f"{_fmt(s_term, 10, 4)} | {_fmt(p_term, 8, 4)} | "
            f"{_fmt(s_closed, 12, 1)} | {_fmt(p_closed, 10, 1)}",
            flush=True,
        )

    print("-" * 118, flush=True)
    print(f"Run folder: {run_dir.relative_to(PROJECT_ROOT)}", flush=True)
    print(f"Episodes CSV: {episodes_csv.relative_to(PROJECT_ROOT)}", flush=True)
    print("=" * 118, flush=True)


def main() -> None:
    args = parse_args()
    total_t0 = time.time()
    run_id, run_dir = prepare_run_dir(args.run_tag)
    write_run_config(run_dir / "run_config.json", {
        "run_id": run_id,
        "run_tag": args.run_tag,
        "timestamp": run_id.split("_", 1)[1] if run_id.startswith("run_") else run_id,
        "algorithm": "train_all",
        "quick": args.quick,
        "only": args.only,
        "episodes": args.episodes,
        "steps": args.steps,
        "seeds": args.seeds,
        "agents": args.agents,
        "zi_episodes": args.zi_episodes,
        "eval_episodes": args.eval_episodes,
        "workers": args.workers,
        "agent_id_features": args.agent_id_features,
    })

    if args.only in {"all", "sarsa"}:
        run_command("Deep SARSA", build_sarsa_cmd(args, run_id, run_dir))
    if args.only in {"all", "ppo"}:
        run_command("PPO", build_ppo_cmd(args, run_id, run_dir))

    print_eval_comparison(args.quick)

    print()
    print("=" * 78, flush=True)
    print(f"WSZYSTKIE WYBRANE BENCHMARKI ZAKOŃCZONE: {time.time() - total_t0:.0f}s", flush=True)
    print(f"Wyniki: {run_dir.relative_to(PROJECT_ROOT)}/episodes.csv", flush=True)
    print(f"Próbka: {run_dir.relative_to(PROJECT_ROOT)}/agents_sample.csv", flush=True)
    print("=" * 78, flush=True)


if __name__ == "__main__":
    main()
