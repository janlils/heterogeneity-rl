"""
Uruchamia benchmarki Deep SARSA, PPO, IPPO i SignalRule jednym poleceniem.

Przykłady:
    python -m codes.train_all --quick
    python -m codes.train_all --medium
    python -m codes.train_all --quick --episodes 20 --steps 200 --eval-episodes 10
    python -m codes.train_all --quick --agent-id-features
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from codes.experiment import init_run_artifacts
from codes.results import write_run_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_cli_benchmark_mode(args: argparse.Namespace) -> str:
    if getattr(args, "quick", False):
        return "quick"
    if getattr(args, "medium", False):
        return "medium"
    return "full"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--quick", action="store_true", help="Uruchom wybrane benchmarki w trybie quick.")
    mode_group.add_argument("--medium", action="store_true", help="Uruchom wybrane benchmarki w trybie pośrednim.")
    parser.add_argument(
        "--only",
        choices=["all", "sarsa", "ppo", "ippo", "signal_rule"],
        default="all",
        help="Który benchmark uruchomić.",
    )
    parser.add_argument("--episodes", type=int, help="Override liczby epizodów dla wszystkich algorytmów.")
    parser.add_argument("--steps", type=int, help="Override liczby kroków w epizodzie dla wszystkich algorytmów.")
    parser.add_argument("--seeds", type=int, help="Override liczby seedów dla wszystkich algorytmów.")
    parser.add_argument("--agents", type=int, help="Override liczby agentów dla wszystkich algorytmów.")
    parser.add_argument("--max-position", type=int, help="Override maksymalnej pozycji |position| per agent dla wszystkich algorytmów.")
    parser.add_argument("--gamma-spread", action="store_true", help="Włącz dodatkową heterogeniczność gamma obok sigma_i.")
    parser.add_argument("--fixed-gamma", type=float, help="Ustaw wspólną gammę wszystkich agentów, np. 0.90.")
    parser.add_argument("--transaction-cost", type=float, help="Koszt transakcyjny odejmowany od każdego wykonanego filla.")
    parser.add_argument("--zi-episodes", type=int, help="Override liczby epizodów ZI baseline dla wszystkich algorytmów.")
    parser.add_argument("--eval-episodes", type=int, help="Override liczby epizodów eval dla wszystkich algorytmów.")
    parser.add_argument("--workers", type=int, help="Override workerów dla SARSA, PPO, IPPO i SignalRule.")
    parser.add_argument(
        "--agent-id-features",
        action="store_true",
        help="Uruchom PPO z one-hot agent_id doklejonym do obserwacji.",
    )
    parser.add_argument("--run-tag", type=str, default="run", help="Krótki tag do nazwy folderu run.")
    parser.add_argument(
        "--parallel-algorithms",
        type=int,
        help="Maksymalna liczba algorytmów uruchamianych równolegle w train_all.",
    )
    return parser.parse_args(argv)


def _add_optional(cmd: List[str], flag: str, value) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def build_sarsa_cmd(args: argparse.Namespace, run_id: str, run_dir: Path) -> List[str]:
    cmd = [sys.executable, "-m", "codes.main", "train", "--algo", "sarsa"]
    if args.quick:
        cmd.append("--quick")
    elif args.medium:
        cmd.append("--medium")
    _add_optional(cmd, "--episodes", args.episodes)
    _add_optional(cmd, "--steps", args.steps)
    _add_optional(cmd, "--seeds", args.seeds)
    _add_optional(cmd, "--agents", args.agents)
    _add_optional(cmd, "--max-position", args.max_position)
    if args.gamma_spread:
        cmd.append("--gamma-spread")
    _add_optional(cmd, "--fixed-gamma", args.fixed_gamma)
    _add_optional(cmd, "--transaction-cost", args.transaction_cost)
    _add_optional(cmd, "--zi-episodes", args.zi_episodes)
    _add_optional(cmd, "--eval-episodes", args.eval_episodes)
    _add_optional(cmd, "--workers", args.workers)
    cmd.extend(["--run-tag", args.run_tag, "--run-id", run_id, "--run-dir", str(run_dir)])
    return cmd


def build_ppo_cmd(args: argparse.Namespace, run_id: str, run_dir: Path) -> List[str]:
    cmd = [sys.executable, "-m", "codes.main", "train", "--algo", "ppo"]
    if args.quick:
        cmd.append("--quick")
    elif args.medium:
        cmd.append("--medium")
    if args.agent_id_features:
        cmd.append("--agent-id-features")
    _add_optional(cmd, "--episodes", args.episodes)
    _add_optional(cmd, "--steps", args.steps)
    _add_optional(cmd, "--seeds", args.seeds)
    _add_optional(cmd, "--agents", args.agents)
    _add_optional(cmd, "--max-position", args.max_position)
    if args.gamma_spread:
        cmd.append("--gamma-spread")
    _add_optional(cmd, "--fixed-gamma", args.fixed_gamma)
    _add_optional(cmd, "--transaction-cost", args.transaction_cost)
    _add_optional(cmd, "--zi-episodes", args.zi_episodes)
    _add_optional(cmd, "--eval-episodes", args.eval_episodes)
    _add_optional(cmd, "--workers", args.workers)
    cmd.extend(["--run-tag", args.run_tag, "--run-id", run_id, "--run-dir", str(run_dir)])
    return cmd


def build_ippo_cmd(args: argparse.Namespace, run_id: str, run_dir: Path) -> List[str]:
    cmd = [sys.executable, "-m", "codes.main", "train", "--algo", "ippo"]
    if args.quick:
        cmd.append("--quick")
    elif args.medium:
        cmd.append("--medium")
    _add_optional(cmd, "--episodes", args.episodes)
    _add_optional(cmd, "--steps", args.steps)
    _add_optional(cmd, "--seeds", args.seeds)
    _add_optional(cmd, "--agents", args.agents)
    _add_optional(cmd, "--max-position", args.max_position)
    if args.gamma_spread:
        cmd.append("--gamma-spread")
    _add_optional(cmd, "--fixed-gamma", args.fixed_gamma)
    _add_optional(cmd, "--transaction-cost", args.transaction_cost)
    _add_optional(cmd, "--zi-episodes", args.zi_episodes)
    _add_optional(cmd, "--eval-episodes", args.eval_episodes)
    _add_optional(cmd, "--workers", args.workers)
    cmd.extend(["--run-tag", args.run_tag, "--run-id", run_id, "--run-dir", str(run_dir)])
    return cmd


def build_signal_rule_cmd(args: argparse.Namespace, run_id: str, run_dir: Path) -> List[str]:
    cmd = [sys.executable, "-m", "codes.main", "train", "--algo", "signal_rule"]
    if args.quick:
        cmd.append("--quick")
    elif args.medium:
        cmd.append("--medium")
    _add_optional(cmd, "--steps", args.steps)
    _add_optional(cmd, "--seeds", args.seeds)
    _add_optional(cmd, "--agents", args.agents)
    _add_optional(cmd, "--max-position", args.max_position)
    if args.gamma_spread:
        cmd.append("--gamma-spread")
    _add_optional(cmd, "--fixed-gamma", args.fixed_gamma)
    _add_optional(cmd, "--transaction-cost", args.transaction_cost)
    _add_optional(cmd, "--zi-episodes", args.zi_episodes)
    _add_optional(cmd, "--eval-episodes", args.eval_episodes)
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


def _parallel_default(args: argparse.Namespace) -> int:
    if args.parallel_algorithms is not None:
        return max(1, args.parallel_algorithms)
    if args.quick and args.only == "all":
        return 4
    return 1


def run_commands_parallel(commands: Sequence[Tuple[str, List[str]]], max_parallel: int) -> None:
    if not commands:
        return
    if max_parallel <= 1 or len(commands) == 1:
        for label, cmd in commands:
            run_command(label, cmd)
        return

    active: List[dict] = []
    pending = list(commands)
    heartbeat_interval_s = 10.0

    def _start(label: str, cmd: List[str]) -> None:
        log_path = PROJECT_ROOT / "logs" / f"{label.lower()}_train_all.out"
        print()
        print("=" * 78, flush=True)
        print(f"START {label}: {' '.join(cmd)}", flush=True)
        print(f"LOG {label}: {log_path}", flush=True)
        print("=" * 78, flush=True)
        handle = open(log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        active.append({
            "label": label,
            "cmd": cmd,
            "proc": proc,
            "handle": handle,
            "t0": time.time(),
            "log_path": log_path,
            "last_heartbeat": 0.0,
            "last_log_line": None,
        })

    def _read_last_nonempty_line(log_path: Path) -> Optional[str]:
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            return None
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return None
        return lines[-1]

    try:
        while pending or active:
            while pending and len(active) < max_parallel:
                label, cmd = pending.pop(0)
                _start(label, cmd)

            time.sleep(1.0)
            still_active: List[dict] = []
            for job in active:
                rc = job["proc"].poll()
                if rc is None:
                    now = time.time()
                    if now - job["last_heartbeat"] >= heartbeat_interval_s:
                        last_line = _read_last_nonempty_line(job["log_path"])
                        if last_line and last_line != job["last_log_line"]:
                            print(f"[{job['label']}] {last_line}", flush=True)
                            job["last_log_line"] = last_line
                        job["last_heartbeat"] = now
                    still_active.append(job)
                    continue
                job["handle"].close()
                elapsed = time.time() - job["t0"]
                print("=" * 78, flush=True)
                print(f"KONIEC {job['label']}: {elapsed:.0f}s | log: {job['log_path']}", flush=True)
                print("=" * 78, flush=True)
                if rc != 0:
                    raise subprocess.CalledProcessError(rc, job["cmd"])
            active = still_active
    finally:
        for job in active:
            try:
                if job["proc"].poll() is None:
                    job["proc"].terminate()
            except Exception:
                pass
            try:
                job["handle"].close()
            except Exception:
                pass


def _mean_or_none(df, column: str) -> Optional[float]:
    if column not in df.columns or df.empty:
        return None
    return float(df[column].mean())


def _fmt(value: Optional[float], width: int, decimals: int = 3) -> str:
    if value is None:
        return f"{'n/a':>{width}}"
    return f"{value:>{width}.{decimals}f}"


def print_eval_comparison(run_dir: Path) -> None:
    episodes_csv = run_dir / "episodes.csv"
    if not episodes_csv.exists():
        print()
        print("=" * 78, flush=True)
        print("PORÓWNANIE EVAL POMINIĘTE", flush=True)
        print(f"Brak {episodes_csv.relative_to(PROJECT_ROOT)}", flush=True)
        print("=" * 78, flush=True)
        return

    import pandas as pd

    episodes = pd.read_csv(episodes_csv)
    eval_df = episodes[episodes["phase"].astype(str).str.startswith("eval")].copy()
    algo_col = eval_df["algorithm"].astype(str)
    sarsa = eval_df[algo_col.str.contains("SARSA", case=False, na=False)]
    ippo = eval_df[eval_df["algorithm"].astype(str).str.contains("IPPO", case=False, na=False)]
    signal_rule = eval_df[eval_df["algorithm"].astype(str).str.contains("SIGNAL_RULE", case=False, na=False)]
    ppo = eval_df[eval_df["algorithm"].astype(str).str.match(r"^PPO", case=False, na=False)]
    ppo = ppo[~ppo["algorithm"].astype(str).str.contains("NO_IMPACT", case=False, na=False)]
    if sarsa.empty and ppo.empty and ippo.empty and signal_rule.empty:
        print()
        print("=" * 78, flush=True)
        print("PORÓWNANIE EVAL POMINIĘTE", flush=True)
        print(f"Brak odpowiednich rekordów eval w {episodes_csv}", flush=True)
        print("=" * 78, flush=True)
        return
    d_vals = sorted(
        set(sarsa["diversity_score"].unique())
        | set(ppo["diversity_score"].unique())
        | set(ippo["diversity_score"].unique())
        | set(signal_rule["diversity_score"].unique())
    )

    print()
    print("=" * 194, flush=True)
    print("PORÓWNANIE EVAL — SARSA vs PPO vs IPPO vs SignalRule", flush=True)
    print(
        f"{'D':>5} | {'ZI':>6} | {'SARSA acc':>10} | {'PPO acc':>7} | {'IPPO acc':>8} | {'Rule acc':>8} | "
        f"{'SARSA pnl':>10} | {'PPO pnl':>8} | {'IPPO pnl':>9} | {'Rule pnl':>9} | "
        f"{'SARSA Closed':>13} | {'PPO Closed':>10} | {'IPPO Closed':>11} | {'Rule Closed':>11}",
        flush=True,
    )
    print("-" * 194, flush=True)

    for d in d_vals:
        s_d = sarsa[sarsa["diversity_score"] == d]
        p_d = ppo[ppo["diversity_score"] == d]
        i_d = ippo[ippo["diversity_score"] == d]
        r_d = signal_rule[signal_rule["diversity_score"] == d]

        s_acc = _mean_or_none(s_d, "trade_accuracy")
        p_acc = _mean_or_none(p_d, "trade_accuracy")
        i_acc = _mean_or_none(i_d, "trade_accuracy")
        r_acc = _mean_or_none(r_d, "trade_accuracy")
        zi = _mean_or_none(s_d, "zi_baseline_trade_accuracy")
        if zi is None:
            zi = _mean_or_none(p_d, "zi_baseline_trade_accuracy")
        if zi is None:
            zi = _mean_or_none(i_d, "zi_baseline_trade_accuracy")
        if zi is None:
            zi = _mean_or_none(r_d, "zi_baseline_trade_accuracy")

        s_pnl = _mean_or_none(s_d, "mean_total_pnl")
        p_pnl = _mean_or_none(p_d, "mean_total_pnl")
        i_pnl = _mean_or_none(i_d, "mean_total_pnl")
        r_pnl = _mean_or_none(r_d, "mean_total_pnl")
        s_closed = _mean_or_none(s_d, "n_trades_closed")
        p_closed = _mean_or_none(p_d, "n_trades_closed")
        i_closed = _mean_or_none(i_d, "n_trades_closed")
        r_closed = _mean_or_none(r_d, "n_trades_closed")

        print(
            f"{d:5.1f} | {_fmt(zi, 6)} | {_fmt(s_acc, 10)} | {_fmt(p_acc, 7)} | {_fmt(i_acc, 8)} | {_fmt(r_acc, 8)} | "
            f"{_fmt(s_pnl, 10, 4)} | {_fmt(p_pnl, 8, 4)} | {_fmt(i_pnl, 9, 4)} | {_fmt(r_pnl, 9, 4)} | "
            f"{_fmt(s_closed, 13, 1)} | {_fmt(p_closed, 10, 1)} | {_fmt(i_closed, 11, 1)} | {_fmt(r_closed, 11, 1)}",
            flush=True,
        )

    print("-" * 194, flush=True)
    print(f"Run folder: {run_dir.relative_to(PROJECT_ROOT)}", flush=True)
    print(f"Episodes CSV: {episodes_csv.relative_to(PROJECT_ROOT)}", flush=True)
    print("=" * 194, flush=True)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    total_t0 = time.time()
    artifacts = init_run_artifacts(args.run_tag, None, None)
    run_id = artifacts.run_id
    run_dir = artifacts.run_dir
    write_run_config(run_dir / "train_all_config.json", {
        "run_id": artifacts.run_id,
        "run_tag": args.run_tag,
        "timestamp": run_id.split("_", 1)[1] if run_id.startswith("run_") else run_id,
        "algorithm": "train_all",
        "mode": resolve_cli_benchmark_mode(args),
        "quick": args.quick,
        "medium": args.medium,
        "only": args.only,
        "episodes": args.episodes,
        "steps": args.steps,
        "seeds": args.seeds,
        "agents": args.agents,
        "max_position": args.max_position,
        "gamma_spread": args.gamma_spread,
        "fixed_gamma": args.fixed_gamma,
        "transaction_cost": args.transaction_cost,
        "zi_episodes": args.zi_episodes,
        "eval_episodes": args.eval_episodes,
        "workers": args.workers,
        "parallel_algorithms": _parallel_default(args),
        "agent_id_features": args.agent_id_features,
    })

    commands: List[Tuple[str, List[str]]] = []
    if args.only in {"all", "sarsa"}:
        commands.append(("Deep SARSA", build_sarsa_cmd(args, run_id, run_dir)))
    if args.only in {"all", "ppo"}:
        commands.append(("PPO", build_ppo_cmd(args, run_id, run_dir)))
    if args.only in {"all", "ippo"}:
        commands.append(("IPPO", build_ippo_cmd(args, run_id, run_dir)))
    if args.only in {"all", "signal_rule"}:
        commands.append(("SignalRule", build_signal_rule_cmd(args, run_id, run_dir)))

    run_commands_parallel(commands, _parallel_default(args))

    print_eval_comparison(run_dir)

    print()
    print("=" * 78, flush=True)
    print(f"WSZYSTKIE WYBRANE BENCHMARKI ZAKOŃCZONE: {time.time() - total_t0:.0f}s", flush=True)
    print(f"Wyniki: {run_dir.relative_to(PROJECT_ROOT)}/episodes.csv", flush=True)
    print(f"Próbka: {run_dir.relative_to(PROJECT_ROOT)}/agents_sample.csv", flush=True)
    print("=" * 78, flush=True)


if __name__ == "__main__":
    main()
