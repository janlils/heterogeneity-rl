from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

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

from codes.results_store import latest_run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analiza debugowego epizodu HTM.")
    parser.add_argument("--run-dir", type=str, help="Ścieżka do results/run_*.")
    parser.add_argument("--algorithm", type=str, help="Filtruj po algorithm.")
    parser.add_argument("--phase", type=str, help="Filtruj po phase.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--diversity-score", type=float, default=1.0)
    parser.add_argument("--episode", type=int, default=0)
    return parser.parse_args()


def _pick_run_dir(run_dir_arg: str | None) -> Path:
    if run_dir_arg:
        run_dir = Path(run_dir_arg)
    else:
        run_dir = latest_run_dir()
        if run_dir is None:
            raise FileNotFoundError("Brak results/run_*/")
    if not run_dir.exists():
        raise FileNotFoundError(f"Brak katalogu run: {run_dir}")
    return run_dir


def _corr(a: pd.Series, b: pd.Series) -> float:
    joined = pd.concat([a, b], axis=1).dropna()
    if len(joined) < 2:
        return 0.0
    if joined.iloc[:, 0].std() < 1e-12 or joined.iloc[:, 1].std() < 1e-12:
        return 0.0
    return float(joined.iloc[:, 0].corr(joined.iloc[:, 1]))


def _decision_alignment(df_agents: pd.DataFrame) -> dict:
    actionable = df_agents[df_agents["action_name"].isin(["BUY", "SELL"])].copy()
    if actionable.empty:
        return {
            "actionable_rows": 0,
            "signal_action_alignment": 0.0,
            "buy_given_positive_signal": 0.0,
            "sell_given_negative_signal": 0.0,
            "reward_when_aligned": 0.0,
            "reward_when_misaligned": 0.0,
            "realized_when_aligned": 0.0,
            "realized_when_misaligned": 0.0,
        }

    actionable["aligned"] = (
        ((actionable["signal_i"] > 0) & (actionable["action_name"] == "BUY"))
        | ((actionable["signal_i"] < 0) & (actionable["action_name"] == "SELL"))
    )
    pos_sig = actionable[actionable["signal_i"] > 0]
    neg_sig = actionable[actionable["signal_i"] < 0]
    return {
        "actionable_rows": int(len(actionable)),
        "signal_action_alignment": float(actionable["aligned"].mean()),
        "buy_given_positive_signal": float((pos_sig["action_name"] == "BUY").mean()) if not pos_sig.empty else 0.0,
        "sell_given_negative_signal": float((neg_sig["action_name"] == "SELL").mean()) if not neg_sig.empty else 0.0,
        "reward_when_aligned": float(actionable.loc[actionable["aligned"], "reward_this_step"].mean()) if actionable["aligned"].any() else 0.0,
        "reward_when_misaligned": float(actionable.loc[~actionable["aligned"], "reward_this_step"].mean()) if (~actionable["aligned"]).any() else 0.0,
        "realized_when_aligned": float(actionable.loc[actionable["aligned"], "realized_pnl_this_step"].mean()) if actionable["aligned"].any() else 0.0,
        "realized_when_misaligned": float(actionable.loc[~actionable["aligned"], "realized_pnl_this_step"].mean()) if (~actionable["aligned"]).any() else 0.0,
    }


def _sigma_bucket_summary(df_agents: pd.DataFrame) -> pd.DataFrame:
    actionable = df_agents[df_agents["action_name"].isin(["BUY", "SELL"])].copy()
    if actionable.empty:
        return pd.DataFrame()
    actionable["aligned"] = (
        ((actionable["signal_i"] > 0) & (actionable["action_name"] == "BUY"))
        | ((actionable["signal_i"] < 0) & (actionable["action_name"] == "SELL"))
    )
    actionable["sigma_bucket"] = pd.qcut(
        actionable["sigma_i"],
        q=min(4, actionable["sigma_i"].nunique()),
        duplicates="drop",
    )
    summary = (
        actionable.groupby("sigma_bucket", observed=False)
        .agg(
            n=("agent_id", "size"),
            alignment=("aligned", "mean"),
            mean_reward=("reward_this_step", "mean"),
            mean_realized=("realized_pnl_this_step", "mean"),
            mean_signal=("signal_i", "mean"),
        )
        .reset_index()
    )
    return summary


def _plot_env_timeseries(df_env: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    x = df_env["step"].to_numpy()

    axes[0].plot(x, df_env["eq_price_before"], label="V before", lw=2)
    axes[0].plot(x, df_env["ref_price_before"], label="P before", lw=2)
    axes[0].set_ylabel("Price")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, df_env["public_gap_before"], label="public_gap_before", lw=2)
    axes[1].plot(x, df_env["mean_signal"], label="mean_signal", lw=2)
    axes[1].set_ylabel("Signal")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(x, df_env["net_flow"], label="net_flow", lw=2)
    axes[2].plot(x, df_env["price_delta_step"], label="price_delta_step", lw=2)
    axes[2].set_ylabel("Flow / return")
    axes[2].set_xlabel("Step")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Debug epizodu — środowisko")
    fig.tight_layout()
    fig.savefig(out_dir / "01_env_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_signal_vs_action(df_agents: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    actionable = df_agents[df_agents["action_name"].isin(["BUY", "SELL"])].copy()
    if actionable.empty:
        return

    action_y = actionable["action_name"].map({"SELL": -1, "BUY": 1}).to_numpy()
    colors = actionable["sigma_i"].to_numpy()
    sc = axes[0].scatter(
        actionable["signal_i"],
        action_y,
        c=colors,
        cmap="coolwarm",
        alpha=0.6,
        s=18,
    )
    axes[0].set_title("signal_i vs action")
    axes[0].set_xlabel("signal_i")
    axes[0].set_ylabel("action (-1 SELL, +1 BUY)")
    axes[0].grid(True, alpha=0.3)
    cbar = fig.colorbar(sc, ax=axes[0])
    cbar.set_label("sigma_i")

    bins = np.linspace(-1.0, 1.0, 11)
    actionable["signal_bin"] = pd.cut(actionable["signal_i"], bins=bins, include_lowest=True)
    buy_share = actionable.groupby("signal_bin", observed=False)["action_name"].apply(lambda s: float((s == "BUY").mean()))
    sell_share = actionable.groupby("signal_bin", observed=False)["action_name"].apply(lambda s: float((s == "SELL").mean()))
    x = np.arange(len(buy_share))
    axes[1].plot(x, buy_share.values, label="BUY share", lw=2)
    axes[1].plot(x, sell_share.values, label="SELL share", lw=2)
    axes[1].set_title("Udział BUY/SELL w koszykach signal_i")
    axes[1].set_xlabel("signal_i bin")
    axes[1].set_ylabel("share")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(i) for i in buy_share.index], rotation=35, ha="right", fontsize=8)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_dir / "02_signal_vs_action.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_sigma_vs_outcomes(df_agents: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    actionable = df_agents[df_agents["action_name"].isin(["BUY", "SELL"])].copy()
    if actionable.empty:
        return

    actionable["aligned"] = (
        ((actionable["signal_i"] > 0) & (actionable["action_name"] == "BUY"))
        | ((actionable["signal_i"] < 0) & (actionable["action_name"] == "SELL"))
    )

    axes[0].scatter(
        actionable["sigma_i"],
        actionable["reward_this_step"],
        c=actionable["aligned"].map({True: 1.0, False: 0.0}),
        cmap="coolwarm",
        alpha=0.6,
        s=18,
    )
    axes[0].set_title("sigma_i vs reward_this_step")
    axes[0].set_xlabel("sigma_i")
    axes[0].set_ylabel("reward_this_step")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(
        actionable["sigma_i"],
        actionable["realized_pnl_this_step"],
        c=actionable["aligned"].map({True: 1.0, False: 0.0}),
        cmap="coolwarm",
        alpha=0.6,
        s=18,
    )
    axes[1].set_title("sigma_i vs realized_pnl_this_step")
    axes[1].set_xlabel("sigma_i")
    axes[1].set_ylabel("realized_pnl_this_step")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "03_sigma_vs_outcomes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_cumulative_pnl(df_agents: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    last = (
        df_agents.sort_values(["agent_id", "step"])
        .groupby("agent_id", as_index=False)
        .tail(1)
        .sort_values("sigma_i")
    )
    axes[0].scatter(last["sigma_i"], last["realized_pnl_cum"], alpha=0.75, s=22)
    axes[0].set_title("Końcowy realized_pnl_cum vs sigma_i")
    axes[0].set_xlabel("sigma_i")
    axes[0].set_ylabel("realized_pnl_cum")
    axes[0].grid(True, alpha=0.3)

    top = last.head(min(10, len(last)))
    bottom = last.tail(min(10, len(last)))
    axes[1].bar(
        [f"low_{i}" for i in range(len(top))],
        top["realized_pnl_cum"],
        alpha=0.7,
        label="lowest sigma_i",
    )
    axes[1].bar(
        [f"high_{i}" for i in range(len(bottom))],
        bottom["realized_pnl_cum"],
        alpha=0.7,
        label="highest sigma_i",
    )
    axes[1].set_title("Końcowy PnL skrajnych agentów")
    axes[1].set_ylabel("realized_pnl_cum")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_dir / "04_cumulative_pnl.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_dir = _pick_run_dir(args.run_dir)
    agents_path = run_dir / "agents_sample.csv"
    env_path = run_dir / "env_steps.csv"
    if not agents_path.exists():
        raise FileNotFoundError(f"Brak {agents_path}")
    if not env_path.exists():
        raise FileNotFoundError(f"Brak {env_path}")

    agents = pd.read_csv(agents_path)
    env_steps = pd.read_csv(env_path)

    agents = agents[
        (agents["seed"] == args.seed)
        & (agents["diversity_score"] == args.diversity_score)
        & (agents["episode"] == args.episode)
    ].copy()
    env_steps = env_steps[
        (env_steps["seed"] == args.seed)
        & (env_steps["diversity_score"] == args.diversity_score)
        & (env_steps["episode"] == args.episode)
    ].copy()

    if args.algorithm:
        agents = agents[agents["algorithm"] == args.algorithm].copy()
        env_steps = env_steps[env_steps["algorithm"] == args.algorithm].copy()
    if args.phase:
        agents = agents[agents["phase"] == args.phase].copy()
        env_steps = env_steps[env_steps["phase"] == args.phase].copy()

    if agents.empty or env_steps.empty:
        raise ValueError("Po filtracji brak danych debugowych.")

    out_dir = run_dir / "debug_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "run_dir": str(run_dir),
        "agents_rows": int(len(agents)),
        "env_rows": int(len(env_steps)),
        "algorithm": str(agents["algorithm"].iloc[0]),
        "phase": str(agents["phase"].iloc[0]),
        "diversity_score": float(args.diversity_score),
        "seed": int(args.seed),
        "episode": int(args.episode),
        "env_signal": {
            "corr_public_gap_to_price_delta": _corr(env_steps["public_gap_before"], env_steps["price_delta_step"]),
            "corr_mean_signal_to_price_delta": _corr(env_steps["mean_signal"], env_steps["price_delta_step"]),
            "corr_mean_signal_to_net_flow": _corr(env_steps["mean_signal"], env_steps["net_flow"]),
            "corr_public_gap_to_net_flow": _corr(env_steps["public_gap_before"], env_steps["net_flow"]),
        },
        "agent_decisions": _decision_alignment(agents),
        "agent_reward": {
            "corr_signal_to_reward": _corr(agents["signal_i"], agents["reward_this_step"]),
            "corr_signal_to_realized": _corr(agents["signal_i"], agents["realized_pnl_this_step"]),
            "corr_sigma_to_reward": _corr(agents["sigma_i"], agents["reward_this_step"]),
            "corr_sigma_to_realized": _corr(agents["sigma_i"], agents["realized_pnl_this_step"]),
        },
    }

    sigma_bucket_summary = _sigma_bucket_summary(agents)
    sigma_bucket_path = out_dir / "sigma_bucket_summary.csv"
    if not sigma_bucket_summary.empty:
        sigma_bucket_summary.to_csv(sigma_bucket_path, index=False)

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True, sort_keys=True)

    _plot_env_timeseries(env_steps, out_dir)
    _plot_signal_vs_action(agents, out_dir)
    _plot_sigma_vs_outcomes(agents, out_dir)
    _plot_cumulative_pnl(agents, out_dir)

    print(f"Run: {run_dir}")
    print(f"Debug output: {out_dir}")
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    if not sigma_bucket_summary.empty:
        print(f"sigma bucket summary: {sigma_bucket_path}")


if __name__ == "__main__":
    main()
