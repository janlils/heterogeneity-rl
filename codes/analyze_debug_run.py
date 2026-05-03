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

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib = None
    plt = None

try:
    import pandas as pd
except ImportError:
    pd = None

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from codes.results_store import latest_run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analiza debugowego epizodu HTM.")
    parser.add_argument("--run-dir", type=str, help="Ścieżka do results/run_*.")
    parser.add_argument("--algorithm", type=str, help="Filtruj po algorithm.")
    parser.add_argument("--phase", type=str, help="Filtruj po phase.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--diversity-score", type=float, help="Filtruj po diversity_score. Domyślnie skrypt spróbuje wybrać jedyną dostępną wartość.")
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


def _require_pandas() -> None:
    if pd is None:
        raise RuntimeError("pandas nie jest dostępny w aktywnym interpreterze.")


def _corr(a: pd.Series, b: pd.Series) -> float:
    joined = pd.concat([a, b], axis=1).dropna()
    if len(joined) < 2:
        return 0.0
    if joined.iloc[:, 0].std() < 1e-12 or joined.iloc[:, 1].std() < 1e-12:
        return 0.0
    return float(joined.iloc[:, 0].corr(joined.iloc[:, 1]))


def _require_plotting() -> bool:
    if plt is None:
        print("  [!] matplotlib nie jest dostępny; zapisuję tylko summary.json i CSV bez wykresów.")
        return False
    return True


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
        "hold_given_positive_signal": float((df_agents[df_agents["signal_i"] > 0]["action_name"] == "HOLD").mean()) if not df_agents[df_agents["signal_i"] > 0].empty else 0.0,
        "hold_given_negative_signal": float((df_agents[df_agents["signal_i"] < 0]["action_name"] == "HOLD").mean()) if not df_agents[df_agents["signal_i"] < 0].empty else 0.0,
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


def _signal_quality_by_sigma(df_agents: pd.DataFrame, df_env: pd.DataFrame) -> pd.DataFrame:
    merged = df_agents.merge(
        df_env[["step", "price_delta_step"]],
        on="step",
        how="left",
    )
    if merged.empty:
        return pd.DataFrame()
    q = min(4, merged["sigma_i"].nunique())
    if q < 1:
        return pd.DataFrame()
    merged["sigma_bucket"] = pd.qcut(merged["sigma_i"], q=q, duplicates="drop")
    rows = []
    for bucket, grp in merged.groupby("sigma_bucket", observed=False):
        if grp.empty:
            continue
        signal = grp["signal_i"]
        delta = grp["price_delta_step"]
        sign_match = (
            np.sign(signal.to_numpy()) == np.sign(delta.to_numpy())
        )
        nonzero = (np.sign(signal.to_numpy()) != 0) & (np.sign(delta.to_numpy()) != 0)
        rows.append({
            "sigma_bucket": str(bucket),
            "n": int(len(grp)),
            "corr_signal_to_price_delta": _corr(signal, delta),
            "mean_abs_signal": float(signal.abs().mean()),
            "mean_abs_price_delta": float(delta.abs().mean()),
            "sign_match_rate": float(sign_match[nonzero].mean()) if np.any(nonzero) else 0.0,
        })
    return pd.DataFrame(rows)


def _agent_level_summary(df_agents: pd.DataFrame) -> pd.DataFrame:
    actionable = df_agents[df_agents["action_name"].isin(["BUY", "SELL"])].copy()
    actionable["aligned"] = (
        ((actionable["signal_i"] > 0) & (actionable["action_name"] == "BUY"))
        | ((actionable["signal_i"] < 0) & (actionable["action_name"] == "SELL"))
    )
    if actionable.empty:
        return pd.DataFrame()
    out = (
        actionable.groupby(["agent_id", "sigma_i"], as_index=False)
        .agg(
            n_actions=("action_name", "size"),
            alignment_rate=("aligned", "mean"),
            mean_reward=("reward_this_step", "mean"),
            mean_realized=("realized_pnl_this_step", "mean"),
            buy_rate=("action_name", lambda s: float((s == "BUY").mean())),
            sell_rate=("action_name", lambda s: float((s == "SELL").mean())),
        )
        .sort_values("sigma_i")
    )
    return out


def _diagnostic_checks(summary: dict) -> dict:
    env = summary["env_signal"]
    agent = summary["agent_decisions"]
    reward = summary["agent_reward"]
    return {
        "signal_predicts_price_move": bool(env["corr_mean_signal_to_price_delta"] > 0.15),
        "agents_follow_signal": bool(agent["signal_action_alignment"] > 0.55),
        "aligned_decisions_improve_reward": bool(agent["reward_when_aligned"] > agent["reward_when_misaligned"]),
        "low_sigma_should_help": bool(reward["corr_sigma_to_reward"] < 0.0),
    }


def _plot_env_timeseries(df_env: pd.DataFrame, out_dir: Path) -> None:
    if not _require_plotting():
        return
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
    if not _require_plotting():
        return
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
    if not _require_plotting():
        return
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


def _plot_signal_quality_by_sigma(df_signal_quality: pd.DataFrame, out_dir: Path) -> None:
    if not _require_plotting():
        return
    if df_signal_quality.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x = np.arange(len(df_signal_quality))
    labels = df_signal_quality["sigma_bucket"].astype(str).tolist()

    axes[0].bar(x, df_signal_quality["corr_signal_to_price_delta"], alpha=0.8)
    axes[0].axhline(0.0, color="gray", lw=1)
    axes[0].set_title("Jakosc sygnalu vs sigma_i")
    axes[0].set_ylabel("corr(signal_i, price_delta_step)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=25, ha="right")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(x, df_signal_quality["sign_match_rate"], alpha=0.8)
    axes[1].axhline(0.5, color="gray", lw=1, ls="--")
    axes[1].set_title("Zgodnosc znaku sygnalu z ruchem ceny")
    axes[1].set_ylabel("sign match rate")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=25, ha="right")
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "05_signal_quality_by_sigma.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_agent_alignment(df_agent_summary: pd.DataFrame, out_dir: Path) -> None:
    if not _require_plotting():
        return
    if df_agent_summary.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].scatter(
        df_agent_summary["sigma_i"],
        df_agent_summary["alignment_rate"],
        c=df_agent_summary["mean_reward"],
        cmap="coolwarm",
        alpha=0.75,
        s=24,
    )
    axes[0].axhline(0.5, color="gray", ls="--", lw=1)
    axes[0].set_title("Czy agent podaza za sygnalem?")
    axes[0].set_xlabel("sigma_i")
    axes[0].set_ylabel("alignment_rate")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(
        df_agent_summary["alignment_rate"],
        df_agent_summary["mean_realized"],
        c=df_agent_summary["sigma_i"],
        cmap="viridis",
        alpha=0.75,
        s=24,
    )
    axes[1].axvline(0.5, color="gray", ls="--", lw=1)
    axes[1].axhline(0.0, color="gray", lw=1)
    axes[1].set_title("Czy alignment daje zysk?")
    axes[1].set_xlabel("alignment_rate")
    axes[1].set_ylabel("mean_realized_pnl_this_step")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "06_agent_alignment.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_cumulative_pnl(df_agents: pd.DataFrame, out_dir: Path) -> None:
    if not _require_plotting():
        return
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
    _require_pandas()
    run_dir = _pick_run_dir(args.run_dir)
    agents_path = run_dir / "agents_sample.csv"
    env_path = run_dir / "env_steps.csv"
    if not agents_path.exists():
        raise FileNotFoundError(f"Brak {agents_path}")
    if not env_path.exists():
        raise FileNotFoundError(f"Brak {env_path}")

    agents = pd.read_csv(agents_path)
    env_steps = pd.read_csv(env_path)

    if agents.empty:
        raise ValueError(
            f"{agents_path} jest puste. Ten run nie zawiera danych debugowych per-agent. "
            "Sprawdź, czy podczas ewaluacji został zapisany pełny debug episode."
        )
    if env_steps.empty:
        raise ValueError(
            f"{env_path} jest puste. Ten run nie zawiera danych debugowych per-step środowiska."
        )

    required_agent_cols = {"diversity_score", "seed", "episode", "algorithm", "phase"}
    required_env_cols = {"diversity_score", "seed", "episode", "algorithm", "phase"}
    missing_agent_cols = sorted(required_agent_cols - set(agents.columns))
    missing_env_cols = sorted(required_env_cols - set(env_steps.columns))
    if missing_agent_cols:
        raise ValueError(f"Brak wymaganych kolumn w {agents_path.name}: {missing_agent_cols}")
    if missing_env_cols:
        raise ValueError(f"Brak wymaganych kolumn w {env_path.name}: {missing_env_cols}")

    diversity_score = args.diversity_score
    if diversity_score is None:
        available_d = sorted(pd.unique(agents["diversity_score"]))
        if len(available_d) == 0:
            raise ValueError(
                f"Brak wartości diversity_score w {agents_path}. "
                "Plik istnieje, ale nie zawiera wierszy z poprawnym debug logiem."
            )
        if len(available_d) == 1:
            diversity_score = float(available_d[0])
        elif 0.5 in set(float(x) for x in available_d):
            diversity_score = 0.5
        else:
            diversity_score = float(available_d[0])

    agents = agents[
        (agents["seed"] == args.seed)
        & (agents["diversity_score"] == diversity_score)
        & (agents["episode"] == args.episode)
    ].copy()
    env_steps = env_steps[
        (env_steps["seed"] == args.seed)
        & (env_steps["diversity_score"] == diversity_score)
        & (env_steps["episode"] == args.episode)
    ].copy()

    if args.algorithm:
        agents = agents[agents["algorithm"] == args.algorithm].copy()
        env_steps = env_steps[env_steps["algorithm"] == args.algorithm].copy()
    if args.phase:
        agents = agents[agents["phase"] == args.phase].copy()
        env_steps = env_steps[env_steps["phase"] == args.phase].copy()

    if agents.empty or env_steps.empty:
        available = {
            "algorithms": sorted(pd.unique(pd.read_csv(agents_path)["algorithm"])) if agents_path.exists() else [],
            "phases": sorted(pd.unique(pd.read_csv(agents_path)["phase"])) if agents_path.exists() else [],
            "diversity_scores": sorted(pd.unique(pd.read_csv(agents_path)["diversity_score"])) if agents_path.exists() else [],
            "seeds": sorted(pd.unique(pd.read_csv(agents_path)["seed"])) if agents_path.exists() else [],
            "episodes": sorted(pd.unique(pd.read_csv(agents_path)["episode"])) if agents_path.exists() else [],
        }
        raise ValueError(
            "Po filtracji brak danych debugowych. "
            f"Filtry: algorithm={args.algorithm}, phase={args.phase}, seed={args.seed}, "
            f"diversity_score={diversity_score}, episode={args.episode}. "
            f"Dostępne: {available}"
        )

    out_dir = run_dir / "debug_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "run_dir": str(run_dir),
        "agents_rows": int(len(agents)),
        "env_rows": int(len(env_steps)),
        "algorithm": str(agents["algorithm"].iloc[0]),
        "phase": str(agents["phase"].iloc[0]),
        "diversity_score": float(diversity_score),
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
    summary["diagnostic_checks"] = _diagnostic_checks(summary)

    sigma_bucket_summary = _sigma_bucket_summary(agents)
    signal_quality_by_sigma = _signal_quality_by_sigma(agents, env_steps)
    agent_level_summary = _agent_level_summary(agents)
    sigma_bucket_path = out_dir / "sigma_bucket_summary.csv"
    if not sigma_bucket_summary.empty:
        sigma_bucket_summary.to_csv(sigma_bucket_path, index=False)
    signal_quality_path = out_dir / "signal_quality_by_sigma.csv"
    if not signal_quality_by_sigma.empty:
        signal_quality_by_sigma.to_csv(signal_quality_path, index=False)
    agent_summary_path = out_dir / "agent_decision_summary.csv"
    if not agent_level_summary.empty:
        agent_level_summary.to_csv(agent_summary_path, index=False)

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True, sort_keys=True)

    _plot_env_timeseries(env_steps, out_dir)
    _plot_signal_vs_action(agents, out_dir)
    _plot_sigma_vs_outcomes(agents, out_dir)
    _plot_cumulative_pnl(agents, out_dir)
    _plot_signal_quality_by_sigma(signal_quality_by_sigma, out_dir)
    _plot_agent_alignment(agent_level_summary, out_dir)

    print(f"Run: {run_dir}")
    print(f"Debug output: {out_dir}")
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    if not sigma_bucket_summary.empty:
        print(f"sigma bucket summary: {sigma_bucket_path}")
    if not signal_quality_by_sigma.empty:
        print(f"signal quality by sigma: {signal_quality_path}")
    if not agent_level_summary.empty:
        print(f"agent decision summary: {agent_summary_path}")


if __name__ == "__main__":
    main()
