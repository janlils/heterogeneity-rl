from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _log_plot_saved(save_path: Path, logger: Optional[logging.Logger]) -> None:
    if logger is not None:
        logger.info(f"Wykres: {save_path}")


def plot_policy_learning_curves(
    records: List[dict],
    zi_baselines: Dict[float, float],
    save_path: Path,
    diversity_scores: List[float],
    rolling_window: int,
    algorithm_label: str,
    logger: Optional[logging.Logger] = None,
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
        axes[0].plot(acc.index, acc.values, lw=2, label=f"{algorithm_label} D={d:.1f}")
        axes[1].plot(pnl.index, pnl.values, lw=2, label=f"D={d:.1f}")
        if d in zi_baselines:
            axes[0].axhline(zi_baselines[d], color="gray", ls="--", lw=1, alpha=0.5)

    axes[0].set_title(f"{algorithm_label} trade_accuracy")
    axes[0].set_xlabel("Epizod")
    axes[0].set_ylabel("trade_accuracy")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].set_title(f"{algorithm_label} mean_total_pnl")
    axes[1].set_xlabel("Epizod")
    axes[1].set_ylabel("mean_total_pnl")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log_plot_saved(save_path, logger)


def plot_sarsa_learning_curves(
    all_records: List[dict],
    zi_baselines: Dict[float, float],
    save_path: Path,
    diversity_scores: List[float],
    rolling_window: int,
    n_agents: int,
    algorithm_label: str = "Deep SARSA",
    logger: Optional[logging.Logger] = None,
) -> None:
    df = pd.DataFrame(all_records)
    n_cols = len(diversity_scores)

    cmap = plt.cm.coolwarm
    colors = {d: cmap(i / max(n_cols - 1, 1)) for i, d in enumerate(diversity_scores)}

    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle(
        f"{algorithm_label} — krzywe uczenia | N={n_agents} agentów | model spekulacyjny",
        fontsize=13,
        fontweight="bold",
    )

    for col, d in enumerate(diversity_scores):
        df_d = df[df["diversity_score"] == d]
        zi = zi_baselines.get(d)
        color = colors[d]

        ax = axes[0, col]
        grouped = df_d.groupby("episode")["trade_accuracy"]
        mean_e = grouped.mean()
        std_e = grouped.std().fillna(0)
        smooth = mean_e.rolling(rolling_window, min_periods=1).mean()
        eps_idx = mean_e.index

        ax.fill_between(
            eps_idx,
            np.clip(smooth - std_e, 0, 1),
            np.clip(smooth + std_e, 0, 1),
            alpha=0.15,
            color=color,
        )
        ax.plot(eps_idx, smooth, color=color, lw=2.5, label=algorithm_label)
        if zi is not None:
            ax.axhline(zi, color="gray", ls="--", lw=1.5, label=f"ZI ({zi:.3f})", alpha=0.8)

        final_eff = float(smooth.iloc[-1])
        if zi is not None:
            delta = final_eff - zi
            c_delta = "#2E7D32" if delta > 0 else "#C62828"
            ax.annotate(
                f"Δ = {delta:+.3f}",
                xy=(eps_idx[-1], final_eff),
                fontsize=9,
                color=c_delta,
                fontweight="bold",
                ha="right",
                va="bottom",
            )

        ax.set_title(f"D = {d:.1f}", fontsize=11, color=color, fontweight="bold")
        ax.set_xlabel("Epizod")
        ax.set_ylabel("Trade Accuracy")
        ax.set_ylim(0.0, 1.0)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax2 = axes[1, col]
        mean_td = df_d.groupby("episode")["mean_td_error"].mean()
        mean_eps = df_d.groupby("episode")["mean_epsilon"].mean()
        smooth_td = mean_td.rolling(rolling_window, min_periods=1).mean()

        ax2_twin = ax2.twinx()
        ax2.plot(eps_idx, smooth_td, color=color, lw=2, label="TD error")
        ax2_twin.plot(eps_idx, mean_eps, color="gray", lw=1.5, ls=":", label="ε (epsilon)")

        ax2.set_xlabel("Epizod")
        ax2.set_ylabel("Mean TD Error", color=color)
        ax2_twin.set_ylabel("Epsilon", color="gray")
        ax2.set_title(f"Zbieżność (D={d:.1f})", fontsize=9)
        ax2.grid(True, alpha=0.3)

        lines1, labs1 = ax2.get_legend_handles_labels()
        lines2, labs2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labs1 + labs2, fontsize=7, loc="upper right")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log_plot_saved(save_path, logger)


def plot_sarsa_final_comparison(
    all_records: List[dict],
    zi_baselines: Dict[float, float],
    save_path: Path,
    diversity_scores: List[float],
    n_episodes: int,
    algorithm_label: str = "Deep SARSA",
    logger: Optional[logging.Logger] = None,
) -> None:
    df = pd.DataFrame(all_records)
    final_window = min(50, max(1, n_episodes // 3))
    final = df[df["episode"] >= n_episodes - final_window]

    sarsa_eff = final.groupby("diversity_score")["trade_accuracy"].agg(["mean", "std"])
    sarsa_gini = final.groupby("diversity_score")["gini"].agg(["mean", "std"])
    sarsa_trd = final.groupby("diversity_score")["n_trades"].agg(["mean", "std"])

    d_vals = diversity_scores
    x = np.arange(len(d_vals))
    w = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"{algorithm_label} vs ZI Baseline — końcowe wyniki (ostatnie {final_window} epizodów)\n"
        "Model spekulacyjny: N agentów z prywatnymi wycenami",
        fontsize=12,
        fontweight="bold",
    )

    cmap = plt.cm.coolwarm
    colors = [cmap(i / max(len(d_vals) - 1, 1)) for i in range(len(d_vals))]

    ax = axes[0]
    sarsa_means = [sarsa_eff.loc[d, "mean"] if d in sarsa_eff.index else 0 for d in d_vals]
    sarsa_stds = [sarsa_eff.loc[d, "std"] if d in sarsa_eff.index else 0 for d in d_vals]
    zi_means = [zi_baselines.get(d, 0) for d in d_vals]

    ax.bar(x - w / 2, sarsa_means, w, yerr=sarsa_stds, label=algorithm_label, color="#1565C0", alpha=0.85, capsize=4, error_kw={"lw": 1.5})
    ax.bar(x + w / 2, zi_means, w, label="ZI Baseline", color="#616161", alpha=0.75)

    for i, (sm, zm) in enumerate(zip(sarsa_means, zi_means)):
        delta = sm - zm
        c = "#2E7D32" if delta > 0 else "#C62828"
        ax.text(i - w / 2, min(sm + 0.02, 0.98), f"{sm:.3f}", ha="center", fontsize=7.5)
        ax.text(i, min(max(sm, zm) + 0.06, 0.98), f"Δ={delta:+.2f}", ha="center", fontsize=7.5, color=c, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"D={d}" for d in d_vals], rotation=30)
    ax.set_ylabel("Trade Accuracy")
    ax.set_title("Metryka główna vs empiryczny ZI")
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1]
    gini_means = [sarsa_gini.loc[d, "mean"] if d in sarsa_gini.index else 0 for d in d_vals]
    gini_stds = [sarsa_gini.loc[d, "std"] if d in sarsa_gini.index else 0 for d in d_vals]
    bars = ax.bar(x, gini_means, w * 2, yerr=gini_stds, color=colors, alpha=0.85, capsize=4, error_kw={"lw": 1.5})
    ax.set_xticks(x)
    ax.set_xticklabels([f"D={d}" for d in d_vals], rotation=30)
    ax.set_ylabel("Gini Coefficient")
    ax.set_title("Nierówność wyników agentów")
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.3)
    for bar, m in zip(bars, gini_means):
        ax.text(bar.get_x() + bar.get_width() / 2, m + 0.01, f"{m:.3f}", ha="center", fontsize=8)

    ax = axes[2]
    trd_means = [sarsa_trd.loc[d, "mean"] if d in sarsa_trd.index else 0 for d in d_vals]
    trd_stds = [sarsa_trd.loc[d, "std"] if d in sarsa_trd.index else 0 for d in d_vals]
    bars = ax.bar(x, trd_means, w * 2, yerr=trd_stds, color=colors, alpha=0.85, capsize=4, error_kw={"lw": 1.5})
    ax.set_xticks(x)
    ax.set_xticklabels([f"D={d}" for d in d_vals], rotation=30)
    ax.set_ylabel("Średnia liczba transakcji / epizod")
    ax.set_title("Aktywność rynkowa")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, m in zip(bars, trd_means):
        ax.text(bar.get_x() + bar.get_width() / 2, m + 0.05, f"{m:.1f}", ha="center", fontsize=8)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log_plot_saved(save_path, logger)


def plot_agent_eval_distribution(
    agent_eval_rows: List[dict],
    save_path: Path,
    diversity_scores: List[float],
    title: str,
    logger: Optional[logging.Logger] = None,
) -> None:
    if not agent_eval_rows:
        return

    df = pd.DataFrame(agent_eval_rows)
    if df.empty:
        return

    grouped = (
        df.groupby(["diversity_score", "seed", "agent_id"], as_index=False)
        .agg({
            "sigma_i": "first",
            "trader_type": "first",
            "mean_trade_accuracy_agent": "mean",
            "mean_realized_pnl": "mean",
            "mean_n_trades_closed": "mean",
        })
    )

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(title, fontsize=13, fontweight="bold")
    cmap = plt.cm.coolwarm
    colors = {d: cmap(i / max(len(diversity_scores) - 1, 1)) for i, d in enumerate(diversity_scores)}

    ax = axes[0, 0]
    for d in diversity_scores:
        sub = grouped[grouped["diversity_score"] == d]
        if sub.empty:
            continue
        ax.scatter(sub["sigma_i"], sub["mean_trade_accuracy_agent"], s=24, alpha=0.7, color=colors[d], label=f"D={d:.1f}")
    ax.set_title("sigma_i vs mean_trade_accuracy_agent")
    ax.set_xlabel("sigma_i")
    ax.set_ylabel("mean_trade_accuracy_agent")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for d in diversity_scores:
        sub = grouped[grouped["diversity_score"] == d]
        if sub.empty:
            continue
        ax.scatter(sub["sigma_i"], sub["mean_realized_pnl"], s=24, alpha=0.7, color=colors[d], label=f"D={d:.1f}")
    ax.set_title("sigma_i vs mean_realized_pnl")
    ax.set_xlabel("sigma_i")
    ax.set_ylabel("mean_realized_pnl")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    acc_data = [
        grouped.loc[grouped["diversity_score"] == d, "mean_trade_accuracy_agent"].values
        for d in diversity_scores
        if not grouped.loc[grouped["diversity_score"] == d].empty
    ]
    acc_labels = [f"D={d:.1f}" for d in diversity_scores if not grouped.loc[grouped["diversity_score"] == d].empty]
    if acc_data:
        bp = ax.boxplot(acc_data, patch_artist=True, labels=acc_labels)
        for patch, label in zip(bp["boxes"], acc_labels):
            d = float(label.split("=")[1])
            patch.set_facecolor(colors[d])
            patch.set_alpha(0.55)
    ax.set_title("Rozkład mean_trade_accuracy_agent")
    ax.set_ylabel("mean_trade_accuracy_agent")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1, 1]
    pnl_data = [
        grouped.loc[grouped["diversity_score"] == d, "mean_realized_pnl"].values
        for d in diversity_scores
        if not grouped.loc[grouped["diversity_score"] == d].empty
    ]
    pnl_labels = [f"D={d:.1f}" for d in diversity_scores if not grouped.loc[grouped["diversity_score"] == d].empty]
    if pnl_data:
        bp = ax.boxplot(pnl_data, patch_artist=True, labels=pnl_labels)
        for patch, label in zip(bp["boxes"], pnl_labels):
            d = float(label.split("=")[1])
            patch.set_facecolor(colors[d])
            patch.set_alpha(0.55)
    ax.set_title("Rozkład mean_realized_pnl per agent")
    ax.set_ylabel("mean_realized_pnl")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log_plot_saved(save_path, logger)
