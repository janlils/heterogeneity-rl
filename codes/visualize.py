"""
codes/visualize.py — Wykresy diagnostyczne HTM (model spekulacyjny CT)
=========================================================================
Wersja zsynchronizowana z aktualnym API:
  - Równoległe wykonanie: execute_parallel_actions + compute_step_rewards
  - Realized PnL jako główny składnik rewardu
  - trade_accuracy jako glowna metryka, porownywana z empirycznym ZI
  - 3 akcje: HOLD/BUY/SELL
  - Position model: position in [-max_pos, +max_pos]
  - obs[1] = position_norm

Wykresy:
  01  Rozklady sentimentu agentow przy roznych D
  02  Parametry heterogenicznosci (gamma, threshold, risk_aversion, sentiment)
  03  Emergencja rol BUY/SELL/HOLD jako funkcja D
  04  Dynamika ceny rynkowej w jednym epizodzie
  05  Sentiment vs realized P&L (kolorowany trade_accuracy per agent)
  06  ZI baseline walidacja srodowiska (trade_accuracy, pos_agents, trades, hold_frac)
  07  Heatmapa akcji per agent (CT, 3 akcje)
  08  Rozklad majatku agentow (Pareto -> max_position)
  09  Ewolucja cen i wycen przez epizody
  10  Ewolucja pozycji przez epizod (CT)
  11  Rozklad P&L per agent vs D (box + violin)
  12  Aktywnosc handlowa przez epizod
  13  SARSA vs ZI: trade_accuracy przez epizody (z CSV)
  14  Trade accuracy curves — glowny wykres artykuluop (z CSV)
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
(PROJECT_ROOT / ".matplotlib_cache").mkdir(exist_ok=True)
(PROJECT_ROOT / ".cache").mkdir(exist_ok=True)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from codes.config import HTMConfig, EnvConfig, MarketDynamics, LogConfig
from codes.double_auction import (
    DoubleAuction, AgentPopulation, ZeroIntelligenceAgent,
    run_zi_baseline, _gini,
)

logging.basicConfig(level=logging.WARNING)
PLOTS_DIR = Path(__file__).parent.parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Paleta — spojna przez caly artykul
# ---------------------------------------------------------------------------

COLORS = {
    "D0.0": "#1A237E",
    "D0.2": "#1565C0",
    "D0.4": "#0288D1",
    "D0.6": "#00897B",
    "D0.8": "#F57F17",
    "D1.0": "#B71C1C",
    "buy":  "#2E7D32",
    "sell": "#C62828",
    "hold": "#78909C",
    "eq":   "#E65100",
    "zi":   "#616161",
    "sarsa":"#1565C0",
}

D_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
D_COLORS = [COLORS[f"D{d:.1f}"] for d in D_VALUES]


def _cfg(n_agents: int = 20) -> HTMConfig:
    return HTMConfig(
        env=EnvConfig(n_agents=n_agents),
        market=MarketDynamics.stable(),
        log=LogConfig(level="WARNING"),
    )


def _run_zi_episode(da: DoubleAuction, cfg: HTMConfig) -> dict:
    """Jeden epizod ZI w CT (równoległe wykonanie). Zwraca episode_metrics()."""
    agent_ids = list(da.population.agents.keys())
    zi = {aid: ZeroIntelligenceAgent(p, cfg.env)
          for aid, p in da.population.agents.items()}
    da.reset_episode()
    for _ in range(cfg.env.episode_steps):
        if da.done:
            break
        actions = {
            aid: zi[aid].act(da.get_observation(aid))
            for aid in agent_ids
        }
        da.execute_parallel_actions(actions)
        _, dones = da.compute_step_rewards()
        if dones.get(agent_ids[0], False):
            break
    return da.episode_metrics()


def _save(fig, filename: str) -> None:
    path = PLOTS_DIR / filename
    try:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    except Exception:
        fig.savefig(path, dpi=100)
    print(f"  zapisano: {path}")
    plt.close(fig)


def _tight():
    try:
        plt.tight_layout(pad=0.8)
    except Exception:
        pass


def _load_results_csv(csv_path: Optional[str] = None, quick: bool = False):
    try:
        import pandas as pd
    except ImportError:
        print("  [!] pandas nie dostepny")
        return None, None

    if csv_path is None:
        candidates = []
        if quick:
            candidates.append(PROJECT_ROOT / "results" / "deep_sarsa_quick_results.csv")
        else:
            candidates.extend([
                PROJECT_ROOT / "results" / "deep_sarsa_results.csv",
                PROJECT_ROOT / "results" / "deep_sarsa_quick_results.csv",
            ])
        existing = [p for p in candidates if p.exists()]
        if not existing:
            print("  [!] Brak CSV treningu. Uruchom najpierw train_deep_sarsa.")
            return None, None
        path = max(existing, key=lambda p: p.stat().st_mtime)
    else:
        path = Path(csv_path)
        if not path.exists():
            print(f"  [!] Brak CSV: {path}")
            return None, None

    return pd.read_csv(path), path


def _result_colors(d_vals):
    cmap = plt.cm.coolwarm
    return {d: cmap(i / max(len(d_vals) - 1, 1)) for i, d in enumerate(d_vals)}


def _baseline_by_d(df):
    if "zi_baseline_trade_accuracy" not in df.columns:
        print("  [!] Brak zi_baseline_trade_accuracy w CSV — pomijam linię ZI.")
        return {}
    d_vals = sorted(df["diversity_score"].unique())
    baseline = float(df["zi_baseline_trade_accuracy"].dropna().iloc[0])
    return {d: baseline for d in d_vals}


# ===========================================================================
# 01. Rozklady wycen agentow przy roznych D
# ===========================================================================

def plot_valuation_distributions(n_agents: int = 40, n_seeds: int = 20) -> None:
    """Rozklad sentimentu — glowna cecha modelu spekulacyjnego."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(
        "Rozklad sentimentu agentow — model spekulacyjny\n"
        "(handel wynika z roznych nastrojow, nie stalych rol)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    cfg = _cfg(n_agents=n_agents)
    eq  = 0.5

    for ax, (d, color) in zip(axes.flat, zip(D_VALUES, D_COLORS)):
        vals = []
        for s in range(n_seeds):
            pop = AgentPopulation(
                n_agents=n_agents, diversity_score=d,
                diversity_cfg=cfg.diversity, sentiment_cfg=cfg.sentiment,
                env_cfg=cfg.env, eq_price=eq, seed=s,
            )
            vals.extend(p.sentiment for p in pop.agents.values())
        vals = np.array(vals)

        ax.hist(vals, bins=25, color=color, alpha=0.75, density=True,
                edgecolor="white", linewidth=0.5)
        ax.axvline(eq, color=COLORS["eq"], ls="--", lw=2, label=f"eq={eq:.2f}")
        ax.axvspan(eq, 1.0, alpha=0.05, color=COLORS["buy"])
        ax.axvspan(0.0, eq, alpha=0.05, color=COLORS["sell"])

        ax.text(0.05, 0.95,
                f"sigma={vals.std():.3f}\nbull: {(vals>0).mean()*100:.0f}%\n"
                f"bear: {(vals<0).mean()*100:.0f}%",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        ax.text(0.75, 0.82, "KUP", transform=ax.transAxes,
                color=COLORS["buy"], fontsize=9, fontweight="bold", ha="center")
        ax.text(0.25, 0.82, "SPRZEDAJ", transform=ax.transAxes,
                color=COLORS["sell"], fontsize=9, fontweight="bold", ha="center")
        ax.set_title(f"D = {d:.1f}", fontsize=12, color=color, fontweight="bold")
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Gestosc" if ax in axes[:, 0] else "")
        ax.set_xlim(-1, 1)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    _tight()
    _save(fig, "01_valuation_distributions.png")


# ===========================================================================
# 02. Rozklady parametrow heterogenicznosci
# ===========================================================================

def plot_heterogeneity_parameters(n_agents: int = 60, n_seeds: int = 10) -> None:
    """Rozklady gamma, threshold, risk_aversion i parametrow behawioralnych."""
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle("Rozklady parametrow heterogenicznosci — D = 0 / 0.5 / 1.0",
                 fontsize=13, fontweight="bold")
    cfg = _cfg(n_agents=n_agents)

    params = [
        ("gamma",          "Discount factor gamma",     [0.5, 1.0],  "Horyzont czasowy"),
        ("threshold",      "Prog decyzji",              [0.0, 0.30], "Min |val-price| do handlu"),
        ("risk_aversion",  "Awersja do ryzyka lambda",  [0.0, 3.0],  "Kara za duza pozycje (CT)"),
        ("sentiment",      "Sentiment",                 [-1.0, 1.0], "Nastroj poczatkowy"),
        ("alpha_i",        "alpha_i",                   [0.0, 0.4],  "Momentum ceny"),
        ("beta_i",         "beta_i",                    [0.0, 0.25], "Powrot do neutralu"),
        ("news_sensitivity","news_sensitivity",         [0.0, 0.5],  "Reakcja na news"),
    ]
    d_sub    = [0.0, 0.5, 1.0]
    d_colors = [COLORS["D0.0"], COLORS["D0.4"], COLORS["D1.0"]]

    for ax, (param, label, xlim, desc) in zip(axes.flat, params):
        for d, color in zip(d_sub, d_colors):
            values = []
            for s in range(n_seeds):
                pop = AgentPopulation(
                    n_agents=n_agents, diversity_score=d,
                    diversity_cfg=cfg.diversity, sentiment_cfg=cfg.sentiment,
                    env_cfg=cfg.env, eq_price=0.5, seed=s,
                )
                for p in pop.agents.values():
                    values.append(getattr(p, param))
            ax.hist(values, bins=20, color=color, alpha=0.55, density=True,
                    label=f"D={d:.1f}", edgecolor="none")

        ax.set_xlabel(label, fontsize=9)
        ax.set_ylabel("Gestosc", fontsize=8)
        ax.set_title(desc, fontsize=9, style="italic")
        ax.set_xlim(xlim)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    _tight()
    _save(fig, "02_heterogeneity_parameters.png")


# ===========================================================================
# 03. Emergencja rol BUY/SELL/HOLD jako funkcja D
# ===========================================================================

def plot_role_emergence(n_episodes: int = 100) -> None:
    """Proporcje akcji, mean_pnl i liczba transakcji vs D (ZI baseline)."""
    cfg = _cfg()
    rng = np.random.default_rng(42)
    results = {d: {"buy": [], "sell": [], "hold": [], "pnl": [], "trades": []}
               for d in D_VALUES}

    for d in D_VALUES:
        da = DoubleAuction(cfg, seed=42)
        da.reset(diversity_score=d, seed=42)
        for ep in range(n_episodes):
            ep_seed = int(rng.integers(0, 100_000))
            da.reset(diversity_score=d, seed=ep_seed)
            m = _run_zi_episode(da, cfg)
            results[d]["buy"].append(m["action_buy_frac"])
            results[d]["sell"].append(m["action_sell_frac"])
            results[d]["hold"].append(m["action_hold_frac"])
            results[d]["pnl"].append(m["mean_pnl"])
            results[d]["trades"].append(m["n_trades"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Emergencja rol BUY/SELL/HOLD jako funkcja D (ZI baseline)",
        fontsize=12, fontweight="bold",
    )
    x = np.arange(len(D_VALUES))
    w = 0.6

    ax = axes[0]
    buy_m  = [np.mean(results[d]["buy"])  for d in D_VALUES]
    sell_m = [np.mean(results[d]["sell"]) for d in D_VALUES]
    hold_m = [np.mean(results[d]["hold"]) for d in D_VALUES]
    ax.bar(x, buy_m,  w, label="BUY",  color=COLORS["buy"],  alpha=0.85)
    ax.bar(x, sell_m, w, bottom=buy_m, label="SELL", color=COLORS["sell"], alpha=0.85)
    bot = [b + s for b, s in zip(buy_m, sell_m)]
    ax.bar(x, hold_m, w, bottom=bot,   label="HOLD", color=COLORS["hold"], alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels([f"D={d}" for d in D_VALUES], rotation=30)
    ax.set_ylabel("Frakcja akcji"); ax.set_ylim(0, 1)
    ax.set_title("Proporcje akcji agentow")
    ax.legend(fontsize=9); ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1]
    pnl_m = [np.mean(results[d]["pnl"]) for d in D_VALUES]
    pnl_s = [np.std(results[d]["pnl"])  for d in D_VALUES]
    ax.errorbar(D_VALUES, pnl_m, yerr=pnl_s, fmt="o-",
                color=COLORS["zi"], lw=2, capsize=5, ms=8)
    ax.fill_between(D_VALUES,
                    [m - s for m, s in zip(pnl_m, pnl_s)],
                    [m + s for m, s in zip(pnl_m, pnl_s)],
                    alpha=0.15, color=COLORS["zi"])
    ax.set_xlabel("Diversity Score D"); ax.set_ylabel("Mean realized P&L")
    ax.set_title("Realized P&L jako funkcja D"); ax.grid(True, alpha=0.3)

    ax = axes[2]
    trd_m = [np.mean(results[d]["trades"]) for d in D_VALUES]
    trd_s = [np.std(results[d]["trades"])  for d in D_VALUES]
    bars = ax.bar(D_VALUES, trd_m, width=0.12, color=D_COLORS, alpha=0.85,
                  edgecolor="white")
    ax.errorbar(D_VALUES, trd_m, yerr=trd_s, fmt="none",
                color="gray", capsize=4, lw=1.5)
    for bar, m in zip(bars, trd_m):
        ax.text(bar.get_x() + bar.get_width() / 2, m + 0.1,
                f"{m:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Diversity Score D"); ax.set_ylabel("Transakcji / epizod")
    ax.set_title("Aktywnosc rynkowa vs D"); ax.grid(True, axis="y", alpha=0.3)

    _tight()
    _save(fig, "03_role_emergence.png")


# ===========================================================================
# 04. Dynamika ceny rynkowej w jednym epizodzie
# ===========================================================================

def plot_price_dynamics(diversity_scores: List[float] = None) -> None:
    """ref_price przez 200 krokow epizodu przy roznych D."""
    if diversity_scores is None:
        diversity_scores = [0.0, 0.5, 1.0]
    cfg = _cfg(n_agents=30)
    fig, axes = plt.subplots(1, len(diversity_scores),
                             figsize=(5 * len(diversity_scores), 5))
    fig.suptitle("Dynamika ceny rynkowej — jeden epizod CT (T=200)",
                 fontsize=12, fontweight="bold")
    if len(diversity_scores) == 1:
        axes = [axes]

    for ax, d in zip(axes, diversity_scores):
        color = D_COLORS[min(int(d * 5), 5)]
        da    = DoubleAuction(cfg, seed=42)
        da.reset(diversity_score=d, seed=42)
        eq    = da.eq_price

        zi = {aid: ZeroIntelligenceAgent(p, cfg.env)
              for aid, p in da.population.agents.items()}
        agent_ids = list(da.population.agents.keys())
        da.reset_episode()

        prices, trade_steps = [], []
        for step in range(cfg.env.episode_steps):
            if da.done:
                break
            prices.append(da.ref_price)
            prev_n = da.episode_metrics()["n_trades"]
            actions = {
                aid: zi[aid].act(da.get_observation(aid))
                for aid in agent_ids
            }
            da.execute_parallel_actions(actions)
            _, dones = da.compute_step_rewards()
            if da.episode_metrics()["n_trades"] > prev_n:
                trade_steps.append(step)
            if dones.get(agent_ids[0], False):
                break

        steps = np.arange(len(prices))
        ax.axhline(eq, color=COLORS["eq"], ls="--", lw=2,
                   label=f"eq={eq:.2f}", alpha=0.8)
        ax.axhspan(eq - 0.05, eq + 0.05, alpha=0.07, color=COLORS["eq"])
        ax.plot(steps, prices, color=color, lw=1.8, label="ref_price")
        for ts in trade_steps:
            ax.axvline(ts, color=color, alpha=0.12, lw=0.7)

        m = da.episode_metrics()
        ax.set_title(
            f"D={d:.1f} | trades={m['n_trades']} | "
            f"trade_acc={m['trade_accuracy']:.3f}",
            fontsize=10, color=color,
        )
        ax.set_xlabel("Krok epizodu"); ax.set_ylabel("Cena referencyjna")
        ax.set_ylim(0.2, 0.8); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    _tight()
    _save(fig, "04_price_dynamics.png")


# ===========================================================================
# 05. Sentiment vs realized P&L (kolorowany trade_accuracy)
# ===========================================================================

def plot_valuation_vs_pnl(n_episodes: int = 30) -> None:
    """Scatter: sentiment vs P&L, kolor = trade_accuracy per agent."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Sentiment vs Realized P&L (kolor = trade_accuracy per agent)\n"
        "Zielony = agent podejmowal dobre decyzje (acc>0.5), czerwony = zle",
        fontsize=12, fontweight="bold",
    )
    cfg = _cfg()
    rng = np.random.default_rng(42)

    for ax, d in zip(axes, [0.2, 0.5, 1.0]):
        all_vals, all_pnl, all_acc = [], [], []

        da = DoubleAuction(cfg, seed=42)
        da.reset(diversity_score=d, seed=42)

        for ep in range(n_episodes):
            ep_seed = int(rng.integers(0, 100_000))
            da.reset(diversity_score=d, seed=ep_seed)
            _run_zi_episode(da, cfg)
            am = da.agent_metrics()
            for m in am.values():
                all_vals.append(m["sentiment"])
                all_pnl.append(m["ep_pnl"])
                all_acc.append(m["trade_accuracy"])

        vals = np.array(all_vals)
        pnl  = np.array(all_pnl)
        acc  = np.array(all_acc)

        sc = ax.scatter(vals, pnl, c=acc, cmap="RdYlGn",
                        vmin=0.0, vmax=1.0, alpha=0.5, s=22, edgecolors="none")
        ax.axvline(0.0, color=COLORS["eq"], ls="--", lw=1.5, alpha=0.7, label="neutral")
        ax.axhline(0.0, color="gray", lw=0.8, alpha=0.5)

        if len(vals) > 5:
            z  = np.polyfit(vals, pnl, 1)
            xr = np.linspace(vals.min(), vals.max(), 100)
            ax.plot(xr, np.poly1d(z)(xr), "k--", lw=1.5, alpha=0.6, label="trend")

        plt.colorbar(sc, ax=ax, label="trade_accuracy")
        ax.legend(fontsize=8)
        ax.set_xlabel("Sentiment agenta")
        ax.set_ylabel("Realized P&L epizodu")
        ax.set_title(f"D={d:.1f} | N={len(all_vals)} obs.")
        ax.set_xlim(-1.0, 1.0)
        ax.grid(True, alpha=0.3)

    _tight()
    _save(fig, "05_valuation_vs_pnl.png")


# ===========================================================================
# 06. ZI baseline — walidacja srodowiska
# ===========================================================================

def plot_zi_validation(n_episodes: int = 150) -> None:
    """Walidacja: wspólny ZI baseline dla zadanej konfiguracji środowiska."""
    cfg = _cfg()
    print("  Licze ZI baseline...")
    ref_d = float(D_VALUES[0])
    baseline = run_zi_baseline(cfg, diversity_score=ref_d,
                               n_episodes=n_episodes, seed=42)
    results = {d: baseline for d in D_VALUES}
    print(f"    wspolny | D_ref={ref_d:.1f} | "
          f"acc={baseline['trade_accuracy']['mean']:.3f} | "
          f"trades={baseline['n_trades']['mean']:.0f}")

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle(
        "ZI Baseline — wspolny punkt odniesienia dla CT\n"
        "(ta sama wartosc dla wszystkich D, bo ZI nie korzysta z typu agenta)",
        fontsize=12, fontweight="bold",
    )

    metrics = [
        ("trade_accuracy",     "Trade accuracy",          "Empiryczny wynik ZI"),
        ("pnl_positive_agents","Agentow z dodatnim P&L",  "Ile agentow zarabia"),
        ("n_trades",           "Transakcji / epizod",      "Aktywnosc rynkowa"),
        ("action_hold_frac",   "Frakcja HOLD",             "Inercja rynku"),
    ]

    for ax, (key, ylabel, title) in zip(axes, metrics):
        means = [results[d][key]["mean"] for d in D_VALUES]
        stds  = [results[d][key]["std"]  for d in D_VALUES]
        ax.errorbar(D_VALUES, means, yerr=stds, fmt="o-",
                    color=COLORS["zi"], lw=2.5, capsize=5, ms=8)
        ax.fill_between(D_VALUES,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.15, color=COLORS["zi"])
        for dv, mv, c in zip(D_VALUES, means, D_COLORS):
            ax.scatter([dv], [mv], color=c, s=80, zorder=6)
        ax.set_xlabel("Diversity Score D")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)
        if key == "trade_accuracy":
            ax.axhline(0.5, color="gray", ls="--", lw=1.5, label="neutral reference")
            ax.set_ylim(0.0, 1.0)
            ax.legend(fontsize=8)

    _tight()
    _save(fig, "06_zi_validation.png")


# ===========================================================================
# 07. Heatmapa akcji per agent (CT, 3 akcje)
# ===========================================================================

def plot_action_heatmap(n_episodes: int = 50, d: float = 0.7) -> None:
    """Ktory agent wybiera jaka akcje — posortowane wg sentimentu."""
    cfg = _cfg(n_agents=20)
    da  = DoubleAuction(cfg, seed=42)
    rng = np.random.default_rng(42)

    n_actions = cfg.env.n_actions  # 3
    n_ag      = cfg.env.n_agents   # 20
    action_matrix = np.zeros((n_ag, n_actions))
    valuations    = None

    for ep in range(n_episodes):
        ep_seed = int(rng.integers(0, 100_000))
        da.reset(diversity_score=d, seed=ep_seed)
        agent_ids = list(da.population.agents.keys())
        if valuations is None:
            valuations = [da.population.agents[aid].sentiment
                          for aid in agent_ids]

        zi = {aid: ZeroIntelligenceAgent(p, cfg.env)
              for aid, p in da.population.agents.items()}
        da.reset_episode()

        for _ in range(cfg.env.episode_steps):
            if da.done:
                break
            actions = {}
            for aid in agent_ids:
                obs    = da.get_observation(aid)
                action = zi[aid].act(obs)
                i      = agent_ids.index(aid)
                if i < n_ag:
                    action_matrix[i, action] += 1
                actions[aid] = action
            da.execute_parallel_actions(actions)
            _, dones = da.compute_step_rewards()
            if dones.get(agent_ids[0], False):
                break

    sort_idx      = np.argsort(valuations)
    action_matrix = action_matrix[sort_idx]
    sorted_vals   = np.array(valuations)[sort_idx]

    row_sums = action_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    action_norm = action_matrix / row_sums

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 7),
                                    gridspec_kw={"width_ratios": [3, 1]})
    fig.suptitle(
        f"Rozklad akcji per agent | D={d:.1f} | {n_episodes} epizodow (ZI baseline)",
        fontsize=12, fontweight="bold",
    )

    x_labels = [cfg.env.action_name(i) for i in range(n_actions)]
    im = ax1.imshow(action_norm, aspect="auto", cmap="YlOrRd", vmin=0, vmax=0.5)
    ax1.set_xticks(range(n_actions))
    ax1.set_xticklabels(x_labels, fontsize=10)
    ax1.set_xlabel("Akcja")
    ax1.set_ylabel("Agent (posortowany wg sentimentu rosnaco)")
    ax1.set_title("Heatmapa akcji (znorm. per agent)")
    plt.colorbar(im, ax=ax1, label="Frakcja wyborow")

    colors_v = [COLORS["buy"] if v > 0.0 else COLORS["sell"]
                for v in sorted_vals]
    ax2.barh(range(len(sorted_vals)), sorted_vals, color=colors_v, alpha=0.8)
    ax2.axvline(0.0, color=COLORS["eq"], ls="--", lw=2)
    ax2.set_xlabel("Sentiment agenta")
    ax2.set_title("Sentiment (rosnaco)")
    ax2.set_xlim(-1.0, 1.0)
    ax2.grid(True, axis="x", alpha=0.3)

    _tight()
    _save(fig, "07_action_heatmap.png")


# ===========================================================================
# 08. Ewolucja cen i sentimentu przez kolejne epizody
# ===========================================================================

def plot_price_valuation_evolution(n_episodes: int = 60,
                                   n_agents:   int = 20) -> None:
    """Jak ref_price i sentiment ewoluuja przez wiele epizodow."""
    cfg    = _cfg(n_agents=n_agents)
    D_SHOW = [0.3, 0.7, 1.0]
    D_COL3 = [COLORS["D0.2"], COLORS["D0.6"], COLORS["D1.0"]]

    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle(
        "Ewolucja cen i sentimentu przez kolejne epizody CT\n"
        "(cena NIE resetuje sie miedzy epizodami — ciaglsc historii)",
        fontsize=13, fontweight="bold",
    )

    for col, (d, color) in enumerate(zip(D_SHOW, D_COL3)):
        da = DoubleAuction(cfg, seed=42)
        da.reset(diversity_score=d, seed=42)

        agents_sorted = sorted(da.population.agents.items(),
                               key=lambda x: x[1].sentiment)
        n = len(agents_sorted)
        tracked = {
            "HIGH":  agents_sorted[-1][0],
            "HIGH2": agents_sorted[max(0, n - 4)][0],
            "MID":   agents_sorted[n // 2][0],
            "LOW2":  agents_sorted[min(3, n - 1)][0],
            "LOW":   agents_sorted[0][0],
        }
        track_colors = {
            "HIGH":  "#B71C1C", "HIGH2": "#E57373",
            "MID":   "#455A64",
            "LOW2":  "#64B5F6", "LOW":   "#1A237E",
        }

        ref_prices = [da.ref_price]
        val_all    = [[p.sentiment for p in da.population.agents.values()]]
        val_tracked = {k: [da.population.agents[v].sentiment]
                       for k, v in tracked.items()}
        trades_per_ep = []

        for ep in range(n_episodes):
            m = _run_zi_episode(da, cfg)
            ref_prices.append(da.ref_price)
            val_all.append([p.sentiment for p in da.population.agents.values()])
            for k, aid in tracked.items():
                val_tracked[k].append(da.population.agents[aid].sentiment)
            trades_per_ep.append(m["n_trades"])

        rounds  = np.arange(len(ref_prices))
        val_arr = np.array(val_all)
        val_mean, val_std = val_arr.mean(axis=1), val_arr.std(axis=1)

        ax0 = axes[0, col]
        ax0.fill_between(rounds,
                         np.clip(val_mean - val_std, 0.05, 0.95),
                         np.clip(val_mean + val_std, 0.05, 0.95),
                         alpha=0.15, color=color, label="±1sigma sentiment")
        ax0.plot(rounds, val_mean, color=color, lw=1.5, ls="--",
                 alpha=0.7, label="Sredni sentiment")
        ax0.plot(rounds, ref_prices, color=COLORS["eq"], lw=2.5,
                 label="ref_price")
        ax0.axhline(0.5, color="gray", ls=":", lw=1, alpha=0.5)
        ax0.set_title(f"D={d:.1f} — Cena vs sentiment",
                      fontsize=11, color=color, fontweight="bold")
        ax0.set_ylabel("Cena / sentiment")
        ax0.legend(fontsize=7)
        ax0.grid(True, alpha=0.3)

        ax1 = axes[1, col]
        for k, vals in val_tracked.items():
            ax1.plot(rounds, vals, color=track_colors[k], lw=1.8,
                     label=f"{k}")
        ax1.plot(rounds, ref_prices, color=COLORS["eq"], lw=1.5,
                 ls="--", alpha=0.5, label="ref_price")
        ax1.set_ylabel("Sentiment agenta")
        ax1.set_ylim(-1.0, 1.0)
        ax1.set_title(f"D={d:.1f} — Sentiment 5 agentow", fontsize=10)
        ax1.legend(fontsize=7)
        ax1.grid(True, alpha=0.3)
        fv = {k: val_tracked[k][-1] for k in tracked}
        order_ok = fv["HIGH"] > fv["MID"] > fv["LOW"]
        ax1.text(0.02, 0.05,
                 f"Kolejnosc HIGH>MID>LOW: {'TAK' if order_ok else 'NIE'}",
                 transform=ax1.transAxes, fontsize=8,
                 color="#2E7D32" if order_ok else "#C62828",
                 fontweight="bold")

        ax2 = axes[2, col]
        rolling = np.convolve(trades_per_ep, np.ones(10) / 10, mode="valid")
        ax2.bar(range(len(trades_per_ep)), trades_per_ep,
                color=color, alpha=0.3, width=1.0)
        ax2.plot(range(len(rolling)), rolling, color=color, lw=2)
        ax2.set_xlabel("Epizod")
        ax2.set_ylabel("Transakcji")
        ax2.set_title(f"D={d:.1f} — Aktywnosc rynku", fontsize=10)
        ax2.grid(True, alpha=0.3)

    _tight()
    _save(fig, "09_price_valuation_evolution.png")


# ===========================================================================
# 09. Ewolucja pozycji przez epizod (CT)
# ===========================================================================

def plot_position_evolution(diversity_scores: List[float] = None,
                             n_agents: int = 20) -> None:
    """Sladzi pozycje i realized P&L przez 200 krokow epizodu."""
    if diversity_scores is None:
        diversity_scores = [0.3, 0.7, 1.0]
    cfg = _cfg(n_agents=n_agents)
    fig, axes = plt.subplots(3, len(diversity_scores), figsize=(16, 10))
    fig.suptitle(
        "Ewolucja pozycji i Realized P&L przez epizod CT (T=200, ZI baseline)",
        fontsize=13, fontweight="bold",
    )

    for col, d in enumerate(diversity_scores):
        color = D_COLORS[min(int(d * 5), 5)]
        da    = DoubleAuction(cfg, seed=42)
        da.reset(diversity_score=d, seed=42)
        rng   = np.random.default_rng(42)

        zi = {aid: ZeroIntelligenceAgent(p, cfg.env)
              for aid, p in da.population.agents.items()}
        agent_ids = list(da.population.agents.keys())
        da.reset_episode()

        prices, mean_pos, cum_pnl_mean, cum_pnl_std = [], [], [], []
        cum_pnl = {aid: 0.0 for aid in agent_ids}

        for step in range(cfg.env.episode_steps):
            if da.done:
                break
            prices.append(da.ref_price)
            positions = [da.population.agents[a].position for a in agent_ids]
            mean_pos.append(float(np.mean(positions)))

            actions = {
                aid: zi[aid].act(da.get_observation(aid))
                for aid in agent_ids
            }
            da.execute_parallel_actions(actions)
            rewards, dones = da.compute_step_rewards()

            for aid in agent_ids:
                cum_pnl[aid] += rewards.get(aid, 0.0)
            pnl_vals = list(cum_pnl.values())
            cum_pnl_mean.append(float(np.mean(pnl_vals)))
            cum_pnl_std.append(float(np.std(pnl_vals)))

            if dones.get(agent_ids[0], False):
                break

        steps = np.arange(len(prices))

        ax0 = axes[0, col]
        ax0.plot(steps, prices, color=COLORS["eq"], lw=2)
        ax0.axhline(da.eq_price, color="gray", ls=":", alpha=0.5, label="eq")
        ax0.set_title(f"D={d:.1f} — Cena rynkowa",
                      fontsize=10, color=color, fontweight="bold")
        ax0.set_ylabel("Cena")
        ax0.set_ylim(0.25, 0.75)
        ax0.legend(fontsize=8)
        ax0.grid(True, alpha=0.3)

        ax1 = axes[1, col]
        ax1.plot(steps, mean_pos, color=color, lw=2)
        ax1.axhline(0, color="gray", ls="--", lw=1, alpha=0.7, label="neutral=0")
        ax1.fill_between(steps, mean_pos, 0, alpha=0.2, color=color)
        ax1.set_ylabel("Sr. pozycja populacji")
        ax1.set_title(f"D={d:.1f} — Pozycja (long>0, short<0)", fontsize=10)
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        ax2 = axes[2, col]
        mp = np.array(cum_pnl_mean)
        sp = np.array(cum_pnl_std)
        ax2.fill_between(steps, mp - sp, mp + sp, alpha=0.2, color=color)
        ax2.plot(steps, mp, color=color, lw=2, label="mean realized P&L")
        ax2.axhline(0, color="gray", ls=":", lw=1)
        m_fin = da.episode_metrics()
        pos_frac = m_fin["pnl_positive_agents"] / max(n_agents, 1)
        tacc     = m_fin["trade_accuracy"]
        ax2.text(0.98, 0.05,
                 f"pos_agents={pos_frac * 100:.0f}%\ntrade_acc={tacc:.3f}",
                 transform=ax2.transAxes, ha="right", fontsize=9,
                 color=COLORS["buy"] if pos_frac > 0.5 else COLORS["sell"])
        ax2.set_xlabel("Krok epizodu")
        ax2.set_ylabel("Kumulatywny realized P&L")
        ax2.set_title(f"D={d:.1f} — Realized P&L przez epizod", fontsize=10)
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

    _tight()
    _save(fig, "10_position_evolution.png")


# ===========================================================================
# 11. Rozklad P&L agentow vs D
# ===========================================================================

def plot_pnl_distribution(n_agents: int = 20, n_episodes: int = 30) -> None:
    """Box plot + violin: realized P&L agentow przy roznych D (ZI baseline)."""
    cfg    = _cfg(n_agents=n_agents)
    rng    = np.random.default_rng(42)
    D_SHOW = [0.0, 0.3, 0.5, 0.7, 1.0]

    all_pnls = {d: [] for d in D_SHOW}
    all_accs = {d: [] for d in D_SHOW}

    for d in D_SHOW:
        da = DoubleAuction(cfg, seed=42)
        da.reset(diversity_score=d, seed=42)
        for ep in range(n_episodes):
            ep_seed = int(rng.integers(0, 100_000))
            da.reset(diversity_score=d, seed=ep_seed)
            _run_zi_episode(da, cfg)
            am = da.agent_metrics()
            all_pnls[d].extend(m["ep_pnl"]        for m in am.values())
            all_accs[d].extend(m["trade_accuracy"] for m in am.values())

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "Rozklad Realized P&L i Trade Accuracy agentow przy roznych D (ZI baseline, CT)",
        fontsize=12, fontweight="bold",
    )

    data   = [all_pnls[d] for d in D_SHOW]
    colors = [D_COLORS[min(int(d * 5), 5)] for d in D_SHOW]
    labels = [f"D={d}" for d in D_SHOW]

    ax = axes[0]
    bp = ax.boxplot(data, patch_artist=True, labels=labels)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.set_title("Box plot — Realized P&L per agent per epizod")
    ax.set_ylabel("Realized P&L")
    ax.set_xlabel("Diversity Score D")
    ax.grid(True, alpha=0.3, axis="y")

    ax2 = axes[1]
    parts = ax2.violinplot(data, positions=range(len(D_SHOW)), showmeans=True)
    for pc, c in zip(parts["bodies"], colors):
        pc.set_facecolor(c)
        pc.set_alpha(0.7)
    ax2.set_xticks(range(len(D_SHOW)))
    ax2.set_xticklabels(labels)
    ax2.axhline(0, color="gray", ls="--", lw=1)
    ax2.set_title("Violin — rozklad Realized P&L")
    ax2.set_ylabel("Realized P&L")
    ax2.grid(True, alpha=0.3, axis="y")

    ax3 = axes[2]
    acc_data = [all_accs[d] for d in D_SHOW]
    parts3 = ax3.violinplot(acc_data, positions=range(len(D_SHOW)), showmeans=True)
    for pc, c in zip(parts3["bodies"], colors):
        pc.set_facecolor(c)
        pc.set_alpha(0.7)
    ax3.axhline(0.5, color="gray", ls="--", lw=1.5, label="neutral reference")
    ax3.set_xticks(range(len(D_SHOW)))
    ax3.set_xticklabels(labels)
    ax3.set_title("Trade accuracy per agent")
    ax3.set_ylabel("Trade accuracy")
    ax3.set_ylim(0, 1)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, axis="y")

    _tight()
    _save(fig, "11_pnl_distribution.png")


# ===========================================================================
# 12. Aktywnosc handlowa przez epizod
# ===========================================================================

def plot_trading_activity(diversity_scores: List[float] = None,
                           n_agents: int = 20) -> None:
    """Rozklad akcji HOLD/BUY/SELL i transakcji przez epizod (ZI)."""
    if diversity_scores is None:
        diversity_scores = [0.3, 0.7, 1.0]
    cfg = _cfg(n_agents=n_agents)
    fig, axes = plt.subplots(2, len(diversity_scores), figsize=(16, 8))
    fig.suptitle("Aktywnosc handlowa — CT (ZI baseline, T=200)",
                 fontsize=12, fontweight="bold")

    action_names  = ["HOLD", "BUY", "SELL"]
    action_colors = ["#90A4AE", "#2E7D32", "#C62828"]
    window = 10

    for col, d in enumerate(diversity_scores):
        color = D_COLORS[min(int(d * 5), 5)]
        da    = DoubleAuction(cfg, seed=42)
        da.reset(diversity_score=d, seed=42)
        rng   = np.random.default_rng(col)
        zi    = {aid: ZeroIntelligenceAgent(p, cfg.env)
                 for aid, p in da.population.agents.items()}
        agent_ids = list(da.population.agents.keys())
        da.reset_episode()

        action_counts = [[] for _ in range(3)]
        trade_counts  = []
        prev_trades   = 0

        for step in range(cfg.env.episode_steps):
            if da.done:
                break
            step_actions = {}
            for aid in agent_ids:
                obs    = da.get_observation(aid)
                action = zi[aid].act(obs)
                step_actions[aid] = action
            da.execute_parallel_actions(step_actions)
            _, dones = da.compute_step_rewards()

            for a_idx in range(3):
                action_counts[a_idx].append(
                    sum(1 for a in step_actions.values() if a == a_idx))
            curr = da.episode_metrics()["n_trades"]
            trade_counts.append(curr - prev_trades)
            prev_trades = curr

            if dones.get(agent_ids[0], False):
                break

        steps = np.arange(len(trade_counts))

        ax0 = axes[0, col]
        bottom = np.zeros(len(steps))
        for counts, aname, acolor in zip(action_counts, action_names, action_colors):
            smooth = np.convolve(counts, np.ones(window) / window, mode="same")
            ax0.fill_between(steps, bottom, bottom + smooth,
                             alpha=0.75, color=acolor, label=aname)
            bottom += smooth
        ax0.set_title(f"D={d:.1f} — Akcje (rolling {window})",
                      fontsize=10, color=color, fontweight="bold")
        if col == 0:
            ax0.set_ylabel("Liczba agentow")
        ax0.set_ylim(0, n_agents)
        if col == len(diversity_scores) - 1:
            ax0.legend(loc="upper right", fontsize=8)
        ax0.grid(True, alpha=0.3)

        ax1 = axes[1, col]
        smooth_t = np.convolve(trade_counts, np.ones(window) / window, mode="same")
        ax1.bar(steps, trade_counts, alpha=0.3, color=color, width=1)
        ax1.plot(steps, smooth_t, color=color, lw=2)
        ax1.set_xlabel("Krok epizodu")
        ax1.set_title(f"D={d:.1f} — Transakcje per krok", fontsize=10)
        if col == 0:
            ax1.set_ylabel("Trades")
        ax1.grid(True, alpha=0.3)

    _tight()
    _save(fig, "12_trading_activity.png")


# ===========================================================================
# Wykresy z wynikow treningu CSV
# ===========================================================================

def plot_role_emergence_from_results(csv_path: Optional[str] = None,
                                     quick: bool = False,
                                     final_window: Optional[int] = None) -> None:
    """Proporcje akcji, P&L i transakcje vs D bez ponownej symulacji."""
    df, path = _load_results_csv(csv_path, quick=quick)
    if df is None:
        return

    needed = {"action_buy_frac", "action_sell_frac", "action_hold_frac", "mean_pnl", "n_trades"}
    missing = sorted(needed - set(df.columns))
    if missing:
        print(f"  [!] CSV nie ma kolumn: {missing}")
        return

    max_episode = int(df["episode"].max())
    if final_window is None:
        final_window = min(50, max(1, (max_episode + 1) // 3))
    final = df[df["episode"] >= max_episode - final_window + 1]

    d_vals = sorted(final["diversity_score"].unique())
    colors = _result_colors(d_vals)
    x = np.arange(len(d_vals))
    w = 0.6

    grouped = final.groupby("diversity_score")
    buy_m = grouped["action_buy_frac"].mean().reindex(d_vals).to_numpy()
    sell_m = grouped["action_sell_frac"].mean().reindex(d_vals).to_numpy()
    hold_m = grouped["action_hold_frac"].mean().reindex(d_vals).to_numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Akcje i metryki Deep SARSA z wynikow treningu\n"
        f"{path.name}, ostatnie {final_window} epizodow",
        fontsize=12, fontweight="bold",
    )

    ax = axes[0]
    ax.bar(x, buy_m, w, label="BUY", color=COLORS["buy"], alpha=0.85)
    ax.bar(x, sell_m, w, bottom=buy_m, label="SELL", color=COLORS["sell"], alpha=0.85)
    ax.bar(x, hold_m, w, bottom=buy_m + sell_m, label="HOLD", color=COLORS["hold"], alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels([f"D={d:.1f}" for d in d_vals], rotation=30)
    ax.set_ylabel("Frakcja akcji"); ax.set_ylim(0, 1)
    ax.set_title("Proporcje akcji agentow")
    ax.legend(fontsize=9); ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1]
    pnl = grouped["mean_pnl"].agg(["mean", "std"]).reindex(d_vals).fillna(0)
    ax.errorbar(d_vals, pnl["mean"], yerr=pnl["std"], fmt="o-",
                color=COLORS["zi"], lw=2, capsize=5, ms=8)
    ax.axhline(0, color="gray", lw=1, alpha=0.5)
    ax.set_xlabel("Diversity Score D"); ax.set_ylabel("Mean realized P&L")
    ax.set_title("P&L jako funkcja D"); ax.grid(True, alpha=0.3)

    ax = axes[2]
    trades = grouped["n_trades"].agg(["mean", "std"]).reindex(d_vals).fillna(0)
    bars = ax.bar(d_vals, trades["mean"], width=0.12,
                  color=[colors[d] for d in d_vals], alpha=0.85, edgecolor="white")
    ax.errorbar(d_vals, trades["mean"], yerr=trades["std"], fmt="none",
                color="gray", capsize=4, lw=1.5)
    for bar, m in zip(bars, trades["mean"]):
        ax.text(bar.get_x() + bar.get_width() / 2, m + 0.1,
                f"{m:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Diversity Score D"); ax.set_ylabel("Transakcji / epizod")
    ax.set_title("Aktywnosc rynkowa vs D"); ax.grid(True, axis="y", alpha=0.3)

    _tight()
    _save(fig, "03_role_emergence.png")


def plot_training_metrics_from_results(csv_path: Optional[str] = None,
                                       quick: bool = False,
                                       rolling_window: int = 10) -> None:
    """Kompaktowe krzywe z CSV: P&L, transakcje, epsilon i TD error."""
    df, path = _load_results_csv(csv_path, quick=quick)
    if df is None:
        return

    d_vals = sorted(df["diversity_score"].unique())
    colors = _result_colors(d_vals)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    fig.suptitle(f"Deep SARSA — metryki treningu z CSV ({path.name})",
                 fontsize=12, fontweight="bold")

    panels = [
        ("mean_pnl", "Mean P&L"),
        ("n_trades", "Transakcje / epizod"),
        ("mean_epsilon", "Epsilon"),
        ("mean_td_error", "TD error"),
    ]

    for ax, (col, title) in zip(axes.flat, panels):
        if col not in df.columns:
            ax.set_visible(False)
            continue
        for d in d_vals:
            df_d = df[df["diversity_score"] == d]
            mean = df_d.groupby("episode")[col].mean()
            smooth = mean.rolling(rolling_window, min_periods=1).mean()
            ax.plot(mean.index, smooth, color=colors[d], lw=2, label=f"D={d:.1f}")
        if col == "mean_pnl":
            ax.axhline(0, color="gray", lw=1, alpha=0.5)
        ax.set_title(title)
        ax.set_xlabel("Epizod")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    _tight()
    _save(fig, "15_training_metrics_from_results.png")


# ===========================================================================
# 13. SARSA vs ZI — trade_accuracy przez epizody (czyta CSV)
# ===========================================================================

def plot_sarsa_vs_zi(csv_path: Optional[str] = None,
                     rolling_window: int = 20) -> None:
    """
    Glowny wykres artykulu (wersja rozbudowana z krzywymi uczenia i TD error).
    Czyta wyniki treningu z CSV.
    """
    try:
        import pandas as pd
    except ImportError:
        print("  [!] pandas nie dostepny")
        return

    if csv_path is None:
        csv_path = str(
            Path(__file__).parent.parent / "results" / "deep_sarsa_results.csv"
        )

    if not Path(csv_path).exists():
        print(f"  [!] Brak CSV: {csv_path} — uruchom trening najpierw")
        return

    df = pd.read_csv(csv_path)
    metric_col = "trade_accuracy"
    if metric_col not in df.columns:
        print("  [!] Brak trade_accuracy w CSV")
        return

    d_vals = sorted(df["diversity_score"].unique())
    n_cols = len(d_vals)
    cmap   = plt.cm.coolwarm
    colors = {d: cmap(i / max(n_cols - 1, 1)) for i, d in enumerate(d_vals)}

    zi_by_d = _baseline_by_d(df)

    fig = plt.figure(figsize=(5 * n_cols + 4, 8))
    gs  = fig.add_gridspec(2, n_cols + 1,
                            width_ratios=[1] * n_cols + [1.2])
    fig.suptitle(
        "SARSA vs ZI Baseline — trade accuracy przez epizody\n"
        "(linia ZI = empiryczny baseline z tego samego CSV)",
        fontsize=13, fontweight="bold",
    )

    sarsa_final = {}

    for i, d in enumerate(d_vals):
        df_d  = df[df["diversity_score"] == d]
        color = colors[d]

        grouped = df_d.groupby("episode")[metric_col]
        mean_s  = grouped.mean()
        std_s   = grouped.std().fillna(0)
        smooth  = mean_s.rolling(rolling_window, min_periods=1).mean()
        ep_idx  = mean_s.index

        sarsa_final[d] = float(smooth.iloc[-min(30, len(smooth)):].mean())

        ax_top = fig.add_subplot(gs[0, i])
        ax_top.fill_between(ep_idx,
                            np.clip(smooth - std_s, 0, 1),
                            np.clip(smooth + std_s, 0, 1),
                            alpha=0.15, color=color)
        ax_top.plot(ep_idx, smooth, color=color, lw=2.5, label="SARSA")
        zi_level = zi_by_d.get(d)
        if zi_level is not None:
            ax_top.axhline(zi_level, color=COLORS["zi"], ls="--", lw=1.5,
                           label=f"ZI={zi_level:.3f}", alpha=0.8)
            ax_top.axhspan(zi_level, 1.0, alpha=0.04, color="#2E7D32")

        final = float(smooth.iloc[-1])
        delta = final - zi_level if zi_level is not None else float("nan")
        c_d   = "#2E7D32" if delta > 0 else "#C62828"
        ax_top.annotate(
            f"delta={delta:+.3f}" if zi_level is not None else "brak ZI",
            xy=(ep_idx[-1], final),
            fontsize=9, color=c_d, fontweight="bold",
            ha="right", va="bottom",
        )

        ax_top.set_title(f"D = {d:.1f}", fontsize=11, color=color, fontweight="bold")
        ax_top.set_xlabel("Epizod")
        if i == 0:
            ax_top.set_ylabel("trade_accuracy")
        ax_top.set_ylim(0.0, 1.0)
        ax_top.legend(fontsize=8)
        ax_top.grid(True, alpha=0.3)

        ax_bot = fig.add_subplot(gs[1, i])
        if "mean_td_error" in df_d.columns:
            mean_td   = df_d.groupby("episode")["mean_td_error"].mean()
            smooth_td = mean_td.rolling(rolling_window, min_periods=1).mean()
            ax_bot.plot(mean_td.index, smooth_td, color=color, lw=2, label="TD error")
        if "mean_epsilon" in df_d.columns:
            mean_eps  = df_d.groupby("episode")["mean_epsilon"].mean()
            ax_bot2   = ax_bot.twinx()
            ax_bot2.plot(mean_eps.index, mean_eps, color="gray",
                         lw=1.5, ls=":", label="epsilon")
            ax_bot2.set_ylabel("epsilon", fontsize=8)
            ax_bot2.set_ylim(0, 0.4)
        ax_bot.set_xlabel("Epizod")
        if i == 0:
            ax_bot.set_ylabel("TD error")
        ax_bot.legend(fontsize=8)
        ax_bot.grid(True, alpha=0.3)

    # Prawy panel: slupki koncowe
    ax_bar = fig.add_subplot(gs[:, -1])
    x      = np.arange(len(d_vals))
    w      = 0.5
    s_vals = [sarsa_final.get(d, 0) for d in d_vals]

    ax_bar.bar(x, s_vals, w,
               color=[colors[d] for d in d_vals], alpha=0.85)
    zi_vals = [zi_by_d.get(d) for d in d_vals]
    if all(v is not None for v in zi_vals):
        ax_bar.plot(x, zi_vals, color=COLORS["zi"], ls="--", lw=2,
                    marker="o", label="ZI empirical", alpha=0.9)

    for xi, sv in enumerate(s_vals):
        zi_level = zi_by_d.get(d_vals[xi])
        delta = sv - zi_level if zi_level is not None else float("nan")
        c_d   = "#2E7D32" if delta > 0 else "#C62828"
        ax_bar.text(xi, min(sv + 0.02, 0.98),
                    f"{sv:.3f}  d={delta:+.3f}" if zi_level is not None else f"{sv:.3f}",
                    ha="center", fontsize=7, color=c_d, fontweight="bold")

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([f"D={d:.1f}" for d in d_vals], rotation=30)
    ax_bar.set_ylabel("trade_accuracy (ostatnie 30 ep)")
    ax_bar.set_title("Wynik koncowy vs empiryczny ZI", fontsize=10)
    ax_bar.set_ylim(0.0, 1.0)
    ax_bar.legend(fontsize=9)
    ax_bar.grid(True, axis="y", alpha=0.3)

    _tight()
    _save(fig, "13_sarsa_vs_zi_main.png")


# ===========================================================================
# 14. Trade accuracy curves — glowny wykres artykulu (prosty)
# ===========================================================================

def plot_trade_accuracy_curves(csv_path: Optional[str] = None,
                                rolling_window: int = 20) -> None:
    """
    Jeden panel per D — trade_accuracy SARSA vs empiryczny baseline ZI.
    Najbardziej czytelny wykres do artykulu.
    """
    try:
        import pandas as pd
    except ImportError:
        print("  [!] pandas nie dostepny")
        return

    if csv_path is None:
        csv_path = str(
            Path(__file__).parent.parent / "results" / "deep_sarsa_results.csv"
        )

    if not Path(csv_path).exists():
        print(f"  [!] Brak CSV: {csv_path} — uruchom trening najpierw")
        return

    df = pd.read_csv(csv_path)
    if "trade_accuracy" not in df.columns:
        print("  [!] Brak trade_accuracy w CSV — uruchom nowy trening")
        return

    d_vals = sorted(df["diversity_score"].unique())
    n_cols = len(d_vals)
    cmap   = plt.cm.coolwarm
    colors = {d: cmap(i / max(n_cols - 1, 1)) for i, d in enumerate(d_vals)}
    zi_by_d = _baseline_by_d(df)

    fig, axes = plt.subplots(1, n_cols, figsize=(4.5 * n_cols, 5), sharey=True)
    if n_cols == 1:
        axes = [axes]

    fig.suptitle(
        "Trade Accuracy — Deep SARSA per D\n"
        "(porównanie do empirycznego baseline'u ZI z tych samych warunków)",
        fontsize=13, fontweight="bold",
    )

    for ax, d in zip(axes, d_vals):
        color = colors[d]
        df_d  = df[df["diversity_score"] == d]

        mean_acc = df_d.groupby("episode")["trade_accuracy"].mean()
        std_acc  = df_d.groupby("episode")["trade_accuracy"].std().fillna(0)
        smooth   = mean_acc.rolling(rolling_window, min_periods=1).mean()
        ep_idx   = mean_acc.index

        ax.fill_between(ep_idx,
                        np.clip(smooth - std_acc, 0, 1),
                        np.clip(smooth + std_acc, 0, 1),
                        alpha=0.15, color=color)
        ax.plot(ep_idx, smooth, color=color, lw=2.5, label="SARSA")
        zi_level = zi_by_d.get(d)
        if zi_level is not None:
            ax.axhline(zi_level, color="gray", ls="--", lw=1.5,
                       alpha=0.8, label=f"ZI={zi_level:.3f}")
            ax.axhspan(zi_level, 1.0, alpha=0.04, color="#2E7D32")

        final = float(smooth.iloc[-min(30, len(smooth)):].mean())
        delta = final - zi_level if zi_level is not None else float("nan")
        c_d   = "#2E7D32" if delta > 0 else "#C62828"
        ax.text(0.97, 0.05,
                f"koncowe: {final:.3f}\ndelta={delta:+.3f}" if zi_level is not None
                else f"koncowe: {final:.3f}\nbrak ZI",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=9, color=c_d, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

        ax.set_title(f"D = {d:.1f}", fontsize=12, color=color, fontweight="bold")
        ax.set_xlabel("Epizod")
        ax.set_ylim(0.0, 1.0)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Trade accuracy")
    _tight()
    _save(fig, "14_trade_accuracy_curves.png")


# ===========================================================================
# Main
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wykresy HTM. Domyslnie czyta wyniki train_deep_sarsa z CSV.",
    )
    parser.add_argument("--csv", type=str, help="Sciezka do CSV z treningu.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Uzyj results/deep_sarsa_quick_results.csv.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=10,
        help="Okno wygładzania krzywych z CSV.",
    )
    parser.add_argument(
        "--simulate-diagnostics",
        action="store_true",
        help="Stary wolny tryb: generuje wykresy 1-12 przez nowe symulacje.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("HTM — Wykresy z wynikow treningu")
    print("=" * 60)

    df, csv_path = _load_results_csv(args.csv, quick=args.quick)
    if df is None:
        return
    print(f"CSV: {csv_path}")
    d_list = [float(d) for d in sorted(df["diversity_score"].unique())]
    print(f"Wiersze: {len(df)} | D={d_list}")

    print("\n[03] Akcje/P&L/transakcje z CSV...")
    plot_role_emergence_from_results(str(csv_path), final_window=args.rolling_window)

    print("[13] SARSA vs ZI (trade_accuracy, z CSV)...")
    plot_sarsa_vs_zi(str(csv_path), rolling_window=args.rolling_window)

    print("[14] Trade accuracy curves (z CSV)...")
    plot_trade_accuracy_curves(str(csv_path), rolling_window=args.rolling_window)

    print("[15] Metryki treningu (z CSV)...")
    plot_training_metrics_from_results(str(csv_path), rolling_window=args.rolling_window)

    if args.simulate_diagnostics:
        print("\n[simulate] Rozklady wycen agentow...")
        plot_valuation_distributions(n_agents=40, n_seeds=15)

        print("[simulate] Parametry heterogenicznosci...")
        plot_heterogeneity_parameters(n_agents=50, n_seeds=8)

        print("[simulate] Emergencja rol BUY/SELL/HOLD...")
        plot_role_emergence(n_episodes=80)

        print("[simulate] Dynamika ceny...")
        plot_price_dynamics([0.0, 0.5, 1.0])

        print("[simulate] Valuation vs Realized P&L (trade_accuracy)...")
        plot_valuation_vs_pnl(n_episodes=25)

        print("[simulate] ZI walidacja srodowiska...")
        plot_zi_validation(n_episodes=100)

        print("[simulate] Heatmapa akcji...")
        plot_action_heatmap(n_episodes=50, d=0.7)

        print("[simulate] Ewolucja cen i wycen przez epizody...")
        plot_price_valuation_evolution(n_episodes=60)

        print("[simulate] Ewolucja pozycji przez epizod...")
        plot_position_evolution()

        print("[simulate] Rozklad P&L i trade_accuracy vs D...")
        plot_pnl_distribution(n_episodes=20)

        print("[simulate] Aktywnosc handlowa...")
        plot_trading_activity()

    print("\n" + "=" * 60)
    print(f"Wykresy zapisane w: {PLOTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
