"""
analysis/visualize.py — Kompleksowe wykresy dla benchmarku HTM
==============================================================
Generuje wykresy pokazujące:
  1. Rozkłady wycen agentów przy różnych D (główna nowa cecha)
  2. Rozkłady gamma, wealth, threshold, parametrów behawioralnych
  3. Porównanie populacji D=0 vs D=1 (radar chart + histogramy)
  4. Dynamikę ceny i price discovery
  5. Rozkład surplusów (kto zarobił ile)
  6. Rozkład akcji buy/sell/pass (proporcje)
  7. Krzywe uczenia Deep SARSA vs ZI baseline
  8. Scatter: valuation vs surplus (kto skorzystał na heterogeniczności)

Uruchomienie:
    cd htm_project
    python analysis/visualize.py
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as ticker

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import HTMConfig, EnvConfig, MarketDynamics, LogConfig, ExpConfig
from envs.double_auction import (
    DoubleAuction, AgentPopulation, ZeroIntelligenceAgent,
    run_zi_baseline, _gini
)

# ===========================================================================
# Helper — uruchamia jeden epizod ZI (używany przez wszystkie wykresy)
# ===========================================================================

def _run_zi_episode(da, zi, cfg):
    """
    Uruchamia jeden epizod ZI baseline.
    Zwraca episode_metrics() po zakończeniu.
    Bezpieczna wersja: inkrementuje da._step przy PASS żeby epizod
    miał szansę się skończyć (bez tego pętla jest nieskończona).
    """
    step       = 0
    max_steps  = cfg.env.max_steps * 3   # limit bezpieczeństwa
    active_ids = list(da.population.agents.keys())

    while not da.done and step < max_steps:
        active = da.active_agents
        if not active:
            break

        aid    = active[step % len(active)]
        obs    = da.get_observation(aid)
        act_i, signal = zi[aid].act(obs, da.ref_price)

        _apply_zi_order(da, aid, act_i, signal, cfg)
        step += 1

    return da.episode_metrics()


def _apply_zi_order(da, aid, act_i, signal, cfg):
    """Mapuje akcję ZI na ofertę cenową i składa ją do rynku."""
    if act_i == cfg.env.ACTION_PASS or signal == "none":
        da._step += 1
        if da._step >= cfg.env.max_steps:
            da._done = True
        return
    p   = da.population.agents[aid]
    ref = da.ref_price
    T, M, F = cfg.env.limit_tight_offset, cfg.env.limit_med_offset, cfg.env.limit_far_offset
    if signal == "buy":
        gap = max(0.0, p.valuation - da.ref_price)
        price = {cfg.env.ACTION_MARKET:      p.valuation,
                 cfg.env.ACTION_LIMIT_TIGHT: da.ref_price + gap * 0.67,
                 cfg.env.ACTION_LIMIT_MED:   da.ref_price + gap * 0.33,
                 cfg.env.ACTION_LIMIT_FAR:   da.ref_price}.get(act_i, p.valuation)
        price = min(price, p.max_affordable_bid())
        da.submit(aid, float(np.clip(price, 0.001, 0.999)), "bid")
    else:
        gap = max(0.0, da.ref_price - p.valuation)
        price = {cfg.env.ACTION_MARKET:      p.valuation,
                 cfg.env.ACTION_LIMIT_TIGHT: da.ref_price - gap * 0.67,
                 cfg.env.ACTION_LIMIT_MED:   da.ref_price - gap * 0.33,
                 cfg.env.ACTION_LIMIT_FAR:   da.ref_price}.get(act_i, p.valuation)
        da.submit(aid, float(np.clip(price, 0.001, 0.999)), "ask")


logging.basicConfig(level=logging.WARNING)
PLOTS_DIR = Path(__file__).parent.parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Paleta kolorów — spójna przez cały artykuł
# ---------------------------------------------------------------------------

COLORS = {
    "D0.0": "#1A237E",   # ciemny niebieski
    "D0.2": "#1565C0",
    "D0.4": "#0288D1",
    "D0.6": "#00897B",
    "D0.8": "#F57F17",
    "D1.0": "#B71C1C",   # ciemna czerwień
    "buy":  "#2E7D32",
    "sell": "#C62828",
    "pass": "#78909C",
    "none": "#B0BEC5",
    "eq":   "#E65100",
    "zi":   "#616161",
    "sarsa":"#1565C0",
}

D_COLORS = [COLORS[f"D{d:.1f}"] for d in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]]
D_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def _cfg(n_agents=20, eq_spread=0.0) -> HTMConfig:
    """Pomocnicza konfiguracja do wykresów."""
    return HTMConfig(
        env=EnvConfig(n_agents=n_agents),
        market=MarketDynamics(eq_spread=eq_spread),
        log=LogConfig(level="WARNING"),
    )


# ===========================================================================
# 1. Rozkłady wycen agentów przy różnych D
# ===========================================================================

def plot_valuation_distributions(
    n_agents: int = 40,
    n_seeds:  int = 20,
    save:     bool = True,
) -> None:
    """
    Główny wykres modelu spekulacyjnego.

    Pokazuje jak rozkład prywatnych wycen zmienia się z D:
    D=0: wszyscy mają tę samą wycenę (no-trade theorem)
    D=1: szeroki rozkład → dużo transakcji, bogata dynamika

    Układ: 2×3 — jeden panel per D.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(
        "Rozkład prywatnych wycen agentów — model spekulacyjny\n"
        "(brak stałych ról: każdy agent może kupować lub sprzedawać)",
        fontsize=13, fontweight="bold", y=1.02
    )

    cfg = _cfg(n_agents=n_agents)
    eq  = 0.5
    rng = np.random.default_rng(42)

    for ax, (d, color) in zip(axes.flat, zip(D_VALUES, D_COLORS)):
        all_vals = []
        for s in range(n_seeds):
            pop = AgentPopulation(
                n_agents=n_agents, diversity_score=d,
                diversity_cfg=cfg.diversity, belief_cfg=cfg.beliefs,
                env_cfg=cfg.env, eq_price=eq, seed=s,
            )
            all_vals.extend([p.valuation for p in pop.agents.values()])

        all_vals = np.array(all_vals)

        # Histogram
        ax.hist(all_vals, bins=25, color=color, alpha=0.7, density=True,
                edgecolor="white", linewidth=0.5)

        # Linia eq_price
        ax.axvline(eq, color=COLORS["eq"], ls="--", lw=2,
                   label=f"eq = {eq:.2f}", zorder=5)

        # Obszary buy/sell
        ax.axvspan(eq, 1.0, alpha=0.05, color=COLORS["buy"])
        ax.axvspan(0.0, eq, alpha=0.05, color=COLORS["sell"])

        # Statystyki
        n_buy  = (all_vals > eq).sum()
        n_sell = (all_vals < eq).sum()
        n_none = (all_vals == eq).sum()

        ax.text(0.05, 0.95,
                f"σ = {all_vals.std():.3f}\n"
                f"↑buy: {n_buy/len(all_vals)*100:.0f}%\n"
                f"↓sell: {n_sell/len(all_vals)*100:.0f}%",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        ax.set_title(f"D = {d:.1f}", fontsize=12, color=color, fontweight="bold")
        ax.set_xlabel("Prywatna wycena aktywa")
        ax.set_ylabel("Gęstość" if ax in axes[:, 0] else "")
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Adnotacje buy/sell
        ax.text(0.75, 0.85, "KUP", transform=ax.transAxes,
                color=COLORS["buy"], fontsize=9, fontweight="bold", ha="center")
        ax.text(0.25, 0.85, "SPRZEDAJ", transform=ax.transAxes,
                color=COLORS["sell"], fontsize=9, fontweight="bold", ha="center")

    plt.tight_layout()
    _save(fig, "01_valuation_distributions.png", save)


# ===========================================================================
# 2. Rozkłady parametrów heterogeniczności
# ===========================================================================

def plot_heterogeneity_parameters(
    n_agents: int = 60,
    n_seeds:  int = 10,
    save:     bool = True,
) -> None:
    """
    4 panele pokazujące rozkłady kluczowych parametrów heterogeniczności
    przy D=0, D=0.5, D=1.0.
    """
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle("Rozkłady parametrów heterogeniczności przy D = 0 / 0.5 / 1.0",
                 fontsize=13, fontweight="bold")

    cfg = _cfg(n_agents=n_agents)

    params_to_plot = [
        ("gamma",        "Discount factor γ",        [0.5, 0.99], "Horyzont czasowy"),
        ("wealth",       "Majątek (wealth)",          [0, 5],      "Pareto wealth dist."),
        ("threshold",    "Próg decyzji (threshold)",  [0, 0.15],   "Min. |val-price| do handlu"),
        ("update_speed", "Szybkość uczenia (beliefs)",[0, 1],      "EMA alpha"),
        ("anchoring_bias","Zakotwiczenie",            [0, 1],      "Anchoring bias (Kahneman)"),
        ("loss_aversion","Awersja do strat",          [1, 3],      "Loss aversion (Kahneman)"),
        ("panic_factor", "Panic factor",              [0, 0.5],    "Panika przy spadku"),
        ("patience",     "Cierpliwość",               [0, 0.5],    "Czekanie na niższą cenę"),
    ]

    d_vals_sub = [0.0, 0.5, 1.0]
    d_colors   = [COLORS["D0.0"], COLORS["D0.4"], COLORS["D1.0"]]

    for ax, (param, label, xlim, desc) in zip(axes.flat, params_to_plot):
        for d, color in zip(d_vals_sub, d_colors):
            values = []
            for s in range(n_seeds):
                pop = AgentPopulation(
                    n_agents=n_agents, diversity_score=d,
                    diversity_cfg=cfg.diversity, belief_cfg=cfg.beliefs,
                    env_cfg=cfg.env, eq_price=0.5, seed=s,
                )
                for p in pop.agents.values():
                    if param in ("gamma", "threshold"):
                        values.append(getattr(p, param))
                    elif param == "wealth":
                        values.append(p.wealth)
                    else:
                        values.append(getattr(p.belief, param))

            values = np.array(values)
            ax.hist(values, bins=20, color=color, alpha=0.55, density=True,
                    label=f"D={d:.1f}", edgecolor="none")

        ax.set_xlabel(label, fontsize=9)
        ax.set_ylabel("Gęstość", fontsize=8)
        ax.set_title(desc, fontsize=9, style="italic")
        ax.set_xlim(xlim)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, "02_heterogeneity_parameters.png", save)


# ===========================================================================
# 3. Emergencja ról buy/sell/pass w zależności od D
# ===========================================================================

def plot_role_emergence(
    n_episodes: int = 300,
    save:       bool = True,
) -> None:
    """
    Pokazuje jak proporcje akcji buy/sell/pass/none zmieniają się z D.
    Kluczowy wykres dla modelu spekulacyjnego:
    D=0: prawie wszystko PASS/NONE (brak sygnałów)
    D=1: dużo BUY i SELL, mało PASS
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Emergencja ról buy/sell/pass jako funkcja Diversity Score D\n"
        "(ZI baseline — kierunek z sygnału, agresywność losowa)",
        fontsize=12, fontweight="bold"
    )

    cfg = _cfg()

    results = {d: {"buy": [], "sell": [], "pass": [], "none": [], "eff": [], "trades": []}
               for d in D_VALUES}

    rng = np.random.default_rng(42)

    for d in D_VALUES:
        for ep in range(n_episodes):
            ep_seed = int(rng.integers(0, 100_000))
            da  = DoubleAuction(cfg, seed=ep_seed)
            da.reset(diversity_score=d, seed=ep_seed)
            zi  = {
                aid: ZeroIntelligenceAgent(p, cfg.env, seed=ep_seed + i)
                for i, (aid, p) in enumerate(da.population.agents.items())
            }

            step = 0


            m = _run_zi_episode(da, zi, cfg)

            m = da.episode_metrics()
            results[d]["buy"].append(m["action_buy_frac"])
            results[d]["sell"].append(m["action_sell_frac"])
            results[d]["pass"].append(m["action_pass_frac"])
            results[d]["none"].append(m["action_none_frac"])
            results[d]["eff"].append(m["allocative_efficiency"])
            results[d]["trades"].append(m["n_trades"])

    d_arr = np.array(D_VALUES)

    # Panel 1: Proporcje akcji (stacked bar)
    ax = axes[0]
    buy_m   = [np.mean(results[d]["buy"])  for d in D_VALUES]
    sell_m  = [np.mean(results[d]["sell"]) for d in D_VALUES]
    pass_m  = [np.mean(results[d]["pass"]) for d in D_VALUES]
    none_m  = [np.mean(results[d]["none"]) for d in D_VALUES]

    x = np.arange(len(D_VALUES))
    w = 0.6
    ax.bar(x, buy_m,  w, label="BUY",  color=COLORS["buy"],  alpha=0.85)
    ax.bar(x, sell_m, w, bottom=buy_m, label="SELL", color=COLORS["sell"], alpha=0.85)
    bot2 = [b + s for b, s in zip(buy_m, sell_m)]
    ax.bar(x, pass_m, w, bottom=bot2, label="PASS", color=COLORS["pass"], alpha=0.85)
    bot3 = [b + p for b, p in zip(bot2, pass_m)]
    ax.bar(x, none_m, w, bottom=bot3, label="NONE\n(brak sygnału)", color=COLORS["none"], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"D={d}" for d in D_VALUES], rotation=30)
    ax.set_ylabel("Frakcja akcji")
    ax.set_title("Proporcje akcji agentów")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.3)

    # Panel 2: Efficiency vs D
    ax = axes[1]
    eff_m  = [np.mean(results[d]["eff"]) for d in D_VALUES]
    eff_s  = [np.std(results[d]["eff"])  for d in D_VALUES]
    ax.errorbar(D_VALUES, eff_m, yerr=eff_s, fmt="o-",
                color=COLORS["sarsa"], lw=2, capsize=5, ms=8, label="ZI baseline")
    ax.fill_between(D_VALUES,
                    [m - s for m, s in zip(eff_m, eff_s)],
                    [m + s for m, s in zip(eff_m, eff_s)],
                    alpha=0.15, color=COLORS["sarsa"])
    ax.set_xlabel("Diversity Score D")
    ax.set_ylabel("Allocative Efficiency")
    ax.set_title("Efficiency jako funkcja D")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # Adnotacja przy D=0
    ax.annotate("D=0: no-trade\n(Milgrom-Stokey)",
                xy=(0.0, eff_m[0]), xytext=(0.15, eff_m[0] - 0.25),
                arrowprops=dict(arrowstyle="->", color="gray"),
                fontsize=8, color="gray")

    # Panel 3: Liczba transakcji vs D
    ax = axes[2]
    trd_m = [np.mean(results[d]["trades"]) for d in D_VALUES]
    trd_s = [np.std(results[d]["trades"])  for d in D_VALUES]

    bars = ax.bar(D_VALUES, trd_m, width=0.12, color=D_COLORS, alpha=0.85,
                  edgecolor="white")
    ax.errorbar(D_VALUES, trd_m, yerr=trd_s, fmt="none",
                color="gray", capsize=4, lw=1.5)
    ax.set_xlabel("Diversity Score D")
    ax.set_ylabel("Średnia liczba transakcji / epizod")
    ax.set_title("Handel jako funkcja D")
    ax.grid(True, axis="y", alpha=0.3)

    for bar, m in zip(bars, trd_m):
        ax.text(bar.get_x() + bar.get_width()/2, m + 0.1,
                f"{m:.1f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    _save(fig, "03_role_emergence.png", save)


# ===========================================================================
# 4. Dynamika ceny rynkowej w jednym epizodzie
# ===========================================================================

def plot_price_dynamics(
    diversity_scores: List[float] = [0.0, 0.5, 1.0],
    save: bool = True,
) -> None:
    """
    Cena transakcyjna w czasie dla jednego epizodu przy różnych D.
    Pokazuje price discovery — jak szybko cena zbliża się do eq.
    """
    cfg = _cfg(n_agents=30)
    fig, axes = plt.subplots(1, len(diversity_scores), figsize=(5*len(diversity_scores), 5))
    fig.suptitle("Dynamika ceny rynkowej w jednym epizodzie", fontsize=12, fontweight="bold")

    if len(diversity_scores) == 1:
        axes = [axes]

    for ax, d in zip(axes, diversity_scores):
        da = DoubleAuction(cfg, seed=42)
        da.reset(diversity_score=d, seed=42)
        eq = da.eq_price

        zi = {
            aid: ZeroIntelligenceAgent(p, cfg.env, seed=i)
            for i, (aid, p) in enumerate(da.population.agents.items())
        }

        step = 0
        step_prices = []
        step_idx    = []

        step_limit = cfg.env.max_steps * 4
        while not da.done and step < step_limit:
            active = da.active_agents
            if not active: break
            aid = active[step % len(active)]
            obs = da.get_observation(aid)
            action_idx, signal = zi[aid].act(obs, da.ref_price)

            prev_n = len(da.order_book.price_history)

            _apply_zi_order(da, aid, action_idx, signal, cfg)

            new_n = len(da.order_book.price_history)
            if new_n > prev_n:
                step_prices.extend(da.order_book.price_history[prev_n:])
                step_idx.extend([step] * (new_n - prev_n))

            step += 1

        # Rysuj
        ax.axhline(eq, color=COLORS["eq"], ls="--", lw=2, label=f"eq = {eq:.2f}", alpha=0.8)
        ax.axhspan(eq - 0.05, eq + 0.05, alpha=0.08, color=COLORS["eq"])

        if step_prices:
            ax.scatter(step_idx, step_prices, s=60, zorder=5,
                       color=D_COLORS[int(d * 5)], label="Ceny transakcji")

            if len(step_prices) > 2:
                z    = np.polyfit(step_idx, step_prices, 1)
                poly = np.poly1d(z)
                x_tr = np.linspace(min(step_idx), max(step_idx), 100)
                ax.plot(x_tr, poly(x_tr), "--", color="gray", alpha=0.5, lw=1.5,
                        label="Trend")

        m = da.episode_metrics()
        ax.set_title(
            f"D = {d:.1f} | trades = {m['n_trades']} | "
            f"eff = {m['allocative_efficiency']:.3f}",
            fontsize=10
        )
        ax.set_xlabel("Krok (tura agenta)")
        ax.set_ylabel("Cena transakcyjna")
        ax.set_ylim(0.1, 0.9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, "04_price_dynamics.png", save)


# ===========================================================================
# 5. Valuation vs surplus — kto zarobił i dlaczego
# ===========================================================================

def plot_valuation_vs_surplus(
    n_episodes: int = 50,
    save:       bool = True,
) -> None:
    """
    Scatter plot: prywatna wycena agenta vs jego surplus.
    Pokazuje czy agenci z ekstremalnymi wycenami zyskują więcej.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Prywatna wycena vs surplus — kto korzysta na heterogeniczności?",
        fontsize=12, fontweight="bold"
    )

    cfg = _cfg()
    rng = np.random.default_rng(42)

    for ax, d in zip(axes, [0.2, 0.5, 1.0]):
        all_vals     = []
        all_surplus  = []
        all_signals  = []

        for ep in range(n_episodes):
            ep_seed = int(rng.integers(0, 100_000))
            da = DoubleAuction(cfg, seed=ep_seed)
            da.reset(diversity_score=d, seed=ep_seed)
            zi = {
                aid: ZeroIntelligenceAgent(p, cfg.env, seed=ep_seed + i)
                for i, (aid, p) in enumerate(da.population.agents.items())
            }

            step = 0


            m = _run_zi_episode(da, zi, cfg)

            am = da.agent_metrics()
            for aid, m in am.items():
                all_vals.append(m["valuation"])
                all_surplus.append(m["surplus"])
                all_signals.append(m["signal"])

        vals    = np.array(all_vals)
        surplus = np.array(all_surplus)
        signals = all_signals

        # Koloruj według sygnału (buy/sell/none)
        colors_scatter = [
            COLORS["buy"]  if s == "buy"  else
            COLORS["sell"] if s == "sell" else
            COLORS["none"]
            for s in signals
        ]

        ax.scatter(vals, surplus, c=colors_scatter, alpha=0.4, s=20, edgecolors="none")
        ax.axvline(0.5, color=COLORS["eq"], ls="--", lw=1.5, alpha=0.7, label="eq=0.5")
        ax.axhline(0.0, color="gray",        ls="-",  lw=0.8, alpha=0.5)

        # Trend liniowy
        if len(vals) > 5:
            z    = np.polyfit(vals, surplus, 1)
            poly = np.poly1d(z)
            xr   = np.linspace(vals.min(), vals.max(), 100)
            ax.plot(xr, poly(xr), "k--", lw=1.5, alpha=0.7, label="Trend")

        # Legenda
        patches = [
            mpatches.Patch(color=COLORS["buy"],  label=f"BUY signal"),
            mpatches.Patch(color=COLORS["sell"], label=f"SELL signal"),
            mpatches.Patch(color=COLORS["none"], label=f"PASS/NONE"),
        ]
        ax.legend(handles=patches, fontsize=8, loc="upper left")

        ax.set_xlabel("Prywatna wycena aktywa")
        ax.set_ylabel("Surplus z transakcji (0 = brak transakcji)")
        ax.set_title(f"D = {d:.1f} | N={len(all_vals)} obserwacji")
        ax.set_xlim(0.05, 0.95)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, "05_valuation_vs_surplus.png", save)


# ===========================================================================
# 6. ZI baseline — efficiency i trades vs D (walidacja środowiska)
# ===========================================================================

def plot_zi_validation(
    n_episodes: int = 300,
    save:       bool = True,
) -> None:
    """
    Walidacja środowiska: ZI baseline metrics vs D.
    Analogia do wykresu Gode & Sunder ale dla modelu spekulacyjnego.
    """
    cfg = _cfg()

    results = {}
    print("Liczę ZI baseline validation (to zajmie chwilę)...")
    for d in D_VALUES:
        results[d] = run_zi_baseline(cfg, diversity_score=d, n_episodes=n_episodes, seed=42)
        print(f"  D={d:.1f} | eff={results[d]['allocative_efficiency']['mean']:.3f}")

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle(
        "ZI Baseline — walidacja środowiska spekulacyjnego\n"
        "(analogia do Gode & Sunder 1993, model spekulacyjny)",
        fontsize=12, fontweight="bold"
    )

    metrics = [
        ("allocative_efficiency", "Allocative Efficiency", "Główna metryka"),
        ("gini_coefficient",      "Gini Coefficient",      "Nierówność wyników"),
        ("n_trades",              "Liczba transakcji",     "Aktywność rynkowa"),
        ("action_pass_frac",      "Frakcja akcji PASS",    "Inercja rynku"),
    ]

    for ax, (key, ylabel, title) in zip(axes, metrics):
        means = [results[d][key]["mean"] for d in D_VALUES]
        stds  = [results[d][key]["std"]  for d in D_VALUES]

        ax.errorbar(D_VALUES, means, yerr=stds,
                    fmt="o-", color=COLORS["sarsa"], lw=2.5, capsize=5,
                    ms=8, label="ZI baseline")
        ax.fill_between(D_VALUES,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.15, color=COLORS["sarsa"])

        for d_val, m_val, color in zip(D_VALUES, means, D_COLORS):
            ax.scatter([d_val], [m_val], color=color, s=80, zorder=6)

        ax.set_xlabel("Diversity Score D")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)

        if key == "allocative_efficiency":
            ax.set_ylim(0, 1.05)
            ax.annotate("D=0:\nbrak handlu", xy=(0, means[0]),
                        xytext=(0.1, means[0] + 0.1),
                        arrowprops=dict(arrowstyle="->", color="gray"),
                        fontsize=8, color="gray")
        elif key == "gini_coefficient":
            ax.set_ylim(0, 1)

    plt.tight_layout()
    _save(fig, "06_zi_validation.png", save)


# ===========================================================================
# 7. Rozkład akcji per agent — heatmapa agresywności
# ===========================================================================

def plot_action_heatmap(
    n_episodes: int = 100,
    d:          float = 0.7,
    save:       bool = True,
) -> None:
    """
    Heatmapa: który agent wybiera jaką agresywność oferty.
    Pokazuje że heterogeniczni agenci uczą się różnych strategii.
    """
    cfg = _cfg(n_agents=20)
    da  = DoubleAuction(cfg, seed=42)
    rng = np.random.default_rng(42)

    n_actions = cfg.env.n_actions
    n_agents  = cfg.env.n_agents
    action_matrix = np.zeros((n_agents, n_actions))
    valuations    = None

    for ep in range(n_episodes):
        ep_seed = int(rng.integers(0, 100_000))
        da.reset(diversity_score=d, seed=ep_seed)

        if valuations is None:
            agent_ids  = list(da.population.agents.keys())
            valuations = [da.population.agents[aid].valuation for aid in agent_ids]

        zi = {
            aid: ZeroIntelligenceAgent(p, cfg.env, seed=ep_seed + i)
            for i, (aid, p) in enumerate(da.population.agents.items())
        }

        step = 0
        agent_ids_ep = list(da.population.agents.keys())

        step_limit = cfg.env.max_steps * 4
        while not da.done and step < step_limit:
            active = da.active_agents
            if not active: break
            aid = active[step % len(active)]
            obs = da.get_observation(aid)
            action_idx, signal = zi[aid].act(obs, da.ref_price)

            i_agent = agent_ids_ep.index(aid) if aid in agent_ids_ep else 0
            if i_agent < n_agents:
                action_matrix[i_agent, action_idx] += 1

            _apply_zi_order(da, aid, action_idx, signal, cfg)
            step += 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7),
                                    gridspec_kw={"width_ratios": [3, 1]})
    fig.suptitle(
        f"Rozkład akcji per agent | D={d:.1f} | {n_episodes} epizodów\n"
        f"(ZI baseline: losowa agresywność, kierunek z sygnału)",
        fontsize=12, fontweight="bold"
    )

    # Znormalizuj wierszami
    row_sums = action_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    action_norm = action_matrix / row_sums

    # Etykiety osi X
    x_labels = [cfg.env.action_name(i) for i in range(cfg.env.n_actions)]

    im = ax1.imshow(action_norm, aspect="auto", cmap="YlOrRd", vmin=0, vmax=0.4)
    ax1.set_xticks(range(n_actions))
    ax1.set_xticklabels(x_labels, fontsize=8)
    ax1.set_xlabel("Akcja (agresywność → PASS)")
    ax1.set_ylabel("Agent (posortowany wg wyceny)")
    ax1.set_title("Heatmapa akcji (znorm. per agent)")

    # Zaznacz PASS kolumnę
    ax1.axvline(cfg.env.ACTION_PASS + 0.5, color="blue", lw=2, ls="--", alpha=0.7)
    ax1.text(0, -0.8, "PASS", color="blue", fontsize=9, ha="center")

    plt.colorbar(im, ax=ax1, label="Frakcja wyborów")

    # Panel 2: wyceny agentów
    if valuations:
        sorted_vals = sorted(valuations)
        colors_v = [COLORS["buy"] if v > 0.5 else COLORS["sell"] for v in sorted_vals]
        ax2.barh(range(len(sorted_vals)), sorted_vals, color=colors_v, alpha=0.8)
        ax2.axvline(0.5, color=COLORS["eq"], ls="--", lw=2, label="eq=0.5")
        ax2.set_xlabel("Wycena agenta")
        ax2.set_title("Wyceny agentów")
        ax2.legend(fontsize=8)
        ax2.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    _save(fig, "07_action_heatmap.png", save)


# ===========================================================================
# 8. Wealth distribution — Pareto
# ===========================================================================

def plot_wealth_distribution(save: bool = True) -> None:
    """
    Rozkład majątku agentów przy D=0, 0.5, 1.0 (rozkład Pareto).
    """
    cfg = _cfg(n_agents=200)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Rozkład majątku agentów — interpolacja między równym a Pareto",
                 fontsize=12, fontweight="bold")

    for ax, (d, color) in zip(axes, [(0.0, D_COLORS[0]), (0.5, D_COLORS[3]), (1.0, D_COLORS[5])]):
        pop = AgentPopulation(
            n_agents=200, diversity_score=d,
            diversity_cfg=cfg.diversity, belief_cfg=cfg.beliefs,
            env_cfg=cfg.env, eq_price=0.5, seed=42,
        )
        wealth = np.array([p.wealth for p in pop.agents.values()])
        gini_w = _gini(wealth.tolist())

        ax.hist(np.clip(wealth, 0, 8), bins=40, color=color, alpha=0.8,
                density=True, edgecolor="white", lw=0.5)
        ax.set_xlabel("Majątek agenta (wealth)")
        ax.set_ylabel("Gęstość")
        ax.set_title(f"D = {d:.1f} | Gini = {gini_w:.3f}")
        ax.text(0.7, 0.95, f"σ={wealth.std():.2f}\nmax={wealth.max():.1f}",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, "08_wealth_distribution.png", save)


# ===========================================================================
# Pomocnicze
# ===========================================================================

# ===========================================================================
# 9. Ewolucja cen i wycen przez rundy — continuous market dynamics
# ===========================================================================

def plot_price_valuation_evolution(
    n_rounds:   int   = 150,
    n_agents:   int   = 20,
    save:       bool  = True,
) -> None:
    """
    Pokazuje jak cena rynkowa i wyceny agentów ewoluują przez wiele rund
    bez resetu — kluczowy wykres dla modelu continuous market dynamics.

    Górny panel: ref_price przez rundy + pasmo ±1σ wycen
    Środkowy panel: wyceny 5 wybranych agentów (HIGH/LOW/MID + 2 losowych)
    Dolny panel: liczba transakcji per runda
    """
    cfg = _cfg(n_agents=n_agents)
    rng = np.random.default_rng(42)

    D_SHOW   = [0.3, 0.7, 1.0]
    D_COLORS_3 = [COLORS["D0.2"], COLORS["D0.6"], COLORS["D1.0"]]

    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle(
        "Ewolucja cen i wycen przez kolejne rundy handlowe\n(bez resetu ceny i wycen — continuous market dynamics)",
        fontsize=13, fontweight="bold"
    )

    for col, (d, color) in enumerate(zip(D_SHOW, D_COLORS_3)):
        da = DoubleAuction(cfg, seed=42)
        da.reset(diversity_score=d, seed=42)

        # Wybierz reprezentatywnych agentów do śledzenia
        agents_by_val = sorted(
            da.population.agents.items(), key=lambda x: x[1].valuation
        )
        tracked = {
            "HIGH":  agents_by_val[-1][0],
            "HIGH2": agents_by_val[-4][0],
            "MID":   agents_by_val[len(agents_by_val)//2][0],
            "LOW2":  agents_by_val[3][0],
            "LOW":   agents_by_val[0][0],
        }
        track_colors = {
            "HIGH":  "#B71C1C",
            "HIGH2": "#E57373",
            "MID":   "#455A64",
            "LOW2":  "#64B5F6",
            "LOW":   "#1A237E",
        }

        # Zbierz dane przez n_rounds rund
        ref_prices  = [da.ref_price]
        val_all     = [[p.valuation for p in da.population.agents.values()]]
        val_tracked = {k: [da.population.agents[v].valuation] for k, v in tracked.items()}
        trades_per_round = []

        for rnd in range(n_rounds):
            obs = da.reset_market_only()
            step = 0
            n_trades = 0
            while not da.done and step < 30:
                active = da.active_agents
                if not active: break
                actions = {a: cfg.env.ACTION_MARKET
                           if da.population.agents[a].trade_signal(da.ref_price) != "none"
                           else cfg.env.ACTION_PASS for a in active}
                _, rewards, _, _ = da.parallel_step(actions)
                n_trades += sum(1 for r in rewards.values() if r > 0)
                step += 1

            ref_prices.append(da.ref_price)
            val_all.append([p.valuation for p in da.population.agents.values()])
            for k, aid in tracked.items():
                val_tracked[k].append(da.population.agents[aid].valuation)
            trades_per_round.append(n_trades)

        rounds = np.arange(len(ref_prices))
        val_arr = np.array(val_all)
        val_mean = val_arr.mean(axis=1)
        val_std  = val_arr.std(axis=1)

        # ── Panel górny: ref_price + pasmo wycen ──────────────────────
        ax0 = axes[0, col]
        ax0.fill_between(rounds,
                         np.clip(val_mean - val_std, 0.05, 0.95),
                         np.clip(val_mean + val_std, 0.05, 0.95),
                         alpha=0.15, color=color, label="±1σ wycen")
        ax0.plot(rounds, val_mean, color=color, lw=1.5, ls="--",
                 alpha=0.7, label="Średnia wycena")
        ax0.plot(rounds, ref_prices, color="#E65100", lw=2.5,
                 label="Cena rynkowa")
        ax0.axhline(0.5, color="gray", ls=":", lw=1, alpha=0.5, label="eq=0.5")

        ax0.set_title(f"D={d:.1f} — Cena vs wyceny",
                      fontsize=11, color=color, fontweight="bold")
        ax0.set_ylabel("Cena / Wycena")
        ax0.set_ylim(0.1, 0.9)
        ax0.legend(fontsize=7, loc="upper right")
        ax0.grid(True, alpha=0.3)

        # ── Panel środkowy: wyceny 5 agentów ──────────────────────────
        ax1 = axes[1, col]
        for k, vals in val_tracked.items():
            base = da.population.agents[tracked[k]].base_valuation
            ax1.plot(rounds, vals, color=track_colors[k], lw=1.8,
                     label=f"{k} (base={base:.2f})")
        ax1.plot(rounds, ref_prices, color="#E65100", lw=1.5,
                 ls="--", alpha=0.6, label="ref_price")

        ax1.set_ylabel("Wycena agenta")
        ax1.set_title(f"D={d:.1f} — Ewolucja wycen 5 agentów", fontsize=10)
        ax1.set_ylim(0.1, 0.9)
        ax1.legend(fontsize=7, loc="upper right")
        ax1.grid(True, alpha=0.3)

        # Adnotacja: czy kolejność się utrzymuje?
        final_vals = {k: val_tracked[k][-1] for k in tracked}
        order_ok = final_vals["HIGH"] > final_vals["MID"] > final_vals["LOW"]
        ax1.text(0.02, 0.05,
                 f"Kolejność HIGH>MID>LOW: {'✓' if order_ok else '✗'}",
                 transform=ax1.transAxes, fontsize=8,
                 color="#2E7D32" if order_ok else "#C62828",
                 fontweight="bold")

        # ── Panel dolny: transakcje ────────────────────────────────────
        ax2 = axes[2, col]
        rolling_trades = np.convolve(trades_per_round, np.ones(10)/10, mode="valid")
        ax2.bar(range(len(trades_per_round)), trades_per_round,
                color=color, alpha=0.3, width=1.0)
        ax2.plot(range(len(rolling_trades)), rolling_trades,
                 color=color, lw=2, label="Śr. krocząca (10)")
        ax2.set_xlabel("Runda")
        ax2.set_ylabel("Transakcji / runda")
        ax2.set_title(f"D={d:.1f} — Aktywność rynku", fontsize=10)
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        # Sprawdź czy rynek umiera
        mean_early = np.mean(trades_per_round[:20]) if len(trades_per_round) >= 20 else 0
        mean_late  = np.mean(trades_per_round[-20:]) if len(trades_per_round) >= 20 else 0
        status = "aktywny ✓" if mean_late > 0.5 else "umiera ✗"
        ax2.text(0.98, 0.95, f"{mean_early:.1f}→{mean_late:.1f} {status}",
                 transform=ax2.transAxes, fontsize=8, ha="right", va="top",
                 color="#2E7D32" if mean_late > 0.5 else "#C62828")

    plt.tight_layout()
    _save(fig, "09_price_valuation_evolution.png", save)


def _save(fig, filename: str, save: bool) -> None:
    if save:
        path = PLOTS_DIR / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  ✓ Zapisano: {path}")
    plt.close(fig)


# ===========================================================================
# Main — generuj wszystkie wykresy
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("HTM — Generowanie wykresów diagnostycznych")
    print("=" * 60)

    print("\n[1/8] Rozkłady wycen agentów...")
    plot_valuation_distributions(n_agents=40, n_seeds=15)

    print("[2/8] Parametry heterogeniczności...")
    plot_heterogeneity_parameters(n_agents=50, n_seeds=8)

    print("[3/8] Emergencja ról buy/sell/pass...")
    plot_role_emergence(n_episodes=200)

    print("[4/8] Dynamika ceny...")
    plot_price_dynamics(diversity_scores=[0.0, 0.5, 1.0])

    print("[5/8] Valuation vs surplus...")
    plot_valuation_vs_surplus(n_episodes=30)

    print("[6/8] ZI validation (najdłuższy)...")
    plot_zi_validation(n_episodes=200)

    print("[7/8] Heatmapa akcji...")
    plot_action_heatmap(n_episodes=80, d=0.7)

    print("[8/8] Rozkład majątku...")
    plot_wealth_distribution()

    print("[9/9] Ewolucja cen i wycen...")
    plot_price_valuation_evolution(n_rounds=150)

    print("\n" + "=" * 60)
    print(f"Wszystkie wykresy zapisane w: {PLOTS_DIR}")
    print("=" * 60)