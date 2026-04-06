"""
experiments/train_deep_sarsa.py
================================
Pętla treningowa Deep SARSA na benchmarku HTM (model spekulacyjny).

Uruchomienie:
    cd htm_project
    python experiments/train_deep_sarsa.py

Co robi:
  1. Liczy ZI baseline dla każdego D (punkt odniesienia)
  2. Trenuje Deep SARSA przez N_EPISODES epizodów per D
  3. Loguje metryki co LOG_EVERY epizodów
  4. Generuje wykresy krzywych uczenia
  5. Zapisuje wyniki do results/deep_sarsa_results.csv

Parametry do zmiany na górze pliku — nie trzeba grzebać w kodzie.
"""

import sys
import logging
import time
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Upewnij się że Python widzi katalog projektu
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from multiprocessing import Pool, cpu_count
from config import HTMConfig, EnvConfig, MarketDynamics, LogConfig, DeepSARSAConfig
from envs.double_auction import DoubleAuction, ZeroIntelligenceAgent, run_zi_baseline
from agents.deep_sarsa import DeepSARSAMultiAgent

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "logs" / "deep_sarsa.log", mode="w"),
    ],
)
UPDATE_EVERY = 1   # aktualizuj sieć co krok — przy batch=1 numpy jest wystarczająco szybki
log = logging.getLogger("htm.train")


# ---------------------------------------------------------------------------
# Parametry eksperymentu — zmień tu
# ---------------------------------------------------------------------------

DIVERSITY_SCORES = [0.0, 0.3, 0.5, 0.7, 1.0]   # wartości D do przetestowania
N_AGENTS         = 20                             # liczba agentów
N_EPISODES       = 300                            # epizodów CT (każdy = 200 kroków)
# N_ROUNDS usunięty — CT nie ma rund, epizod = T=200 kroków
N_SEEDS          = 3                              # powtórzeń (dla std)
N_WORKERS        = min(cpu_count(), N_SEEDS * len([0.0, 0.3, 0.5, 0.7, 1.0]))
                                                  # równoległe procesy (auto: liczba corów)
LOG_EVERY        = 25                             # loguj co ile epizodów
ROLLING_WINDOW   = 30                             # okno wygładzania krzywych
ZI_EPISODES      = 30                             # epizodów do policzenia ZI baseline (populacja nie resetuje się co ep)

# Warunek rynkowy: stable / random_eq / drifting
MARKET = MarketDynamics.stable()

# Hiperparametry sieci
SARSA_CFG = DeepSARSAConfig(
    hidden_size   = 32,
    lr            = 1e-3,
    epsilon_start = 0.35,
    epsilon_end   = 0.05,
    epsilon_decay = 0.993,
    grad_clip     = 1.0,
)


# ---------------------------------------------------------------------------
# ZI baseline — liczy się raz przed treningiem
# ---------------------------------------------------------------------------

def compute_zi_baseline(
    diversity_score: float,
    cfg:             HTMConfig,
    n_episodes:      int = 100,
    seed:            int = 42,
) -> float:
    """
    Wrapper wokoł run_zi_baseline() z double_auction.py.
    Zwraca mean allocative_efficiency jako float (punkt odniesienia dla SARSA).
    """
    result = run_zi_baseline(cfg, diversity_score=diversity_score,
                             n_episodes=n_episodes, seed=seed)
    return result["allocative_efficiency"]["mean"]


# ---------------------------------------------------------------------------
# Jeden run treningowy Deep SARSA
# ---------------------------------------------------------------------------

def run_training(
    diversity_score: float,
    n_episodes:      int,
    seed:            int,
    cfg:             HTMConfig,
    zi_baseline:     float,
) -> List[dict]:
    """
    Trenuje Deep SARSA — Continuous Trading.

    Jeden epizod = T=200 kroków (wszyscy agenci aktywni przez cały czas).
    Między epizodami: portfele resetowane, wyceny dryfują (pamięć rynku).
    Gamma jest istotna w każdym kroku (done=True dopiero po T krokach).
    """
    da = DoubleAuction(cfg, seed=seed)
    da.reset(diversity_score=diversity_score, seed=seed)

    agent_ids    = list(da.population.agents.keys())
    agent_gammas = np.array([da.population.agents[aid].gamma for aid in agent_ids])

    log.info(
        f"  Populacja | N={len(agent_ids)} | "
        f"val=[{min(da.population.agents[a].valuation for a in agent_ids):.2f}, "
        f"{max(da.population.agents[a].valuation for a in agent_ids):.2f}] | "
        f"gamma=[{agent_gammas.min():.2f}, {agent_gammas.max():.2f}]"
    )

    sarsa = DeepSARSAMultiAgent(
        agent_ids=agent_ids, agent_gammas=agent_gammas,
        n_obs=cfg.env.n_obs, n_actions=cfg.env.n_actions,
        cfg=SARSA_CFG, seed=seed,
    )

    # Parametry populacji — stałe przez cały run, zapisywane do każdego rekordu
    _ap = da.population.agents
    pop_meta = {
        "pop_mean_valuation":     float(np.mean([_ap[a].valuation     for a in agent_ids])),
        "pop_std_valuation":      float(np.std( [_ap[a].valuation     for a in agent_ids])),
        "pop_mean_gamma":         float(np.mean([_ap[a].gamma         for a in agent_ids])),
        "pop_std_gamma":          float(np.std( [_ap[a].gamma         for a in agent_ids])),
        "pop_mean_risk_aversion": float(np.mean([_ap[a].risk_aversion for a in agent_ids])),
        "pop_mean_threshold":     float(np.mean([_ap[a].threshold     for a in agent_ids])),
        "pop_mean_max_position":  float(np.mean([_ap[a].max_position  for a in agent_ids])),
    }

    records  = []
    t_start  = time.time()

    step_rng = np.random.default_rng(seed + 1000)

    for episode in range(n_episodes):

        # Nowy epizod: reset portfeli, wyceny dryfują, cena zostaje
        da.reset_episode()
        ep_rewards = {aid: 0.0 for aid in agent_ids}
        step = 0

        while not da.done:
            # Sekwencyjne wykonanie — losowa kolejność agentów per krok
            # Każdy agent obserwuje rynek ZAKTUALIZOWANY przez poprzedników
            agent_order = step_rng.permutation(agent_ids)
            obs_at_action  = {}
            actions_taken  = {}

            for aid in agent_order:
                obs = da.get_observation(aid)          # aktualny stan rynku
                obs_at_action[aid] = obs
                action = sarsa.agents[aid].act(obs, explore=True)
                actions_taken[aid] = action
                da.execute_single_action(aid, action)  # natychmiastowe wykonanie

            # Nagrody na końcu kroku (po wszystkich agentach)
            rewards, dones = da.compute_step_rewards()

            # Aktualizacja SARSA dla każdego agenta
            for aid in agent_ids:
                r = rewards.get(aid, 0.0)
                ep_rewards[aid] += r
                next_obs = da.get_observation(aid)
                sarsa.agents[aid].update(
                    obs      = obs_at_action[aid],
                    action   = actions_taken[aid],
                    reward   = r,
                    next_obs = next_obs,
                    done     = dones.get(aid, False),
                )

            step += 1

        # Metryki epizodu
        m     = da.episode_metrics()
        pop_s = sarsa.population_stats()
        sarsa.end_episode()

        mean_pnl       = m.get("mean_pnl", 0.0)
        n_trades       = m.get("n_trades", 0)
        eff            = m.get("allocative_efficiency", 0.0)
        n_pos          = m.get("pnl_positive_agents", 0)
        pnl_pos_frac   = n_pos / max(N_AGENTS, 1)
        trade_acc      = m.get("trade_accuracy", 0.5)

        record = {
            "episode":           episode,
            "diversity_score":   diversity_score,
            "seed":              seed,
            "algorithm":         "DeepSARSA_CT",
            "n_agents":          N_AGENTS,
            "eq_price":          m.get("eq_price", 0.5),
            "ref_price_final":   m.get("ref_price_final", 0.5),
            "mean_pnl":          mean_pnl,
            "pnl_positive_frac": pnl_pos_frac,
            "trade_accuracy":    trade_acc,        # GŁÓWNA METRYKA: > 0.5 = lepszy niż ZI
            "n_trades":          n_trades,
            "n_trades_closed":   m.get("n_trades_closed", 0),
            "price_volatility":  m.get("price_volatility", 0.0),
            "open_positions":    m.get("open_positions_end", 0),
            "action_buy_frac":   m.get("action_buy_frac", 0.0),
            "action_sell_frac":  m.get("action_sell_frac", 0.0),
            "action_hold_frac":  m.get("action_hold_frac", 0.0),
            "mean_reward":       float(np.mean(list(ep_rewards.values()))),
            "mean_epsilon":      pop_s["mean_epsilon"],
            "mean_td_error":     pop_s["mean_td_error"],
            "mean_grad_norm":    pop_s.get("mean_grad_norm", 0.0),
            "zi_baseline":       zi_baseline,
            "beats_zi":          trade_acc > 0.5,  # bije ZI gdy trade_accuracy > 50%
            "allocative_efficiency": eff,
            "gini":              m.get("gini_pnl", 0.0),
            **pop_meta,
        }
        records.append(record)

        if (episode + 1) % LOG_EVERY == 0:
            recent     = records[-LOG_EVERY:]
            r_tacc  = np.mean([r["trade_accuracy"]    for r in recent])
            r_pnl   = np.mean([r["mean_pnl"]          for r in recent])
            r_td    = np.mean([r["mean_td_error"]      for r in recent])
            elapsed = time.time() - t_start
            log.info(
                f"  [D={diversity_score:.1f} s={seed}] "
                f"ep={episode+1:4d}/{n_episodes} | "
                f"acc={r_tacc:.3f} | "
                f"pnl={r_pnl:.4f} | "
                f"eps={pop_s['mean_epsilon']:.3f} | "
                f"td={r_td:.5f} | "
                f"t={elapsed:.0f}s"
            )

    return records


# ---------------------------------------------------------------------------
# Wykresy
# ---------------------------------------------------------------------------

def plot_learning_curves(
    all_records:  List[dict],
    zi_baselines: Dict[float, float],
    save_path:    Path,
) -> None:
    """Krzywe uczenia Deep SARSA vs ZI baseline dla każdego D."""

    df     = pd.DataFrame(all_records)
    n_cols = len(DIVERSITY_SCORES)

    # Kolory per D
    cmap   = plt.cm.coolwarm
    colors = {d: cmap(i / max(n_cols - 1, 1)) for i, d in enumerate(DIVERSITY_SCORES)}

    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle(
        f"Deep SARSA — krzywe uczenia | N={N_AGENTS} agentów | "
        f"model spekulacyjny",
        fontsize=13, fontweight="bold"
    )

    for col, d in enumerate(DIVERSITY_SCORES):
        df_d  = df[df["diversity_score"] == d]
        zi    = zi_baselines.get(d, 0.5)
        color = colors[d]

        # ── Panel górny: Efficiency ───────────────────────────
        ax = axes[0, col]

        grouped = df_d.groupby("episode")["allocative_efficiency"]
        mean_e  = grouped.mean()
        std_e   = grouped.std().fillna(0)
        smooth  = mean_e.rolling(ROLLING_WINDOW, min_periods=1).mean()
        eps_idx = mean_e.index

        ax.fill_between(
            eps_idx,
            np.clip(smooth - std_e, 0, 1),
            np.clip(smooth + std_e, 0, 1),
            alpha=0.15, color=color
        )
        ax.plot(eps_idx, smooth, color=color, lw=2.5, label="Deep SARSA")
        ax.axhline(zi, color="gray", ls="--", lw=1.5,
                   label=f"ZI ({zi:.3f})", alpha=0.8)

        # Adnotacja czy SARSA bije ZI
        final_eff = float(smooth.iloc[-1])
        delta     = final_eff - zi
        c_delta   = "#2E7D32" if delta > 0 else "#C62828"
        ax.annotate(
            f"Δ = {delta:+.3f}",
            xy=(eps_idx[-1], final_eff),
            fontsize=9, color=c_delta, fontweight="bold",
            ha="right", va="bottom"
        )

        ax.set_title(f"D = {d:.1f}", fontsize=11, color=color, fontweight="bold")
        ax.set_xlabel("Epizod")
        ax.set_ylabel("Allocative Efficiency")
        ax.set_ylim(-0.05, 1.1)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ── Panel dolny: TD error i epsilon ──────────────────
        ax2 = axes[1, col]

        mean_td  = df_d.groupby("episode")["mean_td_error"].mean()
        mean_eps = df_d.groupby("episode")["mean_epsilon"].mean()
        smooth_td= mean_td.rolling(ROLLING_WINDOW, min_periods=1).mean()

        ax2_twin = ax2.twinx()
        ax2.plot(eps_idx, smooth_td,  color=color,   lw=2,   label="TD error")
        ax2_twin.plot(eps_idx, mean_eps, color="gray", lw=1.5, ls=":", label="ε (epsilon)")

        ax2.set_xlabel("Epizod")
        ax2.set_ylabel("Mean TD Error", color=color)
        ax2_twin.set_ylabel("Epsilon", color="gray")
        ax2.set_title(f"Zbieżność (D={d:.1f})", fontsize=9)
        ax2.grid(True, alpha=0.3)

        # Połączona legenda
        lines1, labs1 = ax2.get_legend_handles_labels()
        lines2, labs2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labs1 + labs2, fontsize=7, loc="upper right")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Wykres: {save_path}")


def plot_final_comparison(
    all_records:  List[dict],
    zi_baselines: Dict[float, float],
    save_path:    Path,
) -> None:
    """
    Główny wykres artykułu:
    Deep SARSA vs ZI baseline — końcowe wyniki dla każdego D.
    """
    df    = pd.DataFrame(all_records)
    final = df[df["episode"] >= N_EPISODES - 50]  # ostatnie 50 epizodów

    sarsa_eff  = final.groupby("diversity_score")["allocative_efficiency"].agg(["mean", "std"])
    sarsa_gini = final.groupby("diversity_score")["gini"].agg(["mean", "std"])
    sarsa_trd  = final.groupby("diversity_score")["n_trades"].agg(["mean", "std"])

    d_vals = DIVERSITY_SCORES
    x      = np.arange(len(d_vals))
    w      = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Deep SARSA vs ZI Baseline — końcowe wyniki (ostatnie 50 epizodów)\n"
        "Model spekulacyjny: N agentów z prywatnymi wycenami",
        fontsize=12, fontweight="bold"
    )

    cmap   = plt.cm.coolwarm
    colors = [cmap(i / max(len(d_vals) - 1, 1)) for i in range(len(d_vals))]

    # ── Efficiency ────────────────────────────────────────────
    ax = axes[0]
    sarsa_means = [sarsa_eff.loc[d, "mean"] if d in sarsa_eff.index else 0 for d in d_vals]
    sarsa_stds  = [sarsa_eff.loc[d, "std"]  if d in sarsa_eff.index else 0 for d in d_vals]
    zi_means    = [zi_baselines.get(d, 0)                                   for d in d_vals]

    ax.bar(x - w/2, sarsa_means, w,
           yerr=sarsa_stds, label="Deep SARSA",
           color="#1565C0", alpha=0.85, capsize=4, error_kw={"lw": 1.5})
    ax.bar(x + w/2, zi_means, w,
           label="ZI Baseline",
           color="#616161", alpha=0.75)

    for i, (sm, zm) in enumerate(zip(sarsa_means, zi_means)):
        delta = sm - zm
        c     = "#2E7D32" if delta > 0 else "#C62828"
        ax.text(i - w/2, sm + 0.02, f"{sm:.3f}", ha="center", fontsize=7.5)
        ax.text(i,       max(sm, zm) + 0.06,
                f"Δ={delta:+.2f}", ha="center", fontsize=7.5, color=c, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"D={d}" for d in d_vals], rotation=30)
    ax.set_ylabel("Allocative Efficiency")
    ax.set_title("Efficiency (główna metryka)")
    ax.set_ylim(0, 1.2)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    # ── Gini ──────────────────────────────────────────────────
    ax = axes[1]
    gini_means = [sarsa_gini.loc[d, "mean"] if d in sarsa_gini.index else 0 for d in d_vals]
    gini_stds  = [sarsa_gini.loc[d, "std"]  if d in sarsa_gini.index else 0 for d in d_vals]

    bars = ax.bar(x, gini_means, w * 2,
                  yerr=gini_stds, color=colors, alpha=0.85,
                  capsize=4, error_kw={"lw": 1.5})
    ax.set_xticks(x)
    ax.set_xticklabels([f"D={d}" for d in d_vals], rotation=30)
    ax.set_ylabel("Gini Coefficient")
    ax.set_title("Nierówność wyników agentów")
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.3)
    for bar, m in zip(bars, gini_means):
        ax.text(bar.get_x() + bar.get_width() / 2, m + 0.01,
                f"{m:.3f}", ha="center", fontsize=8)

    # ── Transakcje ────────────────────────────────────────────
    ax = axes[2]
    trd_means = [sarsa_trd.loc[d, "mean"] if d in sarsa_trd.index else 0 for d in d_vals]
    trd_stds  = [sarsa_trd.loc[d, "std"]  if d in sarsa_trd.index else 0 for d in d_vals]

    bars = ax.bar(x, trd_means, w * 2,
                  yerr=trd_stds, color=colors, alpha=0.85,
                  capsize=4, error_kw={"lw": 1.5})
    ax.set_xticks(x)
    ax.set_xticklabels([f"D={d}" for d in d_vals], rotation=30)
    ax.set_ylabel("Średnia liczba transakcji / epizod")
    ax.set_title("Aktywność rynkowa")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, m in zip(bars, trd_means):
        ax.text(bar.get_x() + bar.get_width() / 2, m + 0.05,
                f"{m:.1f}", ha="center", fontsize=8)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Wykres: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Worker — jeden task (D, seed) dla multiprocessing.Pool
# ---------------------------------------------------------------------------

def _train_worker(args: tuple) -> list:
    """
    Uruchamia run_training dla jednej kombinacji (D, seed).
    Musi być funkcją modułu (nie lambda) żeby pickle działał.
    Każdy worker ma własny proces — brak konfliktów między sieciami.
    """
    # numpy-only — brak zależności PyTorch, nic do konfigurowania
    diversity_score, seed, cfg, zi_baseline = args
    return run_training(
        diversity_score = diversity_score,
        n_episodes      = N_EPISODES,
        seed            = seed,
        cfg             = cfg,
        zi_baseline     = zi_baseline,
    )



def main():
    log.info("=" * 65)
    log.info("HTM Benchmark — Deep SARSA (model spekulacyjny)")
    log.info(f"N={N_AGENTS} | D={DIVERSITY_SCORES} | ep={N_EPISODES} | steps/ep=200 | seeds={N_SEEDS}")
    log.info(f"Łączne kroki per agent per D: {N_EPISODES}×200={N_EPISODES*200} | update_every={UPDATE_EVERY}")
    log.info(f"Market: eq±{MARKET.eq_spread} drift={MARKET.drift_enabled}")
    log.info("=" * 65)

    # Upewnij się że katalogi istnieją
    for d in ["logs", "plots", "results", "experiments"]:
        (PROJECT_ROOT / d).mkdir(exist_ok=True)

    cfg = HTMConfig(
        env    = EnvConfig(n_agents=N_AGENTS),
        market = MARKET,
        log    = LogConfig(level="WARNING"),
        sarsa  = SARSA_CFG,
    )

    log.info(cfg.summary())
    log.info(f"episode_steps={cfg.env.episode_steps} | n_actions={cfg.env.n_actions}")

    # 1. ZI baseline — raz przed treningiem
    log.info("\n--- Liczę ZI baseline ---")
    zi_baselines = {}
    for d in DIVERSITY_SCORES:
        zi = compute_zi_baseline(d, cfg, n_episodes=ZI_EPISODES, seed=42)
        zi_baselines[d] = zi
        log.info(f"  ZI | D={d:.1f} | eff={zi:.3f}")

    # 2. Trening — równoległy (multiprocessing.Pool)
    log.info(f"\n--- Start treningu ({N_WORKERS} równoległych procesów) ---")
    t_total = time.time()

    # Stwórz listę zadań: (D, seed) dla wszystkich kombinacji
    tasks = [
        (d, seed, cfg, zi_baselines[d])
        for d in DIVERSITY_SCORES
        for seed in range(N_SEEDS)
    ]
    n_tasks = len(tasks)
    log.info(f"Łącznie zadań: {n_tasks} ({len(DIVERSITY_SCORES)} D × {N_SEEDS} seeds)")

    # Pool: każde zadanie to osobny proces, brak konfliktów między sieciami
    # imap_unordered: zapisuje wyniki na bieżąco — nie traci danych gdy worker pada
    all_records = []
    with Pool(processes=N_WORKERS) as pool:
        for i, task_records in enumerate(pool.imap_unordered(_train_worker, tasks)):
            all_records.extend(task_records)
            d_done = task_records[0]["diversity_score"] if task_records else "?"
            s_done = task_records[0]["seed"]            if task_records else "?"
            log.info(f"  Zakończono: D={d_done} seed={s_done} "
                     f"({i+1}/{n_tasks}) | rekordy: {len(all_records)}")

            # Zapisuj częściowe wyniki co task — bezpieczeństwo
            df_partial = pd.DataFrame(all_records)
            df_partial.to_csv(
                PROJECT_ROOT / "results" / "deep_sarsa_results_partial.csv",
                index=False
            )

    log.info(f"Trening zakończony — {len(all_records)} rekordów")

    # 3. Zapisz CSV
    df       = pd.DataFrame(all_records)
    csv_path = PROJECT_ROOT / "results" / "deep_sarsa_results.csv"
    df.to_csv(csv_path, index=False)
    log.info(f"\nWyniki: {csv_path} ({len(df)} wierszy)")

    # 4. Wykresy
    plot_learning_curves(
        all_records, zi_baselines,
        PROJECT_ROOT / "plots" / "deep_sarsa_learning_curves.png"
    )
    plot_final_comparison(
        all_records, zi_baselines,
        PROJECT_ROOT / "plots" / "deep_sarsa_final_comparison.png"
    )

    # 5. Podsumowanie w konsoli
    log.info("\n" + "=" * 65)
    log.info("PODSUMOWANIE — ostatnie 50 epizodów, uśrednione po seedach")
    log.info(f"{'D':>5} | {'acc (>0.5=good)':>16} | {'pnl':>8} | {'Trades':>7}")
    log.info("-" * 48)

    final = df[df["episode"] >= N_EPISODES - 50]
    for d in DIVERSITY_SCORES:
        sub   = final[final["diversity_score"] == d]
        if sub.empty:
            continue
        pos_frac = sub["pnl_positive_frac"].mean()
        s_pnl    = sub["mean_pnl"].mean()
        tacc     = sub["trade_accuracy"].mean()
        zi       = zi_baselines.get(d, 0.0)
        trades   = sub["n_trades"].mean()
        sign     = "↑" if tacc > 0.5 else "↓"
        log.info(
            f"{d:>5.1f} | {tacc:>7.3f}{sign} | {s_pnl:>8.4f} | {trades:>7.1f}"
        )

    log.info(f"\nCałkowity czas: {time.time()-t_total:.0f}s")
    log.info("Wykresy: plots/deep_sarsa_learning_curves.png")
    log.info("         plots/deep_sarsa_final_comparison.png")
    log.info("Dane:    results/deep_sarsa_results.csv")


if __name__ == "__main__":
    main()