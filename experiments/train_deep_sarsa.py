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
from envs.double_auction import DoubleAuction, ZeroIntelligenceAgent
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
UPDATE_EVERY = 2   # aktualizuj siec co ile krokow (speedup ~2x)
log = logging.getLogger("htm.train")


# ---------------------------------------------------------------------------
# Parametry eksperymentu — zmień tu
# ---------------------------------------------------------------------------

DIVERSITY_SCORES = [0.0, 0.3, 0.5, 0.7, 1.0]   # wartości D do przetestowania
N_AGENTS         = 20                             # liczba agentów
N_EPISODES       = 500                            # epizodów (każdy = N_ROUNDS rund)
N_ROUNDS         = 3                              # rund handlowych per epizod
                                                  # aktualizacji per agent ≈ N_EP × N_ROUNDS × 2
N_SEEDS          = 3                              # powtórzeń (dla std)
N_WORKERS        = min(cpu_count(), N_SEEDS * len([0.0, 0.3, 0.5, 0.7, 1.0]))
                                                  # równoległe procesy (auto: liczba corów)
LOG_EVERY        = 25                             # loguj co ile epizodów
ROLLING_WINDOW   = 30                             # okno wygładzania krzywych
ZI_EPISODES      = 100                            # epizodów do policzenia ZI baseline

# Warunek rynkowy: stable / random_eq / drifting
MARKET = MarketDynamics.stable()

# Hiperparametry sieci
SARSA_CFG = DeepSARSAConfig(
    hidden_size   = 32,
    lr            = 5e-3,
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
    Uruchamia ZI baseline i zwraca średnią efficiency.
    To jest nasz punkt odniesienia — Deep SARSA musi go pobić.
    """
    rng = np.random.default_rng(seed)
    da  = DoubleAuction(cfg, seed=seed)
    effs = []

    for ep in range(n_episodes):
        ep_seed = int(rng.integers(0, 100_000))
        da.reset(diversity_score=diversity_score, seed=ep_seed)

        zi = {
            aid: ZeroIntelligenceAgent(p, cfg.env, seed=ep_seed + i)
            for i, (aid, p) in enumerate(da.population.agents.items())
        }

        step = 0
        max_steps = cfg.env.max_steps * 3
        while not da.done and step < max_steps:
            active = da.active_agents
            if not active:
                break
            aid   = active[step % len(active)]
            obs   = da.get_observation(aid)
            act_i, signal = zi[aid].act(obs, da.ref_price)
            p     = da.population.agents[aid]
            if act_i != cfg.env.ACTION_PASS and signal != "none":
                if signal == "buy":
                    gap   = max(0.0, p.valuation - da.ref_price)
                    price = {cfg.env.ACTION_MARKET:      p.valuation,
                             cfg.env.ACTION_LIMIT_TIGHT: da.ref_price + gap * 0.67,
                             cfg.env.ACTION_LIMIT_MED:   da.ref_price + gap * 0.33,
                             cfg.env.ACTION_LIMIT_FAR:   da.ref_price}.get(act_i, p.valuation)
                    price = min(price, p.max_affordable_bid())
                    da.submit(aid, float(np.clip(price, 0.001, 0.999)), "bid")
                else:
                    gap   = max(0.0, da.ref_price - p.valuation)
                    price = {cfg.env.ACTION_MARKET:      p.valuation,
                             cfg.env.ACTION_LIMIT_TIGHT: da.ref_price - gap * 0.67,
                             cfg.env.ACTION_LIMIT_MED:   da.ref_price - gap * 0.33,
                             cfg.env.ACTION_LIMIT_FAR:   da.ref_price}.get(act_i, p.valuation)
                    da.submit(aid, float(np.clip(price, 0.001, 0.999)), "ask")
            else:
                da._step += 1
                if da._step >= cfg.env.max_steps:
                    da._done = True
            step += 1

        effs.append(da.episode_metrics()["allocative_efficiency"])

    return float(np.mean(effs))


# ---------------------------------------------------------------------------
# Jeden run treningowy Deep SARSA
# ---------------------------------------------------------------------------

def run_training(
    diversity_score: float,
    n_episodes:      int,
    n_rounds:        int,
    seed:            int,
    cfg:             HTMConfig,
    zi_baseline:     float,
) -> List[dict]:
    """
    Trenuje Deep SARSA — Opcja B + multi-round.

    Opcja B (stała populacja):
      Jedna populacja 20 agentów na cały trening tego (D, seed).
      Każda sieć uczy się strategii JEDNEGO konkretnego agenta
      z niezmiennymi parametrami (valuation, gamma, wealth, threshold).
      Sieć NIE uśrednia po różnych agentach — to byłby de facto model globalny.

    Multi-round (Opcja 1):
      Każdy "epizod" składa się z N_ROUNDS rund handlowych.
      Po każdej rundzie rynek jest resetowany (order book, kto handlował),
      ale agenci zostają. To daje ~N_ROUNDS × 2 aktualizacji per agent
      per epizod zamiast 1-2 w poprzedniej wersji.

    Łącznie aktualizacji per agent:
      N_EPISODES × N_ROUNDS × ~2 = 500 × 5 × 2 = ~5000
      vs. poprzednio: 500 × 1-2 = ~500-1000
    """
    da = DoubleAuction(cfg, seed=seed)

    # ── Opcja B: stwórz populację RAZ przed pętlą epizodów ──────────────
    # Ta sama populacja przez cały trening — każda sieć = jeden konkretny agent
    da.reset(diversity_score=diversity_score, seed=seed)
    agent_ids    = list(da.population.agents.keys())
    agent_gammas = np.array([
        da.population.agents[aid].gamma for aid in agent_ids
    ])

    log.info(
        f"  Populacja | N={len(agent_ids)} | "
        f"val=[{min(da.population.agents[a].valuation for a in agent_ids):.2f}, "
        f"{max(da.population.agents[a].valuation for a in agent_ids):.2f}] | "
        f"gamma=[{agent_gammas.min():.2f}, {agent_gammas.max():.2f}]"
    )

    # Sieci tworzone RAZ — powiązane z konkretną populacją
    sarsa = DeepSARSAMultiAgent(
        agent_ids    = agent_ids,
        agent_gammas = agent_gammas,   # indywidualne gamma, niezmienne
        n_obs        = cfg.env.n_obs,
        n_actions    = cfg.env.n_actions,
        cfg          = SARSA_CFG,
        seed         = seed,
    )

    records = []
    t_start = time.time()

    for episode in range(n_episodes):

        # ── Multi-round: N_ROUNDS rund w jednym epizodzie ───────────────
        # Każda runda = nowa sesja handlowa, ci sami agenci
        round_effs   = []
        round_trades = []
        round_rewards = {aid: 0.0 for aid in agent_ids}
        round_actions_log = []

        for round_idx in range(n_rounds):

            # Reset tylko rynku — agenci niezmienieni (Opcja B)
            obs_dict = da.reset_market_only()

            # ── Pętla kroków w rundzie ───────────────────────
            step = 0
            while not da.done:
                active = da.active_agents
                if not active:
                    break

                cur_obs = {
                    aid: obs_dict[aid]
                    for aid in active
                    if aid in obs_dict
                }
                if not cur_obs:
                    break

                # Deep SARSA: akcje epsilon-greedy
                actions = sarsa.act(cur_obs, explore=True)

                # Równoległy krok
                next_obs, rewards, dones, infos = da.parallel_step(actions)

                # Aktualizacja sieci per agent
                # UPDATE_EVERY: aktualizuj co N kroków — szybsze (~2x)
                # Reward zawsze akumulowany, update tylko co UPDATE_EVERY
                for aid in cur_obs:
                    if aid not in actions:
                        continue
                    reward = rewards.get(aid, 0.0)
                    round_rewards[aid] += reward

                    if step % UPDATE_EVERY == 0:
                        sarsa.agents[aid].update(
                            obs      = cur_obs[aid],
                            action   = actions[aid],
                            reward   = reward,
                            next_obs = next_obs.get(aid, cur_obs[aid]),
                            done     = dones.get(aid, False),
                        )

                obs_dict = next_obs

            # Metryki tej rundy
            m = da.episode_metrics()
            round_effs.append(m.get("allocative_efficiency", 0.0))
            round_trades.append(m.get("n_trades", 0))
            round_actions_log.append(m)

        # ── Koniec epizodu (po wszystkich rundach) ───────────
        # WAŻNE: zbierz statystyki PRZED end_episode()
        # end_episode() resetuje episode_td_errors → stats byłyby zerowe
        pop_s = sarsa.population_stats()

        # Decay epsilon raz per epizod (nie per rundę)
        sarsa.end_episode()

        # Metryki epizodu = średnia po rundach
        avg_eff    = float(np.mean(round_effs))
        avg_trades = float(np.mean(round_trades))
        avg_eff_last = round_effs[-1]   # ostatnia runda (po "rozgrzewce")

        # Proporcje akcji z ostatniej rundy
        last_m = round_actions_log[-1]

        record = {
            "episode":               episode,
            "diversity_score":       diversity_score,
            "seed":                  seed,
            "algorithm":             "DeepSARSA",
            "n_agents":              N_AGENTS,
            "n_rounds":              n_rounds,
            "eq_price":              last_m.get("eq_price", 0.5),
            "allocative_efficiency": avg_eff,         # średnia po rundach
            "eff_last_round":        avg_eff_last,    # ostatnia runda
            "gini":                  last_m.get("gini_coefficient", 0.0),
            "n_trades":              avg_trades,
            "action_buy_frac":       last_m.get("action_buy_frac", 0.0),
            "action_sell_frac":      last_m.get("action_sell_frac", 0.0),
            "action_pass_frac":      last_m.get("action_pass_frac", 0.0),
            "mean_reward":           float(np.mean(list(round_rewards.values()))),
            "mean_epsilon":          pop_s["mean_epsilon"],
            "mean_td_error":         pop_s["mean_td_error"],
            "mean_grad_norm":        pop_s.get("mean_grad_norm", 0.0),
            "zi_baseline":           zi_baseline,
            "beats_zi":              avg_eff > zi_baseline,
        }
        records.append(record)

        # Logowanie co LOG_EVERY epizodów
        if (episode + 1) % LOG_EVERY == 0:
            recent  = records[-LOG_EVERY:]
            r_eff   = np.mean([r["allocative_efficiency"] for r in recent])
            r_gin   = np.mean([r["gini"]                  for r in recent])
            eps     = pop_s["mean_epsilon"]
            td_e    = pop_s["mean_td_error"]
            elapsed = time.time() - t_start
            beats   = "✓" if r_eff > zi_baseline else "✗"

            log.info(
                f"  [D={diversity_score:.1f} s={seed}] "
                f"ep={episode+1:4d}/{n_episodes} | "
                f"eff={r_eff:.3f} {beats}ZI({zi_baseline:.3f}) | "
                f"gini={r_gin:.3f} | "
                f"eps={eps:.3f} | td={td_e:.4f} | "
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
    diversity_score, seed, cfg, zi_baseline = args
    return run_training(
        diversity_score = diversity_score,
        n_episodes      = N_EPISODES,
        n_rounds        = N_ROUNDS,
        seed            = seed,
        cfg             = cfg,
        zi_baseline     = zi_baseline,
    )



def main():
    log.info("=" * 65)
    log.info("HTM Benchmark — Deep SARSA (model spekulacyjny)")
    log.info(f"N={N_AGENTS} | D={DIVERSITY_SCORES} | ep={N_EPISODES} | rounds/ep={N_ROUNDS} | seeds={N_SEEDS}")
    log.info(f"Łączne rundy per agent per D: {N_EPISODES}×{N_ROUNDS}={N_EPISODES*N_ROUNDS} | ~aktualizacji: {N_EPISODES*N_ROUNDS*2}")
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
    log.info(f"max_steps={cfg.env.max_steps} | n_actions={cfg.env.n_actions}")

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
    log.info(f"{'D':>5} | {'SARSA':>8} | {'ZI':>8} | {'Δ':>7} | {'Gini':>6} | {'Trades':>7}")
    log.info("-" * 55)

    final = df[df["episode"] >= N_EPISODES - 50]
    for d in DIVERSITY_SCORES:
        sub   = final[final["diversity_score"] == d]
        if sub.empty:
            continue
        s_eff  = sub["allocative_efficiency"].mean()
        zi     = zi_baselines.get(d, 0.0)
        gini   = sub["gini"].mean()
        trades = sub["n_trades"].mean()
        delta  = s_eff - zi
        sign   = "↑" if delta > 0 else "↓"
        log.info(
            f"{d:>5.1f} | {s_eff:>8.4f} | {zi:>8.4f} | "
            f"{sign}{abs(delta):>6.4f} | {gini:>6.4f} | {trades:>7.1f}"
        )

    log.info(f"\nCałkowity czas: {time.time()-t_total:.0f}s")
    log.info("Wykresy: plots/deep_sarsa_learning_curves.png")
    log.info("         plots/deep_sarsa_final_comparison.png")
    log.info("Dane:    results/deep_sarsa_results.csv")


if __name__ == "__main__":
    main()
