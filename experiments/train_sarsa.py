"""
experiments/train_sarsa.py
──────────────────────────
Pętla treningowa SARSA na benchmarku HTM.

Naprawione względem poprzedniej wersji:
  ✓ Używa parallel_step() zamiast nieistniejących metod AEC
  ✓ Reward pobierany z parallel_step (nie z submit())
  ✓ ZI baseline z run_zi_baseline() (spójne ze środowiskiem)
  ✓ Dynamiczna cena równowagi (MarketDynamics)
  ✓ N agentów jako parametr

Uruchomienie:
    cd htm_project
    python experiments/train_sarsa.py
"""

import sys
import logging
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    HTMConfig, EnvConfig, LogConfig, ExpConfig,
    MarketDynamics, DiversityConfig
)
from envs.double_auction import (
    DoubleAuction, ZeroIntelligenceAgent, run_zi_baseline
)
from agents.sarsa_agent import SARSAMultiAgent, SARSAConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            PROJECT_ROOT / "logs" / "sarsa_training.log", mode="w"
        ),
    ],
)
log = logging.getLogger("htm.train")


# ─────────────────────────────────────────────────────────────
# Parametry eksperymentu
# ─────────────────────────────────────────────────────────────

DIVERSITY_SCORES = [0.0, 0.5, 1.0]
N_EPISODES       = 1000
N_SEEDS          = 3
LOG_EVERY        = 50
ROLLING_WINDOW   = 50

N_BUYERS  = 10
N_SELLERS = 10

MARKET = MarketDynamics.random_eq

()

SARSA_CFG = SARSAConfig(
    alpha=0.1,
    epsilon_start=0.3,
    epsilon_end=0.05,
    epsilon_decay=0.995,
    n_bins=6,
    n_actions=20,
)


# ─────────────────────────────────────────────────────────────
# ZI baseline
# ─────────────────────────────────────────────────────────────

def compute_zi_baselines(
    diversity_scores: List[float],
    n_buyers: int, n_sellers: int,
    market: MarketDynamics,
    n_episodes: int = 200, seed: int = 42,
) -> dict:
    log.info("=" * 50)
    log.info("Liczę ZI baseline (punkt odniesienia)...")

    cfg = HTMConfig(
        env=EnvConfig(n_buyers=n_buyers, n_sellers=n_sellers),
        market=market,
        log=LogConfig(level="WARNING"),
    )

    baselines = {}
    for d in diversity_scores:
        r = run_zi_baseline(cfg, diversity_score=d, n_episodes=n_episodes, seed=seed)
        baselines[d] = r["allocative_efficiency"]["mean"]
        log.info(
            f"ZI | D={d:.1f} | "
            f"eff={r['allocative_efficiency']['mean']:.3f} ± "
            f"{r['allocative_efficiency']['std']:.3f}"
        )
    return baselines


# ─────────────────────────────────────────────────────────────
# Jeden run treningowy
# ─────────────────────────────────────────────────────────────

def run_sarsa_training(
    diversity_score: float,
    n_episodes:      int,
    seed:            int,
    n_buyers:        int,
    n_sellers:       int,
    market:          MarketDynamics,
    sarsa_cfg:       SARSAConfig,
    zi_baseline:     float,
) -> List[dict]:

    cfg = HTMConfig(
        env=EnvConfig(n_buyers=n_buyers, n_sellers=n_sellers),
        market=market,
        log=LogConfig(level="WARNING"),
    )
    env = DoubleAuction(cfg, seed=seed)

    # Inicjalizacja — pobierz gammy z pierwszej populacji
    initial_obs  = env.reset(diversity_score=diversity_score, seed=seed)
    agent_ids    = list(env.population.agents.keys())
    agent_gammas = np.array([
        env.population.agents[aid].belief.gamma
        for aid in agent_ids
    ])

    log.info(
        f"[D={diversity_score:.1f} seed={seed}] "
        f"N={len(agent_ids)} | "
        f"gamma=[{agent_gammas.min():.2f}, {agent_gammas.max():.2f}] | "
        f"max_steps={cfg.env.max_steps}"
    )

    sarsa = SARSAMultiAgent(
        agent_ids=agent_ids,
        agent_gammas=agent_gammas,
        cfg=sarsa_cfg,
        seed=seed,
    )

    episode_records = []
    t_start         = time.time()

    for episode in range(n_episodes):

        # Nowy epizod = nowa populacja (nowe wyceny i ew. nowe eq)
        obs_dict = env.reset(
            diversity_score=diversity_score,
            seed=seed * 10000 + episode,
        )

        # Zaktualizuj gammy agentów SARSA (populacja się zmieniła)
        for aid in agent_ids:
            if aid in sarsa.agents and aid in env.population.agents:
                sarsa.agents[aid].gamma = env.population.agents[aid].belief.gamma

        episode_rewards = {aid: 0.0 for aid in agent_ids}

        # ── Pętla kroków ──────────────────────────────────────
        while not env.done:
            active = env.active_agents
            if not active:
                break

            # Obs tylko dla aktywnych
            current_obs = {
                aid: obs_dict[aid]
                for aid in active
                if aid in obs_dict
            }
            if not current_obs:
                break

            # Akcje epsilon-greedy
            actions = sarsa.act(current_obs, explore=True)

            # Równoległy krok — wszyscy jednocześnie
            next_obs_dict, rewards, dones, infos = env.parallel_step(actions)

            # Aktualizacja Q per agent
            for aid in current_obs:
                if aid not in actions:
                    continue
                sarsa.agents[aid].update(
                    obs=current_obs[aid],
                    action=actions[aid],
                    reward=rewards.get(aid, 0.0),
                    next_obs=next_obs_dict.get(aid, current_obs[aid]),
                    done=dones.get(aid, False),
                )
                episode_rewards[aid] += rewards.get(aid, 0.0)

            obs_dict = next_obs_dict

        # ── Koniec epizodu ────────────────────────────────────
        sarsa.end_episode()

        metrics = env.episode_metrics()
        mean_td = float(np.mean([
            np.mean(a.episode_td_errors) if a.episode_td_errors else 0.0
            for a in sarsa.agents.values()
        ]))

        record = {
            "episode":               episode,
            "diversity_score":       diversity_score,
            "seed":                  seed,
            "algorithm":             "SARSA",
            "n_agents":              n_buyers + n_sellers,
            "eq_price":              metrics.get("equilibrium_price", 0.5),
            "allocative_efficiency": metrics.get("allocative_efficiency", 0.0),
            "gini":                  metrics.get("gini_coefficient", 0.0),
            "n_trades":              metrics.get("n_trades", 0),
            "price_discovery_steps": metrics.get("price_discovery_steps", 999),
            "mean_reward":           float(np.mean(list(episode_rewards.values()))),
            "mean_epsilon":          float(np.mean([
                a.epsilon for a in sarsa.agents.values()
            ])),
            "mean_td_error":         mean_td,
            "zi_baseline":           zi_baseline,
            "beats_zi":              metrics.get("allocative_efficiency", 0.0) > zi_baseline,
        }
        episode_records.append(record)

        if (episode + 1) % LOG_EVERY == 0:
            recent  = episode_records[-LOG_EVERY:]
            avg_eff = np.mean([r["allocative_efficiency"] for r in recent])
            avg_gin = np.mean([r["gini"] for r in recent])
            eps     = sarsa.agents[agent_ids[0]].epsilon
            elapsed = time.time() - t_start
            beats   = "✓" if avg_eff > zi_baseline else "✗"
            log.info(
                f"[D={diversity_score:.1f} s={seed}] "
                f"ep={episode+1}/{n_episodes} | "
                f"eff={avg_eff:.3f} {beats}ZI({zi_baseline:.3f}) | "
                f"gini={avg_gin:.3f} | eps={eps:.3f} | t={elapsed:.0f}s"
            )

    return episode_records


# ─────────────────────────────────────────────────────────────
# Wykresy
# ─────────────────────────────────────────────────────────────

def plot_learning_curves(all_records, zi_baselines, save_path):
    df     = pd.DataFrame(all_records)
    colors = {0.0: "#2196F3", 0.5: "#FF9800", 1.0: "#F44336"}
    n_cols = len(DIVERSITY_SCORES)

    fig = plt.figure(figsize=(5 * n_cols, 8))
    fig.suptitle(
        f"SARSA — krzywe uczenia | N={N_BUYERS+N_SELLERS} agentów",
        fontsize=14, fontweight="bold"
    )
    gs = gridspec.GridSpec(2, n_cols, hspace=0.4, wspace=0.3)

    for col, d in enumerate(DIVERSITY_SCORES):
        df_d = df[df["diversity_score"] == d]
        zi   = zi_baselines.get(d, 0.75)

        ax = fig.add_subplot(gs[0, col])
        grouped = df_d.groupby("episode")["allocative_efficiency"]
        mean_e  = grouped.mean()
        std_e   = grouped.std().fillna(0)
        smooth  = mean_e.rolling(ROLLING_WINDOW, min_periods=1).mean()
        eps_idx = mean_e.index

        ax.fill_between(
            eps_idx,
            (smooth - std_e).clip(0, 1),
            (smooth + std_e).clip(0, 1),
            alpha=0.15, color=colors[d]
        )
        ax.plot(eps_idx, smooth, color=colors[d], lw=2, label=f"SARSA (D={d})")
        ax.axhline(zi, color="gray", ls="--", lw=1.2, label=f"ZI ({zi:.3f})")

        final_e = float(smooth.iloc[-1])
        delta   = final_e - zi
        ax.annotate(
            f"Δ={delta:+.3f}", xy=(eps_idx[-1], final_e),
            fontsize=9,
            color="#4CAF50" if delta > 0 else "#F44336",
            fontweight="bold", ha="right", va="bottom"
        )
        ax.set_title(f"D = {d}", fontsize=11, color=colors[d], fontweight="bold")
        ax.set_xlabel("Epizod")
        ax.set_ylabel("Allocative efficiency")
        ax.set_ylim(0.2, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(gs[1, col])
        mg  = df_d.groupby("episode")["gini"].mean()
        sg  = mg.rolling(ROLLING_WINDOW, min_periods=1).mean()
        ax2.plot(eps_idx, sg, color=colors[d], lw=2)
        ax2.set_xlabel("Epizod")
        ax2.set_ylabel("Gini coefficient")
        ax2.set_ylim(0, 1)
        ax2.set_title(f"Nierówność (D={d})", fontsize=9)
        ax2.grid(True, alpha=0.3)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    log.info(f"Wykres: {save_path}")
    plt.close()


def plot_sarsa_vs_zi(all_records, zi_baselines, save_path):
    df    = pd.DataFrame(all_records)
    final = df[df["episode"] >= N_EPISODES - 100]

    sarsa_eff  = final.groupby("diversity_score")["allocative_efficiency"].agg(["mean","std"])
    sarsa_gini = final.groupby("diversity_score")["gini"].agg(["mean","std"])

    d_vals = sorted(DIVERSITY_SCORES)
    x, w   = np.arange(len(d_vals)), 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("SARSA vs ZI Baseline", fontsize=13, fontweight="bold")

    bars = ax1.bar(
        x - w/2, [sarsa_eff.loc[d,"mean"] for d in d_vals], w,
        yerr=[sarsa_eff.loc[d,"std"] for d in d_vals],
        label="SARSA", color="#2196F3", alpha=0.85, capsize=4
    )
    ax1.bar(
        x + w/2, [zi_baselines.get(d, 0.75) for d in d_vals], w,
        label="ZI Baseline", color="#9E9E9E", alpha=0.85
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"D={d}" for d in d_vals])
    ax1.set_ylabel("Allocative Efficiency")
    ax1.set_ylim(0, 1.1)
    ax1.legend()
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.set_title("Allocative Efficiency")
    for i, d in enumerate(d_vals):
        h = sarsa_eff.loc[d, "mean"]
        ax1.text(i - w/2, h + 0.01, f"{h:.3f}", ha="center", fontsize=8)

    ax2.bar(
        x, [sarsa_gini.loc[d,"mean"] for d in d_vals],
        yerr=[sarsa_gini.loc[d,"std"] for d in d_vals],
        color="#FF9800", alpha=0.85, capsize=4
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"D={d}" for d in d_vals])
    ax2.set_ylabel("Gini Coefficient")
    ax2.set_ylim(0, 1)
    ax2.set_title("Nierówność wyników")
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    log.info(f"Wykres: {save_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("HTM Benchmark — Trening SARSA (parallel step)")
    log.info(f"N={N_BUYERS+N_SELLERS} | D={DIVERSITY_SCORES} | ep={N_EPISODES} | seeds={N_SEEDS}")
    log.info("=" * 60)

    for d in ["logs", "plots", "results"]:
        (PROJECT_ROOT / d).mkdir(exist_ok=True)

    zi_baselines = compute_zi_baselines(
        DIVERSITY_SCORES, N_BUYERS, N_SELLERS, MARKET, n_episodes=200
    )

    all_records = []
    t_total     = time.time()

    for d in DIVERSITY_SCORES:
        for seed in range(N_SEEDS):
            log.info(f"\n{'─'*40}\nSTART: D={d:.1f} | seed={seed}")
            records = run_sarsa_training(
                diversity_score=d, n_episodes=N_EPISODES, seed=seed,
                n_buyers=N_BUYERS, n_sellers=N_SELLERS,
                market=MARKET, sarsa_cfg=SARSA_CFG,
                zi_baseline=zi_baselines[d],
            )
            all_records.extend(records)

    df = pd.DataFrame(all_records)
    df.to_csv(PROJECT_ROOT / "results" / "sarsa_results.csv", index=False)

    plot_learning_curves(all_records, zi_baselines,
                         PROJECT_ROOT / "plots" / "sarsa_learning_curves.png")
    plot_sarsa_vs_zi(all_records, zi_baselines,
                     PROJECT_ROOT / "plots" / "sarsa_vs_zi_comparison.png")

    # Podsumowanie
    log.info("\n" + "=" * 60)
    log.info("PODSUMOWANIE — ostatnie 100 epizodów")
    log.info(f"{'D':>5} | {'SARSA':>8} | {'ZI':>8} | {'Δ':>7} | {'Gini':>6}")
    log.info("-" * 45)
    final = df[df["episode"] >= N_EPISODES - 100]
    for d in DIVERSITY_SCORES:
        sub   = final[final["diversity_score"] == d]
        s_e   = sub["allocative_efficiency"].mean()
        zi    = zi_baselines.get(d, 0.75)
        gini  = sub["gini"].mean()
        delta = s_e - zi
        log.info(
            f"{d:>5.1f} | {s_e:>8.4f} | {zi:>8.4f} | "
            f"{'↑' if delta>0 else '↓'}{abs(delta):>6.4f} | {gini:>6.4f}"
        )
    log.info(f"\nCzas: {time.time()-t_total:.0f}s")


if __name__ == "__main__":
    main()
