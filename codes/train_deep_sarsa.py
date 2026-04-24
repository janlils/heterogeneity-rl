"""
codes/train_deep_sarsa.py
================================
Pętla treningowa Deep SARSA na benchmarku HTM (model spekulacyjny).

Uruchomienie:
    cd htm_project
    python -m codes.train_deep_sarsa

Co robi:
  1. Liczy ZI baseline dla każdego D (punkt odniesienia)
  2. Trenuje Deep SARSA przez N_EPISODES epizodów per D
  3. Loguje metryki co LOG_EVERY epizodów
  4. Generuje wykresy krzywych uczenia
  5. Zapisuje wyniki do results/deep_sarsa_results.csv

Parametry do zmiany na górze pliku — nie trzeba grzebać w kodzie.

Szybki test:
    python -m codes.train_deep_sarsa --quick
"""

import sys
import os
import argparse
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional, Sequence
from collections import deque

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
(PROJECT_ROOT / ".matplotlib_cache").mkdir(exist_ok=True)
(PROJECT_ROOT / ".cache").mkdir(exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Upewnij się że Python widzi katalog projektu
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multiprocessing import Pool, cpu_count
from codes.config import HTMConfig, EnvConfig, MarketDynamics, LogConfig, DeepSARSAConfig
from codes.double_auction import DoubleAuction, ZeroIntelligenceAgent, run_zi_baseline
from codes.deep_sarsa import DeepSARSAMultiAgent
from codes.evaluate_policies import evaluate_sarsa

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
log = logging.getLogger("htm.train")


# ---------------------------------------------------------------------------
# Parametry eksperymentu — zmień tu
# ---------------------------------------------------------------------------

DIVERSITY_SCORES = [0.0, 0.3, 0.5, 0.7, 1.0]   # wartości D do przetestowania
N_AGENTS         = 50                             # liczba agentów
N_EPISODES       = 500                            # epizodów CT (każdy = EPISODE_STEPS kroków)
# N_ROUNDS usunięty — CT nie ma rund, epizod = T kroków
N_SEEDS          = 10                             # powtórzeń (dla std)
N_WORKERS        = min(cpu_count(), N_SEEDS * len([0.0, 0.3, 0.5, 0.7, 1.0]))
                                                  # równoległe procesy (auto: liczba corów)
LOG_EVERY        = 25                             # loguj co ile epizodów
ROLLING_WINDOW   = 30                             # okno wygładzania krzywych
ZI_EPISODES      = 30                             # epizodów do policzenia ZI baseline (populacja nie resetuje się co ep)
EPISODE_STEPS    = 500                            # długość epizodu CT

# Warunek rynkowy: stable / random_eq / drifting
MARKET = MarketDynamics.drifting()

# SARSA_ALGO_GAMMA usunięty — każdy agent używa własnej gamma z populacji

# Hiperparametry sieci
SARSA_CFG = DeepSARSAConfig(
    hidden_size   = 64,
    lr            = 1e-3,
    epsilon_start = 0.35,
    epsilon_end   = 0.05,
    epsilon_decay = 0.993,
    grad_clip     = 1.0,
    n_step        = 10,
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
    Zwraca mean trade_accuracy jako empiryczny punkt odniesienia dla SARSA.
    """
    result = run_zi_baseline(cfg, diversity_score=diversity_score,
                             n_episodes=n_episodes, seed=seed)
    return result["trade_accuracy"]["mean"]


def _build_record(
    episode: int,
    diversity_score: float,
    seed: int,
    cfg: HTMConfig,
    metrics: dict,
    algorithm: str,
    zi_baseline_trade_accuracy: float,
    zi_baseline_positive_pnl_frac: float,
    extra: dict = None,
    agent_gammas: Optional[Sequence[float]] = None,
) -> dict:
    """Kanoniczny rekord metryk epizodu dla CSV."""
    trade_acc = metrics.get("trade_accuracy", 0.0)
    positive_pnl_frac = metrics.get(
        "positive_pnl_frac", metrics.get("allocative_efficiency", 0.0)
    )
    gamma_std = float(np.std(agent_gammas)) if agent_gammas is not None and len(agent_gammas) > 0 else 0.0
    record = {
        "episode":           episode,
        "diversity_score":   diversity_score,
        "seed":              seed,
        "algorithm":         algorithm,
        "n_agents":          cfg.env.n_agents,
        "gamma_std":         gamma_std,
        "eq_price":          metrics.get("eq_price", 0.5),
        "eq_price_start":    metrics.get("eq_price_start", 0.5),
        "ref_price_final":   metrics.get("ref_price_final", 0.5),
        "mean_pnl":          metrics.get("mean_pnl", 0.0),
        "pnl_positive_frac": positive_pnl_frac,
        "trade_accuracy":    trade_acc,
        "n_trades":          metrics.get("n_trades", 0),
        "n_trades_closed":   metrics.get("n_trades_closed", 0),
        "n_position_closes": metrics.get("n_position_closes", 0),
        "price_volatility":  metrics.get("price_volatility", 0.0),
        "price_range":       metrics.get("price_range", 0.0),
        "open_positions":    metrics.get("open_positions_end", 0),
        "mean_abs_position": metrics.get("mean_abs_position", 0.0),
        "mean_value_gap":    metrics.get("mean_value_gap", 0.0),
        "v_perceived_std":   metrics.get("v_perceived_std", 0.0),
        "pct_chartists":     metrics.get("pct_chartists", 0.0),
        "corr_type_pnl":     metrics.get("corr_type_pnl", 0.0),
        "action_buy_frac":   metrics.get("action_buy_frac", 0.0),
        "action_sell_frac":  metrics.get("action_sell_frac", 0.0),
        "action_hold_frac":  metrics.get("action_hold_frac", 0.0),
        "zi_baseline_trade_accuracy": zi_baseline_trade_accuracy,
        "zi_baseline_positive_pnl_frac": zi_baseline_positive_pnl_frac,
        "primary_metric":    "trade_accuracy",
        "beats_zi":          trade_acc > zi_baseline_trade_accuracy,
        "positive_pnl_frac": positive_pnl_frac,
        "allocative_efficiency": positive_pnl_frac,  # deprecated alias
        "zi_baseline":       zi_baseline_trade_accuracy,  # deprecated alias
        "gini":              metrics.get("gini_pnl", 0.0),
        "mean_terminal_pnl":       metrics.get("mean_terminal_pnl", 0.0),
        "terminal_positive_frac":  metrics.get("terminal_positive_frac", 0.0),
        "mean_total_pnl":          metrics.get("mean_total_pnl", 0.0),
    }
    if extra:
        record.update(extra)
    return record


def _agent_diagnostic_rows(
    da: DoubleAuction,
    episode: int,
    diversity_score: float,
    seed: int,
) -> List[dict]:
    rows: List[dict] = []
    agent_metrics = da.agent_metrics()
    for aid, meta in agent_metrics.items():
        alpha_i = float(meta.get("alpha_i", 0.0))
        beta_i = float(meta.get("beta_i", 0.0))
        trader_type = alpha_i / max(alpha_i + beta_i, 1e-9)
        rows.append({
            "episode": episode,
            "diversity_score": diversity_score,
            "seed": seed,
            "agent_id": aid,
            "trader_type": trader_type,
            "alpha_i": alpha_i,
            "beta_i": beta_i,
            "threshold": float(meta.get("threshold", 0.0)),
            "gamma": float(meta.get("gamma", 0.0)),
            "V_perceived": float(meta.get("V_perceived", 0.0)),
            "realized_pnl": float(meta.get("realized_pnl", 0.0)),
            "n_trades_closed": int(meta.get("n_trades_closed", 0)),
            "trade_accuracy_agent": float(meta.get("trade_accuracy", 0.0)),
        })
    return rows


# ---------------------------------------------------------------------------
# Jeden run treningowy Deep SARSA
# ---------------------------------------------------------------------------

def run_training(
    diversity_score: float,
    n_episodes:      int,
    seed:            int,
    cfg:             HTMConfig,
    zi_baseline_trade_accuracy: float,
    zi_baseline_positive_pnl_frac: float,
    sarsa_cfg:       DeepSARSAConfig,
    log_every:       int,
 ) -> tuple[List[dict], List[dict], List[dict], DeepSARSAMultiAgent, DoubleAuction]:
    """
    Trenuje Deep SARSA — Continuous Trading.

    Jeden epizod = T kroków z cfg.env.episode_steps
    (wszyscy agenci aktywni przez cały czas).
    Między epizodami: portfele resetowane, wyceny dryfują (pamięć rynku).
    Gamma jest istotna w każdym kroku (done=True dopiero po T krokach).
    """
    da = DoubleAuction(cfg, seed=seed)
    da.reset(diversity_score=diversity_score, seed=seed)

    agent_ids    = list(da.population.agents.keys())
    agent_gammas = np.array(
        [da.population.agents[aid].gamma for aid in agent_ids],
        dtype=np.float64,
    )

    sentiments = [da.population.agents[a].sentiment for a in agent_ids]
    gammas_pop = [da.population.agents[a].gamma for a in agent_ids]
    log.info(
        f"  Populacja | N={len(agent_ids)} | "
        f"sentiment=[{min(sentiments):.2f}, {max(sentiments):.2f}] | "
        f"gamma=[{min(gammas_pop):.2f}, {max(gammas_pop):.2f}] mean={np.mean(gammas_pop):.3f}"
    )

    sarsa = DeepSARSAMultiAgent(
        agent_ids=agent_ids, agent_gammas=agent_gammas,
        n_obs=cfg.env.n_obs, n_actions=cfg.env.n_actions,
        cfg=sarsa_cfg, seed=seed,
    )

    # Parametry populacji — stałe przez cały run, zapisywane do każdego rekordu
    _ap = da.population.agents
    pop_meta = {
        "pop_mean_sentiment":     float(np.mean([_ap[a].sentiment     for a in agent_ids])),
        "pop_std_sentiment":      float(np.std( [_ap[a].sentiment     for a in agent_ids])),
        "pop_mean_gamma":         float(np.mean([_ap[a].gamma         for a in agent_ids])),
        "pop_std_gamma":          float(np.std( [_ap[a].gamma         for a in agent_ids])),
        "pop_mean_alpha":         float(np.mean([_ap[a].alpha_i       for a in agent_ids])),
        "pop_mean_beta":          float(np.mean([_ap[a].beta_i        for a in agent_ids])),
        "pop_mean_risk_aversion": float(np.mean([_ap[a].risk_aversion for a in agent_ids])),
        "pop_mean_threshold":     float(np.mean([_ap[a].threshold     for a in agent_ids])),
        "pop_mean_max_position":  float(np.mean([_ap[a].max_position  for a in agent_ids])),
    }

    records  = []
    learning_curve_records: List[dict] = []
    agent_diagnostics: List[dict] = []
    t_start  = time.time()

    n_step = sarsa_cfg.n_step
    traj_buffers: Dict[str, deque] = {aid: deque() for aid in agent_ids}

    for episode in range(n_episodes):

        # Nowy epizod: reset portfeli, wyceny dryfują, cena zostaje
        da.reset_episode()
        ep_rewards = {aid: 0.0 for aid in agent_ids}
        for aid in agent_ids:
            traj_buffers[aid].clear()

        while not da.done:
            # Parallel execution: wszyscy obserwują ten sam P_t.
            obs_at_action: Dict[str, np.ndarray] = {}
            actions_taken: Dict[str, int]        = {}

            for aid in agent_ids:
                obs = da.get_observation(aid)
                obs_at_action[aid] = obs
                actions_taken[aid] = sarsa.agents[aid].act(obs, explore=True)

            da.execute_parallel_actions(actions_taken)

            # Nagrody na końcu kroku (po wszystkich agentach)
            rewards, dones = da.compute_step_rewards()
            episode_done = any(dones.values())

            for aid in agent_ids:
                r = rewards.get(aid, 0.0)
                ep_rewards[aid] += r

                traj_buffers[aid].append(
                    (obs_at_action[aid], actions_taken[aid], r)
                )

                if len(traj_buffers[aid]) >= n_step:
                    agent = sarsa.agents[aid]
                    obs_0, action_0, _ = traj_buffers[aid][0]

                    G = sum(
                        agent.gamma ** k * traj_buffers[aid][k][2]
                        for k in range(n_step)
                    )
                    if not episode_done:
                        next_obs = da.get_observation(aid)
                        can_b, can_s = agent._mask(next_obs)
                        q_next = agent.net.predict(next_obs)
                        if not can_b:
                            q_next[1] = -np.inf
                        if not can_s:
                            q_next[2] = -np.inf
                        G += agent.gamma ** n_step * float(np.max(q_next))

                    agent.update_with_target(obs_0, action_0, G)
                    traj_buffers[aid].popleft()

        for aid in agent_ids:
            buf = list(traj_buffers[aid])
            agent = sarsa.agents[aid]
            for i in range(len(buf)):
                obs_i, action_i, _ = buf[i]
                G = sum(
                    agent.gamma ** k * buf[i + k][2]
                    for k in range(len(buf) - i)
                )
                agent.update_with_target(obs_i, action_i, G)
            traj_buffers[aid].clear()

        # Metryki epizodu
        m     = da.episode_metrics()
        pop_s = sarsa.population_stats()
        sarsa.end_episode()

        trade_acc      = m.get("trade_accuracy", 0.0)

        record = _build_record(
            episode=episode,
            diversity_score=diversity_score,
            seed=seed,
            cfg=cfg,
            metrics=m,
            algorithm="DeepSARSA_CT",
            zi_baseline_trade_accuracy=zi_baseline_trade_accuracy,
            zi_baseline_positive_pnl_frac=zi_baseline_positive_pnl_frac,
            extra={
            "mean_reward":       float(np.mean(list(ep_rewards.values()))),
            "mean_epsilon":      pop_s["mean_epsilon"],
            "mean_td_error":     pop_s["mean_td_error"],
            "mean_grad_norm":    pop_s.get("mean_grad_norm", 0.0),
            **pop_meta,
            },
            agent_gammas=agent_gammas,
        )
        records.append(record)

        if (episode + 1) % 50 == 0:
            agent_diagnostics.extend(
                _agent_diagnostic_rows(da, episode + 1, diversity_score, seed)
            )
            short_eval_records = evaluate_trained_sarsa(
                da=da,
                sarsa=sarsa,
                diversity_score=diversity_score,
                seed=seed + 1000,
                cfg=cfg,
                n_eval_episodes=5,
                zi_baseline_trade_accuracy=zi_baseline_trade_accuracy,
                zi_baseline_positive_pnl_frac=zi_baseline_positive_pnl_frac,
                log_trajectories=False,
            )
            learning_curve_records.append({
                "episode": episode + 1,
                "diversity_score": diversity_score,
                "seed": seed,
                "train_trade_accuracy": m.get("trade_accuracy", 0.0),
                "eval_trade_accuracy": float(np.mean([r["trade_accuracy"] for r in short_eval_records])),
                "mean_td_error": pop_s["mean_td_error"],
                "gamma_std": float(np.std(agent_gammas)),
            })

        if (episode + 1) % log_every == 0:
            recent     = records[-log_every:]
            r_tacc  = np.mean([r["trade_accuracy"]    for r in recent])
            r_pnl   = np.mean([r["mean_pnl"]          for r in recent])
            r_td    = np.mean([r["mean_td_error"]      for r in recent])
            elapsed = time.time() - t_start
            r_term = np.mean([r["mean_terminal_pnl"] for r in recent])
            log.info(
                f"  [D={diversity_score:.1f} s={seed}] "
                f"ep={episode+1:4d}/{n_episodes} | "
                f"acc={r_tacc:.3f} | "
                f"pnl={r_pnl:.4f} | "
                f"term={r_term:.4f} | "
                f"eps={pop_s['mean_epsilon']:.3f} | "
                f"td={r_td:.5f} | "
                f"t={elapsed:.0f}s"
            )

    return records, learning_curve_records, agent_diagnostics, sarsa, da


def evaluate_trained_sarsa(
    da: DoubleAuction,
    sarsa: DeepSARSAMultiAgent,
    diversity_score: float,
    seed: int,
    cfg: HTMConfig,
    n_eval_episodes: int,
    zi_baseline_trade_accuracy: float,
    zi_baseline_positive_pnl_frac: float,
    log_trajectories: bool = False,
) -> List[dict]:
    """
    Ewaluacja wytrenowanej polityki z końcowym epsilonem i bez update'ów sieci.
    Używa tego samego równoległego protokołu kroku co trening, ale na
    osobnej populacji ewaluacyjnej z seedem przesuniętym względem treningu.
    """
    del da
    eval_seed = seed if seed >= 1000 else seed + 1000
    return evaluate_sarsa(
        sarsa,
        cfg,
        diversity_score,
        n_eval_episodes,
        eval_seed,
        zi_baseline_trade_accuracy=zi_baseline_trade_accuracy,
        zi_baseline_positive_pnl_frac=zi_baseline_positive_pnl_frac,
        log_trajectories=log_trajectories,
    )


# ---------------------------------------------------------------------------
# Wykresy
# ---------------------------------------------------------------------------

def plot_learning_curves(
    all_records:  List[dict],
    zi_baselines: Dict[float, float],
    save_path:    Path,
    diversity_scores: List[float] = DIVERSITY_SCORES,
    rolling_window: int = ROLLING_WINDOW,
    n_agents: int = N_AGENTS,
) -> None:
    """Krzywe uczenia Deep SARSA vs ZI baseline dla każdego D."""

    df     = pd.DataFrame(all_records)
    n_cols = len(diversity_scores)

    # Kolory per D
    cmap   = plt.cm.coolwarm
    colors = {d: cmap(i / max(n_cols - 1, 1)) for i, d in enumerate(diversity_scores)}

    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle(
        f"Deep SARSA — krzywe uczenia | N={n_agents} agentów | "
        f"model spekulacyjny",
        fontsize=13, fontweight="bold"
    )

    for col, d in enumerate(diversity_scores):
        df_d  = df[df["diversity_score"] == d]
        zi    = zi_baselines.get(d)
        color = colors[d]

        # ── Panel górny: trade accuracy ───────────────────────
        ax = axes[0, col]

        metric_col = "trade_accuracy"
        grouped = df_d.groupby("episode")[metric_col]
        mean_e  = grouped.mean()
        std_e   = grouped.std().fillna(0)
        smooth  = mean_e.rolling(rolling_window, min_periods=1).mean()
        eps_idx = mean_e.index

        ax.fill_between(
            eps_idx,
            np.clip(smooth - std_e, 0, 1),
            np.clip(smooth + std_e, 0, 1),
            alpha=0.15, color=color
        )
        ax.plot(eps_idx, smooth, color=color, lw=2.5, label="Deep SARSA")
        if zi is not None:
            ax.axhline(zi, color="gray", ls="--", lw=1.5,
                       label=f"ZI ({zi:.3f})", alpha=0.8)

        # Adnotacja czy SARSA bije ZI
        final_eff = float(smooth.iloc[-1])
        if zi is not None:
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
        ax.set_ylabel("Trade Accuracy")
        ax.set_ylim(0.0, 1.0)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ── Panel dolny: TD error i epsilon ──────────────────
        ax2 = axes[1, col]

        mean_td  = df_d.groupby("episode")["mean_td_error"].mean()
        mean_eps = df_d.groupby("episode")["mean_epsilon"].mean()
        smooth_td= mean_td.rolling(rolling_window, min_periods=1).mean()

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
    diversity_scores: List[float] = DIVERSITY_SCORES,
    n_episodes: int = N_EPISODES,
) -> None:
    """
    Główny wykres artykułu:
    Deep SARSA vs ZI baseline — końcowe wyniki dla każdego D.
    """
    df    = pd.DataFrame(all_records)
    final_window = min(50, max(1, n_episodes // 3))
    final = df[df["episode"] >= n_episodes - final_window]

    sarsa_eff  = final.groupby("diversity_score")["trade_accuracy"].agg(["mean", "std"])
    sarsa_gini = final.groupby("diversity_score")["gini"].agg(["mean", "std"])
    sarsa_trd  = final.groupby("diversity_score")["n_trades"].agg(["mean", "std"])

    d_vals = diversity_scores
    x      = np.arange(len(d_vals))
    w      = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"Deep SARSA vs ZI Baseline — końcowe wyniki (ostatnie {final_window} epizodów)\n"
        "Model spekulacyjny: N agentów z prywatnymi wycenami",
        fontsize=12, fontweight="bold"
    )

    cmap   = plt.cm.coolwarm
    colors = [cmap(i / max(len(d_vals) - 1, 1)) for i in range(len(d_vals))]

    # ── Trade accuracy ────────────────────────────────────────
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
        ax.text(i - w/2, min(sm + 0.02, 0.98), f"{sm:.3f}", ha="center", fontsize=7.5)
        ax.text(i,       min(max(sm, zm) + 0.06, 0.98),
                f"Δ={delta:+.2f}", ha="center", fontsize=7.5, color=c, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"D={d}" for d in d_vals], rotation=30)
    ax.set_ylabel("Trade Accuracy")
    ax.set_title("Metryka główna vs empiryczny ZI")
    ax.set_ylim(0.0, 1.0)
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
    (
        diversity_score, seed, cfg, zi_baseline_trade_accuracy,
        zi_baseline_positive_pnl_frac, n_episodes, n_eval_episodes,
        sarsa_cfg, log_every, log_trajectories,
    ) = args
    train_records, learning_curve_records, agent_diagnostics, sarsa, da = run_training(
        diversity_score = diversity_score,
        n_episodes      = n_episodes,
        seed            = seed,
        cfg             = cfg,
        zi_baseline_trade_accuracy = zi_baseline_trade_accuracy,
        zi_baseline_positive_pnl_frac = zi_baseline_positive_pnl_frac,
        sarsa_cfg       = sarsa_cfg,
        log_every       = log_every,
    )
    eval_records = evaluate_trained_sarsa(
        da=da,
        sarsa=sarsa,
        diversity_score=diversity_score,
        seed=seed + 1000,
        cfg=cfg,
        n_eval_episodes=n_eval_episodes,
        zi_baseline_trade_accuracy=zi_baseline_trade_accuracy,
        zi_baseline_positive_pnl_frac=zi_baseline_positive_pnl_frac,
        log_trajectories=log_trajectories,
    )
    return train_records, learning_curve_records, agent_diagnostics, eval_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trening Deep SARSA dla HTM.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Szybki smoke trening: mniej D, seedów, epizodów i kroków.",
    )
    parser.add_argument("--episodes", type=int, help="Nadpisz liczbę epizodów treningu.")
    parser.add_argument("--steps", type=int, help="Nadpisz liczbę kroków w epizodzie.")
    parser.add_argument("--seeds", type=int, help="Nadpisz liczbę seedów.")
    parser.add_argument("--agents", type=int, help="Nadpisz liczbę agentów.")
    parser.add_argument("--zi-episodes", type=int, help="Nadpisz liczbę epizodów ZI baseline.")
    parser.add_argument("--eval-episodes", type=int, help="Nadpisz liczbę epizodów ewaluacji bez eksploracji.")
    parser.add_argument("--workers", type=int, help="Nadpisz liczbę workerów.")
    return parser.parse_args()


def build_run_settings(args: argparse.Namespace) -> dict:
    if args.quick:
        settings = {
            "run_name": "quick",
            "diversity_scores": [0.0, 0.3, 0.7, 1.0],
            "n_agents": 50,
            "n_episodes": 40,
            "episode_steps": 800,
            "n_seeds": 1,
            "zi_episodes": 10,
            "eval_episodes": 50,
            "log_every": 10,
            "rolling_window": 5,
            "n_workers": 4,
            "sarsa_cfg": DeepSARSAConfig(
                hidden_size   = 32,
                lr            = 1e-3,
                epsilon_start = 0.30,
                epsilon_end   = 0.1,
                epsilon_decay = 0.97,
                grad_clip     = 1.0,
                n_step        = 5,
            ),
        }
    else:
        settings = {
            "run_name": "full",
            "diversity_scores": DIVERSITY_SCORES,
            "n_agents": N_AGENTS,
            "n_episodes": N_EPISODES,
            "episode_steps": EPISODE_STEPS,
            "n_seeds": N_SEEDS,
            "zi_episodes": ZI_EPISODES,
            "eval_episodes": 30,
            "log_every": LOG_EVERY,
            "rolling_window": ROLLING_WINDOW,
            "n_workers": N_WORKERS,
            "sarsa_cfg": SARSA_CFG,
        }

    if args.episodes is not None:
        settings["n_episodes"] = args.episodes
    if args.steps is not None:
        settings["episode_steps"] = args.steps
    if args.seeds is not None:
        settings["n_seeds"] = args.seeds
    if args.agents is not None:
        settings["n_agents"] = args.agents
    if args.zi_episodes is not None:
        settings["zi_episodes"] = args.zi_episodes
    if args.eval_episodes is not None:
        settings["eval_episodes"] = args.eval_episodes
    if args.workers is not None:
        settings["n_workers"] = args.workers

    settings["n_episodes"] = max(1, settings["n_episodes"])
    settings["episode_steps"] = max(1, settings["episode_steps"])
    settings["n_seeds"] = max(1, settings["n_seeds"])
    settings["n_agents"] = max(1, settings["n_agents"])
    settings["zi_episodes"] = max(1, settings["zi_episodes"])
    settings["eval_episodes"] = max(1, settings["eval_episodes"])
    max_workers = settings["n_seeds"] * len(settings["diversity_scores"])
    settings["n_workers"] = max(1, min(settings["n_workers"], max_workers))
    settings["log_every"] = max(1, min(settings["log_every"], settings["n_episodes"]))
    settings["rolling_window"] = max(1, min(settings["rolling_window"], settings["n_episodes"]))
    return settings



def main():
    args = parse_args()
    settings = build_run_settings(args)
    trajectories_csv = PROJECT_ROOT / "results" / "trajectories_eval.csv"
    if trajectories_csv.exists():
        trajectories_csv.unlink()
    diversity_scores = settings["diversity_scores"]
    n_agents = settings["n_agents"]
    n_episodes = settings["n_episodes"]
    episode_steps = settings["episode_steps"]
    n_seeds = settings["n_seeds"]
    zi_episodes = settings["zi_episodes"]
    n_eval_episodes = settings["eval_episodes"]
    n_workers = settings["n_workers"]
    sarsa_cfg = settings["sarsa_cfg"]
    run_name = settings["run_name"]
    output_stem = "deep_sarsa" if run_name == "full" else f"deep_sarsa_{run_name}"

    log.info("=" * 65)
    log.info("HTM Benchmark — Deep SARSA (model spekulacyjny)")
    log.info(f"Tryb: {run_name}")
    log.info(f"N={n_agents} | D={diversity_scores} | ep={n_episodes} | steps/ep={episode_steps} | seeds={n_seeds}")
    log.info(f"Łączne kroki per agent per D: {n_episodes}×{episode_steps}={n_episodes*episode_steps}")
    log.info(
        f"SARSA epsilon: start={sarsa_cfg.epsilon_start:.3f} | "
        f"end={sarsa_cfg.epsilon_end:.3f} | decay={sarsa_cfg.epsilon_decay:.3f}"
    )
    log.info(f"Market: eq±{MARKET.eq_spread} drift={MARKET.drift_enabled}")
    log.info("=" * 65)

    # Upewnij się że katalogi istnieją
    for d in ["logs", "plots", "results", "experiments"]:
        (PROJECT_ROOT / d).mkdir(exist_ok=True)

    cfg = HTMConfig(
        env    = EnvConfig(n_agents=n_agents, episode_steps=episode_steps),
        market = MARKET,
        log    = LogConfig(level="WARNING"),
        sarsa  = sarsa_cfg,
    )

    log.info(cfg.summary())
    log.info(f"episode_steps={cfg.env.episode_steps} | n_actions={cfg.env.n_actions}")

    # 1. ZI baseline — raz przed treningiem, wspólny dla wszystkich D
    log.info("\n--- Liczę wspólny ZI baseline ---")
    baseline_d = float(diversity_scores[0]) if diversity_scores else 0.0
    zi_result = run_zi_baseline(cfg, diversity_score=baseline_d,
                                n_episodes=zi_episodes, seed=42)
    shared_zi_acc = zi_result.get("trade_accuracy", {}).get("mean", 0.0)
    shared_zi_positive = zi_result.get("positive_pnl_frac", {}).get("mean", 0.0)
    zi_pnl = zi_result.get("mean_pnl", {}).get("mean", 0.0)
    zi_term = zi_result.get("mean_terminal_pnl", {}).get("mean", 0.0)
    zi_baselines = {d: shared_zi_acc for d in diversity_scores}
    zi_positive_baselines = {d: shared_zi_positive for d in diversity_scores}
    log.info(
        f"  ZI | D_ref={baseline_d:.1f} | eff={shared_zi_positive:.3f} | "
        f"acc={shared_zi_acc:.3f} | pnl={zi_pnl:.4f} | term={zi_term:.4f}"
    )

    # 2. Trening — równoległy (multiprocessing.Pool)
    log.info(f"\n--- Start treningu ({n_workers} równoległych procesów) ---")
    t_total = time.time()

    # Stwórz listę zadań: (D, seed) dla wszystkich kombinacji
    tasks = [
        (
            d, seed, cfg, zi_baselines[d], zi_positive_baselines[d],
            n_episodes, n_eval_episodes, sarsa_cfg, settings["log_every"],
            seed == 0 and abs(d - 1.0) < 1e-9,
        )
        for d in diversity_scores
        for seed in range(n_seeds)
    ]
    n_tasks = len(tasks)
    log.info(f"Łącznie zadań: {n_tasks} ({len(diversity_scores)} D × {n_seeds} seeds)")

    # Pool: każde zadanie to osobny proces, brak konfliktów między sieciami
    # imap_unordered: zapisuje wyniki na bieżąco — nie traci danych gdy worker pada
    all_records = []
    all_learning_curve_records = []
    all_agent_diagnostics = []
    all_eval_records = []
    if n_workers == 1:
        iterator = map(_train_worker, tasks)
        pool = None
    else:
        pool = Pool(processes=n_workers)
        iterator = pool.imap_unordered(_train_worker, tasks)

    try:
        for i, worker_result in enumerate(iterator):
            task_records, learning_curve_records, agent_diagnostics, eval_records = worker_result
            all_records.extend(task_records)
            all_learning_curve_records.extend(learning_curve_records)
            all_agent_diagnostics.extend(agent_diagnostics)
            all_eval_records.extend(eval_records)
            d_done = task_records[0]["diversity_score"] if task_records else "?"
            s_done = task_records[0]["seed"]            if task_records else "?"
            log.info(f"  Zakończono: D={d_done} seed={s_done} "
                     f"({i+1}/{n_tasks}) | train: {len(all_records)} | eval: {len(all_eval_records)}")

            # Zapisuj częściowe wyniki co task — bezpieczeństwo
            df_partial = pd.DataFrame(all_records)
            df_partial.to_csv(
                PROJECT_ROOT / "results" / f"{output_stem}_results_partial.csv",
                index=False
            )
            pd.DataFrame(all_eval_records).to_csv(
                PROJECT_ROOT / "results" / f"{output_stem}_eval_results_partial.csv",
                index=False
            )
            pd.DataFrame(all_learning_curve_records).to_csv(
                PROJECT_ROOT / "results" / (
                    "deep_sarsa_learning_curve.csv"
                    if run_name == "full"
                    else f"{output_stem}_learning_curve.csv"
                ),
                index=False
            )
            pd.DataFrame(all_agent_diagnostics).to_csv(
                PROJECT_ROOT / "results" / "deep_sarsa_agent_diagnostics.csv",
                index=False
            )
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    log.info(f"Trening zakończony — {len(all_records)} rekordów")

    # 3. Zapisz CSV
    df       = pd.DataFrame(all_records)
    csv_path = PROJECT_ROOT / "results" / f"{output_stem}_results.csv"
    df.to_csv(csv_path, index=False)
    log.info(f"\nWyniki: {csv_path} ({len(df)} wierszy)")
    eval_df = pd.DataFrame(all_eval_records)
    eval_csv_path = PROJECT_ROOT / "results" / f"{output_stem}_eval_results.csv"
    eval_df.to_csv(eval_csv_path, index=False)
    log.info(f"Ewaluacja: {eval_csv_path} ({len(eval_df)} wierszy)")
    learning_curve_csv = PROJECT_ROOT / "results" / (
        "deep_sarsa_learning_curve.csv"
        if run_name == "full"
        else f"{output_stem}_learning_curve.csv"
    )
    pd.DataFrame(all_learning_curve_records).to_csv(learning_curve_csv, index=False)
    agent_diagnostics_csv = PROJECT_ROOT / "results" / "deep_sarsa_agent_diagnostics.csv"
    pd.DataFrame(all_agent_diagnostics).to_csv(agent_diagnostics_csv, index=False)
    log.info(f"Learning curve: {learning_curve_csv} ({len(all_learning_curve_records)} wierszy)")

    # 4. Wykresy
    plot_learning_curves(
        all_records, zi_baselines,
        PROJECT_ROOT / "plots" / f"{output_stem}_learning_curves.png",
        diversity_scores=diversity_scores,
        rolling_window=settings["rolling_window"],
        n_agents=n_agents,
    )
    plot_final_comparison(
        all_records, zi_baselines,
        PROJECT_ROOT / "plots" / f"{output_stem}_final_comparison.png",
        diversity_scores=diversity_scores,
        n_episodes=n_episodes,
    )

    # 5. Podsumowanie w konsoli
    final_window = min(50, max(1, n_episodes // 3))
    log.info("\n" + "=" * 82)
    log.info(f"PODSUMOWANIE — ostatnie {final_window} epizodów, uśrednione po seedach")
    log.info(
        f"{'D':>5} | {'acc':>7} | {'ZI acc':>7} | {'pnl_tot':>9} | "
        f"{'term':>8} | {'Trades':>7} | {'Closed':>7}"
    )
    log.info("-" * 82)

    final = df[df["episode"] >= n_episodes - final_window]
    for d in diversity_scores:
        sub   = final[final["diversity_score"] == d]
        if sub.empty:
            continue
        s_pnl    = sub["mean_total_pnl"].mean()
        term_pnl = sub["mean_terminal_pnl"].mean()
        tacc     = sub["trade_accuracy"].mean()
        zi       = zi_baselines.get(d, 0.0)
        trades   = sub["n_trades"].mean()
        closed   = sub["n_trades_closed"].mean()
        sign     = "↑" if tacc > zi else "↓"
        log.info(
            f"{d:>5.1f} | {tacc:>7.3f}{sign} | {zi:>7.3f} | "
            f"{s_pnl:>9.4f} | {term_pnl:>8.4f} | {trades:>7.1f} | {closed:>7.1f}"
        )

    if not eval_df.empty:
        log.info("")
        log.info("EWALUACJA SARSA — epsilon policy, explore=True")
        log.info(
            f"{'D':>5} | {'eval acc':>8} | {'ZI acc':>7} | {'eval pnl':>9} | "
            f"{'term':>8} | {'Trades':>7} | {'Closed':>7}"
        )
        log.info("-" * 76)
        for d in diversity_scores:
            sub = eval_df[eval_df["diversity_score"] == d]
            if sub.empty:
                continue
            eval_acc = sub["trade_accuracy"].mean()
            eval_pnl = sub["mean_total_pnl"].mean()
            eval_term = sub["mean_terminal_pnl"].mean()
            eval_trades = sub["n_trades"].mean()
            eval_closed = sub["n_trades_closed"].mean()
            zi = zi_baselines.get(d, 0.0)
            sign = "↑" if eval_acc > zi else "↓"
            log.info(
                f"{d:>5.1f} | {eval_acc:>7.3f}{sign} | {zi:>7.3f} | "
                f"{eval_pnl:>9.4f} | {eval_term:>8.4f} | "
                f"{eval_trades:>7.1f} | {eval_closed:>7.1f}"
            )

    log.info(f"\nCałkowity czas: {time.time()-t_total:.0f}s")
    log.info(f"Wykresy: plots/{output_stem}_learning_curves.png")
    log.info(f"         plots/{output_stem}_final_comparison.png")
    log.info(f"Dane:    results/{output_stem}_results.csv")
    log.info(f"Eval:    results/{output_stem}_eval_results.csv")
    log.info(f"Curve:   {learning_curve_csv.relative_to(PROJECT_ROOT)}")
    log.info(f"Agents:  {agent_diagnostics_csv.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
