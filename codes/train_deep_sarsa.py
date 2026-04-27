"""
codes/train_deep_sarsa.py
================================
Pętla treningowa Deep SARSA na benchmarku HTM (model spekulacyjny).

Uruchomienie:
    cd htm_project
    python -m codes.train_deep_sarsa

Co robi:
  1. Liczy wspólny ZI baseline (punkt odniesienia)
  2. Trenuje Deep SARSA przez N_EPISODES epizodów per D
  3. Loguje metryki co LOG_EVERY epizodów
  4. Generuje wykresy krzywych uczenia
  5. Zapisuje wyniki do results/run_*/episodes.csv i agents_sample.csv

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
from codes.double_auction import DoubleAuction
from codes.deep_sarsa import DeepSARSAMultiAgent
from codes.evaluate_policies import evaluate_sarsa, evaluate_zi
from codes.rl_common import build_agent_sample_row, build_env_step_row, build_episode_record
from codes.results_store import (
    EPISODE_FIELDS,
    AGENT_SAMPLE_FIELDS,
    ENV_STEP_FIELDS,
    append_rows,
    prepare_run_dir,
    write_run_config,
)

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
MARKET = MarketDynamics.stable()

# SARSA_ALGO_GAMMA usunięty — każdy agent używa własnej gamma z populacji

# Hiperparametry sieci
SARSA_CFG = DeepSARSAConfig(
    hidden_size   = 64,
    lr            = 1e-3,
    epsilon_start = 0.35,
    epsilon_end   = 0.05,
    epsilon_decay = 0.993,
    grad_clip     = 1.0,
    n_step        = 1,
)


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
        sigma_i = float(meta.get("sigma_i", 0.0))
        trader_type = sigma_i / max(da.cfg.sentiment.sigma_chart, 1e-9)
        rows.append({
            "episode": episode,
            "diversity_score": diversity_score,
            "seed": seed,
            "agent_id": aid,
            "trader_type": trader_type,
            "sigma_i": sigma_i,
            "threshold": float(meta.get("threshold", 0.0)),
            "gamma": float(meta.get("gamma", 0.0)),
            "realized_pnl": float(meta.get("realized_pnl", 0.0)),
            "n_trades_closed": int(meta.get("n_trades_closed", 0)),
            "trade_accuracy_agent": float(meta.get("trade_accuracy", 0.0)),
        })
    return rows


def _stamp_episode_rows(rows: List[dict], run_id: str, phase: str) -> List[dict]:
    stamped: List[dict] = []
    for row in rows:
        item = dict(row)
        item["run_id"] = run_id
        item["phase"] = phase
        stamped.append(item)
    return stamped


def _stamp_sample_rows(rows: List[dict], run_id: str) -> List[dict]:
    stamped: List[dict] = []
    for row in rows:
        item = dict(row)
        item["run_id"] = run_id
        stamped.append(item)
    return stamped


def _sampled_agent_types(da: DoubleAuction) -> dict[str, tuple[str, float]]:
    trader_meta = []
    for aid, agent in da.population.agents.items():
        trader_type = agent.sigma_i / max(da.cfg.sentiment.sigma_chart, 1e-9)
        trader_meta.append((aid, trader_type))
    trader_meta.sort(key=lambda item: item[1])
    fundamentalist_id, fundamentalist_type = trader_meta[0]
    chartist_id, chartist_type = trader_meta[-1]
    mixed_id, mixed_type = min(trader_meta, key=lambda item: abs(item[1] - 0.5))
    return {
        fundamentalist_id: ("fundamentalista", fundamentalist_type),
        mixed_id: ("mieszany", mixed_type),
        chartist_id: ("chartista", chartist_type),
    }


def evaluate_trained_sarsa_same_population(
    da: DoubleAuction,
    sarsa: DeepSARSAMultiAgent,
    diversity_score: float,
    seed: int,
    cfg: HTMConfig,
    n_eval_episodes: int,
    zi_baseline_trade_accuracy: float,
    zi_baseline_positive_pnl_frac: float,
) -> tuple[List[dict], List[dict], List[dict], List[dict]]:
    agent_ids = list(da.population.agents.keys())
    agent_gammas = [da.population.agents[aid].gamma for aid in agent_ids]
    records: List[dict] = []
    sample_rows: List[dict] = []
    agent_eval_rows: List[dict] = []
    env_step_rows: List[dict] = []

    for episode in range(n_eval_episodes):
        da.reset_episode()
        prev_positions = {aid: da.population.agents[aid].position for aid in agent_ids}
        sample_this_episode = seed == 0 and abs(diversity_score - 1.0) < 1e-9 and episode == 0

        while not da.done:
            obs_by_agent: Dict[str, np.ndarray] = {}
            actions_taken: Dict[str, int] = {}
            positions_before = dict(prev_positions)
            eq_price_before = float(da.eq_price)
            ref_price_before = float(da.ref_price)
            public_gap = float(np.clip(
                (eq_price_before - ref_price_before) / max(cfg.sentiment.signal_scale, 1e-9),
                -1.0,
                1.0,
            ))
            realized_before = {
                aid: da.population.agents[aid].realized_pnl
                for aid in agent_ids
            }
            for aid in agent_ids:
                obs = da.get_observation(aid)
                obs_by_agent[aid] = obs
                actions_taken[aid] = sarsa.agents[aid].act(obs, explore=False)

            da.execute_parallel_actions(actions_taken)
            rewards, _ = da.compute_step_rewards()

            if sample_this_episode:
                public_gap_after = float(np.clip(
                    (da.eq_price - da.ref_price) / max(cfg.sentiment.signal_scale, 1e-9),
                    -1.0,
                    1.0,
                ))
                for aid in agent_ids:
                    agent = da.population.agents[aid]
                    obs = obs_by_agent[aid]
                    executed = agent.position != positions_before[aid]
                    trader_type = agent.sigma_i / max(cfg.sentiment.sigma_chart, 1e-9)
                    if trader_type <= 0.33:
                        agent_type = "fundamentalista"
                    elif trader_type >= 0.67:
                        agent_type = "chartista"
                    else:
                        agent_type = "mieszany"
                    realized_pnl_this_step = float(agent.realized_pnl - realized_before[aid])
                    sample_rows.append(build_agent_sample_row(
                        algorithm="DeepSARSA_EVAL_SAME_POPULATION",
                        phase="eval_same_population",
                        diversity_score=diversity_score,
                        seed=seed,
                        episode=episode,
                        step=da._step,
                        agent_id=aid,
                        trader_type=trader_type,
                        agent_type=agent_type,
                        action=actions_taken[aid],
                        action_name=cfg.env.action_name(actions_taken[aid]),
                        executed=executed,
                        obs=obs,
                        public_gap_before=public_gap,
                        eq_price_before=eq_price_before,
                        ref_price_before=ref_price_before,
                        public_gap_after=public_gap_after,
                        eq_price_after=da.eq_price,
                        ref_price_after=da.ref_price,
                        position_before=positions_before[aid],
                        position=agent.position,
                        entry_price_after=agent.entry_price,
                        reward_this_step=float(rewards.get(aid, 0.0)),
                        realized_pnl_this_step=realized_pnl_this_step,
                        realized_pnl_cum=agent.realized_pnl,
                        n_trades_closed=agent.n_trades_closed,
                        sentiment=agent.sentiment,
                        sigma_i=agent.sigma_i,
                        threshold=agent.threshold,
                    ))
                    prev_positions[aid] = agent.position
                mean_signal = float(np.mean([float(obs_by_agent[aid][0]) for aid in agent_ids]))
                std_signal = float(np.std([float(obs_by_agent[aid][0]) for aid in agent_ids]))
                mean_sigma = float(np.mean([float(da.population.agents[aid].sigma_i) for aid in agent_ids]))
                mean_position_before = float(np.mean([positions_before[aid] for aid in agent_ids]))
                mean_position_after = float(np.mean([da.population.agents[aid].position for aid in agent_ids]))
                n_buy = sum(1 for aid in agent_ids if actions_taken[aid] == cfg.env.ACTION_BUY_MARKET)
                n_sell = sum(1 for aid in agent_ids if actions_taken[aid] == cfg.env.ACTION_SELL_MARKET)
                n_hold = sum(1 for aid in agent_ids if actions_taken[aid] == cfg.env.ACTION_HOLD)
                realized_vals = [float(da.population.agents[aid].realized_pnl - realized_before[aid]) for aid in agent_ids]
                reward_vals = [float(rewards.get(aid, 0.0)) for aid in agent_ids]
                env_step_rows.append(build_env_step_row(
                    algorithm="DeepSARSA_EVAL_SAME_POPULATION",
                    phase="eval_same_population",
                    diversity_score=diversity_score,
                    seed=seed,
                    episode=episode,
                    step=da._step,
                    eq_price_before=eq_price_before,
                    ref_price_before=ref_price_before,
                    public_gap_before=public_gap,
                    eq_price_after=da.eq_price,
                    ref_price_after=da.ref_price,
                    public_gap_after=public_gap_after,
                    price_delta_step=da.ref_price - ref_price_before,
                    mean_signal=mean_signal,
                    std_signal=std_signal,
                    mean_sigma=mean_sigma,
                    mean_position_before=mean_position_before,
                    mean_position_after=mean_position_after,
                    n_buy=n_buy,
                    n_sell=n_sell,
                    n_hold=n_hold,
                    net_flow=n_buy - n_sell,
                    mean_reward=float(np.mean(reward_vals)),
                    mean_realized_pnl=float(np.mean(realized_vals)),
                    mean_mtm=float(np.mean([r - x for r, x in zip(reward_vals, realized_vals)])),
                    n_executed=sum(1 for aid in agent_ids if da.population.agents[aid].position != positions_before[aid]),
                    n_trades_closed_cum=sum(da.population.agents[aid].n_trades_closed for aid in agent_ids),
                ))

        metrics = da.episode_metrics()
        for aid, meta in da.agent_metrics().items():
            sigma_i = float(meta.get("sigma_i", 0.0))
            agent_eval_rows.append({
                "diversity_score": diversity_score,
                "seed": seed,
                "episode": episode,
                "agent_id": aid,
                "sigma_i": sigma_i,
                "trader_type": sigma_i / max(cfg.sentiment.sigma_chart, 1e-9),
                "realized_pnl": float(meta.get("realized_pnl", 0.0)),
                "trade_accuracy_agent": float(meta.get("trade_accuracy", 0.0)),
                "n_trades_closed": int(meta.get("n_trades_closed", 0)),
            })
        records.append(build_episode_record(
            episode=episode,
            diversity_score=diversity_score,
            seed=seed,
            algorithm="DeepSARSA_EVAL_SAME_POPULATION",
            cfg=cfg,
            metrics=metrics,
            extra={
                "zi_baseline_trade_accuracy": zi_baseline_trade_accuracy,
                "zi_baseline_positive_pnl_frac": zi_baseline_positive_pnl_frac,
                "zi_baseline": zi_baseline_trade_accuracy,
                "beats_zi": metrics.get("trade_accuracy", 0.0) > zi_baseline_trade_accuracy,
            },
            agent_gammas=agent_gammas,
        ))

    return records, sample_rows, agent_eval_rows, env_step_rows


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
        "pop_mean_sigma":         float(np.mean([_ap[a].sigma_i       for a in agent_ids])),
        "pop_std_sigma":          float(np.std( [_ap[a].sigma_i       for a in agent_ids])),
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
                        G += agent.gamma ** n_step * agent.expected_next_q(next_obs)

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
            short_eval_records, _, _, _ = evaluate_trained_sarsa(
                da=da,
                sarsa=sarsa,
                diversity_score=diversity_score,
                seed=seed + 1000,
                cfg=cfg,
                n_eval_episodes=5,
                zi_baseline_trade_accuracy=zi_baseline_trade_accuracy,
                zi_baseline_positive_pnl_frac=zi_baseline_positive_pnl_frac,
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
    same_population: bool = True,
) -> tuple[List[dict], List[dict], List[dict], List[dict]]:
    """
    Ewaluacja wytrenowanej polityki z końcowym epsilonem i bez update'ów sieci.
    Używa tego samego równoległego protokołu kroku co trening, ale na
    osobnej populacji ewaluacyjnej z seedem przesuniętym względem treningu.
    """
    if same_population:
        return evaluate_trained_sarsa_same_population(
            da=da,
            sarsa=sarsa,
            diversity_score=diversity_score,
            seed=seed,
            cfg=cfg,
            n_eval_episodes=n_eval_episodes,
            zi_baseline_trade_accuracy=zi_baseline_trade_accuracy,
            zi_baseline_positive_pnl_frac=zi_baseline_positive_pnl_frac,
        )

    eval_seed = seed if seed >= 1000 else seed + 1000
    records, sample_rows, env_step_rows = evaluate_sarsa(
        sarsa,
        cfg,
        diversity_score,
        n_eval_episodes,
        eval_seed,
        zi_baseline_trade_accuracy=zi_baseline_trade_accuracy,
        zi_baseline_positive_pnl_frac=zi_baseline_positive_pnl_frac,
    )
    return records, sample_rows, [], env_step_rows


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


def plot_agent_eval_distribution(
    agent_eval_rows: List[dict],
    save_path: Path,
    diversity_scores: List[float],
) -> None:
    """Per-agent rozkład accuracy/PnL oraz zależność od sigma_i."""
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
            "trade_accuracy_agent": "mean",
            "realized_pnl": "mean",
            "n_trades_closed": "mean",
        })
    )

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "SARSA eval same-population — rozkład wyników per agent",
        fontsize=13,
        fontweight="bold",
    )
    cmap = plt.cm.coolwarm
    colors = {d: cmap(i / max(len(diversity_scores) - 1, 1)) for i, d in enumerate(diversity_scores)}

    ax = axes[0, 0]
    for d in diversity_scores:
        sub = grouped[grouped["diversity_score"] == d]
        if sub.empty:
            continue
        ax.scatter(
            sub["sigma_i"], sub["trade_accuracy_agent"],
            s=24, alpha=0.7, color=colors[d], label=f"D={d:.1f}",
        )
    ax.set_title("sigma_i vs trade_accuracy_agent")
    ax.set_xlabel("sigma_i")
    ax.set_ylabel("trade_accuracy_agent")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for d in diversity_scores:
        sub = grouped[grouped["diversity_score"] == d]
        if sub.empty:
            continue
        ax.scatter(
            sub["sigma_i"], sub["realized_pnl"],
            s=24, alpha=0.7, color=colors[d], label=f"D={d:.1f}",
        )
    ax.set_title("sigma_i vs realized_pnl")
    ax.set_xlabel("sigma_i")
    ax.set_ylabel("mean realized_pnl")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    acc_data = [
        grouped.loc[grouped["diversity_score"] == d, "trade_accuracy_agent"].values
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
    ax.set_title("Rozkład trade_accuracy_agent")
    ax.set_ylabel("trade_accuracy_agent")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1, 1]
    pnl_data = [
        grouped.loc[grouped["diversity_score"] == d, "realized_pnl"].values
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
    ax.set_title("Rozkład realized_pnl per agent")
    ax.set_ylabel("mean realized_pnl")
    ax.grid(True, axis="y", alpha=0.3)

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
        sarsa_cfg, log_every, eval_new_population,
    ) = args
    train_records, _learning_curve_records, _agent_diagnostics, sarsa, da = run_training(
        diversity_score = diversity_score,
        n_episodes      = n_episodes,
        seed            = seed,
        cfg             = cfg,
        zi_baseline_trade_accuracy = zi_baseline_trade_accuracy,
        zi_baseline_positive_pnl_frac = zi_baseline_positive_pnl_frac,
        sarsa_cfg       = sarsa_cfg,
        log_every       = log_every,
    )
    eval_same_population_records, sample_rows, agent_eval_rows, env_step_rows = evaluate_trained_sarsa(
        da=da,
        sarsa=sarsa,
        diversity_score=diversity_score,
        seed=seed,
        cfg=cfg,
        n_eval_episodes=n_eval_episodes,
        zi_baseline_trade_accuracy=zi_baseline_trade_accuracy,
        zi_baseline_positive_pnl_frac=zi_baseline_positive_pnl_frac,
        same_population=True,
    )
    eval_new_population_records: List[dict] = []
    if eval_new_population:
        eval_new_population_records, _, _, _ = evaluate_trained_sarsa(
            da=da,
            sarsa=sarsa,
            diversity_score=diversity_score,
            seed=seed + 1000,
            cfg=cfg,
            n_eval_episodes=n_eval_episodes,
            zi_baseline_trade_accuracy=zi_baseline_trade_accuracy,
            zi_baseline_positive_pnl_frac=zi_baseline_positive_pnl_frac,
            same_population=False,
        )
    return train_records, eval_same_population_records, eval_new_population_records, sample_rows, agent_eval_rows, env_step_rows


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
    parser.add_argument(
        "--eval-new-population",
        action="store_true",
        help="Dodatkowo uruchom osobny eval na nowej populacji z seedem przesuniętym o 1000.",
    )
    parser.add_argument("--run-tag", type=str, default="run", help="Krótki tag do nazwy folderu run.")
    parser.add_argument("--run-id", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=str, help=argparse.SUPPRESS)
    return parser.parse_args()


def build_run_settings(args: argparse.Namespace) -> dict:
    if args.quick:
        settings = {
            "run_name": "quick",
            "diversity_scores": [0.0, 0.5, 1.0],
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
                n_step        = 1,
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
    run_id, run_dir = prepare_run_dir(args.run_tag, args.run_id, args.run_dir)
    episodes_csv = run_dir / "episodes.csv"
    agents_sample_csv = run_dir / "agents_sample.csv"
    env_steps_csv = run_dir / "env_steps.csv"
    run_config_path = run_dir / "run_config.json"
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
    if not run_config_path.exists():
        write_run_config(run_config_path, {
            "run_id": run_id,
            "run_tag": args.run_tag,
            "timestamp": run_id.split("_", 1)[1] if run_id.startswith("run_") else run_id,
            "algorithm": "DeepSARSA",
            "diversity_scores": diversity_scores,
            "n_seeds": n_seeds,
            "n_episodes": n_episodes,
            "n_agents": n_agents,
            "market_condition": {
                "eq_center": cfg.market.eq_center,
                "eq_spread": cfg.market.eq_spread,
                "drift_enabled": cfg.market.drift_enabled,
            },
            "eval_new_population": args.eval_new_population,
        })

    log.info(cfg.summary())
    log.info(f"episode_steps={cfg.env.episode_steps} | n_actions={cfg.env.n_actions}")

    # 1. ZI baseline — raz przed treningiem, wspólny dla wszystkich D
    log.info("\n--- Liczę wspólny ZI baseline ---")
    baseline_d = float(diversity_scores[0]) if diversity_scores else 0.0
    zi_records, _, _ = evaluate_zi(cfg, diversity_score=baseline_d, n_episodes=zi_episodes, seed=42)
    append_rows(episodes_csv, EPISODE_FIELDS, _stamp_episode_rows(zi_records, run_id, "zi_baseline"))
    shared_zi_acc = float(np.mean([r["trade_accuracy"] for r in zi_records])) if zi_records else 0.0
    shared_zi_positive = float(np.mean([r["positive_pnl_frac"] for r in zi_records])) if zi_records else 0.0
    zi_pnl = float(np.mean([r["mean_pnl"] for r in zi_records])) if zi_records else 0.0
    zi_term = float(np.mean([r["mean_terminal_pnl"] for r in zi_records])) if zi_records else 0.0
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
            n_episodes, n_eval_episodes, sarsa_cfg, settings["log_every"], args.eval_new_population,
        )
        for d in diversity_scores
        for seed in range(n_seeds)
    ]
    n_tasks = len(tasks)
    log.info(f"Łącznie zadań: {n_tasks} ({len(diversity_scores)} D × {n_seeds} seeds)")

    # Pool: każde zadanie to osobny proces, brak konfliktów między sieciami
    # imap_unordered: zapisuje wyniki na bieżąco — nie traci danych gdy worker pada
    all_records = []
    all_eval_same_population_records = []
    all_eval_new_population_records = []
    all_agent_eval_rows = []
    all_env_step_rows = []
    if n_workers == 1:
        iterator = map(_train_worker, tasks)
        pool = None
    else:
        pool = Pool(processes=n_workers)
        iterator = pool.imap_unordered(_train_worker, tasks)

    try:
        for i, worker_result in enumerate(iterator):
            task_records, eval_same_population_records, eval_new_population_records, sample_rows, agent_eval_rows, env_step_rows = worker_result
            all_records.extend(task_records)
            all_eval_same_population_records.extend(eval_same_population_records)
            all_eval_new_population_records.extend(eval_new_population_records)
            all_agent_eval_rows.extend(agent_eval_rows)
            all_env_step_rows.extend(env_step_rows)
            append_rows(episodes_csv, EPISODE_FIELDS, _stamp_episode_rows(task_records, run_id, "train"))
            append_rows(episodes_csv, EPISODE_FIELDS, _stamp_episode_rows(eval_same_population_records, run_id, "eval_same_population"))
            append_rows(episodes_csv, EPISODE_FIELDS, _stamp_episode_rows(eval_new_population_records, run_id, "eval_new_population"))
            append_rows(agents_sample_csv, AGENT_SAMPLE_FIELDS, _stamp_sample_rows(sample_rows, run_id))
            append_rows(env_steps_csv, ENV_STEP_FIELDS, _stamp_sample_rows(env_step_rows, run_id))
            d_done = task_records[0]["diversity_score"] if task_records else "?"
            s_done = task_records[0]["seed"]            if task_records else "?"
            log.info(f"  Zakończono: D={d_done} seed={s_done} "
                     f"({i+1}/{n_tasks}) | train: {len(all_records)} | eval_same: {len(all_eval_same_population_records)}")
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    log.info(f"Trening zakończony — {len(all_records)} rekordów")

    # 3. Zapisz CSV
    df       = pd.DataFrame(all_records)
    eval_df = pd.DataFrame(all_eval_same_population_records)
    eval_new_df = pd.DataFrame(all_eval_new_population_records)
    total_rows = len(all_records) + len(all_eval_same_population_records) + len(all_eval_new_population_records) + len(zi_records)
    log.info(f"\nWyniki: {episodes_csv} ({total_rows} wierszy łącznie z fazami)")

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
    plot_agent_eval_distribution(
        all_agent_eval_rows,
        PROJECT_ROOT / "plots" / f"{output_stem}_agent_eval_distribution.png",
        diversity_scores=diversity_scores,
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
        log.info("EWALUACJA SARSA — ta sama populacja treningowa, reset_episode(), explore=False")
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

    if not eval_new_df.empty:
        log.info("")
        log.info("EWALUACJA SARSA — nowa populacja, seed+1000, explore=False")
        log.info(
            f"{'D':>5} | {'eval acc':>8} | {'ZI acc':>7} | {'eval pnl':>9} | "
            f"{'term':>8} | {'Trades':>7} | {'Closed':>7}"
        )
        log.info("-" * 76)
        for d in diversity_scores:
            sub = eval_new_df[eval_new_df["diversity_score"] == d]
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
    log.info(f"         plots/{output_stem}_agent_eval_distribution.png")
    log.info(f"Dane:    {episodes_csv.relative_to(PROJECT_ROOT)}")
    log.info(f"Próbka:  {agents_sample_csv.relative_to(PROJECT_ROOT)}")
    log.info(f"Środow.: {env_steps_csv.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
