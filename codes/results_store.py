from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

from codes.config import RESULTS_DIR


EPISODE_FIELDS = [
    "run_id",
    "phase",
    "algorithm",
    "diversity_score",
    "seed",
    "episode",
    "n_agents",
    "gamma_std",
    "eq_price",
    "eq_price_start",
    "ref_price_final",
    "trade_accuracy",
    "mean_pnl",
    "mean_total_pnl",
    "mean_terminal_pnl",
    "positive_pnl_frac",
    "terminal_positive_frac",
    "n_trades",
    "n_trades_closed",
    "n_position_closes",
    "price_volatility",
    "price_range",
    "mean_abs_position",
    "mean_value_gap",
    "v_perceived_std",
    "pct_chartists",
    "corr_type_pnl",
    "action_buy_frac",
    "action_sell_frac",
    "action_hold_frac",
    "gini",
    "primary_metric",
    "same_action_frac",
    "effective_N",
    "zi_baseline_trade_accuracy",
    "zi_baseline_positive_pnl_frac",
    "zi_baseline",
    "beats_zi",
    "mean_reward",
    "mean_epsilon",
    "mean_td_error",
    "mean_grad_norm",
    "pop_mean_sentiment",
    "pop_std_sentiment",
    "pop_mean_gamma",
    "pop_std_gamma",
    "pop_mean_alpha",
    "pop_mean_beta",
    "pop_mean_risk_aversion",
    "pop_mean_threshold",
    "pop_mean_max_position",
    "policy_loss",
    "value_loss",
    "entropy",
    "approx_kl",
    "clip_fraction",
    "mean_advantage",
    "mean_return",
    "eval_mode",
    "eval_trade_accuracy",
    "eval_mean_total_pnl",
    "eval_n_trades",
    "eval_mean_terminal_pnl",
    "pnl_positive_frac",
    "open_positions",
    "allocative_efficiency",
]

AGENT_SAMPLE_FIELDS = [
    "run_id",
    "algorithm",
    "phase",
    "diversity_score",
    "seed",
    "episode",
    "step",
    "agent_id",
    "trader_type",
    "agent_type",
    "action",
    "action_name",
    "executed",
    "sentiment",
    "value_gap",
    "position",
    "realized_pnl_this_step",
    "prev_net_flow_norm",
    "alpha_i",
    "beta_i",
    "threshold",
]


def make_run_id(run_tag: str, timestamp: Optional[datetime] = None) -> str:
    ts = timestamp or datetime.now()
    return f"run_{ts:%Y%m%d_%H%M%S}_{run_tag}"


def prepare_run_dir(
    run_tag: str,
    run_id: Optional[str] = None,
    run_dir: Optional[str] = None,
) -> tuple[str, Path]:
    if run_dir is not None:
        path = Path(run_dir)
        resolved_run_id = run_id or path.name
    else:
        resolved_run_id = run_id or make_run_id(run_tag)
        path = RESULTS_DIR / resolved_run_id
    path.mkdir(parents=True, exist_ok=True)
    _ensure_csv(path / "episodes.csv", EPISODE_FIELDS)
    _ensure_csv(path / "agents_sample.csv", AGENT_SAMPLE_FIELDS)
    return resolved_run_id, path


def _ensure_csv(path: Path, fieldnames: list[str]) -> None:
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def append_rows(path: Path, fieldnames: list[str], rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    _ensure_csv(path, fieldnames)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def write_run_config(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True, sort_keys=True)


def latest_run_dir() -> Optional[Path]:
    candidates = [
        path for path in RESULTS_DIR.glob("run_*")
        if path.is_dir() and (path / "episodes.csv").exists()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)

