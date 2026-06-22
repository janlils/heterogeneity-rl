"""
codes/double_auction.py — Model spekulacyjny HTM
================================================
Kluczowa zmiana względem modelu G&S:

  Model docelowy:
    - Brak stałych ról i brak statycznej prywatnej wyceny.
    - Każdy agent obserwuje per krok prywatny sygnał:
      (V_t - P_t + noise_i) / signal_scale.
    - Heterogeniczność pochodzi z sigma_i: niski szum = lepszy sygnał.
    - BUY/SELL są wykonywane natychmiast przez market makera.
    - PnL wynika wyłącznie z cen wejścia i wyjścia.

Klasy:
  AgentParams      — profil agenta (sigma_i, gamma)
  AgentPopulation  — generuje N agentów z różnym poziomem szumu sygnału
  DoubleAuction    — główne środowisko CT z równoległą egzekucją market maker
  ZeroIntelligence — baseline (losowa agresywność, losowa decyzja buy/sell/pass)
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from codes.config import HTMConfig, EnvConfig, SentimentConfig, DiversityConfig, MarketDynamics

_log = logging.getLogger("htm.auction")


# ===========================================================================
# AgentParams — kompletny profil agenta
# ===========================================================================

@dataclass
class AgentParams:
    """
    Profil agenta w modelu prywatnego sygnału.
    """
    agent_id:           str
    sigma_i:            float = 0.08   # poziom szumu prywatnego sygnału
    gamma:              float = 0.90   # discount factor (indywidualny, wchodzi do TD)
    max_position:       int   = 5      # maks |position|

    # Stan portfela (reset per epizod)
    position:           int   = 0
    entry_price:        float = 0.0
    realized_pnl:       float = 0.0
    n_trades_closed:    int   = 0
    n_trades_won:       int   = 0

    def reset_position(self) -> None:
        """Reset pozycji i P&L na początku epizodu."""
        self.position        = 0
        self.entry_price     = 0.0
        self.realized_pnl    = 0.0
        self.n_trades_closed = 0
        self.n_trades_won    = 0

    def __repr__(self) -> str:
        return (
            f"Agent({self.agent_id}, γ={self.gamma:.2f}, sigma={self.sigma_i:.3f})"
        )


# ===========================================================================
# AgentPopulation — N agentów bez stałych ról
# ===========================================================================

class AgentPopulation:
    """
    Generuje N agentów z różnym poziomem szumu prywatnego sygnału.

    Mechanizm heterogeniczności:
      D=0: wszyscy mają ten sam sigma_i i trader_type=0.5
      D=1: trader_type jest spolaryzowany, więc sigma_i rozciąga się
           od prawie fundamentalistów do mocno zaszumionych chartystów.
    """

    def __init__(
        self,
        n_agents:        int,
        diversity_score: float,
        diversity_cfg:   DiversityConfig,
        sentiment_cfg:   SentimentConfig,
        env_cfg:         EnvConfig,
        eq_price:        float = 0.50,
        seed:            Optional[int] = None,
    ):
        assert 0.0 <= diversity_score <= 1.0
        self.n_agents        = n_agents
        self.diversity_score = diversity_score
        self.diversity_cfg   = diversity_cfg
        self.sentiment_cfg   = sentiment_cfg
        self.env_cfg         = env_cfg
        self.eq_price        = eq_price
        self.rng             = np.random.default_rng(seed)
        self.agents: Dict[str, AgentParams] = {}
        self._generate()

    def _generate(self):
        d   = self.diversity_score
        cfg = self.diversity_cfg
        sc  = self.sentiment_cfg

        for i in range(self.n_agents):
            aid = f"agent_{i}"

            if d < 1e-6:
                trader_type = 0.5
            else:
                d_knots = np.array([0.2, 0.4, 0.6, 0.8, 1.0], dtype=np.float64)
                a_knots = np.array([2.0, 1.0, 0.7, 0.5, 0.3], dtype=np.float64)
                b_knots = np.array([2.0, 1.0, 0.7, 0.5, 0.3], dtype=np.float64)
                a_beta = float(np.interp(d, d_knots, a_knots))
                b_beta = float(np.interp(d, d_knots, b_knots))
                trader_type = float(self.rng.beta(a_beta, b_beta))

            if cfg.sigma_spread and d > 1e-6:
                sigma_i = sc.sigma_fund + trader_type * (sc.sigma_chart - sc.sigma_fund)
            else:
                sigma_i = (sc.sigma_fund + sc.sigma_chart) / 2.0

            gamma       = self._sample_gamma(d, cfg, trader_type)
            max_pos     = self.env_cfg.max_position

            self.agents[aid] = AgentParams(
                agent_id        = aid,
                sigma_i         = float(sigma_i),
                gamma           = gamma,
                max_position    = max_pos,
            )

    def _sample_gamma(self, d: float, cfg: DiversityConfig, trader_type: float) -> float:
        """
        Discount factor γ — horyzont czasowy agenta.

        Interpretacja:
          γ=0.80 → efektywny horyzont ≈ 5 kroków    (short-term)
          γ=0.90 → horyzont ≈ 15–20 kroków           (swing trader)
          γ=0.99 → horyzont ≈ 100 kroków              (pozycyjny)

        Dolna granica 0.80: agent z γ<0.80 ma bardzo krótki horyzont
        przy T=200 — niestabilna nauka, bez sensu ekonomicznego.
        """
        if not cfg.gamma_spread or d < 1e-6:
            return 0.90
        gamma_lo = 0.95 - trader_type * 0.15
        gamma_hi = 0.99 - trader_type * 0.11
        return float(np.clip(self.rng.uniform(gamma_lo, gamma_hi), 0.80, 0.99))

    def max_theoretical_surplus(self) -> float:
        """Legacy metric: model sentimentu nie ma statycznego surplusu."""
        return 0.0

    def diversity_stats(self) -> dict:
        """Statystyki opisowe populacji — do logowania i wykresów."""
        gammas     = [p.gamma     for p in self.agents.values()]
        sigmas     = [p.sigma_i   for p in self.agents.values()]

        return {
            "D":                self.diversity_score,
            "eq_price":         self.eq_price,
            "gamma_mean":       float(np.mean(gammas)),
            "gamma_std":        float(np.std(gammas)),
            "sigma_mean":       float(np.mean(sigmas)),
            "sigma_std":        float(np.std(sigmas)),
        }


# ===========================================================================
# DoubleAuction — główne środowisko
# ===========================================================================

class DoubleAuction:
    """
    Środowisko spekulacyjne HTM.

    Przepływ jednego epizodu:
      reset(D) → [execute_parallel_actions(...) + compute_step_rewards()] × T → episode_metrics()

    Przestrzeń akcji: 0=HOLD, 1=BUY, 2=SELL.

    Obserwacja (8D):
      patrz get_observation(); wektor opisuje prywatny sygnał,
      pozycję, unrealized P&L, czas, gamma, ruch ceny od startu,
      krótki trend oraz poziom szumu agenta.

    Reward:
      realized_pnl_this_step + składnik MTM (bez kar mid-episode).
    """

    def __init__(self, cfg: HTMConfig, seed: Optional[int] = None):
        self.cfg        = cfg
        self.rng        = np.random.default_rng(seed)
        self.population: Optional[AgentPopulation] = None
        self.agent_ids: List[str] = []
        self._agent_idx: Dict[str, int] = {}
        self._positions_arr = np.zeros(0, dtype=np.int32)
        self._entry_prices_arr = np.zeros(0, dtype=np.float32)
        self._realized_pnl_arr = np.zeros(0, dtype=np.float32)
        self._n_trades_closed_arr = np.zeros(0, dtype=np.int32)
        self._n_trades_won_arr = np.zeros(0, dtype=np.int32)
        self._sigma_i_arr = np.zeros(0, dtype=np.float32)
        self._gamma_i_arr = np.zeros(0, dtype=np.float32)
        self._max_position_arr = np.zeros(0, dtype=np.int32)
        self._eq_price  = cfg.market.eq_center
        self._ref_price = cfg.market.eq_center  # aktualizowany po transakcjach
        self._episode_start_price: float = cfg.market.eq_center
        self._eq_price_start:      float = cfg.market.eq_center
        self._step_prices: List[float]   = []

        # Stan epizodu
        self._step:               int              = 0
        self._done:               bool             = False
        self._V_t_prev:           float            = self._eq_price  # poprzedni V_t dla info signal
        self._rewards:            Dict[str, float] = {}
        self._prev_price:         float            = 0.5
        self._actions_log:        List[dict]       = []
        self._price_window:       List[float]      = []
        self._episode_pnl_arr = np.zeros(0, dtype=np.float32)
        self._realized_this_step_arr = np.zeros(0, dtype=np.float32)
        self._price_history:      List[float]      = []
        self._n_fills:            int              = 0
        self._n_position_closes:  int              = 0
        self._terminal_pnl_arr = np.zeros(0, dtype=np.float32)
        self._prev_net_flow:      int              = 0
        self._V_drift:            float            = 0.0
        self._signal_cache:       Dict[str, float] = {}

    def _sync_agent_from_idx(self, idx: int) -> None:
        if self.population is None or idx >= len(self.agent_ids):
            return
        aid = self.agent_ids[idx]
        p = self.population.agents[aid]
        p.position = int(self._positions_arr[idx])
        p.entry_price = float(self._entry_prices_arr[idx])
        p.realized_pnl = float(self._realized_pnl_arr[idx])
        p.n_trades_closed = int(self._n_trades_closed_arr[idx])
        p.n_trades_won = int(self._n_trades_won_arr[idx])

    def _init_runtime_arrays(self) -> None:
        if self.population is None:
            self.agent_ids = []
            self._agent_idx = {}
            self._positions_arr = np.zeros(0, dtype=np.int32)
            self._entry_prices_arr = np.zeros(0, dtype=np.float32)
            self._realized_pnl_arr = np.zeros(0, dtype=np.float32)
            self._n_trades_closed_arr = np.zeros(0, dtype=np.int32)
            self._n_trades_won_arr = np.zeros(0, dtype=np.int32)
            self._sigma_i_arr = np.zeros(0, dtype=np.float32)
            self._gamma_i_arr = np.zeros(0, dtype=np.float32)
            self._max_position_arr = np.zeros(0, dtype=np.int32)
            self._episode_pnl_arr = np.zeros(0, dtype=np.float32)
            self._realized_this_step_arr = np.zeros(0, dtype=np.float32)
            self._terminal_pnl_arr = np.zeros(0, dtype=np.float32)
            return
        self.agent_ids = list(self.population.agents.keys())
        self._agent_idx = {aid: i for i, aid in enumerate(self.agent_ids)}
        agents = [self.population.agents[aid] for aid in self.agent_ids]
        self._positions_arr = np.array([p.position for p in agents], dtype=np.int32)
        self._entry_prices_arr = np.array([p.entry_price for p in agents], dtype=np.float32)
        self._realized_pnl_arr = np.array([p.realized_pnl for p in agents], dtype=np.float32)
        self._n_trades_closed_arr = np.array([p.n_trades_closed for p in agents], dtype=np.int32)
        self._n_trades_won_arr = np.array([p.n_trades_won for p in agents], dtype=np.int32)
        self._sigma_i_arr = np.array([p.sigma_i for p in agents], dtype=np.float32)
        self._gamma_i_arr = np.array([p.gamma for p in agents], dtype=np.float32)
        self._max_position_arr = np.array([p.max_position for p in agents], dtype=np.int32)
        n = len(agents)
        self._episode_pnl_arr = np.zeros(n, dtype=np.float32)
        self._realized_this_step_arr = np.zeros(n, dtype=np.float32)
        self._terminal_pnl_arr = np.zeros(n, dtype=np.float32)

    # -----------------------------------------------------------------------
    # Reset
    # -----------------------------------------------------------------------

    def reset(
        self,
        diversity_score: float = 0.0,
        seed: Optional[int]    = None,
    ) -> Dict[str, np.ndarray]:
        """
        Reset środowiska. Nowa populacja + nowa (ew. losowa) cena równowagi.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Losuj cenę równowagi dla tego epizodu
        md = self.cfg.market
        if md.eq_spread > 1e-6:
            self._eq_price = float(np.clip(
                self.rng.uniform(md.eq_center - md.eq_spread,
                                 md.eq_center + md.eq_spread),
                0.15, 0.85
            ))
        else:
            self._eq_price = md.eq_center

        self._ref_price = self._eq_price  # start: ref = eq

        self.population = AgentPopulation(
            n_agents        = self.cfg.env.n_agents,
            diversity_score = diversity_score,
            diversity_cfg   = self.cfg.diversity,
            sentiment_cfg   = self.cfg.sentiment,
            env_cfg         = self.cfg.env,
            eq_price        = self._eq_price,
            seed            = seed,
        )
        self.agent_ids = list(self.population.agents.keys())
        self._V_t_prev = self._eq_price

        self._step        = 0
        self._done        = False
        self._actions_log = []
        self._price_window= []
        self._price_history = [self._ref_price]
        self._episode_start_price = self._ref_price
        self._eq_price_start = self._eq_price
        self._step_prices = []
        self._n_fills = 0
        self._n_position_closes = 0

        # Reset pozycji agentów
        for p in self.population.agents.values():
            p.reset_position()
        self._init_runtime_arrays()

        self._rewards    = {aid: 0.0 for aid in self.population.agents}
        self._prev_price = self._ref_price
        self._prev_net_flow = 0
        self._refresh_signal_cache()

        _log.debug(
            f"RESET | D={diversity_score:.2f} | eq={self._eq_price:.3f} | "
f"N={self.cfg.env.n_agents}"
        )

        obs_batch = self.get_all_observations()
        return {aid: obs_batch[i] for i, aid in enumerate(self.agent_ids)}

    def reset_episode(self) -> Dict[str, np.ndarray]:
        """
        Reset na nowy epizod: macro dryft V_t + reset portfeli.

        P_t NIE resetuje się (ciągłość rynku między epizodami).
        """
        assert self.population is not None, "Wywołaj reset() przed reset_episode()"
        sc = self.cfg.sentiment

        # 1. V_t macro dryft (większy skok między epizodami)
        self._V_t_prev = self._eq_price
        V_new = float(np.clip(
            self._eq_price + self.rng.normal(0, sc.sigma_macro),
            self.cfg.env.p_min, self.cfg.env.p_max,
        ))
        self._eq_price = V_new
        self._V_drift = 0.0

        # 2. Reset portfeli agentów na nowy epizod.
        for p in self.population.agents.values():
            p.reset_position()
        self._init_runtime_arrays()

        # 3. Reset stanu epizodu (nie: P_t, wagi sieci, epsilon)
        self._step            = 0
        self._done            = False
        self._actions_log     = []
        self._price_window    = []
        self._price_history   = [self._ref_price]
        self._episode_start_price = self._ref_price
        self._eq_price_start      = self._eq_price
        self._step_prices         = []
        self._n_fills         = 0
        self._n_position_closes = 0
        self._rewards             = {aid: 0.0 for aid in self.population.agents}
        self._prev_price          = self._ref_price
        self._prev_net_flow       = 0
        self._refresh_signal_cache()

        obs_batch = self.get_all_observations()
        return {aid: obs_batch[i] for i, aid in enumerate(self.agent_ids)}

    # backward compat alias
    def reset_market_only(self) -> Dict[str, np.ndarray]:
        return self.reset_episode()

    # -----------------------------------------------------------------------
    # Sekwencyjne wykonanie — właściwy interfejs dla RL
    # -----------------------------------------------------------------------

    def execute_single_action(self, agent_id: str, action_idx: int):
        """
        Natychmiastowe wykonanie akcji jednego agenta.

        Używane w sekwencyjnym training loop:
          for aid in random_permutation(agents):
              obs = da.get_observation(aid)   # aktualny stan rynku
              action = agent.act(obs)
              da.execute_single_action(aid, action)
          rewards, dones = da.compute_step_rewards()

        Każdy następny agent widzi rynek ZAKTUALIZOWANY przez poprzedników.
        To jest realistyczne: w prawdziwym rynku oferty przetwarzane są sekwencyjnie.
        """
        if self._done or agent_id not in self.population.agents:
            return None

        params = self.population.agents[agent_id]
        BUY_M  = self.cfg.env.ACTION_BUY_MARKET
        SELL_M = self.cfg.env.ACTION_SELL_MARKET

        self._actions_log.append({
            "step":       self._step,
            "agent_id":   agent_id,
            "action":     action_idx,
            "action_name":self.cfg.env.action_name(action_idx),
            "position":   params.position,
            "ref_price":  self._ref_price,
        })

        if action_idx == BUY_M:
            if params.position >= params.max_position:
                return None
            p_exec = self._execution_price("buy")
            realized = self._execute_fill(agent_id, "buy", p_exec)
            self._move_ref_price("buy")

        elif action_idx == SELL_M:
            if params.position <= -params.max_position:
                return None
            p_exec = self._execution_price("sell")
            realized = self._execute_fill(agent_id, "sell", p_exec)
            self._move_ref_price("sell")
        else:
            return None  # HOLD

        self._record_fill_price(p_exec)
        return {"agent_id": agent_id, "side": self.cfg.env.action_name(action_idx), "price": p_exec}

    def execute_parallel_actions(self, actions: dict) -> None:
        """
        Matching atomowy w obrębie jednego kroku:
        BUY i SELL są dobierane w pary, a cena przesuwa się tylko od
        niezrealizowanej nadwyżki popytu/podaży.

        Zastępuje sekwencyjne wywołania execute_single_action w pętli.
        Używane przez: train_deep_sarsa.py, train_ppo.py, run_zi_baseline.

        Args:
            actions: {agent_id: action_idx} — mapa akcji dla wszystkich agentów
        """
        if self._done:
            return None

        e = self.cfg.env
        P_before = self._ref_price
        buy_agents = []
        sell_agents = []
        for aid, action in actions.items():
            idx = self._agent_idx.get(aid)
            if idx is None:
                continue
            pos = int(self._positions_arr[idx])
            max_pos = int(self._max_position_arr[idx])
            if action == e.ACTION_BUY_MARKET and pos < max_pos:
                buy_agents.append(aid)
            elif action == e.ACTION_SELL_MARKET and pos > -max_pos:
                sell_agents.append(aid)
        excess = len(buy_agents) - len(sell_agents)
        p_exec_buy = float(np.clip(P_before + e.half_spread, e.p_min, e.p_max))
        p_exec_sell = float(np.clip(P_before - e.half_spread, e.p_min, e.p_max))

        for aid in buy_agents:
            self._execute_fill(aid, "buy", p_exec_buy)
            self._record_fill_price(p_exec_buy)

        for aid in sell_agents:
            self._execute_fill(aid, "sell", p_exec_sell)
            self._record_fill_price(p_exec_sell)

        self._prev_net_flow = excess
        if excess != 0:
            self._ref_price = float(np.clip(
                P_before + e.perm_impact * np.sign(excess) * np.sqrt(abs(excess)),
                e.p_min,
                e.p_max,
            ))
        self._price_history.append(self._ref_price)

        # Zaloguj akcje.
        for aid, action in actions.items():
            if aid not in self.population.agents:
                continue
            self._actions_log.append({
                "step":       self._step,
                "agent_id":   aid,
                "action":     action,
                "action_name":e.action_name(action),
                "position":   int(self._positions_arr[self._agent_idx[aid]]),
                "ref_price":  self._ref_price,
            })

        return None

    def compute_step_rewards(
        self,
    ) -> Tuple[Dict[str, float], Dict[str, bool]]:
        """
        Oblicza nagrody i przygotowuje stan na następny krok.

        Kolejność per krok:
          1. Ruch ceny po decyzjach agentów względem aktualnego V_t
          2. Obliczenie nagród
          3. Drift V_t na potrzeby kolejnego kroku
          4. Terminacja jeśli step >= T
        """
        e  = self.cfg.env

        # 1. Ruch ceny po decyzjach agentów.
        # Agenci wybierają akcje przy bieżącym P_t, po czym rynek przechodzi do P_{t+1}.
        # Reward za ten krok powinien więc używać ruchu ceny, który nastąpił PO akcji.
        ref_price_before = self._ref_price
        noise_P = self.rng.normal(0.0, self.cfg.env.sigma_P_noise)
        self._ref_price = float(np.clip(
            self._ref_price
            + self.cfg.env.mv_speed * (self._eq_price - self._ref_price)
            + noise_P,
            self.cfg.env.p_min,
            self.cfg.env.p_max,
        ))
        price_delta = self._ref_price - ref_price_before

        # 2. Oblicz nagrody
        # MTM = mark-to-market: niezrealizowany zysk/strata z otwartej pozycji
        # po ruchu ceny, który wydarzył się po bieżącej decyzji.
        realized_arr = self._realized_this_step_arr.astype(np.float32, copy=False)
        mtm_arr = e.mtm_weight * self._positions_arr.astype(np.float32, copy=False) * float(price_delta)
        reward_arr = realized_arr + mtm_arr
        self._episode_pnl_arr += realized_arr
        rewards: Dict[str, float] = {
            aid: float(reward_arr[i]) for i, aid in enumerate(self.agent_ids)
        }

        # 3. Drift V_t na kolejny krok.
        self._drift_V_t()

        self._step_prices.append(self._ref_price)
        self._prev_price = self._ref_price
        self._realized_this_step_arr.fill(0.0)
        self._refresh_signal_cache()

        self._step += 1
        done = self._step >= e.episode_steps
        if done:
            if e.auto_liquidate_end:
                terminal = self._liquidate_terminal_positions()
                for aid, realized in terminal.items():
                    rewards[aid]          = rewards.get(aid, 0.0) + realized
                    self._episode_pnl_arr[self._agent_idx[aid]] += float(realized)
            self._done = True

        dones = {aid: done for aid in self.population.agents}
        return rewards, dones

    def _execution_price(self, side: str) -> float:
        """Cena egzekucji market makera z kosztem płynności."""
        e = self.cfg.env
        if side == "buy":
            raw = self._ref_price + e.half_spread + e.temp_impact
        elif side == "sell":
            raw = self._ref_price - e.half_spread - e.temp_impact
        else:
            raise ValueError(f"Unknown side: {side}")
        return float(np.clip(raw, e.p_min, e.p_max))

    def _move_ref_price(self, side: str) -> None:
        """Permanentny wpływ wykonanej akcji na cenę referencyjną."""
        e = self.cfg.env
        if side == "buy":
            self._ref_price = float(np.clip(self._ref_price + e.perm_impact, e.p_min, e.p_max))
        elif side == "sell":
            self._ref_price = float(np.clip(self._ref_price - e.perm_impact, e.p_min, e.p_max))
        else:
            raise ValueError(f"Unknown side: {side}")
        self._price_history.append(self._ref_price)
        self._refresh_signal_cache()

    def _record_fill_price(self, price: float) -> None:
        self._n_fills += 1
        self._price_window.append(price)
        if len(self._price_window) > 20:
            self._price_window.pop(0)

    def _execute_fill(self, agent_id: str, side: str, p_exec: float) -> float:
        """
        Rozlicza jedną jednostkę pozycji po cenie wykonania.

        Sentiment nie bierze udziału w PnL. Zysk/strata wynika wyłącznie
        z average cost basis i ceny zamknięcia.
        """
        idx = self._agent_idx[agent_id]
        old = int(self._positions_arr[idx])
        realized = 0.0
        entry_price = float(self._entry_prices_arr[idx])

        if side == "buy":
            if old < 0:
                realized = entry_price - p_exec
                self._register_close(idx, realized)
                if old + 1 == 0:
                    self._entry_prices_arr[idx] = 0.0
            elif old == 0:
                self._entry_prices_arr[idx] = p_exec
            else:
                self._entry_prices_arr[idx] = (entry_price * old + p_exec) / (old + 1)
            self._positions_arr[idx] = old + 1

        elif side == "sell":
            if old > 0:
                realized = p_exec - entry_price
                self._register_close(idx, realized)
                if old - 1 == 0:
                    self._entry_prices_arr[idx] = 0.0
            elif old == 0:
                self._entry_prices_arr[idx] = p_exec
            else:
                self._entry_prices_arr[idx] = (entry_price * abs(old) + p_exec) / (abs(old) + 1)
            self._positions_arr[idx] = old - 1
        else:
            raise ValueError(f"Unknown side: {side}")

        self._realized_pnl_arr[idx] += float(realized)
        self._realized_this_step_arr[idx] += float(realized)
        self._sync_agent_from_idx(idx)
        _log.debug(
            f"  FILL {agent_id} {side} pos:{old}->{int(self._positions_arr[idx])} "
            f"p={p_exec:.4f} r={realized:.4f}"
        )
        return realized

    def _register_close(self, idx: int, realized: float) -> None:
        self._n_trades_closed_arr[idx] += 1
        self._n_position_closes += 1
        if realized > 0:
            self._n_trades_won_arr[idx] += 1

    def _liquidate_terminal_positions(self) -> Dict[str, float]:
        """
        Wymuszone domknięcie pozycji na końcu epizodu.

        WAŻNE: celowo NIE wywołuje _register_close ani _execute_fill.
        Likwidacja terminalna nie wchodzi do n_trades_closed / n_trades_won
        i tym samym nie zniekształca trade_accuracy (która mierzy tylko
        dobrowolne decyzje agenta).

        Rozliczenie po ref_price bez spreadu — spread jest wyzerowany
        w konfiguracji, narzucanie go tylko na likwidację byłoby sztuczne.

        Wynik trafia do self._terminal_pnl (osobna metryka zarządzania ryzykiem).
        """
        e = self.cfg.env
        realized_by_agent: Dict[str, float] = {}

        for aid in self.agent_ids:
            idx = self._agent_idx[aid]
            total = 0.0
            position = int(self._positions_arr[idx])
            entry_price = float(self._entry_prices_arr[idx])

            while position > 0:
                p_exec = float(np.clip(self._ref_price, e.p_min, e.p_max))
                realized = p_exec - entry_price
                total += realized
                self._realized_pnl_arr[idx] += float(realized)
                position -= 1

            while position < 0:
                p_exec = float(np.clip(self._ref_price, e.p_min, e.p_max))
                realized = entry_price - p_exec
                total += realized
                self._realized_pnl_arr[idx] += float(realized)
                position += 1

            self._positions_arr[idx] = 0
            self._entry_prices_arr[idx] = 0.0
            self._terminal_pnl_arr[idx] = float(total)
            self._sync_agent_from_idx(idx)
            if total != 0.0:
                realized_by_agent[aid] = total

        return realized_by_agent

    def _drift_V_t(self) -> None:
        """
        Intra-episode dryft V_t — wywoływany z compute_step_rewards().

        V_t dryfuje niezależnie od P_t. P_t zmienia się tylko przez transakcje.
        Opcjonalne szoki z MarketDynamics (gdy drift_enabled=True).
        """
        sc  = self.cfg.sentiment
        md  = self.cfg.market
        self._V_t_prev = self._eq_price

        self._V_drift = (
            self.cfg.sentiment.drift_persistence * self._V_drift
            + self.rng.normal(0, sc.sigma_intra)
        )
        drift = self._V_drift
        if md.drift_enabled and self.rng.random() < md.shock_probability:
            drift += float(self.rng.choice([-1, 1])) * md.shock_size

        self._eq_price = float(np.clip(
            self._eq_price + drift,
            self.cfg.env.p_min, self.cfg.env.p_max,
        ))

    def _refresh_signal_cache(self) -> None:
        if self.population is None:
            self._signal_cache = {}
            return
        sc = self.cfg.sentiment
        signal_cache: Dict[str, float] = {}
        for i, aid in enumerate(self.agent_ids):
            noise = float(self.rng.normal(0.0, float(self._sigma_i_arr[i])))
            signal_cache[aid] = float(np.clip(
                (self._eq_price - self._ref_price + noise) / sc.signal_scale,
                -1.0,
                1.0,
            ))
        self._signal_cache = signal_cache

    # -----------------------------------------------------------------------
    # Obserwacja
    # -----------------------------------------------------------------------

    def get_observation(self, agent_id: str) -> np.ndarray:
        """
        8D wektor obserwacji — prywatny sygnał fundamentalny + stan portfela.

          [0] signal_i        prywatny sygnał (V_t - P_t + noise) / scale
          [1] position_norm   position / max_position ∈ [-1, +1]
          [2] unrealized_pnl  znormalizowany niezrealizowany P&L ∈ [-2, +2]
          [3] time_remaining  (T - step) / T ∈ [0, 1]
          [4] gamma           discount factor ∈ [0.80, 0.99]
          [5] price_vs_start  (P_t - P_episode_start) / 0.1, clip [-3, +3]
          [6] trend_short     tanh(ΔP_8steps / σ_P) ∈ (-1, +1)
          [7] sigma_norm      sigma_i / sigma_chart ∈ [0, 1]

        public_gap jest liczony lokalnie, ale celowo pominięty z obserwacji:
        to niezaszumiony sygnał identyczny dla wszystkich agentów, który
        osłabiał heterogeniczność opartą na sigma_i.
        """
        idx = self._agent_idx[agent_id]
        T    = self.cfg.env.episode_steps
        MPOS = max(int(self._max_position_arr[idx]), 1)
        position = int(self._positions_arr[idx])
        entry_price = float(self._entry_prices_arr[idx])

        pos_norm = float(np.clip(position / MPOS, -1.0, 1.0))
        time_rem = float(np.clip(1.0 - self._step / max(T, 1), 0.0, 1.0))

        if position != 0 and entry_price > 0:
            unrealized = float(np.clip(
                (self._ref_price - entry_price) * position / 0.05,
                -2.0, 2.0
            ))
        else:
            unrealized = 0.0

        price_vs_start = float(np.clip(
            (self._ref_price - self._episode_start_price) / 0.1,
            -3.0, 3.0
        ))

        sc = self.cfg.sentiment
        if len(self._step_prices) >= 8:
            old_price = self._step_prices[-8]
        else:
            old_price = self._episode_start_price
        trend_short = float(np.tanh(
            (self._ref_price - old_price) / max(sc.sigma_P, 1e-6)
        ))
        signal_i = self._signal_cache.get(agent_id)
        if signal_i is None:
            self._refresh_signal_cache()
            signal_i = self._signal_cache[agent_id]
        sigma_norm = float(np.clip(float(self._sigma_i_arr[idx]) / sc.sigma_chart, 0.0, 1.0))

        return np.array([
            signal_i,                                           # [0]
            pos_norm,                                           # [1]
            unrealized,                                         # [2]
            time_rem,                                           # [3]
            float(self._gamma_i_arr[idx]),                      # [4]
            price_vs_start,                                     # [5]
            trend_short,                                        # [6]
            sigma_norm,                                         # [7]
        ], dtype=np.float32)

    def get_agent_ids(self) -> List[str]:
        return list(self.agent_ids)

    def get_observations(self, agent_ids: Optional[List[str]] = None) -> np.ndarray:
        """
        Batch API obserwacji.

        Zwraca tablicę [n_agents, obs_dim] w kolejności przekazanych agent_ids
        albo w domyślnej, stabilnej kolejności środowiska.
        """
        if self.population is None:
            return np.zeros((0, self.cfg.env.n_obs), dtype=np.float32)
        ids = agent_ids if agent_ids is not None else self.agent_ids
        if not ids:
            return np.zeros((0, self.cfg.env.n_obs), dtype=np.float32)

        idxs = np.array([self._agent_idx[aid] for aid in ids], dtype=np.int32)
        T = max(self.cfg.env.episode_steps, 1)
        sc = self.cfg.sentiment
        max_pos = np.maximum(self._max_position_arr[idxs].astype(np.float32), 1.0)
        positions = self._positions_arr[idxs].astype(np.float32, copy=False)
        entry_prices = self._entry_prices_arr[idxs].astype(np.float32, copy=False)

        pos_norm = np.clip(positions / max_pos, -1.0, 1.0)
        time_rem = np.full(len(idxs), np.clip(1.0 - self._step / T, 0.0, 1.0), dtype=np.float32)
        active = (positions != 0.0) & (entry_prices > 0.0)
        unrealized = np.zeros(len(idxs), dtype=np.float32)
        unrealized[active] = np.clip(
            (self._ref_price - entry_prices[active]) * positions[active] / 0.05,
            -2.0,
            2.0,
        )
        price_vs_start = np.full(
            len(idxs),
            np.clip((self._ref_price - self._episode_start_price) / 0.1, -3.0, 3.0),
            dtype=np.float32,
        )
        old_price = self._step_prices[-8] if len(self._step_prices) >= 8 else self._episode_start_price
        trend_short = np.full(
            len(idxs),
            np.tanh((self._ref_price - old_price) / max(sc.sigma_P, 1e-6)),
            dtype=np.float32,
        )
        if len(self._signal_cache) != len(self.agent_ids):
            self._refresh_signal_cache()
        signal_i = np.array([self._signal_cache[aid] for aid in ids], dtype=np.float32)
        sigma_norm = np.clip(
            self._sigma_i_arr[idxs].astype(np.float32, copy=False) / max(sc.sigma_chart, 1e-6),
            0.0,
            1.0,
        )
        gamma = self._gamma_i_arr[idxs].astype(np.float32, copy=False)

        return np.column_stack([
            signal_i,
            pos_norm,
            unrealized,
            time_rem,
            gamma,
            price_vs_start,
            trend_short,
            sigma_norm,
        ]).astype(np.float32, copy=False)

    def get_all_observations(self) -> np.ndarray:
        return self.get_observations(self.agent_ids)

    def get_global_state(self) -> np.ndarray:
        """
        Zwięzły globalny stan rynku dla centralized critic (MAPPO).

        To nie jest pełny joint state wszystkich agentów. To zestaw agregatów
        rynku i populacji, który ma pomóc criticowi ocenić wartość stanu:
          [0] eq_price
          [1] ref_price
          [2] public_gap = (V - P) / signal_scale
          [3] price_vs_start
          [4] trend_short
          [5] mean_position_norm
          [6] mean_abs_position_norm
          [7] prev_net_flow_norm
          [8] mean_sigma_norm
          [9] std_sigma_norm
        """
        sc = self.cfg.sentiment
        if self.population is None:
            return np.zeros(10, dtype=np.float32)

        if len(self._step_prices) >= 8:
            old_price = self._step_prices[-8]
        else:
            old_price = self._episode_start_price

        max_position = max(self.cfg.env.max_position, 1)
        positions = self._positions_arr.astype(np.float32, copy=False)
        sigmas = self._sigma_i_arr.astype(np.float32, copy=False)

        public_gap = float(np.clip(
            (self._eq_price - self._ref_price) / max(sc.signal_scale, 1e-6),
            -1.0,
            1.0,
        ))
        price_vs_start = float(np.clip(
            (self._ref_price - self._episode_start_price) / 0.1,
            -3.0,
            3.0,
        ))
        trend_short = float(np.tanh(
            (self._ref_price - old_price) / max(sc.sigma_P, 1e-6)
        ))
        prev_net_flow_norm = float(np.clip(
            self._prev_net_flow / max(self.cfg.env.n_agents, 1),
            -1.0,
            1.0,
        ))
        mean_position_norm = float(np.clip(np.mean(positions) / max_position, -1.0, 1.0))
        mean_abs_position_norm = float(np.clip(np.mean(np.abs(positions)) / max_position, 0.0, 1.0))
        mean_sigma_norm = float(np.clip(np.mean(sigmas) / max(sc.sigma_chart, 1e-6), 0.0, 1.0))
        std_sigma_norm = float(np.clip(np.std(sigmas) / max(sc.sigma_chart, 1e-6), 0.0, 1.0))

        return np.array([
            float(self._eq_price),
            float(self._ref_price),
            public_gap,
            price_vs_start,
            trend_short,
            mean_position_norm,
            mean_abs_position_norm,
            prev_net_flow_norm,
            mean_sigma_norm,
            std_sigma_norm,
        ], dtype=np.float32)

    # -----------------------------------------------------------------------
    # Stan i właściwości
    # -----------------------------------------------------------------------

    @property
    def done(self) -> bool:
        return self._done

    @property
    def active_agents(self) -> List[str]:
        """W CT wszyscy agenci są aktywni przez cały epizod."""
        return list(self.agent_ids) if self.population else []

    @property
    def ref_price(self) -> float:
        return self._ref_price

    @property
    def eq_price(self) -> float:
        return self._eq_price

    # -----------------------------------------------------------------------
    # Metryki
    # -----------------------------------------------------------------------

    def episode_metrics(self) -> dict:
        """Metryki epizodu — Continuous Trading."""
        prices  = self._price_history
        # Realized P&L agentów (z zamkniętych pozycji)
        pnls_arr = self._episode_pnl_arr.astype(np.float32, copy=False)
        pnls = {aid: float(pnls_arr[i]) for i, aid in enumerate(self.agent_ids)}
        pnl_vals = pnls_arr.tolist()
        mean_pnl = float(np.mean(pnl_vals)) if pnl_vals else 0.0
        positive_pnl = int(np.sum(pnls_arr > 0))

        # Trade accuracy: ile zamkniętych transakcji było zyskownych
        total_closed = int(np.sum(self._n_trades_closed_arr))
        total_won    = int(np.sum(self._n_trades_won_arr))
        trade_accuracy = float(total_won / max(total_closed, 1))

        # Zmienność cen
        price_volatility = float(np.std(prices)) if len(prices) >= 2 else 0.0
        mean_dev = (
            float(np.mean(np.abs(np.array(prices) - self._eq_price)))
            if prices else float(abs(self._ref_price - self._eq_price))
        )

        # Aktywność handlowa
        e = self.cfg.env
        n_buy  = sum(1 for a in self._actions_log if a.get("action") == e.ACTION_BUY_MARKET)
        n_sell = sum(1 for a in self._actions_log if a.get("action") == e.ACTION_SELL_MARKET)
        n_hold = sum(1 for a in self._actions_log if a.get("action") == e.ACTION_HOLD)
        n_acts = max(len(self._actions_log), 1)

        # Pozycja na koniec epizodu
        final_pos_arr = self._positions_arr.astype(np.int32, copy=False)
        final_pos = {aid: int(final_pos_arr[i]) for i, aid in enumerate(self.agent_ids)}
        mean_pos    = float(np.mean(final_pos_arr)) if len(final_pos_arr) else 0.0
        mean_abs_pos = float(np.mean(np.abs(final_pos_arr))) if len(final_pos_arr) else 0.0
        open_pos    = int(np.sum(final_pos_arr != 0))
        positive_pnl_frac = float(positive_pnl / max(self.cfg.env.n_agents, 1))
        terminal_pnl_vals  = self._terminal_pnl_arr.tolist()
        mean_terminal_pnl  = float(np.mean(terminal_pnl_vals)) if terminal_pnl_vals else 0.0
        terminal_positive  = int(np.sum(self._terminal_pnl_arr > 0))
        sc = self.cfg.sentiment
        value_gap_value = float(np.clip(
            (self._eq_price - self._ref_price) / sc.signal_scale,
            -1.0,
            1.0,
        ))
        value_gaps = [value_gap_value for _ in self.agent_ids]
        midpoint = (sc.sigma_fund + sc.sigma_chart) / 2.0
        pct_chartists = float(
            np.sum(self._sigma_i_arr > midpoint) / max(len(self.agent_ids), 1)
        )
        type_proxy = self._sigma_i_arr.astype(np.float64, copy=False)
        pnl_array = pnls_arr.astype(np.float64, copy=False)
        if (
            len(type_proxy) >= 2
            and float(np.std(type_proxy)) > 1e-12
            and float(np.std(pnl_array)) > 1e-12
        ):
            corr_type_pnl = float(np.corrcoef(type_proxy, pnl_array)[0, 1])
        else:
            corr_type_pnl = 0.0
        price_range = float(max(prices) - min(prices)) if prices else 0.0

        return {
            "mean_pnl":            mean_pnl,
            "mean_realized_pnl":   mean_pnl,
            "pnl_positive_agents": positive_pnl,
            "positive_pnl_frac":   positive_pnl_frac,
            "gini_pnl":            _gini([max(0, v) for v in pnls.values()]),
            "trade_accuracy":      trade_accuracy,   # główna metryka (> 0.5 = lepszy niż losowy)
            "n_trades_closed":     total_closed,
            "n_position_closes":   self._n_position_closes,
            "n_trades":            self._n_fills,
            "price_volatility":    price_volatility,
            "price_range":         price_range,
            "mean_price_deviation":mean_dev,
            "ref_price_final":     self._ref_price,
            "eq_price":            self._eq_price,
            "eq_price_start":      self._eq_price_start,
            "diversity_score":     self.population.diversity_score,
            "n_agents":            self.cfg.env.n_agents,
            "n_steps":             self._step,
            "open_positions_end":  open_pos,
            "mean_position_end":   mean_pos,
            "mean_abs_position":   mean_abs_pos,
            "mean_value_gap":      float(np.mean(value_gaps)) if value_gaps else 0.0,
            "pct_chartists":       pct_chartists,
            "corr_type_pnl":       corr_type_pnl,
            "action_buy_frac":     n_buy  / n_acts,
            "action_sell_frac":    n_sell / n_acts,
            "action_hold_frac":    n_hold / n_acts,
            "mean_terminal_pnl":       mean_terminal_pnl,
            "terminal_positive_frac":  float(terminal_positive / max(self.cfg.env.n_agents, 1)),
            "mean_total_pnl":          mean_pnl,
            # Backward compat alias
            "allocative_efficiency": positive_pnl_frac,
        }

    def agent_metrics(self) -> Dict[str, dict]:
        return {
            aid: {
                "sigma_i":         p.sigma_i,
                "gamma":           p.gamma,
                "position":        int(self._positions_arr[self._agent_idx[aid]]),
                "entry_price":     float(self._entry_prices_arr[self._agent_idx[aid]]),
                "ep_pnl":          float(self._episode_pnl_arr[self._agent_idx[aid]]),
                "realized_pnl":    float(self._realized_pnl_arr[self._agent_idx[aid]]),
                "n_trades_closed": int(self._n_trades_closed_arr[self._agent_idx[aid]]),
                "n_trades_won":    int(self._n_trades_won_arr[self._agent_idx[aid]]),
                "trade_accuracy":  float(
                    self._n_trades_won_arr[self._agent_idx[aid]]
                    / max(int(self._n_trades_closed_arr[self._agent_idx[aid]]), 1)
                ),
            }
            for aid, p in self.population.agents.items()
        }


# ===========================================================================
# Zero Intelligence baseline
# ===========================================================================

class ZeroIntelligenceAgent:
    """
    ZI baseline dla Continuous Trading.

    Strategia: losuj akcję spośród dostępnych (respektuje maskowanie portfolio).
    Odpowiednik G&S ZI: handluje losowo ale w dozwolonym kierunku.
    """

    def __init__(self, params: AgentParams, env_cfg: EnvConfig,
                 seed: Optional[int] = None):
        self.params  = params
        self.env_cfg = env_cfg
        self.rng     = np.random.default_rng(seed)

    def act(self, obs: np.ndarray) -> int:
        # obs[1] = position_norm ∈ [-1,+1]
        pos_norm = float(obs[1])
        can_buy  = pos_norm < 0.99   # nie na max long
        can_sell = pos_norm > -0.99  # nie na max short
        valid = [0]  # HOLD zawsze
        if can_buy:  valid += [1]    # BUY
        if can_sell: valid += [2]    # SELL
        return int(self.rng.choice(valid))


def run_zi_baseline(
    cfg:             HTMConfig,
    diversity_score: float = 0.5,
    n_episodes:      int   = 30,
    seed:            int   = 42,
) -> dict:
    """
    ZI baseline dla CT.

    Populacja tworzona RAZ — reset_episode() między epizodami zamiast reset().
    reset() co epizod tworzył nową AgentPopulation (sampling z rozkładów) — ~10x wolniej.
    Wyceny dryfują naturalnie przez reset_episode(), więc estymata jest stabilna.
    """
    da = DoubleAuction(cfg, seed=seed)
    da.reset(diversity_score=diversity_score, seed=seed)

    # ZI agenci tworzeni raz dla tej populacji
    zi = {
        aid: ZeroIntelligenceAgent(p, cfg.env, seed=seed + i)
        for i, (aid, p) in enumerate(da.population.agents.items())
    }

    all_m: List[dict] = []
    agent_ids = list(da.population.agents.keys())

    for ep in range(n_episodes):
        da.reset_episode()
        T = cfg.env.episode_steps
        for step in range(T):
            if da.done:
                break
            actions = {
                aid: zi[aid].act(da.get_observation(aid))
                for aid in agent_ids
            }
            da.execute_parallel_actions(actions)
            da.compute_step_rewards()
        all_m.append(da.episode_metrics())

    keys = [k for k, v in all_m[0].items() if isinstance(v, (int, float))]
    result = {
        k: {"mean": float(np.mean([m[k] for m in all_m])),
            "std":  float(np.std( [m[k] for m in all_m]))}
        for k in keys
    }
    return result


# ===========================================================================
# Funkcje pomocnicze
# ===========================================================================

def _gini(values: List[float]) -> float:
    arr = np.array(values, dtype=float)
    arr = arr - arr.min() + 1e-9
    n   = len(arr); arr = np.sort(arr)
    idx = np.arange(1, n + 1)
    return float(
        (2 * np.sum(idx * arr) - (n + 1) * arr.sum()) / (n * arr.sum())
    )

# ===========================================================================
# Walidacja
# ===========================================================================

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    from codes.config import HTMConfig, MarketDynamics, LogConfig, ExpConfig

    cfg = HTMConfig(
        market=MarketDynamics.random_eq(),
        log=LogConfig(level="INFO"),
        exp=ExpConfig.quick_test(),
    )
    print(cfg.summary())
    print()

    for d in [0.0, 0.3, 0.6, 1.0]:
        r = run_zi_baseline(cfg, diversity_score=d, n_episodes=25, seed=42)
        acc   = r["trade_accuracy"]
        pos   = r["positive_pnl_frac"]
        gini  = r["gini_pnl"]
        trades= r["n_trades"]
        hfrac = r["action_hold_frac"]
        print(
            f"D={d:.1f} | acc={acc['mean']:.3f}±{acc['std']:.3f} | "
            f"pos_pnl={pos['mean']:.3f} | "
            f"gini={gini['mean']:.3f} | trades={trades['mean']:.1f} | "
            f"hold%={hfrac['mean']*100:.0f}%"
        )
