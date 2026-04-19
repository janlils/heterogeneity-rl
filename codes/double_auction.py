"""
codes/double_auction.py — Model spekulacyjny HTM
================================================
Kluczowa zmiana względem modelu G&S:

  Model docelowy:
    - Brak stałych ról. Każdy agent ma sentiment ∈ [-1, +1].
    - Jeśli sentiment > threshold → sygnał KUP / long.
    - Jeśli sentiment < -threshold → sygnał SPRZEDAJ / short.
    - BUY/SELL są wykonywane natychmiast przez market makera.
    - PnL wynika wyłącznie z cen wejścia i wyjścia, nie z sentimentu.

Klasy:
  AgentParams      — profil agenta (sentiment, threshold, gamma, wealth)
  AgentPopulation  — generuje N agentów bez stałych ról
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
    Profil agenta w modelu sentiment.

    Główna zmiana vs poprzedniej wersji:
      - expected_price / long_run_fair_price zastąpione przez sentiment ∈ [-1, +1]
      - belief: BeliefState usunięty
      - Dodane: alpha_i, beta_i, news_sensitivity, V_perceived
    """
    agent_id:           str
    sentiment:          float = 0.0    # ∈ [-1, +1]: -1 = silnie niedźwiedzi, +1 = byczo
    alpha_i:            float = 0.08   # momentum: waga sygnału cenowego per krok
    beta_i:             float = 0.06   # mean reversion: powrót do neutralu (0.0)
    news_sensitivity:   float = 0.12   # waga sygnału informacyjnego z V_t
    V_perceived:        float = 0.5    # prywatna wycena fundamentalna agenta

    threshold:          float = 0.20   # min |sentiment| żeby mieć sygnał handlowy
    gamma:              float = 0.90   # discount factor (indywidualny, wchodzi do TD)
    wealth:             float = 1.00   # majątek → max_position
    risk_aversion:      float = 1.00   # λ: kara za otwartą pozycję w reward
    max_position:       int   = 5      # maks |position|

    # Stan portfela (reset per epizod)
    position:           int   = 0
    entry_price:        float = 0.0
    realized_pnl:       float = 0.0
    n_trades_closed:    int   = 0
    n_trades_won:       int   = 0

    def trade_signal(self) -> str:
        """Sygnał handlowy na podstawie sentimentu i progu."""
        if self.sentiment > self.threshold:
            return "buy"
        elif self.sentiment < -self.threshold:
            return "sell"
        return "none"

    def reset_position(self) -> None:
        """Reset pozycji i P&L na początku epizodu."""
        self.position        = 0
        self.entry_price     = 0.0
        self.realized_pnl    = 0.0
        self.n_trades_closed = 0
        self.n_trades_won    = 0

    def __repr__(self) -> str:
        return (
            f"Agent({self.agent_id}, sent={self.sentiment:.3f}, "
            f"V_perc={self.V_perceived:.3f}, "
            f"thr={self.threshold:.3f}, γ={self.gamma:.2f}, "
            f"α={self.alpha_i:.3f}, β={self.beta_i:.3f})"
        )


# ===========================================================================
# AgentPopulation — N agentów bez stałych ról
# ===========================================================================

class AgentPopulation:
    """
    Generuje N agentów z prywatną wyceną fundamentalną aktywa.

    Mechanizm heterogeniczności:
      D=0: wszyscy mają valuation = eq_price → brak transakcji (no-trade theorem)
      D=0.5: wyceny w [eq-0.2, eq+0.2] → umiarkowane różnice, sporo handlu
      D=1: wyceny w [0.05, 0.95] → duże różnice zdań, intensywny handel

    Dlaczego to lepsze niż model G&S:
      W G&S efficiency ≈ 1 zawsze bo wyceny są z definicji po właściwych
      stronach ceny równowagi. Tu agent MOŻE mieć sygnał po złej stronie
      (val > price ale cena potem rośnie) — to jest realne ryzyko rynkowe.
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

            sentiment_0, alpha_i, beta_i, news_sens = self._sample_sentiment_params(d, cfg, sc)
            gamma       = self._sample_gamma(d, cfg)
            threshold   = self._sample_threshold(d, cfg)
            wealth      = self._sample_wealth(d, cfg)
            max_pos     = self._wealth_to_max_position(wealth, d)
            risk_av     = self._sample_risk_aversion(d, cfg)
            V_perceived_init = float(np.clip(
                self.rng.normal(self.eq_price, 0.015 * d + 0.005),
                self.env_cfg.p_min, self.env_cfg.p_max,
            ))

            self.agents[aid] = AgentParams(
                agent_id        = aid,
                sentiment       = sentiment_0,
                alpha_i         = alpha_i,
                beta_i          = beta_i,
                news_sensitivity= news_sens,
                V_perceived     = V_perceived_init,
                threshold       = threshold,
                gamma           = gamma,
                wealth          = wealth,
                risk_aversion   = risk_av,
                max_position    = max_pos,
            )

    def _sample_sentiment_params(
        self, d: float, cfg: DiversityConfig, sc: SentimentConfig
    ) -> tuple:
        """
        Losuje (sentiment_0, alpha_i, beta_i, news_sensitivity) dla agenta.

        D=0: wszyscy identyczni (centra parametrów, sentiment=0).
        D=1: pełny rozrzut → heterogeniczne typy agentów.
        """
        if not cfg.sentiment_spread or d < 1e-6:
            sentiment_0 = 0.0
        else:
            sentiment_0 = float(np.clip(self.rng.normal(0, 0.35 * d), -1.0, 1.0))

        if not cfg.behavioral_spread or d < 1e-6:
            alpha_i = sc.alpha_center
            beta_i  = sc.beta_center
            news    = sc.news_sensitivity_center
        else:
            half_a = sc.alpha_spread / 2.0
            half_b = sc.beta_spread / 2.0
            half_n = sc.news_sensitivity_spread / 2.0
            alpha_i = float(np.clip(
                self.rng.uniform(sc.alpha_center - half_a * d,
                                 sc.alpha_center + half_a * d),
                0.01, 0.40
            ))
            beta_i = float(np.clip(
                self.rng.uniform(sc.beta_center - half_b * d,
                                 sc.beta_center + half_b * d),
                0.01, 0.25
            ))
            news = float(np.clip(
                self.rng.uniform(sc.news_sensitivity_center - half_n * d,
                                 sc.news_sensitivity_center + half_n * d),
                0.01, 0.50
            ))
        return sentiment_0, alpha_i, beta_i, news

    def _sample_gamma(self, d: float, cfg: DiversityConfig) -> float:
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
        low  = 0.90 - 0.10 * d   # D=0.5: 0.85,  D=1.0: 0.80
        high = 0.90 + 0.09 * d   # D=0.5: 0.945, D=1.0: 0.99
        return float(np.clip(self.rng.uniform(low, high), 0.80, 0.99))

    def _sample_threshold(self, d: float, cfg: DiversityConfig) -> float:
        """
        Minimalny |sentiment| żeby agent miał sygnał handlowy.

        D=0: wszyscy = 0.20 (neutralna strefa ±0.20)
        D=1: uniform [0.08, 0.38] — od bardzo reaktywnych do ostrożnych
        """
        base = 0.20
        if not cfg.threshold_spread or d < 1e-6:
            return base
        low  = max(0.05, base * (1.0 - 0.60 * d))
        high = min(0.45, base * (1.0 + 0.90 * d))
        return float(np.clip(self.rng.uniform(low, high), 0.05, 0.45))

    def _sample_risk_aversion(self, d, cfg) -> float:
        """
        Awersja do ryzyka pozycji — λ w: reward -= λ*(inv/max_inv)².

        Interpretacja:
          λ=0.0: agent ignoruje ryzyko — buduje max pozycję
          λ=1.0: umiarkowana kara za dużą pozycję
          λ=3.0: agent konserwatywny — unika dużych pozycji

        Rozkład: LogNormal(log(1.0), 0.5*d) clip[0, 3]
          D=0: wszyscy λ=1.0
          D=1: zakres [0.1, 3.0] — od agresywnych do konserwatywnych
        """
        if not cfg.risk_aversion_spread or d < 1e-6:
            return 1.0
        sigma = 0.5 * d
        raw = self.rng.lognormal(mean=-sigma**2/2, sigma=sigma)  # median=1.0
        return float(np.clip(raw, 0.05, 3.0))

    def _sample_wealth(self, d, cfg) -> float:
        if not cfg.wealth_spread or d < 1e-6:
            return 1.0
        # Pareto(1.5): realistyczna nierówność majątku
        raw = float(self.rng.pareto(1.5) + 1.0)
        return float(np.clip(raw * d + 1.0 * (1.0 - d), 0.05, 20.0))

    def _wealth_to_max_position(self, wealth: float, d: float) -> int:
        """
        Bogaty agent może trzymać większą pozycję.
        D=0: wszyscy = env_cfg.max_position (domyślne 5)
        D>0: max_pos = round(wealth × default), clip [1, default×2]
        wealth=1.0 → max_pos=5 (bez zmiany)
        wealth=2.0 → max_pos=10, wealth=0.5 → max_pos=2
        """
        base = self.env_cfg.max_position
        if d < 1e-6:
            return base
        return max(1, min(base * 2, round(wealth * base)))

    def max_theoretical_surplus(self) -> float:
        """Legacy metric: model sentimentu nie ma statycznego surplusu."""
        return 0.0

    def diversity_stats(self) -> dict:
        """Statystyki opisowe populacji — do logowania i wykresów."""
        sentiments = [p.sentiment for p in self.agents.values()]
        gammas     = [p.gamma     for p in self.agents.values()]
        wealth     = [p.wealth    for p in self.agents.values()]
        thrs       = [p.threshold for p in self.agents.values()]
        alphas     = [p.alpha_i   for p in self.agents.values()]
        betas      = [p.beta_i    for p in self.agents.values()]
        v_perc     = [p.V_perceived for p in self.agents.values()]

        return {
            "D":                self.diversity_score,
            "eq_price":         self.eq_price,
            "sentiment_mean":   float(np.mean(sentiments)),
            "sentiment_std":    float(np.std(sentiments)),
            "sentiment_range":  float(np.ptp(sentiments)),
            "gamma_mean":       float(np.mean(gammas)),
            "gamma_std":        float(np.std(gammas)),
            "wealth_gini":      _gini(wealth),
            "threshold_mean":   float(np.mean(thrs)),
            "threshold_std":    float(np.std(thrs)),
            "alpha_mean":       float(np.mean(alphas)),
            "alpha_std":        float(np.std(alphas)),
            "beta_mean":        float(np.mean(betas)),
            "beta_std":         float(np.std(betas)),
            "v_perceived_mean": float(np.mean(v_perc)),
            "v_perceived_std":  float(np.std(v_perc)),
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
      patrz get_observation(); wektor opisuje sentiment, pozycję,
      unrealized P&L, czas, gamma, ruch ceny od startu epizodu,
      krótki trend i value gap.

    Reward:
      realized_pnl_this_step (bez kar mid-episode).
    """

    def __init__(self, cfg: HTMConfig, seed: Optional[int] = None):
        self.cfg        = cfg
        self.rng        = np.random.default_rng(seed)
        self.population: Optional[AgentPopulation] = None
        self._eq_price  = cfg.market.eq_center
        self._ref_price = cfg.market.eq_center  # aktualizowany po transakcjach
        self._episode_start_price: float = cfg.market.eq_center
        self._step_prices: List[float]   = []

        # Stan epizodu
        self._step:               int              = 0
        self._done:               bool             = False
        self._V_t_prev:           float            = self._eq_price  # poprzedni V_t dla info signal
        self._rewards:            Dict[str, float] = {}
        self._prev_price:         float            = 0.5
        self._actions_log:        List[dict]       = []
        self._price_window:       List[float]      = []
        self._episode_pnl:        Dict[str, float] = {}
        self._realized_this_step: Dict[str, float] = {}  # realized PnL per agent w bieżącym kroku
        self._price_history:      List[float]      = []
        self._n_fills:            int              = 0
        self._n_position_closes:  int              = 0
        self._terminal_pnl:       Dict[str, float] = {}

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
        self._V_t_prev = self._eq_price

        self._step        = 0
        self._done        = False
        self._actions_log = []
        self._price_window= []
        self._price_history = [self._ref_price]
        self._episode_start_price = self._ref_price
        self._step_prices = []
        self._n_fills = 0
        self._n_position_closes = 0

        # Reset pozycji agentów
        for p in self.population.agents.values():
            p.reset_position()

        self._rewards    = {aid: 0.0 for aid in self.population.agents}
        self._prev_price = self._ref_price
        self._episode_pnl = {aid: 0.0 for aid in self.population.agents}
        self._realized_this_step = {}
        self._terminal_pnl = {}

        _log.debug(
            f"RESET | D={diversity_score:.2f} | eq={self._eq_price:.3f} | "
f"N={self.cfg.env.n_agents}"
        )

        return {aid: self.get_observation(aid)
                for aid in self.population.agents}

    def reset_episode(self) -> Dict[str, np.ndarray]:
        """
        Reset na nowy epizod: macro dryft V_t + nowy sygnał sentimentu + reset portfeli.

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

        # Kierunek dryfu V_t jako sygnał dla agentów
        V_change    = self._eq_price - self._V_t_prev
        V_direction = float(np.tanh(V_change / max(sc.sigma_macro, 1e-6)))

        # 2. Aktualizacja sentimentów i V_perceived na nowy epizod
        for p in self.population.agents.values():
            # Prywatna interpretacja kierunku zmiany V_t (dla sentimentu)
            private = float(np.clip(
                V_direction + self.rng.normal(0, sc.sigma_news), -1.0, 1.0
            ))
            # Blend sentimentu: 40% stary + 60% nowy sygnał
            p.sentiment = float(np.clip(
                0.4 * p.sentiment + 0.6 * private, -1.0, 1.0
            ))
            private_V_change = V_change + self.rng.normal(0, sc.sigma_macro)
            implied_V = p.V_perceived + private_V_change
            p.V_perceived = float(np.clip(
                (1.0 - p.news_sensitivity) * p.V_perceived
                + p.news_sensitivity * implied_V,
                self.cfg.env.p_min, self.cfg.env.p_max,
            ))
            p.reset_position()

        # 3. Reset stanu epizodu (nie: P_t, wagi sieci, epsilon)
        self._step            = 0
        self._done            = False
        self._actions_log     = []
        self._price_window    = []
        self._price_history   = [self._ref_price]
        self._episode_start_price = self._ref_price
        self._step_prices         = []
        self._n_fills         = 0
        self._n_position_closes = 0
        self._rewards             = {aid: 0.0 for aid in self.population.agents}
        self._prev_price          = self._ref_price
        self._episode_pnl         = {aid: 0.0 for aid in self.population.agents}
        self._realized_this_step  = {}
        self._terminal_pnl        = {}

        return {aid: self.get_observation(aid) for aid in self.population.agents}

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
            "sentiment":  params.sentiment,
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
        if realized != 0.0:
            self._realized_this_step[agent_id] = (
                self._realized_this_step.get(agent_id, 0.0) + realized
            )

        return {"agent_id": agent_id, "side": self.cfg.env.action_name(action_idx), "price": p_exec}

    def execute_parallel_actions(self, actions: dict) -> None:
        """
        Parallel execution: wszyscy agenci obserwują ten sam P_t,
        składają akcje jednocześnie. Cena przesuwa się RAZ na podstawie
        net_flow = liczba_BUY - liczba_SELL.

        Zastępuje sekwencyjne wywołania execute_single_action w pętli.
        Używane przez: train_deep_sarsa.py, train_ppo.py, run_zi_baseline.

        Args:
            actions: {agent_id: action_idx} — mapa akcji dla wszystkich agentów
        """
        if self._done:
            return None

        e = self.cfg.env

        buys = [
            aid for aid, action in actions.items()
            if aid in self.population.agents
            and action == e.ACTION_BUY_MARKET
            and self.population.agents[aid].position
                < self.population.agents[aid].max_position
        ]
        sells = [
            aid for aid, action in actions.items()
            if aid in self.population.agents
            and action == e.ACTION_SELL_MARKET
            and self.population.agents[aid].position
                > -self.population.agents[aid].max_position
        ]

        net_flow = len(buys) - len(sells)
        P_before = self._ref_price

        # Rozlicz kupujących — wszyscy po tej samej cenie przed ruchem.
        for aid in buys:
            p_exec = float(np.clip(P_before + e.half_spread, e.p_min, e.p_max))
            realized = self._execute_fill(aid, "buy", p_exec)
            self._record_fill_price(p_exec)
            if realized != 0.0:
                self._realized_this_step[aid] = (
                    self._realized_this_step.get(aid, 0.0) + realized
                )

        # Rozlicz sprzedających — wszyscy po tej samej cenie przed ruchem.
        for aid in sells:
            p_exec = float(np.clip(P_before - e.half_spread, e.p_min, e.p_max))
            realized = self._execute_fill(aid, "sell", p_exec)
            self._record_fill_price(p_exec)
            if realized != 0.0:
                self._realized_this_step[aid] = (
                    self._realized_this_step.get(aid, 0.0) + realized
                )

        # Przesuń cenę raz na podstawie net_flow.
        if net_flow != 0:
            self._ref_price = float(np.clip(
                P_before + e.perm_impact * net_flow,
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
                "position":   self.population.agents[aid].position,
                "sentiment":  self.population.agents[aid].sentiment,
                "ref_price":  self._ref_price,
            })

        return None

    def compute_step_rewards(
        self,
    ) -> Tuple[Dict[str, float], Dict[str, bool]]:
        """
        Oblicza nagrody, dryfuje V_t i aktualizuje sentimenty.

        Kolejność per krok:
          1. Intra-episode dryft V_t
          2. Ewentualny sygnał informacyjny (p_info)
          3. Update sentimentów z ruchu ceny (raz per krok)
          4. Obliczenie nagród (3 składniki)
          5. Terminacja jeśli step >= T
        """
        e  = self.cfg.env
        sc = self.cfg.sentiment

        # 1. Dryft V_t
        self._drift_V_t()

        # 2. Sygnał informacyjny z V_t (stochastycznie)
        if self.rng.random() < sc.p_info:
            self._emit_info_signal()

        # 3. Update sentimentów na podstawie ruchu ceny w tym kroku
        self._update_sentiments(self._prev_price)

        # 4. Oblicz nagrody
        # Reward = wyłącznie zrealizowany PnL z tego kroku.
        # Brak kar mid-episode za trzymanie pozycji: kary powodowały że sieć
        # uczyła się zamykać pozycje zbyt wcześnie (po 1-5 krokach), co przy
        # random-walk rynku i koszcie spreadu daje accuracy < 0.5.
        # Naturalna kara za złe pozycje istnieje przez terminal liquidation
        # w _liquidate_terminal_positions() przy done=True.
        rewards: Dict[str, float] = {}
        for aid, p in self.population.agents.items():
            realized = self._realized_this_step.get(aid, 0.0)
            rewards[aid] = realized
            self._episode_pnl[aid] = self._episode_pnl.get(aid, 0.0) + realized

        self._step_prices.append(self._ref_price)
        self._prev_price         = self._ref_price
        self._realized_this_step = {}

        self._step += 1
        done = self._step >= e.episode_steps
        if done:
            if e.auto_liquidate_end:
                terminal = self._liquidate_terminal_positions()
                for aid, realized in terminal.items():
                    rewards[aid]          = rewards.get(aid, 0.0) + realized
                    self._episode_pnl[aid] = self._episode_pnl.get(aid, 0.0) + realized
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
        agent = self.population.agents[agent_id]
        old = agent.position
        realized = 0.0

        if side == "buy":
            if old < 0:
                realized = agent.entry_price - p_exec
                self._register_close(agent, realized)
                if old + 1 == 0:
                    agent.entry_price = 0.0
            elif old == 0:
                agent.entry_price = p_exec
            else:
                agent.entry_price = (agent.entry_price * old + p_exec) / (old + 1)
            agent.position += 1

        elif side == "sell":
            if old > 0:
                realized = p_exec - agent.entry_price
                self._register_close(agent, realized)
                if old - 1 == 0:
                    agent.entry_price = 0.0
            elif old == 0:
                agent.entry_price = p_exec
            else:
                agent.entry_price = (agent.entry_price * abs(old) + p_exec) / (abs(old) + 1)
            agent.position -= 1
        else:
            raise ValueError(f"Unknown side: {side}")

        agent.realized_pnl += realized
        _log.debug(
            f"  FILL {agent_id} {side} pos:{old}->{agent.position} "
            f"p={p_exec:.4f} r={realized:.4f}"
        )
        return realized

    def _register_close(self, agent: AgentParams, realized: float) -> None:
        agent.n_trades_closed += 1
        self._n_position_closes += 1
        if realized > 0:
            agent.n_trades_won += 1

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

        for aid, p in self.population.agents.items():
            total = 0.0

            while p.position > 0:
                p_exec = float(np.clip(self._ref_price, e.p_min, e.p_max))
                realized = p_exec - p.entry_price
                total += realized
                p.realized_pnl += realized
                p.position -= 1

            while p.position < 0:
                p_exec = float(np.clip(self._ref_price, e.p_min, e.p_max))
                realized = p.entry_price - p_exec
                total += realized
                p.realized_pnl += realized
                p.position += 1

            p.entry_price = 0.0
            if total != 0.0:
                realized_by_agent[aid] = total

        self._terminal_pnl = realized_by_agent
        return realized_by_agent

    def _update_sentiments(self, p_t_before: float) -> None:
        """
        Aktualizacja sentimentów na podstawie ruchu ceny i wyceny fundamentalnej.

        Dwa składniki:
          - momentum (alpha_i): reaguje na zmianę ceny w tym kroku
          - value (beta_i): ciągnie sentiment w kierunku (V_perceived - P_t)
            → agent z V_perceived > P_t staje się bardziej byczy (uważa że tanio)
            → agent z V_perceived < P_t staje się bardziej niedźwiedzi (uważa że drogo)

        Wywoływana RAZ per krok z compute_step_rewards().
        """
        sc = self.cfg.sentiment
        delta_P      = self._ref_price - p_t_before
        price_signal = float(np.tanh(delta_P / max(sc.sigma_P, 1e-6)))

        for agent in self.population.agents.values():
            s = agent.sentiment
            s += agent.alpha_i * (price_signal - s)
            value_signal = float(np.tanh(
                (agent.V_perceived - self._ref_price) / 0.05
            ))
            s += agent.beta_i * (value_signal - s)
            agent.sentiment = float(np.clip(s, -1.0, 1.0))

    def _emit_info_signal(self) -> None:
        """
        Periodyczny sygnał informacyjny z V_t do prywatnej wyceny agentów.

        Każdy agent aktualizuje swoje V_perceived — zaszumioną estymację V_t.
        Agenci z wyższym news_sensitivity szybciej śledzą zmiany V_t.
        Sentiment jest aktualizowany przez _update_sentiments() na podstawie
        ruchu ceny — V_perceived to osobny kanał informacyjny.
        """
        sc = self.cfg.sentiment
        V_change = self._eq_price - self._V_t_prev
        if abs(V_change) < 1e-8:
            return

        for agent in self.population.agents.values():
            private_V_change = V_change + self.rng.normal(0, sc.sigma_macro)
            implied_V = agent.V_perceived + private_V_change
            agent.V_perceived = float(np.clip(
                (1.0 - agent.news_sensitivity) * agent.V_perceived
                + agent.news_sensitivity * implied_V,
                self.cfg.env.p_min, self.cfg.env.p_max,
            ))

    def _drift_V_t(self) -> None:
        """
        Intra-episode dryft V_t — wywoływany z compute_step_rewards().

        V_t dryfuje niezależnie od P_t. P_t zmienia się tylko przez transakcje.
        Opcjonalne szoki z MarketDynamics (gdy drift_enabled=True).
        """
        sc  = self.cfg.sentiment
        md  = self.cfg.market
        self._V_t_prev = self._eq_price

        drift = self.rng.normal(0, sc.sigma_intra)
        if md.drift_enabled and self.rng.random() < md.shock_probability:
            drift += float(self.rng.choice([-1, 1])) * md.shock_size

        self._eq_price = float(np.clip(
            self._eq_price + drift,
            self.cfg.env.p_min, self.cfg.env.p_max,
        ))

    # -----------------------------------------------------------------------
    # Obserwacja
    # -----------------------------------------------------------------------

    def get_observation(self, agent_id: str) -> np.ndarray:
        """
        8D wektor obserwacji — Sentiment model z informacją cenową.

          [0] sentiment       bear/bull state ∈ [-1, +1]
          [1] position_norm   position / max_position ∈ [-1, +1]
          [2] unrealized_pnl  znormalizowany niezrealizowany P&L ∈ [-2, +2]
          [3] time_remaining  (T - step) / T ∈ [0, 1]
          [4] gamma           discount factor ∈ [0.80, 0.99]
          [5] price_vs_start  (P_t - P_episode_start) / 0.1, clip [-3, +3]
          [6] trend_short     tanh(ΔP_8steps / σ_P) ∈ (-1, +1)
          [7] value_gap       tanh((V_perceived - P_t) / 0.05) ∈ (-1, +1)
        """
        p    = self.population.agents[agent_id]
        T    = self.cfg.env.episode_steps
        MPOS = max(p.max_position, 1)

        pos_norm = float(np.clip(p.position / MPOS, -1.0, 1.0))
        time_rem = float(np.clip(1.0 - self._step / max(T, 1), 0.0, 1.0))

        if p.position != 0 and p.entry_price > 0:
            unrealized = float(np.clip(
                (self._ref_price - p.entry_price) * p.position / 0.05,
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

        value_gap = float(np.tanh(
            (p.V_perceived - self._ref_price) / 0.05
        ))

        return np.array([
            float(np.clip(p.sentiment, -1.0, 1.0)),            # [0]
            pos_norm,                                           # [1]
            unrealized,                                         # [2]
            time_rem,                                           # [3]
            p.gamma,                                            # [4]
            price_vs_start,                                     # [5]
            trend_short,                                        # [6]
            value_gap,                                          # [7]
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
        return list(self.population.agents.keys()) if self.population else []

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
        pnls     = self._episode_pnl   # realized PnL per agent
        pnl_vals = [v for v in pnls.values()]
        mean_pnl = float(np.mean(pnl_vals)) if pnl_vals else 0.0
        positive_pnl = sum(1 for v in pnls.values() if v > 0)

        # Trade accuracy: ile zamkniętych transakcji było zyskownych
        total_closed = sum(p.n_trades_closed for p in self.population.agents.values())
        total_won    = sum(p.n_trades_won    for p in self.population.agents.values())
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
        final_pos   = {aid: p.position for aid, p in self.population.agents.items()}
        mean_pos    = float(np.mean(list(final_pos.values())))
        mean_abs_pos = float(np.mean([abs(v) for v in final_pos.values()]))
        open_pos    = sum(1 for v in final_pos.values() if v != 0)
        positive_pnl_frac = float(positive_pnl / max(self.cfg.env.n_agents, 1))
        terminal_pnl_vals  = list(self._terminal_pnl.values())
        mean_terminal_pnl  = float(np.mean(terminal_pnl_vals)) if terminal_pnl_vals else 0.0
        terminal_positive  = sum(1 for v in terminal_pnl_vals if v > 0)

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
            "mean_price_deviation":mean_dev,
            "ref_price_final":     self._ref_price,
            "eq_price":            self._eq_price,
            "diversity_score":     self.population.diversity_score,
            "n_agents":            self.cfg.env.n_agents,
            "n_steps":             self._step,
            "open_positions_end":  open_pos,
            "mean_position_end":   mean_pos,
            "mean_abs_position":   mean_abs_pos,
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
                "sentiment":       p.sentiment,
                "alpha_i":         p.alpha_i,
                "beta_i":          p.beta_i,
                "news_sensitivity":p.news_sensitivity,
                "V_perceived":     p.V_perceived,
                "threshold":       p.threshold,
                "gamma":           p.gamma,
                "risk_aversion":   p.risk_aversion,
                "position":        p.position,
                "entry_price":     p.entry_price,
                "ep_pnl":          self._episode_pnl.get(aid, 0.0),
                "realized_pnl":    p.realized_pnl,
                "n_trades_closed": p.n_trades_closed,
                "n_trades_won":    p.n_trades_won,
                "trade_accuracy":  float(p.n_trades_won / max(p.n_trades_closed, 1)),
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
