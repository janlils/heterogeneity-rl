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
  DoubleAuction    — główne środowisko CT z sekwencyjną egzekucją market maker
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
      - Dodane: alpha_i, beta_i, news_sensitivity
    """
    agent_id:           str
    sentiment:          float = 0.0    # ∈ [-1, +1]: -1 = silnie niedźwiedzi, +1 = byczo
    alpha_i:            float = 0.08   # momentum: waga sygnału cenowego per krok
    beta_i:             float = 0.06   # mean reversion: powrót do neutralu (0.0)
    news_sensitivity:   float = 0.12   # waga sygnału informacyjnego z V_t

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

            self.agents[aid] = AgentParams(
                agent_id        = aid,
                sentiment       = sentiment_0,
                alpha_i         = alpha_i,
                beta_i          = beta_i,
                news_sensitivity= news_sens,
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
          γ=0.70 → efektywny horyzont ≈ 3–5 kroków  (scalper)
          γ=0.90 → horyzont ≈ 15–20 kroków           (swing trader)
          γ=0.99 → horyzont ≈ 100 kroków              (pozycyjny)

        Dolna granica 0.70: agent z γ<0.70 ma horyzont <4 kroków
        przy T=200 — niestabilna nauka, bez sensu ekonomicznego.
        """
        if not cfg.gamma_spread or d < 1e-6:
            return 0.90
        low  = 0.90 - 0.20 * d   # D=0.5: 0.80,  D=1.0: 0.70
        high = 0.90 + 0.09 * d   # D=0.5: 0.945, D=1.0: 0.99
        return float(np.clip(self.rng.uniform(low, high), 0.70, 0.99))

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
        }


# ===========================================================================
# Order i Trade
# ===========================================================================

@dataclass
class Order:
    agent_id:   str
    order_type: str    # "bid" lub "ask"
    price:      float
    valuation:  float = 0.0  # legacy: expected_price/fair price przy starym OrderBook
    timestamp:  int = 0


@dataclass
class Trade:
    buyer_id:       str
    seller_id:      str
    price:          float
    timestamp:      int
    buyer_val:      float = 0.0   # wycena kupca
    seller_val:     float = 0.0   # wycena sprzedawcy
    buyer_surplus:  float = 0.0   # buyer_val - price
    seller_surplus: float = 0.0   # price - seller_val

    @property
    def total_surplus(self) -> float:
        return self.buyer_surplus + self.seller_surplus

    @property
    def is_profitable(self) -> bool:
        """True jeśli obie strony zarabiają."""
        return self.buyer_surplus > 0 and self.seller_surplus > 0


# ===========================================================================
# OrderBook
# ===========================================================================

class OrderBook:
    """
    Continuous Double Auction z priorytetem cena-czas.
    submit_batch() — dla parallel step (wszyscy jednocześnie).
    submit()       — dla sekwencyjnego ZI baseline.
    """

    def __init__(self):
        self.bids:          List[Order] = []
        self.asks:          List[Order] = []
        self.trade_history: List[Trade] = []
        self.price_history: List[float] = []
        self._step = 0

    def reset(self):
        self.bids.clear(); self.asks.clear()
        self.trade_history.clear(); self.price_history.clear()
        self._step = 0

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None

    @property
    def last_price(self) -> Optional[float]:
        return self.price_history[-1] if self.price_history else None

    @property
    def spread(self) -> Optional[float]:
        bb, ba = self.best_bid, self.best_ask
        return (ba - bb) if (bb is not None and ba is not None) else None

    def submit_batch(self, orders: List[Order]) -> List[Trade]:
        """
        Parallel step: wszystkie oferty jednocześnie.
        Sortuj bidy malejąco i aski rosnąco, matchuj chciwością.
        """
        bids = sorted(
            [o for o in orders if o.order_type == "bid"], key=lambda o: -o.price
        )
        asks = sorted(
            [o for o in orders if o.order_type == "ask"], key=lambda o: o.price
        )

        trades = []
        bi = ai = 0
        matched_buyers  = set()
        matched_sellers = set()

        while bi < len(bids) and ai < len(asks):
            bid = bids[bi]
            ask = asks[ai]

            # Pomijaj agentów już dopasowanych w tej rundzie
            if bid.agent_id in matched_buyers:
                bi += 1; continue
            if ask.agent_id in matched_sellers:
                ai += 1; continue

            if bid.price >= ask.price:
                trade = self._execute(bid, ask)
                trades.append(trade)
                matched_buyers.add(bid.agent_id)
                matched_sellers.add(ask.agent_id)
                bi += 1; ai += 1
            else:
                break   # najlepszy bid < najlepszy ask → koniec dopasowań

        # Niezrealizowane oferty trafiają do kolejki
        for b in bids:
            if b.agent_id not in matched_buyers:
                b.timestamp = self._step; self._step += 1
                self.bids = [o for o in self.bids if o.agent_id != b.agent_id]
                self.bids.append(b)
                self.bids.sort(key=lambda o: (-o.price, o.timestamp))

        for a in asks:
            if a.agent_id not in matched_sellers:
                a.timestamp = self._step; self._step += 1
                self.asks = [o for o in self.asks if o.agent_id != a.agent_id]
                self.asks.append(a)
                self.asks.sort(key=lambda o: (o.price, o.timestamp))

        return trades

    def submit(self, order: Order) -> Optional[Trade]:
        """Sekwencyjne — dla ZI baseline."""
        order.timestamp = self._step; self._step += 1
        if order.order_type == "bid":
            return self._process_bid(order)
        return self._process_ask(order)

    def remove_agent(self, agent_id: str):
        self.bids = [o for o in self.bids if o.agent_id != agent_id]
        self.asks = [o for o in self.asks if o.agent_id != agent_id]

    def _process_bid(self, bid: Order) -> Optional[Trade]:
        if self.asks and bid.price >= self.asks[0].price:
            ask = self.asks.pop(0)
            return self._execute(bid, ask)
        self.bids = [o for o in self.bids if o.agent_id != bid.agent_id]
        self.bids.append(bid)
        self.bids.sort(key=lambda o: (-o.price, o.timestamp))
        return None

    def _process_ask(self, ask: Order) -> Optional[Trade]:
        if self.bids and self.bids[0].price >= ask.price:
            bid = self.bids.pop(0)
            return self._execute(bid, ask)
        self.asks = [o for o in self.asks if o.agent_id != ask.agent_id]
        self.asks.append(ask)
        self.asks.sort(key=lambda o: (o.price, o.timestamp))
        return None

    def _execute(self, bid: Order, ask: Order) -> Trade:
        price = (bid.price + ask.price) / 2.0
        trade = Trade(
            buyer_id=bid.agent_id, seller_id=ask.agent_id,
            price=price, timestamp=self._step,
            buyer_val=bid.valuation, seller_val=ask.valuation,
        )
        self.price_history.append(price)
        self.trade_history.append(trade)
        _log.debug(
            f"  TRADE {bid.agent_id}(v={bid.valuation:.2f}) × "
            f"{ask.agent_id}(v={ask.valuation:.2f}) @ {price:.3f}"
        )
        return trade


# ===========================================================================
# DoubleAuction — główne środowisko
# ===========================================================================

class DoubleAuction:
    """
    Środowisko spekulacyjne HTM.

    Przepływ jednego epizodu:
      reset(D) → [execute_single_action(...) + compute_step_rewards()] × T → episode_metrics()

    Przestrzeń akcji: 0=HOLD, 1=BUY, 2=SELL.

    Obserwacja (15D):
      patrz get_observation(); wektor opisuje oczekiwaną cenę, cenę rynku,
      sygnał wartości, trend/zmienność, pozycję, unrealized P&L, czas,
      gamma, awersję do ryzyka, kotwicę oczekiwań, próg, drift oczekiwań
      i skumulowany realized P&L epizodu.

    Reward:
      realized_pnl_this_step + alignment - risk_penalty.
    """

    PASS_ACTION = None  # ustawiany z cfg.env.pass_action

    def __init__(self, cfg: HTMConfig, seed: Optional[int] = None):
        self.cfg        = cfg
        self.rng        = np.random.default_rng(seed)
        # OrderBook zostaje jako artefakt kompatybilności, ale główna ścieżka
        # egzekucji używa market makera i nie składa zleceń do książki.
        self.order_book = OrderBook()
        self.population: Optional[AgentPopulation] = None
        self._eq_price  = cfg.market.eq_center
        self._ref_price = cfg.market.eq_center  # aktualizowany po transakcjach


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

        self.order_book.reset()
        self._step        = 0
        self._done        = False
        self._actions_log = []
        self._price_window= []
        self._price_history = [self._ref_price]
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

        # 2. Aktualizacja sentimentów na nowy epizod
        for p in self.population.agents.values():
            # Prywatna interpretacja sygnału z V_t
            private = float(np.clip(
                V_direction + self.rng.normal(0, sc.sigma_news), -1.0, 1.0
            ))
            # Blend: 40% stary sentiment + 60% nowy sygnał
            p.sentiment = float(np.clip(
                0.4 * p.sentiment + 0.6 * private, -1.0, 1.0
            ))
            p.reset_position()

        # 3. Reset stanu epizodu (nie: P_t, wagi sieci, epsilon)
        self.order_book.reset()
        self._step            = 0
        self._done            = False
        self._actions_log     = []
        self._price_window    = []
        self._price_history   = [self._ref_price]
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
    # Parallel step — główny interfejs dla RL
    # -----------------------------------------------------------------------

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
        T              = e.episode_steps
        time_remaining = max(0.0, 1.0 - self._step / max(T, 1))
        time_urgency   = max(0.0, 1.0 - time_remaining / e.holding_urgency_horizon)

        rewards: Dict[str, float] = {}
        for aid, p in self.population.agents.items():
            realized = self._realized_this_step.get(aid, 0.0)

            pos_norm     = p.position / max(p.max_position, 1)
            risk_penalty = e.risk_penalty_kappa * p.risk_aversion * (pos_norm ** 2)

            unrealized_raw = 0.0
            if p.position != 0 and p.entry_price > 0:
                unrealized_raw = (self._ref_price - p.entry_price) * p.position
            holding_cost = e.holding_cost_kappa * max(0.0, unrealized_raw) * time_urgency

            reward      = realized - risk_penalty - holding_cost
            rewards[aid] = reward
            self._episode_pnl[aid] = self._episode_pnl.get(aid, 0.0) + realized

        self._prev_price         = self._ref_price
        self._realized_this_step = {}

        self._step += 1
        done = self._step >= T
        if done:
            if e.auto_liquidate_end:
                terminal = self._liquidate_terminal_positions()
                for aid, realized in terminal.items():
                    rewards[aid]          = rewards.get(aid, 0.0) + realized
                    self._episode_pnl[aid] = self._episode_pnl.get(aid, 0.0) + realized
            self._done = True

        dones = {aid: done for aid in self.population.agents}
        return rewards, dones

    def parallel_step(
        self,
        actions: Dict[str, int],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, dict]]:
        """
        Legacy wrapper dla starszych skryptów. Benchmark ZI i trening SARSA
        używają bezpośrednio execute_single_action + compute_step_rewards.
        """
        if self._done:
            all_agents = list(self.population.agents.keys())
            return (
                {aid: self.get_observation(aid) for aid in all_agents},
                {aid: 0.0 for aid in all_agents},
                {aid: True for aid in all_agents},
                {},
            )

        # Losowa kolejność — każdy agent widzi rynek zaktualizowany przez poprzedników
        agent_list = list(actions.keys())
        self.rng.shuffle(agent_list)
        for aid in agent_list:
            if aid in actions:
                self.execute_single_action(aid, actions[aid])

        rewards, dones = self.compute_step_rewards()

        all_agents = list(self.population.agents.keys())
        obs   = {aid: self.get_observation(aid) for aid in all_agents}
        infos = {
            aid: {
                "n_trades":  self._n_fills,
                "ref_price": self._ref_price,
                "position":  self.population.agents[aid].position,
            }
            for aid in all_agents
        }
        return obs, rewards, dones, infos

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


    def _settle_ct(self, trade: Trade) -> Tuple[float, float]:
        """
        Rozliczenie transakcji z śledzeniem realized PnL.

        Używa average cost basis:
          - Otwierasz: entry_price = trade_price
          - Zwiększasz: entry_price = średnia ważona pozycjami
          - Zamykasz: realized = (trade_price - entry_price) × zamknięte_jednostki
          - Odwracasz (np. long→short): realizujesz całość, otwierasz nową stronę

        Zwraca (buyer_realized, seller_realized).
        """
        buyer  = self.population.agents[trade.buyer_id]
        seller = self.population.agents[trade.seller_id]
        p      = trade.price
        buy_r  = sell_r = 0.0

        # ── Kupujący: position += 1 ──────────────────────────────────────
        old_b = buyer.position
        if old_b < 0:
            # Zamknięcie shorta (lub przejście na long)
            buy_r = buyer.entry_price - p          # profit: sprzedałeś drogo, odkupujesz tanio
            buyer.n_trades_closed += 1
            if buy_r > 0:
                buyer.n_trades_won += 1
            if old_b + 1 == 0:
                buyer.entry_price = 0.0            # pozycja neutralna
            elif old_b + 1 > 0:
                buyer.entry_price = p              # odwrócenie: nowy long po tej cenie
            # else: nadal short ale mniejszy — entry_price bez zmian
        elif old_b == 0:
            buyer.entry_price = p                  # otwierasz long
        else:
            # Zwiększasz long — average cost
            buyer.entry_price = (buyer.entry_price * old_b + p) / (old_b + 1)

        buyer.position  += 1
        buyer.realized_pnl += buy_r

        # ── Sprzedający: position -= 1 ───────────────────────────────────
        old_s = seller.position
        if old_s > 0:
            # Zamknięcie longa (lub przejście na short)
            sell_r = p - seller.entry_price        # profit: kupiłeś tanio, sprzedajesz drogo
            seller.n_trades_closed += 1
            if sell_r > 0:
                seller.n_trades_won += 1
            if old_s - 1 == 0:
                seller.entry_price = 0.0
            elif old_s - 1 < 0:
                seller.entry_price = p             # odwrócenie: nowy short po tej cenie
        elif old_s == 0:
            seller.entry_price = p                 # otwierasz short
        else:
            # Zwiększasz short — average cost
            seller.entry_price = (seller.entry_price * abs(old_s) + p) / (abs(old_s) + 1)

        seller.position -= 1
        seller.realized_pnl += sell_r

        _log.debug(
            f"  SETTLE {trade.buyer_id}(pos:{old_b}→{buyer.position} r={buy_r:.4f}) × "
            f"{trade.seller_id}(pos:{old_s}→{seller.position} r={sell_r:.4f}) @ {p:.3f}"
        )
        return buy_r, sell_r

    def _update_sentiments(self, p_t_before: float) -> None:
        """
        Aktualizacja sentimentów na podstawie ruchu ceny w tym kroku.

        Wywołana RAZ per krok z compute_step_rewards(), po wszystkich transakcjach.
        Używa różnicy między bieżącym ref_price a ceną na początku kroku.

        Mechanizm:
          - price_signal = tanh(ΔP_t / σ_P) ∈ (-1, +1)
          - momentum:  s += α_i * (price_signal - s)
          - reversion: s += β_i * (0 - s)
        """
        sc = self.cfg.sentiment
        delta_P      = self._ref_price - p_t_before
        price_signal = float(np.tanh(delta_P / max(sc.sigma_P, 1e-6)))

        for agent in self.population.agents.values():
            s  = agent.sentiment
            s += agent.alpha_i * (price_signal - s)
            s += agent.beta_i  * (0.0 - s)
            agent.sentiment = float(np.clip(s, -1.0, 1.0))

    def _emit_info_signal(self) -> None:
        """
        Periodyczny sygnał informacyjny z V_t do agentów.

        Wywoływana z compute_step_rewards() z prawdopodobieństwem p_info.
        Każdy agent interpretuje sygnał z własnym szumem prywatnym.
        news_sensitivity kontroluje jak mocno agent reaguje.
        """
        sc = self.cfg.sentiment
        V_change = self._eq_price - self._V_t_prev
        if abs(V_change) < 1e-8:
            return
        info_direction = float(np.tanh(V_change / max(sc.sigma_macro * 0.5, 1e-6)))

        for agent in self.population.agents.values():
            private = float(np.clip(
                info_direction + self.rng.normal(0, sc.sigma_news), -1.0, 1.0
            ))
            agent.sentiment += agent.news_sensitivity * (private - agent.sentiment)
            agent.sentiment  = float(np.clip(agent.sentiment, -1.0, 1.0))

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
    # Sekwencyjne submit — dla ZI baseline
    # -----------------------------------------------------------------------

    def submit(self, agent_id: str, price: float, order_type: str) -> Optional[Trade]:
        """Deprecated adapter: wykonuje natychmiastowy BUY/SELL przez market makera."""
        if self._done:
            return None

        if order_type == "bid":
            self.execute_single_action(agent_id, self.cfg.env.ACTION_BUY_MARKET)
        elif order_type == "ask":
            self.execute_single_action(agent_id, self.cfg.env.ACTION_SELL_MARKET)
        return None

    # -----------------------------------------------------------------------
    # Obserwacja
    # -----------------------------------------------------------------------

    def get_observation(self, agent_id: str) -> np.ndarray:
        """
        7D wektor obserwacji — Sentiment model.

          [0] sentiment       bear/bull state ∈ [-1, +1]
          [1] position_norm   position / max_position ∈ [-1, +1]
          [2] unrealized_pnl  znormalizowany niezrealizowany P&L
          [3] time_remaining  (T - step) / T ∈ [0, 1]
          [4] risk_aversion   znorm. ∈ [0, 1] — wchodzi do reward formula
          [5] threshold       znorm. ∈ [0, 1] — wrażliwość sygnału
          [6] gamma           discount factor ∈ [0.70, 0.99] — wchodzi do TD
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

        return np.array([
            float(np.clip(p.sentiment, -1.0, 1.0)),            # [0]
            pos_norm,                                           # [1]
            unrealized,                                         # [2]
            time_rem,                                           # [3]
            float(np.clip(p.risk_aversion / 3.0, 0.0, 1.0)),  # [4]
            float(np.clip(p.threshold / 0.45, 0.0, 1.0)),     # [5]
            p.gamma,                                            # [6]
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
            order = da.rng.permutation(agent_ids)
            for aid in order:
                obs = da.get_observation(aid)
                action = zi[aid].act(obs)
                da.execute_single_action(aid, action)
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


def _price_disc(prices, eq, threshold=0.05) -> int:
    for t, p in enumerate(prices):
        if abs(p - eq) <= threshold:
            return t
    return len(prices)


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
