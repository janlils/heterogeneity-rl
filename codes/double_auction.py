"""
codes/double_auction.py — Model spekulacyjny HTM
================================================
Kluczowa zmiana względem modelu G&S:

  Model docelowy:
    - Brak stałych ról. Każdy agent ma subiektywną oczekiwaną cenę aktywa.
    - Jeśli expected_price > ref_price → sygnał KUP / long.
    - Jeśli expected_price < ref_price → sygnał SPRZEDAJ / short.
    - BUY/SELL są wykonywane natychmiast przez market makera.
    - PnL wynika wyłącznie z cen wejścia i wyjścia, nie z expected_price.

Klasy:
  BeliefState      — przekonania i biasy behawioralne agenta
  AgentParams      — profil agenta (expected_price, threshold, gamma, wealth, belief)
  AgentPopulation  — generuje N agentów bez stałych ról
  DoubleAuction    — główne środowisko CT z sekwencyjną egzekucją market maker
  ZeroIntelligence — baseline (losowa agresywność, losowa decyzja buy/sell/pass)
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from codes.config import HTMConfig, EnvConfig, BeliefConfig, DiversityConfig, MarketDynamics

_log = logging.getLogger("htm.auction")


# ===========================================================================
# BeliefState — przekonania i biasy behawioralne
# ===========================================================================

@dataclass
class BeliefState:
    """
    Wewnętrzny model przekonań agenta o rynku.

    Stałe cechy (losowane przy tworzeniu, nie zmieniają się w treningu):
      update_speed   — jak szybko agent aktualizuje oczekiwania cenowe (alpha EMA)
                        N(0.30, σ*D); value investor ≈ 0.05, momentum ≈ 0.80
                        Barberis & Thaler (2003)
      anchoring_bias — zakotwiczenie do ceny ustawianej przy reset_dynamic()
                        Beta(2,5) skalowane; Tversky & Kahneman (1974)
                        większość ma umiarkowane zakotwiczenie, nieliczni brak
      loss_aversion  — λ w prospect theory: straty bolą λ× mocniej niż zyski
                        LogNormal(log(2.25), 0.3) clip[1,5]
                        Kahneman & Tversky (1979, 1992): mediana ≈ 2.25

    Usunięte parametry (były martwym kodem):
      panic_factor   — adjusted_aggressiveness() nie jest używana w ścieżce CT
      patience       — j.w.
      gamma          — duplikat AgentParams.gamma, zawsze identyczne

    Stan dynamiczny (aktualizowany po każdej zaobserwowanej transakcji):
      expected_price  — gdzie agent spodziewa się ceny (EMA z obserwacji)
      price_trend     — kierunek ostatniej zmiany EMA
      anchor_price    — kotwica resetowana wraz ze stanem dynamicznym
    """
    # Stałe cechy — heterogeniczne wymiary behawioralne
    update_speed:   float = 0.30   # alpha EMA: jak szybko adaptuje oczekiwania
    anchoring_bias: float = 0.00   # siła zakotwiczenia do bieżącej kotwicy resetu
    loss_aversion:  float = 2.25   # λ prospect theory (Kahneman-Tversky)

    # Stan dynamiczny (reset między rundami)
    expected_price: float = 0.50
    price_trend:    float = 0.00
    anchor_price:   float = 0.50

    def reset_dynamic(self, ref_price: float = 0.50) -> None:
        """Reset stanu dynamicznego między rundami."""
        self.expected_price = ref_price
        self.price_trend    = 0.00
        self.anchor_price   = ref_price

    def observe_price(self, new_price: float) -> None:
        """EMA aktualizacja oczekiwań + zakotwiczenie po każdej transakcji."""
        old              = self.expected_price
        self.price_trend = new_price - old
        self.expected_price = (
            (1 - self.update_speed) * old + self.update_speed * new_price
        )
        # Zakotwiczenie Tversky-Kahneman: przyciągnij z powrotem do kotwicy resetu.
        # TODO: porównać z wariantem persistent anchor jako osobny eksperyment.
        if self.anchoring_bias > 0:
            self.expected_price = (
                (1 - self.anchoring_bias) * self.expected_price
                + self.anchoring_bias * self.anchor_price
            )



    def subjective_surplus(self, raw_surplus: float) -> float:
        """Subiektywna wartość surplusa — straty bolą loss_aversion × mocniej."""
        return raw_surplus if raw_surplus >= 0 else self.loss_aversion * raw_surplus


# ===========================================================================
# AgentParams — kompletny profil agenta
# ===========================================================================

@dataclass
class AgentParams:
    """
    Profil agenta w modelu spekulacyjnym.

    Kluczowa różnica vs G&S:
      - BRAK pola 'role' (kupiec/sprzedawca)
      - 'expected_price' = subiektywna oczekiwana/fair cena aktywa
      - 'threshold' = minimalna różnica |val - price| żeby handlować
      - Rola wynika dynamicznie: val > price → kup, val < price → sprzedaj

    Uzasadnienie ekonomiczne:
      Agent kupuje gdy uważa że aktywo jest tanie (val > price).
      Agent sprzedaje gdy uważa że aktywo jest drogie (val < price).
      Taki handel wynika z heterogenicznych przekonań — klasyczny wynik
      De Long et al. (1990) i Scheinkman & Xiong (2003).
    """
    agent_id:           str
    expected_price:     float  # subiektywna oczekiwana/fair cena [0,1] — dryfuje
    long_run_fair_price:float  # bazowe oczekiwanie — kotwica powrotu
    belief_reversion:   float  # beta: zakotwiczenie do long_run_fair_price
    threshold:          float  # min |val-price| żeby mieć sygnał informacyjny
    gamma:              float = 0.95  # discount factor
    wealth:             float = 1.00  # kapitał (nie używany bezpośrednio w CT)
    risk_aversion:      float = 1.00  # λ: kara za otwartą pozycję (heterogeniczne)
    belief:             BeliefState = field(default_factory=BeliefState)

    # ── Continuous Trading: ekspozycja rynkowa ────────────────────────────
    max_position:       int   = 5      # max |position| (zależy od wealth)
    position:           int   = 0      # bieżąca pozycja ∈ [-max, +max]
    entry_price:        float = 0.0    # średni koszt wejścia (average cost basis)
    realized_pnl:       float = 0.0    # zrealizowany P&L w epizodzie
    n_trades_closed:    int   = 0      # liczba zamkniętych transakcji
    n_trades_won:       int   = 0      # zamknięte z zyskiem (dla trade_accuracy)

    @property
    def valuation(self) -> float:
        """Backward compat: stara nazwa dla expected_price."""
        return self.expected_price

    @valuation.setter
    def valuation(self, value: float) -> None:
        self.expected_price = value

    @property
    def base_valuation(self) -> float:
        """Backward compat: stara nazwa dla long_run_fair_price."""
        return self.long_run_fair_price

    @base_valuation.setter
    def base_valuation(self, value: float) -> None:
        self.long_run_fair_price = value

    def trade_signal(self, ref_price: float) -> str:
        """
        Wyznacza sygnał handlowy na podstawie własnej wyceny i ceny rynkowej.

        Returns: 'buy', 'sell', lub 'none' (różnica za mała)
        """
        diff = self.expected_price - ref_price
        if diff > self.threshold:
            return "buy"
        elif diff < -self.threshold:
            return "sell"
        return "none"

    def max_affordable_bid(self) -> float:
        """Deprecated helper kept for compatibility."""
        return float(np.clip(self.expected_price, 0.01, 0.99))

    def reset_position(self) -> None:
        """Reset pozycji i P&L na początku epizodu."""
        self.position        = 0
        self.entry_price     = 0.0
        self.realized_pnl    = 0.0
        self.n_trades_closed = 0
        self.n_trades_won    = 0

    def __repr__(self) -> str:
        return (
            f"Agent({self.agent_id}, expected={self.expected_price:.3f}, "
            f"thr={self.threshold:.3f}, γ={self.gamma:.2f}, w={self.wealth:.2f})"
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
        belief_cfg:      BeliefConfig,
        env_cfg:         EnvConfig,
        eq_price:        float = 0.50,
        seed:            Optional[int] = None,
    ):
        assert 0.0 <= diversity_score <= 1.0
        self.n_agents        = n_agents
        self.diversity_score = diversity_score
        self.diversity_cfg   = diversity_cfg
        self.belief_cfg      = belief_cfg
        self.env_cfg         = env_cfg
        self.eq_price        = eq_price
        self.rng             = np.random.default_rng(seed)
        self.agents: Dict[str, AgentParams] = {}
        self._generate()

    def _generate(self):
        d   = self.diversity_score
        cfg = self.diversity_cfg
        eq  = self.eq_price

        # ── Wyceny fundamentalne — rozkład NORMALNY ───────────────────────
        # Uzasadnienie: większość agentów ma przekonania bliskie konsensusowi
        # rynkowemu, nieliczni mają skrajne poglądy (De Long et al. 1990).
        # Uniform był nierealistyczny — dawał tyle samo agentów przy 0.1 co przy 0.5.
        #
        # D=0: std=0   → wszyscy val=eq → no-trade theorem (poprawnie)
        # D=0.5: std=0.10 → 68% agentów w [eq-0.10, eq+0.10]
        # D=1.0: std=0.20 → 68% w [eq-0.20, eq+0.20], ogonki do 0.05/0.95
        if cfg.valuation_spread and d > 1e-6:
            # std=0.25*d: przy D=0.3 → σ=0.075, D=1.0 → σ=0.25
            # Kalibrowane żeby ~30-70% agentów miało sygnał handlowy
            std        = 0.25 * d
            valuations = self.rng.normal(eq, std, self.n_agents)
        else:
            valuations = np.full(self.n_agents, eq)

        valuations = np.clip(valuations, 0.05, 0.95)

        for i in range(self.n_agents):
            aid    = f"agent_{i}"
            val    = float(valuations[i])
            gamma  = self._sample_gamma(d, cfg)
            belief = self._sample_belief(d, cfg, eq)

            wealth  = self._sample_wealth(d, cfg)
            max_pos = self._wealth_to_max_position(wealth, d)

            self.agents[aid] = AgentParams(
                agent_id         = aid,
                expected_price   = val,
                long_run_fair_price = val,
                belief_reversion = self._sample_belief_reversion(d, cfg, belief),
                threshold        = self._sample_threshold(d, cfg),
                gamma            = gamma,
                wealth           = wealth,
                risk_aversion    = self._sample_risk_aversion(d, cfg),
                belief           = belief,
                max_position     = max_pos,
            )

    def _sample_gamma(self, d, cfg) -> float:
        """
        Discount factor γ — teraz skalowany LINIOWO z D.

        D=0:   wszyscy = 0.90 (neutralni)
        D=0.5: uniform [0.70, 0.945]
        D=1.0: uniform [0.50, 0.99]

        Poprzedni błąd: przy D=0.1 i D=0.9 rozkład był identyczny ([0.5, 0.99]).
        Teraz rozrzut rośnie proporcjonalnie do D — im bardziej heterogeniczna
        populacja, tym większe różnice w cierpliwości agentów.

        Połączenie z akcjami limit order:
          niska gamma → niecierpliwy → preferuje ACTION_MARKET
          wysoka gamma → cierpliwy  → preferuje ACTION_LIMIT_FAR
        """
        if not cfg.gamma_spread or d < 1e-6:
            return 0.90
        # Zakres rośnie liniowo z D
        low  = 0.90 - 0.40 * d   # D=0.5: 0.70,  D=1.0: 0.50
        high = 0.90 + 0.09 * d   # D=0.5: 0.945, D=1.0: 0.99
        return float(np.clip(self.rng.uniform(low, high), 0.50, 0.99))

    def _sample_threshold(self, d, cfg) -> float:
        """
        Minimalny spread |val - price| żeby handlować.

        Poprzedni błąd: base=0.02 było za małe — prawie każdy agent
        zawsze miał sygnał, threshold nie robił selekcji.

        Nowy base=0.08: agent handluje tylko gdy cena odbiega o min 8%
        od jego wyceny. To tworzy realistyczną "strefę neutralną".

        D=0:   wszyscy = 0.08
        D=0.5: uniform [0.04, 0.16] — różni agenci, różna cierpliwość
        D=1.0: uniform [0.02, 0.26] — od bardzo reaktywnych do bardzo ostrożnych
        """
        base = self.env_cfg.trade_threshold_base  # 0.06
        if not cfg.threshold_spread or d < 1e-6:
            return base
        # Zakres rośnie liniowo z D — kalibrowany do std wycen
        # D=0.3: [0.045, 0.101] mean=0.073, przy std=0.075 → ~33% sygnałów
        # D=1.0: [0.015, 0.195] mean=0.105, przy std=0.250 → ~67% sygnałów
        low  = base * (1.0 - 0.75 * d)
        high = base * (1.0 + 2.25 * d)
        return float(np.clip(self.rng.uniform(low, high), 0.01, 0.40))

    def _sample_belief_reversion(self, d, cfg, belief: 'BeliefState') -> float:
        """
        Jak mocno agent wraca do swoich fundamentalnych przekonań per krok.

        Powiązanie z update_speed (odwrotne):
          Momentum trader (wysoki update_speed) → niskie belief_reversion
            Szybko podąża za ceną, nie wraca do fundamentów
          Value investor (niski update_speed) → wysokie belief_reversion
            Wolno się adaptuje, mocno trzyma fundamenty

        Zakres:
          D=0: wszyscy = base_reversion (0.30)
          D=1: losowane odwrotnie do update_speed
               update_speed ∈ [0.05, 0.95] → reversion ∈ [0.05, 0.75]
        """
        base_reversion = 0.30  # domyślne zakotwiczenie
        if d < 1e-6:
            return base_reversion
        # Odwrotna korelacja z update_speed:
        # momentum trader (upd=0.95) → reversion=0.05
        # value investor (upd=0.05)  → reversion=0.75
        # (1 - update_speed) skalowane do [0.05, 0.75]
        reversion = 0.05 + (1.0 - belief.update_speed) * 0.70
        return float(np.clip(reversion, 0.05, 0.85))

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

    def _sample_belief(self, d, cfg, eq) -> BeliefState:
        bc = self.belief_cfg
        if not cfg.belief_spread or d < 1e-6:
            return BeliefState(
                update_speed=bc.update_speed_center,
                anchoring_bias=0.0,
                loss_aversion=2.25,   # mediana Kahneman-Tversky przy D=0
                expected_price=eq,
                anchor_price=eq,
            )

        def clamp(x, lo, hi): return float(np.clip(x, lo, hi))

        # loss_aversion: LogNormal(log(2.25), 0.3*d) clip[1, 5]
        # Kahneman & Tversky (1992): mediana λ ≈ 2.25, rozkład skośny prawy
        # LogNormal gwarantuje λ > 0, i jest skośny — zgodnie z empirią
        # D skaluje odchylenie: D=0→wszyscy 2.25, D=1→pełny rozkład
        la_sigma = 0.30 * d
        if la_sigma > 1e-6:
            la_raw = self.rng.lognormal(
                mean=np.log(2.25) - la_sigma**2 / 2,  # żeby mediana = 2.25
                sigma=la_sigma
            )
        else:
            la_raw = 2.25
        loss_av = clamp(la_raw, 1.0, 5.0)

        # anchoring_bias: Beta(2,5) skalowane przez d * anchoring_spread
        # Tversky & Kahneman (1974): zakotwiczenie jest powszechne (większość > 0)
        # Beta(2,5): mediana ≈ 0.29, mało masy przy 0 — większość ma zakotwiczenie
        # Skalujemy przez d: D=0→brak, D=1→pełny rozkład Beta
        if d > 1e-6:
            beta_raw = self.rng.beta(2, 5)  # ∈ [0,1], mediana≈0.29
            anchor   = clamp(beta_raw * bc.anchoring_spread * d, 0.0, 1.0)
        else:
            anchor = 0.0

        return BeliefState(
            update_speed=clamp(
                self.rng.normal(bc.update_speed_center, bc.update_speed_spread * d),
                0.05, 0.95
            ),
            anchoring_bias=anchor,
            loss_aversion=loss_av,
            expected_price=eq,
            anchor_price=eq,
        )

    def max_theoretical_surplus(self) -> float:
        """
        Maksymalny możliwy surplus przy optymalnym matchowaniu.

        W modelu spekulacyjnym: parujemy agentów z val > eq (potencjalni kupcy)
        z agentami z val < eq (potencjalni sprzedawcy) malejąco według marży.
        Surplus pary = max(0, val_kupca - val_sprzedawcy).

        To jest < suma wszystkich val bo nie każda para jest profitowna.
        """
        buyers_vals  = sorted(
            [p.valuation for p in self.agents.values() if p.valuation > self.eq_price],
            reverse=True
        )
        sellers_vals = sorted(
            [p.valuation for p in self.agents.values() if p.valuation < self.eq_price]
        )
        n_pairs = min(len(buyers_vals), len(sellers_vals))
        return sum(
            max(0.0, buyers_vals[i] - sellers_vals[i])
            for i in range(n_pairs)
        )

    def diversity_stats(self) -> dict:
        """Statystyki opisowe populacji — do logowania i wykresów."""
        vals    = [p.valuation  for p in self.agents.values()]
        gammas  = [p.gamma      for p in self.agents.values()]
        wealth  = [p.wealth     for p in self.agents.values()]
        thrs    = [p.threshold  for p in self.agents.values()]
        speeds  = [p.belief.update_speed  for p in self.agents.values()]
        anchors = [p.belief.anchoring_bias for p in self.agents.values()]
        las     = [p.belief.loss_aversion  for p in self.agents.values()]

        n_above = sum(1 for v in vals if v > self.eq_price)
        n_below = sum(1 for v in vals if v < self.eq_price)

        return {
            "D":                self.diversity_score,
            "eq_price":         self.eq_price,
            "n_potential_buyers": n_above,
            "n_potential_sellers": n_below,
            "expected_price_mean":   float(np.mean(vals)),
            "expected_price_std":    float(np.std(vals)),
            "expected_price_range":  float(np.ptp(vals)),
            "gamma_mean":       float(np.mean(gammas)),
            "gamma_std":        float(np.std(gammas)),
            "wealth_gini":      _gini(wealth),
            "threshold_mean":   float(np.mean(thrs)),
            "threshold_std":    float(np.std(thrs)),
            "belief_speed_std": float(np.std(speeds)),
            "anchoring_mean":   float(np.mean(anchors)),
            "loss_aversion_mean": float(np.mean(las)),
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
            belief_cfg      = self.cfg.beliefs,
            env_cfg         = self.cfg.env,
            eq_price        = self._eq_price,
            seed            = seed,
        )

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
        Reset na nowy epizod (T=200 kroków) — ta sama populacja, nowe portfele.

        Continuous Trading: każdy epizod = 200 kroków.
        Między epizodami: portfele resetowane, wyceny dryfują (pamięć rynku).
        Cena rynkowa (ref_price) NIE resetuje się — ciągłość historii.
        """
        assert self.population is not None, "Wywołaj reset() przed reset_episode()"

        # Wyceny: drift + zakotwiczenie + szum informacyjny
        sigma_info = 0.01
        for p in self.population.agents.values():
            drift  = p.belief.update_speed * (self._ref_price - p.expected_price)
            anchor = p.belief_reversion    * (p.long_run_fair_price - p.expected_price)
            noise  = self.rng.normal(0, sigma_info)
            p.expected_price = float(np.clip(
                p.expected_price + drift + anchor + noise,
                self.cfg.env.p_min, self.cfg.env.p_max,
            ))
            p.belief.reset_dynamic(p.expected_price)

        # Reset rynku i portfeli
        self.order_book.reset()
        self._step        = 0
        self._done        = False
        self._actions_log = []
        self._price_window= []
        self._price_history = [self._ref_price]
        self._n_fills = 0
        self._n_position_closes = 0

        for p in self.population.agents.values():
            p.reset_position()

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
            "expected_price": params.expected_price,
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
        self._update_beliefs(self._ref_price)
        if realized != 0.0:
            self._realized_this_step[agent_id] = (
                self._realized_this_step.get(agent_id, 0.0) + realized
            )

        return {"agent_id": agent_id, "side": self.cfg.env.action_name(action_idx), "price": p_exec}

    def compute_step_rewards(
        self,
    ) -> Tuple[Dict[str, float], Dict[str, bool]]:
        """
        Oblicza nagrody na końcu kroku dla wszystkich agentów.

        Reward = realized_pnl_this_step + valuation_alignment_signal

        realized_pnl: niezerowa suma (gains from trade z heterogenicznych wycen)
        valuation_alignment: ciągły sygnał który uczy kiedy wycena agenta jest wiarygodna
          signal > 0 gdy agent ma pozycję zgodną z własną wyceną (long gdy val>price)
          Skala: mała (0.005) — nie dominuje nad realized ale daje gradient przy HOLD

        Order flow imbalance i drift aplikowane raz per krok (nie per agent).
        """
        if self.cfg.market.drift_enabled:
            self._apply_drift()
            self._update_beliefs(self._ref_price)

        # Nagrody
        rewards: Dict[str, float] = {}
        for aid, p in self.population.agents.items():
            # Realized PnL z zamkniętych pozycji w tym kroku
            realized = self._realized_this_step.get(aid, 0.0)

            exp_gap = p.expected_price - self._ref_price
            pos_norm = p.position / max(p.max_position, 1)
            alignment = float(
                np.clip(exp_gap / max(p.threshold, 0.001), -1.0, 1.0)
            ) * pos_norm * self.cfg.env.alignment_scale
            risk_penalty = (
                self.cfg.env.risk_penalty_kappa
                * p.risk_aversion
                * (pos_norm ** 2)
            )

            reward = realized + alignment - risk_penalty
            rewards[aid] = reward
            # episode_pnl śledzi tylko realized (nie alignment) — metryka artykułu
            self._episode_pnl[aid] = self._episode_pnl.get(aid, 0.0) + realized

        self._prev_price = self._ref_price
        self._realized_this_step = {}

        self._step += 1
        T    = self.cfg.env.episode_steps
        done = self._step >= T
        if done:
            if self.cfg.env.auto_liquidate_end:
                terminal = self._liquidate_terminal_positions()
                for aid, realized in terminal.items():
                    rewards[aid] = rewards.get(aid, 0.0) + realized
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

        expected_price nie bierze udziału w PnL. Zysk/strata wynika wyłącznie
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

    def _update_beliefs(self, price: float):
        """
        Aktualizuje oczekiwania (EMA) po każdej zmianie ref_price.

        expected_price jest motywem decyzji i shaping rewardu. PnL nadal
        pochodzi wyłącznie z cen wykonania i entry_price.
        """
        for agent in self.population.agents.values():
            agent.belief.observe_price(price)
            agent.expected_price = float(np.clip(
                (1.0 - agent.belief_reversion) * agent.belief.expected_price
                + agent.belief_reversion * agent.long_run_fair_price,
                self.cfg.env.p_min, self.cfg.env.p_max,
            ))

    def _apply_valuation_drift(self, avg_price: float):
        """
        Drift wycen fundamentalnych w kierunku ceny rynkowej.
        Wywoływany RAZ per krok (nie per transakcję) z uśrednioną ceną.

        Dotyczy WSZYSTKICH agentów (aktywnych i handlujących):
          - Cena transakcji jest informacją publiczną — każdy ją obserwuje
          - Poprzedni błąd: drift tylko dla aktywnych → agent cierpliwy tracił
            sygnał szybciej niż niecierpliwy (paradoks!)

        Skala driftu per krok:
          update_speed=0.15: val przesuwa się o ~15% różnicy do avg_price
          update_speed=0.50: val przesuwa się o ~50% różnicy
          Przy avg_price=0.48 i val=0.63: Δval = 0.15×(0.48-0.63) = -0.022 per krok

        Powiązanie z heterogenicznością:
          Niski update_speed = value investor — trzyma przekonania mimo rynku
          Wysoki update_speed = momentum trader — szybko konwerguje do rynku
          SARSA może nauczyć się wykorzystywać tę różnicę (handluj zanim stracisz sygnał)

        No-trade theorem przy D=0:
          Wszyscy val=eq=avg_price → drift: eq*(1-α)+eq*α = eq → bez zmian ✓
        """
        for agent in self.population.agents.values():
            alpha = agent.belief.update_speed
            agent.expected_price = float(np.clip(
                (1.0 - alpha) * agent.expected_price + alpha * avg_price,
                self.cfg.env.p_min, self.cfg.env.p_max
            ))

    def _apply_drift(self):
        """Dryf ceny równowagi w trakcie epizodu."""
        md     = self.cfg.market
        drift  = self.rng.normal(0, md.drift_magnitude * 0.1)
        if self.rng.random() < md.shock_probability:
            drift += self.rng.choice([-1, 1]) * md.shock_size
        self._eq_price  = float(np.clip(self._eq_price  + drift, 0.15, 0.85))
        self._ref_price = float(np.clip(
            self._ref_price + drift * 0.3,
            self.cfg.env.p_min,
            self.cfg.env.p_max,
        ))
        self._price_history.append(self._ref_price)

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
        15D wektor obserwacji — Continuous Trading (position model).

          [0]  expected_price    subiektywna oczekiwana/fair cena
          [1]  ref_price         ostatnia cena rynkowa
          [2]  value_signal      siła sygnału: 0=strong sell, 0.5=neutral, 1=strong buy
          [3]  last_trade_price  cena ostatniej transakcji
          [4]  price_volatility  std ostatnich 5 cen / eq_price
          [5]  price_trend       kierunek trendu (znorm.)
          [6]  position_norm     position / max_position ∈ [-1, +1]
          [7]  position_util     |position| / max_position ∈ [0, 1]
          [8]  time_remaining    (T - step) / T
          [9]  gamma             discount factor agenta
          [10] anchor_price_norm long_run_fair_price
          [11] threshold_norm    threshold / 0.40
          [12] expectation_drift |expected - long_run| / 0.5
          [14] ep_pnl_norm       kumulatywny P&L epizodu (znorm.)

        Usunięte względem starszych wariantów:
          cash_norm i avg_entry_norm — główna ścieżka śledzi pozycję i entry_price.
        """
        p    = self.population.agents[agent_id]
        T    = self.cfg.env.episode_steps
        MPOS = max(p.max_position, 1)   # per-agent (zależy od wealth)

        thr          = max(p.threshold, 0.001)
        value_signal = float(np.clip((p.long_run_fair_price - self._ref_price) / thr / 6.0 + 0.5, 0, 1))

        pw = self._price_window
        volatility = float(np.std(pw[-5:])) / max(self._eq_price, 0.01) if len(pw) >= 2 else 0.0

        pos_norm  = float(np.clip(p.position / MPOS, -1, 1))
        time_rem  = float(np.clip(1.0 - self._step / max(T, 1), 0, 1))
        val_drift = float(np.clip(abs(p.expected_price - p.long_run_fair_price) / 0.5, 0, 1))
        ep_pnl    = float(np.clip(self._episode_pnl.get(agent_id, 0.0) / 5.0, -1, 1))

        # Unrealized P&L: ile zyskałby agent gdyby zamknął teraz pozycję
        # Kluczowe dla decyzji CLOSE: "czy teraz jest dobry moment na wyjście?"
        if p.position != 0 and p.entry_price > 0:
            unrealized = float(np.clip(
                (self._ref_price - p.entry_price) * p.position / max(self.cfg.env.price_norm, 0.01),
                -1, 1
            ))
        else:
            unrealized = 0.0

        return np.array([
            p.expected_price,                                       # [0]  oczekiwana/fair cena
            self._ref_price,                                        # [1]  cena rynkowa
            value_signal,                                           # [2]  siła sygnału
            float(np.clip(p.belief.price_trend + 0.5, 0, 1)),     # [3]  trend ceny
            float(np.clip(volatility, 0, 1)),                      # [4]  zmienność
            pos_norm,                                               # [5]  pozycja ∈ [-1,+1]
            unrealized,                                             # [6]  niezrealizowany P&L ← nowe
            time_rem,                                               # [7]  czas do końca
            p.gamma,                                                # [8]  discount factor
            float(np.clip(p.risk_aversion / 3.0, 0, 1)),          # [9]  awersja do ryzyka
            float(np.clip(p.long_run_fair_price, 0, 1)),           # [10] kotwica oczekiwań
            float(np.clip(p.threshold / 0.40, 0, 1)),              # [11] próg decyzji
            val_drift,                                              # [12] odchylenie oczekiwań od kotwicy
            ep_pnl,                                                 # [13] kumulatywny P&L epizodu
            float(p.position != 0),                                 # [14] czy masz otwartą pozycję
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
                "expected_price":  p.expected_price,
                "valuation":       p.expected_price,
                "long_run_fair_price": p.long_run_fair_price,
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
                "update_speed":    p.belief.update_speed,
                "loss_aversion":   p.belief.loss_aversion,
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
        # obs[5] = position_norm ∈ [-1,+1]
        pos_norm = float(obs[5])
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
