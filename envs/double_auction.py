"""
envs/double_auction.py — Model spekulacyjny HTM
================================================
Kluczowa zmiana względem modelu G&S:

  STARY MODEL (G&S — rynek dóbr):
    - Stałe role: kupiec ma wycenę dobra, sprzedawca ma koszt
    - Problem: efficiency = 1.0 zawsze, brak prawdziwego zadania uczenia

  NOWY MODEL (spekulacyjny — rynek finansowy):
    - Brak stałych ról. Każdy agent ma prywatną wycenę aktywa (valuation)
    - Jeśli valuation > cena_rynkowa → sygnał KUP (aktywo tanie)
    - Jeśli valuation < cena_rynkowa → sygnał SPRZEDAJ (aktywo drogie)
    - Jeśli różnica < threshold → PASS (nie warto handlować)
    - Handel wynika z RÓŻNICY PRZEKONAŃ (heterogeneous beliefs literature)

  UZASADNIENIE EKONOMICZNE:
    - De Long et al. (1990): handel wynika z różnych wycen fundamentalnych
    - Milgrom-Stokey: przy D=0 (wszyscy identyczni) → brak transakcji — POPRAWNIE
    - Santa Fe Artificial Stock Market — klasyczny benchmark ABM

  PRZESTRZEŃ AKCJI (nowa):
    Akcja = agresywność oferty [0, N-1] + PASS [N]
    - Kupujący (val > price): wyższy index → wyższa oferta → łatwiej kupić
    - Sprzedający (val < price): wyższy index → niższa żądana cena → łatwiej sprzedać
    - PASS: agent nie handluje w tym kroku
    Ceny są mapowane RELATYWNIE do własnej wyceny (nie absolutnie).

Klasy:
  BeliefState      — przekonania i biasy behawioralne agenta
  AgentParams      — profil agenta (valuation, threshold, gamma, wealth, belief)
  AgentPopulation  — generuje N agentów bez stałych ról
  OrderBook        — mechanizm matchowania bid/ask
  DoubleAuction    — główne środowisko z parallel_step()
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

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import HTMConfig, EnvConfig, BeliefConfig, DiversityConfig, MarketDynamics

_log = logging.getLogger("htm.auction")


# ===========================================================================
# BeliefState — przekonania i biasy behawioralne
# ===========================================================================

@dataclass
class BeliefState:
    """
    Wewnętrzny model przekonań agenta o rynku.

    Stałe cechy (losowane przy tworzeniu, nie zmieniają się w epizodzie):
      update_speed   — jak szybko agent zmienia oczekiwania (alpha w EMA)
      anchoring_bias — zakotwiczenie do pierwszej zaobserwowanej ceny
      loss_aversion  — straty bolą X razy mocniej niż zyski (Kahneman 1979)
      panic_factor   — przy gwałtownym spadku sprzedaje poniżej wyceny
      patience       — przy spadku wstrzymuje zakupy (czeka na niższą cenę)
      gamma          — indywidualny discount factor [0.5, 0.99]

    Stan dynamiczny (aktualizowany po każdej zaobserwowanej transakcji):
      expected_price  — gdzie agent spodziewa się ceny (EMA z obserwacji)
      price_trend     — szacowany kierunek zmiany (ostatnia delta EMA)
      anchor_price    — pierwsza zaobserwowana cena (zakotwiczenie)
      n_observations  — ile transakcji widział (rośnie pewność)
    """
    # Stałe cechy agenta
    update_speed:   float = 0.30
    anchoring_bias: float = 0.00
    loss_aversion:  float = 1.00
    panic_factor:   float = 0.00
    patience:       float = 0.00
    gamma:          float = 0.90

    # Stan dynamiczny
    expected_price: float = 0.50
    price_trend:    float = 0.00
    anchor_price:   float = 0.50
    n_observations: int   = 0

    def reset_dynamic(self, ref_price: float = 0.50) -> None:
        """Reset stanu dynamicznego przed nowym epizodem."""
        self.expected_price = ref_price
        self.price_trend    = 0.00
        self.anchor_price   = ref_price
        self.n_observations = 0

    def observe_price(self, new_price: float) -> None:
        """EMA aktualizacja oczekiwań po zaobserwowaniu transakcji."""
        if self.n_observations == 0:
            self.anchor_price   = new_price
            self.expected_price = new_price
            self.price_trend    = 0.0
        else:
            old               = self.expected_price
            self.price_trend  = new_price - old
            self.expected_price = (
                (1 - self.update_speed) * old + self.update_speed * new_price
            )
            # Zakotwiczenie: przyciągnij z powrotem do pierwszej ceny
            if self.anchoring_bias > 0:
                self.expected_price = (
                    (1 - self.anchoring_bias) * self.expected_price
                    + self.anchoring_bias * self.anchor_price
                )
        self.n_observations += 1

    def adjusted_aggressiveness(
        self, base_aggressiveness: float, is_buyer: bool
    ) -> float:
        """
        Korekta agresywności oferty na podstawie biasów behawioralnych.

        KUPUJĄCY przy spadającym trendzie (patience > 0):
          → zmniejsz agresywność (poczekaj na niższą cenę)

        SPRZEDAJĄCY przy gwałtownym spadku (panic_factor > 0):
          → zwiększ agresywność (sprzedaj szybciej zanim cena spadnie dalej)
        """
        if is_buyer and self.price_trend < 0 and self.patience > 0:
            reduction = abs(self.price_trend) * self.patience * 2
            return float(np.clip(base_aggressiveness - reduction, 0.0, 1.0))
        if not is_buyer and self.price_trend < -0.04 and self.panic_factor > 0:
            boost = abs(self.price_trend) * self.panic_factor * 2
            return float(np.clip(base_aggressiveness + boost, 0.0, 1.0))
        return base_aggressiveness

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
      - 'valuation' = prywatna wycena fundamentalna aktywa
      - 'threshold' = minimalna różnica |val - price| żeby handlować
      - Rola wynika dynamicznie: val > price → kup, val < price → sprzedaj

    Uzasadnienie ekonomiczne:
      Agent kupuje gdy uważa że aktywo jest tanie (val > price).
      Agent sprzedaje gdy uważa że aktywo jest drogie (val < price).
      Taki handel wynika z heterogenicznych przekonań — klasyczny wynik
      De Long et al. (1990) i Scheinkman & Xiong (2003).
    """
    agent_id:   str
    valuation:  float        # prywatna wycena fundamentalna [0, 1]
    threshold:  float        # min |val - price| do handlu
    gamma:      float = 0.90
    wealth:     float = 1.00 # ograniczenie budżetowe (aktywne!)
    belief:     BeliefState  = field(default_factory=BeliefState)

    def trade_signal(self, ref_price: float) -> str:
        """
        Wyznacza sygnał handlowy na podstawie własnej wyceny i ceny rynkowej.

        Returns: 'buy', 'sell', lub 'none' (różnica za mała)
        """
        diff = self.valuation - ref_price
        if diff > self.threshold:
            return "buy"
        elif diff < -self.threshold:
            return "sell"
        return "none"

    def max_affordable_bid(self) -> float:
        """Kupiec nie może licytować ponad min(valuation, wealth)."""
        return float(np.clip(min(self.valuation, self.wealth), 0.01, 0.99))

    def __repr__(self) -> str:
        return (
            f"Agent({self.agent_id}, val={self.valuation:.3f}, "
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
            belief = self._sample_belief(d, cfg, gamma, eq)

            self.agents[aid] = AgentParams(
                agent_id  = aid,
                valuation = val,
                threshold = self._sample_threshold(d, cfg),
                gamma     = gamma,
                wealth    = self._sample_wealth(d, cfg),
                belief    = belief,
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

    def _sample_wealth(self, d, cfg) -> float:
        if not cfg.wealth_spread or d < 1e-6:
            return 1.0
        # Pareto(1.5): realistyczna nierówność majątku
        raw = float(self.rng.pareto(1.5) + 1.0)
        return float(np.clip(raw * d + 1.0 * (1.0 - d), 0.05, 20.0))

    def _sample_belief(self, d, cfg, gamma, eq) -> BeliefState:
        bc = self.belief_cfg
        if not cfg.belief_spread or d < 1e-6:
            return BeliefState(
                update_speed=bc.update_speed_center, anchoring_bias=0.0,
                loss_aversion=1.0, panic_factor=0.0, patience=0.0,
                gamma=gamma, expected_price=eq, anchor_price=eq,
            )

        def clamp(x, lo, hi): return float(np.clip(x, lo, hi))

        return BeliefState(
            update_speed=clamp(
                self.rng.normal(bc.update_speed_center, bc.update_speed_spread * d),
                0.05, 0.95
            ),
            anchoring_bias=clamp(self.rng.uniform(0, bc.anchoring_spread * d), 0, 1),
            loss_aversion=clamp(
                self.rng.uniform(1.0, 1.0 + bc.loss_aversion_spread * d), 1.0, 3.0
            ),
            panic_factor=clamp(self.rng.uniform(0, bc.panic_spread * d), 0, 0.5),
            patience=clamp(self.rng.uniform(0, bc.patience_spread * d), 0, 0.5),
            gamma=gamma,
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
            "valuation_mean":   float(np.mean(vals)),
            "valuation_std":    float(np.std(vals)),
            "valuation_range":  float(np.ptp(vals)),
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
    valuation:  float  # prywatna wycena agenta (do obliczenia surplusa)
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
      reset(D) → [parallel_step(actions) × max_steps] → episode_metrics()

    Przestrzeń akcji (dla każdego agenta):
      [0, N-1] = agresywność oferty (kierunek wyznaczany przez val vs price)
      [N]      = PASS — nie handluj w tym kroku

    Obserwacja (12D):
      [0]  valuation           własna wycena aktywa
      [1]  ref_price           aktualna cena referencyjna
      [2]  value_signal        (val - price + 0.5), clip [0,1] — siła sygnału
      [3]  best_bid            najlepsza oferta kupna
      [4]  best_ask            najlepsza oferta sprzedaży
      [5]  spread              spread (lub 1.0 jeśli brak)
      [6]  frac_traded         ułamek agentów którzy już handlowali
      [7]  gamma               własny discount factor
      [8]  wealth_norm         majątek (znormalizowany)
      [9]  expected_price      oczekiwana cena wg przekonań
      [10] price_trend         szacowany trend (clip [-1,1] → [0,1])
      [11] price_momentum      momentum z ostatnich transakcji

    Reward:
      Czysty surplus z transakcji. BEZ gamma^step (niszczyło uczenie).
      Subiektywna wartość przez loss_aversion (jeśli strata to boli mocniej).
      Nagroda za PASS = 0 (neutralna).
    """

    PASS_ACTION = None  # ustawiany z cfg.env.pass_action

    def __init__(self, cfg: HTMConfig, seed: Optional[int] = None):
        self.cfg        = cfg
        self.rng        = np.random.default_rng(seed)
        self.order_book = OrderBook()
        self.population: Optional[AgentPopulation] = None
        self._eq_price  = cfg.market.eq_center
        self._ref_price = cfg.market.eq_center  # aktualizowany po transakcjach

        self.PASS_ACTION = cfg.env.pass_action

        # Stan epizodu
        self._step:        int              = 0
        self._done:        bool             = False
        self._traded:      set              = set()
        self._surplus:     Dict[str, float] = {}
        self._rewards:     Dict[str, float] = {}
        self._actions_log: List[dict]       = []  # historia akcji do wykresów
        self._price_window: List[float]     = []

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
        self._traded      = set()
        self._surplus     = {aid: 0.0 for aid in self.population.agents}
        self._rewards     = {aid: 0.0 for aid in self.population.agents}
        self._actions_log = []
        self._price_window= []

        _log.debug(
            f"RESET | D={diversity_score:.2f} | eq={self._eq_price:.3f} | "
            f"N={self.cfg.env.n_agents}"
        )

        return {aid: self.get_observation(aid)
                for aid in self.population.agents}

    def reset_market_only(self) -> Dict[str, np.ndarray]:
        """
        Reset tylko rynku — order book, kroki, kto handlował.
        Populacja agentów zostaje bez zmian (te same wyceny, gamma, wealth).

        Używane w multi-round training (Opcja B + 1):
          - Jedna stała populacja przez cały trening danego (D, seed)
          - Każda runda = nowa sesja handlowa, ci sami agenci
          - Sieć uczy się strategii dla konkretnego agenta, nie uśrednionej

        Eq_price może się losowo zmienić (jeśli MarketDynamics.eq_spread > 0)
        — to jest właściwość rynku, nie agentów.
        """
        assert self.population is not None, "Wywołaj reset() przed reset_market_only()"

        # Opcjonalnie: nowa cena równowagi (właściwość rynku, nie agentów)
        md = self.cfg.market
        if md.eq_spread > 1e-6:
            self._eq_price = float(np.clip(
                self.rng.uniform(md.eq_center - md.eq_spread,
                                 md.eq_center + md.eq_spread),
                0.15, 0.85
            ))
        # else: eq_price zostaje z poprzedniej rundy

        self._ref_price = self._eq_price

        # Reset dynamicznego stanu przekonań każdego agenta
        # (oczekiwana cena, trend) — ale NIE parametrów (gamma, update_speed itp.)
        for p in self.population.agents.values():
            p.belief.reset_dynamic(self._eq_price)

        # Reset rynku
        self.order_book.reset()
        self._step        = 0
        self._done        = False
        self._traded      = set()
        self._surplus     = {aid: 0.0 for aid in self.population.agents}
        self._rewards     = {aid: 0.0 for aid in self.population.agents}
        self._actions_log = []
        self._price_window= []

        return {aid: self.get_observation(aid)
                for aid in self.population.agents}

    # -----------------------------------------------------------------------
    # Parallel step — główny interfejs dla RL
    # -----------------------------------------------------------------------

    def parallel_step(
        self,
        actions: Dict[str, int],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, dict]]:
        """
        Wszyscy aktywni agenci działają jednocześnie.

        Mapowanie akcji na oferty (NOWE — relatywne do własnej wyceny):

          Kupujący (valuation > ref_price + threshold):
            aggressiveness = action / (N-1)  ∈ [0, 1]
            bid = ref_price + (valuation - ref_price) × aggressiveness
            action=0: bid = ref_price  (niska oferta, może nie trafić)
            action=N-1: bid = valuation (maksymalnie ile da, gwarantuje trafienie)

          Sprzedający (valuation < ref_price - threshold):
            aggressiveness = action / (N-1)  ∈ [0, 1]
            ask = ref_price - (ref_price - valuation) × aggressiveness
            action=0: ask = ref_price  (wysoka żądana cena)
            action=N-1: ask = valuation (minimum co przyjmie, gwarantuje trafienie)

          PASS (action == pass_action):
            Brak oferty. Reward = 0. Agent dalej aktywny (może handlować później).
        """
        if self._done:
            all_agents = list(self.population.agents.keys())
            obs   = {aid: self.get_observation(aid) for aid in all_agents}
            rew   = {aid: 0.0 for aid in all_agents}
            dones = {aid: True for aid in all_agents}
            return obs, rew, dones, {}

        # Stałe akcji z konfiguracji
        ACT_PASS   = self.cfg.env.ACTION_PASS
        ACT_MARKET = self.cfg.env.ACTION_MARKET
        ACT_TIGHT  = self.cfg.env.ACTION_LIMIT_TIGHT
        ACT_MED    = self.cfg.env.ACTION_LIMIT_MED
        ACT_FAR    = self.cfg.env.ACTION_LIMIT_FAR

        # Offsety cen dla limit orderów
        TIGHT = self.cfg.env.limit_tight_offset
        MED   = self.cfg.env.limit_med_offset
        FAR   = self.cfg.env.limit_far_offset

        orders = []

        for agent_id, action_idx in actions.items():
            if agent_id in self._traded:
                continue
            if agent_id not in self.population.agents:
                continue

            params = self.population.agents[agent_id]
            signal = params.trade_signal(self._ref_price)

            # Zaloguj akcję
            self._actions_log.append({
                "step":      self._step,
                "agent_id":  agent_id,
                "action":    action_idx,
                "action_name": self.cfg.env.action_name(action_idx),
                "is_pass":   action_idx == ACT_PASS,
                "signal":    signal,
                "valuation": params.valuation,
                "ref_price": self._ref_price,
            })

            # PASS lub brak sygnału → nie handluj
            if action_idx == ACT_PASS or signal == "none":
                continue

            is_buyer = (signal == "buy")

            # ── Nowe mapowanie akcji: limit orders ──────────────────────
            #
            # MARKET:      wykonaj natychmiast po ref_price
            #              → gwarantuje transakcję jeśli jest kontrpartner
            #              → agent rezygnuje z price improvement
            #
            # LIMIT_TIGHT: postaw zlecenie ±TIGHT od ref_price
            #              → małe oczekiwanie na lepszą cenę
            #              → trafi jeśli rynek się lekko poruszy
            #
            # LIMIT_MED:   postaw zlecenie ±MED od ref_price
            #              → umiarkowane oczekiwanie
            #
            # LIMIT_FAR:   postaw zlecenie ±FAR od ref_price
            #              → duże oczekiwanie, agent bardzo cierpliwy
            #              → gwarantuje sporą marżę jeśli wykona
            #
            # Połączenie z gamma:
            #   niska gamma → niecierpliwy → MARKET (chce nagrody teraz)
            #   wysoka gamma → cierpliwy  → LIMIT_FAR (czeka na lepszą cenę)
            #
            # Połączenie z threshold:
            #   duży threshold → agent handluje rzadko, ale gdy handluje
            #   sygnał jest silny → stać go na LIMIT (i tak trafi)

            if is_buyer:
                if action_idx == ACT_MARKET:
                    price = self._ref_price             # kup natychmiast
                elif action_idx == ACT_TIGHT:
                    price = self._ref_price - TIGHT     # czekaj na -2%
                elif action_idx == ACT_MED:
                    price = self._ref_price - MED       # czekaj na -5%
                elif action_idx == ACT_FAR:
                    price = self._ref_price - FAR       # czekaj na -10%
                else:
                    continue

                # Ograniczenia: nie kupuj powyżej własnej wyceny i powyżej wealth
                price = min(price, params.max_affordable_bid())
                price = float(np.clip(price, 0.001, 0.999))

                orders.append(Order(
                    agent_id=agent_id, order_type="bid",
                    price=price, valuation=params.valuation,
                ))

            else:  # seller
                if action_idx == ACT_MARKET:
                    price = self._ref_price             # sprzedaj natychmiast
                elif action_idx == ACT_TIGHT:
                    price = self._ref_price + TIGHT     # czekaj na +2%
                elif action_idx == ACT_MED:
                    price = self._ref_price + MED       # czekaj na +5%
                elif action_idx == ACT_FAR:
                    price = self._ref_price + FAR       # czekaj na +10%
                else:
                    continue

                # Nie sprzedawaj poniżej własnej wyceny
                price = max(price, params.valuation)
                price = float(np.clip(price, 0.001, 0.999))

                orders.append(Order(
                    agent_id=agent_id, order_type="ask",
                    price=price, valuation=params.valuation,
                ))

        # Matchowanie wszystkich ofert jednocześnie
        trades = self.order_book.submit_batch(orders)

        # Rozliczenie transakcji
        for trade in trades:
            self._settle(trade)
            self._update_beliefs(trade.price)
            self._price_window.append(trade.price)
            if len(self._price_window) > 20:
                self._price_window.pop(0)
            # Aktualizuj cenę referencyjną
            self._ref_price = trade.price

        # Drift ceny równowagi (opcjonalnie)
        if self.cfg.market.drift_enabled:
            self._apply_drift()

        self._step += 1
        if self._step >= self.cfg.env.max_steps or not self.active_agents:
            self._done = True

        all_agents = list(self.population.agents.keys())
        obs   = {aid: self.get_observation(aid) for aid in all_agents}
        rew   = {aid: self._rewards.get(aid, 0.0) for aid in all_agents}
        dones = {aid: self._done for aid in all_agents}
        infos = {
            aid: {
                "traded":    aid in self._traded,
                "n_trades":  len(self.order_book.trade_history),
                "ref_price": self._ref_price,
            }
            for aid in all_agents
        }

        # Reset rewards po zebraniu (nie kumuluj)
        self._rewards = {aid: 0.0 for aid in all_agents}

        return obs, rew, dones, infos

    def _settle(self, trade: Trade):
        """Oblicza surplusy i usuwa obu z rynku."""
        buyer_p  = self.population.agents[trade.buyer_id]
        seller_p = self.population.agents[trade.seller_id]

        trade.buyer_surplus  = max(0.0, trade.buyer_val  - trade.price)
        trade.seller_surplus = max(0.0, trade.price - trade.seller_val)

        self._surplus[trade.buyer_id]  = trade.buyer_surplus
        self._surplus[trade.seller_id] = trade.seller_surplus

        # Reward = subiektywny surplus (loss_aversion jeśli strata)
        self._rewards[trade.buyer_id]  = buyer_p.belief.subjective_surplus(
            trade.buyer_surplus
        )
        self._rewards[trade.seller_id] = seller_p.belief.subjective_surplus(
            trade.seller_surplus
        )

        for aid in (trade.buyer_id, trade.seller_id):
            self._traded.add(aid)
            self.order_book.remove_agent(aid)

        _log.debug(
            f"  SETTLE {trade.buyer_id}(v={trade.buyer_val:.3f}) × "
            f"{trade.seller_id}(v={trade.seller_val:.3f}) @ {trade.price:.3f} | "
            f"surplus={trade.total_surplus:.4f}"
        )

    def _update_beliefs(self, price: float):
        """Aktualizuje przekonania wszystkich aktywnych agentów po transakcji."""
        for aid in self.active_agents:
            self.population.agents[aid].belief.observe_price(price)

    def _apply_drift(self):
        """Dryf ceny równowagi w trakcie epizodu."""
        md     = self.cfg.market
        drift  = self.rng.normal(0, md.drift_magnitude * 0.1)
        if self.rng.random() < md.shock_probability:
            drift += self.rng.choice([-1, 1]) * md.shock_size
        self._eq_price  = float(np.clip(self._eq_price  + drift, 0.15, 0.85))
        self._ref_price = float(np.clip(self._ref_price + drift * 0.3, 0.15, 0.85))

    # -----------------------------------------------------------------------
    # Sekwencyjne submit — dla ZI baseline
    # -----------------------------------------------------------------------

    def submit(self, agent_id: str, price: float, order_type: str) -> Optional[Trade]:
        """Sekwencyjne składanie oferty. Używane przez ZI baseline."""
        if self._done or agent_id in self._traded:
            return None

        params = self.population.agents[agent_id]
        # Ograniczenie budżetowe dla kupca
        if order_type == "bid":
            price = min(float(np.clip(price, 0.01, 0.99)), params.max_affordable_bid())
        else:
            price = float(np.clip(price, 0.01, 0.99))

        order = Order(
            agent_id=agent_id, order_type=order_type,
            price=price, valuation=params.valuation,
        )
        trade = self.order_book.submit(order)

        if trade is not None:
            self._settle(trade)
            self._update_beliefs(trade.price)
            self._price_window.append(trade.price)
            self._ref_price = trade.price

        self._step += 1
        if self._step >= self.cfg.env.max_steps or not self.active_agents:
            self._done = True

        return trade

    # -----------------------------------------------------------------------
    # Obserwacja
    # -----------------------------------------------------------------------

    def get_observation(self, agent_id: str) -> np.ndarray:
        """
        12D wektor obserwacji dla agenta.

        Kluczowy nowy wymiar:
          [2] value_signal = clip(valuation - ref_price + 0.5, 0, 1)
          Wartość > 0.5: sygnał kupna (val > price)
          Wartość < 0.5: sygnał sprzedaży (val < price)
          Wartość = 0.5: brak sygnału
        """
        p  = self.population.agents[agent_id]
        ob = self.order_book
        n  = len(self.population.agents)

        if len(self._price_window) >= 2:
            diffs    = np.diff(self._price_window[-5:])
            momentum = float(np.mean(diffs)) if len(diffs) > 0 else 0.0
        else:
            momentum = 0.0

        value_signal = float(np.clip(
            p.valuation - self._ref_price + 0.5, 0.0, 1.0
        ))

        return np.array([
            p.valuation,
            self._ref_price,
            value_signal,                          # NOWY: siła i kierunek sygnału
            ob.best_bid if ob.best_bid is not None else 0.0,
            ob.best_ask if ob.best_ask is not None else 1.0,
            ob.spread   if ob.spread   is not None else 1.0,
            len(self._traded) / max(n, 1),
            p.gamma,
            float(np.clip(p.wealth / 20.0, 0.0, 1.0)),
            p.belief.expected_price,
            float(np.clip(p.belief.price_trend + 0.5, 0.0, 1.0)),  # znorm.
            float(np.clip(momentum + 0.5, 0.0, 1.0)),               # znorm.
        ], dtype=np.float32)

    # -----------------------------------------------------------------------
    # Stan i właściwości
    # -----------------------------------------------------------------------

    @property
    def done(self) -> bool:
        return self._done

    @property
    def active_agents(self) -> List[str]:
        return [a for a in self.population.agents if a not in self._traded]

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
        """Kompletne metryki ekonomiczne po epizodzie."""
        achieved = sum(self._surplus.values())
        max_s    = self.population.max_theoretical_surplus()
        eff      = float(np.clip(achieved / max_s, 0, 1)) if max_s > 1e-9 else 0.0

        prices   = self.order_book.price_history
        mean_dev = (
            float(np.mean(np.abs(np.array(prices) - self._eq_price)))
            if prices else float(abs(self._ref_price - self._eq_price))
        )

        trades     = self.order_book.trade_history
        profitable = sum(1 for t in trades if t.is_profitable)

        # Histogram akcji
        n_buy  = sum(1 for a in self._actions_log if a["signal"] == "buy"  and not a["is_pass"])
        n_sell = sum(1 for a in self._actions_log if a["signal"] == "sell" and not a["is_pass"])
        n_pass = sum(1 for a in self._actions_log if a["is_pass"])
        n_none = sum(1 for a in self._actions_log if a["signal"] == "none" and not a["is_pass"])

        return {
            "allocative_efficiency":   eff,
            "gini_coefficient":        _gini(list(self._surplus.values())),
            "n_trades":                len(trades),
            "n_profitable_trades":     profitable,
            "price_discovery_steps":   _price_disc(
                prices, self._eq_price, self.cfg.env.discovery_threshold
            ),
            "mean_price_deviation":    mean_dev,
            "total_surplus":           float(achieved),
            "max_theoretical_surplus": float(max_s),
            "diversity_score":         self.population.diversity_score,
            "eq_price":                self._eq_price,
            "ref_price_final":         self._ref_price,
            "n_agents":                self.cfg.env.n_agents,
            "n_steps":                 self._step,
            "action_buy_frac":         n_buy  / max(len(self._actions_log), 1),
            "action_sell_frac":        n_sell / max(len(self._actions_log), 1),
            "action_pass_frac":        n_pass / max(len(self._actions_log), 1),
            "action_none_frac":        n_none / max(len(self._actions_log), 1),
        }

    def agent_metrics(self) -> Dict[str, dict]:
        return {
            aid: {
                "valuation":       p.valuation,
                "threshold":       p.threshold,
                "gamma":           p.gamma,
                "wealth":          p.wealth,
                "surplus":         self._surplus.get(aid, 0.0),
                "traded":          aid in self._traded,
                "signal":          p.trade_signal(self._eq_price),
                "update_speed":    p.belief.update_speed,
                "anchoring_bias":  p.belief.anchoring_bias,
                "loss_aversion":   p.belief.loss_aversion,
            }
            for aid, p in self.population.agents.items()
        }


# ===========================================================================
# Zero Intelligence baseline
# ===========================================================================

class ZeroIntelligenceAgent:
    """
    Baseline spekulacyjny — losowa agresywność, kierunek z sygnału.

    Różnica od G&S ZI:
    - G&S ZI: losuj cenę w [0, valuation] (tylko kupujący) lub [cost, 1]
    - Tu ZI: najpierw sprawdź sygnał (val vs price), potem losuj agresywność
    - Jeśli brak sygnału: PASS

    To jest WYŻSZY baseline niż G&S bo agent przynajmniej handluje w dobrym
    kierunku. Ale nie optymalizuje agresywności — to zadanie dla RL.
    """

    def __init__(self, params: AgentParams, env_cfg: EnvConfig,
                 seed: Optional[int] = None):
        self.params  = params
        self.env_cfg = env_cfg
        self.rng     = np.random.default_rng(seed)

    def act(self, obs: np.ndarray, ref_price: float) -> Tuple[int, str]:
        """
        Returns (action_idx, signal).
        action_idx = pass_action jeśli brak sygnału.
        """
        signal = self.params.trade_signal(ref_price)

        if signal == "none":
            return self.env_cfg.ACTION_PASS, "none"

        # ZI losuje jeden z 4 typów zleceń (bez PASS)
        # MARKET=1, LIMIT_TIGHT=2, LIMIT_MED=3, LIMIT_FAR=4
        action = int(self.rng.integers(
            self.env_cfg.ACTION_MARKET,
            self.env_cfg.ACTION_LIMIT_FAR + 1
        ))
        return action, signal


def run_zi_baseline(
    cfg:             HTMConfig,
    diversity_score: float = 0.5,
    n_episodes:      int   = 100,
    seed:            int   = 42,
) -> dict:
    """
    ZI baseline — walidacja środowiska.
    Cel: sensowna efficiency (> 0.4 dla D=0.5, wzrost z D).
    """
    rng = np.random.default_rng(seed)
    da  = DoubleAuction(cfg, seed=seed)
    all_m = []

    for ep in range(n_episodes):
        ep_seed = int(rng.integers(0, 100_000))
        da.reset(diversity_score=diversity_score, seed=ep_seed)

        zi = {
            aid: ZeroIntelligenceAgent(p, cfg.env, seed=ep_seed + i)
            for i, (aid, p) in enumerate(da.population.agents.items())
        }

        step = 0
        max_steps = cfg.env.max_steps * 3  # limit bezpieczeństwa
        while not da.done and step < max_steps:
            active = da.active_agents
            if not active: break
            aid = active[step % len(active)]
            obs = da.get_observation(aid)

            action_idx, signal = zi[aid].act(obs, da.ref_price)

            if action_idx != cfg.env.ACTION_PASS and signal != "none":
                p    = da.population.agents[aid]
                ref  = da.ref_price
                TIGHT = cfg.env.limit_tight_offset
                MED   = cfg.env.limit_med_offset
                FAR   = cfg.env.limit_far_offset

                if signal == "buy":
                    if action_idx == cfg.env.ACTION_MARKET:
                        price = ref
                    elif action_idx == cfg.env.ACTION_LIMIT_TIGHT:
                        price = ref - TIGHT
                    elif action_idx == cfg.env.ACTION_LIMIT_MED:
                        price = ref - MED
                    else:
                        price = ref - FAR
                    price = min(price, p.max_affordable_bid())
                    da.submit(aid, float(np.clip(price, 0.001, 0.999)), "bid")
                else:
                    if action_idx == cfg.env.ACTION_MARKET:
                        price = ref
                    elif action_idx == cfg.env.ACTION_LIMIT_TIGHT:
                        price = ref + TIGHT
                    elif action_idx == cfg.env.ACTION_LIMIT_MED:
                        price = ref + MED
                    else:
                        price = ref + FAR
                    price = max(price, p.valuation)
                    da.submit(aid, float(np.clip(price, 0.001, 0.999)), "ask")
            else:
                # PASS lub brak sygnału — inkrementuj ręcznie
                da._step += 1
                if da._step >= cfg.env.max_steps:
                    da._done = True

            step += 1

        all_m.append(da.episode_metrics())

    keys = [k for k, v in all_m[0].items() if isinstance(v, (int, float))]
    result = {
        k: {"mean": float(np.mean([m[k] for m in all_m])),
            "std":  float(np.std( [m[k] for m in all_m]))}
        for k in keys
    }
    eff = result["allocative_efficiency"]
    _log.info(
        f"ZI | D={diversity_score:.2f} | "
        f"eff={eff['mean']:.3f}±{eff['std']:.3f} | "
        f"trades={result['n_trades']['mean']:.1f}"
    )
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

    from config import HTMConfig, MarketDynamics, LogConfig, ExpConfig

    cfg = HTMConfig(
        market=MarketDynamics.random_eq(),
        log=LogConfig(level="INFO"),
        exp=ExpConfig.quick_test(),
    )
    print(cfg.summary())
    print()

    for d in [0.0, 0.3, 0.6, 1.0]:
        r = run_zi_baseline(cfg, diversity_score=d, n_episodes=25, seed=42)
        eff   = r["allocative_efficiency"]
        gini  = r["gini_coefficient"]
        trades= r["n_trades"]
        pfrac = r["action_pass_frac"]
        print(
            f"D={d:.1f} | eff={eff['mean']:.3f}±{eff['std']:.3f} | "
            f"gini={gini['mean']:.3f} | trades={trades['mean']:.1f} | "
            f"pass%={pfrac['mean']*100:.0f}%"
        )