"""
tests/test_env.py
=================
Testy jednostkowe dla środowiska HTM.

Uruchomienie:
    cd htm_project
    python -m pytest tests/test_env.py -v

Lub bez pytest:
    python tests/test_env.py

Co testujemy i dlaczego:
  1. Order book – czy mechanizm matchowania działa poprawnie
  2. Unit demand – czy agent nie może handlować dwa razy
  3. Surplus – czy efektywność jest poprawnie liczona
  4. Obserwacja – czy wektor ma właściwy kształt i wartości w [0,1]
  5. Reset – czy środowisko czyści się poprawnie między epizodami
  6. Diversity – czy D=0 daje identycznych agentów
  7. Baseline – czy ZI efficiency mieści się w oczekiwanym przedziale
  8. Beliefs – czy przekonania aktualizują się po transakcji
"""

import sys
import logging
from pathlib import Path

import numpy as np

# Dodaj katalog główny projektu do ścieżki
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import HTMConfig, LogConfig, EnvConfig
from envs.double_auction import (
    DoubleAuction,
    OrderBook,
    Order,
    AgentPopulation,
    ZeroIntelligenceAgent,
    BeliefState,
    run_zi_baseline,
)

# Wyłącz logi podczas testów – nie chcemy zaśmiecać outputu
logging.getLogger("htm.auction").setLevel(logging.ERROR)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_cfg(**kwargs) -> HTMConfig:
    """Tworzy konfigurację testową z wyłączonym logowaniem."""
    return HTMConfig(
        log=LogConfig(
            level="ERROR",
            save_to_file=False,
            save_plots=False,
            show_plots=False,
        ),
        **kwargs,
    )


def make_da(n_buyers=5, n_sellers=5, **kwargs) -> DoubleAuction:
    """Tworzy małe środowisko do testów."""
    cfg = make_cfg(env=EnvConfig(n_buyers=n_buyers, n_sellers=n_sellers))
    return DoubleAuction(cfg, seed=42)


# ---------------------------------------------------------------------------
# Test 1: Order book – podstawowy mechanizm matchowania
# ---------------------------------------------------------------------------

class TestOrderBook:

    def test_bid_matches_ask_when_bid_ge_ask(self):
        """
        Transakcja powinna nastąpić gdy bid >= ask.
        To jest serce mechanizmu aukcji – jeśli to nie działa, nic nie działa.
        """
        book = OrderBook()

        # Sprzedawca składa ask 0.40
        ask = Order(agent_id="seller_0", order_type="ask", price=0.40)
        trade = book.submit(ask)
        assert trade is None, "Ask bez matching bida nie powinien dać transakcji"
        assert len(book.asks) == 1, "Ask powinien trafić do kolejki"

        # Kupiec składa bid 0.60 – powinno matchować z askiem 0.40
        bid = Order(agent_id="buyer_0", order_type="bid", price=0.60)
        trade = book.submit(bid)

        assert trade is not None, "bid=0.60 >= ask=0.40 – powinna być transakcja"
        assert trade.buyer_id == "buyer_0"
        assert trade.seller_id == "seller_0"
        assert trade.price == 0.50, f"Cena powinna być średnią: (0.60+0.40)/2=0.50, got {trade.price}"
        print("  OK: bid >= ask → transakcja po cenie średniej")

    def test_bid_does_not_match_when_bid_lt_ask(self):
        """
        Brak transakcji gdy bid < ask (spread jest ujemny).
        Obie oferty trafiają do kolejki i czekają.
        """
        book = OrderBook()

        ask = Order(agent_id="seller_0", order_type="ask", price=0.70)
        book.submit(ask)

        bid = Order(agent_id="buyer_0", order_type="bid", price=0.30)
        trade = book.submit(bid)

        assert trade is None, "bid=0.30 < ask=0.70 – nie powinno być transakcji"
        assert len(book.bids) == 1
        assert len(book.asks) == 1
        print("  OK: bid < ask → brak transakcji, obie oferty w kolejce")

    def test_price_is_midpoint(self):
        """
        Cena transakcyjna = (bid + ask) / 2.
        To jest midpoint rule – standard w CDA.
        """
        book  = OrderBook()
        cases = [(0.80, 0.60), (1.00, 0.00), (0.55, 0.45)]

        for bid_price, ask_price in cases:
            book.reset()
            book.submit(Order("seller", "ask", ask_price))
            trade = book.submit(Order("buyer", "bid", bid_price))

            expected = (bid_price + ask_price) / 2
            assert trade is not None
            assert abs(trade.price - expected) < 1e-9, (
                f"bid={bid_price}, ask={ask_price}: "
                f"expected {expected}, got {trade.price}"
            )

        print("  OK: cena transakcyjna = (bid + ask) / 2")

    def test_price_time_priority(self):
        """
        Przy wielu ofertach w kolejce, pierwsza lepsza cenowo (lub czasowo) jest matchowana.
        Tutaj: dwa aski, niższy powinien być matchowany pierwszym bidem.
        """
        book = OrderBook()

        # Dwa aski: 0.50 i 0.40 – ask 0.40 powinien być na górze kolejki
        book.submit(Order("seller_A", "ask", 0.50))
        book.submit(Order("seller_B", "ask", 0.40))

        assert book.best_ask == 0.40, "Najlepszy ask powinien być 0.40 (niższy)"

        # Bid 0.60 powinien matchować z askiem 0.40 (lepszym dla kupca)
        trade = book.submit(Order("buyer_0", "bid", 0.60))
        assert trade is not None
        assert trade.seller_id == "seller_B", (
            "Powinien matchować z seller_B (ask=0.40), nie seller_A (ask=0.50)"
        )
        print("  OK: price-time priority działa poprawnie")

    def test_remove_agent_clears_orders(self):
        """
        Po transakcji remove_agent() czyści wszystkie oczekujące oferty agenta.
        Ważne żeby agent który już handlował nie mógł matchować ponownie.
        """
        book = OrderBook()
        book.submit(Order("seller_0", "ask", 0.80))  # nie matchuje od razu
        book.submit(Order("seller_0", "ask", 0.85))  # kolejny order tego samego

        # Tylko jeden order per agent (nowy zastępuje stary)
        assert len(book.asks) == 1, "Agent może mieć tylko jeden aktywny order"

        book.remove_agent("seller_0")
        assert len(book.asks) == 0, "remove_agent() powinien wyczyścić wszystkie ordery"
        print("  OK: remove_agent() czyści ordery agenta")


# ---------------------------------------------------------------------------
# Test 2: Unit demand – każdy agent handluje max raz
# ---------------------------------------------------------------------------

class TestUnitDemand:

    def test_agent_cannot_trade_twice(self):
        """
        Po transakcji agent jest usuwany z aktywnych.
        Próba ponownego submit() powinna być ignorowana (z ostrzeżeniem).
        """
        da = make_da()
        da.reset(diversity_score=0.0, seed=42)

        # Znajdź kupca i sprzedawcę którzy na pewno się skrzyżują
        buyer_id  = "buyer_0"   # najwyższa wycena
        seller_id = "seller_0"  # najniższy koszt

        buyer_val  = da.population.agents[buyer_id].private_value
        seller_cost= da.population.agents[seller_id].private_value

        # Zmuś transakcję: buyer licytuje pełną wycenę, seller pyta o koszt
        da.submit(seller_id, seller_cost)   # ask = koszt
        trade = da.submit(buyer_id, buyer_val)   # bid = wycena > ask

        assert trade is not None, "Powinna być transakcja"
        assert buyer_id  in da._traded, "Kupiec powinien być w _traded"
        assert seller_id in da._traded, "Sprzedawca powinien być w _traded"
        assert buyer_id  not in da.active_agents
        assert seller_id not in da.active_agents

        # Próba ponownego handlu – powinno być None
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = da.submit(buyer_id, 0.99)
            assert result is None, "Drugi submit() powinien zwrócić None"
            assert len(w) == 1, "Powinno być ostrzeżenie"

        print("  OK: agent nie może handlować dwa razy w epizodzie")

    def test_active_agents_shrink_after_trade(self):
        """
        Po każdej transakcji liczba aktywnych agentów maleje o 2.
        """
        da = make_da(n_buyers=3, n_sellers=3)
        da.reset(diversity_score=0.0, seed=42)

        n_start = len(da.active_agents)
        assert n_start == 6, f"Powinno być 6 aktywnych agentów, got {n_start}"

        # Zmuś jedną transakcję
        da.submit("seller_0", 0.25)   # ask = 0.25
        trade = da.submit("buyer_0", 0.75)  # bid = 0.75 > ask

        if trade is not None:
            assert len(da.active_agents) == n_start - 2, (
                "Po transakcji powinno być o 2 mniej aktywnych agentów"
            )
            print("  OK: liczba aktywnych agentów maleje o 2 po transakcji")
        else:
            print("  SKIP: transakcja nie nastąpiła (parametry agentów)")


# ---------------------------------------------------------------------------
# Test 3: Surplus i efektywność
# ---------------------------------------------------------------------------

class TestSurplusAndEfficiency:

    def test_buyer_surplus_is_valuation_minus_price(self):
        """
        Surplus kupca = wycena - cena transakcyjna.
        Nie może być ujemny (kupiec nigdy nie płaci więcej niż wycenia).
        """
        da = make_da()
        da.reset(diversity_score=0.0, seed=1)

        buyer_id  = "buyer_0"
        seller_id = "seller_0"
        buyer_val = da.population.agents[buyer_id].private_value  # np. 0.75

        # Wymusz transakcję
        da.submit(seller_id, 0.10)    # ask niski
        trade = da.submit(buyer_id, buyer_val)

        if trade is not None:
            expected_surplus = buyer_val - trade.price
            assert abs(trade.buyer_surplus - expected_surplus) < 1e-9, (
                f"Surplus kupca: expected {expected_surplus:.4f}, "
                f"got {trade.buyer_surplus:.4f}"
            )
            assert trade.buyer_surplus >= 0, "Surplus kupca nie może być ujemny"
            print(f"  OK: surplus kupca = {trade.buyer_surplus:.3f} "
                  f"(val={buyer_val:.3f} - price={trade.price:.3f})")

    def test_max_theoretical_surplus_with_known_values(self):
        """
        Dla znanych wycen i kosztów, max surplus jest policzalny ręcznie.
        Sprawdzamy czy population.max_theoretical_surplus() zgadza się z ręcznym obliczeniem.
        """
        da = make_da(n_buyers=3, n_sellers=3)
        da.reset(diversity_score=0.0, seed=42)

        # Przy D=0: kupcy mają wycenę 0.75, sprzedawcy koszt 0.25
        # Max surplus per para = 0.75 - 0.25 = 0.50
        # 3 pary → max = 3 × 0.50 = 1.50
        pop    = da.population
        max_s  = pop.max_theoretical_surplus()

        buyer_vals  = sorted([p.private_value for p in pop.buyers.values()], reverse=True)
        seller_costs= sorted([p.private_value for p in pop.sellers.values()])
        manual = sum(max(0, v-c) for v, c in zip(buyer_vals, seller_costs))

        assert abs(max_s - manual) < 1e-9, (
            f"max_theoretical_surplus: auto={max_s:.4f}, manual={manual:.4f}"
        )
        print(f"  OK: max_theoretical_surplus = {max_s:.4f} (ręcznie: {manual:.4f})")

    def test_efficiency_between_0_and_1(self):
        """
        Efektywność alokacyjna musi być w [0, 1] zawsze.
        Powyżej 1 oznaczałoby że osiągnęliśmy więcej niż teoretyczne maximum – niemożliwe.
        """
        da  = make_da()
        cfg = make_cfg()
        rng = np.random.default_rng(42)

        for _ in range(50):
            seed = int(rng.integers(0, 10_000))
            da.reset(diversity_score=rng.uniform(0, 1), seed=seed)

            zi = {
                aid: ZeroIntelligenceAgent(p, seed=seed+i)
                for i, (aid, p) in enumerate(da.population.agents.items())
            }

            step = 0
            while not da.done:
                active = da.active_agents
                if not active: break
                aid = active[step % len(active)]
                da.submit(aid, zi[aid].act(da.get_observation(aid)))
                step += 1

            m   = da.episode_metrics()
            eff = m["allocative_efficiency"]
            assert 0.0 <= eff <= 1.0, (
                f"Efficiency poza zakresem: {eff:.4f} (seed={seed})"
            )

        print("  OK: efficiency zawsze w [0, 1] dla 50 losowych epizodów")

    def test_zero_efficiency_impossible(self):
        """
        Przy ZI agentach efficiency nie może być 0 –
        nawet losowe oferty czasem się trafiają.
        Testujemy że po 100 epizodach średnia efficiency > 0.
        """
        cfg = make_cfg()
        r   = run_zi_baseline(cfg, diversity_score=0.5, n_episodes=100, seed=42)
        assert r["allocative_efficiency"]["mean"] > 0.0, (
            "Średnia efficiency ZI powinna być > 0"
        )
        print(f"  OK: mean efficiency ZI = {r['allocative_efficiency']['mean']:.3f} > 0")


# ---------------------------------------------------------------------------
# Test 4: Obserwacja (interfejs RL)
# ---------------------------------------------------------------------------

class TestObservation:

    def test_observation_shape(self):
        """
        Wektor obserwacji musi mieć dokładnie 12 wymiarów.
        Jeśli to zmienisz, wszystkie modele RL przestają działać.
        """
        da = make_da()
        da.reset(diversity_score=0.5, seed=42)

        for aid in da.active_agents:
            obs = da.get_observation(aid)
            assert obs.shape == (12,), (
                f"Obserwacja {aid}: expected (12,), got {obs.shape}"
            )
            assert obs.dtype == np.float32, (
                f"Dtype powinien być float32, got {obs.dtype}"
            )

        print("  OK: obserwacja ma kształt (12,) float32 dla każdego agenta")

    def test_most_observation_dims_in_0_1(self):
        """
        Większość wymiarów obserwacji powinna być w [0, 1].
        Wyjątek: price_trend (dim 9) i momentum (dim 11) mogą być ujemne.
        """
        da = make_da()
        da.reset(diversity_score=0.5, seed=42)

        # Wymiary które MUSZĄ być w [0, 1]
        bounded_dims = {
            0: "private_value",
            1: "last_price",
            2: "best_bid",
            3: "best_ask",
            5: "frac_traded",
            6: "gamma",
            7: "wealth_norm",
            8: "expected_price",
            10: "loss_aversion_norm",
        }

        for aid in da.active_agents:
            obs = da.get_observation(aid)
            for dim, name in bounded_dims.items():
                val = obs[dim]
                assert 0.0 <= val <= 1.0, (
                    f"Agent {aid}, dim {dim} ({name}): "
                    f"value={val:.4f} poza [0,1]"
                )

        print("  OK: wszystkie ograniczone wymiary obserwacji w [0, 1]")

    def test_private_value_in_observation(self):
        """
        Dim 0 obserwacji = prywatna wycena agenta.
        Sprawdzamy że agent widzi własną wartość.
        """
        da = make_da()
        da.reset(diversity_score=0.5, seed=7)

        for aid, params in da.population.agents.items():
            obs = da.get_observation(aid)
            assert abs(obs[0] - params.private_value) < 1e-6, (
                f"{aid}: obs[0]={obs[0]:.4f} != private_value={params.private_value:.4f}"
            )

        print("  OK: obs[0] == private_value dla każdego agenta")


# ---------------------------------------------------------------------------
# Test 5: Reset
# ---------------------------------------------------------------------------

class TestReset:

    def test_reset_clears_traded_set(self):
        """
        Po reset() żaden agent nie powinien być oznaczony jako pohandlowany.
        """
        da = make_da()
        da.reset(diversity_score=0.0, seed=42)

        # Uruchom jeden krok
        active = da.active_agents
        da.submit(active[0], 0.99)

        # Reset
        da.reset(diversity_score=0.0, seed=42)
        assert len(da._traded) == 0, "_traded powinien być pusty po reset()"
        assert len(da.active_agents) == len(da.population.agents)
        print("  OK: reset() czyści _traded i przywraca wszystkich agentów")

    def test_reset_clears_order_book(self):
        """
        Po reset() order book powinien być pusty.
        """
        da = make_da()
        da.reset(diversity_score=0.0, seed=42)

        # Złóż kilka ofert bez transakcji
        da.submit("buyer_0", 0.01)   # bardzo niski bid – nie matchuje
        da.submit("seller_0", 0.99)  # bardzo wysoki ask – nie matchuje

        assert len(da.order_book.bids) > 0 or len(da.order_book.asks) > 0

        da.reset(diversity_score=0.0, seed=42)

        assert len(da.order_book.bids) == 0, "Bids powinny być puste po reset()"
        assert len(da.order_book.asks) == 0, "Asks powinny być puste po reset()"
        assert da.order_book.last_price is None
        print("  OK: reset() czyści order book")

    def test_done_is_false_after_reset(self):
        """
        Po reset() środowisko nie powinno być w stanie done.
        """
        da = make_da(n_buyers=2, n_sellers=2)
        da.cfg.env.max_steps = 1  # jeden krok – szybko się kończy
        da.reset(diversity_score=0.0, seed=42)

        # Uruchom jeden krok żeby done = True
        active = da.active_agents
        da.submit(active[0], 0.5)
        # max_steps=1 → done powinno być True

        # Reset
        da.reset(diversity_score=0.0, seed=42)
        assert not da.done, "done powinno być False po reset()"
        print("  OK: done=False po reset()")


# ---------------------------------------------------------------------------
# Test 6: Diversity Score
# ---------------------------------------------------------------------------

class TestDiversity:

    def test_d0_gives_identical_valuations(self):
        """
        Przy D=0 wszyscy kupcy mają tę samą wycenę (0.75),
        wszyscy sprzedawcy ten sam koszt (0.25).
        """
        da = make_da(n_buyers=5, n_sellers=5)
        da.reset(diversity_score=0.0, seed=42)

        buyer_vals  = [p.private_value for p in da.population.buyers.values()]
        seller_costs= [p.private_value for p in da.population.sellers.values()]

        assert np.std(buyer_vals)   < 1e-9, (
            f"D=0: std wycen kupców powinno być 0, got {np.std(buyer_vals):.6f}"
        )
        assert np.std(seller_costs) < 1e-9, (
            f"D=0: std kosztów sprzedawców powinno być 0, got {np.std(seller_costs):.6f}"
        )
        print(f"  OK: D=0 → wszyscy kupcy val={buyer_vals[0]:.3f}, "
              f"sprzedawcy cost={seller_costs[0]:.3f}")

    def test_d0_gives_neutral_beliefs(self):
        """
        Przy D=0 wszyscy agenci mają neutralne przekonania:
        brak zakotwiczenia, brak paniki, brak awersji do strat.
        """
        da = make_da()
        da.reset(diversity_score=0.0, seed=42)

        for aid, params in da.population.agents.items():
            b = params.belief
            assert b.anchoring_bias == 0.0, f"{aid}: anchoring={b.anchoring_bias}"
            assert b.loss_aversion  == 1.0, f"{aid}: loss_aversion={b.loss_aversion}"
            assert b.panic_factor   == 0.0, f"{aid}: panic={b.panic_factor}"
            assert b.patience       == 0.0, f"{aid}: patience={b.patience}"

        print("  OK: D=0 → neutralne przekonania dla wszystkich agentów")

    def test_d1_increases_valuation_spread(self):
        """
        Przy D=1 rozrzut wycen powinien być znacznie większy niż przy D=0.
        """
        da = make_da(n_buyers=20, n_sellers=20)

        da.reset(diversity_score=0.0, seed=42)
        std_d0 = np.std([p.private_value for p in da.population.agents.values()])

        da.reset(diversity_score=1.0, seed=42)
        std_d1 = np.std([p.private_value for p in da.population.agents.values()])

        assert std_d1 > std_d0, (
            f"D=1 powinno mieć większy std niż D=0: {std_d1:.4f} vs {std_d0:.4f}"
        )
        print(f"  OK: valuation std: D=0={std_d0:.3f} → D=1={std_d1:.3f}")

    def test_d1_gives_diverse_beliefs(self):
        """
        Przy D=1 agenci powinni mieć różne przekonania.
        std parametrów przekonań powinno być > 0.
        """
        da = make_da(n_buyers=10, n_sellers=10)
        da.reset(diversity_score=1.0, seed=42)

        anchor_vals = [p.belief.anchoring_bias for p in da.population.agents.values()]
        loss_vals   = [p.belief.loss_aversion  for p in da.population.agents.values()]

        assert np.std(anchor_vals) > 0, "D=1: anchoring_bias powinien się różnić"
        assert np.std(loss_vals)   > 0, "D=1: loss_aversion powinien się różnić"
        print(f"  OK: D=1 → belief diversity: "
              f"anchor_std={np.std(anchor_vals):.3f}, "
              f"loss_std={np.std(loss_vals):.3f}")


# ---------------------------------------------------------------------------
# Test 7: ZI Baseline – walidacja środowiska
# ---------------------------------------------------------------------------

class TestZIBaseline:

    def test_efficiency_in_expected_range(self):
        """
        ZI efficiency przy dowolnym D powinno być w [0.60, 0.98].
        Poniżej 0.60: środowisko jest zepsute.
        Powyżej 0.98: za dużo kroków (model trywialny).
        """
        cfg = make_cfg()

        for d in [0.0, 0.5, 1.0]:
            r   = run_zi_baseline(cfg, diversity_score=d, n_episodes=200, seed=42)
            eff = r["allocative_efficiency"]["mean"]
            assert 0.60 <= eff <= 0.98, (
                f"D={d}: ZI efficiency={eff:.3f} poza oczekiwanym zakresem [0.60, 0.98]"
            )
            print(f"  OK: D={d:.1f} → ZI efficiency={eff:.3f} ∈ [0.60, 0.98]")

    def test_gini_increases_with_d(self):
        """
        Gini powinien rosnąć z D – większa heterogeniczność → większa nierówność.
        """
        cfg = make_cfg()

        ginis = {}
        for d in [0.0, 0.5, 1.0]:
            r = run_zi_baseline(cfg, diversity_score=d, n_episodes=200, seed=42)
            ginis[d] = r["gini_coefficient"]["mean"]

        assert ginis[1.0] > ginis[0.0], (
            f"Gini przy D=1 ({ginis[1.0]:.3f}) powinien być > Gini przy D=0 ({ginis[0.0]:.3f})"
        )
        print(f"  OK: Gini rośnie z D: "
              f"D=0→{ginis[0.0]:.3f}, D=0.5→{ginis[0.5]:.3f}, D=1→{ginis[1.0]:.3f}")

    def test_n_trades_positive(self):
        """
        Średnia liczba transakcji powinna być > 0.
        Jeśli jest 0, order book nie matchuje w ogóle.
        """
        cfg = make_cfg()
        r   = run_zi_baseline(cfg, diversity_score=0.5, n_episodes=100, seed=42)
        assert r["n_trades"]["mean"] > 0, "Średnia liczba transakcji powinna być > 0"
        print(f"  OK: mean n_trades = {r['n_trades']['mean']:.1f} > 0")


# ---------------------------------------------------------------------------
# Test 8: Beliefs – przekonania agentów
# ---------------------------------------------------------------------------

class TestBeliefs:

    def test_belief_updates_after_observed_price(self):
        """
        Po zaobserwowaniu ceny transakcyjnej, expected_price agenta
        powinna się zmienić w kierunku tej ceny.
        """
        belief = BeliefState(update_speed=0.5, anchoring_bias=0.0)

        initial_price = belief.expected_price  # 0.5 domyślnie
        new_price     = 0.80

        belief.observe_price(new_price)

        # Po pierwszej obserwacji: expected_price = new_price (anchor)
        assert belief.expected_price == new_price, (
            f"Pierwsza obserwacja powinna ustawić expected_price={new_price}, "
            f"got {belief.expected_price}"
        )
        assert belief.n_observations == 1
        print(f"  OK: pierwsza obserwacja ustawia expected_price={belief.expected_price:.3f}")

        # Po drugiej obserwacji: EMA z alpha=0.5
        new_price2 = 0.60
        belief.observe_price(new_price2)
        expected = 0.5 * 0.80 + 0.5 * 0.60  # = 0.70

        assert abs(belief.expected_price - expected) < 1e-6, (
            f"EMA: expected {expected:.4f}, got {belief.expected_price:.4f}"
        )
        print(f"  OK: EMA działa: {0.80:.2f} → obs({new_price2:.2f}) → {belief.expected_price:.3f}")

    def test_anchoring_pulls_toward_anchor(self):
        """
        Z anchoring_bias > 0, expected_price jest przyciągana w stronę anchor_price.
        Im wyższy anchoring_bias, tym słabsza aktualizacja.
        """
        # Agent mocno zakotwiczony (bias=0.9)
        anchored = BeliefState(update_speed=0.5, anchoring_bias=0.9)
        # Agent bez zakotwiczenia (bias=0.0)
        rational = BeliefState(update_speed=0.5, anchoring_bias=0.0)

        # Ustal anchor przez pierwszą obserwację
        anchored.observe_price(0.50)
        rational.observe_price(0.50)

        # Teraz obserwuj cenę daleko od anchora
        anchored.observe_price(0.90)
        rational.observe_price(0.90)

        # Zakotwiczony agent powinien się mniej poruszyć niż racjonalny
        assert anchored.expected_price < rational.expected_price, (
            f"Zakotwiczony ({anchored.expected_price:.3f}) powinien być < "
            f"racjonalny ({rational.expected_price:.3f})"
        )
        print(f"  OK: anchoring: zakotwiczony={anchored.expected_price:.3f} "
              f"< racjonalny={rational.expected_price:.3f}")

    def test_loss_aversion_penalizes_losses(self):
        """
        Z loss_aversion > 1, straty są wyceniane wyżej (boleśniej) niż zyski.
        """
        neutral = BeliefState(loss_aversion=1.0)
        biased  = BeliefState(loss_aversion=2.25)  # wartość Kahneman & Tversky

        gain = 0.10
        loss = -0.10

        # Zysk: obaj wyceniają tak samo
        assert neutral.subjective_surplus(gain) == biased.subjective_surplus(gain)

        # Strata: biased wycenia mocniej (bardziej ujemnie)
        assert biased.subjective_surplus(loss) < neutral.subjective_surplus(loss), (
            f"Loss aversion: biased={biased.subjective_surplus(loss):.3f} "
            f"powinien być < neutral={neutral.subjective_surplus(loss):.3f}"
        )
        print(f"  OK: loss_aversion=2.25: strata -0.10 → "
              f"subiektywna wartość={biased.subjective_surplus(loss):.3f} "
              f"(vs neutral={neutral.subjective_surplus(loss):.3f})")

    def test_all_agents_receive_belief_update_after_trade(self):
        """
        Po transakcji WSZYSCY aktywni agenci powinni zaktualizować przekonania.
        Cena transakcyjna jest informacją publiczną.
        """
        da = make_da(n_buyers=5, n_sellers=5)
        da.reset(diversity_score=1.0, seed=42)  # D=1 żeby belief miały efekt

        # Zapamiętaj stan przekonań przed transakcją
        obs_before = {
            aid: da.population.agents[aid].belief.n_observations
            for aid in da.active_agents
        }

        # Wymusz transakcję
        da.submit("seller_0", 0.01)   # bardzo niski ask
        trade = da.submit("buyer_0", 0.99)  # bardzo wysoki bid

        if trade is None:
            print("  SKIP: transakcja nie nastąpiła")
            return

        # Sprawdź że aktywni agenci zaktualizowali przekonania
        for aid in da.active_agents:
            obs_after = da.population.agents[aid].belief.n_observations
            assert obs_after > obs_before[aid], (
                f"{aid}: n_observations nie wzrosło po transakcji "
                f"({obs_before[aid]} → {obs_after})"
            )

        print(f"  OK: {len(da.active_agents)} aktywnych agentów "
              f"zaktualizowało przekonania po transakcji")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all_tests():
    """Uruchamia wszystkie testy bez pytest."""

    test_classes = [
        ("Order Book",        TestOrderBook),
        ("Unit Demand",       TestUnitDemand),
        ("Surplus & Efficiency", TestSurplusAndEfficiency),
        ("Obserwacja",        TestObservation),
        ("Reset",             TestReset),
        ("Diversity Score",   TestDiversity),
        ("ZI Baseline",       TestZIBaseline),
        ("Beliefs",           TestBeliefs),
    ]

    total_passed = 0
    total_failed = 0
    failures     = []

    print("=" * 60)
    print("HTM Environment – testy jednostkowe")
    print("=" * 60)

    for group_name, cls in test_classes:
        print(f"\n[ {group_name} ]")
        instance = cls()
        methods  = [m for m in dir(instance) if m.startswith("test_")]

        for method_name in methods:
            try:
                getattr(instance, method_name)()
                total_passed += 1
            except Exception as e:
                total_failed += 1
                failures.append((group_name, method_name, str(e)))
                print(f"  FAIL: {method_name}")
                print(f"    → {e}")

    print()
    print("=" * 60)
    print(f"Wyniki: {total_passed} passed, {total_failed} failed")

    if failures:
        print("\nNieudane testy:")
        for group, test, err in failures:
            print(f"  [{group}] {test}: {err}")
    else:
        print("Wszystkie testy przeszły.")

    print("=" * 60)
    return total_failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)