"""
config.py — HTM (Heterogeneous Trader Market)
==============================================
Centralna konfiguracja. Wszystkie parametry żyją tutaj.

Kluczowa zmiana względem G&S:
  Model spekulacyjny — brak stałych ról kupiec/sprzedawca.
  Każdy agent ma prywatną wycenę (valuation) aktywa.
  Rola (buy/sell/pass) wynika z porównania wyceny z ceną rynkową.
"""

from dataclasses import dataclass, field
from typing import List
from pathlib import Path

ROOT_DIR    = Path(__file__).parent
LOGS_DIR    = ROOT_DIR / "logs"
PLOTS_DIR   = ROOT_DIR / "plots"
RESULTS_DIR = ROOT_DIR / "results"

for _d in [LOGS_DIR, PLOTS_DIR, RESULTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Środowisko
# ---------------------------------------------------------------------------

@dataclass
class EnvConfig:
    """
    Parametry rynku spekulacyjnego.

    n_agents zastępuje n_buyers + n_sellers — brak stałych ról.
    max_steps = round_multiplier × n_agents (auto-skalowanie).

    n_aggression_levels: ile poziomów agresywności oferty.
      Łączna przestrzeń akcji = n_aggression_levels + 1 (PASS).
      Przykład: 10 poziomów + PASS = 11 akcji.

    trade_threshold_base: minimalna różnica |valuation - price|
      żeby agent w ogóle rozważał handel (przy D=0).
      Przy D>0 każdy agent ma swój własny próg.
    """
    n_agents:             int   = 20
    round_multiplier:     float = 2.0    # max_steps = round_multiplier × n_agents
    trade_threshold_base: float = 0.06  # min |val - price| żeby handlować
                                         # 0.08 = agent handluje tylko gdy cena
                                         # odbiega o min 8% od jego wyceny
    discovery_threshold:  float = 0.05  # próg do price discovery metric

    # ── Nowa przestrzeń akcji: limit orders ───────────────────────────────
    # Zamiast 10 poziomów agresywności: 4 typy zleceń + PASS = 5 akcji
    #
    # Interpretacja ekonomiczna:
    #   PASS        — nie handluj w tym kroku
    #   MARKET      — zlecenie rynkowe, wykonaj natychmiast po ref_price
    #   LIMIT_TIGHT — zlecenie z limitem ±TIGHT od ref_price, czekaj chwilę
    #   LIMIT_MED   — zlecenie z limitem ±MED, bardziej cierpliwy
    #   LIMIT_FAR   — zlecenie z limitem ±FAR, bardzo cierpliwy
    #
    # Połączenie z gamma: agent z niską gamma (niecierpliwy) wybiera MARKET,
    # agent z wysoką gamma (cierpliwy) woli LIMIT_FAR i czeka na lepszą cenę.
    #
    # Połączenie z threshold: agent z wysokim threshold (konserwatywny)
    # handluje rzadko ale wtedy woli MARKET bo sygnał jest już silny.

    limit_tight_offset: float = 0.02   # ±2% od ref_price
    limit_med_offset:   float = 0.05   # ±5% od ref_price
    limit_far_offset:   float = 0.10   # ±10% od ref_price

    # Stałe indeksów akcji (używaj tych nazw w kodzie, nie magic numbers)
    ACTION_PASS:        int = 0
    ACTION_MARKET:      int = 1
    ACTION_LIMIT_TIGHT: int = 2
    ACTION_LIMIT_MED:   int = 3
    ACTION_LIMIT_FAR:   int = 4

    @property
    def n_agents_per_side(self) -> int:
        return self.n_agents // 2

    @property
    def max_steps(self) -> int:
        return int(self.round_multiplier * self.n_agents)

    @property
    def n_actions(self) -> int:
        """5 akcji: PASS + MARKET + 3 poziomy LIMIT."""
        return 5

    @property
    def pass_action(self) -> int:
        return self.ACTION_PASS

    @property
    def n_obs(self) -> int:
        """Wymiar wektora obserwacji (stały = 12)."""
        return 14

    def action_name(self, action_idx: int) -> str:
        """Nazwa akcji do logowania."""
        names = {0: "PASS", 1: "MARKET", 2: "LIMIT_TIGHT", 3: "LIMIT_MED", 4: "LIMIT_FAR"}
        return names.get(action_idx, f"?{action_idx}")


# ---------------------------------------------------------------------------
# Dynamika rynku
# ---------------------------------------------------------------------------

@dataclass
class MarketDynamics:
    """
    Trzy warunki środowiskowe dla artykułu:
      stable    — stałe eq=0.5 (baseline, jak G&S)
      random_eq — eq losowane per epizod z [eq_center ± eq_spread]
      drifting  — eq zmienia się w trakcie epizodu
    """
    eq_center:         float = 0.5
    eq_spread:         float = 0.0     # 0.0 = stable
    drift_enabled:     bool  = False
    drift_magnitude:   float = 0.015
    shock_probability: float = 0.04
    shock_size:        float = 0.04

    @classmethod
    def stable(cls)    -> "MarketDynamics":
        return cls(eq_spread=0.0,  drift_enabled=False)

    @classmethod
    def random_eq(cls) -> "MarketDynamics":
        return cls(eq_spread=0.18, drift_enabled=False)

    @classmethod
    def drifting(cls)  -> "MarketDynamics":
        return cls(eq_spread=0.18, drift_enabled=True)


# ---------------------------------------------------------------------------
# Przekonania agentów
# ---------------------------------------------------------------------------

@dataclass
class BeliefConfig:
    """
    Parametry behawioralne agentów (literatura z ekonomii behawioralnej).
    Przy D=0: wszyscy neutralni. Przy D=1: losowane z poniższych rozkładów.
    """
    update_speed_center:  float = 0.3    # EMA alpha
    update_speed_spread:  float = 0.25
    anchoring_spread:     float = 0.35   # zakotwiczenie do pierwszej ceny
    loss_aversion_spread: float = 1.2    # straty bolą X razy mocniej (Kahneman)
    panic_spread:         float = 0.12   # panika przy gwałtownym spadku
    patience_spread:      float = 0.18   # czekanie na lepszą cenę


# ---------------------------------------------------------------------------
# Heterogeniczność
# ---------------------------------------------------------------------------

@dataclass
class DiversityConfig:
    """
    Co D kontroluje. Każdy wymiar można włączyć/wyłączyć osobno
    (eksperymenty ablacyjne: który wymiar heterogeniczności jest kluczowy).
    """
    valuation_spread:    bool = True   # główny nowy wymiar: prywatne wyceny
    threshold_spread:    bool = True   # różne progi decyzji o handlu
    gamma_spread:        bool = True   # horyzonty czasowe
    wealth_spread:       bool = True   # majątek (Pareto)
    belief_spread:       bool = True   # parametry behawioralne

    @classmethod
    def valuations_only(cls) -> "DiversityConfig":
        """Tylko wyceny — najczystszy test modelu spekulacyjnego."""
        return cls(threshold_spread=False, gamma_spread=False,
                   wealth_spread=False, belief_spread=False)

    @classmethod
    def full(cls) -> "DiversityConfig":
        return cls()


# ---------------------------------------------------------------------------
# Deep SARSA
# ---------------------------------------------------------------------------

@dataclass
class DeepSARSAConfig:
    """
    Hiperparametry sieci neuronowej per agent.
    Sieć implementowana w czystym numpy (bez zależności GPU).
    """
    hidden_size:    int   = 64      # neurony w warstwie ukrytej
    lr:             float = 3e-3    # learning rate (wyższy niż Adam — numpy SGD)
    epsilon_start:  float = 0.35
    epsilon_end:    float = 0.05
    epsilon_decay:  float = 0.993
    grad_clip:      float = 1.0     # gradient clipping


# ---------------------------------------------------------------------------
# Logowanie
# ---------------------------------------------------------------------------

@dataclass
class LogConfig:
    level:          str  = "INFO"
    save_to_file:   bool = True
    log_filename:   str  = "experiment.log"
    save_plots:     bool = True
    show_plots:     bool = False
    use_wandb:      bool = False
    wandb_project:  str  = "htm-speculative"


# ---------------------------------------------------------------------------
# Eksperyment
# ---------------------------------------------------------------------------

@dataclass
class ExpConfig:
    """
    Grid eksperymentów: D × N × warunek_rynku × algorytm × seed.
    """
    diversity_scores:   List[float] = field(
        default_factory=lambda: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    )
    algorithms:         List[str]   = field(
        default_factory=lambda: ["ZI", "DeepSARSA", "PPO", "IPPO", "MAPPO"]
    )
    n_agents_list:      List[int]   = field(
        default_factory=lambda: [20, 50, 100]
    )
    market_conditions:  List[str]   = field(
        default_factory=lambda: ["stable", "random_eq", "drifting"]
    )
    n_seeds:            int  = 30
    n_train_episodes:   int  = 1000
    n_eval_episodes:    int  = 100
    base_seed:          int  = 42

    @classmethod
    def quick_test(cls) -> "ExpConfig":
        return cls(
            diversity_scores=[0.0, 0.5, 1.0],
            algorithms=["ZI", "DeepSARSA"],
            n_agents_list=[20],
            market_conditions=["stable"],
            n_seeds=3, n_train_episodes=200, n_eval_episodes=30,
        )

    @classmethod
    def conference_paper(cls) -> "ExpConfig":
        return cls(
            diversity_scores=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            algorithms=["ZI", "DeepSARSA", "PPO", "IPPO", "MAPPO"],
            n_agents_list=[20, 50],
            market_conditions=["stable", "random_eq"],
            n_seeds=30, n_train_episodes=1000, n_eval_episodes=100,
        )


# ---------------------------------------------------------------------------
# Główna konfiguracja
# ---------------------------------------------------------------------------

@dataclass
class HTMConfig:
    env:       EnvConfig       = field(default_factory=EnvConfig)
    market:    MarketDynamics  = field(default_factory=MarketDynamics.stable)
    beliefs:   BeliefConfig    = field(default_factory=BeliefConfig)
    diversity: DiversityConfig = field(default_factory=DiversityConfig)
    sarsa:     DeepSARSAConfig = field(default_factory=DeepSARSAConfig)
    log:       LogConfig       = field(default_factory=LogConfig)
    exp:       ExpConfig       = field(default_factory=ExpConfig)

    def summary(self) -> str:
        return (
            f"HTM-Speculative | N={self.env.n_agents} | "
            f"actions={self.env.n_actions} (PASS+MARKET+3×LIMIT) | "
            f"max_steps={self.env.max_steps} | "
            f"eq={self.market.eq_center}±{self.market.eq_spread}"
            f"{'+drift' if self.market.drift_enabled else ''}"
        )