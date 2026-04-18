"""
config.py — HTM (Heterogeneous Trader Market)
==============================================
Centralna konfiguracja. Wszystkie parametry żyją tutaj.

Kluczowa zmiana względem G&S:
  Model spekulacyjny — brak stałych ról kupiec/sprzedawca.
  Każdy agent ma subiektywną oczekiwaną cenę/fair price aktywa.
  Rola (buy/sell/hold) wynika z porównania oczekiwania z ceną rynkową.
"""

from dataclasses import dataclass, field
from typing import List
from pathlib import Path

ROOT_DIR    = Path(__file__).resolve().parent.parent
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
    Parametry środowiska — Continuous Trading.

    Każdy agent handluje przez T=episode_steps kroków.
    Nikt nie 'wychodzi' po transakcji — agenci zarządzają portfolio.
    Reward = realized_pnl_this_step - risk_penalty - holding_cost.
    """
    n_agents:               int   = 20
    episode_steps:          int   = 200    # T: długość epizodu
    max_position:           int   = 3      # domyślne max |position| (może być nadpisane przez wealth)
    risk_aversion_base:     float = 1.0    # bazowa kara za otwartą pozycję

    # Market maker / dealer execution
    use_market_maker:       bool  = True
    half_spread:            float = 0.0008
    temp_impact:            float = 0.000
    perm_impact:            float = 0.0016
    p_min:                  float = 0.05
    p_max:                  float = 0.95
    # alignment_scale usunięty (zawsze był 0)
    risk_penalty_kappa:     float = 0.02
    auto_liquidate_end:     bool  = True

    # Kara za trzymanie zysku bez zamknięcia (rośnie pod koniec epizodu)
    holding_cost_kappa:      float = 0.03
    # Ostatnie X% epizodu to "strefa urgency" dla holding_cost
    holding_urgency_horizon: float = 0.30

    # deprecated fields removed

    # Indeksy akcji — 3 (usunięte limit orders dla prostoty)
    ACTION_HOLD:        int = 0
    ACTION_BUY_MARKET:  int = 1
    ACTION_SELL_MARKET: int = 2

    @property
    def n_actions(self) -> int:
        return 3   # HOLD / BUY / SELL

    @property
    def n_obs(self) -> int:
        return 7

    def action_name(self, idx: int) -> str:
        return {0: "HOLD", 1: "BUY", 2: "SELL"}.get(idx, f"?{idx}")


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
# Sentiment agentów
# ---------------------------------------------------------------------------

@dataclass
class SentimentConfig:
    """
    Parametry dynamiki sentimentu i dyfuzji wartości fundamentalnej V_t.
    """
    # Dynamika V_t
    sigma_intra:             float = 0.003   # dryft V_t per krok (wewnątrz epizodu)
    sigma_macro:             float = 0.015   # skok V_t między epizodami
    p_info:                  float = 0.04    # prawdopodobieństwo sygnału info per krok
    sigma_news:              float = 0.15    # szum prywatnej interpretacji newsa
    sigma_P:                 float = 0.003   # normalizacja zmiany ceny do tanh

    # Centra parametrów behawioralnych (przy D=0 wszyscy mają te wartości)
    alpha_center:            float = 0.08    # momentum: waga sygnału cenowego
    beta_center:             float = 0.06    # mean reversion: powrót do neutralu
    news_sensitivity_center: float = 0.12   # waga sygnału informacyjnego z V_t

    # Zakresy przy D=1 (losowane jednostajnie: center ± half_range)
    alpha_spread:            float = 0.22    # → zakres [0.03, 0.25] przy D=1
    beta_spread:             float = 0.12    # → zakres [0.03, 0.15] przy D=1
    news_sensitivity_spread: float = 0.35   # → zakres [0.05, 0.40] przy D=1


# BeliefConfig usunięty — zastąpiony przez SentimentConfig.


# ---------------------------------------------------------------------------
# Heterogeniczność
# ---------------------------------------------------------------------------

@dataclass
class DiversityConfig:
    """
    Co D kontroluje. Każdy wymiar można włączyć/wyłączyć osobno
    (eksperymenty ablacyjne: który wymiar heterogeniczności jest kluczowy).
    """
    sentiment_spread:    bool = True   # rozrzut początkowego sentimentu agentów
    threshold_spread:    bool = True   # różne progi decyzji o handlu
    gamma_spread:        bool = True   # horyzonty czasowe
    wealth_spread:       bool = True   # majątek (Pareto)
    risk_aversion_spread:bool = True   # awersja do ryzyka pozycji
    behavioral_spread:   bool = True   # parametry α_i, β_i, news_sensitivity

    @classmethod
    def sentiment_only(cls) -> "DiversityConfig":
        """Tylko sentiment — najczystszy test modelu spekulacyjnego."""
        return cls(threshold_spread=False, gamma_spread=False,
                   wealth_spread=False, behavioral_spread=False)

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
# PPO
# ---------------------------------------------------------------------------

@dataclass
class PPOConfig:
    """Hiperparametry shared-policy PPO actor-critic."""
    hidden_size: int = 64
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.95
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    minibatch_size: int = 256
    rollout_episodes: int = 5
    normalize_advantages: bool = True
    device: str = "cpu"
    use_agent_id_features: bool = False


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
    base_seed:          int  = 45

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
    sentiment: SentimentConfig = field(default_factory=SentimentConfig)
    diversity: DiversityConfig = field(default_factory=DiversityConfig)
    sarsa:     DeepSARSAConfig = field(default_factory=DeepSARSAConfig)
    ppo:       PPOConfig       = field(default_factory=PPOConfig)
    log:       LogConfig       = field(default_factory=LogConfig)
    exp:       ExpConfig       = field(default_factory=ExpConfig)

    def summary(self) -> str:
        return (
            f"HTM-Speculative | N={self.env.n_agents} | "
            f"actions={self.env.n_actions} | "
            f"episode_steps={self.env.episode_steps} | "
            f"eq={self.market.eq_center}±{self.market.eq_spread}"
            f"{'+drift' if self.market.drift_enabled else ''}"
        )
