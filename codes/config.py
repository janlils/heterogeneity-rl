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
    Reward = realized_pnl_this_step + MTM; złe otwarte pozycje rozlicza terminal liquidation.
    """
    n_agents:               int   = 50
    episode_steps:          int   = 200    # T: długość epizodu
    max_position:           int   = 1      # domyślne max |position|

    # Market maker / dealer execution
    use_market_maker:       bool  = True
    half_spread:            float = 0.0
    market_impact:          float = 0.0
    temp_impact:            float = 0.000
    perm_impact:            float = 0.0
    k_impact:               float = 0.03
    mtm_weight:             float = 0.3
    p_min:                  float = 0.05
    p_max:                  float = 0.95
    auto_liquidate_end:     bool  = True

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
        return 6   # [signal_i, pos_norm, unrealized, time_rem,
                   #  price_vs_start, trend_short]

    @classmethod
    def no_impact(cls) -> "EnvConfig":
        return cls(half_spread=0.0, temp_impact=0.0, perm_impact=0.0, k_impact=0.0)

    def action_name(self, idx: int) -> str:
        return {0: "HOLD", 1: "BUY", 2: "SELL"}.get(idx, f"?{idx}")


# ---------------------------------------------------------------------------
# Dynamika rynku
# ---------------------------------------------------------------------------

@dataclass
class MarketDynamics:
    """
    Docelowy proces rynku v1:
      - V_t trenduje i doświadcza rzadkich skoków
      - stres s_t przełącza rynek między ciszą i kryzysem
      - P_t jest tłumionym oscylatorem wokół V_t
      - wpływ agentów jest ograniczony przez k_impact * tanh(flow/N)
    """
    init_value:             float = 0.50
    init_mu:                float = 0.0
    init_stress:            float = 0.004
    init_momentum:          float = 0.0

    mu_persistence:         float = 0.99
    mu_innovation_weight:   float = 0.01
    mu_drift_mean:          float = 0.0002
    mu_drift_std:           float = 0.0004

    value_jump_prob:        float = 0.01
    value_jump_min:         float = 0.03
    value_jump_max:         float = 0.07
    value_noise_std:        float = 0.0007
    value_min:              float = 0.25
    value_max:              float = 0.75

    stress_reversion:       float = 0.94
    stress_anchor_weight:   float = 0.06
    stress_low:             float = 0.004
    crisis_prob:            float = 0.012
    crisis_stress_min:      float = 0.020
    crisis_stress_max:      float = 0.035

    psi:                    float = 0.55
    kappa:                  float = 0.16
    nu:                     float = 5.0
    kick_min:               float = 0.03
    kick_max:               float = 0.055

    @classmethod
    def stable(cls)    -> "MarketDynamics":
        return cls()

    @classmethod
    def random_eq(cls) -> "MarketDynamics":
        return cls()

    @classmethod
    def drifting(cls)  -> "MarketDynamics":
        return cls()


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
    sigma_macro:             float = 0.006   # skok V_t między epizodami
    sigma_P:                 float = 0.010   # normalizacja zmiany ceny do tanh
    drift_persistence:       float = 0.80
    sigma_fund:              float = 0.02    # szum fundamentalisty (trader_type=0)
    sigma_chart:             float = 0.15    # szum chartysty (trader_type=1)
    signal_scale:            float = 0.20    # normalizacja sygnału

# ---------------------------------------------------------------------------
# Heterogeniczność
# ---------------------------------------------------------------------------

@dataclass
class DiversityConfig:
    """
    Co D kontroluje w docelowym rdzeniu v1.
    """
    gamma_spread:        bool = True   # horyzonty czasowe
    sigma_spread:        bool = True   # poziom szumu prywatnego sygnału

    @classmethod
    def sentiment_only(cls) -> "DiversityConfig":
        """Kompatybilnosc wsteczna: tylko rozrzut sigma_i bez spreadu gamma."""
        return cls(gamma_spread=False, sigma_spread=True)

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
    n_step:         int   = 1       # liczba kroków do przodu dla n-step returns


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
        default_factory=lambda: ["ZI", "DeepSARSA", "PPO", "IPPO"]
    )
    n_agents_list:      List[int]   = field(
        default_factory=lambda: [20, 50, 100]
    )
    market_conditions:  List[str]   = field(
        default_factory=lambda: ["v1_market"]
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
            n_agents_list=[50],
            market_conditions=["v1_market"],
            n_seeds=3, n_train_episodes=200, n_eval_episodes=30,
        )

    @classmethod
    def conference_paper(cls) -> "ExpConfig":
        return cls(
            diversity_scores=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            algorithms=["ZI", "DeepSARSA", "PPO", "IPPO"],
            n_agents_list=[20, 50],
            market_conditions=["v1_market"],
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
            f"market=v1(psi={self.market.psi:.2f}, kappa={self.market.kappa:.2f}, "
            f"stress_low={self.market.stress_low:.3f})"
        )
