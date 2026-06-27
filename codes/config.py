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
        return cls(half_spread=0.0, temp_impact=0.0, perm_impact=0.0)

    def action_name(self, idx: int) -> str:
        return {0: "HOLD", 1: "BUY", 2: "SELL"}.get(idx, f"?{idx}")


# ---------------------------------------------------------------------------
# Dynamika rynku
# ---------------------------------------------------------------------------

@dataclass
class MarketDynamics:
    """
    Proces rynku v2:
      - V_t jest egzogeniczne: gładkie trendy + newsy + rzadkie załamania,
      - sigma_t jest egzogenicznym reżimem zmienności typu GARCH + kryzysy,
      - P_t reaguje na flow agentów PRZED egzekucją i potem dryfuje luźno do V_t.
    """
    init_value:             float = 0.50
    init_mu:                float = 0.0
    init_variance:          float = 0.0001

    mu_persistence:         float = 0.99
    mu_innov_weight:        float = 0.01
    mu_drift_mean:          float = 0.0002
    mu_drift_std:           float = 0.0004

    value_noise_std:        float = 0.0005
    crash_prob:             float = 0.004
    crash_min:              float = 0.05
    crash_max:              float = 0.11
    news_prob:              float = 0.010
    news_min:               float = 0.02
    news_max:               float = 0.04
    value_min:              float = 0.20
    value_max:              float = 0.80

    garch_w:                float = 8e-6
    garch_a:                float = 0.10
    garch_b:                float = 0.86
    crisis_prob:            float = 0.010
    crisis_stress_min:      float = 0.020
    crisis_stress_max:      float = 0.035

    alpha:                  float = 0.04
    beta:                   float = 0.025
    impact_stress_gain:     float = 8.0
    nu:                     float = 5.0
    kick_min:               float = 0.02
    kick_max:               float = 0.05

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
    reward_scale:   float = 30.0    # skala rewardu tylko dla ścieżki uczenia Q


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
    Docelowa mała siatka benchmarku: D × algorytm × seed.
    """
    diversity_scores:   List[float] = field(
        default_factory=lambda: [0.0, 0.5, 1.0]
    )
    algorithms:         List[str]   = field(
        default_factory=lambda: ["ZI", "DeepSARSA", "PPO", "IPPO"]
    )
    n_agents_list:      List[int]   = field(
        default_factory=lambda: [50]
    )
    market_conditions:  List[str]   = field(
        default_factory=lambda: ["v1_market"]
    )
    n_seeds:            int  = 5
    n_train_episodes:   int  = 500
    n_eval_episodes:    int  = 30
    base_seed:          int  = 45

    @classmethod
    def quick_test(cls) -> "ExpConfig":
        return cls(
            diversity_scores=[0.0, 0.5, 1.0],
            algorithms=["ZI", "DeepSARSA"],
            n_agents_list=[50],
            market_conditions=["v1_market"],
            n_seeds=2, n_train_episodes=200, n_eval_episodes=30,
        )

    @classmethod
    def conference_paper(cls) -> "ExpConfig":
        return cls(
            diversity_scores=[0.0, 0.5, 1.0],
            algorithms=["ZI", "DeepSARSA", "PPO", "IPPO"],
            n_agents_list=[50],
            market_conditions=["v1_market"],
            n_seeds=5, n_train_episodes=500, n_eval_episodes=30,
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
            f"market=v2(alpha={self.market.alpha:.3f}, beta={self.market.beta:.3f}, "
            f"crisis_prob={self.market.crisis_prob:.3f})"
        )


DEFAULT_DIVERSITY_SCORES = [0.0, 0.5, 1.0]
DEFAULT_QUICK_DIVERSITY_SCORES = [0.0, 0.5, 1.0]
DEFAULT_N_AGENTS = 50
DEFAULT_N_EPISODES = 500
DEFAULT_N_SEEDS = 5
DEFAULT_EPISODE_STEPS = 500
DEFAULT_ZI_EPISODES = 30
DEFAULT_EVAL_EPISODES = 30
DEFAULT_LOG_EVERY = 25
DEFAULT_ROLLING_WINDOW = 30
DEFAULT_MARKET = MarketDynamics.stable()


def build_sarsa_benchmark_settings(quick: bool, default_workers: int) -> dict:
    if quick:
        return {
            "run_name": "quick",
            "diversity_scores": DEFAULT_QUICK_DIVERSITY_SCORES,
            "n_agents": DEFAULT_N_AGENTS,
            "n_episodes": 20,
            "episode_steps": 150,
            "n_seeds": 1,
            "zi_episodes": 5,
            "eval_episodes": 10,
            "log_every": 10,
            "rolling_window": 5,
            "n_workers": 4,
            "market": DEFAULT_MARKET,
            "sarsa_cfg": DeepSARSAConfig(
                hidden_size=32,
                lr=1e-3,
                epsilon_start=0.30,
                epsilon_end=0.1,
                epsilon_decay=0.97,
                grad_clip=1.0,
                n_step=4,
                reward_scale=30.0,
            ),
        }

    return {
        "run_name": "full",
        "diversity_scores": DEFAULT_DIVERSITY_SCORES,
        "n_agents": DEFAULT_N_AGENTS,
        "n_episodes": DEFAULT_N_EPISODES,
        "episode_steps": DEFAULT_EPISODE_STEPS,
        "n_seeds": DEFAULT_N_SEEDS,
        "zi_episodes": DEFAULT_ZI_EPISODES,
        "eval_episodes": DEFAULT_EVAL_EPISODES,
        "log_every": DEFAULT_LOG_EVERY,
        "rolling_window": DEFAULT_ROLLING_WINDOW,
        "n_workers": default_workers,
        "market": DEFAULT_MARKET,
        "sarsa_cfg": DeepSARSAConfig(
            hidden_size=64,
            lr=1e-3,
            epsilon_start=0.35,
            epsilon_end=0.05,
            epsilon_decay=0.993,
            grad_clip=1.0,
            n_step=6,
            reward_scale=30.0,
        ),
    }


def build_ppo_benchmark_settings(
    quick: bool,
    use_agent_id_features: bool,
    default_workers: int,
) -> dict:
    if quick:
        return {
            "run_name": "quick",
            "diversity_scores": DEFAULT_QUICK_DIVERSITY_SCORES,
            "n_agents": DEFAULT_N_AGENTS,
            "n_episodes": 20,
            "episode_steps": 150,
            "n_seeds": 1,
            "zi_episodes": 5,
            "eval_episodes": 10,
            "log_every": 1,
            "rolling_window": 2,
            "n_workers": 1,
            "market": DEFAULT_MARKET,
            "ppo_cfg": PPOConfig(
                hidden_size=32,
                update_epochs=4,
                minibatch_size=128,
                rollout_episodes=5,
                use_agent_id_features=use_agent_id_features,
            ),
        }

    return {
        "run_name": "full",
        "diversity_scores": DEFAULT_DIVERSITY_SCORES,
        "n_agents": DEFAULT_N_AGENTS,
        "n_episodes": DEFAULT_N_EPISODES,
        "episode_steps": DEFAULT_EPISODE_STEPS,
        "n_seeds": DEFAULT_N_SEEDS,
        "zi_episodes": DEFAULT_ZI_EPISODES,
        "eval_episodes": DEFAULT_EVAL_EPISODES,
        "log_every": DEFAULT_LOG_EVERY,
        "rolling_window": DEFAULT_ROLLING_WINDOW,
        "n_workers": default_workers,
        "market": DEFAULT_MARKET,
        "ppo_cfg": PPOConfig(use_agent_id_features=use_agent_id_features),
    }


def build_ippo_benchmark_settings(quick: bool, default_workers: int) -> dict:
    if quick:
        return {
            "run_name": "quick",
            "diversity_scores": DEFAULT_QUICK_DIVERSITY_SCORES,
            "n_agents": DEFAULT_N_AGENTS,
            "n_episodes": 20,
            "episode_steps": 150,
            "n_seeds": 1,
            "zi_episodes": 5,
            "eval_episodes": 10,
            "log_every": 1,
            "rolling_window": 2,
            "n_workers": 1,
            "market": DEFAULT_MARKET,
            "ppo_cfg": PPOConfig(
                hidden_size=32,
                update_epochs=3,
                minibatch_size=128,
                rollout_episodes=4,
                use_agent_id_features=False,
            ),
        }

    return {
        "run_name": "full",
        "diversity_scores": DEFAULT_DIVERSITY_SCORES,
        "n_agents": DEFAULT_N_AGENTS,
        "n_episodes": DEFAULT_N_EPISODES,
        "episode_steps": DEFAULT_EPISODE_STEPS,
        "n_seeds": DEFAULT_N_SEEDS,
        "zi_episodes": DEFAULT_ZI_EPISODES,
        "eval_episodes": DEFAULT_EVAL_EPISODES,
        "log_every": DEFAULT_LOG_EVERY,
        "rolling_window": DEFAULT_ROLLING_WINDOW,
        "n_workers": default_workers,
        "market": DEFAULT_MARKET,
        "ppo_cfg": PPOConfig(
            hidden_size=48,
            update_epochs=4,
            minibatch_size=256,
            rollout_episodes=5,
            use_agent_id_features=False,
        ),
    }


def build_signal_rule_benchmark_settings(quick: bool, default_workers: int) -> dict:
    if quick:
        return {
            "run_name": "quick",
            "diversity_scores": DEFAULT_QUICK_DIVERSITY_SCORES,
            "n_agents": DEFAULT_N_AGENTS,
            "n_episodes": 0,
            "episode_steps": 150,
            "n_seeds": 1,
            "zi_episodes": 5,
            "eval_episodes": 10,
            "log_every": 1,
            "rolling_window": 1,
            "n_workers": 1,
            "market": DEFAULT_MARKET,
            "rule_threshold": 0.0,
        }

    return {
        "run_name": "full",
        "diversity_scores": DEFAULT_DIVERSITY_SCORES,
        "n_agents": DEFAULT_N_AGENTS,
        "n_episodes": 0,
        "episode_steps": DEFAULT_EPISODE_STEPS,
        "n_seeds": DEFAULT_N_SEEDS,
        "zi_episodes": DEFAULT_ZI_EPISODES,
        "eval_episodes": DEFAULT_EVAL_EPISODES,
        "log_every": 1,
        "rolling_window": 1,
        "n_workers": default_workers,
        "market": DEFAULT_MARKET,
        "rule_threshold": 0.0,
    }
