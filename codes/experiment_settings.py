"""
Wspólne ustawienia runnerów eksperymentów.

Cel tego modułu:
  - jedno źródło prawdy dla quick/full defaults,
  - shared defaults + algo-specific overrides,
  - brak logiki treningowej i brak zależności od runnerów.
"""

from __future__ import annotations

from typing import Dict

from codes.config import DeepSARSAConfig, MarketDynamics, PPOConfig


DEFAULT_DIVERSITY_SCORES = [0.0, 0.3, 0.5, 0.7, 1.0]
DEFAULT_QUICK_DIVERSITY_SCORES = [0.5]
DEFAULT_N_AGENTS = 50
DEFAULT_N_EPISODES = 500
DEFAULT_N_SEEDS = 10
DEFAULT_EPISODE_STEPS = 500
DEFAULT_ZI_EPISODES = 30
DEFAULT_EVAL_EPISODES = 30
DEFAULT_LOG_EVERY = 25
DEFAULT_ROLLING_WINDOW = 30
DEFAULT_MARKET = MarketDynamics.stable()


def build_sarsa_settings(quick: bool, default_workers: int) -> Dict:
    if quick:
        return {
            "run_name": "quick",
            "diversity_scores": DEFAULT_QUICK_DIVERSITY_SCORES,
            "n_agents": DEFAULT_N_AGENTS,
            "n_episodes": 40,
            "episode_steps": 800,
            "n_seeds": 1,
            "zi_episodes": 10,
            "eval_episodes": 50,
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
                n_step=1,
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
            n_step=1,
        ),
    }


def build_ppo_settings(
    quick: bool,
    use_agent_id_features: bool,
    default_workers: int,
) -> Dict:
    if quick:
        return {
            "run_name": "quick",
            "diversity_scores": DEFAULT_QUICK_DIVERSITY_SCORES,
            "n_agents": DEFAULT_N_AGENTS,
            "n_episodes": 50,
            "episode_steps": 150,
            "n_seeds": 1,
            "zi_episodes": 10,
            "eval_episodes": 50,
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


def build_ippo_settings(quick: bool, default_workers: int) -> Dict:
    if quick:
        return {
            "run_name": "quick",
            "diversity_scores": DEFAULT_QUICK_DIVERSITY_SCORES,
            "n_agents": DEFAULT_N_AGENTS,
            "n_episodes": 40,
            "episode_steps": 150,
            "n_seeds": 1,
            "zi_episodes": 10,
            "eval_episodes": 50,
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


def build_mappo_settings(quick: bool, default_workers: int) -> Dict:
    if quick:
        return {
            "run_name": "quick",
            "diversity_scores": DEFAULT_QUICK_DIVERSITY_SCORES,
            "n_agents": DEFAULT_N_AGENTS,
            "n_episodes": 50,
            "episode_steps": 150,
            "n_seeds": 1,
            "zi_episodes": 10,
            "eval_episodes": 50,
            "log_every": 1,
            "rolling_window": 2,
            "n_workers": 1,
            "market": DEFAULT_MARKET,
            "ppo_cfg": PPOConfig(
                hidden_size=32,
                update_epochs=4,
                minibatch_size=128,
                rollout_episodes=5,
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
            hidden_size=64,
            update_epochs=4,
            minibatch_size=256,
            rollout_episodes=5,
            use_agent_id_features=False,
        ),
    }


def build_signal_rule_settings(quick: bool, default_workers: int) -> Dict:
    if quick:
        return {
            "run_name": "quick",
            "diversity_scores": DEFAULT_QUICK_DIVERSITY_SCORES,
            "n_agents": DEFAULT_N_AGENTS,
            "n_episodes": 0,
            "episode_steps": 150,
            "n_seeds": 1,
            "zi_episodes": 10,
            "eval_episodes": 50,
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
