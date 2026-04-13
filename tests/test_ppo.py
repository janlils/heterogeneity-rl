import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from codes.config import EnvConfig, HTMConfig, LogConfig, MarketDynamics, PPOConfig
from codes.deep_sarsa import DeepSARSAMultiAgent
from codes.double_auction import DoubleAuction
from codes.evaluate_policies import evaluate_policy
from codes.ppo import SharedPPOTrainer
from codes.rl_common import action_mask_from_obs


def make_cfg(**env_kwargs):
    return HTMConfig(
        env=EnvConfig(**env_kwargs),
        market=MarketDynamics.stable(),
        ppo=PPOConfig(hidden_size=16, update_epochs=1, minibatch_size=32),
        log=LogConfig(level="ERROR", save_to_file=False, save_plots=False),
    )


def test_action_mask_blocks_boundary_actions():
    obs = np.zeros(15, dtype=np.float32)
    obs[5] = 1.0
    mask = action_mask_from_obs(obs)
    assert mask.tolist() == [True, False, True]

    obs[5] = -1.0
    mask = action_mask_from_obs(obs)
    assert mask.tolist() == [True, True, False]


def test_ppo_never_selects_masked_action_with_biased_logits():
    cfg = make_cfg(n_agents=2, episode_steps=2)
    trainer = SharedPPOTrainer(cfg.env.n_obs, cfg.env.n_actions, cfg.ppo, seed=1)

    with torch.no_grad():
        trainer.model.policy_head.weight.zero_()
        trainer.model.policy_head.bias[:] = torch.tensor([0.0, 100.0, 90.0])

    obs = np.zeros(cfg.env.n_obs, dtype=np.float32)
    obs[5] = 1.0
    for _ in range(100):
        action, _, _, _ = trainer.act_np(obs, "agent_0", deterministic=False)
        assert action != cfg.env.ACTION_BUY_MARKET

    action, _, _, _ = trainer.act_np(obs, "agent_0", deterministic=True)
    assert action == cfg.env.ACTION_SELL_MARKET


def test_ppo_checkpoint_roundtrip_preserves_deterministic_action(tmp_path):
    cfg = make_cfg(n_agents=2, episode_steps=2)
    trainer = SharedPPOTrainer(cfg.env.n_obs, cfg.env.n_actions, cfg.ppo, seed=2)
    obs = np.linspace(0, 1, cfg.env.n_obs, dtype=np.float32)
    obs[5] = 0.0

    action_before, _, _, _ = trainer.act_np(obs, "agent_0", deterministic=True)
    path = tmp_path / "ppo.pt"
    trainer.save(path, episode=1)

    loaded = SharedPPOTrainer(cfg.env.n_obs, cfg.env.n_actions, cfg.ppo, seed=999)
    loaded.load(path)
    action_after, _, _, _ = loaded.act_np(obs, "agent_0", deterministic=True)

    assert action_after == action_before


def test_common_evaluator_schema_for_zi_sarsa_ppo():
    cfg = make_cfg(n_agents=3, episode_steps=2)
    da = DoubleAuction(cfg, seed=3)
    da.reset(diversity_score=0.5, seed=3)
    agent_ids = list(da.population.agents.keys())
    gammas = np.array([da.population.agents[aid].gamma for aid in agent_ids])

    sarsa = DeepSARSAMultiAgent(
        agent_ids=agent_ids,
        agent_gammas=gammas,
        n_obs=cfg.env.n_obs,
        n_actions=cfg.env.n_actions,
        seed=3,
    )
    ppo = SharedPPOTrainer(
        cfg.env.n_obs,
        cfg.env.n_actions,
        cfg.ppo,
        seed=3,
        agent_ids=agent_ids,
    )

    zi_records = evaluate_policy("ZI", None, cfg, 0.5, n_episodes=1, seed=3)
    sarsa_records = evaluate_policy("DeepSARSA_EVAL", sarsa, cfg, 0.5, n_episodes=1, seed=3)
    ppo_records = evaluate_policy("PPO_EVAL", ppo, cfg, 0.5, n_episodes=1, seed=3)

    required = {
        "episode",
        "diversity_score",
        "seed",
        "algorithm",
        "trade_accuracy",
        "mean_pnl",
        "mean_total_pnl",
        "n_trades",
        "action_buy_frac",
        "action_sell_frac",
        "action_hold_frac",
        "gini",
    }
    schemas = [set(r[0]) for r in (zi_records, sarsa_records, ppo_records)]
    assert all(required.issubset(schema) for schema in schemas)
    assert all(r[0]["primary_metric"] == "trade_accuracy" for r in (zi_records, sarsa_records, ppo_records))
