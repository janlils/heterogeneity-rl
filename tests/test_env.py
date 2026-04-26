"""
Testy środowiska HTM po przejściu na market makera.

Uruchomienie:
    python -m pytest tests/test_env.py -v
"""

import logging
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from codes.config import EnvConfig, HTMConfig, LogConfig, MarketDynamics
from codes.double_auction import DoubleAuction, ZeroIntelligenceAgent, run_zi_baseline

logging.getLogger("htm.auction").setLevel(logging.ERROR)


def make_cfg(**env_kwargs) -> HTMConfig:
    return HTMConfig(
        env=EnvConfig(**env_kwargs),
        market=MarketDynamics.stable(),
        log=LogConfig(level="ERROR", save_to_file=False, save_plots=False),
    )


def test_buy_and_sell_execute_immediately_and_move_ref_price():
    cfg = make_cfg(n_agents=3, episode_steps=5)
    da = DoubleAuction(cfg, seed=1)
    da.reset(diversity_score=0.5, seed=1)
    aid0, aid1 = list(da.population.agents)[:2]

    p0 = da.ref_price
    fill_buy = da.execute_single_action(aid0, cfg.env.ACTION_BUY_MARKET)
    p1 = da.ref_price
    fill_sell = da.execute_single_action(aid1, cfg.env.ACTION_SELL_MARKET)
    p2 = da.ref_price

    assert fill_buy is not None
    assert fill_sell is not None
    assert p1 == np.clip(p0 + cfg.env.perm_impact, cfg.env.p_min, cfg.env.p_max)
    assert p2 == np.clip(p1 - cfg.env.perm_impact, cfg.env.p_min, cfg.env.p_max)
    assert da.episode_metrics()["n_trades"] == 2


def test_position_limits_block_extra_actions():
    cfg = make_cfg(n_agents=2, episode_steps=5, max_position=1)
    da = DoubleAuction(cfg, seed=2)
    da.reset(diversity_score=0.0, seed=2)
    aid = next(iter(da.population.agents))

    assert da.execute_single_action(aid, cfg.env.ACTION_BUY_MARKET) is not None
    assert da.execute_single_action(aid, cfg.env.ACTION_BUY_MARKET) is None
    assert da.population.agents[aid].position == 1


def test_no_impact_env_factory_sets_zero_perm_impact():
    env = EnvConfig.no_impact()
    assert env.perm_impact == 0.0
    assert env.half_spread == 0.0001


def test_parallel_actions_match_pairs_and_move_from_excess():
    cfg = make_cfg(n_agents=4, episode_steps=5, half_spread=0.0, perm_impact=0.01)
    da = DoubleAuction(cfg, seed=8)
    da.reset(diversity_score=0.5, seed=8)
    agent_ids = list(da.population.agents)
    p0 = da.ref_price

    da.execute_parallel_actions({
        agent_ids[0]: cfg.env.ACTION_BUY_MARKET,
        agent_ids[1]: cfg.env.ACTION_BUY_MARKET,
        agent_ids[2]: cfg.env.ACTION_SELL_MARKET,
        agent_ids[3]: cfg.env.ACTION_HOLD,
    })

    buy_positions = [da.population.agents[aid].position for aid in agent_ids[:2]]
    buy_entries = [da.population.agents[aid].entry_price for aid in agent_ids[:2]]
    assert sorted(buy_positions) == [0, 1]
    assert sorted(buy_entries) == [0.0, p0]
    assert da.population.agents[agent_ids[2]].position == -1
    assert da.population.agents[agent_ids[2]].entry_price == p0
    assert da.ref_price == np.clip(p0 + cfg.env.perm_impact, cfg.env.p_min, cfg.env.p_max)
    assert da.episode_metrics()["n_trades"] == 2


def test_parallel_matching_with_excess_keeps_unmatched_buys_unchanged():
    cfg = make_cfg(n_agents=20, episode_steps=5, half_spread=0.0, perm_impact=0.01)
    da = DoubleAuction(cfg, seed=11)
    da.reset(diversity_score=0.5, seed=11)
    agent_ids = list(da.population.agents)
    p0 = da.ref_price
    actions = {
        aid: cfg.env.ACTION_HOLD for aid in agent_ids
    }
    for aid in agent_ids[:10]:
        actions[aid] = cfg.env.ACTION_BUY_MARKET
    for aid in agent_ids[10:15]:
        actions[aid] = cfg.env.ACTION_SELL_MARKET

    da.execute_parallel_actions(actions)

    buy_positions = [da.population.agents[aid].position for aid in agent_ids[:10]]
    sell_positions = [da.population.agents[aid].position for aid in agent_ids[10:15]]
    assert sum(pos == 1 for pos in buy_positions) == 5
    assert sum(pos == 0 for pos in buy_positions) == 5
    assert all(pos == -1 for pos in sell_positions)
    assert da.ref_price == np.clip(
        p0 + cfg.env.perm_impact * np.sqrt(5),
        cfg.env.p_min,
        cfg.env.p_max,
    )


def test_realized_pnl_uses_entry_and_exit_prices_only():
    cfg = make_cfg(n_agents=2, episode_steps=5, half_spread=0.0, temp_impact=0.0, perm_impact=0.01)
    da = DoubleAuction(cfg, seed=3)
    da.reset(diversity_score=1.0, seed=3)
    aid = next(iter(da.population.agents))
    agent = da.population.agents[aid]

    buy_fill = da.execute_single_action(aid, cfg.env.ACTION_BUY_MARKET)
    entry = agent.entry_price
    sell_fill = da.execute_single_action(aid, cfg.env.ACTION_SELL_MARKET)
    rewards, _ = da.compute_step_rewards()

    expected_realized = sell_fill["price"] - buy_fill["price"]
    assert np.isclose(agent.realized_pnl, expected_realized)
    assert np.isclose(agent.realized_pnl, da._episode_pnl[aid])
    assert rewards[aid] != 0.0


def test_mid_episode_reward_has_no_risk_or_holding_penalty():
    cfg = make_cfg(n_agents=2, episode_steps=5)
    da = DoubleAuction(cfg, seed=5)
    da.reset(diversity_score=0.0, seed=5)
    aid0, aid1 = list(da.population.agents)[:2]
    da.population.agents[aid0].risk_aversion = 0.1
    da.population.agents[aid1].risk_aversion = 3.0

    da.execute_single_action(aid0, cfg.env.ACTION_BUY_MARKET)
    da.execute_single_action(aid1, cfg.env.ACTION_BUY_MARKET)
    rewards, _ = da.compute_step_rewards()

    assert rewards[aid0] == 0.0
    assert rewards[aid1] == 0.0


def test_terminal_liquidation_closes_positions():
    cfg = make_cfg(n_agents=4, episode_steps=1)
    da = DoubleAuction(cfg, seed=4)
    da.reset(diversity_score=0.5, seed=4)

    for aid in da.population.agents:
        da.execute_single_action(aid, cfg.env.ACTION_BUY_MARKET)
    _, dones = da.compute_step_rewards()
    m = da.episode_metrics()

    assert all(dones.values())
    assert m["open_positions_end"] == 0
    assert all(p.position == 0 and p.entry_price == 0 for p in da.population.agents.values())
    assert m["n_position_closes"] == 0
    assert "mean_terminal_pnl" in m
    assert "terminal_positive_frac" in m
    assert "mean_total_pnl" in m


def test_observation_shape_and_sentiment_semantics():
    cfg = make_cfg(n_agents=3, episode_steps=5)
    da = DoubleAuction(cfg, seed=6)
    obs = da.reset(diversity_score=0.5, seed=6)
    aid = next(iter(obs))
    agent = da.population.agents[aid]

    assert obs[aid].shape == (cfg.env.n_obs,)
    assert np.isclose(obs[aid][0], agent.sentiment)
    assert np.isclose(obs[aid][4], agent.gamma)


def test_zi_baseline_runs_with_same_action_interface():
    cfg = make_cfg(n_agents=5, episode_steps=4)
    result = run_zi_baseline(cfg, diversity_score=0.5, n_episodes=2, seed=7)

    assert result["n_trades"]["mean"] > 0
    assert result["open_positions_end"]["mean"] == 0
    assert "positive_pnl_frac" in result
