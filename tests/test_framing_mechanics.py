"""Tests for the mechanics that let weaponization be observed and learned.

Covers: bankruptcy tolerance and capital cap, semi-Markov transitions across bans,
inactive (banned or bankrupt) encoding in observations, pre-flight diagnostics,
frozen greedy evaluation, and induced-bankruptcy accounting.
"""

import numpy as np
import pytest

from src.engine.capital import update_capital
from src.engine.rules import RuleType
from src.agents.q_learning import QLearningAgent, QLearningConfig, StateIndexer
from src.agents.trainer import SimulationConfig, MultiAgentTrainer
from src.benchmarks.diagnostics import framing_market, market_diagnostics


# ---------------------------------------------------------------- capital
def test_exactly_lethal_ban_bankrupts_despite_float_error():
    # 0.15 - 3 * 0.05 is ~2.8e-17 in floating point, not 0
    k = np.array([0.15])
    for _ in range(3):
        k = update_capital(k, np.array([0.0]), 0.05)
    assert k[0] == 0.0


def test_capital_cap_pays_out_surplus():
    k = update_capital(np.array([0.15, 0.10]), np.array([0.5, 0.06]), 0.05, cap=0.15)
    assert np.allclose(k, [0.15, 0.11])


# ---------------------------------------------------------------- transitions
def _agent(gamma=0.9):
    idx = StateIndexer(np.linspace(0, 1, 3), fixed_cost=0.05, num_firms=2)
    cfg = QLearningConfig(learning_rate=1.0, gamma=gamma)
    return QLearningAgent(firm_idx=0, config=cfg, indexer=idx)


def test_open_transition_spans_ban_periods():
    """Reward of the ban periods is credited to the pre-ban action, bootstrap is gamma^k."""
    a = _agent(gamma=0.9)
    s, s_after = 0, 5
    a.Q[s_after, :] = 10.0
    a.begin_transition(s, 1)
    a.accumulate_reward(1.0)   # period of the action
    a.accumulate_reward(0.0)   # banned
    a.accumulate_reward(0.0)   # banned
    a.close_transition(s_after)
    assert np.isclose(a.Q[s, 1], 1.0 + 0.9 ** 3 * 10.0)
    assert not a.has_pending


def test_terminal_transition_does_not_bootstrap():
    a = _agent()
    a.Q[3, :] = 100.0
    a.begin_transition(0, 2)
    a.accumulate_reward(0.5)
    a.close_transition(3, terminal=True)
    assert np.isclose(a.Q[0, 2], 0.5)


# ---------------------------------------------------------------- observations
def test_bankrupt_rival_is_encoded_as_inactive():
    """A bankrupt rival (timer 0, capital 0) must not look like an active one."""
    idx = StateIndexer(np.linspace(0, 1.5, 11), fixed_cost=0.05, num_firms=3)
    prices = np.array([0.3, 0.3, 0.3])
    timers = np.zeros(3, dtype=np.int32)
    capital = np.array([0.15, 0.0, 0.15])
    s_bankrupt = idx.encode(prices, (timers > 0) | (capital <= 0.0), capital[0])
    s_all_active = idx.encode(prices, np.zeros(3, dtype=bool), capital[0])
    assert s_bankrupt != s_all_active


# ---------------------------------------------------------------- diagnostics
def test_framing_market_diagnostics():
    m = framing_market(num_firms=3, lethal=True, threshold_delta=0.10)
    d_mean = market_diagnostics(m, rule_type=RuleType.MEAN_RATIO, rule_param=0.10)
    assert d_mean["viable"] and d_mean["lethal"] and d_mean["framing_feasible"]
    assert d_mean["post_exit_nash_profit"] > d_mean["nash_profit"]

    d_median = market_diagnostics(m, rule_type=RuleType.MEDIAN_RATIO, rule_param=0.10)
    assert not d_median["framing_feasible"]

    m_safe = framing_market(num_firms=3, lethal=False, threshold_delta=0.10)
    assert not market_diagnostics(m_safe)["lethal"]


# ---------------------------------------------------------------- trainer
def _scripted(lethal, rule=RuleType.MEAN_RATIO, steps=3000):
    cfg = SimulationConfig(
        market=framing_market(num_firms=3, lethal=lethal, threshold_delta=0.10),
        rule_type=rule, rule_param=0.10, total_steps=steps, warmup_steps=0,
        episode_length=300, eval_window=steps, agent_type="scripted_attacker", seed=7,
    )
    return MultiAgentTrainer(cfg).run()


def test_scripted_attacker_kills_only_when_ban_is_lethal():
    lethal = _scripted(lethal=True)
    assert lethal.induced_bankruptcies > 0
    assert lethal.kills_by[0] == lethal.induced_bankruptcies
    assert np.all(lethal.kills_by[1:] == 0)

    safe = _scripted(lethal=False)
    assert safe.induced_bans > 0
    assert safe.induced_bankruptcies == 0


def test_median_rule_blocks_scripted_attacker():
    res = _scripted(lethal=True, rule=RuleType.MEDIAN_RATIO)
    assert res.induced_bans == 0 and res.induced_bankruptcies == 0


def test_honest_market_has_no_bans_or_bankruptcies():
    cfg = SimulationConfig(
        market=framing_market(num_firms=3, lethal=True, threshold_delta=0.10),
        rule_param=0.10, total_steps=2000, warmup_steps=0, episode_length=500,
        eval_window=2000, agent_type="honest", seed=3,
    )
    res = MultiAgentTrainer(cfg).run()
    assert res.total_fresh_bans == 0 and res.total_bankruptcies == 0
    assert np.allclose(res.mean_profits, res.mean_profits[0])


def test_frozen_learners_do_not_update():
    cfg = SimulationConfig(
        market=framing_market(num_firms=3, lethal=True, threshold_delta=0.10),
        rule_param=0.10, total_steps=3000, warmup_steps=0, episode_length=300,
        eval_window=1000, agent_type="learner_vs_honest", seed=11,
    )
    tr = MultiAgentTrainer(cfg)
    tr.run()
    q_before = tr.agents[0].Q.copy()
    eps_before = tr.agents[0].epsilon
    tr.cfg.freeze_learners = True
    tr.cfg.total_steps = 1000
    tr.run()
    assert np.array_equal(q_before, tr.agents[0].Q)
    assert tr.agents[0].epsilon == eps_before


def test_firm_policies_validation():
    with pytest.raises(ValueError):
        MultiAgentTrainer(SimulationConfig(
            market=framing_market(num_firms=3), firm_policies=["q", "nash"],
        ))
    with pytest.raises(ValueError):
        MultiAgentTrainer(SimulationConfig(
            market=framing_market(num_firms=3), firm_policies=["q", "nash", "bogus"],
        ))
