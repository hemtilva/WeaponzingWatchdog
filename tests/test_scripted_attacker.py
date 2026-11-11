"""Unit tests verifying harm attribution measure W on a scripted predatory attacker (Feedback Item 2).

A scripted attacker deliberately undercuts to marginal cost when rivals are active,
attempting to weaponize the relative regulatory threshold against innocent rivals.
"""

import numpy as np
from src.engine.market import MarketConfig
from src.engine.rules import RuleType
from src.agents.trainer import SimulationConfig, MultiAgentTrainer


def test_scripted_attacker_mean_ratio_induces_bans():
    """Under Mean-Ratio, an intentional undercutter frames rivals. Harm measure W must be ~1.0."""
    m_cfg = MarketConfig(
        num_firms=4,
        price_sensitivity=0.25,
        fixed_cost=0.10,
        initial_capital=1.0,
        ban_duration=3,
        threshold_delta=0.20,
    )
    sim_cfg = SimulationConfig(
        market=m_cfg,
        total_steps=1500,
        warmup_steps=100,
        episode_length=300,
        rule_type=RuleType.MEAN_RATIO,
        rule_param=0.20,
        eval_window=1000,
        agent_type="scripted_attacker",
        seed=42,
    )
    trainer = MultiAgentTrainer(sim_cfg)
    results = trainer.run()

    # The scripted attacker undercuts and causes innocent rivals to get banned
    assert results.total_fresh_bans > 0, "Scripted attacker under Mean-Ratio must cause bans"
    assert results.induced_bans > 0, "Counterfactual check must detect induced bans"
    # Weaponization rate W = induced_bans / total_fresh_bans must be near 1.0
    assert results.weaponization_rate >= 0.80, f"Expected W >= 0.80, got {results.weaponization_rate:.3f}"


def test_scripted_attacker_median_ratio_immunity():
    """Under Median-Ratio with N=4, a single undercutter (k=1 < N/2) cannot frame rivals (Proposition 1)."""
    m_cfg = MarketConfig(
        num_firms=4,
        price_sensitivity=0.25,
        fixed_cost=0.10,
        initial_capital=1.0,
        ban_duration=3,
        threshold_delta=0.20,
    )
    sim_cfg = SimulationConfig(
        market=m_cfg,
        total_steps=1000,
        warmup_steps=100,
        episode_length=300,
        rule_type=RuleType.MEDIAN_RATIO,
        rule_param=0.20,
        eval_window=800,
        agent_type="scripted_attacker",
        seed=42,
    )
    trainer = MultiAgentTrainer(sim_cfg)
    results = trainer.run()

    # Median is unchanged by a single undercutter -> no rivals flagged
    assert results.total_fresh_bans == 0
    assert results.induced_bans == 0
    assert results.weaponization_rate == 0.0


def test_scripted_attacker_price_cap_immunity():
    """Under Price Cap, flags are independent of peer prices. Undercutting cannot frame rivals."""
    m_cfg = MarketConfig(
        num_firms=4,
        price_sensitivity=0.25,
        fixed_cost=0.10,
        initial_capital=1.0,
        ban_duration=3,
        threshold_delta=0.20,
    )
    sim_cfg = SimulationConfig(
        market=m_cfg,
        total_steps=1000,
        warmup_steps=100,
        episode_length=300,
        rule_type=RuleType.PRICE_CAP,
        rule_param=2.20,  # Cap above Nash price
        eval_window=800,
        agent_type="scripted_attacker",
        seed=42,
    )
    trainer = MultiAgentTrainer(sim_cfg)
    results = trainer.run()

    # Absolute price cap is immune to predatory framing
    assert results.total_fresh_bans == 0
    assert results.induced_bans == 0
    assert results.weaponization_rate == 0.0


def test_state_visitation_reporting_in_trainer():
    """Verifies state visitation tracking and coverage reporting (Feedback Item 1)."""
    m_cfg = MarketConfig(num_firms=3, fixed_cost=0.10, initial_capital=1.0, ban_duration=2)
    sim_cfg = SimulationConfig(
        market=m_cfg,
        total_steps=500,
        warmup_steps=100,
        episode_length=200,
        eval_window=300,
        agent_type="adaptive",
        seed=42,
    )
    trainer = MultiAgentTrainer(sim_cfg)
    results = trainer.run()

    assert results.unique_states_visited > 0
    assert results.state_coverage_ratio > 0.0
    assert results.state_coverage_ratio <= 1.0

