"""Unit tests for the MultiAgentTrainer and simulation lifecycle."""

import numpy as np
import pytest
from src.engine.market import MarketConfig
from src.engine.rules import RuleType
from src.agents.q_learning import QLearningConfig
from src.agents.trainer import SimulationConfig, MultiAgentTrainer


def test_simulation_runs_and_produces_valid_metrics():
    # Fast test: 2,000 total steps with 500 step episodes and 500 step warmup
    m_cfg = MarketConfig(
        num_firms=3,
        price_sensitivity=0.25,
        fixed_cost=0.1,
        initial_capital=1.0,
        ban_duration=2,
    )
    q_cfg = QLearningConfig(
        learning_rate=0.1,
        gamma=0.9,
        eps_start=0.5,
        eps_decay=0.999,
    )

    sim_cfg = SimulationConfig(
        market=m_cfg,
        q_learning=q_cfg,
        total_steps=2_000,
        episode_length=500,
        warmup_steps=500,
        rule_type=RuleType.MEAN_RATIO,
        rule_param=0.20,
        eval_window=1_000,
        seed=42,
    )

    trainer = MultiAgentTrainer(sim_cfg)
    results = trainer.run()

    # Check that results are bounded and valid
    assert 0.0 <= results.weaponization_rate <= 1.0
    assert isinstance(results.collusion_index, float)
    assert not np.isnan(results.collusion_index)
    assert isinstance(results.social_welfare_loss, float)
    assert not np.isnan(results.social_welfare_loss)
    assert len(results.mean_prices) == 3
    assert len(results.mean_profits) == 3
    assert 0.0 <= results.bankruptcy_rate <= 1.0


def test_warmup_suppresses_early_bans():
    # Verify that in a pure warmup simulation (warmup >= total_steps), 0 bans occur
    m_cfg = MarketConfig(num_firms=3)
    sim_cfg = SimulationConfig(
        market=m_cfg,
        total_steps=500,
        warmup_steps=1_000,  # Entire run is warmup
        eval_window=400,
        seed=100,
    )

    trainer = MultiAgentTrainer(sim_cfg)
    results = trainer.run()

    assert results.total_fresh_bans == 0
    assert results.weaponization_rate == 0.0
