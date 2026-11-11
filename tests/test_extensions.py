"""Unit tests for Extension A (Learning Regulator) and Extension B (Demand Shocks)."""

try:
    import pytest
except ImportError:
    pass

import numpy as np
from src.extensions.demand_shock import DemandShockConfig, DemandShockEngine, run_cohort_simulation
from src.extensions.learning_regulator import (
    AdaptiveRegulatorConfig,
    SlowTimescaleRegulator,
    evaluate_checkpoint_exploitability,
    simulate_bi_level_dynamics,
)
from src.agents.trainer import SimulationConfig
from src.engine.market import MarketConfig


def test_demand_shock_sampling():
    cfg = DemandShockConfig(base_market_size=1.0, shock_start_step=100, panic_psi=0.5, shock_volatility=0.0)
    engine = DemandShockEngine(cfg)

    # 1. Pre-shock step -> exactly base size
    q_pre = engine.sample_market_size(step=50, current_mean_price=2.0, previous_mean_price=1.5)
    assert np.isclose(q_pre, 1.0)

    # 2. Post-shock step with price hike -> 1.0 * (1 + 0.5) = 1.5
    q_post_hike = engine.sample_market_size(step=150, current_mean_price=2.0, previous_mean_price=1.5)
    assert np.isclose(q_post_hike, 1.5)

    # 3. Post-shock step without price hike -> 1.0 * (1 + 0.0) = 1.0
    q_post_flat = engine.sample_market_size(step=150, current_mean_price=1.5, previous_mean_price=1.5)
    assert np.isclose(q_post_flat, 1.0)


def test_demand_shock_simulation_run():
    shock_cfg = DemandShockConfig(base_market_size=1.0, shock_start_step=200, panic_psi=0.3, shock_volatility=0.05)
    engine = DemandShockEngine(shock_cfg)

    m_cfg = MarketConfig(num_firms=3, fixed_cost=0.1, initial_capital=1.0, ban_duration=2)
    sim_cfg = SimulationConfig(
        market=m_cfg,
        total_steps=500,
        warmup_steps=100,
        eval_window=200,
        seed=123,
    )

    res = engine.evaluate_policy_under_shock(sim_cfg)
    assert "mean_market_size" in res
    assert "mean_social_welfare" in res
    assert "weaponization_rate" in res
    assert res["mean_market_size"] > 0.0


def test_adaptive_regulator_epoch_update():
    reg_cfg = AdaptiveRegulatorConfig(
        macro_epoch=50,
        delta_init=0.20,
        delta_min=0.05,
        delta_max=0.50,
        learning_rate=0.05,
    )
    regulator = SlowTimescaleRegulator(reg_cfg)
    assert np.isclose(regulator.current_delta, 0.20)

    # Simulate periods with high false positives
    prices = np.array([1.5, 1.8, 2.2])
    for _ in range(50):
        regulator.observe_period(
            consumer_surplus=1.2,
            num_audited=1,
            prices=prices,
            false_positive_count=2,
        )

    new_delta = regulator.step_macro_epoch()
    # High FP must cause the regulator to loosen (increase) delta
    assert new_delta > 0.20
    assert new_delta <= 0.50
    assert len(regulator.history_objectives) == 1
    assert len(regulator.history_deltas) == 2


def test_checkpoint_exploitability():
    # Evaluate at a tight delta (delta = 0.15) where Proposition 1 allows single-firm framing
    tight_res = evaluate_checkpoint_exploitability(frozen_delta=0.15, t_ban=2, fixed_cost=0.20, k0_val=0.40)
    assert "exploitability_gain" in tight_res
    assert "optimal_price" in tight_res
    assert "is_framing" in tight_res
    assert tight_res["optimal_price"] <= tight_res["nash_price"]

    # Evaluate at a loose delta (delta = 0.55) where Proposition 1 requires delta < 0.50 for N=3
    # Framing should not be a profitable best response here
    loose_res = evaluate_checkpoint_exploitability(frozen_delta=0.55, t_ban=2, fixed_cost=0.20, k0_val=0.40)
    assert loose_res["exploitability_gain"] <= tight_res["exploitability_gain"] + 1e-4


def test_demand_shock_cohort_simulation():
    # Quick 1000-step runs comparing ethical vs predatory
    res_ethical = run_cohort_simulation(cohort="ethical", panic_psi=0.4, steps=1000, warmup=200, seed=42)
    res_predatory = run_cohort_simulation(cohort="predatory", panic_psi=0.4, steps=1000, warmup=200, seed=42)

    assert "mean_social_welfare" in res_ethical
    assert "mean_social_welfare" in res_predatory
    assert res_ethical["cohort"] == "ethical"
    assert res_predatory["cohort"] == "predatory"
    assert res_ethical["mean_market_size"] > 0.0
