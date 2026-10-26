"""Unit tests for the Exact Best-Response Dynamic Programming (Value Iteration) solver."""

import numpy as np
import pytest
from src.engine.rules import RuleType
from src.benchmarks.exact_dp import DPSolverConfig, ExactDPSolver


def test_exact_dp_solver_convergence_and_absorbing_states():
    # Fast test instance: T_ban = 1, capital grid with 4 levels
    price_grid = np.array([1.0, 1.5, 2.0], dtype=np.float64)
    capital_levels = np.array([0.0, 0.2, 0.4, 0.6], dtype=np.float64)

    config = DPSolverConfig(
        num_firms=3,
        price_grid=price_grid,
        marginal_cost=1.0,
        quality=2.0,
        outside_option=0.0,
        sigma_d=0.25,
        market_size=1.0,
        fixed_cost=0.2,
        T_ban=1,
        audit_budget=1,
        rule_type=RuleType.MEAN_RATIO,
        rule_param=0.2,
        capital_levels=capital_levels,
        gamma=0.85,
        tol=1e-3,
        max_iter=50,
    )

    solver = ExactDPSolver(config)
    V, policy = solver.solve()

    # 1. State space shape verification
    # Shape: (2, 2, 2, 4, 4, 4)
    assert V.shape == (2, 2, 2, 4, 4, 4)
    assert policy.shape == (2, 2, 2, 4, 4, 4)

    # 2. Values must be non-negative everywhere
    assert np.all(V >= -1e-7)

    # 3. Absorbing state: when firm 1 is bankrupt (k1 = 0), V must be identically 0.0
    assert np.all(V[:, :, :, 0, :, :] == 0.0)

    # 4. Monotonicity in capital: higher capital for firm 1 should not decrease firm 1 value
    v_k1 = V[0, 0, 0, 1, 2, 2]
    v_k2 = V[0, 0, 0, 2, 2, 2]
    assert v_k2 >= v_k1 - 1e-6


def test_exploitability_gain_computation():
    price_grid = np.array([1.0, 1.5, 2.0], dtype=np.float64)
    capital_levels = np.array([0.0, 0.2, 0.4, 0.6], dtype=np.float64)

    config = DPSolverConfig(
        num_firms=3,
        price_grid=price_grid,
        marginal_cost=1.0,
        quality=2.0,
        outside_option=0.0,
        sigma_d=0.25,
        market_size=1.0,
        fixed_cost=0.2,
        T_ban=2,
        capital_levels=capital_levels,
        gamma=0.85,
        tol=1e-3,
        max_iter=30,
    )

    solver = ExactDPSolver(config)
    V, policy = solver.solve()

    gain = solver.compute_exploitability_gain(V, initial_capital=0.4)
    # Gain is a finite float
    assert isinstance(gain, float)
    assert not np.isnan(gain)

