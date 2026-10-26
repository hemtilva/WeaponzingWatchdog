"""Unit tests for Bertrand-Nash, Joint Monopoly, and Collusion Index benchmarks."""

import numpy as np
import pytest
from src.engine.market import compute_market_shares, compute_profits
from src.benchmarks.nash import (
    compute_continuous_nash_price,
    find_discrete_nash_equilibrium,
)
from src.benchmarks.monopoly import (
    find_discrete_monopoly_prices,
    compute_collusion_index,
)


def test_continuous_nash_price_satisfies_foc():
    cost = 1.0
    quality = 2.0
    outside_option = 0.0
    sigma_d = 0.25
    N = 3

    p_nash = compute_continuous_nash_price(cost, quality, outside_option, sigma_d, N)

    # Verify FOC: p_nash == cost + sigma_d / (1 - s(p_nash))
    prices = np.full(N, p_nash, dtype=np.float64)
    qualities = np.full(N, quality, dtype=np.float64)
    active = np.ones(N, dtype=np.int32)

    shares = compute_market_shares(prices, qualities, active, outside_option, sigma_d)
    share = shares[0]
    expected_p = cost + sigma_d / (1.0 - share)

    assert np.isclose(p_nash, expected_p, atol=1e-5)


def test_discrete_nash_equilibrium_stability():
    cost = 1.0
    quality = 2.0
    outside_option = 0.0
    sigma_d = 0.25
    N = 3
    costs = np.full(N, cost)
    qualities = np.full(N, quality)

    # Discretized price grid with M = 11 levels
    price_grid = np.linspace(1.0, 2.5, 11)

    nash_prices, nash_profits = find_discrete_nash_equilibrium(
        price_grid, costs, qualities, outside_option, sigma_d, Q=1.0
    )

    # Verify that no firm has a unilateral profitable deviation on the grid
    active = np.ones(N, dtype=np.int32)
    for dev_p in price_grid:
        dev_prices = nash_prices.copy()
        dev_prices[0] = dev_p
        dev_shares = compute_market_shares(dev_prices, qualities, active, outside_option, sigma_d)
        dev_profit = compute_profits(dev_prices, costs, dev_shares, 1.0, active)[0]

        assert dev_profit <= nash_profits[0] + 1e-7, (
            f"Profitable unilateral deviation found: {dev_p} yields {dev_profit} > {nash_profits[0]}"
        )


def test_monopoly_dominates_nash_profit():
    cost = 1.0
    quality = 2.0
    outside_option = 0.0
    sigma_d = 0.25
    N = 3
    costs = np.full(N, cost)
    qualities = np.full(N, quality)
    price_grid = np.linspace(1.0, 2.5, 11)

    nash_prices, nash_profits = find_discrete_nash_equilibrium(
        price_grid, costs, qualities, outside_option, sigma_d, Q=1.0
    )
    mono_prices, mono_profits, total_mono = find_discrete_monopoly_prices(
        price_grid, costs, qualities, outside_option, sigma_d, Q=1.0
    )

    # Monopoly prices must be strictly higher than Nash prices
    assert np.all(mono_prices > nash_prices)
    # Aggregate industry profits must be strictly higher under joint monopoly
    assert np.sum(mono_profits) > np.sum(nash_profits)


def test_collusion_index_bounds():
    p_nash = 0.25
    p_mono = 0.75

    # Bertrand-Nash yields Delta = 0
    assert np.isclose(compute_collusion_index(p_nash, p_nash, p_mono), 0.0)
    # Cartel monopoly yields Delta = 1
    assert np.isclose(compute_collusion_index(p_mono, p_nash, p_mono), 1.0)
    # Intermediate profit yields intermediate Delta
    assert np.isclose(compute_collusion_index(0.50, p_nash, p_mono), 0.5)
