"""Unit tests for capital accumulation and absorbing bankruptcy."""

import numpy as np
import pytest
from src.engine.capital import update_capital, get_bankrupt_mask


def test_solvent_firm_capital_accumulation():
    capital = np.array([10.0, 5.0, 2.0], dtype=np.float64)
    profits = np.array([4.0, 1.0, 0.0], dtype=np.float64)
    fixed_cost = 2.0

    # Expected: [10 + 4 - 2 = 12, 5 + 1 - 2 = 4, max(0, 2 + 0 - 2) = 0]
    new_capital = update_capital(capital, profits, fixed_cost)
    expected = np.array([12.0, 4.0, 0.0], dtype=np.float64)

    assert np.allclose(new_capital, expected)


def test_banned_firm_fixed_cost_drain():
    capital = np.array([3.0, 3.0, 3.0], dtype=np.float64)
    # Banned firm earns 0 profit
    profits = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    fixed_cost = 1.0

    # After 3 periods of bans, capital should hit 0
    cap_t1 = update_capital(capital, profits, fixed_cost)
    assert np.allclose(cap_t1, [2.0, 2.0, 2.0])

    cap_t2 = update_capital(cap_t1, profits, fixed_cost)
    assert np.allclose(cap_t2, [1.0, 1.0, 1.0])

    cap_t3 = update_capital(cap_t2, profits, fixed_cost)
    assert np.allclose(cap_t3, [0.0, 0.0, 0.0])


def test_absorbing_bankruptcy_state():
    # Firm 2 is already bankrupt (0.0)
    capital = np.array([10.0, 0.0, 5.0], dtype=np.float64)
    # Even if firm 2 theoretically had positive profit, absorbing state must keep it at 0.0
    profits = np.array([2.0, 100.0, 2.0], dtype=np.float64)
    fixed_cost = 1.0

    new_capital = update_capital(capital, profits, fixed_cost)
    assert new_capital[1] == 0.0
    assert new_capital[0] == 11.0
    assert new_capital[2] == 6.0

    bankrupt_mask = get_bankrupt_mask(new_capital)
    assert bankrupt_mask[1] == True
    assert bankrupt_mask[0] == False
