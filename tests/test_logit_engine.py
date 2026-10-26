"""Unit tests for multinomial logit demand clearing and consumer surplus."""

import numpy as np
import pytest
from src.engine.market import (
    compute_market_shares,
    compute_consumer_surplus,
    compute_profits,
)


def test_shares_sum_and_outside_option():
    prices = np.array([1.5, 2.0, 2.5], dtype=np.float64)
    qualities = np.array([2.0, 2.0, 2.0], dtype=np.float64)
    active = np.array([1, 1, 1], dtype=np.int32)
    outside_option = 0.0
    sigma_d = 0.25

    shares = compute_market_shares(prices, qualities, active, outside_option, sigma_d)

    # All active shares should be strictly positive
    assert np.all(shares > 0.0)
    # Outside option share
    v_out = np.exp(outside_option / sigma_d)
    v_act = np.exp((qualities - prices) / sigma_d)
    outside_share = v_out / (v_out + np.sum(v_act))

    # Total market choice probabilities must sum to 1.0
    assert np.isclose(np.sum(shares) + outside_share, 1.0, atol=1e-7)


def test_inactive_and_banned_firms_receive_zero_share():
    prices = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    qualities = np.array([2.0, 2.0, 2.0], dtype=np.float64)
    # Firm 1 is banned/inactive
    active = np.array([1, 0, 1], dtype=np.int32)

    shares = compute_market_shares(prices, qualities, active, 0.0, 0.25)
    assert shares[1] == 0.0
    assert shares[0] > 0.0
    assert shares[2] > 0.0
    assert np.isclose(shares[0], shares[2], atol=1e-7)


def test_price_sensitivity_monotonicity():
    qualities = np.array([2.0, 2.0, 2.0], dtype=np.float64)
    active = np.array([1, 1, 1], dtype=np.int32)

    p_low = np.array([1.5, 2.0, 2.0], dtype=np.float64)
    p_high = np.array([2.5, 2.0, 2.0], dtype=np.float64)

    shares_low = compute_market_shares(p_low, qualities, active, 0.0, 0.25)
    shares_high = compute_market_shares(p_high, qualities, active, 0.0, 0.25)

    # Increasing price must strictly reduce own market share
    assert shares_high[0] < shares_low[0]
    # Rivals' shares should increase
    assert shares_high[1] > shares_low[1]
    assert shares_high[2] > shares_low[2]


def test_consumer_surplus_decreases_with_prices():
    qualities = np.array([2.0, 2.0, 2.0], dtype=np.float64)
    active = np.array([1, 1, 1], dtype=np.int32)

    p_low = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    p_high = np.array([2.0, 2.0, 2.0], dtype=np.float64)

    cs_low = compute_consumer_surplus(p_low, qualities, active, 0.0, 0.25, 100.0)
    cs_high = compute_consumer_surplus(p_high, qualities, active, 0.0, 0.25, 100.0)

    assert cs_low > cs_high
