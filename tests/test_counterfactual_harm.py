"""Unit tests for counterfactual harm attribution."""

import numpy as np
import pytest
from src.engine.rules import (
    RuleType,
    compute_counterfactual_harm,
    evaluate_rules,
)


def test_honest_nash_pricing_causes_zero_harm():
    """An agent pricing at Bertrand-Nash must receive zero harm penalty."""
    nash_prices = np.array([2.0, 2.0, 2.0], dtype=np.float64)
    prices = nash_prices.copy()
    active = np.array([1, 1, 1], dtype=np.int32)
    audited = np.array([1, 1, 1], dtype=np.int32)

    harm = compute_counterfactual_harm(
        prices=prices,
        nash_prices=nash_prices,
        active_mask=active,
        audited=audited,
        rule_type=RuleType.MEAN_RATIO,
        param=0.2,
    )

    assert np.all(harm == 0)


def test_predatory_undercutter_attributed_harm():
    """When an undercutter's price crash causes audited rivals to be flagged, harm must be charged."""
    nash_prices = np.array([2.0, 2.0, 2.0], dtype=np.float64)
    # Firm 0 undercuts to 0.5 (inducing rival flags because 2.0 > 1.2 * mean)
    prices = np.array([0.5, 2.0, 2.0], dtype=np.float64)
    active = np.array([1, 1, 1], dtype=np.int32)
    # Both rivals audited
    audited = np.array([0, 1, 1], dtype=np.int32)

    harm = compute_counterfactual_harm(
        prices=prices,
        nash_prices=nash_prices,
        active_mask=active,
        audited=audited,
        rule_type=RuleType.MEAN_RATIO,
        param=0.2,
    )

    # Firm 0 caused bans for both firm 1 and firm 2
    assert harm[0] == 2
    # Innocent firms did not cause anyone's ban
    assert harm[1] == 0
    assert harm[2] == 0


def test_harm_not_charged_if_rival_not_audited():
    """Harm is only charged if the framed rival was actually audited and suspended."""
    nash_prices = np.array([2.0, 2.0, 2.0], dtype=np.float64)
    prices = np.array([0.5, 2.0, 2.0], dtype=np.float64)
    active = np.array([1, 1, 1], dtype=np.int32)
    # Only firm 1 was audited, firm 2 was not
    audited = np.array([0, 1, 0], dtype=np.int32)

    harm = compute_counterfactual_harm(
        prices=prices,
        nash_prices=nash_prices,
        active_mask=active,
        audited=audited,
        rule_type=RuleType.MEAN_RATIO,
        param=0.2,
    )

    assert harm[0] == 1
    assert harm[1] == 0
    assert harm[2] == 0
