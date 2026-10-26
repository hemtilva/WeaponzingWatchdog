"""Rigorous numerical validation of Proposition 1 from the proposal report.

Proposition 1 (Vulnerability of relative rules):
Suppose k firms price at L and the remaining N - k active firms price at H > L.
(i) Under the mean-ratio rule, each high-priced firm satisfies F = 1 iff
    k > delta * (N - k) and H/L > (1 + delta) * k / (k - delta * (N - k)).
    For k = 1 this reads H/L > (1 + delta) / (1 - delta * (N - 1)), which requires delta < 1 / (N - 1).
(ii) Under the median-ratio rule with k < N / 2, the median equals H, so no high-priced firm is flagged.
(iii) Under the z-score rule, each high-priced firm has Z = sqrt(k / (N - k)), independent of L.
"""

import numpy as np
import pytest
from src.engine.rules import RuleType, evaluate_rules


def test_proposition_1_mean_ratio_single_undercutter():
    """Validates Proposition 1 Part (i) for a single undercutter (k = 1)."""
    N = 3
    k = 1
    # For N = 3, delta must be < 1 / (3 - 1) = 0.5
    delta = 0.2
    # Critical threshold ratio: (1 + 0.2) / (1 - 0.2 * 2) = 1.2 / 0.6 = 2.0
    critical_ratio = (1.0 + delta) / (1.0 - delta * (N - 1))
    assert np.isclose(critical_ratio, 2.0)

    active = np.ones(N, dtype=np.int32)
    L = 1.0

    # 1. Below critical threshold: H/L = 1.9 <= 2.0 -> Flags must be 0
    H_sub = 1.9
    prices_sub = np.array([L, H_sub, H_sub], dtype=np.float64)
    flags_sub = evaluate_rules(prices_sub, active, RuleType.MEAN_RATIO, delta)
    assert np.array_equal(flags_sub, [0, 0, 0]), "Rivals should not be flagged when H/L < critical ratio"

    # 2. Exactly at critical threshold: H/L = 2.0 -> Flags must be 0 (strict inequality required)
    H_eq = 2.0
    prices_eq = np.array([L, H_eq, H_eq], dtype=np.float64)
    flags_eq = evaluate_rules(prices_eq, active, RuleType.MEAN_RATIO, delta)
    assert np.array_equal(flags_eq, [0, 0, 0]), "Strict inequality required for flag"

    # 3. Above critical threshold: H/L = 2.1 > 2.0 -> Rivals must be flagged!
    H_sup = 2.1
    prices_sup = np.array([L, H_sup, H_sup], dtype=np.float64)
    flags_sup = evaluate_rules(prices_sup, active, RuleType.MEAN_RATIO, delta)
    assert np.array_equal(flags_sup, [0, 1, 1]), "Rivals MUST be flagged when H/L > critical ratio"


def test_proposition_1_mean_ratio_general_k_and_n():
    """Validates general k and N conditions for the mean-ratio rule."""
    for N in [4, 5]:
        for k in range(1, N):
            delta = 0.15
            # Condition 1: k > delta * (N - k)
            if k <= delta * (N - k):
                continue

            denom = k - delta * (N - k)
            crit_ratio = (1.0 + delta) * k / denom

            active = np.ones(N, dtype=np.int32)
            L = 2.0

            # Test slightly above critical ratio
            H_above = L * (crit_ratio + 0.05)
            prices_above = np.full(N, H_above, dtype=np.float64)
            prices_above[:k] = L

            flags = evaluate_rules(prices_above, active, RuleType.MEAN_RATIO, delta)
            # All high-priced firms must be flagged
            assert np.all(flags[k:] == 1)
            # Undercutters must not be flagged
            assert np.all(flags[:k] == 0)

            # Test slightly below critical ratio
            H_below = L * (crit_ratio - 0.05)
            prices_below = np.full(N, H_below, dtype=np.float64)
            prices_below[:k] = L

            flags_below = evaluate_rules(prices_below, active, RuleType.MEAN_RATIO, delta)
            assert np.all(flags_below == 0)


def test_proposition_1_median_ratio_robustness():
    """Validates Proposition 1 Part (ii): unilateral undercutter CANNOT frame under median rule."""
    N = 3
    delta = 0.1
    active = np.ones(N, dtype=np.int32)

    # Even with an extreme price crash (L = 0.01 vs H = 100.0)
    # The median of [0.01, 100.0, 100.0] is 100.0
    # Condition 100 > (1 + 0.1) * 100 is False, so rivals are NEVER flagged!
    prices = np.array([0.01, 100.0, 100.0], dtype=np.float64)
    flags = evaluate_rules(prices, active, RuleType.MEDIAN_RATIO, delta)
    assert np.array_equal(flags, [0, 0, 0]), "Unilateral undercutter must never frame under median rule"

    # However, if a coalition of k >= N/2 (k = 2) undercuts:
    # The median of [1.0, 1.0, 3.0] is 1.0
    # 3.0 > (1 + 0.1) * 1.0 = 1.1 -> High firm is flagged!
    prices_coalition = np.array([1.0, 1.0, 3.0], dtype=np.float64)
    flags_coalition = evaluate_rules(prices_coalition, active, RuleType.MEDIAN_RATIO, delta)
    assert np.array_equal(flags_coalition, [0, 0, 1]), "Majority coalition frames high-priced rival"


def test_proposition_1_z_score_independence_of_L():
    """Validates Proposition 1 Part (iii): Z is independent of L and equals sqrt(k / (N - k))."""
    N = 4
    k = 1
    expected_z = np.sqrt(k / (N - k))  # sqrt(1/3) ≈ 0.577350269
    H = 10.0
    active = np.ones(N, dtype=np.int32)

    for L in [0.01, 1.0, 5.0, 9.0]:
        prices = np.array([L, H, H, H], dtype=np.float64)
        mean = np.mean(prices)
        std = np.std(prices)
        actual_z = (H - mean) / std

        assert np.isclose(actual_z, expected_z, atol=1e-6), (
            f"Z-score {actual_z} should equal {expected_z} independent of L"
        )


def test_minimum_active_firms_threshold():
    """Rules must disable flagging when N_active < 3."""
    prices = np.array([0.1, 100.0, 100.0], dtype=np.float64)
    # Only 2 firms active
    active = np.array([1, 1, 0], dtype=np.int32)

    for rule in [RuleType.MEAN_RATIO, RuleType.MEDIAN_RATIO, RuleType.PRICE_CAP, RuleType.Z_SCORE]:
        flags = evaluate_rules(prices, active, rule, 0.1)
        assert np.all(flags == 0), f"Rule {rule} should not flag when N_active < 3"

