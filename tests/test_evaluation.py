"""Unit tests for evaluation metrics and statistical hypothesis testing."""

try:
    import pytest
except ImportError:
    pass

import numpy as np
from src.evaluation.metrics import (
    compute_weaponization_rate,
    compute_collusion_index,
    compute_social_welfare_loss,
    compute_exploitability_gain,
)
from src.evaluation.stats import (
    bootstrap_ci,
    compute_summary_stats,
    test_hypothesis_h1,
    test_hypothesis_h2,
    test_hypothesis_h3,
)


def test_metrics_weaponization_rate():
    assert compute_weaponization_rate(0, 0) == 0.0
    assert compute_weaponization_rate(10, 0) == 0.0
    assert np.isclose(compute_weaponization_rate(100, 35), 0.35)
    # Check clipping
    assert compute_weaponization_rate(10, 15) == 1.0


def test_metrics_collusion_index():
    pi_nash = 0.20
    pi_mono = 0.60
    # At Nash profit
    assert np.isclose(compute_collusion_index(0.20, pi_nash, pi_mono), 0.0)
    # At Monopoly profit
    assert np.isclose(compute_collusion_index(0.60, pi_nash, pi_mono), 1.0)
    # Midpoint
    assert np.isclose(compute_collusion_index(0.40, pi_nash, pi_mono), 0.5)


def test_metrics_social_welfare_loss():
    cs_nash = 1.50
    pi_nash = [0.20, 0.20, 0.20]
    # Identical to Nash -> 0 loss
    swl_zero = compute_social_welfare_loss(cs_nash, pi_nash, cs_nash, pi_nash)
    assert np.isclose(swl_zero, 0.0)

    # Harmed welfare
    cs_low = 1.00
    pi_low = [0.10, 0.10, 0.10]
    swl_harm = compute_social_welfare_loss(cs_low, pi_low, cs_nash, pi_nash)
    assert swl_harm > 0.0


def test_metrics_exploitability_gain():
    v_hon = 10.5
    v_opt = 14.2
    assert np.isclose(compute_exploitability_gain(v_opt, v_hon), 3.7)


def test_stats_bootstrap_ci():
    rng = np.random.default_rng(123)
    data = rng.normal(loc=5.0, scale=1.0, size=100)
    low, high = bootstrap_ci(data, num_bootstrap=500, ci=0.95, rng=rng)
    assert low < 5.0 < high
    assert high - low < 1.0


def test_stats_summary_stats():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    stats_dict = compute_summary_stats(data)
    assert np.isclose(stats_dict["mean"], 3.0)
    assert np.isclose(stats_dict["median"], 3.0)
    assert stats_dict["ci_lower"] < 3.0 < stats_dict["ci_upper"]
    assert stats_dict["n"] == 5


def test_stats_hypothesis_h1():
    # Mean has higher W than Median and Cap
    w_mean = np.array([0.45, 0.50, 0.55, 0.48, 0.52, 0.60, 0.58])
    w_median = np.array([0.05, 0.08, 0.06, 0.04, 0.07, 0.05, 0.06])
    w_cap = np.array([0.00, 0.00, 0.01, 0.00, 0.00, 0.00, 0.00])

    res = test_hypothesis_h1(w_mean, w_median, w_cap)
    assert res["hypothesis_supported"] is True
    assert res["mean_w"] > res["median_w"] > res["cap_w"]
    assert res["p_val_mean_gt_median"] < 0.01


def test_stats_hypothesis_h2():
    # High W when T_ban >= K0 / Phi, low W otherwise
    k0_phi = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 5.0, 5.0, 8.0, 8.0, 10.0])
    t_ban = np.array([4, 4, 3, 3, 3, 2, 2, 2, 2, 2])
    # Lethal: [4>=2, 4>=2, 3>=2, 3>=2, 3>=2] (5 items)
    # Sublethal: [2<5, 2<5, 2<8, 2<8, 2<10] (5 items)
    w_rates = np.array([0.65, 0.70, 0.68, 0.72, 0.66, 0.05, 0.08, 0.02, 0.01, 0.03])

    res = test_hypothesis_h2(k0_phi, t_ban, w_rates)
    assert res["hypothesis_supported"] is True
    assert res["difference"] > 0.50


def test_stats_hypothesis_h3():
    # Monotonically decreasing W with ethics mu
    mu_values = np.array([0.0, 0.1, 0.2, 0.5, 1.0])
    w_rates = np.array([0.60, 0.45, 0.30, 0.10, 0.02])

    res = test_hypothesis_h3(mu_values, w_rates, trait_name="ethics_mu")
    assert res["monotonic_decrease_supported"] is True
    assert res["spearman_rho"] < -0.90
