"""Unit tests for agent traits, CRRA utility, and reward calculations."""

import numpy as np
import pytest
from src.agents.traits import AgentTraits, crra_utility, compute_agent_reward


def test_crra_utility_risk_neutral():
    # When alpha = 0: U(x) = (x^1 - 1) / 1 = x - 1
    assert np.isclose(crra_utility(1.5, alpha=0.0), 0.5)
    assert np.isclose(crra_utility(3.0, alpha=0.0), 2.0)


def test_crra_utility_logarithmic():
    # When alpha = 1: U(x) = ln(x)
    assert np.isclose(crra_utility(np.e, alpha=1.0), 1.0)
    assert np.isclose(crra_utility(1.0, alpha=1.0), 0.0)


def test_crra_utility_risk_averse_concavity():
    # When alpha = 2: U(x) = (x^-1 - 1) / (-1) = 1 - 1/x
    assert np.isclose(crra_utility(2.0, alpha=2.0), 0.5)
    assert np.isclose(crra_utility(4.0, alpha=2.0), 0.75)
    # Check diminishing marginal utility: U(2) - U(1) > U(3) - U(2)
    diff1 = crra_utility(2.0, alpha=2.0) - crra_utility(1.0, alpha=2.0)
    diff2 = crra_utility(3.0, alpha=2.0) - crra_utility(2.0, alpha=2.0)
    assert diff1 > diff2


def test_compute_agent_reward_ethics_penalty():
    traits_predator = AgentTraits(alpha=0.0, lambda_altruism=0.0, mu_ethics=0.0)
    traits_ethical = AgentTraits(alpha=0.0, lambda_altruism=0.0, mu_ethics=2.5)

    r_predator = compute_agent_reward(
        profit=1.0, Q=1.0, CS=2.0, other_profits_sum=1.0, N=3,
        harm_count=2, traits=traits_predator, eps=1e-4
    )
    r_ethical = compute_agent_reward(
        profit=1.0, Q=1.0, CS=2.0, other_profits_sum=1.0, N=3,
        harm_count=2, traits=traits_ethical, eps=1e-4
    )

    # Difference must be exactly mu * harm = 2.5 * 2 = 5.0
    assert np.isclose(r_predator - r_ethical, 5.0)


def test_compute_agent_reward_altruism():
    traits_selfish = AgentTraits(alpha=0.0, lambda_altruism=0.0, mu_ethics=0.0)
    traits_altruist = AgentTraits(alpha=0.0, lambda_altruism=0.5, mu_ethics=0.0)

    r_selfish = compute_agent_reward(
        profit=1.0, Q=1.0, CS=3.0, other_profits_sum=3.0, N=3,
        harm_count=0, traits=traits_selfish, eps=1e-4
    )
    r_altruist = compute_agent_reward(
        profit=1.0, Q=1.0, CS=3.0, other_profits_sum=3.0, N=3,
        harm_count=0, traits=traits_altruist, eps=1e-4
    )

    # External surplus = (3.0 + 3.0) / (3 * 1.0) = 2.0
    # Difference should be lambda * W_ext = 0.5 * 2.0 = 1.0
    assert np.isclose(r_altruist - r_selfish, 1.0)

