"""Agent trait parameters, CRRA utility functions, and reward calculations."""

from dataclasses import dataclass
import numpy as np


@dataclass
class AgentTraits:
    """Private behavioral traits for a market seller."""
    alpha: float = 0.0          # Risk aversion parameter in [0, 2] (0 = risk-neutral)
    lambda_altruism: float = 0.0 # Altruism weight on external social surplus in [0, 1]
    mu_ethics: float = 0.0       # Ethics penalty weight for causing rival bans (>= 0)


def crra_utility(x: float, alpha: float) -> float:
    """Evaluates Constant Relative Risk Aversion (CRRA) utility:

    U_alpha(x) = (x^(1 - alpha) - 1) / (1 - alpha)  for alpha != 1
    U_alpha(x) = ln(x)                              for alpha == 1
    """
    if x <= 0.0:
        raise ValueError("CRRA argument must be strictly positive.")

    if abs(alpha - 1.0) < 1e-6:
        return float(np.log(x))
    return float((x ** (1.0 - alpha) - 1.0) / (1.0 - alpha))


def compute_agent_reward(
    profit: float,
    Q: float,
    CS: float,
    other_profits_sum: float,
    N: int,
    harm_count: int,
    traits: AgentTraits,
    eps: float = 1e-4,
) -> float:
    """Computes total per-period agent reward incorporating behavioral traits.

    R_i = U_alpha(pi_i / Q + eps) + lambda_i * W_ext,i - mu_i * C_harm(i, t)
    where W_ext,i = (CS + sum_{j != i} pi_j) / (N * Q).
    """
    # 1. Monetary payoff with shift eps to keep utility finite during zero-profit bans
    monetary = max(0.0, profit) / Q + eps
    u_monetary = crra_utility(monetary, traits.alpha)

    # 2. External surplus
    w_ext = (CS + other_profits_sum) / (N * Q)

    # 3. Harm penalty
    harm_penalty = traits.mu_ethics * float(harm_count)

    # Total reward
    reward = u_monetary + traits.lambda_altruism * w_ext - harm_penalty
    return float(reward)
