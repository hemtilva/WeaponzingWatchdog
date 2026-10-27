"""Core game-theoretic and market evaluation metrics."""

from typing import Sequence, Union
import numpy as np


def compute_weaponization_rate(
    total_fresh_bans: int,
    induced_bans: int,
) -> float:
    """Computes the Weaponization Rate (W):

    W = Induced Bans / Total Fresh Bans in [0, 1].
    If total_fresh_bans == 0, returns 0.0.
    """
    if total_fresh_bans <= 0:
        return 0.0
    rate = float(induced_bans) / float(total_fresh_bans)
    return float(np.clip(rate, 0.0, 1.0))


def compute_collusion_index(
    mean_realized_profit: float,
    nash_profit: float,
    monopoly_profit: float,
) -> float:
    """Computes the normalized Collusion Index (Delta):

    Delta = (pi_realized - pi_Nash) / (pi_Monopoly - pi_Nash).
    At competitive Bertrand-Nash levels, Delta = 0.
    At joint monopoly levels, Delta = 1.
    """
    denom = monopoly_profit - nash_profit
    if abs(denom) < 1e-12:
        return 0.0
    return float((mean_realized_profit - nash_profit) / denom)


def compute_social_welfare_loss(
    cs_realized: float,
    profits_realized: Union[Sequence[float], np.ndarray],
    cs_nash: float,
    profits_nash: Union[Sequence[float], np.ndarray],
) -> float:
    """Computes Social Welfare Loss (SWL):

    SWL = 1 - (CS_realized + sum(pi_realized)) / (CS_Nash + sum(pi_Nash)).
    A positive value reflects deadweight loss or competitive disruption.
    """
    total_realized = float(cs_realized) + float(np.sum(profits_realized))
    total_nash = float(cs_nash) + float(np.sum(profits_nash))

    if total_nash <= 0.0:
        return 0.0

    swl = 1.0 - (total_realized / total_nash)
    return float(swl)


def compute_exploitability_gain(
    v_optimal: float,
    v_honest: float,
) -> float:
    """Computes the Exploitability Gain (E_i(g)) of the best-response policy:

    E_i(g) = V_i(sigma_i^*, sigma_-i^hon; g) - V_i(sigma_i^hon, sigma_-i^hon; g).
    A strictly positive value confirms that the regulatory rule creates an
    incentive for predatory deviation over competitive honest pricing.
    """
    return float(v_optimal - v_honest)
