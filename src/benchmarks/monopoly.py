"""Joint monopoly optimization solver and collusion index metric."""

from typing import Tuple
import itertools
import numpy as np

from src.engine.market import compute_market_shares, compute_profits


def find_discrete_monopoly_prices(
    price_grid: np.ndarray,
    costs: np.ndarray,
    qualities: np.ndarray,
    outside_option: float,
    sigma_d: float,
    Q: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Finds joint profit-maximizing monopoly price vector on the discrete grid.

    Returns:
        monopoly_prices: (N,) optimal price vector
        monopoly_profits: (N,) individual stage profits under cartel
        total_profit: float aggregate industry profit
    """
    num_firms = len(costs)
    active = np.ones(num_firms, dtype=np.int32)

    # In symmetric games, joint monopoly prices are symmetric
    best_p_sym = price_grid[0]
    best_total_profit = -float("inf")
    best_profits = np.zeros(num_firms, dtype=np.float64)

    for p in price_grid:
        p_vec = np.full(num_firms, p, dtype=np.float64)
        shares = compute_market_shares(p_vec, qualities, active, outside_option, sigma_d)
        profits = compute_profits(p_vec, costs, shares, Q, active)
        tot = float(np.sum(profits))

        if tot > best_total_profit:
            best_total_profit = tot
            best_p_sym = p
            best_profits = profits

    opt_prices = np.full(num_firms, best_p_sym, dtype=np.float64)
    return opt_prices, best_profits, best_total_profit


def compute_collusion_index(
    observed_profit: float,
    nash_profit: float,
    monopoly_profit: float,
) -> float:
    """Computes the Calvano et al. collusion index: Delta = (pi_bar - pi^N) / (pi^M - pi^N).

    Delta = 0 corresponds to Bertrand-Nash competition.
    Delta = 1 corresponds to full cartel monopoly.
    """
    denom = monopoly_profit - nash_profit
    if abs(denom) < 1e-9:
        return 0.0
    delta = (observed_profit - nash_profit) / denom
    return float(delta)

