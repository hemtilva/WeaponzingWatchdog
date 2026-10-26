"""Bertrand-Nash stage game equilibrium solver for logit oligopoly."""

from typing import Tuple, Optional
import numpy as np
from scipy.optimize import root_scalar

from src.engine.market import compute_market_shares, compute_profits


def compute_continuous_nash_price(
    cost: float,
    quality: float,
    outside_option: float,
    sigma_d: float,
    num_firms: int,
) -> float:
    """Computes symmetric continuous Bertrand-Nash equilibrium price using FOC fixed point.

    First-order condition: p_i^N = c_i + sigma_d / (1 - s_i(p^N)).
    """
    def foc_residual(p: float) -> float:
        prices = np.full(num_firms, p, dtype=np.float64)
        qualities = np.full(num_firms, quality, dtype=np.float64)
        active = np.ones(num_firms, dtype=np.int32)
        shares = compute_market_shares(prices, qualities, active, outside_option, sigma_d)
        share = shares[0]
        # p - (c + sigma_d / (1 - share)) = 0
        return p - (cost + sigma_d / (1.0 - share))

    res = root_scalar(foc_residual, bracket=[cost, cost + 20.0 * sigma_d], method="brentq")
    return float(res.root)


def find_discrete_nash_equilibrium(
    price_grid: np.ndarray,
    costs: np.ndarray,
    qualities: np.ndarray,
    outside_option: float,
    sigma_d: float,
    Q: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Finds pure-strategy Nash equilibrium on the discrete price grid.

    Returns:
        nash_prices: (N,) optimal discrete price vector
        nash_profits: (N,) stage profits at equilibrium
    """
    num_firms = len(costs)
    active = np.ones(num_firms, dtype=np.int32)

    # For symmetric markets, search symmetric profiles first
    best_symmetric = None
    for p in price_grid:
        p_vec = np.full(num_firms, p, dtype=np.float64)
        shares = compute_market_shares(p_vec, qualities, active, outside_option, sigma_d)
        profits = compute_profits(p_vec, costs, shares, Q, active)

        # Check if firm 0 has any profitable unilateral deviation
        is_nash = True
        for dev_p in price_grid:
            dev_vec = p_vec.copy()
            dev_vec[0] = dev_p
            dev_shares = compute_market_shares(dev_vec, qualities, active, outside_option, sigma_d)
            dev_profit = compute_profits(dev_vec, costs, dev_shares, Q, active)[0]
            if dev_profit > profits[0] + 1e-9:
                is_nash = False
                break

        if is_nash:
            return p_vec, profits

    # Fallback to general best-response dynamics from continuous Nash rounding
    c_nash = compute_continuous_nash_price(costs[0], qualities[0], outside_option, sigma_d, num_firms)
    closest_idx = np.argmin(np.abs(price_grid - c_nash))
    current_prices = np.full(num_firms, price_grid[closest_idx], dtype=np.float64)

    for _ in range(50):
        changed = False
        for i in range(num_firms):
            best_p = current_prices[i]
            # Current profit
            cur_shares = compute_market_shares(current_prices, qualities, active, outside_option, sigma_d)
            best_profit = compute_profits(current_prices, costs, cur_shares, Q, active)[i]

            for cand_p in price_grid:
                test_prices = current_prices.copy()
                test_prices[i] = cand_p
                test_shares = compute_market_shares(test_prices, qualities, active, outside_option, sigma_d)
                cand_profit = compute_profits(test_prices, costs, test_shares, Q, active)[i]

                if cand_profit > best_profit + 1e-9:
                    best_profit = cand_profit
                    best_p = cand_p
                    changed = True

            current_prices[i] = best_p

        if not changed:
            break

    final_shares = compute_market_shares(current_prices, qualities, active, outside_option, sigma_d)
    final_profits = compute_profits(current_prices, costs, final_shares, Q, active)
    return current_prices, final_profits
