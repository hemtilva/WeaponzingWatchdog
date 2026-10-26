"""Theoretical benchmarks: Bertrand-Nash, Joint Monopoly, and Collusion Index."""

from .nash import (
    compute_continuous_nash_price,
    find_discrete_nash_equilibrium,
)
from .monopoly import (
    find_discrete_monopoly_prices,
    compute_collusion_index,
)

__all__ = [
    "compute_continuous_nash_price",
    "find_discrete_nash_equilibrium",
    "find_discrete_monopoly_prices",
    "compute_collusion_index",
]
