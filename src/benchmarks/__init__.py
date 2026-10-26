"""Theoretical benchmarks: Bertrand-Nash, Joint Monopoly, and Exact Dynamic Programming."""

from .nash import (
    compute_continuous_nash_price,
    find_discrete_nash_equilibrium,
)
from .monopoly import (
    find_discrete_monopoly_prices,
    compute_collusion_index,
)
from .exact_dp import (
    DPSolverConfig,
    ExactDPSolver,
)

__all__ = [
    "compute_continuous_nash_price",
    "find_discrete_nash_equilibrium",
    "find_discrete_monopoly_prices",
    "compute_collusion_index",
    "DPSolverConfig",
    "ExactDPSolver",
]
