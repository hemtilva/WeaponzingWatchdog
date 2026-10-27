"""Evaluation metrics, statistical tests, and bootstrapping routines."""

from .metrics import (
    compute_weaponization_rate,
    compute_collusion_index,
    compute_social_welfare_loss,
    compute_exploitability_gain,
)
from .stats import (
    bootstrap_ci,
    compute_summary_stats,
    test_hypothesis_h1,
    test_hypothesis_h2,
    test_hypothesis_h3,
)

__all__ = [
    "compute_weaponization_rate",
    "compute_collusion_index",
    "compute_social_welfare_loss",
    "compute_exploitability_gain",
    "bootstrap_ci",
    "compute_summary_stats",
    "test_hypothesis_h1",
    "test_hypothesis_h2",
    "test_hypothesis_h3",
]
