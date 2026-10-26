"""Market clearinghouse, capital dynamics, and regulatory rule engine."""

from .market import (
    MarketConfig,
    compute_market_shares,
    compute_profits,
    compute_consumer_surplus,
)
from .capital import update_capital
from .rules import (
    RuleType,
    evaluate_rules,
    sample_audit_set,
    update_ban_timers,
    compute_counterfactual_harm,
)

__all__ = [
    "MarketConfig",
    "compute_market_shares",
    "compute_profits",
    "compute_consumer_surplus",
    "update_capital",
    "RuleType",
    "evaluate_rules",
    "sample_audit_set",
    "update_ban_timers",
    "compute_counterfactual_harm",
]

