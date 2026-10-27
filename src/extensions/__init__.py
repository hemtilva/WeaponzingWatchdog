"""Advanced research extensions: learning regulator and non-stationary demand shocks."""

from .demand_shock import DemandShockConfig, DemandShockEngine
from .learning_regulator import AdaptiveRegulatorConfig, SlowTimescaleRegulator

__all__ = [
    "DemandShockConfig",
    "DemandShockEngine",
    "AdaptiveRegulatorConfig",
    "SlowTimescaleRegulator",
]
