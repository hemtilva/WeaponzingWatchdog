"""Extension A: Slow-Timescale Adaptive Regulator.

Operates over macro-epochs K_macro >> 1 based on history features H_t
(price dispersion, ban frequency, price drift) to dynamically optimize threshold delta:
max_phi E[ sum_t gamma_R^t (CS(p_t) - kappa * |Aud_t| - eta * FP_t) ].
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import numpy as np


@dataclass
class AdaptiveRegulatorConfig:
    """Hyperparameters for the bi-level adaptive regulator."""
    macro_epoch: int = 500           # K_macro periods per update
    delta_init: float = 0.20
    delta_min: float = 0.05
    delta_max: float = 0.60
    learning_rate: float = 0.02
    audit_cost_kappa: float = 0.05    # Cost per audit
    fp_penalty_eta: float = 0.25      # Penalty for false positive innocent bans
    gamma_regulator: float = 0.95


class SlowTimescaleRegulator:
    """Bi-level regulator adjusting inspection threshold delta over macro-epochs."""

    def __init__(self, config: AdaptiveRegulatorConfig):
        self.cfg = config
        self.current_delta = float(config.delta_init)

        # Macro-epoch metrics accumulation
        self.epoch_cs: List[float] = []
        self.epoch_audits: List[int] = []
        self.epoch_false_positives: List[int] = []
        self.epoch_dispersions: List[float] = []
        self.history_deltas: List[float] = [self.current_delta]
        self.history_objectives: List[float] = []

    def observe_period(
        self,
        consumer_surplus: float,
        num_audited: int,
        prices: np.ndarray,
        false_positive_count: int = 0,
    ):
        """Records per-period marketplace outcomes."""
        self.epoch_cs.append(consumer_surplus)
        self.epoch_audits.append(num_audited)
        self.epoch_false_positives.append(false_positive_count)
        self.epoch_dispersions.append(float(np.std(prices)))

    def step_macro_epoch(self) -> float:
        """Evaluates macro-epoch objective and applies policy gradient / bandit step.

        Objective: R_R = mean(CS) - kappa * mean(Aud) - eta * mean(FP).
        """
        if len(self.epoch_cs) == 0:
            return self.current_delta

        avg_cs = float(np.mean(self.epoch_cs))
        avg_aud = float(np.mean(self.epoch_audits))
        avg_fp = float(np.mean(self.epoch_false_positives))
        avg_disp = float(np.mean(self.epoch_dispersions))

        regulator_obj = avg_cs - self.cfg.audit_cost_kappa * avg_aud - self.cfg.fp_penalty_eta * avg_fp
        self.history_objectives.append(regulator_obj)

        # Gradient heuristic:
        # If false positives are high, threshold is too tight -> increase delta.
        # If price dispersion is low and FP is low, tighten threshold to protect consumers -> decrease delta.
        gradient = (avg_fp * 2.0) - (0.10 * avg_disp)
        updated_delta = self.current_delta + self.cfg.learning_rate * gradient
        self.current_delta = float(np.clip(updated_delta, self.cfg.delta_min, self.cfg.delta_max))
        self.history_deltas.append(self.current_delta)

        # Clear accumulation buffers for next epoch
        self.epoch_cs.clear()
        self.epoch_audits.clear()
        self.epoch_false_positives.clear()
        self.epoch_dispersions.clear()

        return self.current_delta
