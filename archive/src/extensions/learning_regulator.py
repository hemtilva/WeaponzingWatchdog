"""Extension A: Slow-Timescale Adaptive Regulator.

Bi-level dynamics. On the fast timescale a learning attacker (tabular Q-learner)
prices against honest Nash rivals under the current rule. On the slow timescale,
once per macro-epoch, the regulator adjusts the threshold delta from what it
actually observed in that epoch:

  * false positives  = sanctions of firms whose flag was caused by a rival's
                       deviation (induced sanctions) -> threshold too tight, loosen
  * price dispersion = spread of firms' prices -> evidence of outliers, tighten

Objective reported per epoch: R = mean(CS) - kappa * audits/period - eta * FP/period.
The attacker's Q-table persists across epochs, so it re-adapts to each new delta.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import numpy as np

from src.engine.market import MarketConfig
from src.engine.rules import RuleType
from src.benchmarks.exact_dp import DPSolverConfig, ExactDPSolver
from src.benchmarks.diagnostics import framing_market
from src.agents.q_learning import QLearningConfig
from src.agents.trainer import SimulationConfig, MultiAgentTrainer


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
    fp_weight: float = 2.0            # Loosening pressure per false positive per period
    dispersion_weight: float = 0.10   # Tightening pressure per unit of price dispersion


class SlowTimescaleRegulator:
    """Bi-level regulator adjusting inspection threshold delta over macro-epochs."""

    def __init__(self, config: AdaptiveRegulatorConfig):
        self.cfg = config
        self.current_delta = float(config.delta_init)

        self.epoch_cs: List[float] = []
        self.epoch_audits: List[float] = []
        self.epoch_false_positives: List[float] = []
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
        """Records one period's marketplace outcomes."""
        self.epoch_cs.append(float(consumer_surplus))
        self.epoch_audits.append(float(num_audited))
        self.epoch_false_positives.append(float(false_positive_count))
        self.epoch_dispersions.append(float(np.std(prices)))

    def observe_epoch(
        self,
        mean_consumer_surplus: float,
        audits_per_period: float,
        false_positives_per_period: float,
        mean_dispersion: float,
    ):
        """Records an epoch's per-period averages (equivalent to many observe_period calls)."""
        self.epoch_cs.append(float(mean_consumer_surplus))
        self.epoch_audits.append(float(audits_per_period))
        self.epoch_false_positives.append(float(false_positives_per_period))
        self.epoch_dispersions.append(float(mean_dispersion))

    def step_macro_epoch(self) -> float:
        """Evaluates the epoch objective and moves delta.

        gradient = fp_weight * FP/period - dispersion_weight * dispersion
        """
        if len(self.epoch_cs) == 0:
            return self.current_delta

        avg_cs = float(np.mean(self.epoch_cs))
        avg_aud = float(np.mean(self.epoch_audits))
        avg_fp = float(np.mean(self.epoch_false_positives))
        avg_disp = float(np.mean(self.epoch_dispersions))

        regulator_obj = avg_cs - self.cfg.audit_cost_kappa * avg_aud - self.cfg.fp_penalty_eta * avg_fp
        self.history_objectives.append(regulator_obj)

        gradient = self.cfg.fp_weight * avg_fp - self.cfg.dispersion_weight * avg_disp
        updated_delta = self.current_delta + self.cfg.learning_rate * gradient
        self.current_delta = float(np.clip(updated_delta, self.cfg.delta_min, self.cfg.delta_max))
        self.history_deltas.append(self.current_delta)

        self.epoch_cs.clear()
        self.epoch_audits.clear()
        self.epoch_false_positives.clear()
        self.epoch_dispersions.clear()

        return self.current_delta


def evaluate_checkpoint_exploitability(
    frozen_delta: float,
    t_ban: int = 3,
    fixed_cost: float = 0.20,
    k0_val: float = 0.60,
    rule_type: RuleType = RuleType.MEAN_RATIO,
) -> Dict[str, Any]:
    """Exact best-response exploitability gain against a frozen regulator checkpoint (N = 3).

    Uses the pure-Python DP fallback unless the native engine is compiled; that is
    slow (minutes), so callers in loops should make it optional.
    """
    price_grid = np.array([1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5], dtype=np.float64)
    capital_levels = np.arange(7, dtype=np.float64) * fixed_cost

    config = DPSolverConfig(
        num_firms=3,
        price_grid=price_grid,
        marginal_cost=1.0,
        quality=2.0,
        outside_option=0.0,
        sigma_d=0.25,
        market_size=1.0,
        fixed_cost=fixed_cost,
        T_ban=t_ban,
        audit_budget=1,
        rule_type=rule_type,
        rule_param=frozen_delta,
        capital_levels=capital_levels,
        gamma=0.90,
        tol=1e-3,
        max_iter=80,
    )
    solver = ExactDPSolver(config)
    V, policy = solver.solve()
    gain = solver.compute_exploitability_gain(V, initial_capital=k0_val)

    k0_idx = solver._snap_capital(k0_val)
    opt_action_idx = policy[0, 0, 0, k0_idx, k0_idx, k0_idx]
    opt_p = float(price_grid[opt_action_idx])
    nash_p = float(solver.nash_prices[0])

    return {
        "frozen_delta": float(frozen_delta),
        "exploitability_gain": float(gain),
        "optimal_price": opt_p,
        "nash_price": nash_p,
        "is_framing": bool(opt_p < nash_p),
    }


def _attacker_summary(trainer: MultiAgentTrainer, res, frozen_delta: float) -> Dict[str, Any]:
    nash_p = float(trainer.nash_prices[0])
    return {
        "frozen_delta": float(frozen_delta),
        "attacker_mean_price": float(res.mean_prices[0]),
        "nash_price": nash_p,
        "attacker_mean_profit": float(res.mean_profits[0]),
        "nash_profit": float(trainer.nash_profits[0]),
        "weaponization_rate": float(res.weaponization_rate),
        "total_fresh_bans": int(res.total_fresh_bans),
        "induced_bans": int(res.induced_bans),
        "induced_bankruptcies": int(res.induced_bankruptcies),
        "kills_per_episode": float(res.kills_by[0]) / max(1, res.episodes_in_eval),
        "framing_detected": bool(res.kills_by[0] > 0 and res.mean_prices[0] < nash_p),
    }


def train_adversarial_attacker(
    frozen_delta: float,
    steps: int = 20_000,
    episode_length: int = 500,
    market: Optional[MarketConfig] = None,
    seed: int = 42,
    rule_type: RuleType = RuleType.MEAN_RATIO,
    q_config: Optional[QLearningConfig] = None,
) -> Dict[str, Any]:
    """Trains a Q-learning attacker (firm 0) against honest Nash-pricing rivals under a frozen delta."""
    market = market or framing_market(num_firms=3, lethal=True, threshold_delta=frozen_delta)
    cfg = SimulationConfig(
        market=market,
        q_learning=q_config or QLearningConfig(),
        total_steps=steps,
        warmup_steps=0,
        episode_length=episode_length,
        rule_type=rule_type,
        rule_param=frozen_delta,
        eval_window=max(2_000, steps // 4),
        agent_type="learner_vs_honest",
        seed=seed,
    )
    trainer = MultiAgentTrainer(cfg)
    res = trainer.run()
    return _attacker_summary(trainer, res, frozen_delta)


def simulate_bi_level_dynamics(
    num_epochs: int = 12,
    steps_per_epoch: int = 20_000,
    delta_init: float = 0.10,
    seed: int = 42,
    market: Optional[MarketConfig] = None,
    regulator_config: Optional[AdaptiveRegulatorConfig] = None,
    pretrain_steps: int = 0,
    q_config: Optional[QLearningConfig] = None,
    use_dp: bool = False,
) -> Dict[str, Any]:
    """Co-evolves a persistent learning attacker and the slow-timescale regulator.

    Each epoch runs `steps_per_epoch` market periods at the current delta, measures
    false positives (induced sanctions) and price dispersion, then updates delta.
    `pretrain_steps` lets the attacker learn at delta_init before the regulator moves.
    """
    reg_cfg = regulator_config or AdaptiveRegulatorConfig(
        macro_epoch=steps_per_epoch,
        delta_init=delta_init,
        delta_min=0.05,
        delta_max=0.60,
        learning_rate=0.04,
    )
    regulator = SlowTimescaleRegulator(reg_cfg)
    market = market or framing_market(num_firms=3, lethal=True, threshold_delta=delta_init)

    sim_cfg = SimulationConfig(
        market=market,
        q_learning=q_config or QLearningConfig(),
        total_steps=max(1, pretrain_steps),
        warmup_steps=0,
        episode_length=500,
        rule_type=RuleType.MEAN_RATIO,
        rule_param=regulator.current_delta,
        eval_window=max(1, pretrain_steps),
        agent_type="learner_vs_honest",
        seed=seed,
    )
    trainer = MultiAgentTrainer(sim_cfg)
    if pretrain_steps > 0:
        trainer.run()

    records = []
    for ep in range(num_epochs):
        cur_delta = regulator.current_delta
        trainer.cfg.rule_param = cur_delta
        trainer.cfg.total_steps = steps_per_epoch
        trainer.cfg.eval_window = steps_per_epoch
        res = trainer.run()

        periods = max(1, res.eval_periods)
        fp_rate = res.induced_sanctions / periods
        audit_rate = market.audit_budget * 1.0
        dispersion = float(np.std(res.mean_prices))
        regulator.observe_epoch(res.mean_consumer_surplus, audit_rate, fp_rate, dispersion)
        new_delta = regulator.step_macro_epoch()

        rec = _attacker_summary(trainer, res, cur_delta)
        rec.update({
            "epoch": ep,
            "delta": cur_delta,
            "new_delta": new_delta,
            "regulator_objective": regulator.history_objectives[-1],
            "false_positives_per_period": fp_rate,
            "price_dispersion": dispersion,
            "epsilon": float(trainer.agents[0].epsilon),
        })
        if use_dp:
            dp = evaluate_checkpoint_exploitability(frozen_delta=cur_delta)
            rec["exploitability_gain"] = dp["exploitability_gain"]
            rec["dp_is_framing"] = dp["is_framing"]
        records.append(rec)

    return {
        "epochs": records,
        "final_delta": regulator.current_delta,
        "history_deltas": regulator.history_deltas,
    }
