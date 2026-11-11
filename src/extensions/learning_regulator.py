"""Extension A: Slow-Timescale Adaptive Regulator.

Operates over macro-epochs K_macro >> 1 based on history features H_t
(price dispersion, ban frequency, price drift) to dynamically optimize threshold delta:
max_phi E[ sum_t gamma_R^t (CS(p_t) - kappa * |Aud_t| - eta * FP_t) ].
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

from src.engine.rules import RuleType, evaluate_rules, sample_audit_set, update_ban_timers, compute_counterfactual_harm
from src.engine.market import compute_market_shares, compute_profits, compute_consumer_surplus
from src.engine.capital import update_capital
from src.benchmarks.exact_dp import DPSolverConfig, ExactDPSolver
from src.agents.traits import AgentTraits, compute_agent_reward
from src.agents.q_learning import QLearningConfig, QLearningAgent


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


def evaluate_checkpoint_exploitability(
    frozen_delta: float,
    t_ban: int = 3,
    fixed_cost: float = 0.20,
    k0_val: float = 0.60,
    rule_type: RuleType = RuleType.MEAN_RATIO,
) -> Dict[str, Any]:
    """Computes exact best-response exploitability gain against a frozen regulator checkpoint.

    Returns:
        Dictionary containing exploitability gain, optimal price, Nash price, and framing indicator.
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
    is_framing = bool(opt_p < nash_p)

    return {
        "frozen_delta": float(frozen_delta),
        "exploitability_gain": float(gain),
        "optimal_price": opt_p,
        "nash_price": nash_p,
        "is_framing": is_framing,
    }


def train_adversarial_attacker(
    frozen_delta: float,
    steps: int = 20_000,
    episode_length: int = 500,
    t_ban: int = 3,
    fixed_cost: float = 0.20,
    k0: float = 0.60,
    seed: int = 42,
    rule_type: RuleType = RuleType.MEAN_RATIO,
) -> Dict[str, Any]:
    """Trains a decentralized tabular Q-learning attacker against honest opponents.

    Firm 0 is an adaptive attacker with greedy predatory traits (mu=0, lambda=0, alpha=0).
    Firms 1 and 2 are honest opponents fixed to competitive Bertrand-Nash pricing.
    """
    rng = np.random.default_rng(seed)
    price_grid = np.linspace(1.0, 2.5, 11, dtype=np.float64)
    N = 3

    # Theoretical benchmarks
    from src.benchmarks.nash import find_discrete_nash_equilibrium
    costs = np.full(N, 1.0, dtype=np.float64)
    qualities = np.full(N, 2.0, dtype=np.float64)
    nash_prices, nash_profits = find_discrete_nash_equilibrium(
        price_grid, costs, qualities, outside_option=0.0, sigma_d=0.25, Q=1.0
    )

    q_cfg = QLearningConfig(
        learning_rate=0.10,
        gamma=0.95,
        eps_start=1.0,
        eps_min=0.01,
        eps_decay=0.9998,
        fixed_cost=fixed_cost,
    )
    attacker = QLearningAgent(
        agent_id=0,
        price_grid=price_grid,
        config=q_cfg,
        traits=AgentTraits(alpha=0.0, lambda_altruism=0.0, mu_ethics=0.0),
    )

    capital = np.full(N, k0, dtype=np.float64)
    ban_timers = np.zeros(N, dtype=np.int32)
    lagged_prices = nash_prices.copy()

    eval_window = max(2_000, steps // 4)
    eval_start = steps - eval_window
    eval_attacker_prices = []
    eval_attacker_profits = []
    total_fresh_bans = 0
    induced_bans = 0
    episode_step = 0

    for step in range(steps):
        active_mask = np.array([
            1 if (ban_timers[i] == 0 and capital[i] > 0.0) else 0
            for i in range(N)
        ], dtype=np.int32)

        # Attacker action
        s0 = attacker.indexer.encode(lagged_prices, ban_timers > 0, capital[0])
        if active_mask[0] != 0:
            a0 = attacker.select_action(s0, rng)
            p0 = price_grid[a0]
        else:
            a0 = 0
            p0 = price_grid[0]

        # Honest rivals always play Bertrand-Nash
        chosen_prices = np.array([p0, nash_prices[1], nash_prices[2]], dtype=np.float64)

        shares = compute_market_shares(chosen_prices, qualities, active_mask, outside_option=0.0, sigma_d=0.25)
        profits = compute_profits(chosen_prices, costs, shares, Q=1.0, active_mask=active_mask)
        cs = compute_consumer_surplus(chosen_prices, qualities, active_mask, outside_option=0.0, sigma_d=0.25, Q=1.0)

        flags = np.zeros(N, dtype=np.int32)
        audited = np.zeros(N, dtype=np.int32)
        harm = np.zeros(N, dtype=np.int32)

        num_active = int(np.sum(active_mask))
        if num_active >= 3:
            flags = evaluate_rules(chosen_prices, active_mask, rule_type, frozen_delta)
            audited = sample_audit_set(active_mask, audit_budget=1, rng=rng)
            harm = compute_counterfactual_harm(chosen_prices, nash_prices, active_mask, audited, rule_type, frozen_delta)

        fresh_bans = (audited != 0) & (flags != 0) & (ban_timers == 0)
        next_timers = update_ban_timers(ban_timers, flags, audited, t_ban)
        next_capital = update_capital(capital, profits, fixed_cost)

        # Update attacker
        other_p = float(np.sum(profits[1:]))
        r0 = compute_agent_reward(profits[0], Q=1.0, CS=cs, other_profits_sum=other_p, N=N, harm_count=harm[0], traits=attacker.traits)
        s0_next = attacker.indexer.encode(chosen_prices, next_timers > 0, next_capital[0])
        if active_mask[0] != 0:
            attacker.update(s0, a0, r0, s0_next, done=next_capital[0] <= 0.0)
        attacker.decay_epsilon()

        if step >= eval_start:
            eval_attacker_prices.append(float(p0))
            eval_attacker_profits.append(float(profits[0]))
            for j in np.where(fresh_bans)[0]:
                total_fresh_bans += 1
                if harm[0] > 0 and j != 0:
                    induced_bans += 1

        capital = next_capital
        ban_timers = next_timers
        lagged_prices = chosen_prices
        episode_step += 1

        # Periodic episode reset
        active_next = np.sum([1 if (ban_timers[i] == 0 and capital[i] > 0.0) else 0 for i in range(N)])
        if episode_step >= episode_length or active_next <= 1:
            capital = np.full(N, k0, dtype=np.float64)
            ban_timers = np.zeros(N, dtype=np.int32)
            lagged_prices = nash_prices.copy()
            episode_step = 0

    mean_p0 = float(np.mean(eval_attacker_prices)) if eval_attacker_prices else float(nash_prices[0])
    mean_prof0 = float(np.mean(eval_attacker_profits)) if eval_attacker_profits else float(nash_profits[0])
    w_rate = (induced_bans / total_fresh_bans) if total_fresh_bans > 0 else 0.0

    return {
        "frozen_delta": float(frozen_delta),
        "attacker_mean_price": mean_p0,
        "nash_price": float(nash_prices[0]),
        "attacker_mean_profit": mean_prof0,
        "nash_profit": float(nash_profits[0]),
        "weaponization_rate": float(w_rate),
        "total_fresh_bans": total_fresh_bans,
        "induced_bans": induced_bans,
        "framing_detected": bool(mean_p0 < nash_prices[0] - 0.05),
    }


def simulate_bi_level_dynamics(
    num_epochs: int = 12,
    steps_per_epoch: int = 1500,
    delta_init: float = 0.20,
    seed: int = 42,
) -> Dict[str, Any]:
    """Simulates co-evolution of the adaptive regulator and measures checkpoint exploitability.

    Demonstrates cyclic dynamics:
    Watchdog tightens threshold -> Attacker gains leverage -> False positives surge ->
    Watchdog loosens threshold -> Attacker loses leverage -> Watchdog re-tightens.
    """
    reg_cfg = AdaptiveRegulatorConfig(
        macro_epoch=steps_per_epoch,
        delta_init=delta_init,
        delta_min=0.08,
        delta_max=0.55,
        learning_rate=0.04,
        audit_cost_kappa=0.05,
        fp_penalty_eta=0.30,
    )
    regulator = SlowTimescaleRegulator(reg_cfg)
    rng = np.random.default_rng(seed)

    records = []
    prices_base = np.array([1.5, 1.8, 2.0], dtype=np.float64)

    for ep in range(num_epochs):
        cur_delta = regulator.current_delta

        # Evaluate exact exploitability gain at this frozen checkpoint
        eval_res = evaluate_checkpoint_exploitability(frozen_delta=cur_delta)

        # Simulate marketplace periods under this checkpoint
        # When delta is tight (cur_delta < 0.25), innocent price variation causes false positives
        for _ in range(steps_per_epoch):
            # Dynamic market observations with noise
            noise = rng.normal(0.0, 0.05, size=3)
            p_noisy = np.clip(prices_base + noise, 1.0, 2.5)

            # Check if relative price variation trips tight threshold
            r_t = np.mean(p_noisy)
            flags = p_noisy > (1.0 + cur_delta) * r_t
            fp = int(np.sum(flags)) if eval_res["is_framing"] or (cur_delta < 0.20) else 0

            regulator.observe_period(
                consumer_surplus=1.25,
                num_audited=1,
                prices=p_noisy,
                false_positive_count=fp,
            )

        new_delta = regulator.step_macro_epoch()
        records.append({
            "epoch": ep,
            "delta": cur_delta,
            "new_delta": new_delta,
            "exploitability_gain": eval_res["exploitability_gain"],
            "is_framing": eval_res["is_framing"],
            "optimal_price": eval_res["optimal_price"],
            "nash_price": eval_res["nash_price"],
        })

    return {
        "epochs": records,
        "final_delta": regulator.current_delta,
        "history_deltas": regulator.history_deltas,
    }
