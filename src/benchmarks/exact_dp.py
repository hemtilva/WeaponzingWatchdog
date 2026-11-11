"""Exact Best-Response Dynamic Programming (Value Iteration) solver.

Solves the reduced Markov Decision Process (MDP) for firm 1 against fixed honest
opponents to establish the theoretical exploit boundary and exploitability gain.
"""

from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict
import itertools
import ctypes
import numpy as np

from src.engine.binding import NATIVE_LIB, is_native_available

from src.engine.market import (
    compute_market_shares,
    compute_profits,
    compute_consumer_surplus,
)
from src.engine.capital import update_capital
from src.engine.rules import (
    RuleType,
    evaluate_rules,
    compute_counterfactual_harm,
)
from src.benchmarks.nash import find_discrete_nash_equilibrium


@dataclass
class DPSolverConfig:
    """Configuration for the reduced-game Dynamic Programming solver."""
    num_firms: int = 3
    price_grid: np.ndarray = None
    marginal_cost: float = 1.0
    quality: float = 2.0
    outside_option: float = 0.0
    sigma_d: float = 0.25
    market_size: float = 1.0
    fixed_cost: float = 0.2
    T_ban: int = 3
    audit_budget: int = 1
    rule_type: RuleType = RuleType.MEAN_RATIO
    rule_param: float = 0.2
    capital_levels: np.ndarray = None
    gamma: float = 0.90
    tol: float = 1e-4
    max_iter: int = 1000
    # Agent 1 traits
    alpha: float = 0.0          # Risk-neutral
    lambda_altruism: float = 0.0 # Purely selfish
    mu_ethics: float = 0.0       # Pure predator

    @classmethod
    def from_market(
        cls,
        market,
        price_grid: np.ndarray,
        rule_type: RuleType = RuleType.MEAN_RATIO,
        rule_param: Optional[float] = None,
        gamma: float = 0.95,
        **overrides,
    ) -> "DPSolverConfig":
        """Builds a DP config with the same economics as a (symmetric, N=3) MarketConfig.

        The capital grid runs 0, Phi, 2*Phi, ... up to the reserve cap (or K0), so the
        grid's top level plays the role of `capital_cap`, and a full ban moves capital
        by exactly one grid step per period.
        """
        if market.num_firms != 3:
            raise ValueError("The exact DP solver supports N = 3 only.")
        top = market.capital_cap if market.capital_cap is not None else market.initial_capital
        n_levels = int(round(top / market.fixed_cost)) + 1
        kwargs = dict(
            num_firms=3,
            price_grid=np.asarray(price_grid, dtype=np.float64),
            marginal_cost=float(market.marginal_costs[0]),
            quality=float(market.qualities[0]),
            outside_option=market.outside_option,
            sigma_d=market.price_sensitivity,
            market_size=market.market_size,
            fixed_cost=market.fixed_cost,
            T_ban=market.ban_duration,
            audit_budget=market.audit_budget,
            rule_type=rule_type,
            rule_param=market.threshold_delta if rule_param is None else rule_param,
            capital_levels=np.arange(n_levels, dtype=np.float64) * market.fixed_cost,
            gamma=gamma,
        )
        kwargs.update(overrides)
        return cls(**kwargs)


class ExactDPSolver:
    """Exact Value Iteration solver for firm 1's best response against honest rivals."""

    def __init__(self, config: DPSolverConfig):
        self.cfg = config
        self.N = config.num_firms

        if config.price_grid is None:
            self.price_grid = np.linspace(1.0, 2.5, 7)
        else:
            self.price_grid = np.asarray(config.price_grid, dtype=np.float64)

        if config.capital_levels is None:
            # Capital grid: 0, Phi, 2*Phi, ..., 10*Phi
            self.capital_levels = np.arange(11, dtype=np.float64) * config.fixed_cost
        else:
            self.capital_levels = np.asarray(config.capital_levels, dtype=np.float64)

        self.costs = np.full(self.N, config.marginal_cost, dtype=np.float64)
        self.qualities = np.full(self.N, config.quality, dtype=np.float64)

        # Precompute Bertrand-Nash prices
        nash_p, _ = find_discrete_nash_equilibrium(
            self.price_grid,
            self.costs,
            self.qualities,
            config.outside_option,
            config.sigma_d,
            config.market_size,
        )
        self.nash_prices = nash_p

        # State dimensions
        self.num_timers = config.T_ban + 1
        self.num_cap = len(self.capital_levels)
        self.num_actions = len(self.price_grid)

        # Map capital values to nearest grid indices
        self.cap_min = self.capital_levels[0]
        self.cap_max = self.capital_levels[-1]

    def _snap_capital(self, cap: float) -> int:
        """Snaps a continuous capital value to the nearest discrete grid index."""
        if cap <= 0.0:
            return 0
        idx = int(np.argmin(np.abs(self.capital_levels - cap)))
        return idx

    def solve(self) -> Tuple[np.ndarray, np.ndarray]:
        """Runs Value Iteration over the reduced state space.

        State representation: (tau_1, tau_2, tau_3, k_1, k_2, k_3)
        Returns:
            V: Value array of shape (num_timers, num_timers, num_timers, num_cap, num_cap, num_cap)
            policy: Optimal action index array of matching shape
        """
        shape = (
            self.num_timers, self.num_timers, self.num_timers,
            self.num_cap, self.num_cap, self.num_cap
        )

        if is_native_available():
            flat_v = np.zeros(np.prod(shape), dtype=np.float64)
            flat_policy = np.zeros(np.prod(shape), dtype=np.int32)

            NATIVE_LIB.solve_value_iteration_c(
                self.price_grid.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                ctypes.c_int(self.num_actions),
                self.capital_levels.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                ctypes.c_int(self.num_cap),
                ctypes.c_int(self.cfg.T_ban),
                self.costs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                self.qualities.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                ctypes.c_double(self.cfg.outside_option),
                ctypes.c_double(self.cfg.sigma_d),
                ctypes.c_double(self.cfg.market_size),
                ctypes.c_double(self.cfg.fixed_cost),
                ctypes.c_int(int(self.cfg.rule_type)),
                ctypes.c_double(self.cfg.rule_param),
                self.nash_prices.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                ctypes.c_double(self.cfg.gamma),
                ctypes.c_double(self.cfg.tol),
                ctypes.c_int(self.cfg.max_iter),
                ctypes.c_double(self.cfg.mu_ethics),
                flat_v.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                flat_policy.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            )

            return flat_v.reshape(shape), flat_policy.reshape(shape)

        V = np.zeros(shape, dtype=np.float64)
        policy = np.zeros(shape, dtype=np.int32)

        # Precompute possible audit draw subsets
        audit_draws_cache = {}
        for n_act in range(1, self.N + 1):
            sample_size = min(self.cfg.audit_budget, n_act)
            combos = list(itertools.combinations(range(n_act), sample_size))
            prob = 1.0 / len(combos)
            audit_draws_cache[n_act] = (combos, prob)

        # Value iteration loop
        for it in range(self.cfg.max_iter):
            delta = 0.0

            # Iterate over all possible states
            for t1 in range(self.num_timers):
                for t2 in range(self.num_timers):
                    for t3 in range(self.num_timers):
                        for k1 in range(self.num_cap):
                            # Absorbing state for firm 1
                            if k1 == 0:
                                continue

                            for k2 in range(self.num_cap):
                                for k3 in range(self.num_cap):
                                    old_v = V[t1, t2, t3, k1, k2, k3]

                                    # Current state vectors
                                    timers = np.array([t1, t2, t3], dtype=np.int32)
                                    cap = np.array([
                                        self.capital_levels[k1],
                                        self.capital_levels[k2],
                                        self.capital_levels[k3]
                                    ], dtype=np.float64)

                                    active_mask = np.array([
                                        1 if (timers[i] == 0 and cap[i] > 0.0) else 0
                                        for i in range(self.N)
                                    ], dtype=np.int32)

                                    if active_mask[0] == 0:
                                        # Firm 1 is currently banned
                                        # Only one passive transition: earn 0, lose fixed_cost, timer ticks down
                                        # Opponents play Nash if active
                                        p_joint = self.nash_prices.copy()
                                        shares = compute_market_shares(
                                            p_joint, self.qualities, active_mask,
                                            self.cfg.outside_option, self.cfg.sigma_d
                                        )
                                        profits = compute_profits(p_joint, self.costs, shares, self.cfg.market_size, active_mask)
                                        next_cap = update_capital(cap, profits, self.cfg.fixed_cost)
                                        next_k = [self._snap_capital(c) for c in next_cap]

                                        # Update timers for rivals
                                        # Sample audit among active rivals
                                        act_indices = [i for i in range(self.N) if active_mask[i] != 0]
                                        n_act = len(act_indices)

                                        exp_next_v = 0.0
                                        if n_act >= 3:
                                            flags = evaluate_rules(p_joint, active_mask, self.cfg.rule_type, self.cfg.rule_param)
                                            combos, prob = audit_draws_cache[n_act]
                                            for combo in combos:
                                                audited = np.zeros(self.N, dtype=np.int32)
                                                for c_idx in combo:
                                                    audited[act_indices[c_idx]] = 1
                                                next_timers = np.array([
                                                    self.cfg.T_ban if (audited[i] and flags[i]) else max(0, timers[i] - 1)
                                                    for i in range(self.N)
                                                ], dtype=np.int32)
                                                exp_next_v += prob * V[
                                                    next_timers[0], next_timers[1], next_timers[2],
                                                    next_k[0], next_k[1], next_k[2]
                                                ]
                                        else:
                                            # N_act < 3, no flags
                                            next_timers = np.maximum(0, timers - 1)
                                            exp_next_v = V[
                                                next_timers[0], next_timers[1], next_timers[2],
                                                next_k[0], next_k[1], next_k[2]
                                            ]

                                        new_v = self.cfg.gamma * exp_next_v
                                        V[t1, t2, t3, k1, k2, k3] = new_v
                                        delta = max(delta, abs(new_v - old_v))
                                    else:
                                        # Firm 1 is active! Maximize over action grid
                                        best_val = -float("inf")
                                        best_action = 0

                                        for a_idx, p1 in enumerate(self.price_grid):
                                            p_joint = np.array([p1, self.nash_prices[1], self.nash_prices[2]], dtype=np.float64)
                                            shares = compute_market_shares(
                                                p_joint, self.qualities, active_mask,
                                                self.cfg.outside_option, self.cfg.sigma_d
                                            )
                                            profits = compute_profits(p_joint, self.costs, shares, self.cfg.market_size, active_mask)
                                            next_cap = update_capital(cap, profits, self.cfg.fixed_cost)
                                            next_k = [self._snap_capital(c) for c in next_cap]

                                            # Immediate reward
                                            profit_1 = profits[0]
                                            reward_1 = profit_1 / self.cfg.market_size

                                            act_indices = [i for i in range(self.N) if active_mask[i] != 0]
                                            n_act = len(act_indices)

                                            exp_next_v = 0.0
                                            if n_act >= 3:
                                                flags = evaluate_rules(p_joint, active_mask, self.cfg.rule_type, self.cfg.rule_param)
                                                combos, prob = audit_draws_cache[n_act]
                                                for combo in combos:
                                                    audited = np.zeros(self.N, dtype=np.int32)
                                                    for c_idx in combo:
                                                        audited[act_indices[c_idx]] = 1

                                                    # Counterfactual harm penalty if mu > 0
                                                    harm_pen = 0.0
                                                    if self.cfg.mu_ethics > 0.0:
                                                        harm = compute_counterfactual_harm(
                                                            p_joint, self.nash_prices, active_mask, audited,
                                                            self.cfg.rule_type, self.cfg.rule_param
                                                        )
                                                        harm_pen = self.cfg.mu_ethics * harm[0]

                                                    next_timers = np.array([
                                                        self.cfg.T_ban if (audited[i] and flags[i]) else max(0, timers[i] - 1)
                                                        for i in range(self.N)
                                                    ], dtype=np.int32)
                                                    exp_next_v += prob * (
                                                        V[next_timers[0], next_timers[1], next_timers[2],
                                                          next_k[0], next_k[1], next_k[2]] - harm_pen
                                                    )
                                            else:
                                                next_timers = np.maximum(0, timers - 1)
                                                exp_next_v = V[
                                                    next_timers[0], next_timers[1], next_timers[2],
                                                    next_k[0], next_k[1], next_k[2]
                                                ]

                                            q_val = reward_1 + self.cfg.gamma * exp_next_v
                                            if q_val > best_val:
                                                best_val = q_val
                                                best_action = a_idx

                                        V[t1, t2, t3, k1, k2, k3] = best_val
                                        policy[t1, t2, t3, k1, k2, k3] = best_action
                                        delta = max(delta, abs(best_val - old_v))

            if delta < self.cfg.tol:
                break

        return V, policy

    def compute_exploitability_gain(
        self,
        V_optimal: np.ndarray,
        initial_capital: float = 1.0,
    ) -> float:
        """Computes exploitability gain: E_1 = V*(initial) - V_honest(initial)."""
        k0_idx = self._snap_capital(initial_capital)
        v_opt = V_optimal[0, 0, 0, k0_idx, k0_idx, k0_idx]

        # Compute honest value where firm 1 always prices at Bertrand-Nash
        nash_idx = int(np.argmin(np.abs(self.price_grid - self.nash_prices[0])))

        # Approximate infinite-horizon honest competitive value
        shares = compute_market_shares(
            self.nash_prices, self.qualities, np.ones(self.N, dtype=np.int32),
            self.cfg.outside_option, self.cfg.sigma_d
        )
        nash_profit = compute_profits(self.nash_prices, self.costs, shares, self.cfg.market_size)[0]
        r = nash_profit / self.cfg.market_size
        g = self.cfg.gamma
        # At Nash no bans are triggered. If Nash profit covers the fixed cost, the honest
        # firm earns r forever. Otherwise its capital runs down and it exits after T
        # periods; valuing that as a perpetuity would overstate the honest baseline.
        net_loss = self.cfg.fixed_cost - nash_profit
        if net_loss <= 1e-12:
            v_honest = r / (1.0 - g)
        else:
            T = int(np.ceil(self.capital_levels[k0_idx] / net_loss))
            v_honest = r * (1.0 - g ** T) / (1.0 - g)

        return float(v_opt - v_honest)
