"""Multi-agent continuous Deep Q-Network (DQN) trainer for scalable N-firm markets.

Allows seamless scalability to N >= 5 firms where tabular state spaces exceed 1.6M states,
evaluating neural network function approximation under antitrust regulation.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np
import torch

from src.engine.market import (
    MarketConfig,
    compute_market_shares,
    compute_profits,
    compute_consumer_surplus,
)
from src.engine.capital import update_capital
from src.engine.rules import (
    RuleType,
    evaluate_rules,
    compute_counterfactual_harm,
    update_ban_timers,
    sample_audit_set,
)
from src.benchmarks.nash import find_discrete_nash_equilibrium
from src.benchmarks.monopoly import find_discrete_monopoly_prices, compute_collusion_index
from .traits import AgentTraits, compute_agent_reward
from .trainer import SimulationResults
from .dqn_agent import DQNAgent, DQNConfig, ContinuousStateEncoder


@dataclass
class DQNSimulationConfig:
    """Configuration for continuous Deep Q-Network simulation."""
    market: MarketConfig = field(default_factory=MarketConfig)
    dqn_config: DQNConfig = field(default_factory=DQNConfig)
    price_grid: Optional[np.ndarray] = None
    total_steps: int = 5_000
    warmup_steps: int = 200
    episode_length: int = 200
    rule_type: RuleType = RuleType.MEAN_RATIO
    rule_param: float = 0.20
    agent_traits: Optional[List[AgentTraits]] = None
    eval_window: int = 1_000
    agent_type: str = "adaptive"  # "adaptive", "scripted_attacker"
    penalty_mode: str = "shutdown"  # "shutdown" or "escrow"
    ema_beta: float = 0.95
    seed: int = 42


class DQNTrainer:
    """Orchestrates multi-firm Deep Q-Network agents in continuous state space."""

    def __init__(self, config: DQNSimulationConfig):
        self.cfg = config
        self.m_cfg = config.market
        self.N = self.m_cfg.num_firms
        self.rng = np.random.default_rng(config.seed)

        if config.price_grid is None:
            # Standard M = 7 or 11 price levels
            self.price_grid = np.linspace(
                self.m_cfg.marginal_costs[0],
                self.m_cfg.marginal_costs[0] + 6.0 * self.m_cfg.price_sensitivity,
                7,
                dtype=np.float64,
            )
        else:
            self.price_grid = np.asarray(config.price_grid, dtype=np.float64)

        # Precompute benchmarks
        self.nash_prices, self.nash_profits = find_discrete_nash_equilibrium(
            self.price_grid,
            self.m_cfg.marginal_costs,
            self.m_cfg.qualities,
            self.m_cfg.outside_option,
            self.m_cfg.price_sensitivity,
            self.m_cfg.market_size,
        )
        self.mono_prices, self.mono_profits, _ = find_discrete_monopoly_prices(
            self.price_grid,
            self.m_cfg.marginal_costs,
            self.m_cfg.qualities,
            self.m_cfg.outside_option,
            self.m_cfg.price_sensitivity,
            self.m_cfg.market_size,
        )

        # Initialize N continuous DQN agents
        traits_list = config.agent_traits or [AgentTraits() for _ in range(self.N)]
        p_max = float(np.max(self.price_grid))

        self.agents = [
            DQNAgent(
                firm_idx=i,
                num_firms=self.N,
                price_grid=self.price_grid,
                config=config.dqn_config,
                initial_capital=self.m_cfg.initial_capital,
                traits=traits_list[i],
            )
            for i in range(self.N)
        ]
        self.encoder = ContinuousStateEncoder(self.N, p_max, self.m_cfg.initial_capital)

    def snap_price_to_index(self, price: float) -> int:
        return int(np.argmin(np.abs(self.price_grid - price)))

    def run(self) -> SimulationResults:
        """Executes full continuous DQN training loop."""
        capital = np.full(self.N, self.m_cfg.initial_capital, dtype=np.float64)
        ban_timers = np.zeros(self.N, dtype=np.int32)
        lagged_prices = self.nash_prices.copy()
        ema_ref = float(np.mean(lagged_prices))

        eval_start = max(0, self.cfg.total_steps - self.cfg.eval_window)
        eval_prices = []
        eval_profits = []
        eval_cs = []
        total_fresh_bans = 0
        induced_bans = 0
        bankruptcies_in_eval = 0
        episode_step = 0

        for step in range(self.cfg.total_steps):
            is_warmup = step < self.cfg.warmup_steps
            active_mask = np.array([
                1 if (ban_timers[i] == 0 and capital[i] > 0.0) else 0
                for i in range(self.N)
            ], dtype=np.int32)
            num_active = int(np.sum(active_mask))

            # 1. Continuous state encoding
            current_states = [
                self.encoder.encode(lagged_prices, (ban_timers > 0) | (capital <= 0.0), capital[i])
                for i in range(self.N)
            ]

            # 2. Action selection
            chosen_actions = np.zeros(self.N, dtype=np.int32)
            chosen_prices = lagged_prices.copy()

            for i in range(self.N):
                if active_mask[i] != 0:
                    if self.cfg.agent_type == "scripted_attacker" and i == 0:
                        if num_active >= 3:
                            a_idx = 0  # Undercut to marginal cost floor
                        else:
                            a_idx = len(self.price_grid) - 1
                    else:
                        a_idx = self.agents[i].select_action(current_states[i], self.rng)
                    chosen_actions[i] = a_idx
                    chosen_prices[i] = self.price_grid[a_idx]
                else:
                    chosen_actions[i] = self.snap_price_to_index(chosen_prices[i])

            # 3. Market clearing
            shares = compute_market_shares(
                chosen_prices,
                self.m_cfg.qualities,
                active_mask,
                self.m_cfg.outside_option,
                self.m_cfg.price_sensitivity,
            )
            profits = compute_profits(
                chosen_prices,
                self.m_cfg.marginal_costs,
                shares,
                self.m_cfg.market_size,
                active_mask,
            )
            cs = compute_consumer_surplus(
                chosen_prices,
                self.m_cfg.qualities,
                active_mask,
                self.m_cfg.outside_option,
                self.m_cfg.price_sensitivity,
                self.m_cfg.market_size,
            )

            # 4. Regulatory audit & flagging
            flags = np.zeros(self.N, dtype=np.int32)
            audited = np.zeros(self.N, dtype=np.int32)
            harm = np.zeros(self.N, dtype=np.int32)

            if (not is_warmup) and (num_active >= 3):
                flags = evaluate_rules(
                    chosen_prices,
                    active_mask,
                    self.cfg.rule_type,
                    self.cfg.rule_param,
                    ema_ref=ema_ref,
                    beta=self.cfg.ema_beta,
                )
                audited = sample_audit_set(active_mask, self.m_cfg.audit_budget, self.rng)
                harm = compute_counterfactual_harm(
                    chosen_prices,
                    self.nash_prices,
                    active_mask,
                    audited,
                    self.cfg.rule_type,
                    self.cfg.rule_param,
                    ema_ref=ema_ref,
                    beta=self.cfg.ema_beta,
                )

            # EMA benchmark update
            if num_active > 0:
                act_mean = float(np.mean(chosen_prices[active_mask != 0]))
                ema_ref = float(self.cfg.ema_beta * ema_ref + (1.0 - self.cfg.ema_beta) * act_mean)

            # Penalty dynamics
            if self.cfg.penalty_mode == "escrow":
                next_timers = ban_timers.copy()
                penalized_mask = (audited != 0) & (flags != 0)
                for i in range(self.N):
                    if penalized_mask[i]:
                        excess = max(0.0, profits[i] - self.nash_profits[i])
                        profits[i] -= excess
                fresh_bans = np.zeros(self.N, dtype=bool)
            else:
                fresh_bans = (audited != 0) & (flags != 0) & (ban_timers == 0)
                next_timers = update_ban_timers(ban_timers, flags, audited, self.m_cfg.ban_duration)

            # Capital dynamics
            next_capital = update_capital(capital, profits, self.m_cfg.fixed_cost)

            # 5. Reward calculation & DQN mini-batch updates
            for i in range(self.N):
                other_p_sum = float(np.sum(profits) - profits[i])
                reward_i = compute_agent_reward(
                    profit=profits[i],
                    Q=self.m_cfg.market_size,
                    CS=cs,
                    other_profits_sum=other_p_sum,
                    N=self.N,
                    harm_count=harm[i],
                    traits=self.agents[i].traits,
                )

                next_state_i = self.encoder.encode(
                    chosen_prices, (next_timers > 0) | (next_capital <= 0.0), next_capital[i]
                )
                is_bankrupt = next_capital[i] <= 0.0

                if active_mask[i] != 0:
                    self.agents[i].update(
                        current_states[i],
                        chosen_actions[i],
                        reward_i,
                        next_state_i,
                        done=is_bankrupt,
                    )
                self.agents[i].decay_epsilon()

            # Record stats in evaluation window
            if step >= eval_start:
                eval_prices.append(chosen_prices.copy())
                eval_profits.append(profits.copy())
                eval_cs.append(cs)

                new_ban_indices = np.where(fresh_bans)[0]
                for j in new_ban_indices:
                    total_fresh_bans += 1
                    is_induced = False
                    for k in range(self.N):
                        if k != j and active_mask[k] != 0:
                            cf_p = chosen_prices.copy()
                            cf_p[k] = self.nash_prices[k]
                            cf_flags = evaluate_rules(
                                cf_p,
                                active_mask,
                                self.cfg.rule_type,
                                self.cfg.rule_param,
                                ema_ref=ema_ref,
                                beta=self.cfg.ema_beta,
                            )
                            if cf_flags[j] == 0:
                                is_induced = True
                                break
                    if is_induced:
                        induced_bans += 1

                for i in range(self.N):
                    if capital[i] > 0.0 and next_capital[i] <= 0.0:
                        bankruptcies_in_eval += 1

            # Transition state
            capital = next_capital
            ban_timers = next_timers
            lagged_prices = chosen_prices
            episode_step += 1

            # Reset condition
            active_next = np.sum([1 if (ban_timers[i] == 0 and capital[i] > 0.0) else 0 for i in range(self.N)])
            if episode_step >= self.cfg.episode_length or active_next <= 1:
                capital = np.full(self.N, self.m_cfg.initial_capital, dtype=np.float64)
                ban_timers = np.zeros(self.N, dtype=np.int32)
                lagged_prices = self.nash_prices.copy()
                ema_ref = float(np.mean(lagged_prices))
                episode_step = 0

        # Compute aggregate evaluation metrics
        eval_p_arr = np.array(eval_prices)
        eval_pi_arr = np.array(eval_profits)
        mean_p = np.mean(eval_p_arr, axis=0)
        mean_pi = np.mean(eval_pi_arr, axis=0)

        avg_per_firm_profit = float(np.mean(mean_pi))
        nash_per_firm_profit = float(np.mean(self.nash_profits))
        mono_per_firm_profit = float(np.mean(self.mono_profits))
        delta = compute_collusion_index(
            avg_per_firm_profit, nash_per_firm_profit, mono_per_firm_profit
        )

        w_rate = (induced_bans / total_fresh_bans) if total_fresh_bans > 0 else 0.0
        b_rate = bankruptcies_in_eval / max(1, self.N)

        return SimulationResults(
            weaponization_rate=float(w_rate),
            collusion_index=float(delta),
            social_welfare_loss=0.0,
            mean_prices=mean_p,
            mean_profits=mean_pi,
            bankruptcy_rate=float(b_rate),
            total_fresh_bans=int(total_fresh_bans),
            induced_bans=int(induced_bans),
            unique_states_visited=len(self.agents[0].buffer),
            state_coverage_ratio=1.0,  # Continuous state space has unbounded coverage
            effective_states_visited=len(self.agents[0].buffer),
            effective_state_coverage=1.0,
        )
