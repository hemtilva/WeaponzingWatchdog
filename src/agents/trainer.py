"""Multi-agent market simulation trainer with episodic resets, warmup, and metrics."""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

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
    sample_audit_set,
    update_ban_timers,
    compute_counterfactual_harm,
)
from src.benchmarks.nash import find_discrete_nash_equilibrium
from src.benchmarks.monopoly import find_discrete_monopoly_prices, compute_collusion_index
from .traits import AgentTraits, compute_agent_reward
from .q_learning import QLearningConfig, StateIndexer, QLearningAgent


@dataclass
class SimulationConfig:
    """Configuration for multi-agent pricing simulation."""
    market: MarketConfig = field(default_factory=MarketConfig)
    q_learning: QLearningConfig = field(default_factory=QLearningConfig)
    price_grid: Optional[np.ndarray] = None
    total_steps: int = 200_000
    episode_length: int = 500       # Approach 1: reset horizon
    warmup_steps: int = 20_000      # Approach 2: unregulated warmup phase
    rule_type: RuleType = RuleType.MEAN_RATIO
    rule_param: float = 0.20
    agent_traits: Optional[List[AgentTraits]] = None
    eval_window: int = 30_000       # Trailing window for evaluation metrics
    agent_type: str = "adaptive"    # "adaptive", "honest" (good boy), "random", "scripted_attacker"
    penalty_mode: str = "shutdown"  # "shutdown" (standard ban) or "escrow" (profit diversion)
    ema_beta: float = 0.95          # Memory parameter for EMA_MEAN_RATIO (0.95 retains ~20 periods of inertia)
    record_trace: bool = False      # Whether to collect per-step timeline trace
    trace_steps: int = 200          # Number of steps to record for visualization
    seed: int = 42


@dataclass
class SimulationResults:
    """Evaluation metrics collected post-convergence."""
    weaponization_rate: float       # W in [0, 1]
    collusion_index: float          # Delta
    social_welfare_loss: float      # SWL
    mean_prices: np.ndarray         # (N,)
    mean_profits: np.ndarray        # (N,)
    bankruptcy_rate: float
    total_fresh_bans: int
    induced_bans: int
    unique_states_visited: int = 0
    state_coverage_ratio: float = 0.0
    effective_states_visited: int = 0
    effective_state_coverage: float = 0.0
    history_prices: Optional[np.ndarray] = None
    trace_data: Optional[List[Dict[str, Any]]] = None
    exploit_events: List[Dict[str, Any]] = field(default_factory=list)


class MultiAgentTrainer:
    """Orchestrates decentralized Q-learning agents in the regulated oligopoly market."""

    def __init__(self, config: SimulationConfig):
        self.cfg = config
        self.m_cfg = config.market
        self.N = self.m_cfg.num_firms

        self.rng = np.random.default_rng(config.seed)

        if config.price_grid is None:
            # Default M = 11 price levels spanning [cost, monopoly price approx]
            self.price_grid = np.linspace(
                self.m_cfg.marginal_costs[0],
                self.m_cfg.marginal_costs[0] + 6.0 * self.m_cfg.price_sensitivity,
                11,
                dtype=np.float64
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

        # Baseline competitive consumer surplus
        self.nash_cs = compute_consumer_surplus(
            self.nash_prices,
            self.m_cfg.qualities,
            np.ones(self.N, dtype=np.int32),
            self.m_cfg.outside_option,
            self.m_cfg.price_sensitivity,
            self.m_cfg.market_size,
        )

        # Setup State Indexer & Agents
        self.indexer = StateIndexer(self.price_grid, fixed_cost=self.m_cfg.fixed_cost, num_firms=self.N)
        traits_list = config.agent_traits or [AgentTraits() for _ in range(self.N)]

        self.agents = [
            QLearningAgent(
                firm_idx=i,
                config=config.q_learning,
                indexer=self.indexer,
                traits=traits_list[i],
            )
            for i in range(self.N)
        ]

    def run(self) -> SimulationResults:
        """Executes full simulation training and evaluation loop."""
        # Initial state setup
        capital = np.full(self.N, self.m_cfg.initial_capital, dtype=np.float64)
        ban_timers = np.zeros(self.N, dtype=np.int32)
        lagged_prices = self.nash_prices.copy()
        ema_ref = float(np.mean(lagged_prices))

        # Tracking metrics for evaluation window
        eval_start = max(0, self.cfg.total_steps - self.cfg.eval_window)
        trace_start = max(0, self.cfg.total_steps - self.cfg.trace_steps)
        trace_history = [] if self.cfg.record_trace else None
        trace_exploit_events = []
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

            # 1. State encoding for all agents
            current_states = [
                self.agents[i].indexer.encode(lagged_prices, ban_timers > 0, capital[i])
                for i in range(self.N)
            ]

            # 2. Simultaneous Action Selection
            chosen_actions = np.zeros(self.N, dtype=np.int32)
            chosen_prices = lagged_prices.copy()

            for i in range(self.N):
                if active_mask[i] != 0:
                    if self.cfg.agent_type == "honest":
                        # Ideal good boy: always plays competitive Bertrand-Nash
                        a_idx = self.indexer.snap_price_to_index(self.nash_prices[i])
                    elif self.cfg.agent_type == "random":
                        # Pure random baseline
                        a_idx = int(self.rng.integers(0, len(self.price_grid)))
                    elif self.cfg.agent_type == "scripted_attacker":
                        # Firm 0 is the scripted predatory attacker; rivals are honest Nash pricers
                        if i == 0:
                            if num_active >= 3:
                                # Crash price to minimum grid index (cost floor) to trigger relative rule flags on rivals
                                a_idx = 0
                            else:
                                # When rivals are eliminated/banned, extract joint monopoly profit
                                a_idx = self.indexer.snap_price_to_index(self.mono_prices[0])
                        else:
                            # Innocent rivals price honestly at Bertrand-Nash
                            a_idx = self.indexer.snap_price_to_index(self.nash_prices[i])
                    else:
                        # Adaptive Q-learning agent
                        a_idx = self.agents[i].select_action(current_states[i], self.rng)

                    chosen_actions[i] = a_idx
                    chosen_prices[i] = self.price_grid[a_idx]
                else:
                    # Inactive firms retain previous price
                    chosen_actions[i] = self.indexer.snap_price_to_index(chosen_prices[i])

            # 3. Market Clearing (Demand & Operating Profits)
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

            # 4. Regulatory Audit & Flagging (Only post-warmup and when N_active >= 3)
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

            # Update EMA benchmark reference price
            if num_active > 0:
                act_mean = float(np.mean(chosen_prices[active_mask != 0]))
                ema_ref = float(self.cfg.ema_beta * ema_ref + (1.0 - self.cfg.ema_beta) * act_mean)

            # Detect fresh bans and update penalty states
            if self.cfg.penalty_mode == "escrow":
                # Defended Watchdog Escrow Defense:
                # Flagged firms are NOT banned/shut down. Their excess profits above Nash are escrowed.
                # Because rivals remain in the market, the attacker never secures monopoly rents!
                next_timers = ban_timers.copy()
                penalized_mask = (audited != 0) & (flags != 0)
                for i in range(self.N):
                    if penalized_mask[i]:
                        excess = max(0.0, profits[i] - self.nash_profits[i])
                        profits[i] = profits[i] - excess
                fresh_bans = np.zeros(self.N, dtype=bool)
            else:
                # Standard watchdog shutdown ban
                fresh_bans = (audited != 0) & (flags != 0) & (ban_timers == 0)
                next_timers = update_ban_timers(
                    ban_timers, flags, audited, self.m_cfg.ban_duration
                )

            # 5. Capital & Bankruptcy Dynamics
            next_capital = update_capital(capital, profits, self.m_cfg.fixed_cost)

            # 6. Reward Calculation & Q-Learning Updates
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

                # Next state observation
                next_state_i = self.agents[i].indexer.encode(
                    chosen_prices, next_timers > 0, next_capital[i]
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

            # Record stats if in evaluation window
            if step >= eval_start:
                eval_prices.append(chosen_prices.copy())
                eval_profits.append(profits.copy())
                eval_cs.append(cs)

                new_ban_indices = np.where(fresh_bans)[0]
                for j in new_ban_indices:
                    total_fresh_bans += 1
                    # Counterfactual check: did any rival's deviation cause j's ban?
                    # i.e., sum of harm caused by rivals to firm j
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

                # Check fresh bankruptcies
                for i in range(self.N):
                    if capital[i] > 0.0 and next_capital[i] <= 0.0:
                        bankruptcies_in_eval += 1

            # Record per-step timeline trace if requested
            if self.cfg.record_trace and step >= trace_start:
                # Reference price and threshold
                if num_active >= 3:
                    if self.cfg.rule_type == RuleType.MEAN_RATIO:
                        ref_p = float(np.mean(chosen_prices[active_mask != 0]))
                        thresh_p = float((1.0 + self.cfg.rule_param) * ref_p)
                    elif self.cfg.rule_type == RuleType.EMA_MEAN_RATIO:
                        ref_p = float(ema_ref)
                        thresh_p = float((1.0 + self.cfg.rule_param) * ema_ref)
                    elif self.cfg.rule_type == RuleType.MEDIAN_RATIO:
                        ref_p = float(np.median(chosen_prices[active_mask != 0]))
                        thresh_p = float((1.0 + self.cfg.rule_param) * ref_p)
                    elif self.cfg.rule_type == RuleType.PRICE_CAP:
                        ref_p = float(self.cfg.rule_param)
                        thresh_p = float(self.cfg.rule_param)
                    else:
                        ref_p = float(np.mean(chosen_prices[active_mask != 0]))
                        thresh_p = ref_p + float(self.cfg.rule_param * np.std(chosen_prices[active_mask != 0]))
                else:
                    ref_p = float(np.mean(chosen_prices))
                    thresh_p = ref_p

                # Detect exploit events in trace
                step_events = []
                fresh_banned_indices = np.where(fresh_bans)[0]
                for j in fresh_banned_indices:
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
                                ev = {
                                    "step": int(step),
                                    "step_idx": int(len(trace_history)),
                                    "attacker": int(k),
                                    "victim": int(j),
                                    "attacker_price": float(chosen_prices[k]),
                                    "victim_price": float(chosen_prices[j]),
                                    "nash_price": float(self.nash_prices[k]),
                                    "ref_price": float(ref_p),
                                    "threshold": float(thresh_p),
                                    "ban_duration": int(self.m_cfg.ban_duration),
                                    "fixed_cost": float(self.m_cfg.fixed_cost),
                                }
                                step_events.append(ev)
                                trace_exploit_events.append(ev)
                                break

                trace_history.append({
                    "step": int(step),
                    "prices": [float(p) for p in chosen_prices],
                    "capitals": [float(c) for c in capital],
                    "ban_timers": [int(t) for t in ban_timers],
                    "next_ban_timers": [int(t) for t in next_timers],
                    "flags": [int(f) for f in flags],
                    "audited": [int(a) for a in audited],
                    "fresh_bans": [int(b) for b in fresh_bans],
                    "harm": [int(h) for h in harm],
                    "active_mask": [int(m) for m in active_mask],
                    "active_count": int(num_active),
                    "ref_price": ref_p,
                    "threshold": thresh_p,
                    "exploit_events": step_events,
                })

            # Transition state
            capital = next_capital
            ban_timers = next_timers
            lagged_prices = chosen_prices
            episode_step += 1

            # Approach 1: Episode Reset condition
            active_next = np.sum([1 if (ban_timers[i] == 0 and capital[i] > 0.0) else 0 for i in range(self.N)])
            if episode_step >= self.cfg.episode_length or active_next <= 1:
                # Reset market for new episode while preserving Q-tables!
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

        # Collusion Index
        avg_per_firm_profit = float(np.mean(mean_pi))
        nash_per_firm_profit = float(np.mean(self.nash_profits))
        mono_per_firm_profit = float(np.mean(self.mono_profits))
        delta = compute_collusion_index(
            avg_per_firm_profit, nash_per_firm_profit, mono_per_firm_profit
        )

        # Weaponization Rate
        w_rate = (induced_bans / total_fresh_bans) if total_fresh_bans > 0 else 0.0

        # Social Welfare Loss (SWL)
        mean_cs = float(np.mean(eval_cs))
        total_realized_welfare = mean_cs + float(np.sum(mean_pi))
        nash_welfare = self.nash_cs + float(np.sum(self.nash_profits))
        swl = 1.0 - (total_realized_welfare / nash_welfare) if nash_welfare > 0 else 0.0

        num_episodes_in_eval = max(1, self.cfg.eval_window // self.cfg.episode_length)
        b_rate = bankruptcies_in_eval / (self.N * num_episodes_in_eval)

        # State visitation & exploration coverage (Feedback 1)
        avg_unique = int(np.mean([a.unique_states_visited for a in self.agents]))
        avg_coverage = float(np.mean([a.state_coverage_ratio for a in self.agents]))
        avg_effective = int(np.mean([a.effective_states_visited(min_visits=5) for a in self.agents]))
        avg_effective_cov = float(avg_effective / self.indexer.num_states) if self.indexer.num_states > 0 else 0.0

        return SimulationResults(
            weaponization_rate=float(w_rate),
            collusion_index=float(delta),
            social_welfare_loss=float(swl),
            mean_prices=mean_p,
            mean_profits=mean_pi,
            bankruptcy_rate=float(min(1.0, b_rate)),
            total_fresh_bans=total_fresh_bans,
            induced_bans=induced_bans,
            unique_states_visited=avg_unique,
            state_coverage_ratio=avg_coverage,
            effective_states_visited=avg_effective,
            effective_state_coverage=avg_effective_cov,
            trace_data=trace_history,
            exploit_events=trace_exploit_events,
        )

