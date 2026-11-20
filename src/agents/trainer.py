"""Multi-agent market simulation trainer with episodic resets, warmup, and metrics."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
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
    select_top_audit_set,
    rule_scores,
    update_ban_timers,
    compute_counterfactual_harm,
)
from src.benchmarks.nash import find_discrete_nash_equilibrium
from src.benchmarks.monopoly import find_discrete_monopoly_prices, compute_collusion_index
from .traits import AgentTraits, compute_agent_reward, REWARD_EPS
from .q_learning import QLearningConfig, StateIndexer, QLearningAgent
from src.engine import native_sim


# Per-firm policy identifiers
POLICY_Q = "q"                  # adaptive tabular Q-learner
POLICY_NASH = "nash"            # honest: always plays the Bertrand-Nash price
POLICY_RANDOM = "random"        # uniform random price
POLICY_ATTACKER = "attacker"    # scripted predator: crash price while regulated, monopoly price otherwise
VALID_POLICIES = {POLICY_Q, POLICY_NASH, POLICY_RANDOM, POLICY_ATTACKER}

# Legacy `agent_type` presets -> per-firm policy builder
_POLICY_CODES = {POLICY_Q: 0, POLICY_NASH: 1, POLICY_RANDOM: 2, POLICY_ATTACKER: 3}

_AGENT_TYPE_PRESETS = {
    "adaptive": lambda n: [POLICY_Q] * n,
    "honest": lambda n: [POLICY_NASH] * n,
    "random": lambda n: [POLICY_RANDOM] * n,
    "scripted_attacker": lambda n: [POLICY_ATTACKER] + [POLICY_NASH] * (n - 1),
    "learner_vs_honest": lambda n: [POLICY_Q] + [POLICY_NASH] * (n - 1),
}


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
    # Preset: "adaptive", "honest", "random", "scripted_attacker", "learner_vs_honest".
    # Ignored when `firm_policies` is given.
    agent_type: str = "adaptive"
    # Explicit per-firm policies, e.g. ["q", "nash", "nash"]. See VALID_POLICIES.
    firm_policies: Optional[List[str]] = None
    penalty_mode: str = "shutdown"  # "shutdown" (standard ban) or "escrow" (profit diversion)
    ema_beta: float = 0.95          # Memory parameter for EMA_MEAN_RATIO (0.95 retains ~20 periods of inertia)
    record_trace: bool = False      # Whether to collect per-step timeline trace
    freeze_learners: bool = False   # Q-learners act greedily (no exploration) and do not update
    # "auto": native C++ loop when available (falls back to Python when a per-step trace
    # is requested); "native": require it; "python": always use the reference loop.
    backend: str = "auto"
    # Early episode end. False (default): end early only when no firm is solvent, so a
    # sole survivor keeps earning monopoly rents until episode_length (the payoff of
    # predation). True: legacy rule, end as soon as <= 1 firm is active, which cuts off
    # monopoly rents and makes eliminating all rivals look like a loss to learners.
    legacy_reset: bool = False
    trace_steps: int = 200          # Number of steps to record for visualization
    # Audit selection: "random" (uniform over active firms) or "top" (top-score-first: the
    # highest rule score, e.g. p / reference, is audited first; ties at random)
    audit_mode: str = "random"
    # Ban-timer ablation: learners observe each firm's exact ban timer instead of a bit
    observe_ban_timers: bool = False
    seed: int = 42


@dataclass
class SimulationResults:
    """Evaluation metrics collected post-convergence."""
    weaponization_rate: float       # W in [0, 1] (over bans; over sanctions in escrow mode)
    collusion_index: float          # Delta
    social_welfare_loss: float      # SWL
    mean_prices: np.ndarray         # (N,)
    mean_profits: np.ndarray        # (N,)
    bankruptcy_rate: float
    total_fresh_bans: int           # suspensions started in the eval window
    induced_bans: int               # ... of which a rival's deviation was the trigger
    unique_states_visited: int = 0
    state_coverage_ratio: float = 0.0
    effective_states_visited: int = 0
    effective_state_coverage: float = 0.0
    history_prices: Optional[np.ndarray] = None
    trace_data: Optional[List[Dict[str, Any]]] = None
    exploit_events: List[Dict[str, Any]] = field(default_factory=list)
    # Sanctions = audited & flagged events (bans in shutdown mode, escrow diversions in escrow mode)
    total_sanctions: int = 0
    induced_sanctions: int = 0
    # Bankruptcies in the eval window, and those of firms that were serving an induced ban
    total_bankruptcies: int = 0
    induced_bankruptcies: int = 0
    # Per-firm count of induced sanctions attributed to that firm as the inducer
    induced_by: Optional[np.ndarray] = None
    # Per-firm count of induced bankruptcies attributed to that firm
    kills_by: Optional[np.ndarray] = None
    episodes_in_eval: int = 0
    firm_policies: Optional[List[str]] = None
    mean_consumer_surplus: float = 0.0
    eval_periods: int = 0
    # Coalition-aware attribution: the flag disappears if all rivals post Nash prices.
    # Superset of the unilateral counts above; kills_by uses this notion.
    joint_induced_bans: int = 0
    joint_induced_sanctions: int = 0
    joint_weaponization_rate: float = 0.0
    # Bankruptcies of each firm in the eval window (victims; kills_by counts the inducers)
    bankruptcies_of: Optional[np.ndarray] = None


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

        self.policies = self._resolve_policies()

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
        if config.audit_mode not in ("random", "top"):
            raise ValueError(f"Unknown audit_mode: {config.audit_mode}")
        self.indexer = StateIndexer(self.price_grid, fixed_cost=self.m_cfg.fixed_cost, num_firms=self.N,
                                    timer_levels=self.m_cfg.ban_duration if config.observe_ban_timers else None)
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
        self.nash_idx = np.array(
            [self.indexer.snap_price_to_index(p) for p in self.nash_prices], dtype=np.int32
        )
        self.mono_idx = np.array(
            [self.indexer.snap_price_to_index(p) for p in self.mono_prices], dtype=np.int32
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _resolve_policies(self) -> List[str]:
        if self.cfg.firm_policies is not None:
            policies = list(self.cfg.firm_policies)
            if len(policies) != self.N:
                raise ValueError(f"firm_policies has {len(policies)} entries, expected {self.N}")
        else:
            if self.cfg.agent_type not in _AGENT_TYPE_PRESETS:
                raise ValueError(f"Unknown agent_type: {self.cfg.agent_type}")
            policies = _AGENT_TYPE_PRESETS[self.cfg.agent_type](self.N)
        bad = set(policies) - VALID_POLICIES
        if bad:
            raise ValueError(f"Unknown firm policies: {bad}")
        return policies

    def _select_action(self, i: int, state_idx: int, num_active: int) -> int:
        policy = self.policies[i]
        if policy == POLICY_Q:
            return self.agents[i].select_action(state_idx, self.rng, explore=not self.cfg.freeze_learners)
        if policy == POLICY_NASH:
            return int(self.nash_idx[i])
        if policy == POLICY_RANDOM:
            return int(self.rng.integers(0, len(self.price_grid)))
        # POLICY_ATTACKER: crash to the cost floor while the rule is live, else monopoly price
        if num_active >= self.m_cfg.min_active_for_audit:
            return 0
        return int(self.mono_idx[i])

    def _find_inducer(
        self,
        j: int,
        prices: np.ndarray,
        active_mask: np.ndarray,
        ema_ref: float,
    ) -> int:
        """Returns the rival whose deviation from Nash triggered firm j's flag, or -1.

        A rival k is a trigger if resetting k's price to its Nash price (holding all
        else, including the EMA history, fixed) un-flags j. Among several triggers,
        the one priced furthest below its Nash price is returned.
        """
        best_k, best_gap = -1, -np.inf
        for k in range(self.N):
            if k == j or active_mask[k] == 0:
                continue
            cf_p = prices.copy()
            cf_p[k] = self.nash_prices[k]
            cf_flags = evaluate_rules(
                cf_p, active_mask, self.cfg.rule_type, self.cfg.rule_param,
                ema_ref=ema_ref, beta=self.cfg.ema_beta, min_active=self.m_cfg.min_active_for_audit,
                nash_prices=self.nash_prices,
            )
            if cf_flags[j] == 0:
                gap = self.nash_prices[k] - prices[k]
                if gap > best_gap:
                    best_k, best_gap = k, gap
        return best_k

    def _find_joint_inducer(
        self,
        j: int,
        prices: np.ndarray,
        active_mask: np.ndarray,
        ema_ref: float,
    ) -> int:
        """Coalition attribution: if j would not be flagged had *all* active rivals posted
        their Nash prices, returns the rival priced furthest below Nash (the lead
        undercutter); otherwise -1. Catches framing by several undercutters at once,
        which the one-rival-at-a-time test in _find_inducer misses."""
        rivals = [k for k in range(self.N) if k != j and active_mask[k] != 0]
        if not rivals:
            return -1
        cf_p = prices.copy()
        cf_p[rivals] = self.nash_prices[rivals]
        cf_flags = evaluate_rules(
            cf_p, active_mask, self.cfg.rule_type, self.cfg.rule_param,
            ema_ref=ema_ref, beta=self.cfg.ema_beta, min_active=self.m_cfg.min_active_for_audit,
            nash_prices=self.nash_prices,
        )
        if cf_flags[j] != 0:
            return -1
        gaps = [self.nash_prices[k] - prices[k] for k in rivals]
        return int(rivals[int(np.argmax(gaps))])

    def _reference_and_threshold(self, prices, active_mask, num_active, ema_ref):
        act = prices[active_mask != 0]
        if num_active < self.m_cfg.min_active_for_audit:
            ref = float(np.mean(prices))
            return ref, ref
        rt = self.cfg.rule_type
        if rt == RuleType.MEAN_RATIO:
            ref = float(np.mean(act))
        elif rt == RuleType.EMA_MEAN_RATIO:
            ref = float(self.cfg.ema_beta * ema_ref + (1.0 - self.cfg.ema_beta) * np.mean(act))
        elif rt == RuleType.MEDIAN_RATIO:
            ref = float(np.median(act))
        elif rt == RuleType.PRICE_CAP:
            return float(self.cfg.rule_param), float(self.cfg.rule_param)
        else:
            ref = float(np.mean(act))
            return ref, ref + float(self.cfg.rule_param * np.std(act))
        return ref, float((1.0 + self.cfg.rule_param) * ref)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> SimulationResults:
        """Executes the simulation (training and evaluation window). Q-tables persist across calls."""
        backend = self.cfg.backend
        if backend not in ("auto", "native", "python"):
            raise ValueError(f"Unknown backend: {backend}")
        native_ok = native_sim.is_native_sim_available()
        if backend == "native" and not native_ok:
            raise RuntimeError("Native simulation backend requested but engine is not built "
                               "(run scripts/build_engine.py).")
        if backend == "native" and self.cfg.record_trace:
            raise ValueError("Per-step traces are only recorded by the Python backend.")
        use_native = native_ok and backend != "python" and not self.cfg.record_trace
        return self._run_native() if use_native else self._run_python()

    # ------------------------------------------------------------------
    # Native backend
    # ------------------------------------------------------------------
    def _bind_native_tables(self):
        """Stacks the agents' Q-tables / visit counts into contiguous arrays the C++ loop
        updates in place; each agent's arrays become views into them."""
        if getattr(self, "_qstack", None) is not None:
            return
        S, M = self.indexer.num_states, self.indexer.M
        self._qstack = np.empty((self.N, S, M), dtype=np.float32)
        self._vstack = np.empty((self.N, S), dtype=np.int32)
        for i, a in enumerate(self.agents):
            self._qstack[i] = a.Q
            self._vstack[i] = a.state_visit_counts
            a.Q = self._qstack[i]
            a.state_visit_counts = self._vstack[i]
        self._native_calls = 0

    def _run_native(self) -> SimulationResults:
        self._bind_native_tables()
        m, q = self.m_cfg, self.cfg.q_learning
        if any(a.has_pending for a in self.agents):
            for a in self.agents:  # transitions left open by the Python loop cannot cross backends
                a._pending = None
        traits = [a.traits for a in self.agents]
        eps = np.array([a.epsilon for a in self.agents], dtype=np.float64)
        arrays = dict(
            price_grid=self.price_grid, costs=m.marginal_costs, qualities=m.qualities,
            nash_prices=self.nash_prices, nash_profits=self.nash_profits,
            nash_idx=self.nash_idx, mono_idx=self.mono_idx,
            policies=np.array([_POLICY_CODES[p] for p in self.policies], dtype=np.int32),
            alpha=np.array([t.alpha for t in traits]),
            lambda_=np.array([t.lambda_altruism for t in traits]),
            mu=np.array([t.mu_ethics for t in traits]),
            qtab=self._qstack, visits=self._vstack, epsilon=eps,
        )
        scalars = dict(
            N=self.N, M=len(self.price_grid), rule_type=int(self.cfg.rule_type),
            penalty_mode=1 if self.cfg.penalty_mode == "escrow" else 0,
            total_steps=self.cfg.total_steps, warmup_steps=self.cfg.warmup_steps,
            episode_length=self.cfg.episode_length, eval_window=self.cfg.eval_window,
            audit_budget=m.audit_budget, T_ban=m.ban_duration,
            freeze=int(self.cfg.freeze_learners), has_cap=int(m.capital_cap is not None),
            min_active=int(m.min_active_for_audit), legacy_reset=int(self.cfg.legacy_reset),
            outside=m.outside_option, sigma=m.price_sensitivity, Q=m.market_size,
            fixed_cost=m.fixed_cost, K0=m.initial_capital,
            cap=float(m.capital_cap) if m.capital_cap is not None else 0.0,
            rule_param=float(self.cfg.rule_param), ema_beta=float(self.cfg.ema_beta),
            lr=q.learning_rate, gamma=q.gamma, eps_min=q.eps_min, eps_decay=q.eps_decay,
            reward_eps=REWARD_EPS,
            audit_mode=1 if self.cfg.audit_mode == "top" else 0,
            timer_obs=int(self.cfg.observe_ban_timers),
            seed=(int(self.cfg.seed) * 1_000_003 + self._native_calls) % (2 ** 64),
        )
        self._native_calls += 1
        raw = native_sim.run_native(arrays, scalars)
        for i, a in enumerate(self.agents):
            a.epsilon = float(eps[i])
        return self._finalize(raw)

    # ------------------------------------------------------------------
    # Python reference backend
    # ------------------------------------------------------------------
    def _run_python(self) -> SimulationResults:
        """Reference implementation of the simulation loop (mirrored by src/csrc/sim.cpp)."""
        N = self.N
        m = self.m_cfg
        cap_limit = m.capital_cap
        is_q = np.array([p == POLICY_Q for p in self.policies])

        def fresh_state():
            cap0 = m.initial_capital if cap_limit is None else min(m.initial_capital, cap_limit)
            return (
                np.full(N, cap0, dtype=np.float64),
                np.zeros(N, dtype=np.int32),
                self.nash_prices.copy(),
            )

        capital, ban_timers, lagged_prices = fresh_state()
        ema_ref = float(np.mean(lagged_prices))
        ban_inducer = np.full(N, -1, dtype=np.int32)

        eval_start = max(0, self.cfg.total_steps - self.cfg.eval_window)
        trace_start = max(0, self.cfg.total_steps - self.cfg.trace_steps)
        trace_history = [] if self.cfg.record_trace else None
        trace_exploit_events = []
        eval_prices, eval_profits, eval_cs = [], [], []
        total_fresh_bans = induced_bans = 0
        total_sanctions = induced_sanctions = 0
        joint_induced_sanctions = joint_induced_bans = 0
        n_min = m.min_active_for_audit
        total_bankruptcies = induced_bankruptcies = 0
        induced_by = np.zeros(N, dtype=np.int64)
        kills_by = np.zeros(N, dtype=np.int64)
        bankr_of = np.zeros(N, dtype=np.int64)
        episodes_in_eval = 0
        episode_step = 0

        for step in range(self.cfg.total_steps):
            in_eval = step >= eval_start
            if step == eval_start:
                episodes_in_eval += 1  # episode in progress (or starting) at eval start
            is_warmup = step < self.cfg.warmup_steps
            active_mask = ((ban_timers == 0) & (capital > 0.0)).astype(np.int32)
            num_active = int(active_mask.sum())

            # 1. State encoding; close transitions of learners that are acting again
            current_states = [
                self.indexer.encode(lagged_prices, self.indexer.status(ban_timers, capital), capital[i]) for i in range(N)
            ]
            for i in range(N):
                if is_q[i] and active_mask[i] and self.agents[i].has_pending:
                    self.agents[i].close_transition(current_states[i])

            # 2. Simultaneous action selection (inactive firms keep their last price)
            chosen_actions = np.zeros(N, dtype=np.int32)
            chosen_prices = lagged_prices.copy()
            for i in range(N):
                if active_mask[i]:
                    a_idx = self._select_action(i, current_states[i], num_active)
                    chosen_actions[i] = a_idx
                    chosen_prices[i] = self.price_grid[a_idx]
                    if is_q[i] and not self.cfg.freeze_learners:
                        self.agents[i].begin_transition(current_states[i], a_idx)
                else:
                    chosen_actions[i] = self.indexer.snap_price_to_index(chosen_prices[i])

            # 3. Market clearing
            shares = compute_market_shares(
                chosen_prices, m.qualities, active_mask, m.outside_option, m.price_sensitivity,
            )
            profits = compute_profits(chosen_prices, m.marginal_costs, shares, m.market_size, active_mask)
            cs = compute_consumer_surplus(
                chosen_prices, m.qualities, active_mask, m.outside_option, m.price_sensitivity, m.market_size,
            )

            # 4. Regulatory audit & flagging (post-warmup, N_active >= N_min)
            flags = np.zeros(N, dtype=np.int32)
            audited = np.zeros(N, dtype=np.int32)
            harm = np.zeros(N, dtype=np.int32)
            ema_ref_used = ema_ref  # reference history the rule saw this period
            if (not is_warmup) and (num_active >= n_min):
                flags = evaluate_rules(
                    chosen_prices, active_mask, self.cfg.rule_type, self.cfg.rule_param,
                    ema_ref=ema_ref_used, beta=self.cfg.ema_beta, min_active=n_min,
                    nash_prices=self.nash_prices,
                )
                if self.cfg.audit_mode == "top":
                    scores = rule_scores(chosen_prices, active_mask, self.cfg.rule_type,
                                         ema_ref=ema_ref_used, beta=self.cfg.ema_beta, nash_prices=self.nash_prices)
                    audited = select_top_audit_set(active_mask, scores, m.audit_budget, self.rng)
                else:
                    audited = sample_audit_set(active_mask, m.audit_budget, self.rng)
                harm = compute_counterfactual_harm(
                    chosen_prices, self.nash_prices, active_mask, audited,
                    self.cfg.rule_type, self.cfg.rule_param,
                    ema_ref=ema_ref_used, beta=self.cfg.ema_beta, min_active=n_min,
                )

            if num_active > 0:
                act_mean = float(np.mean(chosen_prices[active_mask != 0]))
                ema_ref = float(self.cfg.ema_beta * ema_ref + (1.0 - self.cfg.ema_beta) * act_mean)

            # Sanctions: audited & flagged firms not already serving a ban
            sanctioned = (audited != 0) & (flags != 0) & (ban_timers == 0)
            sanction_inducer = np.full(N, -1, dtype=np.int32)        # unilateral (defines W)
            sanction_joint_inducer = np.full(N, -1, dtype=np.int32)  # unilateral or coalition
            for j in np.where(sanctioned)[0]:
                k = self._find_inducer(j, chosen_prices, active_mask, ema_ref_used)
                sanction_inducer[j] = k
                sanction_joint_inducer[j] = k if k >= 0 else self._find_joint_inducer(
                    j, chosen_prices, active_mask, ema_ref_used)

            if self.cfg.penalty_mode == "escrow":
                # Flagged firms stay active; profit above the Nash level is diverted to escrow.
                for i in np.where((audited != 0) & (flags != 0))[0]:
                    profits[i] -= max(0.0, profits[i] - self.nash_profits[i])
                fresh_bans = np.zeros(N, dtype=bool)
                next_timers = ban_timers.copy()
            else:
                fresh_bans = sanctioned
                next_timers = update_ban_timers(ban_timers, flags, audited, m.ban_duration)
                for j in np.where(fresh_bans)[0]:
                    ban_inducer[j] = sanction_joint_inducer[j]

            # 5. Capital & bankruptcy dynamics
            next_capital = update_capital(capital, profits, m.fixed_cost, cap=cap_limit)
            newly_bankrupt = (capital > 0.0) & (next_capital <= 0.0)

            # 6. Rewards & learning
            total_profit = float(np.sum(profits))
            for i in range(N):
                if not is_q[i]:
                    continue
                agent = self.agents[i]
                if agent.has_pending:
                    reward_i = compute_agent_reward(
                        profit=profits[i],
                        Q=m.market_size,
                        CS=cs,
                        other_profits_sum=total_profit - float(profits[i]),
                        N=N,
                        harm_count=harm[i],
                        traits=agent.traits,
                    )
                    agent.accumulate_reward(reward_i)
                    if newly_bankrupt[i]:
                        agent.close_transition(None, terminal=True)
                if not self.cfg.freeze_learners:
                    agent.decay_epsilon()

            # Evaluation bookkeeping
            if in_eval:
                eval_prices.append(np.where(active_mask != 0, chosen_prices, np.nan))
                eval_profits.append(profits.copy())
                eval_cs.append(cs)

                for j in np.where(sanctioned)[0]:
                    total_sanctions += 1
                    k = sanction_inducer[j]
                    if k >= 0:
                        induced_sanctions += 1
                        induced_by[k] += 1
                    if sanction_joint_inducer[j] >= 0:
                        joint_induced_sanctions += 1
                n_fresh = int(np.sum(fresh_bans))
                total_fresh_bans += n_fresh
                induced_bans += int(np.sum(fresh_bans & (sanction_inducer >= 0)))
                joint_induced_bans += int(np.sum(fresh_bans & (sanction_joint_inducer >= 0)))

                for j in np.where(newly_bankrupt)[0]:
                    total_bankruptcies += 1
                    bankr_of[j] += 1
                    if ban_timers[j] > 0 and ban_inducer[j] >= 0:
                        induced_bankruptcies += 1
                        kills_by[ban_inducer[j]] += 1

            # Per-step timeline trace
            if self.cfg.record_trace and step >= trace_start:
                ref_p, thresh_p = self._reference_and_threshold(
                    chosen_prices, active_mask, num_active, ema_ref_used
                )
                step_events = []
                for j in np.where(fresh_bans)[0]:
                    k = sanction_joint_inducer[j]
                    if k >= 0:
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
                            "ban_duration": int(m.ban_duration),
                            "fixed_cost": float(m.fixed_cost),
                        }
                        step_events.append(ev)
                        trace_exploit_events.append(ev)
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
                    "active_mask": [int(x) for x in active_mask],
                    "active_count": int(num_active),
                    "ref_price": ref_p,
                    "threshold": thresh_p,
                    "exploit_events": step_events,
                })

            # Transition state
            capital = next_capital
            ban_timers = next_timers
            ban_inducer[ban_timers == 0] = -1
            lagged_prices = chosen_prices
            episode_step += 1

            # Episode reset (Q-tables persist). Open transitions are truncated, not
            # terminated: bootstrap from the last real observation before the reset.
            if self.cfg.legacy_reset:
                early_end = int(np.sum((ban_timers == 0) & (capital > 0.0))) <= 1
            else:
                early_end = not np.any(capital > 0.0)
            if episode_step >= self.cfg.episode_length or early_end:
                for i in range(N):
                    if is_q[i] and self.agents[i].has_pending:
                        s_next = self.indexer.encode(lagged_prices, self.indexer.status(ban_timers, capital), capital[i])
                        self.agents[i].close_transition(s_next)
                capital, ban_timers, lagged_prices = fresh_state()
                ema_ref = float(np.mean(lagged_prices))
                ban_inducer[:] = -1
                episode_step = 0
                if step >= eval_start and step + 1 < self.cfg.total_steps:
                    episodes_in_eval += 1

        # Truncate open transitions so repeated run() calls (e.g. Extension A epochs) start clean
        for i in range(N):
            if is_q[i] and self.agents[i].has_pending:
                self.agents[i].close_transition(
                    self.indexer.encode(lagged_prices, self.indexer.status(ban_timers, capital), capital[i])
                )

        eval_p_arr = np.array(eval_prices)
        active_counts = np.sum(~np.isnan(eval_p_arr), axis=0)
        price_sums = np.nansum(eval_p_arr, axis=0)
        # Average over periods in which each firm was active (inactive firms keep a stale price)
        mean_p = np.where(active_counts > 0, price_sums / np.maximum(active_counts, 1), self.nash_prices)
        return self._finalize(dict(
            mean_prices=mean_p,
            mean_profits=np.mean(np.array(eval_profits), axis=0),
            mean_consumer_surplus=float(np.mean(eval_cs)),
            eval_periods=len(eval_cs),
            total_fresh_bans=total_fresh_bans,
            induced_bans=induced_bans,
            total_sanctions=total_sanctions,
            induced_sanctions=induced_sanctions,
            total_bankruptcies=total_bankruptcies,
            induced_bankruptcies=induced_bankruptcies,
            induced_by=induced_by,
            kills_by=kills_by,
            episodes_in_eval=episodes_in_eval,
            joint_induced_bans=joint_induced_bans,
            joint_induced_sanctions=joint_induced_sanctions,
            bankr_of=bankr_of,
        ), trace_history, trace_exploit_events)

    def _finalize(self, raw: Dict[str, Any], trace_history=None, exploit_events=None) -> SimulationResults:
        """Builds SimulationResults from raw eval-window tallies (shared by both backends)."""
        N = self.N
        mean_pi = raw["mean_profits"]
        delta = compute_collusion_index(
            float(np.mean(mean_pi)), float(np.mean(self.nash_profits)), float(np.mean(self.mono_profits))
        )
        if self.cfg.penalty_mode == "escrow":
            num, den = raw["induced_sanctions"], raw["total_sanctions"]
        else:
            num, den = raw["induced_bans"], raw["total_fresh_bans"]
        w_rate = (num / den) if den > 0 else 0.0
        joint_num = raw["joint_induced_sanctions"] if self.cfg.penalty_mode == "escrow" else raw["joint_induced_bans"]
        w_joint = (joint_num / den) if den > 0 else 0.0

        mean_cs = float(raw["mean_consumer_surplus"])
        nash_welfare = self.nash_cs + float(np.sum(self.nash_profits))
        swl = 1.0 - ((mean_cs + float(np.sum(mean_pi))) / nash_welfare) if nash_welfare > 0 else 0.0
        b_rate = raw["total_bankruptcies"] / (N * max(1, raw["episodes_in_eval"]))

        q_agents = [self.agents[i] for i in range(N) if self.policies[i] == POLICY_Q]
        if q_agents:
            avg_unique = int(np.mean([a.unique_states_visited for a in q_agents]))
            avg_coverage = float(np.mean([a.state_coverage_ratio for a in q_agents]))
            avg_effective = int(np.mean([a.effective_states_visited(min_visits=5) for a in q_agents]))
        else:
            avg_unique, avg_coverage, avg_effective = 0, 0.0, 0
        avg_effective_cov = float(avg_effective / self.indexer.num_states) if self.indexer.num_states > 0 else 0.0

        return SimulationResults(
            weaponization_rate=float(w_rate),
            collusion_index=float(delta),
            social_welfare_loss=float(swl),
            mean_prices=np.asarray(raw["mean_prices"], dtype=np.float64),
            mean_profits=np.asarray(mean_pi, dtype=np.float64),
            bankruptcy_rate=float(min(1.0, b_rate)),
            total_fresh_bans=int(raw["total_fresh_bans"]),
            induced_bans=int(raw["induced_bans"]),
            unique_states_visited=avg_unique,
            state_coverage_ratio=avg_coverage,
            effective_states_visited=avg_effective,
            effective_state_coverage=avg_effective_cov,
            trace_data=trace_history,
            exploit_events=exploit_events or [],
            total_sanctions=int(raw["total_sanctions"]),
            induced_sanctions=int(raw["induced_sanctions"]),
            total_bankruptcies=int(raw["total_bankruptcies"]),
            induced_bankruptcies=int(raw["induced_bankruptcies"]),
            induced_by=np.asarray(raw["induced_by"], dtype=np.int64),
            kills_by=np.asarray(raw["kills_by"], dtype=np.int64),
            episodes_in_eval=int(raw["episodes_in_eval"]),
            firm_policies=list(self.policies),
            mean_consumer_surplus=mean_cs,
            eval_periods=int(raw["eval_periods"]),
            joint_induced_bans=int(raw["joint_induced_bans"]),
            joint_induced_sanctions=int(raw["joint_induced_sanctions"]),
            joint_weaponization_rate=float(w_joint),
            bankruptcies_of=np.asarray(raw["bankr_of"], dtype=np.int64) if "bankr_of" in raw else None,
        )
