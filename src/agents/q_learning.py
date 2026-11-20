"""Tabular Q-learning agent with compact state indexer and exploration schedule."""

from dataclasses import dataclass
from typing import Tuple, Sequence, Optional
import numpy as np

from .traits import AgentTraits


@dataclass
class QLearningConfig:
    """Hyperparameters for tabular Q-learning pricing agents."""
    learning_rate: float = 0.10
    gamma: float = 0.95
    eps_start: float = 1.0
    eps_min: float = 0.01
    eps_decay: float = 0.99995  # Exponential decay multiplier per step
    fixed_cost: float = 0.2     # Phi for capital binning


class StateIndexer:
    """Compact bijection between discrete market observations and flat state indices.

    Supports arbitrary N active firms (e.g. N=3 or N=4 standard):
      1. Lagged joint prices: p_{t-1} in {0, ..., M-1}^N -> M^N values
      2. Inactivity indicators: b_t in {0, 1}^N (banned OR bankrupt) -> 2^N values
      3. Capital bin: kappa in {low=0, mid=1, high=2} -> 3 values

    Total states: (M^N) * (2^N) * 3.
      - N=3, M=11: 11^3 * 8 * 3 = 31,944 states.
      - N=4, M=7:   7^4 * 16 * 3 = 115,248 states (~110,000 states).

    Ban-timer ablation (`timer_levels` = T_ban): the inactivity bit is replaced by the exact
    status of each firm, 0 (active), 1..T_ban (remaining ban) or T_ban + 1 (bankrupt), so the
    status part has (T_ban + 2)^N values.
    """

    def __init__(self, price_grid: np.ndarray, fixed_cost: float = 0.2, num_firms: int = 3,
                 timer_levels: Optional[int] = None):
        self.price_grid = np.asarray(price_grid, dtype=np.float64)
        self.M = len(self.price_grid)
        self.fixed_cost = fixed_cost
        self.num_firms = num_firms
        self.timer_levels = timer_levels
        self.base = 2 if timer_levels is None else int(timer_levels) + 2
        self.num_bans = self.base ** num_firms
        self.num_states = (self.M ** num_firms) * self.num_bans * 3

    def status(self, timers: np.ndarray, capital: np.ndarray) -> np.ndarray:
        """Per-firm status codes for `encode` (mirrors Sim::status in src/csrc/sim.cpp)."""
        timers = np.asarray(timers)
        capital = np.asarray(capital)
        if self.timer_levels is None:
            return ((timers > 0) | (capital <= 0.0)).astype(np.int64)
        return np.where(capital <= 0.0, self.timer_levels + 1, np.minimum(timers, self.timer_levels)).astype(np.int64)

    def get_capital_bin(self, capital: float) -> int:
        """Discretizes continuous capital into 3 bins:

        0 (low):  K <= 2*Phi (in immediate danger of ban-induced bankruptcy)
        1 (mid):  2*Phi < K <= 5*Phi
        2 (high): K > 5*Phi
        """
        if capital <= 2.0 * self.fixed_cost:
            return 0
        elif capital <= 5.0 * self.fixed_cost:
            return 1
        return 2

    def snap_price_to_index(self, price: float) -> int:
        """Finds nearest price grid index."""
        return int(np.argmin(np.abs(self.price_grid - price)))

    def encode(
        self,
        lagged_prices: Sequence[float],
        ban_indicators: Sequence[int],
        capital: float,
    ) -> int:
        """Encodes observation into flat integer state index via Horner's method."""
        price_combo = 0
        for p in lagged_prices[:self.num_firms]:
            p_idx = self.snap_price_to_index(p)
            price_combo = price_combo * self.M + p_idx

        ban_combo = 0
        for b in ban_indicators[:self.num_firms]:
            code = int(b > 0) if self.base == 2 else int(b)
            ban_combo = ban_combo * self.base + code

        cap_bin = self.get_capital_bin(capital)
        state_idx = (price_combo * self.num_bans + ban_combo) * 3 + cap_bin
        return int(state_idx)

    def decode(self, state_idx: int) -> Tuple[Tuple[int, ...], Tuple[int, ...], int]:
        """Decodes flat state index back into (prices, bans, cap_bin)."""
        rem = state_idx
        cap_bin = rem % 3
        rem //= 3

        ban_combo = rem % self.num_bans
        price_combo = rem // self.num_bans

        bans = []
        for _ in range(self.num_firms):
            bans.append(ban_combo % self.base)
            ban_combo //= self.base
        bans = tuple(reversed(bans))

        prices = []
        for _ in range(self.num_firms):
            prices.append(price_combo % self.M)
            price_combo //= self.M
        prices = tuple(reversed(prices))

        return (prices, bans, cap_bin)


class QLearningAgent:
    """Decentralized tabular Q-learning seller with private behavioral traits."""

    def __init__(
        self,
        firm_idx: int,
        config: QLearningConfig,
        indexer: StateIndexer,
        traits: Optional[AgentTraits] = None,
        init_q: float = 0.0,
    ):
        self.firm_idx = firm_idx
        self.cfg = config
        self.indexer = indexer
        self.traits = traits or AgentTraits()

        self.num_actions = indexer.M
        self.num_states = indexer.num_states

        # Q-table shape: (num_states, num_actions)
        self.Q = np.full((self.num_states, self.num_actions), init_q, dtype=np.float32)
        self.epsilon = float(config.eps_start)

        # State visitation tracking (Feedback 1)
        self.state_visit_counts = np.zeros(self.num_states, dtype=np.int32)

        # Open (semi-Markov) transition: (state, action, accumulated reward, discount).
        # A transition stays open while the agent is suspended, so the reward of the
        # zero-revenue ban periods is credited to the action that preceded the ban,
        # and bootstrapping happens from the next state in which the agent acts again.
        self._pending: Optional[Tuple[int, int, float, float]] = None

    @property
    def unique_states_visited(self) -> int:
        """Returns total distinct states visited by this agent."""
        return int(np.count_nonzero(self.state_visit_counts))

    @property
    def state_coverage_ratio(self) -> float:
        """Returns fraction of total state space explored in [0, 1]."""
        if self.num_states == 0:
            return 0.0
        return float(self.unique_states_visited) / float(self.num_states)

    def effective_states_visited(self, min_visits: int = 5) -> int:
        """Returns count of states visited at least min_visits times."""
        return int(np.count_nonzero(self.state_visit_counts >= min_visits))

    def select_action(self, state_idx: int, rng: np.random.Generator, explore: bool = True) -> int:
        """Selects action via epsilon-greedy policy with uniform random tie-breaking.

        explore=False gives the greedy policy (used for frozen evaluation).
        """
        if explore and rng.random() < self.epsilon:
            return int(rng.integers(0, self.num_actions))

        # Greedy choice with uniform random tie breaking
        q_row = self.Q[state_idx]
        max_q = np.max(q_row)
        best_actions = np.where(np.abs(q_row - max_q) < 1e-6)[0]
        if len(best_actions) == 1:
            return int(best_actions[0])
        return int(rng.choice(best_actions))

    def update(
        self,
        state_idx: int,
        action_idx: int,
        reward: float,
        next_state_idx: int,
        done: bool = False,
    ):
        """Standard Bellman temporal difference update:

        Q(s, a) <- Q(s, a) + alpha * [reward + gamma * max_a' Q(s', a') * (1 - done) - Q(s, a)]
        """
        # Record state visit
        self.state_visit_counts[state_idx] += 1

        target = reward
        if not done:
            target += self.cfg.gamma * float(np.max(self.Q[next_state_idx]))

        current_q = self.Q[state_idx, action_idx]
        self.Q[state_idx, action_idx] += np.float32(
            self.cfg.learning_rate * (target - current_q)
        )

    # ------------------------------------------------------------------
    # Semi-Markov transition interface (used by the trainers)
    # ------------------------------------------------------------------
    @property
    def has_pending(self) -> bool:
        return self._pending is not None

    def begin_transition(self, state_idx: int, action_idx: int) -> None:
        """Opens a transition for the action just taken in `state_idx`."""
        self._pending = (int(state_idx), int(action_idx), 0.0, 1.0)

    def accumulate_reward(self, reward: float) -> None:
        """Adds one period's reward to the open transition (discounted by elapsed periods)."""
        if self._pending is None:
            return
        s, a, acc, disc = self._pending
        self._pending = (s, a, acc + disc * float(reward), disc * self.cfg.gamma)

    def close_transition(self, next_state_idx: Optional[int], terminal: bool = False) -> None:
        """Closes the open transition and applies the TD update.

        target = R_acc + gamma^k * max_a' Q(s', a')   (no bootstrap if terminal)
        """
        if self._pending is None:
            return
        s, a, acc, disc = self._pending
        self._pending = None

        self.state_visit_counts[s] += 1
        target = acc
        if not terminal and next_state_idx is not None:
            target += disc * float(np.max(self.Q[next_state_idx]))
        self.Q[s, a] += np.float32(self.cfg.learning_rate * (target - self.Q[s, a]))

    def greedy_action(self, state_idx: int) -> int:
        """Greedy action (lowest index on ties, deterministic)."""
        return int(np.argmax(self.Q[state_idx]))

    def decay_epsilon(self):
        """Decays epsilon towards eps_min."""
        self.epsilon = max(self.cfg.eps_min, self.epsilon * self.cfg.eps_decay)

