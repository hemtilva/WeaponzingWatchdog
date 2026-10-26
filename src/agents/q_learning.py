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

    Observation components:
      1. Lagged joint prices: p_{t-1} in {0, ..., M-1}^3 -> M^3 values
      2. Ban indicators: b_t in {0, 1}^3 -> 8 values
      3. Capital bin: kappa in {low=0, mid=1, high=2} -> 3 values

    Total states: M^3 * 8 * 3. With M=11, total = 31,944 states.
    """

    def __init__(self, price_grid: np.ndarray, fixed_cost: float = 0.2):
        self.price_grid = np.asarray(price_grid, dtype=np.float64)
        self.M = len(self.price_grid)
        self.fixed_cost = fixed_cost
        self.num_states = (self.M ** 3) * 8 * 3

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
        p_idx0 = self.snap_price_to_index(lagged_prices[0])
        p_idx1 = self.snap_price_to_index(lagged_prices[1])
        p_idx2 = self.snap_price_to_index(lagged_prices[2])

        price_combo = (p_idx0 * self.M + p_idx1) * self.M + p_idx2
        ban_combo = ((int(ban_indicators[0] > 0) * 2) + int(ban_indicators[1] > 0)) * 2 + int(ban_indicators[2] > 0)
        cap_bin = self.get_capital_bin(capital)

        # Flat state index: (price_combo * 8 + ban_combo) * 3 + cap_bin
        state_idx = (price_combo * 8 + ban_combo) * 3 + cap_bin
        return int(state_idx)

    def decode(self, state_idx: int) -> Tuple[Tuple[int, int, int], Tuple[int, int, int], int]:
        """Decodes flat state index back into ((p0, p1, p2), (b0, b1, b2), cap_bin)."""
        rem = state_idx
        cap_bin = rem % 3
        rem //= 3
        ban_combo = rem % 8
        price_combo = rem // 8

        b2 = ban_combo % 2
        ban_combo //= 2
        b1 = ban_combo % 2
        b0 = ban_combo // 2

        p2 = price_combo % self.M
        price_combo //= self.M
        p1 = price_combo % self.M
        p0 = price_combo // self.M

        return ((p0, p1, p2), (b0, b1, b2), cap_bin)


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

    def select_action(self, state_idx: int, rng: np.random.Generator) -> int:
        """Selects action via epsilon-greedy policy with uniform random tie-breaking."""
        if rng.random() < self.epsilon:
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
        target = reward
        if not done:
            target += self.cfg.gamma * float(np.max(self.Q[next_state_idx]))

        current_q = self.Q[state_idx, action_idx]
        self.Q[state_idx, action_idx] += np.float32(
            self.cfg.learning_rate * (target - current_q)
        )

    def decay_epsilon(self):
        """Decays epsilon towards eps_min."""
        self.epsilon = max(self.cfg.eps_min, self.epsilon * self.cfg.eps_decay)

