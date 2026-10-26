"""Unit tests for StateIndexer and QLearningAgent."""

import numpy as np
import pytest
from src.agents.q_learning import StateIndexer, QLearningAgent, QLearningConfig
from src.agents.traits import AgentTraits


def test_state_indexer_encode_decode_roundtrip():
    price_grid = np.linspace(1.0, 2.5, 7)
    indexer = StateIndexer(price_grid, fixed_cost=0.2)

    # Test edge states and random states
    rng = np.random.default_rng(42)
    for _ in range(50):
        p_indices = rng.integers(0, 7, size=3)
        prices = [price_grid[idx] for idx in p_indices]
        bans = rng.integers(0, 2, size=3)
        cap = rng.uniform(0.0, 3.0)

        s_idx = indexer.encode(prices, bans, cap)
        assert 0 <= s_idx < indexer.num_states

        dec_p, dec_b, dec_cap = indexer.decode(s_idx)
        assert dec_p == tuple(p_indices)
        assert dec_b == tuple(bans)
        assert dec_cap == indexer.get_capital_bin(cap)


def test_capital_binning():
    indexer = StateIndexer(np.linspace(1.0, 2.0, 5), fixed_cost=0.2)
    # Low: <= 2 * 0.2 = 0.4
    assert indexer.get_capital_bin(0.0) == 0
    assert indexer.get_capital_bin(0.4) == 0
    # Mid: 0.4 < K <= 5 * 0.2 = 1.0
    assert indexer.get_capital_bin(0.5) == 1
    assert indexer.get_capital_bin(1.0) == 1
    # High: > 1.0
    assert indexer.get_capital_bin(1.01) == 2
    assert indexer.get_capital_bin(5.0) == 2


def test_q_learning_action_selection():
    price_grid = np.linspace(1.0, 2.0, 5)
    indexer = StateIndexer(price_grid, fixed_cost=0.2)
    cfg = QLearningConfig(eps_start=0.0)  # Pure greedy
    agent = QLearningAgent(firm_idx=0, config=cfg, indexer=indexer)

    # Set Q-values for state 100
    agent.Q[100, :] = [0.1, 0.5, 0.9, 0.2, 0.4]
    rng = np.random.default_rng(123)

    # Greedy action must select index 2 (Q=0.9)
    action = agent.select_action(state_idx=100, rng=rng)
    assert action == 2


def test_q_learning_bellman_update():
    price_grid = np.linspace(1.0, 2.0, 5)
    indexer = StateIndexer(price_grid, fixed_cost=0.2)
    cfg = QLearningConfig(learning_rate=0.5, gamma=0.8)
    agent = QLearningAgent(firm_idx=0, config=cfg, indexer=indexer)

    # Initial Q is 0
    assert agent.Q[10, 1] == 0.0
    # Let max Q at s'=20 be 10.0
    agent.Q[20, 3] = 10.0

    # Update with reward = 2.0
    # Target = 2.0 + 0.8 * 10.0 = 10.0
    # New Q = 0 + 0.5 * (10.0 - 0) = 5.0
    agent.update(state_idx=10, action_idx=1, reward=2.0, next_state_idx=20, done=False)
    assert np.isclose(agent.Q[10, 1], 5.0)


def test_epsilon_decay():
    price_grid = np.linspace(1.0, 2.0, 5)
    indexer = StateIndexer(price_grid, fixed_cost=0.2)
    cfg = QLearningConfig(eps_start=1.0, eps_min=0.1, eps_decay=0.9)
    agent = QLearningAgent(firm_idx=0, config=cfg, indexer=indexer)

    assert agent.epsilon == 1.0
    agent.decay_epsilon()
    assert np.isclose(agent.epsilon, 0.9)

    for _ in range(50):
        agent.decay_epsilon()
    assert agent.epsilon == 0.1  # Bounded below by eps_min
