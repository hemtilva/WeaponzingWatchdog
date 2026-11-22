"""Unit tests for Deep Q-Network (DQN) continuous scalability ablation (Option 2).

Verifies:
1. ContinuousStateEncoder dimensions for N = 3, 4, 5.
2. DQNNetwork forward pass and tensor shapes.
3. ReplayBuffer storage and batch sampling.
4. DQNAgent action selection and Bellman loss computation.
5. DQNTrainer end-to-end execution for N = 5 continuous oligopoly.
"""

import numpy as np
import torch

from src.engine.market import MarketConfig
from src.engine.rules import RuleType
from src.agents.dqn_agent import DQNConfig, ContinuousStateEncoder, DQNNetwork, ReplayBuffer, DQNAgent
from src.agents.dqn_trainer import DQNSimulationConfig, DQNTrainer


def test_continuous_state_encoder_dimensions():
    """ContinuousStateEncoder produces 2*N + 1 feature vectors."""
    for n in [3, 4, 5]:
        enc = ContinuousStateEncoder(num_firms=n, p_max=2.5, initial_capital=1.0)
        assert enc.state_dim == 2 * n + 1

        prices = [1.5] * n
        bans = [0] * n
        cap = 1.0
        vec = enc.encode(prices, bans, cap)
        assert vec.shape == (2 * n + 1,)
        assert np.all(vec >= 0.0)


def test_dqn_network_forward():
    """DQNNetwork correctly projects continuous state to M action values."""
    net = DQNNetwork(input_dim=11, num_actions=7, hidden_dims=(64, 64))
    x = torch.randn(16, 11)
    out = net(x)
    assert out.shape == (16, 7)


def test_replay_buffer_sampling():
    """ReplayBuffer stores transitions and samples mini-batches accurately."""
    device = torch.device("cpu")
    buf = ReplayBuffer(capacity=100, state_dim=9, device=device)
    for i in range(20):
        s = np.zeros(9, dtype=np.float32) + i
        ns = np.zeros(9, dtype=np.float32) + i + 1
        buf.push(s, i % 5, float(i), ns, done=(i == 19))

    assert len(buf) == 20
    s_b, a_b, r_b, ns_b, d_b = buf.sample(batch_size=8)
    assert s_b.shape == (8, 9)
    assert a_b.shape == (8,)
    assert r_b.shape == (8,)
    assert ns_b.shape == (8, 9)
    assert d_b.shape == (8,)


def test_dqn_agent_update():
    """DQNAgent performs Bellman gradient update when buffer has enough samples."""
    cfg = DQNConfig(batch_size=8, buffer_capacity=100, learning_rate=1e-3)
    price_grid = np.linspace(0.5, 2.5, 7)
    agent = DQNAgent(firm_idx=0, num_firms=4, price_grid=price_grid, config=cfg)

    rng = np.random.default_rng(42)
    s = np.ones(agent.state_dim, dtype=np.float32)

    # Action selection
    a = agent.select_action(s, rng)
    assert 0 <= a < len(price_grid)

    # Fill buffer
    loss = None
    for step in range(12):
        ns = s.copy()
        l = agent.update(s, a, 1.0, ns, done=False)
        if l is not None:
            loss = l

    assert loss is not None, "Agent must compute Bellman loss once batch_size is reached"
    assert loss >= 0.0


def test_dqn_trainer_n5_simulation():
    """DQNTrainer executes N = 5 multi-firm market clearing and produces valid metrics."""
    m_cfg = MarketConfig(
        num_firms=5,
        marginal_costs=[0.5] * 5,
        qualities=[2.0] * 5,
        price_sensitivity=0.25,
        fixed_cost=0.10,
        initial_capital=1.0,
        ban_duration=3,
        threshold_delta=0.20,
    )
    dqn_cfg = DQNConfig(batch_size=16, buffer_capacity=500, learning_rate=1e-3)
    sim_cfg = DQNSimulationConfig(
        market=m_cfg,
        dqn_config=dqn_cfg,
        total_steps=120,
        warmup_steps=20,
        episode_length=50,
        eval_window=50,
        rule_type=RuleType.MEAN_RATIO,
        rule_param=0.20,
        seed=42,
    )
    trainer = DQNTrainer(sim_cfg)
    results = trainer.run()

    assert results.mean_prices.shape == (5,)
    assert results.mean_profits.shape == (5,)
    assert 0.0 <= results.weaponization_rate <= 1.0
    assert not np.isnan(results.collusion_index)
