"""Deep Q-Network (DQN) agent with continuous state representations and experience replay.

Provides scalable neural network function approximation for multi-agent pricing games
where discrete state indexers suffer the curse of dimensionality (e.g. N = 5 firms,
where discrete tabular state space exceeds 1,600,000 states).
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Sequence
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .traits import AgentTraits


@dataclass
class DQNConfig:
    """Hyperparameters for Deep Q-Network pricing agents."""
    learning_rate: float = 1e-3
    gamma: float = 0.95
    eps_start: float = 1.0
    eps_min: float = 0.02
    eps_decay: float = 0.9995
    buffer_capacity: int = 20_000
    batch_size: int = 64
    target_sync_freq: int = 100
    hidden_dims: Tuple[int, ...] = (128, 128)
    device: str = "cpu"


class DQNNetwork(nn.Module):
    """Multi-Layer Perceptron (MLP) for continuous-state Q-function approximation."""

    def __init__(self, input_dim: int, num_actions: int, hidden_dims: Tuple[int, ...] = (128, 128)):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, num_actions))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ReplayBuffer:
    """Fixed-capacity experience replay buffer with uniform random sampling."""

    def __init__(self, capacity: int, state_dim: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = 1.0 if done else 0.0

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.states[indices], dtype=torch.float32, device=self.device),
            torch.as_tensor(self.actions[indices], dtype=torch.int64, device=self.device),
            torch.as_tensor(self.rewards[indices], dtype=torch.float32, device=self.device),
            torch.as_tensor(self.next_states[indices], dtype=torch.float32, device=self.device),
            torch.as_tensor(self.dones[indices], dtype=torch.float32, device=self.device),
        )

    def __len__(self) -> int:
        return self.size


class ContinuousStateEncoder:
    """Constructs normalized continuous state vectors for arbitrary N firms.

    State representation s_t = [p_1/p_max, ..., p_N/p_max, b_1, ..., b_N, K_i/K0]
    Dimension = 2 * N + 1.
    """

    def __init__(self, num_firms: int, p_max: float, initial_capital: float):
        self.num_firms = num_firms
        self.p_max = float(p_max) if p_max > 0.0 else 1.0
        self.initial_capital = float(initial_capital) if initial_capital > 0.0 else 1.0
        self.state_dim = 2 * num_firms + 1

    def encode(
        self,
        lagged_prices: Sequence[float],
        ban_indicators: Sequence[int],
        capital: float,
    ) -> np.ndarray:
        norm_prices = [p / self.p_max for p in lagged_prices[:self.num_firms]]
        norm_bans = [1.0 if b > 0 else 0.0 for b in ban_indicators[:self.num_firms]]
        norm_cap = capital / self.initial_capital
        vec = np.array(norm_prices + norm_bans + [norm_cap], dtype=np.float32)
        return vec


class DQNAgent:
    """Deep Q-Network pricing agent with private traits and target network stabilization."""

    def __init__(
        self,
        firm_idx: int,
        num_firms: int,
        price_grid: np.ndarray,
        config: DQNConfig,
        initial_capital: float = 1.0,
        traits: Optional[AgentTraits] = None,
    ):
        self.firm_idx = firm_idx
        self.num_firms = num_firms
        self.price_grid = np.asarray(price_grid, dtype=np.float64)
        self.num_actions = len(self.price_grid)
        self.cfg = config
        self.traits = traits or AgentTraits()

        self.device = torch.device(config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu")

        p_max = float(np.max(self.price_grid))
        self.encoder = ContinuousStateEncoder(num_firms, p_max, initial_capital)
        self.state_dim = self.encoder.state_dim

        # Policy and Target Networks
        self.policy_net = DQNNetwork(self.state_dim, self.num_actions, config.hidden_dims).to(self.device)
        self.target_net = DQNNetwork(self.state_dim, self.num_actions, config.hidden_dims).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.learning_rate)
        self.criterion = nn.SmoothL1Loss()
        self.buffer = ReplayBuffer(config.buffer_capacity, self.state_dim, self.device)

        self.epsilon = float(config.eps_start)
        self.total_steps = 0
        self.recent_losses: List[float] = []

    def select_action(self, state_vec: np.ndarray, rng: np.random.Generator) -> int:
        """Selects action via epsilon-greedy policy over continuous state vector."""
        if rng.random() < self.epsilon:
            return int(rng.integers(0, self.num_actions))

        with torch.no_grad():
            s_tensor = torch.as_tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_vals = self.policy_net(s_tensor).cpu().numpy().squeeze(0)
            # Break ties randomly among max Q actions
            max_q = np.max(q_vals)
            best_actions = np.where(np.isclose(q_vals, max_q, atol=1e-6))[0]
            return int(rng.choice(best_actions))

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> Optional[float]:
        """Stores transition in buffer and performs mini-batch gradient descent update."""
        self.buffer.push(state, action, reward, next_state, done)
        self.total_steps += 1

        if len(self.buffer) < self.cfg.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.cfg.batch_size)

        # Compute current Q(s, a)
        q_current = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target y = r + gamma * max_a Q_target(s', a) * (1 - done)
        with torch.no_grad():
            q_next_max = self.target_net(next_states).max(1)[0]
            q_targets = rewards + self.cfg.gamma * q_next_max * (1.0 - dones)

        loss = self.criterion(q_current, q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        loss_val = float(loss.item())
        self.recent_losses.append(loss_val)
        if len(self.recent_losses) > 100:
            self.recent_losses.pop(0)

        # Target network sync
        if self.total_steps % self.cfg.target_sync_freq == 0:
            self.sync_target()

        return loss_val

    def sync_target(self) -> None:
        """Copies policy network weights to target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self) -> None:
        """Decays exploration parameter epsilon exponentially."""
        self.epsilon = max(self.cfg.eps_min, self.epsilon * self.cfg.eps_decay)
