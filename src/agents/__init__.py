"""Multi-agent learning, behavioral traits, and Q-learning agents."""

from .traits import AgentTraits, crra_utility, compute_agent_reward
from .q_learning import QLearningConfig, StateIndexer, QLearningAgent
from .trainer import SimulationConfig, SimulationResults, MultiAgentTrainer

__all__ = [
    "AgentTraits",
    "crra_utility",
    "compute_agent_reward",
    "QLearningConfig",
    "StateIndexer",
    "QLearningAgent",
    "SimulationConfig",
    "SimulationResults",
    "MultiAgentTrainer",
]

