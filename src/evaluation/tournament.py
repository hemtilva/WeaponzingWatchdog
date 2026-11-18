"""Axelrod-style Evolutionary Strategy Tournament for Heterogeneous Sellers (Option 3).

Evaluates the evolutionary stability and population ecology of four seller archetypes:
1. Pure Predator (mu = 0.0, lambda = 0.0, alpha = 0.0): Unconstrained profit seeker
2. Ethical Seller (mu = 1.0, lambda = 0.0, alpha = 0.0): Internalizes regulatory harm
3. Altruist Seller (mu = 0.0, lambda = 0.5, alpha = 0.0): Cares for consumer surplus
4. Risk-Averse Seller (mu = 0.0, lambda = 0.0, alpha = 1.5): Concave CRRA utility

Fitness is realized monetary profit over the match's evaluation window (not the
trait-adjusted reward), so archetypes compete on what they actually earn.
Each match trains fresh Q-learners; exploration decays to eps_min by
`explore_fraction` of the match so that the evaluation window reflects learned play.

Simulates multi-generation selection via discrete Replicator Dynamics:
  x_k^{t+1} = (1 - epsilon_m) * x_k^t * (f_k / bar{f}) + epsilon_m / K
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np

from src.engine.market import MarketConfig
from src.engine.rules import RuleType
from src.agents.traits import AgentTraits
from src.agents.q_learning import QLearningConfig
from src.agents.trainer import SimulationConfig, MultiAgentTrainer
from src.benchmarks.diagnostics import framing_market


ARCHETYPES: Dict[str, AgentTraits] = {
    "Predator": AgentTraits(alpha=0.0, lambda_altruism=0.0, mu_ethics=0.0),
    "Ethical": AgentTraits(alpha=0.0, lambda_altruism=0.0, mu_ethics=1.0),
    "Altruist": AgentTraits(alpha=0.0, lambda_altruism=0.5, mu_ethics=0.0),
    "RiskAverse": AgentTraits(alpha=1.5, lambda_altruism=0.0, mu_ethics=0.0),
}

ARCHETYPE_NAMES = list(ARCHETYPES.keys())
NUM_ARCHETYPES = len(ARCHETYPE_NAMES)


@dataclass
class TournamentConfig:
    """Hyperparameters for evolutionary strategy tournament."""
    num_generations: int = 30
    cohorts_per_gen: int = 6
    steps_per_match: int = 20_000
    warmup_steps: int = 0
    episode_length: int = 500
    eval_window: int = 5_000
    explore_fraction: float = 0.6     # epsilon reaches eps_min at this fraction of the match
    eps_min: float = 0.01
    num_firms: int = 4
    mutation_rate: float = 0.02
    rule_type: RuleType = RuleType.MEAN_RATIO
    rule_param: float = 0.20
    penalty_mode: str = "shutdown"
    ema_beta: float = 0.95
    seed: int = 42


@dataclass
class TournamentResults:
    """Results and ecological history of evolutionary tournament."""
    generations: int
    archetype_names: List[str]
    history_shares: np.ndarray        # Shape: (num_generations + 1, num_archetypes)
    history_fitness: np.ndarray       # Shape: (num_generations, num_archetypes)
    final_shares: Dict[str, float]
    dominant_archetype: str
    extinct_archetypes: List[str]


class EvolutionaryTournament:
    """Simulates population ecology of heterogeneous pricing strategies over generations."""

    def __init__(self, config: TournamentConfig, market_config: Optional[MarketConfig] = None):
        self.cfg = config
        # Viable (Nash profit > fixed cost) and lethal market, so bankruptcies come from bans
        self.m_cfg = market_config or framing_market(
            num_firms=config.num_firms, lethal=True, threshold_delta=config.rule_param,
        )
        self.rng = np.random.default_rng(config.seed)
        explore_steps = max(1, int(config.explore_fraction * config.steps_per_match))
        self.q_cfg = QLearningConfig(
            eps_min=config.eps_min,
            eps_decay=float(config.eps_min ** (1.0 / explore_steps)),
        )

    def run(self) -> TournamentResults:
        """Runs full multi-generation tournament with replicator dynamics."""
        K = NUM_ARCHETYPES
        # Initial uniform population distribution
        shares = np.full(K, 1.0 / K, dtype=np.float64)

        history_shares = [shares.copy()]
        history_fitness = []

        baseline_fitness = np.ones(K, dtype=np.float64)

        for gen in range(self.cfg.num_generations):
            payoffs_acc = [[] for _ in range(K)]

            for _ in range(self.cfg.cohorts_per_gen):
                # Sample 4 firms from current population distribution
                chosen_indices = self.rng.choice(K, size=self.cfg.num_firms, p=shares)
                cohort_traits = [ARCHETYPES[ARCHETYPE_NAMES[idx]] for idx in chosen_indices]

                sim_cfg = SimulationConfig(
                    market=self.m_cfg,
                    q_learning=self.q_cfg,
                    total_steps=self.cfg.steps_per_match,
                    warmup_steps=self.cfg.warmup_steps,
                    episode_length=self.cfg.episode_length,
                    rule_type=self.cfg.rule_type,
                    rule_param=self.cfg.rule_param,
                    penalty_mode=self.cfg.penalty_mode,
                    ema_beta=self.cfg.ema_beta,
                    eval_window=self.cfg.eval_window,
                    agent_traits=cohort_traits,
                    seed=int(self.rng.integers(0, 1_000_000)),
                )

                trainer = MultiAgentTrainer(sim_cfg)
                res = trainer.run()

                # Accumulate realized profits for each firm's archetype
                for firm_i, arch_idx in enumerate(chosen_indices):
                    pi_i = float(res.mean_profits[firm_i])
                    payoffs_acc[arch_idx].append(pi_i)

            # Compute generation fitness vector
            gen_fitness = np.zeros(K, dtype=np.float64)
            for k in range(K):
                if len(payoffs_acc[k]) > 0:
                    f_val = float(np.mean(payoffs_acc[k]))
                    # Shift fitness positive to ensure non-negative replicator dynamics
                    gen_fitness[k] = max(0.01, f_val + 0.10)
                    baseline_fitness[k] = gen_fitness[k]
                else:
                    gen_fitness[k] = baseline_fitness[k]

            history_fitness.append(gen_fitness.copy())

            # Replicator Dynamics update
            avg_pop_fitness = float(np.sum(shares * gen_fitness))
            if avg_pop_fitness > 0.0:
                raw_next_shares = shares * (gen_fitness / avg_pop_fitness)
            else:
                raw_next_shares = shares.copy()

            # Apply mutation drift to preserve ecological viability
            mut = self.cfg.mutation_rate
            next_shares = (1.0 - mut) * raw_next_shares + (mut / K)
            next_shares = next_shares / np.sum(next_shares)

            shares = next_shares
            history_shares.append(shares.copy())

        hist_shares_arr = np.array(history_shares)
        hist_fitness_arr = np.array(history_fitness)

        final_dict = {
            ARCHETYPE_NAMES[i]: float(shares[i]) for i in range(K)
        }
        dominant = max(final_dict, key=final_dict.get)
        extinct = [name for name, s in final_dict.items() if s < 0.05]

        return TournamentResults(
            generations=self.cfg.num_generations,
            archetype_names=ARCHETYPE_NAMES,
            history_shares=hist_shares_arr,
            history_fitness=hist_fitness_arr,
            final_shares=final_dict,
            dominant_archetype=dominant,
            extinct_archetypes=extinct,
        )
