"""Unit tests for Axelrod-style Evolutionary Strategy Tournament (Option 3).

Verifies:
1. Archetype definitions (Predator, Ethical, Altruist, RiskAverse).
2. Population share conservation (sum = 1.0) under replicator dynamics.
3. Tournament execution and generation tracking.
"""

import numpy as np
from src.engine.rules import RuleType
from src.evaluation.tournament import (
    TournamentConfig,
    EvolutionaryTournament,
    ARCHETYPES,
    ARCHETYPE_NAMES,
)


def test_archetype_definitions():
    """Verifies all four seller archetypes possess distinct behavioral traits."""
    assert len(ARCHETYPES) == 4
    assert ARCHETYPES["Predator"].mu_ethics == 0.0
    assert ARCHETYPES["Ethical"].mu_ethics == 1.0
    assert ARCHETYPES["Altruist"].lambda_altruism == 0.5
    assert ARCHETYPES["RiskAverse"].alpha == 1.5


def test_tournament_population_conservation():
    """Replicator dynamics must conserve total population share (sum == 1.0)."""
    cfg = TournamentConfig(
        num_generations=5,
        cohorts_per_gen=2,
        steps_per_match=150,
        warmup_steps=20,
        episode_length=50,
        eval_window=50,
        seed=42,
    )
    tournament = EvolutionaryTournament(cfg)
    results = tournament.run()

    # Shape must be (num_generations + 1, num_archetypes)
    assert results.history_shares.shape == (6, 4)
    for g in range(6):
        assert np.isclose(np.sum(results.history_shares[g]), 1.0, atol=1e-5), f"Shares at gen {g} must sum to 1.0"
        assert np.all(results.history_shares[g] >= 0.0), f"Shares at gen {g} must be non-negative"


def test_tournament_runs_under_rules():
    """Tournament produces valid dominant and extinct archetypes."""
    cfg = TournamentConfig(
        num_generations=3,
        cohorts_per_gen=2,
        steps_per_match=150,
        warmup_steps=20,
        episode_length=50,
        eval_window=50,
        rule_type=RuleType.MEDIAN_RATIO,
        seed=101,
    )
    tournament = EvolutionaryTournament(cfg)
    results = tournament.run()

    assert results.dominant_archetype in ARCHETYPE_NAMES
    assert isinstance(results.final_shares, dict)
    assert len(results.final_shares) == 4
