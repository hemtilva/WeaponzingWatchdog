"""Tests for trajectory-level (but-for) attribution, src/evaluation/replay_attribution.py."""
import numpy as np
import pytest

from experiments.common import Job, build
from src.engine.rules import RuleType
from src.evaluation.replay_attribution import ReplayEngine, attribute


def _engine(rule, beta=0.95, policies=("nash", "nash", "nash")):
    tr = build(Job(tag="t", seed=1, rule=int(rule), rule_param=0.10, ema_beta=beta,
                   policies=list(policies), train_steps=1000))
    return ReplayEngine(tr)


def _patient(e, low=0.15):
    lo = int(np.argmin(np.abs(e.grid - low)))
    hi = len(e.grid) - 1
    e.policies[0] = lambda st, n: lo if n >= e.n_min else hi
    return e


def test_honest_market_has_nothing_to_attribute():
    r = attribute(_engine(RuleType.MEAN_RATIO), episodes=3)
    assert r["bans_per_ep"] == 0.0 and r["causal_W_wNone"] == 0.0


def test_crash_attacker_under_mean_rule_is_fully_attributed_at_every_window():
    r = attribute(_engine(RuleType.MEAN_RATIO, policies=("attacker", "nash", "nash")), episodes=3)
    assert r["bans_per_ep"] > 0
    for key in ("one_period_W", "causal_W_w1", "causal_W_w5", "causal_W_wNone"):
        assert r[key] == pytest.approx(1.0)
    assert r["causal_bankruptcies_per_ep_wNone"] == pytest.approx(r["bankruptcies_during_bans_per_ep"])


@pytest.mark.parametrize("beta", [0.80, 0.95])
def test_patient_attacker_under_ema_is_missed_by_one_period_test_but_caught_by_replay(beta):
    r = attribute(_patient(_engine(RuleType.EMA_MEAN_RATIO, beta)), episodes=3)
    assert r["bans_per_ep"] == pytest.approx(2.0)
    assert r["one_period_W"] < 1.0                       # the simulator's metric undercounts
    assert r["causal_W_w5"] == pytest.approx(1.0)        # a short replay window recovers it
    assert r["causal_bankruptcies_per_ep_wNone"] == pytest.approx(2.0)


def test_replay_dynamics_match_the_engine_for_a_patient_attacker():
    """A seller holding 0.15 earns more than its fixed cost, so it survives, while both
    Nash rivals are framed into bankruptcy (as in experiments/patient_attacker.py)."""
    e = _patient(_engine(RuleType.MEAN_RATIO))
    events, st = e.run_episode(np.random.default_rng(0).random(500))
    assert st.capital[0] > 0.0
    assert st.capital[1] == 0.0 and st.capital[2] == 0.0
    assert sum(int(ev["bankrupt"][1:].sum()) for ev in events) == 2
