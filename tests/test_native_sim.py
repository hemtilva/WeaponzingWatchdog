"""The native C++ loop (src/csrc/sim.cpp) must reproduce the Python reference loop.

Randomness is removed so the two backends can be compared exactly: every active firm is
audited each period (audit_budget = N) and learners act greedily on Q-tables with
distinct random values (no exploration, no ties). Then counts must match exactly and
Q-tables to float32 precision.
"""

import copy
import numpy as np
import pytest

from src.engine.rules import RuleType
from src.engine.native_sim import is_native_sim_available
from src.agents.q_learning import QLearningConfig
from src.agents.traits import AgentTraits
from src.agents.trainer import SimulationConfig, MultiAgentTrainer
from src.benchmarks.diagnostics import framing_market

pytestmark = pytest.mark.skipif(not is_native_sim_available(), reason="native engine not built")

COUNT_FIELDS = (
    "total_fresh_bans", "induced_bans", "total_sanctions", "induced_sanctions",
    "total_bankruptcies", "induced_bankruptcies", "episodes_in_eval", "eval_periods",
)


def _deterministic_cfg(n=3, rule=RuleType.MEAN_RATIO, param=0.10, policies=None, penalty="shutdown",
                       traits=None, steps=3000, lethal=True):
    market = framing_market(num_firms=n, lethal=lethal, threshold_delta=param if rule != RuleType.PRICE_CAP else 0.10)
    market.audit_budget = n  # audit everyone: no audit-sampling randomness
    return SimulationConfig(
        market=market,
        q_learning=QLearningConfig(eps_start=0.0, eps_min=0.0, learning_rate=0.1, gamma=0.95),
        total_steps=steps, warmup_steps=200, episode_length=300, eval_window=steps // 2,
        rule_type=rule, rule_param=param, firm_policies=policies or ["q"] * n,
        penalty_mode=penalty, agent_traits=traits, ema_beta=0.9, seed=5,
    )


def _pair(cfg):
    trainers = []
    for backend in ("python", "native"):
        c = copy.deepcopy(cfg)
        c.backend = backend
        tr = MultiAgentTrainer(c)
        rng = np.random.default_rng(123)
        for a in tr.agents:  # distinct values -> deterministic greedy choices
            a.Q[:] = rng.random(a.Q.shape, dtype=np.float32)
        trainers.append(tr)
    return trainers


def _assert_same(res_py, res_c):
    for f in COUNT_FIELDS:
        assert getattr(res_py, f) == getattr(res_c, f), f
    np.testing.assert_array_equal(res_py.induced_by, res_c.induced_by)
    np.testing.assert_array_equal(res_py.kills_by, res_c.kills_by)
    np.testing.assert_allclose(res_py.mean_prices, res_c.mean_prices, rtol=1e-9)
    np.testing.assert_allclose(res_py.mean_profits, res_c.mean_profits, rtol=1e-9, atol=1e-12)
    assert np.isclose(res_py.mean_consumer_surplus, res_c.mean_consumer_surplus, rtol=1e-9)


@pytest.mark.parametrize("rule,param", [
    (RuleType.MEAN_RATIO, 0.10), (RuleType.MEDIAN_RATIO, 0.10),
    (RuleType.PRICE_CAP, 0.60), (RuleType.Z_SCORE, 1.0), (RuleType.EMA_MEAN_RATIO, 0.10),
])
def test_scripted_attacker_matches(rule, param):
    cfg = _deterministic_cfg(rule=rule, param=param, policies=["attacker", "nash", "nash"])
    py, c = _pair(cfg)
    _assert_same(py.run(), c.run())


def test_escrow_matches():
    cfg = _deterministic_cfg(policies=["attacker", "nash", "nash"], penalty="escrow")
    py, c = _pair(cfg)
    res_py, res_c = py.run(), c.run()
    _assert_same(res_py, res_c)
    assert res_c.total_sanctions > 0


@pytest.mark.parametrize("traits", [
    None,
    [AgentTraits(alpha=1.5), AgentTraits(lambda_altruism=0.5), AgentTraits(mu_ethics=1.0)],
    [AgentTraits(alpha=1.0, mu_ethics=0.5)] * 3,
])
def test_learners_match_including_q_tables(traits):
    # delta = 0.5 gives a non-degenerate deterministic run (bans, kills, episode resets)
    cfg = _deterministic_cfg(traits=traits, param=0.5)
    py, c = _pair(cfg)
    res_py, res_c = py.run(), c.run()
    _assert_same(res_py, res_c)
    assert res_c.total_fresh_bans > 0 and res_c.total_bankruptcies > 0
    for a_py, a_c in zip(py.agents, c.agents):
        np.testing.assert_allclose(a_py.Q, a_c.Q, rtol=0, atol=1e-5)
        np.testing.assert_array_equal(a_py.state_visit_counts, a_c.state_visit_counts)


def test_learners_match_n4_and_repeated_runs():
    cfg = _deterministic_cfg(n=4, steps=1500, param=0.5)
    py, c = _pair(cfg)
    for _ in range(2):  # second call continues from persisted Q-tables
        res_py, res_c = py.run(), c.run()
        _assert_same(res_py, res_c)
    assert res_c.induced_bankruptcies > 0
    for a_py, a_c in zip(py.agents, c.agents):
        np.testing.assert_allclose(a_py.Q, a_c.Q, rtol=0, atol=1e-5)


def test_native_learner_discovers_framing():
    """Stochastic end-to-end check on the native backend: one learner vs honest rivals."""
    cfg = SimulationConfig(
        market=framing_market(num_firms=3, lethal=True, threshold_delta=0.10),
        rule_param=0.10, total_steps=200_000, warmup_steps=0, episode_length=500,
        eval_window=20_000, agent_type="learner_vs_honest", seed=1001, backend="native",
    )
    tr = MultiAgentTrainer(cfg)
    tr.run()
    tr.cfg.freeze_learners, tr.cfg.total_steps, tr.cfg.eval_window = True, 20_000, 20_000
    g = tr.run()
    assert g.kills_by[0] / g.episodes_in_eval > 0.5
    assert g.mean_profits[0] > tr.nash_profits[0]


@pytest.mark.parametrize("n_min", [2, 3])
def test_n_min_and_joint_attribution_match(n_min):
    cfg = _deterministic_cfg(param=0.5)
    cfg.market.min_active_for_audit = n_min
    py, c = _pair(cfg)
    res_py, res_c = py.run(), c.run()
    _assert_same(res_py, res_c)
    assert res_py.joint_induced_bans == res_c.joint_induced_bans
    assert res_py.joint_induced_sanctions == res_c.joint_induced_sanctions
    assert res_c.joint_induced_bans >= res_c.induced_bans
