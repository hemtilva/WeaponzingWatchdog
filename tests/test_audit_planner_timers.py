"""Proposal items added late: z-score rule (Proposition 1 iii), omniscient planner rule,
top-score-first audits, the ban-timer observation ablation, and per-firm bankruptcies.

Native (src/csrc/sim.cpp) and Python reference backends must agree. Where audit draws are
random, agreement is checked on configurations whose aggregate counts do not depend on the
draw (everyone audited, or ties only between interchangeable honest rivals).
"""
import copy

import numpy as np
import pytest

from src.agents.q_learning import QLearningConfig, StateIndexer
from src.agents.trainer import MultiAgentTrainer, SimulationConfig
from src.benchmarks.diagnostics import framing_market
from src.engine.native_sim import is_native_sim_available
from src.engine.rules import RuleType, evaluate_rules, rule_scores, select_top_audit_set
from src.evaluation.replay_attribution import ReplayEngine

native = pytest.mark.skipif(not is_native_sim_available(), reason="native engine not built")

COUNTS = ("total_fresh_bans", "induced_bans", "total_sanctions", "induced_sanctions",
          "total_bankruptcies", "induced_bankruptcies", "episodes_in_eval", "eval_periods",
          "joint_induced_bans", "joint_induced_sanctions")


def _cfg(rule=RuleType.MEAN_RATIO, param=0.10, policies=None, budget=3, audit_mode="random",
         timers=False, steps=3000, n_min=2):
    market = framing_market(num_firms=3, lethal=True, threshold_delta=param if rule != RuleType.PRICE_CAP else 0.10)
    market.audit_budget = budget
    market.min_active_for_audit = n_min
    return SimulationConfig(
        market=market, q_learning=QLearningConfig(eps_start=0.0, eps_min=0.0, learning_rate=0.1, gamma=0.95),
        price_grid=np.linspace(0.0, 1.5, 11), total_steps=steps, warmup_steps=0, episode_length=300,
        eval_window=steps // 2, rule_type=rule, rule_param=param, firm_policies=policies or ["q"] * 3,
        audit_mode=audit_mode, observe_ban_timers=timers, seed=7,
    )


def _pair(cfg):
    out = []
    for backend in ("python", "native"):
        c = copy.deepcopy(cfg)
        c.backend = backend
        tr = MultiAgentTrainer(c)
        rng = np.random.default_rng(11)
        for a in tr.agents:  # distinct values: deterministic greedy play
            a.Q[:] = rng.random(a.Q.shape, dtype=np.float32)
        out.append(tr)
    return out


def _same(a, b, per_firm=True):
    for f in COUNTS:
        assert getattr(a, f) == getattr(b, f), f
    if per_firm:
        np.testing.assert_array_equal(a.kills_by, b.kills_by)
        np.testing.assert_array_equal(a.bankruptcies_of, b.bankruptcies_of)
        np.testing.assert_allclose(a.mean_profits, b.mean_profits, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(a.mean_consumer_surplus, b.mean_consumer_surplus, rtol=1e-9)


# ---------------------------------------------------------------- rules
def test_proposition1_zscore_value():
    """Proposition 1 (iii): k firms at L, N - k at H > L: each high firm has Z = sqrt(k/(N-k))."""
    for N in (3, 4, 5, 8):
        for k in range(1, N):
            p = np.array([0.3] * k + [0.9] * (N - k))
            z = rule_scores(p, np.ones(N), RuleType.Z_SCORE)
            assert np.allclose(z[k:], np.sqrt(k / (N - k)))


def test_zscore_independent_of_crash_depth():
    """Any undercut, however small, gives the high-priced rivals the same z-score."""
    for low in (0.0, 0.15, 0.3):
        z = rule_scores(np.array([low, 0.45, 0.45]), np.ones(3), RuleType.Z_SCORE)
        assert np.allclose(z[1:], np.sqrt(0.5))
    f = evaluate_rules(np.array([0.3, 0.45, 0.45]), np.ones(3), RuleType.Z_SCORE, 0.5, min_active=2)
    assert list(f) == [0, 1, 1]
    f = evaluate_rules(np.array([0.0, 0.45, 0.45]), np.ones(3), RuleType.Z_SCORE, 1.0, min_active=2)
    assert list(f) == [0, 0, 0]  # sqrt(1/2) < 1: one undercutter cannot frame
    f = evaluate_rules(np.array([0.3, 0.3, 0.45]), np.ones(3), RuleType.Z_SCORE, 1.0, min_active=2)
    assert list(f) == [0, 0, 1]  # sqrt(2) > 1: two undercutters can


def test_planner_flags_only_above_nash():
    nash = np.array([0.3, 0.3, 0.3])
    f = evaluate_rules(np.array([0.0, 0.3, 0.45]), np.ones(3), RuleType.PLANNER, 0.0, min_active=2, nash_prices=nash)
    assert list(f) == [0, 0, 1]
    # A rival's price never changes another firm's flag: undercutting cannot frame
    for low in (0.0, 0.15, 0.3):
        f = evaluate_rules(np.array([low, 0.3, 0.3]), np.ones(3), RuleType.PLANNER, 0.0, min_active=2, nash_prices=nash)
        assert f.sum() == 0


def test_top_audit_picks_highest_score_and_breaks_ties_at_random():
    rng = np.random.default_rng(0)
    act = np.ones(3)
    assert list(select_top_audit_set(act, np.array([0.1, 0.9, 0.5]), 1, rng)) == [0, 1, 0]
    assert list(select_top_audit_set(act, np.array([0.1, 0.9, 0.5]), 2, rng)) == [0, 1, 1]
    picks = [int(np.argmax(select_top_audit_set(act, np.array([0.0, 1.0, 1.0]), 1, rng))) for _ in range(2000)]
    assert set(picks) == {1, 2} and 0.45 < np.mean(np.array(picks) == 1) < 0.55
    # inactive firms are never audited
    assert list(select_top_audit_set(np.array([1, 0, 1]), np.array([0.0, 9.0, 1.0]), 1, rng)) == [0, 0, 1]


# ---------------------------------------------------------------- state indexer
def test_timer_indexer_roundtrip_and_codes():
    ix = StateIndexer(np.linspace(0, 1.5, 11), fixed_cost=0.05, num_firms=3, timer_levels=4)
    assert ix.base == 6 and ix.num_states == 11 ** 3 * 6 ** 3 * 3
    st = ix.status(np.array([0, 3, 2]), np.array([0.1, 0.1, 0.0]))
    assert list(st) == [0, 3, 5]  # active, 3 periods left, bankrupt
    s = ix.encode([0.3, 0.6, 1.5], st, 0.3)
    assert ix.decode(s) == ((2, 4, 10), (0, 3, 5), 2)
    # default indexer unchanged: one bit per firm
    ix2 = StateIndexer(np.linspace(0, 1.5, 11), fixed_cost=0.05, num_firms=3)
    assert ix2.num_states == 31_944
    assert list(ix2.status(np.array([0, 3, 2]), np.array([0.1, 0.1, 0.0]))) == [0, 1, 1]


# ---------------------------------------------------------------- native == python
@native
@pytest.mark.parametrize("rule,param", [(RuleType.PLANNER, 0.0), (RuleType.Z_SCORE, 0.5), (RuleType.Z_SCORE, 1.0)])
def test_new_rules_native_matches_python(rule, param):
    py, c = _pair(_cfg(rule=rule, param=param))
    _same(py.run(), c.run())


@native
def test_timer_observation_native_matches_python():
    py, c = _pair(_cfg(timers=True))
    assert py.indexer.num_states == c.indexer.num_states == 11 ** 3 * 6 ** 3 * 3
    a, b = py.run(), c.run()
    _same(a, b)
    assert a.total_fresh_bans > 0
    np.testing.assert_allclose(py.agents[0].Q, c.agents[0].Q, rtol=1e-5, atol=1e-6)


@native
@pytest.mark.parametrize("rule,param", [(RuleType.MEAN_RATIO, 0.10), (RuleType.Z_SCORE, 0.5)])
def test_top_audit_native_matches_python(rule, param):
    """Crash attacker vs two identical honest rivals, one audit per period. Top-score-first
    always audits a flagged rival (the tie between the rivals cannot change any count)."""
    cfg = _cfg(rule=rule, param=param, policies=["attacker", "nash", "nash"], budget=1, audit_mode="top")
    py, c = _pair(cfg)
    a, b = py.run(), c.run()
    _same(a, b, per_firm=False)
    assert a.kills_by[0] == b.kills_by[0] == a.total_bankruptcies > 0
    # with random audits the attacker is often audited instead, so the kill comes later
    rnd = copy.deepcopy(cfg)
    rnd.audit_mode = "random"
    rnd.backend = "native"
    r = MultiAgentTrainer(rnd).run()
    assert r.total_sanctions <= b.total_sanctions


@native
def test_planner_has_no_induced_bans():
    cfg = _cfg(rule=RuleType.PLANNER, param=0.0, budget=1, audit_mode="top", steps=20_000)
    cfg.backend = "native"
    tr = MultiAgentTrainer(cfg)
    rng = np.random.default_rng(3)
    for a in tr.agents:
        a.Q[:] = rng.random(a.Q.shape, dtype=np.float32)
    r = tr.run()
    assert r.total_fresh_bans > 0
    assert r.induced_bans == r.joint_induced_bans == 0 and r.kills_by.sum() == 0


@native
def test_bankruptcies_of_sums_to_total():
    py, c = _pair(_cfg())
    a, b = py.run(), c.run()
    assert a.bankruptcies_of.sum() == a.total_bankruptcies
    np.testing.assert_array_equal(a.bankruptcies_of, b.bankruptcies_of)


# ---------------------------------------------------------------- replay engine
def test_replay_top_audit_frames_every_period():
    """Replay engine, top-score-first: the crash attacker gets a rival banned at t = 0 and a
    lethal ban bankrupts it; with the planner nobody is ever sanctioned."""
    cfg = _cfg(policies=["attacker", "nash", "nash"], budget=1, audit_mode="top")
    eng = ReplayEngine(MultiAgentTrainer(cfg))
    st = eng.initial_state()
    ev = eng.step(st, 0.99)
    assert ev["sanction"][1:].sum() == 1 and not ev["sanction"][0]
    events, _ = eng.run_episode(np.random.default_rng(0).random(60))
    assert sum(e["bankrupt"][1:].sum() for e in events) >= 1

    cfg = _cfg(rule=RuleType.PLANNER, param=0.0, policies=["attacker", "nash", "nash"], budget=1, audit_mode="top")
    eng = ReplayEngine(MultiAgentTrainer(cfg))
    events, _ = eng.run_episode(np.random.default_rng(0).random(60))
    assert sum(e["sanction"].sum() for e in events) == 0


def test_replay_timer_observation_policy_runs():
    cfg = _cfg(timers=True)
    tr = MultiAgentTrainer(cfg)
    rng = np.random.default_rng(1)
    for a in tr.agents:
        a.Q[:] = rng.random(a.Q.shape, dtype=np.float32)
    events, _ = ReplayEngine(tr).run_episode(rng.random(100))
    assert len(events) > 0
