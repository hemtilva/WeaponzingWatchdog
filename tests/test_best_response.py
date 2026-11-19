"""Tests for the exact single-seller best response (src/benchmarks/best_response.py)."""
from collections import deque

import numpy as np
import pytest

from src.benchmarks.best_response import BRConfig, BestResponseMDP, solve_best_response
from src.engine.rules import RuleType

MEAN, MEDIAN, CAP = int(RuleType.MEAN_RATIO), int(RuleType.MEDIAN_RATIO), int(RuleType.PRICE_CAP)


def test_honest_value_is_nash_perpetuity():
    r = solve_best_response(BRConfig(rule=MEDIAN))
    mdp = BestResponseMDP(BRConfig(rule=MEDIAN))
    assert r["V_honest"] == pytest.approx(mdp.nash_profits[0] / (1 - 0.95), abs=1e-6)


@pytest.mark.parametrize("rule,param", [(MEAN, 0.10), (MEDIAN, 0.10), (CAP, 0.60)])
def test_gain_is_never_negative(rule, param):
    r = solve_best_response(BRConfig(rule=rule, rule_param=param))
    assert r["gain"] >= -1e-9


def test_mean_rule_best_response_frames_both_rivals():
    r = solve_best_response(BRConfig(rule=MEAN, rule_param=0.10))
    assert r["gain"] > 1.0
    assert r["first_price"] < r["nash_price"]
    assert r["opt_rival_bankruptcies"] == pytest.approx(2.0, abs=1e-6)
    assert r["opt_own_bankruptcy"] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("rule,param", [(MEDIAN, 0.10), (CAP, 0.60)])
def test_median_and_cap_best_response_is_nash(rule, param):
    r = solve_best_response(BRConfig(rule=rule, rule_param=param))
    assert r["gain"] == pytest.approx(0.0, abs=1e-9)
    assert r["opt_rival_bankruptcies"] == pytest.approx(0.0, abs=1e-9)


def test_capital_grid_does_not_change_the_answer():
    a = solve_best_response(BRConfig(rule=MEAN, cap_step=0.001))
    b = solve_best_response(BRConfig(rule=MEAN, cap_step=0.005))
    for k in ("gain", "first_price", "opt_rival_bankruptcies", "opt_profit_per_period"):
        assert a[k] == pytest.approx(b[k], abs=1e-6)


def test_fast_build_matches_direct_transitions():
    """Value of the start state from the fast build equals value iteration on a graph
    built directly from transitions()."""
    cfg = BRConfig(rule=MEAN, initial_capital=0.05, ban_duration=1, cap_step=0.005)
    fast = BestResponseMDP(cfg).build()
    v_fast = fast.solve()[0]

    ref = BestResponseMDP(cfg)
    s0 = ref.initial_state()
    idx, states, edges = {s0: 0}, [s0], {}
    q = deque([s0])
    while q:
        s = q.popleft()
        for a in ref._actions(s):
            outs = []
            for prob, ns, r, _ in ref.transitions(s, a):
                if ns not in idx:
                    idx[ns] = len(states)
                    states.append(ns)
                    q.append(ns)
                outs.append((prob, idx[ns], r))
            edges.setdefault(idx[s], []).append(outs)
    V = np.zeros(len(states))
    for _ in range(5000):
        Vn = np.zeros_like(V)
        for s, acts in edges.items():
            Vn[s] = max(sum(p * (r + 0.95 * V[n]) for p, n, r in outs) for outs in acts)
        if np.max(np.abs(Vn - V)) < 1e-11:
            break
        V = Vn
    assert len(states) == len(fast.states)
    assert v_fast == pytest.approx(V[0], abs=1e-8)
