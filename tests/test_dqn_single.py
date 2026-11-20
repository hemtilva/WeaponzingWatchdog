"""Tests for the single DQN seller (src/agents/dqn_single.py)."""
import pytest

torch = pytest.importorskip("torch")

from experiments.common import Job, build  # noqa: E402
from src.agents.dqn_single import DQNSingleConfig, DQNSingleSeller  # noqa: E402
from src.engine.rules import RuleType  # noqa: E402


def _agent(rule, param, steps=20_000, seed=1):
    torch.set_num_threads(1)
    tr = build(Job(tag="t", seed=seed, rule=int(rule), rule_param=param,
                   policies=["q", "nash", "nash"], train_steps=1000))
    agent = DQNSingleSeller(tr, DQNSingleConfig(train_steps=steps, eval_steps=2_000, seed=seed))
    agent.train()
    return agent.evaluate(attribution_episodes=2)


def test_observation_matches_the_tabular_information():
    tr = build(Job(tag="t", seed=1, policies=["q", "nash", "nash"], train_steps=1000))
    a = DQNSingleSeller(tr, DQNSingleConfig(train_steps=10, seed=1))
    o = a.obs(a.engine.initial_state())
    assert o.shape == (7,)                     # 3 prices, 3 inactive flags, own capital
    assert o[-1] == pytest.approx(1.0)          # starts at full capital


def test_dqn_learns_to_frame_under_the_mean_rule():
    r = _agent(RuleType.MEAN_RATIO, 0.10)
    assert r["rival_bankruptcies_per_ep"] == pytest.approx(2.0)
    assert r["replay_causal_bankruptcies_per_ep_wNone"] == pytest.approx(2.0)
    assert r["profit_per_period"] > 1.0


def test_dqn_does_not_frame_under_the_median_rule():
    r = _agent(RuleType.MEDIAN_RATIO, 0.10)
    assert r["rival_bankruptcies_per_ep"] == pytest.approx(0.0)
