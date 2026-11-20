"""Tests for independent DQN sellers (src/agents/dqn_multi.py)."""
import pytest

torch = pytest.importorskip("torch")

from src.agents.dqn_multi import DQNMultiSellers  # noqa: E402
from src.agents.dqn_single import DQNSingleConfig, make_market  # noqa: E402
from src.engine.rules import RuleType  # noqa: E402


def _sellers(steps=3_000):
    torch.set_num_threads(1)
    mk = make_market(3, int(RuleType.MEAN_RATIO), 0.10, fixed_cost=0.05, initial_capital=0.15)
    return DQNMultiSellers(mk, learners=[0, 1], cfg=DQNSingleConfig(train_steps=steps, eval_steps=1_000,
                                                                    learning_starts=500, seed=1))


def test_each_learner_sees_its_own_capital():
    s = _sellers()
    st = s.engine.initial_state()
    st.capital[1] = 0.05
    assert s.obs(st, 0)[-1] == pytest.approx(1.0)
    assert s.obs(st, 1)[-1] == pytest.approx(0.05 / 0.15)


def test_learners_get_independent_networks():
    s = _sellers()
    assert s.L[0].q is not s.L[1].q
    assert s.engine.policies[2] is not s.engine.policies[0]      # seller 2 stays a Nash rival


def test_short_training_and_evaluation_run():
    s = _sellers()
    s.train()
    r = s.evaluate(attribution_episodes=1)
    assert 0.0 <= r["survivors_per_ep"] <= 3.0
    assert r["fixed_rival_bankruptcies_per_ep"] <= 1.0
    assert "learner0_profit" in r and "learner1_profit" in r
