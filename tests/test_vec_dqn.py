"""Vectorised DQN (src/gpu/vec_dqn.py): reproduces the single-learner result on CPU."""
import pytest

torch = pytest.importorskip("torch")

from src.agents.dqn_single import make_market  # noqa: E402
from src.engine.rules import RuleType  # noqa: E402
from src.gpu.vec_dqn import VecDQN, VecDQNConfig, evaluate  # noqa: E402


def _train(rule, learners=(0,), steps=300_000):
    torch.set_num_threads(2)
    mk = make_market(3, int(rule), 0.10, fixed_cost=0.05, initial_capital=0.15)
    cfg = VecDQNConfig(env_steps=steps, num_envs=128, batch_size=256, buffer_size=100_000,
                       learning_starts=2_000, updates_per_iter=2, seed=1, device="cpu")
    agent = VecDQN(mk, list(learners), cfg)
    agent.train()
    return evaluate(mk, agent.cpu_policies(), eval_episodes=5, attribution_episodes=1, seed=1)


def test_single_learner_frames_under_mean_rule():
    r = _train(RuleType.MEAN_RATIO)
    assert r["fixed_rival_bankruptcies_per_ep"] == pytest.approx(2.0)
    assert r["learner0_profit"] > 1.2


def test_single_learner_prices_honestly_under_median_rule():
    r = _train(RuleType.MEDIAN_RATIO)
    assert r["fixed_rival_bankruptcies_per_ep"] == pytest.approx(0.0)
    assert r["learner0_profit"] == pytest.approx(0.1, abs=1e-3)


def test_training_resumes_in_chunks():
    mk = make_market(3, int(RuleType.MEAN_RATIO), 0.10, fixed_cost=0.05, initial_capital=0.15)
    agent = VecDQN(mk, [0, 1], VecDQNConfig(env_steps=20_000, num_envs=64, learning_starts=500, device="cpu"))
    agent.train(until=0.5)
    assert agent._it == agent.iters // 2
    agent.train(until=1.0)
    assert agent._it == agent.iters
