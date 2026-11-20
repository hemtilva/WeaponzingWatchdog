"""The vectorised market (src/gpu/vec_market.py) must reproduce ReplayEngine exactly."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.agents.dqn_single import make_market  # noqa: E402
from src.engine.rules import RuleType  # noqa: E402
from src.evaluation.replay_attribution import ReplayEngine  # noqa: E402
from src.gpu.vec_market import VecMarket  # noqa: E402

CASES = [(3, RuleType.MEAN_RATIO, 0.10, "shutdown"), (3, RuleType.MEDIAN_RATIO, 0.10, "shutdown"),
         (3, RuleType.PRICE_CAP, 0.60, "shutdown"), (3, RuleType.EMA_MEAN_RATIO, 0.10, "shutdown"),
         (3, RuleType.MEAN_RATIO, 0.10, "escrow"), (4, RuleType.MEDIAN_RATIO, 0.10, "shutdown"),
         (5, RuleType.MEAN_RATIO, 0.10, "shutdown"), (8, RuleType.MEAN_RATIO, 0.10, "shutdown")]


@pytest.mark.parametrize("n,rule,param,penalty", CASES)
def test_vectorised_market_matches_replay_engine(n, rule, param, penalty):
    phi = 0.05 if n <= 3 else (0.04 if n <= 5 else 0.02)
    mk = make_market(n, int(rule), param, fixed_cost=phi, initial_capital=3 * phi, penalty_mode=penalty,
                     ema_beta=0.8)
    B, T = 16, 300
    rng = np.random.default_rng(0)
    # actions biased toward low prices so that bans and bankruptcies actually happen
    acts = rng.integers(0, len(mk.price_grid), size=(T, B, n))
    acts = np.where(rng.random((T, B, n)) < 0.5, rng.integers(0, 3, size=(T, B, n)), acts)
    us = rng.random((T, B))

    vm = VecMarket(mk, B)
    vst = vm.reset()
    eng = ReplayEngine(mk)
    sts = [eng.initial_state() for _ in range(B)]
    n_events = 0
    for t in range(T):
        vev = vm.step(vst, torch.as_tensor(acts[t]), torch.as_tensor(us[t]))
        for b in range(B):
            if not (sts[b].capital > 0).any():
                continue
            ev = eng.step(sts[b], us[t, b], force={i: int(acts[t, b, i]) for i in range(n)})
            for key in ("sanction", "bankrupt", "active"):
                assert np.array_equal(vev[key][b].numpy(), ev[key]), (t, b, key)
            np.testing.assert_allclose(vev["profits"][b].numpy(), ev["profits"], rtol=0, atol=1e-12)
            np.testing.assert_allclose(vst.capital[b].numpy(), sts[b].capital, rtol=0, atol=1e-12)
            assert np.array_equal(vst.timers[b].numpy(), sts[b].timers)
            n_events += int(ev["sanction"].sum() + ev["bankrupt"].sum())
    assert n_events > 0                      # the test exercised bans and bankruptcies
