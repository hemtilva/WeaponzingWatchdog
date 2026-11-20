"""Two independent DQN sellers plus Nash rivals (src/agents/dqn_multi.py).

Sellers 0 and 1 are DQN learners; the other N - 2 sellers price at Nash. Default N = 3; the
baseline market (delta = 0.10, lethal bans), mean and median rules. Use --seeds 2 for a pilot.

Usage: python experiments/run_dqn_multi.py [--seeds 10] [--steps 2000000] [--num-firms 3] [--tag pilot]
Output: results/dqn_multi/n<N>_<tag>.csv
"""
import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.engine.rules import RuleType  # noqa: E402

RULES = [("mean", int(RuleType.MEAN_RATIO), 0.10), ("median", int(RuleType.MEDIAN_RATIO), 0.10)]


def run_one(args):
    import torch
    torch.set_num_threads(1)
    from src.agents.dqn_multi import DQNMultiSellers
    from src.agents.dqn_single import DQNSingleConfig, make_market
    rname, rule, par, seed, steps, n = args
    # Same viable markets as the single-DQN study: Phi = 0.05 (N = 3), 0.04 (N = 4-5), 0.02 (N > 5),
    # with K0 = 3 Phi so that one ban is lethal.
    phi = 0.05 if n <= 3 else (0.04 if n <= 5 else 0.02)
    mk = make_market(n, rule, par, fixed_cost=phi, initial_capital=3 * phi)
    sellers = DQNMultiSellers(mk, learners=[0, 1], cfg=DQNSingleConfig(train_steps=steps, seed=seed))
    sellers.train()
    row = {"rule": rname, "seed": seed, "num_firms": n, "learners": 2, "train_steps": steps,
           "nash_price": float(mk.nash_prices[0]), "nash_profit": float(mk.nash_profits[0])}
    row.update(sellers.evaluate(attribution_episodes=20 if n <= 5 else 5))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--steps", type=int, default=2_000_000)
    ap.add_argument("--num-firms", type=int, default=3)
    ap.add_argument("--tag", default="full")
    ap.add_argument("--workers", type=int, default=max(1, min(9, (os.cpu_count() or 2) - 1)))
    a = ap.parse_args()
    tasks = [(r, rule, par, 3000 + s, a.steps, a.num_firms) for r, rule, par in RULES for s in range(a.seeds)]
    rows = []
    with ProcessPoolExecutor(min(a.workers, len(tasks))) as ex:
        futs = [ex.submit(run_one, t) for t in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            print(f"  {i}/{len(tasks)} done", flush=True)
    rows.sort(key=lambda r: (r["rule"], r["seed"]))
    out = ROOT / "results" / "dqn_multi"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"n{a.num_firms}_{a.tag}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
