"""Single DQN seller against Nash rivals (src/agents/dqn_single.py).

Same market and conditions as the tabular single-learner study (N = 3, lethal bans,
delta = 0.10, cap 0.60), seeds 3000-3009, 2M training steps; frozen greedy evaluation
over 100 episodes plus but-for attribution over 20 replayed episodes.

Usage: python experiments/run_dqn_single.py [--steps 2000000] [--num-firms 3]
Output: results/dqn_single/n<N>.csv
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

RULES = [("mean", int(RuleType.MEAN_RATIO), 0.10), ("median", int(RuleType.MEDIAN_RATIO), 0.10),
         ("cap", int(RuleType.PRICE_CAP), 0.60)]


def run_one(args):
    import torch
    torch.set_num_threads(1)
    from experiments.common import Job, build
    from src.agents.dqn_single import DQNSingleConfig, DQNSingleSeller
    from src.agents.dqn_single import make_market
    rname, rule, par, seed, steps, n = args
    if n > 5:
        # Large markets: no tabular trainer (it would allocate M^N tables). Phi = 0.02 keeps
        # Nash profit above the fixed cost (0.0375 at N = 8, 0.030 at N = 10), K0 = 3 Phi.
        tr = make_market(n, rule, par, fixed_cost=0.02, initial_capital=0.06)
    else:
        low_phi = dict(fixed_cost=0.04, initial_capital=0.12, price_levels=7) if n >= 5 else {}
        tr = build(Job(tag=rname, seed=seed, num_firms=n, rule=rule, rule_param=par,
                       policies=["q"] + ["nash"] * (n - 1), train_steps=1000, **low_phi))
    agent = DQNSingleSeller(tr, DQNSingleConfig(train_steps=steps, seed=seed))
    agent.train()
    row = {"rule": rname, "seed": seed, "num_firms": n, "train_steps": steps,
           "nash_price": float(tr.nash_prices[0]), "nash_profit": float(tr.nash_profits[0])}
    row.update(agent.evaluate(attribution_episodes=20 if n <= 5 else 5))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2_000_000)
    ap.add_argument("--num-firms", type=int, default=3)
    ap.add_argument("--rules", default="mean,median,cap")
    ap.add_argument("--workers", type=int, default=max(1, min(9, (os.cpu_count() or 2) - 1)))
    a = ap.parse_args()
    keep = set(a.rules.split(","))
    tasks = [(r, rule, par, 3000 + s, a.steps, a.num_firms) for r, rule, par in RULES if r in keep for s in range(10)]
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(run_one, t) for t in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            print(f"  {i}/{len(tasks)} done", flush=True)
    rows.sort(key=lambda r: (r["rule"], r["seed"]))
    out = ROOT / "results" / "dqn_single"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"n{a.num_firms}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
