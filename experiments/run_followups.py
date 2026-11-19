"""Follow-ups to the sensitivity and exact-vs-learned studies.

1. Single learner with N = 5 (7 prices, Phi = 0.04, K0 = 0.12), mean and median, at 50M and
   300M steps: is the absence of framing at 2M steps a training-length effect?
2. Single learner in the one-period-ban column of the K0/Phi x T_ban grid (mean rule), at
   300M steps: does the learnability gap against the exact best response close?

Usage: python experiments/run_followups.py
Output: results/followups/n5_single.csv, results/followups/tban1_single_300M.csv
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.common import Job, run_jobs  # noqa: E402
from src.engine.rules import RuleType  # noqa: E402

MEAN, MEDIAN = int(RuleType.MEAN_RATIO), int(RuleType.MEDIAN_RATIO)


def save(rows, name):
    out = ROOT / "results" / "followups"
    out.mkdir(parents=True, exist_ok=True)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(out / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out / name}", flush=True)


def main():
    n5 = [Job(tag=f"N=5 M=7 {rn} {st // 1_000_000}M", seed=3000 + s, num_firms=5, price_levels=7, rule=r,
              rule_param=0.10, fixed_cost=0.04, initial_capital=0.12, policies=["q"] + ["nash"] * 4,
              train_steps=st, extra=dict(rule_name=rn))
          for st in (50_000_000, 300_000_000) for rn, r in (("mean", MEAN), ("median", MEDIAN)) for s in range(10)]
    save(run_jobs(n5), "n5_single.csv")
    t1 = [Job(tag=f"mean K0/Phi={k} T=1", seed=3000 + s, rule=MEAN, rule_param=0.10, initial_capital=k * 0.05,
              fixed_cost=0.05, ban_duration=1, policies=["q", "nash", "nash"], train_steps=300_000_000,
              extra=dict(rule_name="mean", K0_over_Phi=k))
          for k in (2, 3, 4, 6, 8) for s in range(10)]
    save(run_jobs(t1), "tban1_single_300M.csv")


if __name__ == "__main__":
    main()
