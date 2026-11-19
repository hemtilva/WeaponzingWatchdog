"""Sensitivity of the mean-vs-median comparison to market and learning parameters.

Each condition changes one thing from the baseline market (N = 3, M = 11, a0 = 0,
sigma = 0.25, reserves capped at K0 = 0.15, eta = 0.10, gamma = 0.95, delta = 0.10,
N_min = 2) and is run under the mean and the median rule.

Axes:
  a0       outside option 1.0, 1.5 (baseline 0: almost every customer buys)
  sigma    price sensitivity 0.15, 0.25, 0.40 (at Phi = 0.04, K0 = 0.12; see below)
  N        3, 4, 5 sellers on a 7-price grid (Phi = 0.04, K0 = 0.12) (tabular Q tables grow as M^N 2^N 3;
           N = 6 would need ~2e7 states per learner even at M = 7, so it is out of reach)
  capital  reserves not capped (capital accumulates over the episode)
  eta      learning rate 0.05, 0.20
  gamma    discount 0.90, 0.99

Part "all": all sellers learn, seeds 6000-6009, 300M steps (the converged regime of the
training-length check). Part "single": one learner vs Nash rivals, seeds 3000-3009, 2M steps.
Each output row records the market diagnostics (viable, lethal, framing_feasible).

Usage: python experiments/run_sensitivity.py all|single [--steps N] [--workers W]
Output: results/sensitivity/<part>.csv
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.common import Job, run_jobs  # noqa: E402
from src.engine.rules import RuleType  # noqa: E402

RULES = (("mean", int(RuleType.MEAN_RATIO)), ("median", int(RuleType.MEDIAN_RATIO)))
LOW_PHI = dict(fixed_cost=0.04, initial_capital=0.12)
CONDITIONS = [
    ("a0=1.0", "a0", 1.0, dict(outside_option=1.0)),
    ("a0=1.5", "a0", 1.5, dict(outside_option=1.5)),
    # sigma and N axes use Phi = 0.04, K0 = 0.12 (same K0/Phi = 3, one ban still lethal), so
    # that every point on the axis is viable (at Phi = 0.05, sigma = 0.15 and N = 5 have
    # Nash profit = Phi exactly and honest firms could not cover their fixed cost)
    ("sigma=0.15", "sigma", 0.15, dict(price_sensitivity=0.15, **LOW_PHI)),
    ("sigma=0.25", "sigma", 0.25, dict(price_sensitivity=0.25, **LOW_PHI)),
    ("sigma=0.40", "sigma", 0.40, dict(price_sensitivity=0.40, **LOW_PHI)),
    ("N=3 M=7", "N", 3, dict(num_firms=3, price_levels=7, **LOW_PHI)),
    ("N=4 M=7", "N", 4, dict(num_firms=4, price_levels=7, **LOW_PHI)),
    ("N=5 M=7", "N", 5, dict(num_firms=5, price_levels=7, **LOW_PHI)),
    ("uncapped capital", "capital", 0, dict(capped_capital=False)),
    ("eta=0.05", "eta", 0.05, dict(learning_rate=0.05)),
    ("eta=0.20", "eta", 0.20, dict(learning_rate=0.20)),
    ("gamma=0.90", "gamma", 0.90, dict(gamma=0.90)),
    ("gamma=0.99", "gamma", 0.99, dict(gamma=0.99)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["all", "single"])
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    single = a.part == "single"
    steps = a.steps or (2_000_000 if single else 300_000_000)
    jobs = []
    for tag, axis, value, kw in CONDITIONS:
        n = kw.get("num_firms", 3)
        for rname, rule in RULES:
            jobs += [Job(tag=f"{tag} {rname}", seed=(3000 if single else 6000) + s, rule=rule, rule_param=0.10,
                         train_steps=steps, policies=(["q"] + ["nash"] * (n - 1)) if single else None,
                         extra=dict(axis=axis, value=value, rule_name=rname), **kw)
                     for s in range(10)]
    rows = run_jobs(jobs, a.workers)
    out = ROOT / "results" / "sensitivity"
    out.mkdir(parents=True, exist_ok=True)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    path = out / f"{a.part}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
