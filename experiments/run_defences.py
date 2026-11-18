"""Defences against regulator framing, tested against Q-learners.

Two defences, both implemented in the native engine:
  EMA reference : flags are judged against r_t = beta * r_{t-1} + (1 - beta) * mean(p_t),
                  so a one-period undercut moves the reference only (1 - beta) as far.
  Escrow        : a flagged, audited seller is not suspended; its profit above the Nash
                  level is diverted for that period. No bans, hence no ban-induced exits.

Part "single": one learner vs two fixed Nash rivals (seeds 3000-3019, 2M steps), the
               setting where the mean rule is exploited in every seed.
Part "all"   : all three sellers learning (seeds 6000-6009, 300M steps).

Usage: python experiments/run_defences.py single|all [--steps N]
Output: results/defences/<part>.csv
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.common import Job, run_jobs  # noqa: E402
from src.engine.rules import RuleType  # noqa: E402

MEAN, MEDIAN, EMA = int(RuleType.MEAN_RATIO), int(RuleType.MEDIAN_RATIO), int(RuleType.EMA_MEAN_RATIO)

# (tag, rule, ema_beta, penalty_mode)
SINGLE = [
    ("mean", MEAN, 0.95, "shutdown"),
    ("EMA beta=0.80", EMA, 0.80, "shutdown"),
    ("EMA beta=0.95", EMA, 0.95, "shutdown"),
    ("EMA beta=0.99", EMA, 0.99, "shutdown"),
    ("mean + escrow", MEAN, 0.95, "escrow"),
]
ALL = [
    ("mean", MEAN, 0.95, "shutdown"),
    ("median", MEDIAN, 0.95, "shutdown"),
    ("EMA beta=0.80", EMA, 0.80, "shutdown"),
    ("EMA beta=0.95", EMA, 0.95, "shutdown"),
    ("mean + escrow", MEAN, 0.95, "escrow"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["single", "all"])
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    if a.part == "single":
        steps = a.steps or 2_000_000
        jobs = [Job(tag=t, seed=3000 + s, rule=r, rule_param=0.10, ema_beta=b, penalty_mode=pm,
                    policies=["q", "nash", "nash"], train_steps=steps)
                for t, r, b, pm in SINGLE for s in range(20)]
    else:
        steps = a.steps or 300_000_000
        jobs = [Job(tag=t, seed=6000 + s, rule=r, rule_param=0.10, ema_beta=b, penalty_mode=pm,
                    train_steps=steps)
                for t, r, b, pm in ALL for s in range(10)]
    rows = run_jobs(jobs, a.workers)
    out = ROOT / "results" / "defences"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{a.part}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
