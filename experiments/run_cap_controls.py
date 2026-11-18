"""All-learner price-cap controls (replaces the uncommitted ad-hoc driver used for
results/athena/controls/).

Three conditions, all three sellers learning, seeds 6000-6009, 10M training steps:
  cap p=0.60             N_min = 2
  no regulation (cap=10) N_min = 2   (cap above the whole price grid)
  cap p=0.60 Nmin=1      N_min = 1

Usage: python experiments/run_cap_controls.py [--steps 10000000]
Output: results/controls_rerun/cap_controls.csv
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.common import Job, run_jobs  # noqa: E402
from src.engine.rules import RuleType  # noqa: E402

CONDITIONS = [("cap p=0.60", 0.60, 2), ("no regulation (cap=10)", 10.0, 2), ("cap p=0.60 Nmin=1", 0.60, 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=10_000_000)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    jobs = [Job(tag=tag, seed=6000 + s, rule=int(RuleType.PRICE_CAP), rule_param=cap, n_min=n_min,
                train_steps=args.steps)
            for tag, cap, n_min in CONDITIONS for s in range(10)]
    rows = run_jobs(jobs, args.workers)
    out = ROOT / "results" / "controls_rerun"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "cap_controls.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
