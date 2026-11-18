"""Training-length check for the all-learner market (reviewer feedback item 1).

Reruns the robustness study's delta=0.10 mean and median conditions (same seeds 6000-6009,
same market) with 10x the training budget, so outcomes can be compared seed by seed
with the 10M-step results in results/Krishs-MacBook-Air/all/study_robust.csv.

Usage: python experiments/check_training_length.py [--steps 100000000]
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.common import Job, run_jobs  # noqa: E402
from src.engine.rules import RuleType  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100_000_000)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    jobs = [Job(tag=f"delta=0.1 {name} {args.steps // 1_000_000}M", seed=6000 + s, rule=int(rule),
                rule_param=0.10, train_steps=args.steps)
            for name, rule in (("mean", RuleType.MEAN_RATIO), ("median", RuleType.MEDIAN_RATIO))
            for s in range(10)]
    rows = run_jobs(jobs, args.workers)
    out = ROOT / "results" / "training_length"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"all_learner_{args.steps // 1_000_000}M.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
