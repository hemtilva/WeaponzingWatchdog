"""Repeat the five single-learner conditions on new seeds for the final report.

This is an exploratory replication: the conditions were selected after inspecting
the existing study. Outputs are kept separate from the original study.
"""
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Job, run_jobs, ROOT
from src.engine.rules import RuleType


def main():
    conditions = [
        ("mean d=0.10 lethal", RuleType.MEAN_RATIO, 0.10, True),
        ("mean d=0.20 lethal", RuleType.MEAN_RATIO, 0.20, True),
        ("mean d=0.10 non-lethal", RuleType.MEAN_RATIO, 0.10, False),
        ("median d=0.10 lethal", RuleType.MEDIAN_RATIO, 0.10, True),
        ("cap p=0.60 lethal", RuleType.PRICE_CAP, 0.60, True),
    ]
    jobs = [Job(tag=tag, seed=seed, rule=int(rule), rule_param=param,
                lethal=lethal, policies=["q", "nash", "nash"],
                train_steps=2_000_000, eval_steps=50_000)
            for tag, rule, param, lethal in conditions for seed in range(9020, 9025)]
    rows = run_jobs(jobs, workers=4)
    dest = ROOT / "results" / "report_replication" / "single_learner.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} runs to {dest}")


if __name__ == "__main__":
    main()
