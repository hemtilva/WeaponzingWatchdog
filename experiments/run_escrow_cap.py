"""Price cap enforced with escrow instead of suspension.

A flagged, audited seller (price above the 0.60 cap) keeps selling but loses its profit
above the Nash level for that period. Compared against the cap with suspensions
(results/controls_rerun/cap_controls_300M.csv) and the mean rule with escrow
(results/defences/all.csv): does a cap keep prices down without the lethal bans that
make framing possible under relative rules?

Part "all": all sellers learn, seeds 6000-6009, 300M steps. Part "single": one learner vs
Nash rivals, seeds 3000-3019, 2M steps.

Usage: python experiments/run_escrow_cap.py all|single
Output: results/defences/escrow_cap_<part>.csv
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
    ap.add_argument("part", choices=["all", "single"])
    a = ap.parse_args()
    if a.part == "all":
        jobs = [Job(tag="cap 0.60 + escrow", seed=6000 + s, rule=int(RuleType.PRICE_CAP), rule_param=0.60,
                    penalty_mode="escrow", train_steps=300_000_000) for s in range(10)]
    else:
        jobs = [Job(tag="cap 0.60 + escrow", seed=3000 + s, rule=int(RuleType.PRICE_CAP), rule_param=0.60,
                    penalty_mode="escrow", policies=["q", "nash", "nash"], train_steps=2_000_000)
                for s in range(20)]
    rows = run_jobs(jobs)
    out = ROOT / "results" / "defences"
    with open(out / f"escrow_cap_{a.part}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out / f'escrow_cap_{a.part}.csv'}")


if __name__ == "__main__":
    main()
