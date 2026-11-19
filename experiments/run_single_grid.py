"""Single learner over the K0/Phi x T_ban grid, for comparison with the exact best response.

Same cells and rules as experiments/run_best_response.py: one Q-learner against two fixed
Nash rivals, seeds 3000-3009.

Usage: python experiments/run_single_grid.py [--steps 2000000]
Output: results/best_response/learned_grid_<steps>.csv
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.common import Job, run_jobs  # noqa: E402
from experiments.run_best_response import PHI, RATIOS, TBANS, RULES  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2_000_000)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    jobs = [Job(tag=f"{rname} K0/Phi={r} T={tb}", seed=3000 + s, rule=rule, rule_param=par,
                initial_capital=r * PHI, fixed_cost=PHI, ban_duration=tb, policies=["q", "nash", "nash"],
                train_steps=a.steps, extra=dict(rule_name=rname, K0_over_Phi=r))
            for rname, rule, par in RULES for r in RATIOS for tb in TBANS for s in range(a.seeds)]
    rows = run_jobs(jobs, a.workers)
    out = ROOT / "results" / "best_response"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"learned_grid_{a.steps // 1_000_000}M.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
