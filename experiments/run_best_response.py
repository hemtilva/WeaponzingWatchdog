"""Exact best response over the K0/Phi x T_ban grid (the phase-diagram grid of run_study h2).

One seller against two fixed Nash rivals, solved exactly by src/benchmarks/best_response.py.
Cells where a banned rival can survive the ban (K0 > T_ban * Phi) have very large state
spaces; any cell above --max-states is recorded as unsolved instead of being forced.

Usage: python experiments/run_best_response.py [--max-states 300000] [--cap-step 0.005]
Output: results/best_response/exact_grid.csv
"""
import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.benchmarks.best_response import BRConfig, solve_best_response  # noqa: E402
from src.engine.rules import RuleType  # noqa: E402

PHI = 0.05
RATIOS = [1, 2, 3, 4, 6, 8]
TBANS = [1, 2, 3, 4, 6]
RULES = [("mean", int(RuleType.MEAN_RATIO), 0.10), ("median", int(RuleType.MEDIAN_RATIO), 0.10),
         ("cap", int(RuleType.PRICE_CAP), 0.60)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-states", type=int, default=300_000)
    ap.add_argument("--cap-step", type=float, default=0.005)
    a = ap.parse_args()
    rows = []
    for rname, rule, par in RULES:
        for ratio in RATIOS:
            for tb in TBANS:
                lethal = ratio * PHI <= tb * PHI + 1e-12
                row = {"rule": rname, "K0_over_Phi": ratio, "T_ban": tb, "lethal_one_ban": lethal,
                       "cap_step": a.cap_step}
                t0 = time.time()
                try:
                    r = solve_best_response(BRConfig(rule=rule, rule_param=par, initial_capital=ratio * PHI,
                                                     fixed_cost=PHI, ban_duration=tb, cap_step=a.cap_step,
                                                     max_states=a.max_states))
                    row.update(solved=True, **r)
                except RuntimeError:
                    row.update(solved=False)
                row["seconds"] = round(time.time() - t0, 2)
                rows.append(row)
                print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()
                       if k in ("rule", "K0_over_Phi", "T_ban", "lethal_one_ban", "solved", "states", "gain",
                                "opt_rival_bankruptcies", "seconds")}, flush=True)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    out = ROOT / "results" / "best_response"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "exact_grid.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out / 'exact_grid.csv'}")


if __name__ == "__main__":
    main()
