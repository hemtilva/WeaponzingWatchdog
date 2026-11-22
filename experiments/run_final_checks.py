"""Final checks requested in review (protocol identical to the paper's 300M-step studies).

1. Ethics at stronger penalties: mu in {3, 10}, all sellers learning, mean rule.
   Same seeds (5000-5009) and settings as results/traits_300M, so rows are comparable
   with the mu <= 1 sweep there.
2. Omniscient-planner baseline from the proposal (Table 1): audit only sellers priced
   above Nash, never punish undercutting. With delta = 0.10 this is a fixed cap at
   (1 + delta) * p^N = 0.33. Run with all sellers learning (300M steps, seeds 5000-5009)
   and with a single learner against Nash rivals (2M steps, 20 seeds, as in Table 1).

  python experiments/run_final_checks.py            # writes results/final_checks/*.csv
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Job, run_jobs, ROOT  # noqa: E402

from src.engine.rules import RuleType  # noqa: E402

STEPS_ALL = 300_000_000
PLANNER_CAP = 0.33  # (1 + delta) * p^N with delta = 0.10, p^N = 0.30


def main():
    jobs = []
    for mu in (3.0, 10.0):
        jobs += [Job(tag=f"mu_ethics={mu}", seed=5000 + s, traits=[{"mu_ethics": mu}] * 3,
                     train_steps=STEPS_ALL, extra=dict(study="ethics", trait_value=mu))
                 for s in range(10)]
    jobs += [Job(tag="planner all-learners", seed=5000 + s, rule=int(RuleType.PRICE_CAP),
                 rule_param=PLANNER_CAP, train_steps=STEPS_ALL, extra=dict(study="planner_all"))
             for s in range(10)]
    jobs += [Job(tag="planner single", seed=3000 + s, rule=int(RuleType.PRICE_CAP),
                 rule_param=PLANNER_CAP, policies=["q", "nash", "nash"], train_steps=2_000_000,
                 extra=dict(study="planner_single"))
             for s in range(20)]
    # longest jobs first so the pool finishes evenly
    jobs.sort(key=lambda j: -j.train_steps)
    rows = run_jobs(jobs)

    out = ROOT / "results" / "final_checks"
    out.mkdir(parents=True, exist_ok=True)
    import csv
    for study in ("ethics", "planner_all", "planner_single"):
        rs = [r for r in rows if r["study"] == study]
        keys = list(dict.fromkeys(k for r in rs for k in r))
        with open(out / f"{study}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rs)
    for tag in dict.fromkeys(r["tag"] for r in rows):
        rs = [r for r in rows if r["tag"] == tag]
        def m(k):
            v = np.array([float(r[k]) for r in rs])
            return f"{v.mean():.3f} (se {v.std(ddof=1) / np.sqrt(len(v)):.3f})"
        print(f"{tag:<24} n={len(rs):2d} kills/ep {m('kills_per_ep')}  survivors {m('survivors')}  "
              f"price {m('mean_price')}  swl {m('swl')}  firm0 profit {m('firm0_profit')}")
    print(f"[SAVED] {out}")


if __name__ == "__main__":
    main()
