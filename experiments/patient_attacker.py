"""Is framing still *possible* under each defence? A scripted 'patient' attacker.

Firm 0 holds a low price L while at least N_min firms are active and charges the top grid
price otherwise; firms 1-2 price at Nash (0.30). Unlike the crash-to-zero scripted attacker
in the tests, L = 0.15 earns more than the fixed cost, so the attacker can wait as long as an
EMA reference needs to drift. Runs on the Python reference backend (no learning involved).

Usage: python experiments/patient_attacker.py
Output: results/defences/patient_attacker.csv
"""
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.common import Job, build  # noqa: E402
from src.engine.rules import RuleType  # noqa: E402

MEAN, EMA = int(RuleType.MEAN_RATIO), int(RuleType.EMA_MEAN_RATIO)
CONDITIONS = [("mean", MEAN, 0.95, "shutdown"), ("EMA beta=0.80", EMA, 0.80, "shutdown"),
              ("EMA beta=0.95", EMA, 0.95, "shutdown"), ("EMA beta=0.99", EMA, 0.99, "shutdown"),
              ("mean + escrow", MEAN, 0.95, "escrow")]


def run(tag, rule, beta, pm, low, seed, steps=50_000):
    tr = build(Job(tag=tag, seed=seed, rule=rule, rule_param=0.10, ema_beta=beta, penalty_mode=pm,
                   policies=["nash", "nash", "nash"], train_steps=steps))
    tr.cfg.backend = "python"
    tr.cfg.eval_window = steps
    lo = int(np.argmin(np.abs(tr.price_grid - low)))
    hi = len(tr.price_grid) - 1
    base = tr._select_action

    def select(i, state_idx, num_active):
        if i == 0:
            return lo if num_active >= tr.m_cfg.min_active_for_audit else hi
        return base(i, state_idx, num_active)

    tr._select_action = select
    r = tr.run()
    ep = max(1, r.episodes_in_eval)
    return {"tag": tag, "low_price": float(tr.price_grid[lo]), "seed": seed,
            "bans_per_ep": r.total_fresh_bans / ep, "kills_per_ep": float(r.kills_by.sum()) / ep,
            "bankruptcies_per_ep": r.total_bankruptcies / ep,
            "attacker_profit": float(r.mean_profits[0]), "rival_profit": float(np.mean(r.mean_profits[1:])),
            "swl": r.social_welfare_loss}


def main():
    rows = [run(t, r, b, pm, low, seed) for t, r, b, pm in CONDITIONS for low in (0.0, 0.15)
            for seed in range(5)]
    out = ROOT / "results" / "defences"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "patient_attacker.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    import pandas as pd
    d = pd.DataFrame(rows)
    print(d.groupby(["tag", "low_price"], sort=False)[["bans_per_ep", "kills_per_ep", "bankruptcies_per_ep", "attacker_profit",
                                                       "rival_profit", "swl"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
