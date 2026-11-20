"""Step-by-step trace of the framing attack by a single learner (the paper's attack figure).

Trains one Q-learner against two Nash rivals with the main-study protocol (seed 3000, mean
rule, delta = 0.10, lethal bans, 2M steps on the native engine), freezes it, and records the
first periods of one frozen-policy episode on the Python reference backend.

Usage: python experiments/trace_single_learner.py
Output: results/single_trace/trace.csv
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.common import Job, build  # noqa: E402
from src.engine.rules import RuleType  # noqa: E402


def main(seed=3000, periods=40):
    job = Job(tag="trace", seed=seed, rule=int(RuleType.MEAN_RATIO), rule_param=0.10,
              policies=["q", "nash", "nash"], train_steps=2_000_000)
    tr = build(job)
    tr.run()
    tr.cfg.freeze_learners = True
    tr.cfg.backend = "python"
    tr.cfg.record_trace = True
    tr.cfg.total_steps = tr.cfg.trace_steps = tr.cfg.eval_window = periods
    res = tr.run()
    out = ROOT / "results" / "single_trace"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for t, r in enumerate(res.trace_data):
        row = {"t": t}
        for i in range(3):
            row[f"p{i}"] = round(float(r["prices"][i]), 4)
            row[f"K{i}"] = round(float(r["capitals"][i]), 4)
            row[f"ban{i}"] = int(r["ban_timers"][i])
        rows.append(row)
    with open(out / "trace.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out / 'trace.csv'}")


if __name__ == "__main__":
    main()
