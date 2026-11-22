"""Step-by-step trace of two learners among honest sellers (the paper's two-learner figure).

N = 4: two tabular Q-learners (firms 0, 1) and two Nash-pricing sellers (firms 2, 3), mean
rule, delta = 0.10, lethal bans, 100M steps on the native engine. The traced seed (8003) is
representative of all ten probe seeds: in every seed about one seller survives per episode,
both honest sellers are bankrupted, and both learners take part in the kills.

Usage: python experiments/trace_two_learners.py
Output: results/two_trace/trace.csv (one frozen-policy episode)
        results/two_trace/probe.csv (10 seeds, mean and median rule, greedy outcomes)
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.common import Job, build, run_jobs  # noqa: E402
from src.engine.rules import RuleType  # noqa: E402

POLICIES = ["q", "q", "nash", "nash"]
STEPS = 100_000_000


def probe():
    jobs = [Job(tag=f"two learners N=4 {name}", seed=8000 + s, num_firms=4, rule=int(rule),
                policies=POLICIES, train_steps=STEPS, extra=dict(rule_name=name))
            for name, rule in (("mean", RuleType.MEAN_RATIO), ("median", RuleType.MEDIAN_RATIO))
            for s in range(10)]
    return run_jobs(jobs)


def trace(seed=8003, periods=40):
    job = Job(tag="trace", seed=seed, num_firms=4, rule=int(RuleType.MEAN_RATIO), rule_param=0.10,
              policies=POLICIES, train_steps=STEPS)
    tr = build(job)
    tr.run()
    tr.cfg.freeze_learners = True
    tr.cfg.backend = "python"
    tr.cfg.record_trace = True
    tr.cfg.total_steps = tr.cfg.trace_steps = tr.cfg.eval_window = periods
    res = tr.run()
    rows = []
    for t, r in enumerate(res.trace_data):
        row = {"t": t}
        for i in range(4):
            row[f"p{i}"] = round(float(r["prices"][i]), 4)
            row[f"K{i}"] = round(float(r["capitals"][i]), 4)
            row[f"ban{i}"] = int(r["ban_timers"][i])
        row["inducers"] = ";".join(f"{e['attacker']}->{e['victim']}" for e in r["exploit_events"])
        rows.append(row)
    return rows


def write(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


def main():
    out = ROOT / "results" / "two_trace"
    out.mkdir(parents=True, exist_ok=True)
    write(out / "probe.csv", probe())
    write(out / "trace.csv", trace())


if __name__ == "__main__":
    main()
