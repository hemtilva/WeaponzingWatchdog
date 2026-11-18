"""Trace the all-learner price-cap control (cap 0.60) to see how firms go bankrupt.

For each seed: train with the control protocol (native backend, 10M steps), freeze the
learners, then replay frozen greedy play on the Python backend with a per-step trace.
Writes one row per step to results/cap_trace/trace.csv and a summary per seed/episode.

Usage: python experiments/trace_cap_control.py [--n-min 2] [--seeds 6000 6001 ...]
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.common import Job, build  # noqa: E402
from src.engine.rules import RuleType  # noqa: E402

CAP = 0.60


def trace_seed(seed: int, n_min: int, trace_steps: int):
    job = Job(tag=f"cap p=0.60 Nmin={n_min}", seed=seed, rule=int(RuleType.PRICE_CAP),
              rule_param=CAP, n_min=n_min)
    tr = build(job)
    tr.run()  # training, identical protocol to the saved controls
    tr.cfg.freeze_learners = True
    tr.cfg.backend = "python"
    tr.cfg.record_trace = True
    tr.cfg.total_steps = trace_steps
    tr.cfg.trace_steps = trace_steps
    tr.cfg.eval_window = trace_steps
    res = tr.run()
    rows = []
    for t, r in enumerate(res.trace_data):
        row = {"seed": seed, "n_min": n_min, "t": t, "episode": t // job.episode_length,
               "ep_step": t % job.episode_length}
        for i in range(job.num_firms):
            row[f"p{i}"] = round(float(r["prices"][i]), 4)
            row[f"K{i}"] = round(float(r["capitals"][i]), 4)
            row[f"ban{i}"] = int(r["ban_timers"][i])
        rows.append(row)
    return rows, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-min", type=int, default=2)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(6000, 6010)))
    ap.add_argument("--steps", type=int, default=1000, help="frozen steps to trace (2 episodes)")
    args = ap.parse_args()

    out = ROOT / "results" / "cap_trace"
    out.mkdir(parents=True, exist_ok=True)
    all_rows, summary = [], []
    for seed in args.seeds:
        rows, res = trace_seed(seed, args.n_min, args.steps)
        all_rows += rows
        for ep in sorted({r["episode"] for r in rows}):
            er = [r for r in rows if r["episode"] == ep]
            n = sum(1 for k in er[0] if k.startswith("p"))
            deaths = []
            for i in range(n):
                dead = next((r for r in er if r[f"K{i}"] <= 0), None)
                if dead is not None:
                    # price the firm set on the step its ban began (last step with ban timer 0 before ban)
                    banned = next((r for r in er if r[f"ban{i}"] > 0), None)
                    prev = er[max(0, banned["ep_step"] - 1)] if banned else None
                    deaths.append((i, dead["ep_step"], banned["ep_step"] if banned else None,
                                   prev[f"p{i}"] if prev else None))
            first_steps = [r for r in er if r["ep_step"] < 5]
            last = er[-1]
            summary.append({
                "seed": seed, "n_min": args.n_min, "episode": ep,
                "bankruptcies": len(deaths),
                "deaths(firm,step_dead,step_banned,price_before_ban)": ";".join(map(str, deaths)),
                "prices_first_5_steps": "|".join(",".join(str(r[f"p{i}"]) for i in range(n))
                                                 for r in first_steps),
                "survivor_prices_last_step": ",".join(str(last[f"p{i}"]) for i in range(n)),
                "fresh_bans": res.total_fresh_bans, "attributed_bans": res.induced_bans,
            })
        print(summary[-1], flush=True)

    suffix = f"nmin{args.n_min}"
    with open(out / f"trace_{suffix}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        w.writeheader()
        w.writerows(all_rows)
    with open(out / f"summary_{suffix}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    print(f"wrote {out}/trace_{suffix}.csv and summary_{suffix}.csv")


if __name__ == "__main__":
    main()
