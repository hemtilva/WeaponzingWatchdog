"""But-for attribution of framing by trained learners (src/evaluation/replay_attribution.py).

Trains each condition on the native engine (same seeds and steps as experiments/run_defences.py
part "all", so engine statistics are comparable), freezes the learners, records the engine's
own frozen-play statistics, then replays frozen episodes with common random numbers and asks,
for every ban: would it (and the victim's bankruptcy) still have happened had a rival priced at
Nash over the previous w periods?

Usage: python experiments/run_replay_attribution.py [--steps 300000000] [--episodes 20]
Output: results/replay_attribution/all_learners.csv
"""
import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.common import Job, build  # noqa: E402
from src.engine.rules import RuleType  # noqa: E402
from src.evaluation.replay_attribution import ReplayEngine, attribute  # noqa: E402

CONDITIONS = [("mean", int(RuleType.MEAN_RATIO), 0.95), ("median", int(RuleType.MEDIAN_RATIO), 0.95),
              ("EMA beta=0.80", int(RuleType.EMA_MEAN_RATIO), 0.80), ("EMA beta=0.95", int(RuleType.EMA_MEAN_RATIO), 0.95)]


def run_one(args):
    tag, rule, beta, seed, steps, episodes = args
    job = Job(tag=tag, seed=seed, rule=rule, rule_param=0.10, ema_beta=beta, train_steps=steps)
    tr = build(job)
    tr.run()
    tr.cfg.freeze_learners = True
    tr.cfg.total_steps = job.eval_steps
    tr.cfg.eval_window = job.eval_steps
    g = tr.run()
    ep = max(1, g.episodes_in_eval)
    row = {"tag": tag, "seed": seed, "train_steps": steps,
           "engine_bans_per_ep": g.total_fresh_bans / ep, "engine_W": g.weaponization_rate,
           "engine_attributed_bankruptcies_per_ep": float(g.kills_by.sum()) / ep,
           "engine_bankruptcies_per_ep": g.total_bankruptcies / ep}
    row.update({f"replay_{k}": v for k, v in attribute(ReplayEngine(tr), episodes=episodes, seed=seed).items()})
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300_000_000)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--workers", type=int, default=max(1, min(9, (os.cpu_count() or 2) - 1)))
    a = ap.parse_args()
    tasks = [(t, r, b, 6000 + s, a.steps, a.episodes) for t, r, b in CONDITIONS for s in range(10)]
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(run_one, t) for t in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            print(f"  {i}/{len(tasks)} done", flush=True)
    rows.sort(key=lambda r: (r["tag"], r["seed"]))
    out = ROOT / "results" / "replay_attribution"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "all_learners.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out / 'all_learners.csv'}")


if __name__ == "__main__":
    main()
