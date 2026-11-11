"""Step 1: do independent learners weaponize the watchdog when every firm adapts?

All N firms are tabular Q-learners with identical (purely self-interested) traits.
Unlike the single-learner experiment, the victims also learn, so framing has to
emerge despite rivals that may undercut, retaliate, or collude.

Each run:
  1. train   : epsilon-greedy, exploration decays to eps_min at `--explore-frac` of training
  2. greedy  : all learners frozen and greedy; this is the measured outcome

Reported per configuration (mean +- std over seeds), greedy phase:
  kills/ep      rival bankruptcies caused by an induced ban, per episode (all firms)
  framed eps    fraction of seeds whose greedy play kills at least 0.5 rivals per episode
  W             share of bans induced by a rival's deviation
  Delta         collusion index (0 = Nash, 1 = joint monopoly)
  p_mean        average posted price (Nash shown for reference)
  survivors     firms still solvent at episode end, on average

Requires the native engine (python scripts/build_engine.py).
  python experiments/run_all_learners.py                 # full
  python experiments/run_all_learners.py --quick         # smoke test
"""

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.engine.rules import RuleType
from src.agents.q_learning import QLearningConfig
from src.agents.trainer import SimulationConfig, MultiAgentTrainer
from src.benchmarks.diagnostics import framing_market, market_diagnostics, format_diagnostics

# name, N, rule, rule_param, lethal
CONFIGS = [
    ("N=3 mean d=0.10 lethal", 3, RuleType.MEAN_RATIO, 0.10, True),
    ("N=3 mean d=0.10 non-lethal", 3, RuleType.MEAN_RATIO, 0.10, False),
    ("N=3 median d=0.10 lethal", 3, RuleType.MEDIAN_RATIO, 0.10, True),
    ("N=3 cap p=0.60 lethal", 3, RuleType.PRICE_CAP, 0.60, True),
    ("N=4 mean d=0.10 lethal", 4, RuleType.MEAN_RATIO, 0.10, True),
    ("N=4 median d=0.10 lethal", 4, RuleType.MEDIAN_RATIO, 0.10, True),
]


def make_market(n, rule, param, lethal):
    return framing_market(num_firms=n, lethal=lethal,
                          threshold_delta=param if rule != RuleType.PRICE_CAP else 0.10)


def run_one(job):
    name, n, rule, param, lethal, seed, train_steps, eval_steps, explore_frac, episode_length = job
    market = make_market(n, rule, param, lethal)
    eps_min = 0.01
    q_cfg = QLearningConfig(
        learning_rate=0.10, gamma=0.95, eps_start=1.0, eps_min=eps_min,
        eps_decay=float(eps_min ** (1.0 / max(1, int(explore_frac * train_steps)))),
    )
    cfg = SimulationConfig(
        market=market, q_learning=q_cfg, total_steps=train_steps, warmup_steps=0,
        episode_length=episode_length, rule_type=rule, rule_param=param,
        eval_window=min(train_steps, 200_000), agent_type="adaptive", seed=seed, backend="native",
    )
    tr = MultiAgentTrainer(cfg)
    t0 = time.time()
    train_res = tr.run()

    tr.cfg.freeze_learners = True
    tr.cfg.total_steps = eval_steps
    tr.cfg.eval_window = eval_steps
    g = tr.run()
    episodes = max(1, g.episodes_in_eval)
    kills = g.kills_by.astype(float) / episodes

    return {
        "config": name, "N": n, "rule": RuleType(rule).name, "rule_param": param, "lethal": lethal,
        "seed": seed, "train_steps": train_steps,
        "nash_price": float(tr.nash_prices[0]), "nash_profit": float(tr.nash_profits[0]),
        "mono_price": float(tr.mono_prices[0]),
        "greedy_kills_per_episode": float(kills.sum()),
        "greedy_max_firm_kills_per_episode": float(kills.max()),
        "greedy_bankruptcies_per_episode": g.total_bankruptcies / episodes,
        "greedy_survivors": n - g.total_bankruptcies / episodes,
        "greedy_W": g.weaponization_rate,
        "greedy_bans_per_episode": g.total_fresh_bans / episodes,
        "greedy_collusion_index": g.collusion_index,
        "greedy_mean_price": float(np.mean(g.mean_prices)),
        "greedy_mean_profit": float(np.mean(g.mean_profits)),
        "greedy_swl": g.social_welfare_loss,
        "train_kills_per_episode": float(train_res.kills_by.sum()) / max(1, train_res.episodes_in_eval),
        "train_W": train_res.weaponization_rate,
        "coverage": train_res.state_coverage_ratio,
        "seconds": time.time() - t0,
    }


def summarize(rows):
    hdr = (f"{'config':<30}{'kills/ep':>13}{'framed':>8}{'W':>12}{'Delta':>13}"
           f"{'p_mean':>9}{'Nash':>6}{'survive':>9}{'coverage':>10}")
    print("\n" + "=" * len(hdr)); print(hdr); print("-" * len(hdr))
    out = []
    for name, n, *_ in CONFIGS:
        rs = [r for r in rows if r["config"] == name]
        if not rs:
            continue

        def ms(k):
            v = np.array([r[k] for r in rs], dtype=float)
            return v.mean(), v.std()

        k, ks = ms("greedy_kills_per_episode")
        w, ws = ms("greedy_W")
        d, ds = ms("greedy_collusion_index")
        p, _ = ms("greedy_mean_price")
        s, _ = ms("greedy_survivors")
        cov, _ = ms("coverage")
        framed = sum(r["greedy_kills_per_episode"] >= 0.5 for r in rs)
        print(f"{name:<30}{f'{k:.2f}+-{ks:.2f}':>13}{f'{framed}/{len(rs)}':>8}{f'{w:.2f}+-{ws:.2f}':>12}"
              f"{f'{d:+.2f}+-{ds:.2f}':>13}{p:>9.3f}{rs[0]['nash_price']:>6.2f}{s:>9.2f}{cov:>10.3f}")
        out.append(dict(config=name, kills=k, kills_sd=ks, framed=framed, n=len(rs), W=w, delta=d, delta_sd=ds))
    print("=" * len(hdr))
    return out


def plot(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [s["config"] for s in summary]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    ax1.bar(names, [s["kills"] for s in summary], yerr=[s["kills_sd"] for s in summary],
            capsize=4, color="#c0392b")
    ax1.set_ylabel("induced rival bankruptcies per episode (greedy)")
    ax1.set_title("All firms learning: framing kills")
    ax1.tick_params(axis="x", rotation=30)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.5)

    ax2.bar(names, [s["delta"] for s in summary], yerr=[s["delta_sd"] for s in summary],
            capsize=4, color="#8e44ad")
    ax2.axhline(0, color="gray", linewidth=1)
    ax2.set_ylabel("collusion index Delta (0 = Nash, 1 = monopoly)")
    ax2.set_title("All firms learning: pricing outcome")
    ax2.tick_params(axis="x", rotation=30)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--train-steps-n3", type=int, default=10_000_000)
    ap.add_argument("--train-steps-n4", type=int, default=50_000_000)
    ap.add_argument("--eval-steps", type=int, default=50_000)
    ap.add_argument("--explore-frac", type=float, default=0.6)
    ap.add_argument("--episode-length", type=int, default=500)
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--only", type=str, default=None, help="substring filter on config names")
    ap.add_argument("--quick", action="store_true", help="2 seeds, 300k/1M steps")
    args = ap.parse_args()
    if args.quick:
        args.seeds, args.train_steps_n3, args.train_steps_n4, args.eval_steps = 2, 300_000, 1_000_000, 10_000

    configs = [c for c in CONFIGS if args.only is None or args.only in c[0]]
    print("Market diagnostics")
    for name, n, rule, param, lethal in configs:
        print(f"[{name}]")
        print(format_diagnostics(market_diagnostics(make_market(n, rule, param, lethal), rule_type=rule, rule_param=param)))

    jobs = [
        (name, n, rule, param, lethal, 2000 + s,
         args.train_steps_n3 if n == 3 else args.train_steps_n4,
         args.eval_steps, args.explore_frac, args.episode_length)
        for name, n, rule, param, lethal in configs for s in range(args.seeds)
    ]
    print(f"\nRunning {len(jobs)} jobs on {args.workers} workers...")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(run_one, jobs))
    print(f"done in {time.time() - t0:.0f}s")

    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "plots").mkdir(exist_ok=True)
    suffix = "_quick" if args.quick else ""
    csv_path = ROOT / "data" / f"all_learners{suffix}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary = summarize(rows)
    plot_path = ROOT / "plots" / f"all_learners{suffix}.png"
    plot(summary, plot_path)
    print(f"\n[SAVED] {csv_path}\n[SAVED] {plot_path}")


if __name__ == "__main__":
    main()
