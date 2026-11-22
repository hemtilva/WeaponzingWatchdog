"""Does a single Q-learner discover watchdog framing on its own?

Firm 0 is a tabular Q-learner; its rivals are honest and always post the Nash price.
This is a stationary single-agent problem, so if framing is the best response the
learner should find it, and if it is not (median rule, price cap, non-lethal bans)
it should not. Covers H1 (rule ranking) and H2 (lethal vs non-lethal bans).

Each run has two phases:
  1. train   : epsilon-greedy Q-learning for --train-steps periods
  2. greedy  : learning frozen, epsilon = 0, for --eval-steps periods

The greedy phase is the clean measurement: exploration noise cannot cause bans or
deaths there, so every induced bankruptcy is the learned policy's doing.

Usage:
  python experiments/run_framing_discovery.py               # full (5 configs x 5 seeds)
  python experiments/run_framing_discovery.py --quick       # smoke test
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
from src.agents.trainer import SimulationConfig, MultiAgentTrainer
from src.benchmarks.diagnostics import framing_market, market_diagnostics, format_diagnostics

# name, rule, rule_param, lethal
CONFIGS = [
    ("mean d=0.10 lethal", RuleType.MEAN_RATIO, 0.10, True),
    ("mean d=0.20 lethal", RuleType.MEAN_RATIO, 0.20, True),
    ("mean d=0.10 non-lethal", RuleType.MEAN_RATIO, 0.10, False),
    ("median d=0.10 lethal", RuleType.MEDIAN_RATIO, 0.10, True),
    ("cap p=0.60 lethal", RuleType.PRICE_CAP, 0.60, True),
]


def run_one(args):
    name, rule, param, lethal, seed, train_steps, eval_steps, episode_length = args
    market = framing_market(num_firms=3, lethal=lethal, threshold_delta=param if rule != RuleType.PRICE_CAP else 0.10)
    cfg = SimulationConfig(
        market=market,
        total_steps=train_steps,
        warmup_steps=0,
        episode_length=episode_length,
        rule_type=rule,
        rule_param=param,
        eval_window=min(train_steps, 40_000),
        agent_type="learner_vs_honest",
        seed=seed,
    )
    trainer = MultiAgentTrainer(cfg)
    t0 = time.time()
    train_res = trainer.run()

    # Greedy policy at the episode-start state
    s0 = trainer.indexer.encode(trainer.nash_prices, np.zeros(3, dtype=bool), market.initial_capital)
    start_price = float(trainer.price_grid[trainer.agents[0].greedy_action(s0)])

    trainer.cfg.freeze_learners = True
    trainer.cfg.total_steps = eval_steps
    trainer.cfg.eval_window = eval_steps
    g = trainer.run()
    episodes = max(1, g.episodes_in_eval)

    return {
        "config": name,
        "seed": seed,
        "rule": RuleType(rule).name,
        "rule_param": param,
        "lethal": lethal,
        "nash_price": float(trainer.nash_prices[0]),
        "nash_profit": float(trainer.nash_profits[0]),
        "start_greedy_price": start_price,
        "frames_at_start": bool(start_price < trainer.nash_prices[0]),
        # greedy (frozen) phase
        "greedy_learner_price": float(g.mean_prices[0]),
        "greedy_learner_profit": float(g.mean_profits[0]),
        "greedy_rival_profit": float(np.mean(g.mean_profits[1:])),
        "greedy_W": float(g.weaponization_rate),
        "greedy_bans": int(g.total_fresh_bans),
        "greedy_induced_bans": int(g.induced_bans),
        "greedy_kills_per_episode": float(g.kills_by[0]) / episodes,
        "greedy_other_bankruptcies_per_episode": float(
            g.total_bankruptcies - g.induced_bankruptcies
        ) / episodes,
        "greedy_episodes": episodes,
        # training tail (epsilon-greedy)
        "train_W": float(train_res.weaponization_rate),
        "train_kills_per_episode": float(train_res.kills_by[0]) / max(1, train_res.episodes_in_eval),
        "seconds": time.time() - t0,
    }


def summarize(rows):
    print("\n" + "=" * 112)
    print(f"{'config':<24}{'frames@start':>13}{'kills/ep':>14}{'W':>14}{'learner pi':>14}{'rival pi':>12}{'Nash pi':>9}{'other bankr/ep':>15}")
    print("-" * 112)
    out = []
    for name, *_ in CONFIGS:
        rs = [r for r in rows if r["config"] == name]
        if not rs:
            continue

        def ms(key):
            v = np.array([r[key] for r in rs], dtype=float)
            return v.mean(), v.std()

        k, ks = ms("greedy_kills_per_episode")
        w, ws = ms("greedy_W")
        lp, lps = ms("greedy_learner_profit")
        rp, _ = ms("greedy_rival_profit")
        d, _ = ms("greedy_other_bankruptcies_per_episode")
        frames = sum(r["frames_at_start"] for r in rs)
        print(f"{name:<24}{f'{frames}/{len(rs)}':>13}{f'{k:.2f}+-{ks:.2f}':>14}{f'{w:.2f}+-{ws:.2f}':>14}"
              f"{f'{lp:.3f}+-{lps:.3f}':>14}{rp:>12.3f}{rs[0]['nash_profit']:>9.3f}{d:>15.2f}")
        out.append((name, frames, len(rs), k, ks, w, lp, rp))
    print("=" * 112)
    return out


def plot(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [s[0] for s in summary]
    kills = [s[3] for s in summary]
    kills_sd = [s[4] for s in summary]
    lp = [s[6] for s in summary]
    nash_pi = 0.10

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.bar(names, kills, yerr=kills_sd, capsize=4, color="#c0392b")
    ax1.set_ylabel("rival bankruptcies caused per episode (greedy policy)")
    ax1.set_title("Learned framing: induced kills")
    ax1.tick_params(axis="x", rotation=25)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.5)

    ax2.bar(names, lp, color="#2980b9")
    ax2.axhline(nash_pi, color="gray", linestyle="--", label="honest Nash profit")
    ax2.set_ylabel("learner mean profit per period (greedy policy)")
    ax2.set_title("Does framing pay?")
    ax2.tick_params(axis="x", rotation=25)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--train-steps", type=int, default=200_000)
    ap.add_argument("--eval-steps", type=int, default=20_000)
    ap.add_argument("--episode-length", type=int, default=500)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--quick", action="store_true", help="2 seeds x 30k steps smoke test")
    args = ap.parse_args()
    if args.quick:
        args.seeds, args.train_steps, args.eval_steps = 2, 30_000, 5_000

    print("Market diagnostics")
    for name, rule, param, lethal in CONFIGS:
        m = framing_market(num_firms=3, lethal=lethal, threshold_delta=param if rule != RuleType.PRICE_CAP else 0.10)
        print(f"[{name}]")
        print(format_diagnostics(market_diagnostics(m, rule_type=rule, rule_param=param)))

    jobs = [
        (name, rule, param, lethal, 1000 + s, args.train_steps, args.eval_steps, args.episode_length)
        for name, rule, param, lethal in CONFIGS
        for s in range(args.seeds)
    ]
    print(f"\nRunning {len(jobs)} jobs on {args.workers} workers "
          f"({args.train_steps} train + {args.eval_steps} greedy steps each)...")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(run_one, jobs))
    print(f"done in {time.time() - t0:.0f}s")

    data_dir = ROOT / "data"
    plots_dir = ROOT / "plots"
    data_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)
    csv_path = data_dir / "framing_discovery.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = summarize(rows)
    plot_path = plots_dir / "framing_discovery.png"
    plot(summary, plot_path)
    print(f"\n[SAVED] {csv_path}\n[SAVED] {plot_path}")


if __name__ == "__main__":
    main()
