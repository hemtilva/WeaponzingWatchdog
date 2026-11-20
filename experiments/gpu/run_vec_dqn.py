"""Vectorised DQN experiments (src/gpu/vec_dqn.py) - designed for a CUDA GPU, runs on CPU too.

Sellers 0..K-1 are independent DQN learners; the rest price at Nash. Each (rule, seed) run
is evaluated at several fractions of its training budget, so every result shows whether it
had converged. Results are appended to a CSV after each evaluation and finished
(rule, seed, checkpoint) rows are skipped on restart, so an interrupted run resumes.

Usage:
  python experiments/gpu/run_vec_dqn.py --num-firms 3 --learners 3 --rules mean,median \\
      --seeds 10 --env-steps 100000000 --tag allN3
Output: results/gpu/<tag>.csv, trained networks in results/gpu/checkpoints/
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.agents.dqn_single import make_market  # noqa: E402
from src.benchmarks.diagnostics import market_diagnostics  # noqa: E402
from src.engine.rules import RuleType  # noqa: E402
from src.gpu.vec_dqn import VecDQN, VecDQNConfig, evaluate, pick_device  # noqa: E402

# name -> (rule, rule_param, penalty_mode, ema_beta)
RULES = {
    "mean": (RuleType.MEAN_RATIO, 0.10, "shutdown", 0.95),
    "median": (RuleType.MEDIAN_RATIO, 0.10, "shutdown", 0.95),
    "cap": (RuleType.PRICE_CAP, 0.60, "shutdown", 0.95),
    "ema95": (RuleType.EMA_MEAN_RATIO, 0.10, "shutdown", 0.95),
    "mean_escrow": (RuleType.MEAN_RATIO, 0.10, "escrow", 0.95),
    "cap_escrow": (RuleType.PRICE_CAP, 0.60, "escrow", 0.95),
}


def phi_for(n: int) -> float:
    """Fixed cost keeping honest (Nash) profit above Phi as N grows; K0 = 3 Phi (one ban lethal)."""
    return 0.05 if n <= 3 else 0.04 if n <= 5 else 0.02 if n <= 10 else 0.01


def build_market(n, rule_name):
    rule, par, pen, beta = RULES[rule_name]
    phi = phi_for(n)
    mk = make_market(n, int(rule), par, fixed_cost=phi, initial_capital=3 * phi, penalty_mode=pen, ema_beta=beta)
    d = market_diagnostics(mk.m_cfg, price_grid=mk.price_grid, rule_type=RuleType.MEAN_RATIO, rule_param=0.10)
    if not d["viable"]:
        raise ValueError(f"N={n}: Nash profit {mk.nash_profits[0]:.4f} does not cover Phi={phi}")
    return mk


def done_keys(path):
    if not path.exists():
        return set()
    with open(path) as f:
        return {(r["rule"], int(r["seed"]), float(r["checkpoint"])) for r in csv.DictReader(f)}


def append(path, row):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def run(num_firms, learners, rules, seeds, seed0, env_steps, num_envs, checkpoints, tag, device="auto",
        eval_episodes=100, attr_episodes=None, updates_per_iter=4, batch_size=1024):
    out_dir = ROOT / "results" / "gpu"
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{tag}.csv"
    done = done_keys(path)
    attr = attr_episodes if attr_episodes is not None else (20 if num_firms <= 5 else 5)
    dev = pick_device(device)
    print(f"[{tag}] device={dev} N={num_firms} learners={learners} rules={rules} seeds={seeds} "
          f"env_steps={env_steps:,} envs={num_envs}", flush=True)
    for rule in rules:
        mk = build_market(num_firms, rule)
        for s in range(seeds):
            seed = seed0 + s
            if all((rule, seed, float(c)) in done for c in checkpoints):
                continue
            cfg = VecDQNConfig(env_steps=env_steps, num_envs=num_envs, seed=seed, device=device,
                               updates_per_iter=updates_per_iter, batch_size=batch_size)
            agent = VecDQN(mk, list(range(learners)), cfg)
            t0 = time.time()
            for c in checkpoints:
                agent.train(until=c)
                if (rule, seed, float(c)) in done:
                    continue
                train_s = time.time() - t0
                res = evaluate(mk, agent.cpu_policies(), eval_episodes=eval_episodes,
                               attribution_episodes=attr if c == checkpoints[-1] else 0, seed=seed)
                row = {"tag": tag, "rule": rule, "seed": seed, "num_firms": num_firms, "learners": learners,
                       "checkpoint": float(c), "env_steps_done": int(agent._it * num_envs), "epsilon": agent.eps,
                       "train_seconds": round(train_s, 1), "device": str(dev)}
                row.update(res)
                append(path, row)
                print(f"  {rule} seed {seed} at {c:.0%}: fixed-rival bankr/ep "
                      f"{res['fixed_rival_bankruptcies_per_ep']:.2f}, learner bankr/ep "
                      f"{res['learner_bankruptcies_per_ep']:.2f}, survivors {res['survivors_per_ep']:.2f}, "
                      f"mean learner profit {res['learner_profit_mean']:.3f} ({train_s / 60:.1f} min)", flush=True)
            torch.save(agent.state_dicts(), out_dir / "checkpoints" / f"{tag}_{rule}_s{seed}.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-firms", type=int, default=3)
    ap.add_argument("--learners", type=int, default=3)
    ap.add_argument("--rules", default="mean,median")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=7000)
    ap.add_argument("--env-steps", type=int, default=100_000_000)
    ap.add_argument("--num-envs", type=int, default=1024)
    ap.add_argument("--checkpoints", default="0.25,0.5,1.0")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    run(a.num_firms, a.learners, a.rules.split(","), a.seeds, a.seed0, a.env_steps, a.num_envs,
        [float(c) for c in a.checkpoints.split(",")], a.tag, a.device)


if __name__ == "__main__":
    main()
