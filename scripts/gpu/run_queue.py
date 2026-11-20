"""Run the GPU experiment queue in priority order (resumable).

  python scripts/gpu/run_queue.py --benchmark        # measure speed, print time estimates
  python scripts/gpu/run_queue.py                     # run everything (skips finished work)
  python scripts/gpu/run_queue.py --only Q2,Q3        # run selected items

Every item writes results/gpu/<tag>.csv after each evaluation; stopping and restarting the
command continues where it left off. Commit and push results/gpu/ when done (see GPU_SETUP.md).
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.gpu.run_vec_dqn import build_market, run  # noqa: E402
from src.gpu.vec_dqn import VecDQN, VecDQNConfig, pick_device  # noqa: E402

M = 1_000_000
# id: (tag, N, learners, rules, seeds, env_steps, question)
QUEUE = {
    "Q1": ("sanity_N3_K1", 3, 1, "mean,median,cap", 5, 2 * M,
           "Sanity: one learner vs Nash rivals must reproduce the CPU result (frames under mean only)."),
    "Q2": ("all_N3_K3", 3, 3, "mean,median", 10, 100 * M,
           "All three sellers are DQN learners: does framing emerge with deep RL, mean vs median?"),
    "Q3": ("N10_K2", 10, 2, "mean,median", 10, 50 * M,
           "Two learners among 8 Nash sellers: shared framing, then a duel?"),
    "Q4": ("N10_K4", 10, 4, "mean,median", 10, 50 * M,
           "Four learners among 6 Nash sellers."),
    "Q5": ("N20_K1", 20, 1, "mean,median", 5, 30 * M,
           "N = 20: Proposition 1 says one undercutter cannot frame (1 < 0.1 x 19). Expect none."),
    "Q6": ("N20_K2", 20, 2, "mean,median", 5, 30 * M,
           "N = 20 with two learners: framing needs both together. Do they coordinate?"),
    "Q7": ("all_N5_K5", 5, 5, "mean,median", 10, 100 * M,
           "All five sellers learning."),
    "Q8": ("all_N3_K3_defences", 3, 3, "ema95,mean_escrow,cap_escrow", 10, 100 * M,
           "Defences against all-DQN markets: EMA, mean+escrow, cap+escrow."),
}


def benchmark(seconds=20):
    dev = pick_device()
    print(f"device: {dev}")
    rates = {}
    for n, k in ((3, 1), (3, 3), (10, 4), (20, 2)):
        mk = build_market(n, "mean")
        agent = VecDQN(mk, list(range(k)), VecDQNConfig(env_steps=10**12, learning_starts=5_000))
        agent.train(until=2_000 / agent.iters)            # warm-up past learning_starts
        t0, it0 = time.time(), agent._it
        while time.time() - t0 < seconds:
            agent.train(until=(agent._it + 50) / agent.iters)
        rates[(n, k)] = (agent._it - it0) * agent.cfg.num_envs / (time.time() - t0)
        print(f"  N={n:2} learners={k}: {rates[(n, k)]:,.0f} market-steps/s")
    print("\nEstimated time per queue item (training only; evaluation adds a few minutes per run):")
    total = 0.0
    for q, (tag, n, k, rules, seeds, steps, _) in QUEUE.items():
        key = min(rates, key=lambda r: abs(r[0] - n) + abs(r[1] - k))
        hours = len(rules.split(",")) * seeds * steps / rates[key] / 3600
        total += hours
        print(f"  {q} {tag:22} {hours:6.1f} h")
    print(f"  total {total:.1f} h")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    if a.benchmark:
        benchmark()
        return
    items = a.only.split(",") if a.only else list(QUEUE)
    for q in items:
        tag, n, k, rules, seeds, steps, question = QUEUE[q]
        print(f"\n===== {q}: {question}", flush=True)
        run(n, k, rules.split(","), seeds, 7000, steps, 1024, [0.25, 0.5, 1.0], tag)


if __name__ == "__main__":
    main()
