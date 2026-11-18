"""Measure training throughput of the Python reference loop vs the native C++ loop.

Same market and protocol as the all-learner studies (3 Q-learners, mean rule).
Usage: python scripts/benchmark_backends.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.common import Job, build  # noqa: E402


def main():
    rates = {}
    for backend, steps in (("python", 100_000), ("native", 10_000_000)):
        tr = build(Job(tag="bench", seed=1, train_steps=steps))
        tr.cfg.backend = backend
        t0 = time.time()
        tr.run()
        rates[backend] = steps / (time.time() - t0)
        print(f"{backend:>6}: {rates[backend]:>14,.0f} steps/s")
    print(f"speed-up: {rates['native'] / rates['python']:,.0f}x")


if __name__ == "__main__":
    main()
