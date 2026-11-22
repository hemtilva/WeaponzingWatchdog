"""Experiment script to compute and map the exact theoretical exploit boundary.

Sweeps over rule types, delta, T_ban, and K_0 / Phi using Value Iteration to
identify parameter regimes where regulator framing yields positive exploitability gain.
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.engine.rules import RuleType
from src.benchmarks.exact_dp import DPSolverConfig, ExactDPSolver


def run_boundary_sweep(rule_type: RuleType = RuleType.MEAN_RATIO, delta: float = 0.20):
    print(f"=== Mapping Exact Exploit Boundary for Rule: {rule_type.name} (delta={delta}) ===")

    price_grid = np.array([1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5], dtype=np.float64)
    fixed_cost = 0.2
    # Capital levels: 0 to 6 in multiples of Phi
    capital_levels = np.arange(7, dtype=np.float64) * fixed_cost

    t_ban_values = [1, 2, 3, 4]
    k0_ratios = [1.0, 2.0, 3.0, 4.0, 5.0]

    records = []

    for t_ban in t_ban_values:
        print(f"\nEvaluating T_ban = {t_ban}...")
        config = DPSolverConfig(
            num_firms=3,
            price_grid=price_grid,
            marginal_cost=1.0,
            quality=2.0,
            outside_option=0.0,
            sigma_d=0.25,
            market_size=1.0,
            fixed_cost=fixed_cost,
            T_ban=t_ban,
            audit_budget=1,
            rule_type=rule_type,
            rule_param=delta,
            capital_levels=capital_levels,
            gamma=0.90,
            tol=1e-3,
            max_iter=100,
        )

        solver = ExactDPSolver(config)
        V, policy = solver.solve()

        for k0_ratio in k0_ratios:
            k0_val = k0_ratio * fixed_cost
            gain = solver.compute_exploitability_gain(V, initial_capital=k0_val)

            # Check if framing is chosen at initial state (p1* < p^N)
            k0_idx = solver._snap_capital(k0_val)
            opt_action_idx = policy[0, 0, 0, k0_idx, k0_idx, k0_idx]
            opt_p = price_grid[opt_action_idx]
            nash_p = solver.nash_prices[0]
            is_framing = opt_p < nash_p

            records.append({
                "rule": rule_type.name,
                "delta": delta,
                "T_ban": t_ban,
                "K0_over_Phi": k0_ratio,
                "optimal_price": opt_p,
                "nash_price": nash_p,
                "is_framing": is_framing,
                "exploitability_gain": gain,
            })
            print(f"  K0/Phi = {k0_ratio:.1f} | opt_p = {opt_p:.2f} (Nash={nash_p:.2f}) | Gain = {gain:+.4f} | Framing = {is_framing}")

    df = pd.DataFrame(records)
    os.makedirs("data", exist_ok=True)
    out_path = f"data/exploit_boundary_{rule_type.name.lower()}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nResults successfully saved to {out_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Map exact theoretical exploit boundary.")
    parser.add_argument("--rule", type=str, default="mean", choices=["mean", "median", "cap"])
    parser.add_argument("--delta", type=float, default=0.20)
    args = parser.parse_args()

    rule_map = {
        "mean": RuleType.MEAN_RATIO,
        "median": RuleType.MEDIAN_RATIO,
        "cap": RuleType.PRICE_CAP,
    }
    run_boundary_sweep(rule_map[args.rule], args.delta)
