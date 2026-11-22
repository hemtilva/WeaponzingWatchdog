"""Experiment runner for Extension B: Out-of-Distribution Non-Stationary Demand Shock.

Evaluates how behavioral cohorts (Ethical vs Predatory vs Mixed) perform under
increasing panic buying surges psi in {0.0, 0.25, 0.50, 0.75}, measuring:
1. Social Welfare Retention
2. Bankruptcy Frequency
3. Weaponization Rate under Demand Disruption
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.extensions.demand_shock import run_cohort_simulation
from src.engine.rules import RuleType


def run_extension_b_experiment(
    psi_values: list = None,
    cohorts: list = None,
    steps: int = 25_000,
    warmup: int = 5_000,
    seeds: int = 3,
    output_dir: str = "plots",
    data_dir: str = "data",
):
    if psi_values is None:
        psi_values = [0.0, 0.25, 0.50, 0.75]
    if cohorts is None:
        cohorts = ["ethical", "predatory", "mixed"]

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    print("\n=======================================================")
    print("STARTING EXTENSION B: OUT-OF-DISTRIBUTION DEMAND SHOCKS")
    print(f"Cohorts: {cohorts} | Psi sweep: {psi_values} | Seeds: {seeds} | Steps: {steps:,}")
    print("=======================================================\n")

    records = []

    for cohort in cohorts:
        for psi in psi_values:
            welfares = []
            bankruptcies = []
            w_rates = []
            qs = []

            for s_idx in range(seeds):
                seed = 100 + s_idx
                res = run_cohort_simulation(
                    cohort=cohort,
                    panic_psi=psi,
                    shock_volatility=0.10,
                    steps=steps,
                    warmup=warmup,
                    rule_type=RuleType.MEAN_RATIO,
                    delta=0.20,
                    seed=seed,
                )
                welfares.append(res["mean_social_welfare"])
                bankruptcies.append(res["bankruptcy_rate"])
                w_rates.append(res["weaponization_rate"])
                qs.append(res["mean_market_size"])

            records.append({
                "cohort": cohort,
                "panic_psi": psi,
                "mean_welfare": float(np.mean(welfares)),
                "welfare_std": float(np.std(welfares)),
                "bankruptcy_rate": float(np.mean(bankruptcies)),
                "weaponization_rate": float(np.mean(w_rates)),
                "mean_market_size": float(np.mean(qs)),
            })
            print(f"Cohort: {cohort:<9} | psi: {psi:.2f} | Welfare: {np.mean(welfares):.3f} | B-Rate: {np.mean(bankruptcies):.4f} | W: {np.mean(w_rates):.3f}")

    df = pd.DataFrame(records)
    csv_path = Path(data_dir) / "extension_b_shock_cohorts.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[OK] Demand shock data saved to: {csv_path}")

    # Generate Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    cohort_colors = {
        "ethical": "#2ca02c",
        "predatory": "#d62728",
        "mixed": "#ff7f0e",
    }
    cohort_markers = {
        "ethical": "o",
        "predatory": "s",
        "mixed": "^",
    }

    # Subplot 1: Social Welfare Retention
    for cohort in cohorts:
        cdf = df[df["cohort"] == cohort]
        ax1.plot(
            cdf["panic_psi"],
            cdf["mean_welfare"],
            label=f"Cohort: {cohort.capitalize()}",
            color=cohort_colors[cohort],
            marker=cohort_markers[cohort],
            linewidth=2.5,
            markersize=7,
        )
    ax1.set_xlabel("Panic Buying Multiplier (psi)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Mean Social Welfare", fontsize=11, fontweight="bold")
    ax1.set_title("Welfare Retention Under Non-Stationary Shock", fontsize=12, fontweight="bold", pad=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="best", frameon=True)

    # Subplot 2: Bankruptcy Rate
    for cohort in cohorts:
        cdf = df[df["cohort"] == cohort]
        ax2.plot(
            cdf["panic_psi"],
            cdf["bankruptcy_rate"],
            label=f"Cohort: {cohort.capitalize()}",
            color=cohort_colors[cohort],
            marker=cohort_markers[cohort],
            linewidth=2.5,
            markersize=7,
        )
    ax2.set_xlabel("Panic Buying Multiplier (psi)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Bankruptcy Rate (Liquidations / Period)", fontsize=11, fontweight="bold")
    ax2.set_title("Market Insolvency Vulnerability", fontsize=12, fontweight="bold", pad=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="best", frameon=True)

    plt.tight_layout()
    plot_path = Path(output_dir) / "extension_b_demand_shock.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[OK] High-resolution visualization saved to: {plot_path}\n")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Extension B: Demand Shock Cohort Sweeps.")
    parser.add_argument("--steps", type=int, default=20_000, help="Total simulation steps.")
    parser.add_argument("--warmup", type=int, default=4_000, help="Warmup steps before shock.")
    parser.add_argument("--seeds", type=int, default=3, help="Seeds per configuration.")
    parser.add_argument("--quick", action="store_true", help="Run 3000 steps quick smoke run.")
    args = parser.parse_args()

    steps = 3000 if args.quick else args.steps
    warmup = 600 if args.quick else args.warmup

    run_extension_b_experiment(
        steps=steps,
        warmup=warmup,
        seeds=args.seeds,
    )

