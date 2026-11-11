"""Master production experiment runner: executes all sweeps and statistical tests."""

import argparse
import os
import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.engine.rules import RuleType
from experiments.run_exact_boundary import run_boundary_sweep
from experiments.run_q_sweeps import run_parallel_sweep
from experiments.run_trait_ablations import run_trait_ablation_sweep
from experiments.run_extension_a import run_extension_a_experiment
from experiments.run_extension_b import run_extension_b_experiment
from scripts.generate_phase_diagrams import generate_phase_diagrams
from src.evaluation.stats import (
    compute_summary_stats,
    test_hypothesis_h1,
    test_hypothesis_h2,
    test_hypothesis_h3,
)


class SweepArgs:
    """Mock argument container for run_parallel_sweep."""
    def __init__(self, rule, delta=0.20, t_ban=3, k0=1.0, phi=0.10, steps=100_000, warmup=20_000, seeds=20, workers=8, output=None):
        self.rule = rule
        self.delta = delta
        self.t_ban = t_ban
        self.k0 = k0
        self.phi = phi
        self.steps = steps
        self.warmup = warmup
        self.episode_length = 500
        self.seeds = seeds
        self.workers = workers
        self.agent_type = "adaptive"
        self.ethics = 0.0
        self.altruism = 0.0
        self.alpha = 0.0
        self.output = output


def run_full_suite(smoke_test: bool = False):
    os.makedirs("data", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    start_time = time.time()

    workers = min(8, os.cpu_count() or 4)
    q_seeds = 3 if smoke_test else 20
    q_steps = 15_000 if smoke_test else 100_000
    q_warmup = 3_000 if smoke_test else 20_000
    trait_seeds = 2 if smoke_test else 5
    trait_steps = 10_000 if smoke_test else 60_000
    trait_warmup = 2_000 if smoke_test else 15_000

    print("\n=======================================================")
    print("STARTING EXPERIMENT SUITE (WEAPONIZING THE WATCHDOG)")
    print(f"Mode: {'SMOKE TEST (Fast Validation)' if smoke_test else 'PRODUCTION (Full Paper Suite)'}")
    print(f"Cores: {os.cpu_count()} (Using {workers} workers) | Seeds per Rule: {q_seeds} | Steps: {q_steps:,}")
    print("=======================================================\n")

    # -----------------------------------------------------------------
    # STEP 1: Exact DP Exploit Boundary Mapping
    # -----------------------------------------------------------------
    print("\n>>> STEP 1: Dynamic Programming Exploit Boundaries (Value Iteration)...")
    for rule in [RuleType.MEAN_RATIO, RuleType.MEDIAN_RATIO, RuleType.PRICE_CAP]:
        run_boundary_sweep(rule, delta=0.20)

    # -----------------------------------------------------------------
    # STEP 2: Decentralized Multi-Seed Q-Learning Sweeps (Hypothesis H1)
    # -----------------------------------------------------------------
    print("\n>>> STEP 2: Multi-Seed Q-Learning Sweeps (Mean vs Median vs Cap)...")
    rule_dfs = {}
    for rule_name in ["mean", "median", "cap"]:
        args = SweepArgs(
            rule=rule_name,
            seeds=q_seeds,
            steps=q_steps,
            warmup=q_warmup,
            workers=workers,
        )
        run_parallel_sweep(args)
        rule_dfs[rule_name] = pd.read_csv(f"data/q_sweep_{rule_name}_tban3.csv")

    # -----------------------------------------------------------------
    # STEP 3: Parametric Trait Ablations (Hypothesis H3)
    # -----------------------------------------------------------------
    print("\n>>> STEP 3: Parametric Trait Ablations (Ethics, Altruism, Risk)...")
    run_trait_ablation_sweep(trait_name="ethics", seeds=trait_seeds, steps=trait_steps, warmup=trait_warmup, workers=workers)
    run_trait_ablation_sweep(trait_name="altruism", seeds=trait_seeds, steps=trait_steps, warmup=trait_warmup, workers=workers)
    run_trait_ablation_sweep(trait_name="alpha", seeds=trait_seeds, steps=trait_steps, warmup=trait_warmup, workers=workers)

    # -----------------------------------------------------------------
    # STEP 4: Phase Diagrams & Learnability Gap Generation
    # -----------------------------------------------------------------
    print("\n>>> STEP 4: Generating High-Resolution 2D Phase Diagrams...")
    generate_phase_diagrams(output_dir="plots", rule_type=RuleType.MEAN_RATIO, delta=0.20, quick_mode=smoke_test)

    # -----------------------------------------------------------------
    # STEP 5: Extension A (Slow-Timescale Adaptive Regulator)
    # -----------------------------------------------------------------
    print("\n>>> STEP 5: Extension A - Adaptive Regulator & Adversarial Attacker...")
    run_extension_a_experiment(
        num_epochs=6 if smoke_test else 14,
        steps_per_epoch=500 if smoke_test else 1500,
        evaluate_empirical_attacker=not smoke_test,
    )

    # -----------------------------------------------------------------
    # STEP 6: Extension B (Out-of-Distribution Demand Shocks)
    # -----------------------------------------------------------------
    print("\n>>> STEP 6: Extension B - Out-of-Distribution Demand Shock Cohorts...")
    run_extension_b_experiment(
        steps=3_000 if smoke_test else 20_000,
        warmup=500 if smoke_test else 4_000,
        seeds=1 if smoke_test else 3,
    )

    # -----------------------------------------------------------------
    # STEP 7: Statistical Hypothesis Testing Summary
    # -----------------------------------------------------------------
    print("\n=======================================================")
    print("STATISTICAL HYPOTHESIS TESTING REPORT")
    print("=======================================================\n")

    df_mean = rule_dfs["mean"]
    df_median = rule_dfs["median"]
    df_cap = rule_dfs["cap"]

    h1_results = test_hypothesis_h1(
        df_mean["weaponization_rate"].values,
        df_median["weaponization_rate"].values,
        df_cap["weaponization_rate"].values,
    )

    print("--- HYPOTHESIS H1: Rule Vulnerability Ranking ---")
    print(f"Mean Rule W:   {h1_results['mean_w']:.4f}")
    print(f"Median Rule W: {h1_results['median_w']:.4f}")
    print(f"Price Cap W:   {h1_results['cap_w']:.4f}")
    print(f"p-val (Mean > Median): {h1_results['p_val_mean_gt_median']:.4e}")
    print(f"p-val (Mean > Cap):    {h1_results['p_val_mean_gt_cap']:.4e}")
    print(f"Supported:             {h1_results['hypothesis_supported']}")

    # Trait Ethics Test (H3)
    df_ethics = pd.read_csv("data/trait_ablation_ethics.csv")
    h3_ethics = test_hypothesis_h3(df_ethics["trait_val"].values, df_ethics["weaponization_rate"].values, trait_name="ethics_mu")
    print("\n--- HYPOTHESIS H3: Trait Modulation (Ethics mu) ---")
    print(f"Spearman Rho: {h3_ethics['spearman_rho']:.4f}")
    print(f"p-value:      {h3_ethics['p_value']:.4e}")
    print(f"Supported:    {h3_ethics['monotonic_decrease_supported']}")

    # State Visitation & Coverage Summary (Feedback Item 1)
    if "unique_states" in df_mean.columns:
        mean_visited = df_mean["unique_states"].mean()
        mean_coverage = df_mean["state_coverage"].mean() * 100.0
        print(f"\n--- STATE VISITATION & COVERAGE REPORT ---")
        print(f"Mean Distinct States Visited: {mean_visited:.0f}")
        print(f"State Space Exploration Coverage: {mean_coverage:.2f}%")

    elapsed = time.time() - start_time
    print(f"\n=======================================================")
    print(f"ALL EXPERIMENTS COMPLETED IN {elapsed / 60.0:.2f} MINUTES")
    print("=======================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master Experiment Runner (Weaponizing the Watchdog).")
    parser.add_argument("--smoke-test", action="store_true", help="Execute rapid smoke test validation (~1.5 min).")
    parser.add_argument("--production", action="store_true", help="Execute full 20-seed production suite.")
    args = parser.parse_args()

    # Default to smoke-test if neither specified, to prevent accidental 40-minute blocking runs
    is_smoke = not args.production
    run_full_suite(smoke_test=is_smoke)
