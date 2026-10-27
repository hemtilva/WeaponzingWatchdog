"""Parametric trait sweep runner over ethics, altruism, and risk aversion."""

import argparse
import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.engine.market import MarketConfig
from src.engine.rules import RuleType
from src.agents.traits import AgentTraits
from src.agents.q_learning import QLearningConfig
from src.agents.trainer import SimulationConfig, MultiAgentTrainer
from src.evaluation.stats import compute_summary_stats, test_hypothesis_h3


def run_single_trait_seed(
    seed: int,
    rule_type: RuleType,
    delta: float,
    t_ban: int,
    k0: float,
    phi: float,
    steps: int,
    warmup: int,
    ethics_mu: float,
    altruism_lambda: float,
    alpha_risk: float,
    target_trait: str,
    trait_val: float,
) -> dict:
    """Runs a single simulation seed with specified trait vector."""
    m_cfg = MarketConfig(
        num_firms=3,
        price_sensitivity=0.25,
        fixed_cost=phi,
        initial_capital=k0,
        ban_duration=t_ban,
        threshold_delta=delta,
    )
    q_cfg = QLearningConfig(
        learning_rate=0.10,
        gamma=0.95,
        eps_start=1.0,
        eps_min=0.01,
        eps_decay=0.99995,
        fixed_cost=phi,
    )
    traits = [
        AgentTraits(alpha=alpha_risk, lambda_altruism=altruism_lambda, mu_ethics=ethics_mu)
        for _ in range(3)
    ]
    sim_cfg = SimulationConfig(
        market=m_cfg,
        q_learning=q_cfg,
        total_steps=steps,
        episode_length=500,
        warmup_steps=warmup,
        rule_type=rule_type,
        rule_param=delta,
        agent_traits=traits,
        eval_window=min(steps // 3, 50_000),
        seed=seed,
    )

    trainer = MultiAgentTrainer(sim_cfg)
    res = trainer.run()

    return {
        "seed": seed,
        "target_trait": target_trait,
        "trait_val": trait_val,
        "ethics_mu": ethics_mu,
        "altruism_lambda": altruism_lambda,
        "alpha_risk": alpha_risk,
        "weaponization_rate": res.weaponization_rate,
        "collusion_index": res.collusion_index,
        "social_welfare_loss": res.social_welfare_loss,
        "bankruptcy_rate": res.bankruptcy_rate,
        "total_fresh_bans": res.total_fresh_bans,
        "induced_bans": res.induced_bans,
    }


def run_trait_ablation_sweep(
    trait_name: str = "ethics",
    values: list = None,
    seeds: int = 5,
    steps: int = 60_000,
    warmup: int = 15_000,
    workers: int = 8,
    out_file: str = None,
):
    """Orchestrates parallel multi-seed trait sweeps."""
    if values is None:
        if trait_name == "ethics":
            values = [0.0, 0.10, 0.25, 0.50, 1.00]
        elif trait_name == "altruism":
            values = [0.0, 0.10, 0.20, 0.35, 0.50]
        elif trait_name == "alpha":
            values = [0.0, 0.50, 1.00, 1.50, 2.00]
        else:
            values = [0.0, 0.25, 0.50]

    print(f"\n=======================================================")
    print(f"Launching Trait Ablation Sweep: {trait_name}")
    print(f"Values: {values} | Seeds per value: {seeds} | Total tasks: {len(values) * seeds}")
    print(f"=======================================================\n")

    tasks = []
    rule_type = RuleType.MEAN_RATIO
    delta = 0.20
    t_ban = 3
    k0 = 1.0
    phi = 0.10

    max_cpu = min(workers, os.cpu_count() or 4)
    results = []

    with ProcessPoolExecutor(max_workers=max_cpu) as executor:
        future_map = {}
        for val in values:
            mu = val if trait_name == "ethics" else 0.0
            lam = val if trait_name == "altruism" else 0.0
            alp = val if trait_name == "alpha" else 0.0

            for s_idx in range(seeds):
                seed = 2000 + int(val * 100) + s_idx
                fut = executor.submit(
                    run_single_trait_seed,
                    seed, rule_type, delta, t_ban, k0, phi,
                    steps, warmup, mu, lam, alp, trait_name, val
                )
                future_map[fut] = (val, seed)

        for fut in as_completed(future_map):
            val, seed = future_map[fut]
            try:
                res = fut.result()
                results.append(res)
                print(f"  [{trait_name}={val:.2f} | Seed {seed}] W = {res['weaponization_rate']:.3f} | Delta = {res['collusion_index']:+.3f}")
            except Exception as e:
                print(f"  [{trait_name}={val:.2f} | Seed {seed}] Failed: {e}")

    df = pd.DataFrame(results)
    os.makedirs("data", exist_ok=True)
    out_path = out_file or f"data/trait_ablation_{trait_name}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSweep completed. Results saved to: {out_path}")

    # Statistical Evaluation of Hypothesis H3
    if len(df) > 0:
        h3_test = test_hypothesis_h3(df["trait_val"].values, df["weaponization_rate"].values, trait_name=trait_name)
        print("\n--- Hypothesis H3 Test (Monotonic Decrease) ---")
        print(f"Trait: {trait_name}")
        print(f"Spearman Correlation (rho): {h3_test['spearman_rho']:.4f}")
        print(f"P-value:                    {h3_test['p_value']:.4e}")
        print(f"Monotonic Decrease Holds:   {h3_test['monotonic_decrease_supported']}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parametric trait sweep runner.")
    parser.add_argument("--trait", type=str, default="ethics", choices=["ethics", "altruism", "alpha"])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--warmup", type=int, default=8_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    run_trait_ablation_sweep(
        trait_name=args.trait,
        seeds=args.seeds,
        steps=args.steps,
        warmup=args.warmup,
        workers=args.workers,
        out_file=args.output,
    )
