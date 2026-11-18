"""Parallel multi-seed sweep runner for tabular Q-learning agents across rules and traits."""

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
from src.benchmarks.diagnostics import framing_market
from src.engine.rules import RuleType
from src.agents.traits import AgentTraits
from src.agents.q_learning import QLearningConfig
from src.agents.trainer import SimulationConfig, MultiAgentTrainer, SimulationResults


def run_single_seed(seed: int, rule_type: RuleType, delta: float, t_ban: int,
                    total_steps: int, warmup_steps: int, episode_length: int,
                    k0: float, phi: float, ethics_mu: float, altruism_lambda: float,
                    alpha_risk: float, agent_type: str = "adaptive") -> dict:
    """Executes a single simulation seed."""
    # Viable market (Nash profit 0.10 > phi for the defaults) with the reserve capped at K0
    m_cfg = framing_market(
        num_firms=3,
        threshold_delta=delta,
        ban_duration=t_ban,
        fixed_cost=phi,
        initial_capital=k0,
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
        total_steps=total_steps,
        episode_length=episode_length,
        warmup_steps=warmup_steps,
        rule_type=rule_type,
        rule_param=delta,
        agent_traits=traits,
        eval_window=min(total_steps // 3, 50_000),
        agent_type=agent_type,
        seed=seed,
    )

    trainer = MultiAgentTrainer(sim_cfg)
    res = trainer.run()

    return {
        "seed": seed,
        "rule": rule_type.name,
        "agent_type": agent_type,
        "delta": delta,
        "T_ban": t_ban,
        "K0_over_Phi": k0 / phi,
        "ethics_mu": ethics_mu,
        "altruism_lambda": altruism_lambda,
        "alpha_risk": alpha_risk,
        "weaponization_rate": res.weaponization_rate,
        "collusion_index": res.collusion_index,
        "social_welfare_loss": res.social_welfare_loss,
        "bankruptcy_rate": res.bankruptcy_rate,
        "mean_p0": float(res.mean_prices[0]),
        "mean_p1": float(res.mean_prices[1]),
        "mean_p2": float(res.mean_prices[2]),
        "total_fresh_bans": res.total_fresh_bans,
        "induced_bans": res.induced_bans,
        "unique_states": res.unique_states_visited,
        "state_coverage": res.state_coverage_ratio,
        "effective_states": res.effective_states_visited,
        "effective_coverage": res.effective_state_coverage,
    }


def run_parallel_sweep(args):
    rule_map = {
        "mean": RuleType.MEAN_RATIO,
        "median": RuleType.MEDIAN_RATIO,
        "cap": RuleType.PRICE_CAP,
    }
    rule = rule_map[args.rule]

    print(f"\n==================================================")
    print(f"Launching Multi-Seed Q-Learning Sweep")
    print(f"Rule: {rule.name} | delta={args.delta} | T_ban={args.t_ban} | Agent: {args.agent_type}")
    print(f"Steps: {args.steps} (Warmup: {args.warmup}) | Seeds: {args.seeds}")
    print(f"==================================================\n")

    tasks = []
    seeds = [1000 + i for i in range(args.seeds)]

    workers = min(args.workers, os.cpu_count() or 4)
    print(f"Using {workers} parallel CPU workers across {args.seeds} seeds...")

    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                run_single_seed,
                seed, rule, args.delta, args.t_ban,
                args.steps, args.warmup, args.episode_length,
                args.k0, args.phi, args.ethics, args.altruism, args.alpha,
                args.agent_type
            ): seed for seed in seeds
        }

        for fut in as_completed(futures):
            s = futures[fut]
            try:
                data = fut.result()
                results.append(data)
                print(f"  [Seed {s:4d} Complete] W = {data['weaponization_rate']:.3f} | Delta = {data['collusion_index']:+.3f} | SWL = {data['social_welfare_loss']:+.3f}")
            except Exception as e:
                print(f"  [Seed {s:4d} Failed]: {e}")

    df = pd.DataFrame(results)
    os.makedirs("data", exist_ok=True)
    out_file = args.output or f"data/q_sweep_{args.rule}_tban{args.t_ban}.csv"
    df.to_csv(out_file, index=False)

    print("\n---------------- Aggregate Summary ----------------")
    print(f"Total Completed Seeds: {len(df)}")
    print(f"Weaponization Rate (W): {df['weaponization_rate'].mean():.4f} +/- {df['weaponization_rate'].std():.4f}")
    print(f"Collusion Index (Delta): {df['collusion_index'].mean():.4f} +/- {df['collusion_index'].std():.4f}")
    print(f"Social Welfare Loss:    {df['social_welfare_loss'].mean():.4f} +/- {df['social_welfare_loss'].std():.4f}")
    print(f"Bankruptcy Rate:        {df['bankruptcy_rate'].mean():.4f} +/- {df['bankruptcy_rate'].std():.4f}")
    print(f"Results saved to: {out_file}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-seed Q-learning sweep runner.")
    parser.add_argument("--rule", type=str, default="mean", choices=["mean", "median", "cap"])
    parser.add_argument("--delta", type=float, default=0.10)
    parser.add_argument("--t-ban", type=int, default=4)
    parser.add_argument("--k0", type=float, default=0.15)
    parser.add_argument("--phi", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--warmup", type=int, default=20_000)
    parser.add_argument("--episode-length", type=int, default=500)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--agent-type", type=str, default="adaptive", choices=["adaptive", "honest", "random"])
    parser.add_argument("--ethics", type=float, default=0.0)
    parser.add_argument("--altruism", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    run_parallel_sweep(args)

