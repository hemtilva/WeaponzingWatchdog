#!/usr/bin/env python3
"""Deep Q-Network (DQN) Continuous Scalability Ablation (Option 2).

Evaluates whether predatory framing and market dynamics emerge under neural network
function approximation in continuous state space, scaling to N = 5 firms where
tabular methods suffer exponential state space explosion (|S| > 1.6M).

Generates:
- data/dqn_continuous_ablation.csv
- plots/dqn_continuous_ablation.png
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.engine.market import MarketConfig
from src.engine.rules import RuleType
from src.agents.trainer import SimulationConfig, MultiAgentTrainer
from src.agents.dqn_agent import DQNConfig
from src.agents.dqn_trainer import DQNSimulationConfig, DQNTrainer


def run_experiment():
    print("=" * 65)
    print("Running Option 2: DQN Continuous Scalability Ablation (N=3, 4, 5)")
    print("=" * 65)

    data_dir = root_dir / "data"
    plots_dir = root_dir / "plots"
    data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    records = []

    # Configurations for N = 3, 4, 5
    n_settings = [
        (3, "Tabular Q-Learning", True),
        (3, "Deep Q-Network (DQN)", False),
        (4, "Tabular Q-Learning", True),
        (4, "Deep Q-Network (DQN)", False),
        (5, "Deep Q-Network (DQN)", False),  # Tabular intractable (|S| = 1.6M)
    ]

    rules = [
        ("Mean-Ratio", RuleType.MEAN_RATIO),
        ("Median-Ratio", RuleType.MEDIAN_RATIO),
    ]

    for n_firms, model_type, is_tabular in n_settings:
        m_cfg = MarketConfig(
            num_firms=n_firms,
            marginal_costs=[0.5] * n_firms,
            qualities=[2.0] * n_firms,
            price_sensitivity=0.25,
            fixed_cost=0.10,
            initial_capital=1.0,
            ban_duration=3,
            threshold_delta=0.20,
        )

        for rule_label, rule_type in rules:
            print(f"\nEvaluating N = {n_firms} | Model = {model_type} | Rule = {rule_label}...")

            if is_tabular:
                # Tabular runner
                sim_cfg = SimulationConfig(
                    market=m_cfg,
                    total_steps=5_000,
                    warmup_steps=200,
                    episode_length=200,
                    rule_type=rule_type,
                    rule_param=0.20,
                    eval_window=1_000,
                    seed=42,
                )
                trainer = MultiAgentTrainer(sim_cfg)
                res = trainer.run()
                coverage = res.state_coverage_ratio
            else:
                # DQN runner
                dqn_cfg = DQNConfig(
                    batch_size=32,
                    buffer_capacity=5_000,
                    learning_rate=1e-3,
                    eps_decay=0.9992,
                )
                sim_cfg = DQNSimulationConfig(
                    market=m_cfg,
                    dqn_config=dqn_cfg,
                    total_steps=5_000,
                    warmup_steps=200,
                    episode_length=200,
                    rule_type=rule_type,
                    rule_param=0.20,
                    eval_window=1_000,
                    seed=42,
                )
                trainer = DQNTrainer(sim_cfg)
                res = trainer.run()
                coverage = 1.0  # Continuous function approximation spans continuous space

            rec = {
                "NumFirms": n_firms,
                "Model": model_type,
                "Rule": rule_label,
                "CollusionIndex": float(res.collusion_index),
                "WeaponizationRate": float(res.weaponization_rate),
                "MeanPrice": float(np.mean(res.mean_prices)),
                "MeanProfit": float(np.mean(res.mean_profits)),
                "StateCoverage": float(coverage),
                "TotalFreshBans": int(res.total_fresh_bans),
            }
            records.append(rec)
            print(f"  -> Collusion Δ = {res.collusion_index:.3f}, W = {res.weaponization_rate:.3f}, Bans = {res.total_fresh_bans}")

    df = pd.DataFrame(records)
    csv_path = data_dir / "dqn_continuous_ablation.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[SAVED] {csv_path}")

    # Plot 2-panel figure: (a) Collusion Index Δ across N; (b) Scalability & States Visited
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Collusion Index by Model & N (Mean-Ratio)
    df_mean = df[df["Rule"] == "Mean-Ratio"]
    tab_data = df_mean[df_mean["Model"] == "Tabular Q-Learning"]
    dqn_data = df_mean[df_mean["Model"] == "Deep Q-Network (DQN)"]

    ax1.plot(
        tab_data["NumFirms"],
        tab_data["CollusionIndex"],
        marker="s",
        linewidth=2.2,
        markersize=8,
        color="#e67e22",
        label="Tabular Q-Learning",
    )
    ax1.plot(
        dqn_data["NumFirms"],
        dqn_data["CollusionIndex"],
        marker="o",
        linewidth=2.5,
        markersize=9,
        color="#2980b9",
        label="Deep Q-Network (DQN)",
    )
    ax1.set_xlabel("Number of Oligopoly Firms (N)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Collusion Index (Δ)", fontsize=11, fontweight="bold")
    ax1.set_title("(a) Collusive Pricing under Function Approximation", fontsize=12, fontweight="bold")
    ax1.set_xticks([3, 4, 5])
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(frameon=True)

    # Panel 2: State Space Scalability: Discrete |S| vs Continuous DQN State Vector Dimension
    n_axis = np.array([3, 4, 5, 6])
    discrete_states = (7 ** n_axis) * (2 ** n_axis) * 3
    dqn_dims = 2 * n_axis + 1

    color = "#c0392b"
    ax2.set_xlabel("Number of Oligopoly Firms (N)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Discrete Tabular States |S| (Log Scale)", color=color, fontsize=11, fontweight="bold")
    ax2.semilogy(n_axis, discrete_states, color=color, marker="^", linewidth=2.5, markersize=8, label="Discrete |S| = (M^N)(2^N)*3")
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.set_xticks([3, 4, 5, 6])
    ax2.grid(True, linestyle="--", alpha=0.5)

    ax2_twin = ax2.twinx()
    color_twin = "#27ae60"
    ax2_twin.set_ylabel("DQN Continuous State Vector Dim (2N + 1)", color=color_twin, fontsize=11, fontweight="bold")
    ax2_twin.plot(n_axis, dqn_dims, color=color_twin, marker="o", linewidth=2.5, linestyle="--", markersize=8, label="DQN Continuous State Dim")
    ax2_twin.tick_params(axis="y", labelcolor=color_twin)
    ax2.set_title("(b) Curse of Dimensionality vs DQN Linear Scaling", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plot_path = plots_dir / "dqn_continuous_ablation.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[SAVED] {plot_path}")
    print("\nOption 2 DQN Scalability Ablation Complete!")


if __name__ == "__main__":
    run_experiment()
