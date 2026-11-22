#!/usr/bin/env python3
"""Defended Watchdog Experiment Runner (Option 1).

Evaluates the efficacy of:
1. Exponential Moving Average (EMA) smoothed reference benchmark
2. Escrow penalty defense (diverting excess profit rather than market shutdown)
3. Combined EMA + Escrow Defense

Generates:
- data/defended_watchdog_comparison.csv
- plots/defended_watchdog_comparison.png
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


def run_experiment():
    print("=" * 65)
    print("Running Option 1: Defended Watchdog Evaluation Suite")
    print("=" * 65)

    data_dir = root_dir / "data"
    plots_dir = root_dir / "plots"
    data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    m_cfg = MarketConfig(
        num_firms=4,
        price_sensitivity=0.25,
        fixed_cost=0.10,
        initial_capital=1.0,
        ban_duration=3,
        threshold_delta=0.20,
    )

    regimes = [
        ("Vulnerable Watchdog (Mean-Ratio, Shutdown)", RuleType.MEAN_RATIO, "shutdown", 0.0),
        ("Defended Watchdog (EMA beta=0.85, Shutdown)", RuleType.EMA_MEAN_RATIO, "shutdown", 0.85),
        ("Defended Watchdog (EMA beta=0.95, Shutdown)", RuleType.EMA_MEAN_RATIO, "shutdown", 0.95),
        ("Defended Watchdog (Mean-Ratio, Escrow)", RuleType.MEAN_RATIO, "escrow", 0.0),
        ("Full Defended Watchdog (EMA beta=0.95, Escrow)", RuleType.EMA_MEAN_RATIO, "escrow", 0.95),
    ]

    records = []
    seeds = [42, 101, 202]

    for label, rule_type, pen_mode, beta_val in regimes:
        w_list = []
        bans_list = []
        coll_list = []
        for s in seeds:
            sim_cfg = SimulationConfig(
                market=m_cfg,
                total_steps=2000,
                warmup_steps=100,
                episode_length=300,
                rule_type=rule_type,
                rule_param=0.20,
                penalty_mode=pen_mode,
                ema_beta=beta_val,
                eval_window=1000,
                agent_type="scripted_attacker",
                seed=s,
            )
            trainer = MultiAgentTrainer(sim_cfg)
            res = trainer.run()
            w_list.append(res.weaponization_rate)
            bans_list.append(res.total_fresh_bans)
            coll_list.append(res.collusion_index)

        records.append({
            "Regime": label,
            "RuleType": rule_type.name,
            "PenaltyMode": pen_mode,
            "EMA_Beta": beta_val,
            "WeaponizationRate": float(np.mean(w_list)),
            "WeaponizationRate_Std": float(np.std(w_list)),
            "TotalFreshBans": float(np.mean(bans_list)),
            "TotalFreshBans_Std": float(np.std(bans_list)),
            "CollusionIndex": float(np.mean(coll_list)),
        })
        print(f"[{label}] -> W = {np.mean(w_list):.4f}, Fresh Bans = {np.mean(bans_list):.1f}")

    # Beta sensitivity sweep
    beta_sweep = [0.0, 0.50, 0.70, 0.85, 0.90, 0.95, 0.98]
    beta_records = []
    for b in beta_sweep:
        b_w, b_bans = [], []
        for s in seeds:
            sim_cfg = SimulationConfig(
                market=m_cfg,
                total_steps=2000,
                warmup_steps=100,
                episode_length=300,
                rule_type=RuleType.EMA_MEAN_RATIO,
                rule_param=0.20,
                penalty_mode="shutdown",
                ema_beta=b,
                eval_window=1000,
                agent_type="scripted_attacker",
                seed=s,
            )
            res = MultiAgentTrainer(sim_cfg).run()
            b_w.append(res.weaponization_rate)
            b_bans.append(res.total_fresh_bans)
        beta_records.append({
            "beta": b,
            "W": float(np.mean(b_w)),
            "bans": float(np.mean(b_bans)),
        })

    # Save summary dataframe
    df = pd.DataFrame(records)
    csv_path = data_dir / "defended_watchdog_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[SAVED] {csv_path}")

    # Plotting 2-panel comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Harm / Bans across Regimes
    short_labels = [
        "Vulnerable\n(Shutdown)",
        "EMA (β=0.85)\n(Shutdown)",
        "EMA (β=0.95)\n(Shutdown)",
        "Escrow\n(No Shutdown)",
        "Full Defended\n(EMA + Escrow)",
    ]
    bans = [r["TotalFreshBans"] for r in records]
    colors = ["#e74c3c", "#e67e22", "#2ecc71", "#3498db", "#1abc9c"]

    bars = ax1.bar(short_labels, bans, color=colors, edgecolor="black", alpha=0.85)
    ax1.set_ylabel("Total Induced Bans in Eval Window", fontsize=11, fontweight="bold")
    ax1.set_title("(a) Regulatory Bans Under Scripted Attack", fontsize=12, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, b_val in zip(bars, bans):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{b_val:.1f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    # Panel 2: Inertia Beta Sweep
    b_vals = [r["beta"] for r in beta_records]
    b_bans = [r["bans"] for r in beta_records]

    ax2.plot(b_vals, b_bans, marker="o", linewidth=2.5, color="#2980b9", markersize=8)
    ax2.set_xlabel("EMA Smoothing Memory (β)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Total Induced Bans", fontsize=11, fontweight="bold")
    ax2.set_title("(b) Memory Inertia Protection Curve", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.axvline(0.90, color="#27ae60", linestyle=":", label="Immunization Threshold (β ≥ 0.90)")
    ax2.legend(frameon=True)

    plt.tight_layout()
    plot_path = plots_dir / "defended_watchdog_comparison.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[SAVED] {plot_path}")
    print("\nDefended Watchdog Experiment Complete!")


if __name__ == "__main__":
    run_experiment()
