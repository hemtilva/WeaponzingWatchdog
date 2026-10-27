"""Generates 2D phase diagrams and maps the Learnability Gap between DP and Q-learning."""

import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.engine.rules import RuleType
from src.benchmarks.exact_dp import DPSolverConfig, ExactDPSolver
from src.agents.trainer import SimulationConfig, MultiAgentTrainer
from src.engine.market import MarketConfig


def generate_phase_diagrams(
    output_dir: str = "plots",
    rule_type: RuleType = RuleType.MEAN_RATIO,
    delta: float = 0.20,
    quick_mode: bool = False,
):
    """Computes and renders 2D Phase Diagrams with the DP Exploitability Frontier."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n=======================================================")
    print(f"Generating Phase Diagrams for {rule_type.name} (delta={delta})")
    print(f"=======================================================\n")

    # Grid definition
    t_ban_vals = np.array([1, 2, 3, 4, 5]) if not quick_mode else np.array([1, 2, 3, 4])
    k0_phi_vals = np.linspace(1.5, 8.0, 6) if not quick_mode else np.linspace(2.0, 6.0, 4)

    T_grid, K_grid = np.meshgrid(t_ban_vals, k0_phi_vals)
    W_grid = np.zeros_like(T_grid, dtype=np.float64)
    Delta_grid = np.zeros_like(T_grid, dtype=np.float64)
    Gain_grid = np.zeros_like(T_grid, dtype=np.float64)

    fixed_cost = 0.20
    price_grid = np.linspace(1.0, 2.5, 7)
    capital_levels = np.arange(9, dtype=np.float64) * fixed_cost

    # 1. Compute Exact DP Frontier across T_ban and K0 / Phi
    print("Computing exact Dynamic Programming exploit frontier...")
    for i, t_b in enumerate(t_ban_vals):
        dp_cfg = DPSolverConfig(
            num_firms=3,
            price_grid=price_grid,
            marginal_cost=1.0,
            quality=2.0,
            outside_option=0.0,
            sigma_d=0.25,
            fixed_cost=fixed_cost,
            T_ban=int(t_b),
            rule_type=rule_type,
            rule_param=delta,
            capital_levels=capital_levels,
            gamma=0.90,
            tol=1e-3,
            max_iter=100,
        )
        solver = ExactDPSolver(dp_cfg)
        V, _ = solver.solve()

        for j, k0_ratio in enumerate(k0_phi_vals):
            k0_val = k0_ratio * fixed_cost
            gain = solver.compute_exploitability_gain(V, initial_capital=k0_val)
            Gain_grid[j, i] = gain

    # 2. Compute Decentralized Q-Learning Weaponization Surface
    print("Simulating decentralized Q-learning response surface...")
    steps = 40_000 if not quick_mode else 20_000
    warmup = 10_000 if not quick_mode else 5_000

    for i, t_b in enumerate(t_ban_vals):
        for j, k0_ratio in enumerate(k0_phi_vals):
            m_cfg = MarketConfig(
                num_firms=3,
                price_sensitivity=0.25,
                fixed_cost=fixed_cost,
                initial_capital=k0_ratio * fixed_cost,
                ban_duration=int(t_b),
                threshold_delta=delta,
            )
            sim_cfg = SimulationConfig(
                market=m_cfg,
                total_steps=steps,
                warmup_steps=warmup,
                episode_length=500,
                rule_type=rule_type,
                rule_param=delta,
                eval_window=15_000,
                seed=42 + i * 10 + j,
            )
            trainer = MultiAgentTrainer(sim_cfg)
            res = trainer.run()

            W_grid[j, i] = res.weaponization_rate
            Delta_grid[j, i] = res.collusion_index
            print(f"  [T_ban={t_b:d}, K0/Phi={k0_ratio:.1f}] -> W = {res.weaponization_rate:.3f} | Gain = {Gain_grid[j, i]:+.3f}")

    # Plot 1: Weaponization Rate Phase Diagram & Learnability Gap
    fig, ax = plt.subplots(figsize=(9, 7))
    cf = ax.contourf(T_grid, K_grid, W_grid, levels=np.linspace(0, 1.0, 11), cmap="YlOrRd", alpha=0.85)
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("Weaponization Rate (W)", fontsize=11, fontweight="bold")

    # Exact DP Frontier: Contour line at E_1 = 0
    cp = ax.contour(T_grid, K_grid, Gain_grid, levels=[0.0], colors="#2563eb", linewidths=3.0, linestyles="--")
    ax.clabel(cp, inline=True, fontsize=10, fmt={0.0: "Exact DP Frontier (Gain = 0)"})

    # Shaded Learnability Gap annotation
    ax.set_title(
        f"Phase Diagram: Weaponization Rate (W) & Learnability Gap\n"
        f"Rule: {rule_type.name} (delta={delta})",
        fontsize=13, fontweight="bold", pad=12
    )
    ax.set_xlabel("Ban Duration (T_ban)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Capital Reserves Ratio (K0 / Phi)", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.4)

    # Plot theoretical threshold T_ban = K0 / Phi
    t_diag = np.linspace(1, 5, 50)
    ax.plot(t_diag, t_diag, color="#10b981", linestyle=":", linewidth=2.0, label="Theoretical Liquidation Barrier (T_ban = K0 / Phi)")
    ax.legend(loc="upper left", frameon=True, fontsize=9)

    out_w = os.path.join(output_dir, "phase_diagram_weaponization.png")
    plt.tight_layout()
    plt.savefig(out_w, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nWeaponization phase diagram saved to: {out_w}")

    # Plot 2: Collusion Index Phase Diagram
    fig, ax = plt.subplots(figsize=(9, 7))
    cf = ax.contourf(T_grid, K_grid, Delta_grid, levels=12, cmap="viridis", alpha=0.85)
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("Collusion Index (Delta)", fontsize=11, fontweight="bold")

    cp = ax.contour(T_grid, K_grid, Gain_grid, levels=[0.0], colors="#ffffff", linewidths=2.5, linestyles="--")
    ax.clabel(cp, inline=True, fontsize=10, fmt={0.0: "Gain = 0"})

    ax.set_title(
        f"Phase Diagram: Collusion Index (Delta)\n"
        f"Rule: {rule_type.name} (delta={delta})",
        fontsize=13, fontweight="bold", pad=12
    )
    ax.set_xlabel("Ban Duration (T_ban)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Capital Reserves Ratio (K0 / Phi)", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.4)

    out_c = os.path.join(output_dir, "phase_diagram_collusion.png")
    plt.tight_layout()
    plt.savefig(out_c, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Collusion phase diagram saved to: {out_c}")


if __name__ == "__main__":
    generate_phase_diagrams(quick_mode=True)
