"""Generates a high-resolution, multi-panel visual trace plot of agent interactions and bans."""

import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.engine.market import MarketConfig
from src.engine.rules import RuleType
from src.agents.trainer import SimulationConfig, MultiAgentTrainer


def generate_trace_plot(
    output_path: str = "plots/trace_plot.png",
    rule_type: RuleType = RuleType.MEAN_RATIO,
    delta: float = 0.20,
    steps: int = 50_000,
    trace_window: int = 250,
    seed: int = 1000,
):
    print(f"Running simulation to collect trace data (Rule: {rule_type.name}, delta={delta})...")

    sim_cfg = SimulationConfig(
        total_steps=steps,
        warmup_steps=10_000,
        episode_length=500,
        rule_type=rule_type,
        rule_param=delta,
        record_trace=True,
        trace_steps=trace_window,
        seed=seed,
    )

    trainer = MultiAgentTrainer(sim_cfg)
    results = trainer.run()

    trace = results.trace_data
    if not trace:
        print("Error: No trace data collected.", file=sys.stderr)
        return

    # Extract arrays
    timesteps = np.array([r["step"] for r in trace])
    # Normalize time axis to [0, T]
    t_norm = timesteps - timesteps[0]

    p0 = np.array([r["prices"][0] for r in trace])
    p1 = np.array([r["prices"][1] for r in trace])
    p2 = np.array([r["prices"][2] for r in trace])

    k0 = np.array([r["capitals"][0] for r in trace])
    k1 = np.array([r["capitals"][1] for r in trace])
    k2 = np.array([r["capitals"][2] for r in trace])

    b0 = np.array([r["ban_timers"][0] for r in trace])
    b1 = np.array([r["ban_timers"][1] for r in trace])
    b2 = np.array([r["ban_timers"][2] for r in trace])

    ref_p = np.array([r["ref_price"] for r in trace])
    thresh_p = np.array([r["threshold"] for r in trace])
    n_act = np.array([r["active_count"] for r in trace])

    # Plot setup: 3 subplots with shared x-axis
    fig, (ax_p, ax_k, ax_n) = plt.subplots(3, 1, figsize=(14, 10), sharex=True, gridspec_kw={"height_ratios": [2.5, 1.5, 1]})

    # 1. Price Timeline
    ax_p.plot(t_norm, p0, label="Seller 0 (Attacker)", color="#1f77b4", linewidth=2.0)
    ax_p.plot(t_norm, p1, label="Seller 1 (Rival)", color="#ff7f0e", linewidth=2.0, alpha=0.9)
    ax_p.plot(t_norm, p2, label="Seller 2 (Rival)", color="#2ca02c", linewidth=2.0, alpha=0.9)
    ax_p.plot(t_norm, ref_p, label="Reference Price (Mean)", color="#7f7f7f", linestyle="--", linewidth=1.5, alpha=0.7)
    ax_p.plot(t_norm, thresh_p, label=f"Flag Threshold (1+{delta})*Mean", color="#d62728", linestyle=":", linewidth=1.8, alpha=0.8)

    # Shade ban intervals for rivals
    is_rival_banned = (b1 > 0) | (b2 > 0)
    in_ban = False
    ban_start = 0
    for idx, banned in enumerate(is_rival_banned):
        if banned and not in_ban:
            in_ban = True
            ban_start = t_norm[idx]
        elif not banned and in_ban:
            in_ban = False
            ax_p.axvspan(ban_start, t_norm[idx], color="#ff9896", alpha=0.25, label="Rival Banned" if "Rival Banned" not in ax_p.get_legend_handles_labels()[1] else "")
    if in_ban:
        ax_p.axvspan(ban_start, t_norm[-1], color="#ff9896", alpha=0.25)

    ax_p.set_ylabel("Price", fontsize=12, fontweight="bold")
    ax_p.set_title(f"Market Timeline: Predatory Price Deflation & Rival Suspension\nRule: {rule_type.name} (delta={delta}) | Weaponization Rate W = {results.weaponization_rate:.3f}", fontsize=14, fontweight="bold")
    ax_p.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=True, fontsize=10)
    ax_p.grid(True, linestyle="--", alpha=0.4)

    # 2. Capital Reserves Timeline
    ax_k.plot(t_norm, k0, label="Seller 0 Capital", color="#1f77b4", linewidth=1.8)
    ax_k.plot(t_norm, k1, label="Seller 1 Capital", color="#ff7f0e", linewidth=1.8)
    ax_k.plot(t_norm, k2, label="Seller 2 Capital", color="#2ca02c", linewidth=1.8)
    ax_k.axhline(0.0, color="#d62728", linestyle="--", linewidth=1.2, label="Bankruptcy Barrier (K=0)")
    ax_k.set_ylabel("Capital Reserves", fontsize=12, fontweight="bold")
    ax_k.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=True, fontsize=10)
    ax_k.grid(True, linestyle="--", alpha=0.4)

    # 3. Active Seller Count
    ax_n.step(t_norm, n_act, color="#9467bd", linewidth=2.0, where="post", label="Active Sellers (N_act)")
    ax_n.set_ylabel("Active Firms", fontsize=12, fontweight="bold")
    ax_n.set_xlabel("Relative Timeline Steps (Periods)", fontsize=12, fontweight="bold")
    ax_n.set_yticks([1, 2, 3])
    ax_n.set_ylim(0.5, 3.5)
    ax_n.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=True, fontsize=10)
    ax_n.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Visual trace plot successfully generated and saved to: {output_path}")


if __name__ == "__main__":
    generate_trace_plot()
