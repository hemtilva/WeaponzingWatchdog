"""Experiment runner for Extension A: Slow-Timescale Adaptive Regulator and Adversarial Attacker.

Demonstrates cyclic policy dynamics:
1. Tight watchdog threshold triggers innocent false-positive bans and creates framing surface.
2. Adversarial attacker exploits relative rule to induce rival bans.
3. Watchdog adapts over macro-epochs, loosening threshold delta to curb wrongful suspensions.
4. Exploitability drops, prompting the watchdog to tighten delta again to protect consumer surplus.
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

from src.extensions.learning_regulator import (
    AdaptiveRegulatorConfig,
    SlowTimescaleRegulator,
    evaluate_checkpoint_exploitability,
    train_adversarial_attacker,
)
from src.engine.rules import RuleType


def run_extension_a_experiment(
    num_epochs: int = 14,
    steps_per_epoch: int = 1500,
    delta_init: float = 0.18,
    seed: int = 42,
    evaluate_empirical_attacker: bool = True,
    output_dir: str = "plots",
    data_dir: str = "data",
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    print("\n=======================================================")
    print("STARTING EXTENSION A: ADAPTIVE REGULATOR & ADVERSARY")
    print(f"Macro-Epochs: {num_epochs} | Steps/Epoch: {steps_per_epoch} | Initial delta: {delta_init:.2f}")
    print("=======================================================\n")

    reg_cfg = AdaptiveRegulatorConfig(
        macro_epoch=steps_per_epoch,
        delta_init=delta_init,
        delta_min=0.08,
        delta_max=0.55,
        learning_rate=0.04,
        audit_cost_kappa=0.05,
        fp_penalty_eta=0.30,
    )
    regulator = SlowTimescaleRegulator(reg_cfg)
    rng = np.random.default_rng(seed)

    records = []
    prices_base = np.array([1.5, 1.8, 2.0], dtype=np.float64)

    for ep in range(num_epochs):
        cur_delta = regulator.current_delta

        # 1. Exact Dynamic Programming Exploitability at this frozen checkpoint
        dp_res = evaluate_checkpoint_exploitability(frozen_delta=cur_delta, t_ban=3, fixed_cost=0.20, k0_val=0.60)

        # 2. Optional empirical Q-learning attacker against this frozen checkpoint
        empirical_w = 0.0
        attacker_p = dp_res["optimal_price"]
        if evaluate_empirical_attacker and ep % 2 == 0:
            att_res = train_adversarial_attacker(
                frozen_delta=cur_delta,
                steps=min(12_000, steps_per_epoch * 4),
                seed=seed + ep,
            )
            empirical_w = att_res["weaponization_rate"]
            attacker_p = att_res["attacker_mean_price"]

        # 3. Simulate marketplace observations under this checkpoint
        for _ in range(steps_per_epoch):
            noise = rng.normal(0.0, 0.04, size=3)
            p_noisy = np.clip(prices_base + noise, 1.0, 2.5)
            r_t = np.mean(p_noisy)
            flags = p_noisy > (1.0 + cur_delta) * r_t
            fp = int(np.sum(flags)) if dp_res["is_framing"] or (cur_delta < 0.20) else 0

            regulator.observe_period(
                consumer_surplus=1.25,
                num_audited=1,
                prices=p_noisy,
                false_positive_count=fp,
            )

        new_delta = regulator.step_macro_epoch()
        reg_obj = regulator.history_objectives[-1] if regulator.history_objectives else 0.0

        records.append({
            "epoch": ep,
            "delta": cur_delta,
            "new_delta": new_delta,
            "regulator_objective": reg_obj,
            "exploitability_gain": dp_res["exploitability_gain"],
            "is_framing": dp_res["is_framing"],
            "optimal_price": dp_res["optimal_price"],
            "nash_price": dp_res["nash_price"],
            "empirical_weaponization_rate": empirical_w,
            "empirical_attacker_price": attacker_p,
        })

        status_str = "ATTACK / FRAMING" if dp_res["is_framing"] else "STABLE PRICING"
        print(f"Epoch {ep:02d} | delta: {cur_delta:.3f} -> {new_delta:.3f} | Gain: {dp_res['exploitability_gain']:+.4f} | State: {status_str}")

    df = pd.DataFrame(records)
    csv_path = Path(data_dir) / "extension_a_cyclic_dynamics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[OK] Dynamics data saved to: {csv_path}")

    # Generate Visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    epochs = df["epoch"].values
    deltas = df["delta"].values
    gains = df["exploitability_gain"].values

    # Panel 1: Watchdog Adaptation
    ax1.plot(epochs, deltas, color="#1f77b4", marker="o", linewidth=2.5, label="Watchdog Threshold (delta)")
    ax1.axhline(y=0.50, color="gray", linestyle="--", alpha=0.7, label="Theoretical Bound (1/(N-1) = 0.50)")
    ax1.axhline(y=0.20, color="red", linestyle=":", alpha=0.6, label="Initial Tight Threshold (0.20)")
    ax1.set_ylabel("Threshold delta", fontsize=12, fontweight="bold")
    ax1.set_title("Extension A: Watchdog Threshold Adaptation & Adversarial Exploitability Cycles", fontsize=13, fontweight="bold", pad=12)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", frameon=True)

    # Panel 2: Adversarial Best-Response Exploitability Gain
    colors = ["#d62728" if f else "#2ca02c" for f in df["is_framing"].values]
    ax2.bar(epochs, gains, color=colors, alpha=0.75, width=0.55, edgecolor="black", label="Exploitability Gain E_1")
    ax2.set_xlabel("Regulator Macro-Epoch", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Exploitability Gain E_1", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)

    # Annotate legend for exploitability
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#d62728", edgecolor="black", label="Framing Active (opt_p < p^N)"),
        Patch(facecolor="#2ca02c", edgecolor="black", label="Framing Unprofitable (opt_p >= p^N)"),
    ]
    ax2.legend(handles=legend_elements, loc="upper right", frameon=True)

    plt.tight_layout()
    plot_path = Path(output_dir) / "extension_a_cyclic_dynamics.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[OK] High-resolution visualization saved to: {plot_path}\n")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Extension A: Adaptive Regulator & Best-Response Attacker.")
    parser.add_argument("--epochs", type=int, default=14, help="Number of macro-epochs.")
    parser.add_argument("--steps", type=int, default=1500, help="Market periods per macro-epoch.")
    parser.add_argument("--delta_init", type=float, default=0.18, help="Initial threshold delta.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--no_empirical", action="store_true", help="Skip empirical Q-attacker for faster run.")
    args = parser.parse_args()

    run_extension_a_experiment(
        num_epochs=args.epochs,
        steps_per_epoch=args.steps,
        delta_init=args.delta_init,
        seed=args.seed,
        evaluate_empirical_attacker=not args.no_empirical,
    )
