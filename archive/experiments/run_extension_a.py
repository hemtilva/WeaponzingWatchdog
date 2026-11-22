"""Experiment runner for Extension A: Slow-Timescale Adaptive Regulator vs a learning attacker.

A persistent Q-learning attacker (firm 0) prices against honest Nash rivals in a viable,
lethal market. Every macro-epoch the regulator moves delta from the false positives
(induced sanctions) and price dispersion it actually observed. Whether the resulting
delta path cycles, converges, or drifts is an empirical outcome, not built in.
"""

import argparse
import csv
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.extensions.learning_regulator import simulate_bi_level_dynamics
from src.benchmarks.diagnostics import framing_market, market_diagnostics, format_diagnostics
from src.engine.rules import RuleType


def run_extension_a_experiment(
    num_epochs: int = 15,
    steps_per_epoch: int = 20_000,
    pretrain_steps: int = 100_000,
    delta_init: float = 0.10,
    seed: int = 42,
    use_dp: bool = False,
    output_dir: str = "plots",
    data_dir: str = "data",
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    market = framing_market(num_firms=3, lethal=True, threshold_delta=delta_init)
    print("\nEXTENSION A: ADAPTIVE REGULATOR vs LEARNING ATTACKER")
    print(format_diagnostics(market_diagnostics(market, rule_type=RuleType.MEAN_RATIO, rule_param=delta_init)))
    print(f"Epochs: {num_epochs} x {steps_per_epoch} steps (pretrain {pretrain_steps}) | delta_init={delta_init}\n")

    out = simulate_bi_level_dynamics(
        num_epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        delta_init=delta_init,
        seed=seed,
        market=market,
        pretrain_steps=pretrain_steps,
        use_dp=use_dp,
    )
    records = out["epochs"]
    for r in records:
        print(
            f"Epoch {r['epoch']:02d} | delta {r['delta']:.3f} -> {r['new_delta']:.3f} | "
            f"attacker p={r['attacker_mean_price']:.3f} (Nash {r['nash_price']:.3f}) "
            f"profit={r['attacker_mean_profit']:.4f} | W={r['weaponization_rate']:.2f} "
            f"FP/period={r['false_positives_per_period']:.4f} kills/ep={r['kills_per_episode']:.2f}"
        )

    csv_path = Path(data_dir) / "extension_a_cyclic_dynamics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"\n[OK] Dynamics data saved to: {csv_path}")

    epochs = np.array([r["epoch"] for r in records])
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    ax1.plot(epochs, [r["delta"] for r in records], marker="o", linewidth=2.2, label="threshold delta")
    ax1.set_ylabel("delta")
    ax1.set_title("Extension A: adaptive watchdog vs learning attacker (mean-ratio rule, N=3)")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    ax2.plot(epochs, [r["attacker_mean_price"] for r in records], marker="s", label="attacker mean price")
    ax2.axhline(records[0]["nash_price"], color="gray", linestyle="--", label="Nash price")
    ax2.set_ylabel("price")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    ax3.bar(epochs - 0.2, [r["weaponization_rate"] for r in records], width=0.4, label="W")
    ax3.bar(epochs + 0.2, [r["kills_per_episode"] for r in records], width=0.4, label="induced kills / episode")
    ax3.set_xlabel("regulator macro-epoch")
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend()

    plt.tight_layout()
    plot_path = Path(output_dir) / "extension_a_cyclic_dynamics.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"[OK] Plot saved to: {plot_path}\n")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Extension A: adaptive regulator vs learning attacker.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--steps", type=int, default=20_000, help="Market periods per macro-epoch.")
    parser.add_argument("--pretrain", type=int, default=100_000, help="Attacker pretraining steps at delta_init.")
    parser.add_argument("--delta_init", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dp", action="store_true", help="Also compute exact DP gain per epoch (slow without native engine).")
    args = parser.parse_args()

    run_extension_a_experiment(
        num_epochs=args.epochs,
        steps_per_epoch=args.steps,
        pretrain_steps=args.pretrain,
        delta_init=args.delta_init,
        seed=args.seed,
        use_dp=args.dp,
    )
