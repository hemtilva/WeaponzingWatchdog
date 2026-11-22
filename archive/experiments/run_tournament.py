#!/usr/bin/env python3
"""Axelrod-style Evolutionary Strategy Tournament Runner (Option 3).

Evaluates the multi-generation population ecology of heterogeneous sellers
(Predator, Ethical, Altruist, Risk-Averse) under:
1. Vulnerable Watchdog (Mean-Ratio)
2. Robust Watchdog (Median-Ratio)
3. Defended Watchdog (EMA Mean-Ratio + Escrow)

Generates:
- data/evolutionary_tournament_dynamics.csv
- plots/evolutionary_tournament_dynamics.png
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.engine.rules import RuleType
from src.evaluation.tournament import TournamentConfig, EvolutionaryTournament, ARCHETYPE_NAMES


def run_experiment():
    print("=" * 65)
    print("Running Option 3: Evolutionary Strategy Tournament")
    print("=" * 65)

    data_dir = root_dir / "data"
    plots_dir = root_dir / "plots"
    data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    experiments = [
        ("Vulnerable Watchdog (Mean-Ratio)", RuleType.MEAN_RATIO, "shutdown", 0.0),
        ("Robust Watchdog (Median-Ratio)", RuleType.MEDIAN_RATIO, "shutdown", 0.0),
        ("Defended Watchdog (EMA + Escrow)", RuleType.EMA_MEAN_RATIO, "escrow", 0.95),
    ]

    all_records = []
    tournament_histories = {}

    for label, rule_type, pen_mode, beta_val in experiments:
        print(f"\nSimulating Evolutionary Tournament: {label}...")
        cfg = TournamentConfig(
            num_generations=25,
            cohorts_per_gen=6,
            steps_per_match=20_000,
            warmup_steps=0,
            episode_length=500,
            eval_window=5_000,
            num_firms=4,
            mutation_rate=0.02,
            rule_type=rule_type,
            rule_param=0.20,
            penalty_mode=pen_mode,
            ema_beta=beta_val,
            seed=42,
        )
        tournament = EvolutionaryTournament(cfg)
        res = tournament.run()

        tournament_histories[label] = res

        print(f"  -> Dominant Archetype: {res.dominant_archetype}")
        for name, sh in res.final_shares.items():
            print(f"     {name}: {sh * 100:.1f}%")

        for g in range(res.generations + 1):
            row = {"Regime": label, "Generation": g}
            for i, name in enumerate(ARCHETYPE_NAMES):
                row[name] = float(res.history_shares[g, i])
            all_records.append(row)

    # Save to CSV
    df = pd.DataFrame(all_records)
    csv_path = data_dir / "evolutionary_tournament_dynamics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[SAVED] {csv_path}")

    # Plot 3-panel figure: Population trajectory for each regime
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    colors = {
        "Predator": "#e74c3c",    # Red
        "Ethical": "#2ecc71",     # Green
        "Altruist": "#3498db",    # Blue
        "RiskAverse": "#9b59b6",  # Purple
    }

    for idx, (label, _, _, _) in enumerate(experiments):
        ax = axes[idx]
        res = tournament_histories[label]
        gens = np.arange(res.generations + 1)

        for i, name in enumerate(ARCHETYPE_NAMES):
            shares = res.history_shares[:, i] * 100.0
            ax.plot(
                gens,
                shares,
                label=name,
                color=colors[name],
                linewidth=2.5,
                marker="o" if idx == 0 else ("s" if idx == 1 else "^"),
                markersize=5,
            )

        ax.set_title(f"({chr(97 + idx)}) {label}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Generation (Replicator Dynamics)", fontsize=10, fontweight="bold")
        if idx == 0:
            ax.set_ylabel("Population Share (%)", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_ylim(-2, 102)
        ax.axhline(25, color="gray", linestyle=":", alpha=0.7, label="Initial Share (25%)" if idx == 0 else "")
        ax.legend(frameon=True, fontsize=9, loc="upper right")

    plt.tight_layout()
    plot_path = plots_dir / "evolutionary_tournament_dynamics.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[SAVED] {plot_path}")
    print("\nOption 3 Evolutionary Strategy Tournament Complete!")


if __name__ == "__main__":
    run_experiment()
