"""Generate report figures from committed seed-level experiment results."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "plots" / "final_report"
OUT.mkdir(parents=True, exist_ok=True)
original = pd.read_csv(ROOT / "results/Krishs-MacBook-Air/all/study_single.csv")
replication = pd.read_csv(ROOT / "results/report_replication/single_learner.csv")
conditions = ["mean d=0.10 lethal", "mean d=0.20 lethal", "mean d=0.10 non-lethal",
              "median d=0.10 lethal", "cap p=0.60 lethal"]
labels = ["Mean 0.10\nlethal", "Mean 0.20\nlethal", "Mean 0.10\nnon-lethal",
          "Median 0.10\nlethal", "Cap 0.60\nlethal"]
fig, ax = plt.subplots(figsize=(9, 4.3), layout="constrained")
x = np.arange(len(conditions))
for off, df, label, color in [(-.18, original, "Original (20 seeds)", "#235a81"),
                               (.18, replication, "New seeds (5)", "#da8a32")]:
    means = [df.loc[df.tag == c, "kills_per_ep"].mean() for c in conditions]
    ax.bar(x + off, means, width=.36, label=label, color=color)
ax.set_xticks(x, labels)
ax.set_ylabel("Attributed bankruptcies per episode")
ax.set_ylim(0, 2.2)
ax.set_title("One learner against two fixed Nash-price rivals")
ax.legend(frameon=False)
ax.grid(axis="y", alpha=.2)
fig.savefig(OUT / "single_learner.png", dpi=180)
plt.close(fig)

a = pd.read_csv(ROOT / "results/athena/controls/control_cap_noreg.csv")
b = pd.read_csv(ROOT / "results/athena/controls/control_cap_nmin1.csv")
controls = pd.concat([a, b], ignore_index=True)
tags = ["cap p=0.60", "no regulation (cap=10)", "cap p=0.60 Nmin=1"]
fig, ax = plt.subplots(figsize=(7.2, 4), layout="constrained")
ax.bar(range(3), [controls.loc[controls.tag == t, "survivors"].mean() for t in tags],
       color=["#b55646", "#507e71", "#507e71"])
ax.set_xticks(range(3), ["Cap 0.60\nN_min = 2", "Cap 10\n(no effective cap)", "Cap 0.60\nN_min = 1"])
ax.set_ylabel("Mean surviving firms per episode")
ax.set_ylim(0, 3.3)
ax.set_title("All-learning controls (10 seeds per condition)")
ax.grid(axis="y", alpha=.2)
fig.savefig(OUT / "controls.png", dpi=180)
plt.close(fig)

traits = pd.read_csv(ROOT / "results/Krishs-MacBook-Air/all/study_h3.csv")
ethics = traits[traits.trait == "mu_ethics"]
fig, ax = plt.subplots(figsize=(7.2, 4), layout="constrained")
for value, group in ethics.groupby("trait_value"):
    ax.scatter(np.full(len(group), value), group.W, alpha=.4, color="#6389a1", s=26)
means = ethics.groupby("trait_value").W.mean()
ax.plot(means.index, means.values, marker="o", color="#b55646", label="Seed mean")
ax.set_xlabel("Ethics penalty weight")
ax.set_ylabel("Recorded unilateral W")
ax.set_title("Ethics sweep: individual seeds and means")
ax.set_ylim(-.03, 1.03)
ax.grid(alpha=.2)
ax.legend(frameon=False)
fig.savefig(OUT / "ethics.png", dpi=180)
plt.close(fig)
