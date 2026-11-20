"""Generate every figure in the LaTeX paper (paper/figures/) from committed result CSVs.

Usage: python scripts/make_paper_figures.py
"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Validated categorical slots 1-3 (light mode), text and grid tokens.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "lines.linewidth": 2, "legend.frameon": False, "savefig.bbox": "tight",
})


def se(x):
    x = np.asarray(x, dtype=float)
    return x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def bar(ax, x, h, err, color, label=None, width=0.36):
    ax.bar(x, h, width=width, color=color, label=label, edgecolor="white", linewidth=1)
    ax.errorbar(x, h, yerr=err, fmt="none", ecolor=INK, elinewidth=1, capsize=2)


# ------------------------------------------------------------------ Fig: single learner
def fig_single():
    orig = pd.read_csv(RES / "Krishs-MacBook-Air/all/study_single.csv")
    rep = pd.read_csv(RES / "report_replication/single_learner.csv")
    conds = ["mean d=0.10 lethal", "mean d=0.20 lethal", "mean d=0.10 non-lethal",
             "median d=0.10 lethal", "cap p=0.60 lethal"]
    labels = ["Mean\n$\\delta$=0.10", "Mean\n$\\delta$=0.20", "Mean $\\delta$=0.10\nnon-lethal",
              "Median\n$\\delta$=0.10", "Cap\n0.60"]
    fig, ax = plt.subplots(figsize=(5.2, 2.4))
    x = np.arange(len(conds))
    for off, df, lab, col in [(-0.19, orig, "Original seeds (20)", BLUE),
                              (0.19, rep, "New seeds (5)", ORANGE)]:
        g = [df.loc[df.tag == c, "kills_per_ep"] for c in conds]
        bar(ax, x + off, [v.mean() for v in g], [se(v) for v in g], col, lab)
    for xi in (3, 4):
        ax.text(xi, 0.05, "0", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Attributed rival\nbankruptcies / episode")
    ax.set_ylim(0, 2.3)
    ax.legend(loc="upper right")
    fig.savefig(OUT / "single_learner.pdf")
    fig.savefig(OUT / "single_learner.png", dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ Fig: training length
def fig_training():
    rob = pd.read_csv(RES / "Krishs-MacBook-Air/all/study_robust.csv")
    rows = [(1e7, rob[rob.tag.isin(["delta=0.1 mean", "delta=0.1 median"])])]
    for s, v in [("100M", 1e8), ("300M", 3e8), ("1000M", 1e9)]:
        rows.append((v, pd.read_csv(RES / f"training_length/all_learner_{s}.csv")))
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.3))
    for ax, col, ylab in [(axes[0], "kills_per_ep", "Attributed bankruptcies\n/ episode"),
                          (axes[1], "W", "Weaponization rate $W$")]:
        for rule, name, c, mk in [("MEAN_RATIO", "Mean rule", BLUE, "o"),
                                  ("MEDIAN_RATIO", "Median rule", ORANGE, "s")]:
            xs = [s for s, _ in rows]
            ys = [d.loc[d.rule == rule, col].mean() for _, d in rows]
            es = [se(d.loc[d.rule == rule, col]) for _, d in rows]
            ax.errorbar(xs, ys, yerr=es, color=c, marker=mk, ms=5, capsize=2, label=name,
                        markeredgecolor="white", markeredgewidth=1)
        ax.set_xscale("log")
        ax.set_xticks([1e7, 1e8, 3e8, 1e9], ["10M", "100M", "300M", "1B"])
        ax.minorticks_off()
        ax.set_xlabel("Training steps")
        ax.set_ylabel(ylab)
        ax.set_ylim(bottom=0)
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "training_length.pdf")
    fig.savefig(OUT / "training_length.png", dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ Fig: cap trace
def fig_trace(seed=6000, steps=25):
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.3), sharey=True)
    for ax, nmin in zip(axes, (2, 1)):
        t = pd.read_csv(RES / f"cap_trace/trace_nmin{nmin}.csv")
        t = t[(t.seed == seed) & (t.episode == 0) & (t.ep_step < steps)]
        for i, c, mk in [(0, BLUE, "o"), (1, ORANGE, "s"), (2, AQUA, "^")]:
            alive = t[f"K{i}"] > 0
            ax.plot(t.ep_step[alive], t[f"p{i}"][alive], color=c, marker=mk, ms=4,
                    lw=1.5, label=f"Firm {i + 1}", markeredgecolor="white", markeredgewidth=0.6)
            banned = t[(t[f"ban{i}"] > 0) & alive]
            if len(banned):
                ax.scatter(banned.ep_step, banned[f"p{i}"], s=60, facecolors="none",
                           edgecolors=c, linewidths=1.2, zorder=4)
            dead = t[~alive]
            if len(dead):
                ax.scatter(dead.ep_step.iloc[:1], dead[f"p{i}"].iloc[:1], marker="x", s=55,
                           color=c, linewidths=2, zorder=5)
        ax.axhline(0.6, color=INK2, ls="--", lw=1)
        ax.text(steps - 0.5, 0.63, "cap 0.60", ha="right", fontsize=7, color=INK2)
        ax.set_title(f"$N_{{\\min}}={nmin}$" + (" (audits stop at 1 firm)" if nmin == 2 else " (audits never stop)"))
        ax.set_xlabel("Period within episode")
    axes[0].set_ylabel("Price")
    axes[0].set_ylim(-0.05, 1.65)
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "cap_trace.pdf")
    fig.savefig(OUT / "cap_trace.png", dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ Fig: traits
def fig_traits():
    path300 = RES / "traits_300M/study_h3_300M.csv"
    sets = [("10M steps", pd.read_csv(RES / "Krishs-MacBook-Air/all/study_h3.csv"), ORANGE, "s")]
    if path300.exists():
        sets.append(("300M steps", pd.read_csv(path300), BLUE, "o"))
    traits = [("mu_ethics", "Ethics weight $\\mu$"), ("lambda_altruism", "Altruism $\\lambda$"),
              ("alpha", "Risk aversion $\\alpha$")]
    fig, axes = plt.subplots(1, 3, figsize=(6.4, 2.2), sharey=True)
    for ax, (tr, lab) in zip(axes, traits):
        for name, d, c, mk in sets:
            d = d[d.trait == tr]
            g = d.groupby("trait_value").kills_per_ep
            vals = sorted(d.trait_value.unique())
            pos = np.arange(len(vals))
            ax.errorbar(pos, g.mean().loc[vals], yerr=g.apply(se).loc[vals], color=c, marker=mk,
                        ms=4, capsize=2, label=name, markeredgecolor="white", markeredgewidth=0.6)
            ax.set_xticks(pos, [f"{v:g}" for v in vals])
        ax.set_xlabel(lab)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("Attributed bankruptcies\n/ episode")
    axes[0].legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "traits.pdf")
    fig.savefig(OUT / "traits.png", dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ Fig: phase diagram (H2)
def fig_phase():
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle
    d = pd.read_csv(RES / "long_300M/study_h2_300M.csv")
    cmap = LinearSegmentedColormap.from_list("seq", ["#f4f7fc", "#9cc0ec", BLUE, "#123e75"])
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7), sharey=True)
    for ax, rule, title in zip(axes, ("mean", "median"), ("Mean rule", "Median rule")):
        piv = d[d.rule_name == rule].pivot_table(index="ratio", columns="T_ban", values="kills_per_ep")
        im = ax.imshow(piv.values, cmap=cmap, vmin=0, vmax=1.5, origin="lower", aspect="auto")
        ax.grid(False)
        ax.set_xticks(range(len(piv.columns)), piv.columns)
        ax.set_yticks(range(len(piv.index)), piv.index)
        for i, k in enumerate(piv.index):
            for j, t in enumerate(piv.columns):
                v = piv.values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.8 else INK)
                if k <= t:   # one ban is lethal: K0 <= T_ban * Phi
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, lw=1.2, ec=ORANGE))
        ax.set_title(title)
        ax.set_xlabel("Ban length $T_{\\mathrm{ban}}$")
    axes[0].set_ylabel("Capital buffer $K_0/\\Phi$")
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cb.set_label("Attributed bankruptcies / episode", fontsize=8)
    cb.outline.set_visible(False)
    fig.savefig(OUT / "phase_diagram.pdf")
    fig.savefig(OUT / "phase_diagram.png", dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ Fig: sensitivity
def fig_sensitivity():
    d = pd.read_csv(RES / "sensitivity/all.csv")
    d["cond"] = d.tag.str.rsplit(" ", n=1).str[0]
    order = list(dict.fromkeys(d.cond))
    labels = {"a0=1.0": "$a_0=1.0$", "a0=1.5": "$a_0=1.5$", "sigma=0.15": "$\\sigma=0.15$",
              "sigma=0.25": "$\\sigma=0.25$", "sigma=0.40": "$\\sigma=0.40$", "N=3 M=7": "$N=3$",
              "N=4 M=7": "$N=4$", "N=5 M=7": "$N=5$", "uncapped capital": "uncapped $K$",
              "eta=0.05": "$\\eta=0.05$", "eta=0.20": "$\\eta=0.20$", "gamma=0.90": "$\\gamma=0.90$",
              "gamma=0.99": "$\\gamma=0.99$"}
    fig, ax = plt.subplots(figsize=(6.2, 2.4))
    x = np.arange(len(order))
    for off, rule, c, mk, lab in [(-0.12, "mean", BLUE, "o", "Mean rule"), (0.12, "median", ORANGE, "s", "Median rule")]:
        g = [d[(d.cond == k) & (d.rule_name == rule)].kills_per_ep for k in order]
        ax.errorbar(x + off, [v.mean() for v in g], yerr=[se(v) for v in g], fmt=mk, color=c, ms=5,
                    capsize=2, label=lab, markeredgecolor="white", markeredgewidth=0.6)
    ax.set_xticks(x, [labels[k] for k in order], rotation=45, ha="right")
    ax.set_ylabel("Attributed bankruptcies\n/ episode")
    ax.set_ylim(bottom=0)
    ax.grid(axis="x", visible=False)
    axes_of = [k.split("=")[0] if "=" in k else k for k in order]
    for i in range(1, len(order)):
        if axes_of[i] != axes_of[i - 1]:
            ax.axvline(i - 0.5, color=GRID, lw=1)
    ax.legend(loc="upper right", ncol=2)
    fig.savefig(OUT / "sensitivity.pdf")
    fig.savefig(OUT / "sensitivity.png", dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ Fig: two deep learners
def fig_two_learners():
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.4))
    ns = [3, 5, 8]
    x = np.arange(len(ns))
    for off, rule, c, lab in [(-0.19, "mean", BLUE, "Mean rule"), (0.19, "median", ORANGE, "Median rule")]:
        surv, honest = [], []
        for n in ns:
            d = pd.read_csv(RES / f"dqn_multi/n{n}_full.csv")
            g = d[d.rule == rule]
            surv.append((g.survivors_per_ep.mean(), se(g.survivors_per_ep)))
            honest.append((g.fixed_rival_bankruptcies_per_ep.mean() / (n - 2),
                           se(g.fixed_rival_bankruptcies_per_ep / (n - 2))))
        bar(axes[0], x + off, [m for m, _ in honest], [e for _, e in honest], c, lab)
        bar(axes[1], x + off, [m for m, _ in surv], [e for _, e in surv], c, lab)
        for xi, (m, _) in zip(x + off, honest):
            if m == 0:
                axes[0].text(xi, 0.02, "0", ha="center", fontsize=8, color=INK2)
    for i, n in enumerate(ns):
        axes[1].hlines(n, i - 0.4, i + 0.4, colors=INK2, linestyles=":", lw=1)
    axes[0].set_ylabel("Share of honest sellers\nbankrupted per episode")
    axes[0].set_ylim(0, 1.1)
    axes[1].set_ylabel("Survivors per episode")
    axes[1].set_ylim(0, 8.8)
    for ax in axes:
        ax.set_xticks(x, [f"$N={n}$" for n in ns])
    axes[1].text(2.42, 8.15, "all sellers", fontsize=7, color=INK2, ha="right")
    axes[1].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "two_learners.pdf")
    fig.savefig(OUT / "two_learners.png", dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ Fig: the attack + training length
def fig_attack_training():
    t = pd.read_csv(RES / "single_trace/trace.csv")
    t = t[t.t <= 20]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 1.95), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    for i, c, mk, lab in [(0, BLUE, "o", "Learner"), (1, ORANGE, "s", "Rival 1 (Nash)"), (2, AQUA, "^", "Rival 2 (Nash)")]:
        alive = t[f"K{i}"] > 0
        ax.plot(t.t[alive], t[f"p{i}"][alive], color=c, marker=mk, ms=3.5, lw=1.4, label=lab,
                markeredgecolor="white", markeredgewidth=0.5)
        banned = t[(t[f"ban{i}"] > 0) & alive]
        if len(banned):
            ax.scatter(banned.t, banned[f"p{i}"], s=55, facecolors="none", edgecolors=c, linewidths=1.1, zorder=4)
        dead = t[~alive]
        if len(dead):
            ax.scatter(dead.t.iloc[:1], dead[f"p{i}"].iloc[:1], marker="x", s=45, color=c, linewidths=1.8, zorder=5)
    ax.axhline(0.30, color=INK2, ls=":", lw=1)
    ax.text(20, 0.36, "Nash 0.30", ha="right", fontsize=7, color=INK2)
    ax.set_xlabel("Period within episode")
    ax.set_ylabel("Price")
    ax.set_ylim(-0.08, 1.65)
    ax.set_title("(a) A learned framing attack (mean rule)")
    ax.legend(loc="center right", fontsize=7)

    ax = axes[1]
    rob = pd.read_csv(RES / "Krishs-MacBook-Air/all/study_robust.csv")
    rows = [(1e7, rob[rob.tag.isin(["delta=0.1 mean", "delta=0.1 median"])])]
    for s_, v in [("100M", 1e8), ("300M", 3e8), ("1000M", 1e9)]:
        rows.append((v, pd.read_csv(RES / f"training_length/all_learner_{s_}.csv")))
    for rule, name, c, mk in [("MEAN_RATIO", "Mean rule", BLUE, "o"), ("MEDIAN_RATIO", "Median rule", ORANGE, "s")]:
        ax.errorbar([s_ for s_, _ in rows], [d.loc[d.rule == rule, "kills_per_ep"].mean() for _, d in rows],
                    yerr=[se(d.loc[d.rule == rule, "kills_per_ep"]) for _, d in rows], color=c, marker=mk, ms=4,
                    capsize=2, label=name, markeredgecolor="white", markeredgewidth=0.6)
    ax.set_xscale("log")
    ax.set_xticks([1e7, 1e8, 3e8, 1e9], ["10M", "100M", "300M", "1B"])
    ax.minorticks_off()
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Attributed bankr. / ep.")
    ax.set_ylim(bottom=0)
    ax.set_title("(b) All sellers learning")
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "attack_training.pdf")
    fig.savefig(OUT / "attack_training.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig_single()
    fig_training()
    fig_trace()
    fig_traits()
    fig_phase()
    fig_sensitivity()
    fig_two_learners()
    fig_attack_training()
    print(f"figures written to {OUT}")
