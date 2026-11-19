"""Main study driver (native C++ backend). One subcommand per study:

  single   One Q-learner vs honest Nash rivals (H1, H2 against non-adaptive victims)
  h2       Phase diagram: K0/Phi x T_ban, all firms learning, mean vs median (H2)
  h3       Trait sweeps: ethics mu, altruism lambda, risk aversion alpha (H3)
  robust   Mean vs median across delta, price-grid size M and audit budget B (H1 robustness)
  n4long   N = 4 with long training, N_min 3 vs 2 (convergence check)

Defaults: N = 3, all firms learning, mean rule delta = 0.10, N_min = 2 (watchdog keeps
auditing 2-firm markets), viable market (Nash profit 0.10 > Phi = 0.05), lethal bans
(K0 = 0.15, T_ban = 4). Outcome = greedy (frozen) play after training; see common.py.

  python experiments/run_study.py single [--seeds 20]
  python experiments/run_study.py all          # every study in sequence
  python experiments/run_study.py h3 --quick   # smoke test
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Job, run_jobs, save_csv, group_stats, ROOT  # noqa: E402

from src.engine.rules import RuleType  # noqa: E402

MEAN, MEDIAN, CAP = int(RuleType.MEAN_RATIO), int(RuleType.MEDIAN_RATIO), int(RuleType.PRICE_CAP)


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _mwu(a, b):
    from scipy.stats import mannwhitneyu
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.all(a == a[0]) and np.all(b == b[0]) and a[0] == b[0]:
        return 1.0
    return float(mannwhitneyu(a, b, alternative="greater").pvalue)


def _vals(rows, tag, metric="kills_per_ep"):
    return [float(r[metric]) for r in rows if r["tag"] == tag]


def _table(rows, metrics=("kills_per_ep", "self_bankr_per_ep", "W", "W_joint", "delta", "firm0_profit")):
    tags = list(dict.fromkeys(r["tag"] for r in rows))
    head = f"{'config':<34}" + "".join(f"{m:>18}" for m in metrics)
    print("\n" + "=" * len(head)); print(head); print("-" * len(head))
    for t in tags:
        cells = []
        for m in metrics:
            v = np.array(_vals(rows, t, m))
            cells.append(f"{v.mean():.3f}+-{v.std():.3f}")
        print(f"{t:<34}" + "".join(f"{c:>18}" for c in cells))
    print("=" * len(head))


def _save_fig(fig, name):
    path = ROOT / "plots" / name
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"[SAVED] {path}")


# ---------------------------------------------------------------------------- single
def study_single(a):
    honest = ["q", "nash", "nash"]
    cfgs = [
        ("mean d=0.10 lethal", dict(rule=MEAN, rule_param=0.10, lethal=True)),
        ("mean d=0.20 lethal", dict(rule=MEAN, rule_param=0.20, lethal=True)),
        ("mean d=0.10 non-lethal", dict(rule=MEAN, rule_param=0.10, lethal=False)),
        ("median d=0.10 lethal", dict(rule=MEDIAN, rule_param=0.10, lethal=True)),
        ("cap p=0.60 lethal", dict(rule=CAP, rule_param=0.60, lethal=True)),
    ]
    jobs = [Job(tag=t, seed=3000 + s, policies=honest, train_steps=a.steps or 2_000_000,
                eval_steps=a.eval_steps, **kw)
            for t, kw in cfgs for s in range(a.seeds)]
    rows = run_jobs(jobs, a.workers)
    save_csv(rows, f"study_single{a.sfx}.csv")
    _table(rows, ("kills_by_q_per_ep", "W", "firm0_price", "firm0_profit", "rival_profit"))
    m = _vals(rows, "mean d=0.10 lethal", "kills_by_q_per_ep")
    for other in ("median d=0.10 lethal", "cap p=0.60 lethal", "mean d=0.10 non-lethal"):
        print(f"  mean lethal > {other:<24} p = {_mwu(m, _vals(rows, other, 'kills_by_q_per_ep')):.2e}")
    nash = float(rows[0]["nash_profit"])
    p = np.array(_vals(rows, "mean d=0.10 lethal", "firm0_profit"))
    print(f"  learner profit under mean/lethal {p.mean():.4f} vs Nash {nash:.4f}; "
          f"seeds above Nash: {(p > nash).sum()}/{len(p)}")

    plt = _plt()
    tags = [t for t, _ in cfgs]
    k = [group_stats(rows, "tag", "kills_by_q_per_ep")[t] for t in tags]
    pr = [group_stats(rows, "tag", "firm0_profit")[t] for t in tags]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.bar(tags, [x[0] for x in k], yerr=[x[1] for x in k], capsize=4, color="#c0392b")
    ax1.set_ylabel("rival bankruptcies caused per episode")
    ax1.set_title(f"Single learner vs honest rivals ({a.seeds} seeds)")
    ax2.bar(tags, [x[0] for x in pr], yerr=[x[1] for x in pr], capsize=4, color="#2980b9")
    ax2.axhline(nash, color="gray", ls="--", label="honest Nash profit")
    ax2.set_ylabel("learner profit per period")
    ax2.legend()
    for ax in (ax1, ax2):
        ax.tick_params(axis="x", rotation=25)
        ax.grid(True, axis="y", ls="--", alpha=0.5)
    _save_fig(fig, f"study_single{a.sfx}.png")


# ---------------------------------------------------------------------------- h2
def study_h2(a):
    phi = 0.05
    ratios = [1, 2, 3, 4, 6, 8]            # K0 / Phi
    tbans = [1, 2, 3, 4, 6]
    jobs = []
    for rule, rname in ((MEAN, "mean"), (MEDIAN, "median")):
        for r in ratios:
            for tb in tbans:
                tag = f"{rname} K0/Phi={r} T={tb}"
                jobs += [Job(tag=tag, seed=4000 + s, rule=rule, initial_capital=r * phi, fixed_cost=phi,
                             ban_duration=tb, train_steps=a.steps or 10_000_000, eval_steps=a.eval_steps,
                             extra=dict(rule_name=rname, ratio=r))
                         for s in range(a.seeds)]
    rows = run_jobs(jobs, a.workers)
    save_csv(rows, f"study_h2{a.sfx}.csv")

    from scipy.stats import spearmanr
    print("\nH2: kills/episode (mean rule), rows K0/Phi, cols T_ban")
    for rname in ("mean", "median"):
        print(f"[{rname}]  T_ban: " + " ".join(f"{t:>6}" for t in tbans))
        for r in ratios:
            cells = [np.mean(_vals(rows, f"{rname} K0/Phi={r} T={tb}")) for tb in tbans]
            print(f"  K0/Phi={r:<2}        " + " ".join(f"{c:6.2f}" for c in cells))
    mrows = [r for r in rows if r["rule_name"] == "mean"]
    leth = [r["kills_per_ep"] for r in mrows if r["T_ban"] >= r["ratio"]]
    sub = [r["kills_per_ep"] for r in mrows if r["T_ban"] < r["ratio"]]
    print(f"  mean rule: lethal cells {np.mean(leth):.3f} vs sub-lethal {np.mean(sub):.3f}, "
          f"p = {_mwu(leth, sub):.2e}")
    rho, p = spearmanr([r["T_ban"] * r["phi"] / r["K0"] for r in mrows], [r["kills_per_ep"] for r in mrows])
    print(f"  Spearman(kills, T_ban*Phi/K0) = {rho:+.3f} (p = {p:.2e})")

    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    vmax = max(np.mean(_vals(rows, t)) for t in dict.fromkeys(r["tag"] for r in rows))
    for ax, rname in zip(axes, ("mean", "median")):
        grid = np.array([[np.mean(_vals(rows, f"{rname} K0/Phi={r} T={tb}")) for tb in tbans] for r in ratios])
        im = ax.imshow(grid, origin="lower", cmap="Reds", vmin=0, vmax=max(vmax, 1e-9), aspect="auto")
        ax.set_xticks(range(len(tbans)), tbans)
        ax.set_yticks(range(len(ratios)), ratios)
        ax.set_xlabel("ban duration T_ban")
        ax.set_title(f"{rname} rule: induced kills / episode")
        for i in range(len(ratios)):
            for j in range(len(tbans)):
                ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=8)
        # lethal boundary T_ban * Phi >= K0  <=>  T_ban >= K0/Phi
        xs = [j for j, tb in enumerate(tbans) for i, r in enumerate(ratios) if tb == r]
        ys = [i for j, tb in enumerate(tbans) for i, r in enumerate(ratios) if tb == r]
        ax.plot(xs, ys, "k--", lw=1.2, label="T_ban = K0/Phi")
        ax.legend(loc="upper left", fontsize=8)
    axes[0].set_ylabel("capital buffer K0 / Phi")
    fig.colorbar(im, ax=axes, shrink=0.8)
    _save_fig(fig, f"study_h2{a.sfx}.png")


# ---------------------------------------------------------------------------- h3
H3_SWEEPS = {
    "mu_ethics": [0.0, 0.01, 0.03, 0.1, 0.3, 1.0],
    "lambda_altruism": [0.0, 0.1, 0.25, 0.5, 1.0],
    "alpha": [0.0, 0.5, 1.0, 1.5, 2.0],
}


def study_h3(a):
    jobs = []
    for trait, values in H3_SWEEPS.items():
        for v in values:
            jobs += [Job(tag=f"{trait}={v}", seed=5000 + s, traits=[{trait: v}] * 3,
                         train_steps=a.steps or 10_000_000, eval_steps=a.eval_steps,
                         extra=dict(trait=trait, trait_value=v))
                     for s in range(a.seeds)]
    rows = run_jobs(jobs, a.workers)
    save_csv(rows, f"study_h3{a.sfx}.csv")
    _table(rows, ("kills_per_ep", "W", "W_joint", "delta", "mean_price"))

    from scipy.stats import spearmanr
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    for ax, (trait, values) in zip(axes, H3_SWEEPS.items()):
        tr_rows = [r for r in rows if r["trait"] == trait]
        rho, p = spearmanr([r["trait_value"] for r in tr_rows], [r["kills_per_ep"] for r in tr_rows])
        print(f"  H3 {trait:<16} Spearman rho = {rho:+.3f}  p = {p:.2e}")
        means = [np.mean(_vals(rows, f"{trait}={v}")) for v in values]
        sds = [np.std(_vals(rows, f"{trait}={v}")) for v in values]
        xs = range(len(values))
        ax.errorbar(xs, means, yerr=sds, marker="o", capsize=4, color="#c0392b")
        ax.set_xticks(list(xs), [str(v) for v in values])
        ax.set_xlabel(trait)
        ax.set_title(f"{trait}: rho={rho:+.2f}, p={p:.1e}")
        ax.grid(True, ls="--", alpha=0.5)
    axes[0].set_ylabel("induced kills / episode (all firms learning)")
    _save_fig(fig, f"study_h3{a.sfx}.png")


# ---------------------------------------------------------------------------- robust
def study_robust(a):
    steps = a.steps or 10_000_000
    jobs = []
    for d in (0.05, 0.10, 0.15, 0.20, 0.30):
        for rule, rn in ((MEAN, "mean"), (MEDIAN, "median")):
            jobs += [Job(tag=f"delta={d} {rn}", seed=6000 + s, rule=rule, rule_param=d, train_steps=steps,
                         eval_steps=a.eval_steps, extra=dict(axis="delta", value=d, rule_name=rn))
                     for s in range(a.seeds)]
    for M in (7, 15):
        for rule, rn in ((MEAN, "mean"), (MEDIAN, "median")):
            jobs += [Job(tag=f"M={M} {rn}", seed=6000 + s, rule=rule, price_levels=M, train_steps=steps,
                         eval_steps=a.eval_steps, extra=dict(axis="M", value=M, rule_name=rn))
                     for s in range(a.seeds)]
    for B in (2,):
        for rule, rn in ((MEAN, "mean"), (MEDIAN, "median")):
            jobs += [Job(tag=f"B={B} {rn}", seed=6000 + s, rule=rule, audit_budget=B, train_steps=steps,
                         eval_steps=a.eval_steps, extra=dict(axis="B", value=B, rule_name=rn))
                     for s in range(a.seeds)]
    rows = run_jobs(jobs, a.workers)
    save_csv(rows, f"study_robust{a.sfx}.csv")
    _table(rows, ("kills_per_ep", "W", "W_joint", "delta", "framing_feasible"))
    print("\nH1 robustness (mean > median, induced kills/episode):")
    for t in dict.fromkeys(r["tag"].rsplit(" ", 1)[0] for r in rows):
        m, md = _vals(rows, f"{t} mean"), _vals(rows, f"{t} median")
        print(f"  {t:<12} mean {np.mean(m):.3f} vs median {np.mean(md):.3f}   p = {_mwu(m, md):.2e}")

    plt = _plt()
    fig, ax = plt.subplots(figsize=(11, 4.5))
    labels = list(dict.fromkeys(r["tag"].rsplit(" ", 1)[0] for r in rows))
    x = np.arange(len(labels))
    for off, rn, col in ((-0.2, "mean", "#c0392b"), (0.2, "median", "#2980b9")):
        st = [np.mean(_vals(rows, f"{l} {rn}")) for l in labels]
        sd = [np.std(_vals(rows, f"{l} {rn}")) for l in labels]
        ax.bar(x + off, st, width=0.4, yerr=sd, capsize=3, color=col, label=rn)
    ax.set_xticks(x, labels)
    ax.set_ylabel("induced kills / episode")
    ax.set_title("H1 robustness: mean vs median (all firms learning, N=3)")
    ax.legend()
    ax.grid(True, axis="y", ls="--", alpha=0.5)
    _save_fig(fig, f"study_robust{a.sfx}.png")


# ---------------------------------------------------------------------------- n4long
def study_n4long(a):
    jobs = []
    for n_min in (3, 2):
        for rule, rn in ((MEAN, "mean"), (MEDIAN, "median")):
            jobs += [Job(tag=f"N=4 {rn} Nmin={n_min}", seed=7000 + s, num_firms=4, rule=rule, n_min=n_min,
                         train_steps=a.steps or 300_000_000, eval_steps=a.eval_steps,
                         extra=dict(rule_name=rn))
                     for s in range(a.seeds)]
    rows = run_jobs(jobs, a.workers)
    save_csv(rows, f"study_n4long{a.sfx}.csv")
    _table(rows, ("kills_per_ep", "self_bankr_per_ep", "W_joint", "delta", "survivors", "coverage"))
    for n_min in (3, 2):
        m, md = _vals(rows, f"N=4 mean Nmin={n_min}"), _vals(rows, f"N=4 median Nmin={n_min}")
        print(f"  N_min={n_min}: mean {np.mean(m):.3f} vs median {np.mean(md):.3f}  p = {_mwu(m, md):.2e}")

    plt = _plt()
    tags = list(dict.fromkeys(r["tag"] for r in rows))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    k = [group_stats(rows, "tag", "kills_per_ep")[t] for t in tags]
    s = [group_stats(rows, "tag", "self_bankr_per_ep")[t] for t in tags]
    x = np.arange(len(tags))
    ax.bar(x - 0.2, [v[0] for v in k], 0.4, yerr=[v[1] for v in k], capsize=3, color="#c0392b", label="framed")
    ax.bar(x + 0.2, [v[0] for v in s], 0.4, yerr=[v[1] for v in s], capsize=3, color="#7f8c8d",
           label="self-inflicted")
    ax.set_xticks(x, tags)
    ax.set_ylabel("bankruptcies / episode")
    ax.set_title("N=4, long training")
    ax.legend()
    ax.grid(True, axis="y", ls="--", alpha=0.5)
    _save_fig(fig, f"study_n4long{a.sfx}.png")


STUDIES = {"single": study_single, "h2": study_h2, "h3": study_h3, "robust": study_robust, "n4long": study_n4long}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("study", choices=list(STUDIES) + ["all"])
    ap.add_argument("--seeds", type=int, default=None, help="default: 20 for single, 10 otherwise")
    ap.add_argument("--steps", type=int, default=None, help="training steps (default per study)")
    ap.add_argument("--eval-steps", type=int, default=50_000)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--quick", action="store_true", help="2 seeds, 200k steps, outputs suffixed _quick")
    ap.add_argument("--suffix", default="", help="suffix for output CSV/plot names, e.g. _300M")
    a = ap.parse_args()
    a.sfx = "_quick" if a.quick else a.suffix
    if a.quick:
        a.steps, a.eval_steps = 200_000, 5_000
    for name in (STUDIES if a.study == "all" else [a.study]):
        a_seeds = a.seeds
        if a.seeds is None:
            a.seeds = 2 if a.quick else (20 if name == "single" else 10)
        print(f"\n######## study: {name} ({a.seeds} seeds) ########", flush=True)
        STUDIES[name](a)
        a.seeds = a_seeds


if __name__ == "__main__":
    main()
