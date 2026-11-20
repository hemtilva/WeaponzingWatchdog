"""Confidence intervals and tests for the proposal items completed late (results/promises).

Same method as scripts/stats_report.py: difference in seed means, 95% bootstrap CI (10,000
resamples), two-sided permutation p-value (10,000 relabellings), Holm's correction across
this family. Within-run comparisons between firms of one market (trait mixtures) are
paired by seed: bootstrap of the mean paired difference and a sign-flip permutation test.

Usage: python scripts/promises_report.py
Output: results/stats/promises_comparisons.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "results" / "promises"
RNG = np.random.default_rng(20261120)
B = 10_000


def diff_test(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    obs = x.mean() - y.mean()
    boots = np.array([RNG.choice(x, len(x)).mean() - RNG.choice(y, len(y)).mean() for _ in range(B)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    pooled = np.concatenate([x, y])
    if np.ptp(pooled) == 0:
        return obs, lo, hi, 1.0
    perm = np.empty(B)
    for i in range(B):
        RNG.shuffle(pooled)
        perm[i] = pooled[:len(x)].mean() - pooled[len(x):].mean()
    return obs, lo, hi, (np.sum(np.abs(perm) >= abs(obs) - 1e-12) + 1) / (B + 1)


def paired_test(x, y):
    d = np.asarray(x, float) - np.asarray(y, float)
    obs = d.mean()
    boots = np.array([RNG.choice(d, len(d)).mean() for _ in range(B)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    if np.all(d == 0):
        return obs, lo, hi, 1.0
    flips = RNG.choice([-1.0, 1.0], size=(B, len(d)))
    perm = (flips * d).mean(axis=1)
    return obs, lo, hi, (np.sum(np.abs(perm) >= abs(obs) - 1e-12) + 1) / (B + 1)


def holm(p):
    p = np.asarray(p)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = run
    return adj


def main():
    rules = pd.read_csv(P / "rules.csv")
    tban = pd.read_csv(P / "tban.csv")
    mix = pd.read_csv(P / "mixtures.csv")

    def v(d, tag, m="kills_per_ep"):
        x = d[d["tag"] == tag].sort_values("seed")[m].to_numpy()
        assert len(x) > 0, tag
        return x

    rows = []

    def add(group, label, x, y, metric, paired=False):
        obs, lo, hi, p = (paired_test if paired else diff_test)(x, y)
        rows.append(dict(group=group, comparison=label, metric=metric, a_mean=np.mean(x), b_mean=np.mean(y),
                         diff=obs, ci_lo=lo, ci_hi=hi, p=p, n_a=len(x), n_b=len(y)))

    K = "kills_per_ep"
    # Rules, all learners, random audits
    for r in ("z0.5", "z1.0", "planner"):
        add("rules", f"all: {r} vs mean (random audit)", v(rules, f"all {r} random"), v(rules, "all mean random"), K)
    add("rules", "all: planner vs median (random audit)", v(rules, "all planner random"), v(rules, "all median random"), K)
    add("rules", "all: planner vs mean, welfare loss", v(rules, "all planner random", "swl"), v(rules, "all mean random", "swl"), "swl")
    add("rules", "single: z0.5 vs median", v(rules, "single z0.5 random", "kills_by_q_per_ep"),
        v(rules, "single median random", "kills_by_q_per_ep"), "kills_by_q_per_ep")
    # Audit selection
    for r in ("mean", "median", "cap", "z0.5", "z1.0"):
        add("audit", f"all {r}: top vs random", v(rules, f"all {r} top"), v(rules, f"all {r} random"), K)
    add("audit", "all mean: top vs random, mean price", v(rules, "all mean top", "mean_price"),
        v(rules, "all mean random", "mean_price"), "mean_price")
    add("audit", "single mean: top vs random", v(rules, "single mean top", "kills_by_q_per_ep"),
        v(rules, "single mean random", "kills_by_q_per_ep"), "kills_by_q_per_ep")
    # Ban length and timer observation
    for r in ("mean", "median"):
        add("tban", f"{r}: T_ban=3 vs 1", v(tban, f"{r} T_ban=3"), v(tban, f"{r} T_ban=1"), K)
        add("timers", f"all {r}: timers seen vs bit", v(tban, f"{r} T_ban=4 timers=seen"),
            v(tban, f"{r} T_ban=4 timers=bit"), K)
    add("timers", "single mean: timers seen vs bit (2M)", v(tban, "single mean T_ban=4 timers=seen", "kills_by_q_per_ep"),
        v(tban, "single mean T_ban=4 timers=bit", "kills_by_q_per_ep"), "kills_by_q_per_ep")
    tl = P / "timerlen.csv"
    if tl.exists():
        t = pd.read_csv(tl)
        for st in ("20M", "100M"):
            add("timers", f"single mean: timers seen vs bit ({st})",
                v(t, f"single mean timers=seen steps={st}", "kills_by_q_per_ep"),
                v(t, f"single mean timers=bit steps={st}", "kills_by_q_per_ep"), "kills_by_q_per_ep")
    # Trait mixtures
    add("mixtures", "all risk-averse vs all neutral", v(mix, "alpha: all risk-averse"), v(mix, "all neutral"), K)
    add("mixtures", "1 neutral + 2 averse vs all averse", v(mix, "alpha: 1 neutral + 2 averse"), v(mix, "alpha: all risk-averse"), K)
    add("mixtures", "all altruistic vs all neutral", v(mix, "lambda: all altruistic"), v(mix, "all neutral"), K)
    add("mixtures", "1 selfish + 2 altruistic vs all altruistic", v(mix, "lambda: 1 selfish + 2 altruistic"),
        v(mix, "lambda: all altruistic"), K)
    add("mixtures", "all ethical vs all neutral", v(mix, "mu: all ethical"), v(mix, "all neutral"), K)
    for tag, label in (("alpha: 1 neutral + 2 averse", "neutral firm vs averse firms (paired)"),
                       ("lambda: 1 selfish + 2 altruistic", "selfish firm vs altruistic firms (paired)"),
                       ("mu: 1 predator + 2 ethical", "predator vs ethical firms (paired)")):
        d = mix[mix["tag"] == tag].sort_values("seed")
        add("mixtures", f"{tag}: {label}", d["kills_firm0_per_ep"].to_numpy(),
            ((d["kills_firm1_per_ep"] + d["kills_firm2_per_ep"]) / 2).to_numpy(), "kills by firm", paired=True)

    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p"].to_numpy())
    (ROOT / "results" / "stats").mkdir(parents=True, exist_ok=True)
    path = ROOT / "results" / "stats" / "promises_comparisons.csv"
    out.to_csv(path, index=False, float_format="%.4f")
    pd.set_option("display.width", 200)
    print(out[["comparison", "a_mean", "b_mean", "diff", "ci_lo", "ci_hi", "p_holm"]].to_string(
        index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"[SAVED] {path}  ({len(out)} comparisons, Holm-corrected)")


if __name__ == "__main__":
    main()
