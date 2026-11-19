"""Confidence intervals and tests for the paper's key comparisons (seed-level).

For each comparison: difference in means between two conditions, a 95% bootstrap CI
(10,000 resamples of seeds within each condition), and a two-sided permutation p-value
(10,000 relabellings). Seeds share numbers across conditions but their trajectories
diverge, so conditions are treated as independent samples. For trait sweeps: Spearman
correlation with a permutation p-value. Holm's correction is applied across the family.

Usage: python scripts/stats_report.py
Output: results/stats/key_comparisons.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
RNG = np.random.default_rng(20261119)
B = 10_000


def load(path, **filt):
    d = pd.read_csv(R / path)
    for k, v in filt.items():
        d = d[d[k] == v]
    return d


def diff_test(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    obs = x.mean() - y.mean()
    boots = np.array([RNG.choice(x, len(x)).mean() - RNG.choice(y, len(y)).mean() for _ in range(B)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    pooled = np.concatenate([x, y])
    if np.ptp(pooled) == 0:
        p = 1.0
    else:
        perm = np.empty(B)
        for i in range(B):
            RNG.shuffle(pooled)
            perm[i] = pooled[:len(x)].mean() - pooled[len(x):].mean()
        p = (np.sum(np.abs(perm) >= abs(obs) - 1e-12) + 1) / (B + 1)
    return obs, lo, hi, p


def spearman_test(v, y):
    v, y = np.asarray(v, float), np.asarray(y, float)
    rho = spearmanr(v, y).statistic
    perm = np.array([spearmanr(v, RNG.permutation(y)).statistic for _ in range(B)])
    p = (np.sum(np.abs(perm) >= abs(rho) - 1e-12) + 1) / (B + 1)
    return rho, p


def main():
    rows = []

    def add(claim, a, b, col, label_a, label_b):
        obs, lo, hi, p = diff_test(a[col], b[col])
        rows.append(dict(claim=claim, metric=col, a=label_a, b=label_b, n_a=len(a), n_b=len(b),
                         mean_a=a[col].mean(), mean_b=b[col].mean(), diff=obs, ci_lo=lo, ci_hi=hi, p=p))

    s = load("Krishs-MacBook-Air/all/study_single.csv")
    add("Single learner: mean vs median", s[s.tag == "mean d=0.10 lethal"], s[s.tag == "median d=0.10 lethal"],
        "kills_per_ep", "mean", "median")
    rob = load("Krishs-MacBook-Air/all/study_robust.csv")
    add("All learners 10M: mean vs median", rob[rob.tag == "delta=0.1 mean"], rob[rob.tag == "delta=0.1 median"],
        "kills_per_ep", "mean", "median")
    for steps in ("300M", "1000M"):
        t = load(f"training_length/all_learner_{steps}.csv")
        add(f"All learners {steps}: mean vs median", t[t.rule == "MEAN_RATIO"], t[t.rule == "MEDIAN_RATIO"],
            "kills_per_ep", "mean", "median")
    t10 = rob[rob.tag == "delta=0.1 mean"]
    t1b = load("training_length/all_learner_1000M.csv", rule="MEAN_RATIO")
    add("Mean rule: 1B vs 10M training", t1b, t10, "kills_per_ep", "1B", "10M")
    n4 = load("Krishs-MacBook-Air/all/study_n4long.csv")
    add("N=4, N_min=3: mean vs median", n4[n4.tag == "N=4 mean Nmin=3"], n4[n4.tag == "N=4 median Nmin=3"],
        "kills_per_ep", "mean", "median")
    d = load("defences/all.csv")
    d = d.assign(total_bankr=d.kills_per_ep + d.self_bankr_per_ep)
    for t in ("EMA beta=0.80", "EMA beta=0.95", "mean + escrow"):
        add(f"Defence vs mean (all learners): {t}", d[d.tag == t], d[d.tag == "mean"], "total_bankr", t, "mean")
    add("Escrow vs mean: mean price", d[d.tag == "mean + escrow"], d[d.tag == "mean"], "mean_price", "escrow", "mean")

    rob3 = load("long_300M/study_robust_300M.csv")
    for t in dict.fromkeys(rob3.tag.str.rsplit(" ", n=1).str[0]):
        add(f"Robustness 300M ({t}): mean vs median", rob3[rob3.tag == f"{t} mean"], rob3[rob3.tag == f"{t} median"],
            "kills_per_ep", "mean", "median")
    sens = load("sensitivity/all.csv")
    sens = sens.assign(cond=sens.tag.str.rsplit(" ", n=1).str[0])
    for c in dict.fromkeys(sens.cond):
        g = sens[sens.cond == c]
        add(f"Sensitivity 300M ({c}): mean vs median", g[g.rule_name == "mean"], g[g.rule_name == "median"],
            "kills_per_ep", "mean", "median")
    rep = load("replay_attribution/all_learners.csv")
    add("Replay: causally caused bankruptcies (joint, whole episode), mean vs median",
        rep[rep.tag == "mean"], rep[rep.tag == "median"], "replay_causal_bankruptcies_joint_per_ep_wNone", "mean", "median")
    add("Replay, mean rule: causal W (w=5) vs one-period W", rep[rep.tag == "mean"].rename(
        columns={"replay_causal_W_w5": "W"}), rep[rep.tag == "mean"].rename(columns={"replay_one_period_W": "W"}),
        "W", "causal", "one-period")

    h3 = load("traits_300M/study_h3_300M.csv")
    for trait in ("alpha", "lambda_altruism", "mu_ethics"):
        g = h3[h3.trait == trait]
        rho, p = spearman_test(g.trait_value, g.kills_per_ep)
        rows.append(dict(claim=f"Traits 300M: Spearman({trait}, attributed bankruptcies)", metric="kills_per_ep",
                         a=trait, b="", n_a=len(g), n_b=0, mean_a=np.nan, mean_b=np.nan, diff=rho,
                         ci_lo=np.nan, ci_hi=np.nan, p=p))

    out = pd.DataFrame(rows)
    order = np.argsort(out.p.values)               # Holm step-down
    m = len(out)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * out.p.values[i]))
        adj[i] = running
    out["p_holm"] = adj
    (R / "stats").mkdir(exist_ok=True)
    out.to_csv(R / "stats" / "key_comparisons.csv", index=False)
    pd.set_option("display.width", 200)
    print(out[["claim", "mean_a", "mean_b", "diff", "ci_lo", "ci_hi", "p", "p_holm"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
