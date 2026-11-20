"""Proposal items completed after the main study (native backend, main-study protocol).

  rules     Rule x audit selection: mean, median, cap, z-score (tau = 0.5, 1.0) and the
            omniscient planner (flags only prices above Nash), each with uniform-random and
            top-score-first audits (proposal 2.4 and Table 1). Single learner vs two Nash
            rivals (2M steps, 20 seeds) and all learners (300M steps, 10 seeds).
  tban      Ban length T_ban in {1, 3, 5, 10} (proposal Step 3) at K0 = 3 Phi, mean vs median,
            and the ban-timer observation ablation (proposal 4.3): learners see exact ban
            timers instead of an inactive bit, T_ban = 4. All learners, 300M steps, 10 seeds.
  timerlen  Follow-up to the timer ablation: single learner vs two Nash rivals with and
            without exact ban timers at 2M, 20M and 100M steps (20 seeds), to separate
            information from the slower learning of a 27x larger state space.
  mixtures  Trait mixtures (proposal Step 3 and Table 1, homogeneous all-ethical vs
            all-predatory): heterogeneous populations in mu, alpha and lambda under the mean
            rule; per-firm kills and bankruptcies show who frames whom. 300M steps, 10 seeds.

Seeds match the main study (3000+ single learner, 4000+ rules/T_ban, 5000+ traits), so
control cells reproduce stored results exactly.

  python experiments/run_promises.py rules|tban|timerlen|mixtures|all [--seeds N] [--steps S] [--quick]
Output: results/promises/<study>.csv
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Job, ROOT, run_jobs  # noqa: E402

from src.engine.rules import RuleType  # noqa: E402

OUT = ROOT / "results" / "promises"
RULES = [  # (name, rule, param)
    ("mean", int(RuleType.MEAN_RATIO), 0.10),
    ("median", int(RuleType.MEDIAN_RATIO), 0.10),
    ("cap", int(RuleType.PRICE_CAP), 0.60),
    ("z0.5", int(RuleType.Z_SCORE), 0.5),
    ("z1.0", int(RuleType.Z_SCORE), 1.0),
    ("planner", int(RuleType.PLANNER), 0.0),
]
MIXTURES = {  # name -> per-firm traits (firm order matters for the per-firm columns)
    "all neutral": [{}, {}, {}],
    "mu: all ethical": [{"mu_ethics": 1.0}] * 3,
    "mu: 1 predator + 2 ethical": [{}, {"mu_ethics": 1.0}, {"mu_ethics": 1.0}],
    "mu: 2 predators + 1 ethical": [{}, {}, {"mu_ethics": 1.0}],
    "alpha: all risk-averse": [{"alpha": 2.0}] * 3,
    "alpha: 1 neutral + 2 averse": [{}, {"alpha": 2.0}, {"alpha": 2.0}],
    "alpha: 2 neutral + 1 averse": [{}, {}, {"alpha": 2.0}],
    "lambda: all altruistic": [{"lambda_altruism": 1.0}] * 3,
    "lambda: 1 selfish + 2 altruistic": [{}, {"lambda_altruism": 1.0}, {"lambda_altruism": 1.0}],
    "lambda: 2 selfish + 1 altruistic": [{}, {}, {"lambda_altruism": 1.0}],
}


def save(rows, name):
    OUT.mkdir(parents=True, exist_ok=True)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    path = OUT / name
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"[SAVED] {path}", flush=True)


def summary(rows, metrics=("kills_per_ep", "W_joint", "delta", "swl", "mean_price", "survivors")):
    tags = list(dict.fromkeys(r["tag"] for r in rows))
    print(f"{'config':<44}" + "".join(f"{m:>12}" for m in metrics))
    for t in tags:
        sel = [r for r in rows if r["tag"] == t]
        print(f"{t:<44}" + "".join(f"{sum(float(r[m]) for r in sel) / len(sel):12.3f}" for m in metrics))


def study_rules(a):
    jobs = []
    for audit in ("random", "top"):
        for name, rule, par in RULES:
            jobs += [Job(tag=f"single {name} {audit}", seed=3000 + s, rule=rule, rule_param=par,
                         policies=["q", "nash", "nash"], train_steps=a.single_steps, audit_mode=audit,
                         extra=dict(study="single", rule_name=name))
                     for s in range(a.single_seeds)]
            jobs += [Job(tag=f"all {name} {audit}", seed=4000 + s, rule=rule, rule_param=par,
                         train_steps=a.steps, audit_mode=audit, extra=dict(study="all", rule_name=name))
                     for s in range(a.seeds)]
    rows = run_jobs(jobs, a.workers)
    save(rows, "rules.csv")
    summary([r for r in rows if r["study"] == "single"],
            ("kills_by_q_per_ep", "W_joint", "firm0_price", "firm0_profit", "rival_profit"))
    summary([r for r in rows if r["study"] == "all"])


def study_tban(a):
    jobs = []
    for name, rule, par in RULES[:2]:
        for tb in (1, 3, 5, 10):
            jobs += [Job(tag=f"{name} T_ban={tb}", seed=4000 + s, rule=rule, rule_param=par,
                         initial_capital=0.15, ban_duration=tb, train_steps=a.steps,
                         extra=dict(study="tban", rule_name=name))
                     for s in range(a.seeds)]
        for obs in (False, True):
            jobs += [Job(tag=f"{name} T_ban=4 timers={'seen' if obs else 'bit'}", seed=4000 + s, rule=rule,
                         rule_param=par, initial_capital=0.15, ban_duration=4, train_steps=a.steps,
                         observe_ban_timers=obs, extra=dict(study="timer_obs", rule_name=name))
                     for s in range(a.seeds)]
            jobs += [Job(tag=f"single {name} T_ban=4 timers={'seen' if obs else 'bit'}", seed=3000 + s,
                         rule=rule, rule_param=par, policies=["q", "nash", "nash"], ban_duration=4,
                         train_steps=a.single_steps, observe_ban_timers=obs,
                         extra=dict(study="timer_obs_single", rule_name=name))
                     for s in range(a.single_seeds)]
    rows = run_jobs(jobs, a.workers)
    save(rows, "tban.csv")
    summary([r for r in rows if r["study"] != "timer_obs_single"])
    summary([r for r in rows if r["study"] == "timer_obs_single"], ("kills_by_q_per_ep", "firm0_profit"))


def study_timerlen(a):
    jobs = [Job(tag=f"single mean timers={'seen' if obs else 'bit'} steps={st // 1_000_000}M", seed=3000 + s,
                policies=["q", "nash", "nash"], train_steps=st, observe_ban_timers=obs,
                extra=dict(study="timerlen"))
            for st in (2_000_000, 20_000_000, 100_000_000) for obs in (False, True)
            for s in range(a.single_seeds)]
    rows = run_jobs(jobs, a.workers)
    save(rows, "timerlen.csv")
    summary(rows, ("kills_by_q_per_ep", "firm0_profit", "firm0_price", "coverage"))


def study_mixtures(a):
    jobs = [Job(tag=name, seed=5000 + s, traits=traits, train_steps=a.steps, extra=dict(study="mixture"))
            for name, traits in MIXTURES.items() for s in range(a.seeds)]
    rows = run_jobs(jobs, a.workers)
    save(rows, "mixtures.csv")
    summary(rows, ("kills_per_ep",) + tuple(f"kills_firm{i}_per_ep" for i in range(3))
            + tuple(f"bankr_firm{i}_per_ep" for i in range(3)) + ("mean_price",))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("study", choices=["rules", "tban", "timerlen", "mixtures", "all"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--single-seeds", type=int, default=20)
    ap.add_argument("--steps", type=int, default=300_000_000)
    ap.add_argument("--single-steps", type=int, default=2_000_000)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--quick", action="store_true", help="smoke test: 2 seeds, 200k steps")
    a = ap.parse_args()
    if a.quick:
        a.seeds = a.single_seeds = 2
        a.steps = a.single_steps = 200_000
        global OUT
        OUT = ROOT / "results" / "promises_quick"
    studies = {"rules": study_rules, "tban": study_tban, "timerlen": study_timerlen, "mixtures": study_mixtures}
    for name in (studies if a.study == "all" else [a.study]):
        print(f"\n=== {name} ===", flush=True)
        studies[name](a)


if __name__ == "__main__":
    main()
