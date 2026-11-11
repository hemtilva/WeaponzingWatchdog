"""Shared protocol for the learning experiments.

Every experiment measures the same way:
  1. train   : epsilon-greedy Q-learning; epsilon decays to eps_min at `explore_frac`
               of training (so exploration scales with run length)
  2. greedy  : all learners frozen (epsilon = 0, no updates); this is the measured outcome

`Job` describes one run; `run_job` executes it on the native backend and returns a flat
dict of metrics; `run_jobs` fans jobs out over processes; `save_csv` writes results.
"""

import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.engine.rules import RuleType
from src.agents.q_learning import QLearningConfig
from src.agents.traits import AgentTraits
from src.agents.trainer import SimulationConfig, MultiAgentTrainer
from src.benchmarks.diagnostics import framing_market, market_diagnostics


@dataclass
class Job:
    """One training run. Market fields not given fall back to framing_market() defaults."""
    tag: str                                   # configuration label (rows are grouped by it)
    seed: int
    num_firms: int = 3
    rule: int = int(RuleType.MEAN_RATIO)
    rule_param: float = 0.10
    lethal: bool = True
    initial_capital: Optional[float] = None    # overrides `lethal`
    fixed_cost: float = 0.05
    ban_duration: int = 4
    audit_budget: int = 1
    n_min: int = 2
    price_levels: int = 11
    policies: Optional[List[str]] = None       # default: all Q-learners
    traits: Optional[List[Dict[str, float]]] = None  # per-firm AgentTraits kwargs
    train_steps: int = 10_000_000
    eval_steps: int = 50_000
    explore_frac: float = 0.6
    episode_length: int = 500
    extra: Dict[str, object] = field(default_factory=dict)  # copied into the output row


def build(job: Job) -> MultiAgentTrainer:
    market = framing_market(
        num_firms=job.num_firms,
        lethal=job.lethal,
        threshold_delta=job.rule_param if job.rule != RuleType.PRICE_CAP else 0.10,
        ban_duration=job.ban_duration,
        fixed_cost=job.fixed_cost,
        initial_capital=job.initial_capital,
    )
    market.audit_budget = job.audit_budget
    market.min_active_for_audit = job.n_min
    grid = np.linspace(0.0, 1.5, job.price_levels)
    eps_min = 0.01
    q_cfg = QLearningConfig(
        learning_rate=0.10, gamma=0.95, eps_start=1.0, eps_min=eps_min,
        eps_decay=float(eps_min ** (1.0 / max(1, int(job.explore_frac * job.train_steps)))),
    )
    traits = [AgentTraits(**t) for t in job.traits] if job.traits else None
    cfg = SimulationConfig(
        market=market, q_learning=q_cfg, price_grid=grid, total_steps=job.train_steps,
        warmup_steps=0, episode_length=job.episode_length, rule_type=RuleType(job.rule),
        rule_param=job.rule_param, eval_window=min(job.train_steps, 200_000),
        firm_policies=job.policies or ["q"] * job.num_firms, agent_traits=traits,
        seed=job.seed, backend="native",
    )
    return MultiAgentTrainer(cfg)


def run_job(job: Job) -> Dict[str, object]:
    t0 = time.time()
    tr = build(job)
    train = tr.run()
    tr.cfg.freeze_learners = True
    tr.cfg.total_steps = job.eval_steps
    tr.cfg.eval_window = job.eval_steps
    g = tr.run()
    ep = max(1, g.episodes_in_eval)
    q_firms = [i for i, p in enumerate(tr.policies) if p == "q"]
    row = {
        "tag": job.tag, "seed": job.seed, "N": job.num_firms, "rule": RuleType(job.rule).name,
        "rule_param": job.rule_param, "n_min": job.n_min, "K0": tr.m_cfg.initial_capital,
        "phi": job.fixed_cost, "T_ban": job.ban_duration, "B": job.audit_budget, "M": job.price_levels,
        "train_steps": job.train_steps,
        "nash_price": float(tr.nash_prices[0]), "nash_profit": float(tr.nash_profits[0]),
        # greedy (frozen) outcome
        "kills_per_ep": float(g.kills_by.sum()) / ep,
        "kills_by_q_per_ep": float(g.kills_by[q_firms].sum()) / ep if q_firms else 0.0,
        "self_bankr_per_ep": (g.total_bankruptcies - g.induced_bankruptcies) / ep,
        "bans_per_ep": g.total_fresh_bans / ep,
        "W": g.weaponization_rate,
        "W_joint": g.joint_weaponization_rate,
        "delta": g.collusion_index,
        "swl": g.social_welfare_loss,
        "mean_price": float(np.mean(g.mean_prices)),
        "firm0_price": float(g.mean_prices[0]),
        "firm0_profit": float(g.mean_profits[0]),
        "rival_profit": float(np.mean(g.mean_profits[1:])),
        "survivors": job.num_firms - g.total_bankruptcies / ep,
        "episodes": ep,
        # training diagnostics
        "train_kills_per_ep": float(train.kills_by.sum()) / max(1, train.episodes_in_eval),
        "coverage": train.state_coverage_ratio,
        "seconds": time.time() - t0,
    }
    for i, k in enumerate(g.kills_by):
        row[f"kills_firm{i}_per_ep"] = float(k) / ep
    diag = market_diagnostics(tr.m_cfg, price_grid=tr.price_grid,
                              rule_type=RuleType(job.rule), rule_param=job.rule_param)
    row.update(viable=diag["viable"], lethal=diag["lethal"], framing_feasible=diag["framing_feasible"])
    row.update(job.extra)
    return row


def default_workers() -> int:
    return max(1, min(10, (os.cpu_count() or 2) - 1))


def run_jobs(jobs: Sequence[Job], workers: Optional[int] = None) -> List[Dict[str, object]]:
    workers = workers or default_workers()
    print(f"Running {len(jobs)} jobs on {workers} workers...", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(run_job, jobs, chunksize=1))
    print(f"done in {time.time() - t0:.0f}s", flush=True)
    return rows


def save_csv(rows: List[Dict[str, object]], name: str) -> Path:
    (ROOT / "data").mkdir(exist_ok=True)
    path = ROOT / "data" / name
    keys: List[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"[SAVED] {path}")
    return path


def group_stats(rows, key: str, metric: str):
    """{tag: (mean, std, n)} of `metric` grouped by `key`, in first-seen order."""
    out = {}
    for r in rows:
        out.setdefault(r[key], []).append(float(r[metric]))
    return {k: (float(np.mean(v)), float(np.std(v)), len(v)) for k, v in out.items()}
