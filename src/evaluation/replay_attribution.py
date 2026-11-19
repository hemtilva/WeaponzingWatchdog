"""Trajectory-level ("but-for") attribution of bans and bankruptcies.

The simulator's built-in attribution asks a one-period question: would this ban have
happened if rival k had priced at Nash *this period*? That misses framing built up over
several periods, for example against an EMA reference, where the reference was dragged
down earlier.

This module replays frozen-policy episodes and asks the causal question directly: if rival
k had priced at Nash over the last w periods before the ban (and everyone, k included,
followed their policies otherwise), would the ban, or the victim's bankruptcy, still have
happened at the same time? Replays use common random numbers (one pre-drawn uniform per
period decides the audit), so the factual and counterfactual episodes face the same luck,
and every period's full state is checkpointed so a counterfactual restarts at the start of
its window.

The replay engine mirrors the native loop (src/csrc/sim.cpp) period by period: actions
from the encoded observation, logit clearing, regulation when N_active >= N_min (flags,
one uniformly drawn audit, suspension or escrow), the EMA reference update, and capital
with the reserve cap. Frozen Q-learners act greedily; ties go to the lowest price index
(the native engine breaks ties at random, so a few states can differ from engine runs).
"""
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from src.engine.market import compute_market_shares, compute_profits
from src.engine.rules import RuleType

BANKRUPTCY_TOL = 1e-9


@dataclass
class _State:
    lag_idx: np.ndarray
    timers: np.ndarray
    capital: np.ndarray
    ema: float

    def copy(self):
        return _State(self.lag_idx.copy(), self.timers.copy(), self.capital.copy(), self.ema)


class ReplayEngine:
    """Deterministic replays of a trainer's market under fixed policies."""

    def __init__(self, trainer, policies: Optional[Sequence[Callable]] = None):
        tr = trainer
        self.tr = tr
        self.N = tr.N
        self.m = tr.m_cfg
        self.grid = tr.price_grid
        self.nash_idx = np.asarray(tr.nash_idx, dtype=np.int64)
        self.nash_profits = np.asarray(tr.nash_profits, dtype=np.float64)
        self.rule = int(tr.cfg.rule_type)
        self.param = float(tr.cfg.rule_param)
        self.beta = float(tr.cfg.ema_beta)
        self.escrow = tr.cfg.penalty_mode == "escrow"
        self.n_min = int(self.m.min_active_for_audit)
        self.cap = self.m.capital_cap
        self.cap0 = self.m.initial_capital if self.cap is None else min(self.m.initial_capital, self.cap)
        self.T = int(self.m.ban_duration)
        self.policies = list(policies) if policies is not None else [self.policy_for(i) for i in range(self.N)]

    # ---------------------------------------------------------------- policies
    def policy_for(self, i: int) -> Callable:
        p = self.tr.policies[i]
        if p == "q":
            Q = self.tr.agents[i].Q
            enc = self.tr.indexer.encode

            def act(st: _State, num_active: int, i=i, Q=Q, enc=enc):
                inactive = (st.timers > 0) | (st.capital <= 0.0)
                s = enc(self.grid[st.lag_idx], inactive, st.capital[i])
                return int(np.argmax(Q[s]))
            return act
        if p == "nash":
            return lambda st, num_active, i=i: int(self.nash_idx[i])
        if p == "attacker":
            mono = int(self.tr.mono_idx[i])
            return lambda st, num_active: 0 if num_active >= self.n_min else mono
        raise ValueError(f"no replay policy for '{p}'")

    # ---------------------------------------------------------------- rules
    def _flags(self, p, act, ema):
        out = np.zeros(self.N, dtype=bool)
        idx = np.where(act)[0]
        if len(idx) < self.n_min:
            return out
        ap = p[idx]
        mean = ap.mean()
        d = self.param
        if self.rule == RuleType.MEAN_RATIO:
            ref = mean
        elif self.rule == RuleType.EMA_MEAN_RATIO:
            ref = self.beta * ema + (1.0 - self.beta) * mean if ema > 0.0 else mean
        elif self.rule == RuleType.MEDIAN_RATIO:
            s = np.sort(ap)
            n = len(s)
            ref = s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])
        elif self.rule == RuleType.PRICE_CAP:
            out[idx] = ap > d
            return out
        else:
            raise ValueError("unsupported rule for replay")
        out[idx] = ap > (1.0 + d) * ref
        return out

    # ---------------------------------------------------------------- one period
    def initial_state(self) -> _State:
        return _State(self.nash_idx.copy(), np.zeros(self.N, dtype=np.int64),
                      np.full(self.N, self.cap0), float(np.mean(self.grid[self.nash_idx])))

    def step(self, st: _State, u: float, force: Optional[Dict[int, int]] = None):
        """Advance one period in place. Returns per-firm event arrays."""
        N = self.N
        act = (st.timers == 0) & (st.capital > 0.0)
        n_act = int(act.sum())
        a = st.lag_idx.copy()
        for i in range(N):
            if act[i]:
                a[i] = force[i] if (force and i in force) else self.policies[i](st, n_act)
        p = self.grid[a]
        sh = compute_market_shares(p, self.m.qualities, act.astype(np.int32), self.m.outside_option,
                                   self.m.price_sensitivity)
        pr = compute_profits(p, self.m.marginal_costs, sh, self.m.market_size, act.astype(np.int32))
        audited = np.zeros(N, dtype=bool)
        flags = np.zeros(N, dtype=bool)
        if n_act >= self.n_min:
            flags = self._flags(p, act, st.ema)
            idx = np.where(act)[0]
            audited[idx[min(int(u * len(idx)), len(idx) - 1)]] = True
        hit = audited & flags
        sanction = hit & (st.timers == 0)
        if n_act > 0:
            st.ema = self.beta * st.ema + (1.0 - self.beta) * float(p[act].mean())
        if self.escrow:
            for i in np.where(hit)[0]:
                pr[i] -= max(0.0, pr[i] - self.nash_profits[i])
            nt = st.timers.copy()
        else:
            nt = np.where(hit, self.T, np.maximum(0, st.timers - 1))
        nk = np.where(st.capital > 0.0, np.maximum(0.0, st.capital + pr - self.m.fixed_cost), 0.0)
        nk[nk <= BANKRUPTCY_TOL] = 0.0
        if self.cap is not None:
            nk = np.minimum(nk, self.cap)
        bankrupt = (st.capital > 0.0) & (nk <= 0.0)
        st.lag_idx, st.timers, st.capital = a, nt, nk
        return {"sanction": sanction, "bankrupt": bankrupt, "prices": p, "active": act}

    # ---------------------------------------------------------------- episodes
    def run_episode(self, us: np.ndarray, start: Optional[_State] = None, t0: int = 0, t1: Optional[int] = None,
                    force_from: Optional[int] = None, force_to: Optional[int] = None, force_firm: Optional[int] = None,
                    checkpoints: Optional[List[_State]] = None):
        """Run periods t0..t1-1 from `start`. Firm force_firm is held at Nash in [force_from, force_to]."""
        st = (start or self.initial_state()).copy()
        t1 = len(us) if t1 is None else t1
        events = []
        for t in range(t0, t1):
            if checkpoints is not None:
                checkpoints.append(st.copy())
            force = None
            if force_firm is not None and force_from <= t <= force_to:
                firms = (force_firm,) if np.isscalar(force_firm) else tuple(force_firm)
                force = {int(f): int(self.nash_idx[f]) for f in firms}
            ev = self.step(st, us[t], force)
            events.append(ev)
            if not (st.capital > 0.0).any():
                break
        return events, st


def attribute(engine: ReplayEngine, episodes: int = 20, length: int = 500, windows=(1, 5, 20, None),
              seed: int = 0) -> Dict[str, float]:
    """But-for attribution over `episodes` frozen-policy episodes.

    For every fresh ban of firm j at period t and each rival k, and each window w
    (None = since the start of the episode), replay periods [t - w + 1, t] with k at Nash.
    The ban is caused by k (window w) if j is not freshly sanctioned at t in that replay.
    A bankruptcy of j during a ban that began at t is caused by k (window w) if, with k
    honest (at Nash) from period t - w + 1 to the end of the episode, j survives the
    episode. Holding k at Nash afterwards stops it from simply framing j again later, and
    judging survival at the end of the episode, not at the original bankruptcy period,
    stops a delayed bankruptcy from counting as an avoided one.
    Also returns the simulator's one-period attribution on the same bans for comparison.
    """
    rng = np.random.default_rng(seed)
    N = engine.N
    out = {"episodes": 0, "bans": 0, "bankruptcies": 0, "one_period_bans": 0}
    for w in windows:
        out[f"caused_bans_w{w}"] = 0
        out[f"caused_bankruptcies_w{w}"] = 0
        out[f"caused_bankruptcies_joint_w{w}"] = 0
    for _ in range(episodes):
        us = rng.random(length)
        ckpt: List[_State] = []
        events, _ = engine.run_episode(us, checkpoints=ckpt)
        out["episodes"] += 1
        T = len(events)
        for t, ev in enumerate(events):
            for j in np.where(ev["sanction"])[0]:
                out["bans"] += 1
                # one-period counterfactual (the simulator's W): any rival at Nash this period
                if any(_still_sanctioned(engine, ckpt, us, t, t, k, j) is False for k in range(N) if k != j):
                    out["one_period_bans"] += 1
                # find this ban's bankruptcy, if any (victim dies before the ban ends)
                t_b = next((s for s in range(t + 1, min(T, t + engine.T + 1)) if events[s]["bankrupt"][j]), None)
                if t_b is not None:
                    out["bankruptcies"] += 1
                for w in windows:
                    lo = 0 if w is None else max(0, t - w + 1)
                    rivals = [k for k in range(N) if k != j]
                    if any(_still_sanctioned(engine, ckpt, us, lo, t, k, j) is False for k in rivals):
                        out[f"caused_bans_w{w}"] += 1
                    if t_b is not None:
                        if any(_still_bankrupt(engine, ckpt, us, lo, k, j) is False for k in rivals):
                            out[f"caused_bankruptcies_w{w}"] += 1
                        if _still_bankrupt(engine, ckpt, us, lo, tuple(rivals), j) is False:
                            out[f"caused_bankruptcies_joint_w{w}"] += 1
    bans, bk, ep = out["bans"], out["bankruptcies"], out["episodes"]
    res = {"episodes": ep, "bans_per_ep": bans / ep, "bankruptcies_during_bans_per_ep": bk / ep,
           "one_period_W": out["one_period_bans"] / bans if bans else 0.0}
    for w in windows:
        res[f"causal_W_w{w}"] = out[f"caused_bans_w{w}"] / bans if bans else 0.0
        res[f"causal_bankruptcies_per_ep_w{w}"] = out[f"caused_bankruptcies_w{w}"] / ep
        res[f"causal_bankruptcies_joint_per_ep_w{w}"] = out[f"caused_bankruptcies_joint_w{w}"] / ep
    return res


def _still_sanctioned(engine, ckpt, us, lo, t, k, j) -> bool:
    """Is j freshly sanctioned at period t when k is held at Nash over [lo, t]?"""
    ev, _ = engine.run_episode(us, start=ckpt[lo], t0=lo, t1=t + 1, force_from=lo, force_to=t, force_firm=k)
    return len(ev) == t + 1 - lo and bool(ev[-1]["sanction"][j])


def _still_bankrupt(engine, ckpt, us, lo, k, j) -> bool:
    """Is j bankrupt at the end of the episode when k (a firm or a tuple of firms) is honest
    (at Nash) from lo onward?"""
    ev, st = engine.run_episode(us, start=ckpt[lo], t0=lo, t1=len(us), force_from=lo, force_to=len(us),
                                force_firm=k)
    return bool(st.capital[j] <= 0.0)
