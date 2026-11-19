"""Exact best response of one seller against fixed Nash-pricing rivals.

With the rivals' strategies fixed, the seller faces a single-agent MDP. This module builds
that MDP exactly, by breadth-first search over the states reachable from the start of an
episode, and solves it by value iteration. It mirrors the native simulation loop
(src/csrc/sim.cpp) period by period:

  1. active sellers set prices (rivals: their grid Nash price; seller 0: the action);
  2. logit shares and operating profits for active sellers;
  3. if N_active >= N_min: flags under the rule, one of the active sellers is audited
     uniformly at random (B = 1), and audited + flagged sellers are suspended for T_ban
     periods (shutdown) or lose their profit above Nash for the period (escrow);
  4. capital: K' = min(cap, max(0, K + pi - Phi)), bankrupt when K' <= 1e-9.

State: (ban timers, capitals) of all sellers. Profits are irrational, so exact capitals
take unboundedly many values; capitals are therefore kept on a grid of step `cap_step`
(default 0.001, against Phi = 0.05 and K0 = 0.15), which keeps the reachable set finite.
Each period's profit is rounded to the grid, an error of at most cap_step / 2. Seller 0's reward is its
operating profit / Q (the risk-neutral, trait-free reward used by the learners), and its
bankruptcy is terminal with value 0. The rivals' fixed policy makes this the
full-information best response, which upper-bounds what a learner can achieve.

This replaces src/benchmarks/exact_dp.py for the single-learner comparison: that solver
hard-codes audits at 3 active firms, rounds capital to multiples of Phi, and compares its
value against an analytic 'honest' formula instead of the same MDP.
"""
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.engine.market import compute_market_shares, compute_profits, compute_consumer_surplus
from src.engine.rules import RuleType
from src.benchmarks.nash import find_discrete_nash_equilibrium

BANKRUPTCY_TOL = 1e-9


@dataclass
class BRConfig:
    num_firms: int = 3
    price_grid: np.ndarray = field(default_factory=lambda: np.linspace(0.0, 1.5, 11))
    quality: float = 2.0
    marginal_cost: float = 0.0
    outside_option: float = 0.0
    sigma: float = 0.25
    market_size: float = 1.0
    fixed_cost: float = 0.05
    initial_capital: float = 0.15
    capital_cap: Optional[float] = None      # default: initial_capital (as in the experiments)
    ban_duration: int = 4
    rule: int = int(RuleType.MEAN_RATIO)
    rule_param: float = 0.10
    n_min: int = 2
    penalty_mode: str = "shutdown"           # or "escrow"
    gamma: float = 0.95
    cap_step: float = 0.001                  # capital grid; Phi and the cap must be multiples
    max_states: int = 1_000_000


class BestResponseMDP:
    def __init__(self, cfg: BRConfig):
        self.cfg = cfg
        self.N = cfg.num_firms
        self.grid = np.asarray(cfg.price_grid, dtype=np.float64)
        self.costs = np.full(self.N, cfg.marginal_cost)
        self.qual = np.full(self.N, cfg.quality)
        self.cap = cfg.initial_capital if cfg.capital_cap is None else cfg.capital_cap
        nash_p, nash_pi = find_discrete_nash_equilibrium(
            self.grid, self.costs, self.qual, cfg.outside_option, cfg.sigma, cfg.market_size)
        self.nash_prices = np.asarray(nash_p, dtype=np.float64)
        self.nash_profits = np.asarray(nash_pi, dtype=np.float64)
        self.nash_idx = int(np.argmin(np.abs(self.grid - self.nash_prices[0])))
        self._built = False

    # ------------------------------------------------------------------ rules
    def _flags(self, p: np.ndarray, act: np.ndarray) -> np.ndarray:
        out = np.zeros(self.N, dtype=bool)
        idx = np.where(act)[0]
        if len(idx) < self.cfg.n_min:
            return out
        ap = p[idx]
        d = self.cfg.rule_param
        if self.cfg.rule == RuleType.MEAN_RATIO:
            out[idx] = ap > (1.0 + d) * ap.mean()
        elif self.cfg.rule == RuleType.MEDIAN_RATIO:
            s = np.sort(ap)
            n = len(s)
            med = s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])
            out[idx] = ap > (1.0 + d) * med
        elif self.cfg.rule == RuleType.PRICE_CAP:
            out[idx] = ap > d
        else:
            raise ValueError("best_response supports MEAN_RATIO, MEDIAN_RATIO and PRICE_CAP")
        return out

    # ------------------------------------------------------------------ dynamics
    def _key(self, timers, capital) -> Tuple:
        return tuple(int(t) for t in timers) + tuple(round(float(k), 12) for k in capital)

    def initial_state(self) -> Tuple:
        return self._key(np.zeros(self.N), np.full(self.N, self.cfg.initial_capital))

    def _split(self, s: Tuple):
        return np.array(s[:self.N], dtype=np.int64), np.array(s[self.N:], dtype=np.float64)

    def transitions(self, s: Tuple, a: Optional[int]) -> List[Tuple[float, Tuple, float, Dict]]:
        """All outcomes of taking action a in state s: (prob, next_state, reward, info)."""
        timers, K = self._split(s)
        act = (timers == 0) & (K > 0.0)
        p = self.nash_prices.copy()
        if act[0]:
            p[0] = self.grid[a]
        shares = compute_market_shares(p, self.qual, act.astype(np.int32), self.cfg.outside_option, self.cfg.sigma)
        profits = compute_profits(p, self.costs, shares, self.cfg.market_size, act.astype(np.int32))
        idx = np.where(act)[0]
        regulated = len(idx) >= self.cfg.n_min
        flags = self._flags(p, act) if regulated else np.zeros(self.N, dtype=bool)
        audits = [(1.0 / len(idx), j) for j in idx] if regulated else [(1.0, None)]
        out = []
        for prob, j in audits:
            pr = profits.copy()
            nt = timers.copy()
            sanctioned = np.zeros(self.N, dtype=bool)
            if j is not None and flags[j]:
                sanctioned[j] = timers[j] == 0
            if self.cfg.penalty_mode == "escrow":
                if j is not None and flags[j]:
                    pr[j] -= max(0.0, pr[j] - self.nash_profits[j])
            else:
                nt = np.array([self.cfg.ban_duration if (j is not None and i == j and flags[i]) else max(0, timers[i] - 1)
                               for i in range(self.N)], dtype=np.int64)
            nK = np.where(K > 0.0, np.maximum(0.0, K + pr - self.cfg.fixed_cost), 0.0)
            nK[nK <= BANKRUPTCY_TOL] = 0.0
            nK = np.minimum(nK, self.cap)
            nK = np.round(nK / self.cfg.cap_step) * self.cfg.cap_step
            nK[nK <= BANKRUPTCY_TOL] = 0.0
            info = {"sanction": sanctioned, "bankrupt": (K > 0.0) & (nK <= 0.0), "profit0": pr[0]}
            out.append((prob, self._key(nt, nK), pr[0] / self.cfg.market_size, info))
        return out

    def _actions(self, s: Tuple) -> List[Optional[int]]:
        timers, K = self._split(s)
        if K[0] <= 0.0:
            return []                       # seller 0 bankrupt: terminal
        if timers[0] > 0:
            return [None]                   # suspended: no choice
        return list(range(len(self.grid)))

    # ------------------------------------------------------------------ build + solve
    def _table(self):
        """Per (active mask, seller-0 action): audit outcomes as integer capital increments.

        Profits, flags and audits depend only on who is active and seller 0's price, so they
        are computed once. Capital lives on a grid of step cap_step; because K is
        already on the grid and Phi is a multiple of the step, rounding K + pi - Phi equals
        K + round(pi / step) - Phi exactly, so the fast build matches `transitions()`.
        """
        step = self.cfg.cap_step
        self._step = step
        self._phi_u = int(round(self.cfg.fixed_cost / step))
        self._cap_u = int(round(self.cap / step))
        if abs(self._phi_u * step - self.cfg.fixed_cost) > 1e-12 or abs(self._cap_u * step - self.cap) > 1e-12:
            raise ValueError("fixed cost and capital cap must be multiples of the capital grid step")
        table = {}
        for mask in range(1 << self.N):
            act = np.array([(mask >> i) & 1 for i in range(self.N)], dtype=bool)
            acts = range(len(self.grid)) if act[0] else [None]
            for a in acts:
                p = self.nash_prices.copy()
                if act[0]:
                    p[0] = self.grid[a]
                sh = compute_market_shares(p, self.qual, act.astype(np.int32), self.cfg.outside_option, self.cfg.sigma)
                pr = compute_profits(p, self.costs, sh, self.cfg.market_size, act.astype(np.int32))
                idx = np.where(act)[0]
                regulated = len(idx) >= self.cfg.n_min
                flags = self._flags(p, act) if regulated else np.zeros(self.N, dtype=bool)
                outs = []
                for prob, j in ([(1.0 / len(idx), int(j)) for j in idx] if regulated else [(1.0, None)]):
                    prj = pr.copy()
                    hit = j is not None and bool(flags[j])
                    if hit and self.cfg.penalty_mode == "escrow":
                        prj[j] -= max(0.0, prj[j] - self.nash_profits[j])
                    inc = tuple(int(round(x / step)) for x in prj)
                    outs.append((prob, j if hit else None, inc, prj[0] / self.cfg.market_size))
                table[(mask, a)] = outs
        return table

    def build(self):
        """Breadth-first search over reachable states (integer capital units)."""
        N, T = self.N, self.cfg.ban_duration
        table = self._table()
        phi, capu, escrow = self._phi_u, self._cap_u, self.cfg.penalty_mode == "escrow"
        k0 = int(round(self.cfg.initial_capital / self._step))
        s0 = (0,) * N + (k0,) * N
        index = {s0: 0}
        states = [s0]
        sa_state, sa_action, o_sa, o_prob, o_next, o_rew, o_sanc, o_bank, o_pi0 = ([] for _ in range(9))
        q = deque([s0])
        while q:
            s = q.popleft()
            si = index[s]
            tm, ku = s[:N], s[N:]
            if ku[0] <= 0:
                continue                                     # seller 0 bankrupt: terminal
            mask = sum(1 << i for i in range(N) if tm[i] == 0 and ku[i] > 0)
            acts = range(len(self.grid)) if mask & 1 else [None]
            for a in acts:
                sa = len(sa_state)
                sa_state.append(si)
                sa_action.append(-1 if a is None else a)
                for prob, j, inc, r0 in table[(mask, a)]:
                    if escrow or j is None:
                        nt = tuple(max(0, t - 1) for t in tm) if not escrow else tm
                    else:
                        nt = tuple(T if i == j else max(0, tm[i] - 1) for i in range(N))
                    nk = tuple(min(capu, max(0, ku[i] + inc[i] - phi)) if ku[i] > 0 else 0 for i in range(N))
                    ns = nt + nk
                    ni = index.get(ns)
                    if ni is None:
                        if len(states) >= self.cfg.max_states:
                            raise RuntimeError("reachable state space exceeds max_states")
                        ni = index[ns] = len(states)
                        states.append(ns)
                        q.append(ns)
                    o_sa.append(sa); o_prob.append(prob); o_next.append(ni); o_rew.append(r0)
                    o_sanc.append(tuple(j is not None and i == j and tm[i] == 0 for i in range(N)))
                    o_bank.append(tuple(ku[i] > 0 and nk[i] == 0 for i in range(N)))
                    o_pi0.append(r0 * self.cfg.market_size)
        self.states = [st[:N] + tuple(k * self._step for k in st[N:]) for st in states]
        self.index = index
        self.sa_state = np.array(sa_state, dtype=np.int64)
        self.sa_action = np.array(sa_action, dtype=np.int64)
        self.o_sa = np.array(o_sa, dtype=np.int64)
        self.o_prob = np.array(o_prob)
        self.o_next = np.array(o_next, dtype=np.int64)
        self.o_rew = np.array(o_rew)
        self.o_sanction = np.array(o_sanc, dtype=bool).reshape(-1, N)
        self.o_bankrupt = np.array(o_bank, dtype=bool).reshape(-1, N)
        self.o_profit0 = np.array(o_pi0)
        self._built = True
        return self

    def _q(self, V):
        return np.bincount(self.o_sa, weights=self.o_prob * (self.o_rew + self.cfg.gamma * V[self.o_next]),
                           minlength=len(self.sa_state))

    def solve(self, tol: float = 1e-10, max_iter: int = 100_000):
        """Value iteration for the optimal value V* and a greedy optimal policy."""
        if not self._built:
            self.build()
        S = len(self.states)
        V = np.zeros(S)
        for _ in range(max_iter):
            Q = self._q(V)
            Vn = np.zeros(S)
            np.maximum.at(Vn, self.sa_state, Q)          # terminal states keep 0
            if np.max(np.abs(Vn - V)) < tol:
                V = Vn
                break
            V = Vn
        Q = self._q(V)
        best = np.full(S, -1, dtype=np.int64)            # index into sa arrays
        order = np.lexsort((self.sa_action, -np.round(Q, 12), self.sa_state))
        first = np.ones(len(order), dtype=bool)
        first[1:] = self.sa_state[order[1:]] != self.sa_state[order[:-1]]
        best[self.sa_state[order[first]]] = order[first]  # highest Q; ties: lowest action
        self.V, self.Q, self.best_sa = V, Q, best
        return V

    def evaluate(self, choose) -> np.ndarray:
        """Value of a fixed policy; choose(state_tuple) -> action index (ignored while suspended)."""
        if not self._built:
            self.build()
        S = len(self.states)
        pick = np.full(S, -1, dtype=np.int64)
        want = np.array([choose(st) for st in self.states])
        ok = (self.sa_action == -1) | (self.sa_action == want[self.sa_state])
        pick[self.sa_state[ok]] = np.where(ok)[0]
        V = np.zeros(S)
        mask = pick >= 0
        for _ in range(100_000):
            Vn = np.zeros(S)
            Vn[mask] = self._q(V)[pick[mask]]
            if np.max(np.abs(Vn - V)) < 1e-10:
                return Vn
            V = Vn
        return V

    def episode_stats(self, sa_of_state: np.ndarray, horizon: int = 500) -> Dict[str, float]:
        """Expected outcomes over one episode from s0 when following sa_of_state (-1: terminal)."""
        from scipy.sparse import csr_matrix
        S = len(self.states)
        chosen = np.zeros(len(self.sa_state), dtype=bool)
        live = sa_of_state >= 0
        chosen[sa_of_state[live]] = True
        o = np.where(chosen[self.o_sa])[0]
        src = self.sa_state[self.o_sa[o]]
        M = csr_matrix((self.o_prob[o], (self.o_next[o], src)), shape=(S, S))
        term = np.where(~live)[0]
        M = M + csr_matrix((np.ones(len(term)), (term, term)), shape=(S, S))
        w = self.o_prob[o]
        f_sanc = np.zeros((S, self.N)); f_bank = np.zeros((S, self.N)); f_pi = np.zeros(S)
        np.add.at(f_sanc, src, w[:, None] * self.o_sanction[o])
        np.add.at(f_bank, src, w[:, None] * self.o_bankrupt[o])
        np.add.at(f_pi, src, w * self.o_profit0[o])
        P = np.zeros(S); P[0] = 1.0
        sanc = np.zeros(self.N); bank = np.zeros(self.N); pi0 = 0.0
        for _ in range(horizon):
            sanc += P @ f_sanc; bank += P @ f_bank; pi0 += P @ f_pi
            P = M @ P
        return {"rival_bankruptcies": float(bank[1:].sum()), "own_bankruptcy": float(bank[0]),
                "rival_sanctions": float(sanc[1:].sum()), "own_sanctions": float(sanc[0]),
                "profit_per_period": float(pi0) / horizon}


def solve_best_response(cfg: BRConfig) -> Dict[str, float]:
    """Exact best response vs. always-Nash for one configuration."""
    mdp = BestResponseMDP(cfg).build()
    V = mdp.solve()
    V_hon = mdp.evaluate(lambda s: mdp.nash_idx)
    S = len(mdp.states)
    hon_sa = np.full(S, -1, dtype=np.int64)
    ok = (mdp.sa_action == -1) | (mdp.sa_action == mdp.nash_idx)
    hon_sa[mdp.sa_state[ok]] = np.where(ok)[0]
    opt = mdp.episode_stats(mdp.best_sa)
    hon = mdp.episode_stats(hon_sa)
    a0 = int(mdp.sa_action[mdp.best_sa[0]])
    return {
        "states": S,
        "V_opt": float(V[0]), "V_honest": float(V_hon[0]),
        "gain": float(V[0] - V_hon[0]),
        "first_price": float(mdp.grid[a0]),
        "nash_price": float(mdp.nash_prices[0]),
        **{f"opt_{k}": v for k, v in opt.items()},
        **{f"honest_{k}": v for k, v in hon.items()},
    }
