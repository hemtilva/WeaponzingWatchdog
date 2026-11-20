"""Vectorised market: B independent copies of the watchdog market stepped at once in PyTorch.

Mirrors ReplayEngine.step (src/evaluation/replay_attribution.py), which mirrors the native
loop (src/csrc/sim.cpp), period by period:
  1. active sellers (not banned, solvent) set prices; inactive sellers keep their last price
  2. logit shares with an outside option, operating profits for active sellers
  3. if N_active >= N_min: flags under the rule (mean, median, cap, EMA), one active seller
     audited uniformly at random; audited + flagged sellers are banned for T_ban periods
     (shutdown) or lose their profit above Nash for the period (escrow)
  4. EMA reference update, capital K' = min(cap, max(0, K + pi - Phi)), bankrupt when K' <= 1e-9
The audit draw takes one uniform per market and period, exactly as ReplayEngine.step does, so
the two produce identical trajectories from identical actions and uniforms
(tests/test_vec_market.py). Runs on CPU or CUDA.
"""
from dataclasses import dataclass

import numpy as np
import torch

BANKRUPTCY_TOL = 1e-9
MEAN, MEDIAN, CAP, ZSCORE, EMA = 0, 1, 2, 3, 4


@dataclass
class VecState:
    lag_idx: torch.Tensor    # (B, N) long
    timers: torch.Tensor     # (B, N) long
    capital: torch.Tensor    # (B, N) float
    ema: torch.Tensor        # (B,) float
    t: torch.Tensor          # (B,) long, period within the episode


class VecMarket:
    def __init__(self, market, batch: int, device="cpu", dtype=torch.float64):
        """`market`: an object with the fields of src.agents.dqn_single.make_market()
        (N, m_cfg, price_grid, nash_idx, nash_profits, cfg.rule_type / rule_param / ema_beta /
        penalty_mode)."""
        self.B, self.N = batch, market.N
        self.dev, self.dt = torch.device(device), dtype
        m = market.m_cfg
        f = lambda x: torch.as_tensor(np.asarray(x, dtype=np.float64), dtype=dtype, device=self.dev)
        self.grid = f(market.price_grid)
        self.M = len(market.price_grid)
        self.q = f(m.qualities)
        self.c = f(m.marginal_costs)
        self.a0, self.sigma, self.Q = float(m.outside_option), float(m.price_sensitivity), float(m.market_size)
        self.phi = float(m.fixed_cost)
        self.cap = None if m.capital_cap is None else float(m.capital_cap)
        self.cap0 = m.initial_capital if self.cap is None else min(m.initial_capital, self.cap)
        self.T = int(m.ban_duration)
        self.n_min = int(m.min_active_for_audit)
        self.rule = int(market.cfg.rule_type)
        self.d = float(market.cfg.rule_param)
        self.beta = float(market.cfg.ema_beta)
        self.escrow = market.cfg.penalty_mode == "escrow"
        self.nash_idx = torch.as_tensor(np.asarray(market.nash_idx), dtype=torch.long, device=self.dev)
        self.nash_pi = f(market.nash_profits)
        self.ema0 = float(np.mean(np.asarray(market.price_grid)[np.asarray(market.nash_idx)]))

    def reset(self) -> VecState:
        B, N = self.B, self.N
        return VecState(self.nash_idx.expand(B, N).clone(),
                        torch.zeros(B, N, dtype=torch.long, device=self.dev),
                        torch.full((B, N), self.cap0, dtype=self.dt, device=self.dev),
                        torch.full((B,), self.ema0, dtype=self.dt, device=self.dev),
                        torch.zeros(B, dtype=torch.long, device=self.dev))

    def reset_rows(self, st: VecState, rows: torch.Tensor):
        """Reset the markets where `rows` (B,) bool is True."""
        if not bool(rows.any()):
            return
        r = rows[:, None]
        st.lag_idx = torch.where(r, self.nash_idx.expand_as(st.lag_idx), st.lag_idx)
        st.timers = torch.where(r, torch.zeros_like(st.timers), st.timers)
        st.capital = torch.where(r, torch.full_like(st.capital, self.cap0), st.capital)
        st.ema = torch.where(rows, torch.full_like(st.ema, self.ema0), st.ema)
        st.t = torch.where(rows, torch.zeros_like(st.t), st.t)

    # ------------------------------------------------------------------ rules
    def _flags(self, p, act, n_act, ema):
        actf = act.to(self.dt)
        mean = (p * actf).sum(1) / n_act.clamp(min=1).to(self.dt)
        if self.rule == CAP:
            return act & (p > self.d)
        if self.rule == MEAN:
            ref = mean
        elif self.rule == EMA:
            ref = torch.where(ema > 0.0, self.beta * ema + (1.0 - self.beta) * mean, mean)
        elif self.rule == MEDIAN:
            big = torch.finfo(self.dt).max
            s = torch.sort(torch.where(act, p, torch.full_like(p, big)), dim=1).values
            n = n_act.clamp(min=1)
            hi = s.gather(1, (n // 2)[:, None]).squeeze(1)
            lo = s.gather(1, ((n - 1) // 2)[:, None]).squeeze(1)
            ref = torch.where(n % 2 == 1, hi, 0.5 * (lo + hi))
        else:
            raise ValueError("unsupported rule")
        return act & (p > (1.0 + self.d) * ref[:, None])

    # ------------------------------------------------------------------ step
    def step(self, st: VecState, actions: torch.Tensor, u: torch.Tensor):
        """Advance every market one period in place. `actions` (B, N) price indices (ignored for
        inactive sellers); `u` (B,) uniforms for the audit draw. Returns per-seller events."""
        act = (st.timers == 0) & (st.capital > 0.0)
        n_act = act.sum(1)
        a = torch.where(act, actions, st.lag_idx)
        p = self.grid[a]
        v = torch.where(act, (self.q - p) / self.sigma, torch.full_like(p, -torch.inf))
        v_out = torch.full_like(st.ema, self.a0 / self.sigma)
        mx = torch.maximum(v.max(1).values, v_out)
        ev = torch.exp(v - mx[:, None])
        shares = ev / (torch.exp(v_out - mx) + ev.sum(1))[:, None]
        profit = torch.where(act, self.Q * (p - self.c) * shares, torch.zeros_like(p))

        regulated = n_act >= self.n_min
        flags = self._flags(p, act, n_act, st.ema) & regulated[:, None]
        k = torch.clamp((u * n_act.to(u.dtype)).long(), max=(n_act - 1).clamp(min=0))
        rank = torch.cumsum(act.long(), 1) - 1
        audited = act & (rank == k[:, None]) & regulated[:, None]
        hit = audited & flags
        sanction = hit & (st.timers == 0)

        mean_act = (p * act.to(self.dt)).sum(1) / n_act.clamp(min=1).to(self.dt)
        st.ema = torch.where(n_act > 0, self.beta * st.ema + (1.0 - self.beta) * mean_act, st.ema)

        if self.escrow:
            profit = profit - torch.where(hit, torch.relu(profit - self.nash_pi), torch.zeros_like(profit))
            nt = st.timers
        else:
            nt = torch.where(hit, torch.full_like(st.timers, self.T), (st.timers - 1).clamp(min=0))
        alive = st.capital > 0.0
        nk = torch.where(alive, (st.capital + profit - self.phi).clamp(min=0.0), torch.zeros_like(st.capital))
        nk = torch.where(nk <= BANKRUPTCY_TOL, torch.zeros_like(nk), nk)
        if self.cap is not None:
            nk = nk.clamp(max=self.cap)
        bankrupt = alive & (nk <= 0.0)
        st.lag_idx, st.timers, st.capital = a, nt, nk
        st.t = st.t + 1
        return {"profits": profit, "sanction": sanction, "bankrupt": bankrupt, "prices": p, "active": act}
