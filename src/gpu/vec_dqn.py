"""Independent DQN sellers trained on B vectorised markets (GPU or CPU).

Same learner as src/agents/dqn_single.py, batched over markets:
  observation   last-period prices / grid max, inactive flags of all sellers, own capital / K0
  reward        operating profit / Q
  transitions   suspension-aware: each market keeps an open transition per learner that
                accumulates discounted reward while the learner is banned and is closed with a
                multi-step target when it acts again; bankruptcy is terminal
  exploration   epsilon decays geometrically from 1 to eps_min over explore_frac of training
  learning      double DQN, hard target updates, Adam, Huber loss, large batches
Sellers that are not learners price at Nash. Evaluation copies the trained networks to the
CPU and uses the validated ReplayEngine and but-for attribution, so results are comparable
with every other experiment in the repository.
"""
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn

from src.gpu.vec_market import VecMarket


@dataclass
class VecDQNConfig:
    env_steps: int = 50_000_000        # total market-periods across all parallel markets
    num_envs: int = 1024
    episode_length: int = 500
    gamma: float = 0.95
    lr: float = 1e-3
    hidden: int = 64
    batch_size: int = 1024
    buffer_size: int = 500_000
    updates_per_iter: int = 4          # gradient steps per vectorised step (replay ratio = U*batch/num_envs)
    target_every: int = 250            # in gradient steps
    learning_starts: int = 20_000      # transitions in the buffer before learning
    eps_min: float = 0.01
    explore_frac: float = 0.6
    seed: int = 0
    device: str = "auto"


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QNet(nn.Module):
    def __init__(self, d_in: int, n_act: int, h: int):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d_in, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, n_act))

    def forward(self, x):
        return self.f(x)


class _Learner:
    def __init__(self, idx, d, n_act, cfg, dev, B):
        self.i = idx
        self.q = QNet(d, n_act, cfg.hidden).to(dev)
        self.tgt = QNet(d, n_act, cfg.hidden).to(dev)
        self.tgt.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        C = cfg.buffer_size
        z = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device=dev)
        self.bs, self.ba, self.bg, self.bd, self.bs2 = z(C, d), z(C, dt=torch.long), z(C), z(C), z(C, d)
        self.n, self.ptr = 0, 0
        self.ps, self.pa, self.pg, self.pd = z(B, d), z(B, dt=torch.long), z(B), z(B)
        self.pv = torch.zeros(B, dtype=torch.bool, device=dev)
        self.updates = 0

    def push(self, rows, s, a, g, disc, s2, C):
        k = int(rows.sum())
        if k == 0:
            return
        idx = (self.ptr + torch.arange(k, device=rows.device)) % C
        self.bs[idx], self.ba[idx], self.bg[idx] = s[rows], a[rows], g[rows]
        self.bd[idx], self.bs2[idx] = disc[rows], s2[rows]
        self.ptr = (self.ptr + k) % C
        self.n = min(self.n + k, C)


class VecDQN:
    def __init__(self, market, learners: Sequence[int], cfg: VecDQNConfig):
        self.cfg = cfg
        self.dev = pick_device(cfg.device)
        torch.manual_seed(cfg.seed)
        self.gen = torch.Generator(device=self.dev).manual_seed(cfg.seed)
        self.market = market
        # Market in float64: capital bookkeeping needs it (e.g. 0.15 - 3 * 0.05 must reach exactly 0
        # for the bankruptcy test); the networks run in float32.
        self.vm = VecMarket(market, cfg.num_envs, device=self.dev, dtype=torch.float64)
        self.N, self.M = self.vm.N, self.vm.M
        self.pmax = float(self.vm.grid[-1])
        self.k0 = float(self.vm.cap0)
        d = 2 * self.N + 1
        self.L: List[_Learner] = [_Learner(i, d, self.M, cfg, self.dev, cfg.num_envs) for i in learners]
        self.iters = max(1, cfg.env_steps // cfg.num_envs)
        self.eps = 1.0
        self.eps_decay = cfg.eps_min ** (1.0 / max(1, int(cfg.explore_frac * self.iters)))

    def obs(self, st, i):
        inactive = ((st.timers > 0) | (st.capital <= 0.0)).float()
        return torch.cat([self.vm.grid[st.lag_idx].float() / self.pmax, inactive,
                          (st.capital[:, i:i + 1].float() / self.k0)], dim=1)

    def _update(self, l: _Learner):
        idx = torch.randint(l.n, (self.cfg.batch_size,), device=self.dev, generator=self.gen)
        s, a, g, disc, s2 = l.bs[idx], l.ba[idx], l.bg[idx], l.bd[idx], l.bs2[idx]
        with torch.no_grad():
            a2 = l.q(s2).argmax(1, keepdim=True)
            target = g + disc * l.tgt(s2).gather(1, a2).squeeze(1)
        pred = l.q(s).gather(1, a[:, None]).squeeze(1)
        loss = nn.functional.smooth_l1_loss(pred, target)
        l.opt.zero_grad(set_to_none=True)
        loss.backward()
        l.opt.step()
        l.updates += 1
        if l.updates % self.cfg.target_every == 0:
            l.tgt.load_state_dict(l.q.state_dict())

    def train(self, until: float = 1.0, progress=None):
        """Train up to fraction `until` of the budget; call again with a larger fraction to
        continue (the market state, buffers and exploration schedule carry over)."""
        cfg, vm, C = self.cfg, self.vm, self.cfg.buffer_size
        B = cfg.num_envs
        if not hasattr(self, "_st"):
            self._st, self._it = vm.reset(), 0
        st = self._st
        nash = vm.nash_idx.expand(B, self.N)
        stop = min(self.iters, int(round(until * self.iters)))
        while self._it < stop:
            it = self._it
            self._it += 1
            actions = nash.clone()
            for l in self.L:
                o = self.obs(st, l.i)
                act = (st.timers[:, l.i] == 0) & (st.capital[:, l.i] > 0.0)
                l.push(act & l.pv, l.ps, l.pa, l.pg, l.pd, o, C)
                with torch.no_grad():
                    greedy = l.q(o).argmax(1)
                rand = torch.randint(self.M, (B,), device=self.dev, generator=self.gen)
                explore = torch.rand(B, device=self.dev, generator=self.gen) < self.eps
                a = torch.where(explore, rand, greedy)
                actions[:, l.i] = a
                l.ps = torch.where(act[:, None], o, l.ps)
                l.pa = torch.where(act, a, l.pa)
                l.pg = torch.where(act, torch.zeros_like(l.pg), l.pg)
                l.pd = torch.where(act, torch.ones_like(l.pd), l.pd)
                l.pv = l.pv | act
            u = torch.rand(B, device=self.dev, generator=self.gen, dtype=vm.dt)
            ev = vm.step(st, actions, u)
            for l in self.L:
                r = ev["profits"][:, l.i].float() / vm.Q
                l.pg = torch.where(l.pv, l.pg + l.pd * r, l.pg)
                l.pd = torch.where(l.pv, l.pd * cfg.gamma, l.pd)
                dead = ev["bankrupt"][:, l.i] & l.pv
                l.push(dead, l.ps, l.pa, l.pg, torch.zeros_like(l.pd), l.ps, C)
                l.pv = l.pv & ~dead
            done = (st.t >= cfg.episode_length) | ~(st.capital > 0.0).any(1)
            if bool(done.any()):
                vm.reset_rows(st, done)
                for l in self.L:
                    l.pv = l.pv & ~done
            for l in self.L:
                if l.n >= cfg.learning_starts:
                    for _ in range(cfg.updates_per_iter):
                        self._update(l)
            self.eps = max(cfg.eps_min, self.eps * self.eps_decay)
            if progress is not None and it % max(1, self.iters // 20) == 0:
                progress(it, self.iters, self.eps)

    # ------------------------------------------------------------------ evaluation (CPU)
    def cpu_policies(self):
        """Greedy CPU policies for ReplayEngine (same observation as training)."""
        grid = np.asarray(self.market.price_grid, dtype=np.float32)
        out = {}
        for l in self.L:
            net = QNet(2 * self.N + 1, self.M, self.cfg.hidden)
            net.load_state_dict({k: v.detach().cpu() for k, v in l.q.state_dict().items()})
            net.eval()

            def act(st, num_active, net=net, i=l.i):
                inactive = ((st.timers > 0) | (st.capital <= 0.0)).astype(np.float32)
                o = np.concatenate([grid[st.lag_idx] / self.pmax, inactive, [st.capital[i] / self.k0]])
                with torch.no_grad():
                    return int(net(torch.from_numpy(o.astype(np.float32))[None]).argmax(1).item())
            out[l.i] = act
        return out

    def state_dicts(self):
        return {l.i: {k: v.detach().cpu() for k, v in l.q.state_dict().items()} for l in self.L}


def evaluate(market, policies: Dict[int, callable], eval_episodes: int = 100, episode_length: int = 500,
             attribution_episodes: int = 20, seed: int = 0) -> Dict[str, float]:
    """Frozen-policy outcomes and but-for attribution on the validated CPU ReplayEngine."""
    from src.evaluation.replay_attribution import ReplayEngine, attribute
    eng = ReplayEngine(market)
    for i, f in policies.items():
        eng.policies[i] = f
    learners = sorted(policies)
    fixed = [i for i in range(eng.N) if i not in learners]
    rng = np.random.default_rng(seed + 10_000)
    bankr, profit = np.zeros(eng.N), np.zeros(eng.N)
    survivors, bans_fixed = 0.0, 0.0
    p_with, p_after = {i: [] for i in learners}, {i: [] for i in learners}
    for _ in range(eval_episodes):
        events, st = eng.run_episode(rng.random(episode_length))
        for ev in events:
            bankr += ev["bankrupt"]
            profit += ev["profits"]
            bans_fixed += float(ev["sanction"][fixed].sum()) if fixed else 0.0
            fixed_alive = any(ev["active"][j] for j in fixed) if fixed else False
            for i in learners:
                if ev["active"][i]:
                    (p_with if fixed_alive else p_after)[i].append(float(ev["prices"][i]))
        survivors += float((st.capital > 0.0).sum())
    T = eval_episodes * episode_length
    out = {"eval_episodes": eval_episodes, "survivors_per_ep": survivors / eval_episodes,
           "fixed_rival_bankruptcies_per_ep": float(bankr[fixed].sum()) / eval_episodes if fixed else 0.0,
           "fixed_rival_bans_per_ep": bans_fixed / eval_episodes,
           "learner_bankruptcies_per_ep": float(bankr[learners].sum()) / eval_episodes,
           "learner_profit_mean": float(profit[learners].mean()) / T,
           "nash_profit": float(eng.nash_profits[0])}
    for k, i in enumerate(learners):
        out[f"learner{k}_profit"] = profit[i] / T
        out[f"learner{k}_bankruptcies_per_ep"] = bankr[i] / eval_episodes
        out[f"learner{k}_price_while_fixed_active"] = float(np.mean(p_with[i])) if p_with[i] else float("nan")
        out[f"learner{k}_price_after"] = float(np.mean(p_after[i])) if p_after[i] else float("nan")
    if attribution_episodes > 0:
        out.update({f"replay_{k}": v for k, v in attribute(eng, episodes=attribution_episodes, seed=seed).items()})
    return out
