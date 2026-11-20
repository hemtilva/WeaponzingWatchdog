"""Several independent DQN sellers (plus fixed rivals) on the native engine's dynamics.

Generalises src/agents/dqn_single.py: every learning seller has its own network, target
network, replay buffer, exploration schedule and suspension-aware open transition, and
treats every other seller as part of the environment (independent learners). Observation
of learner i: last-period prices / grid max, inactive flags of all sellers, own capital / K0.
"""
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn

from src.agents.dqn_single import DQNSingleConfig, QNet
from src.evaluation.replay_attribution import ReplayEngine, attribute


class _Learner:
    def __init__(self, idx: int, d: int, n_act: int, cfg: DQNSingleConfig, seed: int):
        self.i = idx
        torch.manual_seed(seed)
        self.q = QNet(d, n_act, cfg.hidden)
        self.tgt = QNet(d, n_act, cfg.hidden)
        self.tgt.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        B = cfg.buffer_size
        self.s = np.zeros((B, d), np.float32); self.a = np.zeros(B, np.int64)
        self.g = np.zeros(B, np.float32); self.dsc = np.zeros(B, np.float32)
        self.s2 = np.zeros((B, d), np.float32)
        self.n = 0; self.ptr = 0
        self.eps = 1.0
        self.pend = None

    def store(self, s, a, g, disc, s2, B):
        k = self.ptr
        self.s[k], self.a[k], self.g[k], self.dsc[k], self.s2[k] = s, a, g, disc, s2
        self.ptr = (k + 1) % B
        self.n = min(self.n + 1, B)


class DQNMultiSellers:
    def __init__(self, trainer, learners: Sequence[int], cfg: DQNSingleConfig):
        """`trainer`: a MultiAgentTrainer or make_market() object; sellers not in `learners`
        follow their fixed policies (Nash in the experiments)."""
        self.cfg = cfg
        self.engine = ReplayEngine(trainer)
        self.N = self.engine.N
        self.M = len(self.engine.grid)
        self.pmax = float(self.engine.grid[-1])
        self.k0 = float(self.engine.cap0)
        self.rng = np.random.default_rng(cfg.seed)
        d = 2 * self.N + 1
        self.L: List[_Learner] = [_Learner(i, d, self.M, cfg, cfg.seed * 100 + i) for i in learners]
        self.eps_decay = cfg.eps_min ** (1.0 / max(1, int(cfg.explore_frac * cfg.train_steps)))
        for l in self.L:
            self.engine.policies[l.i] = self._greedy_for(l)

    def obs(self, st, i: int) -> np.ndarray:
        inactive = ((st.timers > 0) | (st.capital <= 0.0)).astype(np.float32)
        return np.concatenate([self.engine.grid[st.lag_idx] / self.pmax, inactive,
                               [st.capital[i] / self.k0]]).astype(np.float32)

    def _greedy_for(self, l: _Learner):
        def act(st, num_active):
            with torch.no_grad():
                return int(l.q(torch.from_numpy(self.obs(st, l.i))[None]).argmax(1).item())
        return act

    def _update(self, l: _Learner):
        idx = self.rng.integers(l.n, size=self.cfg.batch_size)
        s = torch.from_numpy(l.s[idx]); a = torch.from_numpy(l.a[idx])
        g = torch.from_numpy(l.g[idx]); disc = torch.from_numpy(l.dsc[idx]); s2 = torch.from_numpy(l.s2[idx])
        with torch.no_grad():
            a2 = l.q(s2).argmax(1, keepdim=True)
            target = g + disc * l.tgt(s2).gather(1, a2).squeeze(1)
        pred = l.q(s).gather(1, a[:, None]).squeeze(1)
        loss = nn.functional.smooth_l1_loss(pred, target)
        l.opt.zero_grad(); loss.backward(); l.opt.step()

    def train(self):
        cfg, eng, B = self.cfg, self.engine, self.cfg.buffer_size
        st = eng.initial_state()
        ep_t = 0
        for step in range(cfg.train_steps):
            force = {}
            for l in self.L:
                if st.timers[l.i] == 0 and st.capital[l.i] > 0.0:
                    o = self.obs(st, l.i)
                    if l.pend is not None:
                        l.store(l.pend[0], l.pend[1], l.pend[2], l.pend[3], o, B)
                    if self.rng.random() < l.eps:
                        a = int(self.rng.integers(self.M))
                    else:
                        with torch.no_grad():
                            a = int(l.q(torch.from_numpy(o)[None]).argmax(1).item())
                    l.pend = [o, a, 0.0, 1.0]
                    force[l.i] = a
            ev = eng.step(st, float(self.rng.random()), force or None)
            for l in self.L:
                if l.pend is not None:
                    l.pend[2] += l.pend[3] * float(ev["profits"][l.i]) / eng.m.market_size
                    l.pend[3] *= cfg.gamma
                    if ev["bankrupt"][l.i]:
                        l.store(l.pend[0], l.pend[1], l.pend[2], 0.0, l.pend[0], B)
                        l.pend = None
                l.eps = max(cfg.eps_min, l.eps * self.eps_decay)
                if step >= cfg.learning_starts and step % cfg.train_every == 0 and l.n >= cfg.batch_size:
                    self._update(l)
                if step % cfg.target_every == 0:
                    l.tgt.load_state_dict(l.q.state_dict())
            ep_t += 1
            if ep_t >= cfg.episode_length or not (st.capital > 0.0).any():
                st, ep_t = eng.initial_state(), 0
                for l in self.L:
                    l.pend = None

    def evaluate(self, attribution_episodes: int = 20) -> Dict[str, float]:
        cfg, eng = self.cfg, self.engine
        n_ep = max(1, cfg.eval_steps // cfg.episode_length)
        rng = np.random.default_rng(cfg.seed + 10_000)
        learners = [l.i for l in self.L]
        fixed = [i for i in range(self.N) if i not in learners]
        bankr = np.zeros(self.N); profit = np.zeros(self.N)
        survivors = 0.0
        p_with, p_end = {i: [] for i in learners}, {i: [] for i in learners}
        for _ in range(n_ep):
            events, st = eng.run_episode(rng.random(cfg.episode_length))
            for ev in events:
                bankr += ev["bankrupt"]
                profit += ev["profits"]
                fixed_alive = any(ev["active"][j] for j in fixed)
                for i in learners:
                    if ev["active"][i]:
                        (p_with if fixed_alive else p_end)[i].append(float(ev["prices"][i]))
            survivors += float((st.capital > 0.0).sum())
        T = n_ep * cfg.episode_length
        out = {"episodes": n_ep, "survivors_per_ep": survivors / n_ep,
               "fixed_rival_bankruptcies_per_ep": float(bankr[fixed].sum()) / n_ep,
               "learner_bankruptcies_per_ep": float(bankr[learners].sum()) / n_ep}
        for k, i in enumerate(learners):
            out[f"learner{k}_profit"] = profit[i] / T
            out[f"learner{k}_bankruptcies_per_ep"] = bankr[i] / n_ep
            out[f"learner{k}_price_while_fixed_rivals_active"] = float(np.mean(p_with[i])) if p_with[i] else np.nan
            out[f"learner{k}_price_after"] = float(np.mean(p_end[i])) if p_end[i] else np.nan
        out.update({f"replay_{k}": v for k, v in attribute(eng, episodes=attribution_episodes, seed=cfg.seed).items()})
        return out
