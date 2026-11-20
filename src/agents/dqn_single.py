"""Single DQN seller against fixed rivals, on the same dynamics as the native engine.

The market is stepped by ReplayEngine (src/evaluation/replay_attribution.py), which mirrors
src/csrc/sim.cpp period by period; seller 0's price comes from a deep Q-network, the other
sellers follow their fixed policies (Nash in the experiments).

The learner matches the tabular protocol except for the function approximator:
  observation  last-period prices / grid max, inactive flags of all sellers, own capital / K0
               (the tabular learner sees the same, with capital in three bins)
  reward       operating profit / Q (the trait-free reward of the tabular learners)
  transitions  suspension-aware: while seller 0 is banned, rewards accumulate (discounted)
               on the open transition, which is closed with a multi-step target when it acts
               again; bankruptcy is terminal with no bootstrap
  exploration  epsilon decays geometrically from 1 to 0.01 over the first 60% of training
  learning     double DQN with a hard-updated target network, Adam, Huber loss
Episodes last `episode_length` periods and end early only if no seller is solvent.
"""
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from src.evaluation.replay_attribution import ReplayEngine, attribute


@dataclass
class DQNSingleConfig:
    train_steps: int = 2_000_000
    eval_steps: int = 50_000
    episode_length: int = 500
    gamma: float = 0.95
    lr: float = 1e-3
    hidden: int = 64
    batch_size: int = 64
    buffer_size: int = 100_000
    train_every: int = 4
    target_every: int = 2_000
    learning_starts: int = 5_000
    eps_min: float = 0.01
    explore_frac: float = 0.6
    seed: int = 0


class QNet(nn.Module):
    def __init__(self, d_in: int, n_act: int, h: int):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d_in, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, n_act))

    def forward(self, x):
        return self.f(x)


class DQNSingleSeller:
    def __init__(self, trainer, cfg: DQNSingleConfig):
        """`trainer` is a MultiAgentTrainer built for the market (experiments.common.build);
        its policies for sellers 1..N-1 are used as the fixed rivals."""
        self.cfg = cfg
        self.engine = ReplayEngine(trainer)
        self.N = self.engine.N
        self.M = len(self.engine.grid)
        self.pmax = float(self.engine.grid[-1])
        self.k0 = float(self.engine.cap0)
        torch.manual_seed(cfg.seed)
        self.rng = np.random.default_rng(cfg.seed)
        d = 2 * self.N + 1
        self.q = QNet(d, self.M, cfg.hidden)
        self.tgt = QNet(d, self.M, cfg.hidden)
        self.tgt.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        B = cfg.buffer_size
        self.buf_s = np.zeros((B, d), np.float32)
        self.buf_a = np.zeros(B, np.int64)
        self.buf_g = np.zeros(B, np.float32)
        self.buf_d = np.zeros(B, np.float32)       # gamma^k of the multi-step transition (0 if terminal)
        self.buf_s2 = np.zeros((B, d), np.float32)
        self.n_buf = 0
        self.ptr = 0
        self.eps = 1.0
        self.eps_decay = cfg.eps_min ** (1.0 / max(1, int(cfg.explore_frac * cfg.train_steps)))
        self.engine.policies[0] = self._greedy_policy

    # ------------------------------------------------------------------ observation / policy
    def obs(self, st) -> np.ndarray:
        inactive = ((st.timers > 0) | (st.capital <= 0.0)).astype(np.float32)
        return np.concatenate([self.engine.grid[st.lag_idx] / self.pmax, inactive,
                               [st.capital[0] / self.k0]]).astype(np.float32)

    def _greedy_policy(self, st, num_active):
        with torch.no_grad():
            return int(self.q(torch.from_numpy(self.obs(st))[None]).argmax(1).item())

    def _act(self, o):
        if self.rng.random() < self.eps:
            return int(self.rng.integers(self.M))
        with torch.no_grad():
            return int(self.q(torch.from_numpy(o)[None]).argmax(1).item())

    # ------------------------------------------------------------------ learning
    def _store(self, s, a, g, disc, s2):
        i = self.ptr
        self.buf_s[i], self.buf_a[i], self.buf_g[i], self.buf_d[i], self.buf_s2[i] = s, a, g, disc, s2
        self.ptr = (i + 1) % self.cfg.buffer_size
        self.n_buf = min(self.n_buf + 1, self.cfg.buffer_size)

    def _update(self):
        idx = self.rng.integers(self.n_buf, size=self.cfg.batch_size)
        s = torch.from_numpy(self.buf_s[idx]); a = torch.from_numpy(self.buf_a[idx])
        g = torch.from_numpy(self.buf_g[idx]); disc = torch.from_numpy(self.buf_d[idx])
        s2 = torch.from_numpy(self.buf_s2[idx])
        with torch.no_grad():
            a2 = self.q(s2).argmax(1, keepdim=True)                     # double DQN
            target = g + disc * self.tgt(s2).gather(1, a2).squeeze(1)
        pred = self.q(s).gather(1, a[:, None]).squeeze(1)
        loss = nn.functional.smooth_l1_loss(pred, target)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def train(self):
        cfg, eng = self.cfg, self.engine
        st = eng.initial_state()
        pend = None                                                     # (s, a, G, disc)
        ep_t = 0
        for step in range(cfg.train_steps):
            active0 = st.timers[0] == 0 and st.capital[0] > 0.0
            force = None
            if active0:
                o = self.obs(st)
                if pend is not None:
                    self._store(pend[0], pend[1], pend[2], pend[3], o)
                a = self._act(o)
                pend = [o, a, 0.0, 1.0]
                force = {0: a}
            ev = eng.step(st, float(self.rng.random()), force)
            if pend is not None:
                pend[2] += pend[3] * float(ev["profits"][0]) / eng.m.market_size
                pend[3] *= cfg.gamma
                if ev["bankrupt"][0]:
                    self._store(pend[0], pend[1], pend[2], 0.0, pend[0])
                    pend = None
            self.eps = max(cfg.eps_min, self.eps * self.eps_decay)
            if step >= cfg.learning_starts and step % cfg.train_every == 0 and self.n_buf >= cfg.batch_size:
                self._update()
            if step % cfg.target_every == 0:
                self.tgt.load_state_dict(self.q.state_dict())
            ep_t += 1
            if ep_t >= cfg.episode_length or not (st.capital > 0.0).any():
                st, pend, ep_t = eng.initial_state(), None, 0         # truncate at the episode end

    # ------------------------------------------------------------------ evaluation
    def evaluate(self, attribution_episodes: int = 20) -> Dict[str, float]:
        """Frozen greedy play: per-episode statistics plus but-for attribution."""
        cfg, eng = self.cfg, self.engine
        n_ep = max(1, cfg.eval_steps // cfg.episode_length)
        rng = np.random.default_rng(cfg.seed + 10_000)
        bans = bankr_rivals = own_bankr = 0
        profit0 = 0.0
        price_active = []
        price_alone = []
        for _ in range(n_ep):
            events, _ = eng.run_episode(rng.random(cfg.episode_length))
            for ev in events:
                bans += int(ev["sanction"][1:].sum())
                bankr_rivals += int(ev["bankrupt"][1:].sum())
                own_bankr += int(ev["bankrupt"][0])
                profit0 += float(ev["profits"][0])
                if ev["active"][0]:
                    (price_alone if ev["active"].sum() == 1 else price_active).append(float(ev["prices"][0]))
        T = n_ep * cfg.episode_length
        out = {"episodes": n_ep, "rival_bans_per_ep": bans / n_ep, "rival_bankruptcies_per_ep": bankr_rivals / n_ep,
               "own_bankruptcy_per_ep": own_bankr / n_ep, "profit_per_period": profit0 / T,
               "mean_price_with_rivals": float(np.mean(price_active)) if price_active else np.nan,
               "mean_price_alone": float(np.mean(price_alone)) if price_alone else np.nan}
        out.update({f"replay_{k}": v for k, v in attribute(eng, episodes=attribution_episodes,
                                                           seed=cfg.seed).items()})
        return out
