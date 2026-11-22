/* Browser implementation of the project's static audit-and-ban market. */
(() => {
  'use strict';
  const grid = Array.from({length: 11}, (_, i) => i * 0.15);
  const zero = 1e-9;
  const sum = xs => xs.reduce((a, b) => a + b, 0);
  function rng(seed) {
    let state = seed >>> 0;
    return () => {
      state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
      return state / 4294967296;
    };
  }
  function demand(prices, cfg) {
    const outside = cfg.outside / cfg.scale;
    const scores = prices.map(p => p === null ? -Infinity : (cfg.quality - p) / cfg.scale);
    const max = Math.max(outside, ...scores);
    const weights = scores.map(v => Math.exp(v - max));
    const out = Math.exp(outside - max);
    const total = sum(weights) + out;
    const shares = weights.map(w => w / total);
    return {shares, outsideShare: out / total,
      surplus: cfg.buyers * cfg.scale * (max + Math.log(total) - outside),
      profits: prices.map((p, i) => p === null ? 0 : cfg.buyers * p * shares[i])};
  }
  function nash(cfg) {
    for (const p of grid) {
      const prices = Array(cfg.count).fill(p);
      const baseline = demand(prices, cfg).profits[0];
      if (grid.every(a => demand([a, ...prices.slice(1)], cfg).profits[0] <= baseline + zero)) return p;
    }
    return null;
  }
  function rule(prices, cfg) {
    const active = prices.filter(p => p !== null).sort((a, b) => a - b);
    const enabled = active.length >= cfg.minimum && cfg.rule !== 'none';
    if (!enabled) return {threshold: null, flags: prices.map(() => false)};
    const mid = Math.floor(active.length / 2);
    const median = active.length % 2 ? active[mid] : (active[mid - 1] + active[mid]) / 2;
    const reference = cfg.rule === 'median' ? median : sum(active) / active.length;
    const threshold = cfg.rule === 'cap' ? cfg.param : reference * (1 + cfg.param);
    return {threshold, flags: prices.map(p => p !== null && p > threshold)};
  }
  function create(cfg) {
    const sellers = Array.from({length: cfg.count}, () => ({cash: cfg.capital, ban: 0}));
    const auditRandom = rng(cfg.seed);
    const priceRandom = rng(cfg.seed ^ 0x9e3779b9);
    const benchmark = nash(cfg);
    const history = [];
    function step(policies) {
      const active = sellers.flatMap((s, i) => s.cash > zero && s.ban === 0 ? [i] : []);
      const activeSet = new Set(active);
      const prices = sellers.map((_, i) => {
        if (!activeSet.has(i)) return null;
        const {policy, price} = policies[i];
        if (policy === 'random') return grid[Math.floor(priceRandom() * grid.length)];
        if (policy === 'nash' && benchmark !== null) return benchmark;
        return price;
      });
      const {threshold, flags} = rule(prices, cfg);
      const {shares, outsideShare, profits, surplus} = demand(prices, cfg);
      const pool = active.slice();
      const audited = new Set();
      if (threshold !== null) {
        for (let i = 0; i < Math.min(cfg.budget, active.length); i++) {
          audited.add(pool.splice(Math.floor(auditRandom() * pool.length), 1)[0]);
        }
      }
      const bans = flags.map((flag, i) => flag && audited.has(i));
      const attributed = bans.map((ban, victim) => ban && benchmark !== null && active.some(i => {
        if (i === victim || prices[i] === benchmark) return false;
        const alternative = prices.slice();
        alternative[i] = benchmark;
        return !rule(alternative, cfg).flags[victim];
      }));
      const exits = sellers.map((s, i) => {
        if (s.cash <= zero) return false;
        const next = Math.max(0, s.cash + profits[i] - cfg.overhead);
        s.cash = next <= zero ? 0 : Math.min(next, cfg.capped ? cfg.capital : Infinity);
        s.ban = s.cash === 0 ? 0 : bans[i] ? cfg.duration : Math.max(0, s.ban - 1);
        return s.cash === 0;
      });
      const row = {period: history.length + 1, prices, shares, profits, outsideShare, surplus, threshold, flags,
        audited: sellers.map((_, i) => audited.has(i)), bans, attributed, exits, cash: sellers.map(s => s.cash)};
      history.push(row);
      return row;
    }
    return {sellers, benchmark, history, step};
  }
  globalThis.Watchdog = {create, demand, grid, rule};
})();
