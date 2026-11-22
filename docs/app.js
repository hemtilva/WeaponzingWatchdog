(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const form = $('setup');
  const inputs = form.elements;
  const colors = ['#364cc8','#c44c61','#169274','#b36b09','#8445b1','#067a9e'];
  const label = i => `Seller ${String.fromCharCode(65 + i)}`;
  const num = v => v === null ? '—' : v.toFixed(3);
  const pct = v => `${(v * 100).toFixed(1)}%`;
  const maxPeriods = 1000;
  let market, policies, metric = 'prices', dirty = false;
  const counts = rows => rows.reduce((a, row) => {
    a.bans += row.bans.filter(Boolean).length;
    a.attributed += row.attributed.filter(Boolean).length;
    a.exits += row.exits.filter(Boolean).length;
    return a;
  }, {bans: 0, attributed: 0, exits: 0});
  function syncRule() {
    $('param-label').hidden = inputs.rule.value === 'none';
    inputs.param.disabled = inputs.rule.value === 'none';
    $('param-text').textContent = inputs.rule.value === 'cap' ? 'Price cap' : 'Markup (%)';
  }
  function config() {
    const value = name => Number(inputs[name].value);
    return {rule: inputs.rule.value, count: value('count'), budget: value('budget'), minimum: value('minimum'),
      duration: value('duration'), capital: value('capital'), overhead: value('overhead'), seed: value('seed'),
      quality: value('quality'), scale: value('scale'), outside: value('outside'), buyers: value('buyers'),
      capped: inputs.capped.checked, param: value('param') / (inputs.rule.value === 'cap' ? 1 : 100)};
  }
  function reset(scenario) {
    const cfg = config();
    if (cfg.minimum > cfg.count || cfg.budget > cfg.count) {
      $('setup-message').textContent = 'Audit budget and minimum active sellers must be at most the number of sellers.';
      dirty = true;
      if (market) update();
      return;
    }
    market = Watchdog.create(cfg);
    policies = Array.from({length: cfg.count}, (_, i) => ({
      policy: i === 0 && scenario !== 'nash' || market.benchmark === null ? 'manual' : 'nash',
      price: i === 0 && scenario !== 'nash' ? 0.15 : market.benchmark ?? 0.30,
    }));
    dirty = false;
    $('setup-message').textContent = '';
    $('benchmark').textContent = market.benchmark === null ? 'No symmetric grid Nash found' : `Grid Nash ${num(market.benchmark)}`;
    $('sellers').innerHTML = policies.map(({policy, price}, i) => `
      <tr><td><span class="dot" style="background:${colors[i]}"></span>${label(i)}</td>
      <td><select aria-label="${label(i)} policy" data-policy="${i}">
      <option value="manual" ${policy === 'manual' ? 'selected' : ''}>Manual</option>
      <option value="nash" ${policy === 'nash' ? 'selected' : ''} ${market.benchmark === null ? 'disabled' : ''}>Fixed Nash</option>
      <option value="random">Random grid</option></select></td>
      <td><input type="number" data-price="${i}" aria-label="${label(i)} manual price" value="${price.toFixed(2)}" min="0" max="100" step="any" required></td>
      <td data-cash="${i}"></td><td data-status="${i}"></td></tr>`).join('');
    $('legend').innerHTML = policies.map((_,i) => `<span><i class="dot" style="background:${colors[i]}"></i>${label(i)}</span>`).join('');
    $('inspect').innerHTML = '<option>No periods yet</option>';
    update();
  }
  function update() {
    const totals = counts(market.history);
    const length = market.history.length;
    const canRun = !dirty && length < maxPeriods && market.sellers.some(s => s.cash > 0);
    $('period').textContent = length;
    $('active').textContent = market.sellers.filter(s => s.cash > 0 && s.ban === 0).length;
    $('bans').textContent = totals.bans;
    $('exits').textContent = totals.exits;
    $('ratio').textContent = totals.bans && market.benchmark !== null ? pct(totals.attributed / totals.bans) : '—';
    $('ratio-info').textContent = market.benchmark === null ? 'Nash reference unavailable' : totals.bans ? `${totals.attributed} of ${totals.bans} bans` : 'No bans yet';
    $('step').disabled = $('run').disabled = !canRun;
    $('download').disabled = $('inspect').disabled = !length;
    market.sellers.forEach((s, i) => {
      document.querySelector(`[data-cash="${i}"]`).textContent = num(s.cash);
      const td = document.querySelector(`[data-status="${i}"]`);
      const kind = s.cash === 0 ? 'exited' : s.ban ? 'banned' : 'active';
      td.className = `state ${kind}`;
      td.textContent = kind === 'exited' ? 'Bankrupt' : kind === 'banned' ? `Banned (${s.ban} left)` : 'Active';
      document.querySelector(`[data-policy="${i}"]`).disabled = s.cash === 0;
      document.querySelector(`[data-price="${i}"]`).disabled = policies[i].policy !== 'manual' || s.cash === 0;
    });
    $('chart-note').textContent = !length ? 'Step the market to start tracking prices.' :
      length >= maxPeriods ? 'Run complete. Reset to start a new simulation.' : `Last ${Math.min(length, 100)} periods. ${metric === 'prices' ? 'Inactive sellers leave gaps.' : 'Balances are measured after each period.'}`;
    inspect();
    draw();
  }
  function inspect() {
    const row = market.history[Number($('inspect').value) - 1];
    if (!row) {
      $('inspect-summary').textContent = 'Run a period to inspect prices, flags, and sanctions.';
      $('events').innerHTML = '<tr><td colspan="6">No period data yet.</td></tr>';
      return;
    }
    $('inspect-summary').textContent = `${row.threshold === null ? 'Rule inactive' : `Flag threshold ${num(row.threshold)}`} · Outside option ${pct(row.outsideShare)} · Buyer surplus ${num(row.surplus)}`;
    $('events').innerHTML = row.prices.map((price, i) => {
      const status = price === null ? 'Inactive' : row.bans[i] ? row.attributed[i] ? 'Banned · attributed' : 'Banned' :
        row.flags[i] ? 'Flagged · unaudited' : row.audited[i] ? 'Audited · clear' : 'No audit';
      return `<tr><td>${label(i)}</td><td>${num(price)}</td><td>${pct(row.shares[i])}</td><td>${num(row.profits[i])}</td><td>${status}</td><td>${num(row.cash[i])}${row.exits[i] ? ' · exited' : ''}</td></tr>`;
    }).join('');
  }
  function advance(steps) {
    const fields = [...$('sellers').querySelectorAll('[data-price]:enabled')];
    for (const input of fields) {
      if (!input.reportValidity()) return;
      policies[Number(input.dataset.price)].price = Number(input.value);
    }
    if (dirty) return;
    for (let i = 0; i < steps && market.history.length < maxPeriods && market.sellers.some(s => s.cash > 0); i++) {
      const row = market.step(policies);
      if (row.period === 1) $('inspect').innerHTML = '';
      $('inspect').add(new Option(`Period ${row.period}`, row.period));
      $('inspect').value = row.period;
    }
    update();
  }
  function draw() {
    const canvas = $('chart');
    const {width, height} = canvas.getBoundingClientRect();
    if (!width || !height) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr);
    const ctx = canvas.getContext('2d'); ctx.scale(dpr, dpr);
    const rows = market.history.slice(-100);
    const values = rows.flatMap(row => row[metric]).filter(v => v !== null);
    const yMax = Math.max(0.1, ...values) * 1.1;
    const left = 48, right = width - 14, bottom = height - 24, top = 10;
    const x = j => rows.length <= 1 ? left + (right-left)/2 : left + j * (right-left)/(rows.length-1);
    const y = v => bottom - v/yMax * (bottom-top);
    ctx.font = '12px system-ui'; ctx.fillStyle = '#52637a';
    for (let j = 0; j <= 4; j++) {
      const val = j * yMax / 4;
      ctx.strokeStyle = '#e2e9f2'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(left, y(val)); ctx.lineTo(right, y(val)); ctx.stroke();
      ctx.fillText(val.toFixed(val >= 10 ? 1 : 2), 3, y(val) + 4);
    }
    if (!rows.length) { ctx.fillText('No periods recorded', left + 12, top + 24); return; }
    market.sellers.forEach((_, i) => {
      ctx.strokeStyle = ctx.fillStyle = colors[i]; ctx.lineWidth = 2;
      let started = false;
      ctx.beginPath();
      rows.forEach((row, j) => {
        const val = row[metric][i];
        if (val === null) { started = false; return; }
        if (started) ctx.lineTo(x(j), y(val)); else ctx.moveTo(x(j), y(val));
        started = true;
      });
      ctx.stroke();
      rows.forEach((row, j) => {
        const val = row[metric][i];
        if (val === null || (j > 0 && j < rows.length - 1 && row[metric][i] === rows[j-1][metric][i])) return;
        ctx.beginPath(); ctx.arc(x(j), y(val), 2.6, 0, 2*Math.PI); ctx.fill();
      });
    });
    ctx.fillStyle = '#52637a';
    ctx.fillText(String(rows[0].period), left, height - 5);
    if (rows.length > 1) { ctx.textAlign = 'right'; ctx.fillText(String(rows.at(-1).period), right, height - 5); }
  }
  function preset() {
    const scenario = $('preset').value;
    if (!scenario) return;
    form.reset();
    inputs.rule.value = scenario === 'nash' ? 'none' : scenario;
    inputs.param.value = scenario === 'cap' ? 0.59 : 10;
    syncRule(); reset(scenario);
  }
  $('preset').addEventListener('change', preset);
  form.addEventListener('input', e => {
    if (e.target === inputs.rule) {
      inputs.param.value = inputs.rule.value === 'cap' ? 0.59 : 10;
      syncRule();
    }
    $('preset').value = ''; dirty = true;
    $('setup-message').textContent = 'Reset with setup to apply these changes.';
    update();
  });
  form.addEventListener('submit', e => { e.preventDefault(); reset(''); });
  $('sellers').addEventListener('change', e => {
    if (e.target.dataset.policy !== undefined) {
      policies[Number(e.target.dataset.policy)].policy = e.target.value;
      $('preset').value = ''; update();
    }
    if (e.target.dataset.price !== undefined) $('preset').value = '';
  });
  $('step').addEventListener('click', () => advance(1));
  $('run').addEventListener('click', () => advance(25));
  $('inspect').addEventListener('change', inspect);
  document.querySelectorAll('[data-metric]').forEach(button => button.addEventListener('click', () => {
    metric = button.dataset.metric;
    document.querySelectorAll('[data-metric]').forEach(b => b.setAttribute('aria-pressed', b === button));
    update();
  }));
  $('download').addEventListener('click', () => {
    const header = 'period,seller,price,share,profit,cash_after,flag_threshold,audited,flagged,new_ban,attributed_ban,exit,outside_share,consumer_surplus';
    const lines = market.history.flatMap(row => row.prices.map((price, i) => [row.period,label(i),price,row.shares[i],row.profits[i],row.cash[i],
      row.threshold,+row.audited[i],+row.flags[i],+row.bans[i],+row.attributed[i],+row.exits[i],row.outsideShare,row.surplus].join(',')));
    const url = URL.createObjectURL(new Blob([[header, ...lines].join('\n')], {type:'text/csv'}));
    const link = document.createElement('a'); link.href = url; link.download = 'watchdog-simulation.csv'; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  window.addEventListener('resize', draw);
  preset();
})();
