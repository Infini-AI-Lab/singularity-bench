/* Leaderboard bars, performance-vs-cost scatter, and the full results table. */
(function () {
  'use strict';
  const SB = window.SB;
  const svgEl = SB.svgEl;
  const still = SB.still;
  const AG = SB.AGENTS;
  const LIFE = ['pre', 'mid', 'post', 'inf', 'clu', 'har'];

  const easeOut = x => 1 - Math.pow(1 - x, 3);
  const easeInOut = x => (x < .5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2);
  const easeBack = x => { const c1 = 1.4, c3 = c1 + 1; return 1 + c3 * Math.pow(x - 1, 3) + c1 * Math.pow(x - 1, 2); };
  // run fn(p) for p in [0,1] over dur ms after delay ms
  function tween(dur, fn, { delay = 0, ease = easeOut, done } = {}) {
    if (still || dur <= 0) { fn(1, 1); done && done(); return () => {}; }
    let raf = 0, t0 = null, dead = false;
    const step = now => {
      if (dead) return;
      if (t0 === null) t0 = now + delay;
      const raw = Math.max(0, Math.min(1, (now - t0) / dur));
      fn(ease(raw), raw);
      if (raw < 1) raf = requestAnimationFrame(step); else done && done();
    };
    raf = requestAnimationFrame(step);
    return () => { dead = true; cancelAnimationFrame(raf); };
  }
  const onceVisible = (el, fn, threshold = .3) => {
    if (still || !('IntersectionObserver' in window)) { fn(); return; }
    const io = new IntersectionObserver(es => { if (es.some(e => e.isIntersecting)) { io.disconnect(); fn(); } }, { threshold });
    io.observe(el);
  };
  const debounce = (fn, ms) => { let t; return () => { clearTimeout(t); t = setTimeout(fn, ms); }; };
  const money = v => `$${(+v).toFixed(2)}`;
  const tipPlace = (plot, tip, x, y) => {
    tip.hidden = false;
    const w = tip.offsetWidth, pw = plot.clientWidth;
    const left = Math.max(w / 2 + 4, Math.min(pw - w / 2 - 4, x));
    tip.style.left = `${left}px`;
    tip.style.top = `${y}px`;
  };
  const head = a => `<div class="th"><img src="${a.logo}" alt="">${a.model}<span style="font-weight:400;color:var(--ink-3)">+ ${a.harness}</span></div>`;

  /* ================= leaderboard ================= */
  function Leaderboard(root) {
    const plot = root.querySelector('.plot');
    const svg = plot.querySelector('svg');
    const tip = plot.querySelector('.tip');
    const sub = root.querySelector('[data-lb-sub]');
    const info = root.querySelector('[data-index-info]');
    if (info) {
      info.addEventListener('pointerenter', () => info.classList.remove('dismissed'));
      info.addEventListener('focusin', () => info.classList.remove('dismissed'));
      info.addEventListener('keydown', e => {
        if (e.key === 'Escape') {
          info.classList.add('dismissed');
          e.stopPropagation();
        }
      });
    }
    const metric = 'index';
    let L = null, entered = false, cancel = [];

    const val = (a, m) => (m === 'index' ? a.idx : a.ratio[m]);
    const ymaxOf = m => {
      const mx = Math.max(1, ...AG.map(a => val(a, m))) * 1.12;
      return [1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.5, 3].find(s => s >= mx) || Math.ceil(mx);
    };
    const tickVals = ym => { let st = ym > 1.6 ? .5 : .2; if (L && !L.v && span() / (ym / st) < 34) st = ym > 1.6 ? 1 : .5; const out = []; for (let v = 0; v <= ym + 1e-9; v += st) out.push(+v.toFixed(2)); return out; };

    const defs = svgEl('defs', {}, svg);
    defs.innerHTML = Object.entries(SB.DEV).map(([k, d]) => `<pattern id="st-${k}" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="7" height="7" fill="${d.color}"/><rect width="2.4" height="7" fill="#fff" fill-opacity=".9"/></pattern>`).join('');
    const gGrid = svgEl('g', {}, svg), gBars = svgEl('g', {}, svg), gTop = svgEl('g', {}, svg);

    const bars = AG.map(a => {
      const g = svgEl('g', { class: 'bar' }, gBars);
      const shape = svgEl('path', { fill: a.striped ? `url(#st-${a.dev})` : a.color }, g);
      const edge = svgEl('path', { fill: 'none', stroke: a.color, 'stroke-width': a.striped ? 1.4 : 0 }, g);
      const vt = svgEl('text', { class: 'val' }, g);
      const ft = svgEl('text', { class: 'fail' }, g);
      const tile = svgEl('rect', { rx: 8, fill: '#f1f3f4' }, g);
      const img = svgEl('image', { href: a.logo, preserveAspectRatio: 'xMidYMid meet' }, g);
      const n1 = svgEl('text', { class: 'nm' }, g), n2 = svgEl('text', { class: 'nm' }, g), hn = svgEl('text', { class: 'hn' }, g);
      const hit = svgEl('rect', { class: 'hit' }, g);
      return { a, g, shape, edge, vt, ft, tile, img, n1, n2, hn, hit, pos: 0, len: 0 };
    });

    function layout() {
      const W = Math.max(300, plot.clientWidth);
      if (W >= 720) {
        const m = { l: 42, r: 8, t: 34, b: 108 }, H = 440;
        return { W, H, v: true, m, slot: (W - m.l - m.r) / AG.length, base: H - m.b, top: m.t };
      }
      const m = { l: 172, r: 52, t: 30, b: 14 }, row = 40;
      return { W, H: m.t + row * AG.length + m.b, v: false, m, row, x0: m.l, x1: W - m.r };
    }
    const span = () => (L.v ? L.base - L.top : L.x1 - L.x0);
    const lenOf = (v, ym) => Math.max(0, v / ym * span());

    function placeInner(b) {
      const a = b.a;
      [b.n1, b.n2, b.hn].forEach(t => { t.textContent = ''; });
      if (L.v) {
        const wrap = L.slot < 104 && a.model.includes(' V4.1');
        const lines = wrap ? ['DeepSeek', 'V4.1 Flash'] : [a.model];
        b.tile.setAttribute('x', -16); b.tile.setAttribute('y', L.base + 10); b.tile.setAttribute('width', 32); b.tile.setAttribute('height', 32);
        b.img.setAttribute('x', -10); b.img.setAttribute('y', L.base + 16); b.img.setAttribute('width', 20); b.img.setAttribute('height', 20);
        [b.n1, b.n2, b.hn, b.vt, b.ft].forEach(t => t.setAttribute('text-anchor', 'middle'));
        b.n1.setAttribute('x', 0); b.n1.setAttribute('y', L.base + 60); b.n1.textContent = lines[0];
        if (lines[1]) { b.n2.setAttribute('x', 0); b.n2.setAttribute('y', L.base + 75); b.n2.textContent = lines[1]; }
        b.hn.setAttribute('x', 0); b.hn.setAttribute('y', L.base + (lines[1] ? 90 : 75)); b.hn.textContent = a.harness;
        b.hit.setAttribute('x', -L.slot / 2); b.hit.setAttribute('y', L.top - 20); b.hit.setAttribute('width', L.slot); b.hit.setAttribute('height', L.H - L.top + 20);
      } else {
        b.tile.setAttribute('x', 0); b.tile.setAttribute('y', -14); b.tile.setAttribute('width', 28); b.tile.setAttribute('height', 28);
        b.img.setAttribute('x', 6); b.img.setAttribute('y', -8); b.img.setAttribute('width', 16); b.img.setAttribute('height', 16);
        [b.n1, b.hn].forEach(t => t.setAttribute('text-anchor', 'start'));
        b.n1.setAttribute('x', 38); b.n1.setAttribute('y', -2); b.n1.textContent = a.model;
        b.hn.setAttribute('x', 38); b.hn.setAttribute('y', 13); b.hn.textContent = a.harness;
        b.vt.setAttribute('text-anchor', 'start'); b.ft.setAttribute('text-anchor', 'start');
        b.hit.setAttribute('x', 0); b.hit.setAttribute('y', -L.row / 2); b.hit.setAttribute('width', L.W); b.hit.setAttribute('height', L.row);
      }
    }

    function drawBar(b) {
      const len = b.len;
      if (L.v) {
        const cx = L.m.l + L.slot * (b.pos + .5);
        b.g.setAttribute('transform', `translate(${cx.toFixed(2)} 0)`);
        const w = Math.min(56, L.slot * .62), x0 = -w / 2, x1 = w / 2, y = L.base - len, r = Math.min(5, len / 2, w / 2);
        const d = len < .5 ? '' : `M${x0} ${L.base}V${(y + r).toFixed(2)}Q${x0} ${y.toFixed(2)} ${x0 + r} ${y.toFixed(2)}H${x1 - r}Q${x1} ${y.toFixed(2)} ${x1} ${(y + r).toFixed(2)}V${L.base}Z`;
        b.shape.setAttribute('d', d); b.edge.setAttribute('d', d);
        b.vt.setAttribute('x', 0); b.vt.setAttribute('y', (y - 8).toFixed(2));
        b.ft.setAttribute('x', 0); b.ft.setAttribute('y', L.base - 8);
      } else {
        const cy = L.m.t + L.row * (b.pos + .5);
        b.g.setAttribute('transform', `translate(0 ${cy.toFixed(2)})`);
        const h = Math.min(22, L.row * .56), y0 = -h / 2, y1 = h / 2, x0 = L.x0, x = L.x0 + len, r = Math.min(5, len / 2, h / 2);
        const d = len < .5 ? '' : `M${x0} ${y0}H${(x - r).toFixed(2)}Q${x.toFixed(2)} ${y0} ${x.toFixed(2)} ${y0 + r}V${y1 - r}Q${x.toFixed(2)} ${y1} ${(x - r).toFixed(2)} ${y1}H${x0}Z`;
        b.shape.setAttribute('d', d); b.edge.setAttribute('d', d);
        b.vt.setAttribute('x', (x + 6).toFixed(2)); b.vt.setAttribute('y', 4.5);
        b.ft.setAttribute('x', x0 + 6); b.ft.setAttribute('y', 4.5);
      }
    }

    function drawStatic(m) {
      const ym = ymaxOf(m);
      gGrid.innerHTML = ''; gTop.innerHTML = '';
      svg.setAttribute('viewBox', `0 0 ${L.W} ${L.H}`);
      svg.setAttribute('height', L.H);
      for (const v of tickVals(ym)) {
        if (L.v) {
          const y = L.base - v / ym * span();
          if (v > 0) svgEl('line', { x1: L.m.l, x2: L.W - L.m.r, y1: y, y2: y, class: 'gridl' }, gGrid);
          svgEl('text', { x: L.m.l - 9, y: y + 4, 'text-anchor': 'end', class: 'tick' }, gGrid).textContent = v.toFixed(1);
        } else {
          const x = L.x0 + v / ym * span();
          if (v > 0) svgEl('line', { y1: L.m.t - 6, y2: L.H - L.m.b, x1: x, x2: x, class: 'gridl' }, gGrid);
          svgEl('text', { x, y: L.m.t - 12, 'text-anchor': 'middle', class: 'tick' }, gGrid).textContent = v.toFixed(1);
        }
      }
      const t = m === 'index' ? null : SB.task(m);
      const cap = t ? '× human baseline' : 'Singularity-Index';
      if (L.v) svgEl('text', { x: L.m.l - 9, y: L.top - 18, 'text-anchor': 'start', class: 'tick', style: 'font-weight:600' }, gGrid).textContent = cap;
      const baseLab = t ? `Human baseline (${t.baseShort}) = ${t.humanText}` : 'Human baseline = 1.000';
      if (L.v) {
        svgEl('line', { x1: L.m.l, x2: L.W - L.m.r, y1: L.base, y2: L.base, class: 'axis' }, gGrid);
        const y = L.base - 1 / ym * span();
        svgEl('line', { x1: L.m.l, x2: L.W - L.m.r, y1: y, y2: y, class: 'base-l' }, gTop);
        svgEl('text', { x: L.W - L.m.r, y: y - 7, 'text-anchor': 'end', class: 'base-t' }, gTop).textContent = baseLab;
      } else {
        svgEl('line', { y1: L.m.t - 6, y2: L.H - L.m.b, x1: L.x0, x2: L.x0, class: 'axis' }, gGrid);
        const x = L.x0 + 1 / ym * span();
        svgEl('line', { y1: L.m.t - 6, y2: L.H - L.m.b, x1: x, x2: x, class: 'base-l' }, gTop);
      }
      if (!still) gGrid.animate?.([{ opacity: 0 }, { opacity: 1 }], { duration: 350 });
      return ym;
    }

    function labelsFor(m) {
      bars.forEach(b => {
        const s = m === 'index' ? b.a.index : b.a.score[m];
        const failed = s === null;
        b.vt.style.opacity = '';
        b.vt.textContent = failed ? '' : s;
        b.ft.textContent = failed ? (L.v ? '✕' : '✕ failed') : '';
        b.g.setAttribute('aria-label', `${b.a.name}: ${failed ? 'failed' : s}`);
      });
    }
    const ranks = m => {
      const order = [...AG].sort((x, y) => val(y, m) - val(x, m) || y.idx - x.idx);
      return new Map(order.map((a, i) => [a.key, i]));
    };

    function render(m, animate) {
      cancel.forEach(c => c()); cancel = [];
      const ym = drawStatic(m);
      const rk = ranks(m);
      labelsFor(m);
      bars.forEach(b => placeInner(b));
      const from = bars.map(b => ({ pos: b.pos, len: b.len }));
      const to = bars.map(b => ({ pos: rk.get(b.a.key), len: lenOf(val(b.a, m), ym) }));
      if (!animate) { bars.forEach((b, i) => { b.pos = to[i].pos; b.len = to[i].len; drawBar(b); }); return; }
      cancel.push(tween(800, p => {
        bars.forEach((b, i) => { b.pos = from[i].pos + (to[i].pos - from[i].pos) * p; b.len = from[i].len + (to[i].len - from[i].len) * p; drawBar(b); });
      }, { ease: easeInOut }));
    }

    function enter() {
      entered = true;
      const ym = ymaxOf(metric), rk = ranks(metric);
      bars.forEach(b => { b.pos = rk.get(b.a.key); b.len = 0; drawBar(b); b.vt.style.opacity = 0; });
      bars.forEach((b, i) => {
        const target = lenOf(val(b.a, metric), ym);
        const final = metric === 'index' ? b.a.idx : null;
        cancel.push(tween(950, (p, raw) => {
          b.len = target * p; drawBar(b);
          b.vt.style.opacity = Math.min(1, raw * 2.2);
          if (final !== null && raw < 1) b.vt.textContent = (final * Math.min(1, raw * 1.15)).toFixed(3);
          if (raw >= 1 && final !== null) b.vt.textContent = b.a.index;
        }, { delay: 120 + b.pos * 75, ease: easeBack }));
      });
    }

    sub.textContent = 'Mean of the six task scores, each relative to its human baseline. 1.0 matches the human baselines on average; a failed task counts as zero.';

    // tooltip
    bars.forEach(b => {
      const show = () => {
        plot.classList.add('dim'); b.g.classList.add('hot');
        const a = b.a, m = metric;
        let body;
        if (m === 'index') body = `<div class="row"><span>Singularity-Index</span><span>${a.index}</span></div><div class="row"><span>API cost, all six tasks</span><span>${money(a.total)}</span></div>`;
        else {
          const t = SB.task(m), s = a.score[m];
          body = `<div class="row"><span>${t.metric}</span><span>${s === null ? 'failed' : s}</span></div>` +
            `<div class="row"><span>Relative to human (${t.humanText})</span><span>${s === null ? '0' : a.ratio[m].toFixed(2)}×</span></div>` +
            `<div class="row"><span>API cost on this task</span><span>${money(a.taskCost[m])}</span></div>`;
        }
        tip.innerHTML = head(a) + body;
        const box = b.shape.getBBox();
        const ctm = b.g.transform.baseVal.consolidate();
        const tx = ctm ? ctm.matrix.e : 0, ty = ctm ? ctm.matrix.f : 0;
        if (L.v) tipPlace(plot, tip, tx, Math.max(16, (b.len ? box.y : L.base) - 18));
        else tipPlace(plot, tip, Math.min(L.x0 + b.len, L.W - 120), ty - 14);
      };
      const hide = () => { plot.classList.remove('dim'); b.g.classList.remove('hot'); tip.hidden = true; };
      b.hit.addEventListener('mouseenter', show);
      b.hit.addEventListener('mouseleave', hide);
      b.hit.addEventListener('touchstart', show, { passive: true });
    });
    plot.addEventListener('mouseleave', () => { tip.hidden = true; plot.classList.remove('dim'); });

    L = layout();
    render(metric, false);
    bars.forEach(b => { b.len = 0; drawBar(b); });
    onceVisible(plot, enter);
    addEventListener('resize', debounce(() => {
      const n = layout();
      if (n.W === L.W) return;
      L = n;
      render(metric, false);
    }, 150));
  }

  /* ================= performance vs cost ================= */
  function CostChart(root) {
    const plot = root.querySelector('.plot');
    const svg = plot.querySelector('svg');
    const tip = plot.querySelector('.tip');
    const chips = root.querySelector('[data-pc-chips]');
    const sub = root.querySelector('[data-pc-sub]');
    const failures = root.querySelector('[data-pc-failed]');
    let metric = 'all', L = null, entered = false, cancel = [];
    const gGrid = svgEl('g', {}, svg), gFront = svgEl('g', {}, svg), gPts = svgEl('g', {}, svg), gLab = svgEl('g', {}, svg);
    const clip = svgEl('clipPath', { id: 'pcclip' }, svgEl('defs', {}, svg));
    const clipRect = svgEl('rect', { x: 0, y: 0, width: 0, height: 2000 }, clip);
    const TICKS = [.05, .1, .2, .5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000];
    const fmtT = v => (v < 1 ? `$${v}` : `$${v}`);

    const pts = AG.map(a => {
      const g = svgEl('g', { class: 'pt' }, gPts);
      const tri = a.harness === 'Codex';
      const mk = svgEl('path', { d: tri ? 'M0 -10 L9 6 L-9 6 Z' : 'M-7.5 0a7.5 7.5 0 1 0 15 0a7.5 7.5 0 1 0 -15 0', fill: a.color, stroke: '#fff', 'stroke-width': 2, 'stroke-linejoin': 'round' }, g);
      const hit = svgEl('circle', { r: 16, class: 'hit' }, g);
      return { a, g, mk, hit, x: 0, y: 0 };
    });

    const dataFor = m => AG.map(a => ({ a, x: m === 'all' ? a.cost : a.taskCost[m], y: m === 'all' ? a.idx : (a.score[m] === null ? null : +a.score[m]), failed: m !== 'all' && a.score[m] === null }));
    const baselineFor = m => m === 'all' ? 1 : SB.task(m).human;
    const baselineLabel = m => m === 'all' ? 'Human baseline = 1' : L.small ? `Human baseline = ${SB.task(m).humanText}` : `Human baseline (${SB.task(m).baseShort}) = ${SB.task(m).humanText}`;
    const lowerBetter = m => m !== 'all' && !!SB.task(m).lower;
    const yLabel = m => {
      if (m === 'all') return 'Singularity-Index';
      const t = SB.task(m);
      const name = m === 'pre' ? 'Perplexity (PPL)' : m === 'inf' ? 'Income (USD / GPU-h)' : t.metric;
      return `${name} · ${t.lower ? 'lower is better ↓' : 'higher is better ↑'}`;
    };
    function layout() {
      const W = Math.max(300, plot.clientWidth), small = W < 640;
      const m = { l: small ? 66 : 76, r: small ? 14 : 28, t: 20, b: 58 };
      const H = small ? 400 : 470;
      return { W, H, m, small, x0: m.l, x1: W - m.r, y0: H - m.b, y1: m.t };
    }
    let sx, sy, dom;
    function scales(m) {
      const d = dataFor(m);
      const xs = d.map(p => p.x);
      let lo = Math.min(...xs) / 1.7, hi = Math.max(...xs) * 1.9;
      let ymin = 0, ymax = 1.45, step = .2;
      if (m !== 'all') {
        const values = d.filter(p => !p.failed).map(p => p.y).concat(baselineFor(m));
        const min = lowerBetter(m) ? Math.min(...values) : 0;
        const max = Math.max(...values);
        const padding = (max - min || max || 1) * .1;
        const bottom = lowerBetter(m) ? Math.max(0, min - padding) : 0;
        const top = max + padding;
        const rawStep = (top - bottom) / 6;
        const power = 10 ** Math.floor(Math.log10(rawStep));
        step = [1, 2, 2.5, 5, 10].find(n => n * power >= rawStep) * power;
        ymin = Math.floor(bottom / step) * step;
        ymax = Math.ceil(top / step) * step;
      }
      const power = 10 ** Math.floor(Math.log10(step));
      const decimals = Math.max(0, -Math.floor(Math.log10(step)) + (Math.abs(step / power - 2.5) < 1e-9 ? 1 : 0));
      dom = { lo, hi, ymin, ymax, step, decimals };
      const llo = Math.log10(lo), lhi = Math.log10(hi);
      sx = v => L.x0 + (Math.log10(v) - llo) / (lhi - llo) * (L.x1 - L.x0);
      sy = v => L.y0 - (v - ymin) / (ymax - ymin) * (L.y0 - L.y1);
      return d;
    }
    function drawStatic(m) {
      gGrid.innerHTML = '';
      svg.setAttribute('viewBox', `0 0 ${L.W} ${L.H}`);
      svg.setAttribute('height', L.H);
      let ticks = TICKS.filter(v => v >= dom.lo && v <= dom.hi);
      if (L.small && ticks.length > 6) ticks = ticks.filter((v, i) => i % 2 === (ticks.length % 2 ? 0 : 1));
      ticks.forEach(v => {
        const x = sx(v);
        svgEl('line', { x1: x, x2: x, y1: L.y1, y2: L.y0, class: 'gridl' }, gGrid);
        svgEl('text', { x, y: L.y0 + 19, 'text-anchor': 'middle', class: 'tick' }, gGrid).textContent = fmtT(v);
      });
      for (let i = 0; i <= Math.floor((dom.ymax - dom.ymin) / dom.step + 1e-9); i++) {
        const v = dom.ymin + i * dom.step;
        const y = sy(v);
        if (v > 0) svgEl('line', { x1: L.x0, x2: L.x1, y1: y, y2: y, class: 'gridl' }, gGrid);
        svgEl('text', { x: L.x0 - 9, y: y + 4, 'text-anchor': 'end', class: 'tick' }, gGrid).textContent = v.toFixed(dom.decimals);
      }
      svgEl('line', { x1: L.x0, x2: L.x1, y1: L.y0, y2: L.y0, class: 'axis' }, gGrid);
      const yb = sy(baselineFor(m));
      svgEl('line', { x1: L.x0, x2: L.x1, y1: yb, y2: yb, class: 'base-l' }, gGrid);
      svgEl('text', { x: L.x1 - 4, y: yb - 7, 'text-anchor': 'end', class: 'base-t' }, gGrid).textContent = baselineLabel(m);
      const xt = L.small ? 'Agent API cost (USD, log scale)' : m === 'all' ? 'Total agent API cost, all six tasks (USD, log scale)' : `Agent API cost on ${SB.task(m).name.toLowerCase()} (USD, log scale)`;
      svgEl('text', { x: (L.x0 + L.x1) / 2, y: L.H - 12, 'text-anchor': 'middle', class: 'axl' }, gGrid).textContent = xt;
      const yt = yLabel(m);
      svgEl('text', { x: 0, y: 0, transform: `translate(${L.small ? 12 : 16} ${(L.y0 + L.y1) / 2}) rotate(-90)`, 'text-anchor': 'middle', class: 'axl' }, gGrid).textContent = yt;
      svg.setAttribute('aria-label', `${yt} against agent API cost. ${m === 'all' ? '' : 'Only valid submissions are plotted.'}`);
      if (!still) gGrid.animate?.([{ opacity: 0 }, { opacity: 1 }], { duration: 350 });
    }
    function frontier(d, lower = lowerBetter(metric)) {
      const ok = d.filter(p => !p.failed && Number.isFinite(p.y)).sort((p, q) => p.x - q.x || (lower ? p.y - q.y : q.y - p.y));
      const out = [];
      let best = lower ? Infinity : -Infinity;
      ok.forEach(p => { if (lower ? p.y < best : p.y > best) { out.push(p); best = p.y; } });
      return out;
    }
    function drawFrontier(d, animate) {
      gFront.innerHTML = '';
      const f = frontier(d);
      if (f.length < 2) return;
      const dd = f.map((p, i) => `${i ? 'L' : 'M'}${sx(p.x).toFixed(1)} ${sy(p.y).toFixed(1)}`).join('');
      svgEl('path', { d: dd, fill: 'none', stroke: '#252525', 'stroke-width': 2.6, 'stroke-dasharray': '.6 5.4', 'stroke-linecap': 'round', 'clip-path': 'url(#pcclip)' }, gFront);
      const x0 = sx(f[0].x) - 4, x1 = sx(f[f.length - 1].x) + 4;
      clipRect.setAttribute('x', x0);
      if (!animate) { clipRect.setAttribute('width', x1 - x0); return; }
      clipRect.setAttribute('width', 0);
      cancel.push(tween(900, p => clipRect.setAttribute('width', (x1 - x0) * p), { ease: easeInOut }));
    }
    // greedy label placement around each point
    function drawLabels(d) {
      gLab.innerHTML = '';
      const placed = [];
      const pbox = d.filter(p => !p.failed).map(p => ({ x0: sx(p.x) - 10, x1: sx(p.x) + 10, y0: sy(p.y) - 11, y1: sy(p.y) + 8 }));
      // the frontier line and the baseline label are obstacles too
      const f = frontier(d);
      for (let i = 1; i < f.length; i++) {
        const ax = sx(f[i - 1].x), ay = sy(f[i - 1].y), bx = sx(f[i].x), by = sy(f[i].y);
        const n = Math.ceil(Math.hypot(bx - ax, by - ay) / 6);
        for (let k = 0; k <= n; k++) { const x = ax + (bx - ax) * k / n, y = ay + (by - ay) * k / n; pbox.push({ x0: x - 2, x1: x + 2, y0: y - 2, y1: y + 2 }); }
      }
      pbox.push({ x0: L.x1 - baselineLabel(metric).length * 6.2, x1: L.x1, y0: sy(baselineFor(metric)) - 20, y1: sy(baselineFor(metric)) - 2 });
      const hits = (b) => b.x0 < L.x0 - 6 || b.x1 > L.x1 + 2 || b.y0 < L.y1 - 14 || b.y1 > L.y0 - 2 || placed.some(o => !(b.x1 < o.x0 || b.x0 > o.x1 || b.y1 < o.y0 || b.y0 > o.y1)) || pbox.some(o => !(b.x1 < o.x0 || b.x0 > o.x1 || b.y1 < o.y0 || b.y0 > o.y1));
      const order = d.filter(p => !p.failed).sort((p, q) => (lowerBetter(metric) ? p.y - q.y : q.y - p.y) || p.x - q.x);
      order.forEach(p => {
        if (p.failed) return;
        const x = sx(p.x), y = sy(p.y);
        const name = p.a.model, value = metricVal(p);
        const w = Math.max(name.length * 6.7, value.length * 7.4) + 4;
        const cands = [[13, -3, 'start'], [-13, -3, 'end'], [0, -27, 'middle'], [0, 21, 'middle'], [11, -21, 'start'], [11, 15, 'start'], [-11, -21, 'end'], [-11, 15, 'end']];
        for (const [dx, dy, an] of cands) {
          const lx = x + dx, ly = y + dy;
          const bx0 = an === 'start' ? lx : an === 'end' ? lx - w : lx - w / 2;
          const box = { x0: bx0, x1: bx0 + w, y0: ly - 11, y1: ly + 17 };
          if (hits(box)) continue;
          placed.push(box);
          const t = svgEl('text', { x: lx, y: ly, 'text-anchor': an, class: 'nm' }, gLab);
          const s1 = svgEl('tspan', { x: lx }, t); s1.textContent = name;
          const s2 = svgEl('tspan', { x: lx, dy: 15, 'font-weight': 700 }, t); s2.textContent = value;
          break;
        }
      });
      if (!still) gLab.animate?.([{ opacity: 0 }, { opacity: 1 }], { duration: 300 });
    }
    const metricVal = p => (metric === 'all' ? p.a.index : `${p.a.score[metric]}`);

    function render(m, animate) {
      cancel.forEach(c => c()); cancel = [];
      const d = scales(m);
      drawStatic(m);
      gLab.innerHTML = '';
      const from = pts.map(p => ({ x: p.x, y: p.y }));
      const to = d.map((q, i) => q.failed ? from[i] : ({ x: sx(q.x), y: sy(q.y) }));
      pts.forEach((p, i) => {
        const f = d[i].failed;
        p.g.style.display = f ? 'none' : '';
        p.mk.setAttribute('fill', f ? '#fff' : p.a.color);
        p.mk.setAttribute('stroke', f ? p.a.color : '#fff');
        p.g.style.opacity = f ? .55 : 1;
      });
      if (failures) {
        const failed = d.filter(p => p.failed);
        failures.hidden = failed.length === 0;
        failures.textContent = failed.length ? `No valid metric (not plotted): ${failed.map(p => `${p.a.name} (${money(p.x)})`).join('; ')}.` : '';
      }
      const fin = () => { drawFrontier(d, animate); drawLabels(d); };
      if (!animate) { pts.forEach((p, i) => { p.x = to[i].x; p.y = to[i].y; p.g.setAttribute('transform', `translate(${p.x.toFixed(1)} ${p.y.toFixed(1)})`); }); fin(); return; }
      gFront.innerHTML = '';
      cancel.push(tween(750, q => {
        pts.forEach((p, i) => { p.x = from[i].x + (to[i].x - from[i].x) * q; p.y = from[i].y + (to[i].y - from[i].y) * q; p.g.setAttribute('transform', `translate(${p.x.toFixed(1)} ${p.y.toFixed(1)})`); });
      }, { ease: easeInOut, done: fin }));
    }

    function enter() {
      entered = true;
      const d = scales(metric);
      drawStatic(metric);
      pts.forEach((p, i) => {
        if (d[i].failed) return;
        p.x = sx(d[i].x); p.y = sy(d[i].y);
        p.g.setAttribute('transform', `translate(${p.x.toFixed(1)} ${p.y.toFixed(1)}) scale(0)`);
      });
      let left = pts.length;
      pts.forEach((p, i) => {
        cancel.push(tween(600, q => p.g.setAttribute('transform', `translate(${p.x.toFixed(1)} ${(p.y - 26 * (1 - q)).toFixed(1)}) scale(${q.toFixed(3)})`), {
          delay: 80 + i * 70, ease: easeBack,
          done: () => { if (--left === 0) { drawFrontier(d, true); drawLabels(d); } },
        }));
      });
    }

    chips.innerHTML = [['all', 'All six tasks', null], ...LIFE.map(k => [k, SB.task(k).name, SB.task(k).color])]
      .map(([k, n, c]) => `<button class="chip" type="button" data-m="${k}" aria-pressed="${k === 'all'}"${c ? ` style="--c:${c}"` : ''}>${c ? '<i></i>' : ''}${n}</button>`).join('');
    const subText = m => (m === 'all'
      ? 'Singularity-Index against total agent API spend across all six tasks. The dotted line is the cost–index Pareto frontier.'
      : `${SB.task(m).name}: ${SB.task(m).metric} against agent API spend on this task. ${lowerBetter(m) ? 'Lower perplexity is better: the Pareto frontier favors lower cost and lower perplexity (bottom-left).' : 'Higher values are better: the Pareto frontier favors lower cost and higher performance (top-left).'} Only valid submissions are plotted.`);
    sub.textContent = subText(metric);
    chips.addEventListener('click', e => {
      const btn = e.target.closest('.chip');
      if (!btn || btn.dataset.m === metric) return;
      metric = btn.dataset.m;
      chips.querySelectorAll('.chip').forEach(c => c.setAttribute('aria-pressed', c === btn));
      sub.textContent = subText(metric);
      tip.hidden = true;
      render(metric, entered);
    });

    pts.forEach(p => {
      const show = () => {
        const a = p.a;
        let body;
        if (metric === 'all') {
          body = `<div class="row"><span>Singularity-Index</span><span>${a.index}</span></div><div class="row"><span>API cost, all six tasks</span><span>${money(a.total)}</span></div><div class="mini">` +
            LIFE.map(k => { const r = a.ratio[k], t = SB.task(k); return `<span>${t.name}</span><span class="tr" style="--h:50%"><i style="--c:${t.color};width:${Math.min(100, r / 2 * 100).toFixed(1)}%"></i></span><span>${a.score[k] === null ? '✕' : r.toFixed(2) + '×'}</span>`; }).join('') + '</div>';
        } else {
          const t = SB.task(metric), s = a.score[metric];
          body = `<div class="row"><span>${t.metric}${t.lower ? ' (lower is better)' : ''}</span><span>${s === null ? 'failed' : s}</span></div><div class="row"><span>Human baseline (${t.baseShort})</span><span>${t.humanText}</span></div><div class="row"><span>API cost on this task</span><span>${money(a.taskCost[metric])}</span></div>`;
        }
        tip.innerHTML = head(a) + body;
        tipPlace(plot, tip, p.x, p.y - 16);
        p.mk.setAttribute('stroke-width', 3.5);
      };
      const hide = () => { tip.hidden = true; p.mk.setAttribute('stroke-width', 2); };
      p.hit.addEventListener('mouseenter', show);
      p.hit.addEventListener('mouseleave', hide);
      p.hit.addEventListener('touchstart', show, { passive: true });
    });

    L = layout();
    render(metric, false);
    if (!still) {
      pts.forEach(p => p.g.setAttribute('transform', `translate(${p.x.toFixed(1)} ${p.y.toFixed(1)}) scale(0)`));
      gFront.innerHTML = '';
      gLab.innerHTML = '';
    }
    onceVisible(plot, enter);
    addEventListener('resize', debounce(() => { const n = layout(); if (n.W === L.W) return; L = n; render(metric, false); }, 150));
  }

  /* ================= inference reasoning effort ================= */
  function EffortChart(root) {
    const plot = root.querySelector('.plot');
    const svg = plot.querySelector('svg');
    const tip = plot.querySelector('.tip');
    const grid = svgEl('g', {}, svg);
    const clip = svgEl('clipPath', { id: 'effort-reveal' }, svgEl('defs', {}, svg));
    const reveal = svgEl('rect', {}, clip);
    const lines = svgEl('g', { 'clip-path': 'url(#effort-reveal)' }, svg);
    const points = svgEl('g', {}, svg);
    let L, progress = still ? 1 : 0, entered = false, stop = () => {}, active = null;
    const series = SB.EFFORT_RUNS.map((s, i) => {
      const color = SB.DEV[s.dev].color;
      const line = svgEl('path', { fill: 'none', stroke: color, 'stroke-width': 2.5,
        'stroke-dasharray': i ? '6 4' : '', 'stroke-linejoin': 'round' }, lines);
      const runs = [...s.runs].sort((a, b) => a.cost - b.cost).map(run => {
        const group = svgEl('g', { class: 'effort-point' }, points);
        const marker = svgEl('path', { class: 'effort-marker',
          d: i ? 'M0 -7L6.5 5.5H-6.5Z' : 'M-6 0a6 6 0 1 0 12 0a6 6 0 1 0 -12 0',
          fill: color, stroke: '#fff', 'stroke-width': 1.8, 'stroke-linejoin': 'round' }, group);
        const hit = svgEl('circle', { r: 12, class: 'hit', tabindex: 0, role: 'button',
          'aria-label': `${s.agent}, ${run.effort} reasoning effort: performance ${run.performance.toFixed(2)} USD per GPU-hour, API cost ${money(run.cost)}` }, group);
        return { ...run, group, marker, hit, series: s, x: 0, y: 0 };
      });
      return { line, runs };
    });
    const all = series.flatMap(s => s.runs);
    const sx = cost => L.x0 + Math.log(cost / 14) / Math.log(165 / 14) * (L.x1 - L.x0);
    const sy = score => L.y0 - score / 13 * (L.y0 - L.y1);

    function revealTo(p) {
      progress = p;
      const edge = L.x0 - 14 + p * (L.x1 - L.x0 + 28);
      reveal.setAttribute('width', Math.max(0, edge - (L.x0 - 14)));
      all.forEach(point => {
        const pop = Math.max(0, Math.min(1, (edge - point.x) / 12));
        point.marker.setAttribute('transform', `scale(${pop})`);
        point.group.style.opacity = pop;
        point.hit.style.pointerEvents = pop > .8 ? '' : 'none';
      });
    }
    function hide() {
      tip.hidden = true;
      active?.hit.removeAttribute('aria-describedby');
      if (active) active.marker.setAttribute('stroke-width', 1.8);
      active = null;
    }
    function show(point) {
      hide();
      stop(); revealTo(1);
      active = point;
      tip.innerHTML = `<div class="th"><img src="${SB.DEV[point.series.dev].logo}" alt="">${point.series.agent}</div>` +
        `<div class="row"><span>Reasoning effort</span><span>${point.effort}</span></div>` +
        `<div class="row"><span>Performance</span><span>$${point.performance.toFixed(2)} / GPU-hour</span></div>` +
        `<div class="row"><span>Agent API cost</span><span>${money(point.cost)}</span></div>`;
      const box = svg.getBoundingClientRect();
      const x = point.x * box.width / L.W, y = point.y * box.height / L.H;
      tip.hidden = false;
      tip.style.transform = 'translateX(-50%)';
      tipPlace(plot, tip, x, y);
      const top = y - tip.offsetHeight - 16;
      tip.style.top = `${top < 4 ? y + 18 : top}px`;
      point.hit.setAttribute('aria-describedby', tip.id);
      point.marker.setAttribute('stroke-width', 3);
    }
    function draw() {
      const W = Math.max(280, plot.clientWidth), small = W < 480;
      L = { W, H: small ? 330 : 360, x0: 40, x1: W - 18, y1: 34, y0: small ? 278 : 308 };
      svg.setAttribute('viewBox', `0 0 ${W} ${L.H}`);
      svg.setAttribute('height', L.H);
      grid.innerHTML = '';
      [0, 2, 4, 6, 8, 10, 12].forEach(value => {
        const y = sy(value);
        svgEl('line', { x1: L.x0, x2: L.x1, y1: y, y2: y, class: value ? 'gridl' : 'axis' }, grid);
        svgEl('text', { x: L.x0 - 9, y: y + 4, 'text-anchor': 'end', class: 'tick' }, grid).textContent = value;
      });
      (small ? [15, 30, 70, 150] : [15, 20, 30, 50, 70, 100, 150]).forEach(value => {
        const x = sx(value);
        svgEl('line', { x1: x, x2: x, y1: L.y1, y2: L.y0, class: 'gridl' }, grid);
        svgEl('text', { x, y: L.y0 + 20, 'text-anchor': 'middle', class: 'tick' }, grid).textContent = `$${value}`;
      });
      svgEl('text', { x: L.x0, y: 14, class: 'axl' }, grid).textContent = 'Serving income (USD / GPU-hour)';
      svgEl('text', { x: (L.x0 + L.x1) / 2, y: L.H - 6, 'text-anchor': 'middle', class: 'axl' }, grid).textContent = 'Agent API cost (USD, log scale)';
      const baseline = sy(SB.task('inf').human);
      svgEl('line', { x1: L.x0, x2: L.x1, y1: baseline, y2: baseline, class: 'base-l' }, grid);
      svgEl('text', { x: L.x1, y: baseline - 8, 'text-anchor': 'end', class: 'base-t' }, grid).textContent = 'SGLang baseline = 8.18';
      series.forEach(s => {
        s.runs.forEach(point => {
          point.x = sx(point.cost); point.y = sy(point.performance);
          point.group.setAttribute('transform', `translate(${point.x} ${point.y})`);
        });
        s.line.setAttribute('d', s.runs.map((p, i) => `${i ? 'L' : 'M'}${p.x} ${p.y}`).join(' '));
      });
      reveal.setAttribute('x', L.x0 - 14);
      reveal.setAttribute('y', 0);
      reveal.setAttribute('height', L.H);
      revealTo(progress);
      hide();
    }
    all.forEach(point => {
      point.hit.addEventListener('pointerenter', e => { if (e.pointerType !== 'touch') show(point); });
      point.hit.addEventListener('pointerleave', () => { if (document.activeElement !== point.hit) hide(); });
      point.hit.addEventListener('focus', () => show(point));
      point.hit.addEventListener('blur', hide);
      point.hit.addEventListener('click', () => show(point));
      point.hit.addEventListener('keydown', e => {
        if (e.key === 'Escape') { hide(); e.stopPropagation(); }
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); show(point); }
      });
    });
    document.addEventListener('pointerdown', e => { if (!root.contains(e.target)) hide(); });
    draw();
    onceVisible(plot, () => {
      if (entered) return;
      entered = true;
      stop = tween(2000, revealTo, { ease: easeInOut });
    });
    const resize = debounce(() => { if (Math.max(280, plot.clientWidth) !== L.W) draw(); }, 100);
    if ('ResizeObserver' in window) new ResizeObserver(resize).observe(plot);
    else addEventListener('resize', resize);
  }

  /* ================= full results table ================= */
  function ResultsTable(table) {
    const COLS = SB.COLS;
    const human = { s: COLS.map(k => SB.task(k).humanText), index: '1.000' };
    const better = (k, x, y) => (SB.task(k).lower ? x < y : x > y);
    const rankCol = k => {
      const vals = AG.map(a => a.score[k]).concat([human.s[COLS.indexOf(k)]]).filter(v => v !== null).map(Number);
      const uniq = [...new Set(vals)].sort((x, y) => (SB.task(k).lower ? x - y : y - x));
      return { best: uniq[0], second: uniq[1] };
    };
    const ranks = Object.fromEntries(COLS.map(k => [k, rankCol(k)]));
    const costs = AG.map(a => a.cost).sort((x, y) => x - y);
    const idxs = AG.map(a => a.idx).concat([1]).sort((x, y) => y - x);
    const cls = (v, r) => (v === r.best ? 'best' : v === r.second ? 'second' : '');
    const hdr = [['Pre-training', 'PPL ↓'], ['Mid-training', 'Acc ↑'], ['Post-training', 'Resolve ↑'], ['Inference', 'Income ↑'], ['Harness', 'Score ↑'], ['Cluster', 'SLO ↑']];
    let h = `<thead><tr><th>Agent</th>${hdr.map(([a, b]) => `<th>${a}<small>${b}</small></th>`).join('')}<th>Total<small>cost ↓</small></th><th>Singularity-<small>Index ↑</small></th></tr></thead><tbody>`;
    AG.forEach(a => {
      h += `<tr class="${a.prop ? 'prop' : ''}"><td><span class="an"><img src="${a.logo}" alt="">${a.model}<span class="h">+ ${a.harness}</span></span></td>`;
      COLS.forEach((k, j) => {
        const s = a.score[k];
        if (s === null) { h += `<td><span class="x">✕</span><span class="c">${money(a.c[j])}</span></td>`; return; }
        const beat = better(k, +s, +SB.task(k).humanText);
        h += `<td class="${cls(+s, ranks[k])}">${s}${beat ? '<span class="star">★</span>' : ''}<span class="c">${money(a.c[j])}</span></td>`;
      });
      h += `<td class="${a.cost === costs[0] ? 'best' : a.cost === costs[1] ? 'second' : ''}">${money(a.total)}</td>`;
      h += `<td class="${a.idx === idxs[0] ? 'best' : a.idx === idxs[1] ? 'second' : ''}"><b>${a.index}</b></td></tr>`;
    });
    h += `<tr class="human"><td>Human baseline</td>${COLS.map((k, j) => `<td class="${cls(+human.s[j], ranks[k])}">${human.s[j]}</td>`).join('')}<td>—</td><td class="${idxs[1] === 1 ? 'second' : ''}">1.000</td></tr></tbody>`;
    table.innerHTML = h;
  }

  SB.charts = { Leaderboard, CostChart, EffortChart, ResultsTable, tween, onceVisible };
})();
