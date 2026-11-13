/* Decorative and animated art: hero technique search, lifecycle loop, task-card art, locked tiles. */
(function () {
  'use strict';
  const SB = window.SB;
  const NS = 'http://www.w3.org/2000/svg';
  const still = matchMedia('(prefers-reduced-motion: reduce)').matches || /[?&]still\b/.test(location.search);
  SB.still = still;

  const svgEl = (tag, attrs = {}, parent) => {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    if (parent) parent.appendChild(e);
    return e;
  };
  SB.svgEl = svgEl;
  // mix a hex color with white: amt = share of the color
  const tint = (hex, amt) => {
    const n = parseInt(hex.slice(1), 16), c = [n >> 16, (n >> 8) & 255, n & 255];
    return '#' + c.map(v => Math.round(255 - (255 - v) * amt).toString(16).padStart(2, '0')).join('');
  };
  SB.tint = tint;

  /* ---------- task art (white strokes on the task color) ---------- */
  const W = 'fill="#fff"', S = 'stroke="#fff"';
  const dots = (op = .22) => {
    let d = '';
    for (let x = 12; x < 400; x += 22) for (let y = 12; y < 150; y += 22) d += `M${x} ${y}h.01`;
    return `<path d="${d}" ${S} stroke-opacity="${op}" stroke-width="2.6" stroke-linecap="round"/>`;
  };
  const ART = {
    pre() {
      const L = [[170, [38, 75, 112]], [245, [26, 60, 94, 128]], [320, [56, 98]]];
      let s = dots();
      for (let i = 0; i < L.length - 1; i++)
        for (const y1 of L[i][1]) for (const y2 of L[i + 1][1])
          s += `<path d="M${L[i][0]} ${y1}L${L[i + 1][0]} ${y2}" ${S} stroke-opacity=".38" stroke-width="2"/>`;
      L.forEach(([x, ys], i) => ys.forEach((y, j) => {
        const hot = (i + j) % 3 === 0;
        s += `<circle cx="${x}" cy="${y}" r="${hot ? 9 : 7}" ${W} fill-opacity="${hot ? 1 : .7}"/>`;
      }));
      return s + `<path d="M350 118c10-30 22-48 40-56" ${S} stroke-opacity=".6" stroke-width="3" fill="none" stroke-dasharray="2 7" stroke-linecap="round"/>`;
    },
    mid() {
      const rows = [[.32, .18, .3, .2], [.22, .4, .14, .24], [.5, .12, .2, .18], [.16, .26, .36, .22], [.28, .3, .12, .3]];
      let s = dots(.18);
      rows.forEach((r, i) => {
        let x = 150;
        const y = 18 + i * 25;
        r.forEach((w, j) => {
          const ww = w * 228 - 4;
          s += `<rect x="${x.toFixed(1)}" y="${y}" width="${ww.toFixed(1)}" height="15" rx="7.5" ${W} fill-opacity="${[1, .62, .38, .2][(i + j) % 4]}"/>`;
          x += w * 228;
        });
      });
      return s;
    },
    post() {
      let s = dots(.18);
      [26, 40, 52, 70, 84, 104].forEach((h, i) => { s += `<rect x="${150 + i * 18}" y="${132 - h}" width="11" height="${h}" rx="4" ${W} fill-opacity="${.35 + i * .11}"/>`; });
      s += `<circle cx="318" cy="76" r="46" fill="none" ${S} stroke-opacity=".45" stroke-width="6" stroke-dasharray="210 80" stroke-linecap="round" transform="rotate(-40 318 76)"/>`;
      s += `<path d="M352 34l12 4-4 12" fill="none" ${S} stroke-width="5" stroke-linecap="round" stroke-linejoin="round" stroke-opacity=".8"/>`;
      s += `<path d="m318 56 6 12.6 13.8 2-10 9.8 2.4 13.8-12.2-6.6-12.4 6.6 2.4-13.8-10-9.8 13.8-2z" ${W}/>`;
      return s;
    },
    inf() {
      let s = dots(.16);
      const rows = [[18, 26, 14, 34, 22, 30], [30, 16, 24, 18, 36], [14, 34, 20, 28, 16, 22], [24, 18, 32, 14, 26]];
      rows.forEach((r, i) => {
        let x = 130 + (i % 2) * 14;
        const y = 22 + i * 28;
        r.forEach((w, j) => { s += `<rect x="${x}" y="${y}" width="${w}" height="14" rx="7" ${W} fill-opacity="${(.18 + j * .14).toFixed(2)}"/>`; x += w + 7; });
      });
      s += `<rect x="326" y="36" width="58" height="78" rx="10" fill="none" ${S} stroke-width="4" stroke-opacity=".85"/>`;
      for (let k = 0; k < 4; k++) s += `<path d="M${336 + k * 13} 28v8M${336 + k * 13} 114v8" ${S} stroke-width="3" stroke-linecap="round" stroke-opacity=".7"/>`;
      return s + `<path d="m359 52-12 22h14l-10 22" fill="none" ${S} stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>`;
    },
    clu() {
      let s = dots(.16);
      const busy = ['1111', '1101', '0111', '1011'];
      for (let n = 0; n < 4; n++) {
        const x0 = 150 + n * 60;
        s += `<rect x="${x0 - 6}" y="10" width="52" height="132" rx="10" fill="none" ${S} stroke-opacity=".45" stroke-width="2"/>`;
        for (let g = 0; g < 4; g++) {
          const on = busy[n][g] === '1';
          s += `<rect x="${x0}" y="${17 + g * 31}" width="40" height="25" rx="5" ${on ? `${W} fill-opacity="${(.55 + ((n + g) % 3) * .2).toFixed(2)}"` : `fill="none" ${S} stroke-opacity=".7" stroke-width="2"`}/>`;
        }
      }
      return s;
    },
    har() {
      let s = dots(.16);
      s += `<rect x="150" y="16" width="236" height="118" rx="12" ${W} fill-opacity=".14" ${S} stroke-opacity=".55" stroke-width="2"/>`;
      s += `<path d="M150 40h236" ${S} stroke-opacity=".45" stroke-width="2"/>`;
      [166, 180, 194].forEach(x => { s += `<circle cx="${x}" cy="28" r="4" ${W} fill-opacity=".8"/>`; });
      s += `<path d="m168 58 9 7-9 7" fill="none" ${S} stroke-width="3.4" stroke-linecap="round" stroke-linejoin="round"/>`;
      [[186, 58, 120, 1], [168, 82, 160, .6], [168, 100, 96, .6], [168, 118, 140, .4]].forEach(([x, y, w, o]) => { s += `<rect x="${x}" y="${y + 1}" width="${w}" height="10" rx="5" ${W} fill-opacity="${o}"/>`; });
      return s + `<rect class="cursor" x="312" y="57" width="9" height="14" rx="1.5" ${W}/>`;
    },
  };
  const artSVG = key => `<svg class="pat" viewBox="0 0 400 150" preserveAspectRatio="xMidYMid slice" aria-hidden="true">${ART[key]()}</svg>`;
  SB.artSVG = artSVG;

  function paintCards() {
    document.querySelectorAll('[data-art]').forEach(el => { el.insertAdjacentHTML('afterbegin', artSVG(el.dataset.art)); });
  }

  /* ---------- hero: an agent searching each phase's techniques ---------- */
  // Each column feeds technique cards up through a lens. The agent tries the card in the lens, keeps or drops
  // it, and moves on to the next; hovering or scrolling away pauses the search.
  function buildSearch() {
    const plane = document.getElementById('fplane');
    if (!plane) return;
    const frames = plane.closest('.frames') || plane;
    const order = ['pre', 'mid', 'post', 'inf', 'clu', 'har'];
    const SLOTS = 5, LENS = 2;                    // cards per column; the lens holds the middle one
    const TRY = 1300, HOLD = 1150, SLIDE = 650;   // ms: trying, showing the outcome, moving on
    const wait = ms => new Promise(res => setTimeout(res, ms));
    const paused = () => document.hidden || frames.classList.contains('off') || frames.matches(':hover');
    const free = async () => { while (paused()) await wait(300); };
    plane.style.setProperty('--try', `${TRY}ms`);   // the progress bar fills over the trying time

    const reels = order.map((k, i) => {
      const t = SB.task(k);
      const list = SB.TECHNIQUES[k];
      const reel = document.createElement('div');
      reel.className = 'freel';
      reel.style.setProperty('--c', t.color);
      reel.innerHTML = `<div class="freel-label"><span>0${i + 1}</span>${t.name}</div>` +
        '<div class="fwindow"><div class="fcol"></div><div class="flens"><i></i><i></i><i></i><i></i></div></div>';
      plane.appendChild(reel);
      const col = reel.querySelector('.fcol');
      // stagger the columns' starting points, with a technique the agent keeps in the lens
      let next = (i * 3) % list.length;
      while (!list[(next + LENS) % list.length][2]) next = (next + 1) % list.length;
      const card = () => {
        const [name, kind, kept] = list[next];
        next = (next + 1) % list.length;
        const el = document.createElement('div');
        el.className = 'fcard';
        el.dataset.out = kept ? 'ok' : 'no';
        el.innerHTML = `<span class="ic">${SB.icon(k, 'i-task')}${SB.icon('spin', 'i-try')}${SB.icon('check', 'i-ok')}${SB.icon('x', 'i-no')}</span>` +
          `<span><b>${name}</b><small><i class="c-kind">${kind}</i><i class="c-try">Trying…</i><i class="c-ok">Kept</i><i class="c-no">Dropped</i></small></span>`;
        return el;
      };
      for (let s = 0; s < SLOTS; s++) col.appendChild(card());
      // the cards above the lens were tried before the page loaded
      [...col.children].slice(0, LENS).forEach(el => el.classList.add(el.dataset.out, 'past'));
      if (still) col.children[LENS].classList.add('ok');
      return { col, card, i, moving: false };
    });

    // center the lens card, or the one after it when shift = 1
    const place = (r, shift = 0, animate = false) => {
      const cards = r.col.children;
      const pitch = cards[1].offsetTop - cards[0].offsetTop;
      const y = (r.col.parentNode.clientHeight - cards[0].offsetHeight) / 2 - (LENS + shift) * pitch;
      r.col.style.transition = animate ? `transform ${SLIDE}ms cubic-bezier(.65, 0, .35, 1)` : 'none';
      r.col.style.transform = `translateY(${y.toFixed(1)}px)`;
    };
    reels.forEach(r => place(r));
    addEventListener('resize', () => reels.forEach(r => { if (!r.moving) place(r); }));
    if (still) return;

    reels.forEach(async r => {
      await wait(300 + r.i * 520);
      for (;;) {
        await free();
        const c = r.col.children[LENS];
        c.classList.add('try');
        await wait(TRY);
        c.classList.replace('try', c.dataset.out);
        await wait(HOLD);
        await free();
        c.classList.add('past');
        r.moving = true;
        place(r, 1, true);
        await wait(SLIDE + 40);
        r.col.firstElementChild.remove();
        r.col.appendChild(r.card());
        place(r);
        r.moving = false;
      }
    });
  }

  /* ---------- lifecycle loop ---------- */
  function buildLifecycle() {
    const svg = document.getElementById('lifesvg');
    if (!svg) return;
    const order = ['pre', 'mid', 'post', 'inf', 'clu', 'har'];
    const CX = 340, CY = 312, R = 190, NR = 40;
    const rad = deg => deg * Math.PI / 180;
    const angOf = i => rad(-90 + i * 60);
    const pt = (a, r = R) => [CX + r * Math.cos(a), CY + r * Math.sin(a)];
    const f = n => n.toFixed(2);
    const arcD = (a0, a1, r = R) => {
      const [x0, y0] = pt(a0, r), [x1, y1] = pt(a1, r);
      return `M${f(x0)} ${f(y0)}A${r} ${r} 0 ${a1 - a0 > Math.PI ? 1 : 0} 1 ${f(x1)} ${f(y1)}`;
    };
    svg.setAttribute('viewBox', '0 0 680 624');
    const defs = svgEl('defs', {}, svg);
    defs.innerHTML = '<filter id="glow" x="-80%" y="-80%" width="260%" height="260%"><feGaussianBlur stdDeviation="5" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>' +
      '<radialGradient id="core" cx="50%" cy="50%" r="50%"><stop offset="0" stop-color="#eef0ff"/><stop offset=".7" stop-color="#f7f7fd"/><stop offset="1" stop-color="#fff" stop-opacity="0"/></radialGradient>';

    svgEl('circle', { cx: CX, cy: CY, r: R - NR - 14, fill: 'url(#core)' }, svg);
    svgEl('circle', { cx: CX, cy: CY, r: R, fill: 'none', stroke: '#e3e6ec', 'stroke-width': 3 }, svg);
    // direction chevrons midway between nodes
    for (let i = 0; i < 6; i++) {
      const a = angOf(i) + rad(30);
      const [x, y] = pt(a);
      const deg = -90 + i * 60 + 30 + 90;
      svgEl('path', { d: 'M-4 -5 L2 0 L-4 5', transform: `translate(${f(x)} ${f(y)}) rotate(${deg})`, fill: 'none', stroke: '#b4bac6', 'stroke-width': 2, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }, svg);
    }
    // progress arcs from node i to node i+1, revealed as the pulse passes
    const gap = NR / R + 0.03;
    const arcs = order.map((k, i) => {
      const p = svgEl('path', { d: arcD(angOf(i) + gap, angOf(i) + rad(60) - gap), fill: 'none', stroke: SB.task(k).color, 'stroke-width': 5, 'stroke-linecap': 'round' }, svg);
      const len = p.getTotalLength();
      p.style.strokeDasharray = `${len} ${len}`;
      p.style.strokeDashoffset = still ? 0 : len;
      p.dataset.len = len;
      return p;
    });
    // comet trail and head
    const trails = [[.5, .12, 9], [.3, .25, 7], [.14, .5, 5.5]].map(([span, op, w]) => ({ span, el: svgEl('path', { fill: 'none', stroke: '#4f46e5', 'stroke-opacity': op, 'stroke-width': w, 'stroke-linecap': 'round' }, svg) }));
    const head = svgEl('circle', { r: 8, fill: '#fff', stroke: '#4f46e5', 'stroke-width': 4, filter: 'url(#glow)' }, svg);

    // nodes
    const nodes = order.map((k, i) => {
      const t = SB.task(k);
      const a = angOf(i);
      const [x, y] = pt(a);
      const g = svgEl('g', { class: 'node', transform: `translate(${f(x)} ${f(y)})`, tabindex: 0, role: 'button', 'aria-label': `${t.name}: show the task card` }, svg);
      g.style.setProperty('--c', t.color);
      const halo = svgEl('circle', { r: NR, fill: t.color, 'fill-opacity': 0, class: 'halo' }, g);
      const disc = svgEl('circle', { r: NR, fill: '#fff', stroke: t.color, 'stroke-width': 2.5, class: 'disc' }, g);
      const ic = svgEl('svg', { x: -14, y: -14, width: 28, height: 28, viewBox: '0 0 24 24', fill: 'none', stroke: t.color, 'stroke-width': 1.8, 'stroke-linecap': 'round', 'stroke-linejoin': 'round', class: 'nic' }, g);
      ic.innerHTML = SB.ICON[k];
      // label outside the ring
      const lx = Math.cos(a), ly = Math.sin(a);
      const lab = svgEl('g', {}, g);
      let tx, ty, anchor;
      if (Math.abs(lx) < .2) { tx = 0; ty = ly < 0 ? -NR - 36 : NR + 26; anchor = 'middle'; }
      else { tx = lx > 0 ? NR + 14 : -NR - 14; ty = -6; anchor = lx > 0 ? 'start' : 'end'; }
      svgEl('text', { x: tx, y: ty, 'text-anchor': anchor, 'font-size': 11, 'font-weight': 700, 'letter-spacing': '.09em', fill: '#80868b' }, lab).textContent = t.group.toUpperCase();
      svgEl('text', { x: tx, y: ty + 20, 'text-anchor': anchor, 'font-size': 18, 'font-weight': 600, fill: '#1f1f1f' }, lab).textContent = t.name;
      return { g, halo, disc, ic, t, i };
    });

    // center readout
    const center = svgEl('g', { 'text-anchor': 'middle' }, svg);
    svgEl('text', { x: CX, y: CY - 50, 'font-size': 12, 'font-weight': 700, 'letter-spacing': '.14em', fill: '#80868b' }, center).textContent = 'AGENT GENERATION';
    const ver = svgEl('text', { x: CX, y: CY + 18, 'font-size': 70, 'font-weight': 700, 'letter-spacing': '-.04em', fill: '#1f1f1f' }, center);
    const phase = svgEl('text', { x: CX, y: CY + 52, 'font-size': 17, 'font-weight': 600 }, center);
    const desc = svgEl('text', { x: CX, y: CY + 74, 'font-size': 13.5, fill: '#5f6368' }, center);

    const groups = document.querySelectorAll('.life-groups li');
    let gen = 1, active = -1, held = null;
    const setActive = i => {
      if (i === active) return;
      active = i;
      nodes.forEach((n, j) => {
        const on = j === i;
        n.disc.setAttribute('fill', on ? n.t.color : (j < i ? tint(n.t.color, .14) : '#fff'));
        n.ic.setAttribute('stroke', on ? '#fff' : n.t.color);
        n.g.classList.toggle('on', on);
      });
      const t = nodes[i].t;
      phase.textContent = t.name;
      phase.setAttribute('fill', t.color);
      desc.textContent = t.short;
      groups.forEach(li => li.classList.toggle('on', li.dataset.group === t.group));
    };
    const setVer = () => { ver.textContent = `v${gen}`; };

    nodes.forEach(n => {
      const go = () => {
        const card = document.querySelector(`.fc[data-task="${n.t.key}"]`);
        if (!card) return;
        card.scrollIntoView({ behavior: still ? 'auto' : 'smooth', block: 'center' });
        setTimeout(() => SB.flipCard && SB.flipCard(n.t.key, true), still ? 0 : 450);
      };
      n.g.addEventListener('click', go);
      n.g.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); go(); } });
      n.g.addEventListener('mouseenter', () => { held = n.i; setActive(n.i); });
      n.g.addEventListener('mouseleave', () => { held = null; });
      n.g.addEventListener('focus', () => { held = n.i; setActive(n.i); });
      n.g.addEventListener('blur', () => { held = null; });
    });

    if (still) {
      head.style.display = 'none';
      trails.forEach(tr => { tr.el.style.display = 'none'; });
      nodes.forEach(n => n.disc.setAttribute('fill', tint(n.t.color, .14)));
      ver.textContent = 'v1→v2';
      ver.setAttribute('font-size', 50);
      phase.textContent = 'Six phases, one loop';
      phase.setAttribute('fill', '#4f46e5');
      desc.textContent = 'each generation builds the next';
      return;
    }

    const DWELL = 950, TRAVEL = 850, STEP = DWELL + TRAVEL, LAP = STEP * 6;
    const ease = x => x < .5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2;
    let t0 = null, lastLap = 0, running = false, raf = 0, pausedAt = 0;
    setVer();
    const frame = now => {
      raf = requestAnimationFrame(frame);
      if (t0 === null) t0 = now;
      const e = now - t0;
      const lap = Math.floor(e / LAP);
      if (lap !== lastLap) {
        lastLap = lap;
        gen = lap + 1;
        setVer();
        ver.animate?.([{ transform: 'scale(1.25)', opacity: .4 }, { transform: 'scale(1)', opacity: 1 }], { duration: 450, easing: 'cubic-bezier(.3,1.5,.5,1)' });
        active = -1;
      }
      const le = e - lap * LAP;
      const seg = Math.min(5, Math.floor(le / STEP));
      const local = le - seg * STEP;
      let theta = angOf(seg), prog = 0;
      if (local > DWELL) { prog = ease((local - DWELL) / TRAVEL); theta = angOf(seg) + prog * rad(60); }
      arcs.forEach((p, j) => {
        const len = +p.dataset.len;
        const shown = j < seg ? 1 : j === seg ? prog : 0;
        p.style.strokeDashoffset = len * (1 - shown);
        p.style.opacity = seg === 5 && local > DWELL ? 1 - prog * .6 : 1;
      });
      const [hx, hy] = pt(theta);
      head.setAttribute('cx', f(hx));
      head.setAttribute('cy', f(hy));
      trails.forEach(tr => tr.el.setAttribute('d', arcD(theta - tr.span, theta)));
      if (held === null) setActive(local > DWELL && prog > .85 ? (seg + 1) % 6 : seg);
    };
    const start = () => { if (running) return; running = true; if (t0 !== null) t0 += performance.now() - pausedAt; raf = requestAnimationFrame(frame); };
    const stop = () => { if (!running) return; running = false; pausedAt = performance.now(); cancelAnimationFrame(raf); };
    new IntersectionObserver(es => es.forEach(en => (en.isIntersecting ? start() : stop())), { threshold: .15 }).observe(svg);
    document.addEventListener('visibilitychange', () => (document.hidden ? stop() : (svg.getBoundingClientRect().top < innerHeight && start())));
  }

  /* ---------- locked tiles ---------- */
  function buildLocked() {
    const box = document.getElementById('locked');
    if (!box) return;
    const diagnoses = SB.TASKS.filter(t => t.key !== 'mid').map(t =>
      t.key === 'pre' ? { ...t, name: 'Language modeling training' } : t);
    box.innerHTML = diagnoses.map((t, i) => `<div class="lk"><div class="lk-tile" style="--c:${t.color};--d:${(i * .35).toFixed(2)}s"><div class="lk-art">${artSVG(t.key)}</div><div class="lk-lock">${SB.icon('lock')}</div></div><span>${t.name}</span></div>`).join('');
  }

  SB.art = { paintCards, buildSearch, buildLifecycle, buildLocked };
})();
