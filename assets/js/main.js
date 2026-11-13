/* Page wiring: links, section nav, counters, task cards, red-teaming explorer, case studies. */
(function () {
  'use strict';
  const SB = window.SB;
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const still = SB.still;
  const { tween, onceVisible } = SB.charts;

  /* ---------- paper and code links ---------- */
  function links() {
    $$('[data-link]').forEach(a => {
      const url = SB.LINKS[a.dataset.link];
      if (url) {
        a.href = url;
        a.target = '_blank';
        a.rel = 'noopener noreferrer';
        a.removeAttribute('aria-disabled');
        $$('.soon', a).forEach(s => s.remove());
      } else {
        a.removeAttribute('href');
        a.setAttribute('aria-disabled', 'true');
        a.title = 'Coming soon';
      }
    });
    const mailto = `mailto:${SB.EMAILS.join(',')}`;
    $$('[data-mail]').forEach(el => {
      el.innerHTML = SB.EMAILS.map(e => `<a href="${mailto}">${e}</a>`).join(', ').replace(/, ([^,]*)$/, ', and $1');
    });
  }

  /* ---------- section nav ---------- */
  function subnav() {
    const nav = $('#subnav');
    if (!nav) return;
    const items = $$('a', nav).map(a => ({ a, sec: document.getElementById(a.getAttribute('href').slice(1)) })).filter(x => x.sec);
    const setOn = id => {
      items.forEach(({ a, sec }) => {
        const on = sec.id === id;
        a.classList.toggle('on', on);
        if (on) {
          a.setAttribute('aria-current', 'true');
          const r = a.getBoundingClientRect(), nr = nav.getBoundingClientRect();
          if (r.left < nr.left || r.right > nr.right) nav.scrollBy({ left: r.left - nr.left - 24, behavior: still ? 'auto' : 'smooth' });
        } else a.removeAttribute('aria-current');
      });
    };
    const io = new IntersectionObserver(es => {
      es.forEach(e => { if (e.isIntersecting) setOn(e.target.id); });
    }, { rootMargin: '-40% 0px -55% 0px' });
    items.forEach(({ sec }) => io.observe(sec));
  }

  /* ---------- counters ---------- */
  function countUp(el) {
    const target = el.dataset.count;
    const dec = (target.split('.')[1] || '').length;
    const end = parseFloat(target);
    const suffix = el.dataset.suffix || '';
    onceVisible(el, () => tween(1300, p => { el.textContent = (end * p).toFixed(dec) + suffix; }, { done: () => { el.textContent = target + suffix; } }), .6);
  }

  /* ---------- task cards ---------- */
  function cards() {
    const all = $$('.fc');
    const set = (card, on) => {
      card.classList.toggle('flipped', on);
      const b = $('.fc-hit', card);
      b.setAttribute('aria-pressed', on);
      $('.fc-front', card).inert = on;
      $('.fc-back', card).inert = !on;
    };
    all.forEach(card => {
      set(card, false);
      $('.fc-hit', card).addEventListener('click', () => set(card, !card.classList.contains('flipped')));
    });
    SB.flipCard = (key, on = true) => { const c = $(`.fc[data-task="${key}"]`); if (c) set(c, on); };
    const btn = $('#flipall');
    if (btn) btn.addEventListener('click', () => {
      const anyFront = all.some(c => !c.classList.contains('flipped'));
      all.forEach(c => set(c, anyFront));
      btn.textContent = anyFront ? 'Flip all back' : 'Flip all cards';
    });
  }

  /* ---------- red-teaming ---------- */
  function redteam() {
    const box = $('#owasp'), list = $('#atklist'), filt = $('#atkfilter');
    if (!box || !list) return;
    const disclosure = $('#attack-details'), toggleLabel = $('[data-atk-toggle]');
    const updateToggle = () => {
      if (disclosure && toggleLabel) toggleLabel.textContent = disclosure.open ? 'Collapse attack table' : 'Expand attack table';
    };
    if (disclosure) disclosure.addEventListener('toggle', updateToggle);
    updateToggle();
    const H = SB.ATTACKS_HARBOR, J = SB.ATTACKS_JUDGE;
    let tab = 'harbor', cat = null;
    const done = { harbor: false, judge: false };

    box.innerHTML = SB.OWASP.map(o => {
      if (o.na) return `<div class="ow na" title="Our benchmark has no inter-agent communication channel"><span class="id">${o.id}</span><span class="nm">${o.name}</span><span class="ct">Not applicable</span></div>`;
      const n = H.filter(a => a.cat === o.id).length;
      const judge = o.id === 'ASI01' || o.id === 'ASI06';
      const ct = [n ? `${n} vs. Harbor` : '', judge ? 'judge attacks' : ''].filter(Boolean).join(' · ');
      return `<button class="ow" type="button" data-cat="${o.id}" aria-pressed="false"><span class="dot" aria-hidden="true"></span><span class="id">${o.id}</span><span class="nm">${o.name}</span><span class="ct">${ct}</span></button>`;
    }).join('');

    const pill = (kind) => kind === 'succ'
      ? `<span class="pill succ">${SB.icon('alert')}Successful</span>`
      : kind === 'caught' ? `<span class="pill caught">${SB.icon('shield')}Caught</span>`
        : `<span class="pill wait">${SB.icon('spin')}Running</span>`;
    const row = (a, live) => {
      const t = SB.task(a.task);
      return `<div class="atk-r"><span class="aid">${a.id}</span><span class="ad">${a.desc}</span><span class="tag" style="--c:${t.color}"><i></i>${t.name}</span><span class="p1">${pill(live ? 'wait' : 'succ')}</span><span class="p2">${pill(live ? 'wait' : 'caught')}</span></div>`;
    };
    const heads = () => `<div class="atk-h" aria-hidden="true"><span>Attack</span><span>What the malicious agent does</span><span>Task</span><span>${tab === 'harbor' ? 'Plain Harbor' : 'Harbor + LLM judge'}</span><span><span class="bw">${SB.icon('eye')}Breakwater</span></span></div>`;

    function draw(replay) {
      let html = heads();
      if (tab === 'judge') {
        html += `<div class="atk-g"><span>ASI01 · ASI06</span>Agent goal hijack and memory &amp; context poisoning; HJ = judge hijacking</div>` + J.map(a => row(a, replay)).join('');
      } else {
        const cats = SB.OWASP.filter(o => H.some(a => a.cat === o.id) && (!cat || o.id === cat));
        cats.forEach(o => { html += `<div class="atk-g"><span>${o.id}</span>${o.name}</div>` + H.filter(a => a.cat === o.id).map(a => row(a, replay)).join(''); });
      }
      list.innerHTML = html;
      filt.innerHTML = cat && tab === 'harbor'
        ? `Showing ${cat} only · <button type="button" id="atkall">Show all</button>`
        : tab === 'harbor' ? `${H.length} attacks against plain Harbor, all caught by Breakwater` : `${J.length} attacks against Harbor with an LLM judge, all caught by Breakwater`;
      const all = $('#atkall');
      if (all) all.addEventListener('click', () => setCat(null));
      if (replay) play();
    }
    function play() {
      const rows = $$('.atk-r', list);
      rows.forEach((r, i) => {
        setTimeout(() => {
          const p1 = $('.p1', r); p1.innerHTML = pill('succ'); r.classList.add('pop');
          setTimeout(() => { $('.p2', r).innerHTML = pill('caught'); }, 380);
        }, 260 + i * 110);
      });
    }
    function setTab(t) {
      tab = t;
      $$('#atktabs button').forEach(b => b.setAttribute('aria-selected', b.dataset.tab === t));
      const replay = !still && !done[t];
      done[t] = true;
      draw(replay);
    }
    function setCat(c) {
      if (disclosure) { disclosure.open = true; updateToggle(); }
      cat = c;
      $$('.ow[data-cat]', box).forEach(b => b.setAttribute('aria-pressed', b.dataset.cat === c));
      if (c === 'ASI01') { cat = null; setTab('judge'); $$('.ow[data-cat]', box).forEach(b => b.setAttribute('aria-pressed', b.dataset.cat === 'ASI01')); return; }
      if (tab !== 'harbor') setTab('harbor'); else draw(false);
    }
    box.addEventListener('click', e => {
      const b = e.target.closest('.ow[data-cat]');
      if (!b) return;
      setCat(b.getAttribute('aria-pressed') === 'true' ? null : b.dataset.cat);
    });
    $$('#atktabs button').forEach(b => b.addEventListener('click', () => { cat = null; $$('.ow[data-cat]', box).forEach(x => x.setAttribute('aria-pressed', 'false')); setTab(b.dataset.tab); }));
    draw(false);
    if (!still) {
      onceVisible(list, () => {
        if (done[tab]) return;
        done[tab] = true;
        draw(true);
      }, .15);
    } else done.harbor = true;

    // architecture diagram and safeguard list highlight each other
    const arch = $('#arch');
    const focus = (n, on) => {
      if (!arch) return;
      arch.classList.toggle('focus', on);
      $$(`.g`, arch).forEach(g => g.classList.toggle('hot', on && g.classList.contains(`g${n}`)));
      $$('.guards li').forEach(li => li.classList.toggle('hot', on && li.dataset.g === String(n)));
    };
    $$('.guards li').forEach(li => {
      li.addEventListener('mouseenter', () => focus(li.dataset.g, true));
      li.addEventListener('mouseleave', () => focus(li.dataset.g, false));
      li.addEventListener('click', () => focus(li.dataset.g, !li.classList.contains('hot')));
    });
    if (arch) $$('[data-g]', arch).forEach(g => {
      g.addEventListener('mouseenter', () => focus(g.dataset.g, true));
      g.addEventListener('mouseleave', () => focus(g.dataset.g, false));
    });
    if (arch && still) arch.querySelector('svg')?.pauseAnimations?.();
  }

  /* ---------- case studies ---------- */
  function cases() {
    $$('.case').forEach(c => {
      const btn = $('.cmore', c), det = $('.cdet', c);
      if (!btn || !det) return;
      btn.addEventListener('click', () => {
        const open = det.hidden;
        det.hidden = !open;
        c.classList.toggle('open', open);
        btn.setAttribute('aria-expanded', open);
        $('span', btn).textContent = open ? 'Show less' : 'Read the case';
      });
    });
  }

  /* ---------- pause decorative motion off-screen ---------- */
  function offscreen() {
    if (still || !('IntersectionObserver' in window)) return;
    const io = new IntersectionObserver(es => es.forEach(e => {
      const el = e.target;
      if (el.tagName.toLowerCase() === 'svg') (e.isIntersecting ? el.unpauseAnimations() : el.pauseAnimations());
      else el.classList.toggle('off', !e.isIntersecting);
    }));
    ['.frames', '#locked', '#arch svg'].forEach(s => { const el = $(s); if (el) io.observe(el); });
  }

  /* ---------- boot ---------- */
  function boot() {
    links();
    SB.art.paintCards();
    SB.art.buildSearch();
    SB.art.buildLifecycle();
    SB.art.buildLocked();
    cards();
    $$('[data-count]').forEach(countUp);
    const lb = $('#lbchart'); if (lb) SB.charts.Leaderboard(lb);
    const pc = $('#pcchart'); if (pc) SB.charts.CostChart(pc);
    const effort = $('#effortchart'); if (effort) SB.charts.EffortChart(effort);
    const rt = $('#restbl'); if (rt) SB.charts.ResultsTable(rt);
    redteam();
    cases();
    subnav();
    offscreen();
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot); else boot();
})();
