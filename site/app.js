/* 발표 슬라이드 엔진 + 차트 (외부 라이브러리 없음).
   데이터는 data/data.js 가 window.DATA 로 넣어 준다 (scripts/06_build_site.py 가 생성). */
(function () {
  "use strict";
  const D = window.DATA || {};
  const NS = "http://www.w3.org/2000/svg";

  /* ------------------------------------------------------------ 슬라이드 엔진 */
  const slides = Array.from(document.querySelectorAll(".slide"));
  const counter = document.getElementById("counter");
  const bar = document.getElementById("progress");
  let cur = 0;

  function fit() {
    const stage = document.getElementById("stage");
    const s = Math.min(window.innerWidth / 1280, window.innerHeight / 720);
    stage.style.transform = `translate(-50%, -50%) scale(${s})`;
  }
  function show(i) {
    cur = Math.max(0, Math.min(slides.length - 1, i));
    slides.forEach((s, k) => s.classList.toggle("active", k === cur));
    counter.textContent = `${cur + 1} / ${slides.length}`;
    bar.style.width = `${((cur + 1) / slides.length) * 100}%`;
    if (location.hash !== `#s${cur + 1}`) history.replaceState(null, "", `#s${cur + 1}`);
    const v = slides[cur].querySelector("video");
    document.querySelectorAll("video").forEach((x) => { if (x !== v) x.pause(); });
    if (v) { v.currentTime = 0; v.play().catch(() => {}); }
  }
  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT") return;
    if (["ArrowRight", "PageDown", " "].includes(e.key)) { e.preventDefault(); show(cur + 1); }
    if (["ArrowLeft", "PageUp"].includes(e.key)) { e.preventDefault(); show(cur - 1); }
    if (e.key === "Home") show(0);
    if (e.key === "End") show(slides.length - 1);
    if (e.key === "t") toggleTheme();
  });
  document.getElementById("prev").onclick = () => show(cur - 1);
  document.getElementById("next").onclick = () => show(cur + 1);
  document.getElementById("theme").onclick = toggleTheme;
  function toggleTheme() {
    const r = document.documentElement;
    const dark = r.dataset.theme === "dark" ||
      (!r.dataset.theme && matchMedia("(prefers-color-scheme: dark)").matches);
    r.dataset.theme = dark ? "light" : "dark";
  }
  window.addEventListener("resize", fit);
  window.addEventListener("hashchange", () => {
    const m = location.hash.match(/^#s(\d+)$/);
    if (m) show(+m[1] - 1);
  });
  fit();
  const m0 = location.hash.match(/^#s(\d+)$/);
  show(m0 ? +m0[1] - 1 : 0);

  /* ------------------------------------------------------------ SVG 도우미 */
  function el(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function txt(parent, x, y, s, attrs) {
    const t = el("text", Object.assign({ x, y, class: "ct" }, attrs || {}), parent);
    t.textContent = s;
    return t;
  }
  function svgIn(id, w, h) {
    const host = document.getElementById(id);
    if (!host) return null;
    host.innerHTML = "";
    return el("svg", { viewBox: `0 0 ${w} ${h}`, class: "chart" }, host);
  }
  const pct = (v) => (v * 100).toFixed(1) + "%";

  /* 극좌표 튜닝 곡선 */
  function polar(svg, cx, cy, R, series, opts) {
    opts = opts || {};
    [0.25, 0.5, 0.75, 1].forEach((f) => el("circle", { cx, cy, r: R * f, class: "grid", fill: "none" }, svg));
    [0, 90, 180, 270].forEach((a) => {
      const r = (a * Math.PI) / 180;
      el("line", { x1: cx, y1: cy, x2: cx + R * Math.cos(r), y2: cy - R * Math.sin(r), class: "grid" }, svg);
      if (opts.labels !== false) {
        const lx = cx + (R + 13) * Math.cos(r), ly = cy - (R + 13) * Math.sin(r) + 4;
        txt(svg, lx, ly, a + "°", { "text-anchor": "middle", class: "ct small" });
      }
    });
    series.forEach((s) => {
      const pts = s.r.map((v, k) => {
        const a = (s.angles[k] * Math.PI) / 180;
        return [cx + R * v * Math.cos(a), cy - R * v * Math.sin(a)];
      });
      pts.push(pts[0]);
      el("path", {
        d: "M" + pts.map((p) => p.join(",")).join("L"),
        fill: s.fill ? s.color : "none", "fill-opacity": s.fill ? 0.15 : 0,
        stroke: s.color, "stroke-width": s.width || 2, "stroke-dasharray": s.dash || "",
        "stroke-opacity": s.opacity ?? 1,
        "stroke-linejoin": "round",
      }, svg);
    });
    if (opts.marker != null) {
      const a = (opts.marker * Math.PI) / 180;
      el("line", { x1: cx, y1: cy, x2: cx + R * 1.08 * Math.cos(a), y2: cy - R * 1.08 * Math.sin(a), class: "marker" }, svg);
    }
  }

  /* 선 그래프 (오차 막대) */
  function lineChart(id, cfg) {
    const W = cfg.w || 640, H = cfg.h || 400, m = { l: 62, r: 20, t: 16, b: 56 };
    const svg = svgIn(id, W, H);
    if (!svg) return;
    const xs = cfg.x, n = xs.length;
    const X = (i) => m.l + (cfg.xcat ? (i + 0.5) / n : i / (n - 1)) * (W - m.l - m.r);
    const ymin = cfg.ymin ?? 0, ymax = cfg.ymax ?? 100;
    const Y = (v) => H - m.b - ((v - ymin) / (ymax - ymin)) * (H - m.t - m.b);
    for (let v = ymin; v <= ymax; v += cfg.ystep || 25) {
      el("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), class: "grid" }, svg);
      txt(svg, m.l - 8, Y(v) + 4, v + (cfg.yunit || ""), { "text-anchor": "end" });
    }
    xs.forEach((x, i) => txt(svg, X(i), H - m.b + 20, cfg.xlabels ? cfg.xlabels[i] : x, { "text-anchor": "middle" }));
    if (cfg.xtitle) txt(svg, (m.l + W - m.r) / 2, H - 10, cfg.xtitle, { "text-anchor": "middle", class: "ct axis" });
    if (cfg.ytitle) txt(svg, 16, (m.t + H - m.b) / 2, cfg.ytitle, { "text-anchor": "middle", class: "ct axis", transform: `rotate(-90 16 ${(m.t + H - m.b) / 2})` });
    (cfg.hlines || []).forEach((h) => {
      el("line", { x1: m.l, x2: W - m.r, y1: Y(h.y), y2: Y(h.y), stroke: h.color || "var(--muted)", "stroke-dasharray": h.dash || "6 5", "stroke-width": 1.5 }, svg);
      txt(svg, W - m.r - 4, Y(h.y) - 6, h.label, { "text-anchor": "end", class: "ct small", fill: h.color || "var(--muted)" });
    });
    const g = el("g", {}, svg);
    cfg.series.forEach((s) => {
      const p = s.y.map((v, i) => [X(i), Y(v)]);
      el("path", { d: "M" + p.map((q) => q.join(",")).join("L"), fill: "none", stroke: s.color, "stroke-width": 3, "stroke-dasharray": s.dash || "" }, g);
      s.y.forEach((v, i) => {
        if (s.err) el("line", { x1: X(i), x2: X(i), y1: Y(v - s.err[i]), y2: Y(v + s.err[i]), stroke: s.color, "stroke-width": 2 }, g);
        el("circle", { cx: X(i), cy: Y(v), r: cfg.highlight === i ? 7 : 5, fill: s.color, stroke: "var(--bg)", "stroke-width": 2 }, g);
      });
    });
    return { svg, X, Y };
  }

  /* 혼동 행렬 */
  function heatmap(id, cm, labels, color) {
    const n = labels.length, c = 34, m = 52, W = m + n * c + 8, H = m + n * c + 8;
    const svg = svgIn(id, W, H);
    if (!svg) return;
    cm.forEach((row, i) => {
      const s = row.reduce((a, b) => a + b, 0) || 1;
      row.forEach((v, j) => {
        const f = v / s;
        el("rect", { x: m + j * c, y: m + i * c, width: c - 2, height: c - 2, rx: 3, fill: color, "fill-opacity": 0.06 + 0.94 * f }, svg);
        if (f >= 0.05) txt(svg, m + j * c + c / 2 - 1, m + i * c + c / 2 + 4, Math.round(f * 100), { "text-anchor": "middle", class: f > 0.55 ? "ct small inv" : "ct small" });
      });
    });
    labels.forEach((l, k) => {
      txt(svg, m + k * c + c / 2, m - 8, l, { "text-anchor": "middle", class: "ct small" });
      txt(svg, m - 6, m + k * c + c / 2 + 4, l, { "text-anchor": "end", class: "ct small" });
    });
    txt(svg, m + (n * c) / 2, 14, "예측한 방향 →", { "text-anchor": "middle", class: "ct small muted" });
    txt(svg, 10, m + (n * c) / 2, "실제 방향 →", { "text-anchor": "middle", class: "ct small muted", transform: `rotate(-90 10 ${m + (n * c) / 2})` });
  }

  /* 묶음 막대 */
  function groupedBars(id, groups, methods, colors, cfg) {
    cfg = cfg || {};
    const W = cfg.w || 640, H = cfg.h || 360, m = { l: 52, r: 10, t: 48, b: 46 };
    const svg = svgIn(id, W, H);
    if (!svg) return;
    const Y = (v) => H - m.b - (v / 100) * (H - m.t - m.b);
    for (let v = 0; v <= 100; v += 25) {
      el("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), class: "grid" }, svg);
      txt(svg, m.l - 8, Y(v) + 4, v + "%", { "text-anchor": "end" });
    }
    const gw = (W - m.l - m.r) / groups.length, bw = (gw * 0.78) / methods.length;
    groups.forEach((gname, gi) => {
      methods.forEach((mt, mi) => {
        const d = cfg.get(gname, mt);
        const x = m.l + gi * gw + gw * 0.11 + mi * bw;
        el("rect", { x, y: Y(d.mean * 100), width: bw - 4, height: Y(0) - Y(d.mean * 100), rx: 3, fill: colors[mi] }, svg);
        txt(svg, x + (bw - 4) / 2, Y(d.mean * 100) - 6, Math.round(d.mean * 100), { "text-anchor": "middle", class: "ct small strong" });
      });
      txt(svg, m.l + gi * gw + gw / 2, H - m.b + 22, cfg.glabel ? cfg.glabel(gname) : gname, { "text-anchor": "middle" });
    });
    if (cfg.chance) {
      el("line", { x1: m.l, x2: W - m.r, y1: Y(cfg.chance * 100), y2: Y(cfg.chance * 100), stroke: "var(--muted)", "stroke-dasharray": "6 5", "stroke-width": 1.5 }, svg);
    }
    // 범례는 그래프 위 별도 줄에 (막대 숫자와 겹치지 않게), 우연 수준 점선도 범례에 넣는다
    const lg = el("g", {}, svg);
    let lx = m.l;
    methods.forEach((mt, mi) => {
      el("rect", { x: lx, y: 6, width: 12, height: 12, rx: 2, fill: colors[mi] }, lg);
      txt(lg, lx + 18, 17, mt, { class: "ct small" });
      lx += 30 + mt.length * 11.5;
    });
    if (cfg.chance) {
      el("line", { x1: lx, x2: lx + 20, y1: 12, y2: 12, stroke: "var(--muted)", "stroke-dasharray": "5 4", "stroke-width": 1.5 }, lg);
      txt(lg, lx + 26, 17, "우연 12.5%", { class: "ct small muted" });
    }
  }

  /* 히스토그램 */
  function hist(id, values, cfg) {
    const W = cfg.w || 640, H = cfg.h || 340, m = { l: 48, r: 14, t: 14, b: 50 };
    const svg = svgIn(id, W, H);
    if (!svg) return;
    const lo = 0, hi = 100, nb = 40, bw = (hi - lo) / nb;
    const counts = new Array(nb).fill(0);
    values.forEach((v) => counts[Math.min(nb - 1, Math.floor((v * 100 - lo) / bw))]++);
    const cmax = Math.max(...counts);
    const X = (v) => m.l + ((v - lo) / (hi - lo)) * (W - m.l - m.r);
    const Y = (c) => H - m.b - (c / cmax) * (H - m.t - m.b);
    for (let v = 0; v <= 100; v += 25) {
      txt(svg, X(v), H - m.b + 18, v + "%", { "text-anchor": "middle" });
      el("line", { x1: X(v), x2: X(v), y1: m.t, y2: H - m.b, class: "grid" }, svg);
    }
    counts.forEach((c, i) => {
      if (c) el("rect", { x: X(lo + i * bw) + 1, y: Y(c), width: X(bw) - X(0) - 2, height: Y(0) - Y(c), fill: cfg.color }, svg);
    });
    (cfg.lines || []).forEach((l) => {
      el("line", { x1: X(l.x * 100), x2: X(l.x * 100), y1: m.t, y2: H - m.b, stroke: l.color, "stroke-width": l.w || 3, "stroke-dasharray": l.dash || "" }, svg);
      txt(svg, X(l.x * 100) + (l.anchor === "end" ? -6 : 6), m.t + 14 + (l.dy || 0), l.label, { "text-anchor": l.anchor || "start", class: "ct small strong", fill: l.color });
    });
    txt(svg, (m.l + W - m.r) / 2, H - 8, cfg.xtitle, { "text-anchor": "middle", class: "ct axis" });
  }

  /* 산점도 */
  function scatter(id, pts, cfg) {
    const W = cfg.w || 560, H = cfg.h || 360, m = { l: 60, r: 14, t: 14, b: 52 };
    const svg = svgIn(id, W, H);
    if (!svg) return;
    const [x0, x1] = cfg.xr, [y0, y1] = cfg.yr;
    const X = (v) => m.l + ((v - x0) / (x1 - x0)) * (W - m.l - m.r);
    const Y = (v) => H - m.b - ((v - y0) / (y1 - y0)) * (H - m.t - m.b);
    cfg.xt.forEach((v) => { el("line", { x1: X(v), x2: X(v), y1: m.t, y2: H - m.b, class: "grid" }, svg); txt(svg, X(v), H - m.b + 18, v, { "text-anchor": "middle" }); });
    cfg.yt.forEach((v) => { el("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), class: "grid" }, svg); txt(svg, m.l - 8, Y(v) + 4, v, { "text-anchor": "end" }); });
    pts.forEach((p) => el("circle", { cx: X(p.x), cy: Y(p.y), r: p.hi ? 7 : 5, fill: p.hi ? "var(--accent)" : "var(--muted)", "fill-opacity": p.hi ? 0.9 : 0.45, stroke: "var(--bg)", "stroke-width": 1.5 }, svg));
    txt(svg, (m.l + W - m.r) / 2, H - 10, cfg.xtitle, { "text-anchor": "middle", class: "ct axis" });
    txt(svg, 16, (m.t + H - m.b) / 2, cfg.ytitle, { "text-anchor": "middle", class: "ct axis", transform: `rotate(-90 16 ${(m.t + H - m.b) / 2})` });
  }

  /* ------------------------------------------------------------ 표지: 겹눈 위를 지나가는 막대 */
  function heroEye() {
    const cv = document.getElementById("eye");
    if (!cv || !D.eye) return;
    const ctx = cv.getContext("2d");
    const { x, y } = D.eye;
    const W = cv.width, H = cv.height;
    const xr = Math.max(...x.map(Math.abs)), yr = Math.max(...y.map(Math.abs));
    const s = Math.min(W / (2 * xr + 3), H / (2 * yr + 3));
    const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
    let t0 = performance.now();
    function hexPath(cx, cy, r) {
      ctx.beginPath();
      for (let k = 0; k < 6; k++) {
        const a = (Math.PI / 3) * k;
        ctx.lineTo(cx + r * Math.cos(a), cy + r * Math.sin(a));
      }
      ctx.closePath();
    }
    function frame(now) {
      const css = getComputedStyle(document.documentElement);
      const on = css.getPropertyValue("--on").trim(), dim = css.getPropertyValue("--hex").trim();
      ctx.clearRect(0, 0, W, H);
      const ph = reduce ? 0.35 : ((now - t0) / 3600) % 1;
      const ang = reduce ? 0 : Math.floor((now - t0) / 3600) % 8 * 45;
      const ca = Math.cos((ang * Math.PI) / 180), sa = Math.sin((ang * Math.PI) / 180);
      const pos = -xr * 1.2 + ph * xr * 2.4;
      for (let i = 0; i < x.length; i++) {
        const d = x[i] * ca + y[i] * sa - pos;
        const v = Math.exp(-(d * d) / 3);
        ctx.fillStyle = v > 0.02 ? on : dim;
        ctx.globalAlpha = v > 0.02 ? 0.25 + 0.75 * v : 1;
        hexPath(W / 2 + x[i] * s, H / 2 - y[i] * s, s * 0.95);
        ctx.fill();
      }
      ctx.globalAlpha = 1;
      if (!reduce) requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  }

  /* ------------------------------------------------------------ 재현 슬라이드 */
  function reproduce() {
    const R = D.reproduce;
    if (!R) return;
    const angles = R.angles_deg;
    const host = document.getElementById("polars");
    const order = ["T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"];
    let mode = "best";
    let sel = 0;
    function draw() {
      host.innerHTML = "";
      order.forEach((ct) => {
        const c = R.cell_types[ct];
        const box = document.createElement("div");
        box.className = "polar " + (c.edge === "bright" ? "on" : "off");
        const svg = el("svg", { viewBox: "0 0 150 150" }, box);
        const col = c.edge === "bright" ? "var(--on)" : "var(--off)";
        const series = [{ r: c.measured_norm, angles, color: "var(--muted)", dash: "4 3", width: 1.6 }];
        if (mode === "top") {
          c.top_models_norm.forEach((r) => series.push({ r, angles, color: col, width: 1, opacity: 0.4 }));
        } else {
          series.push({ r: c.best_model_norm, angles, color: col, width: 2.6, fill: true });
        }
        polar(svg, 75, 77, 56, series, { labels: false, marker: sel });
        const cap = document.createElement("div");
        cap.className = "pcap";
        const pref = R.known_preferred_deg[ct];
        cap.innerHTML = mode === "top"
          ? `<b>${ct}</b><span>${c.top_correct}/10 정답</span>`
          : `<b>${ct}</b><span>${pref}° / ${Math.round(c.best_pref_deg)}°</span>`;
        box.appendChild(cap);
        host.appendChild(box);
      });
    }
    let edge = "bright";
    function stim() {
      document.getElementById("stimgif").src = `media/stimulus_edge_${edge}_${String(sel).padStart(3, "0")}.gif`;
      document.getElementById("stimnote").textContent =
        `${edge === "bright" ? "밝은" : "어두운"} 가장자리가 ${sel}° 방향으로 이동`;
    }
    document.querySelectorAll("input[name=edge]").forEach((r) => (r.onchange = () => { edge = r.value; stim(); }));
    const btns = document.getElementById("dirbtns");
    angles.forEach((a, k) => {
      const b = document.createElement("button");
      b.textContent = a + "°";
      b.onclick = () => {
        sel = a;
        btns.querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
        stim();
        draw();
      };
      if (a === 0) b.classList.add("on");
      btns.appendChild(b);
    });
    document.querySelectorAll("input[name=pmode]").forEach((r) => (r.onchange = () => { mode = r.value; draw(); }));
    draw();

    // 5부 논문 결과: 실제 측정 튜닝 곡선만 (Maisak 2013)
    const gt = document.getElementById("gtpolars");
    if (gt) order.forEach((ct) => {
      const c = R.cell_types[ct];
      const box = document.createElement("div");
      box.className = "polar " + (c.edge === "bright" ? "on" : "off");
      const svg = el("svg", { viewBox: "0 0 150 150" }, box);
      polar(svg, 75, 77, 56, [{ r: c.measured_norm, angles, color: c.edge === "bright" ? "var(--on)" : "var(--off)", width: 2.4, fill: true }], { labels: false });
      const cap = document.createElement("div");
      cap.className = "pcap";
      cap.innerHTML = `<b>${ct}</b><span>${R.known_preferred_deg[ct]}°</span>`;
      box.appendChild(cap);
      gt.appendChild(box);
    });
  }

  /* ------------------------------------------------------------ 앙상블 */
  function ensemble() {
    const E = D.ensemble;
    if (!E) return;
    const ms = E.models;
    const xs = ms.map((m) => m.val_loss);
    const lo = Math.floor(Math.min(...xs) * 10) / 10, hi = Math.ceil(Math.max(...xs) * 10) / 10;
    const xt = [];
    for (let v = lo; v <= hi + 1e-9; v += 0.1) xt.push(+v.toFixed(1));
    // 같은 정답 수에 점이 겹치지 않게 결정적인 작은 흔들림
    scatter("ens", ms.map((m, i) => ({ x: m.val_loss, y: m.n_correct + (((i * 37) % 11) - 5) * 0.03, hi: i === 0 })), {
      xr: [lo, hi], yr: [-0.5, 8.5], xt, yt: [0, 2, 4, 6, 8], w: 520, h: 380,
      xtitle: "과제 오차 (검증 손실, 낮을수록 좋음)", ytitle: "맞힌 하위 유형 수 (8개 중)",
    });
    // 순위 구간별 평균 정답 수 막대
    const W = 560, H = 380, m = { l: 52, r: 10, t: 24, b: 56 };
    const svg = svgIn("ensbar", W, H);
    const Y = (v) => H - m.b - (v / 8) * (H - m.t - m.b);
    [0, 2, 4, 6, 8].forEach((v) => { el("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), class: "grid" }, svg); txt(svg, m.l - 8, Y(v) + 4, v, { "text-anchor": "end" }); });
    const n = E.group_labels.length, gw = (W - m.l - m.r) / n;
    E.group_mean_correct.forEach((v, i) => {
      const x = m.l + i * gw + gw * 0.18;
      el("rect", { x, y: Y(v), width: gw * 0.64, height: Y(0) - Y(v), rx: 4, fill: i === 0 ? "var(--accent)" : "var(--sb1)" }, svg);
      txt(svg, x + gw * 0.32, Y(v) - 8, v.toFixed(1), { "text-anchor": "middle", class: "ct strong" });
      txt(svg, x + gw * 0.32, H - m.b + 20, E.group_labels[i], { "text-anchor": "middle" });
    });
    txt(svg, (m.l + W - m.r) / 2, H - 12, "검증 오차 순위 구간 (모델 10개씩)", { "text-anchor": "middle", class: "ct axis" });
    txt(svg, 16, (m.t + H - m.b) / 2, "평균 정답 수 (8개 중)", { "text-anchor": "middle", class: "ct axis", transform: `rotate(-90 16 ${(m.t + H - m.b) / 2})` });
  }

  /* 파라미터 수 비교 (로그 눈금 막대) */
  function paramBar() {
    const W = 620, H = 210, m = { l: 150, r: 90, t: 16, b: 36 };
    const svg = svgIn("parambar", W, H);
    if (!svg) return;
    const X = (v) => m.l + (Math.log10(v) / 7) * (W - m.l - m.r);
    [1, 10, 100, 1e3, 1e4, 1e5, 1e6, 1e7].forEach((v) => {
      el("line", { x1: X(v), x2: X(v), y1: m.t, y2: H - m.b, class: "grid" }, svg);
      txt(svg, X(v), H - m.b + 18, v >= 1e6 ? v / 1e6 + "백만" : v >= 1e4 ? v / 1e4 + "만" : v, { "text-anchor": "middle", class: "ct small" });
    });
    [["일반 RNN (연결마다)", 1513000, "var(--sb1)", "약 151만"], ["DMN (공유)", 734, "var(--accent)", "734"]].forEach(([l, v, c, s], i) => {
      const y = m.t + 18 + i * 74;
      el("rect", { x: X(1), y, width: X(v) - X(1), height: 40, rx: 4, fill: c }, svg);
      txt(svg, m.l - 10, y + 25, l, { "text-anchor": "end", class: "ct axis" });
      txt(svg, X(v) + 8, y + 26, s, { class: "ct strong" });
    });
    txt(svg, (m.l + W - m.r) / 2, H - 4, "학습하는 값의 개수 (로그 눈금)", { "text-anchor": "middle", class: "ct small muted" });
  }

  /* ------------------------------------------------------------ 디코딩 메인 */
  function decoding() {
    const R = D.decoding;
    if (!R) return;
    const lv = R.noise_levels;
    const keys = ["A", "B4", "B1"];
    const col = { A: "var(--sa)", B4: "var(--sb4)", B1: "var(--sb1)" };
    const slider = document.getElementById("noise");
    function draw() {
      const k = +slider.value;
      lineChart("acc", {
        x: lv, xlabels: lv.map((v) => v + "배"), xcat: true, ymin: 0, ymax: 100, yunit: "%", highlight: k,
        xtitle: "측정 잡음 (채널 신호 표준편차의 배수)", ytitle: "8방향 분류 정확도",
        hlines: [{ y: 12.5, label: "우연 수준 12.5%" }],
        series: keys.map((s) => ({ color: col[s], y: R.results[s].map((r) => r.acc_mean * 100), err: R.results[s].map((r) => r.acc_std * 100) })),
        w: 640, h: 420,
      });
      document.getElementById("noiseval").textContent = lv[k] + "배";
      keys.forEach((s) => (document.getElementById("v" + s).textContent = pct(R.results[s][k].acc_mean)));
      const labels = R.directions.map((d) => d + "°");
      heatmap("cmA", R.results.A[k].confusion, labels, "var(--sa)");
      heatmap("cmB", R.results.B4[k].confusion, labels, "var(--sb4)");
    }
    slider.oninput = draw;
    draw();
  }

  /* ------------------------------------------------------------ 가상 EEG 파형 */
  function signals() {
    const S = D.signals;
    if (!S) return;
    const btns = document.getElementById("sigbtns");
    const cols = ["var(--z1)", "var(--z2)", "var(--z3)", "var(--z4)"];
    function draw(a) {
      const s = S.signals[String(a)];
      const n = s.B4[0].length;
      const W = 600, H = 300, m = { l: 40, r: 10, t: 10, b: 36 };
      const svg = svgIn("sigplot", W, H);
      const all = s.B4.flat();
      const lo = Math.min(...all), hi = Math.max(...all);
      const X = (i) => m.l + (i / (n - 1)) * (W - m.l - m.r);
      const Y = (v) => H - m.b - ((v - lo) / (hi - lo)) * (H - m.t - m.b);
      el("line", { x1: m.l, x2: W - m.r, y1: Y(0), y2: Y(0), class: "grid" }, svg);
      s.B4.forEach((ch, k) => el("path", { d: "M" + ch.map((v, i) => `${X(i)},${Y(v)}`).join("L"), fill: "none", stroke: cols[k], "stroke-width": 2.2 }, svg));
      for (let t = 0; t <= n / S.fs; t += 0.5) txt(svg, X(t * S.fs), H - m.b + 18, t.toFixed(1) + "초", { "text-anchor": "middle", class: "ct small" });
      document.getElementById("sigimg").src = `media/stimulus_bright_${String(a).padStart(3, "0")}.gif`;
    }
    S.directions.forEach((a) => {
      const b = document.createElement("button");
      b.textContent = a + "°";
      b.onclick = () => { btns.querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b)); draw(a); };
      if (a === 0) b.classList.add("on");
      btns.appendChild(b);
    });
    draw(0);
  }

  /* ------------------------------------------------------------ 머신러닝 실험 */
  function ml() {
    const M = D.ml;
    if (!M) return;
    const fs = M.feature_selection;
    lineChart("sfs", {
      x: fs.k, xcat: false, ymin: 0, ymax: 100, yunit: "%", xtitle: "고른 세포 유형 수", ytitle: "테스트 정확도", w: 620, h: 380,
      hlines: [
        { y: M.full_A * 100, label: `65개 전부 ${pct(M.full_A)}`, color: "var(--sa)", dash: "2 4" },
        { y: M.B4 * 100, label: `가상 EEG 4채널 ${pct(M.B4)}`, color: "var(--sb4)", dash: "2 4" },
        { y: 12.5, label: "우연 12.5%" },
      ],
      series: [
        { color: "var(--bad)", y: fs.leaky_mean.map((v) => v * 100), err: fs.leaky_std.map((v) => v * 100), dash: "7 5" },
        { color: "var(--sa)", y: fs.honest_mean.map((v) => v * 100), err: fs.honest_std.map((v) => v * 100) },
      ],
    });
    const r4 = M.random4;
    hist("rand4", r4.accs, {
      color: "var(--sa-soft)", xtitle: "테스트 정확도 (무작위 4개 유형 200번)", w: 620, h: 330,
      lines: [{ x: r4.b4_acc, color: "var(--sb4)", label: `가상 EEG 4채널 ${pct(r4.b4_acc)}`, anchor: r4.b4_acc > 0.6 ? "end" : "start" },
              { x: 0.125, color: "var(--muted)", label: "우연", w: 1.5, dash: "5 4", dy: 18 }],
    });
    const sig = ["A", "B4", "B1"], gl = (g) => ({ A: "신호 A (65채널)", B4: "가상 EEG 4채널", B1: "가상 EEG 1채널" }[g]);
    groupedBars("feat", sig, ["시간 영역", "PSD", "둘 다"], ["var(--sa)", "var(--c2)", "var(--c3)"], { get: (g, m) => M.features[g][m], glabel: gl, chance: 0.125, w: 600, h: 330 });
    groupedBars("clf", sig, ["로지스틱 회귀", "선형 SVM", "K-NN"], ["var(--sa)", "var(--c2)", "var(--c3)"], { get: (g, m) => M.classifiers[g][m], glabel: gl, chance: 0.125, w: 600, h: 330 });
    groupedBars("leak", sig, ["속도 기준 fold (올바름)", "전체 데이터로 정규화", "무작위 5-fold (반복 측정 섞임)"], ["var(--sa)", "var(--c2)", "var(--bad)"], { get: (g, m) => M.leakage[g][m], glabel: gl, chance: 0.125, w: 640, h: 340 });
    const sh = M.shuffled;
    if (sh) {
      lineChart("shuf", {
        x: sh.k, xcat: false, ymin: 0, ymax: 50, ystep: 10, yunit: "%", xtitle: "고른 세포 유형 수", ytitle: "테스트 정확도", w: 600, h: 380,
        hlines: [{ y: 12.5, label: "우연 12.5% (가짜 정답이므로 이것이 정직한 값)" }],
        series: [
          { color: "var(--bad)", y: sh.leaky_mean.map((v) => v * 100), err: sh.leaky_std.map((v) => v * 100), dash: "7 5" },
          { color: "var(--sa)", y: sh.honest_mean.map((v) => v * 100), err: sh.honest_std.map((v) => v * 100) },
        ],
      });
      const lk = Math.max(...sh.leaky_mean), hn = sh.honest_mean.reduce((a, b) => a + b, 0) / sh.honest_mean.length;
      document.getElementById("shuf-summary").innerHTML =
        `학습 fold 안에서 선택: 평균 <b>${pct(hn)}</b> (우연 수준 이하)<br>테스트 정답을 보고 선택: 최대 <b class="bad-t">${pct(lk)}</b>`;
    }
    const sel = document.getElementById("seltypes");
    if (sel) sel.innerHTML = fs.chosen_per_fold.map((f, i) => `<li><span class="mono">fold ${i + 1}</span> ${f.slice(0, 4).map((t) => `<b>${t}</b>`).join(" → ")}</li>`).join("");
  }

  heroEye();
  paramBar();
  reproduce();
  ensemble();
  signals();
  decoding();
  ml();
})();
