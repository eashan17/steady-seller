// Time-driven composition for the 3-minute demo video. window.renderAt(t) draws the frame at t seconds; every frame is a
// pure function of t. Uses the demo app's own CSS and render functions (ui.js) and the real negotiation data.
(function () {
  const stage = document.getElementById("stage");
  const D = {};
  const clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
  const ease = (x) => 1 - Math.pow(1 - clamp(x), 3);
  const vis = (t, t0, t1, fin = 0.45, fout = 0.35) => (t < t0 || t > t1 ? 0 : Math.min(ease((t - t0) / fin), ease((t1 - t) / fout)));
  const st = (o, dy = 14) => `style="opacity:${o.toFixed(3)};transform:translateY(${((1 - o) * dy).toFixed(1)}px)"`;
  const typed = (s, t, t0, dur) => s.slice(0, Math.round(s.length * clamp((t - t0) / dur)));
  const TEAM = "TM-8E2C352EF10";

  const SCENES = [
    { id: "hook", t0: 0, t1: 14 }, { id: "reveal", t0: 14, t1: 34 }, { id: "replay", t0: 34, t1: 94 },
    { id: "break", t0: 94, t1: 126 }, { id: "sbs", t0: 126, t1: 146 }, { id: "evidence", t0: 146, t1: 170 },
    { id: "close", t0: 170, t1: 180 },
  ];
  window.SCENES = SCENES;
  window.DURATION = 180;

  function head(t, items) {
    // items: [{t0, t1, kicker, text}]
    return items.map((h) => { const o = vis(t, h.t0, h.t1); return o <= 0 ? "" :
      `<div class="head" ${st(o, 10)}><div class="kicker">${h.kicker || "&nbsp;"}</div><div class="h1">${h.text}</div></div>`; }).join("");
  }
  function top() { return `<div class="vtop"><div class="brand"><span class="logo">🤝</span> Steady</div><div class="tid">AgentX · Selling to Machines · ${TEAM}</div></div>`; }
  function win(o, { chan, url, title, sub, right = "", feed = "", panel = "", composer = "", split = null }) {
    const chans = [["Replays", ["llm-take-it-or-leave-it", "llm-fake-approvals", "llm-final-offer"]],
      ["Compare", ["same-buyer-two-sellers", "no-deal-possible"]], ["Live", ["try-to-break-it", "how-we-got-here"]]];
    const side = chans.map(([h, cs]) => `<h5 style="margin-top:12px">${h}</h5>` + cs.map((c) => `<div class="chan ${c === chan ? "on" : ""}"><b>#</b>${c}</div>`).join("")).join("");
    const main = split ? split : `<div class="mhead"><span class="title"># ${chan}</span><span class="sub">${sub || ""}</span><div class="right">${right}</div></div>
      <div class="feed" data-bottom="1">${feed}</div>${composer}`;
    return `<section class="window vwin" ${st(o, 24)}><div class="chrome"><i></i><i></i><i></i><div class="url">127.0.0.1:8765/#${url || chan}</div></div>
      <div class="cols"><aside class="side"><div class="brand"><span class="logo">🤝</span> Steady</div>${side}
      <div class="foot">Real negotiations, real seller. Replays feed recorded buyer messages to the submitted seller.</div></aside>
      <main class="main">${main}</main><aside class="panel">${panel}</aside></div></section>`;
  }

  // ------------------------------------------------------------ scenes
  function hook(t) {
    const g = D.kit[0], k = g.ours.turns.findIndex((x) => /SYSTEM/.test(x.buyer.text || ""));
    const turn = g.ours.turns[k], prev = g.ours.turns[k - 1];
    const text = turn.buyer.text || "";
    let feed = UI.sys("Kit manipulator · round " + turn.round);
    if (prev) feed += UI.sellerMsg(prev.reply);
    if (t >= 2.2) {
      const full = t >= 6.2;
      feed += UI.buyerMsg("kit", full ? turn.buyer : { type: "counter" }, typed(text, t, 2.2, 4), t >= 6.6, turn.tags);
    }
    if (t >= 8.2) feed += UI.sellerMsg(turn.reply);
    const panel = t >= 8.6 ? UI.insight(turn) : UI.insight(null);
    return top() + head(t, [{ t0: 0.3, t1: 6.9, kicker: "SELLING TO MACHINES", text: "Shoppers are sending AI agents to buy for them." },
      { t0: 7.2, t1: 13.7, kicker: "SELLING TO MACHINES", text: "Some of them try to <em>hack the seller</em>." }]) +
      win(vis(t, 0.6, 13.7), { chan: "same-buyer-two-sellers", sub: "· round " + turn.round + " of 10",
        right: `<span class="pill green">0 violations</span>`, feed, panel });
  }

  function reveal(t) {
    const cards = [
      ["01", "Read", "A small trained classifier tags intent, claims and attacks in each message.", "MiniLM · int8 ONNX · 23 MB"],
      ["02", "Believe", "A Bayesian grid over the buyer's hidden limit, deadline and stubbornness.", "updated with every bid"],
      ["03", "Decide", "An expected-value planner picks offer, accept or walk.", "deadlock cost built in"],
      ["04", "Word", "Templates use only numbers from our own offer.", "content-seeded · fair"],
      ["05", "Guard", "Every reply is checked against policy and truth rules before it is sent.", "fallback if anything fails"]];
    const pipe = cards.map((c, i) => { const o = vis(t, 16 + 2.5 * i, 33.7);
      return `<div class="pcard ${i === 4 ? "dark" : ""}" ${st(o, 30)}><div class="n">${c[0]}</div><h3>${c[1]}</h3><p>${c[2]}</p><div class="spec">${c[3]}</div></div>`; }).join("");
    const b = ["No LLM at runtime", "≈10 ms per turn", "0 API calls", "Deterministic"];
    const badges = b.map((x, i) => { const o = vis(t, 28.4 + 0.5 * i, 33.7); return `<span class="pill ${i ? "ghost" : "green"}" ${st(o, 10)}>${x}</span>`; }).join("");
    return top() + head(t, [{ t0: 14.3, t1: 33.7, kicker: "OUR SELLER", text: "A seller that reads buyers, <em>not prompts</em>." }]) +
      `<div class="pipe">${pipe}</div><div class="badges">${badges}</div>`;
  }

  function replaySeg(t, g, chan, T0, T1, tEnd, endText) {
    const n = g.turns.length, step = (T1 - T0) / n;
    let feed = UI.sys(g.title), shown = null;
    for (let i = 0; i < n; i++) {
      const tb = T0 + i * step, ts = tb + step * 0.5, tr = g.turns[i];
      if (t >= tb) feed += UI.buyerMsg("llm", tr.buyer, null, t >= tb + Math.min(0.6, step * 0.3), tr.tags);
      if (t >= ts) { feed += UI.sellerMsg(tr.reply); shown = tr; }
    }
    if (g.final_buyer && t >= T1 + 0.6) feed += UI.buyerMsg("llm", g.final_buyer, null, false);
    if (t >= T1 + 1.6) feed += UI.sys(endText);
    const right = t >= T1 + 2.2 && g.reproduced ? `<span class="pill green">✓ Replayed: same decisions</span>` : `<span class="pill ghost">Groq · gpt-oss-120b · recorded</span>`;
    return { chan, sub: `· round ${shown ? shown.round : 0} of 10`, right, feed, panel: UI.insight(shown) };
  }

  function replay(t) {
    const A = t < 70;
    const g = D.llm[A ? 0 : 1];
    const w = A ? replaySeg(t, g, "llm-take-it-or-leave-it", 36.5, 63.5, 69.7,
                           `Deal at ${UI.rs(g.result.deal_total)} · buyer's hidden limit was ${UI.rs(g.buyer_limit)} · 0 violations`)
                : replaySeg(t, g, "llm-fake-approvals", 71.2, 87.5, 93.7, `Walked away in time · buyer's hidden limit ${UI.rs(g.buyer_limit)} was below our floor · 0 violations`);
    return top() + head(t, D.heads.replay(t)) + win(A ? vis(t, 34.3, 69.7) : vis(t, 70.1, 93.7), w);
  }

  function breakIt(t) {
    const b = D.brk, T = [95, 101, 107, 113];
    let feed = UI.sys(`You are the buyer · ${b.qty} × ${b.name}`), shown = null, typing = "";
    b.turns.forEach((tr, i) => {
      const t0 = T[i];
      if (t >= t0 && t < t0 + 2.8) typing = typed(tr.buyer.text || "", t, t0, 2.4);
      if (t >= t0 + 2.8) feed += UI.buyerMsg("you", tr.buyer, null, t >= t0 + 3.3, tr.tags);
      if (t >= t0 + 4.0) { feed += UI.sellerMsg(tr.reply); shown = tr; }
    });
    if (t >= 119 && t < 121.8) typing = typed(b.naive.text, t, 119, 2.4);
    let right = `<span class="pill green">0 violations</span>`;
    if (t >= 122) {
      const kinds = b.naive.violations.map((v) => ({ pressure: "pressure words", "house:stock": "stock claim", "house:urgency": "urgency" }[v.kind] || v.kind));
      feed += UI.sys(`What an LLM sales line would cost: "${b.naive.text}"`);
      feed += `<div class="tagrow blocked" style="justify-content:center">${kinds.map((k) => `<span class="tag attack">✕ ${k}</span>`).join("")}<span class="tag ok">blocked before sending</span></div>`;
      right = `<span class="pill red">Guard: blocked</span>`;
    }
    const composer = `<div class="composer"><div class="box">${typing ? UI.esc(typing) + '<span style="color:#16171a">|</span>' : "Say something to the seller…"}</div>
      <div class="row"><span class="btn small dark">Send</span><span>The seller answers in its next turn.</span><span class="btn small" style="margin-left:auto">Check an LLM-style line</span></div></div>`;
    return top() + head(t, [{ t0: 94.3, t1: 125.7, kicker: "LIVE · YOU PLAY THE BUYER", text: "Try to <em>break it</em>." }]) +
      win(vis(t, 94.3, 125.7), { chan: "try-to-break-it", sub: shown ? `· round ${shown.round} of 10` : "", right, feed, panel: UI.insight(shown), composer });
  }

  function sbs(t) {
    const first = t < 140, g = D.kit[first ? 0 : 1], ts = first ? 126.5 : 140.2, te = first ? 136 : 142.5;
    const col = (side, who) => {
      const d = g[side], msgs = d.transcript, n = msgs.length;
      let h = "";
      msgs.forEach((m, i) => { if (t >= ts + (te - ts) * (i / n)) h += m.from === "buyer" ? UI.buyerMsg("kit", m, null, false) : UI.sellerMsg(m, who); });
      const label = { deal: "deal", seller_walked: "walked away in time", deadlock: "deadlock · −25 points", buyer_walked: "buyer left" }[d.ended] || d.ended;
      const so = vis(t, te + 0.3, first ? 139.7 : 145.7);
      return `<div><div class="mhead"><span class="title">${who.name}</span><span class="sub" ${st(so, 6)}>${label}</span>
        <div class="right"><span class="score" ${st(so, 8)}>${(100 * d.score).toFixed(1)}</span></div></div><div class="feed" data-bottom="1">${h}</div></div>`;
    };
    const panel = `<h5>Same buyer, two sellers</h5><div class="card"><div class="k">Buyer</div><div class="v" style="font-size:17px">Kit manipulator</div>
      <div class="muted" style="font-size:13px">Fake SYSTEM notes, injected JSON, "you already agreed", fake budgets.</div></div>
      <div class="kv"><div class="card"><div class="k">Baseline</div><div class="v">${(100 * g.baseline.score).toFixed(1)}</div></div>
      <div class="card"><div class="k">Ours</div><div class="v">${(100 * g.ours.score).toFixed(1)}</div></div></div>`;
    return top() + head(t, [{ t0: 126.3, t1: 139.8, kicker: "KIT MANIPULATOR · SAME SCENARIO", text: "Same buyer, <em>two sellers</em>." },
      { t0: 140.1, t1: 145.7, kicker: "NO COMPLIANT DEAL EXISTS", text: "Walk away in time. <em>Never deadlock.</em>" }]) +
      win(vis(t, 126.3, 145.7), { chan: first ? "same-buyer-two-sellers" : "no-deal-possible", split: `<div class="split">${col("baseline", UI.BASE)}${col("ours", Object.assign({}, UI.SELLER, { name: "Ours (V11)" }))}</div>`, panel });
  }

  function evidence(t) {
    const e = D.evid;
    const tl = e.timeline.map((x, i) => { const o = vis(t, 147.5 + 1.75 * i, 169.7);
      return `<div class="tl" ${st(o, 10)}><div class="ver ${x.cls || ""}">${x.v}</div><div class="t"><b>${UI.esc(x.what)}</b> ${UI.esc(x.why)}</div></div>`; }).join("");
    const cards = e.cards.map((c, i) => { const o = vis(t, 149 + 2.2 * i, 169.7);
      return `<div class="card" ${st(o, 10)}><div class="k">${UI.esc(c.k)}</div><div class="v">${UI.esc(c.v)} <small>${UI.esc(c.s || "")}</small></div></div>`; }).join("");
    const feed = `<div class="evid" style="padding:4px 0"><div class="timeline">${tl}</div></div>`;
    return top() + head(t, [{ t0: 146.3, t1: 169.7, kicker: "EVERY VERSION HAD TO BEAT THE LAST · ZERO VIOLATIONS", text: "Not guessed. <em>Derived.</em>" }]) +
      win(vis(t, 146.3, 169.7), { chan: "how-we-got-here", sub: "· V0 → V11", feed, panel: `<h5>Results</h5>${cards}` });
  }

  function close(t) {
    const s = [["51.4", "kit score · baseline 48.5"], ["0", "policy or truth violations"], ["≈10 ms", "per turn, one CPU thread"], ["₹0", "API cost · no LLM at runtime"]];
    const stats = s.map((x, i) => { const o = vis(t, 170.4 + 0.45 * i, 180.5); return `<div class="stat" ${st(o, 18)}><div class="v">${x[0]}</div><div class="k">${x[1]}</div></div>`; }).join("");
    const o = vis(t, 172.6, 180.5);
    return top() + head(t, [{ t0: 170.2, t1: 180.5, kicker: "STEADY", text: "Calm under pressure. <em>Honest by construction.</em>" }]) +
      `<div class="stats">${stats}</div><div class="closing" ${st(o, 10)}><div class="t">A seller that reads buyers, not prompts.</div>
      <div class="s">Team ${TEAM} · AgentX Challenge 2 · Selling to Machines</div></div>`;
  }

  const DRAW = { hook, reveal, replay, break: breakIt, sbs, evidence, close };
  window.renderAt = function (t) {
    const sc = SCENES.find((s) => t >= s.t0 && t < s.t1) || SCENES[SCENES.length - 1];
    stage.innerHTML = DRAW[sc.id](t);
    stage.querySelectorAll('[data-bottom="1"]').forEach((f) => (f.scrollTop = f.scrollHeight));
    return sc.id;
  };

  window.videoReady = (async () => {
    const j = (f) => fetch("data/" + f).then((r) => r.json());
    [D.llm, D.kit, D.brk, D.evid] = await Promise.all([j("llm_replays.json"), j("kit_side_by_side.json"), j("break_it.json"), j("evidence.json")]);
    D.heads = { replay: (t) => [{ t0: 34.3, t1: 51.7, kicker: "RECORDED · LLM BUYER · GROQ GPT-OSS-120B", text: "A real negotiation with an <em>LLM buyer</em>." },
      { t0: 52, t1: 69.7, kicker: "TAGS ARE EVIDENCE, NEVER LEVERS", text: "It hears the bluff. <em>It holds the price.</em>" },
      { t0: 70.1, t1: 93.7, kicker: "LLM MANIPULATOR · DEAL POSSIBLE 73% → 0%", text: "No honest deal? <em>It knows, and walks.</em>" }] };
    await document.fonts.ready;
    const q = new URLSearchParams(location.search);
    renderAt(parseFloat(q.get("t") || "0"));
    return true;
  })();
})();
