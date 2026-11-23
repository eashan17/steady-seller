// Interactive demo app. Data comes from data/*.json (built by demo/make_traces.py) and the live /api endpoints.
(async function () {
  const $ = (s) => document.querySelector(s);
  const main = $("#main"), panel = $("#panel");
  const load = (f) => fetch("data/" + f).then((r) => (r.ok ? r.json() : null)).catch(() => null);
  const [llm, kit, evid] = await Promise.all([load("llm_replays.json"), load("kit_side_by_side.json"), load("evidence.json")]);
  const REPLAYS = { "llm-take-it-or-leave-it": 0, "llm-fake-approvals": 1, "llm-final-offer": 2 };
  let timer = null;

  function setStep(n) { document.querySelectorAll(".step").forEach((s) => s.classList.toggle("on", +s.dataset.step === n)); }
  function go(name) {
    clearInterval(timer);
    location.hash = name;
    $("#url").textContent = "127.0.0.1:8765/#" + name;
    document.querySelectorAll(".chan").forEach((c) => c.classList.toggle("on", c.dataset.go === name));
    document.querySelectorAll(".nav a").forEach((a) => a.classList.toggle("active", a.dataset.go === name ||
      (a.dataset.go === "llm-take-it-or-leave-it" && name in REPLAYS) || (a.dataset.go === "same-buyer-two-sellers" && name === "no-deal-possible")));
    if (name in REPLAYS) return replay(llm && llm[REPLAYS[name]], name);
    if (name === "same-buyer-two-sellers") return sideBySide(kit && kit[0], name);
    if (name === "no-deal-possible") return sideBySide(kit && kit[1], name);
    if (name === "try-to-break-it") return breakIt();
    if (name === "how-we-got-here") return evidence();
  }
  document.querySelectorAll("[data-go]").forEach((el) => el.addEventListener("click", () => go(el.dataset.go)));

  // ---------------------------------------------------------------- replay
  function replay(g, name) {
    if (!g) { main.innerHTML = `<div class="feed">${UI.sys("Run demo/make_traces.py to build the replay data.")}</div>`; return; }
    let k = 0;
    const n = g.turns.length;
    main.innerHTML = `<div class="mhead"><span class="title"># ${name}</span><span class="sub" id="rnd"></span>
      <div class="right"><span class="pill ghost">${g.reproduced ? "Replayed: same decisions" : "Replay"}</span>
      <button class="btn small" id="prev">◀</button><button class="btn small" id="next">Next ▶</button><button class="btn small dark" id="play">Play</button></div></div>
      <div class="feed" id="feed"></div>`;
    const draw = () => {
      let h = UI.sys(g.title);
      for (let i = 0; i < k; i++) {
        const t = g.turns[i];
        h += UI.buyerMsg("llm", t.buyer, null, true, t.tags);
        h += UI.sellerMsg(t.reply);
      }
      if (k === n && g.final_buyer) h += UI.buyerMsg("llm", g.final_buyer, null, false);
      if (k === n) h += UI.sys(`${g.result.ended === "deal" ? "Deal at " + UI.rs(g.result.deal_total) : "Walked away in time"} · buyer's hidden limit ${UI.rs(g.buyer_limit)} · ${g.result.violations} violations`);
      $("#feed").innerHTML = h;
      $("#feed").scrollTop = 1e6;
      $("#rnd").textContent = `· round ${Math.min(k, n)} of ${n}`;
      panel.innerHTML = UI.insight(g.turns[Math.max(0, k - 1)] && k ? g.turns[k - 1] : null);
      setStep(k ? 3 : 1);
    };
    $("#next").onclick = () => { k = Math.min(n, k + 1); draw(); };
    $("#prev").onclick = () => { k = Math.max(0, k - 1); draw(); };
    $("#play").onclick = () => { clearInterval(timer); k = 0; draw(); timer = setInterval(() => { if (k >= n) return clearInterval(timer); k++; draw(); }, 1600); };
    draw();
  }

  // ---------------------------------------------------------------- side by side
  function sideBySide(g, name) {
    if (!g) { main.innerHTML = ""; return; }
    const col = (side, who) => {
      const d = g[side];
      let h = "";
      for (const m of d.transcript) {
        if (m.from === "buyer") h += UI.buyerMsg("kit", m, null, false);
        else h += UI.sellerMsg(m, who);
      }
      const label = { deal: "deal", seller_walked: "walked away in time", deadlock: "deadlock (−0.25)", buyer_walked: "buyer left" }[d.ended] || d.ended;
      return `<div><div class="mhead"><span class="title">${who.name}</span><span class="sub">${label}</span>
        <div class="right"><span class="score">${(100 * d.score).toFixed(1)}</span></div></div><div class="feed">${h}</div></div>`;
    };
    main.innerHTML = `<div class="split">${col("baseline", UI.BASE)}${col("ours", Object.assign({}, UI.SELLER, { name: "Ours (V11)" }))}</div>`;
    main.querySelectorAll(".feed").forEach((f) => (f.scrollTop = f.scrollHeight));
    const last = g.ours.turns[g.ours.turns.length - 1];
    panel.innerHTML = `<h5>Same buyer, two sellers</h5>
      <div class="card"><div class="k">Buyer</div><div class="v" style="font-size:16px">Kit manipulator</div>
      <div class="muted" style="font-size:12.5px">Fake SYSTEM notes, injected JSON, "you already agreed", fake budgets.</div></div>
      <div class="kv"><div class="card"><div class="k">Baseline</div><div class="v">${(100 * g.baseline.score).toFixed(1)}</div></div>
      <div class="card"><div class="k">Ours (V11)</div><div class="v">${(100 * g.ours.score).toFixed(1)}</div></div></div>` + UI.insight(last).replace(/<h5>.*?<\/h5>/, "");
    setStep(3);
  }

  // ---------------------------------------------------------------- live
  async function breakIt() {
    const cat = await fetch("/api/catalogue").then((r) => r.json()).catch(() => null);
    if (!cat) { main.innerHTML = `<div class="feed">${UI.sys("Start the server: python -X utf8 demo/server.py")}</div>`; panel.innerHTML = UI.insight(null); return; }
    const opts = cat.items.map((i) => `<option value="${i.sku}">${UI.esc(i.name)} · ${UI.rs(i.price)}</option>`).join("");
    main.innerHTML = `<div class="mhead"><span class="title"># try-to-break-it</span><span class="sub" id="rnd">· you are the buyer</span>
      <div class="right"><span class="pill green" id="vio">0 violations</span></div></div>
      <div class="feed" id="feed">${UI.sys("Pick a product, then bargain, bluff or attack. The seller is the submitted V11.")}</div>
      <div class="composer">
        <div class="box"><select id="sku">${opts}</select><input id="qty" type="number" min="1" value="1" style="width:46px"><button class="btn small dark" id="start">Start</button></div>
        <div class="box" style="margin-top:8px"><input class="text" id="text" placeholder="Say something to the seller…"><input class="price" id="price" placeholder="your bid ₹"></div>
        <div class="row"><button class="btn small dark" id="send" disabled>Send</button><button class="btn small" id="acc" disabled>Accept offer</button>
          <span>The seller answers in its next turn.</span><button class="btn small" id="naive" style="margin-left:auto" disabled>Check an LLM-style line</button></div>
      </div>`;
    panel.innerHTML = UI.insight(null);
    let sid = null, feed = $("#feed");
    const add = (h) => { feed.insertAdjacentHTML("beforeend", h); feed.scrollTop = 1e6; };
    $("#start").onclick = async () => {
      const r = await fetch("/api/new", { method: "POST", body: JSON.stringify({ sku: $("#sku").value, qty: +$("#qty").value }) }).then((r) => r.json());
      sid = r.sid;
      add(UI.sys(`New negotiation · list ${UI.rs(r.list_total)} for ${r.qty}`));
      ["#send", "#acc", "#naive"].forEach((s) => ($(s).disabled = false));
    };
    $("#send").onclick = async () => {
      const text = $("#text").value, price = $("#price").value;
      const r = await fetch("/api/turn", { method: "POST", body: JSON.stringify({ sid, text, price: price || null, kind: "counter" }) }).then((r) => r.json());
      if (r.error) return add(UI.sys(r.error));
      const t = r.turn;
      add(UI.buyerMsg("you", t.buyer, null, true, t.tags));
      add(UI.sellerMsg(t.reply));
      panel.innerHTML = UI.insight(t);
      $("#rnd").textContent = `· round ${t.round} of 10`;
      $("#text").value = "";
      setStep(3);
    };
    $("#acc").onclick = async () => {
      const r = await fetch("/api/turn", { method: "POST", body: JSON.stringify({ sid, kind: "accept" }) }).then((r) => r.json());
      add(UI.sys(r.deal ? `Deal at ${UI.rs(UI.itemsTotal(r.deal.items))}` : r.error || "Ended"));
    };
    $("#naive").onclick = async () => {
      const line = $("#text").value || "Only 2 left and this offer expires today, so hurry!";
      const r = await fetch("/api/check", { method: "POST", body: JSON.stringify({ sid, text: line }) }).then((r) => r.json());
      const v = r.violations || [];
      add(UI.sys(`Guard on "${line}": ${v.length ? v.map((x) => x.kind).join(", ") + " · blocked" : "passes"}`));
    };
  }

  // ---------------------------------------------------------------- evidence
  function evidence() {
    const e = evid || { timeline: [] };
    main.innerHTML = `<div class="mhead"><span class="title"># how-we-got-here</span><span class="sub">· every version had to beat the last</span></div>
      <div class="evid"><div class="timeline">${e.timeline.map((x) => `<div class="tl"><div class="ver ${x.cls || ""}">${x.v}</div><div class="t"><b>${UI.esc(x.what)}</b> ${UI.esc(x.why)}</div></div>`).join("")}</div></div>`;
    panel.innerHTML = `<h5>Results</h5>` + (e.cards || []).map((c) => `<div class="card"><div class="k">${UI.esc(c.k)}</div><div class="v">${UI.esc(c.v)} <small>${UI.esc(c.s || "")}</small></div></div>`).join("");
    setStep(2);
  }

  go((location.hash || "#llm-take-it-or-leave-it").slice(1));
})();
