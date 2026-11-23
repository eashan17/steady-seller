// Shared, pure render functions for the demo app (app.js) and the video composition (video.js).
(function () {
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const rs = (n) => (n == null || isNaN(n) ? "—" : "₹" + Math.round(n).toLocaleString("en-IN"));
  const pct = (p) => (p == null ? "—" : Math.round(100 * p) + "%");

  const BUYER = {
    llm: { av: "LB", cls: "llm", name: "LLM buyer", meta: "gpt-oss-120b" },
    kit: { av: "KB", cls: "buyer", name: "Kit manipulator", meta: "buyers.py" },
    you: { av: "Y", cls: "you", name: "You", meta: "playing the buyer" },
  };
  const SELLER = { av: "S", cls: "seller", name: "Seller", meta: "V11 · no LLM" };
  const BASE = { av: "B", cls: "base", name: "Baseline seller", meta: "example_seller.py" };

  function msg(p, body, extra = "") {
    return `<div class="msg"><div class="av ${p.cls}">${p.av}</div><div>
      <div class="who">${esc(p.name)} <span class="meta">${esc(p.meta)}</span>${p.time ? `<span class="meta">${p.time}</span>` : ""}</div>
      <div class="body">${body}</div>${extra}</div></div>`;
  }

  function tagChips(tags) {
    if (!tags || !tags.length) return "";
    return `<div class="tagrow">${tags.map((t) => `<span class="tag ${t.kind}">${esc(t.label)}</span>`).join("")}</div>`;
  }

  function itemsTotal(items) {
    if (!Array.isArray(items)) return null;
    let s = 0, ok = false;
    for (const i of items) if (i && i.unit_price != null) { s += (i.qty || 0) * i.unit_price; ok = true; }
    return ok ? s : null;
  }

  function buyerMsg(kind, b, text, showTags, tags) {
    const p = BUYER[kind] || BUYER.llm;
    const total = itemsTotal(b.items);
    let body = esc(text ?? b.text ?? "");
    if (b.type === "accept") body = (body ? body + " " : "") + `<span class="code">accept</span>`;
    else if (b.type === "walk_away") body = (body ? body + " " : "") + `<span class="code">walk_away</span>`;
    else if (total != null) body += ` <span class="code">bid ${rs(total)}</span>`;
    return msg(p, body, showTags ? tagChips(tags) : "");
  }

  function sellerMsg(reply, who = SELLER, text) {
    const total = itemsTotal(reply.items);
    let body = esc(text ?? reply.text ?? "");
    if (reply.type === "offer" && total != null) body += ` <span class="code offer">offer ${rs(total)}</span>`;
    else if (reply.type !== "offer" && reply.type !== "message") body += ` <span class="code">${esc(reply.type)}</span>`;
    return msg(who, body);
  }

  function sys(text) { return `<div class="sys">${esc(text)}</div>`; }

  // Belief density of the buyer's limit, with our floor, the buyer's bid and our offer marked.
  function beliefSVG(t, w = 272, h = 118) {
    const c = (t.belief && t.belief.curve) || [];
    if (!c.length) return `<div class="muted" style="font-size:12px">The belief starts after the buyer's first bid.</div>`;
    const L = t.list_total, F = t.floor_total;
    const x0 = Math.min(c[0].x, t.buyer_total ?? c[0].x), x1 = L;
    const X = (x) => 8 + ((x - x0) / (x1 - x0)) * (w - 16);
    const pm = Math.max(...c.map((d) => d.p)) || 1;
    const Y = (p) => h - 22 - (p / pm) * (h - 34);
    let d = `M ${X(c[0].x)} ${h - 22}`;
    for (const q of c) d += ` L ${X(q.x).toFixed(1)} ${Y(q.p).toFixed(1)}`;
    d += ` L ${X(c[c.length - 1].x)} ${h - 22} Z`;
    const vline = (x, col, dash, lab, dy) => x == null || x < x0 || x > x1 ? "" :
      `<line x1="${X(x)}" x2="${X(x)}" y1="8" y2="${h - 22}" stroke="${col}" stroke-width="1.5" ${dash ? 'stroke-dasharray="3 3"' : ""}/>
       <text x="${Math.min(w - 34, Math.max(2, X(x) + 3))}" y="${dy ? 18 : h - 7}" font-size="10" fill="${col}">${lab}</text>`;
    return `<svg viewBox="0 0 ${w} ${h}" width="100%" height="${h}">
      <line x1="8" x2="${w - 8}" y1="${h - 22}" y2="${h - 22}" stroke="#e8e8ec"/>
      <path d="${d}" fill="rgba(122,90,245,.16)" stroke="#7a5af5" stroke-width="1.4"/>
      ${vline(F, "#d4553a", true, "floor", 0)}
      ${vline(t.buyer_total, "#8a8d96", true, "bid", 1)}
      ${vline(t.reply_total, "#16171a", false, "offer", 1)}
    </svg>`;
  }

  function insight(t, opts = {}) {
    if (!t) return `<h5>Seller's view</h5><div class="card muted" style="font-size:13px">Waiting for the buyer's first message.</div>`;
    const b = t.belief || {};
    const g = t.guard || {};
    const checks = ["Price between floor and list", "Stock and quantities", "Every number is in play",
      "No pressure or scarcity words", "No fees or unsupported claims"];
    const bad = g.violations_in_sent_reply > 0;
    return `<h5>Seller's view · round ${t.round ?? "—"}</h5>
      <div class="card"><div class="k">Buyer's hidden limit (estimate)</div>
        <div class="v">${rs(b.limit_med)} <small>${rs(b.limit_lo)} – ${rs(b.limit_hi)}</small></div>
        ${beliefSVG(t)}
        <div class="legend"><span><i style="background:#7a5af5"></i>belief</span><span><i style="background:#d4553a"></i>our floor</span><span><i style="background:#16171a"></i>our offer</span></div>
      </div>
      <div class="kv">
        <div class="card"><div class="k">Won't walk away</div><div class="v">${pct(b.p_stubborn)}</div><div class="meter"><span style="width:${100 * (b.p_stubborn || 0)}%"></span></div></div>
        <div class="card"><div class="k">Deal possible</div><div class="v">${pct(b.p_deal_possible)}</div><div class="meter"><span style="width:${100 * (b.p_deal_possible || 0)}%;background:#1f9d55"></span></div></div>
      </div>
      <div class="card"><div class="k">What the text says</div>${t.tags && t.tags.length ? tagChips(t.tags) : '<div class="muted" style="font-size:12.5px;margin-top:4px">Nothing notable</div>'}
        <div class="muted" style="font-size:11.5px;margin-top:7px">Evidence about the buyer, never a lever on the price.</div></div>
      <div class="card"><div class="k">Guard · checked before sending</div>
        <div class="checks" style="margin-top:6px">${checks.map((c) => `<div class="${bad ? "bad" : ""}">${c}</div>`).join("")}</div></div>`;
  }

  window.UI = { esc, rs, pct, msg, buyerMsg, sellerMsg, sys, insight, beliefSVG, tagChips, itemsTotal, BUYER, SELLER, BASE };
})();
