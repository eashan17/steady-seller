"""Demo server: the replay viewer, the side-by-side view, the evidence page, and a live "try to break it" channel
where you play the buyer against the real seller (default mode, V11).

    python -X utf8 demo/server.py            # then open http://127.0.0.1:8765

Standard library only. Binds to 127.0.0.1. The seller runs in this process exactly as the kit harness calls it.
"""
import json
import sys
import threading
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from demo.insight import POLICY, STORE, InsightSeller, basket_totals  # noqa: E402
from negotiator import guard  # noqa: E402
from seller import DEFAULT_MODE  # noqa: E402

STATIC = HERE / "static"
SESSIONS, LOCK = {}, threading.Lock()


def catalogue():
    return [{"sku": s, "name": r["name"], "price": r["price"], "stock": r["stock"], "category": r["category"]}
            for s, r in STORE.items() if r["stock"] > 0]


def new_session(sku, qty):
    s = InsightSeller(DEFAULT_MODE)
    sid = uuid.uuid4().hex[:10]
    s.start(f"live-{sid}")
    with LOCK:
        SESSIONS[sid] = {"seller": s, "round": 0, "sku": sku, "qty": qty, "done": False}
    return sid


def buyer_turn(sid, text, price, kind):
    st = SESSIONS[sid]
    if st["done"]:
        return {"error": "This negotiation has ended. Start a new one."}
    st["round"] += 1
    r = st["round"]
    if kind == "accept":
        msg = {"type": "accept", "text": text or "Deal.", "round": r}
    elif kind == "walk_away":
        msg = {"type": "walk_away", "text": text or "No thanks.", "round": r}
    else:
        item = {"sku": st["sku"], "qty": st["qty"]}
        if price:
            item["unit_price"] = round(float(price) / st["qty"])
        msg = {"type": "request" if r == 1 else "counter", "items": [item], "text": text, "round": r}
    if kind in ("accept", "walk_away"):
        st["done"] = True
        last = next((t["reply"] for t in reversed(st["seller"].turns) if t["reply"].get("type") == "offer"), None)
        return {"ended": "deal" if kind == "accept" and last else "buyer_walked", "deal": last}
    st["seller"].respond(msg)
    rec = st["seller"].turns[-1]
    if rec["reply"]["type"] in ("accept", "walk_away") or r >= POLICY["max_rounds"]:
        st["done"] = rec["reply"]["type"] in ("accept", "walk_away")
    return {"turn": rec}


def check_line(sid, text):
    """Run the guard on a hypothetical seller line (what an LLM seller might write) in this negotiation's context."""
    st = SESSIONS.get(sid)
    state = st["seller"].s._guard_state() if st else None
    if state is None:
        return {"error": "Start a negotiation first."}
    v = guard.check({"type": "message", "text": text}, state)
    return {"violations": v}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(STATIC), **k)

    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/catalogue":
            return self._json({"items": catalogue(), "mode": DEFAULT_MODE})
        return super().do_GET()

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            data = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/api/new":
                sku, qty = data["sku"], max(1, min(int(data.get("qty", 1)), STORE[data["sku"]]["stock"]))
                L, F = basket_totals({sku: qty})
                return self._json({"sid": new_session(sku, qty), "qty": qty, "list_total": L, "floor_total": F})
            if self.path == "/api/turn":
                return self._json(buyer_turn(data["sid"], str(data.get("text") or "")[:600],
                                             data.get("price"), data.get("kind", "counter")))
            if self.path == "/api/check":
                return self._json(check_line(data.get("sid"), str(data.get("text") or "")[:600]))
        except Exception as e:                      # demo server: report, never crash
            return self._json({"error": f"{type(e).__name__}: {e}"}, 400)
        return self._json({"error": "unknown endpoint"}, 404)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    print(f"Demo on http://127.0.0.1:{port}  (seller mode {DEFAULT_MODE})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
