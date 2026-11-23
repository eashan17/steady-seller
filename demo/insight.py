"""Demo only: run the real seller and record, after every turn, what it read, believed and decided.

Nothing here changes the seller. `InsightSeller` wraps `seller.Seller` exactly as the harness would call it
and reads the engine's state after each `respond`, for the replay viewer and the live "try to break it" page.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import harness  # noqa: E402
from negotiator import guard  # noqa: E402
from seller import DEFAULT_MODE, Seller  # noqa: E402

TAG_LABELS = {   # negotiator/textsignals.py TAGS -> plain words; kind: intent, claim, attack, tone
    "accept_signal": ("leaning to accept", "intent"), "price_objection": ("price objection", "intent"),
    "walk_threat": ("walk-away threat", "intent"), "bundle_request": ("bundle request", "intent"),
    "scope_change": ("changes the basket", "intent"), "info_request": ("asks a question", "intent"),
    "urgency": ("time pressure", "intent"),
    "claim_budget": ("budget claim", "claim"), "claim_competitor": ("competitor quote", "claim"),
    "claim_final": ("\"final offer\" claim", "claim"), "claim_seller_commitment": ("\"you already agreed\"", "claim"),
    "claim_hardship": ("hardship story", "claim"),
    "attack_injection": ("prompt injection", "attack"), "attack_authority": ("fake authority", "attack"),
    "attack_threat": ("threat", "attack"), "attack_extraction": ("asks for internal data", "attack"),
    "tone_friendly": ("friendly tone", "tone"), "tone_hostile": ("hostile tone", "tone"),
}


STORE, POLICY = harness.load_store(), harness.load_policy()


def basket_totals(basket):
    """(list total, floor total) of a {sku: qty} basket."""
    L = sum(STORE[s]["price"] * q for s, q in basket.items())
    F = sum(harness.floor_price(STORE[s], POLICY) * q for s, q in basket.items())
    return L, F


def items_total(items):
    try:
        return sum(int(i["qty"]) * float(i["unit_price"]) for i in items or [])
    except Exception:
        return None


class InsightSeller:
    """The V9 seller (or any mode) plus a per-turn record of its reading, belief and decision."""

    def __init__(self, mode=None):
        self.mode = mode or DEFAULT_MODE
        self.s = Seller(STORE, POLICY, mode=self.mode)
        self.turns = []

    def start(self, nid):
        self.s.start(nid)
        self.turns = []

    def respond(self, msg):
        before = dict(self.s.stats)
        out = self.s.respond(msg)
        self.turns.append(self._record(msg, out, before))
        return out

    def _record(self, msg, out, before):
        e = getattr(self.s, "engine", None)
        rec = {"round": msg.get("round") if isinstance(msg, dict) else None,
               "buyer": {k: msg.get(k) for k in ("type", "items", "text")} if isinstance(msg, dict) else {},
               "reply": out,
               "guard": {"rejected": self.s.stats["guard_rejections"] - before.get("guard_rejections", 0),
                         "text_retries": self.s.stats["text_retries"] - before.get("text_retries", 0),
                         "violations_in_sent_reply": len(guard.check(out, self.s._guard_state()))
                         if out.get("type") != "accept" else 0}}
        rec["buyer_total"] = items_total(msg.get("items")) if isinstance(msg, dict) else None
        rec["reply_total"] = items_total(out.get("items")) if out.get("type") == "offer" else None
        basket = getattr(e, "basket", None)
        if basket:
            L, F = basket_totals(basket)
            rec["basket"] = [{"sku": s, "qty": q, "name": STORE[s]["name"], "list": STORE[s]["price"]}
                             for s, q in basket.items()]
            rec["list_total"], rec["floor_total"] = L, F
        tags = list(getattr(e, "current_tags", []) or [])
        rec["tags"] = [{"tag": t, "label": TAG_LABELS.get(t, (t.replace("_", " "), "intent"))[0],
                        "kind": TAG_LABELS.get(t, ("", "intent"))[1]} for t in tags]
        bel = getattr(e, "belief", None)
        if bel is not None and basket:
            L, F = rec["list_total"], rec["floor_total"]
            med, lo, hi = bel.limit_estimate(effective=True)
            rec["belief"] = {
                "limit_med": med * L, "limit_lo": lo * L, "limit_hi": hi * L,
                "p_stubborn": sum(c.w for c in bel.cells if c.stub),
                "p_deal_possible": bel.p_feasible(effective=True),
                "leave_risk": getattr(e, "leave_risk", None),
                "curve": self._density(bel, L, F),
            }
        return rec

    @staticmethod
    def _density(bel, L, F):
        """Belief density of the buyer's effective limit, on 60 points from 60% of the floor to list."""
        lo, n = max(bel.opening, 0.6 * F / L), 60
        xs = [lo + (1.0 - lo) * i / n for i in range(n + 1)]
        sf = [bel.p_limit_at_least(x, effective=True) for x in xs]
        return [{"x": round(xs[i] * L, 1), "p": max(0.0, sf[i] - sf[i + 1])} for i in range(n)]


def replay(buyer_msgs, mode=None, nid="replay"):
    """Feed recorded buyer messages to a fresh seller; returns (turns, replies)."""
    s = InsightSeller(mode)
    s.start(nid)
    replies = []
    for m in buyer_msgs:
        replies.append(s.respond(json.loads(json.dumps(m))))
    return s.turns, replies
