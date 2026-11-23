"""Deterministic fallback engine (plan v3, D2): the seller's safety net and the V0 reference.

Ported from `mvp_seller.py` (frozen). With `beta=1.0` it plays exactly like the MVP on
well-formed input. Changes from the MVP:
  - final-turn logic no longer crashes on non-counter messages (`mvp_seller.py:45`);
  - requests are read leniently: SKU case/whitespace, integer-valued quantities ("2", 2.0),
    a missing quantity means 1; a request with nothing readable gets a clarifying message,
    not a walk-away (it walks at the final turn if still nothing is readable);
  - bids are still accepted strictly: only int quantities equal to ours and numeric prices
    between floor and list, exactly as the harness will check them;
  - defaults are the measured robust settings: front-loaded concession (beta 0.5), floor at the
    final round, walk-away cutoff 0.90 (0.80 after an incomplete counter).
"""
import math
import re

from negotiator.guard import floor_price

TEXT_OFFER = "Here is my offer."
TEXT_ACCEPT = "I can accept that offer."
TEXT_WALK = "I cannot make a compliant offer."
TEXT_NO_DEAL = "No deal this time."
TEXT_CLARIFY = "Could you tell me which items and quantities you would like?"

_INT_STR = re.compile(r"\s*\+?(\d{1,6})\s*")


def norm_sku(sku, store):
    """A store SKU from loosely written input (case, whitespace), or None."""
    if not isinstance(sku, str):
        return None
    s = sku.strip().upper()
    return s if s in store else None


def norm_qty(qty):
    """A positive int from loosely written input ("2", 2.0, missing -> 1), or None."""
    if qty is None:
        return 1
    if isinstance(qty, bool):
        return None
    if isinstance(qty, int):
        return qty if qty >= 1 else None
    if isinstance(qty, float):
        return int(qty) if math.isfinite(qty) and qty.is_integer() and qty >= 1 else None
    if isinstance(qty, str) and (m := _INT_STR.fullmatch(qty)):
        n = int(m[1])
        return n if n >= 1 else None
    return None


def _is_price(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and not (isinstance(v, float) and not math.isfinite(v))


class FallbackEngine:
    def __init__(self, store, policy, beta=0.5, floor_round=None, cut_full=0.90, cut_part=0.80,
                 m1=1.2, m2=2.4):
        self.store, self.policy = store, policy
        self.max_rounds = policy["max_rounds"]
        self.beta = beta
        self.floor_round = floor_round or self.max_rounds
        self.cut_full, self.cut_part, self.m1, self.m2 = cut_full, cut_part, m1, m2
        self.floors = {s: floor_price(r, policy) for s, r in store.items()}
        self.start("init")

    # ------------------------------------------------------------ state
    def start(self, negotiation_id):
        self.basket = None        # None: not read yet; {}: nothing sellable; else sku -> qty
        self.hist = {}            # sku -> [(round, bid price)]

    def prices(self, r):
        prog = 1.0 if self.floor_round <= 1 else min(1.0, max(0, r - 1) / (self.floor_round - 1)) ** self.beta
        return {s: math.ceil(self.store[s]["price"] - (self.store[s]["price"] - self.floors[s]) * prog)
                for s in self.basket}

    def offer_items(self, r):
        """Our offer for round r, or None before a sellable basket exists."""
        if not self.basket:
            return None
        p = self.prices(r)
        return [{"sku": s, "qty": q, "unit_price": p[s]} for s, q in self.basket.items()]

    # ------------------------------------------------------------ turn
    def respond(self, msg):
        early, t, r = self._prepare(msg)
        return early if early is not None else self._decide(msg, t, r)

    def _prepare(self, msg):
        """Shared input handling: (early reply or None, message type, clamped round).

        An early reply is returned for non-dict input, a buyer walk-away, an unsellable basket, and
        an unreadable request (clarify; walk at the final turn). Otherwise the basket is set and
        the subclass-specific `_decide` chooses the reply.
        """
        if not isinstance(msg, dict):
            return {"type": "message", "text": TEXT_CLARIFY}, None, 1
        t = msg.get("type")
        if t == "walk_away":
            return {"type": "walk_away", "text": TEXT_NO_DEAL}, t, 1
        r = msg.get("round")
        r = r if isinstance(r, int) and not isinstance(r, bool) else 1
        r = min(max(r, 1), self.max_rounds)

        if self.basket is None and t in ("request", "counter"):
            readable, sellable = self.read_request(msg.get("items"))
            if readable:
                self.basket = sellable
                if not sellable:
                    return {"type": "walk_away", "text": TEXT_WALK}, t, r
        if not self.basket:
            if r >= self.max_rounds:
                return {"type": "walk_away", "text": TEXT_WALK}, t, r
            return {"type": "message", "text": TEXT_CLARIFY}, t, r
        return None, t, r

    def _decide(self, msg, t, r):
        """The fallback's pricing: time-based concession, projected final-turn walk, accept at ask."""
        p = self.prices(r)
        counter = self.counter_prices(msg.get("items")) if t == "counter" else {}
        for s, v in counter.items():
            self.hist.setdefault(s, []).append((r, v))
        if r == self.max_rounds:
            est = self.project(r, counter)
            if est is not None:
                proj_total, complete = est
                floor_total = sum(q * self.floors[s] for s, q in self.basket.items())
                if proj_total < (self.cut_full if complete else self.cut_part) * floor_total:
                    return {"type": "walk_away", "text": TEXT_WALK}
        bid = self.valid_bid(msg.get("items")) if t in ("request", "counter") else None
        if bid and all(bid[s] >= p[s] for s in self.basket):
            return {"type": "accept", "text": TEXT_ACCEPT}
        return {"type": "offer", "items": [{"sku": s, "qty": q, "unit_price": p[s]} for s, q in self.basket.items()],
                "text": TEXT_OFFER}

    # ------------------------------------------------------------ reading
    def read_request(self, items):
        """(anything readable?, sellable basket) from a request's items."""
        readable, basket = False, {}
        for it in (items if isinstance(items, list) else [])[:len(self.store)]:
            if not isinstance(it, dict):
                continue
            s, q = norm_sku(it.get("sku"), self.store), norm_qty(it.get("qty"))
            if s is None or q is None:
                continue
            readable = True
            if self.store[s]["stock"] > 0:
                basket[s] = min(self.store[s]["stock"], basket.get(s, 0) + q)
        return readable, basket

    def counter_prices(self, items):
        """Usable bid prices for our SKUs, for the final-turn projection only (never for accepting)."""
        out, bad = {}, set()
        for it in (items if isinstance(items, list) else [])[:len(self.store) + len(self.basket)]:
            if not isinstance(it, dict):
                continue
            s, v = norm_sku(it.get("sku"), self.store), it.get("unit_price")
            if s is None or s not in self.basket:
                continue
            if s in out or not _is_price(v):
                bad.add(s)
                out.pop(s, None)
                continue
            out[s] = v
        for s in bad:
            out.pop(s, None)
        return out

    def project(self, r, counter):
        """(projected final bid total, whether the last counter covered every line), or None."""
        total, complete = 0.0, counter.keys() == self.basket.keys()
        for s, q in self.basket.items():
            h = [e for e in self.hist.get(s, []) if e[0] < r]
            if s in counter and h:
                pr, pp = h[-1]
                proj = counter[s] + self.m1 * max(0, (counter[s] - pp) / (r - pr))
            elif len(h) >= 2:
                (pr, pp), (lr, lp) = h[-2], h[-1]
                proj = lp + self.m2 * max(0, (lp - pp) / (lr - pr))
                complete = False
            else:
                return None
            total += q * proj
        return total, complete

    def valid_bid(self, items):
        """The bid as {sku: price} if it is exactly our basket at policy-valid prices, else None."""
        if not isinstance(items, list) or not items:
            return None
        bid = {}
        for it in items[:len(self.store)]:
            if not isinstance(it, dict):
                return None
            s, q, v = it.get("sku"), it.get("qty"), it.get("unit_price")
            if not isinstance(s, str) or s not in self.basket or s in bid:
                return None
            if not isinstance(q, int) or isinstance(q, bool) or q != self.basket[s]:
                return None
            if not _is_price(v) or not self.floors[s] <= v <= self.store[s]["price"]:
                return None
            bid[s] = v
        return bid if bid.keys() == self.basket.keys() else None
