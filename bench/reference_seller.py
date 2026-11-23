"""Reference seller for the bench: the MVP's logic with knobs.

Copied from `context/experiments/tradeoffs.py:31-178` (`ParamSeller`). The only change is that
the floor prices and round limit, module globals there, are computed per instance here so the
file imports without running anything. Defaults reproduce `mvp_seller.py` on the public bots
(50.41 / 50.94 / 44.65 on seeds 0-2); `beta=0.5` is the front-loaded fallback reference
(`tradeoff-analysis.md` §3-§5).

Knobs:
    beta          concession shape: price = list - (list - floor) * progress**beta
    floor_round   round at which the offer reaches the floor
    accept        "ask" (bid >= current ask), "next" (bid >= next ask), "floor" (any valid bid)
    walk          "mvp" (projected-bid cutoff), "never", "walk10" (walk unless the bid is acceptable)
    cut_full / cut_part   walk if projected final bid < cut x floor total (complete / partial projection)
    m1, m2        projection multipliers on the last price step
    partial       at the final turn, offer the best sub-basket instead of walking
    final_kappa   price the last offer at kappa x projected buyer limit

Usable as a seller spec: `bench/reference_seller.py:ParamSeller beta=0.5`.
"""
import itertools
import math

from harness import floor_price


class ParamSeller:
    """MVP logic (mvp_seller.py) with knobs. Defaults reproduce the MVP on the public bots."""

    def __init__(self, store, policy, beta=1.0, floor_round=10, accept="ask", walk="mvp",
                 cut_full=0.90, cut_part=0.80, m1=1.2, m2=2.4, partial=False, final_kappa=None):
        self.final_kappa = final_kappa
        self.store, self.policy = store, policy
        self.floor = {s: floor_price(r, policy) for s, r in store.items()}
        self.maxr = policy["max_rounds"]
        self.beta, self.floor_round, self.accept, self.walk = beta, floor_round, accept, walk
        self.cut_full, self.cut_part, self.m1, self.m2, self.partial = cut_full, cut_part, m1, m2, partial
        self.basket, self.hist = None, {}

    def start(self, nid):
        self.basket, self.hist = None, {}

    def prices(self, r):
        prog = 1.0 if self.floor_round <= 1 else min(1.0, (r - 1) / (self.floor_round - 1)) ** self.beta
        return {s: math.ceil(self.store[s]["price"] - (self.store[s]["price"] - self.floor[s]) * prog)
                for s in self.basket}

    def respond(self, msg):
        FLOOR, MAXR = self.floor, self.maxr
        t = msg.get("type")
        if t == "walk_away":
            return {"type": "walk_away", "text": "No deal for now."}
        if self.basket is None and t in ("request", "counter"):
            self.basket = {}
            for it in (msg.get("items") or [])[:len(self.store)]:
                if not isinstance(it, dict):
                    continue
                s, q = it.get("sku"), it.get("qty")
                if isinstance(s, str) and s in self.store and isinstance(q, int) and not isinstance(q, bool) \
                        and q >= 1 and self.store[s]["stock"] > 0:
                    self.basket[s] = min(self.store[s]["stock"], self.basket.get(s, 0) + q)
            if not self.basket:
                return {"type": "walk_away", "text": "I cannot make a compliant offer."}
        if not self.basket:
            return {"type": "message", "text": "Please send the items you'd like to buy."}
        r = msg.get("round") if isinstance(msg.get("round"), int) else 1
        r = min(max(r, 1), MAXR)
        p = self.prices(r)
        counter = self.counter(msg.get("items")) if t == "counter" else {}
        for s, v in counter.items():
            self.hist.setdefault(s, []).append((r, v))
        if r == MAXR and self.walk != "never":
            if self.walk == "walk10":
                bid = self.valid_bid(msg.get("items")) if t in ("request", "counter") else None
                if not (bid and self.bid_ok(bid, p, r)):
                    return {"type": "walk_away", "text": "I cannot make a compliant offer."}
            else:
                est = self.project(r, counter)
                if est is not None:
                    proj_total, complete = est
                    floor_total = sum(q * FLOOR[s] for s, q in self.basket.items())
                    cut = self.cut_full if complete else self.cut_part
                    if proj_total < cut * floor_total:
                        sub = self.subset(proj_total) if self.partial else None
                        if sub:
                            return {"type": "offer", "items": sub, "text": "Here is my offer."}
                        return {"type": "walk_away", "text": "I cannot make a compliant offer."}
        if r == MAXR and self.final_kappa:
            # price the last offer at the projected buyer limit instead of the floor
            est = self.project(r, counter)
            if est is not None:
                listed = sum(q * self.store[s]["price"] for s, q in self.basket.items())
                rho = self.final_kappa * est[0] / listed
                prev = self.prices(MAXR - 1)
                p = {s: max(FLOOR[s], min(prev[s], math.ceil(rho * self.store[s]["price"]))) for s in self.basket}
        bid = self.valid_bid(msg.get("items")) if t in ("request", "counter") else None
        if bid and self.bid_ok(bid, p, r):
            return {"type": "accept", "text": "I can accept that offer."}
        return {"type": "offer", "items": [{"sku": s, "qty": q, "unit_price": p[s]} for s, q in self.basket.items()],
                "text": "Here is my offer."}

    def bid_ok(self, bid, p, r):
        if self.accept == "ask":
            return all(bid[s] >= p[s] for s in self.basket)
        if self.accept == "next":
            nxt = self.prices(min(r + 1, self.maxr))
            return sum(q * bid[s] for s, q in self.basket.items()) >= sum(q * nxt[s] for s, q in self.basket.items())
        return True                                   # "floor": any valid bid (valid_bid already checks floor)

    def project(self, r, counter):
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

    def subset(self, proj_total):
        """Best sub-basket whose floor fits under the projected buyer ratio (lines/qty may be cut)."""
        FLOOR = self.floor
        listed = sum(q * self.store[s]["price"] for s, q in self.basket.items())
        rho = proj_total / listed
        best, best_profit = None, 0
        skus = list(self.basket)
        for qs in itertools.product(*[range(self.basket[s] + 1) for s in skus]):
            lines = [(s, q) for s, q in zip(skus, qs) if q]
            if not lines or len(lines) == len(skus) and all(q == self.basket[s] for s, q in lines):
                continue
            fl = sum(q * FLOOR[s] for s, q in lines)
            li = sum(q * self.store[s]["price"] for s, q in lines)
            if fl <= rho * li:
                profit = sum(q * (FLOOR[s] - self.store[s]["cost"]) for s, q in lines)
                if profit > best_profit:
                    best, best_profit = lines, profit
        return [{"sku": s, "qty": q, "unit_price": FLOOR[s]} for s, q in best] if best else None

    def counter(self, items):
        out, bad = {}, set()
        for it in (items if isinstance(items, list) else [])[:len(self.store) + len(self.basket)]:
            if not isinstance(it, dict):
                continue
            s, v = it.get("sku"), it.get("unit_price")
            if not isinstance(s, str) or s not in self.basket:
                continue
            if s in out or not isinstance(v, (int, float)) or isinstance(v, bool) or \
                    (isinstance(v, float) and not math.isfinite(v)):
                bad.add(s)
                out.pop(s, None)
                continue
            out[s] = v
        for s in bad:
            out.pop(s, None)
        return out

    def valid_bid(self, items):
        FLOOR = self.floor
        if not isinstance(items, list) or not items:
            return None
        bid = {}
        for it in items[:len(self.store)]:
            if not isinstance(it, dict):
                return None
            s, q, v = it.get("sku"), it.get("qty"), it.get("unit_price")
            if not isinstance(s, str) or s not in self.basket or s in bid:
                return None
            if q != self.basket[s] or not isinstance(q, int) or isinstance(q, bool):
                return None
            if not isinstance(v, (int, float)) or isinstance(v, bool) or (isinstance(v, float) and not math.isfinite(v)):
                return None
            if not FLOOR[s] <= v <= self.store[s]["price"]:
                return None
            bid[s] = v
        return bid if bid.keys() == self.basket.keys() else None


Seller = ParamSeller
