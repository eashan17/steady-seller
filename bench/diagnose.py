"""Bottleneck diagnosis: where does the seller lose points, by negotiation type and by pipeline step?

    python bench/diagnose.py bots  --modes V0,V1,V5 --out <dir>      # kit bots, seeds 0-4 (free)
    python bench/diagnose.py llm   --rows <llm-eval dir> ...          # LLM-buyer transcripts already played
    python bench/diagnose.py llm-oracle --n 50 --workers 3 --out <dir> # oracle vs Codex buyers (costs calls)

Oracle seller: knows the buyer's private limit R and deadline. Against the kit bots it offers list until the
buyer's last useful round (min(deadline, 10)), then the highest basket total the buyer accepts there
(R x list + 1, every line at or above its floor); it walks at round 1 when R x list is below the floor total.
Played through the real harness, so its score is the achievable maximum, not a formula.

Loss attribution for the kit bots, per feasible negotiation, in score points (value x 100), using the bot's
true willingness w(j) (buyers.Buyer.ratio):
  missed_walk / missed_deadlock / missed_buyer_left   no deal although one was possible
  accepted_bid_below_limit                            we accepted the buyer's bid b < R      (R - b)
  offer_below_willingness                             our accepted offer x was below w(j)    (w(j) - x)
  closed_before_limit                                 the deal round's w(j) was below R      (R - w(j))
plus deadlock penalties on infeasible buyers. The pieces add up to oracle - achieved.
"""
import argparse
import json
import math
import random
import statistics
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
KIT = HERE.parent
sys.path.insert(0, str(KIT))
sys.path.insert(0, str(HERE))

import buyers  # noqa: E402
import evaluate  # noqa: E402
import harness  # noqa: E402
from negotiator.guard import floor_price  # noqa: E402

STORE, POLICY = harness.load_store(), harness.load_policy()
FLOOR = {s: floor_price(r, POLICY) for s, r in STORE.items()}
M = POLICY["max_rounds"]


def basket_of(sc):
    return {i["sku"]: min(i["qty"], STORE[i["sku"]]["stock"]) for i in sc["items"] if STORE[i["sku"]]["stock"] > 0}


def totals(basket):
    L = sum(q * STORE[s]["price"] for s, q in basket.items())
    C = sum(q * STORE[s]["cost"] for s, q in basket.items())
    F = sum(q * FLOOR[s] for s, q in basket.items())
    return L, C, F


def lines_at_total(basket, target):
    """Integer per-line prices, each between floor and list, summing to at most `target` (as high as possible)."""
    L, _, F = totals(basket)
    lam = 0.0 if L <= F else min(1.0, max(0.0, (target - F) / (L - F)))
    items = [{"sku": s, "qty": q, "unit_price": min(STORE[s]["price"],
                                                    math.floor(FLOOR[s] + lam * (STORE[s]["price"] - FLOOR[s])))}
             for s, q in basket.items()]
    total = sum(i["qty"] * i["unit_price"] for i in items)
    for i in items:                                    # top up with spare rupees where a line still has room
        while total + i["qty"] <= target and i["unit_price"] < STORE[i["sku"]]["price"]:
            i["unit_price"] += 1
            total += i["qty"]
    return items


# ---------------------------------------------------------------- oracle sellers
class OracleSeller:
    """Knows the scenario (set .scenario before each negotiation). Kit-bot timing: hold list, strike at the last useful round."""
    hold_until_last = True
    limit_slack = 1

    def __init__(self, store, policy):
        self.scenario = None

    def start(self, nid):
        sc = self.scenario
        self.basket = basket_of(sc)
        L, _, F = totals(self.basket) if self.basket else (0, 0, 0)
        # kit bots accept up to R x list + 1 (buyers.py:80); LLM buyers are told int(R x list) as their maximum
        # (llm_buyers.LLMBuyer.limit) and refuse even a rupee more, so the LLM oracle offers exactly that
        self.limit_total = (int(sc["reservation"] * L) + self.limit_slack) if self.basket else 0
        self.feasible = bool(self.basket) and self.limit_total >= F
        self.strike = min(sc["deadline"], M) if self.hold_until_last else 1

    def respond(self, msg):
        r = msg.get("round", 1)
        if not self.feasible:
            return {"type": "walk_away", "text": "I cannot make a compliant offer."}
        if r < self.strike:
            items = [{"sku": s, "qty": q, "unit_price": STORE[s]["price"]} for s, q in self.basket.items()]
        else:
            items = lines_at_total(self.basket, self.limit_total)
        return {"type": "offer", "items": items, "text": "Here is my offer."}


class OracleLLMSeller(OracleSeller):
    """Against LLM buyers: offer the buyer's stated limit from round 1 and hold it (they may accept any time)."""
    hold_until_last = False
    limit_slack = 0


# ---------------------------------------------------------------- tracing our seller
class Traced:
    """Wraps seller.Seller to record, after every turn, our ask and what the belief thought."""

    def __init__(self, mode, **params):
        from seller import Seller
        self.s = Seller(STORE, POLICY, mode=mode, **params)
        self.turns = []

    def start(self, nid):
        self.s.start(nid)
        self.turns = []

    def respond(self, msg):
        out = self.s.respond(msg)
        e, rec = self.s.engine, {"round": msg.get("round") if isinstance(msg, dict) else None, "type": out["type"]}
        bel = getattr(e, "belief", None)
        if out["type"] == "offer" and e.basket:
            L, _, _ = totals(e.basket)
            rec["ask"] = sum(i["qty"] * i["unit_price"] for i in out["items"]) / L
        if bel is not None:
            med, lo, hi = bel.limit_estimate()
            emed, elo, ehi = bel.limit_estimate(effective=True)
            rec.update(R_med=med, R_lo=lo, R_hi=hi, p_stub=sum(c.w for c in bel.cells if c.stub),
                       p_feasible=bel.p_feasible(), rel=bel.reliability,
                       Re_med=emed, Re_lo=elo, Re_hi=ehi, pe_feasible=bel.p_feasible(effective=True))
        rec["tags"] = list(getattr(e, "current_tags", []) or [])
        self.turns.append(rec)
        return out


def play(seller_factory, suite_classes, seeds, n, oracle=False):
    out = []
    for seed in seeds:
        seller = seller_factory()
        for sc in buyers.make_scenarios(STORE, POLICY, n, seed, suite_classes):
            b = suite_classes[sc["buyer"]](sc, STORE, random.Random(sc["seed"]))
            if oracle:
                seller.scenario = sc
            res = harness.run_negotiation(seller, b, STORE, POLICY, sc["id"])
            row = evaluate.score_one(res, sc, STORE, POLICY)
            out.append({"sc": sc, "res": res, "row": row, "turns": list(getattr(seller, "turns", []))})
    return out


# ---------------------------------------------------------------- attribution (kit bots)
def attribute(rec):
    """Points lost against the oracle value, by cause (score points = value x 100)."""
    sc, res, row = rec["sc"], rec["res"], rec["row"]
    basket = basket_of(sc)
    if not basket:
        return {}, {}
    L, C, F = totals(basket)
    R = sc["reservation"]
    to_pts = 100.0 * L / (L - C) if L > C else 0.0
    feasible = R * L >= F
    bot = buyers.BUYERS.get(sc["buyer"], buyers.Buyer)(sc, STORE, random.Random(0))
    w = lambda j: bot.ratio(j)
    loss, info = {}, {}
    if not feasible:
        if res["ended"] == "deadlock":
            loss["deadlock_penalty"] = 25.0
        return loss, info
    oracle_pts = 100.0 * (R * L - C) / (L - C)
    if not res["deal"]:
        key = {"seller_walked": "missed_walk", "deadlock": "missed_deadlock", "buyer_walked": "missed_buyer_left"}
        loss[key.get(res["ended"], "missed_other")] = oracle_pts
        if res["ended"] == "deadlock":
            loss["deadlock_penalty"] = 25.0
        return loss, info
    t = res["transcript"]
    last = t[-1]
    deal_share = sum(i["qty"] * i["unit_price"] for i in res["deal"]) / L
    j = res["rounds"]
    if last["from"] == "seller" and last["type"] == "accept":          # we accepted the buyer's bid
        loss["accepted_bid_below_limit"] = max(0.0, R - deal_share) * to_pts
        info["closed_by"] = "we accepted bid"
    else:                                                              # the buyer accepted our offer
        offer_round = max(m["round"] for m in t if m["from"] == "seller" and m["type"] == "offer")
        wj = min(R, w(offer_round))
        loss["offer_below_willingness"] = max(0.0, wj - deal_share) * to_pts
        loss["closed_before_limit"] = max(0.0, R - wj) * to_pts
        info["closed_by"] = "buyer accepted offer"
        j = offer_round
    info.update(deal_round=j, deal_share=deal_share, R=R)
    return loss, info


def summarize_bots(records, label):
    by = {}
    for rec in records:
        loss, info = attribute(rec)
        b = rec["sc"]["buyer"]
        d = by.setdefault(b, {"n": 0, "feasible": 0, "score": 0.0, "losses": {}, "deal_rounds": [], "closed_by": {}})
        d["n"] += 1
        d["score"] += 100 * rec["row"]["score"]
        d["feasible"] += int(rec["row"]["feasible"])
        for k, v in loss.items():
            d["losses"][k] = d["losses"].get(k, 0.0) + v
        if "deal_round" in info:
            d["deal_rounds"].append(info["deal_round"])
            d["closed_by"][info["closed_by"]] = d["closed_by"].get(info["closed_by"], 0) + 1
    out = {}
    for b, d in by.items():
        out[b] = {"n": d["n"], "feasible": d["feasible"], "score": d["score"] / d["n"],
                  "losses_per_negotiation": {k: v / d["n"] for k, v in sorted(d["losses"].items(), key=lambda kv: -kv[1])},
                  "deal_round_mean": statistics.mean(d["deal_rounds"]) if d["deal_rounds"] else None,
                  "deal_round_hist": {r: d["deal_rounds"].count(r) for r in sorted(set(d["deal_rounds"]))},
                  "closed_by": d["closed_by"]}
    return {"label": label, "by_buyer": out}


def step_metrics(records):
    """Per buyer type: opening ask, ask path vs limit, belief error and type identification, final-turn cap."""
    by = {}
    for rec in records:
        sc, turns = rec["sc"], rec["turns"]
        basket = basket_of(sc)
        if not basket or not turns:
            continue
        L, C, F = totals(basket)
        R = sc["reservation"]
        d = by.setdefault(sc["buyer"], {"open": [], "below_R_round": [], "R_err": {3: [], 6: [], 9: []},
                                       "pstub_err": [], "ask_at": {5: [], 8: [], 9: [], 10: []}, "n": 0})
        d["n"] += 1
        asks = {t["round"]: t["ask"] for t in turns if "ask" in t}
        if 1 in asks:
            d["open"].append(asks[1])
        if R * L >= F:
            below = [r for r, a in sorted(asks.items()) if a < R - 1e-9]
            d["below_R_round"].append(below[0] if below else None)
            for r in d["ask_at"]:
                if r in asks:
                    d["ask_at"][r].append(asks[r] - R)
        for t in turns:
            if t.get("round") in d["R_err"] and "R_med" in t:
                d["R_err"][t["round"]].append(t["R_med"] - R)
            if t.get("round") == 9 and "p_stub" in t:
                d["pstub_err"].append(t["p_stub"] - float(sc["stubborn"]))
    mean = lambda xs: round(statistics.mean(xs), 4) if xs else None
    return {b: {"n": d["n"], "opening_ask_share": mean(d["open"]),
                "first_round_ask_below_limit": mean([x for x in d["below_R_round"] if x is not None]),
                "never_below_limit_share": round(sum(x is None for x in d["below_R_round"]) / max(1, len(d["below_R_round"])), 3),
                "ask_minus_limit": {r: mean(v) for r, v in d["ask_at"].items()},
                "belief_limit_error": {r: mean(v) for r, v in d["R_err"].items()},
                "belief_limit_abs_error": {r: mean([abs(x) for x in v]) for r, v in d["R_err"].items()},
                "p_stubborn_error_round9": mean(d["pstub_err"])} for b, d in by.items()}


# ---------------------------------------------------------------- identification quality
def identification_metrics(records, rounds=(3, 6, 9), effective=False):
    """How well the belief identifies each buyer, per buyer type, at the given rounds.

    limit_mae / limit_bias: |median - true limit| and (median - true limit), share of list
    coverage90: share of buyers whose true limit lies in the belief's 5-95% range
    feasible_brier: (P(deal possible) - truth)^2; stubborn_brier: (P(stubborn) - truth)^2. Lower is better.
    effective=True scores the belief's effective limit (what the buyer pays by our last round) instead of R. Every
    bench buyer reaches its limit by round 10, so the true value is R either way; the belief's R is not identified
    for cells with a deadline past our last round (a late deadline and a high limit give the same bids).
    """
    by = {}
    for rec in records:
        sc, turns = rec["sc"], rec["turns"]
        basket = basket_of(sc)
        if not basket:
            continue
        L, C, F = totals(basket)
        R, feas, stub = sc["reservation"], float(sc["reservation"] * L >= F), float(sc["stubborn"])
        d = by.setdefault(sc["buyer"], {r: {"err": [], "cov": [], "fb": [], "sb": []} for r in rounds})
        pre = "Re_" if effective else "R_"
        for t in turns:
            r = t.get("round")
            if r in d and pre + "med" in t:
                d[r]["err"].append(t[pre + "med"] - R)
                d[r]["cov"].append(float(t[pre + "lo"] - 0.005 <= R <= t[pre + "hi"] + 0.005))
                d[r]["fb"].append((t["pe_feasible" if effective else "p_feasible"] - feas) ** 2)
                d[r]["sb"].append((t["p_stub"] - stub) ** 2)
    m = lambda xs: round(statistics.mean(xs), 4) if xs else None
    return {b: {r: {"n": len(v["err"]), "limit_mae": m([abs(x) for x in v["err"]]), "limit_bias": m(v["err"]),
                    "coverage90": m(v["cov"]), "feasible_brier": m(v["fb"]), "stubborn_brier": m(v["sb"])}
                for r, v in d.items()} for b, d in by.items()}


# ---------------------------------------------------------------- LLM transcripts
def llm_attribution(ev_dir, labels):
    import llm_buyers
    scen = {sc["id"]: sc for sc in buyers.make_scenarios(STORE, POLICY, 50, 0, llm_buyers.LLM_TYPES)}
    out = {}
    for label in labels:
        path = Path(ev_dir) / "transcripts" / f"{label}.jsonl"
        if not path.exists():
            continue
        by = {}
        for line in open(path, encoding="utf-8"):
            rec = json.loads(line)
            sc = scen[rec["id"]]
            basket = basket_of(sc)
            if not basket:
                continue
            L, C, F = totals(basket)
            R, to_pts = sc["reservation"], 100.0 * L / (L - C)
            d = by.setdefault(sc["buyer"], {"n": 0, "losses": {}, "premium": [], "first_offer_accepts": 0,
                                           "open": [], "deal_rounds": []})
            d["n"] += 1
            t = rec["transcript"]
            offers = [m for m in t if m["from"] == "seller" and m["type"] == "offer"]
            if offers:
                d["open"].append(sum(i["qty"] * i["unit_price"] for i in offers[0]["items"]) / L)
            if R * L < F:
                if rec["ended"] == "deadlock":
                    d["losses"]["deadlock_penalty"] = d["losses"].get("deadlock_penalty", 0) + 25
                continue
            oracle_pts = 100.0 * (R * L - C) / (L - C)
            if not rec["deal"]:
                k = {"seller_walked": "missed_walk", "deadlock": "missed_deadlock", "buyer_walked": "missed_buyer_left"}
                key = k.get(rec["ended"], "missed_other")
                d["losses"][key] = d["losses"].get(key, 0) + oracle_pts
                continue
            share = sum(i["qty"] * i["unit_price"] for i in rec["deal"]) / L
            last = t[-1]
            key = "accepted_bid_below_limit" if last["from"] == "seller" else "offer_accepted_below_limit"
            d["losses"][key] = d["losses"].get(key, 0) + max(0.0, R - share) * to_pts
            d["deal_rounds"].append(last["round"])
            if last["from"] == "buyer":
                bids = [m for m in t if m["from"] == "buyer" and m.get("items")]
                prev = [m for m in bids if m["round"] < last["round"]]
                if prev:
                    b = sum(i["qty"] * i["unit_price"] for i in prev[-1]["items"]) / L
                    d["premium"].append(share - b)
                if len(offers) == 1:
                    d["first_offer_accepts"] += 1
        out[label] = {b: {"n": d["n"], "losses_per_negotiation": {k: round(v / d["n"], 2) for k, v in
                                                                 sorted(d["losses"].items(), key=lambda kv: -kv[1])},
                          "accept_premium_over_last_bid": round(statistics.mean(d["premium"]), 4) if d["premium"] else None,
                          "first_offer_accepts": d["first_offer_accepts"],
                          "opening_ask_share": round(statistics.mean(d["open"]), 4) if d["open"] else None,
                          "deal_round_mean": round(statistics.mean(d["deal_rounds"]), 2) if d["deal_rounds"] else None}
                      for b, d in by.items()}
    return out


def llm_oracle(n, workers, out_dir):
    import llm_buyers
    scen = buyers.make_scenarios(STORE, POLICY, n, 0, llm_buyers.LLM_TYPES)
    lock, rows = threading.Lock(), []

    def one(sc):
        s = OracleLLMSeller(STORE, POLICY)
        s.scenario = sc
        b = llm_buyers.LLM_TYPES[sc["buyer"]](sc, STORE, random.Random(sc["seed"]))
        res = harness.run_negotiation(s, b, STORE, POLICY, sc["id"])
        row = evaluate.score_one(res, sc, STORE, POLICY)
        with lock:
            rows.append({"id": sc["id"], "buyer": sc["buyer"], "score": row["score"], "ended": res["ended"],
                         "rounds": res["rounds"], "feasible": row["feasible"], "transcript": res["transcript"]})

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(one, scen))
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(out_dir) / "llm_oracle_rows.jsonl", "w", encoding="utf-8") as fh:
        for r in sorted(rows, key=lambda r: r["id"]):
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {}
    for b in llm_buyers.LLM_TYPES:
        rs = [r for r in rows if r["buyer"] == b]
        fe = [r for r in rs if r["feasible"]]
        summary[b] = {"n": len(rs), "score": round(100 * statistics.mean(r["score"] for r in rs), 2),
                      "feasible": len(fe), "feasible_deals": sum(r["ended"] == "deal" for r in fe),
                      "deal_rounds": sorted(r["rounds"] for r in fe if r["ended"] == "deal"),
                      "ended": {k: sum(r["ended"] == k for r in rs) for k in {r["ended"] for r in rs}}}
    summary["all"] = {"score": round(100 * statistics.mean(r["score"] for r in rows), 2), "n": len(rows)}
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["bots", "llm", "llm-oracle", "identify"])
    ap.add_argument("--suites", default="public,shifted,proxy")
    ap.add_argument("--param", action="append", default=[], help="key=value passed to Seller (identify)")
    ap.add_argument("--modes", default="V0,V1,V5")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--n", type=int, default=210)
    ap.add_argument("--rows", nargs="*", default=[])
    ap.add_argument("--labels", default="v0,v1,v2,v3,v4,v5")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default=str(HERE / "runs" / "diagnose"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.what == "bots":
        seeds = [int(s) for s in a.seeds.split(",")]
        report = {"oracle": summarize_bots(play(lambda: OracleSeller(STORE, POLICY), buyers.BUYERS, seeds, a.n, oracle=True), "oracle")}
        for mode in a.modes.split(","):
            recs = play(lambda: Traced(mode), buyers.BUYERS, seeds, a.n)
            report[mode] = summarize_bots(recs, mode)
            report[mode]["steps"] = step_metrics(recs)
        (out / "bots.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(json.dumps(report, indent=1)[:200000])
    elif a.what == "identify":
        import suites
        params = {}
        for kv in a.param:
            k, v = kv.split("=", 1)
            params[k] = json.loads(v) if v[:1] in "0123456789-[{tf" and v not in ("true", "false") else \
                {"true": True, "false": False}.get(v, v)
        report = {}
        for mode in a.modes.split(","):
            report[mode] = {}
            for name in a.suites.split(","):
                s = suites.get(name)
                recs = play(lambda: Traced(mode, **params), s.buyer_classes(), s.seeds, s.n)
                report[mode][name] = {"identification": identification_metrics(recs),
                                      "identification_effective": identification_metrics(recs, effective=True),
                                      "score": round(100 * statistics.mean(r["row"]["score"] for r in recs), 2),
                                      "score_by_buyer": {b: round(100 * statistics.mean(
                                          r["row"]["score"] for r in recs if r["sc"]["buyer"] == b), 2)
                                          for b in sorted({r["sc"]["buyer"] for r in recs})}}
        (out / "identify.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
        for mode, per in report.items():
            for name, rep in per.items():
                print(f"{mode} {name} score {rep['score']} {rep['score_by_buyer']}")
                for b, rr in rep["identification_effective"].items():
                    print(f"   {b:16} " + " | ".join(
                        f"r{r}: mae {v['limit_mae']} bias {v['limit_bias']} cov {v['coverage90']} "
                        f"feasB {v['feasible_brier']} stubB {v['stubborn_brier']}" for r, v in rr.items()))
    elif a.what == "llm":
        report = {d: llm_attribution(d, a.labels.split(",")) for d in a.rows}
        (out / "llm.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(json.dumps(report, indent=1))
    else:
        s = llm_oracle(a.n, a.workers, out)
        (out / "llm_oracle.json").write_text(json.dumps(s, indent=1), encoding="utf-8")
        print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
