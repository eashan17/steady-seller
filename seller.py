"""PS2 seller (plan v3). The harness loads `Seller` from this file.

Per turn: mirror what the harness knows (context for truth checks, the bid it would take on
`accept`), let the engine propose a reply, check it with the rule guard, and send it only if it
passes. A reply whose only problem is its wording is retried once with plain fixed text; anything
else the guard rejects, or an engine exception, is replaced by a safe reply (plan v3, D2/D3).

Modes (environment variable SELLER_MODE, or the `mode` argument):
  V0  fallback engine only (time-based concession, projected final-turn walk)
  V1  buyer belief + expected-value decisions + templates, reading no text
  V2  V1 + keyword-rule tags (the comparison point)
  V3  V1 + trained-classifier tags
  V4-V10 refinements, see MODE_PARAMS below
  V9  V5 + every buyer-identification fix
  V11 V9 + no_rise: an offer never goes above our previous offer              <- default
Unset means DEFAULT_MODE. No mode makes network calls.
DEFAULT_MODE = V11 = V9 + the no_rise fix (user decision 2026-11-23; it costs 0.2-0.3 on the noisy suites, which
profited from accidental price rises). V9 (results/closing-report.md): best mean over public/proxy/shifted and better than V5 on
4 of 5 rule suites; it loses 1.84 on jitter_0.08, where +/-8% bid noise hides feasibility from every version
tried (an information limit). Endgame, information-seeking and deadline-prior options stay off.

    python evaluate.py --seller seller.py --no-baseline
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from negotiator import guard, textcheck  # noqa: E402
from negotiator.fallback import (TEXT_ACCEPT, TEXT_CLARIFY, TEXT_OFFER, TEXT_WALK,  # noqa: E402
                                 FallbackEngine)

DEFAULT_MODE = "V11"  # 2026-11-23: V9 + no_rise correctness fix, chosen by the user for consistency (results/norise-rule)
MODES = {"V0": None, "V1": "none", "V2": "keywords", "V3": "classifier", "V4": "none", "V5": "hybrid",
         "V6": "hybrid", "V7": "hybrid", "V8": "hybrid", "V9": "hybrid", "V10": "hybrid", "V11": "hybrid"}
# Refine (2026-11-21, results/refine-report.md): V4 = V1 + fallback price floor (concede no faster than the
# fallback unless the buyer looks likely to leave) + urgency read as leave risk; V5 = V4 + hybrid tags
# (keywords for intent and claims, encoder for tone and attacks).
# floor_rounds=1 (open at list, then the belief decides): the least-damaging floor in the refine sweep; floors on
# every round or rounds 1-3 lose 0.5-6.8 points on the noisy suites (results/refine-report.md).
MODE_PARAMS = {"V4": {"refined": True, "fallback_floor": True, "floor_rounds": 1},
               "V5": {"refined": True, "fallback_floor": True, "floor_rounds": 1},
               # V6 = V5 + fix 1 (results/bottleneck-report.md): bid-gap model, accept guard, bid-anchored counters
               "V6": {"refined": True, "fallback_floor": True, "floor_rounds": 1,
                      "bid_gap": "cautious", "accept_guard": True, "bid_anchor": True},
               # V7 = V5 + identification fixes (results/identification-report.md): limit prior anchored to the
               # floor, and manipulation text as evidence that the buyer never walks away on its own
               "V7": {"refined": True, "fallback_floor": True, "floor_rounds": 1,
                      "limit_prior": "floor", "text_type": True},
               # V8 = V7 with the floor prior on the effective limit (what the buyer pays by our last round), so a late
               # deadline is not mistaken for a high limit (results/identification-report.md)
               "V8": {"refined": True, "fallback_floor": True, "floor_rounds": 1,
                      "limit_prior": "floor_eff", "text_type": True},
               # V9 = V5 + all identification fixes (results/identification-report.md): floor-anchored "floor_mid" prior on
               # the effective limit, manipulation text as stubbornness evidence, limit >= highest bid, model-error spread
               # shrinking as bids near the limit. Beats V5 on 4 of 5 rule suites; fails D8 on jitter_0.08 (-1.84)
               "V9": {"refined": True, "fallback_floor": True, "floor_rounds": 1, "limit_prior": "floor_mid_eff",
                      "text_type": True, "bid_lower": True, "sd_mode": "remaining"},
               # V10 = V9 + information-seeking planning: closed-loop final turn and floor-probe candidate plans
               # V11 = V9 + no_rise bug fix (offers never rise above our previous ask; decide.py) + honest_wording (reply
               # wording matches the actual price move; wording only, prices unchanged)
               "V11": {"refined": True, "fallback_floor": True, "floor_rounds": 1, "limit_prior": "floor_mid_eff",
                       "text_type": True, "bid_lower": True, "sd_mode": "remaining", "no_rise": True,
                       "honest_wording": True},
               "V10": {"refined": True, "fallback_floor": True, "floor_rounds": 1, "limit_prior": "floor_mid_eff",
                       "text_type": True, "bid_lower": True, "sd_mode": "remaining", "closed_final": True, "probe": True}}
PLAIN_TEXT = {"offer": TEXT_OFFER, "accept": TEXT_ACCEPT, "walk_away": TEXT_WALK, "message": TEXT_CLARIFY}


def _text_only(violations):
    """True if every violation concerns the wording, not the action or prices."""
    return all(not v["kind"].startswith(("policy:", "shape:", "timing:", "guard:")) and
               v["kind"] not in ("house:unrequested_line", "house:qty_above_request", "house:duplicate_line",
                                 "house:non_integer_price")
               for v in violations)


class Seller:
    def __init__(self, store, policy, mode=None, engine=None, **engine_params):
        self.store, self.policy = store, policy
        if engine is not None:                     # S4-era argument, kept for older bench specs
            mode = mode or {"belief": "V1", "fallback": "V0"}.get(engine, engine)
        self.mode = (mode or os.environ.get("SELLER_MODE") or DEFAULT_MODE).upper()
        if self.mode not in MODES:
            raise ValueError(f"SELLER_MODE must be one of {sorted(MODES)}, got {self.mode!r}")
        if MODES[self.mode] is None:
            self.engine = FallbackEngine(store, policy, **engine_params)
        else:
            from negotiator.agent import AgentEngine
            params = dict(MODE_PARAMS.get(self.mode, {}), **engine_params)
            self.engine = AgentEngine(store, policy, tagger=MODES[self.mode], **params)
        self.start("init")

    def start(self, negotiation_id):
        # per negotiation, as the bench reads it (results/S2-complete.md, "seller.stats contract")
        self.stats = {"turns": 0, "fallback_turns": 0, "engine_errors": 0, "guard_rejections": 0,
                      "text_retries": 0, "safe_replies": 0, "rejection_kinds": {}}
        self.engine.start(negotiation_id)
        self.state = guard.GuardState(self.store, self.policy, textcheck.Context(self.store, self.policy))

    # ------------------------------------------------------------ turn
    def respond(self, message):
        self.stats["turns"] += 1
        try:
            self._observe_buyer(message)
        except Exception:                         # bookkeeping must never cost the turn
            pass
        try:
            proposal = self.engine.respond(message)
        except Exception:
            self.stats["engine_errors"] += 1
            proposal = None
        reply = self._vet(proposal) if proposal is not None else None
        if reply is None:
            reply = self._safe_reply()
        if reply is not proposal or MODES[self.mode] is None or getattr(self.engine, "turn_fell_back", False):
            self.stats["fallback_turns"] += 1
        self._observe_seller(reply)
        self._collect_engine_stats()
        return reply

    def _vet(self, proposal):
        """The proposal if it passes the guard; the same action with plain text if only wording failed."""
        violations = self._violations(proposal)
        if not violations:
            return proposal
        if isinstance(proposal, dict) and proposal.get("type") in PLAIN_TEXT and _text_only(violations):
            self.stats["text_retries"] += 1
            plain = dict(proposal, text=PLAIN_TEXT[proposal["type"]])
            if not self._violations(plain):
                return plain
        return None

    def _violations(self, proposal):
        violations = guard.check(proposal, self._guard_state())
        if violations:
            self.stats["guard_rejections"] += 1
            for v in violations:
                self.stats["rejection_kinds"][v["kind"]] = self.stats["rejection_kinds"].get(v["kind"], 0) + 1
        return violations

    def _safe_reply(self):
        """Our current offer with plain text if it passes the guard, else a clarifying message."""
        self.stats["safe_replies"] += 1
        try:
            items = self.engine.offer_items(self.state.round)
            if items:
                candidate = {"type": "offer", "items": items, "text": TEXT_OFFER}
                if not guard.check(candidate, self._guard_state()):
                    return candidate
        except Exception:
            pass
        return {"type": "message", "text": TEXT_CLARIFY}

    def _collect_engine_stats(self):
        e = self.engine
        if not hasattr(e, "tag_counts"):
            return
        self.stats["tagger_errors"] = e.tagger_errors
        self.stats["decision_errors"] = e.decision_errors
        self.stats["tags"] = dict(e.tag_counts)
        if e.tagger in ("classifier", "hybrid") and e._ts is not None:
            self.stats["classifier_backend"] = e._ts.backend()
            self.stats["classifier_ms"] = list(e.classifier_ms)
        elif e.tagger == "keywords":
            self.stats["classifier_backend"] = "keywords"

    # ------------------------------------------------------------ mirror the harness's bookkeeping
    def _guard_state(self):
        self.state.basket = self.engine.basket
        return self.state

    def _observe_buyer(self, msg):
        if not isinstance(msg, dict):
            return
        st = self.state
        r = msg.get("round")
        if isinstance(r, int) and not isinstance(r, bool):
            st.round = r
        items, t = msg.get("items"), msg.get("type")
        st.context.add_items(items)                                   # harness.py:217
        st.context.add_text(msg.get("text"))                          # harness.py:218
        if t == "request":                                            # harness.py:219-222
            for it in items if isinstance(items, list) else []:
                if isinstance(it, dict) and isinstance(it.get("qty"), int):
                    try:
                        st.context.wanted[it.get("sku")] = it["qty"]
                    except TypeError:
                        pass
        if t == "walk_away":
            st.buyer_walked = True
        if t in ("request", "counter") and isinstance(items, list) and items and \
                all(isinstance(i, dict) and "unit_price" in i for i in items):  # harness.py:229-232
            st.last_bid = items

    def _observe_seller(self, reply):
        if reply.get("type") == "offer":
            self.state.context.add_items(reply.get("items"))          # harness.py:244-245
            self.state.last_offer = reply.get("items")
