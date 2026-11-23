"""The full per-turn pipeline for modes V1-V3 (plan v3, D2 / S5).

    read text (tags) -> update the buyer belief -> expected-value decision -> template wording

`seller.py` wraps this with the rule guard and the safe reply, so whatever happens here, nothing
rule-breaking is sent. Within this engine:
  - a tagger error means "no tags this turn" (the belief simply gets no text evidence);
  - a belief or decision error makes this turn the fallback engine's decision (counted);
  - a wording error keeps the engine's plain fixed text.

Tags enter the belief as evidence with learned reliability (D4) and choose the template purpose;
they never set a price. S3's tag names map to the belief's evidence names in TAG_MAP.
"""
import re
import time

from negotiator import keywords, templates
from negotiator.decide import BeliefEngine
from negotiator.fallback import FallbackEngine

# S3 tag -> (belief evidence name, weight). walk_threat is weighted low: precision 0.41 held-out (S3).
TAG_MAP = {
    "claim_final": ("final_offer", 1.0),
    "walk_threat": ("walk_threat", 0.6),
    "claim_budget": ("budget_claim", 1.0),
    "claim_seller_commitment": ("false_agreement", 1.0),
    "attack_injection": ("injection", 1.0),
    "attack_authority": ("authority_claim", 1.0),
    "claim_competitor": ("competitor_quote", 1.0),
    "attack_threat": ("manipulation", 1.0),
    "attack_extraction": ("manipulation", 1.0),
}
# "₹3,000", "Rs. 3000", "INR 3,000": the currency marker as a whole word, then a number starting with a digit
# (a looser "rs" pattern matched inside words such as "offers," and captured an empty amount).
# Refined modes (V4/V5) also read the buyer's own time pressure as a sign it may leave soon.
REFINED_EXTRA_MAP = {"urgency": ("walk_threat", 0.5)}
# Hybrid reader (V5): S6 scored the encoder against keyword rules on 150 real LLM-buyer messages. Keywords were
# better on intent (0.56 vs 0.47) and claims (0.57 vs 0.31); the encoder on tone (0.98 vs 0.29) and raised
# no false attack flags (keywords 6). So each tag group is read by the reader that did better on it.
HYBRID_KEYWORD_GROUPS = ("intent", "claims")
HYBRID_ENCODER_GROUPS = ("attacks", "tone")
_RUPEES = re.compile(r"(?:₹|\brs\b\.?|\binr\b)\s*(\d[\d,]*)", re.I)

TAGGERS = ("none", "keywords", "classifier", "hybrid")


class AgentEngine(BeliefEngine):
    def __init__(self, store, policy, tagger="none", refined=False, honest_wording=False, **params):
        super().__init__(store, policy, **params)
        # honest_wording (2026-11-23): pick wording from what the price actually did, so a reply never says
        # "I've moved on price" at an unchanged price or "my offer stands" after a drop (templates.match_move)
        self.honest_wording = honest_wording
        if tagger not in TAGGERS:
            raise ValueError(f"tagger must be one of {TAGGERS}")
        self.tagger, self.refined = tagger, refined
        self.tag_map = dict(TAG_MAP, **REFINED_EXTRA_MAP) if refined else TAG_MAP
        self._ts = None
        if tagger in ("classifier", "hybrid"):
            from negotiator import textsignals
            # the module picks its backend once per process; re-read TEXTSIGNALS_BACKEND for this seller so
            # two sellers built with different settings in one process (the S6 bench) each get their own
            textsignals.reset()
            self._ts = textsignals

    def start(self, negotiation_id):
        super().start(negotiation_id)
        self.turn_fell_back = False
        self.current_tags = []
        self._prev_offer_total = None
        self.tag_counts, self.classifier_ms, self.tagger_errors, self.decision_errors = {}, [], 0, 0

    # ------------------------------------------------------------ text
    def read_tags(self, text):
        """Active tags for the buyer's text; [] on any error (no tags this turn)."""
        if self.tagger == "none" or not isinstance(text, str) or not text:
            return []
        try:
            if self.tagger == "keywords":
                return keywords.active(text)
            t0 = time.perf_counter()
            tags = list(self._ts.active(text))
            self.classifier_ms.append(1000 * (time.perf_counter() - t0))
            if self.tagger == "hybrid":
                groups = self._ts.GROUPS
                enc = {t for g in HYBRID_ENCODER_GROUPS for t in groups[g]}
                kw = {t for g in HYBRID_KEYWORD_GROUPS for t in groups[g]}
                tags = [t for t in tags if t in enc] + [t for t in keywords.active(text) if t in kw]
            return tags
        except Exception:
            self.tagger_errors += 1
            return []

    def _observe_text(self, msg, r, bid):
        tags = self.read_tags(msg.get("text"))
        self.current_tags = tags
        for t in tags:
            self.tag_counts[t] = self.tag_counts.get(t, 0) + 1
        evidence = {}
        for t in tags:
            if t in self.tag_map:
                name, weight = self.tag_map[t]
                evidence[name] = max(evidence.get(name, 0.0), weight)
        if not evidence:
            return
        value = None
        if "budget_claim" in evidence:
            amounts = [int(m[1].replace(",", "")) for m in _RUPEES.finditer(msg.get("text") or "")]
            L, _, _ = self._totals()
            if amounts and L > 0:
                value = max(amounts) / L
        self.belief.observe_tags(r - 1, evidence, bid_share=bid, value_share=value)

    # ------------------------------------------------------------ turn
    def _decide(self, msg, t, r):
        self.turn_fell_back = False
        first_offer = not self.offers
        try:
            reply = super()._decide(msg, t, r)
        except Exception:
            self.decision_errors += 1
            self.turn_fell_back = True
            return FallbackEngine._decide(self, msg, t, r)
        try:
            items = reply.get("items") or self.offer_items_for_text()
            # wording only: "close" comes from the round and the buyer's accept signals (a p_accept call
            # here cost ~4.6 ms per turn for no change in price or action)
            purpose = templates.choose_purpose(reply["type"], r, self.max_rounds, self.current_tags, first_offer)
            if self.honest_wording and reply["type"] == "offer" and items:
                prev = self._prev_offer_total
                now = sum(i["qty"] * i["unit_price"] for i in items)
                move = "first" if prev is None else ("down" if now < prev else "same")
                purpose = templates.match_move(purpose, move)
                self._prev_offer_total = now
            reply["text"] = templates.fill(purpose, items or [], self.store, (self.seed if self.belief else 0) + r)
        except Exception:
            pass                                       # keep the engine's plain fixed text
        return reply

    def offer_items_for_text(self):
        return self.allocate(self.ask) if self.ask is not None and self.basket else None
