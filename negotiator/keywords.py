"""Keyword-rule tagger: the V2 comparison point for the trained classifier (plan v3, S5 / S6).

Same tag names as `negotiator/textsignals.py`, so V2 and V3 differ only in how text is read.
These rules are deliberately plain, the kind a team would write by hand; S6 measures whether
the classifier (V3) beats them on LLM buyers, which was the reason for training one.
"""
import re

RULES = {
    "accept_signal": r"\b(?:deal|agreed?|works for me|i'?ll take it|sounds good|i accept|let'?s do it)\b",
    "price_objection": r"\b(?:too (?:high|expensive|much)|costly|pricey|overpriced|lower|cheaper|come down|discount)\b",
    "walk_threat": r"\b(?:walk|leave|leaving|i'?m out|pass|elsewhere|no deal|go somewhere)\b",
    "bundle_request": r"\b(?:bundle|bulk|throw in|extra|for free|together|combo)\b",
    "scope_change": r"\b(?:instead|actually make it|change (?:it|the)|add (?:one|a|an)|remove|only one)\b",
    "info_request": r"\?|\b(?:what|how|why|which|can you tell)\b",
    "urgency": r"\b(?:now|quick(?:ly)?|today|hurry|asap|right away)\b",
    "claim_budget": r"\b(?:budget|afford|can only pay|maximum|max i can)\b",
    "claim_competitor": r"\b(?:amazon|flipkart|myntra|quickkart|another (?:shop|store|site|seller)|other (?:shop|store|site))\b",
    "claim_final": r"\b(?:final|last offer|take it or leave it|best i can|won'?t go (?:higher|up)|not a rupee more)\b",
    "claim_seller_commitment": r"\b(?:you (?:already )?(?:agreed|said|promised)|honou?r (?:it|that|the)|"
                               r"(?:your )?(?:website|site|listing|ad) says)\b",
    "claim_hardship": r"\b(?:student|tight|broke|struggling|can'?t afford|hard times)\b",
    "attack_injection": r"(?:\bignore (?:all |your |the )?(?:previous |earlier )?(?:instructions|rules)|\bsystem:|developer mode|\{\s*\"type\")",
    "attack_authority": r"\b(?:manager|authori[sz]e[ds]?|policy (?:updated|says|allows)|admin|approval granted)\b",
    "attack_threat": r"\b(?:report|complain|bad review|sue|consumer court|gouging)\b",
    "attack_extraction": r"\b(?:cost price|your cost|margin|floor price|print your|reveal)\b",
    "tone_friendly": r"\b(?:thanks|thank you|appreciate|please|great|lovely|perfect)\b",
    "tone_hostile": r"\b(?:ridiculous|scam|waste of time|rip ?off|stupid|joke|not even close)\b",
}
_COMPILED = {tag: re.compile(p, re.I) for tag, p in RULES.items()}


def active(text):
    """Tags whose rule matches the text."""
    if not isinstance(text, str) or not text:
        return []
    return [tag for tag, rx in _COMPILED.items() if rx.search(text[:2000])]
