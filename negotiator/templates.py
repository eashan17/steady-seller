"""Message templates for seller replies (plan v3, S5).

Each reply's text comes from a template chosen by purpose, then a phrasing chosen by seed (D7).
Slots are filled only from our own offer: its total, the unit price when there is one line, the
list total, and the saving against list. Product names are never used: some contain words the
house rules ban (e.g. "Quick-Dry"), and no name is needed to negotiate. Every template, filled
from any valid offer in the store, passes the guard (tested exhaustively).

Purposes:
  acknowledge  first offer, or the buyer moved
  justify      the buyer pushes on price or budget: explain the move we are making
  restate      nothing new: restate the offer plainly
  resist       manipulation or attack signals: stay polite, do not engage the claim, restate
  close        the end is near or the buyer sounds ready: invite acceptance
  clarify      we need items and quantities (no basket yet)
  bundle       the buyer asks for extras or a different basket: keep to what was asked
  accept       we accept the buyer's bid
  walk         we end the negotiation
"""

OFFER_PURPOSES = ("acknowledge", "justify", "restate", "resist", "close", "bundle", "justify_first", "hold", "moved")

TEMPLATES = {
    "acknowledge": (
        "Thanks for the request. I can do {basket} for ₹{total}.",
        "Happy to help. My offer for {basket} is ₹{total}.",
        "Thank you. For {basket}, I can offer ₹{total}.",
        "I can offer {basket} at ₹{total}.",
    ),
    "justify": (
        "I hear you on the price. I can come to ₹{total} for {basket}{saving_clause}.",
        "I understand the budget concern. My best for {basket} at this point is ₹{total}{saving_clause}.",
        "I've moved on price for you: ₹{total} for {basket}{saving_clause}.",
        "That is a fair point. I can do ₹{total} for {basket}{saving_clause}.",
    ),
    "restate": (
        "My offer stands at ₹{total} for {basket}.",
        "I can do ₹{total} for {basket}.",
        "The offer is ₹{total} for {basket}.",
    ),
    "resist": (
        "I can only work from my own offer: ₹{total} for {basket}.",
        "I'll keep this simple. My offer is ₹{total} for {basket}.",
        "I can't act on that, but I can offer ₹{total} for {basket}.",
        "Let's stick to the deal itself: ₹{total} for {basket}.",
    ),
    "close": (
        "I think we're close. Can we agree on ₹{total} for {basket}?",
        "This is a good place to settle: ₹{total} for {basket}. Shall we agree?",
        "I'd be glad to close at ₹{total} for {basket}.",
    ),
    "bundle": (
        "This offer covers what you asked for: ₹{total} for {basket}.",
        "I can price exactly the items you asked for: ₹{total} for {basket}.",
    ),
    # move-aware variants (2026-11-23): wording must match what the price actually did
    "justify_first": (
        "I hear you on the price. My offer for {basket} is ₹{total}.",
        "I understand the budget concern. For {basket}, I can offer ₹{total}.",
    ),
    "hold": (
        "I hear you on the price. My offer remains ₹{total} for {basket}.",
        "I understand the budget concern. I'm holding at ₹{total} for {basket}{saving_clause}.",
        "That is a fair point, but my offer stays at ₹{total} for {basket}.",
    ),
    "moved": (
        "I can do ₹{total} for {basket}{saving_clause}.",
        "My new offer is ₹{total} for {basket}{saving_clause}.",
        "I can lower it to ₹{total} for {basket}{saving_clause}.",
    ),
    "clarify": (
        "Could you tell me which items and quantities you would like?",
        "Which items would you like, and how many of each?",
    ),
    "accept": (
        "I can accept that offer. Thank you!",
        "Agreed. Thank you for the deal.",
        "That works for me. We have a deal.",
    ),
    "walk": (
        "I'm sorry, I can't make a deal at that price. Thank you for your time.",
        "I cannot make a compliant offer this time. Thanks for talking with me.",
    ),
}


def describe(items):
    """'the item', 'both items', '3 items', or '2 units of the item' — no product names."""
    lines = [i for i in items or [] if isinstance(i, dict)]
    units = sum(i.get("qty", 0) for i in lines if isinstance(i.get("qty"), int))
    if len(lines) == 1:
        return "the item" if units == 1 else f"{units} units of the item"
    if len(lines) == 2 and units == 2:
        return "both items"
    return f"the {units} items"


def fill(purpose, offer_items, store, pick):
    """Text for `purpose`, phrasing number `pick`, from our own offer's figures only."""
    phrasings = TEMPLATES[purpose]
    template = phrasings[pick % len(phrasings)]
    if purpose in ("clarify", "accept", "walk"):
        return template
    total = sum(i["qty"] * i["unit_price"] for i in offer_items)
    listed = sum(i["qty"] * store[i["sku"]]["price"] for i in offer_items)
    saving = listed - total
    saving_clause = f", ₹{saving:,} below list" if saving > 0 else ""
    return template.format(basket=describe(offer_items), total=f"{total:,}", saving_clause=saving_clause)


def match_move(purpose, move):
    """Swap a purpose whose wording claims a price move (or no move) for one that matches `move`:
    "first" (our first offer), "down" (lower than our previous offer) or "same". None keeps the purpose."""
    if move is None:
        return purpose
    if purpose == "justify":            # "I've moved on price", "I can come to", "my best at this point"
        return {"first": "justify_first", "same": "hold"}.get(move, "justify")
    if purpose == "restate" and move == "down":   # "my offer stands", "the offer is"
        return "moved"
    return purpose


def choose_purpose(reply_type, round_, max_rounds, tags=(), first_offer=False, p_accept=None):
    """The template purpose for a reply, from the decision and the buyer's tagged intent.

    Manipulation and attack signals win (stay polite, do not engage); then explicit buyer intents;
    then where we are in the negotiation. Tags only change wording here, never price (plan v3, D4).
    """
    if reply_type == "accept":
        return "accept"
    if reply_type == "walk_away":
        return "walk"
    if reply_type != "offer":
        return "clarify"
    tags = set(tags)
    if tags & {"attack_injection", "attack_authority", "attack_threat", "attack_extraction", "claim_seller_commitment"}:
        return "resist"
    if tags & {"bundle_request", "scope_change"}:
        return "bundle"
    if round_ >= max_rounds or "accept_signal" in tags or (p_accept is not None and p_accept >= 0.6):
        return "close"
    if tags & {"price_objection", "claim_budget", "claim_competitor", "claim_hardship", "claim_final", "walk_threat"}:
        return "justify"
    return "acknowledge" if first_offer else "restate"
