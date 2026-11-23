"""Rule guard: checks any proposed seller message before it is sent (plan v3, D3).

`check(message, state)` returns a list of violations; an empty list means the message may be sent.
It covers policy (mirroring `check_items`, `harness.py:60-88`), message shape, walk-away timing,
acceptance of the bid the harness will actually take as the deal (`harness.py:229-232,250-256`),
the kit's truth checks and the stricter house rules (`textcheck.py`).
"""
import math
from dataclasses import dataclass, field
from typing import Optional

from negotiator import textcheck

ALLOWED_TYPES = ("offer", "accept", "walk_away", "message")


def floor_price(row, policy):
    """The lowest unit price policy allows for this product (copy of `harness.floor_price`)."""
    margin_floor = row["cost"] / (1 - policy["min_margin"][row["category"]])
    discount_floor = row["price"] * (1 - policy["max_discount"])
    return math.ceil(max(margin_floor, discount_floor))


def check_items(items, store, policy):
    """Policy violations in a set of priced lines (copy of `harness.check_items`)."""
    out = []
    if not isinstance(items, list) or not items:
        return [{"kind": "policy:malformed", "detail": "offer has no items"}]
    for it in items:
        if not isinstance(it, dict):
            out.append({"kind": "policy:malformed", "detail": f"item {it!r}"})
            continue
        sku, qty, price = it.get("sku"), it.get("qty"), it.get("unit_price")
        try:
            row = store.get(sku)
        except TypeError:                       # unhashable sku; the kit would raise here
            row = None
        if row is None:
            out.append({"kind": "policy:unknown_sku", "detail": f"{sku!r}"})
            continue
        if not isinstance(qty, int) or isinstance(qty, bool) or qty < 1:
            out.append({"kind": "policy:malformed", "detail": f"{sku} qty {qty!r}"})
            continue
        if not isinstance(price, (int, float)) or isinstance(price, bool):
            out.append({"kind": "policy:malformed", "detail": f"{sku} unit_price {price!r}"})
            continue
        if qty > row["stock"]:
            out.append({"kind": "policy:out_of_stock", "detail": f"{sku} qty {qty}, stock {row['stock']}"})
        if price < row["cost"] / (1 - policy["min_margin"][row["category"]]):
            out.append({"kind": "policy:below_min_margin", "detail": f"{sku} at {price}"})
        if price < row["price"] * (1 - policy["max_discount"]):
            out.append({"kind": "policy:over_max_discount", "detail": f"{sku} at {price}, list {row['price']}"})
        if price > row["price"]:
            out.append({"kind": "policy:above_list_price", "detail": f"{sku} at {price}, list {row['price']}"})
    return out


@dataclass
class GuardState:
    """What the guard needs to know about the negotiation so far."""
    store: dict
    policy: dict
    context: textcheck.Context
    round: int = 1
    basket: Optional[dict] = None        # None: not parsed yet; {}: nothing sellable; else sku -> qty
    last_bid: Optional[list] = None      # the bid the harness would take as the deal on `accept`
    last_offer: Optional[list] = None    # our previous offer's items
    buyer_walked: bool = False
    extra: dict = field(default_factory=dict)

    @property
    def final_turn(self):
        return self.round >= self.policy["max_rounds"]


def _offer_shape(items, state):
    """House rules on offers: integer prices, requested sellable lines only, no duplicates."""
    out, seen = [], set()
    basket = state.basket or {}
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict):
            continue
        sku, qty, price = it.get("sku"), it.get("qty"), it.get("unit_price")
        if not isinstance(sku, str):
            continue
        if sku in seen:
            out.append({"kind": "house:duplicate_line", "detail": sku})
        seen.add(sku)
        if sku not in basket:
            out.append({"kind": "house:unrequested_line", "detail": sku})
        elif isinstance(qty, int) and qty > basket[sku]:
            out.append({"kind": "house:qty_above_request", "detail": f"{sku} qty {qty} > {basket[sku]}"})
        if isinstance(price, float):              # other non-int prices are already policy:malformed
            out.append({"kind": "house:non_integer_price", "detail": f"{sku} at {price!r}"})
    return out


def _allowed_figures(message, state):
    t = message.get("type")
    if t == "offer":
        return textcheck.offer_figures(message.get("items"), state.store)
    if t == "accept":
        return textcheck.offer_figures(state.last_bid, state.store)
    return textcheck.offer_figures(state.last_offer, state.store)


def check(message, state):
    """Every rule the proposed `message` breaks, as [{"kind", "detail"}]. Empty means it may be sent.

    Fails closed: if checking itself raises (e.g. an unhashable SKU reaching the kit's context
    code), the message is rejected rather than let through.
    """
    try:
        return _check(message, state)
    except Exception as e:
        return [{"kind": "guard:error", "detail": f"{type(e).__name__}: {e}"}]


def _check(message, state):
    if not isinstance(message, dict):
        return [{"kind": "shape:not_a_dict", "detail": type(message).__name__}]
    t = message.get("type")
    if t not in ALLOWED_TYPES:
        return [{"kind": "shape:bad_type", "detail": repr(t)}]
    out = []
    ctx = state.context.copy()
    if t == "offer":
        items = message.get("items")
        out += check_items(items, state.store, state.policy)
        out += _offer_shape(items, state)
        if isinstance(items, list):                # the harness adds the offer before checking text
            ctx.add_items([it for it in items if isinstance(it, dict) and isinstance(it.get("sku"), str)])
    elif t == "accept":
        if state.last_bid is None:
            out.append({"kind": "policy:malformed", "detail": "accept with no priced bid to accept"})
        else:
            out += check_items(state.last_bid, state.store, state.policy)
    elif t == "walk_away":
        allowed = state.final_turn or state.buyer_walked or state.basket == {}
        if not allowed:
            out.append({"kind": "timing:early_walk", "detail": f"walk at round {state.round}"})
    text = message.get("text")
    if text is not None and not isinstance(text, str):
        out.append({"kind": "shape:text_type", "detail": type(text).__name__})
    else:
        out += textcheck.check_text(text, ctx.as_dict())
        out += textcheck.house_violations(text, _allowed_figures(message, state))
    return out
