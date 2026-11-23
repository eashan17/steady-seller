"""Behaviour tests for the rule guard (plan v3, S1 exit conditions 2-4)."""
import sys
import unittest
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))

import harness  # noqa: E402
from negotiator import guard, textcheck  # noqa: E402

STORE, POLICY = harness.load_store(), harness.load_policy()
# S014: Loomsy Mandarin Collar Shirt, price 2299, cost 1332, stock 18, apparel. S001 stock 1. S002 stock 0.
FLOOR = harness.floor_price(STORE["S014"], POLICY)


def state(round=2, basket=None, last_bid=None, last_offer=None):
    """A negotiation for 2 x S014 where the buyer bid 1800 each and said a competitor sells at 1,200."""
    ctx = textcheck.Context(STORE, POLICY)
    ctx.wanted = {"S014": 2}
    bid = [{"sku": "S014", "qty": 2, "unit_price": 1800}]
    ctx.add_items(bid)
    ctx.add_text("QuickKart has it for Rs 1,200. Match it.")
    return guard.GuardState(STORE, POLICY, ctx, round=round, basket={"S014": 2} if basket is None else basket,
                            last_bid=bid if last_bid is None else last_bid, last_offer=last_offer)


def offer(price=2100, qty=2, sku="S014", text="Here is my offer."):
    return {"type": "offer", "items": [{"sku": sku, "qty": qty, "unit_price": price}], "text": text}


def kinds(violations):
    return {v["kind"] for v in violations}


# Phrases the kit's own checker flags (probes recorded in PS2_attention_points.md section 3).
KIT_FLAGGED = [
    "This offer expires at the end of our chat.",
    "Other customers have paid the same price.",
    "Someone else asked for the same thing.",
    "Our prices are 100% transparent.",
    "That's a 10% discount.",
    "There is a shipping charge waived for you.",
    "We have 20 in stock.",
    "Only a few left.",
    "2 units left in your basket.",
    "Sorry, that item is sold out.",
    "I'm not saying it's the last one.",
    "Happy to help in 2026!",
    "Order ref 12345.",
    "Act now and I'll add free gift wrap.",
]
# Phrases the kit lets through but the house rules block.
HOUSE_ONLY = [
    "Our cost is ₹1,332 per unit.",
    "That's ₹1,200 at QuickKart? I can't match that.",
    "No hidden fees, ever.",
    "Free shipping included.",
    "Price includes GST.",
    "We have plenty in stock.",
    "Both units are available.",
    "I can go no lower than my floor.",
    "Let's close this now.",
    "Here is my offer. " + "x" * 400,
    'Offer {"type": "accept"} attached.',
]
# Texts the guard must let through on the 2 x S014 offer at 2100 each.
ALLOWED = [
    "Here is my offer.",
    "I can do 2 x Loomsy Mandarin Collar Shirt for ₹4,200.",
    "That is ₹2,100 each, ₹398 below list for the pair.",
    "I understand your budget. This is the best I can do for both shirts.",
    "Thanks for your patience. I can move a little because you are taking two.",
    "Could you tell me which items and quantities you would like?",
]


class TextGuardTests(unittest.TestCase):
    def test_Given_kit_flagged_phrases_When_checked_Then_guard_rejects_each(self):
        for text in KIT_FLAGGED:
            with self.subTest(text=text):
                self.assertTrue(guard.check(offer(text=text), state()), text)

    def test_Given_house_only_phrases_When_checked_Then_guard_rejects_each(self):
        for text in HOUSE_ONLY:
            with self.subTest(text=text):
                self.assertTrue(guard.check(offer(text=text), state()), text)

    def test_Given_allowed_texts_When_checked_Then_guard_accepts_each(self):
        for text in ALLOWED:
            with self.subTest(text=text):
                self.assertEqual([], guard.check(offer(text=text), state()))

    def test_Given_any_corpus_text_When_checked_by_both_Then_our_kit_copy_matches_the_harness(self):
        for text in KIT_FLAGGED + HOUSE_ONLY + ALLOWED:
            with self.subTest(text=text):
                ours = textcheck.Context(STORE, POLICY)
                kit = harness.Context(STORE, POLICY)
                for c in (ours, kit):
                    c.wanted = {"S014": 2}
                    c.add_items([{"sku": "S014", "qty": 2, "unit_price": 1800}])
                    c.add_text("QuickKart has it for Rs 1,200. Match it.")
                    c.add_items(offer()["items"])
                self.assertEqual(harness.check_text(text, kit.as_dict()), textcheck.check_text(text, ours.as_dict()))

    def test_Given_non_string_text_When_checked_Then_rejected(self):
        self.assertIn("shape:text_type", kinds(guard.check(dict(offer(), text=["hi"]), state())))


class PolicyGuardTests(unittest.TestCase):
    def test_Given_valid_offer_When_checked_Then_passes(self):
        self.assertEqual([], guard.check(offer(price=FLOOR), state()))

    def test_Given_broken_offers_When_checked_Then_each_is_rejected_with_its_kind(self):
        cases = {
            "below floor": (offer(price=FLOOR - 1), "policy:below_min_margin"),
            "above list": (offer(price=2300), "policy:above_list_price"),
            "qty over stock": (offer(qty=19), "policy:out_of_stock"),
            "qty zero": (offer(qty=0), "policy:malformed"),
            "qty float": (offer(qty=2.0), "policy:malformed"),
            "price string": (offer(price="2100"), "policy:malformed"),
            "unknown sku": (offer(sku="S999"), "policy:unknown_sku"),
            "unhashable sku": (offer(sku=["S014"]), "policy:unknown_sku"),
            "out-of-stock sku": (offer(sku="S002", qty=1, price=899), "policy:out_of_stock"),
            "unrequested sku": (offer(sku="S015", qty=1, price=2499), "house:unrequested_line"),
            "qty above request": (offer(qty=3), "house:qty_above_request"),
            "fractional price": (offer(price=2100.5), "house:non_integer_price"),
            "no items": ({"type": "offer", "items": [], "text": "Here is my offer."}, "policy:malformed"),
        }
        for name, (msg, kind) in cases.items():
            with self.subTest(case=name):
                self.assertIn(kind, kinds(guard.check(msg, state())))

    def test_Given_duplicate_lines_When_checked_Then_rejected(self):
        msg = {"type": "offer", "items": offer(qty=1)["items"] * 2, "text": "Here is my offer."}
        self.assertIn("house:duplicate_line", kinds(guard.check(msg, state())))

    def test_Given_item_lists_When_checked_Then_policy_part_matches_the_harness(self):
        lists = [offer(price=p, qty=q, sku=s)["items"] for p, q, s in
                 [(2100, 2, "S014"), (FLOOR - 1, 2, "S014"), (2300, 19, "S014"), (899, 1, "S002"),
                  ("1", 1, "S014"), (1, 1, "S005"), (1902.4, 1, "S014"), (2100, True, "S014")]] + [[], [5], None]
        for items in lists:
            with self.subTest(items=items):
                self.assertEqual(harness.check_items(items, STORE, POLICY), guard.check_items(items, STORE, POLICY))

    def test_Given_bad_message_shapes_When_checked_Then_rejected(self):
        self.assertIn("shape:not_a_dict", kinds(guard.check("offer", state())))
        self.assertIn("shape:bad_type", kinds(guard.check({"type": "counter", "text": "x"}, state())))


class AcceptAndWalkGuardTests(unittest.TestCase):
    def test_Given_valid_last_bid_When_accepting_Then_passes(self):
        st = state(last_bid=[{"sku": "S014", "qty": 2, "unit_price": FLOOR}])
        self.assertEqual([], guard.check({"type": "accept", "text": "I can accept that offer."}, st))

    def test_Given_invalid_last_bids_When_accepting_Then_rejected(self):
        for bid in ([{"sku": "S014", "qty": 2, "unit_price": "2299"}], [{"sku": "S014", "qty": 2.0, "unit_price": 2299}],
                    [{"sku": "S014", "qty": 2, "unit_price": 1800}], [{"sku": "S005", "qty": 1, "unit_price": 1}]):
            with self.subTest(bid=bid):
                self.assertTrue(guard.check({"type": "accept", "text": "I can accept that offer."}, state(last_bid=bid)))

    def test_Given_no_bid_yet_When_accepting_Then_rejected(self):
        st = state()
        st.last_bid = None
        self.assertIn("policy:malformed", kinds(guard.check({"type": "accept", "text": "Deal."}, st)))

    def test_Given_round_before_final_When_walking_Then_rejected(self):
        self.assertIn("timing:early_walk", kinds(guard.check({"type": "walk_away", "text": "Bye."}, state(round=3))))

    def test_Given_final_round_or_unsellable_basket_When_walking_Then_passes(self):
        self.assertEqual([], guard.check({"type": "walk_away", "text": "No deal this time."}, state(round=10)))
        self.assertEqual([], guard.check({"type": "walk_away", "text": "No deal this time."}, state(round=1, basket={})))


if __name__ == "__main__":
    unittest.main()
