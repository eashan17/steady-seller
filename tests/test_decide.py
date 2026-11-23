"""Behaviour tests for the expected-value decision engine, variant V1 (plan v3, S4 exit conditions 3, 6)."""
import random
import sys
import unittest
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))

import buyers  # noqa: E402
import harness  # noqa: E402
from seller import Seller  # noqa: E402

STORE, POLICY = harness.load_store(), harness.load_policy()
SKU = "S014"                                   # list 2299, floor share about 0.827
LIST = STORE[SKU]["price"]
FLOOR = harness.floor_price(STORE[SKU], POLICY)


def v1():
    s = Seller(STORE, POLICY, engine="belief")
    s.start("t")
    return s


def bid(share, r):
    return {"type": "counter", "round": r, "items": [{"sku": SKU, "qty": 1, "unit_price": int(round(share * LIST / 10)) * 10}]}


def play(shares):
    """Opening request at shares[0], then counters at shares[1:]; returns every seller reply."""
    s, out = v1(), []
    out.append(s.respond(dict(bid(shares[0], 1), type="request")))
    for r, sh in enumerate(shares[1:], start=2):
        out.append(s.respond(bid(sh, r)))
        if out[-1]["type"] != "offer":
            break
    return out


class WalkAwayTests(unittest.TestCase):
    def test_Given_hopeless_buyer_When_rounds_pass_Then_walks_only_at_the_final_turn(self):
        replies = play([0.40 + 0.01 * k for k in range(10)])         # crawls up, far below the floor
        self.assertEqual(["offer"] * 9, [m["type"] for m in replies[:9]])
        self.assertEqual("walk_away", replies[9]["type"])

    def test_Given_buyer_steadily_closing_on_the_floor_When_final_turn_Then_makes_the_offer(self):
        # steady +0.03 per round: the round-9 bid is 0.82 and the trajectory reaches 0.85 > floor (0.828)
        shares = [0.55 + 0.03 * k for k in range(10)]
        replies = play(shares)
        self.assertEqual("offer", replies[-1]["type"])
        self.assertEqual(10, len(replies))


class OfferShapeTests(unittest.TestCase):
    def test_Given_any_buyer_When_offering_Then_prices_within_floor_and_list_and_asks_never_rise(self):
        rng = random.Random(3)
        for _ in range(20):
            start = rng.uniform(0.3, 0.8)
            shares = [min(0.99, start + rng.uniform(0, 0.04) * k) for k in range(10)]
            asks = [m["items"][0]["unit_price"] for m in play(shares) if m["type"] == "offer"]
            with self.subTest(shares=[round(x, 3) for x in shares]):
                self.assertTrue(all(FLOOR <= p <= LIST for p in asks))
                self.assertEqual(asks, sorted(asks, reverse=True))

    def test_Given_bid_at_list_When_countering_early_Then_accepts(self):
        s = v1()
        s.respond(dict(bid(0.6, 1), type="request"))
        self.assertEqual("accept", s.respond({"type": "counter", "round": 2,
                                              "items": [{"sku": SKU, "qty": 1, "unit_price": LIST}]})["type"])


class DeterminismAndHarnessTests(unittest.TestCase):
    def test_Given_identical_negotiations_When_played_twice_Then_identical_replies(self):
        shares = [0.5, 0.55, 0.6, 0.66, 0.7, 0.74, 0.77, 0.8, 0.82, 0.83]
        self.assertEqual(play(shares), play(shares))

    def test_Given_kit_bots_When_played_through_the_harness_Then_no_violations_or_errors(self):
        seller = Seller(STORE, POLICY, engine="belief")
        for sc in buyers.make_scenarios(STORE, POLICY, 45, 11):
            b = buyers.BUYERS[sc["buyer"]](sc, STORE, random.Random(sc["seed"]))
            res = harness.run_negotiation(seller, b, STORE, POLICY, sc["id"])
            with self.subTest(id=sc["id"]):
                self.assertEqual([], res["violations"])
                self.assertEqual([], [m["error"] for m in res["transcript"] if m.get("error")])


if __name__ == "__main__":
    unittest.main()
