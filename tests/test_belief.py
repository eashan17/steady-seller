"""Behaviour tests for the buyer belief (plan v3, S4 exit conditions 1, 2, 6)."""
import random
import sys
import unittest
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))

import buyers  # noqa: E402
import harness  # noqa: E402
from negotiator.belief import BuyerBelief  # noqa: E402

STORE, POLICY = harness.load_store(), harness.load_policy()


def synthetic_buyers(classes=("hard_bargainer", "manipulator"), n=150, seed=7):
    """Kit buyers with known parameters, used only as test truth (priors are not fitted to them)."""
    out = []
    for sc in buyers.make_scenarios(STORE, POLICY, n, seed):
        if sc["buyer"] not in classes:
            continue
        b = buyers.BUYERS[sc["buyer"]](sc, STORE, random.Random(sc["seed"]))
        sellable = [i for i in sc["items"] if STORE[i["sku"]]["stock"] > 0]
        if not sellable:
            continue
        listed = sum(STORE[i["sku"]]["price"] * i["qty"] for i in sellable)
        floor = sum(harness.floor_price(STORE[i["sku"]], POLICY) * i["qty"] for i in sellable) / listed
        share = lambda j: b.total(b.priced(sellable, b.ratio(j))) / listed      # the bot's own ₹10 rounding
        out.append((sc, share, floor))
    return out


def belief_after(share, floor, upto):
    """Belief after the opening bid and the counters for rounds 1..upto-1, with our offers rejected above them."""
    bel = BuyerBelief(floor, POLICY["max_rounds"], opening=share(0))
    for j in range(1, upto):
        bel.observe_rejection(j - 1, min(1.0, share(j - 1) + 0.15))   # we asked well above, buyer countered
        bel.observe_bid(j, share(j))
    bel.update()
    return bel


class LimitEstimateTests(unittest.TestCase):
    def test_Given_known_buyers_When_rounds_pass_Then_limit_estimate_gets_closer(self):
        errs = {4: [], 8: []}
        for sc, share, floor in synthetic_buyers():
            for k in errs:
                errs[k].append(abs(belief_after(share, floor, k).limit_estimate()[0] - sc["reservation"]))
        self.assertLess(sum(errs[8]) / len(errs[8]), sum(errs[4]) / len(errs[4]))

    def test_Given_known_buyers_When_at_round_8_Then_90pct_range_covers_the_limit_at_least_80pct(self):
        hits = total = 0
        for sc, share, floor in synthetic_buyers():
            _, lo, hi = belief_after(share, floor, 8).limit_estimate(0.05, 0.95)
            hits += lo - 0.005 <= sc["reservation"] <= hi + 0.005
            total += 1
        self.assertGreaterEqual(hits / total, 0.80, f"{hits}/{total}")

    def test_Given_buyer_stays_past_round_When_not_stubborn_cells_have_passed_deadline_Then_leave_chance_drops(self):
        bel = BuyerBelief(0.8, 10, opening=0.6)
        before = bel.p_leave_by(6)
        for j in range(6):
            bel.observe_rejection(j, 0.95)
        bel.update()
        self.assertLess(bel.p_leave_by(6), before)


class TagEvidenceTests(unittest.TestCase):
    def test_Given_final_offer_claim_When_a_higher_bid_follows_Then_reliability_drops(self):
        bel = BuyerBelief(0.8, 10, opening=0.6)
        start = bel.reliability
        bel.observe_tags(3, {"final_offer": 0.9}, bid_share=0.70)
        bel.observe_bid(4, 0.75)                       # the "final" offer was a bluff
        self.assertLess(bel.reliability, start)

    def test_Given_bluffs_When_a_later_claim_arrives_Then_it_moves_the_belief_less(self):
        def shift(bel):                                # a "final offer" claim says the deadline is near
            bel.update()
            before = bel.p_leave_by(8)
            bel.observe_tags(6, {"final_offer": 0.9}, bid_share=0.70)
            bel.update()
            return abs(bel.p_leave_by(8) - before)
        honest = BuyerBelief(0.8, 10, opening=0.6)
        bluffer = BuyerBelief(0.8, 10, opening=0.6)
        for j, b in ((1, 0.62), (2, 0.64)):
            bluffer.observe_tags(j, {"final_offer": 0.9}, bid_share=b)
            bluffer.observe_bid(j + 1, b + 0.02)       # every claim is followed by a higher bid
        self.assertLess(shift(bluffer), shift(honest))

    def test_Given_manipulation_tag_When_observed_Then_reliability_drops(self):
        bel = BuyerBelief(0.8, 10, opening=0.6)
        start = bel.reliability
        bel.observe_tags(2, {"manipulation": 0.9})
        self.assertLess(bel.reliability, start)


class DeterminismTests(unittest.TestCase):
    def test_Given_identical_evidence_When_sampled_twice_Then_identical_draws(self):
        def run():
            bel = BuyerBelief(0.8, 10, opening=0.6)
            bel.observe_bid(1, 0.63)
            bel.update()
            return [(c.T, round(c.e, 6), c.sigma, c.stub, round(R, 9)) for c, R in bel.sample(50, seed=123)]
        self.assertEqual(run(), run())


if __name__ == "__main__":
    unittest.main()
