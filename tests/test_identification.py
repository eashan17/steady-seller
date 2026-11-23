"""Identification fixes to the buyer belief (2026-11-23, results/identification-report.md)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from negotiator.belief import BuyerBelief, prior_segments  # noqa: E402

F = 0.75


def belief(**kw):
    b = BuyerBelief(F, 10, 0.35, 0.05, **kw)
    for j, x in enumerate((0.36, 0.38, 0.42, 0.48, 0.55, 0.62), start=1):
        b.observe_bid(j, x)
    b.update()
    return b


class PriorTests(unittest.TestCase):
    def test_Given_floor_priors_When_built_Then_each_sums_to_one_with_25pct_below_the_floor(self):
        for name in ("floor", "floor_low", "floor_mid"):
            segs = prior_segments(0.35, F, name)
            total = sum(d * (hi - lo) for lo, hi, d in segs)
            below = sum(d * (min(hi, F - 0.35) - lo) for lo, hi, d in segs if lo < F - 0.35)
            self.assertAlmostEqual(total, 1.0, places=9, msg=name)
            self.assertAlmostEqual(below, 0.25, places=9, msg=name)

    def test_Given_floor_low_When_compared_with_floor_Then_more_mass_sits_just_above_the_floor(self):
        def near(name):
            hi = F + 0.3 * (1 - F) - 0.35
            return sum(d * max(0.0, min(h, hi) - max(lo, F - 0.35)) for lo, h, d in prior_segments(0.35, F, name))
        self.assertGreater(near("floor_low"), near("floor_mid"))
        self.assertGreater(near("floor_mid"), near("floor"))


class EvidenceTests(unittest.TestCase):
    def test_Given_bid_lower_When_limit_estimated_Then_no_mass_below_the_highest_bid(self):
        b = belief(limit_prior="floor", bid_lower=True)
        self.assertLess(1.0 - b.p_limit_at_least(0.62 - 0.006), 1e-6)
        self.assertGreater(1.0 - belief(limit_prior="floor").p_limit_at_least(0.62 - 0.006), 1e-3)

    def test_Given_remaining_spread_When_bids_progress_Then_cell_spread_shrinks_but_keeps_a_minimum(self):
        # per cell: model_sd * max(SD_MIN_SHARE, 1 - g(last bid round)). (The mixture's 5-95% range need not narrow:
        # narrower cells can separate a multi-modal mixture.)
        b = belief(limit_prior="floor", sd_mode="remaining")
        ms = [b._m(c) for c in b.cells]
        self.assertTrue(all(0.25 * 0.05 - 1e-12 <= m <= 0.05 + 1e-12 for m in ms))
        fast = next(c for c in b.cells if c.T == 5)            # reached its limit by round 5: minimum spread
        self.assertAlmostEqual(b._m(fast), 0.25 * 0.05)
        self.assertEqual({belief(limit_prior="floor")._m(c) for c in b.cells}, {0.05})

    def test_Given_effective_prior_When_no_bids_Then_effective_feasibility_matches_the_table(self):
        b = BuyerBelief(F, 10, 0.35, 0.0, limit_prior="floor_eff")
        self.assertGreater(b.p_feasible(effective=True), 0.6)
        r = BuyerBelief(F, 10, 0.35, 0.0, limit_prior="floor")
        self.assertGreater(b.p_feasible(effective=True), r.p_feasible(effective=True))

    def test_Given_manipulation_text_When_text_type_on_Then_stubbornness_rises(self):
        b = belief(limit_prior="floor", text_type=True)
        before = sum(c.w for c in b.cells if c.stub)
        b.observe_tags(6, {"manipulation": 0.9})
        b.update()
        after = sum(c.w for c in b.cells if c.stub)
        self.assertAlmostEqual(after / (1 - after), 2.0 * before / (1 - before), places=6)   # odds x STUB_LR
        off = belief(limit_prior="floor")
        off.observe_tags(6, {"manipulation": 0.9})
        off.update()
        self.assertAlmostEqual(sum(c.w for c in off.cells if c.stub), before, places=6)


if __name__ == "__main__":
    unittest.main()
