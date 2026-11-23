"""no_rise (V11): our offer for the same basket never goes above our previous offer."""
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import buyers  # noqa: E402
import harness  # noqa: E402
from seller import Seller  # noqa: E402

STORE, POLICY = harness.load_store(), harness.load_policy()


def rises(mode, n=210, seed=0):
    seller, count = Seller(STORE, POLICY, mode=mode), 0
    for sc in buyers.make_scenarios(STORE, POLICY, n, seed):
        res = harness.run_negotiation(seller, buyers.BUYERS[sc["buyer"]](sc, STORE, random.Random(sc["seed"])),
                                      STORE, POLICY, sc["id"])
        prev = None
        for m in res["transcript"]:
            if m["from"] == "seller" and m["type"] == "offer":
                key = tuple(sorted((i["sku"], i["qty"]) for i in m["items"]))
                tot = sum(i["qty"] * i["unit_price"] for i in m["items"])
                if prev and prev[0] == key and tot > prev[1]:
                    count += 1
                prev = (key, tot)
    return count


class NoRiseTests(unittest.TestCase):
    def test_v11_offers_never_rise(self):
        self.assertEqual(rises("V11"), 0)

    def test_v9_bug_is_present_without_the_flag(self):
        # documents the bug the flag fixes (rounding up + fallback substitution)
        self.assertGreater(rises("V9", n=60), 0)


if __name__ == "__main__":
    unittest.main()
