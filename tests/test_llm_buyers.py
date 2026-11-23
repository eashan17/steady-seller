"""Offline tests for the LLM test buyers (S6): Codex is mocked, so these spend no quota."""
import random
import sys
import unittest
from pathlib import Path
from unittest import mock

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))
sys.path.insert(0, str(KIT / "bench"))

import buyers  # noqa: E402
import harness  # noqa: E402
import llm_buyers  # noqa: E402
from seller import Seller  # noqa: E402

STORE, POLICY = harness.load_store(), harness.load_policy()


def scenarios(cls, n=40, seed=3):
    return buyers.make_scenarios(STORE, POLICY, n, seed, {"x": cls})


def play(cls, llm_reply, n=40):
    """Negotiations against V1 with every LLM call answered by `llm_reply(prompt)`."""
    out = []
    with mock.patch.object(llm_buyers, "codex_json", side_effect=lambda prompt, tag: llm_reply(prompt)):
        seller = Seller(STORE, POLICY, mode="V1")
        for sc in scenarios(cls, n):
            b = cls(sc, STORE, random.Random(sc["seed"]))
            out.append((sc, b, harness.run_negotiation(seller, b, STORE, POLICY, sc["id"])))
    return out


def buyer_bids(result):
    """(bid total, list total of those same lines) for every priced buyer message."""
    return [(sum(i["qty"] * i["unit_price"] for i in m["items"]),
             sum(i["qty"] * STORE[i["sku"]]["price"] for i in m["items"])) for m in result["transcript"]
            if m["from"] == "buyer" and m.get("items") and all("unit_price" in i for i in m["items"])]


class EnforcementTests(unittest.TestCase):
    def test_Given_an_LLM_that_always_bids_and_accepts_huge_When_playing_Then_never_pays_above_the_limit(self):
        greedy = lambda p: {"action": "accept", "total": 10 ** 7, "message": "Deal at any price!"}
        for sc, b, res in play(LLMManip := llm_buyers.LLMManipulator, greedy):
            with self.subTest(id=sc["id"]):
                if res["deal"]:
                    paid = sum(i["qty"] * i["unit_price"] for i in res["deal"])
                    self.assertLessEqual(paid, sc["reservation"] * sum(STORE[i["sku"]]["price"] * i["qty"]
                                                                       for i in res["deal"]) + 1)
                # each bid against the limit for its own lines (the basket can shrink to what is sellable)
                self.assertTrue(all(t <= sc["reservation"] * listed + 1 for t, listed in buyer_bids(res)))

    def test_Given_garbage_LLM_output_When_playing_Then_safe_counters_and_no_crash(self):
        for reply in (lambda p: None, lambda p: {"action": "dance"}, lambda p: {"action": "counter", "total": "x"}):
            for sc, b, res in play(llm_buyers.LLMHardBargainer, reply, n=10):
                with self.subTest(id=sc["id"]):
                    self.assertIn(res["ended"], ("deal", "deadlock", "buyer_walked", "seller_walked"))

    def test_Given_stubborn_buyer_that_wants_to_walk_When_playing_Then_it_never_walks(self):
        walker = lambda p: {"action": "walk_away", "total": 0, "message": "Bye."}
        for sc, b, res in play(llm_buyers.LLMImpossibleBudget, walker, n=10):
            with self.subTest(id=sc["id"]):
                self.assertNotEqual("buyer_walked", res["ended"])

    def test_Given_impatient_buyer_When_its_deadline_passes_Then_it_walks_or_has_dealt(self):
        lowball = lambda p: {"action": "counter", "total": 10, "message": "Ten rupees."}
        for sc, b, res in play(llm_buyers.LLMEarlyLeaver, lowball, n=15):
            with self.subTest(id=sc["id"]):
                # it answers our round-D offer by walking, which the harness logs as round D + 1 (as for kit bots)
                self.assertLessEqual(res["rounds"], llm_buyers.LLMEarlyLeaver.deadline + 1)

    def test_Given_suite_registry_When_imported_Then_llm_mix_is_an_llm_suite_not_in_defaults(self):
        import suites
        s = suites.get("llm_mix")
        self.assertEqual("llm", s.kind)
        self.assertNotIn("llm_mix", suites.default_names())
        self.assertEqual(5, len(s.buyer_classes()))


if __name__ == "__main__":
    unittest.main()
