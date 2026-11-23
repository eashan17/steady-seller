"""S2 bench: fidelity against the trade-off experiments, plus resume, workers, stats and the registry.

    python -X utf8 -m unittest tests.test_bench -v        (from ps2-selling-to-machines/)

Exit conditions (plan.md S2): 1 fidelity, 2 resume, 3 workers, 4 "n/a" stats, 5 registry, 6 speed.
"""
import io
import json
import os
import shutil
import sys
import tempfile
import textwrap
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
for _p in (str(KIT), str(KIT / "bench")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import buyers               # noqa: E402
import run_bench            # noqa: E402
import suites               # noqa: E402
from reference_seller import ParamSeller  # noqa: E402

EXPERIMENTS = KIT.parent / ".sessions" / "ps2-seller-optimize-20261120" / "context" / "experiments"
QUIET = dict(log=lambda m: None)


def spec(tokens):
    return run_bench.parse_seller(tokens if isinstance(tokens, list) else tokens.split())


class BenchCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="bench-test-"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def bench(self, specs, suite_names, out="out", **kw):
        return run_bench.run_bench(specs, suite_names, self.tmp / out, **{**QUIET, **kw})


# ---------------------------------------------------------------- exit condition 1: fidelity
class Fidelity(BenchCase):
    def test_given_mvp_on_public_seeds_0_to_2_when_benched_then_scores_match_the_mvp_session(self):
        expected = {0: (50.41, 5), 1: (50.94, 8), 2: (44.65, 11)}   # fidelity.json
        for seed, (score, deadlocks) in expected.items():
            rep = self.bench([spec("mvp"), spec("ref")], ["public"], out=f"s{seed}", seeds=[seed])
            for label in ("mvp", "ref"):
                got = rep["sellers"][label]["suites"]["public"]
                self.assertEqual(round(got["score"], 2), score, f"{label} seed {seed}")
                self.assertEqual(got["deadlocks"], deadlocks, f"{label} seed {seed}")
            cmp = rep["comparisons"][0]
            self.assertEqual((cmp["diff"], cmp["wins"], cmp["losses"]), (0, 0, 0), "ParamSeller defaults == MVP")

    @unittest.skipUnless((EXPERIMENTS / "tradeoffs.json").exists(), "tradeoffs.json not present")
    def test_given_baseline_and_mvp_when_benched_then_loss_tables_match_tradeoffs_json(self):
        gap = json.loads((EXPERIMENTS / "tradeoffs.json").read_text(encoding="utf-8"))["gap"]
        rep = self.bench([spec("baseline"), spec("mvp")], ["public", "proxy", "shifted"])
        checked = 0
        for label in ("baseline", "mvp"):
            for suite in ("public", "proxy", "shifted"):
                ref, got = gap[label][suite], rep["sellers"][label]["suites"][suite]
                pairs = [(k, ref[k], got[k]) for k in ref if k != "by_buyer"]
                pairs += [(f"{b}.{k}", v, got["by_buyer"][b][k]) for b, x in ref["by_buyer"].items() for k, v in x.items()]
                for key, want, have in pairs:
                    self.assertLessEqual(abs(want - have), 0.05, f"{label}/{suite}/{key}: {have} vs {want}")
                    checked += 1
                total = got["loss_margin"] + got["loss_missed"] + got["loss_deadlock"] + got["loss_violation"]
                self.assertAlmostEqual(got["oracle_full"] - total + got["gain_infeasible"], got["score"], places=6)
            self.assertAlmostEqual(rep["sellers"][label]["mean_score"], gap[label]["mean_score"], delta=0.05)
        self.assertGreater(checked, 200)

    def test_given_shape_05_reference_when_benched_on_noise_suites_then_scores_are_30_6_and_29_1(self):
        rep = self.bench([spec("ref beta=0.5")], ["jitter_0.05", "jitter_0.08"])
        got = rep["sellers"]["ref_beta=0.5"]["suites"]
        self.assertLessEqual(abs(got["jitter_0.05"]["score"] - 30.6), 0.05, got["jitter_0.05"]["score"])
        self.assertLessEqual(abs(got["jitter_0.08"]["score"] - 29.1), 0.05, got["jitter_0.08"]["score"])

    def test_given_the_registry_then_builtin_suites_match_tradeoff_analysis_section_1(self):
        want = {"public": ([0, 1, 2, 3, 4], 210), "proxy": ([0, 1, 2], 200), "shifted": ([0, 1, 2], 210),
                "jitter_0.015": ([0, 1, 2], 200), "jitter_0.05": ([0, 1, 2], 200), "jitter_0.08": ([0, 1, 2], 200)}
        for name, (seeds, n) in want.items():
            s = suites.get(name)
            self.assertEqual((s.seeds, s.n, s.kind, s.default), (seeds, n, "rule", True), name)
        jit = suites.get("jitter_0.05").buyer_classes()
        self.assertEqual(list(jit), list(suites.get("proxy").buyer_classes()))
        self.assertTrue(all(c.price_jitter == 0.05 for c in jit.values()))


# ---------------------------------------------------------------- exit condition 2: resume
class Resume(BenchCase):
    def test_given_a_finished_run_when_resumed_then_nothing_is_replayed_and_the_score_is_unchanged(self):
        first = self.bench([spec("ref")], ["public"], seeds=[0], n=12)
        again = self.bench([spec("ref")], ["public"], seeds=[0], n=12, resume=True)
        self.assertEqual(first["sellers"]["ref"]["run"]["played"], 12)
        self.assertEqual((again["sellers"]["ref"]["run"]["played"], again["sellers"]["ref"]["run"]["reused"]), (0, 12))
        self.assertEqual(again["sellers"]["ref"]["suites"]["public"]["score"],
                         first["sellers"]["ref"]["suites"]["public"]["score"])

    def test_given_an_interrupted_ledger_when_resumed_then_only_the_missing_negotiations_are_played(self):
        first = self.bench([spec("ref")], ["public"], seeds=[0], n=12)
        ledger = self.tmp / "out" / "rows" / "ref.jsonl"
        lines = ledger.read_text(encoding="utf-8").splitlines()
        ledger.write_text("\n".join(lines[:9]) + "\n" + lines[9][:40], encoding="utf-8")   # 9 rows + a torn line
        again = self.bench([spec("ref")], ["public"], seeds=[0], n=12, resume=True)
        self.assertEqual((again["sellers"]["ref"]["run"]["played"], again["sellers"]["ref"]["run"]["reused"]), (3, 9))
        self.assertEqual(again["sellers"]["ref"]["suites"]["public"]["score"],
                         first["sellers"]["ref"]["suites"]["public"]["score"])
        self.assertEqual(len(ledger.read_text(encoding="utf-8").splitlines()), 12)

    def test_given_a_changed_seller_config_when_resumed_then_stale_rows_are_replayed(self):
        self.bench([spec("ref label=x")], ["public"], seeds=[0], n=6)
        again = self.bench([spec("ref beta=0.5 label=x")], ["public"], seeds=[0], n=6, resume=True)
        run = again["sellers"]["x"]["run"]
        self.assertEqual((run["played"], run["reused"], run["stale_dropped"]), (6, 0, 6))

    def test_given_no_resume_flag_when_rerun_then_everything_is_replayed(self):
        self.bench([spec("ref")], ["public"], seeds=[0], n=6)
        again = self.bench([spec("ref")], ["public"], seeds=[0], n=6)
        self.assertEqual(again["sellers"]["ref"]["run"]["played"], 6)


# ---------------------------------------------------------------- exit condition 3: workers
class SlowSeller(ParamSeller):
    """Sleeps per turn and fails loudly if two negotiations ever share this instance at once."""
    delay = 0.01

    def __init__(self, store, policy, **kw):
        super().__init__(store, policy, **kw)
        self.busy = False

    def respond(self, msg):
        if self.busy:
            raise RuntimeError("seller instance shared between workers")
        self.busy = True
        try:
            time.sleep(self.delay)
            return super().respond(msg)
        finally:
            self.busy = False


class Workers(BenchCase):
    def test_given_a_slow_seller_when_run_with_4_workers_then_it_is_faster_and_identical(self):
        slow = run_bench.SellerSpec("slow", factory=SlowSeller)
        t0 = time.time()
        one = self.bench([slow], ["public"], out="w1", seeds=[0], n=16, workers=1)
        t1 = time.time()
        four = self.bench([slow], ["public"], out="w4", seeds=[0], n=16, workers=4)
        t4 = time.time() - t1
        t1 -= t0
        self.assertLess(t4, 0.6 * t1, f"4 workers {t4:.2f}s vs 1 worker {t1:.2f}s")
        a, b = one["sellers"]["slow"]["suites"]["public"], four["sellers"]["slow"]["suites"]["public"]
        self.assertEqual((a["score"], a["deals"], a["deadlocks"]), (b["score"], b["deals"], b["deadlocks"]))
        self.assertEqual(b["crashes"], 0, "each worker must own its seller")
        rows = lambda d: {json.loads(x)["key"]: json.loads(x)["score"]
                          for x in (self.tmp / d / "rows" / "slow.jsonl").read_text(encoding="utf-8").splitlines()}
        self.assertEqual(rows("w1"), rows("w4"))

    def test_given_a_buyer_that_raises_when_benched_then_the_error_is_reported_and_retried_on_resume(self):
        class Flaky(buyers.Cooperative):
            fail = True

            def respond(self, msg):
                if Flaky.fail and self.s["id"] == "neg-002":
                    raise ConnectionError("LLM buyer timed out")
                return super().respond(msg)

        suites.register("test_flaky", {"flaky": Flaky}, [0], 4, kind="llm", replace=True)
        try:
            first = self.bench([spec("ref")], ["test_flaky"], workers=2)
            run = first["sellers"]["ref"]["run"]
            self.assertEqual((run["played"], run["missing"], len(run["errors"])), (3, 1, 1))
            self.assertIn("timed out", run["errors"][0]["error"])
            Flaky.fail = False
            again = self.bench([spec("ref")], ["test_flaky"], resume=True)
            self.assertEqual((again["sellers"]["ref"]["run"]["played"], again["sellers"]["ref"]["run"]["missing"]), (1, 0))
        finally:
            suites.REGISTRY.pop("test_flaky", None)


# ---------------------------------------------------------------- exit condition 4: component metrics
class StatsSeller(ParamSeller):
    def start(self, nid):
        super().start(nid)
        self.stats = {"turns": 0, "fallback_turns": 0, "guard_rejections": 0, "classifier_backend": "linear",
                      "classifier_ms": [], "belief_updates": 0}

    def respond(self, msg):
        self.stats["turns"] += 1
        self.stats["fallback_turns"] += self.stats["turns"] == 1
        self.stats["guard_rejections"] += 2
        self.stats["classifier_ms"] += [1.0, 3.0]
        self.stats["belief_updates"] += 1
        return super().respond(msg)


class BrokenStatsSeller(ParamSeller):
    @property
    def stats(self):
        raise RuntimeError("boom")


class Components(BenchCase):
    def test_given_a_seller_without_stats_when_benched_then_metrics_show_na(self):
        rep = self.bench([spec("ref")], ["public"], seeds=[0], n=6)
        self.assertEqual(rep["sellers"]["ref"]["suites"]["public"]["components"], "n/a")
        md = (self.tmp / "out" / "report.md").read_text(encoding="utf-8")
        line = next(x for x in md.splitlines() if x.startswith("| `public` | `ref` | n/a"))
        self.assertGreaterEqual(line.count("n/a"), 5)

    def test_given_a_seller_with_stats_when_benched_then_rates_and_totals_are_reported(self):
        rep = self.bench([run_bench.SellerSpec("st", factory=StatsSeller)], ["public"], seeds=[0], n=9)
        c = rep["sellers"]["st"]["suites"]["public"]["components"]
        turns = rep["sellers"]["st"]["suites"]["public"]["latency"]["turns"]
        self.assertEqual((c["negotiations_with_stats"], c["turns"]), (9, turns))
        self.assertEqual(c["fallback_turns"], 9)
        self.assertAlmostEqual(c["fallback_rate"], 9 / turns)
        self.assertEqual((c["guard_rejections"], c["guard_rejections_per_negotiation"]), (2 * turns, 2 * turns / 9))
        self.assertEqual(c["classifier_backend"], {"linear": 9})
        self.assertEqual((c["classifier_ms_p50"], c["classifier_ms_p95"]), (1.0, 3.0))
        self.assertEqual(c["other_totals"], {"belief_updates": turns})

    def test_given_a_stats_property_that_raises_when_benched_then_metrics_show_na_and_nothing_crashes(self):
        rep = self.bench([run_bench.SellerSpec("bad", factory=BrokenStatsSeller)], ["public"], seeds=[0], n=6)
        got = rep["sellers"]["bad"]["suites"]["public"]
        self.assertEqual((got["components"], got["crashes"], got["n"]), ("n/a", 0, 6))


# ---------------------------------------------------------------- exit condition 5: registry
class DealAtListBuyer(buyers.Buyer):
    feasible, zone, anchor = 1.0, (0.9, 1.0), (0.5, 0.6)

    def respond(self, msg):
        if msg.get("type") == "offer" and self.usable(msg.get("items")):
            return {"type": "accept", "text": "ok"}
        return {"type": "counter", "items": self.priced(self.basket, self.ratio(msg["round"])), "text": "?"}


class Registry(BenchCase):
    def tearDown(self):
        for name in ("test_llm", "test_lazy", "test_plugin"):
            suites.REGISTRY.pop(name, None)
        super().tearDown()

    def test_given_a_registered_llm_suite_then_it_runs_but_is_not_in_the_default_set(self):
        suites.register("test_llm", {"llm_like": DealAtListBuyer}, seeds=[3], n=5, kind="llm")
        self.assertNotIn("test_llm", suites.default_names())
        rep = self.bench([spec("ref")], ["test_llm"])
        got = rep["sellers"]["ref"]["suites"]["test_llm"]
        self.assertEqual((got["n"], list(got["by_buyer"])), (5, ["llm_like"]))
        self.assertEqual(rep["meta"]["suites"]["test_llm"]["kind"], "llm")
        with self.assertRaises(ValueError):
            suites.register("test_llm", {"x": DealAtListBuyer}, [0], 1)

    def test_given_lazy_classes_then_they_are_built_only_when_the_suite_runs(self):
        calls = []
        suites.register("test_lazy", lambda: calls.append(1) or {"a": DealAtListBuyer}, [0], 2, kind="llm")
        self.assertEqual(calls, [])
        self.bench([spec("ref")], ["test_lazy"])
        self.assertEqual(calls, [1])

    def test_given_a_non_buyer_class_then_the_suite_is_rejected(self):
        suites.register("test_lazy", {"bad": dict}, [0], 2)
        with self.assertRaises(TypeError):
            self.bench([spec("ref")], ["test_lazy"])

    def test_given_a_suite_module_on_the_cli_then_its_suites_are_listed_and_runnable(self):
        mod = self.tmp / "my_suites.py"
        mod.write_text(textwrap.dedent("""
            import buyers, suites
            suites.register("test_plugin", {"coop": buyers.Cooperative}, [0], 3, kind="llm", description="plugin")
        """), encoding="utf-8")
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = run_bench.main(["--suite-module", str(mod), "--list-suites"])
        self.assertEqual(code, 0)
        self.assertIn("test_plugin", buf.getvalue())
        with redirect_stdout(io.StringIO()):
            code = run_bench.main(["--seller", "ref", "--suites", "test_plugin", "--out", str(self.tmp / "cli")])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads((self.tmp / "cli" / "report.json").read_text(encoding="utf-8"))
                         ["sellers"]["ref"]["suites"]["test_plugin"]["n"], 3)


# ---------------------------------------------------------------- seller specs
class Specs(BenchCase):
    def test_given_an_absolute_path_with_class_kwargs_and_env_then_the_seller_loads_with_them(self):
        f = self.tmp / "my_seller.py"
        f.write_text(textwrap.dedent(f"""
            import os, sys
            sys.path.insert(0, {str(KIT / 'bench')!r})
            from reference_seller import ParamSeller
            class Mine(ParamSeller):
                def start(self, nid):
                    super().start(nid)
                    self.stats = {{"mode_v1": int(os.environ.get("BENCH_TEST_MODE") == "V1"), "beta_x10": self.beta * 10}}
        """), encoding="utf-8")
        s = spec([f"{f.resolve()}:Mine", "beta=0.5", "env.BENCH_TEST_MODE=V1"])
        self.assertEqual((s.class_name, s.kwargs, s.env, s.label), ("Mine", {"beta": 0.5}, {"BENCH_TEST_MODE": "V1"},
                                                                    "my_seller_beta=0.5_BENCH_TEST_MODE=V1"))
        rep = self.bench([s], ["public"], seeds=[0], n=4)
        self.assertEqual(rep["sellers"][s.label]["suites"]["public"]["components"]["other_totals"],
                         {"mode_v1": 4, "beta_x10": 20.0})
        self.assertNotIn("BENCH_TEST_MODE", os.environ, "env vars are restored after the run")

    def test_given_bad_specs_then_they_fail_clearly(self):
        with self.assertRaises(FileNotFoundError):
            spec("no_such_seller.py")
        with self.assertRaises(ValueError):
            spec("ref beta")
        with self.assertRaises(ValueError):
            self.bench([spec("ref"), spec("ref")], ["public"], n=1)
        with self.assertRaises(KeyError):
            self.bench([spec("ref")], ["no_such_suite"])


# ---------------------------------------------------------------- exit condition 6: speed
class Speed(BenchCase):
    def test_given_a_deterministic_seller_when_the_default_set_runs_then_it_takes_under_2_minutes(self):
        t0 = time.time()
        rep = self.bench([spec("mvp")], None)
        elapsed = time.time() - t0
        self.assertEqual(list(rep["meta"]["suites"]), suites.default_names())
        self.assertEqual(sum(v["n"] for v in rep["sellers"]["mvp"]["suites"].values()), 4080)
        self.assertLess(elapsed, 120)
        print(f"\n  default rule-based set (4080 negotiations, mvp): {elapsed:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    unittest.main()
