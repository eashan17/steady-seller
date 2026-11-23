"""S2 bench: score seller configurations on registered buyer suites against an oracle.

    python bench/run_bench.py --seller mvp --seller ref beta=0.5 --out bench/runs/demo
    python bench/run_bench.py --seller seller.py env.SELLER_MODE=V1 --suites public,proxy --workers 4 --resume
    python bench/run_bench.py --list-suites

Seller specs (`--seller`, repeatable; the first one is the base for paired comparisons):

    PATH[:Class] [label=NAME] [KEY=VALUE ...] [env.NAME=VALUE ...]

    PATH      a .py file (relative to the working directory or the kit folder, or absolute; Windows
              drive letters are fine) or an alias: baseline, mvp, ref (bench/reference_seller.py), seller.
    KEY=VALUE constructor keyword arguments, parsed as JSON when possible (0.5, true, null, "x").
    env.N=V   environment variables set while this seller is imported, built and played.
    label=    the name used in the report and the resume file (default: derived from the spec).

Output in --out: report.json, report.md, rows/<label>.jsonl (one line per negotiation, the
resume ledger) and, with --transcripts, transcripts/<label>.jsonl.

--resume keeps every row of rows/<label>.jsonl whose fingerprint (spec + seller source hash)
matches and plays only the missing negotiations. Without it the ledger is started afresh.
--workers N plays N negotiations at once; each worker thread owns its own Seller instance, and
sellers are assumed not to learn across negotiations (plan NON_GOALS), so results do not depend
on N. Use it for slow sellers or LLM buyers; deterministic sellers are CPU-bound and gain little.

Component metrics come from an optional `seller.stats` dict, read after every negotiation and
describing that negotiation only (reset it in `start()`). Recognised keys, all optional:

    turns              int    respond() calls (default: the bench's own count)
    fallback_turns     int    turns answered by the deterministic fallback
    guard_rejections   int    candidate messages the guard rejected
    classifier_backend str    e.g. "onnx", "linear", "none"
    classifier_ms      float or list of floats, per-call classifier latency

Other numeric keys are summed and reported as totals. Anything missing shows "n/a".

Scores are points per 100 negotiations, as evaluate.py prints them. Loss columns are measured
against the full-basket oracle (sells the sellable basket at the buyer's private limit when that
is at or above the floor): oracle - margin - missed - deadlock - violation + infeasible gain = score.
"""
import argparse
import hashlib
import importlib
import importlib.util
import itertools
import json
import math
import os
import random
import re
import statistics
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

BENCH = Path(__file__).resolve().parent
KIT = BENCH.parent
for _p in (str(KIT), str(BENCH)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import buyers                                     # noqa: E402
import evaluate                                   # noqa: E402
import suites                                     # noqa: E402
from harness import floor_price, run_negotiation  # noqa: E402
from harness import load_policy, load_store       # noqa: E402

ALIASES = {
    "baseline": "example_seller.py:Seller",
    "mvp": "mvp_seller.py:Seller",
    "ref": "bench/reference_seller.py:ParamSeller",
    "seller": "seller.py:Seller",
}
STATS_KNOWN = ("turns", "fallback_turns", "guard_rejections", "classifier_backend", "classifier_ms")
NA = "n/a"


# ---------------------------------------------------------------- seller specs
def _value(text):
    for parse in (json.loads, {"None": None, "True": True, "False": False}.__getitem__):
        try:
            return parse(text)
        except (ValueError, KeyError):
            pass
    return text


def _safe(text):
    return re.sub(r"[^A-Za-z0-9._=-]+", "_", text).strip("_") or "seller"


@dataclass
class SellerSpec:
    """One seller configuration. Build from CLI tokens with `parse_seller`, or pass `factory`."""
    label: str
    path: str = None
    class_name: str = "Seller"
    kwargs: dict = field(default_factory=dict)
    env: dict = field(default_factory=dict)
    factory: object = None        # callable(store, policy) -> seller; bypasses path loading

    def describe(self):
        if self.factory is not None:
            return {"label": self.label, "factory": getattr(self.factory, "__qualname__", repr(self.factory)),
                    "kwargs": self.kwargs, "env": self.env}
        return {"label": self.label, "path": self.path, "class": self.class_name, "kwargs": self.kwargs, "env": self.env}

    def fingerprint(self):
        h = hashlib.sha256(json.dumps(self.describe(), sort_keys=True, default=str).encode())
        if self.factory is None:
            h.update(Path(self.path).read_bytes())
        return h.hexdigest()[:16]


def _resolve_path(path):
    p = Path(path)
    if p.is_absolute():
        return p
    for base in (Path.cwd(), KIT):
        if (base / p).exists():
            return (base / p).resolve()
    return (Path.cwd() / p).resolve()


def parse_seller(tokens):
    """['mvp'] or ['C:\\x\\seller.py:Seller', 'beta=0.5', 'env.SELLER_MODE=V1', 'label=v1'] -> SellerSpec."""
    head, rest = tokens[0], tokens[1:]
    alias = head.partition(":")[0] if head.partition(":")[0] in ALIASES else None
    target = ALIASES[alias] if alias else head
    m = re.match(r"^(?P<path>.+?)(?::(?P<cls>[A-Za-z_]\w*))?$", target)
    path, cls = m.group("path"), m.group("cls") or "Seller"
    if alias and ":" in head:
        cls = head.partition(":")[2]
    kwargs, env, label = {}, {}, None
    for tok in rest:
        key, eq, val = tok.partition("=")
        if not eq or not key:
            raise ValueError(f"seller argument {tok!r} is not KEY=VALUE")
        if key == "label":
            label = val
        elif key.startswith("env."):
            env[key[4:]] = val
        else:
            kwargs[key] = _value(val)
    resolved = _resolve_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"seller file not found: {path} (looked in {Path.cwd()} and {KIT})")
    if label is None:
        parts = [alias or resolved.stem] + [f"{k}={v}" for k, v in kwargs.items()] + \
                [f"{k}={v}" for k, v in env.items()]
        label = "_".join(parts)
    return SellerSpec(_safe(label), str(resolved), cls, kwargs, env)


@contextmanager
def _env(values):
    old = {k: os.environ.get(k) for k in values}
    os.environ.update({k: str(v) for k, v in values.items()})
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


_MODULES = {}


def seller_factory(spec):
    """callable(store, policy) -> a fresh seller. Call inside `_env(spec.env)`."""
    if spec.factory is not None:
        return lambda store, policy: spec.factory(store, policy, **spec.kwargs)
    path = Path(spec.path)
    key = (str(path), tuple(sorted(spec.env.items())))
    if key not in _MODULES:
        if str(path.parent) not in sys.path:
            sys.path.insert(0, str(path.parent))      # like harness.load_class: lets seller.py import negotiator/
        name = "bench_seller_" + hashlib.sha1(repr(key).encode()).hexdigest()[:10]
        mod_spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(module)
        _MODULES[key] = module
    cls = getattr(_MODULES[key], spec.class_name)
    return lambda store, policy: cls(store, policy, **spec.kwargs)


# ---------------------------------------------------------------- oracle and playing
def oracle(sc, store, floors):
    """(full-basket oracle value, sub-basket oracle value, value at the floor). tradeoffs.py:229-247."""
    lines = [(store[i["sku"]], min(i["qty"], store[i["sku"]]["stock"])) for i in sc["items"]]
    potential = sum(q * (r["price"] - r["cost"]) for r, q in lines)
    if potential <= 0:
        return 0.0, 0.0, 0.0
    R = sc["reservation"]
    top = sum(q * r["price"] for r, q in lines)
    fl = sum(q * floors[r["sku"]] for r, q in lines)
    full = (R * top - sum(q * r["cost"] for r, q in lines)) / potential if R * top >= fl else 0.0
    floor_v = (fl - sum(q * r["cost"] for r, q in lines)) / potential
    best = full
    for qs in itertools.product(*[range(q + 1) for _, q in lines]):
        sel = [(r, k) for (r, _), k in zip(lines, qs) if k]
        if not sel:
            continue
        li = sum(k * r["price"] for r, k in sel)
        if R * li >= sum(k * floors[r["sku"]] for r, k in sel):
            best = max(best, (R * li - sum(k * r["cost"] for r, k in sel)) / potential)
    return full, best, floor_v


class _Timed:
    """Times every respond() call (evaluate.Timed) for one negotiation at a time."""

    def __init__(self, seller):
        self.seller, self.times = seller, []

    def start(self, negotiation_id):
        self.times = []
        self.seller.start(negotiation_id)

    def respond(self, message):
        t0 = time.perf_counter()
        try:
            return self.seller.respond(message)
        finally:
            self.times.append(time.perf_counter() - t0)


def _stats_snapshot(seller):
    try:
        stats = getattr(seller, "stats", None)
        if not isinstance(stats, dict):
            return None
        return json.loads(json.dumps(stats, default=str))
    except Exception:            # a broken stats property must not cost the negotiation
        return None


def task_key(suite, seed, sc):
    return f"{suite}|{seed}|{sc['id']}"


def play_one(timed, suite, seed, sc, classes, store, policy, floors):
    buyer = classes[sc["buyer"]](sc, store, random.Random(sc["seed"]))
    res = run_negotiation(timed, buyer, store, policy, sc["id"])
    row = evaluate.score_one(res, sc, store, policy)
    row["oracle_full"], row["oracle_sub"], row["floor_value"] = oracle(sc, store, floors)
    row["crash"] = any(m.get("error") for m in res["transcript"])
    row["stubborn"] = sc["stubborn"]
    row.update(suite=suite, seed=seed, key=task_key(suite, seed, sc),
               turn_s=[round(t, 6) for t in timed.times], stats=_stats_snapshot(timed.seller))
    return row, res


# ---------------------------------------------------------------- aggregation
def _mean(xs):
    return statistics.mean(xs) if xs else 0.0


def agg(rows):
    """tradeoffs.py:265-290 plus the infeasible-deadlock split and buyer walks."""
    n = len(rows)
    if not n:
        return None
    deals = [r for r in rows if r["deal"]]
    feas = [r for r in rows if r["feasible"]]
    return {
        "n": n, "score": 100 * statistics.mean(r["score"] for r in rows),
        "oracle_full": 100 * statistics.mean(r["oracle_full"] for r in rows),
        "oracle_sub": 100 * statistics.mean(r["oracle_sub"] for r in rows),
        "deals": len(deals), "feasible": len(feas),
        "feasible_deals": sum(bool(r["deal"]) for r in feas),
        "deadlocks": sum(r["deadlock"] for r in rows),
        "deadlocks_feasible": sum(r["deadlock"] and r["feasible"] for r in rows),
        "deadlocks_infeasible": sum(r["deadlock"] and not r["feasible"] for r in rows),
        "seller_walked": sum(r["ended"] == "seller_walked" for r in rows),
        "wrong_walks": sum(r["ended"] == "seller_walked" and r["feasible"] for r in rows),
        "buyer_walked": sum(r["ended"] == "buyer_walked" for r in rows),
        "buyer_walked_feasible": sum(r["ended"] == "buyer_walked" and r["feasible"] for r in rows),
        "infeasible_deals": sum(bool(r["deal"]) and not r["feasible"] for r in rows),
        "margin_kept": statistics.mean(r["margin_kept"] for r in deals) if deals else 0.0,
        "policy": sum(r["policy_violations"] for r in rows), "truth": sum(r["truth_violations"] for r in rows),
        "crashes": sum(r["crash"] for r in rows),
        "loss_margin": 100 * sum(r["oracle_full"] - r["value"] for r in feas if r["deal"]) / n,
        "loss_missed": 100 * sum(r["oracle_full"] for r in feas if not r["deal"]) / n,
        "loss_deadlock": 100 * evaluate.P_DEADLOCK * sum(r["deadlock"] for r in rows) / n,
        "loss_violation": 100 * sum(evaluate.P_POLICY * r["policy_violations"] +
                                    evaluate.P_TRUTH * r["truth_violations"] for r in rows) / n,
        "gain_infeasible": 100 * sum(r["value"] for r in rows if r["deal"] and not r["feasible"]) / n,
    }


def _pct(xs, p):
    xs = sorted(xs)
    return xs[int(p * (len(xs) - 1))] if xs else None


def latency(rows):
    ts = [t for r in rows for t in r.get("turn_s") or []]
    if not ts:
        return NA
    return {"turns": len(ts), "mean_ms": 1000 * statistics.mean(ts), "p95_ms": 1000 * _pct(ts, .95),
            "max_ms": 1000 * max(ts)}


def components(rows):
    """Aggregate per-negotiation `seller.stats`. "n/a" when no row carries a stats dict."""
    with_stats = [r for r in rows if isinstance(r.get("stats"), dict)]
    if not with_stats:
        return NA
    st = [r["stats"] for r in with_stats]

    def num(v):
        return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)

    def total(key):
        vals = [s[key] for s in st if num(s.get(key))]
        return sum(vals) if vals else None

    turns = sum(s["turns"] if num(s.get("turns")) else len(r.get("turn_s") or []) for s, r in zip(st, with_stats))
    fb, gr = total("fallback_turns"), total("guard_rejections")
    backends = {}
    for s in st:
        if "classifier_backend" in s:
            backends[str(s["classifier_backend"])] = backends.get(str(s["classifier_backend"]), 0) + 1
    cms = []
    for s in st:
        v = s.get("classifier_ms")
        cms += [x for x in (v if isinstance(v, list) else [v]) if num(x)]
    other = {}
    for s in st:
        for k, v in s.items():
            if k not in STATS_KNOWN and num(v):
                other[k] = other.get(k, 0) + v
    return {
        "negotiations_with_stats": len(st), "turns": turns,
        "fallback_turns": NA if fb is None else fb,
        "fallback_rate": NA if fb is None or not turns else fb / turns,
        "guard_rejections": NA if gr is None else gr,
        "guard_rejections_per_negotiation": NA if gr is None else gr / len(st),
        "classifier_backend": backends or NA,
        "classifier_ms_p50": _pct(cms, .5) if cms else NA,
        "classifier_ms_p95": _pct(cms, .95) if cms else NA,
        "other_totals": other or NA,
    }


def summarise_suite(rows):
    out = agg(rows)
    out["by_buyer"] = {b: agg([r for r in rows if r["buyer"] == b]) for b in dict.fromkeys(r["buyer"] for r in rows)}
    out["latency"] = latency(rows)
    out["components"] = components(rows)
    return out


def paired(base_rows, other_rows):
    """Paired score difference (other - base) over the negotiations both played, with a normal 95% CI."""
    a = {r["key"]: r["score"] for r in base_rows}
    d = [100 * (r["score"] - a[r["key"]]) for r in other_rows if r["key"] in a]
    if not d:
        return None
    mean = statistics.mean(d)
    half = 1.96 * statistics.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else None
    return {"n_paired": len(d), "diff": mean, "ci95": None if half is None else [mean - half, mean + half],
            "wins": sum(x > 1e-12 for x in d), "losses": sum(x < -1e-12 for x in d)}


def d8_check(base, cand, suite_names):
    """plan.md D8 on rule suites: mean over public/proxy/shifted up, no suite drops > 0.2, zero-violation gates."""
    rule = [s for s in suite_names if suites.get(s).kind == "rule" and s in base and s in cand]
    if not rule:
        return None
    delta = {s: cand[s]["score"] - base[s]["score"] for s in rule}
    mean_suites = [s for s in suites.D8_MEAN_SUITES if s in rule]
    mean_delta = statistics.mean(delta[s] for s in mean_suites) if mean_suites else None
    gates = {k: sum(cand[s][k] for s in rule) for k in ("policy", "truth", "crashes")}
    worst = min(delta.values())
    ok = (mean_delta is None or mean_delta > 0) and worst >= -0.2 and not any(gates.values())
    return {"suite_delta": delta, "mean_suites": mean_suites, "mean_delta": mean_delta, "worst_delta": worst,
            "gates": gates, "pass": ok, "complete": len(mean_suites) == len(suites.D8_MEAN_SUITES)}


# ---------------------------------------------------------------- the run
def _load_ledger(path, fp):
    """Rows in `path` with fingerprint `fp`, by key. A torn last line (crash mid-write) is skipped."""
    keep, stale = {}, 0
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("fp") == fp:
                keep[row["key"]] = row
            else:
                stale += 1
    return keep, stale


def run_seller(spec, suite_names, out, store, policy, workers=1, resume=False, seeds=None, n=None,
               transcripts=False, log=print):
    floors = {s: floor_price(r, policy) for s, r in store.items()}
    fp = spec.fingerprint()
    rows_path = out / "rows" / f"{spec.label}.jsonl"
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    done, stale = _load_ledger(rows_path, fp) if resume else ({}, 0)
    with open(rows_path, "w", encoding="utf-8") as fh:        # rewrite: matching rows only, deduplicated
        for row in done.values():
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    tr_fh = None
    if transcripts:
        (out / "transcripts").mkdir(parents=True, exist_ok=True)
        tr_fh = open(out / "transcripts" / f"{spec.label}.jsonl", "a" if resume else "w", encoding="utf-8")

    plan, pending = {}, []
    for name in suite_names:
        suite = suites.get(name)
        classes = suite.buyer_classes()
        keys = []
        for seed in (seeds if seeds is not None else suite.seeds):
            for sc in buyers.make_scenarios(store, policy, n or suite.n, seed, classes):
                k = task_key(name, seed, sc)
                keys.append(k)
                if k not in done:
                    pending.append((name, seed, sc, classes))
        plan[name] = keys
    reused = sum(k in done for keys in plan.values() for k in keys)
    log(f"[{spec.label}] {len(pending)} to play, {reused} reused from {rows_path.name}"
        + (f", {stale} stale rows dropped (fingerprint changed)" if stale else ""))

    lock, cursor, errors, fresh = threading.Lock(), [0], [], {}
    t0, last_note = time.time(), [time.time()]
    total = len(pending)

    def worker():
        try:
            timed = _Timed(factory(store, policy))
        except Exception as e:
            with lock:
                errors.append({"key": None, "error": f"seller construction: {type(e).__name__}: {e}"})
            return
        while True:
            with lock:
                if cursor[0] >= total:
                    return
                name, seed, sc, classes = pending[cursor[0]]
                cursor[0] += 1
            try:
                row, res = play_one(timed, name, seed, sc, classes, store, policy, floors)
            except Exception as e:             # buyer or harness failure: not recorded, so --resume retries it
                with lock:
                    errors.append({"key": task_key(name, seed, sc), "error": f"{type(e).__name__}: {e}"})
                continue
            row["fp"], row["seller"] = fp, spec.label
            with lock:
                fresh[row["key"]] = row
                rows_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows_fh.flush()
                if tr_fh:
                    tr_fh.write(json.dumps({"key": row["key"], "seller": spec.label, **res}, ensure_ascii=False) + "\n")
                    tr_fh.flush()
                if time.time() - last_note[0] > 30:
                    last_note[0] = time.time()
                    log(f"[{spec.label}] {len(fresh)}/{total} played, {time.time() - t0:.0f}s")

    rows_fh = open(rows_path, "a", encoding="utf-8")
    try:
        with _env(spec.env):
            factory = seller_factory(spec)
            threads = [threading.Thread(target=worker, daemon=True) for _ in range(max(1, min(workers, total or 1)))]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
    finally:
        rows_fh.close()
        if tr_fh:
            tr_fh.close()
    elapsed = time.time() - t0
    rows_by_key = {**done, **fresh}
    by_suite = {}
    for name, keys in plan.items():
        rows = [rows_by_key[k] for k in keys if k in rows_by_key]
        if rows:
            by_suite[name] = summarise_suite(rows)
            log(f"[{spec.label}] {name:14} score {by_suite[name]['score']:6.2f}  oracle {by_suite[name]['oracle_full']:6.2f}"
                f"  n {len(rows)}/{len(keys)}")
    missing = sum(k not in rows_by_key for keys in plan.values() for k in keys)
    mean_suites = [s for s in suites.D8_MEAN_SUITES if s in by_suite]
    return {
        "spec": spec.describe(), "fp": fp,
        "run": {"played": len(fresh), "reused": reused, "stale_dropped": stale, "missing": missing,
                "errors": errors, "elapsed_s": elapsed, "workers": workers},
        "suites": by_suite,
        "mean_score": statistics.mean(by_suite[s]["score"] for s in mean_suites) if mean_suites else None,
        "worst_score": min((v["score"] for v in by_suite.values()), default=None),
        "_rows": {name: [rows_by_key[k] for k in keys if k in rows_by_key] for name, keys in plan.items()},
    }


def run_bench(specs, suite_names=None, out="bench/runs/default", workers=1, resume=False, seeds=None, n=None,
              transcripts=False, log=print):
    """Run every seller spec on every suite; write report.json and report.md to `out`; return the report."""
    labels = [s.label for s in specs]
    if len(set(labels)) != len(labels):
        raise ValueError(f"duplicate seller labels {labels}; set label=NAME")
    suite_names = list(suite_names or suites.default_names())
    for name in suite_names:
        suites.get(name)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    store, policy = load_store(), load_policy()
    t0 = time.time()
    results = {s.label: run_seller(s, suite_names, out, store, policy, workers, resume, seeds, n, transcripts, log)
               for s in specs}
    rows = {label: r.pop("_rows") for label, r in results.items()}
    comparisons, d8 = [], {}
    base = labels[0]
    for other in labels[1:]:
        for name in suite_names:
            p = paired(rows[base].get(name, []), rows[other].get(name, []))
            if p:
                comparisons.append({"suite": name, "base": base, "other": other, **p})
        d8[other] = d8_check(results[base]["suites"], results[other]["suites"], suite_names)
    report = {
        "meta": {"created": datetime.now().isoformat(timespec="seconds"), "kit": str(KIT), "out": str(out),
                 "elapsed_s": time.time() - t0, "workers": workers, "resume": resume,
                 "suites": {nm: {"seeds": seeds if seeds is not None else suites.get(nm).seeds,
                                 "n": n or suites.get(nm).n, "kind": suites.get(nm).kind,
                                 "description": suites.get(nm).description} for nm in suite_names}},
        "sellers": results, "comparisons": comparisons, "d8_vs_base": d8 or None,
    }
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    (out / "report.md").write_text(render_markdown(report), encoding="utf-8")
    log(f"wrote {out / 'report.json'} and {out / 'report.md'} ({report['meta']['elapsed_s']:.1f}s)")
    return report


# ---------------------------------------------------------------- markdown
def _f(x, nd=2):
    if x is None or x == NA:
        return NA
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    if isinstance(x, dict):
        return ", ".join(f"{k}: {v}" for k, v in x.items())
    return str(x)


def _table(header, rows):
    def numeric(c):
        return isinstance(c, (int, float)) or bool(re.match(r"^(-?[\d.]+|n/a|\[|$)", str(c)))
    align = ["---:" if rows and all(numeric(r[i]) for r in rows) else "---" for i in range(len(header))]
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(align) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def render_markdown(report):
    S = report["sellers"]
    labels = list(S)
    names = list(report["meta"]["suites"])
    md = ["# Bench report", "",
          f"- **Created:** {report['meta']['created']}  ",
          f"- **Suites:** " + ", ".join(f"`{nm}` (seeds {v['seeds']}, n {v['n']})" for nm, v in report['meta']['suites'].items()),
          f"- **Sellers:** " + ", ".join(f"`{lb}`" for lb in labels) + f" (base for comparisons: `{labels[0]}`)",
          f"- **Elapsed:** {report['meta']['elapsed_s']:.1f}s, workers {report['meta']['workers']}", "",
          "Scores are points per 100 negotiations. Losses are measured against the full-basket oracle: "
          "oracle - margin - missed - deadlock - violations + infeasible gain = score.", "",
          "## Scores", ""]
    rows = []
    for nm in names:
        first = next((S[lb]["suites"][nm] for lb in labels if nm in S[lb]["suites"]), None)
        rows.append([f"`{nm}`", _f(first and first["oracle_full"])] +
                    [_f(S[lb]["suites"][nm]["score"]) if nm in S[lb]["suites"] else NA for lb in labels])
    rows.append(["mean (public/proxy/shifted)", ""] + [_f(S[lb]["mean_score"]) for lb in labels])
    rows.append(["worst suite", ""] + [_f(S[lb]["worst_score"]) for lb in labels])
    md += [_table(["Suite", "Oracle"] + [f"`{lb}`" for lb in labels], rows), "", "## Points lost, by cause", ""]
    md.append(_table(["Suite", "Seller", "Score", "Oracle", "Margin", "Missed feasible", "Deadlocks", "Violations",
                      "Infeasible gain"],
                     [[f"`{nm}`", f"`{lb}`"] + [_f(x[k]) for k in ("score", "oracle_full", "loss_margin", "loss_missed",
                                                                  "loss_deadlock", "loss_violation", "gain_infeasible")]
                      for nm in names for lb in labels for x in [S[lb]["suites"].get(nm)] if x]))
    md += ["", "## Outcomes", ""]
    md.append(_table(["Suite", "Seller", "n", "Deals", "Feasible deals", "Deadlocks (feasible / infeasible)",
                      "Wrongful walk-aways", "Seller walked", "Buyer walked (feasible)", "Policy", "Truth", "Crashes",
                      "Margin kept"],
                     [[f"`{nm}`", f"`{lb}`", x["n"], x["deals"], f"{x['feasible_deals']}/{x['feasible']}",
                       f"{x['deadlocks_feasible']} / {x['deadlocks_infeasible']}", x["wrong_walks"], x["seller_walked"],
                       f"{x['buyer_walked']} ({x['buyer_walked_feasible']})", x["policy"], x["truth"], x["crashes"],
                       _f(x["margin_kept"], 3)]
                      for nm in names for lb in labels for x in [S[lb]["suites"].get(nm)] if x]))
    md += ["", "## Components (from `seller.stats`) and latency", ""]
    crow = []
    for nm in names:
        for lb in labels:
            x = S[lb]["suites"].get(nm)
            if not x:
                continue
            c, lat = x["components"], x["latency"]
            g = (lambda k: NA) if c == NA else (lambda k: _f(c[k], 3))
            crow.append([f"`{nm}`", f"`{lb}`", g("fallback_rate"), g("guard_rejections"),
                         g("guard_rejections_per_negotiation"), g("classifier_backend"), g("classifier_ms_p95"),
                         NA if lat == NA else _f(lat["mean_ms"], 3), NA if lat == NA else _f(lat["p95_ms"], 3),
                         g("other_totals")])
    md.append(_table(["Suite", "Seller", "Fallback rate", "Guard rejections", "Rejections / negotiation",
                      "Classifier backend", "Classifier p95 ms", "Turn mean ms", "Turn p95 ms", "Other stats"], crow))
    if report["comparisons"]:
        md += ["", "## Paired comparisons (other - base, same scenarios)", ""]
        md.append(_table(["Suite", "Base", "Other", "n", "Diff", "95% CI", "Wins / losses"],
                         [[f"`{c['suite']}`", f"`{c['base']}`", f"`{c['other']}`", c["n_paired"], _f(c["diff"]),
                           NA if not c["ci95"] else f"[{c['ci95'][0]:.2f}, {c['ci95'][1]:.2f}]",
                           f"{c['wins']} / {c['losses']}"] for c in report["comparisons"]]))
    if report["d8_vs_base"]:
        md += ["", "## D8 check on rule suites (vs base)", "",
               "Pass = mean over public/proxy/shifted goes up, no rule suite drops more than 0.2, "
               "and zero policy violations, truth violations and crashes.", ""]
        md.append(_table(["Seller", "Mean delta", "Worst suite delta", "Policy / truth / crashes", "Pass", "All 3 mean suites run"],
                         [[f"`{lb}`", _f(d["mean_delta"]), _f(d["worst_delta"]),
                           f"{d['gates']['policy']} / {d['gates']['truth']} / {d['gates']['crashes']}",
                           "yes" if d["pass"] else "no", "yes" if d["complete"] else "no"]
                          for lb, d in report["d8_vs_base"].items() if d]))
    md += ["", "## By buyer", ""]
    md.append(_table(["Suite", "Seller", "Buyer", "n", "Score", "Oracle", "Margin lost", "Missed", "Deals",
                      "Deadlocks (feas / infeas)", "Wrongful walk-aways"],
                     [[f"`{nm}`", f"`{lb}`", b, y["n"], _f(y["score"]), _f(y["oracle_full"]), _f(y["loss_margin"]),
                       _f(y["loss_missed"]), y["deals"], f"{y['deadlocks_feasible']} / {y['deadlocks_infeasible']}",
                       y["wrong_walks"]]
                      for nm in names for lb in labels for x in [S[lb]["suites"].get(nm)] if x
                      for b, y in x["by_buyer"].items()]))
    md += ["", "## Run", ""]
    md.append(_table(["Seller", "Played", "Reused", "Stale dropped", "Missing", "Errors", "Elapsed s", "Fingerprint"],
                     [[f"`{lb}`", r["run"]["played"], r["run"]["reused"], r["run"]["stale_dropped"], r["run"]["missing"],
                       len(r["run"]["errors"]), _f(r["run"]["elapsed_s"], 1), r["fp"]] for lb, r in S.items()]))
    errs = [(lb, e) for lb, r in S.items() for e in r["run"]["errors"]]
    if errs:
        md += ["", "Errors (first 20):", ""] + [f"- `{lb}` {e['key']}: {e['error']}" for lb, e in errs[:20]]
    return "\n".join(md) + "\n"


# ---------------------------------------------------------------- plugins and CLI
def load_suite_plugins(extra=(), log=print):
    """Import bench-local plugin modules (suites.PLUGIN_MODULES) and any --suite-module files."""
    for name in suites.PLUGIN_MODULES:
        if (BENCH / f"{name}.py").exists() and name not in sys.modules:
            try:
                importlib.import_module(name)
            except Exception as e:
                log(f"warning: suite plugin {name}.py failed to import: {type(e).__name__}: {e}")
    for path in extra:
        p = _resolve_path(path)
        if str(p.parent) not in sys.path:
            sys.path.insert(0, str(p.parent))
        spec = importlib.util.spec_from_file_location(p.stem, p)
        module = importlib.util.module_from_spec(spec)
        sys.modules[p.stem] = module
        spec.loader.exec_module(module)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n", 1)[1])
    ap.add_argument("--seller", nargs="+", action="append", metavar="TOKEN",
                    help="PATH[:Class] [label=NAME] [KEY=VALUE ...] [env.NAME=VALUE ...]; repeatable")
    ap.add_argument("--suites", help="comma-separated suite names (default: every default suite)")
    ap.add_argument("--suite-module", action="append", default=[], help="a .py file that calls suites.register()")
    ap.add_argument("--seeds", help="override every suite's seeds, e.g. 0,1")
    ap.add_argument("--n", type=int, help="override every suite's negotiations per seed")
    ap.add_argument("--workers", type=int, default=1, help="negotiations played at once (one Seller per worker)")
    ap.add_argument("--resume", action="store_true", help="keep finished negotiations in --out and play the rest")
    ap.add_argument("--transcripts", action="store_true", help="also write transcripts/<label>.jsonl")
    ap.add_argument("--out", default=str(BENCH / "runs" / "default"), help="output folder")
    ap.add_argument("--list-suites", action="store_true")
    args = ap.parse_args(argv)
    load_suite_plugins(args.suite_module)
    if args.list_suites:
        for s in suites.REGISTRY.values():
            print(f"{s.name:14} {s.kind:5} {'default' if s.default else '       '} seeds {s.seeds} x {s.n}  {s.description}")
        return 0
    if not args.seller:
        ap.error("give at least one --seller (aliases: " + ", ".join(ALIASES) + ")")
    specs = [parse_seller(tokens) for tokens in args.seller]
    names = [s.strip() for s in args.suites.split(",") if s.strip()] if args.suites else None
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None
    report = run_bench(specs, names, args.out, args.workers, args.resume, seeds, args.n, args.transcripts,
                       log=lambda m: print(m, file=sys.stderr, flush=True))
    bad = sum(len(r["run"]["errors"]) + r["run"]["missing"] for r in report["sellers"].values())
    print(Path(args.out, "report.md").read_text(encoding="utf-8"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
