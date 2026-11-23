"""Evaluate both runtime backends on the held-out set and write the S3 report (S3 / T03-T05 evidence).

    python -X utf8 training/eval_classifier.py      (any Python with onnxruntime; no training packages needed)

Measures, through negotiator/textsignals.py exactly as the seller will call it:
  - per-tag precision / recall / F1 and group macro-F1 on heldout.jsonl (and valid.jsonl for reference)
  - linear-vs-encoder agreement (element-wise, and per-tag agreement F1) on held-out + valid
  - model file sizes; cold-load time; warm p50 / p95 / max latency per message
Writes .sessions/<id>/results/classifier-eval.md and classifier-eval.json.
"""
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, MODELS, RESULTS, TAGS, load_textsignals, read_jsonl  # noqa: E402

GROUP_TARGETS = [("intent", 0.70), ("claims+attacks", 0.80), ("tone", None)]


def prf(pred, gold, j):
    tp = sum(1 for p, g in zip(pred, gold) if p[j] and g[j])
    fp = sum(1 for p, g in zip(pred, gold) if p[j] and not g[j])
    fn = sum(1 for p, g in zip(pred, gold) if not p[j] and g[j])
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0
    return prec, rec, f1, tp + fn


def group_tags(ts, name):
    return list(ts.GROUPS["claims"]) + list(ts.GROUPS["attacks"]) if name == "claims+attacks" else list(ts.GROUPS[name])


def run_backend(ts, choice, rows):
    ts.reset(choice)
    t0 = time.perf_counter()
    name = ts.backend()
    load_ms = (time.perf_counter() - t0) * 1000
    preds, lat = [], []
    for r in rows:
        t = time.perf_counter()
        act = set(ts.active(r["text"]))
        lat.append((time.perf_counter() - t) * 1000)
        preds.append([1 if k in act else 0 for k in TAGS])
    return name, load_ms, preds, lat


def scores(preds, rows, ts):
    gold = [[1 if k in set(r["tags"]) else 0 for k in TAGS] for r in rows]
    per = {k: prf(preds, gold, j) for j, k in enumerate(TAGS)}
    groups = {}
    for g, _ in GROUP_TARGETS:
        tags = [t for t in group_tags(ts, g) if per[t][3] > 0]
        groups[g] = statistics.mean(per[t][2] for t in tags) if tags else None
    return per, groups


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def main():
    ts = load_textsignals()
    heldout, valid = read_jsonl(DATA / "heldout.jsonl"), read_jsonl(DATA / "valid.jsonl")
    hand = read_jsonl(DATA / "heldout_handwritten.jsonl")
    out = {"heldout_lines": len(heldout), "valid_lines": len(valid), "handwritten_lines": len(hand), "backends": {}}
    # long-ish probe set for latency: held-out + valid, plus a 2,000-char message
    probe = heldout + valid + [{"text": ("I really want this but the price is high. " * 60)[:2000], "tags": []}]
    for choice in ("onnx", "linear"):
        name, load_ms, ph, lat_h = run_backend(ts, choice, heldout)
        if name != choice:
            out["backends"][choice] = {"error": f"backend resolved to {name}"}
            continue
        _, _, pv, _ = run_backend(ts, choice, valid)
        _, _, pw, _ = run_backend(ts, choice, hand)
        per_w, grp_w = scores(pw, hand, ts)
        _, _, _, lat = run_backend(ts, choice, probe)
        per_h, grp_h = scores(ph, heldout, ts)
        per_v, grp_v = scores(pv, valid, ts)
        out["backends"][choice] = {
            "cold_load_ms": round(load_ms, 1), "p50_ms": round(pct(lat, 0.5), 2), "p95_ms": round(pct(lat, 0.95), 2),
            "max_ms": round(max(lat), 2), "heldout_groups": grp_h, "valid_groups": grp_v,
            "handwritten_groups": grp_w, "handwritten_per_tag": {k: [round(x, 3) for x in v[:3]] + [v[3]]
                                                                for k, v in per_w.items()},
            "heldout_per_tag": {k: [round(x, 3) for x in v[:3]] + [v[3]] for k, v in per_h.items()},
            "valid_per_tag": {k: [round(x, 3) for x in v[:3]] + [v[3]] for k, v in per_v.items()},
            "_preds": ph + pv}
    if all("error" not in out["backends"].get(b, {"error": 1}) for b in ("onnx", "linear")):
        a, b = out["backends"]["onnx"].pop("_preds"), out["backends"]["linear"].pop("_preds")
        el = sum(x == y for ra, rb in zip(a, b) for x, y in zip(ra, rb)) / (len(a) * len(TAGS))
        exact = sum(ra == rb for ra, rb in zip(a, b)) / len(a)
        tagf1 = {k: round(prf(b, a, j)[2], 3) for j, k in enumerate(TAGS)}
        out["agreement"] = {"elementwise": round(el, 4), "exact_message": round(exact, 4),
                            "per_tag_f1_vs_encoder": tagf1,
                            "macro_f1_vs_encoder": round(statistics.mean(tagf1.values()), 3)}
    for b in out["backends"].values():
        b.pop("_preds", None)
    sizes = {f.name: round(f.stat().st_size / 1e6, 2) for f in sorted(MODELS.glob("*")) if f.is_file()}
    out["sizes_mb"] = sizes
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "classifier-eval.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    write_md(out, ts)
    print(json.dumps({k: v for k, v in out.items() if k != "backends"}, indent=1))
    for k, v in out["backends"].items():
        print(k, {x: v.get(x) for x in ("heldout_groups", "p95_ms", "cold_load_ms", "error")})
    return 0


def fmt(x):
    return "n/a" if x is None else f"{x:.3f}"


def write_md(out, ts):
    L = ["# S3 classifier evaluation", "",
         f"Generated by `training/eval_classifier.py`. Held-out lines: {out['heldout_lines']} (different prompt "
         f"template, never trained on). Valid lines: {out['valid_lines']} (whole generation groups unseen in train; used for "
         "threshold calibration, shown for reference).", "",
         "## Group macro-F1 (held-out)", "", "| Backend | intent (≥0.70) | claims+attacks (≥0.80) | tone |",
         "|---|---|---|---|"]
    for name, b in out["backends"].items():
        if "error" in b:
            L.append(f"| {name} | {b['error']} | | |")
            continue
        g = b["heldout_groups"]
        L.append(f"| {name} | {fmt(g['intent'])} | {fmt(g['claims+attacks'])} | {fmt(g['tone'])} |")
    L += ["", "Group macro-F1 averages the tags that have at least one held-out positive.", "",
          "## Second held-out set: hand-written, independent wording (robustness check, not the gate)", "",
          f"{out['handwritten_lines']} lines written by hand without reusing TAGS.md examples. Support is 1-5 "
          "lines per tag, so treat these as indicative only.", "",
          "| Backend | intent | claims+attacks | tone | all-tag macro-F1 |", "|---|---|---|---|---|"]
    for name, b in out["backends"].items():
        if "error" not in b:
            g = b["handwritten_groups"]
            per = [v[2] for v in b["handwritten_per_tag"].values() if v[3] > 0]
            L.append(f"| {name} | {fmt(g['intent'])} | {fmt(g['claims+attacks'])} | {fmt(g['tone'])} | "
                     f"{fmt(statistics.mean(per) if per else None)} |")
    L += ["",
          "## Per-tag held-out P / R / F1 (support)", "", "| Tag | onnx P | onnx R | onnx F1 | linear F1 | support |",
          "|---|---|---|---|---|---|"]
    on, li = out["backends"].get("onnx", {}), out["backends"].get("linear", {})
    for k in TAGS:
        o, l_ = on.get("heldout_per_tag", {}).get(k), li.get("heldout_per_tag", {}).get(k)
        if o:
            L.append(f"| `{k}` | {o[0]:.2f} | {o[1]:.2f} | {o[2]:.2f} | {l_[2]:.2f} | {o[3]} |" if l_ else
                     f"| `{k}` | {o[0]:.2f} | {o[1]:.2f} | {o[2]:.2f} | n/a | {o[3]} |")
    L += ["", "## Latency and size", "", "| Backend | cold load ms | p50 ms | p95 ms | max ms |", "|---|---|---|---|---|"]
    for name, b in out["backends"].items():
        if "error" not in b:
            L.append(f"| {name} | {b['cold_load_ms']} | {b['p50_ms']} | {b['p95_ms']} | {b['max_ms']} |")
    L += ["", "Latency is per message through `textsignals.active()`, warm, one thread, over held-out + valid "
          "lines plus one 2,000-character message.", "", "| File | MB |", "|---|---|"]
    L += [f"| `models/{f}` | {mb} |" for f, mb in out["sizes_mb"].items()]
    if "agreement" in out:
        ag = out["agreement"]
        L += ["", "## Linear fallback vs encoder (held-out + valid)", "",
              f"- Element-wise tag agreement: **{ag['elementwise']:.4f}** (exit condition 5 asks ≥ 0.85).",
              f"- Whole-message agreement (all 18 decisions equal): {ag['exact_message']:.4f}.",
              f"- Macro agreement-F1 (encoder decisions as truth): {ag['macro_f1_vs_encoder']:.3f}. Element-wise "
              "agreement is inflated by the many true negatives, so this is the stricter view.", ""]
    L += history(out)
    (RESULTS / "classifier-eval.md").write_text("\n".join(L) + "\n", encoding="utf-8")


HISTORY = [  # (saved json, what changed) - every look at the held-out set, so none is hidden
    ("classifier-eval-v1-randomsplit.json", "v1: thresholds fit on a random 15% line split of the training pool"),
    ("classifier-eval-v2-groupsplit.json", "v2: thresholds fit on a stratified split of whole generation groups"),
]


def history(current):
    rows = [(json.loads((RESULTS / f).read_text(encoding="utf-8")), why) for f, why in HISTORY if (RESULTS / f).exists()]
    rows.append((current, "v3 (this report): head refit on train+valid, thresholds from grouped 5-fold "
                          "out-of-fold probabilities"))
    L = ["", "## Evaluation history (every look at the held-out set)", "",
         "The held-out set was scored three times. Each change after v1 replaced the threshold-calibration "
         "method because of a flaw visible in the thresholds themselves (0.85-0.94 from an in-style line split "
         "in v1; swinging 0.24-0.95 between splits in v2). No threshold or hyper-parameter was fitted to "
         "held-out lines, and the encoder body is the same in v2 and v3. But the v2 -> v3 switch was prompted "
         "by v2's held-out drop, so the held-out set has influenced which method was kept: treat its scores as "
         "slightly optimistic. The hand-written set and S6's real LLM-buyer messages are the independent checks.",
         "", "| Version | encoder intent | encoder claims+attacks | linear intent | linear claims+attacks | "
         "hand-written all-tag (encoder) |", "|---|---|---|---|---|---|"]
    for o, why in rows:
        on, li = o["backends"].get("onnx", {}), o["backends"].get("linear", {})
        hw = on.get("handwritten_per_tag")
        hw_all = statistics.mean(v[2] for v in hw.values() if v[3] > 0) if hw else None
        L.append(f"| {why} | {fmt(on.get('heldout_groups', {}).get('intent'))} | "
                 f"{fmt(on.get('heldout_groups', {}).get('claims+attacks'))} | "
                 f"{fmt(li.get('heldout_groups', {}).get('intent'))} | "
                 f"{fmt(li.get('heldout_groups', {}).get('claims+attacks'))} | {fmt(hw_all)} |")
    return L


if __name__ == "__main__":
    sys.exit(main())
