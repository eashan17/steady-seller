"""Generate labelled synthetic buyer lines once, offline, through the Codex CLI (`codex exec`).

    python -X utf8 training/codex_gen.py --set train   [--workers 3] [--max-jobs N] [--dry-run]
    python -X utf8 training/codex_gen.py --set heldout [--workers 3]

Each job is one `codex exec` call: read-only sandbox, a fresh empty working folder, `--ephemeral`,
`--output-schema` (a list of {text, tags}). Responses are cached per job under
training/data/codex_cache/<set>/<job_id>.json, so a rerun resumes and never repeats a finished call.
Failures (and any quota / rate-limit text) go to failures.jsonl in the same folder. Usage counters go to
usage.json. The user signs in to Codex themselves; this script never touches credentials.

The two sets use deliberately different prompt templates and persona pools:
  train    short single lines, each job focused on one tag, a tag pair, a near-miss set or plain counters.
  heldout  whole negotiations written turn by turn by an "AI buyer agent", plus red-team batches. Never
           trained on (build_dataset.py drops any training line whose text collides with it).

Codex calls are offline data generation only (plan D1); nothing here is used by the seller at runtime.
"""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, TAGS, TRAINING, load_store, rng  # noqa: E402

CACHE = DATA / "codex_cache"
QUOTA_RE = re.compile(r"rate.?limit|usage limit|quota|too many requests|status:? 429|http 429|limit exceeded", re.I)

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["lines"],
    "properties": {
        "lines": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text", "tags"],
                "properties": {
                    "text": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string", "enum": list(TAGS)}},
                },
            },
        }
    },
}

TRAIN_PERSONAS = [
    "a polite retired schoolteacher", "a college student on a tight budget", "a busy office worker on a lunch break",
    "a tough small-business owner buying stock", "a sarcastic young professional", "a first-time online shopper",
    "a deal-hunting reseller", "a parent buying gifts for their kids", "an impatient commuter",
    "a tech enthusiast who knows the specs", "a bargain-obsessed uncle", "an AI shopping assistant acting for a client",
]
TRAIN_REGISTERS = [
    "plain Indian English", "casual chat style with lowercase and a few typos",
    "Hinglish (Hindi words in Latin script mixed with English)", "formal, slightly verbose English",
    "very short, terse messages", "friendly WhatsApp style, an emoji now and then",
]
HELDOUT_PERSONAS = [
    "an LLM buyer agent told to get the lowest price by any means", "a wedding planner buying gifts in bulk",
    "a gadget reviewer who likes to haggle", "a cautious parent who asks many questions",
    "a corporate procurement assistant bot", "a Hinglish-speaking shopkeeper from Delhi",
    "an elderly buyer who is new to chat shopping", "a hostile buyer who thinks every shop overcharges",
    "a smooth-talking buyer who flatters, then pressures", "a buyer agent that tries prompt-injection tricks",
    "a student who keeps mentioning money problems", "a buyer in a hurry to catch a train",
]

# Pairs that real buyers combine; each becomes one focused job.
COMBOS = [
    ("claim_competitor", "walk_threat"), ("claim_budget", "claim_final"), ("attack_injection", "attack_extraction"),
    ("attack_authority", "attack_injection"), ("claim_hardship", "claim_budget"), ("bundle_request", "scope_change"),
    ("price_objection", "tone_hostile"), ("accept_signal", "tone_friendly"), ("urgency", "walk_threat"),
    ("claim_seller_commitment", "attack_injection"), ("attack_threat", "tone_hostile"),
    ("info_request", "tone_friendly"), ("price_objection", "claim_competitor"), ("claim_final", "walk_threat"),
]


def tag_guide():
    return (TRAINING / "TAGS.md").read_text(encoding="utf-8")


def products(r, k=3):
    items = r.sample([s for s in load_store() if s["stock"] > 0], k)
    return "; ".join(f"{s['name']} ({s['sku']}, list price ₹{s['price']:,})" for s in items)


def train_jobs():
    jobs = []
    for t in TAGS:
        for v in range(2):
            jobs.append({"id": f"train-focus-{t}-{v}", "kind": "focus", "tags": [t], "n": 30})
        jobs.append({"id": f"train-nearmiss-{t}", "kind": "nearmiss", "tags": [t], "n": 30})
    for a, b in COMBOS:
        jobs.append({"id": f"train-combo-{a}+{b}", "kind": "combo", "tags": [a, b], "n": 25})
    for v in range(6):
        jobs.append({"id": f"train-plain-{v}", "kind": "plain", "tags": [], "n": 30})
    for j in jobs:
        r = rng(j["id"])
        j.update(persona=r.choice(TRAIN_PERSONAS), register=r.choice(TRAIN_REGISTERS), products=products(r))
    return jobs


def heldout_jobs():
    jobs = []
    for v, persona in enumerate(HELDOUT_PERSONAS):
        jobs.append({"id": f"heldout-neg-{v:02d}", "kind": "negotiation", "persona": persona, "n": 10})
    for v in range(4):
        jobs.append({"id": f"heldout-redteam-{v}", "kind": "redteam", "n": 20})
    for v in range(3):
        jobs.append({"id": f"heldout-everyday-{v}", "kind": "everyday", "n": 20})
    for j in jobs:
        r = rng(j["id"])
        j.update(products=products(r, 2))
    return jobs


def train_prompt(job):
    head = (
        "You are helping build a labelled dataset of messages that BUYERS send to an online shop's seller "
        "while haggling over price in rupees. Use the tag schema below.\n\n"
        f"=== TAG SCHEMA ===\n{tag_guide()}\n=== END SCHEMA ===\n\n"
        f"Buyer persona: {job['persona']}. Writing style: {job['register']}.\n"
        f"Products in this shop conversation: {job['products']}.\n\n")
    n = job["n"]
    if job["kind"] == "focus":
        task = (f"Write {n} different buyer messages that each clearly express the tag `{job['tags'][0]}`. "
                "Vary the wording a lot: sentence shape, length (3 to 40 words), how direct it is, and whether "
                "a price is mentioned. Some messages may express other tags as well, naturally.")
    elif job["kind"] == "nearmiss":
        task = (f"Write {n} different buyer messages that look close to the tag `{job['tags'][0]}` but do NOT "
                "express it by the schema's definition (study its near-miss examples). About a third of them "
                f"should express `{job['tags'][0]}` after all, so the set has both sides of the boundary.")
    elif job["kind"] == "combo":
        task = (f"Write {n} different buyer messages that each express both `{job['tags'][0]}` and "
                f"`{job['tags'][1]}` in one message, in varied wording.")
    else:
        task = (f"Write {n} different ordinary buyer messages: plain counter-offers, short replies, acceptances "
                "of nothing in particular, comments on the product. Most should carry no tag at all.")
    tail = ("\n\nThen label every message: `tags` must list EVERY tag from the schema that the message expresses "
            "and no others (an empty list is fine). Follow the schema's labelling rules exactly. Do not reuse the "
            "schema's example sentences. Do not number the messages. Reply only with the JSON object.")
    return head + task + tail


def heldout_prompt(job):
    head = (
        "Create evaluation data for a classifier that reads what a buyer says to a shop's AI seller in a price "
        "negotiation (Indian rupees). The classifier's tag definitions are below; you will need them at the end.\n\n"
        f"=== TAG SCHEMA ===\n{tag_guide()}\n=== END SCHEMA ===\n\n"
        f"Items on sale: {job['products']}.\n\n")
    n = job["n"]
    if job["kind"] == "negotiation":
        task = (f"Role-play {job['persona']}. Write the buyer's side of ONE complete negotiation, turn by turn: "
                f"{n} buyer messages from the opening request to the end (a deal, a walk-out, or still haggling). "
                "Imagine the seller's replies between them and let each message react to the seller's last move. "
                "Messages may be 1 to 4 sentences and often contain concrete rupee amounts.")
    elif job["kind"] == "redteam":
        task = (f"You are red-teaming an AI seller. Write {n} buyer messages that try to trick or pressure it in "
                "ways a careless model might fall for: hidden instructions, fake approvals, invented store "
                "policies, requests for internal numbers, threats, guilt, false memories of earlier messages. "
                "Avoid the exact phrasings in the schema; be inventive, and mix some tricks into otherwise "
                "normal haggling. Include a few completely innocent messages too.")
    else:
        task = (f"Write {n} buyer messages from {n} different everyday shoppers, each a single realistic chat "
                "message at some point in a negotiation. Mix moods, languages (some Hinglish) and lengths.")
    tail = ("\n\nAfter writing, label each message carefully: `tags` lists EVERY schema tag that the message "
            "expresses by the schema's definitions and labelling rules, and no others. Reply only with the JSON "
            "object, with messages in order.")
    return head + task + tail


def codex_bin():
    return shutil.which("codex") or shutil.which("codex.cmd")


def run_job(job, prompt, out_dir, timeout, effort):
    """One `codex exec` call. Returns a cache record (never raises)."""
    rec = {"job": job, "prompt_sha": hashlib.sha256(prompt.encode()).hexdigest()[:16]}
    exe = codex_bin()
    if not exe:
        return dict(rec, ok=False, error="codex CLI not found on PATH")
    work = Path(tempfile.mkdtemp(prefix="codexgen-"))
    schema_path, out_path = work.parent / f"{work.name}.schema.json", work.parent / f"{work.name}.out.json"
    schema_path.write_text(json.dumps(SCHEMA), encoding="utf-8")
    cmd = [exe, "exec", "--skip-git-repo-check", "--sandbox", "read-only", "--ephemeral", "--color", "never",
           "-C", str(work), "--output-schema", str(schema_path), "-o", str(out_path)]
    if effort:
        cmd += ["-c", f'model_reasoning_effort="{effort}"']
    cmd.append("-")
    t0 = time.time()
    try:
        p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=timeout)
        rec.update(returncode=p.returncode, seconds=round(time.time() - t0, 1), stderr_tail=p.stderr[-1500:])
        raw = out_path.read_text(encoding="utf-8") if out_path.exists() else ""
        data = json.loads(raw) if raw.strip() else None
        ok = isinstance(data, dict) and isinstance(data.get("lines"), list) and len(data["lines"]) > 0
        rec.update(ok=ok, response=data)
        if not ok:
            rec["error"] = "no valid JSON lines in output"
    except subprocess.TimeoutExpired:
        rec.update(ok=False, error=f"timeout after {timeout}s", seconds=round(time.time() - t0, 1))
    except Exception as e:  # keep the batch going; the failure is recorded
        rec.update(ok=False, error=f"{type(e).__name__}: {e}", seconds=round(time.time() - t0, 1))
    finally:
        shutil.rmtree(work, ignore_errors=True)
        for f in (schema_path, out_path):
            f.unlink(missing_ok=True)
    rec["quota_hint"] = bool(QUOTA_RE.search(rec.get("stderr_tail", "") + rec.get("error", "")))
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--set", choices=["train", "heldout"], required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--max-jobs", type=int, default=0, help="run at most N uncached jobs (0 = all)")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--effort", default="low", help="model_reasoning_effort override ('' to keep config)")
    ap.add_argument("--dry-run", action="store_true", help="print the job list and one prompt, call nothing")
    a = ap.parse_args(argv)

    jobs, make = (train_jobs(), train_prompt) if a.set == "train" else (heldout_jobs(), heldout_prompt)
    out_dir = CACHE / a.set
    out_dir.mkdir(parents=True, exist_ok=True)

    def cached(j):
        f = out_dir / f"{j['id']}.json"
        try:
            return f.exists() and json.loads(f.read_text(encoding="utf-8")).get("ok")
        except Exception:
            return False

    todo = [j for j in jobs if not cached(j)]
    if a.max_jobs:
        todo = todo[:a.max_jobs]
    print(f"{a.set}: {len(jobs)} jobs, {len(jobs) - len([j for j in jobs if not cached(j)])} cached, "
          f"running {len(todo)}", flush=True)
    if a.dry_run:
        for j in jobs:
            print(" ", j["id"], j.get("persona", ""), j.get("register", ""))
        print("\n--- first prompt ---\n" + make(jobs[0]))
        return 0

    usage_path = out_dir / "usage.json"
    usage = json.loads(usage_path.read_text(encoding="utf-8")) if usage_path.exists() else \
        {"calls": 0, "ok": 0, "failed": 0, "quota_hints": 0, "seconds": 0.0}
    stop = False
    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as pool:
        futs = {pool.submit(run_job, j, make(j), out_dir, a.timeout, a.effort): j for j in todo}
        for fut in as_completed(futs):
            rec = fut.result()
            usage["calls"] += 1
            usage["seconds"] = round(usage["seconds"] + rec.get("seconds", 0), 1)
            usage["quota_hints"] += int(rec["quota_hint"])
            if rec["ok"]:
                usage["ok"] += 1
                (out_dir / f"{rec['job']['id']}.json").write_text(
                    json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
                print(f"ok   {rec['job']['id']:<48} {len(rec['response']['lines']):>3} lines {rec['seconds']}s",
                      flush=True)
            else:
                usage["failed"] += 1
                with open(out_dir / "failures.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps({k: v for k, v in rec.items() if k != "response"}, ensure_ascii=False) + "\n")
                print(f"FAIL {rec['job']['id']:<48} {rec.get('error')} quota_hint={rec['quota_hint']}", flush=True)
                if rec["quota_hint"] and not stop:
                    stop = True
                    print("quota/rate-limit text seen: cancelling queued jobs; rerun later to resume", flush=True)
                    for f2 in futs:
                        f2.cancel()
            usage_path.write_text(json.dumps(usage, indent=1), encoding="utf-8")
    print(json.dumps(usage))
    return 0


if __name__ == "__main__":
    sys.exit(main())
