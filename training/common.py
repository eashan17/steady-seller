"""Shared paths and helpers for the S3 training scripts (offline only; never imported by the seller).

The tag list has one source of truth: negotiator/textsignals.py. It is loaded here by file path so the
training scripts do not depend on negotiator/__init__.py (owned by slice S1).
"""
import csv
import hashlib
import importlib.util
import json
import random
from pathlib import Path

TRAINING = Path(__file__).resolve().parent
KIT = TRAINING.parent
DATA = TRAINING / "data"
MODELS = KIT / "models"
RESULTS = KIT.parent / ".sessions" / "ps2-seller-optimize-20261120" / "results"

SEED = 20261120


def load_textsignals():
    """Import negotiator/textsignals.py standalone (no package __init__)."""
    spec = importlib.util.spec_from_file_location("textsignals", KIT / "negotiator" / "textsignals.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TAGS = tuple(load_textsignals().TAGS)


def load_store():
    with open(KIT / "store.csv", newline="", encoding="utf-8") as f:
        return [dict(r, price=int(r["price"]), stock=int(r["stock"])) for r in csv.DictReader(f)]


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def norm_text(text):
    return " ".join(str(text).lower().split())


def text_key(text):
    """Dedup / leakage key: case- and whitespace-insensitive hash."""
    return hashlib.sha1(norm_text(text).encode("utf-8")).hexdigest()[:16]


def clean_tags(tags):
    """Keep known tags only, in canonical order."""
    s = set(tags or ())
    return [t for t in TAGS if t in s]


def rng(*parts):
    return random.Random(f"{SEED}:" + ":".join(map(str, parts)))
