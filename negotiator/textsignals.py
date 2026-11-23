"""Buyer-text signals for the seller: tag probabilities and numbers from one buyer message.

    from negotiator import textsignals          # or load this file by path; it imports standalone
    textsignals.tags("My budget is 4k, final.")  # {"accept_signal": 0.01, ..., "claim_budget": 0.97, ...}
    textsignals.signals(text)                    # tags + active tags + numbers + backend name

Backends, chosen once at first use (plan D5):
  onnx    a MiniLM encoder + multi-label head, int8 ONNX (models/tags_encoder.int8.onnx), used only when
          `onnxruntime` imports and the model files are present.
  linear  a character n-gram linear model distilled from the encoder (models/linear_tags.json), pure Python.
  none    no model files at all: every probability is 0.0.
Set TEXTSIGNALS_BACKEND=onnx|linear|none to force one (auto is the default).

Guarantees: standard library only (plus optional onnxruntime/numpy); no public function raises; input of any
type or size is accepted (non-strings are JSON-encoded, text is cut to MAX_CHARS).

Tags are evidence for the buyer model, never price levers (plan D4). Definitions: training/TAGS.md.
"""
import json
import math
import os
import re
import time
import unicodedata
from pathlib import Path

TAGS = (
    # intent
    "accept_signal", "price_objection", "walk_threat", "bundle_request", "scope_change", "info_request",
    "urgency",
    # claims
    "claim_budget", "claim_competitor", "claim_final", "claim_seller_commitment", "claim_hardship",
    # attacks
    "attack_injection", "attack_authority", "attack_threat", "attack_extraction",
    # tone
    "tone_friendly", "tone_hostile",
)
GROUPS = {
    "intent": TAGS[0:7],
    "claims": TAGS[7:12],
    "attacks": TAGS[12:16],
    "tone": TAGS[16:18],
}

MAX_CHARS = 2000          # longer text is cut; keeps latency bounded for huge inputs
MAX_TOKENS = 128          # encoder sequence length, [CLS] and [SEP] included
MODELS_DIR = Path(os.environ.get("TEXTSIGNALS_MODELS") or Path(__file__).resolve().parent.parent / "models")
ONNX_FILE, VOCAB_FILE, META_FILE, LINEAR_FILE = (
    "tags_encoder.int8.onnx", "vocab.txt", "tags_meta.json", "linear_tags.json")

stats = {"backend": None, "calls": 0, "errors": 0, "total_ms": 0.0, "max_ms": 0.0}


# ---------------------------------------------------------------- input


def _coerce(text):
    """Any input -> str of at most MAX_CHARS. Never raises."""
    try:
        if text is None:
            s = ""
        elif isinstance(text, str):
            s = text
        else:
            s = json.dumps(text, ensure_ascii=False, default=str)
    except Exception:
        try:
            s = str(text)
        except Exception:
            s = ""
    return s[:MAX_CHARS]


# ---------------------------------------------------------------- linear fallback features
# training/distill_linear.py imports this function, so training and runtime features are identical.

_DIGITS = re.compile(r"\d+")
CHAR_NGRAMS = (2, 3, 4, 5)


def features(text):
    """Sparse feature vector {name: value}: words, word bigrams and per-word character n-grams.

    Text is NFKC-normalised, lower-cased, digits collapsed to '0'. Counts are sublinear (1 + log c) and the
    vector is L2-normalised.
    """
    s = _DIGITS.sub("0", unicodedata.normalize("NFKC", _coerce(text)).lower())
    words = s.split()
    counts = {}

    def add(key):
        counts[key] = counts.get(key, 0) + 1

    prev = "<s>"
    for w in words:
        add("w:" + w)
        add("b:" + prev + " " + w)
        prev = w
        padded = " " + w + " "
        for n in CHAR_NGRAMS:
            if len(padded) < n:
                break
            for i in range(len(padded) - n + 1):
                add("c:" + padded[i:i + n])
    if words:
        add("b:" + prev + " </s>")
    if not counts:
        return {}
    vec = {k: 1.0 + math.log(c) for k, c in counts.items()}
    norm = math.sqrt(sum(v * v for v in vec.values()))
    return {k: v / norm for k, v in vec.items()}


def _sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


class _LinearBackend:
    name = "linear"

    def __init__(self, path):
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
        if list(m["tags"]) != list(TAGS):
            raise ValueError("linear model tag order does not match TAGS")
        self.bias = [float(b) for b in m["bias"]]
        scale = float(m.get("scale", 1.0))
        # weights: {feature: [tag_idx, w_int, tag_idx, w_int, ...]}
        self.weights = {f: [(row[i], row[i + 1] * scale) for i in range(0, len(row), 2)]
                        for f, row in m["weights"].items()}
        self.thresholds = [float(t) for t in m["thresholds"]]

    def probs(self, text):
        z = list(self.bias)
        w = self.weights
        for feat, x in features(text).items():
            row = w.get(feat)
            if row:
                for j, wj in row:
                    z[j] += x * wj
        return [_sigmoid(v) for v in z]


# ---------------------------------------------------------------- BERT WordPiece tokenizer (pure Python)


def _is_whitespace(ch):
    return ch in " \t\n\r" or unicodedata.category(ch) == "Zs"


def _is_control(ch):
    return ch not in "\t\n\r" and unicodedata.category(ch).startswith("C")


def _is_punctuation(ch):
    cp = ord(ch)
    if 33 <= cp <= 47 or 58 <= cp <= 64 or 91 <= cp <= 96 or 123 <= cp <= 126:
        return True
    return unicodedata.category(ch).startswith("P")


def _is_cjk(cp):
    return (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or 0x20000 <= cp <= 0x2A6DF
            or 0x2A700 <= cp <= 0x2B73F or 0x2B740 <= cp <= 0x2B81F or 0x2B820 <= cp <= 0x2CEAF
            or 0xF900 <= cp <= 0xFAFF or 0x2F800 <= cp <= 0x2FA1F)


class WordPiece:
    """Uncased BERT tokenizer, equivalent to HF BertTokenizer(do_lower_case=True) for MiniLM's vocab."""

    def __init__(self, vocab_path):
        with open(vocab_path, encoding="utf-8") as f:
            self.vocab = {line.rstrip("\n"): i for i, line in enumerate(f)}
        self.unk, self.cls, self.sep = self.vocab["[UNK]"], self.vocab["[CLS]"], self.vocab["[SEP]"]

    @staticmethod
    def basic(text):
        out = []
        for ch in text:
            cp = ord(ch)
            if cp == 0 or cp == 0xFFFD or _is_control(ch):
                continue
            if _is_whitespace(ch):
                out.append(" ")
            elif _is_cjk(cp):
                out.append(" " + ch + " ")
            else:
                out.append(ch)
        text = unicodedata.normalize("NFD", "".join(out).lower())
        text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
        tokens = []
        for word in text.split():
            cur = []
            for ch in word:
                if _is_punctuation(ch):
                    if cur:
                        tokens.append("".join(cur))
                        cur = []
                    tokens.append(ch)
                else:
                    cur.append(ch)
            if cur:
                tokens.append("".join(cur))
        return tokens

    def wordpiece(self, word):
        if len(word) > 100:
            return [self.unk]
        ids, start = [], 0
        while start < len(word):
            end, found = len(word), None
            while start < end:
                piece = word[start:end] if start == 0 else "##" + word[start:end]
                if piece in self.vocab:
                    found = self.vocab[piece]
                    break
                end -= 1
            if found is None:
                return [self.unk]
            ids.append(found)
            start = end
        return ids

    def encode(self, text, max_len=MAX_TOKENS):
        ids = []
        for word in self.basic(text):
            ids.extend(self.wordpiece(word))
            if len(ids) >= max_len - 2:
                break
        return [self.cls] + ids[:max_len - 2] + [self.sep]


class _OnnxBackend:
    name = "onnx"

    def __init__(self, models_dir, meta):
        import numpy as np
        import onnxruntime as ort
        self.np = np
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(os.environ.get("TEXTSIGNALS_THREADS", "1"))
        so.inter_op_num_threads = 1
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(str(models_dir / ONNX_FILE), so, providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}
        self.tok = WordPiece(models_dir / VOCAB_FILE)
        self.max_len = int(meta.get("max_tokens", MAX_TOKENS))
        if list(meta["tags"]) != list(TAGS):
            raise ValueError("encoder tag order does not match TAGS")
        self.thresholds = [float(t) for t in meta["thresholds"]]

    def probs(self, text):
        np = self.np
        ids = np.array([self.tok.encode(text, self.max_len)], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": np.ones_like(ids), "token_type_ids": np.zeros_like(ids)}
        out = self.sess.run(None, {k: v for k, v in feed.items() if k in self.inputs})[0][0]
        return [float(x) for x in out]


class _NoBackend:
    name = "none"
    thresholds = [1.0] * len(TAGS)

    def probs(self, text):
        return [0.0] * len(TAGS)


# ---------------------------------------------------------------- backend selection

_backend = None
_forced = None        # set by reset(choice); overrides TEXTSIGNALS_BACKEND
_linear = None          # kept as the per-call fallback when the encoder errors
_MAX_ENCODER_ERRORS = 3


def _load_linear():
    global _linear
    if _linear is None:
        path = MODELS_DIR / LINEAR_FILE
        _linear = _LinearBackend(path) if path.exists() else _NoBackend()
    return _linear


def _choose(choice):
    if choice in ("auto", "onnx"):
        try:
            with open(MODELS_DIR / META_FILE, encoding="utf-8") as f:
                return _OnnxBackend(MODELS_DIR, json.load(f))
        except Exception:
            if choice == "onnx":
                stats["errors"] += 1
    if choice in ("auto", "onnx", "linear"):
        try:
            return _load_linear()
        except Exception:
            stats["errors"] += 1
    return _NoBackend()


def _get_backend():
    global _backend
    if _backend is None:
        _backend = _choose((_forced or os.environ.get("TEXTSIGNALS_BACKEND", "auto")).strip().lower())
        stats["backend"] = _backend.name
    return _backend


def backend():
    """Name of the backend in use: 'onnx', 'linear' or 'none'. Never raises."""
    try:
        return _get_backend().name
    except Exception:
        return "none"


def reset(choice=None):
    """Forget the chosen backend (tests and ablations). `choice` ('auto', 'onnx', 'linear', 'none') forces one
    for this module only; None goes back to TEXTSIGNALS_BACKEND. The process environment is not modified."""
    global _backend, _linear, _forced
    _forced = choice
    _backend = _linear = None
    stats.update(backend=None, calls=0, errors=0, total_ms=0.0, max_ms=0.0)


# ---------------------------------------------------------------- public API


def _probs_and_thresholds(text):
    global _backend
    b = _get_backend()
    try:
        p = b.probs(text)
        if len(p) == len(TAGS) and all(0.0 <= x <= 1.0 for x in p):
            return p, b.thresholds
        raise ValueError("bad probability vector")
    except Exception:
        stats["errors"] += 1
        if b.name == "onnx" and stats["errors"] >= _MAX_ENCODER_ERRORS:
            _backend = _load_linear()
            stats["backend"] = _backend.name
        try:
            fb = _load_linear()
            return fb.probs(text), fb.thresholds
        except Exception:
            return [0.0] * len(TAGS), [1.0] * len(TAGS)


def tags(text):
    """{tag: probability} for every tag in TAGS. Never raises."""
    return signals(text)["tags"]


def active(text):
    """Tags whose probability clears the backend's calibrated threshold. Never raises."""
    return signals(text)["active"]


def signals(text):
    """{"tags": {tag: p}, "active": [tag...], "numbers": {...}, "backend": name}. Never raises."""
    t0 = time.perf_counter()
    try:
        s = _coerce(text)
        p, th = _probs_and_thresholds(s)
        out = {"tags": {t: round(float(x), 4) for t, x in zip(TAGS, p)},
               "active": [t for t, x, h in zip(TAGS, p, th) if x >= h],
               "numbers": numbers(s),
               "backend": stats["backend"] or "none"}
    except Exception:
        stats["errors"] += 1
        out = {"tags": {t: 0.0 for t in TAGS}, "active": [], "numbers": _EMPTY_NUMBERS(), "backend": "none"}
    ms = (time.perf_counter() - t0) * 1000.0
    stats["calls"] += 1
    stats["total_ms"] += ms
    stats["max_ms"] = max(stats["max_ms"], ms)
    return out


# ---------------------------------------------------------------- numbers by regex

_RUPEE = r"(?:₹|rs\.?|inr|rupees?)"
_NUM = r"(?P<int>\d{1,3}(?:,\d{2,3})+|\d+)(?:\.(?P<frac>\d+))?"
_AMOUNT_RE = re.compile(
    r"(?<![a-z0-9])(?P<cur>" + _RUPEE + r")?\s*" + _NUM + r"\s*(?P<mult>k\b|lakh|lac|thousand)?"
    r"(?P<pct>\s*(?:%|percent))?(?P<after>\s*(?:/-|rupees?|rs\b|inr\b))?",
    re.IGNORECASE)
_MULT = {"k": 1000, "thousand": 1000, "lakh": 100000, "lac": 100000}


def _EMPTY_NUMBERS():
    return {"rupees": [], "percents": [], "numbers": []}


def numbers(text):
    """Numbers in the text: {"rupees": [...], "percents": [...], "numbers": [...all values]}.

    'rupees' are values marked as money (₹, Rs, INR, rupees, /-) or written with a k/lakh multiplier.
    Digits glued to letters (SKU codes like S001) are skipped. Never raises.
    """
    out = _EMPTY_NUMBERS()
    try:
        for m in _AMOUNT_RE.finditer(_coerce(text)):
            whole = m.group("int").replace(",", "")
            value = float(whole + ("." + m.group("frac") if m.group("frac") else ""))
            mult = (m.group("mult") or "").lower()
            if mult:
                value *= _MULT[mult]
            value = int(value) if value == int(value) else round(value, 2)
            out["numbers"].append(value)
            if m.group("pct"):
                out["percents"].append(value)
            elif m.group("cur") or m.group("after") or mult:
                out["rupees"].append(value)
    except Exception:
        stats["errors"] += 1
    return out
