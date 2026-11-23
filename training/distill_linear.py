"""Distil the ONNX encoder into a pure-Python character n-gram linear model (S3 / T05).

    <venv>/python -X utf8 training/distill_linear.py [--c 8] [--max-mb 4.5]

Teacher: the int8 encoder through negotiator/textsignals.py (backend "onnx"). Corpus: train + valid +
unlabelled (never the held-out set). For each tag, a logistic regression learns the teacher's SOFT probability
with the standard duplication trick (each line once as positive with weight p, once as negative with weight
1 - p). Features come from textsignals.features(), the same function the runtime uses.

Weights are magnitude-pruned until models/linear_tags.json fits --max-mb, stored as integers x scale.
Thresholds maximise F1 against the teacher's decisions on valid.jsonl, so the fallback mimics the encoder.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, MODELS, TAGS, load_textsignals, read_jsonl  # noqa: E402
from export_onnx import vector  # noqa: E402
from train_setfit import best_thresholds  # noqa: E402

SCALE = 0.001


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--c", type=float, default=8.0, help="inverse L2 regularisation")
    ap.add_argument("--min-df", type=int, default=2)
    ap.add_argument("--max-mb", type=float, default=4.5)
    a = ap.parse_args(argv)

    import numpy as np
    from scipy.sparse import csr_matrix, vstack
    from sklearn.linear_model import LogisticRegression

    ts = load_textsignals()
    ts.reset("onnx")
    assert ts.backend() == "onnx", "teacher (ONNX encoder) not available; run export_onnx.py first"

    train = read_jsonl(DATA / "train.jsonl")
    valid = read_jsonl(DATA / "valid.jsonl")
    unlab = read_jsonl(DATA / "unlabelled.jsonl")
    corpus = [r["text"] for r in train + unlab]
    t0 = time.time()
    teach = np.array([vector(ts, t) for t in corpus], dtype=np.float64)
    teach_valid = np.array([vector(ts, r["text"]) for r in valid], dtype=np.float64)
    enc_th = np.array(json.loads((MODELS / "tags_meta.json").read_text(encoding="utf-8"))["thresholds"])
    print(f"teacher labelled {len(corpus) + len(valid)} lines in {time.time() - t0:.0f}s", flush=True)

    feats = [ts.features(t) for t in corpus]
    df = {}
    for f in feats:
        for k in f:
            df[k] = df.get(k, 0) + 1
    vocab = {k: i for i, k in enumerate(sorted(k for k, c in df.items() if c >= a.min_df))}

    def matrix(rows):
        data, ind, ptr = [], [], [0]
        for f in rows:
            for k, v in f.items():
                j = vocab.get(k)
                if j is not None:
                    ind.append(j)
                    data.append(v)
            ptr.append(len(ind))
        return csr_matrix((data, ind, ptr), shape=(len(rows), len(vocab)))

    X = matrix(feats)
    X2 = vstack([X, X]).tocsr()
    print(f"features: {len(vocab)} (min_df {a.min_df}), lines {X.shape[0]}", flush=True)
    W = np.zeros((len(TAGS), len(vocab)))
    bias = np.zeros(len(TAGS))
    for j, tag in enumerate(TAGS):
        p = np.clip(teach[:, j], 1e-4, 1 - 1e-4)
        y = np.r_[np.ones(len(p)), np.zeros(len(p))]
        w = np.r_[p, 1 - p]
        clf = LogisticRegression(C=a.c, max_iter=2000, solver="lbfgs")
        clf.fit(X2, y, sample_weight=w)
        W[j], bias[j] = clf.coef_[0], clf.intercept_[0]

    # prune by magnitude until the JSON fits the budget
    order = np.sort(np.abs(W[W != 0]))[::-1]
    keep_n = len(order)
    while True:
        cut = order[keep_n - 1] if keep_n < len(order) else 0.0
        cut = max(cut, SCALE)
        weights = {}
        inv = {i: k for k, i in vocab.items()}
        rows, cols = np.nonzero(np.abs(W) >= cut)
        for t, f in zip(rows, cols):
            weights.setdefault(inv[f], []).extend([int(t), int(round(W[t, f] / SCALE))])
        blob = json.dumps({"weights": weights}, ensure_ascii=False, separators=(",", ":"))
        mb = len(blob.encode("utf-8")) / 1e6
        if mb <= a.max_mb:
            break
        keep_n = int(keep_n * 0.85)
    print(f"kept {sum(len(v) // 2 for v in weights.values())} weights on {len(weights)} features, {mb:.2f} MB")

    model = {"version": 1, "tags": list(TAGS), "scale": SCALE, "bias": [round(float(x), 5) for x in bias],
             "thresholds": [0.5] * len(TAGS), "weights": weights,
             "features": "textsignals.features: words, word bigrams, char 2-5 grams per word; sublinear, L2"}
    out = MODELS / "linear_tags.json"
    out.write_text(json.dumps(model, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    # thresholds: mimic the teacher's decisions on valid
    ts.reset("linear")
    assert ts.backend() == "linear"
    student = [vector(ts, r["text"]) for r in valid]
    teacher_dec = (teach_valid >= enc_th).astype(int).tolist()
    model["thresholds"] = best_thresholds(student, teacher_dec)
    out.write_text(json.dumps(model, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    ts.reset("linear")
    dec = np.array([[1 if k in set(ts.active(r["text"])) else 0 for k in TAGS] for r in valid])
    agree = float((dec == np.array(teacher_dec)).mean())
    print(f"valid element-wise agreement with the encoder: {agree:.4f}; file {out.stat().st_size / 1e6:.2f} MB")
    ts.reset("auto")
    return 0


if __name__ == "__main__":
    sys.exit(main())
