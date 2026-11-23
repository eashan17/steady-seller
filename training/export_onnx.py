"""Export the trained SetFit model to one int8 ONNX graph plus tokenizer files (S3 / T04).

    <venv>/python -X utf8 training/export_onnx.py [--src training/artifacts/setfit]

The graph is the whole classifier: BERT body -> mean pooling -> (L2 normalise, if the sentence-transformer
does) -> the one-vs-rest logistic-regression head -> sigmoid. Inputs input_ids / attention_mask /
token_type_ids (int64, [batch, seq]); output probs [batch, 18] in TAGS order.

Writes to models/: tags_encoder.int8.onnx, vocab.txt, tags_meta.json (tags, thresholds, max_tokens).
Thresholds come from training (grouped out-of-fold calibration, thresholds.json); int8 drift against the
fp32 graph is measured on valid.jsonl and reported (--recalibrate-valid re-fits them on valid instead). Also checks that the pure-Python WordPiece in
negotiator/textsignals.py tokenises every corpus text exactly like the HF tokenizer.
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, MODELS, TAGS, TRAINING, load_textsignals, read_jsonl  # noqa: E402
from train_setfit import best_thresholds, multi_hot  # noqa: E402


def head_weights(head, dim):
    """(W [n_tags, dim], b [n_tags]) from a fitted OneVsRestClassifier of LogisticRegression."""
    import numpy as np
    W, b = np.zeros((len(TAGS), dim), dtype=np.float32), np.zeros(len(TAGS), dtype=np.float32)
    for j, est in enumerate(head.estimators_):
        if hasattr(est, "coef_"):
            W[j], b[j] = est.coef_[0], est.intercept_[0]
        else:  # sklearn _ConstantPredictor: the tag had a single class in training
            b[j] = 20.0 if float(np.ravel(est.y_)[0]) > 0.5 else -20.0
    return W, b


def vector(ts, text):
    """Probabilities in TAGS order from one runtime call."""
    p = ts.tags(text)
    return [p[t] for t in TAGS]


def head_nodes(path):
    """MatMul/Gemm nodes whose weight has an 18-wide axis: the tag head, kept in fp32 (it is tiny and the
    sigmoid right after it amplifies quantisation error)."""
    import onnx
    from onnx import numpy_helper
    m = onnx.load(str(path))
    shapes = {i.name: numpy_helper.to_array(i).shape for i in m.graph.initializer}
    return [n.name for n in m.graph.node if n.op_type in ("MatMul", "Gemm")
            and any(len(TAGS) in shapes.get(x, ()) for x in n.input)]


def drift(fp32_path, ts, texts, thresholds):
    """int8 runtime vs the fp32 graph on the same token ids: mean/max |dp| and decision agreement."""
    import numpy as np
    import onnxruntime as ort
    sess = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    wp = ts.WordPiece(MODELS / "vocab.txt")
    diffs, agree = [], []
    for t in texts:
        ids = np.array([wp.encode(ts._coerce(t), 128)], dtype=np.int64)
        p32 = sess.run(None, {"input_ids": ids, "attention_mask": np.ones_like(ids),
                              "token_type_ids": np.zeros_like(ids)})[0][0]
        tg = ts.tags(t)
        p8 = np.array([tg[k] for k in TAGS])
        diffs.append(np.abs(p8 - p32))
        agree.append((p8 >= thresholds) == (p32 >= thresholds))
    d, g = np.array(diffs), np.array(agree)
    return {"mean_abs_dp": round(float(d.mean()), 4), "max_abs_dp": round(float(d.max()), 4),
            "decision_agreement": round(float(g.mean()), 4)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=TRAINING / "artifacts" / "setfit")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--per-channel", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--recalibrate-valid", action="store_true")
    a = ap.parse_args(argv)

    import numpy as np
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from setfit import SetFitModel

    model = SetFitModel.from_pretrained(str(a.src / "model"))
    body = model.model_body
    mods = [type(m).__name__ for m in body]
    hf, tok = body[0].auto_model.eval(), body[0].tokenizer
    pooling = body[1]
    mean = getattr(pooling, "pooling_mode", None) == "mean" or getattr(pooling, "pooling_mode_mean_tokens", False)
    assert mean, f"expected mean pooling, got {pooling}"
    normalize = "Normalize" in mods
    W, b = head_weights(model.model_head, hf.config.hidden_size)

    class Graph(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.hf = hf
            self.W = torch.nn.Parameter(torch.from_numpy(W), requires_grad=False)
            self.b = torch.nn.Parameter(torch.from_numpy(b), requires_grad=False)

        def forward(self, input_ids, attention_mask, token_type_ids):
            h = self.hf(input_ids=input_ids, attention_mask=attention_mask,
                        token_type_ids=token_type_ids).last_hidden_state
            m = attention_mask.unsqueeze(-1).to(h.dtype)
            e = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
            if normalize:
                e = torch.nn.functional.normalize(e, p=2, dim=1)
            return torch.sigmoid(e @ self.W.T + self.b)

    g = Graph().eval()
    tmp = a.src / "onnx"
    tmp.mkdir(parents=True, exist_ok=True)
    fp32 = tmp / "tags_encoder.fp32.onnx"
    enc = tok(["hello there, can you do 4,500?"], return_tensors="pt")
    feed = (enc["input_ids"], enc["attention_mask"], enc.get("token_type_ids", torch.zeros_like(enc["input_ids"])))
    names = ["input_ids", "attention_mask", "token_type_ids"]
    axes = {n: {0: "batch", 1: "seq"} for n in names}
    axes["probs"] = {0: "batch"}
    with torch.no_grad():
        ref = g(*feed).numpy()
        try:
            torch.onnx.export(g, feed, str(fp32), input_names=names, output_names=["probs"],
                              dynamic_axes=axes, opset_version=a.opset, dynamo=False)
        except TypeError:  # older torch without the dynamo flag
            torch.onnx.export(g, feed, str(fp32), input_names=names, output_names=["probs"],
                              dynamic_axes=axes, opset_version=a.opset)

    MODELS.mkdir(parents=True, exist_ok=True)
    int8 = MODELS / "tags_encoder.int8.onnx"
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8, per_channel=a.per_channel,
                     nodes_to_exclude=head_nodes(fp32))
    vocab = Path(tok.vocab_file) if getattr(tok, "vocab_file", None) else None
    if vocab is not None and vocab.exists():
        shutil.copyfile(vocab, MODELS / "vocab.txt")
    else:  # fast tokenizers may ship only tokenizer.json: write the id-ordered vocabulary ourselves
        by_id = sorted(tok.get_vocab().items(), key=lambda kv: kv[1])
        assert [i for _, i in by_id] == list(range(len(by_id))), "vocabulary ids are not contiguous"
        (MODELS / "vocab.txt").write_text("".join(t + "\n" for t, _ in by_id), encoding="utf-8")

    # tokenizer parity: pure-Python WordPiece vs the HF tokenizer, on every corpus text
    ts = load_textsignals()
    wp = ts.WordPiece(MODELS / "vocab.txt")
    texts = [r["text"] for f in ("train", "valid", "heldout", "unlabelled") for r in read_jsonl(DATA / f"{f}.jsonl")]
    mism = []
    for t in texts:
        mine = wp.encode(ts._coerce(t), 128)
        theirs = tok(ts._coerce(t), truncation=True, max_length=128)["input_ids"]
        if mine != theirs:
            mism.append(t)
    print(f"tokenizer parity: {len(texts) - len(mism)}/{len(texts)} identical", flush=True)
    for t in mism[:5]:
        print("  MISMATCH:", repr(t[:120]))

    # provisional meta (thresholds 0.5) so the runtime can load the graph, then calibrate from its outputs
    meta = {"tags": list(TAGS), "thresholds": [0.5] * len(TAGS), "max_tokens": 128,
            "encoder": json.loads((a.src / "train_report.json").read_text(encoding="utf-8"))["model"],
            "normalize": normalize, "modules": mods}
    (MODELS / "tags_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    ts.reset("onnx")
    assert ts.backend() == "onnx", "runtime could not load the exported graph"
    probe = ts.tags("hello there, can you do 4,500?")
    got = [probe[t] for t in TAGS]
    print(f"int8 vs fp32 torch on the probe: max |dp| = {float(np.max(np.abs(np.array(got) - ref[0]))):.4f}")

    valid = read_jsonl(DATA / "valid.jsonl")
    t0 = time.time()
    if a.recalibrate_valid:
        meta["thresholds"] = best_thresholds([vector(ts, r["text"]) for r in valid], multi_hot(valid))
    else:
        trained = json.loads((a.src / "thresholds.json").read_text(encoding="utf-8"))
        meta["thresholds"] = [float(trained[t]) for t in TAGS]
    (MODELS / "tags_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"thresholds set ({'valid re-fit' if a.recalibrate_valid else 'from training'}) in {time.time() - t0:.1f}s")
    dr = drift(fp32, ts, [r["text"] for r in valid], np.array(meta["thresholds"]))
    print(f"int8 vs fp32 on valid: {dr}")
    print(f"sizes: fp32 {fp32.stat().st_size / 1e6:.1f} MB, int8 {int8.stat().st_size / 1e6:.1f} MB, "
          f"vocab {(MODELS / 'vocab.txt').stat().st_size / 1e6:.2f} MB")
    report = {"tokenizer_parity": [len(texts) - len(mism), len(texts)], "int8_mb": int8.stat().st_size / 1e6,
              "fp32_mb": fp32.stat().st_size / 1e6, "int8_vs_fp32_valid": dr,
              "per_channel": a.per_channel, "fp32_nodes": head_nodes(fp32), "thresholds": dict(zip(TAGS, meta["thresholds"]))}
    (a.src / "export_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
