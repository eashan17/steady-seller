"""Train the multi-label SetFit tag classifier and calibrate one threshold per tag (S3 / T03).

    <venv>/python -X utf8 training/train_setfit.py [--model sentence-transformers/all-MiniLM-L6-v2]
                                                   [--max-steps 1500] [--out training/artifacts/setfit]

Two stages:
  1. SetFit contrastive fine-tuning of the encoder body on train.jsonl.
  2. Head and thresholds (refit_head): embed train + valid with the fine-tuned body, run grouped K-fold
     cross-validation of the one-vs-rest logistic-regression head (groups = Codex job / kit template / CaSiNo
     dialogue), calibrate one threshold per tag on the out-of-fold probabilities of every line, then fit the
     final head on all of train + valid. Calibrating on a single validation split of 20-60 positives per tag
     from a few groups proved high-variance (thresholds swung 0.24-0.95 between splits).
  --refit-head-only reruns stage 2 on an existing --out model.

The held-out sets are never read here. Output folder: the SetFit model, thresholds.json, train_report.json.
export_onnx.py turns it into models/tags_encoder.int8.onnx.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, SEED, TAGS, TRAINING, read_jsonl  # noqa: E402


def multi_hot(rows):
    return [[1 if t in set(r["tags"]) else 0 for t in TAGS] for r in rows]


def best_thresholds(probs, y, grid=None, window=5):
    """Per tag, the threshold maximising F1 smoothed over +/- `window` grid steps (0.05) on (probs, y).

    Smoothing stops a few validation lines from pushing a threshold to an extreme; ties go to the threshold
    closer to 0.5. A tag with no positives keeps 0.5.
    """
    grid = grid or [i / 100 for i in range(5, 96)]
    out = []
    for j in range(len(TAGS)):
        pos = sum(row[j] for row in y)
        if pos == 0:
            out.append(0.5)
            continue
        f1s = []
        for th in grid:
            tp = sum(1 for p, t in zip(probs, y) if p[j] >= th and t[j])
            fp = sum(1 for p, t in zip(probs, y) if p[j] >= th and not t[j])
            f1s.append(2 * tp / (2 * tp + fp + (pos - tp)) if tp else 0.0)
        smooth = [sum(f1s[max(0, i - window):i + window + 1]) / len(f1s[max(0, i - window):i + window + 1])
                  for i in range(len(grid))]
        best = max(range(len(grid)), key=lambda i: (round(smooth[i], 9), -abs(grid[i] - 0.5)))
        out.append(grid[best])
    return out


def refit_head(model, rows, folds=5):
    """Grouped K-fold head CV -> out-of-fold thresholds; then the final head on all rows. Returns thresholds."""
    import numpy as np
    from sklearn.base import clone
    from sklearn.model_selection import GroupKFold

    from build_dataset import group_of
    X = np.asarray(model.model_body.encode([r["text"] for r in rows], convert_to_numpy=True, batch_size=64))
    Y = np.array(multi_hot(rows))
    groups = [group_of(r) for r in rows]
    oof = np.zeros(Y.shape, dtype=float)
    for k, (tr, te) in enumerate(GroupKFold(n_splits=folds).split(X, Y, groups)):
        head = clone(model.model_head).fit(X[tr], Y[tr])
        oof[te] = head.predict_proba(X[te])
        print(f"  head fold {k + 1}/{folds}: fit {len(tr)}, scored {len(te)}", flush=True)
    th = best_thresholds(oof.tolist(), Y.tolist())
    model.model_head = clone(model.model_head).fit(X, Y)
    return th


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--head-c", type=float, default=4.0, help="inverse regularisation of the LR head")
    ap.add_argument("--out", type=Path, default=TRAINING / "artifacts" / "setfit")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--refit-head-only", action="store_true", help="skip stage 1; reuse --out/model")
    a = ap.parse_args(argv)

    from datasets import Dataset
    from setfit import SetFitModel, Trainer, TrainingArguments

    train, valid = read_jsonl(DATA / "train.jsonl"), read_jsonl(DATA / "valid.jsonl")
    print(f"train {len(train)}  valid {len(valid)}  model {a.model}", flush=True)
    if a.refit_head_only:
        model = SetFitModel.from_pretrained(str(a.out / "model"))
        report = json.loads((a.out / "train_report.json").read_text(encoding="utf-8"))
    else:
        ds = Dataset.from_dict({"text": [r["text"] for r in train], "label": multi_hot(train)})
        model = SetFitModel.from_pretrained(
            a.model, multi_target_strategy="one-vs-rest",
            head_params={"C": a.head_c, "max_iter": 3000, "class_weight": "balanced"})
        args = TrainingArguments(output_dir=str(a.out / "checkpoints"), batch_size=a.batch_size, num_epochs=1,
                                 max_steps=a.max_steps, max_length=128, seed=SEED, report_to="none",
                                 save_strategy="no", show_progress_bar=False, logging_steps=100)
        t0 = time.time()
        Trainer(model=model, args=args, train_dataset=ds).train()
        report = {"model": a.model, "train": len(train), "valid": len(valid), "max_steps": a.max_steps,
                  "batch_size": a.batch_size, "head_c": a.head_c, "train_seconds": round(time.time() - t0, 1)}

    th = refit_head(model, train + valid, a.folds)
    a.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(a.out / "model"))
    (a.out / "thresholds.json").write_text(json.dumps(dict(zip(TAGS, th)), indent=1), encoding="utf-8")
    report.update(head="one-vs-rest LR refit on train+valid", calibration=f"grouped {a.folds}-fold out-of-fold",
                  thresholds=dict(zip(TAGS, th)))
    (a.out / "train_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
