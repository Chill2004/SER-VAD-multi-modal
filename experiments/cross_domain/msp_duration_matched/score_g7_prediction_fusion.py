#!/usr/bin/env python3
"""Fit per-target scalar audio/text fusion on MSP validation and score the locked test."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

SHA = "888f6efa5ca9b305eb159b050775484f970a3f10568df775da85f123d889f608"
TARGETS = ("valence", "arousal", "dominance")
TRUES = [f"{t}_true_norm" for t in TARGETS]
PREDS = [f"{t}_pred_norm" for t in TARGETS]
PREFIXES = {"audio": "audio_fusion_model_level", "text": "text_embedding"}


def ccc(y, prediction):
    return 2 * np.mean((y - y.mean()) * (prediction - prediction.mean())) / (
        y.var() + prediction.var() + (y.mean() - prediction.mean()) ** 2 + 1e-12
    )


def aligned(run, manifest, split):
    expected = manifest.loc[manifest.split.eq(split)].set_index("source_utt_id", verify_integrity=True)
    labels = expected[[f"{t}_norm" for t in TARGETS]].to_numpy(float)
    outputs = {}
    for mode, prefix in PREFIXES.items():
        name = f"{prefix}_val_predictions.csv" if split == "val" else f"{prefix}_predictions.csv"
        file = run / mode / name
        frame = pd.read_csv(file).set_index("utterance_id", verify_integrity=True)
        if set(frame.index) != set(expected.index) or len(frame) != len(expected):
            raise ValueError(f"{file}: utterance IDs differ from locked {split} split")
        frame = frame.loc[expected.index]
        y = frame[TRUES].to_numpy(float)
        pred = frame[PREDS].to_numpy(float)
        if not np.allclose(y, labels, atol=1e-7, rtol=0) or not np.isfinite(pred).all():
            raise ValueError(f"{file}: invalid labels or predictions")
        outputs[mode] = pred
    return expected.index, labels, outputs["audio"], outputs["text"]


def run(manifest_path, directory):
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != SHA:
        raise ValueError("Input manifest differs from locked G7 split")
    manifest = pd.read_csv(manifest_path)
    if manifest.groupby("split").size().to_dict() != {"train": 7082, "val": 787, "test": 2170}:
        raise ValueError("Unexpected split sizes")
    _, y_val, audio_val, text_val = aligned(directory, manifest, "val")
    grid = np.arange(101) / 100
    alpha = np.array([
        max(grid, key=lambda a: ccc(y_val[:, i], a * audio_val[:, i] + (1 - a) * text_val[:, i]))
        for i in range(len(TARGETS))
    ])
    ids, y, audio, text = aligned(directory, manifest, "test")
    fusion = alpha * audio + (1 - alpha) * text
    out = directory / "prediction_level"
    out.mkdir(exist_ok=True)
    (out / "fusion_weights.json").write_text(json.dumps({
        "selection_split": "val", "selection_metric": "target CCC",
        "alpha_audio": dict(zip(TARGETS, alpha.tolist())),
        "manifest_sha256": SHA,
    }, indent=2) + "\n")
    pred = pd.DataFrame(y, index=ids, columns=TRUES)
    pred[PREDS] = fusion
    pred.to_csv(out / "test_predictions.csv")
    rows = []
    for method, values in (("audio_only", audio), ("text_only", text), ("prediction_level", fusion)):
        scores = {target: ccc(y[:, i], values[:, i]) for i, target in enumerate(TARGETS)}
        rows.append({"method": method, **scores, "mean_ccc": np.mean(list(scores.values())),
                     "utterances": len(ids)})
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "test_ccc.csv", index=False)
    print(summary.to_string(index=False))
    print(f"G7 BASELINES COMPLETE: {out / 'test_ccc.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run(args.manifest, args.run)
