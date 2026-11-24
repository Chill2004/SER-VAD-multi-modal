#!/usr/bin/env python3
"""Compute the fixed-Ses05 paired dialogue bootstrap from the saved CSVs."""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np

SEEDS = (42, 123, 2024)
TARGETS = ("valence", "arousal", "dominance")
METHODS = ("model_level", "prediction_level")
EXPECTED_ROWS = 2170
PAPER = {
    "model_level": (0.6450721316819865, 0.6122195429479546, 0.6191123703658915),
    "prediction_level": (0.586182, 0.588756, 0.606918),
}
DIALOGUE = re.compile(r"^(Ses05[FM]_(?:impro|script)[\w]+)_[FM]\d+$")
AUG = re.compile(r"__(?:orig|silence|noise|reverb)$")


def load(path):
    raw = path.read_bytes()
    reader = csv.DictReader(raw.decode("utf-8-sig").splitlines(keepends=True))
    rows = list(reader)
    if any(row.get("aug_tag") == "orig" for row in rows):
        rows = [row for row in rows if row.get("aug_tag") == "orig"]
    key = "source_utt_id" if "source_utt_id" in (reader.fieldnames or []) else "utterance_id"
    data = {}
    for row in rows:
        uid = AUG.sub("", row[key].strip())
        match = DIALOGUE.fullmatch(uid)
        if not match or uid in data:
            raise ValueError(f"Unexpected or duplicate fixed-Ses05 ID in {path.name}: {uid}")
        data[uid] = [[float(row[f"{target}_{column}_norm"]) for target in TARGETS]
                     for column in ("true", "pred")]
    if len(data) != EXPECTED_ROWS:
        raise ValueError(f"{path}: expected {EXPECTED_ROWS} original rows, found {len(data)}")
    ids = sorted(data)
    values = np.asarray([data[uid] for uid in ids], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite labels or predictions in {path}")
    return ids, values[:, 0], values[:, 1], hashlib.sha256(raw).hexdigest()


def moments(y, pred):
    y = np.broadcast_to(y, pred.shape)
    return np.stack((np.ones_like(pred), y, pred, y * y, pred * pred, y * pred), axis=-1)


def ccc(stats):
    n, sy, sp, syy, spp, syp = np.moveaxis(stats, -1, 0)
    y, p = sy / n, sp / n
    cov = syp / n - y * p
    denom = syy / n - y * y + spp / n - p * p + (y - p) ** 2
    return 2 * cov / (denom + 1e-12)


def bootstrap(y, pred, dialogue, draws, seed):
    # pred axes are utterance, method, seed, V/A/D.
    groups, codes = np.unique(dialogue, return_inverse=True)
    terms = moments(y[:, None, None, :], pred)
    cluster = np.zeros((len(groups), *terms.shape[1:]))
    np.add.at(cluster, codes, terms)
    point = ccc(cluster.sum(axis=0))
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(groups), size=(draws, len(groups)))
    weights = (sampled[..., None] == np.arange(len(groups))).sum(axis=1)
    sampled_stats = (weights @ cluster.reshape(len(groups), -1)).reshape(draws, *cluster.shape[1:])
    scores = ccc(sampled_stats).mean(axis=2)
    point = point.mean(axis=1)
    delta = scores[:, 0] - scores[:, 1]
    delta = np.column_stack((delta, delta.mean(axis=1)))
    ci = np.quantile(delta, [0.025, 0.975], axis=0)
    by_method = np.column_stack((point, point.mean(axis=1)))
    return len(groups), by_method, ci


def run(args):
    if args.draws < 2:
        raise ValueError("At least two bootstrap draws are required")
    reference_ids = reference_y = None
    predictions, hashes, seed_scores = [], {}, {}
    for method in METHODS:
        method_predictions = []
        seed_scores[method] = {}
        for seed, expected in zip(SEEDS, PAPER[method]):
            path = (
                args.model_root / f"silence_audio_baseline_text_seed{seed}/model_level_predictions.csv"
                if method == "model_level"
                else args.prediction_root / f"seed{seed}/prediction_level_predictions.csv"
            )
            ids, y, pred, digest = load(path)
            if reference_ids is None:
                reference_ids, reference_y = ids, y
            if ids != reference_ids or not np.allclose(y, reference_y, rtol=0, atol=1e-7):
                raise ValueError(f"IDs or labels differ across the paired inputs: {path}")
            value = float(ccc(moments(y, pred).sum(axis=0)).mean())
            print(f"{method}, seed {seed}: Mean CCC {value:.8f} (archived {expected:.8f})")
            if abs(value - expected) > 2e-6:
                raise ValueError("This CSV does not reproduce the archived fixed-Ses05 CCC")
            method_predictions.append(pred)
            hashes[f"{method}_{seed}"] = digest
            seed_scores[method][str(seed)] = value
        predictions.append(np.stack(method_predictions, axis=1))

    dialogues = [DIALOGUE.fullmatch(uid)[1] for uid in reference_ids]
    clusters, by_method, ci = bootstrap(
        reference_y, np.stack(predictions, axis=1), dialogues, args.draws, args.seed
    )
    report = []
    for i, target in enumerate((*TARGETS, "mean")):
        model, scalar = by_method[:, i]
        report.append({
            "target": target, "model_level_ccc": float(model),
            "scalar_ccc": float(scalar), "delta_ccc": float(model - scalar),
            "ci_95_low": float(ci[0, i]), "ci_95_high": float(ci[1, i]),
        })
        print(f"{target:10} delta {model - scalar:+.6f}, 95% CI [{ci[0, i]:+.6f}, {ci[1, i]:+.6f}]")
    print(f"FIXED SES05 COMPLETE: {EXPECTED_ROWS} utterances, {clusters} dialogues, {args.draws} draws")
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump({
                "protocol": "fixed-Ses05; whole-dialogue paired percentile bootstrap",
                "utterances": EXPECTED_ROWS, "dialogues": clusters, "draws": args.draws,
                "bootstrap_seed": args.seed, "seed_ccc": seed_scores, "sha256": hashes,
                "results": report,
            }, stream, indent=2)
            stream.write("\n")


if __name__ == "__main__":
    downloads = Path.home() / "Downloads"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=downloads / "aug_runs_results_multimodal_model_level")
    parser.add_argument("--prediction-root", type=Path, default=downloads / "prediction_level_fusion_3seeds")
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path)
    run(parser.parse_args())
