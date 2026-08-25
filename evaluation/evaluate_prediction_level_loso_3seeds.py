"""Repeated-seed five-fold LOSO evaluation for prediction-level fusion.

Expected directory layout under --root:
  seed_<seed>/audio/<experiment>/
  seed_<seed>/text/<experiment>/

For each held-out fold and V/A/D target, one audio mixing weight alpha is
selected from {0.00, 0.01, ..., 1.00} by maximizing mean validation CCC across
seeds 42, 123, and 2024. The same fold-specific alpha is then applied unchanged
to the three corresponding test predictions. Test labels are never used for
weight selection.

Final reporting first averages the three seeds within each fold, then computes
mean and sample standard deviation (ddof=1) across the five seed-averaged folds.
"""

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


SEEDS = [42, 123, 2024]
FOLDS = ["Ses01", "Ses02", "Ses03", "Ses04", "Ses05"]
TARGETS = ["valence", "arousal", "dominance"]
TRUE_COLS = [f"{t}_true_norm" for t in TARGETS]
PRED_COLS = [f"{t}_pred_norm" for t in TARGETS]

ALPHA_START = 0.0
ALPHA_END = 1.0
ALPHA_STEP = 0.01
ALPHAS = np.round(np.arange(ALPHA_START, ALPHA_END + ALPHA_STEP, ALPHA_STEP), 6)

AUG_SUFFIXES = ["__orig", "__silence", "__noise", "__reverb"]


def strip_aug_suffix(value):
    value = str(value)
    for suffix in AUG_SUFFIXES:
        if value.endswith(suffix):
            return value[:-len(suffix)]
    return value


def source_key(df):
    for col in ["_source_key", "source_utt_id", "_ensemble_key", "_merge_key", "utterance_id"]:
        if col in df.columns:
            return df[col].astype(str).map(strip_aug_suffix)
    raise ValueError(
        "Cannot determine source key; expected one of "
        "_source_key/source_utt_id/_ensemble_key/_merge_key/utterance_id. "
        f"Columns={list(df.columns)}"
    )


def ccc_1d(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    std_true = np.std(y_true)
    std_pred = np.std(y_pred)

    if std_true == 0 or std_pred == 0:
        return 0.0

    rho = np.corrcoef(y_true, y_pred)[0, 1]
    if np.isnan(rho):
        rho = 0.0

    return float(
        (2.0 * rho * std_true * std_pred)
        / (var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-12)
    )


def evaluate(y_true, y_pred):
    rows = []
    for i, target in enumerate(TARGETS):
        mse = float(mean_squared_error(y_true[:, i], y_pred[:, i]))
        rows.append(
            {
                "target": target,
                "mse": mse,
                "rmse": float(np.sqrt(mse)),
                "ccc": ccc_1d(y_true[:, i], y_pred[:, i]),
            }
        )
    df = pd.DataFrame(rows)
    df.loc[len(df)] = {
        "target": "mean",
        "mse": float(df["mse"].mean()),
        "rmse": float(df["rmse"].mean()),
        "ccc": float(df["ccc"].mean()),
    }
    return df


def load_and_align(audio_csv, text_csv):
    if not audio_csv.exists():
        raise FileNotFoundError(audio_csv)
    if not text_csv.exists():
        raise FileNotFoundError(text_csv)

    a = pd.read_csv(audio_csv).copy()
    t = pd.read_csv(text_csv).copy()

    for col in TRUE_COLS + PRED_COLS:
        if col not in a.columns:
            raise ValueError(f"{audio_csv}: missing {col}")
        if col not in t.columns:
            raise ValueError(f"{text_csv}: missing {col}")

    a["_align_key"] = source_key(a)
    t["_align_key"] = source_key(t)

    if a["_align_key"].duplicated().any():
        raise ValueError(f"{audio_csv}: duplicate source keys")
    if t["_align_key"].duplicated().any():
        raise ValueError(f"{text_csv}: duplicate source keys")

    aset = set(a["_align_key"])
    tset = set(t["_align_key"])
    if aset != tset:
        raise ValueError(
            f"Audio/text source mismatch: {audio_csv} vs {text_csv}; "
            f"audio={len(aset)} text={len(tset)}"
        )

    a_keep = a[["_align_key"] + TRUE_COLS + PRED_COLS].rename(
        columns={
            **{c: f"{c}_audio" for c in TRUE_COLS},
            **{c: f"{c}_audio" for c in PRED_COLS},
        }
    )
    t_keep = t[["_align_key"] + TRUE_COLS + PRED_COLS].rename(
        columns={
            **{c: f"{c}_text" for c in TRUE_COLS},
            **{c: f"{c}_text" for c in PRED_COLS},
        }
    )

    merged = a_keep.merge(t_keep, on="_align_key", validate="one_to_one")

    y_true = np.stack(
        [merged[f"{c}_audio"].to_numpy(np.float64) for c in TRUE_COLS], axis=1
    )
    audio_pred = np.stack(
        [merged[f"{c}_audio"].to_numpy(np.float64) for c in PRED_COLS], axis=1
    )
    text_pred = np.stack(
        [merged[f"{c}_text"].to_numpy(np.float64) for c in PRED_COLS], axis=1
    )

    for i, col in enumerate(TRUE_COLS):
        text_true = merged[f"{col}_text"].to_numpy(np.float64)
        diff = float(np.max(np.abs(y_true[:, i] - text_true)))
        if diff >= 1e-6:
            raise ValueError(f"Target mismatch in {col}: max absolute difference={diff}")

    return merged["_align_key"].copy(), y_true, audio_pred, text_pred


def fold_dir(root, seed, branch, fold):
    name = f"silence_audio_baseline_text_loso_fold_{fold}_seed{seed}"
    return root / f"seed_{seed}" / branch / name


def input_paths(root, seed, fold, split):
    adir = fold_dir(root, seed, "audio", fold)
    tdir = fold_dir(root, seed, "text", fold)

    if split == "val":
        return (
            adir / "fusion_val_predictions.csv",
            tdir / "text_embedding_val_predictions.csv",
        )
    if split == "test":
        return (
            adir / "fusion_predictions.csv",
            tdir / "text_embedding_predictions.csv",
        )
    raise ValueError(split)


def prediction_output_dir(root, seed, fold):
    name = f"silence_audio_baseline_text_loso_fold_{fold}_seed{seed}"
    return root / f"seed_{seed}" / "prediction_level" / name


def save_fused_predictions(keys, y_true, y_pred, path):
    out = pd.DataFrame({"_source_key": keys.astype(str).to_numpy()})
    for i, target in enumerate(TARGETS):
        out[f"{target}_true_norm"] = y_true[:, i]
        out[f"{target}_pred_norm"] = y_pred[:, i]
    out.to_csv(path, index=False, encoding="utf-8-sig")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(os.getenv("SER_OUTPUT_ROOT", "outputs")) / "LOSO_prediction_level",
    )
    args = parser.parse_args()
    root = args.root

    evaluation_dir = root / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    cache = {}
    for fold in FOLDS:
        for seed in SEEDS:
            for split in ["val", "test"]:
                acsv, tcsv = input_paths(root, seed, fold, split)
                cache[(fold, seed, split)] = load_and_align(acsv, tcsv)

    alpha_rows = []
    alpha_search_rows = []
    alpha_by_fold = {}

    for fold in FOLDS:
        alpha_by_fold[fold] = {}

        for target_idx, target in enumerate(TARGETS):
            best_alpha = None
            best_mean_val_ccc = -np.inf
            best_seed_cccs = None

            for alpha in ALPHAS:
                seed_cccs = []

                for seed in SEEDS:
                    _, y_val, audio_val, text_val = cache[(fold, seed, "val")]
                    fused = (
                        alpha * audio_val[:, target_idx]
                        + (1.0 - alpha) * text_val[:, target_idx]
                    )
                    seed_cccs.append(ccc_1d(y_val[:, target_idx], fused))

                mean_ccc = float(np.mean(seed_cccs))

                alpha_search_rows.append(
                    {
                        "fold": fold,
                        "target": target,
                        "alpha_audio": float(alpha),
                        "alpha_text": float(1.0 - alpha),
                        "seed42_val_ccc": float(seed_cccs[0]),
                        "seed123_val_ccc": float(seed_cccs[1]),
                        "seed2024_val_ccc": float(seed_cccs[2]),
                        "mean_val_ccc": mean_ccc,
                    }
                )

                if mean_ccc > best_mean_val_ccc:
                    best_mean_val_ccc = mean_ccc
                    best_alpha = float(alpha)
                    best_seed_cccs = list(seed_cccs)

            alpha_by_fold[fold][target] = best_alpha
            alpha_rows.append(
                {
                    "fold": fold,
                    "target": target,
                    "alpha_audio": best_alpha,
                    "alpha_text": 1.0 - best_alpha,
                    "mean_validation_ccc": best_mean_val_ccc,
                    "seed42_validation_ccc": best_seed_cccs[0],
                    "seed123_validation_ccc": best_seed_cccs[1],
                    "seed2024_validation_ccc": best_seed_cccs[2],
                }
            )

            print(
                f"[ALPHA] {fold} {target:10s} | "
                f"audio={best_alpha:.2f} text={1-best_alpha:.2f} | "
                f"mean val CCC={best_mean_val_ccc:.6f}"
            )

    pd.DataFrame(alpha_rows).to_csv(
        evaluation_dir / "prediction_level_alpha_by_fold.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(alpha_search_rows).to_csv(
        evaluation_dir / "prediction_level_alpha_search.csv",
        index=False,
        encoding="utf-8-sig",
    )

    seed_metric_rows = []

    for fold in FOLDS:
        alphas = np.array(
            [alpha_by_fold[fold][target] for target in TARGETS],
            dtype=np.float64,
        )

        for seed in SEEDS:
            keys, y_test, audio_test, text_test = cache[(fold, seed, "test")]
            fused_test = (
                audio_test * alphas.reshape(1, -1)
                + text_test * (1.0 - alphas.reshape(1, -1))
            )

            metrics = evaluate(y_test, fused_test)

            out_dir = prediction_output_dir(root, seed, fold)
            out_dir.mkdir(parents=True, exist_ok=True)

            metrics.to_csv(
                out_dir / "prediction_level_test_metrics.csv",
                index=False,
                encoding="utf-8-sig",
            )
            save_fused_predictions(
                keys,
                y_test,
                fused_test,
                out_dir / "prediction_level_test_predictions.csv",
            )

            with (out_dir / "prediction_level_weights.json").open(
                "w", encoding="utf-8"
            ) as f:
                json.dump(
                    {
                        "fold": fold,
                        "seed": seed,
                        "formula": (
                            "pred = alpha_audio * pred_audio + "
                            "(1-alpha_audio) * pred_text"
                        ),
                        "selection": (
                            "One target-wise alpha per fold selected by "
                            "maximizing mean validation CCC across seeds "
                            "42, 123, and 2024; same fold alpha then applied "
                            "to each seed's test predictions."
                        ),
                        "alpha_grid": {
                            "start": ALPHA_START,
                            "end": ALPHA_END,
                            "step": ALPHA_STEP,
                        },
                        "alpha_audio_by_target": alpha_by_fold[fold],
                        "alpha_text_by_target": {
                            t: 1.0 - alpha_by_fold[fold][t] for t in TARGETS
                        },
                    },
                    f,
                    indent=2,
                )

            for _, row in metrics.iterrows():
                seed_metric_rows.append(
                    {
                        "fold": fold,
                        "seed": seed,
                        "target": row["target"],
                        "mse": float(row["mse"]),
                        "rmse": float(row["rmse"]),
                        "ccc": float(row["ccc"]),
                    }
                )

            mean_ccc = float(
                metrics.loc[metrics["target"] == "mean", "ccc"].iloc[0]
            )
            print(
                f"[TEST] {fold} seed={seed} | "
                f"prediction-level mean CCC={mean_ccc:.6f}"
            )

    by_seed = pd.DataFrame(seed_metric_rows)
    by_seed.to_csv(
        evaluation_dir / "prediction_level_loso_results_by_seed.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fold_avg = (
        by_seed.groupby(["fold", "target"], as_index=False)[["mse", "rmse", "ccc"]]
        .mean()
    )

    fold_order = {fold: i for i, fold in enumerate(FOLDS)}
    target_order = {t: i for i, t in enumerate(TARGETS + ["mean"])}
    fold_avg["_fold_order"] = fold_avg["fold"].map(fold_order)
    fold_avg["_target_order"] = fold_avg["target"].map(target_order)
    fold_avg = (
        fold_avg.sort_values(["_fold_order", "_target_order"])
        .drop(columns=["_fold_order", "_target_order"])
        .reset_index(drop=True)
    )

    fold_avg.to_csv(
        evaluation_dir / "prediction_level_loso_seed_averaged_by_fold.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary_rows = []
    for target in TARGETS + ["mean"]:
        sub = fold_avg.loc[fold_avg["target"] == target]

        summary_rows.append(
            {
                "target": target,
                "mse_mean": float(sub["mse"].mean()),
                "mse_std": float(sub["mse"].std(ddof=1)),
                "rmse_mean": float(sub["rmse"].mean()),
                "rmse_std": float(sub["rmse"].std(ddof=1)),
                "ccc_mean": float(sub["ccc"].mean()),
                "ccc_std": float(sub["ccc"].std(ddof=1)),
                "n_folds": int(len(sub)),
                "aggregation": (
                    "average 3 seeds within each fold, then sample std "
                    "(ddof=1) across 5 seed-averaged folds"
                ),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        evaluation_dir / "prediction_level_loso_mean_std_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with (evaluation_dir / "prediction_level_protocol.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            {
                "seeds": SEEDS,
                "folds": FOLDS,
                "targets": TARGETS,
                "alpha_grid": {
                    "start": ALPHA_START,
                    "end": ALPHA_END,
                    "step": ALPHA_STEP,
                },
                "alpha_selection": (
                    "For each fold and target, select a single audio alpha "
                    "that maximizes the mean validation CCC across seeds "
                    "42/123/2024. Apply that same fold-specific alpha to "
                    "the test predictions of all three seeds."
                ),
                "final_aggregation": (
                    "Average metrics across three seeds within each fold; "
                    "then compute mean and sample standard deviation "
                    "(ddof=1) across the five seed-averaged folds."
                ),
                "test_labels_used_for_selection": False,
            },
            f,
            indent=2,
        )

    print("\n===== SEED-AVERAGED FOLD CCC =====")
    print(
        fold_avg[["fold", "target", "ccc"]]
        .pivot(index="fold", columns="target", values="ccc")
        .reindex(FOLDS)
        .to_string()
    )

    print("\n===== FINAL 5-FOLD MEAN ± STD =====")
    print(summary[["target", "ccc_mean", "ccc_std"]].to_string(index=False))

    print("\nSaved evaluation to:", evaluation_dir)


if __name__ == "__main__":
    main()
