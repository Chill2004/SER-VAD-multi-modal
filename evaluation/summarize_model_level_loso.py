from __future__ import annotations
import os

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ENV_DATA_ROOT = Path(os.getenv("SER_DATA_ROOT", "data"))
ENV_OUTPUT_ROOT = Path(os.getenv("SER_OUTPUT_ROOT", "outputs"))


TARGETS = ["valence", "arousal", "dominance", "mean"]
FOLDS = ["Ses01", "Ses02", "Ses03", "Ses04", "Ses05"]


def ccc_1d(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    return (2.0 * cov) / (
        var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-12
    )


def load_fold_metrics(run_dir: Path, fold: str) -> pd.DataFrame:
    metrics_path = run_dir / "model_level_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics for {fold}: {metrics_path}")

    df = pd.read_csv(metrics_path)
    required = {"target", "mse", "rmse", "ccc"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{fold}: missing columns {sorted(missing)} in {metrics_path}")

    rows = []
    for target in TARGETS:
        match = df[df["target"].astype(str).str.lower() == target]
        if len(match) != 1:
            raise ValueError(
                f"{fold}: expected exactly one row for target={target}, got {len(match)}"
            )
        r = match.iloc[0]
        rows.append({
            "fold": fold,
            "target": target,
            "mse": float(r["mse"]),
            "rmse": float(r["rmse"]),
            "ccc": float(r["ccc"]),
        })

    return pd.DataFrame(rows)


def load_fold_predictions(run_dir: Path, fold: str) -> pd.DataFrame:
    pred_path = run_dir / "model_level_predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing predictions for {fold}: {pred_path}")

    df = pd.read_csv(pred_path)
    df["fold"] = fold

    required = []
    for t in ["valence", "arousal", "dominance"]:
        required += [f"{t}_true_norm", f"{t}_pred_norm"]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{fold}: prediction CSV missing columns: {missing}")

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default=str(ENV_OUTPUT_ROOT / 'aug_runs_results_multimodal_model_level_loso'),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prefix",
        type=str,
        default="silence_audio_baseline_text_loso_fold",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(ENV_OUTPUT_ROOT / 'loso_summary_seed42'),
    )
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_metric_frames = []
    fold_prediction_frames = []

    for fold in FOLDS:
        run_name = f"{args.prefix}_{fold}_seed{args.seed}"
        run_dir = root / run_name

        if not run_dir.exists():
            raise FileNotFoundError(f"Missing LOSO run folder: {run_dir}")

        fold_metric_frames.append(load_fold_metrics(run_dir, fold))
        fold_prediction_frames.append(load_fold_predictions(run_dir, fold))

    fold_metrics = pd.concat(fold_metric_frames, ignore_index=True)
    fold_metrics.to_csv(
        out_dir / "loso_fold_metrics_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    wide = (
        fold_metrics.pivot(index="fold", columns="target", values="ccc")
        .reindex(FOLDS)
        .reset_index()
    )
    wide = wide[["fold", "valence", "arousal", "dominance", "mean"]]
    wide.to_csv(
        out_dir / "loso_ccc_by_fold.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary_rows = []
    for target in TARGETS:
        vals = (
            fold_metrics.loc[fold_metrics["target"] == target, "ccc"]
            .to_numpy(dtype=np.float64)
        )
        summary_rows.append({
            "target": target,
            "n_folds": len(vals),
            "mean_ccc": float(np.mean(vals)),
            "sample_std_ccc": float(np.std(vals, ddof=1)),
            "min_ccc": float(np.min(vals)),
            "max_ccc": float(np.max(vals)),
        })

    mean_std = pd.DataFrame(summary_rows)
    mean_std.to_csv(
        out_dir / "loso_mean_std_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pooled = pd.concat(fold_prediction_frames, ignore_index=True)

    pooled_rows = []
    for target in ["valence", "arousal", "dominance"]:
        yt = pooled[f"{target}_true_norm"].to_numpy(dtype=np.float64)
        yp = pooled[f"{target}_pred_norm"].to_numpy(dtype=np.float64)
        mse = float(np.mean((yt - yp) ** 2))
        rmse = float(np.sqrt(mse))
        ccc = float(ccc_1d(yt, yp))
        pooled_rows.append({
            "target": target,
            "mse": mse,
            "rmse": rmse,
            "ccc": ccc,
        })

    pooled_df = pd.DataFrame(pooled_rows)
    pooled_df.loc[len(pooled_df)] = {
        "target": "mean",
        "mse": float(pooled_df["mse"].mean()),
        "rmse": float(pooled_df["rmse"].mean()),
        "ccc": float(pooled_df["ccc"].mean()),
    }
    pooled_df.to_csv(
        out_dir / "loso_pooled_utterance_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary_json = {
        "seed": int(args.seed),
        "folds": FOLDS,
        "n_test_utterances_pooled": int(len(pooled)),
        "main_statistic": (
            "Mean CCC and sample standard deviation (ddof=1) across the five held-out sessions."
        ),
        "fold_ccc": wide.set_index("fold").to_dict(orient="index"),
        "mean_std": {
            row["target"]: {
                "mean_ccc": row["mean_ccc"],
                "sample_std_ccc": row["sample_std_ccc"],
            }
            for row in summary_rows
        },
        "pooled_utterance_metrics": pooled_df.set_index("target").to_dict(orient="index"),
    }

    with open(out_dir / "loso_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)

    print("\n===== CCC BY HELD-OUT SESSION =====")
    print(wide.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\n===== MAIN LOSO SUMMARY: MEAN ± SAMPLE SD ACROSS 5 FOLDS =====")
    for _, row in mean_std.iterrows():
        print(
            f"{row['target']:10s}: "
            f"{row['mean_ccc']:.6f} ± {row['sample_std_ccc']:.6f}"
        )

    print("\n===== PAPER-READY (3 decimals) =====")
    for _, row in mean_std.iterrows():
        print(
            f"{row['target']:10s}: "
            f"{row['mean_ccc']:.3f} ± {row['sample_std_ccc']:.3f}"
        )

    print("\n===== POOLED UTTERANCE-LEVEL CCC (SECONDARY REFERENCE) =====")
    print(
        pooled_df[["target", "ccc"]].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print("\nSaved summary to:", out_dir)


if __name__ == "__main__":
    main()
