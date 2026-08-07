import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

ENV_DATA_ROOT = Path(os.getenv("SER_DATA_ROOT", "data"))
ENV_OUTPUT_ROOT = Path(os.getenv("SER_OUTPUT_ROOT", "outputs"))



# Configuration
ROOT = Path(str(ENV_OUTPUT_ROOT / 'aug_runs_results_fusion_bilstm_cnn1d'))

SILENCE_DIR = ROOT / "silence"
REVERB_DIR = ROOT / "reverb"

SILENCE_VAL_CSV = SILENCE_DIR / "fusion_val_predictions.csv"
REVERB_VAL_CSV = REVERB_DIR / "fusion_val_predictions.csv"

SILENCE_TEST_CSV = SILENCE_DIR / "fusion_predictions.csv"
REVERB_TEST_CSV = REVERB_DIR / "fusion_predictions.csv"

OUTPUT_DIR = ROOT / "ensemble_silence_reverb"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_NAMES = ["valence", "arousal", "dominance"]

TRUE_COLS = [f"{t}_true_norm" for t in TARGET_NAMES]
PRED_COLS = [f"{t}_pred_norm" for t in TARGET_NAMES]

ALPHA_GRID = np.linspace(0.0, 1.0, 101)
# alpha is the silence weight.
# 1 - alpha is the reverb weight.


# Metrics
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

    return (2.0 * rho * std_true * std_pred) / (
        var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-12
    )


def evaluate_split(y_true, y_pred):
    rows = []

    for i, name in enumerate(TARGET_NAMES):
        mse = mean_squared_error(y_true[:, i], y_pred[:, i])
        rmse = np.sqrt(mse)
        ccc = ccc_1d(y_true[:, i], y_pred[:, i])

        rows.append({
            "target": name,
            "mse": float(mse),
            "rmse": float(rmse),
            "ccc": float(ccc),
        })

    df = pd.DataFrame(rows)

    df.loc["mean"] = {
        "target": "mean",
        "mse": df["mse"].mean(),
        "rmse": df["rmse"].mean(),
        "ccc": df["ccc"].mean(),
    }

    return df


def denormalize_vad(x):
    x = np.asarray(x, dtype=np.float32)
    return ((x + 1.0) / 2.0) * 4.0 + 1.0


# Alignment helpers
def make_key(df):
    if "_merge_key" in df.columns:
        return df["_merge_key"].astype(str)

    if "utterance_id" in df.columns:
        return df["utterance_id"].astype(str)

    if "source_utt_id" in df.columns and "aug_tag" in df.columns:
        return df["source_utt_id"].astype(str) + "__" + df["aug_tag"].astype(str)

    return pd.Series([f"row_{i:06d}" for i in range(len(df))], index=df.index)


def load_and_align(silence_csv, reverb_csv):
    sil = pd.read_csv(silence_csv).copy()
    rev = pd.read_csv(reverb_csv).copy()

    for c in TRUE_COLS + PRED_COLS:
        if c not in sil.columns:
            raise ValueError(f"Missing column in silence file: {c}")
        if c not in rev.columns:
            raise ValueError(f"Missing column in reverb file: {c}")

    sil["_ensemble_key"] = make_key(sil)
    rev["_ensemble_key"] = make_key(rev)

    sil_keep = sil[["_ensemble_key"] + TRUE_COLS + PRED_COLS].copy()
    rev_keep = rev[["_ensemble_key"] + TRUE_COLS + PRED_COLS].copy()

    sil_keep = sil_keep.rename(
        columns={c: f"{c}_silence" for c in TRUE_COLS + PRED_COLS}
    )
    rev_keep = rev_keep.rename(
        columns={c: f"{c}_reverb" for c in TRUE_COLS + PRED_COLS}
    )

    merged = sil_keep.merge(rev_keep, on="_ensemble_key", how="inner")

    if len(merged) == 0:
        raise ValueError("Merged prediction file is empty. Check alignment keys.")

    # Check true label consistency
    mismatch_count = 0
    for c in TRUE_COLS:
        diff = np.abs(
            merged[f"{c}_silence"].to_numpy(dtype=np.float32)
            - merged[f"{c}_reverb"].to_numpy(dtype=np.float32)
        )
        mismatch_count += int(np.sum(diff > 1e-6))

    if mismatch_count > 0:
        print(f"[WARN] True-label mismatch count: {mismatch_count}")

    y_true = np.stack(
        [merged[f"{c}_silence"].to_numpy(dtype=np.float32) for c in TRUE_COLS],
        axis=1,
    )

    pred_silence = np.stack(
        [merged[f"{c}_silence"].to_numpy(dtype=np.float32) for c in PRED_COLS],
        axis=1,
    )

    pred_reverb = np.stack(
        [merged[f"{c}_reverb"].to_numpy(dtype=np.float32) for c in PRED_COLS],
        axis=1,
    )

    return merged, y_true, pred_silence, pred_reverb


# Fusion methods
def ensemble_global(pred_silence, pred_reverb, alpha):
    return alpha * pred_silence + (1.0 - alpha) * pred_reverb


def ensemble_targetwise(pred_silence, pred_reverb, alpha_by_target):
    y_pred = np.zeros_like(pred_silence, dtype=np.float32)

    for i, name in enumerate(TARGET_NAMES):
        alpha = alpha_by_target[name]
        y_pred[:, i] = alpha * pred_silence[:, i] + (1.0 - alpha) * pred_reverb[:, i]

    return y_pred


def find_best_global_alpha(y_true_val, pred_silence_val, pred_reverb_val):
    records = []

    best_alpha = None
    best_ccc = -1e9
    best_metrics = None

    for alpha in ALPHA_GRID:
        pred = ensemble_global(pred_silence_val, pred_reverb_val, alpha)
        metrics = evaluate_split(y_true_val, pred)
        mean_ccc = float(metrics.loc[metrics["target"] == "mean", "ccc"].values[0])
        mean_rmse = float(metrics.loc[metrics["target"] == "mean", "rmse"].values[0])

        records.append({
            "alpha_silence": float(alpha),
            "alpha_reverb": float(1.0 - alpha),
            "val_mean_ccc": mean_ccc,
            "val_mean_rmse": mean_rmse,
        })

        if mean_ccc > best_ccc:
            best_ccc = mean_ccc
            best_alpha = float(alpha)
            best_metrics = metrics

    return best_alpha, best_metrics, pd.DataFrame(records)


def find_best_targetwise_alpha(y_true_val, pred_silence_val, pred_reverb_val):
    alpha_by_target = {}
    records = []

    for i, name in enumerate(TARGET_NAMES):
        best_alpha = None
        best_ccc = -1e9
        best_rmse = None

        for alpha in ALPHA_GRID:
            pred = alpha * pred_silence_val[:, i] + (1.0 - alpha) * pred_reverb_val[:, i]

            mse = mean_squared_error(y_true_val[:, i], pred)
            rmse = np.sqrt(mse)
            ccc = ccc_1d(y_true_val[:, i], pred)

            records.append({
                "target": name,
                "alpha_silence": float(alpha),
                "alpha_reverb": float(1.0 - alpha),
                "val_ccc": float(ccc),
                "val_rmse": float(rmse),
            })

            if ccc > best_ccc:
                best_ccc = ccc
                best_alpha = float(alpha)
                best_rmse = float(rmse)

        alpha_by_target[name] = best_alpha

        print(
            f"[TARGET-WISE BEST] {name}: "
            f"alpha_silence={best_alpha:.2f}, "
            f"alpha_reverb={1.0 - best_alpha:.2f}, "
            f"val_ccc={best_ccc:.6f}, "
            f"val_rmse={best_rmse:.6f}"
        )

    return alpha_by_target, pd.DataFrame(records)


def save_ensemble_predictions(base_merged, y_true, y_pred, out_csv):
    out = base_merged[["_ensemble_key"]].copy()

    for i, name in enumerate(TARGET_NAMES):
        out[f"{name}_true_norm"] = y_true[:, i]
        out[f"{name}_pred_norm"] = y_pred[:, i]

    y_true_denorm = denormalize_vad(y_true)
    y_pred_denorm = denormalize_vad(y_pred)

    for i, name in enumerate(TARGET_NAMES):
        out[f"{name}_true_1to5"] = y_true_denorm[:, i]
        out[f"{name}_pred_1to5"] = y_pred_denorm[:, i]

    out.to_csv(out_csv, index=False, encoding="utf-8-sig")


# Main
def main():
    print("Loading validation predictions...")
    val_merged, y_val, sil_val, rev_val = load_and_align(
        SILENCE_VAL_CSV,
        REVERB_VAL_CSV,
    )

    print("Loading test predictions...")
    test_merged, y_test, sil_test, rev_test = load_and_align(
        SILENCE_TEST_CSV,
        REVERB_TEST_CSV,
    )

    print("Validation rows:", len(val_merged))
    print("Test rows      :", len(test_merged))

    # Baseline: fixed average 0.5 / 0.5
    fixed_alpha = 0.5

    fixed_val_pred = ensemble_global(sil_val, rev_val, fixed_alpha)
    fixed_test_pred = ensemble_global(sil_test, rev_test, fixed_alpha)

    fixed_val_metrics = evaluate_split(y_val, fixed_val_pred)
    fixed_test_metrics = evaluate_split(y_test, fixed_test_pred)

    fixed_val_metrics.to_csv(
        OUTPUT_DIR / "fixed_0505_val_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    fixed_test_metrics.to_csv(
        OUTPUT_DIR / "fixed_0505_test_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    save_ensemble_predictions(
        val_merged,
        y_val,
        fixed_val_pred,
        OUTPUT_DIR / "fixed_0505_val_predictions.csv",
    )
    save_ensemble_predictions(
        test_merged,
        y_test,
        fixed_test_pred,
        OUTPUT_DIR / "fixed_0505_test_predictions.csv",
    )

    # Global alpha tuned on validation mean CCC
    best_global_alpha, best_global_val_metrics, global_search_df = find_best_global_alpha(
        y_val,
        sil_val,
        rev_val,
    )

    global_search_df.to_csv(
        OUTPUT_DIR / "global_alpha_search.csv",
        index=False,
        encoding="utf-8-sig",
    )

    global_val_pred = ensemble_global(sil_val, rev_val, best_global_alpha)
    global_test_pred = ensemble_global(sil_test, rev_test, best_global_alpha)

    global_val_metrics = evaluate_split(y_val, global_val_pred)
    global_test_metrics = evaluate_split(y_test, global_test_pred)

    global_val_metrics.to_csv(
        OUTPUT_DIR / "global_weighted_val_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    global_test_metrics.to_csv(
        OUTPUT_DIR / "global_weighted_test_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    save_ensemble_predictions(
        val_merged,
        y_val,
        global_val_pred,
        OUTPUT_DIR / "global_weighted_val_predictions.csv",
    )
    save_ensemble_predictions(
        test_merged,
        y_test,
        global_test_pred,
        OUTPUT_DIR / "global_weighted_test_predictions.csv",
    )

    # Target-wise alpha tuned on validation CCC per target
    alpha_by_target, targetwise_search_df = find_best_targetwise_alpha(
        y_val,
        sil_val,
        rev_val,
    )

    targetwise_search_df.to_csv(
        OUTPUT_DIR / "targetwise_alpha_search.csv",
        index=False,
        encoding="utf-8-sig",
    )

    targetwise_val_pred = ensemble_targetwise(sil_val, rev_val, alpha_by_target)
    targetwise_test_pred = ensemble_targetwise(sil_test, rev_test, alpha_by_target)

    targetwise_val_metrics = evaluate_split(y_val, targetwise_val_pred)
    targetwise_test_metrics = evaluate_split(y_test, targetwise_test_pred)

    targetwise_val_metrics.to_csv(
        OUTPUT_DIR / "targetwise_weighted_val_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    targetwise_test_metrics.to_csv(
        OUTPUT_DIR / "targetwise_weighted_test_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    save_ensemble_predictions(
        val_merged,
        y_val,
        targetwise_val_pred,
        OUTPUT_DIR / "targetwise_weighted_val_predictions.csv",
    )
    save_ensemble_predictions(
        test_merged,
        y_test,
        targetwise_test_pred,
        OUTPUT_DIR / "targetwise_weighted_test_predictions.csv",
    )

    # Summary
    def get_mean_row(metrics_df):
        row = metrics_df.loc[metrics_df["target"] == "mean"].iloc[0]
        return {
            "mse": float(row["mse"]),
            "rmse": float(row["rmse"]),
            "ccc": float(row["ccc"]),
        }

    summary_rows = []

    summary_rows.append({
        "method": "fixed_0505",
        "alpha_silence": 0.5,
        "alpha_reverb": 0.5,
        "targetwise": False,
        "val_mse": get_mean_row(fixed_val_metrics)["mse"],
        "val_rmse": get_mean_row(fixed_val_metrics)["rmse"],
        "val_ccc": get_mean_row(fixed_val_metrics)["ccc"],
        "test_mse": get_mean_row(fixed_test_metrics)["mse"],
        "test_rmse": get_mean_row(fixed_test_metrics)["rmse"],
        "test_ccc": get_mean_row(fixed_test_metrics)["ccc"],
    })

    summary_rows.append({
        "method": "global_weighted",
        "alpha_silence": best_global_alpha,
        "alpha_reverb": 1.0 - best_global_alpha,
        "targetwise": False,
        "val_mse": get_mean_row(global_val_metrics)["mse"],
        "val_rmse": get_mean_row(global_val_metrics)["rmse"],
        "val_ccc": get_mean_row(global_val_metrics)["ccc"],
        "test_mse": get_mean_row(global_test_metrics)["mse"],
        "test_rmse": get_mean_row(global_test_metrics)["rmse"],
        "test_ccc": get_mean_row(global_test_metrics)["ccc"],
    })

    summary_rows.append({
        "method": "targetwise_weighted",
        "alpha_silence": json.dumps(alpha_by_target),
        "alpha_reverb": json.dumps({
            k: 1.0 - v for k, v in alpha_by_target.items()
        }),
        "targetwise": True,
        "val_mse": get_mean_row(targetwise_val_metrics)["mse"],
        "val_rmse": get_mean_row(targetwise_val_metrics)["rmse"],
        "val_ccc": get_mean_row(targetwise_val_metrics)["ccc"],
        "test_mse": get_mean_row(targetwise_test_metrics)["mse"],
        "test_rmse": get_mean_row(targetwise_test_metrics)["rmse"],
        "test_ccc": get_mean_row(targetwise_test_metrics)["ccc"],
    })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        OUTPUT_DIR / "ensemble_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with open(OUTPUT_DIR / "ensemble_weights.json", "w", encoding="utf-8") as f:
        json.dump({
            "fixed_0505": {
                "alpha_silence": 0.5,
                "alpha_reverb": 0.5,
            },
            "global_weighted": {
                "alpha_silence": best_global_alpha,
                "alpha_reverb": 1.0 - best_global_alpha,
            },
            "targetwise_weighted": {
                "alpha_silence": alpha_by_target,
                "alpha_reverb": {
                    k: 1.0 - v for k, v in alpha_by_target.items()
                },
            },
            "note": "All weights were selected using validation predictions only, then applied to test predictions.",
        }, f, indent=2)

    print("\n===== FIXED 0.5 / 0.5 TEST =====")
    print(fixed_test_metrics)

    print("\n===== GLOBAL WEIGHTED TEST =====")
    print("best_global_alpha_silence:", best_global_alpha)
    print("best_global_alpha_reverb :", 1.0 - best_global_alpha)
    print(global_test_metrics)

    print("\n===== TARGET-WISE WEIGHTED TEST =====")
    print("alpha_by_target_silence:", alpha_by_target)
    print("alpha_by_target_reverb :", {k: 1.0 - v for k, v in alpha_by_target.items()})
    print(targetwise_test_metrics)

    print("\nSaved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()