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

MODEL_A_NAME = "silence"
MODEL_B_NAME = "reverb"

MODEL_A_DIR = ROOT / MODEL_A_NAME
MODEL_B_DIR = ROOT / MODEL_B_NAME

A_VAL_CSV = MODEL_A_DIR / "fusion_val_predictions.csv"
B_VAL_CSV = MODEL_B_DIR / "fusion_val_predictions.csv"

A_TEST_CSV = MODEL_A_DIR / "fusion_predictions.csv"
B_TEST_CSV = MODEL_B_DIR / "fusion_predictions.csv"

OUTPUT_DIR = ROOT / "alpha_tuning_silence_reverb"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_NAMES = ["valence", "arousal", "dominance"]

TRUE_COLS = [f"{t}_true_norm" for t in TARGET_NAMES]
PRED_COLS = [f"{t}_pred_norm" for t in TARGET_NAMES]

# alpha is the silence weight.
# 1 - alpha is the reverb weight.
ALPHA_START = 0.0
ALPHA_END = 1.0
ALPHA_STEP = 0.01

ALPHAS = np.round(
    np.arange(ALPHA_START, ALPHA_END + ALPHA_STEP, ALPHA_STEP),
    6,
)


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

    ccc = (2.0 * rho * std_true * std_pred) / (
        var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-12
    )

    return float(ccc)


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


# Load and align
def strip_aug_suffix(x):
    x = str(x)

    suffixes = [
        "__orig",
        "__silence",
        "__noise",
        "__reverb",
        "__silence_noise",
        "__noise_silence",
        "__silence_reverb",
        "__reverb_silence",
        "__silence_reverb_noise",
        "__orig_silence_noise",
    ]

    for s in suffixes:
        if x.endswith(s):
            return x[: -len(s)]

    return x


def make_key(df):
    # Prefer the shared source key for cross-augmentation alignment.
    if "source_utt_id" in df.columns:
        return df["source_utt_id"].astype(str)

    # Reuse an existing merge key when available.
    if "_merge_key" in df.columns:
        return df["_merge_key"].astype(str).map(strip_aug_suffix)

    if "utterance_id" in df.columns:
        return df["utterance_id"].astype(str).map(strip_aug_suffix)

    # Fallback
    return pd.Series([f"row_{i:06d}" for i in range(len(df))], index=df.index)


def load_and_align(csv_a, csv_b):
    df_a = pd.read_csv(csv_a).copy()
    df_b = pd.read_csv(csv_b).copy()

    required_cols = TRUE_COLS + PRED_COLS

    for c in required_cols:
        if c not in df_a.columns:
            raise ValueError(f"Missing column in {csv_a}: {c}")
        if c not in df_b.columns:
            raise ValueError(f"Missing column in {csv_b}: {c}")

    df_a["_ensemble_key"] = make_key(df_a)
    df_b["_ensemble_key"] = make_key(df_b)

    keep_a = df_a[["_ensemble_key"] + required_cols].copy()
    keep_b = df_b[["_ensemble_key"] + required_cols].copy()

    keep_a = keep_a.rename(columns={c: f"{c}_a" for c in required_cols})
    keep_b = keep_b.rename(columns={c: f"{c}_b" for c in required_cols})

    merged = keep_a.merge(keep_b, on="_ensemble_key", how="inner")

    if len(merged) == 0:
        raise ValueError("Merged file is empty. Check alignment keys.")

    # Use labels from model A after alignment.
    y_true = np.stack(
        [merged[f"{c}_a"].to_numpy(dtype=np.float32) for c in TRUE_COLS],
        axis=1,
    )

    pred_a = np.stack(
        [merged[f"{c}_a"].to_numpy(dtype=np.float32) for c in PRED_COLS],
        axis=1,
    )

    pred_b = np.stack(
        [merged[f"{c}_b"].to_numpy(dtype=np.float32) for c in PRED_COLS],
        axis=1,
    )

    # Confirm labels match across models.
    mismatch_count = 0
    for c in TRUE_COLS:
        diff = np.abs(
            merged[f"{c}_a"].to_numpy(dtype=np.float32)
            - merged[f"{c}_b"].to_numpy(dtype=np.float32)
        )
        mismatch_count += int(np.sum(diff > 1e-6))

    if mismatch_count > 0:
        print(f"[WARN] true-label mismatch count: {mismatch_count}")

    return merged, y_true, pred_a, pred_b


# Fusion
def ensemble(pred_a, pred_b, alpha):
    return alpha * pred_a + (1.0 - alpha) * pred_b


def tune_global_alpha(y_true_val, pred_a_val, pred_b_val):
    records = []

    best_alpha = None
    best_ccc = -1e9
    best_metrics = None

    for alpha in ALPHAS:
        y_pred = ensemble(pred_a_val, pred_b_val, alpha)
        metrics = evaluate_split(y_true_val, y_pred)

        mean_row = metrics.loc[metrics["target"] == "mean"].iloc[0]

        record = {
            "alpha_a": float(alpha),
            "alpha_b": float(1.0 - alpha),
            "val_mse_mean": float(mean_row["mse"]),
            "val_rmse_mean": float(mean_row["rmse"]),
            "val_ccc_mean": float(mean_row["ccc"]),
        }

        for _, row in metrics.iterrows():
            target = row["target"]
            record[f"val_{target}_ccc"] = float(row["ccc"])
            record[f"val_{target}_rmse"] = float(row["rmse"])

        records.append(record)

        if float(mean_row["ccc"]) > best_ccc:
            best_ccc = float(mean_row["ccc"])
            best_alpha = float(alpha)
            best_metrics = metrics

    search_df = pd.DataFrame(records)

    return best_alpha, best_metrics, search_df


def tune_targetwise_alpha(y_true_val, pred_a_val, pred_b_val):
    alpha_by_target = {}
    records = []

    for i, target in enumerate(TARGET_NAMES):
        best_alpha = None
        best_ccc = -1e9
        best_rmse = None

        for alpha in ALPHAS:
            y_pred_one = alpha * pred_a_val[:, i] + (1.0 - alpha) * pred_b_val[:, i]

            mse = mean_squared_error(y_true_val[:, i], y_pred_one)
            rmse = np.sqrt(mse)
            ccc = ccc_1d(y_true_val[:, i], y_pred_one)

            records.append({
                "target": target,
                "alpha_a": float(alpha),
                "alpha_b": float(1.0 - alpha),
                "val_mse": float(mse),
                "val_rmse": float(rmse),
                "val_ccc": float(ccc),
            })

            if ccc > best_ccc:
                best_ccc = float(ccc)
                best_alpha = float(alpha)
                best_rmse = float(rmse)

        alpha_by_target[target] = best_alpha

        print(
            f"[BEST TARGET ALPHA] {target}: "
            f"alpha_{MODEL_A_NAME}={best_alpha:.2f}, "
            f"alpha_{MODEL_B_NAME}={1.0 - best_alpha:.2f}, "
            f"val_ccc={best_ccc:.6f}, "
            f"val_rmse={best_rmse:.6f}"
        )

    return alpha_by_target, pd.DataFrame(records)


def ensemble_targetwise(pred_a, pred_b, alpha_by_target):
    y_pred = np.zeros_like(pred_a, dtype=np.float32)

    for i, target in enumerate(TARGET_NAMES):
        alpha = alpha_by_target[target]
        y_pred[:, i] = alpha * pred_a[:, i] + (1.0 - alpha) * pred_b[:, i]

    return y_pred


def save_predictions(merged, y_true, y_pred, out_csv):
    out = merged[["_ensemble_key"]].copy()

    for i, target in enumerate(TARGET_NAMES):
        out[f"{target}_true_norm"] = y_true[:, i]
        out[f"{target}_pred_norm"] = y_pred[:, i]

    y_true_denorm = denormalize_vad(y_true)
    y_pred_denorm = denormalize_vad(y_pred)

    for i, target in enumerate(TARGET_NAMES):
        out[f"{target}_true_1to5"] = y_true_denorm[:, i]
        out[f"{target}_pred_1to5"] = y_pred_denorm[:, i]

    out.to_csv(out_csv, index=False, encoding="utf-8-sig")


# Main
def main():
    print("MODEL A:", MODEL_A_NAME)
    print("MODEL B:", MODEL_B_NAME)
    print("alpha means:")
    print(f"  final = alpha * {MODEL_A_NAME} + (1-alpha) * {MODEL_B_NAME}")
    print("Alpha grid:", ALPHAS[0], "->", ALPHAS[-1], "step", ALPHA_STEP)

    print("\nLoading validation predictions...")
    val_merged, y_val, pred_a_val, pred_b_val = load_and_align(A_VAL_CSV, B_VAL_CSV)

    print("Loading test predictions...")
    test_merged, y_test, pred_a_test, pred_b_test = load_and_align(A_TEST_CSV, B_TEST_CSV)

    print("Validation rows:", len(val_merged))
    print("Test rows      :", len(test_merged))

    # Global alpha tuning
    best_alpha, best_val_metrics, global_search_df = tune_global_alpha(
        y_val,
        pred_a_val,
        pred_b_val,
    )

    global_search_df.to_csv(
        OUTPUT_DIR / "global_alpha_search.csv",
        index=False,
        encoding="utf-8-sig",
    )

    global_val_pred = ensemble(pred_a_val, pred_b_val, best_alpha)
    global_test_pred = ensemble(pred_a_test, pred_b_test, best_alpha)

    global_val_metrics = evaluate_split(y_val, global_val_pred)
    global_test_metrics = evaluate_split(y_test, global_test_pred)

    global_val_metrics.to_csv(
        OUTPUT_DIR / "global_best_alpha_val_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    global_test_metrics.to_csv(
        OUTPUT_DIR / "global_best_alpha_test_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    save_predictions(
        val_merged,
        y_val,
        global_val_pred,
        OUTPUT_DIR / "global_best_alpha_val_predictions.csv",
    )
    save_predictions(
        test_merged,
        y_test,
        global_test_pred,
        OUTPUT_DIR / "global_best_alpha_test_predictions.csv",
    )

    # Target-wise alpha tuning
    alpha_by_target, targetwise_search_df = tune_targetwise_alpha(
        y_val,
        pred_a_val,
        pred_b_val,
    )

    targetwise_search_df.to_csv(
        OUTPUT_DIR / "targetwise_alpha_search.csv",
        index=False,
        encoding="utf-8-sig",
    )

    targetwise_val_pred = ensemble_targetwise(pred_a_val, pred_b_val, alpha_by_target)
    targetwise_test_pred = ensemble_targetwise(pred_a_test, pred_b_test, alpha_by_target)

    targetwise_val_metrics = evaluate_split(y_val, targetwise_val_pred)
    targetwise_test_metrics = evaluate_split(y_test, targetwise_test_pred)

    targetwise_val_metrics.to_csv(
        OUTPUT_DIR / "targetwise_best_alpha_val_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    targetwise_test_metrics.to_csv(
        OUTPUT_DIR / "targetwise_best_alpha_test_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    save_predictions(
        val_merged,
        y_val,
        targetwise_val_pred,
        OUTPUT_DIR / "targetwise_best_alpha_val_predictions.csv",
    )
    save_predictions(
        test_merged,
        y_test,
        targetwise_test_pred,
        OUTPUT_DIR / "targetwise_best_alpha_test_predictions.csv",
    )

    # Summary
    def mean_row(metrics):
        row = metrics.loc[metrics["target"] == "mean"].iloc[0]
        return {
            "mse": float(row["mse"]),
            "rmse": float(row["rmse"]),
            "ccc": float(row["ccc"]),
        }

    summary = pd.DataFrame([
        {
            "method": "global_alpha",
            "alpha_type": "single_alpha_for_all_targets",
            "alpha_a": best_alpha,
            "alpha_b": 1.0 - best_alpha,
            "alpha_by_target": "",
            "val_mse": mean_row(global_val_metrics)["mse"],
            "val_rmse": mean_row(global_val_metrics)["rmse"],
            "val_ccc": mean_row(global_val_metrics)["ccc"],
            "test_mse": mean_row(global_test_metrics)["mse"],
            "test_rmse": mean_row(global_test_metrics)["rmse"],
            "test_ccc": mean_row(global_test_metrics)["ccc"],
        },
        {
            "method": "targetwise_alpha",
            "alpha_type": "separate_alpha_per_target",
            "alpha_a": "",
            "alpha_b": "",
            "alpha_by_target": json.dumps(alpha_by_target),
            "val_mse": mean_row(targetwise_val_metrics)["mse"],
            "val_rmse": mean_row(targetwise_val_metrics)["rmse"],
            "val_ccc": mean_row(targetwise_val_metrics)["ccc"],
            "test_mse": mean_row(targetwise_test_metrics)["mse"],
            "test_rmse": mean_row(targetwise_test_metrics)["rmse"],
            "test_ccc": mean_row(targetwise_test_metrics)["ccc"],
        },
    ])

    summary.to_csv(
        OUTPUT_DIR / "alpha_tuning_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with open(OUTPUT_DIR / "best_alpha.json", "w", encoding="utf-8") as f:
        json.dump({
            "model_a": MODEL_A_NAME,
            "model_b": MODEL_B_NAME,
            "formula": f"pred = alpha * {MODEL_A_NAME} + (1-alpha) * {MODEL_B_NAME}",
            "alpha_grid": {
                "start": ALPHA_START,
                "end": ALPHA_END,
                "step": ALPHA_STEP,
            },
            "global_alpha": {
                f"alpha_{MODEL_A_NAME}": best_alpha,
                f"alpha_{MODEL_B_NAME}": 1.0 - best_alpha,
            },
            "targetwise_alpha": {
                f"alpha_{MODEL_A_NAME}": alpha_by_target,
                f"alpha_{MODEL_B_NAME}": {
                    k: 1.0 - v for k, v in alpha_by_target.items()
                },
            },
            "note": "Alpha was selected on validation predictions only, then applied to test predictions.",
        }, f, indent=2)

    print("\n===== GLOBAL BEST ALPHA =====")
    print(f"alpha_{MODEL_A_NAME}: {best_alpha}")
    print(f"alpha_{MODEL_B_NAME}: {1.0 - best_alpha}")
    print("\nValidation:")
    print(global_val_metrics)
    print("\nTest:")
    print(global_test_metrics)

    print("\n===== TARGET-WISE BEST ALPHA =====")
    print(f"alpha_{MODEL_A_NAME}:", alpha_by_target)
    print(
        f"alpha_{MODEL_B_NAME}:",
        {k: 1.0 - v for k, v in alpha_by_target.items()},
    )
    print("\nValidation:")
    print(targetwise_val_metrics)
    print("\nTest:")
    print(targetwise_test_metrics)

    print("\nSaved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()