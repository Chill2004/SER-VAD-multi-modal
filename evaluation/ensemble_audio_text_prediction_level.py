import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

ENV_DATA_ROOT = Path(os.getenv("SER_DATA_ROOT", "data"))
ENV_OUTPUT_ROOT = Path(os.getenv("SER_OUTPUT_ROOT", "outputs"))



# Configuration
AUDIO_ROOT = Path(
    str(ENV_OUTPUT_ROOT / 'aug_runs_results_fusion_bilstm_cnn1d')
)

TEXT_ROOT = Path(
    str(ENV_OUTPUT_ROOT / 'aug_runs_results_text_only')
)

# Audio predictions
# Best available audio result:
# Silence/reverb target-wise fusion
AUDIO_DIR = AUDIO_ROOT / "ensemble_silence_reverb"

AUDIO_VAL_CSV = AUDIO_DIR / "targetwise_weighted_val_predictions.csv"
AUDIO_TEST_CSV = AUDIO_DIR / "targetwise_weighted_test_predictions.csv"

# Text predictions
# Use the baseline transcript split because audio augmentation does not change the text.
TEXT_SETTING = "baseline"
TEXT_DIR = TEXT_ROOT / TEXT_SETTING

TEXT_VAL_CSV = TEXT_DIR / "text_embedding_val_predictions.csv"
TEXT_TEST_CSV = TEXT_DIR / "text_embedding_predictions.csv"

# Output
OUTPUT_DIR = Path(
    str(ENV_OUTPUT_ROOT / 'aug_runs_results_multimodal_audio_text_ensemble')
) / "audio_targetwise_silence_reverb__text_baseline"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_NAMES = ["valence", "arousal", "dominance"]

TRUE_COLS = [f"{t}_true_norm" for t in TARGET_NAMES]
PRED_COLS = [f"{t}_pred_norm" for t in TARGET_NAMES]

# alpha is the audio weight.
# 1 - alpha is the text weight.
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


# Alignment helpers
def strip_aug_suffix(x):
    """
    Convert augmented utterance ids to original source ids.

    Examples:
    Ses01F_impro01_F000__orig    -> Ses01F_impro01_F000
    Ses01F_impro01_F000__silence -> Ses01F_impro01_F000
    Ses01F_impro01_F000__reverb  -> Ses01F_impro01_F000
    """
    x = str(x)

    if "__" in x:
        return x.split("__")[0]

    return x


def make_key(df):
    """
    Create a robust utterance key for aligning audio and text predictions.

    Audio ensemble prediction usually contains:
    - _ensemble_key

    Text prediction usually contains:
    - source_utt_id
    - utterance_id

    We normalize all keys by stripping augmentation suffix.
    """
    if "source_utt_id" in df.columns:
        return df["source_utt_id"].astype(str).map(strip_aug_suffix)

    if "_ensemble_key" in df.columns:
        return df["_ensemble_key"].astype(str).map(strip_aug_suffix)

    if "_merge_key" in df.columns:
        return df["_merge_key"].astype(str).map(strip_aug_suffix)

    if "_text_key" in df.columns:
        return df["_text_key"].astype(str).map(strip_aug_suffix)

    if "utterance_id" in df.columns:
        return df["utterance_id"].astype(str).map(strip_aug_suffix)

    return pd.Series(
        [f"row_{i:06d}" for i in range(len(df))],
        index=df.index,
    )


def load_prediction_file(csv_path, prefix):
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing prediction file: {csv_path}")

    df = pd.read_csv(csv_path).copy()

    required_cols = TRUE_COLS + PRED_COLS

    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in {csv_path}")

    df["_align_key"] = make_key(df)

    duplicate_count = int(df["_align_key"].duplicated().sum())
    if duplicate_count > 0:
        print(f"[WARN] {csv_path} has duplicated keys: {duplicate_count}")
        print("[WARN] Keeping the first occurrence for each key.")
        df = df.drop_duplicates(subset=["_align_key"], keep="first").copy()

    keep_cols = ["_align_key"] + required_cols

    # Keep useful metadata when available.
    for optional_col in [
        "utterance_id",
        "source_utt_id",
        "transcript",
        "_ensemble_key",
        "_merge_key",
    ]:
        if optional_col in df.columns and optional_col not in keep_cols:
            keep_cols.append(optional_col)

    out = df[keep_cols].copy()

    rename_map = {}

    for c in TRUE_COLS:
        rename_map[c] = f"{c}_{prefix}"

    for c in PRED_COLS:
        rename_map[c] = f"{c}_{prefix}"

    out = out.rename(columns=rename_map)

    return out


def load_and_align(audio_csv, text_csv):
    audio_df = load_prediction_file(audio_csv, prefix="audio")
    text_df = load_prediction_file(text_csv, prefix="text")

    merged = audio_df.merge(
        text_df,
        on="_align_key",
        how="inner",
        suffixes=("_audio_meta", "_text_meta"),
    )

    if len(merged) == 0:
        raise ValueError(
            "Merged prediction file is empty. "
            "Check alignment keys between audio and text predictions."
        )

    # Check target consistency between audio and text.
    mismatch_count = 0

    for c in TRUE_COLS:
        audio_true = merged[f"{c}_audio"].to_numpy(dtype=np.float32)
        text_true = merged[f"{c}_text"].to_numpy(dtype=np.float32)
        diff = np.abs(audio_true - text_true)
        mismatch_count += int(np.sum(diff > 1e-6))

    if mismatch_count > 0:
        print(f"[WARN] True-label mismatch count: {mismatch_count}")
        print("[WARN] Using audio true labels as y_true.")

    y_true = np.stack(
        [merged[f"{c}_audio"].to_numpy(dtype=np.float32) for c in TRUE_COLS],
        axis=1,
    )

    pred_audio = np.stack(
        [merged[f"{c}_audio"].to_numpy(dtype=np.float32) for c in PRED_COLS],
        axis=1,
    )

    pred_text = np.stack(
        [merged[f"{c}_text"].to_numpy(dtype=np.float32) for c in PRED_COLS],
        axis=1,
    )

    return merged, y_true, pred_audio, pred_text


# Fusion
def ensemble_global(pred_audio, pred_text, alpha_audio):
    """
    pred = alpha_audio * audio + (1-alpha_audio) * text
    """
    return alpha_audio * pred_audio + (1.0 - alpha_audio) * pred_text


def ensemble_targetwise(pred_audio, pred_text, alpha_by_target):
    y_pred = np.zeros_like(pred_audio, dtype=np.float32)

    for i, target in enumerate(TARGET_NAMES):
        alpha = alpha_by_target[target]
        y_pred[:, i] = (
            alpha * pred_audio[:, i]
            + (1.0 - alpha) * pred_text[:, i]
        )

    return y_pred


def tune_global_alpha(y_true_val, pred_audio_val, pred_text_val):
    records = []

    best_alpha = None
    best_ccc = -1e9
    best_metrics = None

    for alpha in ALPHAS:
        y_pred = ensemble_global(
            pred_audio_val,
            pred_text_val,
            alpha_audio=alpha,
        )

        metrics = evaluate_split(y_true_val, y_pred)
        mean_row = metrics.loc[metrics["target"] == "mean"].iloc[0]

        record = {
            "alpha_audio": float(alpha),
            "alpha_text": float(1.0 - alpha),
            "val_mse_mean": float(mean_row["mse"]),
            "val_rmse_mean": float(mean_row["rmse"]),
            "val_ccc_mean": float(mean_row["ccc"]),
        }

        for _, row in metrics.iterrows():
            target = row["target"]
            record[f"val_{target}_mse"] = float(row["mse"])
            record[f"val_{target}_rmse"] = float(row["rmse"])
            record[f"val_{target}_ccc"] = float(row["ccc"])

        records.append(record)

        if float(mean_row["ccc"]) > best_ccc:
            best_ccc = float(mean_row["ccc"])
            best_alpha = float(alpha)
            best_metrics = metrics

    return best_alpha, best_metrics, pd.DataFrame(records)


def tune_targetwise_alpha(y_true_val, pred_audio_val, pred_text_val):
    alpha_by_target = {}
    records = []

    for i, target in enumerate(TARGET_NAMES):
        best_alpha = None
        best_ccc = -1e9
        best_rmse = None
        best_mse = None

        for alpha in ALPHAS:
            y_pred_one = (
                alpha * pred_audio_val[:, i]
                + (1.0 - alpha) * pred_text_val[:, i]
            )

            mse = mean_squared_error(y_true_val[:, i], y_pred_one)
            rmse = np.sqrt(mse)
            ccc = ccc_1d(y_true_val[:, i], y_pred_one)

            records.append({
                "target": target,
                "alpha_audio": float(alpha),
                "alpha_text": float(1.0 - alpha),
                "val_mse": float(mse),
                "val_rmse": float(rmse),
                "val_ccc": float(ccc),
            })

            if ccc > best_ccc:
                best_ccc = float(ccc)
                best_alpha = float(alpha)
                best_rmse = float(rmse)
                best_mse = float(mse)

        alpha_by_target[target] = best_alpha

        print(
            f"[TARGET-WISE BEST] {target}: "
            f"alpha_audio={best_alpha:.2f}, "
            f"alpha_text={1.0 - best_alpha:.2f}, "
            f"val_ccc={best_ccc:.6f}, "
            f"val_rmse={best_rmse:.6f}, "
            f"val_mse={best_mse:.6f}"
        )

    return alpha_by_target, pd.DataFrame(records)


def save_predictions(merged, y_true, y_pred, out_csv):
    out = merged[["_align_key"]].copy()

    # Keep optional metadata when available.
    for col in [
        "utterance_id_audio_meta",
        "utterance_id_text_meta",
        "source_utt_id_audio_meta",
        "source_utt_id_text_meta",
        "transcript",
        "transcript_text_meta",
    ]:
        if col in merged.columns and col not in out.columns:
            out[col] = merged[col]

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
def get_mean_row(metrics_df):
    row = metrics_df.loc[metrics_df["target"] == "mean"].iloc[0]
    return {
        "mse": float(row["mse"]),
        "rmse": float(row["rmse"]),
        "ccc": float(row["ccc"]),
    }


def main():
    print("===== AUDIO-TEXT PREDICTION-LEVEL FUSION =====")
    print("Formula:")
    print("  pred = alpha_audio * pred_audio + (1-alpha_audio) * pred_text")
    print()
    print("AUDIO_VAL_CSV :", AUDIO_VAL_CSV)
    print("AUDIO_TEST_CSV:", AUDIO_TEST_CSV)
    print("TEXT_VAL_CSV  :", TEXT_VAL_CSV)
    print("TEXT_TEST_CSV :", TEXT_TEST_CSV)
    print("OUTPUT_DIR    :", OUTPUT_DIR)
    print()
    print("Alpha grid:")
    print(f"  {ALPHA_START} -> {ALPHA_END}, step={ALPHA_STEP}")
    print()

    # Load and align validation/test predictions
    print("Loading validation predictions...")
    val_merged, y_val, audio_val, text_val = load_and_align(
        AUDIO_VAL_CSV,
        TEXT_VAL_CSV,
    )

    print("Loading test predictions...")
    test_merged, y_test, audio_test, text_test = load_and_align(
        AUDIO_TEST_CSV,
        TEXT_TEST_CSV,
    )

    print("Validation rows:", len(val_merged))
    print("Test rows      :", len(test_merged))

    # Evaluate original audio/text alone after alignment
    audio_val_metrics = evaluate_split(y_val, audio_val)
    audio_test_metrics = evaluate_split(y_test, audio_test)

    text_val_metrics = evaluate_split(y_val, text_val)
    text_test_metrics = evaluate_split(y_test, text_test)

    audio_val_metrics.to_csv(
        OUTPUT_DIR / "audio_aligned_val_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    audio_test_metrics.to_csv(
        OUTPUT_DIR / "audio_aligned_test_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    text_val_metrics.to_csv(
        OUTPUT_DIR / "text_aligned_val_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    text_test_metrics.to_csv(
        OUTPUT_DIR / "text_aligned_test_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Fixed 0.5 / 0.5
    fixed_alpha = 0.5

    fixed_val_pred = ensemble_global(
        audio_val,
        text_val,
        alpha_audio=fixed_alpha,
    )

    fixed_test_pred = ensemble_global(
        audio_test,
        text_test,
        alpha_audio=fixed_alpha,
    )

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

    save_predictions(
        val_merged,
        y_val,
        fixed_val_pred,
        OUTPUT_DIR / "fixed_0505_val_predictions.csv",
    )
    save_predictions(
        test_merged,
        y_test,
        fixed_test_pred,
        OUTPUT_DIR / "fixed_0505_test_predictions.csv",
    )

    # Global alpha tuned on validation mean CCC
    best_global_alpha, best_global_val_metrics, global_search_df = tune_global_alpha(
        y_val,
        audio_val,
        text_val,
    )

    global_search_df.to_csv(
        OUTPUT_DIR / "global_alpha_search.csv",
        index=False,
        encoding="utf-8-sig",
    )

    global_val_pred = ensemble_global(
        audio_val,
        text_val,
        alpha_audio=best_global_alpha,
    )

    global_test_pred = ensemble_global(
        audio_test,
        text_test,
        alpha_audio=best_global_alpha,
    )

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

    save_predictions(
        val_merged,
        y_val,
        global_val_pred,
        OUTPUT_DIR / "global_weighted_val_predictions.csv",
    )
    save_predictions(
        test_merged,
        y_test,
        global_test_pred,
        OUTPUT_DIR / "global_weighted_test_predictions.csv",
    )

    # Target-wise alpha tuned on validation CCC per target
    alpha_by_target, targetwise_search_df = tune_targetwise_alpha(
        y_val,
        audio_val,
        text_val,
    )

    targetwise_search_df.to_csv(
        OUTPUT_DIR / "targetwise_alpha_search.csv",
        index=False,
        encoding="utf-8-sig",
    )

    targetwise_val_pred = ensemble_targetwise(
        audio_val,
        text_val,
        alpha_by_target,
    )

    targetwise_test_pred = ensemble_targetwise(
        audio_test,
        text_test,
        alpha_by_target,
    )

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

    save_predictions(
        val_merged,
        y_val,
        targetwise_val_pred,
        OUTPUT_DIR / "targetwise_weighted_val_predictions.csv",
    )
    save_predictions(
        test_merged,
        y_test,
        targetwise_test_pred,
        OUTPUT_DIR / "targetwise_weighted_test_predictions.csv",
    )

    # Summary
    summary_rows = []

    summary_rows.append({
        "method": "audio_only_aligned",
        "alpha_audio": 1.0,
        "alpha_text": 0.0,
        "alpha_by_target_audio": "",
        "alpha_by_target_text": "",
        "targetwise": False,
        "val_mse": get_mean_row(audio_val_metrics)["mse"],
        "val_rmse": get_mean_row(audio_val_metrics)["rmse"],
        "val_ccc": get_mean_row(audio_val_metrics)["ccc"],
        "test_mse": get_mean_row(audio_test_metrics)["mse"],
        "test_rmse": get_mean_row(audio_test_metrics)["rmse"],
        "test_ccc": get_mean_row(audio_test_metrics)["ccc"],
    })

    summary_rows.append({
        "method": "text_only_aligned",
        "alpha_audio": 0.0,
        "alpha_text": 1.0,
        "alpha_by_target_audio": "",
        "alpha_by_target_text": "",
        "targetwise": False,
        "val_mse": get_mean_row(text_val_metrics)["mse"],
        "val_rmse": get_mean_row(text_val_metrics)["rmse"],
        "val_ccc": get_mean_row(text_val_metrics)["ccc"],
        "test_mse": get_mean_row(text_test_metrics)["mse"],
        "test_rmse": get_mean_row(text_test_metrics)["rmse"],
        "test_ccc": get_mean_row(text_test_metrics)["ccc"],
    })

    summary_rows.append({
        "method": "fixed_0505",
        "alpha_audio": 0.5,
        "alpha_text": 0.5,
        "alpha_by_target_audio": "",
        "alpha_by_target_text": "",
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
        "alpha_audio": best_global_alpha,
        "alpha_text": 1.0 - best_global_alpha,
        "alpha_by_target_audio": "",
        "alpha_by_target_text": "",
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
        "alpha_audio": "",
        "alpha_text": "",
        "alpha_by_target_audio": json.dumps(alpha_by_target),
        "alpha_by_target_text": json.dumps({
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
        OUTPUT_DIR / "audio_text_ensemble_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with open(OUTPUT_DIR / "audio_text_ensemble_weights.json", "w", encoding="utf-8") as f:
        json.dump({
            "formula": "pred = alpha_audio * pred_audio + (1-alpha_audio) * pred_text",
            "audio_prediction": {
                "val_csv": str(AUDIO_VAL_CSV),
                "test_csv": str(AUDIO_TEST_CSV),
            },
            "text_prediction": {
                "val_csv": str(TEXT_VAL_CSV),
                "test_csv": str(TEXT_TEST_CSV),
            },
            "alpha_grid": {
                "start": ALPHA_START,
                "end": ALPHA_END,
                "step": ALPHA_STEP,
            },
            "fixed_0505": {
                "alpha_audio": 0.5,
                "alpha_text": 0.5,
            },
            "global_weighted": {
                "alpha_audio": best_global_alpha,
                "alpha_text": 1.0 - best_global_alpha,
            },
            "targetwise_weighted": {
                "alpha_audio": alpha_by_target,
                "alpha_text": {
                    k: 1.0 - v for k, v in alpha_by_target.items()
                },
            },
            "note": (
                "All alpha weights were selected using validation predictions only, "
                "then applied once to test predictions."
            ),
        }, f, indent=2)

    # Print results
    print("\n===== AUDIO ONLY ALIGNED TEST =====")
    print(audio_test_metrics)

    print("\n===== TEXT ONLY ALIGNED TEST =====")
    print(text_test_metrics)

    print("\n===== FIXED 0.5 AUDIO / 0.5 TEXT TEST =====")
    print(fixed_test_metrics)

    print("\n===== GLOBAL WEIGHTED AUDIO-TEXT TEST =====")
    print("best_global_alpha_audio:", best_global_alpha)
    print("best_global_alpha_text :", 1.0 - best_global_alpha)
    print(global_test_metrics)

    print("\n===== TARGET-WISE WEIGHTED AUDIO-TEXT TEST =====")
    print("alpha_audio_by_target:", alpha_by_target)
    print(
        "alpha_text_by_target :",
        {k: 1.0 - v for k, v in alpha_by_target.items()},
    )
    print(targetwise_test_metrics)

    print("\nSaved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()