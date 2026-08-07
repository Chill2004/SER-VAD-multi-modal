import os
import gc
import json
import warnings
from pathlib import Path

ENV_DATA_ROOT = Path(os.getenv("SER_DATA_ROOT", "data"))
ENV_OUTPUT_ROOT = Path(os.getenv("SER_OUTPUT_ROOT", "outputs"))


warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error


# paths
IEMOCAP_MODEL_DIR = Path(
    str(ENV_OUTPUT_ROOT / 'aug_runs_results_multimodal_model_level/baseline')
)

MODEL_PATH = IEMOCAP_MODEL_DIR / "model_level_best_model.keras"
NORM_STATS_PATH = IEMOCAP_MODEL_DIR / "model_level_norm_stats.npz"

MSP_FEATURE_ROOT = Path(
    str(ENV_OUTPUT_ROOT / 'msp_podcast_iemocap_duration_matched')
)

WAV2VEC2_DIR = MSP_FEATURE_ROOT / "wav2vec2_features"
MFCC_DIR = MSP_FEATURE_ROOT / "mfcc_features"
TEXT_DIR = MSP_FEATURE_ROOT / "text_distilbert_features"

OUTPUT_DIR = Path(
    str(ENV_OUTPUT_ROOT / 'external_eval_iemocap_model_level_baseline_on_msp_duration_matched')
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLS = ["valence_norm", "arousal_norm", "dominance_norm"]
TARGET_NAMES = ["valence", "arousal", "dominance"]

PAD_VALUE = 0.0
BATCH_SIZE = 16


# Metrics and loss
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


def ccc_tf(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    mean_true = tf.reduce_mean(y_true, axis=0)
    mean_pred = tf.reduce_mean(y_pred, axis=0)

    var_true = tf.reduce_mean(tf.square(y_true - mean_true), axis=0)
    var_pred = tf.reduce_mean(tf.square(y_pred - mean_pred), axis=0)

    cov = tf.reduce_mean((y_true - mean_true) * (y_pred - mean_pred), axis=0)

    ccc = (2.0 * cov) / (
        var_true + var_pred + tf.square(mean_true - mean_pred) + 1e-8
    )
    return tf.reduce_mean(ccc)


def mse_ccc_loss(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    ccc_loss = 1.0 - ccc_tf(y_true, y_pred)
    return 0.5 * mse + 0.5 * ccc_loss


@tf.keras.utils.register_keras_serializable()
class SequenceMaskFromPadValue(tf.keras.layers.Layer):
    def __init__(self, pad_value=0.0, **kwargs):
        super().__init__(**kwargs)
        self.pad_value = pad_value

    def call(self, x):
        return tf.reduce_any(tf.not_equal(x, self.pad_value), axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"pad_value": self.pad_value})
        return config


@tf.keras.utils.register_keras_serializable()
class MaskedGlobalAveragePooling1D(tf.keras.layers.Layer):
    def call(self, inputs):
        x, mask = inputs
        mask = tf.cast(mask, x.dtype)
        mask = tf.expand_dims(mask, axis=-1)

        x = x * mask

        denom = tf.reduce_sum(mask, axis=1)
        denom = tf.maximum(denom, tf.constant(1.0, dtype=x.dtype))

        return tf.reduce_sum(x, axis=1) / denom


@tf.keras.utils.register_keras_serializable()
class MaskedGlobalMaxPooling1D(tf.keras.layers.Layer):
    def call(self, inputs):
        x, mask = inputs

        neg_inf = tf.constant(-1e9, dtype=x.dtype)
        mask = tf.expand_dims(mask, axis=-1)

        x = tf.where(mask, x, neg_inf)

        return tf.reduce_max(x, axis=1)


# helpers
def get_npz_value(stats, possible_keys, default=None):
    for key in possible_keys:
        if key in stats.files:
            return stats[key]
    if default is not None:
        return default
    raise KeyError(f"None of these keys found in norm stats: {possible_keys}")


def make_merge_key(df):
    if "utterance_id" in df.columns:
        return df["utterance_id"].astype(str)

    if "source_utt_id" in df.columns and "aug_tag" in df.columns:
        return df["source_utt_id"].astype(str) + "__" + df["aug_tag"].astype(str)

    if "source_file" in df.columns:
        return df["source_file"].astype(str).map(lambda x: Path(x).stem)

    return pd.Series([f"row_{i:06d}" for i in range(len(df))], index=df.index)


def load_feature_manifest(feature_dir, split, feature_col, rename_col):
    p = feature_dir / f"{split}_manifest.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing manifest: {p}")

    df = pd.read_csv(p).copy()

    if feature_col not in df.columns:
        raise ValueError(f"Missing {feature_col} in {p}")

    for c in TARGET_COLS:
        if c not in df.columns:
            raise ValueError(f"Missing target column {c} in {p}")

    df["_merge_key"] = make_merge_key(df)
    df = df.dropna(subset=[feature_col] + TARGET_COLS).copy()
    df = df[df[feature_col].astype(str).str.len() > 0].copy()

    df = df.rename(columns={feature_col: rename_col})

    return df.reset_index(drop=True)


def load_merged_split(split):
    wav_df = load_feature_manifest(
        WAV2VEC2_DIR,
        split,
        feature_col="seq_feat_path",
        rename_col="wav2vec2_seq_feat_path",
    )

    mfcc_df = load_feature_manifest(
        MFCC_DIR,
        split,
        feature_col="seq_feat_path",
        rename_col="mfcc_seq_feat_path",
    )

    text_df = load_feature_manifest(
        TEXT_DIR,
        split,
        feature_col="text_feat_path",
        rename_col="text_feat_path",
    )

    mfcc_keep = ["_merge_key", "mfcc_seq_feat_path"] + TARGET_COLS
    text_keep = ["_merge_key", "text_feat_path"] + TARGET_COLS

    # Keep useful metadata from text or wav manifest if available.
    metadata_cols = [
        "utterance_id",
        "audio_path",
        "duration",
        "transcript",
        "major_emotion",
        "gender",
        "source_file",
    ]

    for col in metadata_cols:
        if col in text_df.columns and col not in text_keep:
            text_keep.append(col)

    merged = wav_df.merge(
        mfcc_df[mfcc_keep],
        on="_merge_key",
        how="inner",
        suffixes=("", "_mfcc"),
    )

    merged = merged.merge(
        text_df[text_keep],
        on="_merge_key",
        how="inner",
        suffixes=("", "_text"),
    )

    print(f"\n===== {split.upper()} MERGE =====")
    print("wav rows :", len(wav_df))
    print("mfcc rows:", len(mfcc_df))
    print("text rows:", len(text_df))
    print("merged   :", len(merged))

    return merged.reset_index(drop=True)


def load_feature_array(path):
    arr = np.load(str(path)).astype(np.float32)
    return arr


def pad_and_normalize_sequences(seq_list, max_len, feat_mean, feat_std):
    n = len(seq_list)
    feat_dim = seq_list[0].shape[1]

    X = np.full((n, max_len, feat_dim), PAD_VALUE, dtype=np.float32)

    for i, seq in enumerate(seq_list):
        seq = seq[:max_len]
        seq = (seq - feat_mean) / feat_std

        L = seq.shape[0]
        X[i, :L, :] = seq

    return X.astype(np.float32)


def normalize_vectors(vec_list, feat_mean, feat_std):
    X = np.stack(vec_list, axis=0).astype(np.float32)
    X = (X - feat_mean) / feat_std
    return X.astype(np.float32)


def build_inputs_from_df(df, stats):
    wav_max_len = int(get_npz_value(stats, ["wav_max_len", "WAV_MAX_LEN"]))
    mfcc_max_len = int(get_npz_value(stats, ["mfcc_max_len", "MFCC_MAX_LEN"]))

    wav_feat_mean = get_npz_value(stats, ["wav_feat_mean", "wav2vec2_feat_mean"])
    wav_feat_std = get_npz_value(stats, ["wav_feat_std", "wav2vec2_feat_std"])

    mfcc_feat_mean = get_npz_value(stats, ["mfcc_feat_mean"])
    mfcc_feat_std = get_npz_value(stats, ["mfcc_feat_std"])

    text_feat_mean = get_npz_value(stats, ["text_feat_mean"])
    text_feat_std = get_npz_value(stats, ["text_feat_std"])

    wav_seqs = []
    mfcc_seqs = []
    text_feats = []
    keep_idx = []

    for i, row in df.iterrows():
        try:
            wav_x = load_feature_array(row["wav2vec2_seq_feat_path"])
            mfcc_x = load_feature_array(row["mfcc_seq_feat_path"])
            text_x = load_feature_array(row["text_feat_path"])

            if wav_x.ndim != 2:
                continue
            if mfcc_x.ndim != 2:
                continue
            if text_x.ndim != 1:
                continue

            wav_seqs.append(wav_x)
            mfcc_seqs.append(mfcc_x)
            text_feats.append(text_x)
            keep_idx.append(i)

        except Exception as e:
            print("[WARN] Failed loading row", i, e)

    kept_df = df.loc[keep_idx].reset_index(drop=True)

    X_wav = pad_and_normalize_sequences(
        wav_seqs,
        wav_max_len,
        wav_feat_mean,
        wav_feat_std,
    )

    X_mfcc = pad_and_normalize_sequences(
        mfcc_seqs,
        mfcc_max_len,
        mfcc_feat_mean,
        mfcc_feat_std,
    )

    X_text = normalize_vectors(
        text_feats,
        text_feat_mean,
        text_feat_std,
    )

    y = kept_df[TARGET_COLS].to_numpy(dtype=np.float32)

    X_wav = np.ascontiguousarray(X_wav.astype("float32"))
    X_mfcc = np.ascontiguousarray(X_mfcc.astype("float32"))
    X_text = np.ascontiguousarray(X_text.astype("float32"))
    y = np.ascontiguousarray(y.astype("float32"))

    return kept_df, X_wav, X_mfcc, X_text, y


def predict_in_batches(model, X_wav, X_mfcc, X_text, batch_size=BATCH_SIZE):
    preds = []

    for start in range(0, len(X_wav), batch_size):
        end = start + batch_size
        batch_pred = model.predict_on_batch(
            [X_wav[start:end], X_mfcc[start:end], X_text[start:end]]
        )
        preds.append(np.asarray(batch_pred, dtype=np.float32))

    return np.concatenate(preds, axis=0)


def evaluate_split(y_true, y_pred):
    rows = []

    for i, name in enumerate(TARGET_NAMES):
        mse = mean_squared_error(y_true[:, i], y_pred[:, i])
        rmse = float(np.sqrt(mse))
        ccc = ccc_1d(y_true[:, i], y_pred[:, i])

        rows.append({
            "target": name,
            "mse": float(mse),
            "rmse": rmse,
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


def save_predictions(base_df, y_true, y_pred, out_csv):
    out_df = base_df.copy()

    for i, name in enumerate(TARGET_NAMES):
        out_df[f"{name}_true_norm"] = y_true[:, i]
        out_df[f"{name}_pred_norm"] = y_pred[:, i]

    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")


# Main
def main():
    print("MODEL_PATH     :", MODEL_PATH)
    print("NORM_STATS_PATH:", NORM_STATS_PATH)
    print("MSP_FEATURE_ROOT:", MSP_FEATURE_ROOT)
    print("OUTPUT_DIR     :", OUTPUT_DIR)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")
    if not NORM_STATS_PATH.exists():
        raise FileNotFoundError(f"Missing norm stats: {NORM_STATS_PATH}")

    stats = np.load(NORM_STATS_PATH, allow_pickle=True)

    print("\n===== NORM STATS KEYS =====")
    print(stats.files)

    tf.keras.backend.clear_session()
    gc.collect()

    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "mse_ccc_loss": mse_ccc_loss,
            "ccc_tf": ccc_tf,
            "SequenceMaskFromPadValue": SequenceMaskFromPadValue,
            "MaskedGlobalAveragePooling1D": MaskedGlobalAveragePooling1D,
            "MaskedGlobalMaxPooling1D": MaskedGlobalMaxPooling1D,
        },
    )

    all_metrics = []

    for split in ["train", "val", "test"]:
        raw_df = load_merged_split(split)
        kept_df, X_wav, X_mfcc, X_text, y = build_inputs_from_df(raw_df, stats)

        print(f"\n===== {split.upper()} INPUT SHAPES =====")
        print("rows  :", len(kept_df))
        print("X_wav :", X_wav.shape)
        print("X_mfcc:", X_mfcc.shape)
        print("X_text:", X_text.shape)
        print("y     :", y.shape)

        y_pred = predict_in_batches(model, X_wav, X_mfcc, X_text)

        metrics = evaluate_split(y, y_pred)
        print(f"\n===== {split.upper()} EXTERNAL EVAL RESULTS =====")
        print(metrics)

        metrics.to_csv(
            OUTPUT_DIR / f"msp_{split}_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )

        save_predictions(
            kept_df,
            y,
            y_pred,
            OUTPUT_DIR / f"msp_{split}_predictions.csv",
        )

        metrics_copy = metrics.copy()
        metrics_copy["split"] = split
        all_metrics.append(metrics_copy)

    all_metrics_df = pd.concat(all_metrics, ignore_index=True)
    all_metrics_df.to_csv(
        OUTPUT_DIR / "external_eval_all_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "experiment": "external_eval_iemocap_model_level_baseline_on_msp_duration_matched",
        "iemocap_model_path": str(MODEL_PATH),
        "iemocap_norm_stats_path": str(NORM_STATS_PATH),
        "msp_feature_root": str(MSP_FEATURE_ROOT),
        "output_dir": str(OUTPUT_DIR),
        "important_note": (
            "This is external evaluation only. The model is trained on IEMOCAP "
            "and directly evaluated on MSP-Podcast duration-matched features. "
            "Normalization statistics and max lengths are loaded from the IEMOCAP training experiment."
        ),
    }

    with open(OUTPUT_DIR / "external_eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()