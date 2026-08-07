import os
import gc
import json
import time
import warnings
from pathlib import Path

ENV_DATA_ROOT = Path(os.getenv("SER_DATA_ROOT", "data"))
ENV_OUTPUT_ROOT = Path(os.getenv("SER_OUTPUT_ROOT", "outputs"))


warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error


# Configuration
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
    str(ENV_OUTPUT_ROOT / 'fine_tune_iemocap_model_level_baseline_on_msp_duration_matched')
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLS = ["valence_norm", "arousal_norm", "dominance_norm"]
TARGET_NAMES = ["valence", "arousal", "dominance"]

PAD_VALUE = 0.0
RANDOM_STATE = 42
BATCH_SIZE = 16
EPOCHS = 40
LEARNING_RATE = 1e-5
MIN_LR = 1e-7

# fine-tuning strategy
# False = fine-tune all layers with small LR.
# True  = freeze audio/text branches and fine-tune only fusion/output layers.
FREEZE_BRANCHES = False

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# metrics / losses / custom layers
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


# Data helpers
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
    return np.load(str(path)).astype(np.float32)


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
    # Reuse IEMOCAP normalization statistics and sequence limits during fine-tuning.
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

    return (
        kept_df,
        np.ascontiguousarray(X_wav.astype("float32")),
        np.ascontiguousarray(X_mfcc.astype("float32")),
        np.ascontiguousarray(X_text.astype("float32")),
        np.ascontiguousarray(y.astype("float32")),
    )


def make_dataset(X_wav, X_mfcc, X_text, y, training):
    ds = tf.data.Dataset.from_tensor_slices(((X_wav, X_mfcc, X_text), y))

    if training:
        ds = ds.shuffle(min(len(y), 4096), seed=RANDOM_STATE)

    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


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


def set_trainable_for_finetune(model):
    if not FREEZE_BRANCHES:
        for layer in model.layers:
            layer.trainable = True
        print("Fine-tuning mode: all layers trainable")
        return

    trainable_keywords = [
        "fusion",
        "vad_output",
        "model_level_audio_text_concat",
    ]

    for layer in model.layers:
        layer.trainable = any(key in layer.name for key in trainable_keywords)

    print("Fine-tuning mode: fusion/output layers only")
    for layer in model.layers:
        if layer.trainable:
            print("trainable:", layer.name)


# Main
def main():
    print("MODEL_PATH      :", MODEL_PATH)
    print("NORM_STATS_PATH :", NORM_STATS_PATH)
    print("MSP_FEATURE_ROOT:", MSP_FEATURE_ROOT)
    print("OUTPUT_DIR      :", OUTPUT_DIR)
    print("LEARNING_RATE   :", LEARNING_RATE)
    print("EPOCHS          :", EPOCHS)
    print("BATCH_SIZE      :", BATCH_SIZE)
    print("FREEZE_BRANCHES :", FREEZE_BRANCHES)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")
    if not NORM_STATS_PATH.exists():
        raise FileNotFoundError(f"Missing norm stats: {NORM_STATS_PATH}")

    stats = np.load(NORM_STATS_PATH, allow_pickle=True)
    print("\n===== IEMOCAP NORM STATS KEYS =====")
    print(stats.files)

    # Load MSP split features.
    train_df_raw = load_merged_split("train")
    val_df_raw = load_merged_split("val")
    test_df_raw = load_merged_split("test")

    train_df, X_wav_train, X_mfcc_train, X_text_train, y_train = build_inputs_from_df(train_df_raw, stats)
    val_df, X_wav_val, X_mfcc_val, X_text_val, y_val = build_inputs_from_df(val_df_raw, stats)
    test_df, X_wav_test, X_mfcc_test, X_text_test, y_test = build_inputs_from_df(test_df_raw, stats)

    print("\n===== INPUT SHAPES =====")
    print("train:", X_wav_train.shape, X_mfcc_train.shape, X_text_train.shape, y_train.shape)
    print("val  :", X_wav_val.shape, X_mfcc_val.shape, X_text_val.shape, y_val.shape)
    print("test :", X_wav_test.shape, X_mfcc_test.shape, X_text_test.shape, y_test.shape)

    train_ds = make_dataset(X_wav_train, X_mfcc_train, X_text_train, y_train, training=True)
    val_ds = make_dataset(X_wav_val, X_mfcc_val, X_text_val, y_val, training=False)

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

    set_trainable_for_finetune(model)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=mse_ccc_loss,
        metrics=[
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
            ccc_tf,
        ],
    )

    best_model_path = OUTPUT_DIR / "fine_tuned_best_model.keras"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_ccc_tf",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_ccc_tf",
            mode="max",
            factor=0.5,
            patience=3,
            min_delta=0.001,
            min_lr=MIN_LR,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_ccc_tf",
            mode="max",
            patience=8,
            min_delta=0.001,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            filename=str(OUTPUT_DIR / "fine_tune_history.csv"),
            append=False,
        ),
    ]

    fit_t0 = time.perf_counter()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        verbose=1,
        callbacks=callbacks,
    )

    fit_seconds = time.perf_counter() - fit_t0
    print("fit_seconds:", fit_seconds)

    # Reload best checkpoint for final evaluation.
    del train_ds, val_ds, history
    gc.collect()
    tf.keras.backend.clear_session()
    gc.collect()

    model = tf.keras.models.load_model(
        best_model_path,
        custom_objects={
            "mse_ccc_loss": mse_ccc_loss,
            "ccc_tf": ccc_tf,
            "SequenceMaskFromPadValue": SequenceMaskFromPadValue,
            "MaskedGlobalAveragePooling1D": MaskedGlobalAveragePooling1D,
            "MaskedGlobalMaxPooling1D": MaskedGlobalMaxPooling1D,
        },
    )

    all_metrics = []

    split_payloads = [
        ("train", train_df, X_wav_train, X_mfcc_train, X_text_train, y_train),
        ("val", val_df, X_wav_val, X_mfcc_val, X_text_val, y_val),
        ("test", test_df, X_wav_test, X_mfcc_test, X_text_test, y_test),
    ]

    for split, split_df, X_wav, X_mfcc, X_text, y in split_payloads:
        y_pred = predict_in_batches(model, X_wav, X_mfcc, X_text)
        metrics = evaluate_split(y, y_pred)

        print(f"\n===== {split.upper()} FINE-TUNED RESULTS =====")
        print(metrics)

        metrics.to_csv(
            OUTPUT_DIR / f"fine_tuned_{split}_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )

        save_predictions(
            split_df,
            y,
            y_pred,
            OUTPUT_DIR / f"fine_tuned_{split}_predictions.csv",
        )

        metrics_copy = metrics.copy()
        metrics_copy["split"] = split
        all_metrics.append(metrics_copy)

    all_metrics_df = pd.concat(all_metrics, ignore_index=True)
    all_metrics_df.to_csv(
        OUTPUT_DIR / "fine_tuned_all_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Keep final model too.
    model.save(OUTPUT_DIR / "fine_tuned_model_final.keras")

    summary = {
        "experiment": "fine_tune_iemocap_model_level_baseline_on_msp_duration_matched",
        "source_iemocap_model_path": str(MODEL_PATH),
        "source_iemocap_norm_stats_path": str(NORM_STATS_PATH),
        "msp_feature_root": str(MSP_FEATURE_ROOT),
        "output_dir": str(OUTPUT_DIR),
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epochs_requested": EPOCHS,
        "freeze_branches": FREEZE_BRANCHES,
        "fit_seconds": fit_seconds,
        "important_note": (
            "This is fine-tuning, not external evaluation. The model is initialized from IEMOCAP weights, "
            "uses IEMOCAP normalization statistics and max lengths, then is fine-tuned on MSP-Podcast "
            "duration-matched train split with validation-based checkpointing."
        ),
    }

    with open(OUTPUT_DIR / "fine_tune_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
