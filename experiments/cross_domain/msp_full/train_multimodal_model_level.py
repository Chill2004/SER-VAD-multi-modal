import os
import gc
import json
import time
import argparse
import warnings
from pathlib import Path

ENV_DATA_ROOT = Path(os.getenv("SER_DATA_ROOT", "data"))
ENV_OUTPUT_ROOT = Path(os.getenv("SER_OUTPUT_ROOT", "outputs"))


# Parse CLI arguments before importing TensorFlow.
parser = argparse.ArgumentParser()

parser.add_argument(
    "--wav2vec2_root",
    type=str,
    default=str(ENV_OUTPUT_ROOT / 'msp_podcast_full/wav2vec2_features'),
    help="Folder containing Wav2Vec2 train/val/test manifests.",
)
parser.add_argument(
    "--mfcc_root",
    type=str,
    default=str(ENV_OUTPUT_ROOT / 'msp_podcast_full/mfcc_features'),
    help="Folder containing MFCC train/val/test manifests.",
)
parser.add_argument(
    "--text_root",
    type=str,
    default=str(ENV_OUTPUT_ROOT / 'msp_podcast_full/text_distilbert_features'),
    help="Folder containing DistilBERT text train/val/test manifests.",
)
parser.add_argument(
    "--out_dir",
    type=str,
    default=str(ENV_OUTPUT_ROOT / 'msp_podcast_full_results/multimodal_model_level'),
    help="Output folder for model-level multi-modal fusion results.",
)

parser.add_argument("--gpu", type=str, default="1")
parser.add_argument("--wav_max_len", type=int, default=512)
parser.add_argument("--mfcc_max_len", type=int, default=1024)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--learning_rate", type=float, default=1e-4)

args = parser.parse_args()

# Runtime setup
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_squared_error


gpus = tf.config.list_physical_devices("GPU")
print("Visible GPUs:", gpus)

if gpus:
    try:
        tf.config.set_visible_devices(gpus[0], "GPU")
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("Using GPU:", gpus[0])
        print("Logical GPUs:", tf.config.list_logical_devices("GPU"))
    except RuntimeError as e:
        print("GPU setup error:", e)
else:
    print("No GPU found. TensorFlow will use CPU.")


# Configuration
WAV2VEC2_ROOT = Path(args.wav2vec2_root)
MFCC_ROOT = Path(args.mfcc_root)
TEXT_ROOT = Path(args.text_root)
OUTPUT_DIR = Path(args.out_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ORIGINAL_SEQ_FEATURE_COL = "seq_feat_path"
ORIGINAL_TEXT_FEATURE_COL = "text_feat_path"
WAV_FEATURE_COL = "wav2vec2_seq_feat_path"
MFCC_FEATURE_COL = "mfcc_seq_feat_path"
TEXT_FEATURE_COL = "text_feat_path"

TARGET_COLS = ["valence_norm", "arousal_norm", "dominance_norm"]
TARGET_NAMES = ["valence", "arousal", "dominance"]

RANDOM_STATE = 42
PAD_VALUE = 0.0
WAV_MAX_LEN = int(args.wav_max_len)
MFCC_MAX_LEN = int(args.mfcc_max_len)
BATCH_SIZE = int(args.batch_size)
EPOCHS = int(args.epochs)
LEARNING_RATE = float(args.learning_rate)

# Wav2Vec2 branch.
WAV_LSTM_UNITS_1 = 256
WAV_LSTM_UNITS_2 = 128
WAV_DROPOUT = 0.25
WAV_BRANCH_DENSE = 128

# MFCC branch.
MFCC_BASE_FILTERS = 128
MFCC_DILATED_FILTERS = 128
MFCC_FINAL_FILTERS = 256
MFCC_KERNEL_SIZE_FIRST = 5
MFCC_KERNEL_SIZE_DILATED = 3
MFCC_DILATION_RATES = [1, 2, 4, 8]
MFCC_CONV_DROPOUT = 0.20
MFCC_BRANCH_DENSE = 128

# Match the DistilBERT text branch used in the IEMOCAP runs.
TEXT_DENSE_1 = 256
TEXT_DENSE_2 = 128
TEXT_DENSE_3 = 64
TEXT_DROPOUT = 0.30

# Fusion head
FUSION_DENSE_1 = 256
FUSION_DENSE_2 = 128
FUSION_DROPOUT = 0.35
BRANCH_DROPOUT = 0.25
L2_REG = 1e-5

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# Metrics and loss
def denormalize_vad(x):
    x = np.asarray(x, dtype=np.float32)
    return ((x + 1.0) / 2.0) * 4.0 + 1.0


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


# Manifest helpers
def infer_key(df):
    for key in ["utterance_id", "source_utt_id", "source_file", "audio_path"]:
        if key in df.columns:
            return key
    raise ValueError("Cannot infer merge key from manifest.")


def load_sequence_manifest(root: Path, split: str, rename_col: str):
    csv_path = root / f"{split}_manifest.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing manifest: {csv_path}")

    df = pd.read_csv(csv_path).copy()
    required = [ORIGINAL_SEQ_FEATURE_COL] + TARGET_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")

    df = df.dropna(subset=required).copy()
    df = df[df[ORIGINAL_SEQ_FEATURE_COL].astype(str).str.len() > 0].copy()

    exists_mask = df[ORIGINAL_SEQ_FEATURE_COL].astype(str).map(lambda p: Path(p).exists())
    missing_count = int((~exists_mask).sum())
    if missing_count > 0:
        print(f"[WARN] Missing sequence features in {csv_path}: {missing_count}")
    df = df[exists_mask].copy()

    key = infer_key(df)
    df = df.rename(columns={ORIGINAL_SEQ_FEATURE_COL: rename_col})

    keep_cols = [key, rename_col] + TARGET_COLS
    for col in [
        "audio_path",
        "duration",
        "sampling_rate",
        "transcript",
        "major_emotion",
        "gender",
        "source_file",
        "split",
    ]:
        if col in df.columns and col not in keep_cols:
            keep_cols.append(col)

    return df[keep_cols].reset_index(drop=True), key


def load_text_manifest(root: Path, split: str):
    csv_path = root / f"{split}_manifest.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing manifest: {csv_path}")

    df = pd.read_csv(csv_path).copy()
    required = [ORIGINAL_TEXT_FEATURE_COL] + TARGET_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")

    df = df.dropna(subset=required).copy()
    df = df[df[ORIGINAL_TEXT_FEATURE_COL].astype(str).str.len() > 0].copy()

    exists_mask = df[ORIGINAL_TEXT_FEATURE_COL].astype(str).map(lambda p: Path(p).exists())
    missing_count = int((~exists_mask).sum())
    if missing_count > 0:
        print(f"[WARN] Missing text features in {csv_path}: {missing_count}")
    df = df[exists_mask].copy()

    key = infer_key(df)
    keep_cols = [key, ORIGINAL_TEXT_FEATURE_COL] + TARGET_COLS
    for col in [
        "transcript",
        "major_emotion",
        "gender",
        "source_file",
        "split",
    ]:
        if col in df.columns and col not in keep_cols:
            keep_cols.append(col)

    return df[keep_cols].reset_index(drop=True), key


def load_merged_split(split: str):
    wav_df, wav_key = load_sequence_manifest(WAV2VEC2_ROOT, split, WAV_FEATURE_COL)
    mfcc_df, mfcc_key = load_sequence_manifest(MFCC_ROOT, split, MFCC_FEATURE_COL)
    text_df, text_key = load_text_manifest(TEXT_ROOT, split)

    key = wav_key

    if mfcc_key != key:
        print(f"[WARN] Different key names: wav={key}, mfcc={mfcc_key}. Renaming MFCC key to wav key.")
        mfcc_df = mfcc_df.rename(columns={mfcc_key: key})
    if text_key != key:
        print(f"[WARN] Different key names: wav={key}, text={text_key}. Renaming text key to wav key.")
        text_df = text_df.rename(columns={text_key: key})

    mfcc_keep = [key, MFCC_FEATURE_COL] + TARGET_COLS
    text_keep = [key, TEXT_FEATURE_COL] + TARGET_COLS
    for col in ["transcript", "major_emotion", "gender", "source_file", "split"]:
        if col in text_df.columns and col not in text_keep:
            text_keep.append(col)

    merged = wav_df.merge(
        mfcc_df[mfcc_keep],
        on=key,
        how="inner",
        suffixes=("_wav", "_mfcc"),
    )

    merged = merged.merge(
        text_df[text_keep],
        on=key,
        how="inner",
        suffixes=("", "_text"),
    )

    # After second merge, text target columns may be unsuffixed if no collision happened,
    # or suffixed with _text depending on previous columns. Normalize for checking only.
    print(f"\n===== {split.upper()} MERGE =====")
    print("wav rows   :", len(wav_df))
    print("mfcc rows  :", len(mfcc_df))
    print("text rows  :", len(text_df))
    print("merged rows:", len(merged))

    mismatch_total = 0
    for c in TARGET_COLS:
        c_wav = f"{c}_wav"
        c_mfcc = f"{c}_mfcc"
        c_text = c if c in merged.columns else f"{c}_text"

        if c_wav in merged.columns and c_mfcc in merged.columns:
            mismatch = int(np.sum(np.abs(merged[c_wav].to_numpy(dtype=np.float32) - merged[c_mfcc].to_numpy(dtype=np.float32)) > 1e-6))
            mismatch_total += mismatch
            if mismatch > 0:
                print(f"[WARN] Target mismatch wav/mfcc for {c}: {mismatch}")

        if c_wav in merged.columns and c_text in merged.columns:
            mismatch = int(np.sum(np.abs(merged[c_wav].to_numpy(dtype=np.float32) - merged[c_text].to_numpy(dtype=np.float32)) > 1e-6))
            mismatch_total += mismatch
            if mismatch > 0:
                print(f"[WARN] Target mismatch wav/text for {c}: {mismatch}")

    if mismatch_total > 0:
        print("[WARN] Using Wav2Vec2 labels as y_true.")

    return merged.reset_index(drop=True), key


def infer_seq_feat_dim_from_paths(paths):
    for p in paths[:100]:
        arr = np.load(str(p))
        if arr.ndim == 2:
            return int(arr.shape[1])
    raise ValueError("Could not infer sequence feature dimension.")


def infer_text_feat_dim_from_paths(paths):
    for p in paths[:100]:
        arr = np.load(str(p))
        if arr.ndim == 1:
            return int(arr.shape[0])
    raise ValueError("Could not infer text feature dimension.")


def compute_seq_norm_stats_streaming(df, feature_col, max_len, label):
    feat_dim = infer_seq_feat_dim_from_paths(df[feature_col].astype(str).tolist())
    total_count = 0
    sum_vec = np.zeros(feat_dim, dtype=np.float64)
    sumsq_vec = np.zeros(feat_dim, dtype=np.float64)

    print(f"\n===== COMPUTING {label} SEQUENCE NORMALIZATION STATS =====")
    print("rows:", len(df))
    print("max_len:", max_len)
    print("feat_dim:", feat_dim)

    for i, p in enumerate(df[feature_col].astype(str)):
        x = np.load(p).astype(np.float32)
        if x.ndim != 2:
            continue

        x = x[:max_len]
        sum_vec += x.sum(axis=0, dtype=np.float64)
        sumsq_vec += np.square(x, dtype=np.float64).sum(axis=0)
        total_count += x.shape[0]

        if (i + 1) % 5000 == 0:
            print(f"Processed {label} stats {i + 1}/{len(df)}")

    if total_count == 0:
        raise ValueError(f"No frames found for {label} normalization.")

    mean = sum_vec / total_count
    var = (sumsq_vec / total_count) - np.square(mean)
    var = np.maximum(var, 1e-12)
    std = np.sqrt(var)
    std = np.where(std < 1e-6, 1.0, std)

    return mean.astype(np.float32), std.astype(np.float32), feat_dim


def compute_text_norm_stats_streaming(df, feature_col):
    feat_dim = infer_text_feat_dim_from_paths(df[feature_col].astype(str).tolist())
    n = 0
    sum_vec = np.zeros(feat_dim, dtype=np.float64)
    sumsq_vec = np.zeros(feat_dim, dtype=np.float64)

    print("\n===== COMPUTING TEXT NORMALIZATION STATS =====")
    print("rows:", len(df))
    print("feat_dim:", feat_dim)

    for i, p in enumerate(df[feature_col].astype(str)):
        x = np.load(p).astype(np.float32)
        if x.ndim != 1:
            continue

        sum_vec += x.astype(np.float64)
        sumsq_vec += np.square(x, dtype=np.float64)
        n += 1

        if (i + 1) % 5000 == 0:
            print(f"Processed text stats {i + 1}/{len(df)}")

    if n == 0:
        raise ValueError("No vectors found for text normalization.")

    mean = sum_vec / n
    var = (sumsq_vec / n) - np.square(mean)
    var = np.maximum(var, 1e-12)
    std = np.sqrt(var)
    std = np.where(std < 1e-6, 1.0, std)

    return mean.astype(np.float32), std.astype(np.float32), feat_dim


def pad_and_normalize_sequence(path, max_len, feat_dim, feat_mean, feat_std):
    x = np.load(str(path)).astype(np.float32)

    if x.ndim != 2:
        raise ValueError(f"Invalid sequence feature shape {x.shape}: {path}")
    if x.shape[1] != feat_dim:
        raise ValueError(f"Unexpected sequence dim {x.shape[1]} != {feat_dim}: {path}")

    x = x[:max_len]
    x = (x - feat_mean) / feat_std

    out = np.full((max_len, feat_dim), PAD_VALUE, dtype=np.float32)
    out[: x.shape[0], :] = x
    return out


def load_and_normalize_text(path, feat_dim, feat_mean, feat_std):
    x = np.load(str(path)).astype(np.float32)

    if x.ndim != 1:
        raise ValueError(f"Invalid text feature shape {x.shape}: {path}")
    if x.shape[0] != feat_dim:
        raise ValueError(f"Unexpected text dim {x.shape[0]} != {feat_dim}: {path}")

    return ((x - feat_mean) / feat_std).astype(np.float32)


class MultimodalBatchGenerator(tf.keras.utils.Sequence):
    def __init__(
        self,
        df,
        wav_mean,
        wav_std,
        wav_dim,
        mfcc_mean,
        mfcc_std,
        mfcc_dim,
        text_mean,
        text_std,
        text_dim,
        batch_size,
        shuffle,
        name,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.df = df.reset_index(drop=True)
        self.wav_mean = wav_mean.astype(np.float32)
        self.wav_std = wav_std.astype(np.float32)
        self.wav_dim = int(wav_dim)
        self.mfcc_mean = mfcc_mean.astype(np.float32)
        self.mfcc_std = mfcc_std.astype(np.float32)
        self.mfcc_dim = int(mfcc_dim)
        self.text_mean = text_mean.astype(np.float32)
        self.text_std = text_std.astype(np.float32)
        self.text_dim = int(text_dim)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.name = name
        self.indices = np.arange(len(self.df))
        self.rng = np.random.default_rng(RANDOM_STATE)
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, batch_idx):
        batch_indices = self.indices[
            batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size
        ]
        batch_df = self.df.iloc[batch_indices]

        X_wav = np.zeros((len(batch_df), WAV_MAX_LEN, self.wav_dim), dtype=np.float32)
        X_mfcc = np.zeros((len(batch_df), MFCC_MAX_LEN, self.mfcc_dim), dtype=np.float32)
        X_text = np.zeros((len(batch_df), self.text_dim), dtype=np.float32)
        y = batch_df[[f"{c}_wav" for c in TARGET_COLS]].to_numpy(dtype=np.float32)

        for j, (_, row) in enumerate(batch_df.iterrows()):
            X_wav[j] = pad_and_normalize_sequence(
                row[WAV_FEATURE_COL],
                WAV_MAX_LEN,
                self.wav_dim,
                self.wav_mean,
                self.wav_std,
            )
            X_mfcc[j] = pad_and_normalize_sequence(
                row[MFCC_FEATURE_COL],
                MFCC_MAX_LEN,
                self.mfcc_dim,
                self.mfcc_mean,
                self.mfcc_std,
            )
            X_text[j] = load_and_normalize_text(
                row[TEXT_FEATURE_COL],
                self.text_dim,
                self.text_mean,
                self.text_std,
            )

        return (X_wav, X_mfcc, X_text), y

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.indices)


def predict_generator_in_order(model, df, wav_mean, wav_std, wav_dim, mfcc_mean, mfcc_std, mfcc_dim, text_mean, text_std, text_dim):
    gen = MultimodalBatchGenerator(
        df=df,
        wav_mean=wav_mean,
        wav_std=wav_std,
        wav_dim=wav_dim,
        mfcc_mean=mfcc_mean,
        mfcc_std=mfcc_std,
        mfcc_dim=mfcc_dim,
        text_mean=text_mean,
        text_std=text_std,
        text_dim=text_dim,
        batch_size=BATCH_SIZE,
        shuffle=False,
        name="predict",
    )
    pred = model.predict(gen, verbose=1)
    return np.asarray(pred, dtype=np.float32)


# Metric helpers
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


def save_predictions_csv(base_df, key, y_true, y_pred, out_csv):
    metadata_cols = []
    for col in [
        key,
        "audio_path",
        "duration",
        "sampling_rate",
        "transcript",
        "major_emotion",
        "gender",
        "source_file",
        "split",
    ]:
        if col in base_df.columns and col not in metadata_cols:
            metadata_cols.append(col)

    out_df = base_df[metadata_cols].copy()

    y_true_denorm = denormalize_vad(y_true)
    y_pred_denorm = denormalize_vad(y_pred)

    for i, name in enumerate(TARGET_NAMES):
        out_df[f"{name}_true_norm"] = y_true[:, i]
        out_df[f"{name}_pred_norm"] = y_pred[:, i]
        out_df[f"{name}_true_1to5"] = y_true_denorm[:, i]
        out_df[f"{name}_pred_1to5"] = y_pred_denorm[:, i]

    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")


# MFCC masked-pooling layers
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


# Model blocks
def conv_bn_relu(x, filters, kernel_size, dilation_rate=1, name_prefix="conv"):
    x = tf.keras.layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        dilation_rate=dilation_rate,
        kernel_regularizer=tf.keras.regularizers.l2(L2_REG),
        name=f"{name_prefix}_conv",
    )(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name_prefix}_bn")(x)
    x = tf.keras.layers.Activation("relu", name=f"{name_prefix}_relu")(x)
    return x


def dilated_residual_block(x, filters, dilation_rate, block_id):
    shortcut = x

    y = conv_bn_relu(
        x,
        filters=filters,
        kernel_size=MFCC_KERNEL_SIZE_DILATED,
        dilation_rate=dilation_rate,
        name_prefix=f"mfcc_dilated_block_{block_id}_a",
    )
    y = tf.keras.layers.SpatialDropout1D(
        MFCC_CONV_DROPOUT,
        name=f"mfcc_dilated_block_{block_id}_dropout",
    )(y)
    y = tf.keras.layers.Conv1D(
        filters=filters,
        kernel_size=MFCC_KERNEL_SIZE_DILATED,
        padding="same",
        dilation_rate=dilation_rate,
        kernel_regularizer=tf.keras.regularizers.l2(L2_REG),
        name=f"mfcc_dilated_block_{block_id}_b_conv",
    )(y)
    y = tf.keras.layers.BatchNormalization(name=f"mfcc_dilated_block_{block_id}_b_bn")(y)

    if shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=1,
            padding="same",
            kernel_regularizer=tf.keras.regularizers.l2(L2_REG),
            name=f"mfcc_dilated_block_{block_id}_shortcut",
        )(shortcut)

    out = tf.keras.layers.Add(name=f"mfcc_dilated_block_{block_id}_add")([shortcut, y])
    out = tf.keras.layers.Activation("relu", name=f"mfcc_dilated_block_{block_id}_relu")(out)
    return out


def build_model_level_multimodal_model(wav_dim, mfcc_dim, text_dim):
    # Wav2Vec2 branch.
    wav_input = tf.keras.Input(shape=(WAV_MAX_LEN, wav_dim), name="wav2vec2_sequence")
    x_wav = tf.keras.layers.Masking(mask_value=PAD_VALUE, name="wav2vec2_masking")(wav_input)

    x_wav = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            WAV_LSTM_UNITS_1,
            return_sequences=True,
            dropout=WAV_DROPOUT,
            recurrent_dropout=0.0,
        ),
        name="wav2vec2_bilstm_1",
    )(x_wav)
    x_wav = tf.keras.layers.LayerNormalization(name="wav2vec2_bilstm_1_norm")(x_wav)

    x_wav = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            WAV_LSTM_UNITS_2,
            return_sequences=True,
            dropout=WAV_DROPOUT,
            recurrent_dropout=0.0,
        ),
        name="wav2vec2_bilstm_2",
    )(x_wav)
    x_wav = tf.keras.layers.LayerNormalization(name="wav2vec2_bilstm_2_norm")(x_wav)

    wav_avg = tf.keras.layers.GlobalAveragePooling1D(name="wav2vec2_avg_pool")(x_wav)
    wav_max = tf.keras.layers.GlobalMaxPooling1D(name="wav2vec2_max_pool")(x_wav)
    wav_repr = tf.keras.layers.Concatenate(name="wav2vec2_concat_pool")([wav_avg, wav_max])
    wav_repr = tf.keras.layers.Dense(WAV_BRANCH_DENSE, activation="relu", name="wav2vec2_branch_dense")(wav_repr)
    wav_repr = tf.keras.layers.Dropout(BRANCH_DROPOUT, name="wav2vec2_branch_dropout")(wav_repr)

    # MFCC branch.
    mfcc_input = tf.keras.Input(shape=(MFCC_MAX_LEN, mfcc_dim), name="mfcc_sequence")
    mfcc_mask = SequenceMaskFromPadValue(PAD_VALUE, name="mfcc_padding_mask")(mfcc_input)
    x_mfcc = tf.keras.layers.Masking(mask_value=PAD_VALUE, name="mfcc_masking")(mfcc_input)

    x_mfcc = conv_bn_relu(
        x_mfcc,
        filters=MFCC_BASE_FILTERS,
        kernel_size=MFCC_KERNEL_SIZE_FIRST,
        dilation_rate=1,
        name_prefix="mfcc_stem",
    )
    x_mfcc = tf.keras.layers.SpatialDropout1D(MFCC_CONV_DROPOUT, name="mfcc_stem_dropout")(x_mfcc)

    for block_id, dilation in enumerate(MFCC_DILATION_RATES, start=1):
        x_mfcc = dilated_residual_block(
            x_mfcc,
            filters=MFCC_DILATED_FILTERS,
            dilation_rate=dilation,
            block_id=block_id,
        )

    x_mfcc = conv_bn_relu(
        x_mfcc,
        filters=MFCC_FINAL_FILTERS,
        kernel_size=1,
        dilation_rate=1,
        name_prefix="mfcc_final_projection",
    )

    mfcc_avg = MaskedGlobalAveragePooling1D(name="mfcc_masked_avg_pool")([x_mfcc, mfcc_mask])
    mfcc_max = MaskedGlobalMaxPooling1D(name="mfcc_masked_max_pool")([x_mfcc, mfcc_mask])
    mfcc_repr = tf.keras.layers.Concatenate(name="mfcc_concat_pool")([mfcc_avg, mfcc_max])
    mfcc_repr = tf.keras.layers.Dense(
        MFCC_BRANCH_DENSE,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(L2_REG),
        name="mfcc_branch_dense",
    )(mfcc_repr)
    mfcc_repr = tf.keras.layers.Dropout(BRANCH_DROPOUT, name="mfcc_branch_dropout")(mfcc_repr)

    # Match the IEMOCAP DistilBERT text branch.
    text_input = tf.keras.Input(shape=(text_dim,), name="text_embedding")
    x_text = tf.keras.layers.Dense(
        TEXT_DENSE_1,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(L2_REG),
        name="text_dense_256",
    )(text_input)
    x_text = tf.keras.layers.BatchNormalization(name="text_bn_256")(x_text)
    x_text = tf.keras.layers.Dropout(TEXT_DROPOUT, name="text_dropout_256")(x_text)

    x_text = tf.keras.layers.Dense(
        TEXT_DENSE_2,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(L2_REG),
        name="text_dense_128",
    )(x_text)
    x_text = tf.keras.layers.BatchNormalization(name="text_bn_128")(x_text)
    x_text = tf.keras.layers.Dropout(TEXT_DROPOUT, name="text_dropout_128")(x_text)

    text_repr = tf.keras.layers.Dense(
        TEXT_DENSE_3,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(L2_REG),
        name="text_dense_64",
    )(x_text)
    text_repr = tf.keras.layers.Dropout(TEXT_DROPOUT, name="text_dropout_64")(text_repr)

    # Model-level audio-text fusion head.
    x = tf.keras.layers.Concatenate(name="model_level_audio_text_concat")([
        wav_repr,
        mfcc_repr,
        text_repr,
    ])
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="multimodal_fusion_norm")(x)
    x = tf.keras.layers.Dense(
        FUSION_DENSE_1,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(L2_REG),
        name="multimodal_fusion_dense_1",
    )(x)
    x = tf.keras.layers.Dropout(FUSION_DROPOUT, name="multimodal_fusion_dropout_1")(x)
    x = tf.keras.layers.Dense(
        FUSION_DENSE_2,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(L2_REG),
        name="multimodal_fusion_dense_2",
    )(x)
    x = tf.keras.layers.Dropout(FUSION_DROPOUT, name="multimodal_fusion_dropout_2")(x)

    outputs = tf.keras.layers.Dense(3, activation="tanh", name="vad_output")(x)

    return tf.keras.Model(
        inputs=[wav_input, mfcc_input, text_input],
        outputs=outputs,
        name="msp_full_model_level_multimodal_vad",
    )


# Main
def main():
    print("WAV2VEC2_ROOT:", WAV2VEC2_ROOT)
    print("MFCC_ROOT    :", MFCC_ROOT)
    print("TEXT_ROOT    :", TEXT_ROOT)
    print("OUTPUT_DIR   :", OUTPUT_DIR)
    print("WAV_MAX_LEN  :", WAV_MAX_LEN)
    print("MFCC_MAX_LEN :", MFCC_MAX_LEN)
    print("BATCH_SIZE   :", BATCH_SIZE)
    print("EPOCHS       :", EPOCHS)
    print("LR           :", LEARNING_RATE)
    print("Fusion type  : model-level multi-modal audio-text fusion")

    train_df, key = load_merged_split("train")
    val_df, _ = load_merged_split("val")
    test_df, _ = load_merged_split("test")

    wav_mean, wav_std, wav_dim = compute_seq_norm_stats_streaming(
        train_df,
        WAV_FEATURE_COL,
        WAV_MAX_LEN,
        "WAV2VEC2",
    )
    mfcc_mean, mfcc_std, mfcc_dim = compute_seq_norm_stats_streaming(
        train_df,
        MFCC_FEATURE_COL,
        MFCC_MAX_LEN,
        "MFCC",
    )
    text_mean, text_std, text_dim = compute_text_norm_stats_streaming(
        train_df,
        TEXT_FEATURE_COL,
    )

    print("\n===== FEATURE SHAPES =====")
    print("wav_dim :", wav_dim)
    print("mfcc_dim:", mfcc_dim)
    print("text_dim:", text_dim)

    np.savez(
        OUTPUT_DIR / "multimodal_norm_stats.npz",
        wav_feat_mean=wav_mean,
        wav_feat_std=wav_std,
        mfcc_feat_mean=mfcc_mean,
        mfcc_feat_std=mfcc_std,
        text_feat_mean=text_mean,
        text_feat_std=text_std,
        wav_max_len=WAV_MAX_LEN,
        mfcc_max_len=MFCC_MAX_LEN,
        wav_feat_dim=wav_dim,
        mfcc_feat_dim=mfcc_dim,
        text_feat_dim=text_dim,
    )

    train_gen = MultimodalBatchGenerator(
        df=train_df,
        wav_mean=wav_mean,
        wav_std=wav_std,
        wav_dim=wav_dim,
        mfcc_mean=mfcc_mean,
        mfcc_std=mfcc_std,
        mfcc_dim=mfcc_dim,
        text_mean=text_mean,
        text_std=text_std,
        text_dim=text_dim,
        batch_size=BATCH_SIZE,
        shuffle=True,
        name="train",
    )

    val_gen = MultimodalBatchGenerator(
        df=val_df,
        wav_mean=wav_mean,
        wav_std=wav_std,
        wav_dim=wav_dim,
        mfcc_mean=mfcc_mean,
        mfcc_std=mfcc_std,
        mfcc_dim=mfcc_dim,
        text_mean=text_mean,
        text_std=text_std,
        text_dim=text_dim,
        batch_size=BATCH_SIZE,
        shuffle=False,
        name="val",
    )

    tf.keras.backend.clear_session()
    gc.collect()

    model = build_model_level_multimodal_model(
        wav_dim=wav_dim,
        mfcc_dim=mfcc_dim,
        text_dim=text_dim,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=mse_ccc_loss,
        metrics=[
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
            ccc_tf,
        ],
    )

    model.summary()

    best_model_path = OUTPUT_DIR / "multimodal_model_level_best_model.keras"
    history_csv = OUTPUT_DIR / "multimodal_model_level_history.csv"

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_ccc_tf",
            mode="max",
            factor=0.5,
            patience=3,
            min_delta=0.001,
            min_lr=1e-6,
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
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_ccc_tf",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(history_csv), append=False),
    ]

    fit_t0 = time.perf_counter()
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        verbose=1,
        callbacks=callbacks,
    )
    fit_seconds = time.perf_counter() - fit_t0
    print("fit_seconds:", fit_seconds)

    del train_gen, val_gen
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

    y_train = train_df[[f"{c}_wav" for c in TARGET_COLS]].to_numpy(dtype=np.float32)
    y_val = val_df[[f"{c}_wav" for c in TARGET_COLS]].to_numpy(dtype=np.float32)
    y_test = test_df[[f"{c}_wav" for c in TARGET_COLS]].to_numpy(dtype=np.float32)

    print("\nPredicting train...")
    y_pred_train = predict_generator_in_order(
        model,
        train_df,
        wav_mean,
        wav_std,
        wav_dim,
        mfcc_mean,
        mfcc_std,
        mfcc_dim,
        text_mean,
        text_std,
        text_dim,
    )
    print("Predicting val...")
    y_pred_val = predict_generator_in_order(
        model,
        val_df,
        wav_mean,
        wav_std,
        wav_dim,
        mfcc_mean,
        mfcc_std,
        mfcc_dim,
        text_mean,
        text_std,
        text_dim,
    )
    print("Predicting test...")
    y_pred_test = predict_generator_in_order(
        model,
        test_df,
        wav_mean,
        wav_std,
        wav_dim,
        mfcc_mean,
        mfcc_std,
        mfcc_dim,
        text_mean,
        text_std,
        text_dim,
    )

    train_metrics = evaluate_split(y_train, y_pred_train)
    val_metrics = evaluate_split(y_val, y_pred_val)
    test_metrics = evaluate_split(y_test, y_pred_test)

    print("\n===== TRAIN RESULTS =====")
    print(train_metrics)

    print("\n===== VALIDATION RESULTS =====")
    print(val_metrics)

    print("\n===== TEST RESULTS =====")
    print(test_metrics)

    train_metrics.to_csv(OUTPUT_DIR / "multimodal_model_level_train_metrics.csv", index=False, encoding="utf-8-sig")
    val_metrics.to_csv(OUTPUT_DIR / "multimodal_model_level_val_metrics.csv", index=False, encoding="utf-8-sig")
    test_metrics.to_csv(OUTPUT_DIR / "multimodal_model_level_metrics.csv", index=False, encoding="utf-8-sig")

    save_predictions_csv(train_df, key, y_train, y_pred_train, OUTPUT_DIR / "multimodal_model_level_train_predictions.csv")
    save_predictions_csv(val_df, key, y_val, y_pred_val, OUTPUT_DIR / "multimodal_model_level_val_predictions.csv")
    save_predictions_csv(test_df, key, y_test, y_pred_test, OUTPUT_DIR / "multimodal_model_level_predictions.csv")

    history_df = pd.read_csv(history_csv)

    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, len(history_df) + 1), history_df["rmse"], label="Train RMSE")
    plt.plot(np.arange(1, len(history_df) + 1), history_df["val_rmse"], label="Validation RMSE")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.title("MSP full Model-level Multi-modal RMSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "multimodal_model_level_rmse_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, len(history_df) + 1), history_df["ccc_tf"], label="Train CCC")
    plt.plot(np.arange(1, len(history_df) + 1), history_df["val_ccc_tf"], label="Validation CCC")
    plt.xlabel("Epoch")
    plt.ylabel("CCC")
    plt.title("MSP full Model-level Multi-modal CCC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "multimodal_model_level_ccc_curve.png", dpi=200)
    plt.close()

    model.save(OUTPUT_DIR / "multimodal_model_level_model_final.keras")

    summary_df = pd.DataFrame([{
        "dataset": "msp_podcast_full",
        "model": "model_level_multimodal_wav2vec2_mfcc_text",
        "fit_seconds": fit_seconds,
        "epochs_ran": int(len(history_df)),
        "batch_size": BATCH_SIZE,
        "learning_rate_init": LEARNING_RATE,
        "wav_max_len": WAV_MAX_LEN,
        "mfcc_max_len": MFCC_MAX_LEN,
        "wav_feat_dim": wav_dim,
        "mfcc_feat_dim": mfcc_dim,
        "text_feat_dim": text_dim,
        "wav_branch": "BiLSTM(256)-BiLSTM(128)-global_avg_max_pool-Dense(128)",
        "mfcc_branch": "CNN1D dilated residual blocks + masked avg/max pooling + Dense(128)",
        "text_branch": "Dense(256)-BN-Dropout-Dense(128)-BN-Dropout-Dense(64)-Dropout",
        "fusion_head": "Concat-Dense(256)-Dense(128)-Dense(3)",
        "mean_mse": float(test_metrics.loc[test_metrics["target"] == "mean", "mse"].values[0]),
        "mean_rmse": float(test_metrics.loc[test_metrics["target"] == "mean", "rmse"].values[0]),
        "mean_ccc": float(test_metrics.loc[test_metrics["target"] == "mean", "ccc"].values[0]),
    }])
    summary_df.to_csv(OUTPUT_DIR / "multimodal_model_level_summary.csv", index=False, encoding="utf-8-sig")

    with open(OUTPUT_DIR / "artifacts.json", "w", encoding="utf-8") as f:
        json.dump({
            "dataset": "msp_podcast_full",
            "model": "model_level_multimodal_wav2vec2_mfcc_text",
            "wav2vec2_root": str(WAV2VEC2_ROOT),
            "mfcc_root": str(MFCC_ROOT),
            "text_root": str(TEXT_ROOT),
            "output_dir": str(OUTPUT_DIR),
            "target_cols": TARGET_COLS,
            "best_model_path": str(best_model_path),
            "norm_stats_path": str(OUTPUT_DIR / "multimodal_norm_stats.npz"),
            "wav_max_len": int(WAV_MAX_LEN),
            "mfcc_max_len": int(MFCC_MAX_LEN),
            "wav_feat_dim": int(wav_dim),
            "mfcc_feat_dim": int(mfcc_dim),
            "text_feat_dim": int(text_dim),
            "batch_size": int(BATCH_SIZE),
            "learning_rate": float(LEARNING_RATE),
            "note": (
                "Model-level multi-modal fusion. Wav2Vec2, MFCC, and text branches are trained jointly in one model. "
                "This is not prediction-level fusion. The Wav2Vec2 branch follows the old BiLSTM architecture; "
                "the MFCC branch follows the old CNN1D dilated residual architecture; the text branch follows the old DistilBERT embedding-only dense architecture. "
                "Data loading uses streaming normalization and batch loading for full MSP."
            ),
        }, f, indent=2)

    print("\nSaved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
