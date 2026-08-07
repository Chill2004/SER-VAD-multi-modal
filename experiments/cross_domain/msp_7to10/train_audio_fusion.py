import os
import gc
import json
import time
import warnings
from pathlib import Path

ENV_DATA_ROOT = Path(os.getenv("SER_DATA_ROOT", "data"))
ENV_OUTPUT_ROOT = Path(os.getenv("SER_OUTPUT_ROOT", "outputs"))


# Runtime setup
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("SER_GPU", "1")

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
FEATURE_ROOT = Path(str(ENV_OUTPUT_ROOT / 'abstracttts_podcast_7to10s'))
WAV2VEC2_DIR = FEATURE_ROOT / "wav2vec2_features"
MFCC_DIR = FEATURE_ROOT / "mfcc_features"

WAV_TRAIN_CSV = WAV2VEC2_DIR / "train_manifest.csv"
WAV_VAL_CSV = WAV2VEC2_DIR / "val_manifest.csv"
WAV_TEST_CSV = WAV2VEC2_DIR / "test_manifest.csv"

MFCC_TRAIN_CSV = MFCC_DIR / "train_manifest.csv"
MFCC_VAL_CSV = MFCC_DIR / "val_manifest.csv"
MFCC_TEST_CSV = MFCC_DIR / "test_manifest.csv"

OUTPUT_DIR = Path(
    str(ENV_OUTPUT_ROOT / 'abstracttts_podcast_7to10s_results/audio_fusion_iemocap_style')
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COL = "seq_feat_path"
TARGET_COLS = ["valence_norm", "arousal_norm", "dominance_norm"]
TARGET_NAMES = ["valence", "arousal", "dominance"]

RANDOM_STATE = 42
MAX_LEN_PERCENTILE = 95
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-4
PAD_VALUE = 0.0

# Wav2Vec2 branch configuration
WAV_DROPOUT_RATE = 0.25

# MFCC branch configuration
BASE_FILTERS = 128
DILATED_FILTERS = 128
FINAL_FILTERS = 256
KERNEL_SIZE_FIRST = 5
KERNEL_SIZE_DILATED = 3
DILATION_RATES = [1, 2, 4, 8]
CONV_DROPOUT = 0.20
HEAD_DROPOUT = 0.30
L2_REG = 1e-5

# Fusion head
FUSION_DENSE = 128
FUSION_DROPOUT = 0.30

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


# Data helpers
def make_merge_key(df: pd.DataFrame) -> pd.Series:
    if "utterance_id" in df.columns:
        return df["utterance_id"].astype(str)
    if "source_file" in df.columns:
        return df["source_file"].astype(str).map(lambda x: Path(x).stem)
    return pd.Series([f"row_{i:06d}" for i in range(len(df))], index=df.index)


def load_manifest(csv_path: Path, feature_name: str) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {feature_name} manifest: {csv_path}")

    df = pd.read_csv(csv_path).dropna(subset=TARGET_COLS).copy()

    if FEATURE_COL not in df.columns:
        raise ValueError(f"Missing feature column {FEATURE_COL} in {csv_path}")

    df = df.dropna(subset=[FEATURE_COL]).reset_index(drop=True)
    df["_merge_key"] = make_merge_key(df)

    duplicate_count = int(df["_merge_key"].duplicated().sum())
    if duplicate_count > 0:
        print(f"[WARN] {feature_name} manifest duplicated merge keys: {duplicate_count}")
        print("[WARN] Keeping first occurrence for each key.")
        df = df.drop_duplicates(subset=["_merge_key"], keep="first").copy()

    return df.reset_index(drop=True)


def load_and_merge_manifests(wav_csv: Path, mfcc_csv: Path) -> pd.DataFrame:
    wav_df = load_manifest(wav_csv, "Wav2Vec2")
    mfcc_df = load_manifest(mfcc_csv, "MFCC")

    wav_df = wav_df.rename(columns={FEATURE_COL: "wav2vec2_seq_feat_path"})

    mfcc_keep_cols = ["_merge_key", FEATURE_COL] + TARGET_COLS
    for optional_col in [
        "utterance_id",
        "audio_path",
        "duration",
        "sampling_rate",
        "transcript",
        "major_emotion",
        "gender",
        "source_file",
    ]:
        if optional_col in mfcc_df.columns and optional_col not in mfcc_keep_cols:
            mfcc_keep_cols.append(optional_col)

    mfcc_keep = mfcc_df[mfcc_keep_cols].copy()
    mfcc_keep = mfcc_keep.rename(columns={FEATURE_COL: "mfcc_seq_feat_path"})

    merged = wav_df.merge(
        mfcc_keep,
        on="_merge_key",
        how="inner",
        suffixes=("", "_mfcc"),
    )

    if len(merged) == 0:
        raise ValueError("Merged manifest is empty. Check utterance_id alignment between Wav2Vec2 and MFCC manifests.")

    mismatch_count = 0
    for c in TARGET_COLS:
        c_mfcc = f"{c}_mfcc"
        if c_mfcc in merged.columns:
            diff = np.abs(
                merged[c].to_numpy(dtype=np.float32)
                - merged[c_mfcc].to_numpy(dtype=np.float32)
            )
            mismatch_count += int(np.sum(diff > 1e-6))

    if mismatch_count > 0:
        print(f"[WARN] Target mismatch count after merge: {mismatch_count}")
        print("[WARN] Using Wav2Vec2 manifest labels as y_true.")

    merged = merged.dropna(
        subset=["wav2vec2_seq_feat_path", "mfcc_seq_feat_path"] + TARGET_COLS
    ).reset_index(drop=True)

    return merged


def load_audio_feature_lists(df: pd.DataFrame):
    wav_seqs = []
    mfcc_seqs = []
    keep_idx = []

    for i, row in df.iterrows():
        wav_path = Path(str(row["wav2vec2_seq_feat_path"]))
        mfcc_path = Path(str(row["mfcc_seq_feat_path"]))

        if not wav_path.exists():
            print(f"[WARN] Missing Wav2Vec2 feature: {wav_path}")
            continue

        if not mfcc_path.exists():
            print(f"[WARN] Missing MFCC feature: {mfcc_path}")
            continue

        try:
            wav_x = np.load(wav_path).astype(np.float32)
            mfcc_x = np.load(mfcc_path).astype(np.float32)

            if wav_x.ndim != 2:
                print(f"[WARN] Invalid Wav2Vec2 shape {wav_x.shape}: {wav_path}")
                continue

            if mfcc_x.ndim != 2:
                print(f"[WARN] Invalid MFCC shape {mfcc_x.shape}: {mfcc_path}")
                continue

            wav_seqs.append(wav_x)
            mfcc_seqs.append(mfcc_x)
            keep_idx.append(i)

        except Exception as e:
            print("[WARN] Failed to load feature pair:")
            print("       wav :", wav_path)
            print("       mfcc:", mfcc_path)
            print("       error:", e)

    if len(wav_seqs) == 0:
        raise ValueError("No valid Wav2Vec2/MFCC feature pairs loaded.")

    kept_df = df.loc[keep_idx].reset_index(drop=True)
    return wav_seqs, mfcc_seqs, kept_df


def compute_max_len(seq_list, percentile=95):
    lengths = np.array([x.shape[0] for x in seq_list], dtype=np.int32)
    max_len = int(np.percentile(lengths, percentile))
    return max(max_len, 8)


def compute_feature_norm_stats(seq_list, max_len):
    clipped = [x[:max_len] for x in seq_list]
    all_frames = np.concatenate(clipped, axis=0)

    feat_mean = all_frames.mean(axis=0).astype(np.float32)
    feat_std = all_frames.std(axis=0).astype(np.float32)
    feat_std = np.where(feat_std < 1e-6, 1.0, feat_std)

    return feat_mean, feat_std


def pad_and_normalize_sequences(seq_list, max_len, feat_mean, feat_std, pad_value=0.0):
    n = len(seq_list)
    feat_dim = seq_list[0].shape[1]

    X = np.full((n, max_len, feat_dim), pad_value, dtype=np.float32)

    for i, seq in enumerate(seq_list):
        seq = seq[:max_len]
        seq = (seq - feat_mean) / feat_std

        L = seq.shape[0]
        X[i, :L, :] = seq

    return X


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


def save_predictions_csv(base_df, y_true, y_pred, out_csv):
    out_df = base_df.copy()

    for i, name in enumerate(TARGET_NAMES):
        out_df[f"{name}_true_norm"] = y_true[:, i]
        out_df[f"{name}_pred_norm"] = y_pred[:, i]

    y_true_denorm = denormalize_vad(y_true)
    y_pred_denorm = denormalize_vad(y_pred)

    for i, name in enumerate(TARGET_NAMES):
        out_df[f"{name}_true_1to5"] = y_true_denorm[:, i]
        out_df[f"{name}_pred_1to5"] = y_pred_denorm[:, i]

    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")


def predict_numpy_in_batches(model, X_wav, X_mfcc, batch_size):
    preds = []

    for start in range(0, len(X_wav), batch_size):
        end = start + batch_size
        batch_pred = model.predict_on_batch([
            X_wav[start:end],
            X_mfcc[start:end],
        ])
        preds.append(np.asarray(batch_pred, dtype=np.float32))

    return np.concatenate(preds, axis=0)


# custom layers for masked pooling in MFCC branch
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


# MFCC CNN blocks
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
        kernel_size=KERNEL_SIZE_DILATED,
        dilation_rate=dilation_rate,
        name_prefix=f"mfcc_dilated_block_{block_id}_a",
    )

    y = tf.keras.layers.SpatialDropout1D(
        CONV_DROPOUT,
        name=f"mfcc_dilated_block_{block_id}_dropout",
    )(y)

    y = tf.keras.layers.Conv1D(
        filters=filters,
        kernel_size=KERNEL_SIZE_DILATED,
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


# model
def build_audio_fusion_model(wav_max_len, wav_feat_dim, mfcc_max_len, mfcc_feat_dim):
    # Wav2Vec2 -> BiLSTM branch
    # Wav2Vec2 branch up to the 128-D projection.
    wav_input = tf.keras.Input(shape=(wav_max_len, wav_feat_dim), name="wav2vec2_sequence")

    x_wav = tf.keras.layers.Masking(mask_value=PAD_VALUE, name="wav2vec2_masking")(wav_input)

    x_wav = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            256,
            return_sequences=True,
            dropout=WAV_DROPOUT_RATE,
            recurrent_dropout=0.0,
        ),
        name="wav2vec2_bilstm_1",
    )(x_wav)
    x_wav = tf.keras.layers.LayerNormalization(name="wav2vec2_bilstm_1_norm")(x_wav)

    x_wav = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            128,
            return_sequences=True,
            dropout=WAV_DROPOUT_RATE,
            recurrent_dropout=0.0,
        ),
        name="wav2vec2_bilstm_2",
    )(x_wav)
    x_wav = tf.keras.layers.LayerNormalization(name="wav2vec2_bilstm_2_norm")(x_wav)

    wav_avg_pool = tf.keras.layers.GlobalAveragePooling1D(name="wav2vec2_avg_pool")(x_wav)
    wav_max_pool = tf.keras.layers.GlobalMaxPooling1D(name="wav2vec2_max_pool")(x_wav)
    wav_repr = tf.keras.layers.Concatenate(name="wav2vec2_concat_pool")([
        wav_avg_pool,
        wav_max_pool,
    ])

    wav_repr = tf.keras.layers.Dense(
        128,
        activation="relu",
        name="wav2vec2_dense_128",
    )(wav_repr)
    wav_repr = tf.keras.layers.Dropout(
        WAV_DROPOUT_RATE,
        name="wav2vec2_dropout_128",
    )(wav_repr)

    # MFCC -> CNN1D branch
    # MFCC branch up to the 128-D projection.
    mfcc_input = tf.keras.Input(shape=(mfcc_max_len, mfcc_feat_dim), name="mfcc_sequence")

    mfcc_padding_mask = SequenceMaskFromPadValue(
        PAD_VALUE,
        name="mfcc_padding_mask",
    )(mfcc_input)

    x_mfcc = tf.keras.layers.Masking(
        mask_value=PAD_VALUE,
        name="mfcc_input_masking",
    )(mfcc_input)

    x_mfcc = conv_bn_relu(
        x_mfcc,
        filters=BASE_FILTERS,
        kernel_size=KERNEL_SIZE_FIRST,
        dilation_rate=1,
        name_prefix="mfcc_stem",
    )

    x_mfcc = tf.keras.layers.SpatialDropout1D(
        CONV_DROPOUT,
        name="mfcc_stem_dropout",
    )(x_mfcc)

    for block_id, dilation in enumerate(DILATION_RATES, start=1):
        x_mfcc = dilated_residual_block(
            x_mfcc,
            filters=DILATED_FILTERS,
            dilation_rate=dilation,
            block_id=block_id,
        )

    x_mfcc = conv_bn_relu(
        x_mfcc,
        filters=FINAL_FILTERS,
        kernel_size=1,
        dilation_rate=1,
        name_prefix="mfcc_final_projection",
    )

    mfcc_avg_pool = MaskedGlobalAveragePooling1D(name="mfcc_masked_avg_pool")([
        x_mfcc,
        mfcc_padding_mask,
    ])
    mfcc_max_pool = MaskedGlobalMaxPooling1D(name="mfcc_masked_max_pool")([
        x_mfcc,
        mfcc_padding_mask,
    ])

    mfcc_repr = tf.keras.layers.Concatenate(name="mfcc_concat_pool")([
        mfcc_avg_pool,
        mfcc_max_pool,
    ])

    mfcc_repr = tf.keras.layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(L2_REG),
        name="mfcc_head_dense_128",
    )(mfcc_repr)

    mfcc_repr = tf.keras.layers.Dropout(
        HEAD_DROPOUT,
        name="mfcc_head_dropout",
    )(mfcc_repr)

    # audio fusion head
    x = tf.keras.layers.Concatenate(name="audio_fusion_concat")([
        wav_repr,
        mfcc_repr,
    ])

    x = tf.keras.layers.Dense(
        FUSION_DENSE,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(L2_REG),
        name="fusion_dense_128",
    )(x)

    x = tf.keras.layers.Dropout(
        FUSION_DROPOUT,
        name="fusion_dropout",
    )(x)

    outputs = tf.keras.layers.Dense(
        3,
        activation="tanh",
        name="vad_output",
    )(x)

    model = tf.keras.Model(
        inputs=[wav_input, mfcc_input],
        outputs=outputs,
        name="abstracttts_audio_fusion_iemocap_style_vad",
    )

    return model


# Load data
print("FEATURE_ROOT:", FEATURE_ROOT)
print("WAV2VEC2_DIR:", WAV2VEC2_DIR)
print("MFCC_DIR    :", MFCC_DIR)
print("OUTPUT_DIR  :", OUTPUT_DIR)

train_df = load_and_merge_manifests(WAV_TRAIN_CSV, MFCC_TRAIN_CSV)
val_df = load_and_merge_manifests(WAV_VAL_CSV, MFCC_VAL_CSV)
test_df = load_and_merge_manifests(WAV_TEST_CSV, MFCC_TEST_CSV)

train_wav_seqs, train_mfcc_seqs, train_df = load_audio_feature_lists(train_df)
val_wav_seqs, val_mfcc_seqs, val_df = load_audio_feature_lists(val_df)
test_wav_seqs, test_mfcc_seqs, test_df = load_audio_feature_lists(test_df)

y_train = train_df[TARGET_COLS].to_numpy(dtype=np.float32)
y_val = val_df[TARGET_COLS].to_numpy(dtype=np.float32)
y_test = test_df[TARGET_COLS].to_numpy(dtype=np.float32)

print("Train rows:", len(train_df))
print("Val rows  :", len(val_df))
print("Test rows :", len(test_df))


# Padding and normalization
WAV_MAX_LEN = compute_max_len(train_wav_seqs, percentile=MAX_LEN_PERCENTILE)
MFCC_MAX_LEN = compute_max_len(train_mfcc_seqs, percentile=MAX_LEN_PERCENTILE)

WAV_FEAT_DIM = train_wav_seqs[0].shape[1]
MFCC_FEAT_DIM = train_mfcc_seqs[0].shape[1]

wav_feat_mean, wav_feat_std = compute_feature_norm_stats(train_wav_seqs, WAV_MAX_LEN)
mfcc_feat_mean, mfcc_feat_std = compute_feature_norm_stats(train_mfcc_seqs, MFCC_MAX_LEN)

X_wav_train = pad_and_normalize_sequences(
    train_wav_seqs,
    WAV_MAX_LEN,
    wav_feat_mean,
    wav_feat_std,
    PAD_VALUE,
)
X_wav_val = pad_and_normalize_sequences(
    val_wav_seqs,
    WAV_MAX_LEN,
    wav_feat_mean,
    wav_feat_std,
    PAD_VALUE,
)
X_wav_test = pad_and_normalize_sequences(
    test_wav_seqs,
    WAV_MAX_LEN,
    wav_feat_mean,
    wav_feat_std,
    PAD_VALUE,
)

X_mfcc_train = pad_and_normalize_sequences(
    train_mfcc_seqs,
    MFCC_MAX_LEN,
    mfcc_feat_mean,
    mfcc_feat_std,
    PAD_VALUE,
)
X_mfcc_val = pad_and_normalize_sequences(
    val_mfcc_seqs,
    MFCC_MAX_LEN,
    mfcc_feat_mean,
    mfcc_feat_std,
    PAD_VALUE,
)
X_mfcc_test = pad_and_normalize_sequences(
    test_mfcc_seqs,
    MFCC_MAX_LEN,
    mfcc_feat_mean,
    mfcc_feat_std,
    PAD_VALUE,
)

X_wav_train = np.ascontiguousarray(X_wav_train.astype("float32"))
X_wav_val = np.ascontiguousarray(X_wav_val.astype("float32"))
X_wav_test = np.ascontiguousarray(X_wav_test.astype("float32"))

X_mfcc_train = np.ascontiguousarray(X_mfcc_train.astype("float32"))
X_mfcc_val = np.ascontiguousarray(X_mfcc_val.astype("float32"))
X_mfcc_test = np.ascontiguousarray(X_mfcc_test.astype("float32"))

y_train = np.ascontiguousarray(y_train.astype("float32"))
y_val = np.ascontiguousarray(y_val.astype("float32"))
y_test = np.ascontiguousarray(y_test.astype("float32"))

print("WAV_MAX_LEN  :", WAV_MAX_LEN)
print("WAV_FEAT_DIM :", WAV_FEAT_DIM)
print("MFCC_MAX_LEN :", MFCC_MAX_LEN)
print("MFCC_FEAT_DIM:", MFCC_FEAT_DIM)
print("X_wav_train :", X_wav_train.shape)
print("X_mfcc_train:", X_mfcc_train.shape)
print("y_train     :", y_train.shape)

np.savez(
    OUTPUT_DIR / "audio_fusion_norm_stats.npz",
    wav_feat_mean=wav_feat_mean,
    wav_feat_std=wav_feat_std,
    mfcc_feat_mean=mfcc_feat_mean,
    mfcc_feat_std=mfcc_feat_std,
    wav_max_len=WAV_MAX_LEN,
    mfcc_max_len=MFCC_MAX_LEN,
    wav_feat_dim=WAV_FEAT_DIM,
    mfcc_feat_dim=MFCC_FEAT_DIM,
)


# Datasets
train_ds = tf.data.Dataset.from_tensor_slices(
    ((X_wav_train, X_mfcc_train), y_train)
)
train_ds = (
    train_ds
    .shuffle(min(len(X_wav_train), 4096), seed=RANDOM_STATE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = tf.data.Dataset.from_tensor_slices(
    ((X_wav_val, X_mfcc_val), y_val)
)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# Build and compile model
tf.keras.backend.clear_session()
gc.collect()

model = build_audio_fusion_model(
    wav_max_len=WAV_MAX_LEN,
    wav_feat_dim=WAV_FEAT_DIM,
    mfcc_max_len=MFCC_MAX_LEN,
    mfcc_feat_dim=MFCC_FEAT_DIM,
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


# Callbacks
best_model_path = OUTPUT_DIR / "audio_fusion_best_model.keras"
history_rows = []

reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_ccc_tf",
    mode="max",
    factor=0.5,
    patience=3,
    min_delta=0.001,
    min_lr=1e-6,
    verbose=1,
)

early_stop_cb = tf.keras.callbacks.EarlyStopping(
    monitor="val_ccc_tf",
    mode="max",
    patience=8,
    min_delta=0.001,
    restore_best_weights=True,
    verbose=1,
)

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=str(best_model_path),
    monitor="val_ccc_tf",
    mode="max",
    save_best_only=True,
    save_weights_only=False,
    verbose=1,
)


class MetricsLoggerCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        current_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))

        history_rows.append({
            "epoch": int(epoch + 1),
            "lr": current_lr,
            "train_loss": float(logs.get("loss", np.nan)),
            "val_loss": float(logs.get("val_loss", np.nan)),
            "train_rmse": float(logs.get("rmse", np.nan)),
            "val_rmse": float(logs.get("val_rmse", np.nan)),
            "train_ccc": float(logs.get("ccc_tf", np.nan)),
            "val_ccc": float(logs.get("val_ccc_tf", np.nan)),
        })

        print(
            f"Epoch {epoch + 1:03d} | "
            f"lr={current_lr:.7f} | "
            f"train_rmse={logs.get('rmse', np.nan):.6f} | "
            f"val_rmse={logs.get('val_rmse', np.nan):.6f} | "
            f"train_ccc={logs.get('ccc_tf', np.nan):.6f} | "
            f"val_ccc={logs.get('val_ccc_tf', np.nan):.6f}"
        )


metrics_logger_cb = MetricsLoggerCallback()


# Training
fit_t0 = time.perf_counter()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1,
    callbacks=[
        reduce_lr_cb,
        early_stop_cb,
        checkpoint_cb,
        metrics_logger_cb,
    ],
)

fit_seconds = time.perf_counter() - fit_t0
print("fit_seconds:", fit_seconds)


# Final predictions
del train_ds, val_ds, history, metrics_logger_cb
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

y_pred_train = predict_numpy_in_batches(
    model,
    X_wav_train,
    X_mfcc_train,
    BATCH_SIZE,
)
y_pred_val = predict_numpy_in_batches(
    model,
    X_wav_val,
    X_mfcc_val,
    BATCH_SIZE,
)
y_pred_test = predict_numpy_in_batches(
    model,
    X_wav_test,
    X_mfcc_test,
    BATCH_SIZE,
)


# Evaluation
train_metrics = evaluate_split(y_train, y_pred_train)
val_metrics = evaluate_split(y_val, y_pred_val)
test_metrics = evaluate_split(y_test, y_pred_test)

print("\n===== TRAIN RESULTS =====")
print(train_metrics)

print("\n===== VALIDATION RESULTS =====")
print(val_metrics)

print("\n===== TEST RESULTS =====")
print(test_metrics)


# Save outputs
train_metrics.to_csv(OUTPUT_DIR / "audio_fusion_train_metrics.csv", index=False, encoding="utf-8-sig")
val_metrics.to_csv(OUTPUT_DIR / "audio_fusion_val_metrics.csv", index=False, encoding="utf-8-sig")
test_metrics.to_csv(OUTPUT_DIR / "audio_fusion_metrics.csv", index=False, encoding="utf-8-sig")

save_predictions_csv(train_df, y_train, y_pred_train, OUTPUT_DIR / "audio_fusion_train_predictions.csv")
save_predictions_csv(val_df, y_val, y_pred_val, OUTPUT_DIR / "audio_fusion_val_predictions.csv")
save_predictions_csv(test_df, y_test, y_pred_test, OUTPUT_DIR / "audio_fusion_predictions.csv")

history_df = pd.DataFrame(history_rows)
history_df.to_csv(OUTPUT_DIR / "audio_fusion_history.csv", index=False, encoding="utf-8-sig")

plt.figure(figsize=(8, 5))
plt.plot(history_df["epoch"], history_df["train_rmse"], label="Train RMSE")
plt.plot(history_df["epoch"], history_df["val_rmse"], label="Validation RMSE")
plt.xlabel("Epoch")
plt.ylabel("RMSE")
plt.title("AbstractTTS/PODCAST Audio Fusion RMSE over Epochs")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "audio_fusion_rmse_curve.png", dpi=200)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(history_df["epoch"], history_df["train_ccc"], label="Train CCC")
plt.plot(history_df["epoch"], history_df["val_ccc"], label="Validation CCC")
plt.xlabel("Epoch")
plt.ylabel("CCC")
plt.title("AbstractTTS/PODCAST Audio Fusion CCC over Epochs")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "audio_fusion_ccc_curve.png", dpi=200)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(history_df["epoch"], history_df["lr"], label="Learning Rate")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title("AbstractTTS/PODCAST Audio Fusion Learning Rate")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "audio_fusion_learning_rate_curve.png", dpi=200)
plt.close()

model.save(OUTPUT_DIR / "audio_fusion_model_final.keras")

summary_df = pd.DataFrame([{
    "dataset": "AbstractTTS/PODCAST 7-10s subset",
    "model": "audio_fusion_iemocap_style_wav2vec2_bilstm_mfcc_cnn1d",
    "feature_root": str(FEATURE_ROOT),
    "wav2vec2_dir": str(WAV2VEC2_DIR),
    "mfcc_dir": str(MFCC_DIR),
    "fit_seconds": fit_seconds,
    "epochs_ran": int(len(history_df)),
    "batch_size": BATCH_SIZE,
    "learning_rate_init": LEARNING_RATE,
    "wav_max_len": WAV_MAX_LEN,
    "wav_feat_dim": WAV_FEAT_DIM,
    "mfcc_max_len": MFCC_MAX_LEN,
    "mfcc_feat_dim": MFCC_FEAT_DIM,
    "fusion_dense": FUSION_DENSE,
    "mean_mse": float(test_metrics.loc[test_metrics["target"] == "mean", "mse"].values[0]),
    "mean_rmse": float(test_metrics.loc[test_metrics["target"] == "mean", "rmse"].values[0]),
    "mean_ccc": float(test_metrics.loc[test_metrics["target"] == "mean", "ccc"].values[0]),
}])
summary_df.to_csv(OUTPUT_DIR / "audio_fusion_summary.csv", index=False, encoding="utf-8-sig")

with open(OUTPUT_DIR / "artifacts.json", "w", encoding="utf-8") as f:
    json.dump({
        "dataset": "AbstractTTS/PODCAST 7-10s subset",
        "model": "audio_fusion_iemocap_style_wav2vec2_bilstm_mfcc_cnn1d",
        "feature_root": str(FEATURE_ROOT),
        "wav2vec2_dir": str(WAV2VEC2_DIR),
        "mfcc_dir": str(MFCC_DIR),
        "wav_train_manifest": str(WAV_TRAIN_CSV),
        "wav_val_manifest": str(WAV_VAL_CSV),
        "wav_test_manifest": str(WAV_TEST_CSV),
        "mfcc_train_manifest": str(MFCC_TRAIN_CSV),
        "mfcc_val_manifest": str(MFCC_VAL_CSV),
        "mfcc_test_manifest": str(MFCC_TEST_CSV),
        "target_cols": TARGET_COLS,
        "model_path": str(best_model_path),
        "norm_stats_path": str(OUTPUT_DIR / "audio_fusion_norm_stats.npz"),
        "wav_max_len": int(WAV_MAX_LEN),
        "wav_feat_dim": int(WAV_FEAT_DIM),
        "mfcc_max_len": int(MFCC_MAX_LEN),
        "mfcc_feat_dim": int(MFCC_FEAT_DIM),
        "batch_size": int(BATCH_SIZE),
        "learning_rate": LEARNING_RATE,
        "note": (
            "Audio-only fusion baseline adapted to AbstractTTS/PODCAST. "
            "The Wav2Vec2 branch follows the IEMOCAP-style BiLSTM baseline, "
            "and the MFCC branch follows the IEMOCAP-style CNN1D baseline."
        ),
    }, f, indent=2)

print("\nSaved to:", OUTPUT_DIR)
