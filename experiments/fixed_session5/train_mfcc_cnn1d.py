import os
import gc
import json
import time
import argparse
import warnings
from pathlib import Path

ENV_DATA_ROOT = Path(os.getenv("SER_DATA_ROOT", "data"))
ENV_OUTPUT_ROOT = Path(os.getenv("SER_OUTPUT_ROOT", "outputs"))


# CLI
parser = argparse.ArgumentParser()

parser.add_argument("--setting", type=str, required=True)

parser.add_argument(
    "--feature_dir",
    type=str,
    default="mfcc39_delta_delta2_features",
    help=(
        "Feature folder inside aug_runs_audio/<setting>/. "
        "Examples: mfcc_features, mfcc39_delta_delta2_features, wav2vec2_features"
    ),
)

parser.add_argument(
    "--gpu",
    type=str,
    default="1",
    help="CUDA_VISIBLE_DEVICES value. Use '0', '1', etc.",
)

parser.add_argument(
    "--max_len",
    type=int,
    default=None,
    help="Optional fixed max length. If not set, use MAX_LEN_PERCENTILE from train lengths.",
)

args = parser.parse_args()

SETTING = args.setting
FEATURE_DIR_NAME = args.feature_dir

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
AUG_ROOT = Path(str(ENV_OUTPUT_ROOT / 'aug_runs_audio'))
SETTING_DIR = AUG_ROOT / SETTING / FEATURE_DIR_NAME

TRAIN_CSV = SETTING_DIR / "train_manifest.csv"
VAL_CSV = SETTING_DIR / "val_manifest.csv"
TEST_CSV = SETTING_DIR / "test_manifest.csv"

OUTPUT_DIR = (
    Path(str(ENV_OUTPUT_ROOT / 'aug_runs_results_cnn1d'))
    / FEATURE_DIR_NAME
    / SETTING
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

# CNN1D configuration
BASE_FILTERS = 128
DILATED_FILTERS = 128
FINAL_FILTERS = 256

KERNEL_SIZE_FIRST = 5
KERNEL_SIZE_DILATED = 3

DILATION_RATES = [1, 2, 4, 8]

CONV_DROPOUT = 0.20
HEAD_DROPOUT = 0.30
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


# Data helpers
def load_sequence_list(df, feature_col):
    seqs = []
    keep_idx = []

    for i, row in df.iterrows():
        p = Path(str(row[feature_col]))

        if not p.exists():
            print(f"[WARN] Missing feature: {p}")
            continue

        try:
            x = np.load(p)
            x = np.asarray(x, dtype=np.float32)

            if x.ndim != 2:
                print(f"[WARN] Invalid feature shape {x.shape}: {p}")
                continue

            seqs.append(x)
            keep_idx.append(i)

        except Exception as e:
            print(f"[WARN] Failed to load feature: {p}")
            print("       Error:", e)

    if len(seqs) == 0:
        raise ValueError(f"No valid sequence features loaded from {feature_col}")

    kept_df = df.loc[keep_idx].reset_index(drop=True)
    return seqs, kept_df


def compute_max_len(train_seqs, percentile=95):
    lengths = np.array([x.shape[0] for x in train_seqs], dtype=np.int32)
    max_len = int(np.percentile(lengths, percentile))
    return max(max_len, 8)


def compute_feature_norm_stats(train_seqs, max_len):
    clipped = [x[:max_len] for x in train_seqs]
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


def predict_numpy_in_batches(model, X, batch_size):
    preds = []

    for start in range(0, len(X), batch_size):
        batch_pred = model.predict_on_batch(X[start:start + batch_size])
        preds.append(np.asarray(batch_pred, dtype=np.float32))

    return np.concatenate(preds, axis=0)


# Masked-pooling layers
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
        kernel_size=KERNEL_SIZE_DILATED,
        dilation_rate=dilation_rate,
        name_prefix=f"dilated_block_{block_id}_a",
    )

    y = tf.keras.layers.SpatialDropout1D(
        CONV_DROPOUT,
        name=f"dilated_block_{block_id}_dropout",
    )(y)

    y = tf.keras.layers.Conv1D(
        filters=filters,
        kernel_size=KERNEL_SIZE_DILATED,
        padding="same",
        dilation_rate=dilation_rate,
        kernel_regularizer=tf.keras.regularizers.l2(L2_REG),
        name=f"dilated_block_{block_id}_b_conv",
    )(y)

    y = tf.keras.layers.BatchNormalization(name=f"dilated_block_{block_id}_b_bn")(y)

    if shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=1,
            padding="same",
            kernel_regularizer=tf.keras.regularizers.l2(L2_REG),
            name=f"dilated_block_{block_id}_shortcut",
        )(shortcut)

    out = tf.keras.layers.Add(name=f"dilated_block_{block_id}_add")([shortcut, y])
    out = tf.keras.layers.Activation("relu", name=f"dilated_block_{block_id}_relu")(out)

    return out


def build_cnn1d_model(max_len, feat_dim):
    inputs = tf.keras.Input(shape=(max_len, feat_dim), name="input_sequence")

    padding_mask = SequenceMaskFromPadValue(
        PAD_VALUE,
        name="padding_mask",
    )(inputs)

    x = tf.keras.layers.Masking(
        mask_value=PAD_VALUE,
        name="input_masking",
    )(inputs)

    # Local feature extractor
    x = conv_bn_relu(
        x,
        filters=BASE_FILTERS,
        kernel_size=KERNEL_SIZE_FIRST,
        dilation_rate=1,
        name_prefix="stem",
    )

    x = tf.keras.layers.SpatialDropout1D(
        CONV_DROPOUT,
        name="stem_dropout",
    )(x)

    # Dilated blocks expand the receptive field without shrinking time.
    for block_id, dilation in enumerate(DILATION_RATES, start=1):
        x = dilated_residual_block(
            x,
            filters=DILATED_FILTERS,
            dilation_rate=dilation,
            block_id=block_id,
        )

    # Final projection before pooling.
    x = conv_bn_relu(
        x,
        filters=FINAL_FILTERS,
        kernel_size=1,
        dilation_rate=1,
        name_prefix="final_projection",
    )

    # Ignore padded timesteps during pooling.
    avg_pool = MaskedGlobalAveragePooling1D(name="masked_avg_pool")([x, padding_mask])
    max_pool = MaskedGlobalMaxPooling1D(name="masked_max_pool")([x, padding_mask])

    x = tf.keras.layers.Concatenate(name="concat_pool")([avg_pool, max_pool])

    x = tf.keras.layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(L2_REG),
        name="head_dense_128",
    )(x)

    x = tf.keras.layers.Dropout(
        HEAD_DROPOUT,
        name="head_dropout",
    )(x)

    outputs = tf.keras.layers.Dense(
        3,
        activation="tanh",
        name="vad_output",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="cnn1d_vad_regression",
    )

    return model


# Load data
print("SETTING_DIR:", SETTING_DIR)

for p in [TRAIN_CSV, VAL_CSV, TEST_CSV]:
    if not p.exists():
        raise FileNotFoundError(f"Missing manifest: {p}")

train_df = pd.read_csv(TRAIN_CSV).dropna(subset=TARGET_COLS).copy()
val_df = pd.read_csv(VAL_CSV).dropna(subset=TARGET_COLS).copy()
test_df = pd.read_csv(TEST_CSV).dropna(subset=TARGET_COLS).copy()

for c in [FEATURE_COL] + TARGET_COLS:
    if c not in train_df.columns:
        raise ValueError(f"Missing column in train manifest: {c}")
    if c not in val_df.columns:
        raise ValueError(f"Missing column in val manifest: {c}")
    if c not in test_df.columns:
        raise ValueError(f"Missing column in test manifest: {c}")

train_seqs, train_df = load_sequence_list(train_df, FEATURE_COL)
val_seqs, val_df = load_sequence_list(val_df, FEATURE_COL)
test_seqs, test_df = load_sequence_list(test_df, FEATURE_COL)

y_train = train_df[TARGET_COLS].to_numpy(dtype=np.float32)
y_val = val_df[TARGET_COLS].to_numpy(dtype=np.float32)
y_test = test_df[TARGET_COLS].to_numpy(dtype=np.float32)

print("SETTING    :", SETTING)
print("FEATURE_DIR:", FEATURE_DIR_NAME)
print("TRAIN_CSV  :", TRAIN_CSV)
print("VAL_CSV    :", VAL_CSV)
print("TEST_CSV   :", TEST_CSV)
print("Train rows :", len(train_df))
print("Val rows   :", len(val_df))
print("Test rows  :", len(test_df))


# Padding and normalization
if args.max_len is not None:
    MAX_LEN = int(args.max_len)
else:
    MAX_LEN = compute_max_len(train_seqs, percentile=MAX_LEN_PERCENTILE)

FEAT_DIM = train_seqs[0].shape[1]

feat_mean, feat_std = compute_feature_norm_stats(train_seqs, MAX_LEN)

X_train = pad_and_normalize_sequences(train_seqs, MAX_LEN, feat_mean, feat_std, PAD_VALUE)
X_val = pad_and_normalize_sequences(val_seqs, MAX_LEN, feat_mean, feat_std, PAD_VALUE)
X_test = pad_and_normalize_sequences(test_seqs, MAX_LEN, feat_mean, feat_std, PAD_VALUE)

X_train = np.ascontiguousarray(X_train.astype("float32"))
X_val = np.ascontiguousarray(X_val.astype("float32"))
X_test = np.ascontiguousarray(X_test.astype("float32"))

y_train = np.ascontiguousarray(y_train.astype("float32"))
y_val = np.ascontiguousarray(y_val.astype("float32"))
y_test = np.ascontiguousarray(y_test.astype("float32"))

print("MAX_LEN :", MAX_LEN)
print("FEAT_DIM:", FEAT_DIM)
print("X_train :", X_train.shape)
print("X_val   :", X_val.shape)
print("X_test  :", X_test.shape)

np.savez(
    OUTPUT_DIR / "feature_norm_stats.npz",
    feat_mean=feat_mean,
    feat_std=feat_std,
    max_len=MAX_LEN,
    feat_dim=FEAT_DIM,
)


# Datasets
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = (
    train_ds
    .shuffle(min(len(X_train), 4096), seed=RANDOM_STATE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# Model
tf.keras.backend.clear_session()
gc.collect()

model = build_cnn1d_model(MAX_LEN, FEAT_DIM)

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
best_model_path = OUTPUT_DIR / "cnn1d_best_model.keras"
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

        train_rmse = float(logs.get("rmse", np.nan))
        val_rmse = float(logs.get("val_rmse", np.nan))
        train_ccc = float(logs.get("ccc_tf", np.nan))
        val_ccc = float(logs.get("val_ccc_tf", np.nan))

        history_rows.append({
            "epoch": int(epoch + 1),
            "lr": current_lr,
            "train_loss": float(logs.get("loss", np.nan)),
            "val_loss": float(logs.get("val_loss", np.nan)),
            "train_rmse": train_rmse,
            "val_rmse": val_rmse,
            "train_ccc": train_ccc,
            "val_ccc": val_ccc,
        })

        print(
            f"Epoch {epoch + 1:03d} | "
            f"lr={current_lr:.7f} | "
            f"train_rmse={train_rmse:.6f} | "
            f"val_rmse={val_rmse:.6f} | "
            f"train_ccc={train_ccc:.6f} | "
            f"val_ccc={val_ccc:.6f}"
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

y_pred_train = predict_numpy_in_batches(model, X_train, BATCH_SIZE)
y_pred_val = predict_numpy_in_batches(model, X_val, BATCH_SIZE)
y_pred_test = predict_numpy_in_batches(model, X_test, BATCH_SIZE)


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
train_metrics.to_csv(OUTPUT_DIR / "cnn1d_train_metrics.csv", index=False, encoding="utf-8-sig")
val_metrics.to_csv(OUTPUT_DIR / "cnn1d_val_metrics.csv", index=False, encoding="utf-8-sig")
test_metrics.to_csv(OUTPUT_DIR / "cnn1d_metrics.csv", index=False, encoding="utf-8-sig")

save_predictions_csv(train_df, y_train, y_pred_train, OUTPUT_DIR / "cnn1d_train_predictions.csv")
save_predictions_csv(val_df, y_val, y_pred_val, OUTPUT_DIR / "cnn1d_val_predictions.csv")
save_predictions_csv(test_df, y_test, y_pred_test, OUTPUT_DIR / "cnn1d_predictions.csv")

history_df = pd.DataFrame(history_rows)
history_df.to_csv(OUTPUT_DIR / "cnn1d_history.csv", index=False, encoding="utf-8-sig")

plt.figure(figsize=(8, 5))
plt.plot(history_df["epoch"], history_df["train_rmse"], label="Train RMSE")
plt.plot(history_df["epoch"], history_df["val_rmse"], label="Validation RMSE")
plt.xlabel("Epoch")
plt.ylabel("RMSE")
plt.title(f"CNN1D RMSE over Epochs ({SETTING}, {FEATURE_DIR_NAME})")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cnn1d_rmse_curve.png", dpi=200)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(history_df["epoch"], history_df["train_ccc"], label="Train CCC")
plt.plot(history_df["epoch"], history_df["val_ccc"], label="Validation CCC")
plt.xlabel("Epoch")
plt.ylabel("CCC")
plt.title(f"CNN1D CCC over Epochs ({SETTING}, {FEATURE_DIR_NAME})")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cnn1d_ccc_curve.png", dpi=200)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(history_df["epoch"], history_df["lr"], label="Learning Rate")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title(f"CNN1D Learning Rate ({SETTING}, {FEATURE_DIR_NAME})")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cnn1d_learning_rate_curve.png", dpi=200)
plt.close()

model.save(OUTPUT_DIR / "cnn1d_model_final.keras")

summary_df = pd.DataFrame([{
    "setting": SETTING,
    "feature_dir": FEATURE_DIR_NAME,
    "model": "cnn1d_dilated_residual",
    "fit_seconds": fit_seconds,
    "epochs_ran": int(len(history_df)),
    "batch_size": BATCH_SIZE,
    "learning_rate_init": LEARNING_RATE,
    "max_len": MAX_LEN,
    "feat_dim": FEAT_DIM,
    "base_filters": BASE_FILTERS,
    "dilated_filters": DILATED_FILTERS,
    "final_filters": FINAL_FILTERS,
    "dilation_rates": str(DILATION_RATES),
    "conv_dropout": CONV_DROPOUT,
    "head_dropout": HEAD_DROPOUT,
    "mean_mse": float(test_metrics.loc[test_metrics["target"] == "mean", "mse"].values[0]),
    "mean_rmse": float(test_metrics.loc[test_metrics["target"] == "mean", "rmse"].values[0]),
    "mean_ccc": float(test_metrics.loc[test_metrics["target"] == "mean", "ccc"].values[0]),
}])
summary_df.to_csv(OUTPUT_DIR / "cnn1d_summary.csv", index=False, encoding="utf-8-sig")

with open(OUTPUT_DIR / "artifacts.json", "w", encoding="utf-8") as f:
    json.dump({
        "setting": SETTING,
        "feature_dir": FEATURE_DIR_NAME,
        "train_manifest": str(TRAIN_CSV),
        "val_manifest": str(VAL_CSV),
        "test_manifest": str(TEST_CSV),
        "feature_col": FEATURE_COL,
        "target_cols": TARGET_COLS,
        "model_path": str(best_model_path),
        "norm_stats_path": str(OUTPUT_DIR / "feature_norm_stats.npz"),
        "max_len": int(MAX_LEN),
        "feat_dim": int(FEAT_DIM),
        "base_filters": BASE_FILTERS,
        "dilated_filters": DILATED_FILTERS,
        "final_filters": FINAL_FILTERS,
        "dilation_rates": DILATION_RATES,
    }, f, indent=2)

print("\nSaved to:", OUTPUT_DIR)