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
args = parser.parse_args()

SETTING = args.setting

GPU_ID = os.getenv("SER_GPU", "1")
MAX_SECONDS_CAP = 10.0
MAX_LEN_SAMPLES_FIXED = None

# Runtime setup
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
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
SETTING_DIR = AUG_ROOT / SETTING

TRAIN_CSV = SETTING_DIR / "train_manifest.csv"
VAL_CSV = SETTING_DIR / "val_manifest.csv"
TEST_CSV = SETTING_DIR / "test_manifest.csv"

OUTPUT_DIR = (
    Path(str(ENV_OUTPUT_ROOT / 'aug_runs_results_cnn1d_raw'))
    / SETTING
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SR = 16000

TARGET_COLS = ["valence_norm", "arousal_norm", "dominance_norm"]
TARGET_NAMES = ["valence", "arousal", "dominance"]

RANDOM_STATE = 42
MAX_LEN_PERCENTILE = 95
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-4
PAD_VALUE = 0.0

# Raw-waveform CNN
L2_REG = 1e-5
CONV_DROPOUT = 0.20
HEAD_DROPOUT = 0.30

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
def find_audio_col(df: pd.DataFrame) -> str:
    candidates = [
        "wav_path",
        "audio_path",
        "wav_abs_path",
        "wav_filepath",
        "path",
    ]

    for c in candidates:
        if c in df.columns:
            return c

    for c in df.columns:
        values = df[c].astype(str)
        if values.str.lower().str.endswith(".wav").mean() > 0.5:
            return c

    raise ValueError("Could not find audio path column in manifest.")


def load_audio(audio_path, target_sr=16000):
    wav, sr = sf.read(str(audio_path))
    wav = np.asarray(wav, dtype=np.float32)

    if wav.ndim == 2:
        wav = wav.mean(axis=1)

    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # Avoid aggressive peak normalization because amplitude carries emotion cues.
    peak = np.max(np.abs(wav)) + 1e-8
    if peak > 1.5:
        wav = wav / peak

    wav = np.clip(wav, -1.0, 1.0).astype(np.float32)
    return wav, sr


def load_waveform_list(df, audio_col):
    waves = []
    keep_idx = []
    lengths = []

    for i, row in df.iterrows():
        audio_path = Path(str(row[audio_col]))

        if not audio_path.exists():
            print(f"[WARN] Missing audio: {audio_path}")
            continue

        try:
            wav, sr = load_audio(audio_path, TARGET_SR)

            if wav.ndim != 1 or len(wav) == 0:
                print(f"[WARN] Invalid waveform: {audio_path}")
                continue

            waves.append(wav)
            keep_idx.append(i)
            lengths.append(len(wav))

        except Exception as e:
            print(f"[WARN] Failed to load audio: {audio_path}")
            print("       Error:", e)

    if len(waves) == 0:
        raise ValueError("No valid waveforms loaded.")

    kept_df = df.loc[keep_idx].reset_index(drop=True)
    return waves, kept_df, np.asarray(lengths, dtype=np.int32)


def compute_max_len_samples(train_lengths):
    """
    Compute max waveform length for padding/truncation.

    Priority:
    1. If MAX_LEN_SAMPLES_FIXED is set, use it directly.
    2. Otherwise, use the 95th percentile of train waveform lengths.
    3. If MAX_SECONDS_CAP is set, cap the max length by that number of seconds.
    """
    if MAX_LEN_SAMPLES_FIXED is not None:
        return int(MAX_LEN_SAMPLES_FIXED)

    p_len = int(np.percentile(train_lengths, MAX_LEN_PERCENTILE))

    if MAX_SECONDS_CAP is not None and MAX_SECONDS_CAP > 0:
        cap_len = int(MAX_SECONDS_CAP * TARGET_SR)
        return max(8000, min(p_len, cap_len))

    return max(8000, p_len)

def pad_waveforms(wave_list, max_len_samples, pad_value=0.0):
    n = len(wave_list)
    X = np.full((n, max_len_samples, 1), pad_value, dtype=np.float32)

    for i, wav in enumerate(wave_list):
        wav = wav[:max_len_samples]
        L = len(wav)
        X[i, :L, 0] = wav

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


# Model blocks
def conv_bn_relu(x, filters, kernel_size, strides=1, name_prefix="conv"):
    x = tf.keras.layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        use_bias=False,
        kernel_regularizer=tf.keras.regularizers.l2(L2_REG),
        name=f"{name_prefix}_conv",
    )(x)

    x = tf.keras.layers.BatchNormalization(name=f"{name_prefix}_bn")(x)
    x = tf.keras.layers.Activation("relu", name=f"{name_prefix}_relu")(x)
    return x


def residual_conv_block(x, filters, kernel_size, block_id):
    shortcut = x

    y = conv_bn_relu(
        x,
        filters=filters,
        kernel_size=kernel_size,
        strides=1,
        name_prefix=f"res_block_{block_id}_a",
    )

    y = tf.keras.layers.SpatialDropout1D(
        CONV_DROPOUT,
        name=f"res_block_{block_id}_dropout",
    )(y)

    y = tf.keras.layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        strides=1,
        padding="same",
        use_bias=False,
        kernel_regularizer=tf.keras.regularizers.l2(L2_REG),
        name=f"res_block_{block_id}_b_conv",
    )(y)

    y = tf.keras.layers.BatchNormalization(name=f"res_block_{block_id}_b_bn")(y)

    if shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=1,
            padding="same",
            use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(L2_REG),
            name=f"res_block_{block_id}_shortcut",
        )(shortcut)

    out = tf.keras.layers.Add(name=f"res_block_{block_id}_add")([shortcut, y])
    out = tf.keras.layers.Activation("relu", name=f"res_block_{block_id}_relu")(out)
    return out


def build_raw_cnn1d_model(max_len_samples):
    inputs = tf.keras.Input(shape=(max_len_samples, 1), name="raw_waveform")

    # Downsample early to keep raw-waveform sequences manageable.
    x = conv_bn_relu(inputs, 32, 9, strides=2, name_prefix="stem_1")
    x = tf.keras.layers.MaxPooling1D(pool_size=2, name="pool_1")(x)

    x = conv_bn_relu(x, 64, 9, strides=2, name_prefix="stem_2")
    x = tf.keras.layers.MaxPooling1D(pool_size=2, name="pool_2")(x)

    x = conv_bn_relu(x, 128, 7, strides=2, name_prefix="stem_3")
    x = tf.keras.layers.MaxPooling1D(pool_size=2, name="pool_3")(x)

    x = residual_conv_block(x, 128, 5, block_id=1)
    x = residual_conv_block(x, 128, 5, block_id=2)

    x = conv_bn_relu(x, 256, 5, strides=1, name_prefix="final_conv")
    x = tf.keras.layers.SpatialDropout1D(CONV_DROPOUT, name="final_spatial_dropout")(x)

    avg_pool = tf.keras.layers.GlobalAveragePooling1D(name="global_avg_pool")(x)
    max_pool = tf.keras.layers.GlobalMaxPooling1D(name="global_max_pool")(x)

    x = tf.keras.layers.Concatenate(name="concat_pool")([avg_pool, max_pool])

    x = tf.keras.layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(L2_REG),
        name="head_dense_128",
    )(x)

    x = tf.keras.layers.Dropout(HEAD_DROPOUT, name="head_dropout")(x)

    outputs = tf.keras.layers.Dense(
        3,
        activation="tanh",
        name="vad_output",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="raw_waveform_cnn1d_vad",
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

audio_col = find_audio_col(train_df)

if audio_col not in val_df.columns:
    raise ValueError(f"Audio column missing in val manifest: {audio_col}")
if audio_col not in test_df.columns:
    raise ValueError(f"Audio column missing in test manifest: {audio_col}")

for c in TARGET_COLS:
    if c not in train_df.columns:
        raise ValueError(f"Missing column in train manifest: {c}")
    if c not in val_df.columns:
        raise ValueError(f"Missing column in val manifest: {c}")
    if c not in test_df.columns:
        raise ValueError(f"Missing column in test manifest: {c}")

train_waves, train_df, train_lengths = load_waveform_list(train_df, audio_col)
val_waves, val_df, val_lengths = load_waveform_list(val_df, audio_col)
test_waves, test_df, test_lengths = load_waveform_list(test_df, audio_col)

y_train = train_df[TARGET_COLS].to_numpy(dtype=np.float32)
y_val = val_df[TARGET_COLS].to_numpy(dtype=np.float32)
y_test = test_df[TARGET_COLS].to_numpy(dtype=np.float32)

print("SETTING    :", SETTING)
print("AUDIO_COL  :", audio_col)
print("TRAIN_CSV  :", TRAIN_CSV)
print("VAL_CSV    :", VAL_CSV)
print("TEST_CSV   :", TEST_CSV)
print("Train rows :", len(train_df))
print("Val rows   :", len(val_df))
print("Test rows  :", len(test_df))

print("Train length stats:")
print("  min:", int(train_lengths.min()))
print("  p50:", int(np.percentile(train_lengths, 50)))
print("  p95:", int(np.percentile(train_lengths, 95)))
print("  max:", int(train_lengths.max()))


# Pad raw waveforms
MAX_LEN_SAMPLES = compute_max_len_samples(train_lengths)
MAX_SECONDS_USED = MAX_LEN_SAMPLES / TARGET_SR

X_train = pad_waveforms(train_waves, MAX_LEN_SAMPLES, PAD_VALUE)
X_val = pad_waveforms(val_waves, MAX_LEN_SAMPLES, PAD_VALUE)
X_test = pad_waveforms(test_waves, MAX_LEN_SAMPLES, PAD_VALUE)

X_train = np.ascontiguousarray(X_train.astype("float32"))
X_val = np.ascontiguousarray(X_val.astype("float32"))
X_test = np.ascontiguousarray(X_test.astype("float32"))

y_train = np.ascontiguousarray(y_train.astype("float32"))
y_val = np.ascontiguousarray(y_val.astype("float32"))
y_test = np.ascontiguousarray(y_test.astype("float32"))

print("MAX_LEN_SAMPLES:", MAX_LEN_SAMPLES)
print("MAX_SECONDS_USED:", MAX_SECONDS_USED)
print("X_train:", X_train.shape)
print("X_val  :", X_val.shape)
print("X_test :", X_test.shape)

np.savez(
    OUTPUT_DIR / "raw_waveform_config.npz",
    max_len_samples=MAX_LEN_SAMPLES,
    target_sr=TARGET_SR,
    max_seconds_used=MAX_SECONDS_USED,
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

model = build_raw_cnn1d_model(MAX_LEN_SAMPLES)

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
best_model_path = OUTPUT_DIR / "raw_cnn1d_best_model.keras"
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
train_metrics.to_csv(OUTPUT_DIR / "raw_cnn1d_train_metrics.csv", index=False, encoding="utf-8-sig")
val_metrics.to_csv(OUTPUT_DIR / "raw_cnn1d_val_metrics.csv", index=False, encoding="utf-8-sig")
test_metrics.to_csv(OUTPUT_DIR / "raw_cnn1d_metrics.csv", index=False, encoding="utf-8-sig")

save_predictions_csv(train_df, y_train, y_pred_train, OUTPUT_DIR / "raw_cnn1d_train_predictions.csv")
save_predictions_csv(val_df, y_val, y_pred_val, OUTPUT_DIR / "raw_cnn1d_val_predictions.csv")
save_predictions_csv(test_df, y_test, y_pred_test, OUTPUT_DIR / "raw_cnn1d_predictions.csv")

history_df = pd.DataFrame(history_rows)
history_df.to_csv(OUTPUT_DIR / "raw_cnn1d_history.csv", index=False, encoding="utf-8-sig")

plt.figure(figsize=(8, 5))
plt.plot(history_df["epoch"], history_df["train_rmse"], label="Train RMSE")
plt.plot(history_df["epoch"], history_df["val_rmse"], label="Validation RMSE")
plt.xlabel("Epoch")
plt.ylabel("RMSE")
plt.title(f"Raw CNN1D RMSE over Epochs ({SETTING})")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "raw_cnn1d_rmse_curve.png", dpi=200)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(history_df["epoch"], history_df["train_ccc"], label="Train CCC")
plt.plot(history_df["epoch"], history_df["val_ccc"], label="Validation CCC")
plt.xlabel("Epoch")
plt.ylabel("CCC")
plt.title(f"Raw CNN1D CCC over Epochs ({SETTING})")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "raw_cnn1d_ccc_curve.png", dpi=200)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(history_df["epoch"], history_df["lr"], label="Learning Rate")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title(f"Raw CNN1D Learning Rate ({SETTING})")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "raw_cnn1d_learning_rate_curve.png", dpi=200)
plt.close()

model.save(OUTPUT_DIR / "raw_cnn1d_model_final.keras")

summary_df = pd.DataFrame([{
    "setting": SETTING,
    "input_type": "raw_waveform",
    "model": "raw_cnn1d",
    "fit_seconds": fit_seconds,
    "epochs_ran": int(len(history_df)),
    "batch_size": BATCH_SIZE,
    "learning_rate_init": LEARNING_RATE,
    "target_sr": TARGET_SR,
    "max_len_samples": MAX_LEN_SAMPLES,
    "max_seconds_used": MAX_SECONDS_USED,
    "mean_mse": float(test_metrics.loc[test_metrics["target"] == "mean", "mse"].values[0]),
    "mean_rmse": float(test_metrics.loc[test_metrics["target"] == "mean", "rmse"].values[0]),
    "mean_ccc": float(test_metrics.loc[test_metrics["target"] == "mean", "ccc"].values[0]),
}])

summary_df.to_csv(OUTPUT_DIR / "raw_cnn1d_summary.csv", index=False, encoding="utf-8-sig")

with open(OUTPUT_DIR / "artifacts.json", "w", encoding="utf-8") as f:
    json.dump({
        "setting": SETTING,
        "input_type": "raw_waveform",
        "train_manifest": str(TRAIN_CSV),
        "val_manifest": str(VAL_CSV),
        "test_manifest": str(TEST_CSV),
        "audio_col": audio_col,
        "target_cols": TARGET_COLS,
        "model_path": str(best_model_path),
        "target_sr": TARGET_SR,
        "max_len_samples": int(MAX_LEN_SAMPLES),
        "max_seconds_used": float(MAX_SECONDS_USED),
        "batch_size": int(BATCH_SIZE),
        "learning_rate": LEARNING_RATE,
    }, f, indent=2)

print("\nSaved to:", OUTPUT_DIR)