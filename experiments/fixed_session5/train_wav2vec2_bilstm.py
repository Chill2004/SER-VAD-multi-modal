import os
import sys
from pathlib import Path

ENV_DATA_ROOT = Path(os.getenv("SER_DATA_ROOT", "data"))
ENV_OUTPUT_ROOT = Path(os.getenv("SER_OUTPUT_ROOT", "outputs"))



def _ensure_cuda_ld_library_path():
    """tensorflow[and-cuda] ships CUDA/cuDNN as nvidia-* wheels but TF 2.21
    does not always auto-wire them. If we detect the wheels and the linker
    cannot yet see them, inject LD_LIBRARY_PATH and re-exec once."""
    try:
        import site
        bases = []
        for sp in site.getsitepackages() + [site.getusersitepackages()]:
            nv = os.path.join(sp, "nvidia")
            if os.path.isdir(nv):
                bases.append(nv)
        lib_dirs = []
        for base in bases:
            for d in sorted(os.listdir(base)):
                p = os.path.join(base, d, "lib")
                if os.path.isdir(p):
                    lib_dirs.append(p)
        if not lib_dirs:
            return
        current = os.environ.get("LD_LIBRARY_PATH", "")
        if lib_dirs[0] in current.split(":"):
            return
        os.environ["LD_LIBRARY_PATH"] = ":".join(lib_dirs + ([current] if current else []))
        if os.environ.get("_TF_CUDA_LD_RELAUNCHED") != "1":
            os.environ["_TF_CUDA_LD_RELAUNCHED"] = "1"
            os.execv(sys.executable, [sys.executable] + sys.argv)
    except Exception as e:
        print("[WARN] CUDA LD path auto-setup failed:", e)


_ensure_cuda_ld_library_path()

import gc
import json
import time
import argparse
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_squared_error


# CLI
parser = argparse.ArgumentParser()
parser.add_argument(
    "--setting",
    type=str,
    required=True,
    choices=["baseline", "silence", "noise", "reverb", "silence_noise"],
)
args = parser.parse_args()
SETTING = args.setting


# Runtime setup
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("SER_GPU", "1")

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
SETTING_DIR = AUG_ROOT / SETTING / "wav2vec2_features"

TRAIN_CSV = SETTING_DIR / "train_manifest.csv"
VAL_CSV = SETTING_DIR / "val_manifest.csv"
TEST_CSV = SETTING_DIR / "test_manifest.csv"

OUTPUT_DIR = Path(str(ENV_OUTPUT_ROOT / 'aug_runs_results')) / SETTING
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COL = "seq_feat_path"
TARGET_COLS = ["valence_norm", "arousal_norm", "dominance_norm"]
TARGET_NAMES = ["valence", "arousal", "dominance"]

RANDOM_STATE = 42
MAX_LEN_PERCENTILE = 95
BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 1e-4
DROPOUT_RATE = 0.25
PAD_VALUE = 0.0

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# Helpers
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

    return (2 * rho * std_true * std_pred) / (
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


def load_sequence_list(df, feature_col):
    seqs = []
    keep_idx = []

    for i, row in df.iterrows():
        p = Path(str(row[feature_col]))
        if not p.exists():
            continue
        try:
            x = np.load(p)
            x = np.asarray(x, dtype=np.float32)
            if x.ndim != 2:
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


# Load data
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

print("SETTING   :", SETTING)
print("Train rows:", len(train_df))
print("Val rows  :", len(val_df))
print("Test rows :", len(test_df))


# Padding and normalization
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
train_ds = train_ds.shuffle(min(len(X_train), 4096), seed=RANDOM_STATE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# Model
tf.keras.backend.clear_session()
gc.collect()

inputs = tf.keras.Input(shape=(MAX_LEN, FEAT_DIM))
x = tf.keras.layers.Masking(mask_value=PAD_VALUE)(inputs)

x = tf.keras.layers.Bidirectional(
    tf.keras.layers.LSTM(
        256,
        return_sequences=True,
        dropout=DROPOUT_RATE,
        recurrent_dropout=0.0,
    )
)(x)
x = tf.keras.layers.LayerNormalization()(x)

x = tf.keras.layers.Bidirectional(
    tf.keras.layers.LSTM(
        128,
        return_sequences=True,
        dropout=DROPOUT_RATE,
        recurrent_dropout=0.0,
    )
)(x)
x = tf.keras.layers.LayerNormalization()(x)

avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
x = tf.keras.layers.Concatenate()([avg_pool, max_pool])

x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)

outputs = tf.keras.layers.Dense(3, activation="tanh")(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=mse_ccc_loss,
    metrics=[
        tf.keras.metrics.RootMeanSquaredError(name="rmse"),
        ccc_tf,
    ]
)

model.summary()


# Callbacks
best_model_path = OUTPUT_DIR / "bilstm_best_model.keras"
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
    patience=6,
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
        train_pred = self.model.predict(X_train, verbose=0)
        val_pred = self.model.predict(X_val, verbose=0)

        train_rmse = float(np.sqrt(mean_squared_error(y_train, train_pred)))
        val_rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))

        train_ccc = float(np.mean([ccc_1d(y_train[:, i], train_pred[:, i]) for i in range(3)]))
        val_ccc = float(np.mean([ccc_1d(y_val[:, i], val_pred[:, i]) for i in range(3)]))

        current_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))

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
            f"Epoch {epoch+1:03d} | "
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
model = tf.keras.models.load_model(
    best_model_path,
    custom_objects={
        "mse_ccc_loss": mse_ccc_loss,
        "ccc_tf": ccc_tf,
    },
)

y_pred_train = model.predict(X_train, verbose=0)
y_pred_val = model.predict(X_val, verbose=0)
y_pred_test = model.predict(X_test, verbose=0)


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
train_metrics.to_csv(OUTPUT_DIR / "bilstm_train_metrics.csv", index=False, encoding="utf-8-sig")
val_metrics.to_csv(OUTPUT_DIR / "bilstm_val_metrics.csv", index=False, encoding="utf-8-sig")
test_metrics.to_csv(OUTPUT_DIR / "bilstm_metrics.csv", index=False, encoding="utf-8-sig")

save_predictions_csv(train_df, y_train, y_pred_train, OUTPUT_DIR / "bilstm_train_predictions.csv")
save_predictions_csv(val_df, y_val, y_pred_val, OUTPUT_DIR / "bilstm_val_predictions.csv")
save_predictions_csv(test_df, y_test, y_pred_test, OUTPUT_DIR / "bilstm_predictions.csv")

history_df = pd.DataFrame(history_rows)
history_df.to_csv(OUTPUT_DIR / "bilstm_history.csv", index=False, encoding="utf-8-sig")

plt.figure(figsize=(8, 5))
plt.plot(history_df["epoch"], history_df["train_rmse"], label="Train RMSE")
plt.plot(history_df["epoch"], history_df["val_rmse"], label="Validation RMSE")
plt.xlabel("Epoch")
plt.ylabel("RMSE")
plt.title(f"BiLSTM RMSE over Epochs ({SETTING})")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "bilstm_rmse_curve.png", dpi=200)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(history_df["epoch"], history_df["train_ccc"], label="Train CCC")
plt.plot(history_df["epoch"], history_df["val_ccc"], label="Validation CCC")
plt.xlabel("Epoch")
plt.ylabel("CCC")
plt.title(f"BiLSTM CCC over Epochs ({SETTING})")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "bilstm_ccc_curve.png", dpi=200)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(history_df["epoch"], history_df["lr"], label="Learning Rate")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title(f"BiLSTM Learning Rate Schedule ({SETTING})")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "bilstm_learning_rate_curve.png", dpi=200)
plt.close()

model.save(OUTPUT_DIR / "bilstm_model_final.keras")

summary_df = pd.DataFrame([{
    "setting": SETTING,
    "model": "bilstm",
    "fit_seconds": fit_seconds,
    "epochs_ran": int(len(history_df)),
    "batch_size": BATCH_SIZE,
    "learning_rate_init": LEARNING_RATE,
    "dropout_rate": DROPOUT_RATE,
    "max_len": MAX_LEN,
    "feat_dim": FEAT_DIM,
    "mean_mse": float(test_metrics.loc[test_metrics["target"] == "mean", "mse"].values[0]),
    "mean_rmse": float(test_metrics.loc[test_metrics["target"] == "mean", "rmse"].values[0]),
    "mean_ccc": float(test_metrics.loc[test_metrics["target"] == "mean", "ccc"].values[0]),
}])
summary_df.to_csv(OUTPUT_DIR / "bilstm_summary.csv", index=False, encoding="utf-8-sig")

with open(OUTPUT_DIR / "artifacts.json", "w", encoding="utf-8") as f:
    json.dump({
        "setting": SETTING,
        "train_manifest": str(TRAIN_CSV),
        "val_manifest": str(VAL_CSV),
        "test_manifest": str(TEST_CSV),
        "feature_col": FEATURE_COL,
        "target_cols": TARGET_COLS,
        "model_path": str(OUTPUT_DIR / "bilstm_best_model.keras"),
        "norm_stats_path": str(OUTPUT_DIR / "feature_norm_stats.npz"),
        "max_len": int(MAX_LEN),
        "feat_dim": int(FEAT_DIM),
    }, f, indent=2)

print("\nSaved to:", OUTPUT_DIR)
