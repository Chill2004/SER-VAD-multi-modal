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

# Runtime setup
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

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
    except RuntimeError as e:
        print("GPU setup error:", e)


# Configuration
AUG_ROOT = Path(str(ENV_OUTPUT_ROOT / 'aug_runs_audio'))
FEATURE_DIR = "text_distilbert_features"

SETTING_DIR = AUG_ROOT / SETTING / FEATURE_DIR

TRAIN_CSV = SETTING_DIR / "train_manifest.csv"
VAL_CSV = SETTING_DIR / "val_manifest.csv"
TEST_CSV = SETTING_DIR / "test_manifest.csv"

OUTPUT_DIR = (
    Path(str(ENV_OUTPUT_ROOT / 'aug_runs_results_text_only'))
    / SETTING
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEXT_FEATURE_COL = "text_feat_path"

TARGET_COLS = ["valence_norm", "arousal_norm", "dominance_norm"]
TARGET_NAMES = ["valence", "arousal", "dominance"]

RANDOM_STATE = 42
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-4
DROPOUT_RATE = 0.30
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
def load_manifest(csv_path):
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing manifest: {csv_path}")

    df = pd.read_csv(csv_path).copy()

    if TEXT_FEATURE_COL not in df.columns:
        raise ValueError(f"Missing column {TEXT_FEATURE_COL} in {csv_path}")

    for c in TARGET_COLS:
        if c not in df.columns:
            raise ValueError(f"Missing target column {c} in {csv_path}")

    df = df.dropna(subset=TARGET_COLS).copy().reset_index(drop=True)
    return df


def load_text_features(df):
    feats = []
    keep_idx = []

    for i, row in df.iterrows():
        p = Path(str(row[TEXT_FEATURE_COL]))

        if not p.exists():
            print(f"[WARN] Missing text feature: {p}")
            continue

        try:
            x = np.load(p).astype(np.float32)

            if x.ndim != 1:
                print(f"[WARN] Invalid text feature shape {x.shape}: {p}")
                continue

            feats.append(x)
            keep_idx.append(i)

        except Exception as e:
            print(f"[WARN] Failed to load {p}: {e}")

    if len(feats) == 0:
        raise ValueError("No valid text features loaded.")

    kept_df = df.loc[keep_idx].reset_index(drop=True)
    X = np.stack(feats, axis=0).astype(np.float32)

    return X, kept_df


def normalize_features(X_train, X_val, X_test):
    mean = X_train.mean(axis=0).astype(np.float32)
    std = X_train.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    return X_train, X_val, X_test, mean, std


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


# Model
def build_model(input_dim):
    inputs = tf.keras.Input(shape=(input_dim,), name="text_embedding")

    x = tf.keras.layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(L2_REG),
        name="dense_256",
    )(inputs)

    x = tf.keras.layers.BatchNormalization(name="bn_256")(x)

    x = tf.keras.layers.Dropout(
        DROPOUT_RATE,
        name="dropout_256",
    )(x)

    x = tf.keras.layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(L2_REG),
        name="dense_128",
    )(x)

    x = tf.keras.layers.BatchNormalization(name="bn_128")(x)

    x = tf.keras.layers.Dropout(
        DROPOUT_RATE,
        name="dropout_128",
    )(x)

    x = tf.keras.layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(L2_REG),
        name="dense_64",
    )(x)

    x = tf.keras.layers.Dropout(
        DROPOUT_RATE,
        name="dropout_64",
    )(x)

    outputs = tf.keras.layers.Dense(
        3,
        activation="tanh",
        name="vad_output",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="text_embedding_only_vad",
    )

    return model


# Load data
print("SETTING:", SETTING)
print("SETTING_DIR:", SETTING_DIR)

train_df = load_manifest(TRAIN_CSV)
val_df = load_manifest(VAL_CSV)
test_df = load_manifest(TEST_CSV)

X_train, train_df = load_text_features(train_df)
X_val, val_df = load_text_features(val_df)
X_test, test_df = load_text_features(test_df)

y_train = train_df[TARGET_COLS].to_numpy(dtype=np.float32)
y_val = val_df[TARGET_COLS].to_numpy(dtype=np.float32)
y_test = test_df[TARGET_COLS].to_numpy(dtype=np.float32)

X_train, X_val, X_test, feat_mean, feat_std = normalize_features(
    X_train,
    X_val,
    X_test,
)

X_train = np.ascontiguousarray(X_train.astype("float32"))
X_val = np.ascontiguousarray(X_val.astype("float32"))
X_test = np.ascontiguousarray(X_test.astype("float32"))

y_train = np.ascontiguousarray(y_train.astype("float32"))
y_val = np.ascontiguousarray(y_val.astype("float32"))
y_test = np.ascontiguousarray(y_test.astype("float32"))

INPUT_DIM = X_train.shape[1]

print("Train rows:", len(train_df))
print("Val rows  :", len(val_df))
print("Test rows :", len(test_df))
print("INPUT_DIM :", INPUT_DIM)

np.savez(
    OUTPUT_DIR / "text_embedding_norm_stats.npz",
    feat_mean=feat_mean,
    feat_std=feat_std,
    input_dim=INPUT_DIM,
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


# Training
tf.keras.backend.clear_session()
gc.collect()

model = build_model(INPUT_DIM)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=mse_ccc_loss,
    metrics=[
        tf.keras.metrics.RootMeanSquaredError(name="rmse"),
        ccc_tf,
    ],
)

model.summary()

best_model_path = OUTPUT_DIR / "text_embedding_best_model.keras"
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
        MetricsLoggerCallback(),
    ],
)

fit_seconds = time.perf_counter() - fit_t0
print("fit_seconds:", fit_seconds)


# Predictions
model = tf.keras.models.load_model(
    best_model_path,
    custom_objects={
        "mse_ccc_loss": mse_ccc_loss,
        "ccc_tf": ccc_tf,
    },
)

y_pred_train = model.predict(X_train, batch_size=BATCH_SIZE, verbose=0).astype(np.float32)
y_pred_val = model.predict(X_val, batch_size=BATCH_SIZE, verbose=0).astype(np.float32)
y_pred_test = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0).astype(np.float32)


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
train_metrics.to_csv(OUTPUT_DIR / "text_embedding_train_metrics.csv", index=False, encoding="utf-8-sig")
val_metrics.to_csv(OUTPUT_DIR / "text_embedding_val_metrics.csv", index=False, encoding="utf-8-sig")
test_metrics.to_csv(OUTPUT_DIR / "text_embedding_metrics.csv", index=False, encoding="utf-8-sig")

save_predictions_csv(train_df, y_train, y_pred_train, OUTPUT_DIR / "text_embedding_train_predictions.csv")
save_predictions_csv(val_df, y_val, y_pred_val, OUTPUT_DIR / "text_embedding_val_predictions.csv")
save_predictions_csv(test_df, y_test, y_pred_test, OUTPUT_DIR / "text_embedding_predictions.csv")

history_df = pd.DataFrame(history_rows)
history_df.to_csv(OUTPUT_DIR / "text_embedding_history.csv", index=False, encoding="utf-8-sig")

plt.figure(figsize=(8, 5))
plt.plot(history_df["epoch"], history_df["train_rmse"], label="Train RMSE")
plt.plot(history_df["epoch"], history_df["val_rmse"], label="Validation RMSE")
plt.xlabel("Epoch")
plt.ylabel("RMSE")
plt.title(f"Text Embedding RMSE over Epochs ({SETTING})")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "text_embedding_rmse_curve.png", dpi=200)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(history_df["epoch"], history_df["train_ccc"], label="Train CCC")
plt.plot(history_df["epoch"], history_df["val_ccc"], label="Validation CCC")
plt.xlabel("Epoch")
plt.ylabel("CCC")
plt.title(f"Text Embedding CCC over Epochs ({SETTING})")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "text_embedding_ccc_curve.png", dpi=200)
plt.close()

summary_df = pd.DataFrame([{
    "setting": SETTING,
    "model": "text_distilbert_embedding_dense",
    "fit_seconds": fit_seconds,
    "epochs_ran": int(len(history_df)),
    "batch_size": BATCH_SIZE,
    "learning_rate_init": LEARNING_RATE,
    "input_dim": INPUT_DIM,
    "mean_mse": float(test_metrics.loc[test_metrics["target"] == "mean", "mse"].values[0]),
    "mean_rmse": float(test_metrics.loc[test_metrics["target"] == "mean", "rmse"].values[0]),
    "mean_ccc": float(test_metrics.loc[test_metrics["target"] == "mean", "ccc"].values[0]),
}])
summary_df.to_csv(OUTPUT_DIR / "text_embedding_summary.csv", index=False, encoding="utf-8-sig")

with open(OUTPUT_DIR / "artifacts.json", "w", encoding="utf-8") as f:
    json.dump({
        "setting": SETTING,
        "feature_dir": FEATURE_DIR,
        "train_manifest": str(TRAIN_CSV),
        "val_manifest": str(VAL_CSV),
        "test_manifest": str(TEST_CSV),
        "text_feature_col": TEXT_FEATURE_COL,
        "target_cols": TARGET_COLS,
        "model_path": str(best_model_path),
        "norm_stats_path": str(OUTPUT_DIR / "text_embedding_norm_stats.npz"),
        "input_dim": int(INPUT_DIM),
        "batch_size": int(BATCH_SIZE),
        "learning_rate": LEARNING_RATE,
    }, f, indent=2)

model.save(OUTPUT_DIR / "text_embedding_model_final.keras")

print("\nSaved to:", OUTPUT_DIR)