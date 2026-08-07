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
    "--feature_root",
    type=str,
    default=str(ENV_OUTPUT_ROOT / 'msp_podcast_full/mfcc_features'),
    help="Folder containing train_manifest.csv, val_manifest.csv, test_manifest.csv for MFCC features.",
)

parser.add_argument(
    "--out_dir",
    type=str,
    default=str(ENV_OUTPUT_ROOT / 'msp_podcast_full_results/mfcc_cnn1d'),
    help="Output folder for full MSP-Podcast MFCC+CNN1D training results.",
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
    default=1024,
    help="Fixed max sequence length. For full MSP MFCC, 1024 covers about 95% of train sequences.",
)

parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--num_workers", type=int, default=0)

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
FEATURE_ROOT = Path(args.feature_root)
TRAIN_CSV = FEATURE_ROOT / "train_manifest.csv"
VAL_CSV = FEATURE_ROOT / "val_manifest.csv"
TEST_CSV = FEATURE_ROOT / "test_manifest.csv"

OUTPUT_DIR = Path(args.out_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COL = "seq_feat_path"
TARGET_COLS = ["valence_norm", "arousal_norm", "dominance_norm"]
TARGET_NAMES = ["valence", "arousal", "dominance"]

RANDOM_STATE = 42
MAX_LEN = int(args.max_len)
BATCH_SIZE = int(args.batch_size)
EPOCHS = int(args.epochs)
LEARNING_RATE = float(args.learning_rate)
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
def load_manifest(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing manifest: {csv_path}")

    df = pd.read_csv(csv_path).copy()

    required_cols = [FEATURE_COL] + TARGET_COLS
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")

    df = df.dropna(subset=required_cols).copy()
    df = df[df[FEATURE_COL].astype(str).str.len() > 0].copy()

    # Keep only rows whose npy feature exists.
    exists_mask = df[FEATURE_COL].astype(str).map(lambda p: Path(p).exists())
    missing_count = int((~exists_mask).sum())
    if missing_count > 0:
        print(f"[WARN] Missing feature files in {csv_path}: {missing_count}")
    df = df[exists_mask].reset_index(drop=True)

    return df


def infer_feat_dim(df: pd.DataFrame) -> int:
    for p in df[FEATURE_COL].astype(str).head(100):
        arr = np.load(p)
        if arr.ndim == 2:
            return int(arr.shape[1])
    raise ValueError("Could not infer feature dimension from manifest.")


def compute_feature_norm_stats_streaming(df: pd.DataFrame, max_len: int, feature_col: str):
    """
    Streaming mean/std computation to avoid concatenating all frames from full MSP train set.
    """
    feat_dim = infer_feat_dim(df)
    total_count = 0
    sum_vec = np.zeros(feat_dim, dtype=np.float64)
    sumsq_vec = np.zeros(feat_dim, dtype=np.float64)

    print("\n===== COMPUTING TRAIN NORMALIZATION STATS =====")
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
            print(f"Processed stats {i + 1}/{len(df)}")

    if total_count == 0:
        raise ValueError("No frames found while computing normalization stats.")

    mean = sum_vec / total_count
    var = (sumsq_vec / total_count) - np.square(mean)
    var = np.maximum(var, 1e-12)
    std = np.sqrt(var)
    std = np.where(std < 1e-6, 1.0, std)

    return mean.astype(np.float32), std.astype(np.float32), feat_dim


def pad_and_normalize_one(path: str, max_len: int, feat_dim: int, feat_mean, feat_std):
    x = np.load(path).astype(np.float32)

    if x.ndim != 2:
        raise ValueError(f"Invalid feature shape {x.shape}: {path}")

    if x.shape[1] != feat_dim:
        raise ValueError(f"Unexpected feature dim {x.shape[1]} != {feat_dim}: {path}")

    x = x[:max_len]
    x = (x - feat_mean) / feat_std

    out = np.full((max_len, feat_dim), PAD_VALUE, dtype=np.float32)
    out[: x.shape[0], :] = x
    return out


class SequenceBatchGenerator(tf.keras.utils.Sequence):
    def __init__(
        self,
        df: pd.DataFrame,
        max_len: int,
        feat_dim: int,
        feat_mean: np.ndarray,
        feat_std: np.ndarray,
        batch_size: int,
        shuffle: bool,
        name: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.df = df.reset_index(drop=True)
        self.max_len = int(max_len)
        self.feat_dim = int(feat_dim)
        self.feat_mean = feat_mean.astype(np.float32)
        self.feat_std = feat_std.astype(np.float32)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.name = name
        self.indices = np.arange(len(self.df))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, batch_idx):
        batch_indices = self.indices[
            batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size
        ]
        batch_df = self.df.iloc[batch_indices]

        X = np.zeros((len(batch_df), self.max_len, self.feat_dim), dtype=np.float32)
        y = batch_df[TARGET_COLS].to_numpy(dtype=np.float32)

        for j, path in enumerate(batch_df[FEATURE_COL].astype(str)):
            X[j] = pad_and_normalize_one(
                path,
                self.max_len,
                self.feat_dim,
                self.feat_mean,
                self.feat_std,
            )

        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            rng = np.random.default_rng(RANDOM_STATE)
            rng.shuffle(self.indices)


def predict_generator_in_order(model, df, feat_mean, feat_std, max_len, feat_dim, batch_size):
    gen = SequenceBatchGenerator(
        df=df,
        max_len=max_len,
        feat_dim=feat_dim,
        feat_mean=feat_mean,
        feat_std=feat_std,
        batch_size=batch_size,
        shuffle=False,
        name="predict",
    )
    pred = model.predict(gen, verbose=1)
    return np.asarray(pred, dtype=np.float32)


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


# custom layers for masked pooling
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

    padding_mask = SequenceMaskFromPadValue(PAD_VALUE, name="padding_mask")(inputs)
    x = tf.keras.layers.Masking(mask_value=PAD_VALUE, name="input_masking")(inputs)

    x = conv_bn_relu(
        x,
        filters=BASE_FILTERS,
        kernel_size=KERNEL_SIZE_FIRST,
        dilation_rate=1,
        name_prefix="stem",
    )
    x = tf.keras.layers.SpatialDropout1D(CONV_DROPOUT, name="stem_dropout")(x)

    for block_id, dilation in enumerate(DILATION_RATES, start=1):
        x = dilated_residual_block(
            x,
            filters=DILATED_FILTERS,
            dilation_rate=dilation,
            block_id=block_id,
        )

    x = conv_bn_relu(
        x,
        filters=FINAL_FILTERS,
        kernel_size=1,
        dilation_rate=1,
        name_prefix="final_projection",
    )

    avg_pool = MaskedGlobalAveragePooling1D(name="masked_avg_pool")([x, padding_mask])
    max_pool = MaskedGlobalMaxPooling1D(name="masked_max_pool")([x, padding_mask])
    x = tf.keras.layers.Concatenate(name="concat_pool")([avg_pool, max_pool])

    x = tf.keras.layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(L2_REG),
        name="head_dense_128",
    )(x)
    x = tf.keras.layers.Dropout(HEAD_DROPOUT, name="head_dropout")(x)

    outputs = tf.keras.layers.Dense(3, activation="tanh", name="vad_output")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="msp_full_mfcc_cnn1d_vad")


# Main
def main():
    print("FEATURE_ROOT:", FEATURE_ROOT)
    print("OUTPUT_DIR  :", OUTPUT_DIR)
    print("MAX_LEN     :", MAX_LEN)
    print("BATCH_SIZE  :", BATCH_SIZE)
    print("EPOCHS      :", EPOCHS)
    print("LR          :", LEARNING_RATE)

    train_df = load_manifest(TRAIN_CSV)
    val_df = load_manifest(VAL_CSV)
    test_df = load_manifest(TEST_CSV)

    print("\n===== DATA =====")
    print("train rows:", len(train_df))
    print("val rows  :", len(val_df))
    print("test rows :", len(test_df))

    feat_mean, feat_std, feat_dim = compute_feature_norm_stats_streaming(
        train_df,
        max_len=MAX_LEN,
        feature_col=FEATURE_COL,
    )

    print("\n===== FEATURE SHAPE =====")
    print("max_len :", MAX_LEN)
    print("feat_dim:", feat_dim)

    np.savez(
        OUTPUT_DIR / "feature_norm_stats.npz",
        feat_mean=feat_mean,
        feat_std=feat_std,
        max_len=MAX_LEN,
        feat_dim=feat_dim,
    )

    train_gen = SequenceBatchGenerator(
        df=train_df,
        max_len=MAX_LEN,
        feat_dim=feat_dim,
        feat_mean=feat_mean,
        feat_std=feat_std,
        batch_size=BATCH_SIZE,
        shuffle=True,
        name="train",
    )

    val_gen = SequenceBatchGenerator(
        df=val_df,
        max_len=MAX_LEN,
        feat_dim=feat_dim,
        feat_mean=feat_mean,
        feat_std=feat_std,
        batch_size=BATCH_SIZE,
        shuffle=False,
        name="val",
    )

    tf.keras.backend.clear_session()
    gc.collect()

    model = build_cnn1d_model(MAX_LEN, feat_dim)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=mse_ccc_loss,
        metrics=[
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
            ccc_tf,
        ],
    )

    model.summary()

    best_model_path = OUTPUT_DIR / "cnn1d_best_model.keras"
    history_csv = OUTPUT_DIR / "cnn1d_history.csv"

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

    y_train = train_df[TARGET_COLS].to_numpy(dtype=np.float32)
    y_val = val_df[TARGET_COLS].to_numpy(dtype=np.float32)
    y_test = test_df[TARGET_COLS].to_numpy(dtype=np.float32)

    print("\nPredicting train...")
    y_pred_train = predict_generator_in_order(
        model, train_df, feat_mean, feat_std, MAX_LEN, feat_dim, BATCH_SIZE
    )
    print("Predicting val...")
    y_pred_val = predict_generator_in_order(
        model, val_df, feat_mean, feat_std, MAX_LEN, feat_dim, BATCH_SIZE
    )
    print("Predicting test...")
    y_pred_test = predict_generator_in_order(
        model, test_df, feat_mean, feat_std, MAX_LEN, feat_dim, BATCH_SIZE
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

    train_metrics.to_csv(OUTPUT_DIR / "cnn1d_train_metrics.csv", index=False, encoding="utf-8-sig")
    val_metrics.to_csv(OUTPUT_DIR / "cnn1d_val_metrics.csv", index=False, encoding="utf-8-sig")
    test_metrics.to_csv(OUTPUT_DIR / "cnn1d_metrics.csv", index=False, encoding="utf-8-sig")

    save_predictions_csv(train_df, y_train, y_pred_train, OUTPUT_DIR / "cnn1d_train_predictions.csv")
    save_predictions_csv(val_df, y_val, y_pred_val, OUTPUT_DIR / "cnn1d_val_predictions.csv")
    save_predictions_csv(test_df, y_test, y_pred_test, OUTPUT_DIR / "cnn1d_predictions.csv")

    history_df = pd.read_csv(history_csv)

    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, len(history_df) + 1), history_df["rmse"], label="Train RMSE")
    plt.plot(np.arange(1, len(history_df) + 1), history_df["val_rmse"], label="Validation RMSE")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.title("MSP full MFCC-CNN1D RMSE over Epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cnn1d_rmse_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, len(history_df) + 1), history_df["ccc_tf"], label="Train CCC")
    plt.plot(np.arange(1, len(history_df) + 1), history_df["val_ccc_tf"], label="Validation CCC")
    plt.xlabel("Epoch")
    plt.ylabel("CCC")
    plt.title("MSP full MFCC-CNN1D CCC over Epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cnn1d_ccc_curve.png", dpi=200)
    plt.close()

    model.save(OUTPUT_DIR / "cnn1d_model_final.keras")

    summary_df = pd.DataFrame([{
        "dataset": "msp_podcast_full",
        "feature_root": str(FEATURE_ROOT),
        "model": "mfcc_cnn1d_dilated_residual_full_msp",
        "fit_seconds": fit_seconds,
        "epochs_ran": int(len(history_df)),
        "batch_size": BATCH_SIZE,
        "learning_rate_init": LEARNING_RATE,
        "max_len": MAX_LEN,
        "feat_dim": feat_dim,
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
            "dataset": "msp_podcast_full",
            "feature_root": str(FEATURE_ROOT),
            "train_manifest": str(TRAIN_CSV),
            "val_manifest": str(VAL_CSV),
            "test_manifest": str(TEST_CSV),
            "feature_col": FEATURE_COL,
            "target_cols": TARGET_COLS,
            "model_path": str(best_model_path),
            "norm_stats_path": str(OUTPUT_DIR / "feature_norm_stats.npz"),
            "max_len": int(MAX_LEN),
            "feat_dim": int(feat_dim),
            "batch_size": int(BATCH_SIZE),
            "learning_rate": float(LEARNING_RATE),
            "dilation_rates": DILATION_RATES,
            "note": "Full MSP-Podcast MFCC 39 + CNN1D training. Uses streaming normalization and batch loading to avoid loading the full padded dataset into RAM.",
        }, f, indent=2)

    print("\nSaved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
