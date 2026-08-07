import os
import argparse
import json
import warnings
from pathlib import Path

ENV_DATA_ROOT = Path(os.getenv("SER_DATA_ROOT", "data"))
ENV_OUTPUT_ROOT = Path(os.getenv("SER_OUTPUT_ROOT", "outputs"))


warnings.filterwarnings("ignore")

import librosa
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Model


# CLI
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_dir",
    type=str,
    default=str(ENV_DATA_ROOT / 'msp_podcast_full'),
)
parser.add_argument(
    "--out_root",
    type=str,
    default=str(ENV_OUTPUT_ROOT / 'msp_podcast_full'),
)
parser.add_argument(
    "--feature",
    type=str,
    default="all",
    choices=["all", "wav2vec2", "mfcc", "text"],
)
args = parser.parse_args()


# Configuration
DATASET_DIR = Path(args.dataset_dir)
OUT_ROOT = Path(args.out_root)
OUT_ROOT.mkdir(parents=True, exist_ok=True)

SPLITS = ["train", "val", "test"]

AUDIO_PATH_COL = "audio_path"
TEXT_COL = "transcript"
TARGET_COLS = ["valence_norm", "arousal_norm", "dominance_norm"]

RANDOM_STATE = 42
TARGET_SR = 16000

# Wav2Vec2
WAV2VEC2_MODEL_NAME = "facebook/wav2vec2-base-960h"
WAV2VEC2_OUTPUT_DIR = OUT_ROOT / "wav2vec2_features"

# MFCC-13 + delta + delta-delta = 39 dim
MFCC_OUTPUT_DIR = OUT_ROOT / "mfcc_features"
N_MFCC = 13
MFCC_N_FFT = 400
MFCC_HOP_LENGTH = 160
MFCC_WIN_LENGTH = 400
USE_DELTA = True
USE_DELTA2 = True

# DistilBERT text embedding
TEXT_MODEL_NAME = "distilbert-base-uncased"
TEXT_OUTPUT_DIR = OUT_ROOT / "text_distilbert_features"
MAX_TEXT_LEN = 96
TEXT_BATCH_SIZE = 32

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Helpers
def clean_text(x):
    if pd.isna(x):
        return ""
    x = str(x).strip()
    x = " ".join(x.split())
    return x


def safe_name(x):
    x = str(x)
    for bad in ["/", "\\", ":", "*", "?", "\"", "<", ">", "|", " "]:
        x = x.replace(bad, "_")
    return x


def get_row_id(row, idx):
    if "utterance_id" in row and pd.notna(row["utterance_id"]):
        return safe_name(row["utterance_id"])
    return f"row_{idx:06d}"


def load_manifest(split):
    csv_path = DATASET_DIR / f"{split}_manifest.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing split manifest: {csv_path}")

    df = pd.read_csv(csv_path).copy()

    required_cols = ["utterance_id", AUDIO_PATH_COL] + TARGET_COLS
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")

    if TEXT_COL not in df.columns:
        df[TEXT_COL] = ""

    df[TEXT_COL] = df[TEXT_COL].apply(clean_text)
    df = df.dropna(subset=[AUDIO_PATH_COL] + TARGET_COLS).reset_index(drop=True)
    return df


def load_audio_mono_16k(audio_path):
    wav, sr = librosa.load(str(audio_path), sr=TARGET_SR, mono=True)
    wav = wav.astype(np.float32)
    return wav, TARGET_SR


def write_meta(out_dir, name, meta):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / name, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


# Wav2Vec2 feature extraction
def extract_wav2vec2_for_split(split, feature_extractor, model):
    out_dir = WAV2VEC2_OUTPUT_DIR
    seq_dir = out_dir / "sequence_features"
    mean_dir = out_dir / "mean_features"
    max_dir = out_dir / "max_features"

    seq_dir.mkdir(parents=True, exist_ok=True)
    mean_dir.mkdir(parents=True, exist_ok=True)
    max_dir.mkdir(parents=True, exist_ok=True)

    df = load_manifest(split)

    seq_paths = []
    mean_paths = []
    max_paths = []
    seq_lens = []
    hidden_sizes = []
    feature_exists = []

    print(f"\n===== WAV2VEC2 {split.upper()} =====")
    print("Rows:", len(df))

    for i, row in df.iterrows():
        row_id = get_row_id(row, i)
        audio_path = Path(str(row[AUDIO_PATH_COL]))

        seq_path = seq_dir / f"{row_id}.npy"
        mean_path = mean_dir / f"{row_id}.npy"
        max_path = max_dir / f"{row_id}.npy"

        try:
            wav, sr = load_audio_mono_16k(audio_path)

            inputs = feature_extractor(
                wav,
                sampling_rate=sr,
                return_tensors="pt",
                padding=False,
            )
            input_values = inputs.input_values.to(DEVICE)

            with torch.no_grad():
                outputs = model(input_values)
                seq_feat = outputs.last_hidden_state.squeeze(0).cpu().numpy().astype(np.float32)

            mean_feat = seq_feat.mean(axis=0).astype(np.float32)
            max_feat = seq_feat.max(axis=0).astype(np.float32)

            np.save(seq_path, seq_feat)
            np.save(mean_path, mean_feat)
            np.save(max_path, max_feat)

            seq_paths.append(str(seq_path))
            mean_paths.append(str(mean_path))
            max_paths.append(str(max_path))
            seq_lens.append(int(seq_feat.shape[0]))
            hidden_sizes.append(int(seq_feat.shape[1]))
            feature_exists.append(True)

        except Exception as e:
            print(f"[WARN] Wav2Vec2 failed: {audio_path} | {e}")
            seq_paths.append("")
            mean_paths.append("")
            max_paths.append("")
            seq_lens.append(0)
            hidden_sizes.append(0)
            feature_exists.append(False)

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(df)}")

    out_df = df.copy()
    out_df["wav2vec2_model"] = WAV2VEC2_MODEL_NAME
    out_df["sampling_rate_used"] = TARGET_SR
    out_df["seq_feat_path"] = seq_paths
    out_df["mean_feat_path"] = mean_paths
    out_df["max_feat_path"] = max_paths
    out_df["seq_len"] = seq_lens
    out_df["hidden_size"] = hidden_sizes
    out_df["feature_exists"] = feature_exists

    out_csv = out_dir / f"{split}_manifest.csv"
    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    write_meta(
        out_dir,
        f"{split}_meta.json",
        {
            "split": split,
            "input_manifest": str(DATASET_DIR / f"{split}_manifest.csv"),
            "output_manifest": str(out_csv),
            "model_name": WAV2VEC2_MODEL_NAME,
            "sampling_rate": TARGET_SR,
            "rows": int(len(out_df)),
            "feature_exists": int(sum(feature_exists)),
        },
    )

    print("Saved:", out_csv)


def extract_wav2vec2_all():
    print("Loading Wav2Vec2:", WAV2VEC2_MODEL_NAME)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(WAV2VEC2_MODEL_NAME)
    model = Wav2Vec2Model.from_pretrained(WAV2VEC2_MODEL_NAME)
    model.to(DEVICE)
    model.eval()

    for split in SPLITS:
        extract_wav2vec2_for_split(split, feature_extractor, model)


# MFCC feature extraction
def compute_mfcc39(wav, sr):
    mfcc = librosa.feature.mfcc(
        y=wav,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=MFCC_N_FFT,
        hop_length=MFCC_HOP_LENGTH,
        win_length=MFCC_WIN_LENGTH,
    )

    features = [mfcc]

    if USE_DELTA:
        features.append(librosa.feature.delta(mfcc))

    if USE_DELTA2:
        features.append(librosa.feature.delta(mfcc, order=2))

    feat = np.concatenate(features, axis=0).T.astype(np.float32)
    return feat


def extract_mfcc_for_split(split):
    out_dir = MFCC_OUTPUT_DIR
    seq_dir = out_dir / "sequence_features"
    seq_dir.mkdir(parents=True, exist_ok=True)

    df = load_manifest(split)

    seq_paths = []
    seq_lens = []
    hidden_sizes = []
    feature_exists = []

    print(f"\n===== MFCC {split.upper()} =====")
    print("Rows:", len(df))

    for i, row in df.iterrows():
        row_id = get_row_id(row, i)
        audio_path = Path(str(row[AUDIO_PATH_COL]))
        seq_path = seq_dir / f"{row_id}.npy"

        try:
            wav, sr = load_audio_mono_16k(audio_path)
            feat = compute_mfcc39(wav, sr)

            np.save(seq_path, feat)

            seq_paths.append(str(seq_path))
            seq_lens.append(int(feat.shape[0]))
            hidden_sizes.append(int(feat.shape[1]))
            feature_exists.append(True)

        except Exception as e:
            print(f"[WARN] MFCC failed: {audio_path} | {e}")
            seq_paths.append("")
            seq_lens.append(0)
            hidden_sizes.append(0)
            feature_exists.append(False)

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(df)}")

    out_df = df.copy()
    out_df["mfcc_n_mfcc"] = N_MFCC
    out_df["mfcc_use_delta"] = USE_DELTA
    out_df["mfcc_use_delta2"] = USE_DELTA2
    out_df["sampling_rate_used"] = TARGET_SR
    out_df["seq_feat_path"] = seq_paths
    out_df["seq_len"] = seq_lens
    out_df["hidden_size"] = hidden_sizes
    out_df["feature_exists"] = feature_exists

    out_csv = out_dir / f"{split}_manifest.csv"
    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    write_meta(
        out_dir,
        f"{split}_meta.json",
        {
            "split": split,
            "input_manifest": str(DATASET_DIR / f"{split}_manifest.csv"),
            "output_manifest": str(out_csv),
            "n_mfcc": N_MFCC,
            "use_delta": USE_DELTA,
            "use_delta2": USE_DELTA2,
            "sampling_rate": TARGET_SR,
            "rows": int(len(out_df)),
            "feature_exists": int(sum(feature_exists)),
        },
    )

    print("Saved:", out_csv)


def extract_mfcc_all():
    for split in SPLITS:
        extract_mfcc_for_split(split)


# DistilBERT text feature extraction
def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    masked_hidden = last_hidden_state * mask
    summed = masked_hidden.sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-9)
    return summed / denom


def encode_texts(texts, tokenizer, model):
    all_embs = []

    for start in range(0, len(texts), TEXT_BATCH_SIZE):
        batch_texts = texts[start:start + TEXT_BATCH_SIZE]

        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_TEXT_LEN,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"].to(DEVICE)
        attention_mask = enc["attention_mask"].to(DEVICE)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            emb = mean_pool(outputs.last_hidden_state, attention_mask)

        emb = emb.detach().cpu().numpy().astype(np.float32)
        all_embs.append(emb)

        print(f"Encoded {min(start + TEXT_BATCH_SIZE, len(texts))}/{len(texts)} texts")

    return np.concatenate(all_embs, axis=0)


def extract_text_for_split(split, tokenizer, model):
    out_dir = TEXT_OUTPUT_DIR
    emb_dir = out_dir / "text_embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)

    df = load_manifest(split)
    df[TEXT_COL] = df[TEXT_COL].apply(clean_text)

    before = len(df)
    df = df[df[TEXT_COL].str.len() > 0].reset_index(drop=True)
    after = len(df)

    print(f"\n===== TEXT {split.upper()} =====")
    print(f"Rows kept: {after}/{before}")

    texts = df[TEXT_COL].tolist()
    embs = encode_texts(texts, tokenizer, model)

    text_feat_paths = []

    for i, row in df.iterrows():
        row_id = get_row_id(row, i)
        out_path = emb_dir / f"{row_id}_text_distilbert.npy"
        np.save(out_path, embs[i])
        text_feat_paths.append(str(out_path))

    out_df = df.copy()
    out_df["text_model_name"] = TEXT_MODEL_NAME
    out_df["text_max_len"] = MAX_TEXT_LEN
    out_df["text_hidden_size"] = int(embs.shape[1])
    out_df["text_feat_path"] = text_feat_paths
    out_df["text_feature_type"] = "distilbert_mean_pool"

    out_csv = out_dir / f"{split}_manifest.csv"
    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    write_meta(
        out_dir,
        f"{split}_meta.json",
        {
            "split": split,
            "input_manifest": str(DATASET_DIR / f"{split}_manifest.csv"),
            "output_manifest": str(out_csv),
            "text_model_name": TEXT_MODEL_NAME,
            "text_feature_type": "distilbert_mean_pool",
            "max_text_len": MAX_TEXT_LEN,
            "hidden_size": int(embs.shape[1]),
            "rows_in": int(before),
            "rows_out": int(after),
        },
    )

    print("Saved:", out_csv)


def extract_text_all():
    print("Loading text model:", TEXT_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    model = AutoModel.from_pretrained(TEXT_MODEL_NAME)
    model.to(DEVICE)
    model.eval()

    for split in SPLITS:
        extract_text_for_split(split, tokenizer, model)


# Main
def main():
    print("DATASET_DIR:", DATASET_DIR)
    print("OUT_ROOT   :", OUT_ROOT)
    print("FEATURE    :", args.feature)
    print("DEVICE     :", DEVICE)

    if args.feature in ["all", "wav2vec2"]:
        extract_wav2vec2_all()

    if args.feature in ["all", "mfcc"]:
        extract_mfcc_all()

    if args.feature in ["all", "text"]:
        extract_text_all()

    print("\nDone. Features saved to:")
    print(OUT_ROOT)


if __name__ == "__main__":
    main()
