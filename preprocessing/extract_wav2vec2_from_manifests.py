from __future__ import annotations

import os
import json
import warnings
from pathlib import Path
from typing import Optional

ENV_DATA_ROOT = Path(os.getenv("SER_DATA_ROOT", "data"))
ENV_OUTPUT_ROOT = Path(os.getenv("SER_OUTPUT_ROOT", "outputs"))


warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch
from transformers import AutoProcessor, Wav2Vec2Model


# Configuration
AUG_ROOT = Path(str(ENV_OUTPUT_ROOT / 'aug_runs_audio'))
SETTINGS = [
    "baseline",
    "silence",
    "noise",
    "reverb",
    "silence_noise",
]

MODEL_NAME = "facebook/wav2vec2-base-960h"
TARGET_SR = 16000
BATCH_INFO_EVERY = 100  # print every N files


# Device setup
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")

print("Using device:", DEVICE)


# Load model
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()


# Helpers
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
        vals = df[c].astype(str)
        if vals.str.lower().str.endswith(".wav").mean() > 0.5:
            return c

    raise ValueError("Could not find audio path column in manifest.")


def load_audio_for_wav2vec2(audio_path: str | Path, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(str(audio_path))
    wav = np.asarray(wav, dtype=np.float32)

    if wav.ndim == 2:
        wav = wav.mean(axis=1)

    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    if np.max(np.abs(wav)) > 1.5:
        wav = wav / max(np.max(np.abs(wav)), 1e-8)

    wav = np.clip(wav, -1.0, 1.0).astype(np.float32)
    return wav, sr


@torch.no_grad()
def extract_wav2vec2_features(wav: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inputs = processor(
        wav,
        sampling_rate=sr,
        return_tensors="pt",
        padding=False,
    )

    input_values = inputs["input_values"].to(DEVICE)
    outputs = model(input_values=input_values)

    seq_feat = outputs.last_hidden_state[0].detach().cpu().numpy().astype(np.float32)   # (T, H)
    mean_feat = seq_feat.mean(axis=0).astype(np.float32)                                  # (H,)
    max_feat = seq_feat.max(axis=0).astype(np.float32)                                    # (H,)
    return seq_feat, mean_feat, max_feat


def get_utt_id(row: pd.Series, audio_path: Path, row_idx: int) -> str:
    if "utterance_id" in row and pd.notna(row["utterance_id"]):
        return str(row["utterance_id"])
    return f"{audio_path.stem}_{row_idx:06d}"


def build_feature_manifest_for_csv(
    manifest_csv: Path,
    out_dir: Path,
) -> Path:
    df = pd.read_csv(manifest_csv).copy()
    audio_col = find_audio_col(df)

    seq_dir = out_dir / "sequence_features"
    pooled_dir = out_dir / "pooled_features"
    seq_dir.mkdir(parents=True, exist_ok=True)
    pooled_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    total = len(df)

    for i, row in df.iterrows():
        audio_path = Path(str(row[audio_col]))

        if not audio_path.exists():
            print(f"[WARN] Missing audio: {audio_path}")
            continue

        try:
            wav, sr = load_audio_for_wav2vec2(audio_path, TARGET_SR)
            seq_feat, mean_feat, max_feat = extract_wav2vec2_features(wav, sr)

            utt_id = get_utt_id(row, audio_path, i)

            seq_path = seq_dir / f"{utt_id}.npy"
            mean_path = pooled_dir / f"{utt_id}_mean.npy"
            max_path = pooled_dir / f"{utt_id}_max.npy"

            np.save(seq_path, seq_feat)
            np.save(mean_path, mean_feat)
            np.save(max_path, max_feat)

            out_row = row.to_dict()
            out_row["model_name"] = MODEL_NAME
            out_row["feature_sr"] = sr
            out_row["wave_num_samples"] = int(len(wav))
            out_row["seq_len"] = int(seq_feat.shape[0])
            out_row["hidden_size"] = int(seq_feat.shape[1])
            out_row["seq_feat_path"] = str(seq_path)
            out_row["mean_feat_path"] = str(mean_path)
            out_row["max_feat_path"] = str(max_path)

            rows.append(out_row)

            if (i + 1) % BATCH_INFO_EVERY == 0 or (i + 1) == total:
                print(f"[INFO] {manifest_csv.name}: processed {i+1}/{total}")

        except Exception as e:
            print(f"[WARN] Failed on: {audio_path}")
            print("       Error:", e)

    out_df = pd.DataFrame(rows)
    out_csv = out_dir / manifest_csv.name
    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    meta = {
        "input_manifest": str(manifest_csv),
        "output_manifest": str(out_csv),
        "model_name": MODEL_NAME,
        "target_sr": TARGET_SR,
        "num_rows_in": int(len(df)),
        "num_rows_out": int(len(out_df)),
    }
    with open(out_dir / f"{manifest_csv.stem}_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return out_csv


def process_setting(setting: str) -> None:
    setting_dir = AUG_ROOT / setting
    if not setting_dir.exists():
        print(f"[WARN] Setting not found: {setting_dir}")
        return

    out_dir = setting_dir / "wav2vec2_features"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifests = [
        setting_dir / "train_manifest.csv",
        setting_dir / "val_manifest.csv",
        setting_dir / "test_manifest.csv",
    ]

    print(f"\n===== PROCESS SETTING: {setting} =====")
    for manifest_csv in manifests:
        if not manifest_csv.exists():
            print(f"[WARN] Missing manifest: {manifest_csv}")
            continue
        build_feature_manifest_for_csv(manifest_csv, out_dir)

    print(f"[DONE] {setting} -> {out_dir}")


# Main
def main():
    for setting in SETTINGS:
        process_setting(setting)


if __name__ == "__main__":
    main()