from __future__ import annotations

import os
import argparse
import time
import shutil
from pathlib import Path

ENV_DATA_ROOT = Path(os.getenv("SER_DATA_ROOT", "data"))
ENV_OUTPUT_ROOT = Path(os.getenv("SER_OUTPUT_ROOT", "outputs"))


# CLI
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=str, default="0")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--limit", type=int, default=0, help="0 = process all")
parser.add_argument("--overwrite_features", action="store_true")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model


# Paths
ROOT = Path(str(ENV_OUTPUT_ROOT / 'aug_runs_audio/silence'))

MISSING_CSV = ROOT / "missing_silence_audit" / "missing_silence_orig_rows.csv"

WAV_AUG_ROOT = ROOT / "wav_aug"

WAV2VEC2_DIR = ROOT / "wav2vec2_features"
MFCC_DIR = ROOT / "mfcc_features"

WAV2VEC2_SEQ_DIR = WAV2VEC2_DIR / "sequence_features"
WAV2VEC2_MEAN_DIR = WAV2VEC2_DIR / "mean_features"
WAV2VEC2_MAX_DIR = WAV2VEC2_DIR / "max_features"

MFCC_SEQ_DIR = MFCC_DIR / "sequence_features"
MFCC_MEAN_DIR = MFCC_DIR / "mean_features"
MFCC_MAX_DIR = MFCC_DIR / "max_features"

TARGET_SR = 16000
MODEL_NAME = "facebook/wav2vec2-base-960h"

SPLITS = ["train", "val", "test"]
AUG_SUFFIXES = ["__orig", "__silence", "__noise", "__reverb"]


# Helpers
def strip_aug_suffix(x: str) -> str:
    x = str(x)
    for suffix in AUG_SUFFIXES:
        if x.endswith(suffix):
            return x[: -len(suffix)]
    return x


def load_audio(path: str | Path, target_sr: int = 16000):
    wav, sr = sf.read(str(path))
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


def save_audio(path: Path, wav: np.ndarray, sr: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    wav = np.clip(np.asarray(wav, dtype=np.float32), -1.0, 1.0)
    sf.write(str(path), wav, sr)


def trim_silence(wav: np.ndarray, sr: int, top_db: int = 30):
    if len(wav) == 0:
        return wav

    trimmed, _ = librosa.effects.trim(wav, top_db=top_db)

    if len(trimmed) < int(0.15 * sr):
        return wav

    return trimmed.astype(np.float32)


def make_source_key(df: pd.DataFrame):
    if "source_utt_id" in df.columns:
        return df["source_utt_id"].astype(str).map(strip_aug_suffix)
    if "utterance_id" in df.columns:
        return df["utterance_id"].astype(str).map(strip_aug_suffix)
    raise ValueError("No source_utt_id or utterance_id column found.")


def get_aug_tag(df: pd.DataFrame):
    if "aug_tag" in df.columns:
        return df["aug_tag"].astype(str)

    if "utterance_id" in df.columns:
        u = df["utterance_id"].astype(str)
        out = pd.Series(["orig"] * len(df), index=df.index)
        out[u.str.endswith("__silence")] = "silence"
        out[u.str.endswith("__noise")] = "noise"
        out[u.str.endswith("__reverb")] = "reverb"
        return out

    raise ValueError("No aug_tag or utterance_id column found.")


def backup_once(path: Path):
    backup = path.with_suffix(path.suffix + ".bak_before_missing_silence_patch")
    if not backup.exists():
        shutil.copy2(path, backup)
        print("[BACKUP]", backup)


def update_common_row(row: pd.Series, base_id: str, split: str, audio_path: Path) -> dict:
    out = row.to_dict()

    out["utterance_id"] = f"{base_id}__silence"
    out["source_utt_id"] = base_id
    out["aug_tag"] = "silence"
    out["is_augmented"] = 1

    if "audio_path" in out:
        out["audio_path"] = str(audio_path)
    elif "wav_path" in out:
        out["wav_path"] = str(audio_path)

    return out


def extract_wav2vec2_sequence(wav: np.ndarray, sr: int, processor, model, device):
    inputs = processor(
        wav,
        sampling_rate=sr,
        return_tensors="pt",
        padding=False,
    )

    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        outputs = model(input_values)
        seq = outputs.last_hidden_state.squeeze(0).detach().cpu().numpy().astype(np.float32)

    return seq


def extract_mfcc39_sequence(wav: np.ndarray, sr: int):
    mfcc = librosa.feature.mfcc(
        y=wav,
        sr=sr,
        n_mfcc=13,
        n_fft=400,
        hop_length=160,
        win_length=400,
        window="hann",
        center=True,
        n_mels=40,
        fmin=0,
        fmax=8000,
    )

    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    feat = np.concatenate([mfcc, delta, delta2], axis=0).T.astype(np.float32)
    return feat


def save_seq_mean_max(seq: np.ndarray, seq_path: Path, mean_path: Path, max_path: Path):
    seq_path.parent.mkdir(parents=True, exist_ok=True)
    mean_path.parent.mkdir(parents=True, exist_ok=True)
    max_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(seq_path, seq.astype(np.float32))
    np.save(mean_path, seq.mean(axis=0).astype(np.float32))
    np.save(max_path, seq.max(axis=0).astype(np.float32))


# Main
def main():
    if not MISSING_CSV.exists():
        raise FileNotFoundError(f"Missing CSV not found: {MISSING_CSV}")

    missing_df = pd.read_csv(MISSING_CSV).copy()

    if args.limit and args.limit > 0:
        missing_df = missing_df.head(args.limit).copy()

    print("Missing rows to process:", len(missing_df))

    for d in [
        WAV2VEC2_SEQ_DIR, WAV2VEC2_MEAN_DIR, WAV2VEC2_MAX_DIR,
        MFCC_SEQ_DIR, MFCC_MEAN_DIR, MFCC_MAX_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    # Load current manifests.
    wav_manifests = {}
    mfcc_manifests = {}

    for split in SPLITS:
        wav_p = WAV2VEC2_DIR / f"{split}_manifest.csv"
        mfcc_p = MFCC_DIR / f"{split}_manifest.csv"

        wav_df = pd.read_csv(wav_p).copy()
        mfcc_df = pd.read_csv(mfcc_p).copy()

        wav_df["_source_key_tmp"] = make_source_key(wav_df)
        wav_df["_aug_tag_tmp"] = get_aug_tag(wav_df)

        mfcc_df["_source_key_tmp"] = make_source_key(mfcc_df)
        mfcc_df["_aug_tag_tmp"] = get_aug_tag(mfcc_df)

        wav_manifests[split] = wav_df
        mfcc_manifests[split] = mfcc_df

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    new_wav_rows_by_split = {s: [] for s in SPLITS}
    new_mfcc_rows_by_split = {s: [] for s in SPLITS}

    t0 = time.perf_counter()

    for idx, row in missing_df.iterrows():
        split = str(row["_old_split"])
        base_id = strip_aug_suffix(str(row["_source_key"]))

        if split not in SPLITS:
            raise ValueError(f"Invalid split: {split}")

        # Skip rows that already exist in both manifests.
        wav_df = wav_manifests[split]
        mfcc_df = mfcc_manifests[split]

        already_wav = ((wav_df["_source_key_tmp"] == base_id) & (wav_df["_aug_tag_tmp"] == "silence")).any()
        already_mfcc = ((mfcc_df["_source_key_tmp"] == base_id) & (mfcc_df["_aug_tag_tmp"] == "silence")).any()

        if already_wav and already_mfcc:
            print(f"[SKIP existing] {base_id}")
            continue

        orig_wav_rows = wav_df[(wav_df["_source_key_tmp"] == base_id) & (wav_df["_aug_tag_tmp"] == "orig")]
        orig_mfcc_rows = mfcc_df[(mfcc_df["_source_key_tmp"] == base_id) & (mfcc_df["_aug_tag_tmp"] == "orig")]

        if len(orig_wav_rows) == 0:
            print(f"[WARN] Missing orig wav2vec2 row for {base_id}")
            continue
        if len(orig_mfcc_rows) == 0:
            print(f"[WARN] Missing orig mfcc row for {base_id}")
            continue

        orig_wav_row = orig_wav_rows.iloc[0]
        orig_mfcc_row = orig_mfcc_rows.iloc[0]

        source_audio_path = Path(str(row["audio_path"]))
        if not source_audio_path.exists():
            print(f"[WARN] Missing source audio: {source_audio_path}")
            continue

        silence_audio_path = WAV_AUG_ROOT / split / f"{base_id}__silence.wav"

        wav, sr = load_audio(source_audio_path, TARGET_SR)
        silence_wav = trim_silence(wav, sr)
        save_audio(silence_audio_path, silence_wav, sr)

        utt_id = f"{base_id}__silence"

        # Wav2Vec2
        wav_seq_path = WAV2VEC2_SEQ_DIR / f"{utt_id}.npy"
        wav_mean_path = WAV2VEC2_MEAN_DIR / f"{utt_id}.npy"
        wav_max_path = WAV2VEC2_MAX_DIR / f"{utt_id}.npy"

        if args.overwrite_features or not wav_seq_path.exists():
            wav_seq = extract_wav2vec2_sequence(silence_wav, sr, processor, model, device)
            save_seq_mean_max(wav_seq, wav_seq_path, wav_mean_path, wav_max_path)
        else:
            wav_seq = np.load(wav_seq_path)

        new_wav_row = update_common_row(orig_wav_row, base_id, split, silence_audio_path)
        new_wav_row["wav2vec2_model"] = MODEL_NAME
        new_wav_row["sampling_rate_used"] = sr
        new_wav_row["n_audio_samples"] = int(len(silence_wav))
        new_wav_row["seq_len"] = int(wav_seq.shape[0])
        new_wav_row["hidden_size"] = int(wav_seq.shape[1])
        new_wav_row["seq_feat_path"] = str(wav_seq_path)
        new_wav_row["mean_feat_path"] = str(wav_mean_path)
        new_wav_row["max_feat_path"] = str(wav_max_path)
        if "model_name" in new_wav_row:
            new_wav_row["model_name"] = MODEL_NAME
        if "feature_sr" in new_wav_row:
            new_wav_row["feature_sr"] = sr
        if "wave_num_samples" in new_wav_row:
            new_wav_row["wave_num_samples"] = int(len(silence_wav))

        new_wav_rows_by_split[split].append(new_wav_row)

        # MFCC
        mfcc_seq_path = MFCC_SEQ_DIR / f"{utt_id}.npy"
        mfcc_mean_path = MFCC_MEAN_DIR / f"{utt_id}.npy"
        mfcc_max_path = MFCC_MAX_DIR / f"{utt_id}.npy"

        if args.overwrite_features or not mfcc_seq_path.exists():
            mfcc_seq = extract_mfcc39_sequence(silence_wav, sr)
            save_seq_mean_max(mfcc_seq, mfcc_seq_path, mfcc_mean_path, mfcc_max_path)
        else:
            mfcc_seq = np.load(mfcc_seq_path)

        new_mfcc_row = update_common_row(orig_mfcc_row, base_id, split, silence_audio_path)
        new_mfcc_row["sampling_rate_used"] = sr
        new_mfcc_row["n_audio_samples"] = int(len(silence_wav))
        new_mfcc_row["seq_len"] = int(mfcc_seq.shape[0])
        new_mfcc_row["hidden_size"] = int(mfcc_seq.shape[1])
        new_mfcc_row["seq_feat_path"] = str(mfcc_seq_path)
        new_mfcc_row["mean_feat_path"] = str(mfcc_mean_path)
        new_mfcc_row["max_feat_path"] = str(mfcc_max_path)
        if "feature_sr" in new_mfcc_row:
            new_mfcc_row["feature_sr"] = sr
        if "wave_num_samples" in new_mfcc_row:
            new_mfcc_row["wave_num_samples"] = int(len(silence_wav))

        new_mfcc_rows_by_split[split].append(new_mfcc_row)

        if (idx + 1) % 50 == 0:
            elapsed = time.perf_counter() - t0
            print(f"[{idx + 1}/{len(missing_df)}] processed, elapsed={elapsed/60:.1f} min")

    # Save the patched manifests.
    for split in SPLITS:
        wav_new = pd.DataFrame(new_wav_rows_by_split[split])
        mfcc_new = pd.DataFrame(new_mfcc_rows_by_split[split])

        wav_p = WAV2VEC2_DIR / f"{split}_manifest.csv"
        mfcc_p = MFCC_DIR / f"{split}_manifest.csv"

        if len(wav_new) > 0:
            backup_once(wav_p)
            wav_df = wav_manifests[split].drop(columns=["_source_key_tmp", "_aug_tag_tmp"], errors="ignore")
            wav_new = wav_new.drop(columns=["_source_key_tmp", "_aug_tag_tmp"], errors="ignore")
            out = pd.concat([wav_df, wav_new], ignore_index=True)
            out.to_csv(wav_p, index=False, encoding="utf-8-sig")
            print(f"[SAVE] {wav_p}: +{len(wav_new)} rows -> {len(out)}")

        if len(mfcc_new) > 0:
            backup_once(mfcc_p)
            mfcc_df = mfcc_manifests[split].drop(columns=["_source_key_tmp", "_aug_tag_tmp"], errors="ignore")
            mfcc_new = mfcc_new.drop(columns=["_source_key_tmp", "_aug_tag_tmp"], errors="ignore")
            out = pd.concat([mfcc_df, mfcc_new], ignore_index=True)
            out.to_csv(mfcc_p, index=False, encoding="utf-8-sig")
            print(f"[SAVE] {mfcc_p}: +{len(mfcc_new)} rows -> {len(out)}")

    print("\n[DONE]")
    print("Run audit again to verify orig=10039 and silence=10039.")


if __name__ == "__main__":
    main()