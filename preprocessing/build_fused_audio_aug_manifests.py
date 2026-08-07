from __future__ import annotations
import os

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from scipy.signal import fftconvolve
from sklearn.model_selection import train_test_split

ENV_DATA_ROOT = Path(os.getenv("SER_DATA_ROOT", "data"))
ENV_OUTPUT_ROOT = Path(os.getenv("SER_OUTPUT_ROOT", "outputs"))



# Configuration
SPLIT_DIR = Path(str(ENV_OUTPUT_ROOT / 'fixed_split_vad'))
TRAIN_CSV = SPLIT_DIR / "train.csv"
TEST_CSV = SPLIT_DIR / "test.csv"

OUT_ROOT = Path(str(ENV_OUTPUT_ROOT / 'aug_runs_audio_fused'))
OUT_ROOT.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
VAL_SIZE = 0.10
TARGET_SR = 16000

# Keep fused settings separate from single-augmentation runs.
FUSED_PIPELINES = {
    "silence_then_noise": ["silence", "noise"],
    "noise_then_silence": ["noise", "silence"],
    "silence_then_reverb": ["silence", "reverb"],
    "reverb_then_silence": ["reverb", "silence"],
    "noise_then_reverb": ["noise", "reverb"],
    "reverb_then_noise": ["reverb", "noise"],
    # Optional three-step variants:
    # "silence_then_noise_then_reverb": ["silence", "noise", "reverb"],
    # "reverb_then_noise_then_silence": ["reverb", "noise", "silence"],
}

TARGET_COLS = ["valence_norm", "arousal_norm", "dominance_norm"]


# Helpers
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


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

    raise ValueError("Could not find audio path column in CSV.")


def load_audio(path: str | Path, target_sr: int = 16000) -> tuple[np.ndarray, int]:
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


def save_audio(path: str | Path, wav: np.ndarray, sr: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wav = np.clip(np.asarray(wav, dtype=np.float32), -1.0, 1.0)
    sf.write(str(path), wav, sr)


def trim_silence(wav: np.ndarray, sr: int, top_db: int = 30) -> np.ndarray:
    if len(wav) == 0:
        return wav
    trimmed, _ = librosa.effects.trim(wav, top_db=top_db)
    if len(trimmed) < int(0.15 * sr):
        return wav
    return trimmed.astype(np.float32)


def add_gaussian_noise_snr(
    wav: np.ndarray,
    snr_db_low: float = 15.0,
    snr_db_high: float = 25.0,
) -> np.ndarray:
    if len(wav) == 0:
        return wav

    snr_db = np.random.uniform(snr_db_low, snr_db_high)
    signal_rms = np.sqrt(np.mean(np.square(wav)) + 1e-12)
    noise_rms = signal_rms / (10 ** (snr_db / 20.0))
    noise = np.random.normal(0.0, noise_rms, size=wav.shape).astype(np.float32)

    out = wav + noise
    peak = np.max(np.abs(out)) + 1e-8
    if peak > 1.0:
        out = out / peak
    return out.astype(np.float32)


def synthetic_ir(sr: int, rt60_range=(0.15, 0.35), early_reflections: int = 5) -> np.ndarray:
    rt60 = np.random.uniform(*rt60_range)
    ir_len = max(int(rt60 * sr), int(0.12 * sr))

    t = np.arange(ir_len, dtype=np.float32) / sr
    decay = np.exp(-3.0 * t / max(rt60, 1e-4))

    ir = np.zeros(ir_len, dtype=np.float32)
    ir[0] = 1.0

    for _ in range(early_reflections):
        delay = np.random.randint(int(0.003 * sr), min(ir_len - 1, int(0.05 * sr)))
        amp = np.random.uniform(0.1, 0.5)
        ir[delay] += amp

    noise_tail = np.random.normal(0.0, 1.0, ir_len).astype(np.float32)
    ir += 0.08 * noise_tail * decay
    ir *= decay

    ir /= np.max(np.abs(ir)) + 1e-8
    return ir.astype(np.float32)


def add_light_reverb(wav: np.ndarray, sr: int, wet_range=(0.08, 0.20)) -> np.ndarray:
    if len(wav) == 0:
        return wav

    ir = synthetic_ir(sr)
    wet = fftconvolve(wav, ir, mode="full")[: len(wav)].astype(np.float32)
    wet_mix = np.random.uniform(*wet_range)

    out = (1.0 - wet_mix) * wav + wet_mix * wet
    peak = np.max(np.abs(out)) + 1e-8
    if peak > 1.0:
        out = out / peak
    return out.astype(np.float32)


def apply_single_op(wav: np.ndarray, sr: int, op_name: str) -> np.ndarray:
    if op_name == "silence":
        return trim_silence(wav, sr)
    if op_name == "noise":
        return add_gaussian_noise_snr(wav)
    if op_name == "reverb":
        return add_light_reverb(wav, sr)
    raise ValueError(f"Unknown op: {op_name}")


def apply_fused_pipeline(wav: np.ndarray, sr: int, ops: list[str]) -> np.ndarray:
    x = wav.copy()
    for op_name in ops:
        x = apply_single_op(x, sr, op_name)
    return x.astype(np.float32)


def make_aug_row(row: pd.Series, new_audio_path: Path, suffix: str) -> dict:
    out = row.to_dict()
    out["aug_tag"] = suffix
    out["is_augmented"] = 0 if suffix == "orig" else 1
    out["source_utt_id"] = str(row.get("utterance_id", row.name))

    if "utterance_id" in out:
        out["utterance_id"] = f"{out['utterance_id']}__{suffix}"

    audio_col = out["_audio_col_name"]
    out[audio_col] = str(new_audio_path)
    return out


def prepare_split(train_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = np.arange(len(train_df))
    train_idx, val_idx = train_test_split(
        idx,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    train_split = train_df.iloc[train_idx].reset_index(drop=True).copy()
    val_split = train_df.iloc[val_idx].reset_index(drop=True).copy()
    return train_split, val_split


# Main
def main():
    seed_everything(RANDOM_STATE)

    train_df = pd.read_csv(TRAIN_CSV).copy()
    test_df = pd.read_csv(TEST_CSV).copy()

    missing_targets = [c for c in TARGET_COLS if c not in train_df.columns or c not in test_df.columns]
    if missing_targets:
        raise ValueError(f"Missing target columns: {missing_targets}")

    audio_col = find_audio_col(train_df)
    if audio_col not in test_df.columns:
        raise ValueError(f"Audio column '{audio_col}' missing in test.csv")

    train_df["_audio_col_name"] = audio_col
    test_df["_audio_col_name"] = audio_col

    train_df = train_df.dropna(subset=TARGET_COLS).reset_index(drop=True)
    test_df = test_df.dropna(subset=TARGET_COLS).reset_index(drop=True)

    train_split, val_split = prepare_split(train_df)

    print("Audio column:", audio_col)
    print("Train split:", len(train_split))
    print("Val split  :", len(val_split))
    print("Test split :", len(test_df))

    for setting, ops in FUSED_PIPELINES.items():
        print(f"\n[INFO] Building fused setting: {setting} | ops={ops}")
        setting_dir = OUT_ROOT / setting
        wav_dir = setting_dir / "wav_aug"
        setting_dir.mkdir(parents=True, exist_ok=True)

        train_rows = []
        val_rows = []
        test_rows = []

        # Training: original + one fused variant
        for _, row in train_split.iterrows():
            wav_path = Path(str(row[audio_col]))
            wav, sr = load_audio(wav_path, TARGET_SR)

            orig_out_path = wav_dir / "train" / f"{wav_path.stem}__orig.wav"
            save_audio(orig_out_path, wav, sr)
            train_rows.append(make_aug_row(row, orig_out_path, "orig"))

            fused_wav = apply_fused_pipeline(wav, sr, ops)
            fused_tag = setting
            fused_out_path = wav_dir / "train" / f"{wav_path.stem}__{fused_tag}.wav"
            save_audio(fused_out_path, fused_wav, sr)
            train_rows.append(make_aug_row(row, fused_out_path, fused_tag))

        # Validation: original only
        for _, row in val_split.iterrows():
            wav_path = Path(str(row[audio_col]))
            wav, sr = load_audio(wav_path, TARGET_SR)

            out_path = wav_dir / "val" / f"{wav_path.stem}__orig.wav"
            save_audio(out_path, wav, sr)
            val_rows.append(make_aug_row(row, out_path, "orig"))

        # Test: original only
        for _, row in test_df.iterrows():
            wav_path = Path(str(row[audio_col]))
            wav, sr = load_audio(wav_path, TARGET_SR)

            out_path = wav_dir / "test" / f"{wav_path.stem}__orig.wav"
            save_audio(out_path, wav, sr)
            test_rows.append(make_aug_row(row, out_path, "orig"))

        train_manifest = pd.DataFrame(train_rows).drop(columns=["_audio_col_name"], errors="ignore")
        val_manifest = pd.DataFrame(val_rows).drop(columns=["_audio_col_name"], errors="ignore")
        test_manifest = pd.DataFrame(test_rows).drop(columns=["_audio_col_name"], errors="ignore")

        train_manifest.to_csv(setting_dir / "train_manifest.csv", index=False, encoding="utf-8-sig")
        val_manifest.to_csv(setting_dir / "val_manifest.csv", index=False, encoding="utf-8-sig")
        test_manifest.to_csv(setting_dir / "test_manifest.csv", index=False, encoding="utf-8-sig")

        split_info = {
            "setting": setting,
            "ops": ops,
            "audio_col": audio_col,
            "random_state": RANDOM_STATE,
            "val_size": VAL_SIZE,
            "target_sr": TARGET_SR,
            "train_rows": int(len(train_manifest)),
            "val_rows": int(len(val_manifest)),
            "test_rows": int(len(test_manifest)),
        }
        with open(setting_dir / "split_info.json", "w", encoding="utf-8") as f:
            json.dump(split_info, f, indent=2)

        print(f"[DONE] {setting}: train={len(train_manifest)}, val={len(val_manifest)}, test={len(test_manifest)}")

    print("\nAll fused settings created in:", OUT_ROOT)


if __name__ == "__main__":
    main()