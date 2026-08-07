import os
from pathlib import Path
import random
import argparse

import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt

ENV_DATA_ROOT = Path(os.getenv("SER_DATA_ROOT", "data"))
ENV_OUTPUT_ROOT = Path(os.getenv("SER_OUTPUT_ROOT", "outputs"))



# CLI
parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=3, help="Number of random utterances to plot")
parser.add_argument("--seed", type=int, default=None, help="Random seed (optional)")
parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
args = parser.parse_args()


# Configuration
AUG_ROOT = Path(str(ENV_OUTPUT_ROOT / 'aug_runs_audio'))

SETTING_TO_TAG = {
    "baseline": "orig",
    "silence": "silence",
    "noise": "noise",
    "reverb": "reverb",
}

SPLIT = args.split
TARGET_SR = 16000
N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024


# Helpers
def get_manifest_path(setting: str) -> Path:
    return AUG_ROOT / setting / f"{SPLIT}_manifest.csv"


def load_manifest(setting: str) -> pd.DataFrame:
    path = get_manifest_path(setting)
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest: {path}")

    df = pd.read_csv(path).copy()

    required_cols = ["source_utt_id", "aug_tag", "audio_path"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"'{col}' not found in {path}")

    df["source_utt_id"] = df["source_utt_id"].astype(str)
    df["aug_tag"] = df["aug_tag"].astype(str)
    df["audio_path"] = df["audio_path"].astype(str)
    return df


def get_common_ids(setting_to_tag: dict[str, str]) -> list[str]:
    common_ids = None

    for setting, tag in setting_to_tag.items():
        df = load_manifest(setting)
        ids = set(df.loc[df["aug_tag"] == tag, "source_utt_id"].tolist())

        if common_ids is None:
            common_ids = ids
        else:
            common_ids = common_ids.intersection(ids)

    if not common_ids:
        raise ValueError("No common source_utt_id found across all selected settings/tags.")

    return sorted(common_ids)


def get_audio_path_for_utt(setting: str, tag: str, source_utt_id: str) -> Path:
    df = load_manifest(setting)
    sub = df[(df["source_utt_id"] == source_utt_id) & (df["aug_tag"] == tag)]

    if len(sub) == 0:
        raise ValueError(
            f"No row found for setting={setting}, tag={tag}, source_utt_id={source_utt_id}"
        )

    audio_path = Path(sub.iloc[0]["audio_path"])
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    return audio_path


def load_audio(path: Path, target_sr: int = 16000):
    wav, sr = librosa.load(path, sr=target_sr, mono=True)
    wav = np.asarray(wav, dtype=np.float32)
    return wav, sr


def compute_log_spectrogram(wav: np.ndarray, sr: int):
    stft = librosa.stft(
        wav,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window="hann",
    )
    mag = np.abs(stft)
    db = librosa.amplitude_to_db(mag, ref=np.max)
    return db


def plot_one_utterance(source_utt_id: str):
    signals = {}
    specs = {}

    for setting, tag in SETTING_TO_TAG.items():
        audio_path = get_audio_path_for_utt(setting, tag, source_utt_id)
        wav, sr = load_audio(audio_path, TARGET_SR)
        spec_db = compute_log_spectrogram(wav, sr)

        label = f"{setting}\n({tag})"
        signals[label] = (wav, sr, audio_path)
        specs[label] = spec_db

    labels = list(signals.keys())
    n_cols = len(labels)

    global_vmin = min(spec.min() for spec in specs.values())
    global_vmax = max(spec.max() for spec in specs.values())

    fig, axes = plt.subplots(
        2, n_cols,
        figsize=(4.8 * n_cols, 7),
        constrained_layout=True
    )

    if n_cols == 1:
        axes = np.array(axes).reshape(2, 1)

    for col, label in enumerate(labels):
        wav, sr, audio_path = signals[label]
        spec_db = specs[label]

        # Waveform
        ax1 = axes[0, col]
        librosa.display.waveshow(wav, sr=sr, ax=ax1)
        ax1.set_title(label, fontsize=14)
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Amplitude" if col == 0 else "")

        # Spectrogram
        ax2 = axes[1, col]
        img = librosa.display.specshow(
            spec_db,
            sr=sr,
            hop_length=HOP_LENGTH,
            x_axis="time",
            y_axis="hz",
            cmap="magma",
            vmin=global_vmin,
            vmax=global_vmax,
            ax=ax2,
        )
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Frequency (Hz)" if col == 0 else "")

    fig.suptitle(f"Utterance Augmentation Comparison: {source_utt_id}", fontsize=18)
    cbar = fig.colorbar(img, ax=axes[1, :], shrink=0.9, location="right")
    cbar.set_label("dB")

    out_path = AUG_ROOT / f"{source_utt_id}_aug_compare.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[SAVED] {out_path}")
    for label in labels:
        print(f"  {label}: {signals[label][2]}")


# Main
def main():
    common_ids = get_common_ids(SETTING_TO_TAG)

    if args.seed is not None:
        random.seed(args.seed)

    n = min(args.n, len(common_ids))
    selected_ids = random.sample(common_ids, n)

    print(f"Split: {SPLIT}")
    print(f"Found {len(common_ids)} common utterances")
    print(f"Selected {n} utterances:\n")

    for utt_id in selected_ids:
        print(" -", utt_id)

    print("\nPlotting...\n")
    for utt_id in selected_ids:
        plot_one_utterance(utt_id)

    print("\nDone.")


if __name__ == "__main__":
    main()