import os
import json
from pathlib import Path

import pandas as pd

ENV_DATA_ROOT = Path(os.getenv("SER_DATA_ROOT", "data"))
ENV_OUTPUT_ROOT = Path(os.getenv("SER_OUTPUT_ROOT", "outputs"))



DATA_DIR = Path(str(ENV_DATA_ROOT / 'msp_podcast_iemocap_duration_matched'))
MANIFEST_PATH = DATA_DIR / "manifest_full.csv"

OUTPUT_DIR = DATA_DIR / "splits"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# Match IEMOCAP split sizes
N_TRAIN = 7082
N_VAL = 787
N_TEST = 2170
N_TOTAL = N_TRAIN + N_VAL + N_TEST

TARGET_COLS = ["valence_norm", "arousal_norm", "dominance_norm"]


def main():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Missing manifest: {MANIFEST_PATH}")

    df = pd.read_csv(MANIFEST_PATH)

    print("Loaded:", MANIFEST_PATH)
    print("Rows before cleaning:", len(df))
    print("Columns:", list(df.columns))

    required_cols = [
        "utterance_id",
        "audio_path",
        "duration",
        "transcript",
        "valence_norm",
        "arousal_norm",
        "dominance_norm",
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df.dropna(subset=required_cols).copy()

    # Keep only rows whose wav file exists
    df = df[df["audio_path"].astype(str).map(lambda p: Path(p).exists())].copy()

    # Numeric cleanup
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce")
    for col in TARGET_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["duration"] + TARGET_COLS).copy()

    # Keep IEMOCAP-duration-matched range
    df = df[(df["duration"] >= 2.0) & (df["duration"] <= 6.0)].copy()

    print("Rows after cleaning/filtering:", len(df))

    if len(df) < N_TOTAL:
        raise ValueError(
            f"Need at least {N_TOTAL} samples, but only found {len(df)}. "
            "Download more samples or relax duration range."
        )

    # Shuffle and keep same total size as IEMOCAP
    df = df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    df = df.iloc[:N_TOTAL].copy().reset_index(drop=True)

    train_df = df.iloc[:N_TRAIN].copy().reset_index(drop=True)
    val_df = df.iloc[N_TRAIN:N_TRAIN + N_VAL].copy().reset_index(drop=True)
    test_df = df.iloc[N_TRAIN + N_VAL:].copy().reset_index(drop=True)

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    # Save inside splits/
    train_df.to_csv(OUTPUT_DIR / "train_manifest.csv", index=False, encoding="utf-8-sig")
    val_df.to_csv(OUTPUT_DIR / "val_manifest.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(OUTPUT_DIR / "test_manifest.csv", index=False, encoding="utf-8-sig")

    # Also save at dataset root for easier feature extraction
    train_df.to_csv(DATA_DIR / "train_manifest.csv", index=False, encoding="utf-8-sig")
    val_df.to_csv(DATA_DIR / "val_manifest.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(DATA_DIR / "test_manifest.csv", index=False, encoding="utf-8-sig")

    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    all_df.to_csv(OUTPUT_DIR / "all_splits_manifest.csv", index=False, encoding="utf-8-sig")

    print("\n===== split sizes =====")
    print("train:", len(train_df))
    print("val  :", len(val_df))
    print("test :", len(test_df))
    print("total:", len(all_df))

    print("\n===== duration by split =====")
    print(all_df.groupby("split")["duration"].describe(percentiles=[0.25, 0.50, 0.75]))

    print("\n===== VAD mean by split =====")
    print(all_df.groupby("split")[TARGET_COLS].mean())

    if "major_emotion" in all_df.columns:
        print("\n===== emotion distribution by split =====")
        print(pd.crosstab(all_df["split"], all_df["major_emotion"]))

    if "gender" in all_df.columns:
        print("\n===== gender distribution by split =====")
        print(pd.crosstab(all_df["split"], all_df["gender"]))

    summary = {
        "dataset": "MSP-Podcast / AbstractTTS-PODCAST IEMOCAP-duration-matched subset",
        "source_manifest": str(MANIFEST_PATH),
        "output_dir": str(OUTPUT_DIR),
        "random_state": RANDOM_STATE,
        "duration_range_sec": [2.0, 6.0],
        "split_sizes": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
            "total": len(all_df),
        },
        "purpose": "Duration-matched comparison with IEMOCAP using the same split sizes.",
    }

    with open(OUTPUT_DIR / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved:")
    print(DATA_DIR / "train_manifest.csv")
    print(DATA_DIR / "val_manifest.csv")
    print(DATA_DIR / "test_manifest.csv")
    print(OUTPUT_DIR / "split_summary.json")


if __name__ == "__main__":
    main()
