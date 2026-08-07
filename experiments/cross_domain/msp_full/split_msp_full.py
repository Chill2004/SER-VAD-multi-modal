import os
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ENV_DATA_ROOT = Path(os.getenv("SER_DATA_ROOT", "data"))
ENV_OUTPUT_ROOT = Path(os.getenv("SER_OUTPUT_ROOT", "outputs"))



DATA_DIR = Path(str(ENV_DATA_ROOT / 'msp_podcast_full'))
MANIFEST_PATH = DATA_DIR / "manifest_full.csv"

RANDOM_STATE = 42
TARGET_COLS = ["valence_norm", "arousal_norm", "dominance_norm"]

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10


def main():
    df = pd.read_csv(MANIFEST_PATH)

    print("Loaded:", MANIFEST_PATH)
    print("Rows before cleaning:", len(df))

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
    df = df[df["audio_path"].astype(str).map(lambda p: Path(p).exists())].copy()

    df["duration"] = pd.to_numeric(df["duration"], errors="coerce")
    for col in TARGET_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["duration"] + TARGET_COLS).copy()

    print("Rows after cleaning:", len(df))

    stratify_col = None
    if "major_emotion" in df.columns:
        counts = df["major_emotion"].value_counts()
        if counts.min() >= 3:
            stratify_col = df["major_emotion"]
            print("Using stratification by major_emotion.")

    train_df, temp_df = train_test_split(
        df,
        train_size=TRAIN_RATIO,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=stratify_col,
    )

    val_fraction_of_temp = VAL_RATIO / (VAL_RATIO + TEST_RATIO)

    temp_stratify = None
    if "major_emotion" in temp_df.columns:
        counts = temp_df["major_emotion"].value_counts()
        if counts.min() >= 2:
            temp_stratify = temp_df["major_emotion"]

    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_fraction_of_temp,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=temp_stratify,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    train_df.to_csv(DATA_DIR / "train_manifest.csv", index=False, encoding="utf-8-sig")
    val_df.to_csv(DATA_DIR / "val_manifest.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(DATA_DIR / "test_manifest.csv", index=False, encoding="utf-8-sig")

    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    all_df.to_csv(DATA_DIR / "all_splits_manifest.csv", index=False, encoding="utf-8-sig")

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
        "dataset": "MSP-Podcast full dataset",
        "source_manifest": str(MANIFEST_PATH),
        "random_state": RANDOM_STATE,
        "split_ratio": {
            "train": TRAIN_RATIO,
            "val": VAL_RATIO,
            "test": TEST_RATIO,
        },
        "split_sizes": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
            "total": len(all_df),
        },
        "purpose": "Full training on real-world podcast-domain SER data.",
    }

    with open(DATA_DIR / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved:")
    print(DATA_DIR / "train_manifest.csv")
    print(DATA_DIR / "val_manifest.csv")
    print(DATA_DIR / "test_manifest.csv")
    print(DATA_DIR / "split_summary.json")


if __name__ == "__main__":
    main()
