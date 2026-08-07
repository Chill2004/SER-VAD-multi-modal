import os
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ENV_DATA_ROOT = Path(os.getenv("SER_DATA_ROOT", "data"))
ENV_OUTPUT_ROOT = Path(os.getenv("SER_OUTPUT_ROOT", "outputs"))



# CLI
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_dir",
    type=str,
    default=str(ENV_DATA_ROOT / 'abstracttts_podcast_7to10s'),
)
parser.add_argument(
    "--manifest",
    type=str,
    default="manifest_7to10s.csv",
)
parser.add_argument(
    "--out_dir",
    type=str,
    default=str(ENV_OUTPUT_ROOT / 'abstracttts_podcast_7to10s_analysis'),
)
args = parser.parse_args()


# Configuration
DATASET_DIR = Path(args.dataset_dir)
MANIFEST_PATH = DATASET_DIR / args.manifest
OUTPUT_DIR = Path(args.out_dir)
FIG_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLS = ["valence_norm", "arousal_norm", "dominance_norm"]
RAW_TARGET_COLS = ["valence", "arousal", "dominance"]


# Helpers
def save_histogram(df, column, title, xlabel, filename, bins=40):
    plt.figure(figsize=(8, 5))
    plt.hist(df[column].dropna(), bins=bins)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, dpi=200)
    plt.close()


def save_bar_counts(df, column, title, xlabel, filename, top_k=None):
    values = df[column].fillna("unknown").astype(str).value_counts()
    if top_k is not None:
        values = values.head(top_k)

    plt.figure(figsize=(10, 5))
    values.plot(kind="bar")
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, dpi=200)
    plt.close()


def save_scatter(df, x_col, y_col, title, filename):
    plt.figure(figsize=(6, 6))
    plt.scatter(df[x_col], df[y_col], s=8, alpha=0.35)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, dpi=200)
    plt.close()


def add_text_length_columns(df):
    if "transcript" not in df.columns:
        df["transcript"] = ""

    text = df["transcript"].fillna("").astype(str)
    df["text_char_len"] = text.str.len()
    df["text_word_len"] = text.apply(lambda x: len(x.split()))
    return df


def summarize_split(split_name, split_df):
    summary = {
        "split": split_name,
        "rows": len(split_df),
        "duration_mean": split_df["duration"].mean() if "duration" in split_df.columns else None,
        "duration_std": split_df["duration"].std() if "duration" in split_df.columns else None,
        "duration_min": split_df["duration"].min() if "duration" in split_df.columns else None,
        "duration_max": split_df["duration"].max() if "duration" in split_df.columns else None,
        "text_word_len_mean": split_df["text_word_len"].mean() if "text_word_len" in split_df.columns else None,
        "text_word_len_std": split_df["text_word_len"].std() if "text_word_len" in split_df.columns else None,
    }

    for col in RAW_TARGET_COLS + TARGET_COLS:
        if col in split_df.columns:
            summary[f"{col}_mean"] = split_df[col].mean()
            summary[f"{col}_std"] = split_df[col].std()
            summary[f"{col}_min"] = split_df[col].min()
            summary[f"{col}_max"] = split_df[col].max()

    return summary


# Main
def main():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Missing manifest: {MANIFEST_PATH}")

    df = pd.read_csv(MANIFEST_PATH).copy()
    df = add_text_length_columns(df)

    print("Manifest:", MANIFEST_PATH)
    print("Rows:", len(df))
    print("Columns:", list(df.columns))

    # Basic CSV summaries
    summary_rows = []
    summary_rows.append(summarize_split("all", df))

    for split in ["train", "val", "test"]:
        split_path = DATASET_DIR / f"{split}_manifest.csv"
        if split_path.exists():
            split_df = pd.read_csv(split_path).copy()
            split_df = add_text_length_columns(split_df)
            summary_rows.append(summarize_split(split, split_df))

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_DIR / "dataset_summary.csv", index=False, encoding="utf-8-sig")

    target_corr_cols = [c for c in TARGET_COLS if c in df.columns]
    if target_corr_cols:
        corr_df = df[target_corr_cols].corr()
        corr_df.to_csv(OUTPUT_DIR / "target_correlation.csv", encoding="utf-8-sig")

    if "major_emotion" in df.columns:
        emotion_counts = df["major_emotion"].fillna("unknown").astype(str).value_counts().reset_index()
        emotion_counts.columns = ["major_emotion", "count"]
        emotion_counts.to_csv(OUTPUT_DIR / "major_emotion_counts.csv", index=False, encoding="utf-8-sig")

    if "gender" in df.columns:
        gender_counts = df["gender"].fillna("unknown").astype(str).value_counts().reset_index()
        gender_counts.columns = ["gender", "count"]
        gender_counts.to_csv(OUTPUT_DIR / "gender_counts.csv", index=False, encoding="utf-8-sig")

    # Figures
    if "duration" in df.columns:
        save_histogram(
            df,
            "duration",
            "Audio Duration Distribution",
            "Duration (seconds)",
            "duration_distribution.png",
            bins=40,
        )

    for col in RAW_TARGET_COLS:
        if col in df.columns:
            save_histogram(
                df,
                col,
                f"{col.capitalize()} Distribution (Raw Scale)",
                col,
                f"{col}_raw_distribution.png",
                bins=35,
            )

    for col in TARGET_COLS:
        if col in df.columns:
            save_histogram(
                df,
                col,
                f"{col} Distribution [-1, 1]",
                col,
                f"{col}_distribution.png",
                bins=35,
            )

    save_histogram(
        df,
        "text_word_len",
        "Transcript Word Count Distribution",
        "Number of words",
        "text_word_count_distribution.png",
        bins=50,
    )

    save_histogram(
        df,
        "text_char_len",
        "Transcript Character Length Distribution",
        "Number of characters",
        "text_character_length_distribution.png",
        bins=50,
    )

    if "major_emotion" in df.columns:
        save_bar_counts(
            df,
            "major_emotion",
            "Major Emotion Distribution",
            "Major emotion",
            "major_emotion_distribution.png",
            top_k=None,
        )

    if "gender" in df.columns:
        save_bar_counts(
            df,
            "gender",
            "Gender Distribution",
            "Gender",
            "gender_distribution.png",
            top_k=None,
        )

    if all(c in df.columns for c in ["valence_norm", "arousal_norm"]):
        save_scatter(
            df,
            "valence_norm",
            "arousal_norm",
            "Valence vs Arousal",
            "valence_vs_arousal.png",
        )

    if all(c in df.columns for c in ["valence_norm", "dominance_norm"]):
        save_scatter(
            df,
            "valence_norm",
            "dominance_norm",
            "Valence vs Dominance",
            "valence_vs_dominance.png",
        )

    if all(c in df.columns for c in ["arousal_norm", "dominance_norm"]):
        save_scatter(
            df,
            "arousal_norm",
            "dominance_norm",
            "Arousal vs Dominance",
            "arousal_vs_dominance.png",
        )

    print("\nSaved analysis to:", OUTPUT_DIR)
    print("Saved figures to :", FIG_DIR)
    print("Summary CSV      :", OUTPUT_DIR / "dataset_summary.csv")


if __name__ == "__main__":
    main()
