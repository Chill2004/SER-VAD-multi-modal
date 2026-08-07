import os
from pathlib import Path

import pandas as pd

ENV_DATA_ROOT = Path(os.getenv("SER_DATA_ROOT", "data"))
ENV_OUTPUT_ROOT = Path(os.getenv("SER_OUTPUT_ROOT", "outputs"))



# Configuration
AUG_ROOT = Path(str(ENV_OUTPUT_ROOT / 'aug_runs_audio'))
TRANSCRIPT_CSV = Path(str(ENV_OUTPUT_ROOT / 'transcripts/iemocap_transcripts.csv'))

FEATURE_DIRS = [
    "wav2vec2_features",
    "mfcc_features",
    "mfcc39_delta_delta2_features",
]

SPLITS = ["train", "val", "test"]


def strip_aug_suffix(x: str) -> str:
    x = str(x)

    suffixes = [
        "__orig",
        "__noise",
        "__reverb",
        "__silence",
        "__silence_noise",
        "__noise_silence",
        "__silence_reverb",
        "__reverb_silence",
        "__reverb_noise",
        "__noise_reverb",
        "__silence_reverb_noise",
    ]

    for suffix in suffixes:
        if x.endswith(suffix):
            return x[: -len(suffix)]

    # Fallback:
    # Ses04F_impro01_M020__orig -> Ses04F_impro01_M020
    if "__" in x:
        return x.split("__")[0]

    return x


def make_text_key(df: pd.DataFrame) -> pd.Series:
    if "source_utt_id" in df.columns:
        return df["source_utt_id"].astype(str)

    if "utterance_id" in df.columns:
        return df["utterance_id"].astype(str).map(strip_aug_suffix)

    raise ValueError("Manifest must contain either source_utt_id or utterance_id.")


def attach_transcript_to_manifest(manifest_csv: Path, transcript_df: pd.DataFrame):
    df = pd.read_csv(manifest_csv).copy()

    df["_text_key"] = make_text_key(df)

    merged = df.merge(
        transcript_df[[
            "utterance_id",
            "dialogue_id",
            "speaker",
            "start_time",
            "end_time",
            "duration",
            "transcript",
        ]].rename(columns={
            "utterance_id": "_text_key",
            "dialogue_id": "text_dialogue_id",
            "speaker": "text_speaker",
            "start_time": "text_start_time",
            "end_time": "text_end_time",
            "duration": "text_duration",
        }),
        on="_text_key",
        how="left",
    )

    missing = merged["transcript"].isna().sum()
    total = len(merged)

    out_csv = manifest_csv.with_name(manifest_csv.stem + "_with_text.csv")
    merged.to_csv(out_csv, index=False, encoding="utf-8-sig")

    if missing > 0:
        missing_csv = manifest_csv.with_name(manifest_csv.stem + "_missing_text.csv")
        merged[merged["transcript"].isna()].to_csv(
            missing_csv,
            index=False,
            encoding="utf-8-sig",
        )
        print(f"[WARN] Missing transcript: {missing}/{total} -> {missing_csv}")

    print(f"[OK] {manifest_csv} -> {out_csv} | missing={missing}/{total}")


def main():
    transcript_df = pd.read_csv(TRANSCRIPT_CSV).copy()

    if "utterance_id" not in transcript_df.columns or "transcript" not in transcript_df.columns:
        raise ValueError("Transcript CSV must contain utterance_id and transcript columns.")

    transcript_df = transcript_df.drop_duplicates(subset=["utterance_id"])

    settings = [
        d.name for d in sorted(AUG_ROOT.iterdir())
        if d.is_dir()
    ]

    print("Settings:", settings)

    for setting in settings:
        setting_dir = AUG_ROOT / setting

        for feature_dir in FEATURE_DIRS:
            feature_path = setting_dir / feature_dir

            if not feature_path.exists():
                continue

            for split in SPLITS:
                manifest_csv = feature_path / f"{split}_manifest.csv"

                if not manifest_csv.exists():
                    continue

                attach_transcript_to_manifest(manifest_csv, transcript_df)


if __name__ == "__main__":
    main()