import os
import re
from pathlib import Path

import pandas as pd

ENV_DATA_ROOT = Path(os.getenv("SER_DATA_ROOT", "data"))
ENV_OUTPUT_ROOT = Path(os.getenv("SER_OUTPUT_ROOT", "outputs"))



# Configuration
DATA_ROOT = Path(str(ENV_DATA_ROOT))

OUTPUT_DIR = Path(str(ENV_OUTPUT_ROOT / 'transcripts'))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "iemocap_transcripts.csv"

TRANSCRIPT_GLOB = "Session*/dialog/transcriptions/*.txt"

LINE_PATTERN = re.compile(
    r"^(?P<utterance_id>\S+)\s+"
    r"\[(?P<start>[0-9.]+)-(?P<end>[0-9.]+)\]:\s*"
    r"(?P<text>.*)$"
)


def parse_speaker_from_utt_id(utterance_id: str) -> str:
    # Example: Ses01F_impro01_F000 -> F
    # Example: Ses01F_impro01_M000 -> M
    last_part = utterance_id.split("_")[-1]
    if len(last_part) > 0 and last_part[0] in ["F", "M"]:
        return last_part[0]
    return ""


def parse_dialogue_id_from_utt_id(utterance_id: str) -> str:
    # Example:
    # Ses01F_impro01_F000 -> Ses01F_impro01
    # Ses01F_script01_2_M000 -> Ses01F_script01_2
    parts = utterance_id.split("_")
    if len(parts) <= 1:
        return utterance_id
    return "_".join(parts[:-1])


def parse_transcript_file(txt_path: Path):
    rows = []

    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            match = LINE_PATTERN.match(line)
            if not match:
                continue

            utterance_id = match.group("utterance_id")
            start_time = float(match.group("start"))
            end_time = float(match.group("end"))
            text = match.group("text").strip()

            dialogue_id = parse_dialogue_id_from_utt_id(utterance_id)
            speaker = parse_speaker_from_utt_id(utterance_id)

            rows.append({
                "utterance_id": utterance_id,
                "dialogue_id": dialogue_id,
                "speaker": speaker,
                "start_time": start_time,
                "end_time": end_time,
                "duration": end_time - start_time,
                "transcript": text,
                "transcript_file": str(txt_path),
                "line_no": line_no,
            })

    return rows


def main():
    txt_files = sorted(DATA_ROOT.glob(TRANSCRIPT_GLOB))

    print("DATA_ROOT:", DATA_ROOT)
    print("Found transcript files:", len(txt_files))

    if len(txt_files) == 0:
        raise FileNotFoundError(
            f"No transcript files found with pattern: {DATA_ROOT / TRANSCRIPT_GLOB}"
        )

    all_rows = []

    for txt_path in txt_files:
        rows = parse_transcript_file(txt_path)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)

    if len(df) == 0:
        raise ValueError("No transcript rows parsed. Check transcript format.")

    df = df.drop_duplicates(subset=["utterance_id"]).reset_index(drop=True)

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("Parsed rows:", len(df))
    print("Saved to:", OUTPUT_CSV)
    print(df.head())


if __name__ == "__main__":
    main()