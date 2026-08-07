import os
import argparse
import io
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from datasets import load_dataset, Audio

ENV_DATA_ROOT = Path(os.getenv("SER_DATA_ROOT", "data"))
ENV_OUTPUT_ROOT = Path(os.getenv("SER_OUTPUT_ROOT", "outputs"))



parser = argparse.ArgumentParser()
parser.add_argument(
    "--out_dir",
    type=str,
    default=str(ENV_DATA_ROOT / 'abstracttts_podcast_7to10s'),
)
parser.add_argument("--min_duration", type=float, default=7.0)
parser.add_argument("--max_duration", type=float, default=10.0)
parser.add_argument("--max_samples", type=int, default=11000)
parser.add_argument("--target_sr", type=int, default=16000)
args = parser.parse_args()


OUT_DIR = Path(args.out_dir)
WAV_DIR = OUT_DIR / "wav"
OUT_DIR.mkdir(parents=True, exist_ok=True)
WAV_DIR.mkdir(parents=True, exist_ok=True)


def normalize_1to7_to_minus1_1(x):
    if x is None:
        return None
    return ((float(x) - 1.0) / 6.0) * 2.0 - 1.0


def get_text(row):
    for key in ["transcription", "transcript", "text", "sentence"]:
        if key in row and row[key] is not None:
            return str(row[key])
    return ""


def get_audio_bytes_and_name(row, seen):
    audio = row["audio"]

    if not isinstance(audio, dict):
        raise ValueError(f"Unsupported audio field type: {type(audio)}")

    audio_bytes = audio.get("bytes", None)
    audio_path = audio.get("path", None)

    if audio_bytes is None:
        raise ValueError(f"Audio bytes is None. audio={audio}")

    if audio_path:
        file_name = Path(str(audio_path)).name
    else:
        file_name = str(row.get("file", f"sample_{seen:08d}.wav"))

    return audio_bytes, file_name


print("Loading AbstractTTS/PODCAST with streaming and decode=False...")

ds = load_dataset(
    "AbstractTTS/PODCAST",
    split="train",
    streaming=True,
)

# Disable Hugging Face audio decoding to avoid extra codec dependencies.
ds = ds.cast_column("audio", Audio(decode=False))

rows = []
kept = 0
seen = 0

for row in ds:
    seen += 1

    try:
        audio_bytes, file_name = get_audio_bytes_and_name(row, seen)

        array, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")

        if array.ndim > 1:
            array = np.mean(array, axis=1)

        duration = float(len(array) / sr)

        if duration < args.min_duration or duration > args.max_duration:
            continue

        utt_id = Path(file_name).stem
        out_wav = WAV_DIR / f"{utt_id}.wav"

        # Resample only when the source rate differs.
        if sr != args.target_sr:
            import librosa
            array = librosa.resample(
                array,
                orig_sr=sr,
                target_sr=args.target_sr,
            ).astype(np.float32)
            sr = args.target_sr
            duration = float(len(array) / sr)

        sf.write(out_wav, array, sr)

        valence = row.get("EmoVal", None)
        arousal = row.get("EmoAct", None)
        dominance = row.get("EmoDom", None)

        rows.append(
            {
                "utterance_id": utt_id,
                "audio_path": str(out_wav),
                "duration": duration,
                "sampling_rate": sr,
                "transcript": get_text(row),
                "valence": valence,
                "arousal": arousal,
                "dominance": dominance,
                "valence_norm": normalize_1to7_to_minus1_1(valence),
                "arousal_norm": normalize_1to7_to_minus1_1(arousal),
                "dominance_norm": normalize_1to7_to_minus1_1(dominance),
                "major_emotion": row.get("major_emotion", None),
                "gender": row.get("gender", None),
                "source_file": file_name,
            }
        )

        kept += 1

        if kept % 10 == 0:
            pd.DataFrame(rows).to_csv(
                OUT_DIR / "manifest_7to10s_partial.csv",
                index=False,
                encoding="utf-8-sig",
            )
            print(f"seen={seen}, kept={kept}, last_duration={duration:.2f}s")

        if args.max_samples > 0 and kept >= args.max_samples:
            break

    except Exception as e:
        print(f"[WARN] failed at seen={seen}: {e}")
        continue


manifest = pd.DataFrame(rows)
manifest_path = OUT_DIR / "manifest_7to10s.csv"
manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")

print("Done.")
print("Seen:", seen)
print("Kept:", kept)
print("Manifest:", manifest_path)
print("Wav dir:", WAV_DIR)