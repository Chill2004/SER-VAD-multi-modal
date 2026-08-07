import os
import argparse
import csv
import shutil
from pathlib import Path

import soundfile as sf
from datasets import load_dataset, Audio
from tqdm import tqdm

ENV_DATA_ROOT = Path(os.getenv("SER_DATA_ROOT", "data"))
ENV_OUTPUT_ROOT = Path(os.getenv("SER_OUTPUT_ROOT", "outputs"))



def normalize_1_to_7(x):
    if x is None:
        return None
    return (float(x) - 4.0) / 3.0


def safe_float(x):
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def write_audio_file(audio_obj, out_wav_path):
    """
    Works with Hugging Face Audio(decode=False).
    Usually audio_obj contains:
      - bytes: raw encoded wav bytes
      - path: original file name or local cache path
    """
    out_wav_path.parent.mkdir(parents=True, exist_ok=True)

    if not isinstance(audio_obj, dict):
        raise ValueError(f"Unexpected audio object type: {type(audio_obj)}")

    audio_bytes = audio_obj.get("bytes", None)
    audio_path = audio_obj.get("path", None)

    if audio_bytes is not None:
        with open(out_wav_path, "wb") as f:
            f.write(audio_bytes)
        return

    if audio_path is not None and Path(audio_path).exists():
        shutil.copyfile(audio_path, out_wav_path)
        return

    raise ValueError("Audio object has neither bytes nor a valid local path.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(ENV_DATA_ROOT / 'msp_podcast_full'),
    )
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--min_duration", type=float, default=None)
    parser.add_argument("--max_duration", type=float, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save_every", type=int, default=1000)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    wav_dir = out_dir / "wav"
    manifest_path = out_dir / "manifest_full.csv"

    out_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)

    seen_files = set()
    write_header = True

    if args.resume and manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                seen_files.add(row["source_file"])
        write_header = False
        print(f"Resume mode: found {len(seen_files)} existing rows in manifest.")

    print("Loading Hugging Face dataset: AbstractTTS/PODCAST")
    print("Using Audio(decode=False) to avoid torchcodec decoding dependency.")

    ds = load_dataset(
        "AbstractTTS/PODCAST",
        split="train",
        streaming=True,
    )

    ds = ds.cast_column("audio", Audio(decode=False))

    fieldnames = [
        "utterance_id",
        "audio_path",
        "duration",
        "sampling_rate",
        "transcript",
        "valence",
        "arousal",
        "dominance",
        "valence_norm",
        "arousal_norm",
        "dominance_norm",
        "major_emotion",
        "gender",
        "source_file",
    ]

    mode = "a" if args.resume and manifest_path.exists() else "w"

    kept = 0
    scanned = 0
    skipped_existing = 0
    skipped_duration = 0
    failed = 0

    with open(manifest_path, mode, encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if write_header:
            writer.writeheader()

        pbar = tqdm(ds, desc="Downloading", unit="sample")

        for row in pbar:
            scanned += 1

            source_file = str(row.get("file", "")).strip()
            if not source_file:
                source_file = f"sample_{scanned:08d}.wav"

            if source_file in seen_files:
                skipped_existing += 1
                continue

            utterance_id = Path(source_file).stem
            out_wav_path = wav_dir / source_file

            try:
                write_audio_file(row["audio"], out_wav_path)

                info = sf.info(str(out_wav_path))
                duration = float(info.duration)
                sampling_rate = int(info.samplerate)

                if args.min_duration is not None and duration < args.min_duration:
                    out_wav_path.unlink(missing_ok=True)
                    skipped_duration += 1
                    continue

                if args.max_duration is not None and duration > args.max_duration:
                    out_wav_path.unlink(missing_ok=True)
                    skipped_duration += 1
                    continue

                valence = safe_float(row.get("EmoVal"))
                arousal = safe_float(row.get("EmoAct"))
                dominance = safe_float(row.get("EmoDom"))

                writer.writerow({
                    "utterance_id": utterance_id,
                    "audio_path": str(out_wav_path),
                    "duration": duration,
                    "sampling_rate": sampling_rate,
                    "transcript": row.get("transcription", ""),
                    "valence": valence,
                    "arousal": arousal,
                    "dominance": dominance,
                    "valence_norm": normalize_1_to_7(valence),
                    "arousal_norm": normalize_1_to_7(arousal),
                    "dominance_norm": normalize_1_to_7(dominance),
                    "major_emotion": row.get("major_emotion", ""),
                    "gender": row.get("gender", ""),
                    "source_file": source_file,
                })

                kept += 1
                seen_files.add(source_file)

                if kept % args.save_every == 0:
                    f.flush()

                pbar.set_postfix({
                    "kept": kept,
                    "failed": failed,
                    "skip_dur": skipped_duration,
                })

                if args.max_samples > 0 and kept >= args.max_samples:
                    break

            except Exception as e:
                failed += 1
                print(f"\n[WARN] Failed sample {source_file}: {e}")

    print("\n===== DONE =====")
    print("Output dir       :", out_dir)
    print("Wav dir          :", wav_dir)
    print("Manifest         :", manifest_path)
    print("Scanned          :", scanned)
    print("Kept             :", kept)
    print("Skipped existing :", skipped_existing)
    print("Skipped duration :", skipped_duration)
    print("Failed           :", failed)


if __name__ == "__main__":
    main()