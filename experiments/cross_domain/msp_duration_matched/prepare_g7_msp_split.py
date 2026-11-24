#!/usr/bin/env python3
"""Prepare a new, reproducible utterance-level MSP 2–6 s split for RIVF G7."""
import argparse
from collections import Counter
import csv
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

REPO = "AbstractTTS/PODCAST"
REVISION = "e791939"
SEED = 42
COUNTS = {"train": 7082, "val": 787, "test": 2170}
LABELS = {"valence": "EmoVal", "arousal": "EmoAct", "dominance": "EmoDom"}
FIELDS = ("split", "source_utt_id", "file", "duration", *LABELS,
          *(target + "_norm" for target in LABELS), "parquet_file", "parquet_row")


def scan_shards(files):
    import pyarrow.parquet as pq
    import soundfile as sf

    candidates = []
    seen = set()
    counts = Counter()
    columns = ["file", "audio", "transcription", *LABELS.values()]
    for name, path in files:
        shard = pq.ParquetFile(path)
        missing = set(columns) - set(shard.schema_arrow.names)
        if missing:
            raise ValueError(f"{name}: missing columns {sorted(missing)}")
        row_index = 0
        print(f"Scanning {name}: {shard.metadata.num_rows:,} rows", flush=True)
        for batch in shard.iter_batches(batch_size=128, columns=columns):
            for row in batch.to_pylist():
                index = row_index
                row_index += 1
                counts["raw_rows"] += 1
                file = row["file"]
                if not isinstance(file, str) or not file.strip():
                    counts["missing_id"] += 1
                    continue
                uid = Path(file.strip()).stem
                if uid in seen:
                    raise ValueError(f"Duplicate utterance ID: {uid}")
                seen.add(uid)
                text = row["transcription"]
                if not isinstance(text, str) or not text.strip():
                    counts["missing_transcript"] += 1
                    continue
                scores = {target: row[column] for target, column in LABELS.items()}
                if any(value is None or not math.isfinite(value) or not 1 <= value <= 7
                       for value in scores.values()):
                    counts["invalid_vad"] += 1
                    continue
                audio = (row["audio"] or {}).get("bytes")
                if not audio:
                    counts["missing_audio_bytes"] += 1
                    continue
                try:
                    info = sf.info(BytesIO(audio))
                except (RuntimeError, ValueError):
                    counts["unreadable_audio"] += 1
                    continue
                if info.samplerate <= 0 or info.frames <= 0:
                    counts["unreadable_audio"] += 1
                    continue
                if not 2 * info.samplerate <= info.frames <= 6 * info.samplerate:
                    counts["outside_2_6_seconds"] += 1
                    continue
                candidates.append({
                    "source_utt_id": uid,
                    "file": file,
                    "duration": info.frames / info.samplerate,
                    **scores,
                    **{target + "_norm": (score - 4) / 3 for target, score in scores.items()},
                    "parquet_file": name,
                    "parquet_row": index,
                })
        if row_index != shard.metadata.num_rows:
            raise ValueError(f"{name}: Parquet row count changed during scan")
    return candidates, counts


def split_candidates(candidates):
    required = sum(COUNTS.values())
    if len(candidates) < required:
        raise ValueError(f"Need {required:,} eligible utterances, found {len(candidates):,}")

    def sort_key(row):
        uid = row["source_utt_id"]
        return hashlib.sha256(f"{SEED}:{uid}".encode()).digest(), uid

    ordered = sorted(candidates, key=sort_key)
    result = []
    offset = 0
    for split, count in COUNTS.items():
        result.extend({"split": split, **row} for row in ordered[offset:offset + count])
        offset += count
    return result


def write_split(output, rows, provenance):
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".g7_split_", dir=output.parent) as tmp:
        staged = Path(tmp) / "result"
        staged.mkdir()
        manifest = staged / "manifest.csv"
        with manifest.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        provenance["manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
        (staged / "split_provenance.json").write_text(
            json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
        )
        if output.exists():
            raise FileExistsError(output)
        staged.rename(output)


def main(args):
    from huggingface_hub import HfApi, hf_hub_download

    if not args.preflight and args.output.exists():
        raise FileExistsError(args.output)
    info = HfApi().dataset_info(REPO, revision=REVISION, files_metadata=True)
    if not info.sha or not info.sha.startswith(REVISION):
        raise ValueError(f"Unexpected snapshot: {info.sha}")
    shards = sorted((entry for entry in info.siblings if entry.rfilename.endswith(".parquet")),
                    key=lambda entry: entry.rfilename)
    if not shards:
        raise ValueError("This snapshot has no Parquet shards")
    print(f"Snapshot: {REPO}@{info.sha}", flush=True)
    for entry in shards:
        size = f"{entry.size / 1e9:.2f} GB" if entry.size is not None else "unknown size"
        print(f"  {entry.rfilename}  {size}", flush=True)
    if args.preflight:
        return
    files = [
        (entry.rfilename, hf_hub_download(
            REPO, entry.rfilename, repo_type="dataset", revision=info.sha,
            cache_dir=args.cache,
        ))
        for entry in shards
    ]
    candidates, counts = scan_shards(files)
    rows = split_candidates(candidates)
    provenance = {
        "dataset": REPO,
        "revision": info.sha,
        "new_split": True,
        "counts": COUNTS,
        "selection": "SHA256 of UTF-8 seed:source_utt_id ascending, then source_utt_id",
        "seed": SEED,
        "duration_range_seconds_inclusive": [2, 6],
        "grouping": "utterance only; speaker and recording overlap not measured",
        "vad_normalization": "(label-4)/3",
        "parquet_row_index_base": 0,
        "parquet_files": [name for name, _ in files],
        "eligible_utterances": len(candidates),
        "scan_counts": dict(counts),
    }
    write_split(args.output, rows, provenance)
    print(f"G7 MANIFEST COMPLETE: {args.output / 'manifest.csv'}", flush=True)
    print(f"Eligible: {len(candidates):,}; selected: {COUNTS}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=lambda path: Path(path).resolve(), required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--preflight", action="store_true")
    main(parser.parse_args())
