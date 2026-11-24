#!/usr/bin/env python3
"""Materialize the locked G7 split and verify upstream feature manifests."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
from tempfile import TemporaryDirectory

COUNTS = {"train": 7082, "val": 787, "test": 2170}
REVISION = "e791939e368b6c26595ca7f2c75aa29ebd92e776"
MANIFEST_SHA = "888f6efa5ca9b305eb159b050775484f970a3f10568df775da85f123d889f608"
LABELS = {"valence": "EmoVal", "arousal": "EmoAct", "dominance": "EmoDom"}
MODALITIES = {
    "wav2vec2_features": ("seq_feat_path", 768, 2),
    "mfcc_features": ("seq_feat_path", 39, 2),
    "text_distilbert_features": ("text_feat_path", 768, 1),
}


def load_split(directory):
    path = directory / "manifest.csv"
    report = json.loads((directory / "split_provenance.json").read_text())
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha != MANIFEST_SHA or sha != report["manifest_sha256"] or report["revision"] != REVISION:
        raise ValueError("G7 manifest hash or snapshot differs from the locked split")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if Counter(row["split"] for row in rows) != COUNTS:
        raise ValueError("Incorrect split sizes")
    if len({row["source_utt_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicated source utterance IDs")
    return rows


def materialize(rows, snapshot, output):
    import pyarrow.parquet as pq

    if output.exists():
        raise FileExistsError(output)
    if snapshot.name != REVISION:
        raise ValueError("Expected the pinned Hugging Face snapshot directory")
    jobs = defaultdict(dict)
    for row in rows:
        name = row["parquet_file"]
        index = int(row["parquet_row"])
        uid = row["source_utt_id"]
        if (not re.fullmatch(r"data/train-\d{5}-of-00068\.parquet", name)
                or not re.fullmatch(r"[A-Za-z0-9_-]+", uid)
                or index < 0 or index in jobs[name]):
            raise ValueError(f"Invalid source location for {uid}")
        jobs[name][index] = row

    results = {}
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".g7_inputs_", dir=output.parent) as tmp:
        stage = Path(tmp) / "inputs"
        (stage / "wav").mkdir(parents=True)
        for name, selected in sorted(jobs.items()):
            parquet = pq.ParquetFile(snapshot / name)
            offset = 0
            last_index = max(selected)
            for batch in parquet.iter_batches(
                batch_size=128, columns=["file", "audio", "transcription", *LABELS.values()]
            ):
                for position, source in enumerate(batch.to_pylist(), start=offset):
                    row = selected.get(position)
                    if row is None:
                        continue
                    uid = row["source_utt_id"]
                    text = source["transcription"]
                    audio = (source["audio"] or {}).get("bytes")
                    if source["file"] != row["file"] or not audio or not isinstance(text, str) or not text.strip():
                        raise ValueError(f"Source record differs for {uid}")
                    if any(not math.isclose(float(row[target]), float(source[column]), abs_tol=1e-7, rel_tol=0)
                           for target, column in LABELS.items()):
                        raise ValueError(f"VAD labels differ for {uid}")
                    relative = Path("wav") / f"{uid}.wav"
                    (stage / relative).write_bytes(audio)
                    results[uid] = dict(row, utterance_id=uid, audio_path=str(output / relative), transcript=text)
                offset += batch.num_rows
                if offset > last_index:
                    break
            print(f"Materialized {len(results):,}/{len(rows):,} after {name}", flush=True)
        if len(results) != len(rows):
            raise ValueError(f"Missing {len(rows) - len(results)} locked samples")
        fields = list(next(iter(results.values())))
        for split in COUNTS:
            with (stage / f"{split}_manifest.csv").open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(results[row["source_utt_id"]] for row in rows if row["split"] == split)
        (stage / "manifest_sha256.txt").write_text(MANIFEST_SHA + "\n")
        if output.exists():
            raise FileExistsError(output)
        stage.rename(output)
    print(f"G7 INPUTS COMPLETE: {len(results):,} samples; {output}", flush=True)


def audit(rows, features):
    import numpy as np

    for split in COUNTS:
        expected = {row["source_utt_id"]: row for row in rows if row["split"] == split}
        for name, (column, dim, rank) in MODALITIES.items():
            path = features / name / f"{split}_manifest.csv"
            with path.open(encoding="utf-8-sig", newline="") as stream:
                observed = list(csv.DictReader(stream))
            if len(observed) != len(expected) or {row["utterance_id"] for row in observed} != expected.keys():
                raise ValueError(f"{path}: sample IDs differ from locked split")
            for row in observed:
                uid = row["utterance_id"]
                for target in LABELS:
                    key = target + "_norm"
                    if not math.isclose(float(row[key]), float(expected[uid][key]), abs_tol=1e-7, rel_tol=0):
                        raise ValueError(f"{path}: label mismatch for {uid}")
                if not row[column]:
                    raise ValueError(f"{path}: missing feature for {uid}")
                array = np.load(row[column], mmap_mode="r", allow_pickle=False)
                if array.ndim != rank or array.shape[-1] != dim or array.size == 0 or not np.isfinite(array).all():
                    raise ValueError(f"{path}: invalid feature for {uid}")
            print(f"AUDIT PASS: {name}/{split} {len(observed):,} samples", flush=True)
    print("G7 FEATURES COMPLETE: all 10,039 utterances in each modality", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, required=True)
    stages = parser.add_subparsers(dest="stage", required=True)
    prepare = stages.add_parser("materialize")
    prepare.add_argument("--snapshot", type=Path, required=True)
    prepare.add_argument("--output", type=lambda value: Path(value).resolve(), required=True)
    check = stages.add_parser("audit")
    check.add_argument("--features", type=Path, required=True)
    args = parser.parse_args()
    rows = load_split(args.split)
    if args.stage == "materialize":
        materialize(rows, args.snapshot, args.output)
    else:
        audit(rows, args.features)
