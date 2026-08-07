from __future__ import annotations
import os

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ENV_DATA_ROOT = Path(os.getenv("SER_DATA_ROOT", "data"))
ENV_OUTPUT_ROOT = Path(os.getenv("SER_OUTPUT_ROOT", "outputs"))



# CLI
parser = argparse.ArgumentParser()

parser.add_argument(
    "--audio_setting",
    "--setting",
    dest="audio_setting",
    type=str,
    required=True,
    help="Audio setting folder name, e.g. silence",
)

parser.add_argument(
    "--text_setting",
    type=str,
    default="baseline",
    help="Text setting folder name, usually baseline",
)

parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--val_ratio", type=float, default=0.10)

parser.add_argument(
    "--aug_root",
    type=str,
    default=str(ENV_OUTPUT_ROOT / 'aug_runs_audio'),
)

parser.add_argument(
    "--out_root",
    type=str,
    default=str(ENV_OUTPUT_ROOT / 'iemocap_loso_manifests_full'),
)

parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Overwrite existing LOSO output folder.",
)

parser.add_argument(
    "--skip_path_check",
    action="store_true",
    help="Skip checking whether every referenced feature file exists.",
)

parser.add_argument(
    "--allow_count_mismatch",
    action="store_true",
    help="Allow source counts that differ from the expected 10,039-utterance IEMOCAP setup.",
)

args = parser.parse_args()


# Configuration
AUG_ROOT = Path(args.aug_root)

AUDIO_SETTING = args.audio_setting
TEXT_SETTING = args.text_setting

WAV2VEC2_DIR = AUG_ROOT / AUDIO_SETTING / "wav2vec2_features"
MFCC_DIR = AUG_ROOT / AUDIO_SETTING / "mfcc_features"
TEXT_DIR = AUG_ROOT / TEXT_SETTING / "text_distilbert_features"

if TEXT_SETTING == AUDIO_SETTING:
    EXPERIMENT_NAME = AUDIO_SETTING
else:
    EXPERIMENT_NAME = f"{AUDIO_SETTING}_audio_{TEXT_SETTING}_text"

OUT_ROOT = Path(args.out_root) / EXPERIMENT_NAME

SPLITS = ["train", "val", "test"]
SESSIONS = ["Ses01", "Ses02", "Ses03", "Ses04", "Ses05"]

AUG_SUFFIXES = ["__orig", "__silence", "__noise", "__reverb"]

TARGET_COLS = ["valence_norm", "arousal_norm", "dominance_norm"]

# Expected original-utterance counts from the paper protocol.
EXPECTED_SESSION_SOURCE_COUNTS = {
    "Ses01": 1819,
    "Ses02": 1811,
    "Ses03": 2136,
    "Ses04": 2103,
    "Ses05": 2170,
}


# Helpers
def strip_aug_suffix(x: str) -> str:
    x = str(x)
    for suffix in AUG_SUFFIXES:
        if x.endswith(suffix):
            return x[: -len(suffix)]
    return x


def normalize_session_value(x: str) -> str:
    x = str(x).strip()

    if x in SESSIONS:
        return x

    lowered = x.lower().replace("session", "").replace("ses", "").strip()
    lowered = lowered.replace("_", "").replace("-", "")

    if lowered.isdigit():
        return f"Ses{int(lowered):02d}"

    for s in SESSIONS:
        if s.lower() in x.lower():
            return s

    raise ValueError(f"Cannot normalize session value: {x}")


def infer_session_from_utterance_id(x: str) -> str:
    x = str(x)
    for s in SESSIONS:
        if x.startswith(s):
            return s
    raise ValueError(f"Cannot infer session from utterance_id/source_utt_id: {x}")


def make_source_key(df: pd.DataFrame) -> pd.Series:
    if "source_utt_id" in df.columns:
        return df["source_utt_id"].astype(str).map(strip_aug_suffix)

    if "utterance_id" in df.columns:
        return df["utterance_id"].astype(str).map(strip_aug_suffix)

    raise ValueError("Manifest must contain source_utt_id or utterance_id.")


def make_aug_tag(df: pd.DataFrame) -> pd.Series:
    if "aug_tag" in df.columns:
        return df["aug_tag"].astype(str)

    if "utterance_id" in df.columns:
        u = df["utterance_id"].astype(str)
        out = pd.Series(["orig"] * len(df), index=df.index)
        out[u.str.endswith("__silence")] = "silence"
        out[u.str.endswith("__noise")] = "noise"
        out[u.str.endswith("__reverb")] = "reverb"
        return out

    raise ValueError("Manifest must contain aug_tag or utterance_id.")


def ensure_session_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "session" in df.columns:
        df["session"] = df["session"].astype(str).map(normalize_session_value)
        return df

    if "source_utt_id" in df.columns:
        df["session"] = df["source_utt_id"].astype(str).map(infer_session_from_utterance_id)
        return df

    if "utterance_id" in df.columns:
        df["session"] = (
            df["utterance_id"]
            .astype(str)
            .map(strip_aug_suffix)
            .map(infer_session_from_utterance_id)
        )
        return df

    raise ValueError("Cannot create session column.")


def load_full_manifest(feature_dir: Path, feature_name: str) -> pd.DataFrame:
    dfs = []

    for split in SPLITS:
        p = feature_dir / f"{split}_manifest.csv"

        if not p.exists():
            raise FileNotFoundError(f"Missing {feature_name} manifest: {p}")

        df = pd.read_csv(p).copy()
        df["_old_split"] = split
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    df = ensure_session_column(df)
    df["_source_key"] = make_source_key(df)
    df["_aug_tag"] = make_aug_tag(df)

    duplicate_mask = df.duplicated(
        subset=["_source_key", "_aug_tag"],
        keep=False,
    )
    if duplicate_mask.any():
        examples = (
            df.loc[duplicate_mask, ["_source_key", "_aug_tag", "_old_split"]]
            .head(20)
            .to_dict("records")
        )
        raise ValueError(
            f"{feature_name}: duplicated source+augmentation rows detected. "
            f"Examples: {examples}"
        )

    return df.reset_index(drop=True)


def load_text_manifest(feature_dir: Path) -> pd.DataFrame:
    dfs = []

    for split in SPLITS:
        p = feature_dir / f"{split}_manifest.csv"

        if not p.exists():
            raise FileNotFoundError(f"Missing text manifest: {p}")

        df = pd.read_csv(p).copy()
        df["_old_split"] = split
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    df = ensure_session_column(df)
    df["_source_key"] = make_source_key(df)

    duplicate_mask = df.duplicated(subset=["_source_key"], keep=False)
    if duplicate_mask.any():
        examples = (
            df.loc[duplicate_mask, ["_source_key", "_old_split"]]
            .head(20)
            .to_dict("records")
        )
        raise ValueError(
            "Text manifests contain duplicated source rows. "
            f"Examples: {examples}"
        )

    return df.reset_index(drop=True)


def assert_full_orig_silence(df: pd.DataFrame, name: str) -> None:
    pivot = (
        df.groupby(["session", "_aug_tag"])
        .size()
        .unstack(fill_value=0)
        .reindex(SESSIONS, fill_value=0)
    )

    print(f"\n================ {name} full audit ================")
    print(pivot)

    for tag in ["orig", "silence"]:
        if tag not in pivot.columns:
            raise ValueError(f"{name}: missing aug_tag={tag} entirely.")

        bad_sessions = [s for s in SESSIONS if int(pivot.loc[s, tag]) == 0]

        if bad_sessions:
            raise ValueError(f"{name}: aug_tag={tag} missing in sessions: {bad_sessions}")

    counts = (
        df[df["_aug_tag"].isin(["orig", "silence"])]
        .groupby("_source_key")["_aug_tag"]
        .nunique()
    )

    missing_pair = counts[counts != 2]

    if len(missing_pair) > 0:
        print(missing_pair.head(20))
        raise ValueError(f"{name}: {len(missing_pair)} source keys do not have both orig and silence.")

    print(f"[OK] {name}: every source key has orig + silence.")


def assert_text_complete(text_df: pd.DataFrame, source_keys: set[str]) -> None:
    text_keys = set(text_df["_source_key"].astype(str))

    missing = sorted(source_keys - text_keys)
    extra = sorted(text_keys - source_keys)

    print("\n================ text full audit ================")
    print("text rows:", len(text_df))
    print("unique text source keys:", len(text_keys))

    if missing:
        raise ValueError(f"Text missing {len(missing)} source keys. Example: {missing[:10]}")

    if extra:
        print(f"[WARN] text has {len(extra)} extra source keys not in audio. Example: {extra[:10]}")

    print("[OK] text covers all audio source keys.")


def split_dev_source_keys(
    dev_keys: list[str],
    val_ratio: float,
    seed: int,
) -> tuple[set[str], set[str]]:
    dev_keys = sorted(list(dev_keys))

    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")

    # Use an integer validation size to reproduce the paper split.
    n_val = int(round(len(dev_keys) * val_ratio))
    n_val = max(1, min(len(dev_keys) - 1, n_val))

    train_keys, val_keys = train_test_split(
        dev_keys,
        test_size=n_val,
        random_state=seed,
        shuffle=True,
    )

    return set(train_keys), set(val_keys)


def make_fold_audio_split(
    audio_df: pd.DataFrame,
    train_keys: set[str],
    val_keys: set[str],
    test_keys: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Train uses original + silence; validation and test use original only.
    train_df = audio_df[
        audio_df["_source_key"].isin(train_keys)
        & audio_df["_aug_tag"].isin(["orig", "silence"])
    ].copy()

    val_df = audio_df[
        audio_df["_source_key"].isin(val_keys)
        & (audio_df["_aug_tag"] == "orig")
    ].copy()

    test_df = audio_df[
        audio_df["_source_key"].isin(test_keys)
        & (audio_df["_aug_tag"] == "orig")
    ].copy()

    return train_df, val_df, test_df


def make_fold_text_split(
    text_df: pd.DataFrame,
    train_keys: set[str],
    val_keys: set[str],
    test_keys: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = text_df[text_df["_source_key"].isin(train_keys)].copy()
    val_df = text_df[text_df["_source_key"].isin(val_keys)].copy()
    test_df = text_df[text_df["_source_key"].isin(test_keys)].copy()

    return train_df, val_df, test_df


def clean_for_save(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["_source_key", "_aug_tag", "_old_split"], errors="ignore").reset_index(drop=True)


def write_manifest_triplet(
    out_dir: Path,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_for_save(train_df).to_csv(out_dir / "train_manifest.csv", index=False, encoding="utf-8-sig")
    clean_for_save(val_df).to_csv(out_dir / "val_manifest.csv", index=False, encoding="utf-8-sig")
    clean_for_save(test_df).to_csv(out_dir / "test_manifest.csv", index=False, encoding="utf-8-sig")


def assert_orig_only(df: pd.DataFrame, fold_name: str, df_name: str) -> None:
    tags = sorted(df["_aug_tag"].astype(str).unique().tolist())
    if tags != ["orig"]:
        raise ValueError(f"{fold_name}: {df_name} must be orig only, but got tags: {tags}")


def assert_required_columns(
    df: pd.DataFrame,
    required_cols: list[str],
    name: str,
) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing required columns: {missing}")


def assert_source_session_consistency(df: pd.DataFrame, name: str) -> None:
    inferred = df["_source_key"].astype(str).map(infer_session_from_utterance_id)
    actual = df["session"].astype(str)
    mismatch = actual != inferred

    if mismatch.any():
        examples = (
            pd.DataFrame({
                "source": df.loc[mismatch, "_source_key"],
                "manifest_session": actual[mismatch],
                "inferred_session": inferred[mismatch],
            })
            .head(20)
            .to_dict("records")
        )
        raise ValueError(f"{name}: source/session mismatch. Examples: {examples}")


def assert_feature_paths(
    df: pd.DataFrame,
    feature_col: str,
    name: str,
) -> None:
    missing = []

    for value in df[feature_col].dropna().astype(str):
        if not Path(value).exists():
            missing.append(value)
            if len(missing) >= 20:
                break

    if missing:
        raise FileNotFoundError(
            f"{name}: referenced feature files are missing. Examples: {missing}"
        )


def assert_targets_consistent_within_source(
    df: pd.DataFrame,
    name: str,
) -> None:
    for col in TARGET_COLS:
        grouped = df.groupby("_source_key")[col].agg(["min", "max"])
        bad = grouped[(grouped["max"] - grouped["min"]).abs() > 1e-6]

        if len(bad) > 0:
            raise ValueError(
                f"{name}: target '{col}' differs across variants for "
                f"{len(bad)} source utterances. Examples: {bad.head(10).index.tolist()}"
            )


def source_target_frame(df: pd.DataFrame) -> pd.DataFrame:
    if "_aug_tag" in df.columns:
        base = df[df["_aug_tag"] == "orig"].copy()
    else:
        base = df.copy()

    return (
        base[["_source_key"] + TARGET_COLS]
        .drop_duplicates(subset=["_source_key"], keep=False)
        .set_index("_source_key")
        .sort_index()
    )


def assert_cross_modal_targets(
    wav_df: pd.DataFrame,
    mfcc_df: pd.DataFrame,
    text_df: pd.DataFrame,
) -> None:
    wav_targets = source_target_frame(wav_df)
    mfcc_targets = source_target_frame(mfcc_df)
    text_targets = source_target_frame(text_df)

    common = wav_targets.index.intersection(mfcc_targets.index).intersection(text_targets.index)
    if len(common) != len(wav_targets):
        raise ValueError(
            "Cannot compare targets for all sources across Wav2Vec2, MFCC, and text."
        )

    for modality_name, other in [
        ("MFCC", mfcc_targets),
        ("Text", text_targets),
    ]:
        diff = (
            wav_targets.loc[common, TARGET_COLS].to_numpy()
            - other.loc[common, TARGET_COLS].to_numpy()
        )
        bad_rows = (abs(diff) > 1e-6).any(axis=1)

        if bad_rows.any():
            bad_keys = common[bad_rows][:10].tolist()
            raise ValueError(
                f"Target mismatch between Wav2Vec2 and {modality_name}. "
                f"Examples: {bad_keys}"
            )


def assert_audio_pair_alignment(
    wav_df: pd.DataFrame,
    mfcc_df: pd.DataFrame,
) -> None:
    def pair_set(df: pd.DataFrame) -> set[tuple[str, str]]:
        selected = df[df["_aug_tag"].isin(["orig", "silence"])]
        return set(
            zip(
                selected["_source_key"].astype(str),
                selected["_aug_tag"].astype(str),
            )
        )

    wav_pairs = pair_set(wav_df)
    mfcc_pairs = pair_set(mfcc_df)

    if wav_pairs != mfcc_pairs:
        raise ValueError(
            "Wav2Vec2 and MFCC orig/silence pairs differ: "
            f"wav_only={len(wav_pairs - mfcc_pairs)}, "
            f"mfcc_only={len(mfcc_pairs - wav_pairs)}"
        )


def assert_expected_original_counts(wav_df: pd.DataFrame) -> None:
    counts = (
        wav_df[wav_df["_aug_tag"] == "orig"]
        .groupby("session")["_source_key"]
        .nunique()
        .reindex(SESSIONS, fill_value=0)
        .astype(int)
        .to_dict()
    )

    print("\n================ expected source-count audit ================")
    print(counts)

    if counts != EXPECTED_SESSION_SOURCE_COUNTS:
        raise ValueError(
            "Original utterance counts differ from the paper protocol. "
            f"got={counts}, expected={EXPECTED_SESSION_SOURCE_COUNTS}"
        )


def write_source_key_lists(
    fold_dir: Path,
    train_keys: set[str],
    val_keys: set[str],
    test_keys: set[str],
) -> None:
    for split_name, keys in [
        ("train", train_keys),
        ("val", val_keys),
        ("test", test_keys),
    ]:
        pd.DataFrame({"source_utt_id": sorted(keys)}).to_csv(
            fold_dir / f"{split_name}_source_keys.csv",
            index=False,
            encoding="utf-8-sig",
        )


# Main
def main() -> None:
    print("AUDIO_SETTING:", AUDIO_SETTING)
    print("TEXT_SETTING :", TEXT_SETTING)
    print("WAV2VEC2_DIR :", WAV2VEC2_DIR)
    print("MFCC_DIR     :", MFCC_DIR)
    print("TEXT_DIR     :", TEXT_DIR)
    print("OUT_ROOT     :", OUT_ROOT)

    if OUT_ROOT.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output folder already exists: {OUT_ROOT}\n"
            f"Use --overwrite to replace it."
        )

    build_root = OUT_ROOT.parent / f".{OUT_ROOT.name}.building"
    if build_root.exists():
        shutil.rmtree(build_root)

    wav_df = load_full_manifest(WAV2VEC2_DIR, "wav2vec2")
    mfcc_df = load_full_manifest(MFCC_DIR, "mfcc")
    text_df = load_text_manifest(TEXT_DIR)

    assert_required_columns(
        wav_df,
        ["_source_key", "_aug_tag", "session", "seq_feat_path"] + TARGET_COLS,
        "Wav2Vec2",
    )
    assert_required_columns(
        mfcc_df,
        ["_source_key", "_aug_tag", "session", "seq_feat_path"] + TARGET_COLS,
        "MFCC",
    )
    assert_required_columns(
        text_df,
        ["_source_key", "session", "text_feat_path"] + TARGET_COLS,
        "Text",
    )

    assert_source_session_consistency(wav_df, "Wav2Vec2")
    assert_source_session_consistency(mfcc_df, "MFCC")
    assert_source_session_consistency(text_df, "Text")

    assert_targets_consistent_within_source(wav_df, "Wav2Vec2")
    assert_targets_consistent_within_source(mfcc_df, "MFCC")
    assert_targets_consistent_within_source(text_df, "Text")

    if not args.skip_path_check:
        assert_feature_paths(wav_df, "seq_feat_path", "Wav2Vec2")
        assert_feature_paths(mfcc_df, "seq_feat_path", "MFCC")
        assert_feature_paths(text_df, "text_feat_path", "Text")

    assert_full_orig_silence(wav_df, "wav2vec2")
    assert_full_orig_silence(mfcc_df, "mfcc")

    wav_keys = set(wav_df["_source_key"].astype(str))
    mfcc_keys = set(mfcc_df["_source_key"].astype(str))

    if wav_keys != mfcc_keys:
        raise ValueError(
            "Wav2Vec2 and MFCC source keys differ: "
            f"wav_only={len(wav_keys - mfcc_keys)}, "
            f"mfcc_only={len(mfcc_keys - wav_keys)}"
        )

    source_keys = wav_keys
    assert_text_complete(text_df, source_keys)
    assert_audio_pair_alignment(wav_df, mfcc_df)
    assert_cross_modal_targets(wav_df, mfcc_df, text_df)

    if not args.allow_count_mismatch:
        assert_expected_original_counts(wav_df)

    build_root.mkdir(parents=True, exist_ok=False)
    summary_rows = []

    for test_session in SESSIONS:
        fold_name = f"fold_{test_session}"
        fold_dir = build_root / fold_name

        print(f"\n================ Creating {fold_name} ================")

        test_keys = set(
            wav_df.loc[wav_df["session"] == test_session, "_source_key"].astype(str)
        )

        dev_keys = sorted(source_keys - test_keys)

        train_keys, val_keys = split_dev_source_keys(
            dev_keys=dev_keys,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )

        # Check source-level overlap.
        assert len(train_keys & val_keys) == 0
        assert len(train_keys & test_keys) == 0
        assert len(val_keys & test_keys) == 0

        wav_train, wav_val, wav_test = make_fold_audio_split(wav_df, train_keys, val_keys, test_keys)
        mfcc_train, mfcc_val, mfcc_test = make_fold_audio_split(mfcc_df, train_keys, val_keys, test_keys)

        text_train, text_val, text_test = make_fold_text_split(text_df, train_keys, val_keys, test_keys)

        # Expected rows under the original + silence training protocol.
        expected_train_audio = 2 * len(train_keys)
        expected_val_audio = len(val_keys)
        expected_test_audio = len(test_keys)

        checks = [
            ("wav_train", len(wav_train), expected_train_audio),
            ("wav_val", len(wav_val), expected_val_audio),
            ("wav_test", len(wav_test), expected_test_audio),
            ("mfcc_train", len(mfcc_train), expected_train_audio),
            ("mfcc_val", len(mfcc_val), expected_val_audio),
            ("mfcc_test", len(mfcc_test), expected_test_audio),
            ("text_train", len(text_train), len(train_keys)),
            ("text_val", len(text_val), len(val_keys)),
            ("text_test", len(text_test), len(test_keys)),
        ]

        for name, got, expected in checks:
            if got != expected:
                raise ValueError(
                    f"{fold_name}: {name} row count mismatch. "
                    f"got={got}, expected={expected}"
                )

        # Keep validation and test original-only.
        assert_orig_only(wav_val, fold_name, "wav_val")
        assert_orig_only(wav_test, fold_name, "wav_test")
        assert_orig_only(mfcc_val, fold_name, "mfcc_val")
        assert_orig_only(mfcc_test, fold_name, "mfcc_test")

        write_manifest_triplet(
            fold_dir / "wav2vec2_features",
            wav_train,
            wav_val,
            wav_test,
        )

        write_manifest_triplet(
            fold_dir / "mfcc_features",
            mfcc_train,
            mfcc_val,
            mfcc_test,
        )

        write_manifest_triplet(
            fold_dir / "text_distilbert_features",
            text_train,
            text_val,
            text_test,
        )

        write_source_key_lists(
            fold_dir,
            train_keys,
            val_keys,
            test_keys,
        )

        fold_info = {
            "fold": fold_name,
            "test_session": test_session,
            "seed": int(args.seed),
            "val_ratio": float(args.val_ratio),
            "train_source_utterances": int(len(train_keys)),
            "val_source_utterances": int(len(val_keys)),
            "test_source_utterances": int(len(test_keys)),
            "train_audio_rows": int(len(wav_train)),
            "val_audio_rows": int(len(wav_val)),
            "test_audio_rows": int(len(wav_test)),
            "train_text_rows": int(len(text_train)),
            "val_text_rows": int(len(text_val)),
            "test_text_rows": int(len(text_test)),
            "train_sessions": sorted(wav_train["session"].astype(str).unique().tolist()),
            "val_sessions": sorted(wav_val["session"].astype(str).unique().tolist()),
            "test_sessions": sorted(wav_test["session"].astype(str).unique().tolist()),
            "protocol_note": (
                "Train audio contains original and silence-removed variants. "
                "Validation and test audio contain original utterances only. "
                "Text uses one baseline DistilBERT embedding per source utterance."
            ),
        }

        with open(fold_dir / "fold_info.json", "w", encoding="utf-8") as f:
            json.dump(fold_info, f, indent=2)

        summary_rows.append(fold_info)

        print("train source utterances:", len(train_keys))
        print("val source utterances  :", len(val_keys))
        print("test source utterances :", len(test_keys))
        print("train audio rows       :", len(wav_train))
        print("val audio rows         :", len(wav_val))
        print("test audio rows        :", len(wav_test))
        print("train text rows        :", len(text_train))
        print("val text rows          :", len(text_val))
        print("test text rows         :", len(text_test))

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(build_root / "loso_manifest_summary.csv", index=False, encoding="utf-8-sig")

    with open(build_root / "loso_manifest_build_info.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "audio_setting": AUDIO_SETTING,
                "text_setting": TEXT_SETTING,
                "experiment_name": EXPERIMENT_NAME,
                "seed": int(args.seed),
                "val_ratio": float(args.val_ratio),
                "aug_root": str(AUG_ROOT),
                "wav2vec2_dir": str(WAV2VEC2_DIR),
                "mfcc_dir": str(MFCC_DIR),
                "text_dir": str(TEXT_DIR),
                "out_root": str(OUT_ROOT),
                "sessions": SESSIONS,
                "protocol": (
                    "Five-fold IEMOCAP LOSO. Each session is used once as the test session. "
                    "The remaining sessions form the development pool. Validation source utterances "
                    "are sampled from the development pool. Training audio uses original and "
                    "silence-removed variants. Validation and test audio use original utterances only."
                ),
            },
            f,
            indent=2,
        )

    # Publish the output folder only after every fold passes validation.
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    build_root.rename(OUT_ROOT)

    print("\n[DONE] LOSO manifests written to:", OUT_ROOT)
    print("Summary:", OUT_ROOT / "loso_manifest_summary.csv")


if __name__ == "__main__":
    main()
