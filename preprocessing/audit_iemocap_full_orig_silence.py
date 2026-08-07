import os
from pathlib import Path
import argparse
import pandas as pd

ENV_DATA_ROOT = Path(os.getenv("SER_DATA_ROOT", "data"))
ENV_OUTPUT_ROOT = Path(os.getenv("SER_OUTPUT_ROOT", "outputs"))


AUG_SUFFIXES = ["__orig", "__silence", "__noise", "__reverb"]


def strip_aug_suffix(x):
    x = str(x)
    for suffix in AUG_SUFFIXES:
        if x.endswith(suffix):
            return x[:-len(suffix)]
    return x


def source_key(df: pd.DataFrame) -> pd.Series:
    if "source_utt_id" in df.columns:
        return df["source_utt_id"].astype(str).map(strip_aug_suffix)
    if "utterance_id" in df.columns:
        return df["utterance_id"].astype(str).map(strip_aug_suffix)
    raise ValueError("Need source_utt_id or utterance_id column to build source key.")


def read_split_manifests(folder: Path):
    frames = []
    for split in ["train", "val", "test"]:
        p = folder / f"{split}_manifest.csv"
        if not p.exists():
            print(f"[WARN] missing {p}")
            continue
        df = pd.read_csv(p)
        df["_old_split"] = split
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No train/val/test manifests found in {folder}")
    out = pd.concat(frames, ignore_index=True)
    out["_source_key"] = source_key(out)
    return out


def infer_aug_tag(df: pd.DataFrame) -> pd.Series:
    if "aug_tag" in df.columns:
        return df["aug_tag"].astype(str)
    if "utterance_id" in df.columns:
        u = df["utterance_id"].astype(str)
        tag = pd.Series(["orig"] * len(df), index=df.index, dtype="object")
        for s in AUG_SUFFIXES:
            name = s.replace("__", "")
            tag[u.str.endswith(s)] = name
        return tag
    return pd.Series(["orig"] * len(df), index=df.index, dtype="object")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aug_root", default=str(ENV_OUTPUT_ROOT / 'aug_runs_audio'))
    ap.add_argument("--audio_setting", default="silence")
    ap.add_argument("--text_setting", default="baseline")
    args = ap.parse_args()

    aug_root = Path(args.aug_root)
    wav_dir = aug_root / args.audio_setting / "wav2vec2_features"
    mfcc_dir = aug_root / args.audio_setting / "mfcc_features"
    text_dir = aug_root / args.text_setting / "text_distilbert_features"

    print("WAV2VEC2_DIR:", wav_dir)
    print("MFCC_DIR    :", mfcc_dir)
    print("TEXT_DIR    :", text_dir)

    data = {
        "wav2vec2": read_split_manifests(wav_dir),
        "mfcc": read_split_manifests(mfcc_dir),
        "text": read_split_manifests(text_dir),
    }

    for name, df in data.items():
        print(f"\n================ {name} ================")
        print("rows:", len(df))
        print("unique source keys:", df["_source_key"].nunique())

        if "session" in df.columns:
            if name in ["wav2vec2", "mfcc"]:
                df = df.copy()
                df["_aug_tag"] = infer_aug_tag(df)
                tab = pd.crosstab(df["session"].astype(str), df["_aug_tag"].astype(str))
                print("\nCount by session and aug_tag:")
                print(tab.to_string())

                for required in ["orig", args.audio_setting]:
                    if required not in tab.columns:
                        print(f"\n[FAIL] Missing aug_tag column: {required}")
                        continue
                    missing_sessions = tab.index[tab[required] == 0].tolist()
                    if missing_sessions:
                        print(f"\n[FAIL] {required} missing in sessions: {missing_sessions}")
                    else:
                        print(f"\n[OK] {required} exists in every session.")

                # Each source should have one original and one silence audio row.
                counts = df[df["_aug_tag"].isin(["orig", args.audio_setting])].groupby(["session", "_source_key"])["_aug_tag"].nunique()
                bad = counts[counts < 2]
                print(f"source keys without both orig and {args.audio_setting}: {len(bad)}")
                if len(bad) > 0:
                    print(bad.head(20).to_string())
            else:
                print("\nText rows by session:")
                print(df["session"].astype(str).value_counts().sort_index().to_string())
        else:
            print("[WARN] no session column")

        # Check feature paths.
        for col in ["seq_feat_path", "text_feat_path"]:
            if col in df.columns:
                existing = df[col].apply(lambda x: Path(str(x)).exists()).sum()
                print(f"{col} existing: {existing}/{len(df)}; missing: {len(df)-existing}")


if __name__ == "__main__":
    main()
