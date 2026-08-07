import os
#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ENV_DATA_ROOT = Path(os.getenv("SER_DATA_ROOT", "data"))
ENV_OUTPUT_ROOT = Path(os.getenv("SER_OUTPUT_ROOT", "outputs"))


SEEDS_DEFAULT = [42, 123, 2024]
AUG_SUFFIXES = ("__orig", "__silence", "__noise", "__reverb")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_root", type=Path, default=Path(str(ENV_OUTPUT_ROOT / 'aug_runs_results_multimodal_model_level')))
    parser.add_argument("--audio_root", type=Path, default=Path(str(ENV_OUTPUT_ROOT / 'aug_runs_results_fusion_bilstm_cnn1d_tuned')))
    parser.add_argument("--text_root", type=Path, default=Path(str(ENV_OUTPUT_ROOT / 'aug_runs_results_text_only')))
    parser.add_argument("--output_dir", type=Path, default=Path(str(ENV_OUTPUT_ROOT / 'valence_case_analysis')))
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS_DEFAULT)
    parser.add_argument("--expected_rows", type=int, default=2170)
    parser.add_argument("--min_consistent_seeds", type=int, default=2)
    parser.add_argument("--min_model_gain", type=float, default=0.10)
    parser.add_argument("--min_text_gain", type=float, default=0.10)
    parser.add_argument("--top_n", type=int, default=30)
    return parser.parse_args()


def strip_aug_suffix(value):
    value = str(value)
    for suffix in AUG_SUFFIXES:
        if value.endswith(suffix):
            return value[:-len(suffix)]
    return value


def make_source_key(df):
    if "source_utt_id" in df.columns:
        return df["source_utt_id"].astype(str).map(strip_aug_suffix)
    if "utterance_id" in df.columns:
        return df["utterance_id"].astype(str).map(strip_aug_suffix)
    raise ValueError("Missing source_utt_id and utterance_id.")


def keep_original_rows(df):
    if "aug_tag" in df.columns:
        tags = df["aug_tag"].astype(str)
        if (tags == "orig").any():
            return df.loc[tags == "orig"].copy()
    return df.copy()


def find_column(df, candidates, label):
    for column in candidates:
        if column in df.columns:
            return column
    raise ValueError(f"Could not find {label}. Tried: {candidates}")


def load_one(path, label, expected_rows):
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")

    df = keep_original_rows(pd.read_csv(path))
    df["_source_key"] = make_source_key(df)

    if df["_source_key"].duplicated().any():
        raise ValueError(f"{label} contains duplicated source utterances.")

    true_col = find_column(df, ["valence_true_norm", "valence_norm"], f"{label} Valence ground truth")
    pred_col = find_column(df, ["valence_pred_norm"], f"{label} Valence prediction")

    if len(df) != expected_rows:
        raise ValueError(f"{label}: expected {expected_rows} rows, found {len(df)}.")

    keep = ["_source_key", true_col, pred_col]
    metadata = ["utterance_id", "source_utt_id", "transcript", "text", "sentence", "dialogue_id", "session", "speaker_gender", "audio_path"]
    keep += [c for c in metadata if c in df.columns]

    out = df[keep].copy().rename(columns={true_col: "valence_true_norm", pred_col: "valence_pred_norm"})
    return out.sort_values("_source_key").reset_index(drop=True)


def merge_seed(model_df, audio_df, text_df, seed):
    model = model_df.rename(columns={"valence_true_norm": f"true_model_seed{seed}", "valence_pred_norm": f"model_seed{seed}"})
    audio = audio_df[["_source_key", "valence_true_norm", "valence_pred_norm"]].rename(columns={"valence_true_norm": f"true_audio_seed{seed}", "valence_pred_norm": f"audio_seed{seed}"})
    text = text_df.rename(columns={"valence_true_norm": f"true_text_seed{seed}", "valence_pred_norm": f"text_seed{seed}"})

    merged = model.merge(audio, on="_source_key", how="inner", validate="one_to_one").merge(text, on="_source_key", how="inner", suffixes=("_model", "_text"), validate="one_to_one")

    true_columns = [f"true_model_seed{seed}", f"true_audio_seed{seed}", f"true_text_seed{seed}"]
    reference = merged[true_columns[0]].to_numpy(float)
    for column in true_columns[1:]:
        if not np.allclose(reference, merged[column].to_numpy(float), atol=1e-7):
            raise ValueError(f"Seed {seed}: Valence ground truth mismatch.")

    for base in ("utterance_id", "source_utt_id", "transcript", "text", "sentence", "dialogue_id", "session", "speaker_gender", "audio_path"):
        existing = [c for c in (base, f"{base}_model", f"{base}_text") if c in merged.columns]
        if existing:
            series = merged[existing[0]]
            for column in existing[1:]:
                series = series.fillna(merged[column])
            merged[base] = series

    return merged


def normalized_to_1to5(values):
    return 2.0 * np.asarray(values, dtype=float) + 3.0


def choose_suggested_shortlist(candidates, n=3):
    if candidates.empty:
        return candidates.copy()

    selected = []
    used_dialogues = set()

    def take(frame):
        for idx, row in frame.iterrows():
            dialogue = str(row.get("dialogue_id", ""))
            if dialogue and dialogue != "nan" and dialogue in used_dialogues:
                continue
            selected.append(idx)
            if dialogue and dialogue != "nan":
                used_dialogues.add(dialogue)
            return

    take(candidates.loc[candidates["true_valence_norm"] <= -0.20])
    take(candidates.loc[candidates["true_valence_norm"] >= 0.20])

    for idx, row in candidates.iterrows():
        if len(selected) >= n:
            break
        if idx in selected:
            continue
        dialogue = str(row.get("dialogue_id", ""))
        if dialogue and dialogue != "nan" and dialogue in used_dialogues:
            continue
        selected.append(idx)
        if dialogue and dialogue != "nan":
            used_dialogues.add(dialogue)

    return candidates.loc[selected[:n]].copy()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged_by_seed = []

    for seed in args.seeds:
        model_path = args.model_root / f"silence_audio_baseline_text_seed{seed}" / "model_level_predictions.csv"
        audio_path = args.audio_root / f"silence_seed{seed}" / "fusion_predictions.csv"
        text_path = args.text_root / f"baseline_seed{seed}" / "text_embedding_predictions.csv"

        model_df = load_one(model_path, f"model-level seed {seed}", args.expected_rows)
        audio_df = load_one(audio_path, f"audio-fusion seed {seed}", args.expected_rows)
        text_df = load_one(text_path, f"text-only seed {seed}", args.expected_rows)
        seed_df = merge_seed(model_df, audio_df, text_df, seed)

        if len(seed_df) != args.expected_rows:
            raise ValueError(f"Seed {seed}: expected {args.expected_rows} aligned rows, found {len(seed_df)}.")

        merged_by_seed.append(seed_df)
        print(f"Seed {seed}: aligned {len(seed_df)} utterances.")

    combined = merged_by_seed[0].copy()
    for seed_df, seed in zip(merged_by_seed[1:], args.seeds[1:]):
        cols = ["_source_key", f"true_model_seed{seed}", f"model_seed{seed}", f"true_audio_seed{seed}", f"audio_seed{seed}", f"true_text_seed{seed}", f"text_seed{seed}"]
        combined = combined.merge(seed_df[cols], on="_source_key", how="inner", validate="one_to_one")

    true_matrix = np.column_stack([combined[f"true_model_seed{seed}"].to_numpy(float) for seed in args.seeds])
    if not np.allclose(true_matrix, true_matrix[:, [0]], atol=1e-7):
        raise ValueError("Ground truth differs across seeds.")

    combined["true_valence_norm"] = true_matrix[:, 0]
    combined["true_valence_1to5"] = normalized_to_1to5(combined["true_valence_norm"])

    audio_pred = np.column_stack([combined[f"audio_seed{seed}"].to_numpy(float) for seed in args.seeds])
    text_pred = np.column_stack([combined[f"text_seed{seed}"].to_numpy(float) for seed in args.seeds])
    model_pred = np.column_stack([combined[f"model_seed{seed}"].to_numpy(float) for seed in args.seeds])
    truth = combined["true_valence_norm"].to_numpy(float)[:, None]

    audio_err = np.abs(audio_pred - truth)
    text_err = np.abs(text_pred - truth)
    model_err = np.abs(model_pred - truth)

    combined["audio_pred_mean_norm"] = audio_pred.mean(axis=1)
    combined["text_pred_mean_norm"] = text_pred.mean(axis=1)
    combined["model_pred_mean_norm"] = model_pred.mean(axis=1)
    combined["audio_pred_mean_1to5"] = normalized_to_1to5(combined["audio_pred_mean_norm"])
    combined["text_pred_mean_1to5"] = normalized_to_1to5(combined["text_pred_mean_norm"])
    combined["model_pred_mean_1to5"] = normalized_to_1to5(combined["model_pred_mean_norm"])

    combined["audio_abs_error_mean"] = audio_err.mean(axis=1)
    combined["text_abs_error_mean"] = text_err.mean(axis=1)
    combined["model_abs_error_mean"] = model_err.mean(axis=1)
    combined["text_gain_over_audio"] = combined["audio_abs_error_mean"] - combined["text_abs_error_mean"]
    combined["model_gain_over_audio"] = combined["audio_abs_error_mean"] - combined["model_abs_error_mean"]

    text_better = text_err < audio_err
    model_better = model_err < audio_err
    both_better = text_better & model_better
    combined["text_better_seed_count"] = text_better.sum(axis=1)
    combined["model_better_seed_count"] = model_better.sum(axis=1)
    combined["both_better_seed_count"] = both_better.sum(axis=1)

    transcript_col = next((c for c in ("transcript", "text", "sentence") if c in combined.columns), None)
    combined["transcript"] = combined[transcript_col].fillna("").astype(str) if transcript_col else ""
    combined["transcript_word_count"] = combined["transcript"].str.split().str.len()
    combined["case_score"] = combined["model_gain_over_audio"] + 0.60 * combined["text_gain_over_audio"] + 0.04 * combined["both_better_seed_count"] + 0.02 * np.abs(combined["true_valence_norm"])

    candidates = combined.loc[
        (combined["both_better_seed_count"] >= args.min_consistent_seeds)
        & (combined["model_gain_over_audio"] >= args.min_model_gain)
        & (combined["text_gain_over_audio"] >= args.min_text_gain)
        & (combined["transcript_word_count"] >= 3)
    ].copy()

    candidates = candidates.sort_values(["case_score", "both_better_seed_count", "model_gain_over_audio"], ascending=False).head(args.top_n)

    preferred = [
        "_source_key", "utterance_id", "dialogue_id", "session", "speaker_gender", "transcript",
        "true_valence_norm", "true_valence_1to5",
        "audio_pred_mean_norm", "text_pred_mean_norm", "model_pred_mean_norm",
        "audio_pred_mean_1to5", "text_pred_mean_1to5", "model_pred_mean_1to5",
        "audio_abs_error_mean", "text_abs_error_mean", "model_abs_error_mean",
        "text_gain_over_audio", "model_gain_over_audio",
        "text_better_seed_count", "model_better_seed_count", "both_better_seed_count", "case_score",
    ]
    for seed in args.seeds:
        preferred += [f"audio_seed{seed}", f"text_seed{seed}", f"model_seed{seed}"]
    output_cols = [c for c in preferred if c in candidates.columns]

    candidates[output_cols].to_csv(args.output_dir / "valence_case_candidates.csv", index=False, encoding="utf-8-sig")
    shortlist = choose_suggested_shortlist(candidates, n=3)
    shortlist[output_cols].to_csv(args.output_dir / "valence_case_suggested_shortlist.csv", index=False, encoding="utf-8-sig")

    print("\n===== SUGGESTED SHORTLIST =====")
    display_cols = [c for c in ["_source_key", "transcript", "true_valence_1to5", "audio_pred_mean_1to5", "text_pred_mean_1to5", "model_pred_mean_1to5", "both_better_seed_count", "model_gain_over_audio"] if c in shortlist.columns]
    print(shortlist[display_cols].to_string(index=False))
    print(f"\nSaved to: {args.output_dir}")


if __name__ == "__main__":
    main()
