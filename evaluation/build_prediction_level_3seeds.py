import os
#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ENV_DATA_ROOT = Path(os.getenv("SER_DATA_ROOT", "data"))
ENV_OUTPUT_ROOT = Path(os.getenv("SER_OUTPUT_ROOT", "outputs"))


TARGETS = ("valence", "arousal", "dominance")
AUG_SUFFIXES = ("__orig", "__silence", "__noise", "__reverb")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio_root",
        type=Path,
        default=Path(str(ENV_OUTPUT_ROOT / 'aug_runs_results_fusion_bilstm_cnn1d_tuned')),
    )
    parser.add_argument(
        "--text_root",
        type=Path,
        default=Path(str(ENV_OUTPUT_ROOT / 'aug_runs_results_text_only')),
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path(str(ENV_OUTPUT_ROOT / 'prediction_level_fusion_3seeds')),
    )
    parser.add_argument("--audio_setting", default="silence")
    parser.add_argument("--text_setting", default="baseline")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024])
    parser.add_argument("--weight_step", type=float, default=0.01)
    parser.add_argument("--expected_val_rows", type=int, default=787)
    parser.add_argument("--expected_test_rows", type=int, default=2170)
    return parser.parse_args()


def strip_suffix(value):
    value = str(value)
    for suffix in AUG_SUFFIXES:
        if value.endswith(suffix):
            return value[:-len(suffix)]
    return value


def source_key(df):
    if "source_utt_id" in df.columns:
        return df["source_utt_id"].astype(str).map(strip_suffix)
    if "utterance_id" in df.columns:
        return df["utterance_id"].astype(str).map(strip_suffix)
    raise ValueError("Missing source_utt_id/utterance_id")


def ccc(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mt, mp = y_true.mean(), y_pred.mean()
    vt, vp = y_true.var(), y_pred.var()
    cov = np.mean((y_true - mt) * (y_pred - mp))
    return float((2 * cov) / (vt + vp + (mt - mp) ** 2 + 1e-12))


def load_predictions(path, expected_rows, label):
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")

    df = pd.read_csv(path).copy()

    if "aug_tag" in df.columns and (df["aug_tag"].astype(str) == "orig").any():
        df = df.loc[df["aug_tag"].astype(str) == "orig"].copy()

    df["_source_key"] = source_key(df)

    if df["_source_key"].duplicated().any():
        raise ValueError(f"{label}: duplicated source utterances")

    for target in TARGETS:
        for column in (f"{target}_true_norm", f"{target}_pred_norm"):
            if column not in df.columns:
                raise ValueError(f"{label}: missing {column}")

    if len(df) != expected_rows:
        raise ValueError(f"{label}: expected {expected_rows}, found {len(df)}")

    return df.reset_index(drop=True)


def align(audio_df, text_df, expected_rows, label):
    audio_cols = ["_source_key"]
    text_cols = ["_source_key"]

    for target in TARGETS:
        audio_cols += [f"{target}_true_norm", f"{target}_pred_norm"]
        text_cols += [f"{target}_true_norm", f"{target}_pred_norm"]

    for column in ("utterance_id", "source_utt_id", "audio_path"):
        if column in audio_df.columns:
            audio_cols.append(column)

    for column in ("transcript", "session", "dialogue_id", "speaker_gender"):
        if column in text_df.columns:
            text_cols.append(column)

    merged = audio_df[audio_cols].merge(
        text_df[text_cols],
        on="_source_key",
        how="inner",
        suffixes=("_audio", "_text"),
        validate="one_to_one",
    )

    if len(merged) != expected_rows:
        raise ValueError(f"{label}: expected {expected_rows} aligned rows, found {len(merged)}")

    for target in TARGETS:
        a = merged[f"{target}_true_norm_audio"].to_numpy(float)
        t = merged[f"{target}_true_norm_text"].to_numpy(float)
        if not np.allclose(a, t, atol=1e-7):
            raise ValueError(f"{label}: ground truth mismatch for {target}")

    return merged.sort_values("_source_key").reset_index(drop=True)


def find_shared_weights(val_frames, step):
    weights = np.arange(0.0, 1.0 + step / 2.0, step)
    result = {}

    for target in TARGETS:
        best_weight = None
        best_score = -np.inf

        for weight in weights:
            scores = []
            for frame in val_frames:
                y = frame[f"{target}_true_norm_audio"].to_numpy(float)
                pa = frame[f"{target}_pred_norm_audio"].to_numpy(float)
                pt = frame[f"{target}_pred_norm_text"].to_numpy(float)
                pred = weight * pa + (1.0 - weight) * pt
                scores.append(ccc(y, pred))

            score = float(np.mean(scores))
            if score > best_score:
                best_score = score
                best_weight = float(weight)

        result[target] = {
            "audio_weight": best_weight,
            "text_weight": 1.0 - best_weight,
            "validation_ccc": best_score,
        }

    return result


def build_predictions(frame, weights):
    output = pd.DataFrame({"source_utt_id": frame["_source_key"].astype(str)})

    for column in ("utterance_id_audio", "audio_path", "transcript", "session", "dialogue_id", "speaker_gender"):
        if column in frame.columns:
            output[column.replace("_audio", "")] = frame[column]

    metrics = []

    for target in TARGETS:
        y = frame[f"{target}_true_norm_audio"].to_numpy(float)
        pa = frame[f"{target}_pred_norm_audio"].to_numpy(float)
        pt = frame[f"{target}_pred_norm_text"].to_numpy(float)
        wa = weights[target]["audio_weight"]
        wt = 1.0 - wa
        pred = wa * pa + wt * pt

        output[f"{target}_true_norm"] = y
        output[f"{target}_audio_pred_norm"] = pa
        output[f"{target}_text_pred_norm"] = pt
        output[f"{target}_pred_norm"] = pred

        metrics.append({
            "target": target,
            "ccc": ccc(y, pred),
            "audio_weight": wa,
            "text_weight": wt,
        })

    metrics_df = pd.DataFrame(metrics)
    metrics_df = pd.concat([
        metrics_df,
        pd.DataFrame([{
            "target": "mean",
            "ccc": float(metrics_df["ccc"].mean()),
            "audio_weight": np.nan,
            "text_weight": np.nan,
        }]),
    ], ignore_index=True)

    return output, metrics_df


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    aligned = {}

    for seed in args.seeds:
        audio_dir = args.audio_root / f"{args.audio_setting}_seed{seed}"
        text_dir = args.text_root / f"{args.text_setting}_seed{seed}"

        audio_val = load_predictions(
            audio_dir / "fusion_val_predictions.csv",
            args.expected_val_rows,
            f"audio val seed {seed}",
        )
        audio_test = load_predictions(
            audio_dir / "fusion_predictions.csv",
            args.expected_test_rows,
            f"audio test seed {seed}",
        )
        text_val = load_predictions(
            text_dir / "text_embedding_val_predictions.csv",
            args.expected_val_rows,
            f"text val seed {seed}",
        )
        text_test = load_predictions(
            text_dir / "text_embedding_predictions.csv",
            args.expected_test_rows,
            f"text test seed {seed}",
        )

        aligned[seed] = {
            "val": align(audio_val, text_val, args.expected_val_rows, f"val seed {seed}"),
            "test": align(audio_test, text_test, args.expected_test_rows, f"test seed {seed}"),
        }

        print(f"Seed {seed}: val={len(aligned[seed]['val'])}, test={len(aligned[seed]['test'])}")

    weights = find_shared_weights(
        [aligned[seed]["val"] for seed in args.seeds],
        args.weight_step,
    )

    print("\nSelected shared target-wise weights:")
    for target in TARGETS:
        item = weights[target]
        print(
            f"{target}: audio={item['audio_weight']:.2f}, "
            f"text={item['text_weight']:.2f}, "
            f"mean val CCC={item['validation_ccc']:.6f}"
        )

    summary_rows = []

    for seed in args.seeds:
        seed_dir = args.output_root / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        val_pred, val_metrics = build_predictions(aligned[seed]["val"], weights)
        test_pred, test_metrics = build_predictions(aligned[seed]["test"], weights)

        val_pred.to_csv(seed_dir / "prediction_level_val_predictions.csv", index=False, encoding="utf-8-sig")
        test_pred.to_csv(seed_dir / "prediction_level_predictions.csv", index=False, encoding="utf-8-sig")
        val_metrics.to_csv(seed_dir / "prediction_level_val_metrics.csv", index=False, encoding="utf-8-sig")
        test_metrics.to_csv(seed_dir / "prediction_level_metrics.csv", index=False, encoding="utf-8-sig")

        mean_ccc = float(test_metrics.loc[test_metrics["target"] == "mean", "ccc"].iloc[0])

        row = {"seed": seed, "mean_ccc": mean_ccc}
        for target in TARGETS:
            row[f"{target}_ccc"] = float(
                test_metrics.loc[test_metrics["target"] == target, "ccc"].iloc[0]
            )
        summary_rows.append(row)

        print(
            f"Seed {seed}: V={row['valence_ccc']:.6f}, "
            f"A={row['arousal_ccc']:.6f}, "
            f"D={row['dominance_ccc']:.6f}, "
            f"Mean={row['mean_ccc']:.6f}"
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        args.output_root / "prediction_level_3seeds_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with open(args.output_root / "prediction_level_weights.json", "w", encoding="utf-8") as f:
        json.dump(weights, f, indent=2)

    aggregate = {
        "mean_ccc": float(summary_df["mean_ccc"].mean()),
        "sample_std": float(summary_df["mean_ccc"].std(ddof=1)),
    }
    with open(args.output_root / "prediction_level_aggregate.json", "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)

    print("\n===== THREE-SEED SUMMARY =====")
    print(summary_df.to_string(index=False))
    print(
        f"\nPaper value: {aggregate['mean_ccc']:.3f} "
        f"± {aggregate['sample_std']:.3f}"
    )
    print(f"Saved to: {args.output_root}")


if __name__ == "__main__":
    main()
