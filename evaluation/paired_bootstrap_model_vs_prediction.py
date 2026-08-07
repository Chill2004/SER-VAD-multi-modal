#!/usr/bin/env python3
"""
Paired bootstrap confidence intervals for model-level vs prediction-level fusion.

For each bootstrap replicate:
1. Resample the same Session-5 utterance indices with replacement.
2. Apply those indices to both fusion methods and all three seeds.
3. Compute CCC for Valence, Arousal, Dominance for each seed.
4. Average target CCCs to obtain each seed's mean CCC.
5. Average across seeds.
6. Record model-level, prediction-level, and their paired difference.

This estimates test-set sampling uncertainty conditional on the three trained
model instances. It does not replace random-seed uncertainty.
"""
import os

import argparse
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
        "--model_root",
        type=Path,
        default=Path(
            str(ENV_OUTPUT_ROOT / 'aug_runs_results_multimodal_model_level')
        ),
    )
    parser.add_argument(
        "--prediction_root",
        type=Path,
        default=Path(
            str(ENV_OUTPUT_ROOT / 'prediction_level_fusion_3seeds')
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(
            str(ENV_OUTPUT_ROOT / 'bootstrap_model_vs_prediction')
        ),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024])
    parser.add_argument("--n_bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap_seed", type=int, default=2026)
    parser.add_argument("--expected_rows", type=int, default=2170)
    return parser.parse_args()


def strip_suffix(value):
    value = str(value)
    for suffix in AUG_SUFFIXES:
        if value.endswith(suffix):
            return value[:-len(suffix)]
    return value


def make_source_key(df):
    if "source_utt_id" in df.columns:
        return df["source_utt_id"].astype(str).map(strip_suffix)
    if "utterance_id" in df.columns:
        return df["utterance_id"].astype(str).map(strip_suffix)
    raise ValueError("Missing source_utt_id or utterance_id.")


def ccc(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    mean_true = y_true.mean()
    mean_pred = y_pred.mean()
    var_true = y_true.var()
    var_pred = y_pred.var()
    covariance = np.mean(
        (y_true - mean_true) * (y_pred - mean_pred)
    )

    return float(
        (2.0 * covariance)
        / (
            var_true
            + var_pred
            + (mean_true - mean_pred) ** 2
            + 1e-12
        )
    )


def load_method_predictions(path, label, expected_rows):
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")

    df = pd.read_csv(path).copy()

    if "aug_tag" in df.columns:
        aug = df["aug_tag"].astype(str)
        if (aug == "orig").any():
            df = df.loc[aug == "orig"].copy()

    df["_source_key"] = make_source_key(df)

    if df["_source_key"].duplicated().any():
        raise ValueError(f"{label}: duplicated source utterances.")

    required = []
    for target in TARGETS:
        required.extend(
            [f"{target}_true_norm", f"{target}_pred_norm"]
        )

    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{label}: missing columns {missing}")

    if len(df) != expected_rows:
        raise ValueError(
            f"{label}: expected {expected_rows} rows, found {len(df)}."
        )

    columns = ["_source_key"] + required
    return (
        df[columns]
        .sort_values("_source_key")
        .reset_index(drop=True)
    )


def load_all(args):
    data = {}

    for seed in args.seeds:
        model_path = (
            args.model_root
            / f"silence_audio_baseline_text_seed{seed}"
            / "model_level_predictions.csv"
        )
        prediction_path = (
            args.prediction_root
            / f"seed{seed}"
            / "prediction_level_predictions.csv"
        )

        model_df = load_method_predictions(
            model_path,
            f"model-level seed {seed}",
            args.expected_rows,
        )
        prediction_df = load_method_predictions(
            prediction_path,
            f"prediction-level seed {seed}",
            args.expected_rows,
        )

        if not model_df["_source_key"].equals(
            prediction_df["_source_key"]
        ):
            raise ValueError(
                f"Seed {seed}: source utterance order differs."
            )

        for target in TARGETS:
            model_true = model_df[
                f"{target}_true_norm"
            ].to_numpy(dtype=np.float64)
            prediction_true = prediction_df[
                f"{target}_true_norm"
            ].to_numpy(dtype=np.float64)

            if not np.allclose(
                model_true,
                prediction_true,
                atol=1e-7,
            ):
                raise ValueError(
                    f"Seed {seed}: ground truth differs for {target}."
                )

        data[seed] = {
            "model": model_df,
            "prediction": prediction_df,
        }

        print(
            f"Seed {seed}: aligned {len(model_df)} utterances."
        )

    reference_keys = data[args.seeds[0]]["model"]["_source_key"]

    for seed in args.seeds[1:]:
        current_keys = data[seed]["model"]["_source_key"]
        if not reference_keys.equals(current_keys):
            raise ValueError(
                f"Source utterance order differs across seeds: "
                f"{args.seeds[0]} vs {seed}."
            )

    return data


def calculate_point_estimates(data, seeds):
    result = {}

    for method in ("model", "prediction"):
        target_values = {target: [] for target in TARGETS}
        seed_means = []

        for seed in seeds:
            df = data[seed][method]
            seed_target_cccs = []

            for target in TARGETS:
                value = ccc(
                    df[f"{target}_true_norm"],
                    df[f"{target}_pred_norm"],
                )
                target_values[target].append(value)
                seed_target_cccs.append(value)

            seed_means.append(float(np.mean(seed_target_cccs)))

        result[method] = {
            "target_mean_across_seeds": {
                target: float(np.mean(target_values[target]))
                for target in TARGETS
            },
            "seed_mean_cccs": seed_means,
            "mean_ccc": float(np.mean(seed_means)),
            "sample_sd_across_seeds": float(
                np.std(seed_means, ddof=1)
            ),
        }

    result["difference"] = {
        "mean_ccc": (
            result["model"]["mean_ccc"]
            - result["prediction"]["mean_ccc"]
        ),
        "target_difference": {
            target: (
                result["model"]["target_mean_across_seeds"][target]
                - result["prediction"]["target_mean_across_seeds"][target]
            )
            for target in TARGETS
        },
    }

    return result


def run_bootstrap(data, args):
    rng = np.random.default_rng(args.bootstrap_seed)
    n = args.expected_rows

    columns = []

    for method in ("model", "prediction"):
        for target in TARGETS:
            columns.append(f"{method}_{target}")
        columns.append(f"{method}_mean")

    for target in TARGETS:
        columns.append(f"difference_{target}")
    columns.append("difference_mean")

    distributions = {
        column: np.empty(args.n_bootstrap, dtype=np.float64)
        for column in columns
    }

    for iteration in range(args.n_bootstrap):
        indices = rng.integers(0, n, size=n)

        method_results = {}

        for method in ("model", "prediction"):
            target_across_seeds = {
                target: [] for target in TARGETS
            }
            seed_means = []

            for seed in args.seeds:
                df = data[seed][method]
                seed_target_values = []

                for target in TARGETS:
                    true_values = df[
                        f"{target}_true_norm"
                    ].to_numpy(dtype=np.float64)[indices]
                    predictions = df[
                        f"{target}_pred_norm"
                    ].to_numpy(dtype=np.float64)[indices]

                    value = ccc(true_values, predictions)
                    target_across_seeds[target].append(value)
                    seed_target_values.append(value)

                seed_means.append(float(np.mean(seed_target_values)))

            method_results[method] = {
                target: float(
                    np.mean(target_across_seeds[target])
                )
                for target in TARGETS
            }
            method_results[method]["mean"] = float(
                np.mean(seed_means)
            )

            for target in TARGETS:
                distributions[
                    f"{method}_{target}"
                ][iteration] = method_results[method][target]

            distributions[
                f"{method}_mean"
            ][iteration] = method_results[method]["mean"]

        for target in TARGETS:
            distributions[
                f"difference_{target}"
            ][iteration] = (
                method_results["model"][target]
                - method_results["prediction"][target]
            )

        distributions["difference_mean"][iteration] = (
            method_results["model"]["mean"]
            - method_results["prediction"]["mean"]
        )

        if (iteration + 1) % 1000 == 0:
            print(
                f"Bootstrap {iteration + 1}/{args.n_bootstrap}"
            )

    return distributions


def percentile_ci(values):
    lower, upper = np.percentile(values, [2.5, 97.5])
    return float(lower), float(upper)


def build_summary(point, distributions):
    rows = []

    mapping = []

    for method in ("model", "prediction"):
        for target in TARGETS:
            mapping.append(
                (
                    method,
                    target,
                    point[method]["target_mean_across_seeds"][target],
                    f"{method}_{target}",
                )
            )

        mapping.append(
            (
                method,
                "mean",
                point[method]["mean_ccc"],
                f"{method}_mean",
            )
        )

    for target in TARGETS:
        mapping.append(
            (
                "model_minus_prediction",
                target,
                point["difference"]["target_difference"][target],
                f"difference_{target}",
            )
        )

    mapping.append(
        (
            "model_minus_prediction",
            "mean",
            point["difference"]["mean_ccc"],
            "difference_mean",
        )
    )

    for method, target, estimate, distribution_name in mapping:
        lower, upper = percentile_ci(
            distributions[distribution_name]
        )

        rows.append(
            {
                "comparison": method,
                "target": target,
                "point_estimate": estimate,
                "ci_95_lower": lower,
                "ci_95_upper": upper,
                "ci_excludes_zero": bool(
                    lower > 0.0 or upper < 0.0
                ),
            }
        )

    return pd.DataFrame(rows)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = load_all(args)
    point = calculate_point_estimates(data, args.seeds)

    print("\n===== POINT ESTIMATES =====")
    print(
        f"Model-level: "
        f"{point['model']['mean_ccc']:.6f} "
        f"± {point['model']['sample_sd_across_seeds']:.6f}"
    )
    print(
        f"Prediction-level: "
        f"{point['prediction']['mean_ccc']:.6f} "
        f"± {point['prediction']['sample_sd_across_seeds']:.6f}"
    )
    print(
        f"Difference: "
        f"{point['difference']['mean_ccc']:.6f}"
    )

    distributions = run_bootstrap(data, args)
    summary = build_summary(point, distributions)

    summary.to_csv(
        args.output_dir / "paired_bootstrap_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(distributions).to_csv(
        args.output_dir / "paired_bootstrap_distributions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n===== 95% PERCENTILE BOOTSTRAP CI =====")
    print(summary.to_string(index=False))

    mean_difference = summary.loc[
        (summary["comparison"] == "model_minus_prediction")
        & (summary["target"] == "mean")
    ].iloc[0]

    print("\nPaper-ready result:")
    print(
        "Model-level minus prediction-level mean CCC = "
        f"{mean_difference['point_estimate']:.3f}, "
        "95% CI ["
        f"{mean_difference['ci_95_lower']:.3f}, "
        f"{mean_difference['ci_95_upper']:.3f}"
        "]."
    )

    print(f"\nSaved to: {args.output_dir}")


if __name__ == "__main__":
    main()
