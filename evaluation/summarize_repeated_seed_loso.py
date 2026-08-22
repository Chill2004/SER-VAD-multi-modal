from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ENV_OUTPUT_ROOT = Path(os.getenv("SER_OUTPUT_ROOT", "outputs"))

FOLDS = ["Ses01", "Ses02", "Ses03", "Ses04", "Ses05"]
TARGETS = ["valence", "arousal", "dominance", "mean"]


def load_run_metrics(run_dir: Path, fold: str, seed: int) -> pd.DataFrame:
    metrics_path = run_dir / "model_level_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Missing metrics for fold={fold}, seed={seed}: {metrics_path}"
        )

    df = pd.read_csv(metrics_path)
    required = {"target", "mse", "rmse", "ccc"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{metrics_path}: missing required columns {sorted(missing)}"
        )

    rows = []
    normalized_target = df["target"].astype(str).str.lower()

    for target in TARGETS:
        match = df.loc[normalized_target == target]
        if len(match) != 1:
            raise ValueError(
                f"{metrics_path}: expected exactly one row for target={target}, "
                f"got {len(match)}"
            )
        row = match.iloc[0]
        rows.append(
            {
                "fold": fold,
                "seed": int(seed),
                "target": target,
                "mse": float(row["mse"]),
                "rmse": float(row["rmse"]),
                "ccc": float(row["ccc"]),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate repeated-seed five-fold LOSO results. "
            "Seeds are averaged within each held-out session first; "
            "the final sample SD is then computed across the five "
            "seed-averaged sessions."
        )
    )
    parser.add_argument(
        "--root",
        type=str,
        default=str(
            ENV_OUTPUT_ROOT / "aug_runs_results_multimodal_model_level_loso"
        ),
        help="Root containing fold/seed run directories.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123, 2024],
        help="Training seeds to aggregate.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="silence_audio_baseline_text_loso_fold",
        help="Run-directory prefix before _SesXX_seedN.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(ENV_OUTPUT_ROOT / "loso_repeated_seed_summary"),
        help="Output directory for aggregate CSV/JSON files.",
    )
    args = parser.parse_args()

    seeds = list(dict.fromkeys(int(s) for s in args.seeds))
    if len(seeds) < 2:
        raise ValueError("Repeated-seed aggregation requires at least two seeds.")

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for fold in FOLDS:
        for seed in seeds:
            run_dir = root / f"{args.prefix}_{fold}_seed{seed}"
            if not run_dir.exists():
                raise FileNotFoundError(f"Missing LOSO run directory: {run_dir}")
            frames.append(load_run_metrics(run_dir, fold, seed))

    metrics = pd.concat(frames, ignore_index=True)

    expected_runs = len(FOLDS) * len(seeds)
    observed_runs = (
        metrics[["fold", "seed"]].drop_duplicates().shape[0]
    )
    if observed_runs != expected_runs:
        raise ValueError(
            f"Expected {expected_runs} fold/seed runs, got {observed_runs}."
        )

    by_seed = (
        metrics.pivot_table(
            index=["fold", "seed"],
            columns="target",
            values="ccc",
            aggfunc="first",
        )
        .reindex(pd.MultiIndex.from_product([FOLDS, seeds], names=["fold", "seed"]))
        .reset_index()
    )
    by_seed = by_seed[
        ["fold", "seed", "valence", "arousal", "dominance", "mean"]
    ].rename(
        columns={
            "valence": "valence_ccc",
            "arousal": "arousal_ccc",
            "dominance": "dominance_ccc",
            "mean": "mean_ccc",
        }
    )
    by_seed.to_csv(out_dir / "loso_results_by_seed.csv", index=False)

    # Average seeds within each fold before cross-fold statistics.
    seed_averaged = (
        by_seed.groupby("fold", sort=False)[
            ["valence_ccc", "arousal_ccc", "dominance_ccc", "mean_ccc"]
        ]
        .mean()
        .reindex(FOLDS)
        .reset_index()
    )
    seed_averaged.to_csv(
        out_dir / "loso_seed_averaged_by_fold.csv", index=False
    )

    seed_averaged.to_csv(out_dir / "loso_results.csv", index=False)

    summary_rows = []
    for target in TARGETS:
        col = f"{target}_ccc"
        values = seed_averaged[col].to_numpy(dtype=np.float64)
        summary_rows.append(
            {
                "target": target,
                "mean_ccc": float(np.mean(values)),
                "sample_std_ccc": float(np.std(values, ddof=1)),
                "min_fold_ccc": float(np.min(values)),
                "max_fold_ccc": float(np.max(values)),
                "n_folds": int(len(values)),
                "n_seeds_per_fold": int(len(seeds)),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "loso_repeated_seed_summary.csv", index=False)
    summary[["target", "mean_ccc", "sample_std_ccc"]].to_csv(
        out_dir / "loso_mean_std_summary.csv", index=False
    )

    metadata = {
        "folds": FOLDS,
        "seeds": seeds,
        "n_runs": expected_runs,
        "aggregation": (
            "Average CCC across seeds within each held-out session, then compute "
            "the mean and sample standard deviation (ddof=1) across the five "
            "seed-averaged held-out sessions."
        ),
        "headline_mean_ccc": float(
            summary.loc[summary["target"] == "mean", "mean_ccc"].iloc[0]
        ),
        "headline_sample_std_ccc": float(
            summary.loc[
                summary["target"] == "mean", "sample_std_ccc"
            ].iloc[0]
        ),
    }
    with (out_dir / "loso_repeated_seed_summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(metadata, f, indent=2)

    print("\n===== REPEATED-SEED CCC BY HELD-OUT SESSION =====")
    print(seed_averaged.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\n===== MEAN ± SAMPLE SD ACROSS 5 SEED-AVERAGED FOLDS =====")
    for _, row in summary.iterrows():
        print(
            f"{row['target']:10s}: "
            f"{row['mean_ccc']:.6f} ± {row['sample_std_ccc']:.6f}"
        )

    print("\n===== PAPER-READY (3 decimals) =====")
    for _, row in summary.iterrows():
        print(
            f"{row['target']:10s}: "
            f"{row['mean_ccc']:.3f} ± {row['sample_std_ccc']:.3f}"
        )


if __name__ == "__main__":
    main()
