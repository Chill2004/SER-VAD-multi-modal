# Results

Paper-facing CSV artifacts.

## Primary repeated-seed LOSO

The final paper uses five held-out IEMOCAP sessions and three training seeds per
fold (`42`, `123`, `2024`). The aggregation order is:

1. average the three seeds within each held-out session;
2. compute the mean and sample standard deviation across the five seed-averaged
   held-out sessions.

Final paper-facing headline:

- Valence: **0.684 ± 0.045**
- Arousal: **0.673 ± 0.018**
- Dominance: **0.548 ± 0.034**
- Mean: **0.635 ± 0.011**

Files:

- `loso_results_by_seed.csv`: all 15 fold × seed CCC rows.
- `loso_seed_averaged_by_fold.csv`: the five seed-averaged fold rows used for
  Table V.
- `loso_repeated_seed_summary.csv`: full-precision repeated-seed summary.
- `loso_results.csv`: compatibility alias of the final seed-averaged fold table.
- `loso_mean_std_summary.csv`: compatibility alias of the final mean/std summary.
- `loso_seed42_results.csv`: archived single-seed LOSO fold values.
- `loso_seed42_mean_std_summary.csv`: archived single-seed LOSO mean/std.

The seed-42 archive is retained only for provenance; it is not the primary LOSO
result reported by the final paper.

Branch/ablation/MSP files are lightweight paper-facing artifacts. Values marked or
described as paper-rounded were transcribed from the paper when only rounded
values were available. Raw datasets, features, checkpoints, and large
per-utterance prediction files are intentionally excluded.
