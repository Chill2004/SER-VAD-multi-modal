# Results

Paper-facing CSV/JSON artifacts.

## Camera-ready artifacts

| Result | File |
|---|---|
| IEMOCAP LOSO, five methods | [iemocap_loso_summary.csv](camera_ready/iemocap_loso_summary.csv) |
| IEMOCAP LOSO, paired mean CCC differences by fold and seed | [iemocap_loso_paired_by_seed.csv](camera_ready/iemocap_loso_paired_by_seed.csv) |
| IEMOCAP outer-fold assignments and waveform settings | [iemocap_loso_protocol.csv](camera_ready/iemocap_loso_protocol.csv) |
| LOSO and fixed-Session-5 dialogue-clustered intervals | [dialogue_clustered_ci.csv](camera_ready/dialogue_clustered_ci.csv) |
| Lightweight MFCC ablation | [mfcc_lite_ablation.csv](camera_ready/mfcc_lite_ablation.csv) |
| Fixed-silence transcript shuffle | [g6_summary.csv](camera_ready/g6_summary.csv) |
| Fixed-silence ASR substitution | [g9_summary.csv](camera_ready/g9_summary.csv), [wer_by_fold.csv](camera_ready/wer_by_fold.csv) |
| New MSP 2–6 s split, seed 42 | [msp_2to6_seed42_baselines.csv](camera_ready/msp_2to6_seed42_baselines.csv) |
| Valence cases | [valence_case_examples.csv](valence_case_examples.csv) |

The LOSO, confidence interval, MFCC ablation, MSP baseline, and Valence case
tables use the precision printed in the camera-ready paper. G6 and G9 come
from the run summaries. The WER table records fold-level counts and model
metadata. For the historical MSP split results, see
[msp_transfer_results.csv](msp_transfer_results.csv).

## Historical repeated-seed model-level LOSO

The historical LOSO experiment uses five held-out IEMOCAP sessions and three training seeds per
fold (`42`, `123`, `2024`). The aggregation order is:

1. average the three seeds within each held-out session;
2. compute the mean and sample standard deviation across the five seed-averaged
   held-out sessions.

Model-level headline:

- Valence: **0.684 ± 0.045**
- Arousal: **0.673 ± 0.018**
- Dominance: **0.548 ± 0.034**
- Mean: **0.635 ± 0.011**

Existing model-level files:

- `loso_results_by_seed.csv`: all 15 fold × seed CCC rows.
- `loso_seed_averaged_by_fold.csv`: five seed-averaged model-level fold rows.
- `loso_repeated_seed_summary.csv`: full-precision repeated-seed summary.
- `loso_results.csv`: compatibility alias of the final seed-averaged fold table.
- `loso_mean_std_summary.csv`: compatibility alias of the final mean/std summary.
- `loso_seed42_results.csv`: archived single-seed LOSO fold values.
- `loso_seed42_mean_std_summary.csv`: archived single-seed LOSO mean/std.

The seed-42 archive records the historical single-seed comparison.

## Repeated-seed prediction-level LOSO

Prediction-level fusion uses the same held-out folds and seeds. Within each fold
and target, the audio weight `alpha` is searched on
`0.00, 0.01, ..., 1.00`. A **single fold-specific weight per target** is selected
by maximizing mean validation CCC across seeds `42`, `123`, and `2024`. The
selected weight is then applied unchanged to the test predictions of all three
seeds. Test labels are never used for weight selection.

Prediction-level headline:

- Valence: **0.651 ± 0.045**
- Arousal: **0.651 ± 0.041**
- Dominance: **0.536 ± 0.041**
- Mean: **0.613 ± 0.024**

Seed-averaged mean CCC by held-out session is
`.649/.605/.590/.624/.594` for Ses01--Ses05, versus
`.650/.635/.623/.642/.625` for model-level fusion. Model-level fusion therefore
has the higher seed-averaged **mean** CCC on all five folds; individual targets
can still differ.

Prediction-level files:

- `prediction_level_alpha_by_fold.csv`: selected alpha for each fold/target.
- `prediction_level_alpha_search.csv`: complete 5 × 3 × 101 validation search.
- `prediction_level_loso_results_by_seed.csv`: fold × seed × target metrics.
- `prediction_level_loso_seed_averaged_by_fold.csv`: three-seed fold averages
  for historical prediction-level fusion.
- `prediction_level_loso_mean_std_summary.csv`: historical five-fold mean/sample SD.
- `prediction_level_protocol.json`: compact machine-readable protocol record.

## Other paper-facing results

Branch/ablation/MSP files are lightweight paper-facing artifacts. Values marked or
described as paper-rounded were transcribed from the paper when only rounded
values were available. Raw datasets, features, checkpoints, and large
per-utterance prediction files are intentionally excluded.
