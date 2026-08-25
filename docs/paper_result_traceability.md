# Paper result traceability

| Result | Code | Artifact |
|---|---|---|
| Branch diagnostics | `experiments/fixed_session5/` | `results/branch_diagnostic_results.csv` |
| Fixed Session-5 prediction-level, 3 seeds | `evaluation/build_prediction_level_3seeds.py` | `results/session5_three_seed_results.csv` |
| Fixed Session-5 model-level, 3 seeds | `experiments/fixed_session5/run_model_level_multimodal_3seeds.py` | `results/session5_three_seed_results.csv` |
| Fixed Session-5 bootstrap | `evaluation/paired_bootstrap_model_vs_prediction.py` | `results/paired_bootstrap_results.csv` |
| Model-level repeated-seed LOSO, all 15 runs | `loso/train_model_level_multimodal_loso_checked.py` | `results/loso_results_by_seed.csv` |
| Model-level LOSO, Table V fold rows | `evaluation/summarize_repeated_seed_loso.py` | `results/loso_seed_averaged_by_fold.csv` |
| Model-level LOSO, final mean/std | `evaluation/summarize_repeated_seed_loso.py` | `results/loso_repeated_seed_summary.csv` |
| Prediction-level LOSO audio branch | `loso/train_audio_fusion_loso_checked.py` | per-run `fusion_val_predictions.csv`, `fusion_predictions.csv` (not redistributed) |
| Prediction-level LOSO text branch | `loso/train_text_distilbert_loso_checked.py` | per-run `text_embedding_val_predictions.csv`, `text_embedding_predictions.csv` (not redistributed) |
| Prediction-level LOSO alpha selection | `evaluation/evaluate_prediction_level_loso_3seeds.py` | `results/prediction_level_alpha_by_fold.csv`, `results/prediction_level_alpha_search.csv` |
| Prediction-level LOSO, Table V fold rows | `evaluation/evaluate_prediction_level_loso_3seeds.py` | `results/prediction_level_loso_seed_averaged_by_fold.csv` |
| Prediction-level LOSO, all seed/target metrics | `evaluation/evaluate_prediction_level_loso_3seeds.py` | `results/prediction_level_loso_results_by_seed.csv` |
| Prediction-level LOSO, final mean/std | `evaluation/evaluate_prediction_level_loso_3seeds.py` | `results/prediction_level_loso_mean_std_summary.csv` |
| Historical seed-42 model-level LOSO | `evaluation/summarize_model_level_loso.py --seed 42` | `results/loso_seed42_results.csv`, `results/loso_seed42_mean_std_summary.csv` |
| Ablation | `ablation/README.md` | `results/ablation_results.csv` |
| Valence cases | `evaluation/select_valence_case_examples.py` | `results/valence_case_examples.csv` |
| MSP | `experiments/cross_domain/` | `results/msp_transfer_results.csv` |

## LOSO aggregation note

Both paper-facing LOSO summaries average seeds `42`, `123`, and `2024` within
each held-out IEMOCAP session first. Mean and sample standard deviation are then
computed across the five seed-averaged session results.

For prediction-level fusion, alpha is selected **within each fold and target**
from validation predictions only, by maximizing mean validation CCC across the
three seeds. The same selected alpha is then used for the three corresponding
test predictions. Test labels are never used for selection.

Final five-fold mean CCC:

- model-level fusion: **0.635 ± 0.011**;
- prediction-level fusion: **0.613 ± 0.024**.

The historical seed-42-only model-level result is retained for provenance but is
not the final headline.
