# Paper result traceability

| Result | Code | Artifact |
|---|---|---|
| Branch diagnostics | `experiments/fixed_session5/` | `results/branch_diagnostic_results.csv` |
| Prediction-level 3 seeds | `evaluation/build_prediction_level_3seeds.py` | `results/session5_three_seed_results.csv` |
| Model-level 3 seeds | `experiments/fixed_session5/run_model_level_multimodal_3seeds.py` | `results/session5_three_seed_results.csv` |
| Bootstrap | `evaluation/paired_bootstrap_model_vs_prediction.py` | `results/paired_bootstrap_results.csv` |
| Repeated-seed LOSO, all 15 runs | `loso/train_model_level_multimodal_loso_checked.py` | `results/loso_results_by_seed.csv` |
| Repeated-seed LOSO, Table V folds | `evaluation/summarize_repeated_seed_loso.py` | `results/loso_seed_averaged_by_fold.csv` |
| Repeated-seed LOSO, final mean/std | `evaluation/summarize_repeated_seed_loso.py` | `results/loso_repeated_seed_summary.csv` |
| Historical seed-42 LOSO | `evaluation/summarize_model_level_loso.py --seed 42` | `results/loso_seed42_results.csv`, `results/loso_seed42_mean_std_summary.csv` |
| Ablation | `ablation/README.md` | `results/ablation_results.csv` |
| Valence cases | `evaluation/select_valence_case_examples.py` | `results/valence_case_examples.csv` |
| MSP | `experiments/cross_domain/` | `results/msp_transfer_results.csv` |

## LOSO aggregation note

The primary paper statistic averages seeds `42`, `123`, and `2024` within each
held-out IEMOCAP session first. Mean and sample standard deviation are then
computed across the five seed-averaged session results. The historical
seed-42-only result is retained for provenance but is not the final headline.
