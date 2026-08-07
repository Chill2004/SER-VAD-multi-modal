# Paper result traceability

| Result | Code | Artifact |
|---|---|---|
| Branch diagnostics | `experiments/fixed_session5/` | `results/branch_diagnostic_results.csv` |
| Prediction-level 3 seeds | `evaluation/build_prediction_level_3seeds.py` | `results/session5_three_seed_results.csv` |
| Model-level 3 seeds | `experiments/fixed_session5/run_model_level_multimodal_3seeds.py` | `results/session5_three_seed_results.csv` |
| Bootstrap | `evaluation/paired_bootstrap_model_vs_prediction.py` | `results/paired_bootstrap_results.csv` |
| LOSO | `loso/*checked.py` + summary | `results/loso_results.csv` |
| Ablation | `ablation/README.md` | `results/ablation_results.csv` |
| Valence cases | `evaluation/select_valence_case_examples.py` | `results/valence_case_examples.csv` |
| MSP | `experiments/cross_domain/` | `results/msp_transfer_results.csv` |
