#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-${SER_GPU:-1}}"
SEEDS="${SEEDS:-42 123 2024}"

for seed in $SEEDS; do
  echo "======================================================"
  echo "Running five-fold LOSO for seed ${seed} on GPU ${GPU}"
  echo "======================================================"
  GPU="$GPU" SEED="$seed" bash loso/run_all_loso_folds.sh
done
