#!/usr/bin/env bash
set -euo pipefail
GPU="${GPU:-1}"
SEED="${SEED:-42}"
for fold in Ses01 Ses02 Ses03 Ses04 Ses05; do
  python -u loso/train_model_level_multimodal_loso_checked.py --setting silence --text_setting baseline --fold "$fold" --seed "$SEED" --gpu "$GPU"
done
