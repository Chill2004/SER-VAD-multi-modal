# Command reference

Run commands from the repository root.

## Environment

```bash
cp config.example.sh config.sh
# Edit config.sh, then:
source config.sh
```

The original experiments used GPU 1 on the server. `SER_GPU` keeps that default configurable for scripts without a `--gpu` option. Scripts that expose `--gpu` can override it directly.

## IEMOCAP preprocessing

```bash
python preprocessing/build_iemocap_master_manifest.py \
  --dataset_root "$SER_DATA_ROOT" \
  --output_csv "$SER_OUTPUT_ROOT/iemocap_master.csv"

python preprocessing/prepare_fixed_session5_split.py \
  --manifest_csv "$SER_OUTPUT_ROOT/iemocap_master.csv" \
  --out_dir "$SER_OUTPUT_ROOT/fixed_split_vad" \
  --test_session Ses05

python preprocessing/build_audio_aug_manifests.py
python preprocessing/extract_wav2vec2_from_manifests.py
python preprocessing/extract_mfcc39_from_manifests.py
python preprocessing/build_transcript_manifest.py
python preprocessing/attach_transcripts_to_manifests.py
python preprocessing/extract_text_distilbert_from_manifests.py --setting baseline
```

The final multimodal setting uses **silence audio + baseline text**.

## Fixed Session-5 repeated-seed evaluation

```bash
python experiments/fixed_session5/run_audio_fusion_3seeds.py \
  --setting silence --seeds 42 123 2024 --gpu "$SER_GPU"

python experiments/fixed_session5/run_text_distilbert_3seeds.py \
  --setting baseline --seeds 42 123 2024 --gpu "$SER_GPU"

python experiments/fixed_session5/run_model_level_multimodal_3seeds.py \
  --setting silence --text_setting baseline \
  --seeds 42 123 2024 --gpu "$SER_GPU"

python evaluation/build_prediction_level_3seeds.py \
  --audio_setting silence --text_setting baseline \
  --seeds 42 123 2024

python evaluation/paired_bootstrap_model_vs_prediction.py \
  --seeds 42 123 2024 --n_bootstrap 10000 --bootstrap_seed 2026
```

## Five-fold LOSO

```bash
python loso/prepare_iemocap_loso_full_orig_silence_checked.py \
  --audio_setting silence \
  --text_setting baseline \
  --seed 42 --val_ratio 0.10 --overwrite

for fold in Ses01 Ses02 Ses03 Ses04 Ses05; do
  python loso/train_model_level_multimodal_loso_checked.py \
    --setting silence --text_setting baseline \
    --fold "$fold" --seed 42 --gpu "$SER_GPU" --audit_only
done

SER_GPU="$SER_GPU" GPU="$SER_GPU" SEED=42 bash loso/run_all_loso_folds.sh

python evaluation/summarize_model_level_loso.py --seed 42
```

Expected headline result: mean CCC **0.642 ± 0.017**, where the standard deviation is the sample SD across the five held-out sessions.

## Ablation entry points

See `ablation/README.md`. The same canonical trainers are reused for several ablations rather than duplicated.

## Cross-domain MSP-Podcast

Install the extras first:

```bash
pip install -r requirements-cross-domain.txt
```

The MSP scripts are grouped by protocol under `experiments/cross_domain/`. Their CLI surface is listed in `docs/cli_audit.csv`.
