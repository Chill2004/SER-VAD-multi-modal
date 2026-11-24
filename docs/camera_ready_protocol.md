# RIVF 2026 camera-ready evaluation

## Primary IEMOCAP LOSO

Each fold holds out one complete session for test and a different complete
session for validation. The remaining sessions are used for training. See the
[fold settings and counts](../results/camera_ready/iemocap_loso_protocol.csv).
The acoustic condition is chosen from baseline, noise, reverberation, and
silence by full-validation mean VAD CCC averaged across seeds 42, 123, and
2024. Augmentation is applied to training only.

Scalar fusion chooses an audio weight per fold and target over 0 to 1 in
steps of .01 by mean validation CCC across seeds. The linear combiner uses
standardized audio and text predictions, an intercept, and L2-regularized
slopes. The regularization strength is selected by dialogue-group
cross-validation within validation. Checkpoints and fusion settings are locked
before outer-test scoring.

The paired analysis resamples whole dialogues within each observed test session
in 10,000 draws, sharing each draw across methods and seeds. It recomputes CCC
before averaging paired differences across targets, seeds, and sessions.
Intervals are conditional on the trained models and observed sessions. The
[camera-ready result index](../results/README.md) gives the released summaries.
The camera-ready locked evaluator, linear combiner implementation, and primary
LOSO clustered-bootstrap runner are pending release. The older LOSO and
utterance-level bootstrap commands document historical experiments.

## Fixed Session 5

For the dialogue-clustered comparison, use the archived fixed-Ses05 prediction
CSVs for the three seeds. The model root contains
`silence_audio_baseline_text_seed<seed>/model_level_predictions.csv`. The
scalar root contains `seed<seed>/prediction_level_predictions.csv`.

```bash
python evaluation/run_fixed_ses05_clustered_ci.py \
  --model-root "$MODEL_ROOT" --prediction-root "$PREDICTION_ROOT" \
  --draws 10000 --seed 2026 --output "$SER_OUTPUT_ROOT/fixed_ses05_ci.json"
```

## Fixed-silence transcript diagnostics

Across five sessions and three seeds, the reference, shuffle, and ASR
conditions share checkpoints, acoustic features, and labels. The shuffle
analysis uses ten no-self-match permutations per session shared across seeds.
ASR uses Wav2Vec2 Base 960h with greedy CTC decoding. The frozen DistilBERT
encoding procedure is shared across the transcript conditions. The released
aggregate CSVs contain the results by fold, seed, and target. The runners
require the archived checkpoints and their locked evaluator, which are
outside the current public package.

## New MSP-Podcast 2–6 s baselines

The seed-42 comparison uses the Hugging Face snapshot
`AbstractTTS/PODCAST@e791939e368b6c26595ca7f2c75aa29ebd92e776`. Utterances are
ranked by SHA-256 of the seed and source utterance ID after filtering, with
7,082/787/2,170 train/validation/test rows. Speaker and recording overlap were
not measured. Direct transfer and fine-tuning in the paper were run on the
historical split.

Set `MSP_CACHE`, `MSP_SPLIT`, `MSP_INPUTS`, `MSP_FEATURES`, and `MSP_RUN`
to absolute local paths and `GPU` to the desired GPU index. Use
`requirements-cross-domain.txt` for the packages below. The preparation stage
downloads the pinned snapshot to `MSP_CACHE` if needed.

```bash
set -e
MSP_CODE=experiments/cross_domain
python "$MSP_CODE/msp_duration_matched/prepare_g7_msp_split.py" \
  --output "$MSP_SPLIT" --cache "$MSP_CACHE"
python "$MSP_CODE/msp_duration_matched/prepare_g7_msp_inputs.py" \
  --split "$MSP_SPLIT" materialize \
  --snapshot "$MSP_CACHE/datasets--AbstractTTS--PODCAST/snapshots/e791939e368b6c26595ca7f2c75aa29ebd92e776" \
  --output "$MSP_INPUTS"
CUDA_VISIBLE_DEVICES="$GPU" python "$MSP_CODE/msp_7to10/extract_msp_features.py" \
  --dataset_dir "$MSP_INPUTS" --out_root "$MSP_FEATURES" --feature all
python "$MSP_CODE/msp_duration_matched/prepare_g7_msp_inputs.py" \
  --split "$MSP_SPLIT" audit --features "$MSP_FEATURES"

python "$MSP_CODE/msp_full/train_text_distilbert.py" \
  --feature_root "$MSP_FEATURES/text_distilbert_features" \
  --out_dir "$MSP_RUN/text" --gpu "$GPU" --epochs 40
python "$MSP_CODE/msp_full/train_audio_fusion.py" \
  --wav2vec2_root "$MSP_FEATURES/wav2vec2_features" \
  --mfcc_root "$MSP_FEATURES/mfcc_features" \
  --out_dir "$MSP_RUN/audio" --gpu "$GPU" --epochs 40
python "$MSP_CODE/msp_duration_matched/score_g7_prediction_fusion.py" \
  --manifest "$MSP_SPLIT/manifest.csv" --run "$MSP_RUN"
```

Preparation and scoring verify the published manifest SHA-256. Use the locked
manifest or reproduce it with the pinned snapshot and the same preprocessing.
Fusion weights are chosen on validation. The test summary is saved to
`$MSP_RUN/prediction_level/test_ccc.csv`.
