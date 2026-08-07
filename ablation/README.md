# Ablation

| Paper ablation | Canonical script |
|---|---|
| Raw waveform | `ablation/train_raw_waveform_cnn1d.py` |
| Wav2Vec2 settings | `experiments/fixed_session5/train_wav2vec2_bilstm.py` |
| MFCC-39d | `experiments/fixed_session5/train_mfcc_cnn1d.py --feature_dir mfcc_features` |
| MFCC-117d | `preprocessing/extract_mfcc117_from_manifests.py` then the same CNN1D trainer with `--feature_dir mfcc39_delta_delta2_features` |
| Audio representation fusion | `experiments/fixed_session5/train_audio_fusion_bilstm_cnn1d.py` |

`fused_augmentation/` contains additional thesis diagnostics.
