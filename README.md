# SER VAD Model-Level Audio-Text Fusion

Reproducibility code for:

**A Controlled Study of Model-Level Audio-Text Fusion for Dimensional Speech Emotion Recognition**

**Authors:** Nguyen The Cuong, Huynh Vinh Nam, and Kieu Quoc Viet

**Affiliation:** Department of Information and Communication Technology, University of Science and Technology of Hanoi, Vietnam Academy of Science and Technology, 18 Hoang Quoc Viet, Nghia Do, Hanoi, Vietnam

**Repository:** https://github.com/Chill2004/SER-VAD-multi-modal

This repository contains preprocessing and evaluation code, historical
experiments, and result tables for the RIVF 2026 camera-ready paper.

## Camera-ready results

The primary IEMOCAP experiment holds out one full session for testing and one
other full session for validation in each fold. It uses seeds `42`, `123`,
and `2024`. Mean CCC and sample SD are computed across the five fold means,
after averaging the three seeds within each fold.

| Method | Valence CCC | Arousal CCC | Dominance CCC | Mean CCC ± SD |
|---|---:|---:|---:|---:|
| Audio only | 0.493 | 0.642 | 0.517 | 0.551 ± 0.042 |
| Text only | 0.610 | 0.465 | 0.444 | 0.506 ± 0.034 |
| Scalar prediction fusion | 0.628 | 0.643 | 0.527 | 0.599 ± 0.022 |
| Linear combiner | 0.597 | 0.612 | 0.468 | 0.559 ± 0.026 |
| Model-level fusion | **0.657** | **0.661** | **0.540** | **0.619 ± 0.016** |

- **Primary LOSO.** The paired mean CCC gain from model-level over scalar
  fusion is **0.020** (dialogue-clustered 95% CI **[0.014, 0.027]**).
- **Fixed Session 5.** Mean CCC is 0.625 for model-level fusion and 0.594 for
  scalar fusion. The paired gain is 0.0315 (dialogue-clustered 95% CI
  [0.0159, 0.0450]). Both intervals are conditional on the trained models
  and observed sessions.
- **Lightweight MFCC ablation.** Retaining MFCCs raises mean CCC from 0.4396
  to 0.5359 in the matched audio-only configuration.
- **Fixed-silence transcript diagnostics.** Across 15 checkpoints, mean CCC
  is 0.6350 with reference transcripts, 0.4494 after shuffling, and 0.5780
  with ASR transcripts. Pooled WER is 30.05%.
- **MSP-Podcast.** On the historical 2–6 s test, fine-tuning raises mean CCC
  from 0.437 to 0.648. On the new 2–6 s split (seed 42, 2,170 test samples),
  mean CCC is 0.5169 for audio only, 0.2799 for text only, and 0.5615 for
  scalar fusion. The new split has no completed model-level comparison.

See [the result index](results/README.md) for camera-ready CSVs and historical
artifacts, and [the protocol notes](docs/camera_ready_protocol.md) for the
evaluation settings and available reproduction commands.

## Repository structure

```text
preprocessing/               IEMOCAP manifests, augmentation, Wav2Vec2/MFCC/DistilBERT features
experiments/
  fixed_session5/            branch, audio-fusion, text, and model-level experiments
  cross_domain/              MSP-Podcast in-domain, transfer, and fine-tuning experiments
ablation/                    paper/thesis ablation entry points
loso/                        audited leave-one-session-out pipeline and seed runners
evaluation/                  LOSO aggregation, prediction fusion, bootstrap, case analysis
results/                     lightweight paper-facing CSV/JSON results
utilities/                   supporting visualization/recovery scripts
docs/                        reproducibility, commands, dependencies, result traceability
environment/                 optional environment metadata notes
```

## Model overview

The final model combines three utterance-level branches:

- **Wav2Vec2 Base 960h** frozen sequence features (`T x 768`) -> two BiLSTM layers -> 128-D;
- **MFCC + delta + delta-delta** (`T x 39`) -> residual/dilated CNN1D -> 128-D;
- **DistilBERT** masked mean-pooled embedding (`768-D`) -> dense text branch -> 64-D.

The three representations are concatenated into a 320-D vector and mapped to
Valence, Arousal, and Dominance with a joint model-level fusion head.

## Quick start

Create a local configuration:

```bash
cp config.example.sh config.sh
# Edit config.sh
source config.sh
```

Install the core dependencies:

```bash
python -m pip install -r requirements.txt
```

For MSP-Podcast experiments:

```bash
python -m pip install -r requirements-cross-domain.txt
```

Then follow:

- [Camera-ready protocol](docs/camera_ready_protocol.md) for the primary comparison,
  fixed-Session-5 dialogue bootstrap, and new MSP baselines.
- [Result traceability](docs/paper_result_traceability.md) for the result-to-code map.
- [Historical commands](docs/command_reference.md) for earlier experiments.

## Data

IEMOCAP and MSP-Podcast are **not redistributed** in this repository. Raw audio,
transcripts distributed with the corpora, extracted `.npy` features, and trained
model checkpoints are excluded. Obtain the datasets from their official sources
and follow their respective licenses and access terms.

## Reproducibility notes

The historical trainers use `SER_DATA_ROOT` and `SER_OUTPUT_ROOT` for dataset
and output paths. The camera-ready entry points published here accept explicit
paths. See the protocol notes for the scripts and inputs needed to reproduce each
result.

## Citation

A `CITATION.cff` file is included for repository citation. Publication metadata
can be updated after the final conference record is available.

## License

No software license has been selected yet. Add an appropriate license before
public release if you intend to grant reuse or redistribution rights.
