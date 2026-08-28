# SER VAD Model-Level Audio-Text Fusion

Reproducibility code for:

**A Controlled Study of Model-Level Audio-Text Fusion for Dimensional Speech Emotion Recognition**

**Authors:** Nguyen The Cuong, Huynh Vinh Nam, and Kieu Quoc Viet

**Affiliation:** Department of Information and Communication Technology, University of Science and Technology of Hanoi (USTH)

**Repository:** https://github.com/Chill2004/SER-VAD-multi-modal

This repository contains preprocessing, fixed Session-5 experiments, ablations,
three-seed five-fold LOSO evaluation for model-level and prediction-level fusion,
cross-domain MSP-Podcast experiments, and statistical analyses used in the study.

## Main results

| Evaluation | Mean CCC |
|---|---:|
| Five-fold IEMOCAP LOSO, model-level fusion, 3 seeds/fold | **0.635 ± 0.011** |
| Five-fold IEMOCAP LOSO, prediction-level fusion, 3 seeds/fold | **0.613 ± 0.024** |
| Fixed Session-5, model-level fusion, 3 seeds | **0.625 ± 0.017** |
| Fixed Session-5, prediction-level fusion, 3 seeds | **0.594 ± 0.011** |
| Fixed Session-5 model-level minus prediction-level | **0.032**, 95% CI **[0.025, 0.039]** over test-set composition |
| IEMOCAP -> MSP-Podcast direct transfer | **0.437** |
| IEMOCAP -> MSP-Podcast fine-tuning | **0.648** |

For both five-fold LOSO fusion strategies, each held-out session is evaluated with
seeds `42`, `123`, and `2024` while keeping that fold's
train/validation/test partition fixed across seeds. Metrics are averaged across
the three seeds within each held-out session first. The reported LOSO deviation
is then the **sample standard deviation across the five seed-averaged held-out
sessions**.

For prediction-level LOSO, the audio mixing weight is selected separately for
Valence, Arousal, and Dominance within each fold from
`{0.00, 0.01, ..., 1.00}`. One weight per fold/target is chosen by maximizing the
**mean validation CCC across the three seeds**, then applied unchanged to the
three corresponding test predictions. Test labels are never used for weight
selection. Model-level fusion has the higher seed-averaged mean CCC on all five
held-out sessions; this does not imply that it is higher for every target on
every fold.

For the fixed Session-5 robustness study, the deviation is the **sample standard
deviation across seeds 42, 123, and 2024**. The paired bootstrap interval
quantifies test-set-composition uncertainty conditional on those trained model
instances, not random-seed uncertainty.

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

- [`docs/reproducibility.md`](docs/reproducibility.md) for the complete research flow;
- [`docs/command_reference.md`](docs/command_reference.md) for runnable commands;
- [`docs/paper_result_traceability.md`](docs/paper_result_traceability.md) to map paper results to scripts and result artifacts.

## Data

IEMOCAP and MSP-Podcast are **not redistributed** in this repository. Raw audio,
transcripts distributed with the corpora, extracted `.npy` features, and trained
model checkpoints are excluded. Obtain the datasets from their official sources
and follow their respective licenses and access terms.

## Reproducibility notes

The canonical code does not depend on the original server's absolute paths.
Dataset and output locations are controlled through `SER_DATA_ROOT` and
`SER_OUTPUT_ROOT`.

The public release focuses on portable code, documented protocols, and
paper-facing result artifacts. Dataset and output paths are configured through
SER_DATA_ROOT and SER_OUTPUT_ROOT; an exact training-server environment lock is
not part of this release.

## Results

`results/` contains small paper-facing artifacts for the repeated-seed model-level
LOSO and prediction-level LOSO comparisons, fixed Session-5 robustness,
bootstrap analysis, ablations, and MSP transfer. Large per-utterance features,
predictions, and model files are intentionally omitted.

## Citation

A `CITATION.cff` file is included for repository citation. Publication metadata
can be updated after the final conference record is available.

## License

No software license has been selected yet. Add an appropriate license before
public release if you intend to grant reuse or redistribution rights.
