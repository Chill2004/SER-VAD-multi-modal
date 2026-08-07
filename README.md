# SER VAD Model-Level Audio-Text Fusion

Reproducibility code for:

**A Controlled Study of Model-Level Audio-Text Fusion for Dimensional Speech Emotion Recognition**

**Authors:** Nguyen The Cuong and Huynh Vinh Nam  
**Affiliation:** Department of Information and Communication Technology, University of Science and Technology of Hanoi (USTH)

This repository contains the preprocessing, fixed Session-5 experiments, ablations,
five-fold LOSO evaluation, cross-domain MSP-Podcast experiments, and statistical
analyses used in the study.

## Main results

| Evaluation | Mean CCC |
|---|---:|
| Five-fold IEMOCAP LOSO | **0.642 ± 0.017** |
| Fixed Session-5, model-level fusion, 3 seeds | **0.625 ± 0.017** |
| Fixed Session-5, prediction-level fusion, 3 seeds | **0.594 ± 0.011** |
| Model-level minus prediction-level | **0.032**, 95% CI **[0.025, 0.039]** |
| IEMOCAP -> MSP-Podcast direct transfer | **0.437** |
| IEMOCAP -> MSP-Podcast fine-tuning | **0.648** |

For LOSO, the deviation is the **sample standard deviation across the five held-out sessions**.
For the fixed Session-5 robustness study, the deviation is the **sample standard deviation
across seeds 42, 123, and 2024**.

## Repository structure

```text
preprocessing/               IEMOCAP manifests, augmentation, Wav2Vec2/MFCC/DistilBERT features
experiments/
  fixed_session5/            branch, audio-fusion, text, and model-level experiments
  cross_domain/              MSP-Podcast in-domain, transfer, and fine-tuning experiments
ablation/                    paper/thesis ablation entry points
loso/                        audited five-fold leave-one-session-out pipeline
evaluation/                  LOSO summary, prediction fusion, bootstrap, case analysis
results/                     lightweight paper-facing CSV results
utilities/                   supporting visualization/recovery scripts
docs/                        reproducibility, commands, dependencies, result traceability
environment/                 instructions for capturing the exact training environment
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
- [`docs/paper_result_traceability.md`](docs/paper_result_traceability.md) to map paper results to scripts and CSV artifacts.

## Data

IEMOCAP and MSP-Podcast are **not redistributed** in this repository. Raw audio,
transcripts distributed with the corpora, extracted `.npy` features, and trained
model checkpoints are excluded. Obtain the datasets from their official sources
and follow their respective licenses and access terms.

## Reproducibility notes

The canonical code no longer depends on the original server's absolute paths.
Dataset and output locations are controlled through `SER_DATA_ROOT` and
`SER_OUTPUT_ROOT`.

The exact package-version lock from the original training server is not included
yet. The repository therefore keeps dependency files unpinned rather than
inventing versions. See [`environment/README.md`](environment/README.md) for the
commands used to capture the original Python/CUDA environment before release.

## Results

`results/` contains small CSV artifacts corresponding to the values reported in
the paper. Large per-utterance features and model files are intentionally omitted.

## Citation

A `CITATION.cff` file is included for repository citation. Update the repository
URL and publication metadata after the public GitHub repository and final paper
record are available.

## License

No software license has been selected yet. Add an appropriate license before
public release if you intend to grant reuse or redistribution rights.
