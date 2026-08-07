# Reproducibility flow

## Step 0: capture/configure the environment

For publication, first capture the exact original training environment on the
server used for the reported experiments:

```bash
bash utilities/capture_environment.sh
```

For a new machine, configure paths:

```bash
cp config.example.sh config.sh
# Edit config.sh
source config.sh
```

The canonical pipeline is:

1. configure dataset/output/GPU roots;
2. build the IEMOCAP master manifest;
3. create the fixed Session-5 development/test split;
4. create training-only acoustic augmentation manifests;
5. extract Wav2Vec2, MFCC-39d, and baseline DistilBERT features;
6. run fixed Session-5 branch/fusion experiments;
7. run the three-seed robustness comparison;
8. build and audit all five LOSO folds;
9. train all LOSO folds with seed 42;
10. aggregate LOSO and run prediction-level/bootstrap/case analyses.

Exact runnable commands are in `command_reference.md`.

## Data policy

IEMOCAP and MSP-Podcast are not redistributed. Large `.wav`, `.npy`, model checkpoint, and per-utterance intermediate files are also excluded from the repository.

## Environment reproducibility

`requirements.txt` lists direct core dependencies, but the exact original version lock was not included in the source archive. Do not infer package versions from current releases. Capture the actual training environment on the original server before final publication; see `dependency_audit.md`.