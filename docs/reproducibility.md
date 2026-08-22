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
7. run the fixed Session-5 three-seed robustness comparison;
8. build and audit all five LOSO folds once, keeping each fold's
   train/validation/test partition fixed across seeds;
9. train every LOSO fold with seeds `42`, `123`, and `2024`;
10. aggregate the three seeds within each held-out session and then compute the
    mean and sample SD across the five seed-averaged sessions;
11. run prediction-level/bootstrap/case analyses and the MSP-Podcast protocols.

The primary LOSO paper statistic is therefore **not** a pooled 15-run standard
deviation and is not the old seed-42-only summary. The final headline is
**0.635 ± 0.011 mean CCC**.

Exact runnable commands are in `command_reference.md`.

## Data policy

IEMOCAP and MSP-Podcast are not redistributed. Large `.wav`, `.npy`, model
checkpoint, and per-utterance intermediate files are also excluded from the
repository.

## Environment reproducibility

`requirements.txt` lists direct core dependencies, but the exact original version
lock was not included in the source archive. Do not infer package versions from
current releases. Capture the actual training environment on the original server
before final publication; see `dependency_audit.md`.
