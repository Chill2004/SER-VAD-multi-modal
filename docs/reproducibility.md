# Reproducibility flow

## Step 0: configure the environment

Configure dataset, output, and GPU paths for the current machine:

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
7. run the fixed Session-5 three-seed robustness comparison and paired bootstrap;
8. build and audit all five LOSO folds once, keeping each fold's
   train/validation/test partition fixed across seeds;
9. train the model-level LOSO system for every fold with seeds `42`, `123`, and
   `2024`;
10. train the fused-audio and text LOSO branches for the prediction-level
    comparison using the same five folds and three seeds;
11. within each fold and V/A/D target, select one prediction-level audio weight
    from `0.00, 0.01, ..., 1.00` by maximizing mean validation CCC across the
    three seeds; apply that same fold-specific weight to the three test
    predictions;
12. for each fusion strategy, average metrics across the three seeds within each
    held-out session, then compute the mean and sample SD across the five
    seed-averaged sessions;
13. run case analyses and the MSP-Podcast protocols.

Test labels are never used for checkpoint, setting, or prediction-level weight
selection.

The primary five-fold paper comparison is:

- **model-level fusion: 0.635 Â± 0.011 mean CCC**;
- **prediction-level fusion: 0.613 Â± 0.024 mean CCC**.

These are **not** pooled 15-run standard deviations. Each standard deviation is
computed across five seed-averaged held-out-session values.

The prediction-level fold means are `.649/.605/.590/.624/.594`; the model-level
fold means are `.650/.635/.623/.642/.625`. Model-level fusion therefore has the
higher seed-averaged mean CCC on all five held-out sessions, while individual
target-level exceptions remain possible.

Exact runnable commands are in `command_reference.md`.

## Data policy

IEMOCAP and MSP-Podcast are not redistributed. Large `.wav`, `.npy`, model
checkpoint, and per-utterance intermediate/prediction files are also excluded
from the repository.

## Environment reproducibility

`requirements.txt` lists the direct core dependencies. The public release is
configured through SER_DATA_ROOT and SER_OUTPUT_ROOT and does not include an
exact lock of the original training-server environment.
