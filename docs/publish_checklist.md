# Pre-publish checklist

## Completed

- [x] Organize scripts by research stage.
- [x] Keep preprocessing code required for Wav2Vec2, MFCC, and DistilBERT features.
- [x] Separate fixed Session-5, LOSO, evaluation, ablation, and cross-domain code.
- [x] Rewrite canonical comments in concise English.
- [x] Remove original absolute server paths from canonical scripts.
- [x] Audit canonical Python syntax.
- [x] Audit direct dependencies and CLI entry points.
- [x] Add paper-facing result CSVs.
- [x] Add three-seed model-level LOSO aggregation artifacts and commands.
- [x] Add full five-fold prediction-level LOSO evaluator and paper-facing artifacts.
- [x] Document fold-specific validation selection for prediction-level alpha.
- [x] Add reproducibility and command documentation for both LOSO fusion strategies.
- [x] Apply the local repeated-seed patch to a clean checkout.
- [x] Remove legacy/copy/notebook code from the public package.
- [x] Scan the public package for obvious secrets and hard-coded personal server paths.
- [x] Create the public GitHub repository.
- [x] Add the repository URL to `README.md`, `CITATION.cff`, and the paper.
- [x] Synchronize README/CITATION author metadata with all three paper authors.
- [x] Keep the public release portable through SER_DATA_ROOT and SER_OUTPUT_ROOT; an exact training-server environment lock is outside the scope of this release.

## Remaining before final release/submission

- [ ] Select a software license if reuse/redistribution is intended.
- [ ] Run the LOSO manifest audit once from the clean final checkout.
- [x] Confirm all commands in `docs/command_reference.md` against the merged final directory layout.
- [x] Re-run the repository validation scan after the final files are merged.
