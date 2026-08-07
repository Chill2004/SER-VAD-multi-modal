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
- [x] Add reproducibility and command documentation.
- [x] Remove legacy/copy/notebook code from the public package.
- [x] Scan the public package for obvious secrets and hard-coded personal server paths.

## Before making the repository public

- [ ] Run `bash utilities/capture_environment.sh` on the original training server.
- [ ] Commit the resulting environment files.
- [ ] Create the public GitHub repository.
- [ ] Add the final repository URL to `README.md`, `CITATION.cff`, and the paper.
- [ ] Select a software license if reuse/redistribution is intended.
- [ ] Run the LOSO manifest audit once from a clean checkout.
- [ ] Confirm the commands in `docs/command_reference.md` against the final directory layout.
