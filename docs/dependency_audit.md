# Dependency audit

The public repository contains only canonical preprocessing, training, evaluation,
ablation, cross-domain, and utility scripts.

## Core dependencies

`requirements.txt` contains the direct packages required by the IEMOCAP pipeline:

- NumPy
- pandas
- SciPy
- scikit-learn
- Matplotlib
- SoundFile
- librosa
- TensorFlow
- PyTorch
- Hugging Face Transformers

## Cross-domain extras

`requirements-cross-domain.txt` adds:

- Hugging Face `datasets`
- `tqdm`

Some scripts contain guarded CUDA-library discovery code for environments where
TensorFlow/PyTorch cannot locate NVIDIA wheel libraries automatically. These
branches are optional and do not introduce a mandatory dependency beyond the
installed CUDA/runtime stack.

## Version policy

The exact package lock from the original training server is not part of this
public release. Package versions are intentionally not fabricated.

The release instead focuses on portable code, documented evaluation protocols,
paper-facing result artifacts, and configurable dataset/output roots.

For maintainers who want additional environment metadata,
`utilities/capture_environment.sh` is retained as an optional helper. Its output
is not required for the paper-facing public release.
