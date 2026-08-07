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

The exact `pip freeze` from the original training server was not present in the
source archive. Package versions are therefore intentionally not fabricated.

Before the repository is published, capture the original environment:

```bash
python --version > environment/python-version.txt
python -m pip freeze > environment/pip-freeze.txt
nvidia-smi > environment/nvidia-smi.txt
```

If Conda was used, also save:

```bash
conda env export --no-builds > environment/conda-environment.yml
```
