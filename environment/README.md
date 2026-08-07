# Training environment

The exact environment should be captured from the machine used for the reported experiments.

From the repository root:

```bash
mkdir -p environment

python --version > environment/python-version.txt
python -m pip freeze > environment/pip-freeze.txt
nvidia-smi > environment/nvidia-smi.txt
```

If Conda was used:

```bash
conda env export --no-builds > environment/conda-environment.yml
```

These files should be committed once they are collected from the original server.
Do not generate a fake lock file from a different machine.
