# Training environment

The public release does not require an exact lock of the original training-server
environment.

Dataset and output locations are configured through `SER_DATA_ROOT` and
`SER_OUTPUT_ROOT`. GPU selection for the documented runners is configured through
`SER_GPU` or an explicit `--gpu` argument where supported.

`utilities/capture_environment.sh` is retained only as an optional helper for
maintainers who want to record Python, package, CUDA, and GPU metadata.

Generated environment-capture files are not required for the paper-facing release.
