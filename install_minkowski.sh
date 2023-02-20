#!/bin/bash
pip3 install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--force_cuda" --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas"
