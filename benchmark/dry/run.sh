#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
MINISGL_VENV="/data1/sandesh/.venvs/mini-sglang"
MINISGL_CUDA_ROOT="/data1/sandesh/.conda/envs/vllm092clean"
MINISGL_PYTHON_INCLUDE="${MINISGL_PYTHON_INCLUDE:-/data1/sandesh/.conda/pkgs/python-3.12.12-hd63d673_1_cpython/include/python3.12}"

export CUDA_HOME="${MINISGL_CUDA_ROOT}"
export PATH="${CUDA_HOME}/bin:${PATH:-/usr/bin:/bin}"
export CC="${CUDA_HOME}/bin/x86_64-conda-linux-gnu-gcc"
export CXX="${CUDA_HOME}/bin/x86_64-conda-linux-gnu-c++"
export NVCC_PREPEND_FLAGS="-ccbin=${CXX}"
export CPATH="${MINISGL_PYTHON_INCLUDE}${CPATH:+:${CPATH}}"
export LIBRARY_PATH="${CUDA_HOME}/lib/stubs:${CUDA_HOME}/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export LD_PRELOAD="${CUDA_HOME}/lib/libstdc++.so.6${LD_PRELOAD:+:${LD_PRELOAD}}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export PYTHONPATH="${REPO_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${REPO_ROOT}/benchmark-results/.matplotlib}"

cd "${REPO_ROOT}"
exec "${MINISGL_VENV}/bin/python" "${SCRIPT_DIR}/bench.py" "$@"
