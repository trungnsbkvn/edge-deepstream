#!/usr/bin/env bash
set -euo pipefail

# Build FAISS (GPU, Python bindings) on Jetson Xavier NX (aarch64, CUDA on JetPack)
#
# Prereqs (apt):
#   sudo apt-get update && sudo apt-get install -y \
#     build-essential cmake swig python3-dev libopenblas-dev
#
# Assumptions:
#   - CUDA from JetPack is installed at /usr/local/cuda (override with CUDA_HOME)
#   - Repo has FAISS source under third_party/faiss
#   - Python uses NumPy 1.x (this project pins to 1.23.5/1.24.x). NumPy 2.x is NOT supported.
#
# Notes:
#   - Xavier NX GPU arch is SM=72. You can override with SM env var, e.g. SM=87 for Orin.
#   - This builds and installs the Python module from the build tree via pip --user.
#   - To rebuild cleanly: delete third_party/faiss/build or run with CLEAN=1.

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
FAISS_DIR="$ROOT_DIR/third_party/faiss"
BUILD_DIR="$ROOT_DIR/third_party/_faiss_build"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
SM_ARCH="${SM:-72}"
PYBIN="${PYBIN:-$(command -v python3)}"

if [[ ! -d "$FAISS_DIR" ]]; then
  echo "FAISS source not found at $FAISS_DIR" >&2
  exit 1
fi

if [[ ! -x "$PYBIN" ]]; then
  echo "Python not found. Set PYBIN to your python3 executable." >&2
  exit 1
fi

echo "Using Python: $PYBIN"
"$PYBIN" - <<'PY'
import sys, numpy
print('Python:', sys.version)
print('NumPy :', numpy.__version__)
maj = int(numpy.__version__.split('.')[0])
if maj >= 2:
  raise SystemExit('ERROR: NumPy {} detected. Please use numpy==1.24.4 (or 1.23.5) before building FAISS.'.format(numpy.__version__))
PY

# CUDA toolchain presence
if [[ ! -d "$CUDA_HOME" ]]; then
  echo "CUDA not found at $CUDA_HOME. Set CUDA_HOME to your CUDA root." >&2
  exit 1
fi
if ! command -v nvcc >/dev/null 2>&1; then
  export PATH="$CUDA_HOME/bin:$PATH"
fi
if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc not found in PATH. Ensure CUDA is installed and PATH includes $CUDA_HOME/bin" >&2
  exit 1
fi

echo "CUDA: $(nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p' | head -n1) at $CUDA_HOME"
echo "Target GPU SM arch: $SM_ARCH"

if [[ "${CLEAN:-0}" == "1" ]]; then
  rm -rf "$BUILD_DIR"
fi

# Configure
mkdir -p "$BUILD_DIR"
echo "Configuring FAISS (GPU=ON, Python=ON)" >&2
cmake -B "$BUILD_DIR" -S "$FAISS_DIR" \
  -DFAISS_ENABLE_GPU=ON \
  -DFAISS_ENABLE_PYTHON=ON \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE="$PYBIN" \
  -DCUDAToolkit_ROOT="$CUDA_HOME" \
  -DCMAKE_CUDA_ARCHITECTURES="$SM_ARCH"

# Build Python bindings (swigfaiss) and core libs
echo "Building swigfaiss (Python module)" >&2
make -C "$BUILD_DIR" -j "$(nproc)" swigfaiss

# Install Python module from build tree
echo "Installing Python package from build tree" >&2
cd "$BUILD_DIR/faiss/python"
"$PYBIN" -m pip install --user .

# Verify GPU API and a simple GPU index roundtrip
echo "Verifying FAISS GPU import and GPU index" >&2
"$PYBIN" - <<'PY'
import numpy as np
import faiss
print('faiss imported ok')
print('has GPU API:', hasattr(faiss, 'StandardGpuResources'))
ng = None
try:
    ng = faiss.get_num_gpus()
except Exception:
    ng = None
print('num_gpus:', ng)
if hasattr(faiss, 'StandardGpuResources') and (ng or 0) > 0:
    res = faiss.StandardGpuResources()
    d = 128
    cpu = faiss.IndexFlatL2(d)
    gpu = faiss.index_cpu_to_gpu(res, 0, cpu)
    x = np.random.RandomState(0).randn(10, d).astype('float32')
    gpu.add(x)
    D, I = gpu.search(x[:1], 1)
    print('gpu_index_ok:', D.shape, I.shape, 'ntotal=', gpu.ntotal)
else:
    print('gpu_index_ok: skipped (no GPU runtime)')
PY

echo "Done."
