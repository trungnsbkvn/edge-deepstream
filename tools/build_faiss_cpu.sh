#!/usr/bin/env bash
set -euo pipefail

# Build FAISS (CPU-only) Python bindings against the current Python and NumPy
# Requirements (install via apt): build-essential cmake swig python3-dev libopenblas-dev

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
FAISS_DIR="$ROOT_DIR/third_party/faiss"
BUILD_DIR="$ROOT_DIR/third_party/_faiss_build"

if [[ ! -d "$FAISS_DIR" ]]; then
  echo "FAISS source not found at $FAISS_DIR" >&2
  exit 1
fi

PYBIN="${PYBIN:-$(command -v python3)}"
echo "Using Python: $PYBIN"
"$PYBIN" - <<'PY'
import sys, numpy
print('Python:', sys.version)
print('NumPy :', numpy.__version__)
maj = int(numpy.__version__.split('.')[0])
if maj >= 2:
  raise SystemExit('ERROR: NumPy {} detected. Please use numpy==1.24.4 before building FAISS.'.format(numpy.__version__))
PY

mkdir -p "$BUILD_DIR"
echo "Configuring FAISS (CPU-only, Python bindings ON)" >&2
cmake -B "$BUILD_DIR" -S "$FAISS_DIR" \
  -DFAISS_ENABLE_GPU=OFF \
  -DFAISS_ENABLE_PYTHON=ON \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE="$PYBIN"

echo "Building swigfaiss (Python module)" >&2
make -C "$BUILD_DIR" -j "$(nproc)" swigfaiss

echo "Installing Python package from build tree" >&2
cd "$BUILD_DIR/faiss/python"
"$PYBIN" -m pip install --user .

echo "Verifying import" >&2
"$PYBIN" - <<'PY'
import faiss, numpy as np
print('faiss imported ok')
print('faiss has GPUs API:', hasattr(faiss, 'get_num_gpus'))
idx = faiss.IndexFlatL2(4)
idx.add(np.zeros((1,4), dtype='float32'))
print('index size:', idx.ntotal)
PY

echo "Done."
