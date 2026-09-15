#!/usr/bin/env bash
set -euo pipefail

python_bin="${GATRANS_PYTHON:-/root/miniconda3/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  echo "Required interpreter is missing or not executable: $python_bin" >&2
  exit 2
fi

python_prefix="$($python_bin -c 'import sys; print(sys.prefix)')"
if [[ "$python_prefix" != "/root/miniconda3" ]]; then
  echo "Refusing to modify $python_prefix; this repair is restricted to /root/miniconda3." >&2
  exit 2
fi

export TMPDIR=/tmp
unset PYTHONPATH PYTHONHOME

# Reinstall NumPy first, then binary consumers. This repairs stale wheels built or
# installed against a different NumPy C ABI. No --target or data-disk path is used.
"$python_bin" -m pip install --no-cache-dir --force-reinstall "numpy==2.1.3"
"$python_bin" -m pip install --no-cache-dir --force-reinstall --no-deps \
  "scipy==1.14.1" "scikit-learn==1.5.2" "pandas==2.2.2" "h5py==3.16.0"

# The base image may contain torchvision from a newer CUDA/PyTorch release.
# Reinstall the official matching 2.4.1/CUDA 12.1 family so torch.onnx and
# compiled torchvision operators come from one release.
"$python_bin" -m pip uninstall --yes torch torchvision torchaudio
"$python_bin" -m pip install --no-cache-dir \
  "torch==2.4.1" "torchvision==0.19.1" "torchaudio==2.4.1" \
  --index-url https://download.pytorch.org/whl/cu121

"$python_bin" -c \
  'import numpy, scipy, sklearn, pandas, h5py, torch, torch.onnx, torchvision, shap; print("environment OK", numpy.__version__, scipy.__version__, torch.__version__, torchvision.__version__, shap.__version__)'
