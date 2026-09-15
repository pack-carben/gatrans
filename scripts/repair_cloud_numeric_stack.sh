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

"$python_bin" -c \
  'import numpy, scipy, sklearn, pandas, h5py; print("numeric stack OK", numpy.__version__, scipy.__version__, sklearn.__version__)'
