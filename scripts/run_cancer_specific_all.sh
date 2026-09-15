#!/usr/bin/env bash
set -euo pipefail

# OpenMP reads this before Python can configure torch's thread pool. Preserve
# valid user values (including nested-team lists); replace empty/invalid values.
if [[ ! "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ ]]; then
  if [[ -v OMP_NUM_THREADS ]]; then
    echo 'Invalid OMP_NUM_THREADS; using 4.' >&2
  fi
  export OMP_NUM_THREADS=4
fi

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${GATRANS_PYTHON:-/root/miniconda3/bin/python}"

if [[ ! -x "$python_bin" ]]; then
  echo "Required interpreter is missing or not executable: $python_bin" >&2
  exit 2
fi

python_prefix="$($python_bin -c 'import sys; print(sys.prefix)')"
if [[ "$python_prefix" != "/root/miniconda3" ]]; then
  echo "Refusing to run with $python_prefix; use /root/miniconda3 and keep packages off the data disk." >&2
  exit 2
fi

cd "$repo_dir"
exec "$python_bin" -u scripts/cancer_specific.py \
  --data dataset/cancer_specific \
  --out results/cancer_specific_all_v2 \
  --kinds homogeneous heterogeneous \
  --require-full-dataset \
  "$@"
