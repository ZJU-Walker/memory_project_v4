#!/usr/bin/env bash
# v4 Stage-1 launcher (V4_PLAN.md §5): fact-head-only training, single GPU.
#   cluster_v4/stage1_train.sh --exp-name v4_stage1_YYYYMMDD_rN [extra train.py args]
set -euo pipefail

v4_launcher_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=env.sh
source "${v4_launcher_dir}/env.sh"

v4_python="${V4_PYTHON:-${MEMORY_PROJECT_ROOT}/openpi/.venv/bin/python}"
if [[ ! -x "${v4_python}" ]]; then
  echo "v4 Python is not executable: ${v4_python}" >&2
  exit 2
fi

export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.92}"
cd -- "${MEMORY_PROJECT_ROOT}/openpi"
exec "${v4_python}" scripts/train.py pi05_yam_mem_v4_stage1 "$@"
