#!/usr/bin/env bash
# Generic v4 launcher (v4 light protocol; see cluster_v4/README.md §8):
#   cluster_v4/train.sh <config-name> --exp-name <exp> [extra train.py args]
# Stage recipes: pi05_yam_mem_v4_stage1 (fact head), pi05_yam_mem_v4_stage2a (oracle
# semantic writes, semantic-only), pi05_yam_mem_v4 (full dual bank).
set -euo pipefail

v4_launcher_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=env.sh
source "${v4_launcher_dir}/env.sh"

if [[ $# -lt 1 ]]; then
  echo "usage: cluster_v4/train.sh <config-name> --exp-name <exp> [args]" >&2
  exit 2
fi
v4_config="$1"; shift

v4_python="${V4_PYTHON:-${MEMORY_PROJECT_ROOT}/openpi/.venv/bin/python}"
if [[ ! -x "${v4_python}" ]]; then
  echo "v4 Python is not executable: ${v4_python}" >&2
  exit 2
fi

export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.92}"
cd -- "${MEMORY_PROJECT_ROOT}/openpi"
exec "${v4_python}" scripts/train.py "${v4_config}" "$@"
