#!/usr/bin/env bash
# One-time environment setup for a fresh machine (no SLURM needed).
#   bash openpi/cluster_v4/coauthor/setup_env.sh
# Installs uv if missing, creates openpi/.venv from the pinned uv.lock (Python 3.11, JAX 0.5.3
# with CUDA 12 wheels), and creates the project-local cache/data directories. Re-running is safe.
# Requirements: Linux x86_64, NVIDIA driver supporting CUDA 12, git, curl, ~30 GB free for the
# environment + JAX/HF caches, ~60 GB for the dataset and checkpoints (see download_data.sh).
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
root="$(cd -- "${here}/../../.." && pwd -P)"   # memory_project/
cd "${root}/openpi"

if ! command -v uv >/dev/null 2>&1; then
  echo "[setup] installing uv (https://astral.sh/uv)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
fi
command -v uv >/dev/null 2>&1 || { echo "[setup] uv not on PATH after install; open a new shell and re-run" >&2; exit 2; }

# Portable environment contract: every cache/data path is derived from the project location.
# (cluster_v35/env.sh is path-relative; cluster_v4/env.sh additionally pins HOME for one cluster
# and is NOT used here.)
source "${root}/openpi/cluster_v35/env.sh"
mkdir -p "${root}/v4/assets" "${root}/v4/checkpoints" "${root}/v4/diagnostics" "${root}/data/lerobot/yam"

echo "[setup] uv sync (frozen lock) in ${root}/openpi"
GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen
.venv/bin/python - <<'EOF'
import jax, importlib
print("[setup] python", __import__("sys").version.split()[0], "| jax", jax.__version__, "| devices:", jax.devices())
for name in ("openpi", "huggingface_hub", "wandb", "lerobot"):
    try:
        importlib.import_module(name); print(f"[setup] import {name}: ok")
    except Exception as e:  # noqa: BLE001
        print(f"[setup] import {name}: FAILED ({e})")
EOF
echo "[setup] done. Next: bash openpi/cluster_v4/coauthor/download_data.sh"
