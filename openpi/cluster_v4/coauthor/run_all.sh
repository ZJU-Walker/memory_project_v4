#!/usr/bin/env bash
# Fresh machine -> trained v4 policy in one command (no SLURM):
#   git clone git@github.com:ZJU-Walker/memory_project.git && cd memory_project && git checkout v4
#   [WANDB=1] [GPUS=8] [BATCH=16] bash openpi/cluster_v4/coauthor/run_all.sh
# Steps: environment (uv, pinned lock) -> data + artifacts from Hugging Face -> 8-GPU training.
# Each step is idempotent; re-run after a failure and it continues where it stopped.
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
bash "${here}/setup_env.sh"
bash "${here}/download_data.sh" --no-policy-checkpoint
exec bash "${here}/train_8gpu.sh" "$@"
