#!/usr/bin/env bash
# Fresh machine -> trained v4 policy -> checkpoint on the Hugging Face Hub, in one command (no SLURM):
#   git clone git@github.com:ZJU-Walker/memory_project_v4.git && cd memory_project_v4
#   [WANDB=0] [UPLOAD=0] [GPUS=8] [BATCH=16] [EXP=<name>] bash openpi/cluster_v4/coauthor/run_all.sh
# Steps: environment (uv, pinned lock) -> Hub login checks -> data + artifacts from Hugging Face
#        -> 8-GPU training -> upload of the final checkpoint (upload_checkpoint_to_hf.py).
# Each step is idempotent; re-run after a failure (same EXP to resume training) and it continues.
# Weights & Biases logging is ON by default (WANDB=0 disables): the training step asks for your
# API key once (https://wandb.ai/authorize) unless WANDB_API_KEY is set or `wandb login` was run.
# The upload is ON by default (UPLOAD=0 disables) and needs a Hugging Face token with write access
# (`openpi/.venv/bin/huggingface-cli login` or HF_TOKEN); it is checked BEFORE training starts.
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
root="$(cd -- "${here}/../../.." && pwd -P)"
config="${CONFIG:-pi05_yam_mem_v4_stage4e}"
export CONFIG="${config}"
export EXP="${EXP:-${config#pi05_yam_mem_}_$(date +%Y%m%d_%H%M)}"
upload="${UPLOAD:-1}"

bash "${here}/setup_env.sh"
py="${root}/openpi/.venv/bin/python"
if [ "${upload}" = "1" ]; then
  if ! "${py}" "${here}/upload_checkpoint_to_hf.py" --check-login; then
    if [ -t 0 ]; then
      echo "[run_all] paste a Hugging Face token with write access (https://huggingface.co/settings/tokens)"
      "${root}/openpi/.venv/bin/huggingface-cli" login
      "${py}" "${here}/upload_checkpoint_to_hf.py" --check-login
    else
      echo "[run_all] no Hub token and no terminal: set HF_TOKEN, run huggingface-cli login, or UPLOAD=0" >&2
      exit 2
    fi
  fi
fi
bash "${here}/download_data.sh" --no-policy-checkpoint
bash "${here}/train_8gpu.sh" "$@"
if [ "${upload}" = "1" ]; then
  "${py}" "${here}/upload_checkpoint_to_hf.py" --exp "${EXP}"
else
  echo "[run_all] training finished; upload later with: openpi/.venv/bin/python ${here#${root}/}/upload_checkpoint_to_hf.py --exp ${EXP}"
fi
