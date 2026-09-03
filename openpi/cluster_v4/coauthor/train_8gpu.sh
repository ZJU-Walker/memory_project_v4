#!/usr/bin/env bash
# Train the v4 dual-bank memory policy on all visible GPUs in ONE process (JAX FSDP; no SLURM,
# no torchrun). Default: 8 GPUs, global batch 16 (2 samples per GPU, the memory budget proven
# on 80 GB H100s), config pi05_yam_mem_v4_stage4e (6000 updates, checkpoints every 500).
#   [GPUS=8] [BATCH=16] [CONFIG=pi05_yam_mem_v4_stage4e] [WANDB=1|0] [EXP=<name>] \
#       [WANDB_ENTITY=<team>] bash openpi/cluster_v4/coauthor/train_8gpu.sh [extra train.py args]
# Resumes automatically if the experiment directory already holds a checkpoint. Logs go to
# v4/diagnostics/train_<EXP>.log; checkpoints to v4/checkpoints/<CONFIG>/<EXP>/<step>/.
# GPU memory: 2 samples/GPU needs ~65 GB per GPU. On 40 GB GPUs use BATCH=8 (1 per GPU).
# Rules: GPUS must divide 2048 and BATCH must be a multiple of GPUS (never 3 GPUs).
# Weights & Biases (default WANDB=1, online): needs an API key once -- `openpi/.venv/bin/wandb
# login`, or WANDB_API_KEY=<key> (https://wandb.ai/authorize); on a terminal the script prompts.
# Runs go to project `openpi` under WANDB_ENTITY (default: the maintainer's team
# kewalk-stanford-university, so the maintainer sees them online). If your account is not a
# member of that team the run falls back to your own entity; share its URL from the log.
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
root="$(cd -- "${here}/../../.." && pwd -P)"
cd "${root}/openpi"
source "${root}/openpi/cluster_v35/env.sh"
mkdir -p "${root}/v4/checkpoints" "${root}/v4/diagnostics"

gpus="${GPUS:-8}"; batch="${BATCH:-16}"; config="${CONFIG:-pi05_yam_mem_v4_stage4e}"
exp="${EXP:-${config#pi05_yam_mem_}_$(date +%Y%m%d_%H%M)}"
WANDB="${WANDB:-1}"
if [ "${WANDB}" = "1" ]; then
  wandb_flag=--wandb-enabled
  if [ -z "${WANDB_API_KEY:-}" ] && ! .venv/bin/wandb login --verify </dev/null >/dev/null 2>&1; then
    if [ -t 0 ]; then
      echo "[train] wandb: no API key stored; paste yours from https://wandb.ai/authorize (stored in ~/.netrc)"
      .venv/bin/wandb login
    else
      echo "[train] wandb: not logged in and no terminal. Run 'openpi/.venv/bin/wandb login', or set WANDB_API_KEY=<key>, or WANDB=0" >&2
      exit 2
    fi
  fi
  export WANDB_ENTITY="${WANDB_ENTITY:-kewalk-stanford-university}"
  if ! .venv/bin/python - <<'EOF' 2>/dev/null
import os, sys, wandb
viewer = wandb.Api().viewer
entity = os.environ["WANDB_ENTITY"]
ok = entity == viewer.entity or entity in (viewer.teams or [])
print(f"[train] wandb: logged in as {viewer.entity}; teams={list(viewer.teams or [])}; entity={entity} {'ok' if ok else 'NOT a member'}")
sys.exit(0 if ok else 1)
EOF
  then
    echo "[train] wandb: not a member of team '${WANDB_ENTITY}' -> logging to your default entity (ask the maintainer to invite you on wandb.ai, or set WANDB_ENTITY)"
    unset WANDB_ENTITY
  fi
else
  wandb_flag=--no-wandb-enabled
fi
if [ $((batch % gpus)) -ne 0 ]; then echo "BATCH=${batch} must be a multiple of GPUS=${gpus}" >&2; exit 2; fi
if [ $((2048 % gpus)) -ne 0 ]; then echo "GPUS=${gpus} must divide 2048 (FSDP layer sharding)" >&2; exit 2; fi
visible=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if [ "${visible}" -lt "${gpus}" ]; then echo "only ${visible} GPUs visible, GPUS=${gpus}" >&2; exit 2; fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((gpus - 1)))}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.92}" PYTHONUNBUFFERED=1

ckdir="${root}/v4/checkpoints/${config}/${exp}"
mode=(); if [ -d "${ckdir}" ]; then
  if ls "${ckdir}" 2>/dev/null | grep -qE '^[0-9]+$'; then mode=(--resume); else mode=(--overwrite); fi
fi
log="${root}/v4/diagnostics/train_${exp}.log"
echo "[train] ${config} exp=${exp} gpus=${gpus} batch=${batch} wandb=${WANDB} entity=${WANDB_ENTITY:-default} ${mode[*]:-fresh} -> ${log}"
[ "${WANDB}" = "1" ] && echo "[train] wandb run URL appears in the first lines of ${log} (project openpi, run name ${exp})"
exec .venv/bin/python scripts/train.py "${config}" --exp-name "${exp}" --batch-size "${batch}" \
  --gradient-accumulation-steps 1 --fsdp-devices "${gpus}" "${wandb_flag}" "${mode[@]}" "$@" 2>&1 | tee -a "${log}"
