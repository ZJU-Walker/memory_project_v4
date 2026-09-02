#!/usr/bin/env bash
# Train the v4 dual-bank memory policy on all visible GPUs in ONE process (JAX FSDP; no SLURM,
# no torchrun). Default: 8 GPUs, global batch 16 (2 samples per GPU, the memory budget proven
# on 80 GB H100s), config pi05_yam_mem_v4_stage4d (6000 updates, checkpoints every 500).
#   [GPUS=8] [BATCH=16] [CONFIG=pi05_yam_mem_v4_stage4d] [WANDB=0|1] [EXP=<name>] \
#       bash openpi/cluster_v4/coauthor/train_8gpu.sh [extra train.py args]
# Resumes automatically if the experiment directory already holds a checkpoint. Logs go to
# v4/diagnostics/train_<EXP>.log; checkpoints to v4/checkpoints/<CONFIG>/<EXP>/<step>/.
# GPU memory: 2 samples/GPU needs ~65 GB per GPU. On 40 GB GPUs use BATCH=8 (1 per GPU).
# Rules: GPUS must divide 2048 and BATCH must be a multiple of GPUS (never 3 GPUs).
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
root="$(cd -- "${here}/../../.." && pwd -P)"
cd "${root}/openpi"
source "${root}/openpi/cluster_v35/env.sh"
mkdir -p "${root}/v4/checkpoints" "${root}/v4/diagnostics"

gpus="${GPUS:-8}"; batch="${BATCH:-16}"; config="${CONFIG:-pi05_yam_mem_v4_stage4d}"
exp="${EXP:-${config#pi05_yam_mem_}_$(date +%Y%m%d_%H%M)}"
if [ "${WANDB:-0}" = "1" ]; then wandb_flag=--wandb-enabled; else wandb_flag=--no-wandb-enabled; fi
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
echo "[train] ${config} exp=${exp} gpus=${gpus} batch=${batch} wandb=${WANDB:-0} ${mode[*]:-fresh} -> ${log}"
exec .venv/bin/python scripts/train.py "${config}" --exp-name "${exp}" --batch-size "${batch}" \
  --gradient-accumulation-steps 1 --fsdp-devices "${gpus}" "${wandb_flag}" "${mode[@]}" "$@" 2>&1 | tee -a "${log}"
