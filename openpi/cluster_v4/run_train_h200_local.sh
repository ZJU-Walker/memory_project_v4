#!/usr/bin/env bash
# v4 training on the single H200 of Slurm job $JOB (default 17267134, iris-hgx-2), preemption-safe,
# from the node-local project copy when staged (cluster_v4/stage_local_project_hgx2.sh).
#   [JOB=17267134] [BATCH=2] [ACCUM=1] [WANDB=1] cluster_v4/run_train_h200_local.sh <config-name> <exp-name> [extra train.py args]
# Run ON the node (a plain `ssh iris-hgx-2` adopts job 17267134); the payload is an --overlap step
# of the job. Kills only this job's busy placeholder (marker gpu_placeholder_marker_<JOB>), never
# the user's train_hs.py keep-alive. Status/log files stay on NFS (v4/diagnostics of the NFS tree);
# checkpoints land under the local root's v4/checkpoints (copy back with cluster_v4/sync_results_from_hgx2.sh).
# Private JAX compile cache under the local root (never shared with another running process).
set -u
config="$1"; exp="$2"; shift 2
batch="${BATCH:-2}"; accum="${ACCUM:-1}"; JOB="${JOB:-17267134}"
nfs_root=/iris/u/kewalk/memory_project_v4
local_root=/scr/kewalk_v4/memory_project_v4
if [ -e "$local_root/.staged" ] && [ -x "$local_root/openpi/.venv/bin/python" ]; then root="$local_root"; else root="$nfs_root"; fi
export HOME=/iris/u/kewalk
export OPENPI_JAX_CACHE_DIR="$root/v35/cache/jax_hgx2"
mkdir -p "$OPENPI_JAX_CACHE_DIR"
if [ "${WANDB:-1}" = "1" ]; then wandb_flag=--wandb-enabled; else wandb_flag=--no-wandb-enabled; fi
diag="$nfs_root/v4/diagnostics"
ckdir="$root/v4/checkpoints/$config/$exp"
mode=fresh; extra=()
if [ -d "$ckdir" ]; then
  if ls "$ckdir" 2>/dev/null | grep -qE '^[0-9]+$'; then mode=resume; extra=(--resume); else mode=overwrite; extra=(--overwrite); fi
fi
ph=$(pgrep -f "gpu_placeholder_marker_${JOB}\b" || true)
[ -n "$ph" ] && { kill $ph 2>/dev/null; sleep 5; echo "killed placeholder pids: $ph"; }
job=$(grep -oE 'job_[0-9]+' /proc/self/cgroup | sort -u | tr '\n' ' ')
echo "launch $(date +%m/%d\ %H:%M) host=$(hostname) job=$job step-of=$JOB config=$config exp=$exp batch=$batch accum=$accum mode=$mode root=$root extra=$*" >> "$diag/train_${exp}_status.log"
cd "$root/openpi" || exit 2
srun --jobid="$JOB" --overlap --nodes=1 --ntasks=1 --cpus-per-task=16 --gres=gpu:1 \
  env CUDA_VISIBLE_DEVICES=0 HOME=/iris/u/kewalk OPENPI_JAX_CACHE_DIR="$OPENPI_JAX_CACHE_DIR" XLA_PYTHON_CLIENT_MEM_FRACTION=0.94 \
  cluster_v4/train.sh "$config" --exp-name "$exp" --batch-size "$batch" --gradient-accumulation-steps "$accum" --fsdp-devices 1 \
  "$wandb_flag" "${extra[@]}" "$@" >> "$diag/train_${exp}.log" 2>&1
echo "exit=$? $(date +%m/%d\ %H:%M)" >> "$diag/train_${exp}_status.log"
