#!/usr/bin/env bash
# Fetch one trained checkpoint from the Hugging Face Hub into the project layout (plus the small
# sidecar files the server needs: episode manifest, fact labels, norm stats), then print the
# serve command. Run inside a clone of the code repo, after setup_env.sh (needs openpi/.venv).
#   bash openpi/cluster_v4/coauthor/download_checkpoint.sh <repo> <project-relative step dir> [--root <dir>]
# e.g.
#   bash openpi/cluster_v4/coauthor/download_checkpoint.sh kewalk123/openpi-v4-memory-artifacts \
#        v4/checkpoints/pi05_yam_mem_v4_stage4e/v4_stage4e_20260903_0900/5999
# The <repo> <step dir> pair is printed by upload_checkpoint_to_hf.py at the end of a training run.
# Public repos need no token; private ones need `openpi/.venv/bin/huggingface-cli login` or HF_TOKEN.
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
root="$(cd -- "${here}/../../.." && pwd -P)"
repo="${1:-}"; rel="${2:-}"
[ -n "${repo}" ] && [ -n "${rel}" ] || { sed -n '2,10p' "${BASH_SOURCE[0]}"; exit 2; }
shift 2
while [ $# -gt 0 ]; do
  case "$1" in
    --root) root="$(cd -- "$2" && pwd -P)"; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
rel="${rel%/}"
case "${rel}" in
  v4/checkpoints/*/*/[0-9]*) ;;
  *) echo "step dir must look like v4/checkpoints/<config>/<exp>/<step>, got '${rel}'" >&2; exit 2 ;;
esac
hf="${HF_CLI:-${here}/../../.venv/bin/huggingface-cli}"
[ -x "${hf}" ] || { echo "huggingface-cli not found at ${hf}; run setup_env.sh or set HF_CLI" >&2; exit 2; }

config="$(echo "${rel}" | cut -d/ -f3)"
echo "[download] ${repo}:${rel} -> ${root}/${rel}"
"${hf}" download "${repo}" --repo-type model --local-dir "${root}" \
  --include "${rel}/params/*" "${rel}/assets/*" "${rel}/_CHECKPOINT_METADATA" \
            "$(dirname "${rel}")/*.json" "$(dirname "${rel}")/wandb_id.txt" \
            "v4/diagnostics/train_$(basename "$(dirname "${rel}")").log" >/dev/null
# Sidecars pinned by the config live in the maintainer's artifacts repo (public); skip if present.
if [ ! -f "${root}/data/v4_fact_labels_0830_0831.json" ] || [ ! -f "${root}/data/0830_0831_episode_manifest_v36_frozen.json" ]; then
  echo "[download] sidecar files (manifest, fact labels, norm stats) from kewalk123/openpi-v4-memory-artifacts"
  "${hf}" download kewalk123/openpi-v4-memory-artifacts --repo-type model --local-dir "${root}" \
    --include "data/*.json" "v4/assets/*" >/dev/null
fi
rm -rf "${root}/.cache"
[ -d "${root}/${rel}/params" ] || { echo "[download] ${rel}/params missing after download" >&2; exit 3; }
[ -f "${root}/data/v4_fact_labels_0830_0831.json" ] || { echo "[download] data/v4_fact_labels_0830_0831.json missing" >&2; exit 3; }
du -sh "${root}/${rel}" | sed 's/^/[download] size: /'

case "${config}" in
  *stage4[de]*) policy=always ;;   # trained with writes on every step: deployment == training
  *) policy=head ;;             # Stage-4c and earlier: evidence-only training, gate writes by the fact head
esac
cat <<EOF
[download] done. Serve it (GPU machine, from the project root):
    cd openpi && source cluster_v35/env.sh
    .venv/bin/python scripts/serve_yam_memory.py --dir ../${rel} --config ${config} --write-policy ${policy} --port 8000
  then on the robot station: python examples/yam/client_memory_v4.py --host <server> --port 8000 (see its docstring).
  Evaluate offline (needs the dataset, download_data.sh): scripts/v4_side_flip_eval.py / v4_closed_loop_eval.py
    --config-name ${config} --params ../${rel}/params (see coauthor/README.md).
EOF
