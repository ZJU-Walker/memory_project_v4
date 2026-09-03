#!/usr/bin/env bash
# Download everything training needs from the public Hugging Face repos into the project layout.
#   bash openpi/cluster_v4/coauthor/download_data.sh [--no-policy-checkpoint]
# Pulls:
#   dataset  kewalk123/bin_memory_0830_0831_v36_subtask @ bd97941e  -> data/lerobot/yam/bin_memory_0830_0831_v36_subtask/{data,meta}  (41.6 GB, 70 episodes)
#   model    kewalk123/openpi-v4-memory-artifacts                    -> data/*.json, v4/assets/..., v4/checkpoints/... (Stage-1 head 5 GB; Stage-4c policy 10 GB unless --no-policy-checkpoint)
# The Pi0.5 base weights (gs://openpi-assets/checkpoints/pi05_base) are fetched by the trainer itself.
# Verifies the manifest / fact-label SHA256 pins the training config enforces. Re-running only fetches missing files.
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
root="$(cd -- "${here}/../../.." && pwd -P)"
cd "${root}/openpi"
source "${root}/openpi/cluster_v35/env.sh"
hf="${root}/openpi/.venv/bin/huggingface-cli"
[ -x "${hf}" ] || { echo "run setup_env.sh first (${hf} missing)" >&2; exit 2; }

DATASET_REPO=kewalk123/bin_memory_0830_0831_v36_subtask
DATASET_REV=bd97941eca402f8be854ee8a0a3bbad14df37292
ARTIFACTS_REPO=kewalk123/openpi-v4-memory-artifacts
dataset_dir="${root}/data/lerobot/yam/bin_memory_0830_0831_v36_subtask"

echo "[data] dataset -> ${dataset_dir}"
mkdir -p "${dataset_dir}"
"${hf}" download "${DATASET_REPO}" --repo-type dataset --revision "${DATASET_REV}" --local-dir "${dataset_dir}" >/dev/null
rm -rf "${dataset_dir}/.cache"
n_parquet=$(find "${dataset_dir}/data" -name '*.parquet' | wc -l)
[ "${n_parquet}" = "70" ] || { echo "[data] expected 70 parquet files, found ${n_parquet}" >&2; exit 3; }
[ -f "${dataset_dir}/meta/info.json" ] || { echo "[data] meta/info.json missing" >&2; exit 3; }

echo "[data] artifacts -> ${root}"
# Never let the Hub repo's own README.md / .gitattributes land in the project root (they would
# overwrite the git-tracked README and change LFS attributes of tracked files).
exclude=(--exclude 'README.md' '.gitattributes')
if [ "${1:-}" = "--no-policy-checkpoint" ]; then exclude+=('v4/checkpoints/pi05_yam_mem_v4_stage4c/*'); fi
"${hf}" download "${ARTIFACTS_REPO}" --repo-type model --local-dir "${root}" "${exclude[@]}" >/dev/null
rm -rf "${root}/.cache"

echo "[data] verifying pinned digests"
"${root}/openpi/.venv/bin/python" - "${root}" <<'EOF'
import hashlib, json, pathlib, re, sys
root = pathlib.Path(sys.argv[1])
cfg = (root / "openpi/src/openpi/training/config.py").read_text()
pins = {
    "data/0830_0831_episode_manifest_v36_frozen.json": re.search(r'memory_episode_manifest_sha256=\("([0-9a-f]{64})"\)', cfg).group(1),
    "data/v4_fact_labels_0830_0831.json": re.search(r'memory_v4_fact_labels_sha256=\("([0-9a-f]{64})"\)', cfg).group(1),
}
ok = True
for rel, expected in pins.items():
    digest = hashlib.sha256((root / rel).read_bytes()).hexdigest()
    print(f"  {rel}: {'OK' if digest == expected else 'MISMATCH ' + digest}")
    ok &= digest == expected
for rel in (
    "v4/assets/pi05_yam_0830_0831_v36/yam/bin_memory_0830_0831_v36_subtask/norm_stats.json",
    "v4/checkpoints/pi05_yam_mem_v4_stage1/v4_stage1_20260901_r3_h100/1000/params",
    "data/0830_0831_episode_manifest_v36_frozen_block_confound.json",
    "data/0830_0831_episode_manifest_v36_frozen_e_visibility.json",
    "data/0830_0831_episode_manifest_v36_frozen_d_valid.json",
):
    present = (root / rel).exists()
    print(f"  {rel}: {'present' if present else 'MISSING'}")
    ok &= present
# The 70 raw per-episode label files (pinned by label_sha256 in the manifest).
manifest = json.loads((root / "data/0830_0831_episode_manifest_v36_frozen.json").read_text())
labels_ok = 0
for episode in manifest["episodes"]:
    if not episode.get("include", True):
        continue
    # raw_root is ".." relative to the manifest (= the project root); raw_dir is "data/<collection>/<demo>".
    path = root / episode["raw_dir"] / episode["label_file"]
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == episode["label_sha256"]:
        labels_ok += 1
    else:
        print(f"  label file for {episode['stable_id']}: MISSING or digest mismatch ({path})")
        ok = False
print(f"  per-episode label files verified: {labels_ok}")
sys.exit(0 if ok else 4)
EOF
echo "[data] done. Next: bash openpi/cluster_v4/coauthor/train_8gpu.sh"
