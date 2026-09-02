"""Publish the v4 training artifacts to the Hugging Face Hub (public), project-relative layout.

Uploads, keeping the paths the code expects relative to the project root:
  data/0830_0831_episode_manifest_v36_frozen.json     episode manifest (SHA pinned in config)
  data/v4_fact_labels_0830_0831.json                  v4 fact-label sidecar (SHA pinned in config)
  v4/assets/pi05_yam_0830_0831_v36/yam/.../norm_stats*.json   normalization statistics
  v4/checkpoints/pi05_yam_mem_v4_stage1/<run>/1000/{params,assets,_CHECKPOINT_METADATA}
                                                      Stage-1 fact head (grafted at Stage 4 init)
  v4/checkpoints/pi05_yam_mem_v4_stage4c/<run>/999/{params,assets,_CHECKPOINT_METADATA}
                                                      Stage-4c policy (serve / evaluate)
and writes a model card with SHA256s. The LeRobot dataset lives in its own dataset repo
(kewalk123/bin_memory_0830_0831_v36_subtask); pi05_base is fetched from gs://openpi-assets.

Usage (token via `huggingface-cli login` or HF_TOKEN):
  python openpi/cluster_v4/coauthor/upload_artifacts_to_hf.py [--repo kewalk123/openpi-v4-memory-artifacts]
      [--skip-checkpoints] [--make-dataset-public]
"""

import argparse
import hashlib
import pathlib
import sys
import time

from huggingface_hub import HfApi

ROOT = pathlib.Path(__file__).resolve().parents[3]  # memory_project/
DATASET_REPO = "kewalk123/bin_memory_0830_0831_v36_subtask"
DATASET_REVISION = "bd97941eca402f8be854ee8a0a3bbad14df37292"
STAGE1 = "v4/checkpoints/pi05_yam_mem_v4_stage1/v4_stage1_20260901_r3_h100/1000"
STAGE4C = "v4/checkpoints/pi05_yam_mem_v4_stage4c/v4_stage4c_20260902_r1/999"
SMALL_FILES = (
    "data/0830_0831_episode_manifest_v36_frozen.json",
    "data/v4_fact_labels_0830_0831.json",
    "v4/assets/pi05_yam_0830_0831_v36/yam/bin_memory_0830_0831_v36_subtask/norm_stats.json",
    "v4/assets/pi05_yam_0830_0831_v36/yam/bin_memory_0830_0831_v36_subtask/norm_stats_provenance.json",
)


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default="kewalk123/openpi-v4-memory-artifacts")
    parser.add_argument("--skip-checkpoints", action="store_true")
    parser.add_argument("--make-dataset-public", action="store_true")
    args = parser.parse_args()
    api = HfApi()
    print("user:", api.whoami()["name"], flush=True)

    if args.make_dataset_public:
        api.update_repo_settings(DATASET_REPO, repo_type="dataset", private=False)
        print(f"dataset {DATASET_REPO} is now public", flush=True)

    api.create_repo(args.repo, repo_type="model", private=False, exist_ok=True)
    digests = {rel: sha256(ROOT / rel) for rel in SMALL_FILES}
    for rel, digest in digests.items():
        print(f"{digest}  {rel}", flush=True)
        api.upload_file(path_or_fileobj=str(ROOT / rel), path_in_repo=rel, repo_id=args.repo, repo_type="model")

    checkpoint_notes = []
    if not args.skip_checkpoints:
        for rel in (STAGE1, STAGE4C):
            src = ROOT / rel
            if not src.is_dir():
                print(f"missing checkpoint dir {src}; skipping", file=sys.stderr)
                continue
            started = time.time()
            api.upload_folder(
                folder_path=str(src),
                path_in_repo=rel,
                repo_id=args.repo,
                repo_type="model",
                allow_patterns=["params/**", "assets/**", "_CHECKPOINT_METADATA"],
                commit_message=f"upload {rel}",
            )
            checkpoint_notes.append(f"- `{rel}` (params + assets + metadata; train_state omitted), uploaded in {time.time() - started:.0f}s")
            print(checkpoint_notes[-1], flush=True)

    card = "\n".join(
        [
            "---",
            "license: mit",
            "tags: [robotics, pi05, memory, openpi]",
            "---",
            "# openpi v4 dual-bank memory policy: training artifacts",
            "",
            "Companion to the `ZJU-Walker/memory_project_v4` code repository. Files keep their",
            "project-relative paths: download with `--local-dir <project root>` and they land where",
            "`openpi/src/openpi/shared/project_paths.py` expects them.",
            "",
            f"Dataset (LeRobot, 70 episodes, 41.6 GB): `{DATASET_REPO}` revision `{DATASET_REVISION}`",
            "-> `data/lerobot/yam/bin_memory_0830_0831_v36_subtask/{data,meta}`.",
            "Base model: `gs://openpi-assets/checkpoints/pi05_base/params` (downloaded by the trainer).",
            "",
            "## Files and SHA256",
            "",
            *[f"- `{rel}`: `{digest}`" for rel, digest in digests.items()],
            *checkpoint_notes,
            "",
            "The manifest and fact-label digests are pinned in `openpi/src/openpi/training/config.py`",
            "(`memory_episode_manifest_sha256`, `memory_v4_fact_labels_sha256`); training refuses to",
            "start if they differ.",
            "",
            "## Quick start",
            "",
            "```bash",
            "git clone git@github.com:ZJU-Walker/memory_project_v4.git && cd memory_project_v4",
            "bash openpi/cluster_v4/coauthor/run_all.sh   # env -> data -> 8-GPU training",
            "```",
            "",
        ]
    )
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=args.repo, repo_type="model")
    print(f"done: https://huggingface.co/{args.repo}", flush=True)


if __name__ == "__main__":
    main()
