# Reproducing the v4 dual-bank memory policy (coauthor quick start)

Everything below runs on one machine with 8 GPUs (80 GB each; 40 GB works with a smaller
batch) and needs no scheduler. Total download ~47 GB (dataset 41.6 GB, Stage-1 head 5 GB;
the Pi0.5 base weights are fetched by the trainer from `gs://openpi-assets`).

```bash
git clone git@github.com:ZJU-Walker/memory_project_v4.git
cd memory_project_v4
bash openpi/cluster_v4/coauthor/run_all.sh                  # env -> data -> training
```

Weights & Biases logging is on by default. The training step asks for your API key once
(https://wandb.ai/authorize; or export `WANDB_API_KEY`, or run `openpi/.venv/bin/wandb login`
beforehand; `WANDB=0` disables logging). Runs land in project `openpi` of the team
`kewalk-stanford-university` (https://wandb.ai/kewalk-stanford-university/openpi) once you are
a member of that team; otherwise they go to your own entity and the run URL is printed in the
first lines of the training log. Metrics every 100 updates, camera views at step 0.

`run_all.sh` is four idempotent steps you can also run one at a time:

| step | script | what it does |
|---|---|---|
| 1 | `setup_env.sh` | installs `uv` if missing, `uv sync --frozen` -> `openpi/.venv` (Python 3.11, JAX 0.5.3 CUDA 12), creates the project-local cache dirs |
| 2 | `download_data.sh` | LeRobot dataset (public HF `kewalk123/bin_memory_0830_0831_v36_subtask`, revision `bd97941e`) -> `data/lerobot/...`; manifest, fact labels, norm stats, Stage-1 head (public HF `kewalk123/openpi-v4-memory-artifacts`) -> project layout; verifies the SHA256 pins the config enforces |
| 3 | `train_8gpu.sh` | `scripts/train.py pi05_yam_mem_v4_stage4d --fsdp-devices 8 --batch-size 16` in one process; auto-resume; log in `v4/diagnostics/` |
| 4 | `upload_checkpoint_to_hf.py` | pushes the final checkpoint (params + assets + metadata, ~10 GB; optimizer state skipped), the run manifests and the training log to the Hugging Face Hub, keeping project-relative paths |

Knobs (environment variables): `GPUS` (default 8; must divide 2048, never 3), `BATCH`
(default 16 = 2 per GPU; use 8 on 40 GB GPUs), `CONFIG` (default `pi05_yam_mem_v4_stage4d`),
`EXP` (experiment name; `run_all.sh` fixes one so a re-run resumes and the upload finds it),
`WANDB` (default 1), `WANDB_ENTITY` (default `kewalk-stanford-university`), `UPLOAD` (default 1).

## After training: hand the checkpoint over

`run_all.sh` uploads automatically; the Hub token is checked before training starts so a
missing token fails in the first minute, not after 15 hours. To upload by hand (for example an
intermediate step, or after `UPLOAD=0`):

```bash
openpi/.venv/bin/python openpi/cluster_v4/coauthor/upload_checkpoint_to_hf.py --exp <EXP>          # newest saved step
openpi/.venv/bin/python openpi/cluster_v4/coauthor/upload_checkpoint_to_hf.py --checkpoint v4/checkpoints/pi05_yam_mem_v4_stage4d/<EXP>/3000
```

Target repo: `kewalk123/openpi-v4-memory-artifacts` when the token can write there (the
maintainer's own token, or a member of that namespace); otherwise the script creates
`<your-user>/openpi-v4-memory-checkpoints` (public unless `--private`) and uploads there. In
both cases it prints the download command to send back:

```bash
bash openpi/cluster_v4/coauthor/download_checkpoint.sh <repo> v4/checkpoints/pi05_yam_mem_v4_stage4d/<EXP>/<step>
```

`download_checkpoint.sh` runs inside any clone of this repo (after `setup_env.sh`), fetches
that step plus the sidecar files serving needs (manifest, fact labels, norm stats; not the
41 GB dataset), and prints the serve command with the right write policy for the config
(`always` for Stage 4d, `head` for Stage 4c).

What the run is: Stage 4d of the v4 recipe -- Pi0.5 base + a semantic fact bank (the fact
head grafted from the Stage-1 checkpoint) + a visual Titans bank, both banks allowed to
commit on every valid step, 6000 updates, cosine LR 5e-5. The full design, every stage's
verdict and the evaluation batteries are documented in `openpi/cluster_v4/README.md`.

Evaluating a checkpoint (development split, 8 held-out episodes):

```bash
cd openpi && source cluster_v35/env.sh
PYTHONPATH=scripts .venv/bin/python scripts/v4_side_flip_eval.py --config-name pi05_yam_mem_v4_stage4d \
    --params ../v4/checkpoints/pi05_yam_mem_v4_stage4d/<exp>/<step>/params --state-mask-prob 0 \
    --output-dir ../v4/diagnostics/side_flip_<exp>_<step>
PYTHONPATH=scripts .venv/bin/python scripts/v4_closed_loop_eval.py --config-name pi05_yam_mem_v4_stage4d \
    --params ../v4/checkpoints/pi05_yam_mem_v4_stage4d/<exp>/<step>/params --write-policy always \
    --output-dir ../v4/diagnostics/closed_loop_<exp>_<step>
```

Serving on a robot: `scripts/serve_yam_memory.py --dir <checkpoint step dir> --config
pi05_yam_mem_v4_stage4d --write-policy always` and the client `examples/yam/client_memory_v4.py`
(see their module docstrings). The pretrained Stage-4c policy is in the artifacts repo
(`v4/checkpoints/pi05_yam_mem_v4_stage4c/.../999`, serve it with `--write-policy head`).

Publishing artifacts (maintainer): `upload_artifacts_to_hf.py` (token via `huggingface-cli login`).
