# Reproducing the v4 dual-bank memory policy (coauthor quick start)

Everything below runs on one machine with 8 GPUs (80 GB each; 40 GB works with a smaller
batch) and needs no scheduler. Total download ~47 GB (dataset 41.6 GB, Stage-1 head 5 GB;
the Pi0.5 base weights are fetched by the trainer from `gs://openpi-assets`).

```bash
git clone git@github.com:ZJU-Walker/memory_project.git
cd memory_project && git checkout v4
WANDB=0 bash openpi/cluster_v4/coauthor/run_all.sh          # env -> data -> training
```

`run_all.sh` is three idempotent steps you can also run one at a time:

| step | script | what it does |
|---|---|---|
| 1 | `setup_env.sh` | installs `uv` if missing, `uv sync --frozen` -> `openpi/.venv` (Python 3.11, JAX 0.5.3 CUDA 12), creates the project-local cache dirs |
| 2 | `download_data.sh` | LeRobot dataset (public HF `kewalk123/bin_memory_0830_0831_v36_subtask`, revision `bd97941e`) -> `data/lerobot/...`; manifest, fact labels, norm stats, Stage-1 head (public HF `kewalk123/openpi-v4-memory-artifacts`) -> project layout; verifies the SHA256 pins the config enforces |
| 3 | `train_8gpu.sh` | `scripts/train.py pi05_yam_mem_v4_stage4d --fsdp-devices 8 --batch-size 16` in one process; auto-resume; log in `v4/diagnostics/` |

Knobs (environment variables): `GPUS` (default 8; must divide 2048, never 3), `BATCH`
(default 16 = 2 per GPU; use 8 on 40 GB GPUs), `CONFIG` (default `pi05_yam_mem_v4_stage4d`),
`EXP` (experiment name), `WANDB=1` (needs `wandb login` first; project `openpi`).

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
