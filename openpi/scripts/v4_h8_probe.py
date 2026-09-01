"""Is the fact linearly present in the frozen layer-8 features the fact head reads?

Companion diagnostic to v4_stage1_eval.py. Collects the layer-8 top-camera hidden states
(h8_top, [256 tokens x 2048], from the FROZEN backbone -- identical for every Stage-1
checkpoint) on evidence frames of the train split and fits the same episode-OOF ridge probe
with stratified permutation null for the banana location, under three feature readouts:

  * mean      -- mean over the 256 top-camera tokens (what a content-free query with
                 diffuse attention effectively sees);
  * grid4x4   -- mean within each cell of a 4x4 grid over the 16x16 SigLIP patch layout
                 (keeps coarse location: left/right bin is a spatial fact);
  * grid8x8   -- 8x8 cells (finer location, 64 x 2048 dims; dual ridge handles it).

High OOF accuracy for a spatial readout but not for the mean says the information is
present and the compressor must learn to localize; chance everywhere says layer-8 features
of the frozen VLM do not expose it and the head must read elsewhere.
"""

import argparse
import json
import pathlib
import sys

import pyarrow.parquet  # noqa: F401  isort: skip  (before openpi/JAX: libarrow segfault order)

import numpy as np

import v4_stage1_eval as battery

PATCH_GRID = 16  # SigLIP So400m/14 at 224x224


def _grid_pool(h8_top: np.ndarray, cells: int) -> np.ndarray:
    """[n, 256, d] -> [n, cells*cells*d] by mean within each grid cell."""
    n, tokens, d = h8_top.shape
    if tokens != PATCH_GRID * PATCH_GRID:
        raise battery.Stage1EvalError(f"expected {PATCH_GRID * PATCH_GRID} top-camera tokens, got {tokens}")
    grid = h8_top.reshape(n, PATCH_GRID, PATCH_GRID, d)
    size = PATCH_GRID // cells
    pooled = grid.reshape(n, cells, size, cells, size, d).mean(axis=(2, 4))
    return pooled.reshape(n, cells * cells * d)


def main(argv=None) -> None:
    from openpi.shared import project_paths

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=pathlib.Path, required=True)
    parser.add_argument("--frames-per-episode", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)

    manifest_path = project_paths.project_path(project_paths.V35_FROZEN_MANIFEST)
    facts_path = project_paths.project_path(project_paths.V4_FACT_LABELS)
    dataset_root = project_paths.project_path(project_paths.V35_DATASET_DIR)
    episodes, provenance = battery._load_manifest_and_facts(manifest_path, facts_path, split="train")
    task_names = battery._load_task_names(dataset_root)
    runtime = battery.Stage1Runtime(params_path=args.params, batch_size=args.batch_size)

    # Reuse the battery's probe, but keep h8_top (the probe step returns it).
    def probe_h8(items):
        real = len(items)
        padded = list(items) + [items[-1]] * (runtime.batch_size - real)
        stacked = runtime._jax.tree.map(lambda *v: runtime._jnp.concatenate(v, axis=0), *padded)
        observation = runtime._model_lib.Observation.from_dict(stacked)
        outputs = runtime._probe(observation)
        return np.asarray(runtime._jax.device_get(outputs["h8_top"][:real]), dtype=np.float32)

    features, labels, stable, collections = [], [], [], []
    pending = []
    pending_meta = []

    def flush():
        if not pending:
            return
        for h8, meta in zip(probe_h8(pending), pending_meta, strict=True):
            features.append(h8)
            labels.append(meta[0])
            stable.append(meta[1])
            collections.append(meta[2])
        pending.clear()
        pending_meta.clear()

    for episode in episodes:
        pools = battery._episode_pool_frames(
            task_names, dataset_root, episode, frames_per_pool=args.frames_per_episode, seed=args.seed
        )
        for row in battery._read_rows(dataset_root, episode, pools["evidence"]):
            pending.append(runtime.observation(row, prompt=episode.prompt, state_override=None))
            pending_meta.append((episode.fact_targets[0], episode.stable_id, episode.collection))
            if len(pending) == runtime.batch_size:
                flush()
    flush()

    h8 = np.stack(features)  # [n, 256, 2048]
    labels_arr = np.asarray(labels, dtype=np.int64)
    stable_arr = np.asarray(stable)
    coll_arr = np.asarray(collections)
    readouts = {
        "mean": h8.mean(axis=1),
        "grid4x4": _grid_pool(h8, 4),
        "grid8x8": _grid_pool(h8, 8),
    }
    results = {}
    for name, feats in readouts.items():
        results[name] = battery.leak_probe(feats, labels_arr, stable_arr, coll_arr)
        results[name]["dims"] = int(feats.shape[1])
        print(
            f"{name:8s} dims={feats.shape[1]:6d}  OOF balanced acc={results[name]['balanced_accuracy']:.3f}  "
            f"perm p={results[name]['permutation_p']:.3f}  null mean={results[name]['null_mean']:.3f}"
        )
    report = {
        "schema_version": "openpi.v4.h8-probe.v1",
        "split": "train",
        "episodes": len(episodes),
        "frames": int(h8.shape[0]),
        "frames_per_episode": args.frames_per_episode,
        "target": "banana_location (slot 0; 0=left_bin, 1=right_bin)",
        "parameter_tree_sha256": runtime.parameter_tree_sha256,
        **provenance,
        "readouts": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    sys.exit(main())
