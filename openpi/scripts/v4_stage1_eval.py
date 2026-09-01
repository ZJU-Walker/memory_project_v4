"""v4 Stage-1 evaluation battery (V4_PLAN.md §5, Stage 1).

Establishes -- BEFORE any memory training is authorizable -- that the memory-blind fact head
is visually grounded on fresh episodes. For the requested manifest split (default:
development; final_test is refused), every included episode contributes three frame pools
selected by the frozen per-frame task labels:

  * pre       -- "open both lids" rows: facts are not yet observable; the head must abstain
                 (`unknown`) and, harder, its slot FEATURES must not decode the answer
                 (episode-OOF ridge probe with stratified permutation null -- Gate-B style).
                 Like Gate B, the leak probes always run over the TRAIN split's episodes: an
                 8-episode split has a discrete permutation space of ~36 arrangements, so its
                 achievable p-value floor (~0.03) cannot support the 0.05 gate;
  * evidence  -- "inspect both bins" rows inside the manifest's reviewed visibility window:
                 fact accuracy, confidence, and write-eligibility live here;
  * post      -- the manifest's d_valid wait rows (bins closed): abstention + feature probe,
                 exactly the occlusion condition the semantic memory exists for.

Interventions on the evidence pool:
  * prompt swap  -- same frames, the OTHER object's prompt: fact predictions should be
                    prompt-invariant (facts describe the scene, not the instruction);
  * state neutral -- same frames, the state vector replaced by the pool's mean state:
                    predictions should not depend on proprioception.

Everything runs through `Pi0.v4_fact_probe_step` against fresh banks: the exact memory-free
boundary Stage-1 claims are scoped to. Outputs one self-hashed JSON report plus a canonical
NPZ of per-frame features/labels under --output-dir (create-only).
"""

import argparse
import dataclasses
import hashlib
import io
import json
import pathlib
import sys
from collections.abc import Mapping, Sequence
from typing import Any

# The installed Arrow/native stack segfaults inside libarrow.so when OpenPI/JAX is imported
# before PyArrow in the same process (the conftest.py-documented safe order). Import first.
import pyarrow.parquet  # noqa: F401  isort: skip

import numpy as np

CONFIG_NAME = "pi05_yam_mem_v4_stage1"
SCHEMA_VERSION = "openpi.v4.stage1-eval.v1"
IMAGE_KEYS = ("image", "left_wrist_image", "right_wrist_image")
PARQUET_COLUMNS = (*IMAGE_KEYS, "state", "frame_index", "episode_index", "task_index")
PRE_TASK = "open both lids"
EVIDENCE_TASK = "inspect both bins"
WAIT_TASKS = {"left": "wait; target bin is left", "right": "wait; target bin is right"}
PROMPTS = {"banana": "find the banana", "grey_pepper_box": "find the grey pepper box"}
POOLS = ("pre", "evidence", "post")
# Episode-OOF folds and permutation protocol (the Gate-B convention, kept small enough for
# an 8-episode development split; the report records the exact values used).
OOF_FOLDS = 4
PERMUTATIONS = 500
PERMUTATION_SEED = 41
RIDGE_LAMBDA = 1.0


class Stage1EvalError(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class EpisodeSpec:
    stable_id: str
    episode_index: int
    collection: str
    object_name: str
    prompt: str
    target_side: str
    split: str
    visibility: tuple[int, int]
    d_window: tuple[int, int]
    fact_targets: tuple[int, ...]


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_dumps(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def _load_manifest_and_facts(
    manifest_path: pathlib.Path, facts_path: pathlib.Path, *, split: str
) -> tuple[list[EpisodeSpec], dict[str, str]]:
    manifest = json.loads(manifest_path.read_text())
    facts = json.loads(facts_path.read_text())
    if manifest.get("review_status") != "frozen":
        raise Stage1EvalError("episode manifest is not frozen")
    if facts.get("source_manifest_sha256") != _sha256_file(manifest_path):
        raise Stage1EvalError("fact sidecar was not derived from this manifest")
    provenance = {
        "manifest_sha256": _sha256_file(manifest_path),
        "fact_labels_sha256": _sha256_file(facts_path),
        "fact_labels_content_sha256": facts["content_sha256"],
    }
    episodes = []
    for record in manifest["episodes"]:
        if not record.get("include") or record.get("split") != split:
            continue
        fact_record = facts["episodes"].get(record["stable_id"])
        if fact_record is None:
            raise Stage1EvalError(f"fact sidecar is missing {record['stable_id']!r}")
        visibility = record["e_visibility"]
        d_valid = record["d_valid"]
        episodes.append(
            EpisodeSpec(
                stable_id=record["stable_id"],
                episode_index=int(record["episode_index"]),
                collection=str(record["collection"]),
                object_name=str(record["object"]),
                prompt=str(record["prompt"]),
                target_side=str(record["target_side"]),
                split=str(record["split"]),
                visibility=(int(visibility["first_valid_visible_frame"]), int(visibility["last_clean_visible_frame"])),
                d_window=(int(d_valid["start"]), int(d_valid["end"])),
                fact_targets=tuple(int(x) for x in fact_record["fact_targets"]),
            )
        )
    if not episodes:
        raise Stage1EvalError(f"no included episodes in split {split!r}")
    return episodes, provenance


def _decode_image(value: Any, *, field: str, frame: int) -> np.ndarray:
    from PIL import Image

    payload = value.get("bytes") if isinstance(value, dict) else value
    if not isinstance(payload, bytes):
        raise Stage1EvalError(f"{field} frame {frame} does not contain inline image bytes")
    with Image.open(io.BytesIO(payload)) as image:
        decoded = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if decoded.shape != (480, 640, 3):
        raise Stage1EvalError(f"{field} frame {frame} has unexpected shape {decoded.shape}")
    return decoded


def _select_frames(frame_indices: np.ndarray, *, count: int, seed: int) -> np.ndarray:
    if frame_indices.size == 0:
        return frame_indices
    if frame_indices.size <= count:
        return np.sort(frame_indices)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(frame_indices, size=count, replace=False))


class Stage1Runtime:
    """Checkpointed stage-1 model + the exact inference-side transform composition."""

    def __init__(self, *, params_path: pathlib.Path, batch_size: int):
        import jax
        import jax.numpy as jnp

        from openpi.models import model as model_lib
        from openpi.shared import nnx_utils
        from openpi.training import config as config_lib
        import openpi.transforms as transforms

        self._jax = jax
        self._jnp = jnp
        self._model_lib = model_lib
        self.batch_size = batch_size

        config = config_lib.get_config(CONFIG_NAME)
        if not getattr(config.model, "memory_v4_dual_bank", False):
            raise Stage1EvalError("registered stage-1 config is no longer dual-bank")
        self.num_fact_targets = config.model.memory_fact_targets
        self.unknown_class = self.num_fact_targets - 1
        self.write_conf = float(config.model.memory_fact_write_conf)
        params = model_lib.restore_params(params_path, restore_type=np.ndarray)
        self.parameter_tree_sha256 = _parameter_tree_sha256_of(params)
        model = config.model.load(params)
        model.eval()
        data_config = config.data.create(config.assets_dirs, config.model)
        if data_config.norm_stats is None:
            raise Stage1EvalError("the registered config did not resolve its pinned norm stats")
        # Hash the exact pinned file the config loaded (the AssetsConfig override directory).
        norm_stats_file = (
            pathlib.Path(config.data.assets.assets_dir or config.assets_dirs)
            / (data_config.asset_id or data_config.repo_id)
            / "norm_stats.json"
        )
        self.norm_stats_sha256 = _sha256_file(norm_stats_file)
        self._data_input_transform = transforms.compose(data_config.data_transforms.inputs)
        self._normalize = transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm)
        self._model_transform = transforms.compose(data_config.model_transforms.inputs)
        self._probe = nnx_utils.module_jit(model.v4_fact_probe_step)

    def observation(self, row: Mapping[str, Any], *, prompt: str, state_override: np.ndarray | None):
        frame = int(row["frame_index"])
        state = np.asarray(row["state"], dtype=np.float32)
        if state_override is not None:
            state = np.asarray(state_override, dtype=np.float32)
        transformed = self._data_input_transform(
            {
                "observation/image": _decode_image(row["image"], field="image", frame=frame),
                "observation/left_wrist_image": _decode_image(
                    row["left_wrist_image"], field="left_wrist_image", frame=frame
                ),
                "observation/right_wrist_image": _decode_image(
                    row["right_wrist_image"], field="right_wrist_image", frame=frame
                ),
                "observation/state": state,
                "prompt": prompt,
            }
        )
        transformed = self._normalize(transformed)
        transformed = self._model_transform(transformed)
        return self._jax.tree.map(lambda value: self._jnp.asarray(value)[None, ...], transformed)

    def probe_batch(self, items: Sequence[Any]) -> dict[str, np.ndarray]:
        real = len(items)
        if not 0 < real <= self.batch_size:
            raise Stage1EvalError("probe batch is empty or oversized")
        padded = list(items) + [items[-1]] * (self.batch_size - real)
        stacked = self._jax.tree.map(lambda *values: self._jnp.concatenate(values, axis=0), *padded)
        observation = self._model_lib.Observation.from_dict(stacked)
        outputs = self._probe(observation)
        return {
            name: np.asarray(self._jax.device_get(outputs[name][:real]))
            for name in ("fact_slots", "fact_logits", "fact_probs", "fact_confidence", "fact_predicted")
        }


def _parameter_tree_sha256_of(params: Any) -> str:
    from openpi.training import weight_loaders

    return weight_loaders.parameter_tree_sha256(params)


def _read_rows(dataset_root: pathlib.Path, episode: EpisodeSpec, frames: np.ndarray) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    if frames.size == 0:
        return []
    path = dataset_root / "data" / "chunk-000" / f"episode_{episode.episode_index:06d}.parquet"
    table = pq.read_table(
        path,
        columns=list(PARQUET_COLUMNS),
        filters=[("frame_index", "in", [int(x) for x in frames])],
    )
    rows = {int(row["frame_index"]): row for row in table.to_pylist()}
    missing = [int(x) for x in frames if int(x) not in rows]
    if missing:
        raise Stage1EvalError(f"{episode.stable_id} is missing requested frames {missing[:5]}")
    for frame, row in rows.items():
        if int(row["episode_index"]) != episode.episode_index:
            raise Stage1EvalError(f"{episode.stable_id} frame {frame} carries a foreign episode_index")
    return [rows[int(x)] for x in frames]


def _episode_pool_frames(
    task_names: Mapping[int, str],
    dataset_root: pathlib.Path,
    episode: EpisodeSpec,
    *,
    frames_per_pool: int,
    seed: int,
) -> dict[str, np.ndarray]:
    import pyarrow.parquet as pq

    path = dataset_root / "data" / "chunk-000" / f"episode_{episode.episode_index:06d}.parquet"
    table = pq.read_table(path, columns=["frame_index", "task_index"])
    frame_index = np.asarray(table["frame_index"], dtype=np.int64)
    task_index = np.asarray(table["task_index"], dtype=np.int64)
    tasks = np.asarray([task_names[int(t)] for t in task_index])
    visibility_lo, visibility_hi = episode.visibility
    d_lo, d_hi = episode.d_window
    wait_task = WAIT_TASKS[episode.target_side]
    pools = {
        "pre": frame_index[tasks == PRE_TASK],
        "evidence": frame_index[
            (tasks == EVIDENCE_TASK) & (frame_index >= visibility_lo) & (frame_index <= visibility_hi)
        ],
        "post": frame_index[(tasks == wait_task) & (frame_index >= d_lo) & (frame_index <= d_hi)],
    }
    return {
        name: _select_frames(np.asarray(values), count=frames_per_pool, seed=seed + hash(name) % 1000)
        for name, values in pools.items()
    }


def _load_task_names(dataset_root: pathlib.Path) -> dict[int, str]:
    names: dict[int, str] = {}
    for line in (dataset_root / "meta" / "tasks.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            names[int(value["task_index"])] = str(value["task"])
    return names


# ------------------------------------------------------------------------------------------
# Episode-OOF ridge probe with stratified permutation null (the Gate-B construction, sized
# for the split at hand).
# ------------------------------------------------------------------------------------------


def _fold_of_episode(stable_ids: Sequence[str]) -> dict[str, int]:
    ranked = sorted(set(stable_ids), key=lambda sid: hashlib.sha256(f"v4stage1:{sid}".encode()).hexdigest())
    return {sid: index % OOF_FOLDS for index, sid in enumerate(ranked)}


@dataclasses.dataclass(frozen=True)
class _PreparedFold:
    """Per-fold dual-ridge factorization, label-independent (the Gate-B reuse pattern).

    The slot features are wide (fact_slots x width = 16k dims), so the primal normal
    equations are infeasible; the dual form needs only the n_train x n_train kernel, and its
    Cholesky factor is reused across every label permutation -- each permutation costs two
    triangular solves and one matvec instead of a fresh factorization.
    """

    test_index: np.ndarray
    train_index: np.ndarray
    cho_factor: tuple[np.ndarray, bool]
    cross_kernel: np.ndarray  # [n_test, n_train] = X_test @ X_train.T


def _prepare_ridge_folds(features: np.ndarray, folds: np.ndarray) -> tuple[_PreparedFold, ...]:
    import scipy.linalg

    x = features.astype(np.float64)
    x = (x - x.mean(axis=0)) / np.maximum(x.std(axis=0), 1e-6)
    prepared = []
    for fold in np.unique(folds):
        test = folds == fold
        train = ~test
        if not np.any(test) or not np.any(train):
            raise Stage1EvalError("degenerate OOF fold (empty train or test)")
        xt = x[train]
        kernel = xt @ xt.T + RIDGE_LAMBDA * np.eye(xt.shape[0])
        prepared.append(
            _PreparedFold(
                test_index=np.nonzero(test)[0],
                train_index=np.nonzero(train)[0],
                cho_factor=scipy.linalg.cho_factor(kernel),
                cross_kernel=x[test] @ xt.T,
            )
        )
    return tuple(prepared)


def _prepared_oof_scores(prepared: Sequence[_PreparedFold], labels: np.ndarray) -> np.ndarray:
    import scipy.linalg

    scores = np.zeros(len(labels), dtype=np.float64)
    y = labels.astype(np.float64) * 2.0 - 1.0
    for fold in prepared:
        if len(np.unique(labels[fold.train_index])) < 2:
            raise Stage1EvalError("degenerate OOF fold (missing class in train)")
        alpha = scipy.linalg.cho_solve(fold.cho_factor, y[fold.train_index])
        scores[fold.test_index] = fold.cross_kernel @ alpha
    return scores


def _episode_balanced_accuracy(
    scores: np.ndarray, labels: np.ndarray, episode_ids: np.ndarray
) -> float:
    episode_scores = {}
    episode_labels = {}
    for sid in np.unique(episode_ids):
        mask = episode_ids == sid
        episode_scores[sid] = float(np.mean(scores[mask]))
        episode_labels[sid] = int(labels[mask][0])
    predictions = {sid: int(score > 0) for sid, score in episode_scores.items()}
    accuracies = []
    for side in (0, 1):
        side_ids = [sid for sid, label in episode_labels.items() if label == side]
        if not side_ids:
            raise Stage1EvalError("leak probe requires both sides present")
        accuracies.append(float(np.mean([predictions[sid] == side for sid in side_ids])))
    return float(np.mean(accuracies))


def leak_probe(
    features: np.ndarray,
    labels: np.ndarray,
    episode_ids: np.ndarray,
    collections: np.ndarray,
) -> dict[str, Any]:
    """Balanced episode-level OOF accuracy + stratified-permutation p-value."""
    stable = np.asarray(episode_ids)
    fold_map = _fold_of_episode([str(x) for x in stable])
    folds = np.asarray([fold_map[str(x)] for x in stable], dtype=np.int64)
    prepared = _prepare_ridge_folds(features, folds)
    observed = _episode_balanced_accuracy(_prepared_oof_scores(prepared, labels), labels, stable)
    rng = np.random.default_rng(PERMUTATION_SEED)
    episode_order = sorted({str(x) for x in stable})
    episode_label = {str(sid): int(labels[stable == sid][0]) for sid in episode_order}
    episode_collection = {str(sid): str(collections[stable == sid][0]) for sid in episode_order}
    null = np.zeros(PERMUTATIONS, dtype=np.float64)
    for index in range(PERMUTATIONS):
        permuted = dict(episode_label)
        for collection in sorted(set(episode_collection.values())):
            ids = [sid for sid in episode_order if episode_collection[sid] == collection]
            values = [episode_label[sid] for sid in ids]
            shuffled = rng.permutation(values)
            for sid, value in zip(ids, shuffled, strict=True):
                permuted[sid] = int(value)
        permuted_labels = np.asarray([permuted[str(x)] for x in stable], dtype=np.int64)
        if len(np.unique(permuted_labels)) < 2:
            null[index] = 0.5
            continue
        null[index] = _episode_balanced_accuracy(
            _prepared_oof_scores(prepared, permuted_labels), permuted_labels, stable
        )
    p_value = float((np.sum(null >= observed) + 1) / (PERMUTATIONS + 1))
    return {
        "balanced_accuracy": observed,
        "permutation_p": p_value,
        "permutations": PERMUTATIONS,
        "null_mean": float(null.mean()),
        "null_p95": float(np.quantile(null, 0.95)),
        "folds": OOF_FOLDS,
    }


def main(argv: Sequence[str] | None = None) -> None:
    from openpi.shared import project_paths

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=pathlib.Path, required=True, help="stage-1 checkpoint params dir")
    parser.add_argument("--split", choices=("development", "train"), default="development")
    parser.add_argument("--frames-per-pool", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)

    manifest_path = project_paths.project_path(project_paths.V35_FROZEN_MANIFEST)
    facts_path = project_paths.project_path(project_paths.V4_FACT_LABELS)
    dataset_root = project_paths.project_path(project_paths.V35_DATASET_DIR)
    episodes, provenance = _load_manifest_and_facts(manifest_path, facts_path, split=args.split)
    if args.split == "train":
        leak_episodes = episodes
    else:
        leak_episodes, _ = _load_manifest_and_facts(manifest_path, facts_path, split="train")
    task_names = _load_task_names(dataset_root)

    output_dir = args.output_dir
    report_path = output_dir / "stage1_eval.json"
    features_path = output_dir / "stage1_features.npz"
    if report_path.exists() or features_path.exists():
        raise Stage1EvalError(f"output artifacts already exist under {output_dir} (create-only)")
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime = Stage1Runtime(params_path=args.params, batch_size=args.batch_size)

    rows_meta: list[dict[str, Any]] = []
    outputs_by_condition: dict[str, list[dict[str, np.ndarray]]] = {"real": [], "prompt_swap": [], "state_neutral": []}

    # Pass 1: collect raw rows per pool (and the evidence-state mean for the neutral probe).
    collected: list[tuple[EpisodeSpec, str, dict[str, Any]]] = []
    for episode in episodes:
        pools = _episode_pool_frames(
            task_names, dataset_root, episode, frames_per_pool=args.frames_per_pool, seed=args.seed
        )
        for pool in POOLS:
            for row in _read_rows(dataset_root, episode, pools[pool]):
                collected.append((episode, pool, row))
    if not collected:
        raise Stage1EvalError("no frames were collected")
    # Leak cohort: pre/post frames over the train split (see module docstring). When the
    # metric split IS train, the same rows serve both purposes.
    if args.split == "train":
        leak_collected = [(episode, pool, row) for episode, pool, row in collected if pool != "evidence"]
    else:
        leak_collected = []
        for episode in leak_episodes:
            pools = _episode_pool_frames(
                task_names, dataset_root, episode, frames_per_pool=args.frames_per_pool, seed=args.seed
            )
            for pool in ("pre", "post"):
                for row in _read_rows(dataset_root, episode, pools[pool]):
                    leak_collected.append((episode, pool, row))
    evidence_states = np.stack(
        [np.asarray(row["state"], dtype=np.float32) for _, pool, row in collected if pool == "evidence"]
    )
    neutral_state = evidence_states.mean(axis=0)

    def other_prompt(episode: EpisodeSpec) -> str:
        others = [prompt for name, prompt in PROMPTS.items() if name != episode.object_name]
        if len(others) != 1:
            raise Stage1EvalError(f"cannot form a prompt swap for object {episode.object_name!r}")
        return others[0]

    # Pass 2: batched probes per condition, plus the train-split leak cohort (real prompts).
    def run_condition(rows, *, prompt_swap: bool, state_neutral: bool) -> list[dict[str, np.ndarray]]:
        batches: list[dict[str, np.ndarray]] = []
        pending_items: list[Any] = []
        for episode, _pool, row in rows:
            prompt = other_prompt(episode) if prompt_swap else episode.prompt
            state_override = neutral_state if state_neutral else None
            pending_items.append(runtime.observation(row, prompt=prompt, state_override=state_override))
            if len(pending_items) == runtime.batch_size:
                batches.append(runtime.probe_batch(pending_items))
                pending_items = []
        if pending_items:
            batches.append(runtime.probe_batch(pending_items))
        return batches

    evidence_rows = [(episode, pool, row) for episode, pool, row in collected if pool == "evidence"]
    outputs_by_condition["real"] = run_condition(collected, prompt_swap=False, state_neutral=False)
    outputs_by_condition["prompt_swap"] = run_condition(evidence_rows, prompt_swap=True, state_neutral=False)
    outputs_by_condition["state_neutral"] = run_condition(evidence_rows, prompt_swap=False, state_neutral=True)
    leak_batches = run_condition(leak_collected, prompt_swap=False, state_neutral=False)

    for episode, pool, row in collected:
        rows_meta.append(
            {
                "stable_id": episode.stable_id,
                "collection": episode.collection,
                "object": episode.object_name,
                "target_side": episode.target_side,
                "pool": pool,
                "frame_index": int(row["frame_index"]),
                "fact_targets": list(episode.fact_targets),
            }
        )

    def stack(condition: str, name: str) -> np.ndarray:
        return np.concatenate([batch[name] for batch in outputs_by_condition[condition]], axis=0)

    real = {name: stack("real", name) for name in ("fact_slots", "fact_probs", "fact_confidence", "fact_predicted")}
    pools_arr = np.asarray([meta["pool"] for meta in rows_meta])
    stable_arr = np.asarray([meta["stable_id"] for meta in rows_meta])
    collection_arr = np.asarray([meta["collection"] for meta in rows_meta])
    targets_arr = np.asarray([meta["fact_targets"] for meta in rows_meta], dtype=np.int64)
    real_slots = min(targets_arr.shape[1], real["fact_predicted"].shape[1])

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "config_name": CONFIG_NAME,
        "split": args.split,
        "frames_per_pool": args.frames_per_pool,
        "seed": args.seed,
        "episodes": len(episodes),
        "frames": len(rows_meta),
        "parameter_tree_sha256": runtime.parameter_tree_sha256,
        "norm_stats_sha256": runtime.norm_stats_sha256,
        **provenance,
        "gates": {},
        "metrics": {},
    }

    # --- evidence accuracy + abstention ---
    evidence = pools_arr == "evidence"
    predicted = real["fact_predicted"][:, :real_slots]
    confident = real["fact_confidence"][:, :real_slots] >= runtime.write_conf
    correct = predicted == targets_arr[:, :real_slots]
    metrics: dict[str, Any] = {}
    metrics["evidence_accuracy_per_slot"] = np.mean(correct[evidence], axis=0).tolist()
    metrics["evidence_write_eligible_rate"] = float(
        np.mean(confident[evidence] & (predicted[evidence] != runtime.unknown_class))
    )
    per_collection = {}
    for collection in sorted(set(collection_arr)):
        mask = evidence & (collection_arr == collection)
        per_collection[collection] = float(np.mean(correct[mask])) if np.any(mask) else None
    metrics["evidence_accuracy_per_collection"] = per_collection
    for pool in ("pre", "post"):
        mask = pools_arr == pool
        metrics[f"{pool}_abstention_rate"] = float(np.mean(predicted[mask] == runtime.unknown_class))
        metrics[f"{pool}_mean_confidence"] = float(np.mean(real["fact_confidence"][mask]))

    # --- interventions (evidence frames only, aligned by construction) ---
    for condition in ("prompt_swap", "state_neutral"):
        swapped = np.concatenate(
            [batch["fact_predicted"] for batch in outputs_by_condition[condition]], axis=0
        )[:, :real_slots]
        metrics[f"{condition}_agreement"] = float(np.mean(swapped == predicted[evidence]))

    # --- leak probes on slot-0 (banana location) features, pre and post pools, TRAIN split ---
    leak_slots = np.concatenate([batch["fact_slots"] for batch in leak_batches], axis=0)
    leak_pools = np.asarray([pool for _, pool, _ in leak_collected])
    leak_stable = np.asarray([episode.stable_id for episode, _, _ in leak_collected])
    leak_collections = np.asarray([episode.collection for episode, _, _ in leak_collected])
    leak_banana_side = np.asarray([episode.fact_targets[0] for episode, _, _ in leak_collected], dtype=np.int64)
    flat_features = leak_slots.reshape(leak_slots.shape[0], -1)
    leak = {"split": "train", "episodes": len({str(x) for x in leak_stable})}
    for pool in ("pre", "post"):
        mask = leak_pools == pool
        leak[pool] = leak_probe(
            flat_features[mask], leak_banana_side[mask], leak_stable[mask], leak_collections[mask]
        )
    metrics["leak_probe"] = leak
    report["metrics"] = metrics

    # --- preregistered gates ---
    gates = {
        "evidence_accuracy": {
            "threshold": 0.9,
            "value": float(np.mean(correct[evidence])),
        },
        "pre_abstention": {"threshold": 0.95, "value": metrics["pre_abstention_rate"]},
        "post_abstention": {"threshold": 0.95, "value": metrics["post_abstention_rate"]},
        "prompt_swap_agreement": {"threshold": 0.95, "value": metrics["prompt_swap_agreement"]},
        "state_neutral_agreement": {"threshold": 0.95, "value": metrics["state_neutral_agreement"]},
        "pre_leak_p": {"threshold": 0.05, "value": leak["pre"]["permutation_p"], "direction": "above"},
        "post_leak_p": {"threshold": 0.05, "value": leak["post"]["permutation_p"], "direction": "above"},
    }
    for gate in gates.values():
        direction = gate.get("direction", "at_least")
        gate["passes"] = (
            bool(gate["value"] > gate["threshold"])
            if direction == "above"
            else bool(gate["value"] >= gate["threshold"])
        )
    report["gates"] = gates
    report["passes"] = bool(all(gate["passes"] for gate in gates.values()))

    np.savez_compressed(
        features_path,
        fact_slots=real["fact_slots"].astype(np.float32),
        fact_probs=real["fact_probs"].astype(np.float32),
        fact_predicted=real["fact_predicted"].astype(np.int32),
        fact_targets=targets_arr.astype(np.int32),
        pool=pools_arr,
        stable_id=stable_arr,
        collection=collection_arr,
    )
    report["features_sha256"] = _sha256_file(features_path)
    body = _canonical_dumps(report)
    report["report_sha256"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
    report_path.write_text(_canonical_dumps(report))
    print(f"wrote {report_path}")
    print(f"PASS={report['passes']}  gates:")
    for name, gate in report["gates"].items():
        print(f"  {name}: value={gate['value']:.4f} threshold={gate['threshold']} passes={gate['passes']}")


if __name__ == "__main__":
    sys.exit(main())
