"""v4 data-pipeline contracts: fact-label sidecar loading and the MemoryV4FactLabels emit."""

import dataclasses
import hashlib
import json

import numpy as np
import pytest

import openpi.training.data_loader as data_loader
import openpi.transforms as transforms


def _sidecar_payload(manifest_sha: str) -> dict:
    payload = {
        "schema_version": "openpi.v4.fact-labels.v1",
        "source_manifest": "manifest.json",
        "source_manifest_sha256": manifest_sha,
        "dataset_version": "v36",
        "fact_slots": [
            {"slot": 0, "entity": "banana", "relation": "located_in"},
            {"slot": 1, "entity": "grey_pepper_box", "relation": "located_in"},
        ],
        "target_vocab": ["left_bin", "right_bin", "unknown"],
        "unknown_target": 2,
        "num_episodes": 2,
        "episodes": {
            "a/demo1": {"split": "train", "fact_targets": [0, 1]},
            "a/demo2": {"split": "train", "fact_targets": [1, 0]},
        },
    }
    body = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    payload["content_sha256"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
    return payload


@dataclasses.dataclass(frozen=True)
class _FakeDataConfig:
    memory_v4_fact_labels_path: str | None
    memory_v4_fact_labels_sha256: str | None
    memory_episode_manifest_sha256: str | None


def _write_sidecar(tmp_path, payload):
    path = tmp_path / "facts.json"
    path.write_text(json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n")
    return path


MANIFEST_SHA = "a" * 64


def test_sidecar_loads_and_aligns_by_stable_id(tmp_path):
    path = _write_sidecar(tmp_path, _sidecar_payload(MANIFEST_SHA))
    config = _FakeDataConfig(
        memory_v4_fact_labels_path=str(path),
        memory_v4_fact_labels_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        memory_episode_manifest_sha256=MANIFEST_SHA,
    )
    table = data_loader._load_v4_fact_labels(config, stable_ids=("a/demo2", "a/demo1"))
    np.testing.assert_array_equal(table, np.asarray([[1, 0], [0, 1]], dtype=np.int32))


def test_sidecar_rejects_wrong_pin_wrong_manifest_and_missing_episode(tmp_path):
    payload = _sidecar_payload(MANIFEST_SHA)
    path = _write_sidecar(tmp_path, payload)
    good_sha = hashlib.sha256(path.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        data_loader._load_v4_fact_labels(
            _FakeDataConfig(str(path), "b" * 64, MANIFEST_SHA), stable_ids=("a/demo1",)
        )
    with pytest.raises(ValueError, match="different manifest"):
        data_loader._load_v4_fact_labels(
            _FakeDataConfig(str(path), good_sha, "c" * 64), stable_ids=("a/demo1",)
        )
    with pytest.raises(ValueError, match="missing episode"):
        data_loader._load_v4_fact_labels(
            _FakeDataConfig(str(path), good_sha, MANIFEST_SHA), stable_ids=("a/demo9",)
        )
    with pytest.raises(ValueError, match="pinned SHA256"):
        data_loader._load_v4_fact_labels(
            _FakeDataConfig(str(path), None, MANIFEST_SHA), stable_ids=("a/demo1",)
        )

    # Tampering with the payload breaks the content self-hash even with a re-pinned file sha.
    tampered = dict(payload)
    tampered["episodes"] = {**payload["episodes"], "a/demo1": {"split": "train", "fact_targets": [1, 0]}}
    tampered_path = tmp_path / "tampered.json"
    tampered_path.write_text(json.dumps(tampered, sort_keys=True, indent=2, ensure_ascii=False) + "\n")
    with pytest.raises(ValueError, match="self-hash"):
        data_loader._load_v4_fact_labels(
            _FakeDataConfig(str(tampered_path), hashlib.sha256(tampered_path.read_bytes()).hexdigest(), MANIFEST_SHA),
            stable_ids=("a/demo1",),
        )


def test_fact_label_transform_pads_and_masks_with_write_steps():
    transform = transforms.MemoryV4FactLabels(num_fact_slots=4, num_fact_targets=3)
    data = {
        "episode_fact_targets": np.asarray([0, 1], dtype=np.int32),
        "seq_write_mask": np.asarray([True, False, True]),
    }
    out = transform(data)
    np.testing.assert_array_equal(out["seq_fact_labels"], np.asarray([0, 1, 2, 2], dtype=np.int32))
    expected = np.zeros((3, 4), dtype=bool)
    expected[0, :2] = True
    expected[2, :2] = True
    np.testing.assert_array_equal(out["seq_fact_observable"], expected)
    assert "episode_fact_targets" not in out


def test_fact_label_transform_uses_the_evidence_mask_under_write_every_step():
    transform = transforms.MemoryV4FactLabels(num_fact_slots=4, num_fact_targets=3)
    data = {
        "episode_fact_targets": np.asarray([0, 1], dtype=np.int32),
        # Write-every-step clock: every valid step may commit ...
        "seq_write_mask": np.asarray([True, True, True, False]),
        # ... but the fact is visible only on the single evidence frame.
        "_v4_evidence_write_mask": np.asarray([False, True, False, False]),
    }
    out = transform(data)
    expected = np.zeros((4, 4), dtype=bool)
    expected[1, :2] = True
    np.testing.assert_array_equal(out["seq_fact_observable"], expected)
    assert "_v4_evidence_write_mask" not in out
    np.testing.assert_array_equal(out["seq_write_mask"], np.asarray([True, True, True, False]))

    with pytest.raises(ValueError, match="subset of the clock write mask"):
        transform(
            {
                "episode_fact_targets": np.asarray([0], dtype=np.int32),
                "seq_write_mask": np.asarray([True, False]),
                "_v4_evidence_write_mask": np.asarray([False, True]),
            }
        )
    # Inference items drop the private selector too.
    out = transform({"_v4_evidence_write_mask": np.asarray([True]), "state": np.zeros(2)})
    assert "_v4_evidence_write_mask" not in out


def test_memory_v34_labels_write_every_step_keeps_observability_on_evidence_frames():
    """MemoryV34Labels(write_every_step=True) -> MemoryV4FactLabels: the clock covers every
    valid step, observability stays exactly the evidence-only write mask (Stage 4e fix; Stage
    4d r1 derived it from the clock and marked every frame observable)."""
    from openpi.training import data_loader_v35_test as v35t

    facts = transforms.MemoryV4FactLabels(num_fact_slots=2, num_fact_targets=3)

    def run(every_step: bool):
        labels = transforms.MemoryV34Labels(
            subtask_vocab=tuple(v35t.VOCAB.values()),
            evidence_subtasks=(v35t.INSPECT,),
            memory_required_subtasks=(v35t.WAIT_L, v35t.WAIT_R),
            write_every_step=every_step,
        )
        # Dense layout for both clocks so the comparison isolates the observability rule.
        build = transforms.BuildMemorySequence(
            stride=15, action_horizon=15, block_steps=4, occlusion_subtasks=(v35t.CLOSE,), allow_sparse_skip_o=False
        )
        item = build(v35t._sequence_item())
        item["episode_fact_targets"] = np.asarray([0], dtype=np.int32)
        return facts(labels(item))

    out, ref = run(True), run(False)
    np.testing.assert_array_equal(out["seq_write_mask"], out["seq_step_mask"])
    np.testing.assert_array_equal(out["seq_fact_observable"][:, 0], ref["seq_write_mask"])
    np.testing.assert_array_equal(out["seq_fact_observable"], ref["seq_fact_observable"])
    assert out["seq_write_mask"].sum() > out["seq_fact_observable"][:, 0].sum() > 0
    assert "_v4_evidence_write_mask" not in out
    assert "_v4_evidence_write_mask" not in ref
    # State validity and D anchors are unchanged by the clock choice.
    for key in ("seq_read_state_valid", "seq_read_credit_reachable", "seq_decision_mask", "seq_decay_gap_before"):
        np.testing.assert_array_equal(out[key], ref[key], err_msg=key)


def test_write_every_step_requires_dense_windows():
    """Stage 4e: the skip-O family is not exact once every tick writes, so the sampler/transform
    must hand MemoryV34Labels(write_every_step=True) dense windows only, and the transform
    refuses an analytic gap if one slips through."""
    from openpi.training import data_loader_v35_test as v35t

    def labels(every_step: bool):
        return transforms.MemoryV34Labels(
            subtask_vocab=tuple(v35t.VOCAB.values()),
            evidence_subtasks=(v35t.INSPECT,),
            memory_required_subtasks=(v35t.WAIT_L, v35t.WAIT_R),
            write_every_step=every_step,
        )

    def build(allow_sparse: bool):
        return transforms.BuildMemorySequence(
            stride=15, action_horizon=15, block_steps=4, occlusion_subtasks=(v35t.CLOSE,), allow_sparse_skip_o=allow_sparse
        )

    # The fixture's default start is a skip-O start under the parity split.
    sparse_item = build(True)(v35t._sequence_item())
    assert bool(sparse_item["seq_sparse_skip_o"])
    assert int(np.asarray(sparse_item["seq_decay_gap_before"]).max()) > 0
    labels(False)(dict(sparse_item))  # evidence-only clock: exact, accepted
    with pytest.raises(ValueError, match="dense memory windows"):
        labels(True)(dict(sparse_item))

    # allow_sparse_skip_o=False turns the same start into the natural (dense) layout.
    dense_item = build(False)(v35t._sequence_item())
    assert not bool(dense_item["seq_sparse_skip_o"])
    np.testing.assert_array_equal(dense_item["seq_decay_gap_before"], 0)
    out = labels(True)(dict(dense_item))
    np.testing.assert_array_equal(out["seq_write_mask"], out["seq_step_mask"])
    assert bool(out["seq_decision_mask"].any())

    # Layout helper: sparse family only when allowed.
    window = v35t._window()
    assert transforms.memory_critical_is_sparse(1, window)
    assert not transforms.memory_critical_is_sparse(1, window, allow_sparse=False)


def test_config_ties_sparse_family_to_the_write_clock():
    import dataclasses

    from openpi.training import config as _config

    base = _config.get_config("pi05_yam_mem_v4_stage4e").data.base_config
    assert base.memory_write_every_step and base.memory_sparse_skip_o_prob == 0.0
    with pytest.raises(ValueError, match="memory_sparse_skip_o_prob=0.0"):
        dataclasses.replace(base, memory_sparse_skip_o_prob=0.5)
    with pytest.raises(ValueError, match="50/50"):
        dataclasses.replace(base, memory_write_every_step=False)
    legacy = _config.get_config("pi05_yam_mem_v4_stage4c").data.base_config
    assert not legacy.memory_write_every_step and legacy.memory_sparse_skip_o_prob == 0.5


def test_fact_label_transform_passes_through_inference_items_and_fails_closed():
    transform = transforms.MemoryV4FactLabels(num_fact_slots=4, num_fact_targets=3)
    # Inference item: no sequence fields; the episode metadata is dropped, nothing emitted.
    out = transform({"episode_fact_targets": np.asarray([0, 1]), "state": np.zeros(3)})
    assert "seq_fact_labels" not in out
    assert "episode_fact_targets" not in out

    with pytest.raises(ValueError, match="require episode_fact_targets"):
        transform({"seq_write_mask": np.asarray([True])})
    with pytest.raises(ValueError, match="out of range"):
        transform(
            {
                "episode_fact_targets": np.asarray([5], dtype=np.int32),
                "seq_write_mask": np.asarray([True]),
            }
        )
