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
