"""Tests for the v4 fact-label derivation (v4_build_fact_labels.py)."""

import hashlib
import json
import pathlib

import pytest

import v4_build_fact_labels as builder


def _manifest(tmp_path: pathlib.Path, episodes: list[dict], **overrides) -> pathlib.Path:
    payload = {
        "review_status": "frozen",
        "dataset_version": "v36",
        "episodes": episodes,
    }
    payload.update(overrides)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload))
    return path


def _episode(stable_id: str, obj: str, side: str, *, split: str = "train", include: bool = True) -> dict:
    return {
        "stable_id": stable_id,
        "object": obj,
        "target_side": side,
        "split": split,
        "include": include,
    }


def test_prompted_object_gets_its_side_and_the_other_object_the_opposite():
    facts = builder.derive_episode_facts(_episode("a/demo1", "banana", "left"))
    # slot 0 = banana -> left_bin (0); slot 1 = grey_pepper_box -> right_bin (1)
    assert facts.fact_targets == (0, 1)
    facts = builder.derive_episode_facts(_episode("a/demo2", "grey_pepper_box", "left"))
    # prompted grey box at left; banana therefore right
    assert facts.fact_targets == (1, 0)
    facts = builder.derive_episode_facts(_episode("a/demo3", "banana", "right"))
    assert facts.fact_targets == (1, 0)


def test_unknown_vocabulary_fails_loudly():
    with pytest.raises(ValueError, match="unknown object"):
        builder.derive_episode_facts(_episode("a/demo1", "apple", "left"))
    with pytest.raises(ValueError, match="unknown target_side"):
        builder.derive_episode_facts(_episode("a/demo1", "banana", "middle"))


def test_build_pins_manifest_hash_skips_excluded_and_is_self_hashed(tmp_path):
    episodes = [
        _episode("a/demo1", "banana", "left"),
        _episode("a/demo2", "grey_pepper_box", "right", split="development"),
        _episode("a/demo3", "banana", "right", include=False),
    ]
    path = _manifest(tmp_path, episodes)
    payload = builder.build_fact_labels(path)

    assert payload["num_episodes"] == 2
    assert "a/demo3" not in payload["episodes"]
    assert payload["episodes"]["a/demo1"] == {"split": "train", "fact_targets": [0, 1]}
    assert payload["episodes"]["a/demo2"] == {"split": "development", "fact_targets": [0, 1]}
    assert payload["source_manifest_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()

    # The self-hash covers exactly the payload minus its own hash field.
    body = dict(payload)
    content_sha = body.pop("content_sha256")
    canonical = json.dumps(body, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    assert content_sha == hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def test_build_rejects_unfrozen_manifests_and_duplicates(tmp_path):
    path = _manifest(tmp_path, [_episode("a/demo1", "banana", "left")], review_status="draft")
    with pytest.raises(ValueError, match="not frozen"):
        builder.build_fact_labels(path)

    path = _manifest(tmp_path, [_episode("a/demo1", "banana", "left")] * 2)
    with pytest.raises(ValueError, match="duplicate stable_id"):
        builder.build_fact_labels(path)


def test_real_frozen_manifest_derivation_when_present():
    real = pathlib.Path(__file__).resolve().parents[2] / "data" / "0830_0831_episode_manifest_v36_frozen.json"
    if not real.exists():
        pytest.skip("frozen v36 manifest not available on this machine")
    payload = builder.build_fact_labels(real)
    assert payload["num_episodes"] == 70
    splits = {"train": 0, "development": 0, "final_test": 0}
    for record in payload["episodes"].values():
        splits[record["split"]] += 1
        targets = record["fact_targets"]
        # Exactly one object per bin in every episode.
        assert sorted(targets) == [0, 1]
    assert splits == {"train": 54, "development": 8, "final_test": 8}
