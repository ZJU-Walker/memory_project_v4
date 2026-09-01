"""Tests for the Stage-1 battery's pure selection/statistics functions."""

import hashlib
import json

import numpy as np
import pytest

import v4_stage1_eval as battery


def _synthetic_cohort(*, informative: bool, episodes: int = 8, frames: int = 10, dims: int = 24):
    rng = np.random.default_rng(7)
    stable, labels, collections, features = [], [], [], []
    for index in range(episodes):
        label = index % 2
        collection = "0830" if index < episodes // 2 else "0831"
        for _ in range(frames):
            noise = rng.normal(size=dims)
            if informative:
                noise[0] += 3.0 * (label * 2 - 1)
            features.append(noise)
            stable.append(f"ep{index}")
            labels.append(label)
            collections.append(collection)
    return (
        np.asarray(features, dtype=np.float64),
        np.asarray(labels, dtype=np.int64),
        np.asarray(stable),
        np.asarray(collections),
    )


def test_leak_probe_detects_an_informative_feature():
    # Sized like the real leak cohort (train split, tens of episodes): with only 8 episodes
    # the stratified permutation space is ~36 arrangements and the p floor sits near the
    # threshold, which is exactly why the battery probes leakage on the train split.
    result = battery.leak_probe(*_synthetic_cohort(informative=True, episodes=16))
    assert result["balanced_accuracy"] == 1.0
    assert result["permutation_p"] <= 0.05


def test_leak_probe_is_at_chance_on_noise():
    result = battery.leak_probe(*_synthetic_cohort(informative=False))
    assert result["permutation_p"] > 0.05
    assert abs(result["null_mean"] - 0.5) < 0.15


def test_select_frames_is_deterministic_sorted_and_bounded():
    frames = np.arange(100, 200)
    a = battery._select_frames(frames, count=12, seed=3)
    b = battery._select_frames(frames, count=12, seed=3)
    np.testing.assert_array_equal(a, b)
    assert len(a) == 12
    assert np.all(np.diff(a) > 0)
    short = battery._select_frames(np.asarray([5, 3]), count=12, seed=3)
    np.testing.assert_array_equal(short, [3, 5])


def test_manifest_fact_cross_check_fails_closed(tmp_path):
    manifest = {
        "review_status": "frozen",
        "episodes": [
            {
                "stable_id": "a/demo1",
                "episode_index": 0,
                "collection": "0830",
                "object": "banana",
                "prompt": "find the banana",
                "target_side": "left",
                "split": "development",
                "include": True,
                "e_visibility": {"first_valid_visible_frame": 10, "last_clean_visible_frame": 20},
                "d_valid": {"start": 30, "end": 40},
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    facts = {
        "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "content_sha256": "x",
        "episodes": {"a/demo1": {"split": "development", "fact_targets": [0, 1]}},
    }
    facts_path = tmp_path / "facts.json"
    facts_path.write_text(json.dumps(facts))

    episodes, provenance = battery._load_manifest_and_facts(manifest_path, facts_path, split="development")
    assert len(episodes) == 1
    assert episodes[0].fact_targets == (0, 1)
    assert provenance["manifest_sha256"] == facts["source_manifest_sha256"]

    with pytest.raises(battery.Stage1EvalError, match="no included episodes"):
        battery._load_manifest_and_facts(manifest_path, facts_path, split="train")

    manifest["episodes"][0]["prompt"] = "tampered"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(battery.Stage1EvalError, match="not derived from this manifest"):
        battery._load_manifest_and_facts(manifest_path, facts_path, split="development")
