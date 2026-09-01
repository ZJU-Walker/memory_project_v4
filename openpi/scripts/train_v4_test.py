"""v4 host-side training contracts: the light v4 protocol, graft overlays, gates."""

import dataclasses
import pathlib

import numpy as np
import pytest

import openpi.models.model as _model
import openpi.training.config as _config
import train as train_script


def _stage1_config() -> _config.TrainConfig:
    return _config.get_config("pi05_yam_mem_v4_stage1")


def _full_v4_config() -> _config.TrainConfig:
    return _config.get_config("pi05_yam_mem_v4")


def test_every_registered_v4_config_is_a_v4_protocol_run():
    assert train_script._is_v4_run(_stage1_config())
    assert train_script._is_v4_run(_full_v4_config())
    assert not train_script._is_v4_run(_config.get_config("pi05_yam_mem_v35"))
    # The Stage-1 shape detector still identifies the fact-head-only recipe.
    assert train_script._is_v4_stage1_config(_stage1_config())
    assert not train_script._is_v4_stage1_config(_full_v4_config())


def test_v4_runs_pass_readiness_without_the_v35_seal_and_v35_runs_still_lock():
    train_script._validate_v35_training_ready(_stage1_config())
    train_script._validate_v35_training_ready(_full_v4_config())
    with pytest.raises(ValueError, match="calibration"):
        train_script._validate_v35_training_ready(_config.get_config("pi05_yam_mem_v35"))


def test_v4_contract_rejects_missing_sidecar_pin_ema_and_bad_graft_sources(tmp_path):
    config = _full_v4_config()
    with pytest.raises(ValueError, match="ema_decay"):
        train_script._validate_v4_run(dataclasses.replace(config, ema_decay=0.99))
    # An unpinned sidecar cannot even be constructed: DataConfig enforces the pin first.
    with pytest.raises(ValueError, match="pinned SHA256"):
        dataclasses.replace(config.data.base_config, memory_v4_fact_labels_sha256=None)
    with pytest.raises(ValueError, match="does not exist"):
        train_script._validate_v4_run(
            dataclasses.replace(config, v4_graft_sources=((r".*fact_.*", str(tmp_path / "missing")),))
        )
    # Requiring v4_protocol on a non-dual-bank model is refused too.
    v35 = _config.get_config("pi05_yam_mem_v35")
    with pytest.raises(ValueError, match="memory_v4_dual_bank"):
        train_script._validate_v4_run(dataclasses.replace(v35, v4_protocol=True))


def test_graft_overlay_replaces_only_matching_leaves_and_never_casts(monkeypatch, tmp_path):
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    shape = {
        "fact_logit_head": {"kernel": np.zeros((4, 3), np.float32), "bias": np.zeros((3,), np.float32)},
        "PaliGemma": {"llm": {"w": np.zeros((2, 2), np.float32)}},
    }
    partial = {"PaliGemma": {"llm": {"w": np.ones((2, 2), np.float32)}}}
    source = {
        "fact_logit_head": {"kernel": np.full((4, 3), 7.0, np.float32), "bias": np.full((3,), 2.0, np.float32)},
        "PaliGemma": {"llm": {"w": np.full((2, 2), 9.0, np.float32)}},
    }
    monkeypatch.setattr(_model, "restore_params", lambda path, restore_type=None: source)
    config = dataclasses.replace(_full_v4_config(), v4_graft_sources=((r"fact_logit_head/.*", str(source_dir)),))
    merged = train_script._apply_v4_graft_sources(config, partial, shape)
    np.testing.assert_array_equal(merged["fact_logit_head"]["kernel"], 7.0)
    np.testing.assert_array_equal(merged["fact_logit_head"]["bias"], 2.0)
    # The backbone leaf keeps the main loader's value: the regex did not match it.
    np.testing.assert_array_equal(merged["PaliGemma"]["llm"]["w"], 1.0)

    # dtype drift is refused, and a regex matching nothing is refused.
    bad = {"fact_logit_head": {"kernel": np.full((4, 3), 7.0, np.float16), "bias": source["fact_logit_head"]["bias"]}}
    monkeypatch.setattr(_model, "restore_params", lambda path, restore_type=None: bad)
    with pytest.raises(ValueError, match="never cast"):
        train_script._apply_v4_graft_sources(config, partial, shape)
    monkeypatch.setattr(_model, "restore_params", lambda path, restore_type=None: source)
    with pytest.raises(ValueError, match="matched no leaf"):
        train_script._apply_v4_graft_sources(
            dataclasses.replace(config, v4_graft_sources=((r"nonexistent/.*", str(source_dir)),)), partial, shape
        )


def test_inject_gate_filter_covers_both_banks_gates_and_nothing_else():
    pattern = train_script.MEMORY_INJECT_GATE_FILTER.pattern
    assert pattern.fullmatch("memory_inject_w")
    assert pattern.fullmatch("memory_sem_inject_w")
    assert not pattern.fullmatch("memory_sem_slot_embedding")
    assert not pattern.fullmatch("memory_gate")


def test_v4_fact_info_exposes_every_raw_numerator_and_denominator():
    chunked = {
        key: np.zeros(())
        for key in (
            "v4_fact_ce_class_sum",
            "v4_fact_count_class",
            "v4_fact_correct_class",
            "v4_fact_read_ce_sum",
            "v4_fact_read_count",
            "v4_fact_read_correct",
            "v4_sem_commit_count",
            "v4_sem_write_eligible_count",
            "v4_sem_degenerate_count",
            "v4_sem_final_residual_sum",
            "v4_sem_final_residual_max",
            "v4_sem_raw_read_rms_sum",
            "v4_sem_injected_pre_cast_rms_sum",
            "v4_sem_injected_post_cast_rms_sum",
        )
    }
    info = train_script._v4_fact_info(chunked)
    assert set(info) == {f"diagnostic/{key}" for key in chunked}
