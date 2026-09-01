"""v4 host-side training contracts: Stage-1 detection, calibration-lock carve-out, gates."""

import dataclasses

import numpy as np
import pytest

import openpi.shared.nnx_utils as nnx_utils
import openpi.training.config as _config
import train as train_script


def _stage1_config() -> _config.TrainConfig:
    return _config.get_config("pi05_yam_mem_v4_stage1")


def _full_v4_config() -> _config.TrainConfig:
    return _config.get_config("pi05_yam_mem_v4")


def test_stage1_shape_is_detected_and_full_v4_is_not():
    assert train_script._is_v4_stage1_config(_stage1_config())
    assert not train_script._is_v4_stage1_config(_full_v4_config())
    assert not train_script._is_v4_stage1_config(_config.get_config("pi05_yam_mem_v35"))


@pytest.mark.parametrize(
    "mutation",
    [
        # Any drift from the sealed Stage-1 shape falls back to the full calibration lock.
        lambda c: dataclasses.replace(c, freeze_filter=nnx_utils.PathRegex(r".*fact_.*")),
        lambda c: dataclasses.replace(
            c, model=dataclasses.replace(c.model, memory_sem_injection_gate_init=0.5)
        ),
        lambda c: dataclasses.replace(
            c, model=dataclasses.replace(c.model, memory_fact_read_loss_weight=0.1)
        ),
        lambda c: dataclasses.replace(
            c, model=dataclasses.replace(c.model, memory_fact_loss_weight=0.0)
        ),
    ],
)
def test_any_stage1_shape_drift_reinstates_the_calibration_lock(mutation):
    drifted = mutation(_stage1_config())
    assert not train_script._is_v4_stage1_config(drifted)
    with pytest.raises(ValueError, match="calibration"):
        train_script._validate_v35_training_ready(drifted)


def test_stage1_passes_training_readiness_without_a_calibration_artifact():
    train_script._validate_v35_training_ready(_stage1_config())


def test_full_v4_stays_behind_the_calibration_lock():
    with pytest.raises(ValueError, match="calibration"):
        train_script._validate_v35_training_ready(_full_v4_config())


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
