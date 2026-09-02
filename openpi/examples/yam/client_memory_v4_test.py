"""Pure-function tests for the v4 YAM client (metadata guard and overlay readout)."""

import importlib.util
import pathlib
import sys
import types
from unittest import mock

import pytest


def _load_client():
    """Import the client with the hardware/network modules stubbed (cv2, openpi_client, tyro)."""
    stubs = {
        "cv2": types.ModuleType("cv2"),
        "tyro": types.ModuleType("tyro"),
        "openpi_client": types.ModuleType("openpi_client"),
        "openpi_client.action_chunk_broker": types.ModuleType("openpi_client.action_chunk_broker"),
        "openpi_client.image_tools": types.ModuleType("openpi_client.image_tools"),
        "openpi_client.websocket_client_policy": types.ModuleType("openpi_client.websocket_client_policy"),
    }
    stubs["openpi_client"].__path__ = []
    path = pathlib.Path(__file__).with_name("client_memory_v4.py")
    spec = importlib.util.spec_from_file_location("_client_memory_v4_under_test", path)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {**stubs, spec.name: module}):
        spec.loader.exec_module(module)
    return module


client = _load_client()


def _metadata(**overrides):
    base = {
        "config_name": "pi05_yam_mem_v4_stage4c",
        "memory_architecture": "v32_layer8_dual_query",
        "memory_v4_dual_bank": True,
        "write_policy": "always",
        "action_horizon": 50,
        "rtc_enabled": True,
        "rtc_delay_semantics": "inclusive_max",
        "rtc_max_delay": 6,
        "memory_stride_frames": 15,
        "fact_slot_names": ["banana", "grey_pepper_box"],
        "fact_target_names": ["left_bin", "right_bin", "unknown"],
    }
    base.update(overrides)
    return base


def test_validate_v4_metadata_accepts_the_stage4c_contract():
    client.validate_v4_metadata(_metadata(), client.Args())


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"memory_v4_dual_bank": False}, "v4 dual-bank"),
        ({"write_policy": "client"}, "write-policy head"),
        ({"memory_stride_frames": 10}, "trained at 15"),
        ({"rtc_max_delay": 4}, "RTC maximum"),
        ({"rtc_enabled": False}, "RTC-trained"),
    ],
)
def test_validate_v4_metadata_rejects_mismatches(override, match):
    with pytest.raises(ValueError, match=match):
        client.validate_v4_metadata(_metadata(**override), client.Args())


def test_validate_v4_metadata_rejects_a_non_training_prompt():
    with pytest.raises(ValueError, match="training prompt"):
        client.validate_v4_metadata(_metadata(), client.Args(prompt="find the bin with banana"))


def test_memory_readout_names_slots_targets_and_commits():
    names = client._Names(_metadata())  # noqa: SLF001
    result = {
        "fact_predicted": [0, 2, 2, 2],
        "fact_confidence": [0.97, 0.5, 0.4, 0.4],
        "sem_commit_now": [True, False, False, False],
        "read_predicted": [0, 1, 2, 2],
        "sem_commits": 3,
    }
    seen, held = client.memory_readout(result, names)
    assert seen == "sees: banana=left(0.97)* grey pepper box=?(0.50)  (* = committed now)"
    assert held == "bank: banana=left grey pepper box=right  commits 3"

    # A never-written slot is blanked even if the read head's argmax names a side.
    _, held = client.memory_readout({**result, "sem_written": [True, False, False, False]}, names)
    assert held == "bank: banana=left grey pepper box=-  commits 3"


def test_memory_readout_falls_back_to_default_names():
    names = client._Names({})  # noqa: SLF001
    seen, held = client.memory_readout({"fact_predicted": [1], "fact_confidence": [0.9], "read_predicted": [1]}, names)
    assert seen.startswith("sees: banana=right(0.90)")
    assert held.startswith("bank: banana=right")
