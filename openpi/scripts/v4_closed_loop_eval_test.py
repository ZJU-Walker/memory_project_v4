"""Pure-function tests for the v4 closed-loop battery (run with PYTHONPATH=scripts)."""

import numpy as np
from v4_closed_loop_eval import BOTH_SIDES
from v4_closed_loop_eval import NO_SIDE
from v4_closed_loop_eval import side_from_tokens
from v4_closed_loop_eval import summarize
from v4_side_flip_eval import LEFT_TOKEN
from v4_side_flip_eval import RIGHT_TOKEN


def test_side_from_tokens_reads_only_live_tokens():
    tokens = np.asarray([5, LEFT_TOKEN, 7, RIGHT_TOKEN], dtype=np.int32)
    assert side_from_tokens(tokens, np.asarray([True, True, True, False])) == 0
    assert side_from_tokens(tokens, np.asarray([True, False, True, True])) == 1
    assert side_from_tokens(tokens, np.asarray([True, True, True, True])) == BOTH_SIDES
    assert side_from_tokens(tokens, np.asarray([True, False, True, False])) == NO_SIDE


def _record(batch, row, order, side, expected, *, pred, d, included=True, read=(1, 1)):
    mismatched = expected in (0, 1) and expected != side
    rec = {
        "batch": batch,
        "row": row,
        "decision_order": order,
        "side": side,
        "expected_donor_side": expected,
        "donor_expected_valid": expected in (0, 1),
        "donor_mismatched": mismatched,
        "included": included,
        "read_correct": read[0],
        "read_count": read[1],
    }
    for cond in ("normal", "reset", "donor"):
        rec[f"pred_side_{cond}"] = pred[cond]
        rec[f"D_{cond}"] = d[cond]
    return rec


def test_summarize_reports_free_and_forced_content_following():
    records = [
        # mismatched pair, follows the donor: free decode names the donor side, D < 0
        _record(
            0,
            0,
            0,
            0,
            1,
            pred={"normal": 0, "reset": NO_SIDE, "donor": 1},
            d={"normal": 5.0, "reset": 0.1, "donor": -4.0},
        ),
        # matched pair, stays correct
        _record(
            0, 1, 0, 1, 1, pred={"normal": 1, "reset": 0, "donor": 1}, d={"normal": 6.0, "reset": -0.5, "donor": 5.0}
        ),
        # second decision step of row 0 (excluded from the first-step view), wrong under donor
        _record(
            0,
            0,
            1,
            0,
            1,
            pred={"normal": 0, "reset": 0, "donor": 0},
            d={"normal": 3.0, "reset": 1.0, "donor": 2.0},
            read=(0, 1),
        ),
        # no side token in the true string: free decode still counts, D excluded
        _record(
            1,
            0,
            0,
            1,
            -1,
            pred={"normal": 1, "reset": 1, "donor": 1},
            d={"normal": 0.0, "reset": 0.0, "donor": 0.0},
            included=False,
            read=(0, 0),
        ),
    ]
    s = summarize(records)
    assert s["decision_steps"] == 4
    assert s["sequences"] == 3
    assert s["included_for_D"] == 3
    assert s["excluded_no_side_token"] == 1
    assert s["normal_free_side_accuracy"] == 1.0
    assert s["reset_free_no_side_rate"] == 0.25
    assert s["reset_free_wrong_side_rate"] == 0.25
    np.testing.assert_allclose(s["normal_D_side_accuracy"], 1.0)
    np.testing.assert_allclose(s["reset_D_side_accuracy"], 2 / 3)
    assert s["donor_expected_valid"] == 3
    assert s["donor_mismatched_pairs"] == 2
    assert s["donor_matched_pairs"] == 1
    np.testing.assert_allclose(s["donor_free_follows_content_rate"], 2 / 3)
    np.testing.assert_allclose(s["donor_D_follows_content_rate"], 2 / 3)
    np.testing.assert_allclose(s["donor_free_flip_rate_mismatched"], 0.5)
    np.testing.assert_allclose(s["donor_D_flip_rate_mismatched"], 0.5)
    assert s["donor_free_side_accuracy_matched"] == 1.0
    np.testing.assert_allclose(s["read_accuracy_normal"], 2 / 3)
    assert s["read_terms"] == 3

    first = summarize(records, first_step_only=True)
    assert first["decision_steps"] == 3
    assert first["donor_mismatched_pairs"] == 1
    assert first["donor_free_flip_rate_mismatched"] == 1.0
    assert first["donor_D_flip_rate_mismatched"] == 1.0

    assert summarize([]) == {"decision_steps": 0, "sequences": 0}
