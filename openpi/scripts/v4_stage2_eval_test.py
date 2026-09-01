"""Tests for the Stage-2 battery's pure helpers (permutation, pairing, gates)."""

import numpy as np

import v4_stage2_eval as battery


def test_alternating_permutation_maximizes_mismatched_donor_pairs():
    sides = np.asarray([0, 0, 1, 1])
    perm = battery.alternate_sides_permutation(sides)
    assert sorted(perm.tolist()) == [0, 1, 2, 3]
    assert battery.mismatched_pair_fraction(sides[perm]) == 1.0
    # Unbalanced draws: as many alternations as possible, the rest appended.
    sides = np.asarray([0, 0, 0, 1])
    perm = battery.alternate_sides_permutation(sides)
    assert sorted(perm.tolist()) == [0, 1, 2, 3]
    assert battery.mismatched_pair_fraction(sides[perm]) == 0.5
    # Invalid sides never form pairs.
    assert battery.mismatched_pair_fraction(np.asarray([-1, -1])) == 0.0


def test_gates_reward_causal_use_and_penalize_deaf_models():
    deaf = {
        "normal": {"decision_ce": 1.0, "use_flow": 0.2, "fact_read_accuracy": 0.95},
        "reset": {"decision_ce": 1.0, "use_flow": 0.2, "fact_read_accuracy": 0.5},
        "donor": {"decision_ce": 1.0, "use_flow": 0.2, "fact_read_accuracy": 0.05},
    }
    gates = battery.evaluate_gates(deaf, mismatch_fraction=1.0)
    assert gates["read_accuracy_normal"]["passes"]
    assert gates["donor_read_accuracy"]["passes"]
    assert not gates["reset_decision_ce_ratio"]["passes"]
    assert not gates["reset_use_flow_ratio"]["passes"]

    using = {
        "normal": {"decision_ce": 1.0, "use_flow": 0.2, "fact_read_accuracy": 0.95},
        "reset": {"decision_ce": 1.6, "use_flow": 0.3, "fact_read_accuracy": 0.5},
        "donor": {"decision_ce": 2.0, "use_flow": 0.4, "fact_read_accuracy": 0.05},
    }
    gates = battery.evaluate_gates(using, mismatch_fraction=1.0)
    assert all(g["passes"] for g in gates.values())
