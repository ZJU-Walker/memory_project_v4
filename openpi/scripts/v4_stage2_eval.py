"""v4 Stage-2 "use" battery (V4_PLAN.md §5 Stage 2, §10 ladder rungs 3-5) -- adversarial.

Runs a trained Stage-2 checkpoint over DEVELOPMENT-split sequences through the real training
data pipeline and the real sequence objective, three times per batch with identical RNG:

  * normal -- the model reads its own semantic bank;
  * reset  -- on decision steps it reads a FRESH (exactly-zero) bank instead;
  * donor  -- on decision steps it reads the batch neighbour's bank (batches are permuted so
              consecutive samples alternate answer sides wherever the draw allows; the
              mismatched-pair fraction is reported and the donor effect scales with it).

The carried state and every write are identical across conditions -- only what the model
READS on decision steps differs -- so any change in the decision-step subtask CE, in the
use-pressure flow loss (steps whose action chunk reaches the side-dependent execute phase),
or in read-side fact accuracy is CAUSED by the semantic memory content. This is the test the
v36 visual pilot failed (writer/reader perfect, actions deaf): here the action pathway must
move.

Gates are provisional thresholds recorded in the report; tighten them once the first Stage-2
run shows the effect sizes.
"""

import argparse
import dataclasses
import hashlib
import json
import pathlib
import sys

import pyarrow.parquet  # noqa: F401  isort: skip  (before openpi/JAX: libarrow segfault order)

import numpy as np

CONDITIONS = ("normal", "reset", "donor")
SCHEMA_VERSION = "openpi.v4.stage2-eval.v1"


def alternate_sides_permutation(sides: np.ndarray) -> np.ndarray:
    """Order a batch so consecutive samples alternate sides as far as the draw allows.

    The donor intervention reads sample i-1's bank into sample i (jnp.roll by one, cyclic),
    so alternation maximizes the number of mismatched (informative) donor pairs.
    """
    sides = np.asarray(sides)
    left = list(np.nonzero(sides == 0)[0])
    right = list(np.nonzero(sides == 1)[0])
    other = list(np.nonzero((sides != 0) & (sides != 1))[0])
    order = []
    take_left = len(left) >= len(right)
    while left or right:
        pool = left if take_left else right
        if pool:
            order.append(pool.pop(0))
        take_left = not take_left
        if not left or not right:
            order.extend(left)
            order.extend(right)
            left, right = [], []
    order.extend(other)
    return np.asarray(order, dtype=np.int64)


def mismatched_pair_fraction(sides: np.ndarray) -> float:
    """Fraction of cyclic (i-1 -> i) donor pairs with different, valid sides."""
    sides = np.asarray(sides)
    donor = np.roll(sides, 1)
    valid = (sides >= 0) & (sides < 2) & (donor >= 0) & (donor < 2)
    if not np.any(valid):
        return 0.0
    return float(np.mean((sides != donor)[valid]))


def evaluate_gates(metrics: dict[str, dict[str, float]], mismatch_fraction: float) -> dict[str, dict]:
    normal, reset, donor = metrics["normal"], metrics["reset"], metrics["donor"]

    def ratio(a: float, b: float) -> float:
        return float(a / b) if b > 0 else float("inf") if a > 0 else 1.0

    gates = {
        # Read: the read head decodes the stored fact on decision steps.
        "read_accuracy_normal": {"value": normal["fact_read_accuracy"], "threshold": 0.9, "direction": "at_least"},
        # Use (text): removing the memory must hurt the decision-step subtask CE.
        "reset_decision_ce_ratio": {
            "value": ratio(reset["decision_ce"], normal["decision_ce"]),
            "threshold": 1.2,
            "direction": "at_least",
        },
        # Use (actions): removing the memory must hurt the use-pressure flow loss.
        "reset_use_flow_ratio": {
            "value": ratio(reset["use_flow"], normal["use_flow"]),
            "threshold": 1.1,
            "direction": "at_least",
        },
        # Causal read: with the donor's bank the read head reports the donor's fact, so
        # accuracy against the OWN label falls toward (1 - mismatch_fraction) * normal.
        "donor_read_accuracy": {
            "value": donor["fact_read_accuracy"],
            "threshold": max(0.0, (1.0 - mismatch_fraction) * normal["fact_read_accuracy"] + 0.15),
            "direction": "at_most",
        },
        # Wrong content must be at least as damaging as no content.
        "donor_decision_ce_ratio": {
            "value": ratio(donor["decision_ce"], normal["decision_ce"]),
            "threshold": 1.2,
            "direction": "at_least",
        },
    }
    for gate in gates.values():
        v, t = gate["value"], gate["threshold"]
        gate["passes"] = bool(v >= t) if gate["direction"] == "at_least" else bool(v <= t)
    return gates


def main(argv=None) -> None:
    import jax
    import jax.numpy as jnp

    from openpi.models import model as model_lib
    from openpi.shared import nnx_utils
    from openpi.training import config as config_lib
    from openpi.training import data_loader as data_loader_lib
    from openpi.training import weight_loaders

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=pathlib.Path, required=True)
    parser.add_argument("--config-name", default="pi05_yam_mem_v4_stage2a")
    parser.add_argument("--split", choices=("development", "train"), default="development")
    parser.add_argument("--batches", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument(
        "--bank",
        choices=("semantic", "visual", "both"),
        default="semantic",
        help="which bank the reset/donor interventions act on (Stage 4: 'visual' shows whether "
        "the decision survives losing the visual bank; 'both' removes all memory).",
    )
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)
    intervention_prefix = "" if args.bank == "semantic" else f"{args.bank}_"

    output_dir = args.output_dir
    report_path = output_dir / "stage2_eval.json"
    if report_path.exists():
        raise SystemExit(f"{report_path} already exists (create-only).")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = config_lib.get_config(args.config_name)
    config = dataclasses.replace(
        config,
        batch_size=args.batch_size,
        num_workers=0,
        seed=args.seed,
        v4_graft_sources=(),
        data=dataclasses.replace(
            config.data,
            base_config=dataclasses.replace(config.data.base_config, memory_manifest_split=args.split),
        ),
    )
    params = model_lib.restore_params(args.params, restore_type=np.ndarray)
    parameter_tree_sha256 = weight_loaders.parameter_tree_sha256(params)
    model = config.model.load(params)
    model.eval()
    loader = data_loader_lib.create_data_loader(
        config, sharding=None, shuffle=True, num_batches=args.batches, exact_resume=False
    )
    sequence_loss = nnx_utils.module_jit(
        model._compute_sequence_loss_v32,  # noqa: SLF001
        static_argnames=("train", "v4_intervention"),
    )

    keys = (
        "v4_decision_ce_sum",
        "v4_decision_count",
        "v4_use_flow_sum",
        "v4_use_count",
        "v4_fact_read_correct",
        "v4_fact_read_count",
        "v4_sem_raw_read_rms_sum",
        "v4_sem_commit_count",
        "v35_transition_count",
    )
    sums = {cond: {k: 0.0 for k in keys} for cond in CONDITIONS}
    sides_seen: list[np.ndarray] = []
    rng = jax.random.key(args.seed)
    for index, (observation, actions) in enumerate(loader):
        sides = np.asarray(jax.device_get(observation.seq_side_label))
        perm = alternate_sides_permutation(sides)
        observation = jax.tree.map(lambda x: x[perm], observation)
        actions = actions[perm]
        sides_seen.append(sides[perm])
        step_rng = jax.random.fold_in(rng, index)
        for cond in CONDITIONS:
            losses = sequence_loss(
                step_rng,
                observation,
                actions,
                train=False,
                v4_intervention=None if cond == "normal" else intervention_prefix + cond,
            )
            for k in keys:
                sums[cond][k] += float(jax.device_get(losses[k]))
        print(f"batch {index + 1}/{args.batches} done", flush=True)

    all_sides = np.concatenate(sides_seen)
    mismatch = float(np.mean([mismatched_pair_fraction(s) for s in sides_seen]))
    metrics = {}
    for cond in CONDITIONS:
        s = sums[cond]
        metrics[cond] = {
            "decision_ce": s["v4_decision_ce_sum"] / max(s["v4_decision_count"], 1.0),
            "use_flow": s["v4_use_flow_sum"] / max(s["v4_use_count"], 1.0),
            "fact_read_accuracy": s["v4_fact_read_correct"] / max(s["v4_fact_read_count"], 1.0),
            "sem_raw_read_rms": s["v4_sem_raw_read_rms_sum"] / max(s["v35_transition_count"], 1.0),
            "decision_steps": s["v4_decision_count"],
            "use_steps": s["v4_use_count"],
            "read_terms": s["v4_fact_read_count"],
            "sem_commits": s["v4_sem_commit_count"],
        }
    gates = evaluate_gates(metrics, mismatch)
    report = {
        "schema_version": SCHEMA_VERSION,
        "config_name": args.config_name,
        "split": args.split,
        "intervention_bank": args.bank,
        "batches": args.batches,
        "batch_size": args.batch_size,
        "sequences": int(all_sides.shape[0]),
        "sides_left_right": [int(np.sum(all_sides == 0)), int(np.sum(all_sides == 1))],
        "donor_mismatched_pair_fraction": mismatch,
        "parameter_tree_sha256": parameter_tree_sha256,
        "metrics": metrics,
        "gates": gates,
        "passes": bool(all(g["passes"] for g in gates.values())),
    }
    body = json.dumps(report, indent=2, sort_keys=True) + "\n"
    report["report_sha256"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {report_path}")
    print(f"PASS={report['passes']}  donor mismatched-pair fraction={mismatch:.2f}")
    for cond in CONDITIONS:
        m = metrics[cond]
        print(
            f"  {cond:6s} decision_ce={m['decision_ce']:.4f} use_flow={m['use_flow']:.4f} "
            f"read_acc={m['fact_read_accuracy']:.3f} raw_read_rms={m['sem_raw_read_rms']:.4f}"
        )
    for name, gate in gates.items():
        print(f"  gate {name}: value={gate['value']:.4f} threshold={gate['threshold']:.4f} passes={gate['passes']}")


if __name__ == "__main__":
    sys.exit(main())
