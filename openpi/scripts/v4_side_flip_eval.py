"""v4 Stage-2 side-flip battery: does the DECISION follow the memory CONTENT?

At the (static) decision step of every held-out sequence the canonical subtask string names
the side ("wait; target bin is left" / "... right"; the side word is a single PaliGemma token,
`▁left`=2731 / `▁right`=1833). The same sequence is scored twice under each memory condition:
with its true string and with the side word swapped, everything else (context, FAST action
tokens, memory writes) identical. The per-sequence statistic is

    D = log p(true string) - log p(side-swapped string)        (nats, decision step only)

so D > 0 means the model names the TRUE side. Conditions (read-side interventions on the
decision step only, exactly as `v4_stage2_eval.py`):

* normal: the sequence's own semantic bank            -> expect D > 0
* reset:  a blank bank                                -> expect D near 0 (no side information)
* donor:  the previous sequence's bank (batch rolled  -> on MISMATCHED pairs a content-using
          by one after the side-alternating permutation)  policy prefers the DONOR's side: D < 0

Reported: side accuracy P(D > 0) per condition, the donor FLIP RATE P(D < 0 | mismatched
pair), the matched-pair control P(D > 0 | matched pair), and mean margins. Sequences whose
decision step carries no side token are excluded and counted. Causal tokens never reach the
memory (the memory tokens attend only to the ar=0 context), so swapping them changes nothing
but the scored string.
"""

# ruff: noqa: I001 - pyarrow must precede the openpi/JAX stack for this dataset (libarrow).
import pyarrow.parquet  # noqa: F401  isort: skip

import argparse
import dataclasses
import hashlib
import json
import pathlib
import sys

import numpy as np

from v4_stage2_eval import alternate_sides_permutation

SCHEMA_VERSION = "v4_side_flip_eval/1"
CONDITIONS = ("normal", "reset", "donor")
LEFT_TOKEN = 2731
RIGHT_TOKEN = 1833


def swap_side_tokens(causal: np.ndarray, causal_mask: np.ndarray, fast_mask: np.ndarray) -> np.ndarray:
    """Swap the side word in the TEXT part of every causal buffer (FAST tokens untouched)."""
    text = causal_mask & ~fast_mask
    swapped = causal.copy()
    swapped[text & (causal == LEFT_TOKEN)] = RIGHT_TOKEN
    swapped[text & (causal == RIGHT_TOKEN)] = LEFT_TOKEN
    return swapped


def decision_token_stats(observation) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per sequence: number of decision steps, causal-token count at the (first) decision
    step, and whether that step's text carries a side token."""
    decision = np.asarray(observation.seq_decision_mask) & np.asarray(observation.seq_step_mask)
    causal = np.asarray(observation.tokenized_causal)
    causal_mask = np.asarray(observation.tokenized_causal_mask)
    fast = np.asarray(observation.causal_fast_mask)
    n_decision = decision.sum(axis=1)
    token_count = np.zeros(decision.shape[0], dtype=np.int64)
    has_side = np.zeros(decision.shape[0], dtype=bool)
    for b in range(decision.shape[0]):
        steps = np.flatnonzero(decision[b])
        if steps.size == 0:
            continue
        t = int(steps[0])
        token_count[b] = int(causal_mask[b, t].sum())
        text = causal_mask[b, t] & ~fast[b, t]
        has_side[b] = bool(np.any(text & np.isin(causal[b, t], (LEFT_TOKEN, RIGHT_TOKEN))))
    return n_decision, token_count, has_side


def summarize(records: list[dict]) -> dict:
    """Aggregate per-sequence D values into the headline statistics."""
    valid = [r for r in records if r["included"]]
    out = {"sequences": len(records), "included": len(valid), "excluded_no_side_token": len(records) - len(valid)}
    if not valid:
        return out
    for cond in CONDITIONS:
        d = np.asarray([r[f"D_{cond}"] for r in valid])
        out[f"{cond}_side_accuracy"] = float(np.mean(d > 0))
        out[f"{cond}_mean_margin"] = float(np.mean(d))
        out[f"{cond}_mean_abs_margin"] = float(np.mean(np.abs(d)))
    mismatched = [r for r in valid if r["donor_mismatched"]]
    matched = [r for r in valid if not r["donor_mismatched"]]
    out["donor_mismatched_pairs"] = len(mismatched)
    out["donor_matched_pairs"] = len(matched)
    if mismatched:
        d = np.asarray([r["D_donor"] for r in mismatched])
        out["donor_flip_rate_mismatched"] = float(np.mean(d < 0))  # follows the DONOR's side
        out["donor_mean_margin_mismatched"] = float(np.mean(d))
        d_normal = np.asarray([r["D_normal"] for r in mismatched])
        out["donor_margin_shift_mismatched"] = float(np.mean(d_normal - d))
    if matched:
        d = np.asarray([r["D_donor"] for r in matched])
        out["donor_side_accuracy_matched"] = float(np.mean(d > 0))
    return out


def main(argv=None) -> None:
    import jax

    from openpi.models import model as model_lib
    from openpi.shared import nnx_utils
    from openpi.training import config as config_lib
    from openpi.training import data_loader as data_loader_lib
    from openpi.training import weight_loaders

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=pathlib.Path, required=True)
    parser.add_argument("--config-name", default="pi05_yam_mem_v4_stage2a")
    parser.add_argument("--split", choices=("development", "train"), default="development")
    parser.add_argument("--batches", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)

    output_dir = args.output_dir
    report_path = output_dir / "side_flip_eval.json"
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

    records: list[dict] = []
    rng = jax.random.key(args.seed)
    for index, (observation, actions) in enumerate(loader):
        sides = np.asarray(jax.device_get(observation.seq_side_label))
        perm = alternate_sides_permutation(sides)
        observation = jax.tree.map(lambda x: x[perm], observation)
        actions = actions[perm]
        sides = sides[perm]
        batch = sides.shape[0]
        # The model's donor intervention is jnp.roll(state, 1, axis=0): sequence b reads b-1.
        donor_sides = np.roll(sides, 1)
        n_decision, token_count, has_side = decision_token_stats(observation)
        swapped_causal = swap_side_tokens(
            np.asarray(observation.tokenized_causal),
            np.asarray(observation.tokenized_causal_mask),
            np.asarray(observation.causal_fast_mask),
        )
        swapped_observation = observation.replace(tokenized_causal=jax.numpy.asarray(swapped_causal))
        step_rng = jax.random.fold_in(rng, index)
        ce = {}
        for cond in CONDITIONS:
            intervention = None if cond == "normal" else cond
            for tag, obs in (("true", observation), ("swap", swapped_observation)):
                losses = sequence_loss(step_rng, obs, actions, train=False, v4_intervention=intervention)
                per_seq = np.asarray(jax.device_get(losses["v4_decision_ce_per_sequence"]))
                count = np.asarray(jax.device_get(losses["v4_decision_count_per_sequence"]))
                ce[(cond, tag)] = per_seq / np.maximum(count, 1.0)
                ce[(cond, tag, "count")] = count
        for b in range(batch):
            count = float(ce[("normal", "true", "count")][b])
            included = bool(has_side[b]) and count == 1.0 and token_count[b] > 0
            record = {
                "batch": index,
                "row": b,
                "side": int(sides[b]),
                "donor_side": int(donor_sides[b]),
                "donor_mismatched": bool(sides[b] != donor_sides[b]),
                "decision_steps": float(count),
                "decision_tokens": int(token_count[b]),
                "has_side_token": bool(has_side[b]),
                "included": included,
            }
            for cond in CONDITIONS:
                # mean-token CE difference x token count = log p(true) - log p(swapped)
                record[f"D_{cond}"] = float((ce[(cond, "swap")][b] - ce[(cond, "true")][b]) * token_count[b])
                record[f"ce_true_{cond}"] = float(ce[(cond, "true")][b])
                record[f"ce_swap_{cond}"] = float(ce[(cond, "swap")][b])
            records.append(record)
        print(f"batch {index + 1}/{args.batches} done", flush=True)

    summary = summarize(records)
    report = {
        "schema_version": SCHEMA_VERSION,
        "config_name": args.config_name,
        "split": args.split,
        "batches": args.batches,
        "batch_size": args.batch_size,
        "parameter_tree_sha256": parameter_tree_sha256,
        "summary": summary,
        "records": records,
    }
    body = json.dumps(report, indent=2, sort_keys=True) + "\n"
    report["report_sha256"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {report_path}")
    print(
        f"sequences={summary['sequences']} included={summary['included']} "
        f"excluded_no_side_token={summary['excluded_no_side_token']}"
    )
    for cond in CONDITIONS:
        if f"{cond}_side_accuracy" in summary:
            print(
                f"  {cond:6s} side_accuracy={summary[f'{cond}_side_accuracy']:.3f} "
                f"mean_margin={summary[f'{cond}_mean_margin']:+.3f} nats "
                f"mean_abs_margin={summary[f'{cond}_mean_abs_margin']:.3f}"
            )
    if "donor_flip_rate_mismatched" in summary:
        print(
            f"  donor mismatched pairs={summary['donor_mismatched_pairs']}: "
            f"FLIP RATE (prefers donor's side)={summary['donor_flip_rate_mismatched']:.3f} "
            f"mean_margin={summary['donor_mean_margin_mismatched']:+.3f} "
            f"margin_shift_vs_normal={summary['donor_margin_shift_mismatched']:+.3f}"
        )
    if "donor_side_accuracy_matched" in summary:
        print(
            f"  donor matched pairs={summary['donor_matched_pairs']}: "
            f"side_accuracy={summary['donor_side_accuracy_matched']:.3f}"
        )


if __name__ == "__main__":
    sys.exit(main())
