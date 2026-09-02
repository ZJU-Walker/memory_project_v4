"""v4 Stage-2 side-flip battery: does the DECISION follow the memory CONTENT?

At the (static) decision step of every held-out sequence the canonical subtask string names
the side ("wait; target bin is left" / "... right"; the side word is a single PaliGemma token,
`▁left`=2731 / `▁right`=1833). The same sequence is scored twice under each memory condition:
with its true string and with the side word swapped, everything else (context, FAST action
tokens, memory writes) identical. The per-DECISION-STEP statistic is

    D = log p(true string) - log p(side-swapped string)        (nats, that step's string)

so D > 0 means the model names the TRUE side. A window may hold several waiting frames
(decision steps); every one is scored, and the report also gives the first-step-only view
(one term per sequence). Steps whose string names no side are excluded and counted. Conditions (read-side interventions on the
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


def summarize(records: list[dict], *, first_step_only: bool = False) -> dict:
    """Aggregate per-decision-step D values into the headline statistics.

    ``first_step_only`` restricts to each sequence's first decision step (one term per
    sequence, the battery's original unit); otherwise every decision step counts."""
    if first_step_only:
        records = [r for r in records if r["decision_order"] == 0]
    valid = [r for r in records if r["included"]]
    out = {
        "decision_steps": len(records),
        "sequences": len({(r["batch"], r["row"]) for r in records}),
        "included": len(valid),
        "excluded_no_side_token": len(records) - len(valid),
    }
    if not valid:
        return out
    for cond in CONDITIONS:
        d = np.asarray([r[f"D_{cond}"] for r in valid])
        out[f"{cond}_side_accuracy"] = float(np.mean(d > 0))
        out[f"{cond}_mean_margin"] = float(np.mean(d))
        out[f"{cond}_mean_abs_margin"] = float(np.mean(np.abs(d)))
    # Content-consistent pairing: expected answer under the donor bank = donor's fact for the
    # OWN prompted object. "Mismatched" = that expectation differs from the own side.
    usable = [r for r in valid if r.get("donor_expected_valid", True)]
    mismatched = [r for r in usable if r["donor_mismatched"]]
    matched = [r for r in usable if not r["donor_mismatched"]]
    out["donor_expected_valid"] = len(usable)
    out["donor_mismatched_pairs"] = len(mismatched)
    out["donor_matched_pairs"] = len(matched)
    if mismatched:
        d = np.asarray([r["D_donor"] for r in mismatched])
        out["donor_flip_rate_mismatched"] = float(np.mean(d < 0))  # names the donor-implied side
        out["donor_mean_margin_mismatched"] = float(np.mean(d))
        d_normal = np.asarray([r["D_normal"] for r in mismatched])
        out["donor_margin_shift_mismatched"] = float(np.mean(d_normal - d))
    if matched:
        d = np.asarray([r["D_donor"] for r in matched])
        out["donor_side_accuracy_matched"] = float(np.mean(d > 0))  # donor implies the SAME side
    # Content-following rate over every usable donor step: names the donor-implied side.
    if usable:
        follows = [
            (r["D_donor"] < 0) if r["donor_mismatched"] else (r["D_donor"] > 0) for r in usable
        ]
        out["donor_follows_content_rate"] = float(np.mean(follows))
    # Legacy pairing by the donor's own TARGET side (what the first report used).
    if any("donor_target_mismatched" in r for r in valid):
        t_mis = [r for r in valid if r["donor_target_mismatched"]]
        t_mat = [r for r in valid if not r["donor_target_mismatched"]]
        if t_mis:
            out["legacy_target_flip_rate_mismatched"] = float(np.mean(np.asarray([r["D_donor"] for r in t_mis]) < 0))
        if t_mat:
            out["legacy_target_side_accuracy_matched"] = float(np.mean(np.asarray([r["D_donor"] for r in t_mat]) > 0))
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
    parser.add_argument(
        "--bank",
        choices=("semantic", "visual", "both"),
        default="semantic",
        help="which bank the reset/donor interventions act on. With 'visual' the donor pairing "
        "still uses the semantic expectation (the visual bank carries no labelled fact), so "
        "read its numbers as: does the decision SURVIVE losing / swapping the visual bank?",
    )
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)
    intervention_prefix = "" if args.bank == "semantic" else f"{args.bank}_"

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
        # The bank stores OBJECT facts (slot 0 banana, slot 1 grey_pepper_box; always on
        # opposite sides in 0830/0831) while the decision names the TARGET side = side of the
        # prompted object. Under a donor bank, the content-consistent answer is the DONOR's
        # fact for the OWN prompted object -- not the donor's target side. The prompted slot
        # is the slot whose own fact equals the own target side.
        fact_labels = np.asarray(jax.device_get(observation.seq_fact_labels))  # [b, F]
        donor_fact_labels = np.roll(fact_labels, 1, axis=0)
        prompted_slot = np.full(batch, -1, dtype=np.int64)
        expected_donor_side = np.full(batch, -1, dtype=np.int64)
        for b in range(batch):
            matches = np.flatnonzero(fact_labels[b, :2] == sides[b])
            if matches.size == 1:
                prompted_slot[b] = int(matches[0])
                expected_donor_side[b] = int(donor_fact_labels[b, prompted_slot[b]])
        causal = np.asarray(observation.tokenized_causal)
        causal_mask = np.asarray(observation.tokenized_causal_mask)
        fast_mask = np.asarray(observation.causal_fast_mask)
        text_mask = causal_mask & ~fast_mask
        token_count = causal_mask.sum(axis=-1)  # [b, T]
        has_side = np.any(text_mask & np.isin(causal, (LEFT_TOKEN, RIGHT_TOKEN)), axis=-1)  # [b, T]
        swapped_causal = swap_side_tokens(causal, causal_mask, fast_mask)
        swapped_observation = observation.replace(tokenized_causal=jax.numpy.asarray(swapped_causal))
        step_rng = jax.random.fold_in(rng, index)
        ce = {}
        active = None
        for cond in CONDITIONS:
            intervention = None if cond == "normal" else intervention_prefix + cond
            for tag, obs in (("true", observation), ("swap", swapped_observation)):
                losses = sequence_loss(step_rng, obs, actions, train=False, v4_intervention=intervention)
                # [T, b] -> [b, T]; the CE is the mean over the step's causal tokens, masked to
                # decision steps by the model (transition-valid waiting frames).
                ce[(cond, tag)] = np.asarray(jax.device_get(losses["v4_decision_ce_steps"])).T
                if active is None:
                    active = np.asarray(jax.device_get(losses["v4_decision_active_steps"])).T > 0.5
        for b in range(batch):
            steps = np.flatnonzero(active[b])
            for order, t in enumerate(steps):
                included = bool(has_side[b, t] and token_count[b, t] > 0)
                expected = int(expected_donor_side[b])
                record = {
                    "batch": index,
                    "row": b,
                    "step": int(t),
                    "decision_order": order,  # 0 = first decision step of the sequence
                    "side": int(sides[b]),
                    "donor_side": int(donor_sides[b]),
                    "fact_labels": fact_labels[b].tolist(),
                    "donor_fact_labels": donor_fact_labels[b].tolist(),
                    "prompted_slot": int(prompted_slot[b]),
                    # Content-consistent expectation under the donor bank (own prompt, donor facts).
                    "expected_donor_side": expected,
                    "donor_expected_valid": bool(expected in (0, 1)),
                    "donor_mismatched": bool(expected in (0, 1) and expected != int(sides[b])),
                    # Legacy pairing by the donor's own TARGET side (kept for comparison).
                    "donor_target_mismatched": bool(sides[b] != donor_sides[b]),
                    "decision_tokens": int(token_count[b, t]),
                    "has_side_token": bool(has_side[b, t]),
                    "included": included,
                }
                for cond in CONDITIONS:
                    # mean-token CE difference x token count = log p(true) - log p(swapped)
                    record[f"D_{cond}"] = float((ce[(cond, "swap")][b, t] - ce[(cond, "true")][b, t]) * token_count[b, t])
                    record[f"ce_true_{cond}"] = float(ce[(cond, "true")][b, t])
                    record[f"ce_swap_{cond}"] = float(ce[(cond, "swap")][b, t])
                records.append(record)
        print(f"batch {index + 1}/{args.batches} done", flush=True)

    summary = summarize(records)
    summary_first = summarize(records, first_step_only=True)

    def show(title: str, s: dict) -> None:
        # Print the headline BEFORE writing the report so the numbers survive any write failure.
        print(
            f"[{title}] decision_steps={s['decision_steps']} sequences={s['sequences']} "
            f"included={s['included']} excluded_no_side_token={s['excluded_no_side_token']}"
        )
        for cond in CONDITIONS:
            if f"{cond}_side_accuracy" in s:
                print(
                    f"  {cond:6s} side_accuracy={s[f'{cond}_side_accuracy']:.3f} "
                    f"mean_margin={s[f'{cond}_mean_margin']:+.3f} nats "
                    f"mean_abs_margin={s[f'{cond}_mean_abs_margin']:.3f}"
                )
        if "donor_flip_rate_mismatched" in s:
            print(
                f"  donor mismatched pairs={s['donor_mismatched_pairs']}: "
                f"FLIP RATE (prefers donor's side)={s['donor_flip_rate_mismatched']:.3f} "
                f"mean_margin={s['donor_mean_margin_mismatched']:+.3f} "
                f"margin_shift_vs_normal={s['donor_margin_shift_mismatched']:+.3f}"
            )
        if "donor_side_accuracy_matched" in s:
            print(
                f"  donor matched pairs={s['donor_matched_pairs']}: "
                f"side_accuracy={s['donor_side_accuracy_matched']:.3f}"
            )
        if "donor_follows_content_rate" in s:
            print(
                f"  donor FOLLOWS-CONTENT rate (names the donor-implied side, all usable steps="
                f"{s['donor_expected_valid']})={s['donor_follows_content_rate']:.3f}"
            )
        if "legacy_target_flip_rate_mismatched" in s:
            print(
                f"  [legacy target-side pairing] flip_rate_mismatched="
                f"{s['legacy_target_flip_rate_mismatched']:.3f} "
                f"side_accuracy_matched={s.get('legacy_target_side_accuracy_matched', float('nan')):.3f}"
            )

    show("all decision steps", summary)
    show("first decision step per sequence", summary_first)

    def json_default(value):
        # numpy scalars that slip through (np.bool_, np.integer, np.floating)
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    report = {
        "schema_version": SCHEMA_VERSION,
        "config_name": args.config_name,
        "split": args.split,
        "intervention_bank": args.bank,
        "batches": args.batches,
        "batch_size": args.batch_size,
        "parameter_tree_sha256": parameter_tree_sha256,
        "summary": summary,
        "summary_first_decision_step": summary_first,
        "records": records,
    }
    body = json.dumps(report, indent=2, sort_keys=True, default=json_default) + "\n"
    report["report_sha256"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=json_default) + "\n")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    sys.exit(main())
