"""v4 closed-loop battery: the DEPLOYMENT path (`Pi0.sample_with_memory`) over development windows.

The Stage-2/4 batteries (`v4_stage2_eval.py`, `v4_side_flip_eval.py`) score the teacher-forced
sequence objective. This one replays every window step by step through the inference path --
prefill, KV-cache greedy subtask decode, action denoising -- threading BOTH banks the model
produces itself under the same sparse clock the sequence path uses (step validity, E-step write
mask, skip-O gap decays). Nothing on the memory side is teacher-forced: the semantic bank holds
whatever the model's own fact head committed on evidence steps.

At every decision step (transition-valid waiting frame) it records, per condition:

* the free-decoded subtask and the side it names (`▁left`=2731 / `▁right`=1833 / none);
* D = log p(true string) - log p(side-swapped string), scored on the same prefix and memory
  through `forced_subtask_tokens` (the side-flip statistic, now on the inference path);
* the read head's per-slot fact prediction from the raw semantic retrieval.

Conditions are read-side only -- the carried states never change -- exactly as the sequence
batteries: normal / reset (a blank bank is read) / donor (the batch neighbour's carried bank is
read), on the semantic, visual or both banks (`--bank`). Donor pairing is content-consistent
(the donor's fact for the OWN prompted object), as in `v4_side_flip_eval.py`.

Headline numbers: free-decode side accuracy per condition, the donor FOLLOWS-CONTENT rate
(free decode and D), semantic commits per window, write accuracy at commits, read accuracy at
decision steps. A closed-loop result that matches the sequence batteries shows the inference
path (cache layout, greedy decode, own-head writes) carries the same memory use.
"""

# ruff: noqa: I001 - pyarrow must precede the openpi/JAX stack for this dataset (libarrow).
import pyarrow.parquet  # noqa: F401  isort: skip

import argparse
import dataclasses
import hashlib
import json
import pathlib
import sys
import time

import numpy as np

from v4_side_flip_eval import LEFT_TOKEN
from v4_side_flip_eval import RIGHT_TOKEN
from v4_side_flip_eval import swap_side_tokens
from v4_stage2_eval import alternate_sides_permutation

SCHEMA_VERSION = "v4_closed_loop_eval/1"
CONDITIONS = ("normal", "reset", "donor")
NO_SIDE = -1
BOTH_SIDES = 2


def side_from_tokens(tokens: np.ndarray, mask: np.ndarray) -> int:
    """The side a generated token buffer names: 0 left, 1 right, -1 none, 2 both."""
    live = np.asarray(tokens)[np.asarray(mask, dtype=bool)]
    has_left = bool(np.any(live == LEFT_TOKEN))
    has_right = bool(np.any(live == RIGHT_TOKEN))
    if has_left and has_right:
        return BOTH_SIDES
    if has_left:
        return 0
    if has_right:
        return 1
    return NO_SIDE


def summarize(records: list[dict], *, first_step_only: bool = False) -> dict:
    """Aggregate per-decision-step records (see main) into the headline statistics."""
    if first_step_only:
        records = [r for r in records if r["decision_order"] == 0]
    out = {
        "decision_steps": len(records),
        "sequences": len({(r["batch"], r["row"]) for r in records}),
    }
    if not records:
        return out
    valid_d = [r for r in records if r["included"]]
    out["included_for_D"] = len(valid_d)
    out["excluded_no_side_token"] = len(records) - len(valid_d)
    for cond in CONDITIONS:
        pred = np.asarray([r[f"pred_side_{cond}"] for r in records])
        side = np.asarray([r["side"] for r in records])
        out[f"{cond}_free_side_accuracy"] = float(np.mean(pred == side))
        out[f"{cond}_free_no_side_rate"] = float(np.mean(pred == NO_SIDE))
        out[f"{cond}_free_wrong_side_rate"] = float(np.mean((pred >= 0) & (pred != BOTH_SIDES) & (pred != side)))
        if valid_d:
            d = np.asarray([r[f"D_{cond}"] for r in valid_d])
            out[f"{cond}_D_side_accuracy"] = float(np.mean(d > 0))
            out[f"{cond}_D_mean_margin"] = float(np.mean(d))
    usable = [r for r in records if r["donor_expected_valid"]]
    mismatched = [r for r in usable if r["donor_mismatched"]]
    matched = [r for r in usable if not r["donor_mismatched"]]
    out["donor_expected_valid"] = len(usable)
    out["donor_mismatched_pairs"] = len(mismatched)
    out["donor_matched_pairs"] = len(matched)
    if usable:
        follows = [r["pred_side_donor"] == r["expected_donor_side"] for r in usable]
        out["donor_free_follows_content_rate"] = float(np.mean(follows))
        usable_d = [r for r in usable if r["included"]]
        if usable_d:
            follows_d = [(r["D_donor"] < 0) if r["donor_mismatched"] else (r["D_donor"] > 0) for r in usable_d]
            out["donor_D_follows_content_rate"] = float(np.mean(follows_d))
    if mismatched:
        out["donor_free_flip_rate_mismatched"] = float(
            np.mean([r["pred_side_donor"] == r["expected_donor_side"] for r in mismatched])
        )
        mis_d = [r for r in mismatched if r["included"]]
        if mis_d:
            out["donor_D_flip_rate_mismatched"] = float(np.mean(np.asarray([r["D_donor"] for r in mis_d]) < 0))
    if matched:
        out["donor_free_side_accuracy_matched"] = float(np.mean([r["pred_side_donor"] == r["side"] for r in matched]))
    read_count = sum(r["read_count"] for r in records)
    if read_count:
        out["read_accuracy_normal"] = float(sum(r["read_correct"] for r in records) / read_count)
        out["read_terms"] = int(read_count)
    return out


def main(argv=None) -> None:
    import jax
    import jax.numpy as jnp

    from openpi.models import model as model_lib
    from openpi.models import tokenizer as tokenizer_lib
    from openpi.shared import nnx_utils
    from openpi.training import config as config_lib
    from openpi.training import data_loader as data_loader_lib
    from openpi.training import weight_loaders

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--params", type=pathlib.Path, required=True)
    parser.add_argument("--config-name", default="pi05_yam_mem_v4_stage4c")
    parser.add_argument("--split", choices=("development", "train"), default="development")
    parser.add_argument("--batches", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--max-decode-steps", type=int, default=12)
    parser.add_argument("--num-steps", type=int, default=10, help="flow denoising steps")
    parser.add_argument(
        "--bank",
        choices=("semantic", "visual", "both"),
        default="semantic",
        help="which bank(s) the reset/donor read overrides act on.",
    )
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)

    output_dir = args.output_dir
    report_path = output_dir / "closed_loop_eval.json"
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
    if not getattr(config.model, "memory_v4_dual_bank", False):
        raise SystemExit(f"{args.config_name} is not a v4 dual-bank config.")
    if getattr(config.model, "memory_fact_oracle_writes", False):
        raise SystemExit("closed-loop replay needs predicted writes; Stage-2a oracle configs are not supported.")
    params = model_lib.restore_params(args.params, restore_type=np.ndarray)
    parameter_tree_sha256 = weight_loaders.parameter_tree_sha256(params)
    model = config.model.load(params)
    model.eval()
    loader = data_loader_lib.create_data_loader(
        config, sharding=None, shuffle=True, num_batches=args.batches, exact_resume=False
    )
    pg = tokenizer_lib.FASTSubtaskTokenizer(config.model.max_token_len)._paligemma_tokenizer  # noqa: SLF001
    # Same terminator the training subtasks were tokenized with (trailing "\n" of the segment).
    stop_token = int(pg.encode("placeholder subtask\n")[-1])
    if args.max_decode_steps > model.causal_token_len:
        raise SystemExit(
            f"--max-decode-steps {args.max_decode_steps} exceeds causal_token_len {model.causal_token_len}."
        )

    sampler = nnx_utils.module_jit(
        model.sample_with_memory, static_argnames=("stop_token", "max_decode_steps", "num_steps", "write_mode")
    )
    visual_gap_decay = nnx_utils.module_jit(model.memory.analytic_decay)
    semantic_gap_decay = nnx_utils.module_jit(model.memory_semantic.analytic_decay)

    def select(apply, new, old):
        return jax.tree.map(lambda n, o: jnp.where(apply.reshape((-1,) + (1,) * (n.ndim - 1)), n, o), new, old)

    def roll(state):
        return jax.tree.map(lambda x: jnp.roll(x, 1, axis=0), state)

    def overrides_for(cond: str, visual, semantic, batch: int) -> dict:
        if cond == "normal":
            return {}
        blank_v, blank_s = model.memory.init_state(batch), model.memory_semantic.init_state(batch)
        alt_v = blank_v if cond == "reset" else roll(visual)
        alt_s = blank_s if cond == "reset" else roll(semantic)
        out = {}
        if args.bank in ("semantic", "both"):
            out["v4_read_semantic_state"] = alt_s
        if args.bank in ("visual", "both"):
            out["v4_read_visual_state"] = alt_v
        return out

    fact_targets = int(model.memory_fact_targets)
    unknown_class = fact_targets - 1
    records: list[dict] = []
    windows: list[dict] = []
    rng = jax.random.key(args.seed)
    t_start = time.perf_counter()
    for index, (raw_observation, raw_actions) in enumerate(loader):
        sides = np.asarray(jax.device_get(raw_observation.seq_side_label))
        perm = alternate_sides_permutation(sides)
        observation = jax.tree.map(lambda x, p=perm: x[p], raw_observation)
        actions = np.asarray(jax.device_get(raw_actions))[perm]
        sides = sides[perm]
        batch, steps = np.asarray(observation.seq_step_mask).shape
        donor_sides = np.roll(sides, 1)
        fact_labels = np.asarray(jax.device_get(observation.seq_fact_labels))  # [b, F]
        donor_fact_labels = np.roll(fact_labels, 1, axis=0)
        real_slot = (fact_labels >= 0) & (fact_labels < fact_targets) & (fact_labels != unknown_class)
        prompted_slot = np.full(batch, -1, dtype=np.int64)
        expected_donor_side = np.full(batch, -1, dtype=np.int64)
        for b in range(batch):
            matches = np.flatnonzero(fact_labels[b, :2] == sides[b])
            if matches.size == 1:
                prompted_slot[b] = int(matches[0])
                expected_donor_side[b] = int(donor_fact_labels[b, prompted_slot[b]])

        causal = np.asarray(jax.device_get(observation.tokenized_causal))  # [b, T, cl]
        causal_mask = np.asarray(jax.device_get(observation.tokenized_causal_mask))
        fast_mask = np.asarray(jax.device_get(observation.causal_fast_mask))
        if causal.shape[-1] != model.causal_token_len:
            raise SystemExit(
                f"dataset causal length {causal.shape[-1]} != model causal_token_len {model.causal_token_len}."
            )
        text_mask = causal_mask & ~fast_mask
        token_count = causal_mask.sum(axis=-1)  # [b, T]
        has_side = np.any(text_mask & np.isin(causal, (LEFT_TOKEN, RIGHT_TOKEN)), axis=-1)  # [b, T]
        swapped_causal = swap_side_tokens(causal, causal_mask, fast_mask)

        step_valid = np.asarray(jax.device_get(observation.seq_step_mask))
        write_mask = np.asarray(jax.device_get(observation.seq_write_mask))
        decision_mask = np.asarray(jax.device_get(observation.seq_decision_mask))
        read_state_valid = np.asarray(jax.device_get(observation.seq_read_state_valid))
        gap_before = np.asarray(jax.device_get(observation.seq_decay_gap_before)).astype(np.int32)
        fact_observable = np.asarray(jax.device_get(observation.seq_fact_observable))  # [b, T, F]
        token_state_mask = observation.token_state_mask

        visual = model.memory.init_state(batch)
        semantic = model.memory_semantic.init_state(batch)
        sem_written = np.zeros(fact_labels.shape, dtype=bool)
        commits = np.zeros(batch, dtype=np.int64)
        eligible = np.zeros(batch, dtype=np.int64)
        commit_correct = np.zeros(batch, dtype=np.int64)
        observable_real = np.zeros(batch, dtype=np.int64)
        observable_predicted_real = np.zeros(batch, dtype=np.int64)
        action_sq_err = np.zeros(batch, dtype=np.float64)
        action_terms = np.zeros(batch, dtype=np.int64)
        decision_order = np.zeros(batch, dtype=np.int64)
        for t in range(steps):
            transition_valid = step_valid[:, t] & (gap_before[:, t] >= 0)
            gap = np.where(transition_valid & (gap_before[:, t] > 0), gap_before[:, t], 0).astype(np.int32)
            if np.any(gap > 0):
                # Skip-O sparse clock: the omitted write-free transitions happen before this
                # step's read, on both banks (shared clock), exactly as the sequence scan.
                apply = jnp.asarray(gap > 0)
                gap_arr = jnp.asarray(gap)
                visual = select(apply, visual_gap_decay(visual, gap_arr)[0], visual)
                semantic = select(apply, semantic_gap_decay(semantic, gap_arr)[0], semantic)
            obs_t = model_lib.Observation(
                images={k: v[:, t] for k, v in observation.images.items()},
                image_masks={k: jnp.ones((batch,), dtype=bool) for k in observation.images},
                state=observation.state[:, t],
                tokenized_prompt=observation.tokenized_prompt[:, t],
                tokenized_prompt_mask=observation.tokenized_prompt_mask[:, t],
                token_state_mask=None if token_state_mask is None else token_state_mask[:, t],
            )
            write_now = write_mask[:, t] & transition_valid
            clock = {
                "v35_transition_valid": jnp.asarray(transition_valid),
                "v35_write_mask": jnp.asarray(write_now),
            }
            common = {
                "stop_token": stop_token,
                "max_decode_steps": args.max_decode_steps,
                "num_steps": args.num_steps,
            }
            step_rng = jax.random.fold_in(rng, index * 10_000 + t)
            sampled_actions, visual_next, aux = sampler(
                step_rng, obs_t, visual, semantic_state=semantic, **clock, **common
            )
            semantic_next = aux["v4_semantic_state"]
            commit_applied = np.asarray(jax.device_get(aux["v4_sem_commit_applied"]))  # [b, F]
            predicted = np.asarray(jax.device_get(aux["v4_fact_predicted"]))  # [b, F]
            write_eligible = np.asarray(jax.device_get(aux["v4_fact_write_eligible"]))
            commits += commit_applied.sum(axis=-1)
            eligible += (write_eligible & write_now[:, None]).sum(axis=-1)
            commit_correct += (commit_applied & (predicted == fact_labels)).sum(axis=-1)
            observable_now = fact_observable[:, t] & real_slot & transition_valid[:, None]
            observable_real += observable_now.sum(axis=-1)
            observable_predicted_real += (observable_now & (predicted == fact_labels)).sum(axis=-1)
            sq = np.mean(np.square(np.asarray(jax.device_get(sampled_actions)) - actions[:, t]), axis=(-1, -2))
            action_sq_err += np.where(transition_valid, sq, 0.0)
            action_terms += transition_valid.astype(np.int64)

            decide = decision_mask[:, t] & transition_valid
            if np.any(decide):
                tokens_by_cond, d_by_cond, logp_by_cond = {}, {}, {}
                forced_true = {
                    "forced_subtask_tokens": jnp.asarray(causal[:, t]),
                    "forced_subtask_mask": jnp.asarray(causal_mask[:, t]),
                }
                forced_swap = {
                    "forced_subtask_tokens": jnp.asarray(swapped_causal[:, t]),
                    "forced_subtask_mask": jnp.asarray(causal_mask[:, t]),
                }
                for cond in CONDITIONS:
                    override = overrides_for(cond, visual, semantic, batch)
                    if cond == "normal":
                        cond_aux = aux
                    else:
                        _, _, cond_aux = sampler(
                            step_rng,
                            obs_t,
                            visual,
                            semantic_state=semantic,
                            write_mode="frozen",
                            **override,
                            **clock,
                            **common,
                        )
                    tokens_by_cond[cond] = (
                        np.asarray(jax.device_get(cond_aux["tokens"])),
                        np.asarray(jax.device_get(cond_aux["token_mask"])),
                    )
                    logps = []
                    for forced in (forced_true, forced_swap):
                        _, _, f_aux = sampler(
                            step_rng,
                            obs_t,
                            visual,
                            semantic_state=semantic,
                            write_mode="frozen",
                            **forced,
                            **override,
                            **clock,
                            **common,
                        )
                        logps.append(np.asarray(jax.device_get(f_aux["conditioned_subtask_logp"])))
                    logp_by_cond[cond] = logps
                    d_by_cond[cond] = logps[0] - logps[1]
                read_logits = np.asarray(jax.device_get(aux["v4_fact_read_logits"]))  # [b, F, T]
                read_pred = np.argmax(read_logits, axis=-1)
                for b in np.flatnonzero(decide):
                    read_slots = real_slot[b] & sem_written[b] & read_state_valid[b, t]
                    record = {
                        "batch": index,
                        "row": int(b),
                        "step": int(t),
                        "decision_order": int(decision_order[b]),
                        "side": int(sides[b]),
                        "donor_side": int(donor_sides[b]),
                        "fact_labels": fact_labels[b].tolist(),
                        "donor_fact_labels": donor_fact_labels[b].tolist(),
                        "prompted_slot": int(prompted_slot[b]),
                        "expected_donor_side": int(expected_donor_side[b]),
                        "donor_expected_valid": bool(expected_donor_side[b] in (0, 1)),
                        "donor_mismatched": bool(
                            expected_donor_side[b] in (0, 1) and expected_donor_side[b] != int(sides[b])
                        ),
                        "true_text": pg.decode(causal[b, t][text_mask[b, t]].tolist()).strip(),
                        "decision_tokens": int(token_count[b, t]),
                        "has_side_token": bool(has_side[b, t]),
                        "included": bool(has_side[b, t] and token_count[b, t] > 0),
                        "read_pred": read_pred[b].tolist(),
                        "read_correct": int(np.sum(read_slots & (read_pred[b] == fact_labels[b]))),
                        "read_count": int(np.sum(read_slots)),
                        "commits_so_far": int(commits[b]),
                    }
                    for cond in CONDITIONS:
                        tokens, mask = tokens_by_cond[cond]
                        record[f"pred_text_{cond}"] = pg.decode(tokens[b][mask[b]].tolist()).strip()
                        record[f"pred_side_{cond}"] = side_from_tokens(tokens[b], mask[b])
                        record[f"D_{cond}"] = float(d_by_cond[cond][b])
                        record[f"logp_true_{cond}"] = float(logp_by_cond[cond][0][b])
                        record[f"logp_swap_{cond}"] = float(logp_by_cond[cond][1][b])
                    records.append(record)
                    decision_order[b] += 1

            # Carry the model's OWN transitions (normal condition) into the next step.
            visual, semantic = visual_next, semantic_next
            sem_written |= commit_applied
        windows.extend(
            {
                "batch": index,
                "row": int(b),
                "side": int(sides[b]),
                "fact_labels": fact_labels[b].tolist(),
                "valid_steps": int(np.sum(step_valid[b] & (gap_before[b] >= 0))),
                "decision_steps": int(decision_order[b]),
                "sem_commits": int(commits[b]),
                "sem_write_eligible": int(eligible[b]),
                "sem_commit_correct": int(commit_correct[b]),
                "observable_real_slot_steps": int(observable_real[b]),
                "observable_predicted_correct": int(observable_predicted_real[b]),
                "action_mse_normal": float(action_sq_err[b] / max(action_terms[b], 1)),
            }
            for b in range(batch)
        )
        elapsed = time.perf_counter() - t_start
        print(
            f"batch {index + 1}/{args.batches} done ({elapsed:.0f}s): decision records so far {len(records)}, "
            f"commits/window {np.mean([w['sem_commits'] for w in windows]):.2f}",
            flush=True,
        )

    summary = summarize(records)
    summary_first = summarize(records, first_step_only=True)
    window_summary = {
        "windows": len(windows),
        "sem_commits_per_window": float(np.mean([w["sem_commits"] for w in windows])) if windows else 0.0,
        "sem_write_eligible_per_window": float(np.mean([w["sem_write_eligible"] for w in windows])) if windows else 0.0,
        "commit_write_accuracy": float(
            sum(w["sem_commit_correct"] for w in windows) / max(sum(w["sem_commits"] for w in windows), 1)
        ),
        "observable_fact_accuracy": float(
            sum(w["observable_predicted_correct"] for w in windows)
            / max(sum(w["observable_real_slot_steps"] for w in windows), 1)
        ),
        "action_mse_normal": float(np.mean([w["action_mse_normal"] for w in windows])) if windows else 0.0,
    }

    def show(title: str, s: dict) -> None:
        # Print the headline BEFORE writing the report so the numbers survive any write failure.
        print(f"[{title}] decision_steps={s['decision_steps']} sequences={s['sequences']}")
        for cond in CONDITIONS:
            if f"{cond}_free_side_accuracy" in s:
                d_acc = s.get(f"{cond}_D_side_accuracy", float("nan"))
                d_margin = s.get(f"{cond}_D_mean_margin", float("nan"))
                print(
                    f"  {cond:6s} free side_accuracy={s[f'{cond}_free_side_accuracy']:.3f} "
                    f"no_side={s[f'{cond}_free_no_side_rate']:.3f} wrong_side={s[f'{cond}_free_wrong_side_rate']:.3f} "
                    f"| D side_accuracy={d_acc:.3f} mean_margin={d_margin:+.3f}"
                )
        if "donor_free_follows_content_rate" in s:
            print(
                f"  donor FOLLOWS-CONTENT free={s['donor_free_follows_content_rate']:.3f} "
                f"D={s.get('donor_D_follows_content_rate', float('nan')):.3f} "
                f"(usable={s['donor_expected_valid']}, mismatched={s['donor_mismatched_pairs']}: "
                f"free flip={s.get('donor_free_flip_rate_mismatched', float('nan')):.3f} "
                f"D flip={s.get('donor_D_flip_rate_mismatched', float('nan')):.3f})"
            )
        if "read_accuracy_normal" in s:
            print(
                f"  read accuracy (normal, written real slots)={s['read_accuracy_normal']:.3f} terms={s['read_terms']}"
            )

    show("all decision steps", summary)
    show("first decision step per sequence", summary_first)
    print(
        f"[windows] n={window_summary['windows']} commits/window={window_summary['sem_commits_per_window']:.2f} "
        f"eligible/window={window_summary['sem_write_eligible_per_window']:.2f} "
        f"commit write acc={window_summary['commit_write_accuracy']:.3f} "
        f"observable fact acc={window_summary['observable_fact_accuracy']:.3f} "
        f"action mse={window_summary['action_mse_normal']:.4f}"
    )

    def json_default(value):
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
        "max_decode_steps": args.max_decode_steps,
        "num_steps": args.num_steps,
        "parameter_tree_sha256": parameter_tree_sha256,
        "summary": summary,
        "summary_first_decision_step": summary_first,
        "window_summary": window_summary,
        "windows": windows,
        "records": records,
    }
    body = json.dumps(report, indent=2, sort_keys=True, default=json_default) + "\n"
    report["report_sha256"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=json_default) + "\n")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    sys.exit(main())
