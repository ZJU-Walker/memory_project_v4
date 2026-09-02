"""Debug probe: what does the read-side fact head actually see and emit?

Hooks `v4_fact_read_logits` during one real dev-split sequence forward (the training
objective, eval mode) and records, per step and slot, the raw semantic retrieval norm and
the read logits, alongside the episode's fact labels and the decision/write masks. Prints a
compact table so a constant per-term read CE can be traced to its cause.
"""

import argparse
import dataclasses
import pathlib
import sys

import pyarrow.parquet  # noqa: F401  isort: skip

import numpy as np


def main(argv=None) -> None:
    import jax
    import jax.numpy as jnp

    from openpi.models import model as model_lib
    from openpi.shared import nnx_utils
    from openpi.training import config as config_lib
    from openpi.training import data_loader as data_loader_lib

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=pathlib.Path, required=True)
    parser.add_argument("--config-name", default="pi05_yam_mem_v4_stage2a")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=4)
    args = parser.parse_args(argv)

    config = config_lib.get_config(args.config_name)
    config = dataclasses.replace(
        config,
        batch_size=args.batch_size,
        num_workers=0,
        seed=args.seed,
        v4_graft_sources=(),
        data=dataclasses.replace(
            config.data,
            base_config=dataclasses.replace(config.data.base_config, memory_manifest_split="development"),
        ),
    )
    params = model_lib.restore_params(args.params, restore_type=np.ndarray)
    model = config.model.load(params)
    model.eval()
    loader = data_loader_lib.create_data_loader(config, sharding=None, shuffle=True, num_batches=1, exact_resume=False)
    observation, actions = next(iter(loader))

    records: list[tuple[np.ndarray, np.ndarray]] = []

    def record(retrieved, logits):
        records.append((np.asarray(retrieved), np.asarray(logits)))

    original = model.v4_fact_read_logits

    def hooked(retrieved):
        logits = original(retrieved)
        jax.debug.callback(record, retrieved, logits)
        return logits

    model.v4_fact_read_logits = hooked
    loss_fn = nnx_utils.module_jit(model._compute_sequence_loss_v32, static_argnames=("train",))  # noqa: SLF001
    losses = loss_fn(jax.random.key(args.seed), observation, actions, train=False)
    jax.block_until_ready(losses["ce"])

    labels = np.asarray(observation.seq_fact_labels)  # [b, f]
    decision = np.asarray(observation.seq_decision_mask)  # [b, t]
    write = np.asarray(observation.seq_write_mask)
    state_valid = np.asarray(observation.seq_read_state_valid)
    observable = np.asarray(observation.seq_fact_observable)  # [b, t, f]
    print(f"labels per sample (slot0, slot1): {labels[:, :2].tolist()}")
    print(f"read terms reported: {float(losses['v4_fact_read_count']):.0f}  read CE sum: {float(losses['v4_fact_read_ce_sum']):.4f}")
    print(f"per-step records captured: {len(records)} (one per scan step)")
    for t, (retrieved, logits) in enumerate(records):
        for b in range(retrieved.shape[0]):
            flags = f"D={int(decision[b, t])} W={int(write[b, t])} SV={int(state_valid[b, t])} obs={observable[b, t, :2].astype(int).tolist()}"
            norms = np.linalg.norm(retrieved[b, :2], axis=-1)
            logp = logits[b, :2] - np.log(np.sum(np.exp(logits[b, :2] - logits[b, :2].max(-1, keepdims=True)), -1, keepdims=True)) - logits[b, :2].max(-1, keepdims=True)
            ce = [-float(logp[s, labels[b, s]]) for s in range(2)]
            print(
                f"t={t:2d} b={b} {flags} |r|=({norms[0]:.3f},{norms[1]:.3f}) "
                f"logits0={np.round(logits[b, 0], 3).tolist()} logits1={np.round(logits[b, 1], 3).tolist()} "
                f"ce=({ce[0]:.4f},{ce[1]:.4f})"
            )


if __name__ == "__main__":
    sys.exit(main())
