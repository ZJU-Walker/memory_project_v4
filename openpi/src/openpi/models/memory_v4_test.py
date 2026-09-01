"""v4 tests for the multi-slot semantic-bank commit (`delta_write_kv_multi`)."""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import memory


def _delta_config(**overrides) -> memory.MemoryConfig:
    values = {
        "d_input": 16,
        "d_key": 8,
        "hidden_dims": (12, 10),
        "d_value": 14,
        "mlp_l2norm": True,
        "blank_initial_output": True,
        "write_rule": "delta_output",
        "association_mode": "pooled_frame",
        "delta_rate": 1.0,
        "alpha_step": 0.01,
    }
    values.update(overrides)
    return memory.MemoryConfig(**values)


def _memory(config: memory.MemoryConfig | None = None) -> memory.TitansMemory:
    return memory.TitansMemory(config or _delta_config(), rngs=nnx.Rngs(0))


def _slot_pairs(config: memory.MemoryConfig, *, batch: int = 2, slots: int = 4):
    k = jax.random.normal(jax.random.key(30), (batch, slots, config.d_key), dtype=jnp.float32)
    v = jax.random.normal(jax.random.key(31), (batch, slots, config.d_value), dtype=jnp.float32)
    return k, v


def _nonzero_state(mem: memory.TitansMemory, *, batch: int = 2) -> memory.MemoryState:
    state = mem.init_state(batch)
    fast = dict(state.fast_weights)
    w3_name = mem._output_weight_name  # noqa: SLF001
    fast[w3_name] = 0.15 * jax.random.normal(jax.random.key(32), fast[w3_name].shape, dtype=jnp.float32)
    return memory.MemoryState(fast, state.momentum)


def _assert_states_equal(actual: memory.MemoryState, expected: memory.MemoryState) -> None:
    for name, leaf in expected.fast_weights.items():
        np.testing.assert_array_equal(np.asarray(actual.fast_weights[name]), np.asarray(leaf), err_msg=name)
    for name, leaf in expected.momentum.items():
        np.testing.assert_array_equal(np.asarray(actual.momentum[name]), np.asarray(leaf), err_msg=name)


def test_multi_commit_requires_the_delta_rule():
    mem = _memory(memory.MemoryConfig(d_input=16, d_key=8, hidden_dims=(12, 10), d_value=14))
    state = mem.init_state(2)
    k, v = _slot_pairs(mem.config)
    with pytest.raises(ValueError, match="delta_write_kv_multi requires"):
        mem.delta_write_kv_multi(state, k, v, jnp.ones((2, 4), dtype=bool))


def test_single_committed_slot_matches_delta_write_kv_bitwise():
    mem = _memory()
    state = _nonzero_state(mem)
    k, v = _slot_pairs(mem.config)
    mask = jnp.zeros((2, 4), dtype=bool).at[:, 2].set(True)
    multi_state, multi_aux = mem.delta_write_kv_multi(state, k, v, mask)
    single_state, single_aux = mem.delta_write_kv(state, k[:, 2:3, :], v[:, 2:3, :])
    _assert_states_equal(multi_state, single_state)
    np.testing.assert_array_equal(np.asarray(multi_aux["commit_applied"][:, 2]), np.asarray(single_aux["commit_applied"]))
    assert not bool(jnp.any(multi_aux["commit_applied"][:, [0, 1, 3]]))
    np.testing.assert_array_equal(np.asarray(multi_aux["num_commits"]), np.ones(2, dtype=np.int32))


def test_all_masked_off_is_bitwise_analytic_decay_of_one_step():
    mem = _memory()
    state = _nonzero_state(mem)
    k, v = _slot_pairs(mem.config)
    multi_state, multi_aux = mem.delta_write_kv_multi(state, k, v, jnp.zeros((2, 4), dtype=bool))
    decay_state, _ = mem.analytic_decay(state, 1)
    _assert_states_equal(multi_state, decay_state)
    assert not bool(jnp.any(multi_aux["commit_applied"]))
    np.testing.assert_array_equal(np.asarray(multi_aux["num_commits"]), np.zeros(2, dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(multi_aux["delta_w3_norm"]), np.zeros(2, dtype=np.float32))


def test_sequential_commits_recall_every_slot_and_last_slot_exactly():
    # Random keys in the default 8-dim test geometry overlap ~0.3 pairwise, which makes the
    # earlier slots' cross-commit interference dominate (measured ~0.76 of the unit target) --
    # correct mechanism, wrong geometry for a recall assertion. Use a moderately wide bank
    # where the overlap statistics resemble the production one (hidden overlap ~1/sqrt(96)).
    mem = _memory(_delta_config(d_key=48, hidden_dims=(96, 96), d_value=40))
    state = mem.init_state(2)
    k, v = _slot_pairs(mem.config)
    new_state, aux = mem.delta_write_kv_multi(state, k, v, jnp.ones((2, 4), dtype=bool))
    assert bool(jnp.all(aux["commit_applied"]))
    # The final slot's residual against the final state is exact (delta_rate=1, nothing after it).
    np.testing.assert_allclose(np.asarray(aux["final_read_residual_norm"][:, 3]), 0.0, atol=1e-5)
    # Earlier slots deviate only through same-step hidden overlap; recall must dominate the
    # unit-norm target scale a fresh (zero-reading) memory would leave untouched (residual 1.0).
    # A single unlucky random key pair can overlap substantially at this width (measured one
    # slot at 0.58), so the interference claim is statistical; production geometry (1024-wide
    # hidden) is pinned by the Stage-0 battery, not this unit test.
    assert float(jnp.max(aux["final_read_residual_norm"])) < 0.8
    assert float(jnp.mean(aux["final_read_residual_norm"])) < 0.3
    retrieved = mem.read_key(new_state, aux["pooled_key"])
    recall_error = jnp.linalg.norm(retrieved - aux["pooled_value"], axis=-1)
    np.testing.assert_array_equal(np.asarray(recall_error), np.asarray(aux["final_read_residual_norm"]))


def test_degenerate_slot_fails_closed_while_others_commit():
    mem = _memory()
    state = mem.init_state(2)
    k, v = _slot_pairs(mem.config)
    k = k.at[:, 1, :].set(0.0)
    new_state, aux = mem.delta_write_kv_multi(state, k, v, jnp.ones((2, 4), dtype=bool))
    assert not bool(jnp.any(aux["association_valid"][:, 1]))
    assert not bool(jnp.any(aux["commit_applied"][:, 1]))
    np.testing.assert_array_equal(np.asarray(aux["num_commits"]), np.full(2, 3, dtype=np.int32))
    assert bool(jnp.all(jnp.isfinite(new_state.fast_weights[mem._output_weight_name])))  # noqa: SLF001


def test_multi_commit_enforces_fp32_state_invariants_under_bf16_inputs():
    mem = _memory()
    state = _nonzero_state(mem)
    k, v = _slot_pairs(mem.config)
    new_state, aux = mem.delta_write_kv_multi(
        state, k.astype(jnp.bfloat16), v.astype(jnp.bfloat16), jnp.ones((2, 4), dtype=bool)
    )
    for leaf in (*new_state.fast_weights.values(), *new_state.momentum.values()):
        assert leaf.dtype == jnp.float32
    np.testing.assert_array_equal(
        np.asarray(new_state.fast_weights[mem._output_bias_name]),  # noqa: SLF001
        np.zeros_like(np.asarray(new_state.fast_weights[mem._output_bias_name])),  # noqa: SLF001
    )
    for leaf in new_state.momentum.values():
        np.testing.assert_array_equal(np.asarray(leaf), np.zeros_like(np.asarray(leaf)))
    for value in aux.values():
        assert value.dtype in (jnp.float32, jnp.bool_, jnp.int32)


def test_masked_slot_values_receive_zero_gradient_and_committed_ones_do_not():
    mem = _memory()
    state = _nonzero_state(mem)
    k, v = _slot_pairs(mem.config)
    mask = jnp.zeros((2, 4), dtype=bool).at[:, 0].set(True)
    queries = jax.random.normal(jax.random.key(33), (2, 3, mem.config.d_key), dtype=jnp.float32)

    def read_norm_after_write(v_in):
        new_state, _ = mem.delta_write_kv_multi(state, k, v_in, mask)
        return jnp.sum(jnp.square(mem.read_key(new_state, memory._l2_norm(queries))))  # noqa: SLF001

    grad = jax.grad(read_norm_after_write)(v)
    assert bool(jnp.all(jnp.isfinite(grad)))
    assert float(jnp.linalg.norm(grad[:, 0])) > 0.0
    np.testing.assert_array_equal(np.asarray(grad[:, 1:]), np.zeros_like(np.asarray(grad[:, 1:])))
