"""v4 dual-bank contracts: config gating, fact-head write intent, semantic bank surface."""

# ruff: noqa: SLF001

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import gemma
from openpi.models import memory
from openpi.models import pi0
from openpi.models import pi0_config
from openpi.models.pi0_v32_test import _single_observation
from openpi.models.pi0_v35_test import _TinyV35
from openpi.models.pi0_v35_test import _v35_sequence_observation


def _delta_memory_config(**overrides) -> memory.MemoryConfig:
    values = {
        "mlp_l2norm": True,
        "blank_initial_output": True,
        "write_rule": "delta_output",
        "association_mode": "pooled_frame",
        "delta_rate": 1.0,
        "alpha_step": 0.01,
    }
    values.update(overrides)
    return memory.MemoryConfig(**values)


def _v35_kwargs(**overrides) -> dict:
    """A minimal legal memory_v35_enabled Pi0Config keyword set (production widths)."""
    values = {
        "pi05": True,
        "predict_subtask": True,
        "predict_with_memory": True,
        "memory_architecture": "v32_layer8_dual_query",
        "memory_layer": 8,
        "memory_write_source": "query_compressed",
        "memory": _delta_memory_config(),
        "memory_v35_enabled": True,
        "memory_injection_mode": "tanh_rms",
        "memory_injection_gate_init": 0.5,
        "memory_freeze_injection_gate": True,
        "memory_write_side_loss_weight": 0.3,
        "memory_read_side_loss_weight": 0.3,
        "memory_time_consistent_augmentation": True,
    }
    values.update(overrides)
    return values


def test_v4_config_requires_delta_semantic_bank_and_v35_semantics():
    # Valid v4 config: dual bank on top of a legal v3.5 config.
    config = pi0_config.Pi0Config(
        **_v35_kwargs(
            memory_v4_dual_bank=True, memory_mask_zero_tokens=True, memory_semantic=_delta_memory_config()
        )
    )
    observation_spec, _ = config.inputs_spec(batch_size=2)
    assert observation_spec.seq_fact_labels.shape == (2, config.memory_fact_slots)
    assert observation_spec.seq_fact_observable.shape == (2, config.memory_seq_steps, config.memory_fact_slots)

    # The v3.5 spec is unchanged when the flag is off.
    off = pi0_config.Pi0Config(**_v35_kwargs())
    off_spec, _ = off.inputs_spec(batch_size=2)
    assert off_spec.seq_fact_labels is None
    assert off_spec.seq_fact_observable is None

    # A blank bank injects exactly-zero tokens: v4 must mask them (Stage 2a r2 regression).
    with pytest.raises(ValueError, match="memory_mask_zero_tokens"):
        pi0_config.Pi0Config(**_v35_kwargs(memory_v4_dual_bank=True, memory_semantic=_delta_memory_config()))
    with pytest.raises(ValueError, match="pooled-frame delta_output"):
        pi0_config.Pi0Config(
            **_v35_kwargs(
                memory_v4_dual_bank=True, memory_mask_zero_tokens=True, memory_semantic=memory.MemoryConfig()
            )
        )
    with pytest.raises(ValueError, match="alpha_step"):
        pi0_config.Pi0Config(
            **_v35_kwargs(
                memory_v4_dual_bank=True,
                memory_mask_zero_tokens=True,
                memory_semantic=_delta_memory_config(alpha_step=0.02),
            )
        )
    with pytest.raises(ValueError, match="memory_v35_enabled"):
        pi0_config.Pi0Config(
            **_v35_kwargs(
                memory_v35_enabled=False,
                memory_injection_gate_init=0.5,
                memory_freeze_injection_gate=False,
                memory_write_side_loss_weight=0.0,
                memory_read_side_loss_weight=0.0,
                memory_time_consistent_augmentation=False,
                memory_v4_dual_bank=True,
                memory_mask_zero_tokens=True,
                memory_semantic=_delta_memory_config(),
            )
        )
    with pytest.raises(ValueError, match="fact losses require"):
        pi0_config.Pi0Config(**_v35_kwargs(memory_fact_loss_weight=0.1))
    with pytest.raises(ValueError, match="PaliGemma width"):
        pi0_config.Pi0Config(
            **_v35_kwargs(
                memory_v4_dual_bank=True,
                memory_mask_zero_tokens=True,
                memory_semantic=_delta_memory_config(d_value=64),
            )
        )


class _TinyV4(nnx.Module):
    """The v4 semantic surface on a 64-wide stand-in (the _TinyV32 pattern)."""

    v4_fact_keys = pi0.Pi0.v4_fact_keys
    _v4_fact_slots = pi0.Pi0._v4_fact_slots
    v4_fact_logits = pi0.Pi0.v4_fact_logits
    v4_fact_write_intent = pi0.Pi0.v4_fact_write_intent
    v4_semantic_write = pi0.Pi0.v4_semantic_write
    v4_semantic_read = pi0.Pi0.v4_semantic_read
    v4_fact_read_logits = pi0.Pi0.v4_fact_read_logits
    _v4_inject_semantic = pi0.Pi0._v4_inject_semantic
    _v32_top_patch_valid = pi0.Pi0._v32_top_patch_valid

    def __init__(self, rngs: nnx.Rngs):
        width = 64
        self.memory_v4_dual_bank = True
        self.memory_fact_slots = 4
        self.memory_fact_targets = 3
        self.memory_fact_write_conf = 0.6
        self.memory_sem_injection_c = 12.4
        self.memory_sem_injection_tau = 0.02
        self.top_patch_valid = None
        semantic_config = memory.MemoryConfig(
            d_input=width,
            d_key=16,
            hidden_dims=(24,),
            d_value=width,
            mlp_l2norm=True,
            blank_initial_output=True,
            write_rule="delta_output",
            association_mode="pooled_frame",
            alpha_step=0.01,
        )
        self.memory_semantic = memory.TitansMemory(semantic_config, rngs=rngs)
        self.fact_keys = nnx.Param(jax.random.normal(rngs.params(), (4, 16), dtype=jnp.float32))
        self.fact_compressor = pi0.MemoryQueryCompressor(num_queries=4, width=width, num_heads=8, rngs=rngs)
        self.fact_logit_head = nnx.Linear(width, 3, rngs=rngs)
        self.fact_value_embed = nnx.Param(jax.random.normal(rngs.params(), (3, width), dtype=jnp.float32))
        self.memory_fact_read_head = nnx.Linear(width, 3, rngs=rngs)
        self.memory_sem_inject_w = nnx.Param(
            jnp.full((width,), jnp.arctanh(jnp.float32(0.5)), dtype=jnp.float32)
        )
        self.memory_sem_slot_embedding = nnx.Param(jnp.zeros((4, width), dtype=jnp.float32))


@pytest.fixture(scope="module")
def tiny_v4():
    return _TinyV4(nnx.Rngs(0))


def test_fact_logits_have_slot_shape_and_are_deterministic(tiny_v4):
    h8_top = jax.random.normal(jax.random.key(1), (2, 9, 64), dtype=jnp.float32)
    logits = tiny_v4.v4_fact_logits(h8_top)
    assert logits.shape == (2, 4, 3)
    assert logits.dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(logits), np.asarray(tiny_v4.v4_fact_logits(h8_top)))


def test_write_intent_gates_on_confidence_and_never_writes_unknown(tiny_v4):
    logits = jnp.zeros((1, 4, 3), dtype=jnp.float32)
    logits = logits.at[0, 0, 0].set(8.0)  # confident real target
    logits = logits.at[0, 1, 2].set(8.0)  # confident `unknown`
    # slot 2: uniform (confidence 1/3 < 0.6); slot 3: mildly target1 (still < 0.6)
    logits = logits.at[0, 3, 1].set(0.5)
    intent = tiny_v4.v4_fact_write_intent(logits)
    np.testing.assert_array_equal(
        np.asarray(intent["write_eligible"][0]), np.asarray([True, False, False, False])
    )
    np.testing.assert_array_equal(np.asarray(intent["predicted"][0, :2]), np.asarray([0, 2], dtype=np.int32))
    np.testing.assert_allclose(np.asarray(jnp.linalg.norm(intent["values"], axis=-1)), 1.0, atol=1e-5)
    np.testing.assert_allclose(np.asarray(jnp.linalg.norm(intent["keys"], axis=-1)), 1.0, atol=1e-5)


def test_semantic_write_then_read_recalls_committed_slots_only(tiny_v4):
    state = tiny_v4.memory_semantic.init_state(2)
    logits = jnp.zeros((2, 4, 3), dtype=jnp.float32)
    logits = logits.at[:, 0, 0].set(8.0)
    logits = logits.at[:, 2, 1].set(8.0)
    write_mask = jnp.asarray([True, False])
    new_state, aux = tiny_v4.v4_semantic_write(state, logits, write_mask)
    np.testing.assert_array_equal(
        np.asarray(aux["commit_applied"]),
        np.asarray([[True, False, True, False], [False, False, False, False]]),
    )
    # The last-committed slot is exact; the earlier one deviates only via same-step overlap.
    np.testing.assert_allclose(np.asarray(aux["final_read_residual_norm"][0, 2]), 0.0, atol=1e-5)
    assert float(aux["final_read_residual_norm"][0, 0]) < 0.5
    retrieved = tiny_v4.v4_semantic_read(new_state)
    intent = tiny_v4.v4_fact_write_intent(logits)
    recall = jnp.sum(retrieved[0] * intent["values"][0], axis=-1)  # cosine: both near unit norm
    assert float(recall[0]) > 0.8
    assert float(recall[2]) > 0.99
    # The masked sample never committed: its bank is fresh and reads exactly zero.
    np.testing.assert_array_equal(np.asarray(retrieved[1]), np.zeros_like(np.asarray(retrieved[1])))
    read_logits = tiny_v4.v4_fact_read_logits(retrieved)
    assert read_logits.shape == (2, 4, 3)


class _TinyV4Seq(_TinyV35):
    """The full v4 sequence path on the tiny v3.5 stand-in."""

    _memory_token_total = pi0.Pi0._memory_token_total
    v4_fact_keys = pi0.Pi0.v4_fact_keys
    _v4_fact_slots = pi0.Pi0._v4_fact_slots
    v4_fact_logits = pi0.Pi0.v4_fact_logits
    v4_fact_write_intent = pi0.Pi0.v4_fact_write_intent
    v4_fact_oracle_intent = pi0.Pi0.v4_fact_oracle_intent
    v4_semantic_write = pi0.Pi0.v4_semantic_write
    v4_semantic_read = pi0.Pi0.v4_semantic_read
    v4_fact_read_logits = pi0.Pi0.v4_fact_read_logits
    _v4_inject_semantic = pi0.Pi0._v4_inject_semantic
    v4_fact_probe_step = pi0.Pi0.v4_fact_probe_step

    def __init__(self, rngs: nnx.Rngs):
        super().__init__(rngs)
        width = 64
        self.memory_v4_dual_bank = True
        self.memory_fact_slots = 3
        self.memory_fact_targets = 3
        self.memory_fact_write_conf = 0.6
        self.memory_fact_loss_weight = 1.0
        self.memory_fact_read_loss_weight = 1.0
        self.memory_sem_injection_c = 12.4
        self.memory_sem_injection_tau = 0.02
        self.memory_semantic = memory.TitansMemory(
            memory.MemoryConfig(
                d_input=width,
                d_key=8,
                hidden_dims=(8,),
                d_value=width,
                mlp_l2norm=True,
                blank_initial_output=True,
                write_rule="delta_output",
                association_mode="pooled_frame",
                delta_rate=1.0,
                alpha_step=0.01,
            ),
            rngs=rngs,
        )
        self.fact_keys = nnx.Param(jax.random.normal(rngs.params(), (3, 8), dtype=jnp.float32))
        self.fact_compressor = pi0.MemoryQueryCompressor(num_queries=3, width=width, num_heads=8, rngs=rngs)
        # Deterministic confident head: every slot predicts target 0 regardless of input, so
        # the sequence contracts below are exact rather than dependent on random init.
        self.fact_logit_head = nnx.Linear(width, 3, rngs=rngs)
        self.fact_logit_head.kernel.value = jnp.zeros_like(self.fact_logit_head.kernel.value)
        self.fact_logit_head.bias.value = jnp.asarray([8.0, 0.0, -8.0], dtype=jnp.float32)
        self.fact_value_embed = nnx.Param(jax.random.normal(rngs.params(), (3, width), dtype=jnp.float32))
        self.memory_fact_read_head = nnx.Linear(width, 3, rngs=rngs)
        self.memory_sem_inject_w = nnx.Param(
            jnp.full((width,), jnp.arctanh(jnp.float32(0.5)), dtype=jnp.float32)
        )
        self.memory_sem_slot_embedding = nnx.Param(jnp.zeros((3, width), dtype=jnp.float32))


@pytest.fixture(scope="module")
def tiny_v4_seq():
    original_vocab = gemma.PALIGEMMA_VOCAB_SIZE
    try:
        gemma.PALIGEMMA_VOCAB_SIZE = 128
        yield _TinyV4Seq(nnx.Rngs(4))
    finally:
        gemma.PALIGEMMA_VOCAB_SIZE = original_vocab


def _v4_sequence_observation(**kwargs):
    observation = _v35_sequence_observation(**kwargs)
    return observation.replace(
        # Slots 0/1 carry real targets 0/1; slot 2 is unpopulated (`unknown`, observable
        # nowhere). Facts are observable only on the E step.
        seq_fact_labels=jnp.asarray([[0, 1, 2]], dtype=jnp.int32),
        seq_fact_observable=jnp.asarray([[[True, True, False], [False] * 3, [False] * 3]]),
    )


def test_v4_sequence_commits_confident_slots_supervises_facts_and_reads_back(tiny_v4_seq):
    assert tiny_v4_seq._memory_token_total == 16 + 3
    observation = _v4_sequence_observation()
    actions = jnp.zeros((1, 3, 4, 2), dtype=jnp.float32)
    losses = tiny_v4_seq._compute_sequence_loss_v32(jax.random.key(44), observation, actions, train=False)

    # The visual v3.5 contract is untouched by the semantic bank.
    np.testing.assert_array_equal(losses["v35_write_eligible_count"], 1.0)
    np.testing.assert_array_equal(losses["v35_commit_success_count"], 1.0)
    np.testing.assert_array_equal(losses["v35_read_state_valid_count"], 1.0)
    assert float(losses["v35_commit_relative_residual_max"]) <= 1e-5

    # The deterministic head is confident in target 0 for every slot: all 3 slots commit on
    # the single E step and nothing commits elsewhere.
    np.testing.assert_array_equal(losses["v4_sem_write_eligible_count"], 3.0)
    np.testing.assert_array_equal(losses["v4_sem_commit_count"], 3.0)
    np.testing.assert_array_equal(losses["v4_sem_degenerate_count"], 0.0)

    # Fact supervision: true targets for slots 0/1 on the E step (classes 0 and 1), the
    # mandatory `unknown` abstention on every other valid (step, slot) pair: 3 steps x 3
    # slots minus the 2 observable rows = 7.
    np.testing.assert_array_equal(losses["v4_fact_count_class"], jnp.asarray([1.0, 1.0, 7.0]))
    # Read-side supervision: the two real slots, written and decodable on the D step.
    np.testing.assert_array_equal(losses["v4_fact_read_count"], 2.0)

    # The D-step retrieval is live: three commits decayed over the sparse gap still read
    # clearly above zero.
    assert float(losses["v4_sem_raw_read_rms_sum"]) > 0.0
    assert float(losses["v4_sem_injected_pre_cast_rms_sum"]) > 0.0
    for key, value in losses.items():
        assert np.isfinite(np.asarray(value)).all(), key


def test_oracle_intent_embeds_the_truth_and_only_for_eligible_real_slots(tiny_v4_seq):
    targets = jnp.asarray([[0, 1, 2], [1, 0, 0]], dtype=jnp.int32)
    slot_mask = jnp.asarray([[True, True, True], [True, False, True]])
    intent = tiny_v4_seq.v4_fact_oracle_intent(targets, slot_mask)
    # `unknown` (2) and masked-off slots are never eligible; real observable slots are.
    np.testing.assert_array_equal(
        np.asarray(intent["write_eligible"]), np.asarray([[True, True, False], [True, False, True]])
    )
    embed = np.asarray(tiny_v4_seq.fact_value_embed.value)
    expected = embed[np.asarray(targets)]
    expected = expected / np.linalg.norm(expected, axis=-1, keepdims=True)
    np.testing.assert_allclose(np.asarray(intent["values"]), expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(np.asarray(intent["confidence"]), np.ones((2, 3), dtype=np.float32))


def test_v4_oracle_writes_commit_exactly_the_observable_real_slots(tiny_v4_seq):
    observation = _v4_sequence_observation()
    actions = jnp.zeros((1, 3, 4, 2), dtype=jnp.float32)
    tiny_v4_seq.memory_fact_oracle_writes = True
    tiny_v4_seq.memory_v4_visual_injection = False
    try:
        losses = tiny_v4_seq._compute_sequence_loss_v32(jax.random.key(46), observation, actions, train=False)
    finally:
        tiny_v4_seq.memory_fact_oracle_writes = False
        tiny_v4_seq.memory_v4_visual_injection = True
    # Slots 0/1 are observable real facts on the E step -> exactly 2 oracle commits (the
    # deterministic head would have committed all 3 in predicted mode).
    np.testing.assert_array_equal(losses["v4_sem_commit_count"], 2.0)
    np.testing.assert_array_equal(losses["v4_sem_write_eligible_count"], 2.0)
    # Semantic-only: the visual bank still commits (state evolves) but injects nothing.
    np.testing.assert_array_equal(losses["v35_commit_success_count"], 1.0)
    np.testing.assert_array_equal(losses["v35_injected_pre_cast_rms_sum"], 0.0)
    assert float(losses["v4_sem_injected_pre_cast_rms_sum"]) > 0.0
    for key, value in losses.items():
        assert np.isfinite(np.asarray(value)).all(), key


def test_v4_reset_intervention_blanks_only_the_decision_step_read(tiny_v4_seq):
    observation = _v4_sequence_observation()
    actions = jnp.zeros((1, 3, 4, 2), dtype=jnp.float32)
    normal = tiny_v4_seq._compute_sequence_loss_v32(jax.random.key(47), observation, actions, train=False)
    reset = tiny_v4_seq._compute_sequence_loss_v32(
        jax.random.key(47), observation, actions, train=False, v4_intervention="reset"
    )
    # Same writes either way (the carried state is untouched by the intervention) ...
    np.testing.assert_array_equal(reset["v4_sem_commit_count"], normal["v4_sem_commit_count"])
    # ... but the decision-step read is a fresh bank: less total raw retrieval, and the
    # read-side fact terms on that step decode from zeros.
    assert float(reset["v4_sem_raw_read_rms_sum"]) < float(normal["v4_sem_raw_read_rms_sum"])
    np.testing.assert_array_equal(reset["v4_decision_count"], 1.0)
    np.testing.assert_array_equal(reset["v4_use_count"], 1.0)
    for key, value in reset.items():
        assert np.isfinite(np.asarray(value)).all(), key
    # Batch of one: the donor roll is the sample itself, so donor == normal bitwise.
    donor = tiny_v4_seq._compute_sequence_loss_v32(
        jax.random.key(47), observation, actions, train=False, v4_intervention="donor"
    )
    np.testing.assert_array_equal(np.asarray(donor["v4_decision_ce_sum"]), np.asarray(normal["v4_decision_ce_sum"]))
    with pytest.raises(ValueError, match="evaluation-only"):
        tiny_v4_seq._compute_sequence_loss_v32(
            jax.random.key(47), observation, actions, train=True, v4_intervention="reset"
        )


def test_v4_visual_and_both_interventions_are_read_side_only(tiny_v4_seq):
    """Stage-4 battery controls: the visual bank can be reset/swapped on decision steps
    without touching the carried state or any write; 'both' composes the two banks."""
    observation = _v4_sequence_observation()
    actions = jnp.zeros((1, 3, 4, 2), dtype=jnp.float32)
    run = lambda name: tiny_v4_seq._compute_sequence_loss_v32(  # noqa: E731
        jax.random.key(49), observation, actions, train=False, v4_intervention=name
    )
    normal = run(None)
    visual_reset = run("visual_reset")
    both_reset = run("both_reset")
    # Writes are identical under every intervention (carried states untouched).
    for key in ("v35_commit_success_count", "v4_sem_commit_count"):
        np.testing.assert_array_equal(visual_reset[key], normal[key])
        np.testing.assert_array_equal(both_reset[key], normal[key])
    # A visual reset blanks only the VISUAL read on the decision step: the visual raw read
    # RMS drops, the semantic raw read is untouched; 'both' blanks the semantic read too.
    assert float(visual_reset["v35_raw_read_rms_sum"]) < float(normal["v35_raw_read_rms_sum"])
    np.testing.assert_allclose(
        np.asarray(visual_reset["v4_sem_raw_read_rms_sum"]), np.asarray(normal["v4_sem_raw_read_rms_sum"])
    )
    assert float(both_reset["v4_sem_raw_read_rms_sum"]) < float(normal["v4_sem_raw_read_rms_sum"])
    for losses in (visual_reset, both_reset):
        for key, value in losses.items():
            assert np.isfinite(np.asarray(value)).all(), key
    # Batch of one: a visual donor roll is the sample itself -> bitwise the normal run.
    visual_donor = run("visual_donor")
    np.testing.assert_array_equal(np.asarray(visual_donor["v4_decision_ce_sum"]), np.asarray(normal["v4_decision_ce_sum"]))
    with pytest.raises(ValueError, match="unsupported v4_intervention"):
        run("audio_reset")


def test_v4_sequence_requires_fact_fields_and_interface_requires_semantic_state(tiny_v4_seq):
    observation = _v35_sequence_observation()
    actions = jnp.zeros((1, 3, 4, 2), dtype=jnp.float32)
    with pytest.raises(ValueError, match="seq_fact_labels"):
        tiny_v4_seq._compute_sequence_loss_v32(jax.random.key(45), observation, actions, train=False)

    prefix = jnp.zeros((1, 4, 64), dtype=jnp.float32)
    mask = jnp.ones((1, 4), dtype=bool)
    ar = jnp.zeros((1, 4), dtype=jnp.int32)
    with pytest.raises(ValueError, match="semantic bank state"):
        tiny_v4_seq._v32_prepare_memory_interface(
            prefix, mask, ar, tiny_v4_seq.memory.init_state(1), top_token_count=2
        )


def test_semantic_injection_is_exactly_zero_for_a_fresh_bank_with_finite_backward(tiny_v4):
    fresh = jnp.zeros((2, 4, 64), dtype=jnp.float32)
    injected = tiny_v4._v4_inject_semantic(fresh)
    np.testing.assert_array_equal(np.asarray(injected), np.zeros_like(np.asarray(injected)))

    grad = jax.grad(lambda r: jnp.sum(tiny_v4._v4_inject_semantic(r)))(fresh)
    assert bool(jnp.all(jnp.isfinite(grad)))

    # Nonzero retrieval is scaled to the calibrated RMS band and gated by tanh(w)=0.5.
    retrieved = jax.random.normal(jax.random.key(2), (2, 4, 64), dtype=jnp.float32)
    injected = tiny_v4._v4_inject_semantic(retrieved)
    rms = jnp.sqrt(jnp.mean(jnp.square(injected), axis=-1))
    np.testing.assert_allclose(np.asarray(rms), 0.5 * 12.4, rtol=1e-4)


def test_mask_zero_tokens_hides_blank_slots_and_cuts_their_gradient(tiny_v4_seq):
    """Stage-2a r2 regression: an exactly-zero memory token must be invisible to every other
    row (no RMSNorm-singularity cotangent into its stream), while nonzero slots stay live."""
    single = _single_observation()
    prefix, mask, ar = tiny_v4_seq.embed_prefix(single)
    prefix_len = mask.shape[1]
    top_tokens = (prefix_len - tiny_v4_seq.max_token_len) // len(single.images)
    original_slots = tiny_v4_seq.memory_sem_slot_embedding.value

    def prepare(flag, slot_value):
        tiny_v4_seq.memory_mask_zero_tokens = flag
        tiny_v4_seq.memory_sem_slot_embedding.value = slot_value
        return tiny_v4_seq._v32_prepare_memory_prefix(
            prefix,
            mask,
            ar,
            tiny_v4_seq.memory.init_state(1),
            top_token_count=top_tokens,
            semantic_state=tiny_v4_seq.memory_semantic.init_state(1),
        )

    def prefix_loss(slot_value, flag):
        out = prepare(flag, slot_value)
        return jnp.sum(jnp.square(out["final_prefix"][:, :prefix_len].astype(jnp.float32)))

    zeros = jnp.zeros_like(original_slots)
    live = jnp.full_like(original_slots, 0.1)
    tiny_v4_seq.memory_v4_visual_injection = False
    try:
        # Fresh banks + visual injection off: every one of the 16 + 3 tokens is exactly zero.
        prepared_off = prepare(False, zeros)
        prepared_on = prepare(True, zeros)
        np.testing.assert_array_equal(np.asarray(prepared_on["memory_tokens"]), 0.0)
        np.testing.assert_array_equal(np.asarray(prepared_off["memory_valid"]), True)  # noqa: FBT003
        np.testing.assert_array_equal(np.asarray(prepared_on["memory_valid"]), False)  # noqa: FBT003
        # A nonzero slot embedding makes only the semantic slots visible again.
        prepared_live = prepare(True, live)
        np.testing.assert_array_equal(np.asarray(prepared_live["memory_valid"][0, :16]), False)  # noqa: FBT003
        np.testing.assert_array_equal(np.asarray(prepared_live["memory_valid"][0, 16:]), True)  # noqa: FBT003
        # Masking removes the zero slots from the non-memory rows: their outputs still
        # depend on the observation and are finite.
        assert bool(jnp.all(jnp.isfinite(prepared_on["final_prefix"])))

        # Gradient contract: the zero-K/V tokens still leak a cotangent into the slot
        # embedding without the mask; with it, the blank slots receive exactly nothing,
        # while a live slot embedding trains normally.
        grad_off = jax.grad(prefix_loss)(zeros, False)
        grad_on = jax.grad(prefix_loss)(zeros, True)
        grad_live = jax.grad(prefix_loss)(live, True)
    finally:
        tiny_v4_seq.memory_v4_visual_injection = True
        tiny_v4_seq.memory_mask_zero_tokens = False
        tiny_v4_seq.memory_sem_slot_embedding.value = original_slots
    assert float(jnp.linalg.norm(grad_off)) > 0.0
    np.testing.assert_array_equal(np.asarray(grad_on), 0.0)
    assert float(jnp.linalg.norm(grad_live)) > 0.0

    # The sequence objective under the Stage-2a recipe stays finite with the mask on and
    # supervises exactly the same read terms.
    observation = _v4_sequence_observation()
    actions = jnp.zeros((1, 3, 4, 2), dtype=jnp.float32)
    tiny_v4_seq.memory_fact_oracle_writes = True
    tiny_v4_seq.memory_v4_visual_injection = False
    tiny_v4_seq.memory_mask_zero_tokens = True
    try:
        losses = tiny_v4_seq._compute_sequence_loss_v32(jax.random.key(48), observation, actions, train=False)
    finally:
        tiny_v4_seq.memory_fact_oracle_writes = False
        tiny_v4_seq.memory_v4_visual_injection = True
        tiny_v4_seq.memory_mask_zero_tokens = False
    np.testing.assert_array_equal(losses["v4_sem_commit_count"], 2.0)
    np.testing.assert_array_equal(losses["v4_fact_read_count"], 2.0)
    for key, value in losses.items():
        assert np.isfinite(np.asarray(value)).all(), key
