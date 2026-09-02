"""Serve a pi05_yam_mem_* memory checkpoint over websocket, threading the Titans memory.

Like scripts/serve_yam_subtask.py, but the policy runs `Pi0.sample_with_memory`: every request
reads the memory, decodes the subtask, denoises the actions and then writes the frame's hidden
representation into the per-episode memory state, which is threaded across requests. v3 writes
the raw layer-8 top-camera states; v3.1 writes the memory-token block's final-normalized output.
Each response
carries "subtask", "surprise" (1-ish = novel, ~0 = recalled), the write gates and the running
write count. A request containing "reset_memory": true re-initializes the memory (send one at
every episode start); a bare {"reset_memory": true} request (no images) just resets and returns.
For RTC-trained checkpoints, an optional "action_prefix" carries the still-executing portion
of the previous chunk plus the anticipated inference delay.

The client controls the write cadence: one infer call = one memory write, so call at the
training stride (memory_stride_frames=10 @ 30 Hz -> ~0.33 s between calls).

v3.5 / v4 checkpoints (memory_v35_enabled, memory_v4_dual_bank): every request is one valid
memory-clock transition (call at memory_stride_frames=15 -> 0.5 s @ 30 Hz). The banks may
COMMIT on a request according to --write-policy: "always" (default) lets every request commit
and leaves the decision to the fact head's confidence gate (a robot has no episode manifest to
mark evidence frames; the head was trained to abstain when the fact is not visible); "client"
commits only when the request carries "memory_write": true (decay-only otherwise). v4 threads
the semantic bank next to the visual one and reports the fact head's per-slot prediction and
confidence, the slots committed on this request, the running commit count and the read head's
per-slot decode of the bank. "reset_memory" resets both banks.

Usage (on the GPU box):
    uv run scripts/serve_yam_memory.py \
        --dir checkpoints/pi05_yam_mem_v31/<exp>/<step> \
        --config pi05_yam_mem_v31
    # v4 (Stage 4c):
    uv run scripts/serve_yam_memory.py \
        --dir v4/checkpoints/pi05_yam_mem_v4_stage4c/v4_stage4c_20260902_r1/999 \
        --config pi05_yam_mem_v4_stage4c
"""

from collections.abc import Mapping
import dataclasses
import json
import logging
import socket
import threading
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.rtc as _rtc
import openpi.models.tokenizer as _tokenizer
import openpi.policies.policy as _policy
from openpi.serving import websocket_policy_server
import openpi.shared.download as download
import openpi.shared.nnx_utils as nnx_utils
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.transforms as transforms


@dataclasses.dataclass
class Args:
    # Checkpoint directory, e.g. checkpoints/pi05_yam_mem_v3/<exp>/<step> or
    # checkpoints/pi05_yam_mem_v31/<exp>/<step>.
    dir: str
    # Keep v3 as the default until an explicitly selected v3.1 checkpoint exists.
    config: str = "pi05_yam_mem_v3"
    port: int = 8000
    max_decode_steps: int = 10
    # v3.5 / v4 only: when a request may COMMIT to the banks ("always" | "client"), see module doc.
    write_policy: str = "always"
    # Run one synthetic request before serving so the JIT compile (minutes) happens here, not on
    # the robot's first request; the memory is reset afterwards.
    warmup: bool = True


WRITE_POLICIES = ("always", "client")


def _build_server_metadata(
    train_config: Any,
    data_config: Any,
    *,
    simulated_delay: int | None,
    write_policy: str = "always",
    fact_names: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Publish the training semantics needed to reject mismatched clients/checkpoints."""
    metadata: dict[str, Any] = dict(train_config.policy_metadata or {})
    memory_architecture = str(getattr(train_config.model, "memory_architecture", "v3_v31"))
    v35 = bool(getattr(train_config.model, "memory_v35_enabled", False))
    v4 = bool(getattr(train_config.model, "memory_v4_dual_bank", False))
    metadata.update(
        {
            "config_name": train_config.name,
            "memory_architecture": memory_architecture,
            "memory_write_source": str(getattr(train_config.model, "memory_write_source", "raw_hidden")),
            "memory_query_tokens": (
                int(getattr(train_config.model, "memory_query_tokens", 0))
                if memory_architecture == "v32_layer8_dual_query"
                else None
            ),
            "action_horizon": int(train_config.model.action_horizon),
            "rtc_enabled": simulated_delay is not None,
            "rtc_max_delay": simulated_delay,
            "rtc_delay_semantics": "inclusive_max",
            "memory_stride_frames": int(data_config.memory_stride_frames),
            "memory_v35_enabled": v35,
            "memory_v4_dual_bank": v4,
            "memory_fact_slots": int(getattr(train_config.model, "memory_fact_slots", 0)) if v4 else None,
            "memory_fact_targets": int(getattr(train_config.model, "memory_fact_targets", 0)) if v4 else None,
            "memory_fact_write_conf": float(getattr(train_config.model, "memory_fact_write_conf", 0.0)) if v4 else None,
            # Human names for the client overlay: slot index -> entity, target index -> name.
            "fact_slot_names": list((fact_names or {}).get("slots", [])) if v4 else None,
            "fact_target_names": list((fact_names or {}).get("targets", [])) if v4 else None,
            "write_policy": write_policy if v35 else "every_request",
        }
    )
    return metadata


def _load_fact_names(data_config: Any) -> dict[str, Any] | None:
    """Slot entities and target vocabulary from the v4 fact-label sidecar, when configured."""
    path = getattr(data_config, "memory_v4_fact_labels_path", None)
    if not path:
        return None
    try:
        with open(path) as f:
            sidecar = json.load(f)
    except OSError as error:
        logging.warning("fact-label sidecar %s unreadable (%s); client overlay will use indices", path, error)
        return None
    slots = sorted(sidecar.get("fact_slots", []), key=lambda s: int(s["slot"]))
    return {
        "slots": [str(s["entity"]) for s in slots],
        "targets": [str(t) for t in sidecar.get("target_vocab", [])],
    }


def _resolve_write_mask(inputs: dict, write_policy: str) -> bool:
    """Whether this request may commit (v3.5/v4). Pops the client's "memory_write" flag."""
    requested = inputs.pop("memory_write", None)
    if write_policy == "always":
        return True
    if write_policy == "client":
        return bool(requested) if requested is not None else False
    raise ValueError(f"unsupported write_policy {write_policy!r}; expected one of {WRITE_POLICIES}")


class MemoryPolicy(_policy.Policy):
    """Policy that threads the Titans memory state across requests.

    Responses carry the decoded subtask, the pre-write surprise and the (frozen) write gates.
    """

    def __init__(
        self,
        model,
        *,
        decode_tokenizer,
        stop_token: int,
        max_decode_steps: int,
        action_horizon: int,
        action_dim: int,
        raw_action_dim: int,
        simulated_delay: int | None,
        write_policy: str = "always",
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        self._decode_tokenizer = decode_tokenizer
        self._stop_token = stop_token
        self._max_decode_steps = max_decode_steps
        self._action_horizon = action_horizon
        self._action_dim = action_dim
        self._raw_action_dim = raw_action_dim
        self._simulated_delay = simulated_delay
        if write_policy not in WRITE_POLICIES:
            raise ValueError(f"unsupported write_policy {write_policy!r}; expected one of {WRITE_POLICIES}")
        self._write_policy = write_policy
        self._v35 = bool(getattr(model, "memory_v35_enabled", False))
        self._v4 = bool(getattr(model, "memory_v4_dual_bank", False))
        self._sample = nnx_utils.module_jit(
            model.sample_with_memory, static_argnames=("stop_token", "max_decode_steps", "num_steps", "write_mode")
        )
        self._init_state = lambda: model.memory.init_state(1)
        self._init_semantic_state = (lambda: model.memory_semantic.init_state(1)) if self._v4 else None
        self._lock = threading.Lock()
        self._memory_state = self._init_state()
        self._semantic_state = self._init_semantic_state() if self._v4 else None
        self._writes = 0
        self._sem_commits = 0

    @staticmethod
    def _integer_scalar(value: Any, *, name: str) -> int:
        """Return a protocol integer without silently truncating floats or accepting booleans."""
        array = np.asarray(value)
        if array.shape != () or array.dtype.kind not in "iu":
            raise ValueError(f"action_prefix.{name} must be an integer scalar, got {value!r}")
        return int(array)

    def _prepare_action_prefix(self, inputs: dict, prefix: Any) -> _rtc.ActionPrefix:
        if not isinstance(prefix, Mapping):
            raise ValueError(f"action_prefix must be a mapping, got {type(prefix).__name__}")

        missing = {"actions", "delay", "prefix_length"} - prefix.keys()
        if missing:
            raise ValueError(f"action_prefix is missing required fields: {sorted(missing)}")

        delay = self._integer_scalar(prefix["delay"], name="delay")
        prefix_length = self._integer_scalar(prefix["prefix_length"], name="prefix_length")
        if not 0 <= delay <= prefix_length <= self._action_horizon:
            raise ValueError(
                "action_prefix must satisfy 0 <= delay <= prefix_length <= action_horizon; "
                f"got delay={delay}, prefix_length={prefix_length}, "
                f"action_horizon={self._action_horizon}"
            )
        if self._simulated_delay is None:
            raise ValueError("action_prefix requires a policy configured with simulated_delay")
        if delay > self._simulated_delay:
            raise ValueError(
                f"action_prefix.delay ({delay}) exceeds the configured inclusive RTC maximum ({self._simulated_delay})"
            )

        raw_actions = np.asarray(prefix["actions"])
        expected_raw_shape = (self._action_horizon, self._raw_action_dim)
        if raw_actions.shape != expected_raw_shape:
            raise ValueError(f"action_prefix.actions must have shape {expected_raw_shape}, got {raw_actions.shape}")
        if raw_actions.dtype.kind not in "fiu" or not np.all(np.isfinite(raw_actions)):
            raise ValueError("action_prefix.actions must contain only finite numeric values")

        # Use the exact observation/action input pipeline used for training. In particular,
        # the client sends absolute actions in robot units; DeltaActions, Normalize and
        # PadStatesAndActions must run before the prefix reaches the model.
        rtc_inputs = jax.tree.map(lambda x: x, inputs)
        rtc_inputs["actions"] = np.asarray(raw_actions, dtype=np.float32).copy()
        rtc_inputs = self._input_transform(rtc_inputs)
        transformed_actions = np.asarray(rtc_inputs["actions"])
        expected_model_shape = (self._action_horizon, self._action_dim)
        if transformed_actions.shape != expected_model_shape:
            raise ValueError(
                "transformed action_prefix.actions has an unexpected shape: "
                f"expected {expected_model_shape}, got {transformed_actions.shape}"
            )
        if not np.all(np.isfinite(transformed_actions)):
            raise ValueError("transformed action_prefix.actions contains non-finite values")

        action_prefix = _rtc.ActionPrefix(
            actions=transformed_actions,
            delay=np.asarray(delay, dtype=np.int32),
            prefix_length=np.asarray(prefix_length, dtype=np.int32),
        )
        _rtc.validate_action_prefix(
            action_prefix,
            action_horizon=self._action_horizon,
            action_dim=self._action_dim,
        )
        return jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], action_prefix)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        inputs = jax.tree.map(lambda x: x, obs)  # copy: transforms may modify in place
        if inputs.pop("reset_memory", False):
            with self._lock:
                self._memory_state = self._init_state()
                if self._v4:
                    self._semantic_state = self._init_semantic_state()
                self._writes = 0
                self._sem_commits = 0
            logging.info("memory reset")
            if "observation/image" not in inputs:  # bare reset ping
                return {"reset": True, "writes": 0, "sem_commits": 0}

        write_allowed = _resolve_write_mask(inputs, self._write_policy)
        prefix = inputs.pop("action_prefix", None)
        action_prefix = self._prepare_action_prefix(inputs, prefix) if prefix is not None else None
        inputs = self._input_transform(inputs)
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        observation = _model.Observation.from_dict(inputs)

        clock: dict[str, Any] = {}
        if self._v35:
            # One request = one valid memory-clock transition (call at memory_stride_frames).
            clock["v35_transition_valid"] = jnp.ones((1,), dtype=bool)
            clock["v35_write_mask"] = jnp.asarray([write_allowed], dtype=bool)
        start_time = time.monotonic()
        with self._lock:
            self._rng, sample_rng = jax.random.split(self._rng)
            if self._v4:
                clock["semantic_state"] = self._semantic_state
            actions, new_state, aux = self._sample(
                sample_rng,
                observation,
                self._memory_state,
                stop_token=self._stop_token,
                max_decode_steps=self._max_decode_steps,
                action_prefix=action_prefix,
                **clock,
            )
            jax.block_until_ready(new_state)
            self._memory_state = new_state
            if self._v4:
                self._semantic_state = aux["v4_semantic_state"]
                commit_now = np.asarray(aux["v4_sem_commit_applied"])[0].astype(bool)
                self._sem_commits += int(commit_now.sum())
            if self._v35:
                self._writes += int(np.asarray(aux["write_occurred"])[0])
            else:
                self._writes += 1
            writes = self._writes
            sem_commits = self._sem_commits
        model_time = time.monotonic() - start_time

        tokens = np.asarray(aux["tokens"])[0]
        mask = np.asarray(aux["token_mask"])[0]
        subtask = self._decode_tokenizer.decode(tokens[mask].tolist()).strip()

        outputs = {"state": inputs["state"], "actions": actions}
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        outputs = self._output_transform(outputs)
        outputs["subtask"] = subtask
        outputs["surprise"] = float(np.asarray(aux["surprise"])[0]) if "surprise" in aux else float("nan")
        outputs["writes"] = writes
        outputs["write_allowed"] = bool(write_allowed)
        outputs["gates"] = {
            k: float(np.asarray(aux[k]).mean()) if k in aux else float("nan") for k in ("theta", "eta", "alpha")
        }
        if self._v4:
            # Fact head (write side, this frame) and read head (bank content BEFORE this
            # request's transition), per slot; the client overlays these next to the subtask.
            outputs["fact_predicted"] = np.asarray(aux["v4_fact_predicted"])[0].astype(int).tolist()
            outputs["fact_confidence"] = np.asarray(aux["v4_fact_confidence"])[0].astype(float).tolist()
            outputs["sem_commit_now"] = commit_now.tolist()
            outputs["sem_commits"] = sem_commits
            read_logits = np.asarray(aux["v4_fact_read_logits"])[0]
            outputs["read_predicted"] = np.argmax(read_logits, axis=-1).astype(int).tolist()
            outputs["sem_injected_rms"] = float(np.asarray(aux["v4_sem_injected_pre_cast_rms"])[0])
        outputs["policy_timing"] = {"infer_ms": model_time * 1000}
        return outputs


def create_policy(args: Args) -> MemoryPolicy:
    train_config = _config.get_config(args.config)
    assert train_config.model.predict_with_memory, f"config {args.config} was not built with predict_with_memory"
    checkpoint_dir = download.maybe_download(args.dir)

    logging.info("Loading model (float32: the memory's inner GD is validated in f32)...")
    model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.float32))
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    memory_architecture = str(getattr(train_config.model, "memory_architecture", "v3_v31"))
    memory_write_source = str(getattr(train_config.model, "memory_write_source", "raw_hidden"))
    if getattr(model, "memory_gate", None) is not None:
        gate = np.asarray(model.memory_gate.value)
        gate_note = (
            f"memory_gate norm {np.linalg.norm(gate):.4f} max|g| {np.abs(gate).max():.5f} (0 = memory content unused)"
        )
    else:
        # v3.5 / v4: frozen tanh injection gates, calibrated per bank.
        visual_gate = float(np.tanh(np.asarray(model.memory_inject_w.value, dtype=np.float32)).mean())
        gate_note = f"visual tanh gate {visual_gate:.3f}"
        if getattr(model, "memory_v4_dual_bank", False):
            sem_gate = float(np.tanh(np.asarray(model.memory_sem_inject_w.value, dtype=np.float32)).mean())
            gate_note += f" | semantic tanh gate {sem_gate:.3f} | fact slots {model.memory_fact_slots} targets {model.memory_fact_targets} write conf {model.memory_fact_write_conf}"
    logging.info(
        "config=%s | memory_architecture=%s | memory_write_source=%s | memory_layer=%d | %s | write_policy=%s",
        train_config.name,
        memory_architecture,
        memory_write_source,
        model.memory_layer,
        gate_note,
        args.write_policy,
    )

    out_norm_stats = dict(norm_stats)

    pg = _tokenizer.FASTSubtaskTokenizer(train_config.model.max_token_len)._paligemma_tokenizer  # noqa: SLF001
    stop_token = int(pg.encode("placeholder subtask\n")[-1])

    configured_delay = getattr(train_config.model, "simulated_delay", None)
    simulated_delay = None if configured_delay is None else int(configured_delay)
    metadata = _build_server_metadata(
        train_config,
        data_config,
        simulated_delay=simulated_delay,
        write_policy=args.write_policy,
        fact_names=_load_fact_names(data_config) if getattr(model, "memory_v4_dual_bank", False) else None,
    )
    return MemoryPolicy(
        model,
        decode_tokenizer=pg,
        stop_token=stop_token,
        max_decode_steps=args.max_decode_steps,
        action_horizon=train_config.model.action_horizon,
        action_dim=train_config.model.action_dim,
        raw_action_dim=int(np.asarray(norm_stats["actions"].mean).shape[-1]),
        simulated_delay=simulated_delay,
        write_policy=args.write_policy,
        transforms=[
            # BuildMemorySequence is the dataset-side sequence builder; live single-frame
            # observations pass through it untouched (no "frame_index"), so no filtering needed.
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(out_norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ],
        metadata=metadata,
    )


def _warmup(policy: MemoryPolicy, *, prompt: str, raw_action_dim: int, action_horizon: int) -> None:
    """Compile every request shape the robot client uses (plain and RTC-prefixed), then reset."""
    rng = np.random.default_rng(0)
    example = {
        "observation/state": rng.random(raw_action_dim).astype(np.float32),
        "observation/image": rng.integers(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/left_wrist_image": rng.integers(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/right_wrist_image": rng.integers(256, size=(480, 640, 3), dtype=np.uint8),
        "prompt": prompt,
    }
    started = time.monotonic()
    first = policy.infer(dict(example))
    logging.info(
        "warmup: plain request compiled + ran in %.1f s (subtask %r)", time.monotonic() - started, first["subtask"]
    )
    if policy._simulated_delay is not None:  # noqa: SLF001
        started = time.monotonic()
        policy.infer(
            {
                **example,
                "action_prefix": {
                    "actions": np.asarray(first["actions"], dtype=np.float32)[:action_horizon],
                    "delay": policy._simulated_delay,  # noqa: SLF001
                    "prefix_length": min(action_horizon, policy._simulated_delay + 10),  # noqa: SLF001
                },
            }
        )
        logging.info("warmup: RTC-prefixed request compiled + ran in %.1f s", time.monotonic() - started)
    policy.infer({"reset_memory": True})
    logging.info("warmup done; memory reset")


def main(args: Args) -> None:
    policy = create_policy(args)
    if args.warmup:
        train_config = _config.get_config(args.config)
        prompt = (
            "find the banana"
            if getattr(train_config.model, "memory_v35_enabled", False)
            else "find the bin with banana"
        )
        _warmup(
            policy,
            prompt=prompt,
            raw_action_dim=policy._raw_action_dim,  # noqa: SLF001
            action_horizon=policy._action_horizon,  # noqa: SLF001
        )

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy.metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
