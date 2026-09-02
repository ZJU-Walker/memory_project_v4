"""Closed-loop robot client for the v4 DUAL-BANK memory model on the bimanual YAM.

Same hardware plumbing as examples/yam/client_memory.py (YAM arms over CAN + three RealSense
cameras, direct in-process) and the same RTC chunk broker, against `scripts/serve_yam_memory.py`
serving a v4 checkpoint (config pi05_yam_mem_v4_stage4c). Differences from the v3.1 client:

* memory clock: v4 trained at memory_stride_frames=15, so the broker replans -- and the server
  ticks both banks -- every 15 control steps (0.5 s @ 30 Hz), not 10;
* prompts: v4 trained on "find the banana" / "find the grey pepper box" (choose with --prompt);
* the server decides writes itself (--write-policy always: the fact head's 0.9 confidence gate),
  so the client sends nothing about phases; it just calls at the stride and resets per episode;
* the overlay shows, next to the subtask, what the fact head sees now, what the semantic bank
  holds (read head decode), and the running commit count -- the numbers the offline batteries
  scored (the decision "wait; target bin is left/right" must follow the bank).

Keys in the window:  r = reset the server-side memory (both banks) AND fetch a fresh chunk
                     q = stop (arms left in place).

Server (GPU box, from the v4 worktree):
    cd /iris/u/kewalk/memory_project_v4/openpi && source cluster_v4/env.sh
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u scripts/serve_yam_memory.py \
        --dir v4/checkpoints/pi05_yam_mem_v4_stage4c/v4_stage4c_20260902_r1/999 \
        --config pi05_yam_mem_v4_stage4c --port 8000

Client (robot computer; needs `gello_software`, `openpi_client` and opencv importable):
    python examples/yam/client_memory_v4.py --host <gpu-host> --port 8000 --prompt "find the banana"

Smoke-test the obs/action/memory contract without hardware:
    python examples/yam/client_memory_v4.py --host <gpu-host> --port 8000 --dry-run
"""

import dataclasses
import datetime
import logging
import os
import shutil
import subprocess
import threading

import cv2
import numpy as np
from openpi_client import action_chunk_broker
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tyro

PROMPTS = ("find the banana", "find the grey pepper box")

# Per-arm DOF: 6 arm joints + 1 gripper. Bimanual state/action is concat(left, right) -> 14.
BIMANUAL_DOF = 14
# Fallback names when the server publishes none (v4 fact-label sidecar order).
DEFAULT_SLOT_NAMES = ("banana", "grey_pepper_box")
DEFAULT_TARGET_NAMES = ("left_bin", "right_bin", "unknown")


@dataclasses.dataclass
class Args:
    # --- Policy server (remote GPU box) ---
    host: str = "10.79.12.252"
    port: int = 8000
    ping_timeout: float = 600.0
    """Websocket keepalive timeout (s). Generous: a server started without --warmup compiles on
    the first request for minutes."""

    # --- Inference / control ---
    action_horizon: int = 50
    """Full model action horizon. Must match the server checkpoint."""
    steps_between_inference: int = 15
    """v4 memory clock: replan (= one memory tick on the server) every 15 controls @ 30 Hz."""
    initial_delay_steps: int = 6
    """Conservative initial latency estimate (6 steps = 200 ms at 30 Hz)."""
    max_async_delay_steps: int = 6
    """Never execute more unconfirmed steps than the largest RTC delay seen in training."""
    delay_tolerance_steps: int = 0
    """Block at the predicted delay rather than leaving the RTC-trained prefix range."""
    delay_buffer_size: int = 8
    """Number of measured inference delays retained by the conservative estimator."""
    max_steps: int = 12000
    hz: float = 30.0
    prompt: str = PROMPTS[0]
    """One of the v4 training prompts: "find the banana" | "find the grey pepper box"."""
    max_joint_delta: float = 1.0
    """Per-step safety clamp: cap |target - current| across all joints to this many radians."""

    # --- Display ---
    show: bool = True
    """Show the top camera + predicted subtask + memory readout in an OpenCV window."""

    # --- Hardware (defaults from gello configs/yam_left.yaml) ---
    can_left: str = "can_left"
    can_right: str = "can_right"
    top_camera_serial: str = "409122273280"
    left_camera_serial: str = "409122271088"
    right_camera_serial: str = "409122271086"

    # --- Recording ---
    record: bool = True
    """Record the top camera view (with the overlay); saved on exit (incl. Ctrl+C)."""
    record_dir: str = "eval"
    record_path: str = ""

    # --- Debug ---
    dry_run: bool = False
    """Skip hardware: validate the RTC replan and obs/action/subtask/memory contract."""
    dry_run_steps: int = 40
    """Control steps in a dry run. Keep above 21 to exercise an asynchronous RTC replan."""


class _H264Writer:
    """Encode RGB frames to an H.264 mp4 via the system ffmpeg (same as client_memory.py)."""

    def __init__(self, path: str, width: int, height: int, fps: float):
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not found on PATH -- needed to encode the recording")
        self._proc = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                f"{fps}",
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-pix_fmt",
                "yuv420p",
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                path,
            ],
            stdin=subprocess.PIPE,
        )

    def write(self, frame_rgb: np.ndarray) -> None:
        self._proc.stdin.write(np.ascontiguousarray(frame_rgb).tobytes())

    def release(self) -> None:
        if self._proc.stdin is not None:
            self._proc.stdin.close()
        self._proc.wait()


class _Names:
    """Slot / target names for the overlay, from server metadata with fallbacks."""

    def __init__(self, metadata: dict):
        slots = metadata.get("fact_slot_names") or list(DEFAULT_SLOT_NAMES)
        targets = metadata.get("fact_target_names") or list(DEFAULT_TARGET_NAMES)
        self.slots = [str(s).replace("_", " ") for s in slots]
        self.targets = [str(t).replace("_bin", "") for t in targets]

    def slot(self, i: int) -> str:
        return self.slots[i] if i < len(self.slots) else f"slot{i}"

    def target(self, i: int) -> str:
        return self.targets[i] if 0 <= i < len(self.targets) else "?"


def memory_readout(result: dict, names: _Names) -> tuple[str, str]:
    """Two overlay lines: what the fact head sees now, and what the semantic bank holds."""
    predicted = result.get("fact_predicted") or []
    confidence = result.get("fact_confidence") or []
    read = result.get("read_predicted") or []
    commit_now = result.get("sem_commit_now") or []
    written = result.get("sem_written")  # None on older servers: show every slot
    unknown = len(names.targets) - 1
    seen = []
    for i, (p, c) in enumerate(zip(predicted, confidence, strict=False)):
        if i >= len(names.slots):
            break
        mark = "*" if i < len(commit_now) and commit_now[i] else ""
        seen.append(f"{names.slot(i)}={names.target(int(p)) if int(p) != unknown else '?'}({float(c):.2f}){mark}")
    held = []
    for i, r in enumerate(read):
        if i >= len(names.slots):
            break
        blank = int(r) == unknown or (written is not None and i < len(written) and not written[i])
        held.append(f"{names.slot(i)}={'-' if blank else names.target(int(r))}")
    line_seen = "sees: " + (" ".join(seen) if seen else "n/a") + "  (* = committed now)"
    line_held = f"bank: {' '.join(held) if held else 'n/a'}  commits {result.get('sem_commits', 0)}"
    return line_seen, line_held


def _overlay(frame_rgb: np.ndarray, subtask: str, lines: list[str]) -> np.ndarray:
    """Subtask (dark green) + memory/status lines (yellow) on an RGB frame."""
    img = np.ascontiguousarray(frame_rgb).copy()
    cv2.putText(img, f"subtask: {subtask}", (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 100, 0), 2, cv2.LINE_AA)
    for k, line in enumerate(lines):
        cv2.putText(img, line, (12, 66 + 26 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 235, 60), 1, cv2.LINE_AA)
    return img


class _Display:
    """Window thread showing the latest overlaid frame; captures 'r' (reset) and 'q' (quit)."""

    def __init__(self, window: str = "pi05 yam memory v4 - closed loop"):
        self._window = window
        self._lock = threading.Lock()
        self._img: np.ndarray | None = None  # RGB, already overlaid
        self.reset_requested = threading.Event()
        self.quit_requested = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def update(self, img_rgb: np.ndarray) -> None:
        with self._lock:
            self._img = img_rgb

    def _loop(self) -> None:
        cv2.namedWindow(self._window, cv2.WINDOW_NORMAL)
        while not self._stop.is_set():
            with self._lock:
                img = None if self._img is None else self._img.copy()
            if img is not None:
                cv2.imshow(self._window, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(30) & 0xFF
            if key == ord("r"):
                self.reset_requested.set()
            elif key == ord("q"):
                self.quit_requested.set()
        cv2.destroyAllWindows()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)


def _obs_to_request(obs: dict, prompt: str) -> dict:
    """Map a `RobotEnv.get_obs()` dict to the websocket observation the server expects."""
    return {
        "observation/state": np.asarray(obs["joint_positions"], dtype=np.float32),
        "observation/image": image_tools.convert_to_uint8(obs["top_camera_rgb"]),
        "observation/left_wrist_image": image_tools.convert_to_uint8(obs["left_camera_rgb"]),
        "observation/right_wrist_image": image_tools.convert_to_uint8(obs["right_camera_rgb"]),
        "prompt": prompt,
    }


def _clamp_joint_delta(target: np.ndarray, current: np.ndarray, max_delta: float) -> np.ndarray:
    """Scale the whole command so no single joint moves more than `max_delta` this step."""
    delta = target - current
    m = float(np.abs(delta).max())
    if m > max_delta:
        delta = delta / m * max_delta
    return current + delta


def validate_v4_metadata(metadata: dict, args: Args) -> None:
    """Fail before touching hardware when the client/server contracts differ."""
    if not metadata.get("memory_v4_dual_bank"):
        raise ValueError(
            f"this client needs a v4 dual-bank server; server advertised config {metadata.get('config_name')!r} "
            f"with memory_v4_dual_bank={metadata.get('memory_v4_dual_bank')!r}"
        )
    if metadata.get("memory_architecture") != "v32_layer8_dual_query":
        raise ValueError(f"unexpected memory_architecture {metadata.get('memory_architecture')!r}")
    if metadata.get("write_policy") != "always":
        raise ValueError(
            f"server write_policy is {metadata.get('write_policy')!r}; this client sends no per-request write "
            "flags, so start the server with --write-policy always"
        )
    horizon = metadata.get("action_horizon")
    if horizon != args.action_horizon:
        raise ValueError(
            f"server action_horizon is {horizon!r}, but the client is configured for {args.action_horizon}"
        )
    if metadata.get("rtc_enabled") is not True:
        raise ValueError("the server checkpoint is not RTC-trained (rtc_enabled must be true)")
    if metadata.get("rtc_delay_semantics") != "inclusive_max":
        raise ValueError(f"unexpected RTC delay semantics {metadata.get('rtc_delay_semantics')!r}")
    trained_max_delay = metadata.get("rtc_max_delay")
    if not isinstance(trained_max_delay, int) or args.max_async_delay_steps > trained_max_delay:
        raise ValueError(
            f"client max_async_delay_steps={args.max_async_delay_steps} exceeds or cannot verify the "
            f"server's trained RTC maximum ({trained_max_delay!r})"
        )
    training_stride = metadata.get("memory_stride_frames")
    if training_stride != args.steps_between_inference:
        raise ValueError(
            f"server memory_stride_frames is {training_stride!r}, but the client replans every "
            f"{args.steps_between_inference} steps (v4 trained at 15)"
        )
    if args.prompt not in PROMPTS:
        raise ValueError(f"prompt {args.prompt!r} is not a v4 training prompt; use one of {PROMPTS}")


def _run_dry(ws_client, policy, args: Args, names: _Names) -> None:
    """Validate the RTC replan and obs/action/subtask/memory contract without hardware."""
    if args.dry_run_steps <= args.steps_between_inference + args.max_async_delay_steps:
        raise ValueError(
            "dry_run_steps must exceed steps_between_inference + max_async_delay_steps "
            "so the test waits for at least one asynchronous RTC replan"
        )
    rng = np.random.default_rng(0)
    logging.info("Dry run: memory reset + %d random control observations...", args.dry_run_steps)
    logging.info("  reset: %s", ws_client.infer({"reset_memory": True}))
    for i in range(args.dry_run_steps):
        example = {
            "observation/state": rng.random(BIMANUAL_DOF).astype(np.float32),
            "observation/image": rng.integers(256, size=(480, 640, 3), dtype=np.uint8),
            "observation/left_wrist_image": rng.integers(256, size=(480, 640, 3), dtype=np.uint8),
            "observation/right_wrist_image": rng.integers(256, size=(480, 640, 3), dtype=np.uint8),
            "prompt": args.prompt,
        }
        result = policy.infer(example)
        action = np.asarray(result["actions"])
        assert action.shape == (BIMANUAL_DOF,), f"expected (14,) per broker step, got {action.shape}"
        assert np.all(np.isfinite(action)), "non-finite action returned"
        assert isinstance(result.get("subtask"), str), f"missing subtask, got {result.get('subtask')!r}"
        for key in ("fact_predicted", "fact_confidence", "read_predicted", "sem_commit_now", "sem_commits"):
            assert key in result, f"v4 server response lacks {key!r}: {sorted(result)}"
        seen, held = memory_readout(result, names)
        logging.info("  step %d: subtask=%r | %s | %s", i, result["subtask"], seen, held)
    logging.info("Dry run OK -- RTC replan and obs/action/subtask/memory contract match.")


def main(args: Args) -> None:
    # --- Connect to the policy server ---
    try:
        ws_client = _websocket_client_policy.WebsocketClientPolicy(
            host=args.host, port=args.port, ping_timeout=args.ping_timeout
        )
    except TypeError:
        # Older openpi_client on the robot computer (no ping_timeout): still works when the
        # server was started with --warmup (default), which keeps every request well under
        # the library's 20 s keepalive timeout.
        logging.warning("openpi_client without ping_timeout support; make sure the server was warmed up")
        ws_client = _websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    metadata = ws_client.get_server_metadata()
    logging.info("Server metadata: %s", metadata)
    validate_v4_metadata(metadata, args)
    names = _Names(metadata)
    policy = action_chunk_broker.RealtimeActionChunkBroker(
        ws_client,
        action_horizon=args.action_horizon,
        steps_between_inference=args.steps_between_inference,
        initial_delay_steps=args.initial_delay_steps,
        delay_tolerance_steps=args.delay_tolerance_steps,
        max_async_delay_steps=args.max_async_delay_steps,
        delay_buffer_size=args.delay_buffer_size,
    )

    if args.dry_run:
        try:
            _run_dry(ws_client, policy, args, names)
        finally:
            policy.close()
        return

    # --- Build hardware (direct, in-process; same as client_memory.py) ---
    from gello.cameras.realsense_camera import RealSenseCamera
    from gello.env import RobotEnv
    from gello.robots.robot import BimanualRobot
    from gello.robots.yam import YAMRobot

    left = YAMRobot(channel=args.can_left)
    right = YAMRobot(channel=args.can_right)
    robot = BimanualRobot(left, right)
    assert robot.num_dofs() == BIMANUAL_DOF, f"expected 14 DOF, got {robot.num_dofs()}"

    camera_dict = {
        "top_camera": RealSenseCamera(device_id=args.top_camera_serial),
        "left_camera": RealSenseCamera(device_id=args.left_camera_serial),
        "right_camera": RealSenseCamera(device_id=args.right_camera_serial),
    }
    env = RobotEnv(robot, control_rate_hz=args.hz, camera_dict=camera_dict)

    obs = env.get_obs()
    for key in ("top_camera_rgb", "left_camera_rgb", "right_camera_rgb"):
        assert key in obs, f"missing camera obs '{key}'"
    assert np.asarray(obs["joint_positions"]).shape == (BIMANUAL_DOF,)

    # --- Ramp to the first inferred target (avoid a large jump from rest) ---
    # The server memory survives websocket reconnects, so clear any previous client/episode
    # before even computing the ramp target. We reset once more after the ramp because that
    # first inference itself is a memory tick (and may commit if a fact is visible).
    ws_client.infer({"reset_memory": True})
    logging.info("Memory reset before computing the ramp target.")
    logging.info("Ramping to first policy target...")
    first_target = np.asarray(policy.infer(_obs_to_request(obs, args.prompt))["actions"], dtype=np.float64)
    for _ in range(25):
        obs = env.get_obs()
        cur = np.asarray(obs["joint_positions"], dtype=np.float64)
        if float(np.abs(first_target - cur).max()) < 1e-2:
            break
        env.step(_clamp_joint_delta(first_target, cur, args.max_joint_delta))
    # Fresh chunk AND fresh banks for the episode (the ramp query already ticked once).
    policy.reset()
    ws_client.infer({"reset_memory": True})
    logging.info("Memory reset -- episode starts fresh (both banks blank).")

    display = _Display() if args.show else None

    # --- Top camera recording (written incrementally so Ctrl+C still saves it) ---
    writer = None
    record_path = ""
    frames_written = 0
    if args.record:
        record_path = args.record_path
        if not record_path:
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # noqa: DTZ005
            os.makedirs(args.record_dir, exist_ok=True)
            record_path = os.path.join(args.record_dir, f"memory_v4_closedloop_{stamp}.mp4")
        else:
            os.makedirs(os.path.dirname(os.path.abspath(record_path)) or ".", exist_ok=True)
        first_frame = image_tools.convert_to_uint8(obs["top_camera_rgb"])
        writer = _H264Writer(record_path, first_frame.shape[1], first_frame.shape[0], args.hz)
        logging.info("Recording top camera (H.264) to %s @ %.1f Hz", record_path, args.hz)

    # --- Control loop (paced to `hz` by RobotEnv.step's internal Rate) ---
    logging.info(
        "Starting RTC control loop: %d steps @ %.1f Hz (replan/memory tick every %d steps, horizon %d), prompt %r",
        args.max_steps,
        args.hz,
        args.steps_between_inference,
        args.action_horizon,
        args.prompt,
    )
    obs = env.get_obs()
    subtask, seen, held = "", "", ""
    try:
        for step in range(args.max_steps):
            if display is not None and display.quit_requested.is_set():
                logging.info("Quit requested from the display window.")
                break
            if display is not None and display.reset_requested.is_set():
                display.reset_requested.clear()
                policy.reset()  # drop the cached chunk: it was computed with the old memory
                ws_client.infer({"reset_memory": True})
                logging.info("Memory reset (keyboard) -- new episode.")

            result = policy.infer(_obs_to_request(obs, args.prompt))
            action = np.asarray(result["actions"], dtype=np.float64)
            subtask = str(result.get("subtask", subtask))
            if "fact_predicted" in result:
                seen, held = memory_readout(result, names)
            status = f"step {step} | ticks {result.get('writes', 0)} | r=reset q=quit"

            frame = image_tools.convert_to_uint8(obs["top_camera_rgb"])
            img = _overlay(frame, subtask, [seen, held, status])
            if display is not None:
                display.update(img)
            if writer is not None:
                writer.write(img)
                frames_written += 1

            cur = np.asarray(obs["joint_positions"], dtype=np.float64)
            action = _clamp_joint_delta(action, cur, args.max_joint_delta)
            obs = env.step(action)
            if step % args.steps_between_inference == 0:
                logging.info("  step %d | subtask: %s | %s | %s", step, subtask, seen, held)
    except KeyboardInterrupt:
        logging.info("Interrupted by user -- stopping (arms left in place).")
    finally:
        if writer is not None:
            writer.release()
            logging.info("Saved recording: %s (%d frames)", record_path, frames_written)
        if display is not None:
            display.close()
        policy.close()
        logging.info("Control loop finished.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
