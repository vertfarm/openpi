"""Serve a Cosmos3 DROID policy through the deployed KETI joint-velocity contract.

Counterpart of ``serve_pi05_droid_jointpos_velocity.py``. The difference is what sits behind it:
pi05 loads an openpi checkpoint into this process, whereas Cosmos3 runs in its own server and
this one is a proxy in front of it.

    [3090 client]  --cosmos request-->  [this proxy]  --cosmos request-->  [cosmos3 server]
                   <--velocity chunk--                <--absolute chunk--

The proxy holds no model and needs no GPU, so it can share a host with the Cosmos server.

Request (from the 3090; see examples/droid/main_cosmos.py)
    observation/image                  uint8 [540, 640, 3]   already composed, or
    observation/exterior_image_1_left  uint8 [H, W, 3]       left exterior  }  composed here
    observation/exterior_image_2_left  uint8 [H, W, 3]       right exterior }  if image is absent
    observation/wrist_image_left       uint8 [H, W, 3]       wrist          }
    observation/joint_position         float [7]
    observation/gripper_position       float [1]
    prompt                             str   -- the bare instruction; the Cosmos server appends
                                              its own description of the camera layout

Response
    actions  float32 [horizon, 8]  -- 7 normalized joint velocities + 1 gripper position

Usage
    python scripts/serve_cosmos_droid_velocity.py --upstream-port 8010 --port 8000
    # or via the launcher, which also starts the Cosmos server: scripts/start_cosmos_droid_velocity.sh 3
"""

from __future__ import annotations

import dataclasses
import logging
import os
import socket
import threading
import time

import numpy as np
import tyro
import websockets.sync.client
from openpi_client import msgpack_numpy
from openpi_client import websocket_client_policy

from openpi.policies import cosmos_droid_policy
from openpi.serving import websocket_policy_server

logger = logging.getLogger(__name__)


class _UpstreamClient(websocket_client_policy.WebsocketClientPolicy):
    """openpi's client with the websocket keepalive disabled on the upstream link.

    The Cosmos server runs inference synchronously inside its asyncio handler, so it cannot
    answer keepalive pings while a request is in flight. openpi's client keeps the websockets
    default (ping every 20 s, drop the link after 20 s without a pong), which turns any inference
    slower than roughly 40 s -- the first request on an A6000 -- into a spurious
    ``ConnectionClosedError: keepalive ping timeout``. Liveness is bounded elsewhere: the
    upstream TCP probe in ``_connect`` and the caller's own timeout.
    """

    def _wait_for_server(self):
        logging.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                    ping_interval=None,
                    ping_timeout=None,
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logging.info("Still waiting for server...")
                time.sleep(5)


@dataclasses.dataclass
class Args:
    # Where the Cosmos3 policy server is listening.
    upstream_host: str = "127.0.0.1"
    upstream_port: int = 8010
    # Where this proxy listens for the robot.
    host: str = "0.0.0.0"
    port: int = 8001
    # Must match the Cosmos checkpoint's chunk length. See cosmos_droid_policy for why this is
    # 32 rather than pi05's 15.
    action_horizon: int = cosmos_droid_policy.COSMOS_ACTION_HORIZON
    joint_delta_scale: float = 0.2
    max_abs_velocity: float = 0.5
    # Seconds to wait for the upstream server. Not unbounded: openpi's client waits forever by
    # default, which turns a dead upstream into a hang with no diagnosis.
    upstream_timeout_s: float = 900.0
    warmup_runs: int = 2


def _validate_output(actions: np.ndarray, expected_horizon: int, max_abs_velocity: float) -> None:
    if actions.shape != (expected_horizon, 8):
        raise RuntimeError(f"Unexpected action shape {actions.shape}; expected ({expected_horizon}, 8)")
    if not np.all(np.isfinite(actions)):
        raise RuntimeError("Actions contain NaN or Inf")
    if np.max(np.abs(actions[:, :7])) > max_abs_velocity + 1e-6:
        raise RuntimeError("Joint velocity exceeded the server safety limit")


class CosmosDroidVelocityPolicy:
    """Translates between the robot's DROID contract and the Cosmos server's own."""

    def __init__(self, args: Args) -> None:
        self._args = args
        self._chain = cosmos_droid_policy.cosmos_velocity_output_chain(
            expected_horizon=args.action_horizon,
            joint_delta_scale=args.joint_delta_scale,
            max_abs_velocity=args.max_abs_velocity,
        )
        # openpi's websocket client is not safe for concurrent use, and the Cosmos server
        # serializes on its model lock anyway, so a second caller would buy nothing.
        self._lock = threading.Lock()
        self._client = self._connect()
        self._requests = 0

    def _connect(self):
        host, port = self._args.upstream_host, self._args.upstream_port
        if self._args.upstream_timeout_s > 0:
            deadline = time.monotonic() + self._args.upstream_timeout_s
            while True:
                try:
                    with socket.create_connection((host, port), timeout=5):
                        break
                except OSError as exc:
                    if time.monotonic() >= deadline:
                        raise RuntimeError(
                            f"Cosmos server at {host}:{port} did not accept a connection within "
                            f"{self._args.upstream_timeout_s:.0f}s; last error: {exc}"
                        ) from exc
                    time.sleep(2.0)
        logger.info("Upstream connected: %s:%d", host, port)
        return _UpstreamClient(host, port)

    def _build_request(self, obs: dict) -> dict:
        if "observation/image" in obs:
            image = np.asarray(obs["observation/image"])
        else:
            try:
                image = cosmos_droid_policy.make_cosmos_observation_image(
                    obs["observation/exterior_image_1_left"],
                    obs["observation/exterior_image_2_left"],
                    obs["observation/wrist_image_left"],
                )
            except KeyError as exc:
                raise ValueError(
                    f"Observation is missing {exc.args[0]!r}. Send either a composed "
                    "'observation/image' or all three camera views. Note that the pi05 request "
                    "carries only ONE exterior view; Cosmos needs both, and feeding it the pi05 "
                    "request shape scored 0/24 in simulation."
                ) from exc

        return {
            "observation/image": image,
            "observation/joint_position": np.asarray(obs["observation/joint_position"]),
            "observation/gripper_position": np.asarray(obs["observation/gripper_position"]),
            "prompt": obs.get("prompt", ""),
        }

    def infer(self, obs: dict) -> dict:
        request = self._build_request(obs)
        with self._lock:
            response = self._client.infer(request)

        state = cosmos_droid_policy.make_state(
            obs["observation/joint_position"], obs["observation/gripper_position"]
        )
        result = self._chain({"state": state, **response})
        actions = np.asarray(result["actions"], dtype=np.float32)

        self._requests += 1
        if self._requests == 1:
            logger.info(
                "First request translated: image=%s %s -> cosmos chunk %s -> velocity %s %s",
                request["observation/image"].shape,
                request["observation/image"].dtype,
                np.asarray(response[cosmos_droid_policy.COSMOS_ACTION_KEY]).shape,
                actions.shape,
                actions.dtype,
            )
        return {"actions": actions}


def _warmup(policy: CosmosDroidVelocityPolicy, args: Args) -> None:
    """Exercise the whole path before the port opens, the way the pi05 server does.

    A synthetic frame is enough: what is being checked is that the upstream answers, that the
    chunk has the horizon this proxy was configured for, and that the converted commands are
    finite and inside the safety limit.
    """
    rng = np.random.default_rng(0)
    observation = {
        "observation/exterior_image_1_left": rng.integers(1, 256, (720, 1280, 3), dtype=np.uint8),
        "observation/exterior_image_2_left": rng.integers(1, 256, (720, 1280, 3), dtype=np.uint8),
        "observation/wrist_image_left": rng.integers(1, 256, (720, 1280, 3), dtype=np.uint8),
        "observation/joint_position": np.zeros(7, dtype=np.float32),
        "observation/gripper_position": np.zeros(1, dtype=np.float32),
        "prompt": "warmup",
    }
    for index in range(args.warmup_runs):
        start = time.monotonic()
        actions = np.asarray(policy.infer(observation)["actions"])
        _validate_output(actions, args.action_horizon, args.max_abs_velocity)
        saturation = float(np.mean(np.abs(actions[:, :7]) >= args.max_abs_velocity - 1e-6))
        logger.info(
            "Warmup %d/%d PASS shape=%s saturation_fraction=%.6f latency_ms=%.1f",
            index + 1,
            args.warmup_runs,
            actions.shape,
            saturation,
            (time.monotonic() - start) * 1000,
        )


def main(args: Args) -> None:
    if args.warmup_runs < 1:
        raise ValueError("warmup_runs must be at least 1")

    policy = CosmosDroidVelocityPolicy(args)
    _warmup(policy, args)

    metadata = {
        "model": "cosmos3_droid",
        "source_action_space": "joint_position",
        "output_action_space": "joint_velocity",
        "action_horizon": args.action_horizon,
        "action_dim": 8,
        "joint_delta_scale": args.joint_delta_scale,
        "max_abs_joint_velocity": args.max_abs_velocity,
        "upstream": f"{args.upstream_host}:{args.upstream_port}",
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", "unset"),
        "warmup_runs": args.warmup_runs,
    }
    logger.info("SERVER_READY host=%s port=%d metadata=%s", socket.gethostname(), args.port, metadata)
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host=args.host, port=args.port, metadata=metadata
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
