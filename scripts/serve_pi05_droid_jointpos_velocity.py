"""Serve the official pi05_droid_jointpos checkpoint through the KETI DROID velocity contract."""

from __future__ import annotations

import dataclasses
import logging
import os
import socket
import time

import numpy as np
import tyro

from openpi.policies import droid_policy
from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


@dataclasses.dataclass
class Args:
    checkpoint_dir: str = "gs://openpi-assets/checkpoints/pi05_droid_jointpos"
    config: str = "pi05_droid_jointpos_velocity"
    host: str = "0.0.0.0"
    port: int = 8000
    warmup_runs: int = 2


def _validate_output(actions: np.ndarray, expected_horizon: int) -> None:
    if actions.shape != (expected_horizon, 8):
        raise RuntimeError(f"Unexpected warmup action shape {actions.shape}; expected ({expected_horizon}, 8)")
    if not np.all(np.isfinite(actions)):
        raise RuntimeError("Warmup actions contain NaN or Inf")
    if np.max(np.abs(actions[:, :7])) > 0.5 + 1e-6:
        raise RuntimeError("Warmup joint velocity exceeded the server safety limit")


def main(args: Args) -> None:
    if args.warmup_runs < 1:
        raise ValueError("warmup_runs must be at least 1")

    train_config = _config.get_config(args.config)
    if train_config.model.action_horizon != 15:
        raise RuntimeError(f"Server requires action horizon 15, got {train_config.model.action_horizon}")

    logging.info("Loading config=%s checkpoint=%s", args.config, args.checkpoint_dir)
    policy = policy_config.create_trained_policy(train_config, args.checkpoint_dir)

    example = droid_policy.make_droid_example()
    for index in range(args.warmup_runs):
        start = time.monotonic()
        result = policy.infer(example)
        actions = np.asarray(result["actions"])
        _validate_output(actions, train_config.model.action_horizon)
        logging.info(
            "Warmup %d/%d PASS shape=%s finite=true latency_ms=%.1f",
            index + 1,
            args.warmup_runs,
            actions.shape,
            (time.monotonic() - start) * 1000,
        )

    metadata = {
        **(policy.metadata or {}),
        "config": args.config,
        "checkpoint": args.checkpoint_dir,
        "physical_gpu": os.getenv("DROID_PHYSICAL_GPU", "unknown"),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", "unset"),
        "warmup_runs": args.warmup_runs,
    }
    hostname = socket.gethostname()
    logging.info("SERVER_READY host=%s port=%d metadata=%s", hostname, args.port, metadata)
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=metadata,
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
