"""Serve a pi05_droid_jointpos checkpoint (JAX params/ or PyTorch model.safetensors) through the KETI DROID velocity contract.

The default checkpoint is the team1 RL fine-tune ``hard5_ours_t2cfg_s20_pytorch``. Any directory that contains
``model.safetensors`` (plus ``assets/droid/norm_stats.json``) is loaded through the upstream PyTorch branch of
``policy_config.create_trained_policy`` -> ``Pi0Config.load_pytorch`` -> ``safetensors.torch.load_model``; a
directory that contains ``params/`` is loaded through the JAX branch. This mirrors the simulation evaluation
setup documented in ``openpi_bhl/docs/260914_checkpoint_loading.md``: the openpi loader itself is not modified,
only the checkpoint directory changes.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import pathlib
import random
import socket
import time

import numpy as np
import tyro

from openpi.policies import droid_policy
from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
from openpi.shared import download
from openpi.training import config as _config

TEAM1_CKPT_ROOT = pathlib.Path("/data/keti/snu/workspace/ckpt/team1")
DEFAULT_CHECKPOINT_DIR = str(TEAM1_CKPT_ROOT / "hard5_ours_t2cfg_s20_pytorch")
# Checkpoints exported for the joint-position DROID model declare this openpi config in config.json.
# The serving config below (pi05_droid_jointpos_velocity) uses the same model and input pipeline and only
# appends the joint-position -> joint-velocity output conversion, so both names are accepted.
COMPATIBLE_CHECKPOINT_CONFIGS = {"pi05_droid_jointpos", "pi05_droid_jointpos_velocity"}


@dataclasses.dataclass
class Args:
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR
    config: str = "pi05_droid_jointpos_velocity"
    host: str = "0.0.0.0"
    port: int = 8001
    # torch.compile finishes during the first two calls (~22s + ~7s on A6000); the third call must be steady-state.
    warmup_runs: int = 3
    # Seed for random/numpy/torch before loading. pi0.5 draws the flow-matching noise from the global RNG,
    # so fixing it makes server restarts reproducible (same approach as the simulation serve_policy_seeded.py).
    seed: int = 0
    # Load the checkpoint, run the warmup inferences, print the metadata and exit without opening the port.
    check_only: bool = False


def _validate_output(actions: np.ndarray, expected_horizon: int) -> None:
    if actions.shape != (expected_horizon, 8):
        raise RuntimeError(f"Unexpected warmup action shape {actions.shape}; expected ({expected_horizon}, 8)")
    if not np.all(np.isfinite(actions)):
        raise RuntimeError("Warmup actions contain NaN or Inf")
    if np.max(np.abs(actions[:, :7])) > 0.5 + 1e-6:
        raise RuntimeError("Warmup joint velocity exceeded the server safety limit")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _inspect_checkpoint(checkpoint_dir: pathlib.Path, train_config: _config.TrainConfig) -> dict:
    """Determine the weight format and cross-check the exported config.json against the serving config."""
    weight_path = checkpoint_dir / "model.safetensors"
    params_dir = checkpoint_dir / "params"
    if weight_path.is_file():
        weight_format = "pytorch"
    elif params_dir.is_dir():
        weight_format = "jax"
    else:
        raise FileNotFoundError(
            f"Checkpoint {checkpoint_dir} has neither PyTorch weights {weight_path} nor JAX params {params_dir}"
        )

    norm_stats_path = checkpoint_dir / "assets" / "droid" / "norm_stats.json"
    if not norm_stats_path.is_file():
        raise FileNotFoundError(f"Missing DROID norm stats: {norm_stats_path}")

    info: dict = {
        "checkpoint": str(checkpoint_dir),
        "checkpoint_name": checkpoint_dir.name,
        "weight_format": weight_format,
    }
    if weight_format == "pytorch":
        stat = weight_path.stat()
        info["weight_bytes"] = stat.st_size
        info["weight_mtime"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(stat.st_mtime))

    config_path = checkpoint_dir / "config.json"
    if config_path.is_file():
        exported = json.loads(config_path.read_text(encoding="utf-8"))
        info["checkpoint_config"] = exported
        exported_config = exported.get("openpi_config")
        if exported_config is not None and exported_config not in COMPATIBLE_CHECKPOINT_CONFIGS:
            raise RuntimeError(
                f"Checkpoint config.json declares openpi_config={exported_config!r}, which is not compatible "
                f"with serving config {train_config.name!r} (expected one of {sorted(COMPATIBLE_CHECKPOINT_CONFIGS)}). "
                "A joint-velocity checkpoint must not be served through the joint-position velocity converter."
            )
        for key, expected in (
            ("action_horizon", train_config.model.action_horizon),
            ("action_dim", train_config.model.action_dim),
        ):
            if key in exported and exported[key] != expected:
                raise RuntimeError(
                    f"Checkpoint config.json {key}={exported[key]} does not match serving config {key}={expected}"
                )
    elif weight_format == "pytorch":
        logging.warning("PyTorch checkpoint %s has no config.json; skipping config cross-check", checkpoint_dir)
    return info


def main(args: Args) -> None:
    if args.warmup_runs < 1:
        raise ValueError("warmup_runs must be at least 1")

    train_config = _config.get_config(args.config)
    if train_config.model.action_horizon != 15:
        raise RuntimeError(f"Server requires action horizon 15, got {train_config.model.action_horizon}")

    checkpoint_dir = download.maybe_download(args.checkpoint_dir)
    checkpoint_info = _inspect_checkpoint(checkpoint_dir, train_config)
    logging.info("Checkpoint info: %s", json.dumps(checkpoint_info, ensure_ascii=False))

    _seed_everything(args.seed)
    logging.info("Loading config=%s checkpoint=%s format=%s seed=%d", args.config, checkpoint_dir,
                 checkpoint_info["weight_format"], args.seed)
    load_start = time.monotonic()
    policy = policy_config.create_trained_policy(train_config, checkpoint_dir)
    logging.info("Model loaded in %.1fs", time.monotonic() - load_start)

    if checkpoint_info["weight_format"] == "pytorch":
        import torch

        checkpoint_info["torch_version"] = torch.__version__
        checkpoint_info["torch_device"] = str(policy._pytorch_device)  # noqa: SLF001
        if torch.cuda.is_available():
            checkpoint_info["torch_device_name"] = torch.cuda.get_device_name(0)

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
        **checkpoint_info,
        "config": args.config,
        "seed": args.seed,
        "physical_gpu": os.getenv("DROID_PHYSICAL_GPU", "unknown"),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", "unset"),
        "warmup_runs": args.warmup_runs,
    }
    if args.check_only:
        logging.info("CHECK_ONLY_PASS metadata=%s", json.dumps(metadata, ensure_ascii=False, default=str))
        return
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
