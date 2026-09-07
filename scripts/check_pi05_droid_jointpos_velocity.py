"""Offline (no RobotEnv) websocket inference check for the DROID velocity adapter."""

from __future__ import annotations

import dataclasses
import pathlib
import time

import numpy as np
from openpi_client import websocket_client_policy
import tyro

from openpi.policies import droid_policy


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 8001
    observation: pathlib.Path | None = None
    save_response: pathlib.Path | None = None
    expected_horizon: int = 15


def _load_observation(path: pathlib.Path | None) -> dict:
    if path is None:
        return droid_policy.make_droid_example()
    loaded = np.load(path, allow_pickle=True)
    if isinstance(loaded, np.ndarray) and loaded.shape == ():
        observation = loaded.item()
    elif isinstance(loaded, np.lib.npyio.NpzFile):
        observation = {key: loaded[key] for key in loaded.files}
    else:
        raise ValueError("Observation must be a scalar object .npy or a flat .npz archive")
    if not isinstance(observation, dict):
        raise ValueError("Loaded observation is not a dictionary")
    return observation


def main(args: Args) -> None:
    observation = _load_observation(args.observation)
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    metadata = client.get_server_metadata()
    start = time.monotonic()
    response = client.infer(observation)
    latency_ms = (time.monotonic() - start) * 1000
    actions = np.asarray(response["actions"])

    if actions.shape != (args.expected_horizon, 8):
        raise RuntimeError(f"Unexpected action shape {actions.shape}; expected ({args.expected_horizon}, 8)")
    if not np.all(np.isfinite(actions)):
        raise RuntimeError("Actions contain NaN or Inf")

    saturation_fraction = float(np.mean(np.abs(actions[:, :7]) >= 0.5 - 1e-6))
    if args.save_response is not None:
        args.save_response.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.save_response, actions=actions)

    print(f"server_metadata={metadata}")
    print(f"action_shape={actions.shape}")
    print("actions_finite=true")
    print(f"joint_min={actions[:, :7].min():.6f}")
    print(f"joint_max={actions[:, :7].max():.6f}")
    print(f"joint_limit_fraction={saturation_fraction:.6f}")
    print(f"round_trip_latency_ms={latency_ms:.1f}")
    print("OFFLINE_INFERENCE_STATUS=PASS")


if __name__ == "__main__":
    main(tyro.cli(Args))
