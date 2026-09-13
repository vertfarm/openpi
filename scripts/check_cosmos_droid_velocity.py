"""Offline (no RobotEnv) websocket check for the Cosmos DROID velocity proxy.

Counterpart of ``check_pi05_droid_jointpos_velocity.py``. Same role: prove the server answers
the deployed contract before anything is allowed to move. Three things differ from the pi05
check and each is a thing that has actually gone wrong:

* the horizon is 32, not 15;
* the request carries **both** exterior views -- sending the pi05 request shape (one exterior
  view at 224x224) scored 0/24 in simulation, and this script fails loudly rather than
  producing a plausible-looking chunk from a blind policy;
* ``--round-trip`` replays the commands the way ``RobotEnv(action_space="joint_velocity")``
  will, so the check reports where the arm would actually be sent, in radians.

    python scripts/check_cosmos_droid_velocity.py --port 8001
"""

from __future__ import annotations

import dataclasses
import pathlib

import numpy as np
from openpi_client import websocket_client_policy
import time
import tyro

from openpi.policies import cosmos_droid_policy


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 8001
    # A saved observation, as a scalar-object .npy or a flat .npz. Without one, a synthetic
    # frame is used -- enough to check the contract, not enough to judge the policy.
    observation: pathlib.Path | None = None
    save_response: pathlib.Path | None = None
    expected_horizon: int = cosmos_droid_policy.COSMOS_ACTION_HORIZON
    joint_delta_scale: float = 0.2
    max_abs_velocity: float = 0.5
    round_trip: bool = True


def _synthetic_observation() -> dict:
    rng = np.random.default_rng(0)
    return {
        "observation/exterior_image_1_left": rng.integers(1, 256, (720, 1280, 3), dtype=np.uint8),
        "observation/exterior_image_2_left": rng.integers(1, 256, (720, 1280, 3), dtype=np.uint8),
        "observation/wrist_image_left": rng.integers(1, 256, (720, 1280, 3), dtype=np.uint8),
        # The Franka reset pose, so joint-limit reporting below means something.
        "observation/joint_position": np.array(
            [0.0, -np.pi / 5, 0.0, -4 * np.pi / 5, 0.0, 3 * np.pi / 5, 0.0], dtype=np.float32
        ),
        "observation/gripper_position": np.zeros(1, dtype=np.float32),
        "prompt": "pick up the object and place it into the container",
    }


def _load_observation(path: pathlib.Path | None) -> dict:
    if path is None:
        return _synthetic_observation()
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

    missing = [
        key
        for key in ("observation/joint_position", "observation/gripper_position")
        if key not in observation
    ]
    if missing:
        raise RuntimeError(f"Observation is missing {missing}")
    if "observation/image" not in observation and "observation/exterior_image_2_left" not in observation:
        raise RuntimeError(
            "Observation carries no second exterior view. Cosmos needs both exterior cameras; "
            "the pi05 request shape (one exterior view) is not a valid Cosmos observation."
        )

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
    if np.max(np.abs(actions[:, :7])) > args.max_abs_velocity + 1e-6:
        raise RuntimeError("Joint velocity exceeded the server safety limit")

    saturation = float(np.mean(np.abs(actions[:, :7]) >= args.max_abs_velocity - 1e-6))
    if args.save_response is not None:
        args.save_response.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.save_response, actions=actions)

    print(f"server_metadata={metadata}")
    print(f"action_shape={actions.shape}")
    print("actions_finite=true")
    print(f"joint_velocity_min={actions[:, :7].min():.6f}")
    print(f"joint_velocity_max={actions[:, :7].max():.6f}")
    print(f"joint_velocity_abs_median={np.median(np.abs(actions[:, :7])):.6f}")
    print(f"joint_limit_fraction={saturation:.6f}")
    print(f"gripper_min={actions[:, 7].min():.6f} gripper_max={actions[:, 7].max():.6f}")
    print(f"round_trip_latency_ms={latency_ms:.1f}")
    # The client needs a fresh chunk every open_loop_horizon/15 seconds. Report the largest
    # cadence this latency can sustain, because the DROID default (8) cannot be served by Cosmos.
    print(f"min_open_loop_horizon_for_realtime={int(np.ceil(latency_ms / 1000 * 15))}")

    if args.round_trip:
        # Replay the commands the way RobotEnv(action_space="joint_velocity") will, assuming the
        # arm tracks each target before the next step. This says where the arm is actually sent.
        state = np.asarray(observation["observation/joint_position"], dtype=np.float64).reshape(-1)
        measured = state[:7].copy()
        targets = []
        for velocity in actions:
            measured = measured + velocity[:7] * args.joint_delta_scale
            targets.append(measured.copy())
        targets = np.stack(targets)
        print(f"reconstructed_target_first={np.array2string(targets[0], precision=4)}")
        print(f"reconstructed_target_last={np.array2string(targets[-1], precision=4)}")
        print(f"reconstructed_total_joint_travel_rad={np.abs(targets[-1] - state[:7]).max():.6f}")

    print("OFFLINE_INFERENCE_STATUS=PASS")


if __name__ == "__main__":
    main(tyro.cli(Args))
