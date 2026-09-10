"""Create clearly synthetic fixtures only in a NEW output directory."""

from pathlib import Path

import h5py
import numpy as np

from .artifacts import write_new_json


def make_demo(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    raw = destination / "raw"
    raw.mkdir()

    def stream(path, names, units):
        return {
            "path": path,
            "timestamps_ns": "timestamps_ns",
            "names": names,
            "units": units,
            "clock_domain": "synthetic_monotonic",
        }

    profile = {
        "schema_version": 1,
        "profile_id": "SYNTHETIC-NOT-HV1-HARDWARE",
        "status": "synthetic",
        "clock_domain": "synthetic_monotonic",
        "fps": 10,
        "action_horizon": 4,
        "max_skew_ms": 50,
        "max_gap_ms": 150,
        "complete_attribute": "complete",
        "episode_id_attribute": "episode_id",
        "session_id_attribute": "session_id",
        "task_attribute": "task",
        "state": stream("observation/state", ["demo_joint_0", "demo_joint_1", "demo_open"], ["rad", "rad", "fraction"]),
        "action": {
            **stream("command/target", ["demo_joint_0", "demo_joint_1", "demo_open"], ["rad", "rad", "fraction"]),
            "semantics": "commanded_target",
            "delta_state_indices": [0, 1, -1],
        },
        "images": {
            name: {
                "path": f"images/{name}",
                "timestamps_ns": "timestamps_ns",
                "shape": [48, 64, 3],
                "color_order": "RGB",
                "clock_domain": "synthetic_monotonic",
            }
            for name in ("head", "hand_l", "hand_r")
        },
    }
    write_new_json(destination / "profile.json", profile)
    for episode in range(3):
        with h5py.File(raw / f"synthetic_{episode}.hdf5", "x") as handle:
            handle.attrs.update(
                complete=True,
                episode_id=f"synthetic_{episode}",
                session_id=f"session_{episode}",
                task="SYNTHETIC pipeline test; not a real robot task",
            )
            handle.create_dataset("timestamps_ns", data=1_000_000_000 + np.arange(12, dtype=np.int64) * 100_000_000)
            state = np.stack([np.linspace(0, 0.1, 12), np.linspace(0.1, 0, 12), np.ones(12) * 0.5], axis=1).astype(
                "float32"
            )
            handle.create_dataset("observation/state", data=state)
            handle.create_dataset("command/target", data=state + np.array([0.01, -0.01, 0], dtype="float32"))
            for slot, name in enumerate(profile["images"]):
                image = np.zeros((12, 48, 64, 3), dtype=np.uint8)
                image[..., slot] = 100 + episode * 40
                image[:, :8, :, :] = np.arange(12, dtype=np.uint8)[:, None, None, None] * 20
                handle.create_dataset(f"images/{name}", data=image, compression="gzip")
    return {"profile": str(destination / "profile.json"), "raw": str(raw)}
