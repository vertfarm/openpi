"""Read-only exhaustive raw/export comparison; use the pinned ML environment."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from .artifacts import ContractError
from .artifacts import file_hash
from .artifacts import read_json
from .workflow import dataset
from .workflow import load_manifest
from .workflow import sampling_plan


def verify(manifest_path, export_path):
    manifest, export = load_manifest(manifest_path), read_json(export_path)
    if export.get("complete") is not True or any(
        export[key] != manifest[key] for key in ("manifest_sha256", "profile_sha256", "profile")
    ):
        raise ContractError("export/manifest contract mismatch")
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    p = manifest["profile"]
    checked = {}
    for split, info in export["splits"].items():
        ds = LeRobotDataset(info["repo_id"], root=Path(info["root"]))
        episodes = [e for e in manifest["episodes"] if e["split"] == split]
        cursor = 0
        for episode_index, episode in enumerate(episodes):
            with h5py.File(episode["path"], "r") as handle:
                grid, indices = sampling_plan(handle, p)
                for frame in range(len(grid)):
                    row = ds[cursor]
                    for key, target in (("state", "observation.state"), ("action", "action")):
                        expected = dataset(handle, p[key]["path"])[indices[key][frame]].astype(np.float32)
                        np.testing.assert_allclose(np.asarray(row[target]), expected, atol=1e-7)
                    for slot, stream in p["images"].items():
                        expected = dataset(handle, stream["path"])[indices[slot][frame]]
                        if stream["color_order"] == "BGR":
                            expected = expected[..., ::-1]
                        actual = np.asarray(row[f"observation.images.{slot}"]).transpose(1, 2, 0)
                        np.testing.assert_array_equal(np.rint(actual * 255).astype(np.uint8), expected)
                    if int(row["episode_index"]) != episode_index or int(row["frame_index"]) != frame:
                        raise ContractError("episode/frame indexing mismatch")
                    if ds.meta.tasks[int(row["task_index"])] != episode["task"]:
                        raise ContractError("task text mismatch")
                    if abs(float(row["timestamp"]) - frame / p["fps"]) > 1e-5:
                        raise ContractError("frame timestamp mismatch")
                    cursor += 1
            if file_hash(episode["path"]) != episode["sha256"]:
                raise ContractError("raw changed during comparison")
        if cursor != len(ds) or cursor != info["frames"]:
            raise ContractError("export frame count mismatch")
        checked[split] = {"episodes": len(episodes), "frames": cursor}
    return {"verified": checked, "all_rgb_state_action_task_time_equal": True, "model_loaded": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--export", required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.manifest, args.export)))


if __name__ == "__main__":
    main()
