"""Local-only pinned LeRobot v2.1 export. Never uploads or deletes output."""

from pathlib import Path

import h5py
import numpy as np

from .artifacts import ContractError
from .artifacts import file_hash
from .artifacts import write_new_json
from .workflow import dataset
from .workflow import load_manifest
from .workflow import sampling_plan


def export_manifest(manifest_path, destination, *, allow_synthetic=False):
    manifest = load_manifest(manifest_path)
    p = manifest["profile"]
    if p["status"] == "synthetic" and not allow_synthetic:
        raise ContractError("explicit --allow-synthetic required")
    destination = Path(destination).resolve()
    if destination.is_relative_to(Path(manifest["raw_root"]).resolve()):
        raise ContractError("export destination must not be inside the immutable raw root")
    if destination.exists():
        raise ContractError("export destination exists; use a new versioned directory")
    # Heavy imports belong to the ML environment, never the recorder/ROS environment.
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    features = {
        "observation.state": {"dtype": "float32", "shape": (len(p["state"]["names"]),), "names": p["state"]["names"]},
        "action": {"dtype": "float32", "shape": (len(p["action"]["names"]),), "names": p["action"]["names"]},
    }
    for slot, stream in p["images"].items():
        features[f"observation.images.{slot}"] = {
            "dtype": "image",
            "shape": tuple(stream["shape"]),
            "names": ["height", "width", "channel"],
        }
    destination.mkdir(parents=True, exist_ok=False)
    exports = {}
    for split in ("train", "validation"):
        episodes = [e for e in manifest["episodes"] if e["split"] == split]
        if not episodes:
            continue
        repo_id = f"hv1/{manifest['manifest_sha256'][:16]}_{split}"
        target = destination / repo_id
        writer = LeRobotDataset.create(
            repo_id=repo_id,
            root=target,
            fps=p["fps"],
            robot_type="hv1",
            features=features,
            use_videos=False,
            image_writer_processes=0,
            image_writer_threads=2,
        )
        provenance = []
        for episode in episodes:
            if file_hash(episode["path"]) != episode["sha256"]:
                raise ContractError("raw source changed before export")
            with h5py.File(episode["path"], "r") as handle:
                grid, indices = sampling_plan(handle, p)
                for i in range(len(grid)):
                    frame = {
                        "observation.state": dataset(handle, p["state"]["path"])[indices["state"][i]].astype(
                            np.float32
                        ),
                        "action": dataset(handle, p["action"]["path"])[indices["action"][i]].astype(np.float32),
                        "task": episode["task"],
                    }
                    for slot, stream in p["images"].items():
                        rgb = dataset(handle, stream["path"])[indices[slot][i]]
                        frame[f"observation.images.{slot}"] = (
                            rgb[..., ::-1].copy() if stream["color_order"] == "BGR" else rgb
                        )
                    writer.add_frame(frame)
                writer.save_episode()
                provenance.append(
                    {
                        "source_id": episode["id"],
                        "source_sha256": episode["sha256"],
                        "session_id": episode["session_id"],
                        "timestamps_ns": grid.tolist(),
                        "source_indices": {k: v.tolist() for k, v in indices.items()},
                    }
                )
            if file_hash(episode["path"]) != episode["sha256"]:
                raise ContractError("raw source changed during export; incomplete export must not be used")
        # Pinned v2.1 writes episodes/meta in save_episode; no v3 finalize/consolidate API.
        reloaded = LeRobotDataset(repo_id=repo_id, root=target)
        if len(reloaded) != sum(len(e["timestamps_ns"]) for e in provenance):
            raise ContractError("LeRobot roundtrip frame count mismatch")
        for key in ("observation.state", "action"):
            if not np.isfinite(np.asarray(reloaded[0][key])).all():
                raise ContractError("invalid roundtrip numeric data")
        write_new_json(target / "hv1_provenance.json", provenance)
        exports[split] = {"repo_id": repo_id, "root": str(target), "frames": len(reloaded)}
    result = {
        "schema_version": 1,
        "manifest_sha256": manifest["manifest_sha256"],
        "profile": p,
        "profile_sha256": manifest["profile_sha256"],
        "hf_lerobot_home": str(destination),
        "splits": exports,
        "lerobot_git_revision": "0cf864870cf29f4738d3ade893e6fd13fbd7cdb5",
        "complete": True,
    }
    # Only a fully successful export receives this completion marker.
    write_new_json(destination / "export.json", result)
    return result
