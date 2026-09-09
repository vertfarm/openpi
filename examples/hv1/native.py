"""Immutable KETI HDF5/HEVC ingestion for the 2026-09-09 pilot. No ROS imports."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path

import h5py
import numpy as np

from .workflow import ContractError
from .workflow import digest
from .workflow import file_hash
from .workflow import read_json
from .workflow import validate_profile
from .workflow import write_new_json

PROMPT = "Pick up the silver cylindrical part from the table with the right hand and place it on the tray."
DUMMIES = {"episode_000000", "episode_000001"}
SUSPECT = {f"episode_{i:06d}" for i in (4, 8, 13, 20)}
VALIDATION = {f"episode_{i:06d}" for i in range(27, 33)}
CAMERAS = ("head", "hand_l", "hand_r")
ARM_STATE = [
    f"position/Right_{x}_Joint"
    for x in (
        "Shoulder_Pitch",
        "Shoulder_Roll",
        "Shoulder_Yaw",
        "Elbow_Pitch",
        "Wrist_Roll",
        "Wrist_Yaw",
        "Wrist_Pitch",
    )
]
ARM_ACTION = [f"points/positions/arm_r_joint{i}" for i in range(1, 8)]
HAND_STATE = [f"position/joint_{i}" for i in (10, 11, 12, 21, 22, 30, 31, 32)]


def profile():
    clock = "recorder_row_alignment_not_sensor_exposure_time"
    stream = lambda path: dict(path=path, timestamps_ns="derived_recorder_row_ns", clock_domain=clock)
    p = dict(
        schema_version=1,
        status="confirmed",
        source_format="keti_hdf5_hevc_v1",
        profile_id="hv1_right_silver_cylinder_3view_v1",
        fps=30,
        action_horizon=15,
        max_skew_ms=100,
        max_gap_ms=100,
        clock_domain=clock,
        state=dict(
            **stream("derived_state"),
            names=[x.removeprefix("position/") for x in ARM_STATE + HAND_STATE],
            units=["rad"] * 15,
        ),
        action=dict(
            **stream("derived_action"),
            names=[f"arm_r_joint{i}" for i in range(1, 8)] + ["right_grasp_intent"],
            units=["rad"] * 7 + ["binary"],
            semantics="commanded_target",
            delta_state_indices=list(range(7)) + [-1],
        ),
        images={c: dict(**stream(c + ".mp4"), shape=[224, 224, 3], color_order="RGB") for c in CAMERAS},
        gripper=dict(
            mode=2,
            wrap=True,
            close_intent=1,
            release_intent=0,
            release_open=0.6,
            provenance="derived_from_fixed_operator_protocol_not_grasp_success",
        ),
        camera_dropout=False,
        robot_motion_authorized=False,
    )
    return validate_profile(p)


def select_columns(f, key, names, labels):
    declared = labels[key]
    if len(declared) != len(set(declared)) or f[key].shape[1] != len(declared):
        raise ContractError(f"{key}: label/width mismatch")
    return f[key][:][:, [declared.index(n) for n in names]].astype(np.float32)


def read_numeric(path):
    with h5py.File(path, "r") as f:
        attrs = f["meta"].attrs
        labels = json.loads(attrs["labels"])
        q = select_columns(f, "observation.state.upper_body.joint", ARM_STATE, labels)
        hand = select_columns(f, "observation.state.hand.joint_r", HAND_STATE, labels)
        command = select_columns(f, "action.upper_body.joint", ARM_ACTION, labels)
        grip = select_columns(f, "action.hand.command_r", ["mode", "open"], labels)
        ts = f["timestamp"][:]
        if len(ts) < 2 or np.any(np.diff(ts) <= 0) or np.max(np.diff(ts)) > 0.1:
            raise ContractError("invalid recorder row timing")
        if not all(np.isfinite(x).all() for x in (q, hand, command, grip, ts)):
            raise ContractError("nonfinite training fields")
        if not np.allclose(grip[:, 0], 2) or not np.all(
            np.isclose(grip[:, 1], 0, atol=1e-4) | np.isclose(grip[:, 1], 0.6, atol=1e-4)
        ):
            raise ContractError("gripper outside confirmed mode2/0/0.6 contract")
        if np.any(f["stale"][:]):
            raise ContractError("stale recorder rows require explicit review")
        if not all(len(x) == len(ts) for x in (q, hand, command, grip)) or int(attrs["num_frames"]) != len(ts):
            raise ContractError("numeric length mismatch")
        state = np.concatenate((q, hand), axis=1)
        action = np.concatenate((command, np.isclose(grip[:, 1:2], 0, atol=1e-4).astype(np.float32)), axis=1)
        changes = np.r_[0, 1 + np.flatnonzero(np.diff(action[:, -1]) != 0)]
        if action[changes, -1].tolist() != [0, 1, 0]:
            raise ContractError("expected one open/close/release cycle")
        info = dict(
            frames=len(ts),
            created_at=str(attrs["created_at"]),
            max_command_step_rad=float(np.max(np.abs(np.diff(command, axis=0)))),
            grasp_frame=int(changes[1]),
            release_frame=int(changes[2]),
            stale_rows=0,
            row_timestamp_seconds=ts.tolist(),
            source_stamp_ns=f["stamp_ns"][:].tolist(),
        )
        return state, action, info


def decode_episode(directory, count):
    import av

    with ExitStack() as stack:
        videos = [stack.enter_context(av.open(str(directory / (c + ".mp4")))) for c in CAMERAS]
        streams = [iter(v.decode(video=0)) for v in videos]
        for i in range(count):
            try:
                frames = [next(s).to_ndarray(format="rgb24") for s in streams]
            except StopIteration as exc:
                raise ContractError(f"{directory.name}: video ends before row {i}") from exc
            if any(x.shape != (480, 640, 3) for x in frames):
                raise ContractError("unexpected source camera dimensions")
            yield dict(zip(CAMERAS, frames, strict=True))
        if any(next(s, None) is not None for s in streams):
            raise ContractError("video contains extra rows")


def source_files(directory):
    return [directory / "data.hdf5", directory / "tasks.json", *[directory / (c + ".mp4") for c in CAMERAS]]


def scan(raw, output):
    from PIL import Image
    from PIL import ImageDraw

    raw, output = Path(raw).resolve(), Path(output).resolve()
    if output.is_relative_to(raw):
        raise ContractError("output must be outside immutable raw")
    output.mkdir(parents=True, exist_ok=True)
    meta = read_json(raw / "metadata.json")
    indexed = {e["id"]: e for e in meta["episodes"]}
    reports = []
    for path in sorted(raw.glob("episodes/episode_*/data.hdf5")):
        eid, directory = path.parent.name, path.parent
        if eid in DUMMIES:
            continue
        before = {p.name: file_hash(p) for p in source_files(directory)}
        state, action, info = read_numeric(path)
        if eid not in indexed or indexed[eid]["num_frames"] != len(state):
            raise ContractError("episode is absent/incomplete in recorder index")
        task = read_json(directory / "tasks.json")
        if task["num_frames"] != len(state) or task["episode_id"] != eid:
            raise ContractError("tasks/index mismatch")
        g, r = info["grasp_frame"], info["release_frame"]
        selected = sorted(
            set(
                [
                    0,
                    max(0, g - 30),
                    g,
                    min(len(state) - 1, g + 60),
                    max(0, r - 60),
                    r,
                    min(len(state) - 1, r + 30),
                    len(state) - 1,
                ]
            )
        )
        sheet = Image.new("RGB", (320 * len(selected), 260 * 3), (255, 255, 255))
        draw = ImageDraw.Draw(sheet)
        freeze = {c: 0 for c in CAMERAS}
        previous = {}
        for i, images in enumerate(decode_episode(directory, len(state))):
            for c, img in images.items():
                if c in previous and np.array_equal(img, previous[c]):
                    freeze[c] += 1
                previous[c] = img
            if i in selected:
                x = selected.index(i) * 320
                for j, c in enumerate(CAMERAS):
                    sheet.paste(Image.fromarray(images[c]).resize((320, 240)), (x, j * 260 + 20))
                    draw.text((x + 3, j * 260 + 3), f"{eid} {c} frame={i}", fill=(0, 0, 0))
        after = {p.name: file_hash(p) for p in source_files(directory)}
        if before != after:
            raise ContractError("raw changed during scan")
        sheet.save(output / (eid + ".jpg"), quality=90)
        reports.append(
            dict(
                id=eid,
                path=str(directory),
                session_id=meta["dataset_name"],
                source_hashes=before,
                source_task=task["main_prompt"],
                task=PROMPT,
                task_override_reason="user confirmed silver cylinder right table-to-tray",
                suspect=eid in SUSPECT,
                validation=eid in VALIDATION,
                duplicate_decoded_frame_counts=freeze,
                visual_success="pending_review",
                **info,
            )
        )
        print(json.dumps(dict(scanned=eid, frames=len(state), duplicate_frames=freeze)), flush=True)
    found = {r["id"] for r in reports}
    if len(found) != 29 or not (SUSPECT | VALIDATION).issubset(found):
        raise ContractError("expected frozen 29-episode inventory; re-plan changed inventory")
    result = dict(
        schema_version=1,
        raw_root=str(raw),
        raw_metadata_sha256=file_hash(raw / "metadata.json"),
        excluded_dummies=sorted(DUMMIES),
        profile=profile(),
        episodes=reports,
        split_unit="episode",
        validation_scope="within_session",
        sensor_sync_verified=False,
    )
    result["manifest_sha256"] = digest(result)
    write_new_json(output / "scan.json", result)
    return result


def export(scan_path, destination, cohort):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from openpi_client import image_tools

    scan_data = read_json(scan_path)
    expected_hash = scan_data["manifest_sha256"]
    if digest({k: v for k, v in scan_data.items() if k != "manifest_sha256"}) != expected_hash:
        raise ContractError("scan manifest changed")
    p = scan_data["profile"]
    destination = Path(destination).resolve()
    if destination.is_relative_to(Path(scan_data["raw_root"])) or destination.exists():
        raise ContractError("new external destination required")
    features = {
        "observation.state": dict(dtype="float32", shape=(15,), names=p["state"]["names"]),
        "action": dict(dtype="float32", shape=(8,), names=p["action"]["names"]),
    }
    features.update(
        {
            f"observation.images.{c}": dict(dtype="image", shape=(224, 224, 3), names=["height", "width", "channel"])
            for c in CAMERAS
        }
    )
    splits = {}
    for split in ("train", "validation"):
        episodes = [
            e
            for e in scan_data["episodes"]
            if e["validation"] == (split == "validation")
            and (split == "validation" or cohort == "inclusive" or not e["suspect"])
        ]
        repo_id = f"hv1/silver_{cohort}_{expected_hash[:8]}_{split}"
        target = destination / repo_id
        writer = LeRobotDataset.create(
            repo_id=repo_id,
            root=target,
            fps=30,
            robot_type="hv1",
            features=features,
            use_videos=False,
            image_writer_processes=0,
            image_writer_threads=4,
        )
        provenance = []
        for e in episodes:
            directory = Path(e["path"])
            if any(file_hash(f) != e["source_hashes"][f.name] for f in source_files(directory)):
                raise ContractError("source changed before export")
            state, action, info = read_numeric(directory / "data.hdf5")
            for i, images in enumerate(decode_episode(directory, len(state))):
                frame = {"observation.state": state[i], "action": action[i], "task": PROMPT}
                for c, img in images.items():
                    frame[f"observation.images.{c}"] = image_tools.resize_with_pad(img, 224, 224)
                writer.add_frame(frame)
            writer.save_episode()
            if any(file_hash(f) != e["source_hashes"][f.name] for f in source_files(directory)):
                raise ContractError("source changed during export")
            provenance.append(e)
            print(json.dumps(dict(exported=e["id"], cohort=cohort, split=split)), flush=True)
        loaded = LeRobotDataset(repo_id=repo_id, root=target)
        if len(loaded) != sum(e["frames"] for e in episodes):
            raise ContractError("export count mismatch")
        write_new_json(target / "hv1_provenance.json", provenance)
        splits[split] = dict(
            repo_id=repo_id, root=str(target), frames=len(loaded), episode_ids=[e["id"] for e in episodes]
        )
    result = dict(
        schema_version=1,
        manifest_sha256=expected_hash,
        profile=p,
        profile_sha256=digest(p),
        hf_lerobot_home=str(destination),
        splits=splits,
        cohort=cohort,
        prompt=PROMPT,
        split_unit="episode",
        validation_scope="within_session",
        complete=True,
    )
    write_new_json(destination / "export.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["scan", "export"])
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cohort", choices=["clean", "inclusive"], default="clean")
    a = parser.parse_args()
    print(json.dumps(scan(a.source, a.output) if a.command == "scan" else export(a.source, a.output, a.cohort)))


if __name__ == "__main__":
    main()
