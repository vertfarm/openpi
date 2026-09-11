"""Source recordings -> sealed manifest -> export -> recipes -> sample schedules.

The module is campaign-neutral; the constants below are not. TODAY30/ALL59 is
the 2026-09-10 campaign, and the next one edits these values rather than
starting a `two_track_*`-style third generation - which is what the
`overnight_*` -> `readapt_*` -> `two_track_*` history cost this project.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random

import numpy as np

from . import native
from .artifacts import ContractError
from .artifacts import checked as checked
from .artifacts import digest
from .artifacts import file_hash
from .artifacts import read_json
from .artifacts import sealed as sealed
from .artifacts import write_new_json

# Keep this string. It is stamped into every manifest, schedule, snapshot and
# checkpoint registry already on disk, and `deploy_server` refuses a registry
# that does not carry it. Renaming the modules does not rename the data; a new
# value here would orphan the campaign the field is running.
SCHEMA = "hv1_two_track_v1"
TRACKS = ("TODAY30", "ALL59")
FILTER_FINETUNES = ("TODAY30_FT", "ALL59_FT")
FILTER_PARENT = {
    "TODAY30_FT": ("TODAY30", 1000),
    "ALL59_FT": ("ALL59", 2000),
}
OLD_SESSION = "keti_humanoid_data_260909"
TODAY_SESSION = "keti_humanoid_data_260910"
OLD_EXCLUDED = {"episode_000000", "episode_000001", "episode_000033"}
OLD_SUSPECT = {f"episode_{i:06d}" for i in (4, 8, 13, 20)}
OLD_DIAGNOSTIC = {f"episode_{i:06d}" for i in range(27, 33)}
TODAY_DIAGNOSTIC = {f"episode_{i:06d}" for i in (2, 12, 20, 22, 25, 32)}
EXPECTED_OLD = {f"episode_{i:06d}" for i in range(3, 33)} - {"episode_000011"}
EXPECTED_TODAY = {f"episode_{i:06d}" for i in range(33)} - {
    "episode_000006",
    "episode_000014",
    "episode_000028",
}
TARGET_STEPS = 2000

# Ablations that ask why the policy ignores its cameras. Each one is a separate
# experiment name, never an edit to a track's recipe: `pipeline_config.configure`
# compares a snapshot's stored recipe against `recipe()`, so changing TODAY30's
# would make `deploy_server` refuse the checkpoint the field is running.
#
# Each entry is the *only* difference from its track's recipe, so the comparison
# stays one-variable. They keep a single snapshot because disk, not time, is the
# binding constraint (2,000 updates measured 1,006 s; a snapshot costs 4.9 GiB).
# Knobs only an ablation may carry. They are *added* to a recipe rather than
# overriding something, so they must be listed here explicitly - a track's recipe
# must keep the exact fields its snapshots were written with, or configure stops
# recognising them.
ABLATION_ONLY_FIELDS = ("state_noise_sigma",)

ABLATIONS = {
    # V1 (run 2026-09-11, no effect). The stopping decision is where object
    # position matters and it is a small share of frames, so weight the close
    # window instead of the shared approach. It did not break the shortcut:
    # arm pose still works as a phase clock inside the close window too.
    "TODAY30_CLOSE45": dict(
        track="TODAY30",
        phase_fractions={"uniform": 0.40, "close": 0.45, "release": 0.15},
    ),
    # V2. Degrade the clock itself. One sigma of each channel's training spread
    # is added to the state the trainer sees, so arm pose stops being a precise
    # statement of progress and the images are the only exact source left.
    "TODAY30_NOISE10": dict(track="TODAY30", state_noise_sigma=1.0),
    # V3. Both levers, to see whether they only work together.
    "TODAY30_CLOSE45_NOISE10": dict(
        track="TODAY30",
        phase_fractions={"uniform": 0.40, "close": 0.45, "release": 0.15},
        state_noise_sigma=1.0,
    ),
}


def uid(session, episode):
    if not session or "::" in session or not episode.startswith("episode_") or "::" in episode:
        raise ContractError("invalid session/episode identity")
    return f"{session}::{episode}"


def _external_output(output, *sources):
    out = Path(output).resolve()
    if any(out == Path(source).resolve() or out.is_relative_to(Path(source).resolve()) for source in sources):
        raise ContractError("campaign must be outside immutable source roots")
    if {part.lower() for part in out.parts} & {"raw", "datasets"}:
        raise ContractError("campaign cannot be stored under raw/datasets")
    return out


def _metadata(root, expected_session):
    root = Path(root).resolve()
    meta = read_json(root / "metadata.json")
    if meta.get("dataset_name") != expected_session or float(meta.get("hz", 0)) != 30:
        raise ContractError("unexpected session identity/rate")
    if set(meta.get("cameras", {})) != set(native.CAMERAS):
        raise ContractError("three required cameras are not declared")
    if any(value.get("resolution") != [640, 480] or value.get("rotate") != 0 for value in meta["cameras"].values()):
        raise ContractError("camera geometry differs from the training contract")
    return root, meta


def _scan(root, meta, selected, cohort):
    indexed = {episode["id"]: episode for episode in meta["episodes"]}
    if set(selected) - set(indexed):
        raise ContractError(f"{cohort}: expected episode is absent from metadata")
    result = []
    for eid in sorted(selected):
        directory = (root / "episodes" / eid).resolve()
        if not directory.is_relative_to(root / "episodes"):
            raise ContractError("episode path escapes source custody")
        hashes = {path.name: file_hash(path) for path in native.source_files(directory)}
        _, _, info = native.read_numeric(directory / "data.hdf5", allow_multiple_cycles=True)
        task = read_json(directory / "tasks.json")
        if (
            indexed[eid]["num_frames"] != info["frames"]
            or task["num_frames"] != info["frames"]
            or task["episode_id"] != eid
        ):
            raise ContractError("metadata/tasks/HDF5 frame mismatch")
        if hashes != {path.name: file_hash(path) for path in native.source_files(directory)}:
            raise ContractError("source changed during scan")
        diagnostic = (
            "old_fixed6"
            if cohort == "old" and eid in OLD_DIAGNOSTIC
            else "today_fixed6"
            if cohort == "today" and eid in TODAY_DIAGNOSTIC
            else None
        )
        tracks = ["ALL59"] + (["TODAY30"] if cohort == "today" else [])
        numeric_summary = {
            key: info[key]
            for key in (
                "frames",
                "created_at",
                "max_command_step_rad",
                "grasp_frame",
                "release_frame",
                "grasp_frames",
                "release_frames",
                "gripper_events",
                "stale_rows",
            )
        }
        result.append(
            dict(
                id=uid(meta["dataset_name"], eid),
                episode_id=eid,
                session_id=meta["dataset_name"],
                cohort=cohort,
                path=str(directory),
                source_hashes=hashes,
                source_task=task["main_prompt"],
                task=native.PROMPT,
                training_tracks=tracks,
                diagnostic_group=diagnostic,
                suspect=cohort == "old" and eid in OLD_SUSPECT,
                numeric_pass=True,
                **numeric_summary,
            )
        )
    return result


def prepare(old_root, today_root, output):
    old_root, old_meta = _metadata(old_root, OLD_SESSION)
    today_root, today_meta = _metadata(today_root, TODAY_SESSION)
    out = _external_output(output, old_root, today_root)
    if out.exists():
        raise ContractError("campaign must be a new directory")
    old_ids = set(old_meta["episodes"][i]["id"] for i in range(len(old_meta["episodes"]))) - OLD_EXCLUDED
    today_ids = {episode["id"] for episode in today_meta["episodes"]}
    if old_ids != EXPECTED_OLD or today_ids != EXPECTED_TODAY:
        raise ContractError("source inventory differs from the approved 29+30 plan")
    episodes = _scan(old_root, old_meta, EXPECTED_OLD, "old") + _scan(today_root, today_meta, EXPECTED_TODAY, "today")
    value = sealed(
        dict(
            schema=SCHEMA,
            profile=native.profile(),
            prompt=native.PROMPT,
            source_roots={"old": str(old_root), "today": str(today_root)},
            source_metadata_sha256={
                "old": file_hash(old_root / "metadata.json"),
                "today": file_hash(today_root / "metadata.json"),
            },
            episodes=episodes,
            tracks={
                "TODAY30": dict(episode_ids=[e["id"] for e in episodes if "TODAY30" in e["training_tracks"]]),
                "ALL59": dict(episode_ids=[e["id"] for e in episodes]),
            },
            diagnostics={
                "old_fixed6": [e["id"] for e in episodes if e["diagnostic_group"] == "old_fixed6"],
                "today_fixed6": [e["id"] for e in episodes if e["diagnostic_group"] == "today_fixed6"],
                "role": "training_overlap_diagnostic_not_common_independent_validation",
            },
            initialization="official_pi05_base_independent_per_track",
            normalization="recomputed_per_track_training_data_only",
            target_steps=TARGET_STEPS,
            selection_basis="user_authorized_all_task_data_pool_2026-09-10",
            excluded={"old": sorted(OLD_EXCLUDED), "today": []},
            robot_motion_authorized=False,
        )
    )
    write_new_json(out / "manifest.json", value)
    return value


def verify_campaign(campaign, *, raw=False):
    campaign = Path(campaign).resolve()
    manifest = checked(campaign / "manifest.json")
    if manifest.get("schema") != SCHEMA or manifest.get("profile") != native.profile():
        raise ContractError("unsupported two-track contract")
    ids = [episode["id"] for episode in manifest["episodes"]]
    if len(ids) != 59 or len(ids) != len(set(ids)):
        raise ContractError("two-track manifest must contain 59 unique episodes")
    for track, expected in (("TODAY30", 30), ("ALL59", 59)):
        selected = manifest["tracks"][track]["episode_ids"]
        if len(selected) != expected or len(selected) != len(set(selected)):
            raise ContractError("track episode count/identity mismatch")
    if raw:
        for episode in manifest["episodes"]:
            directory = Path(episode["path"]).resolve()
            for name, sha256 in episode["source_hashes"].items():
                if file_hash(directory / name) != sha256:
                    raise ContractError(f"raw changed: {episode['id']}/{name}")
    return manifest


def export(campaign):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from openpi_client import image_tools

    campaign = Path(campaign).resolve()
    manifest = verify_campaign(campaign, raw=True)
    destination = campaign / "export"
    if destination.exists():
        raise ContractError("export already exists; use a new campaign")
    profile = manifest["profile"]
    features = {
        "observation.state": dict(dtype="float32", shape=(15,), names=profile["state"]["names"]),
        "action": dict(dtype="float32", shape=(8,), names=profile["action"]["names"]),
    }
    features.update(
        {
            f"observation.images.{camera}": dict(
                dtype="image", shape=(224, 224, 3), names=["height", "width", "channel"]
            )
            for camera in native.CAMERAS
        }
    )
    repo_id = f"hv1/silver_all59_{manifest['sha256'][:8]}"
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
    for episode in manifest["episodes"]:
        directory = Path(episode["path"])
        state, action, _ = native.read_numeric(directory / "data.hdf5", allow_multiple_cycles=True)
        for index, images in enumerate(native.decode_episode(directory, len(state))):
            frame = {"observation.state": state[index], "action": action[index], "task": native.PROMPT}
            for camera, image in images.items():
                frame[f"observation.images.{camera}"] = image_tools.resize_with_pad(image, 224, 224)
            writer.add_frame(frame)
        writer.save_episode()
        if any(file_hash(directory / name) != sha for name, sha in episode["source_hashes"].items()):
            raise ContractError("source changed during export")
        print(json.dumps(dict(exported=episode["id"], frames=len(state))), flush=True)
    loaded = LeRobotDataset(repo_id=repo_id, root=target)
    expected_frames = sum(episode["frames"] for episode in manifest["episodes"])
    if len(loaded) != expected_frames:
        raise ContractError("common export count mismatch")
    write_new_json(target / "hv1_provenance.json", manifest["episodes"])
    value = dict(
        schema_version=1,
        manifest_sha256=manifest["sha256"],
        profile=profile,
        profile_sha256=digest(profile),
        hf_lerobot_home=str(destination),
        splits={
            "train": dict(
                repo_id=repo_id,
                root=str(target),
                frames=len(loaded),
                episode_ids=[episode["id"] for episode in manifest["episodes"]],
            )
        },
        track_episode_ids={track: manifest["tracks"][track]["episode_ids"] for track in TRACKS},
        diagnostics=manifest["diagnostics"],
        prompt=native.PROMPT,
        complete=True,
    )
    write_new_json(destination / "export.json", value)
    return value


def track_ids(manifest, track):
    if track not in TRACKS:
        raise ContractError("unknown training track")
    return set(manifest["tracks"][track]["episode_ids"])


def compute_statistics(campaign, track):
    from openpi.shared import normalize

    campaign = Path(campaign).resolve()
    manifest = verify_campaign(campaign, raw=True)
    selected = track_ids(manifest, track)
    asset_id = f"hv1_{track.lower()}_{manifest['sha256'][:8]}"
    root = campaign / "assets" / asset_id
    if root.exists():
        raise ContractError("track statistics already exist")
    running = {key: normalize.RunningStats() for key in ("state", "actions")}
    frame_count = 0
    for episode in manifest["episodes"]:
        if episode["id"] not in selected:
            continue
        state, action, _ = native.read_numeric(Path(episode["path"]) / "data.hdf5", allow_multiple_cycles=True)
        indices = np.minimum(np.arange(len(state))[:, None] + np.arange(15), len(state) - 1)
        chunks = action[indices].copy()
        chunks[:, :, :7] -= state[:, None, :7]
        running["state"].update(state)
        running["actions"].update(chunks.reshape(-1, 8))
        frame_count += len(state)
    normalize.save(root, {key: value.get_statistics() for key, value in running.items()})
    provenance = dict(
        schema=SCHEMA,
        manifest_sha256=manifest["sha256"],
        track=track,
        asset_id=asset_id,
        episode_ids=sorted(selected),
        episode_count=len(selected),
        training_frames=frame_count,
        action_horizon_basis=15,
        validation_used=False,
        norm_stats_sha256=file_hash(root / "norm_stats.json"),
    )
    write_new_json(root / "provenance.json", provenance)
    return provenance


def recipe(name, manifest_sha, steps=TARGET_STEPS):
    if name not in {*TRACKS, *FILTER_FINETUNES, *ABLATIONS, "SMOKE"}:
        raise ContractError("unknown two-track experiment")
    expected_steps = 50 if name == "SMOKE" else 1000 if name in FILTER_FINETUNES else TARGET_STEPS
    if steps != expected_steps:
        raise ContractError("recipe step count differs from the approved two-track plan")
    snapshots = (
        [50]
        if name == "SMOKE"
        else [500, 1000]
        if name in FILTER_FINETUNES
        else [TARGET_STEPS]
        if name in ABLATIONS
        else [250, 500, 1000, TARGET_STEPS]
    )
    track = (
        "TODAY30"
        if name == "SMOKE"
        else FILTER_PARENT[name][0]
        if name in FILTER_PARENT
        else ABLATIONS[name]["track"]
        if name in ABLATIONS
        else name
    )
    parent = None
    if name in FILTER_PARENT:
        parent_track, parent_step = FILTER_PARENT[name]
        parent = {
            "experiment": parent_track,
            "step": parent_step,
            "snapshot": f"snapshots/{parent_track}/step_{parent_step:06d}",
        }
    value = dict(
        name=name,
        track=track,
        steps=steps,
        batch_size=2,
        seed=42,
        manifest_sha256=manifest_sha,
        peak_lr=2.5e-6 if name in FILTER_FINETUNES else 1e-5,
        decay_lr=2.5e-7 if name in FILTER_FINETUNES else 1e-6,
        warmup_steps=50 if name in FILTER_FINETUNES else 100,
        decay_steps=1000 if name in FILTER_FINETUNES else 5000,
        action_horizon=15,
        lora=False,
        ema_decay=None,
        phase_fractions={"uniform": 0.70, "close": 0.15, "release": 0.15},
        snapshots=snapshots,
        initialization=(
            "BF16_parent_weights_FP32_training_new_optimizer"
            if name in FILTER_FINETUNES
            else "official_pi05_base_new_optimizer"
        ),
        inference_snapshots_priority=True,
        robot_motion_authorized=False,
    )
    if parent is not None:
        value["parent"] = parent
    if name in ABLATIONS:
        # Applied last and by key, so an ablation can only change fields the base
        # recipe already defines - a typo becomes an error, not a silent new knob.
        for key, override in ABLATIONS[name].items():
            if key == "track":
                continue
            if key not in value and key not in ABLATION_ONLY_FIELDS:
                raise ContractError(f"ablation {name} overrides unknown recipe field {key!r}")
            value[key] = override
        value["ablation_of"] = ABLATIONS[name]["track"]
    return value


def sample_schedule(manifest, name, samples):
    if samples <= 0 or samples % 100:
        raise ContractError("sample count must be a positive multiple of 100")
    steps = 50 if name == "SMOKE" else 1000 if name in FILTER_FINETUNES else TARGET_STEPS
    recipe_value = recipe(name, manifest["sha256"], steps)
    selected = track_ids(manifest, recipe_value["track"])
    offsets, start = {}, 0
    for episode in manifest["episodes"]:
        offsets[episode["id"]] = start
        start += episode["frames"]
    pool = sorted((episode for episode in manifest["episodes"] if episode["id"] in selected), key=lambda e: e["id"])
    rng = random.Random(recipe_value["seed"])
    cycles = {phase: [] for phase in recipe_value["phase_fractions"]}
    records = []
    assignments = [phase for phase, share in recipe_value["phase_fractions"].items() for _ in range(round(100 * share))]
    if len(assignments) != 100:
        raise ContractError("phase fractions must allocate exactly 100 samples")
    for _ in range(samples // 100):
        rng.shuffle(assignments)
        for phase in assignments:
            if not cycles[phase]:
                cycles[phase] = list(pool)
                rng.shuffle(cycles[phase])
            episode = cycles[phase].pop()
            lo, hi = 0, episode["frames"] - 1
            event = None
            if phase != "uniform":
                candidates = episode["grasp_frames" if phase == "close" else "release_frames"]
                event = rng.choice(candidates)
                lo, hi = max(0, event - 30), min(hi, event + 30)
            frame = rng.randint(lo, hi)
            records.append(
                dict(
                    index=offsets[episode["id"]] + frame,
                    episode=episode["id"],
                    frame=frame,
                    cohort=episode["cohort"],
                    phase=phase,
                    event_frame=event,
                )
            )
    return sealed(
        dict(
            schema=SCHEMA,
            name=name,
            track=recipe_value["track"],
            samples=samples,
            records=records,
            common_export_episode_order=[episode["id"] for episode in manifest["episodes"]],
            train_episode_ids=sorted(selected),
        )
    )


def coverage(schedule, consumed):
    rows = schedule["records"][:consumed]
    return dict(
        consumed_samples=len(rows),
        unique_anchors=len({(row["episode"], row["frame"]) for row in rows}),
        cohort_counts=dict(Counter(row["cohort"] for row in rows)),
        phase_counts=dict(Counter(row["phase"] for row in rows)),
        episode_counts=dict(Counter(row["episode"] for row in rows)),
        event_anchor_counts=dict(
            Counter(f"{row['phase']}:{row['event_frame']}" for row in rows if row["event_frame"] is not None)
        ),
    )


def write_schedules(campaign):
    campaign = Path(campaign).resolve()
    manifest = verify_campaign(campaign)
    result = {}
    for name in ("SMOKE", *TRACKS):
        value = recipe(name, manifest["sha256"], 50 if name == "SMOKE" else TARGET_STEPS)
        schedule = sample_schedule(manifest, name, value["steps"] * value["batch_size"])
        write_new_json(campaign / f"sampler_{name}.json", schedule)
        result[name] = dict(samples=schedule["samples"], sha256=schedule["sha256"])
    return result


def write_ablation_schedules(campaign):
    """Write any ablation schedule that is missing; verify the ones already there.

    Re-runnable on purpose: the orchestrator calls this when resuming, and an
    existing schedule is evidence rather than an obstacle - but only if it still
    matches what the recipe derives, otherwise the run would train on a plan
    nobody approved.
    """
    campaign = Path(campaign).resolve()
    manifest = verify_campaign(campaign)
    result = {}
    for name in ABLATIONS:
        value = recipe(name, manifest["sha256"], TARGET_STEPS)
        schedule = sample_schedule(manifest, name, value["steps"] * value["batch_size"])
        path = campaign / f"sampler_{name}.json"
        if path.exists():
            if checked(path) != schedule:
                raise ContractError(f"existing sampler_{name}.json differs from the approved schedule")
            state = "verified"
        else:
            write_new_json(path, schedule)
            state = "written"
        result[name] = dict(samples=schedule["samples"], sha256=schedule["sha256"], state=state)
    return result


def write_filter_finetune_schedules(campaign):
    campaign = Path(campaign).resolve()
    manifest = verify_campaign(campaign)
    result = {}
    for name in FILTER_FINETUNES:
        value = recipe(name, manifest["sha256"], 1000)
        schedule = sample_schedule(manifest, name, value["steps"] * value["batch_size"])
        write_new_json(campaign / f"sampler_{name}.json", schedule)
        result[name] = dict(samples=schedule["samples"], sha256=schedule["sha256"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("prepare")
    command.add_argument("--old", required=True)
    command.add_argument("--today", required=True)
    command.add_argument("--campaign", required=True)
    for name in ("export", "stats", "schedules", "filter-schedules", "ablation-schedules", "status"):
        command = sub.add_parser(name)
        command.add_argument("--campaign", required=True)
        if name == "stats":
            command.add_argument("--track", choices=TRACKS, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.old, args.today, args.campaign)
    elif args.command == "export":
        result = export(args.campaign)
    elif args.command == "stats":
        result = compute_statistics(args.campaign, args.track)
    elif args.command == "schedules":
        result = write_schedules(args.campaign)
    elif args.command == "filter-schedules":
        result = write_filter_finetune_schedules(args.campaign)
    elif args.command == "ablation-schedules":
        result = write_ablation_schedules(args.campaign)
    else:
        campaign = Path(args.campaign)
        result = dict(
            manifest=(campaign / "manifest.json").exists(),
            export=(campaign / "export/export.json").exists(),
            tracks={track: (campaign / "runs" / track / "result.json").exists() for track in TRACKS},
            robot_motion_authorized=False,
        )
    if args.command == "prepare":
        result = {
            "schema": result["schema"],
            "sha256": result["sha256"],
            "episodes": len(result["episodes"]),
            "tracks": {track: len(result["tracks"][track]["episode_ids"]) for track in TRACKS},
            "robot_motion_authorized": result["robot_motion_authorized"],
        }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
