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
import tarfile

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
AUGMENTED_SCHEMA = "hv1_augmented_v1"
AUGMENTED_MODELS = ("M0", "M1", "M2", "M3", "M4")
AUGMENTED_CONFIRM_MODELS = ("M0", "M3", "M4")
AUGMENTED_CONFIRM_SEEDS = (42, 43, 44)
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


def _largest_remainder(weights, total):
    raw = {key: value * total for key, value in weights.items()}
    counts = {key: int(value) for key, value in raw.items()}
    order = sorted(weights, key=lambda key: (raw[key] - counts[key], key), reverse=True)
    for key in order[: total - sum(counts.values())]:
        counts[key] += 1
    return counts


def _augmented_model_mix(model):
    mixes = {
        "M0": dict(source={"real": 1.0}, render={"real": 1.0}, pair_fraction=0.0),
        "M1": dict(
            source={"real": 2 / 3, "sim_kinematic": 1 / 3},
            render={"rtx": 0.5, "3dgs": 0.3, "cosmos": 0.2},
            pair_fraction=0.2,
        ),
        "M2": dict(
            source={"real": 0.5, "sim_physics": 0.5},
            render={"rtx": 1.0},
            pair_fraction=0.2,
        ),
        "M3": dict(
            source={"real": 0.5, "sim_physics": 0.5},
            render={"rtx": 0.625, "3dgs": 0.375},
            pair_fraction=0.2,
        ),
        "M4": dict(
            source={"real": 0.4, "sim_physics": 0.4, "sim_kinematic": 0.2},
            render={"rtx": 0.5, "3dgs": 0.3, "cosmos": 0.2},
            pair_fraction=0.2,
        ),
    }
    try:
        return mixes[model]
    except KeyError as exc:
        raise ContractError("unknown augmented model") from exc


def _anchor_metadata_from_tar(path):
    with tarfile.open(path, "r") as archive:
        members = [member for member in archive.getmembers() if member.isfile()]
        by_name = {member.name: member for member in members}
        if len(by_name) != len(members):
            raise ContractError("anchor shard contains duplicate member names")
        for member in members:
            if not member.isfile() or not member.name.endswith(".json"):
                continue
            source = archive.extractfile(member)
            if source is None:
                raise ContractError("anchor metadata member cannot be read")
            value = json.load(source)
            sample_id = member.name.removesuffix(".json")
            expected = {
                "metadata": member.name,
                **{camera: f"{sample_id}.{camera}.jpg" for camera in native.CAMERAS},
            }
            if value.get("image_refs") != {camera: expected[camera] for camera in native.CAMERAS}:
                raise ContractError("anchor metadata image_refs disagree with tar members")
            if any(name not in by_name for name in expected.values()):
                raise ContractError("anchor tar is missing a metadata or camera member")
            offsets = {
                key: {"offset": by_name[name].offset_data, "size": by_name[name].size}
                for key, name in expected.items()
            }
            yield sample_id, value, offsets


def build_real_wrapper_metadata(real_export_path, annotations_path, output):
    """Seal human-reviewed task/phase labels around the unchanged real export."""
    from .transforms import validate_augmented_sample

    real_export_path = Path(real_export_path).resolve()
    export = read_json(real_export_path)
    annotations = checked(annotations_path)
    if annotations.get("schema") != "hv1_augmented_real_annotations_v1":
        raise ContractError("unsupported real annotation schema")
    if export.get("complete") is not True or digest(export["profile"]) != export.get("profile_sha256"):
        raise ContractError("real export is incomplete or changed")
    frame_count = int(export["splits"]["train"]["frames"])
    indices = [record.get("dataset_index") for record in annotations.get("records", [])]
    if (
        not indices
        or any(type(index) is not int or index < 0 or index >= frame_count for index in indices)
        or len(indices) != len(set(indices))
    ):
        raise ContractError("real annotations contain missing, duplicate, or out-of-range indices")
    dummy = {
        "state": np.zeros(15, dtype=np.float32),
        "actions": np.zeros((15, 8), dtype=np.float32),
        "images": {camera: np.zeros((224, 224, 3), dtype=np.uint8) for camera in native.CAMERAS},
    }
    records = []
    for annotation in annotations["records"]:
        metadata = {key: value for key, value in annotation.items() if key != "dataset_index"}
        validate_augmented_sample({**metadata, **dummy}, allow_synthetic=False)
        records.append(dict(metadata, dataset_index=annotation["dataset_index"]))
    value = sealed(
        {
            "schema": "hv1_augmented_real_metadata_v1",
            "real_export_sha256": file_hash(real_export_path),
            "profile_sha256": export["profile_sha256"],
            "annotation_sha256": annotations["sha256"],
            "records": records,
        }
    )
    write_new_json(output, value)
    return value


def build_augmented_index(real_export_path, real_metadata_path, sim_campaign, output, *, allow_synthetic=False):
    """Index unchanged real LeRobot rows and hash-verified local sim shards."""
    if not allow_synthetic:
        raise ContractError("augmented index requires explicit --allow-synthetic")
    real_export_path = Path(real_export_path).resolve()
    real_metadata_path = Path(real_metadata_path).resolve()
    sim_campaign = Path(sim_campaign).resolve()
    export_value = read_json(real_export_path)
    real_metadata = checked(real_metadata_path)
    catalog_path = sim_campaign / "manifests/catalog.json"
    catalog = read_json(catalog_path)
    if real_metadata.get("schema") != "hv1_augmented_real_metadata_v1":
        raise ContractError("unsupported real wrapper metadata")
    if (
        export_value.get("complete") is not True
        or digest(export_value["profile"]) != export_value.get("profile_sha256")
        or real_metadata.get("real_export_sha256") != file_hash(real_export_path)
    ):
        raise ContractError("real wrapper/export identity mismatch")
    real_root = Path(export_value["splits"]["train"]["root"]).resolve()
    if not (real_root / "meta/info.json").is_file():
        raise ContractError("unchanged real LeRobot export is missing")
    records = []
    for metadata in real_metadata["records"]:
        if metadata.get("source_domain") != "real" or metadata.get("synthetic") is not False:
            raise ContractError("real wrapper metadata changed source identity")
        records.append(
            dict(
                data_ref=dict(kind="real_lerobot", index=metadata["dataset_index"]),
                metadata=metadata,
            )
        )
    for shard in catalog.get("shards", []):
        if not str(shard.get("kind", "")).startswith("anchors/"):
            continue
        path = (sim_campaign / shard["relative_path"]).resolve()
        if not path.is_relative_to(sim_campaign) or file_hash(path) != shard["sha256"]:
            raise ContractError("sim anchor shard is missing or changed")
        for sample_id, metadata, members in _anchor_metadata_from_tar(path):
            if metadata.get("synthetic") is not True:
                raise ContractError("sim anchor lacks synthetic=true")
            if not metadata.get("sampleable"):
                continue
            if metadata.get("success") is not True and metadata.get("corrected_recovery") is not True:
                continue
            records.append(
                dict(
                    data_ref=dict(
                        kind="anchor_tar",
                        shard_relative_path=shard["relative_path"],
                        shard_sha256=shard["sha256"],
                        sample_id=sample_id,
                        members=members,
                    ),
                    metadata={
                        key: value
                        for key, value in metadata.items()
                        if key not in {"state", "actions", "image_refs"}
                    },
                )
            )
    if not records:
        raise ContractError("augmented index has no records")
    value = sealed(
        dict(
            schema=AUGMENTED_SCHEMA,
            profile=export_value["profile"],
            profile_sha256=export_value["profile_sha256"],
            real_export=dict(
                root=str(real_root),
                repo_id=export_value["splits"]["train"]["repo_id"],
                export_manifest_path=str(real_export_path),
                export_sha256=file_hash(real_export_path),
                metadata_path=str(real_metadata_path),
                metadata_sha256=real_metadata["sha256"],
            ),
            sim_campaign_root=str(sim_campaign),
            sim_catalog_sha256=file_hash(catalog_path),
            records=records,
            synthetic_included=True,
            robot_motion_authorized=False,
        )
    )
    write_new_json(output, value)
    return value


def load_augmented_index(path, *, allow_synthetic=False):
    value = checked(path)
    if value.get("schema") != AUGMENTED_SCHEMA:
        raise ContractError("unsupported augmented index")
    if value.get("synthetic_included") and not allow_synthetic:
        raise ContractError("synthetic data requires explicit --allow-synthetic")
    if digest(value.get("profile")) != value.get("profile_sha256"):
        raise ContractError("augmented profile identity mismatch")
    if file_hash(value["real_export"]["export_manifest_path"]) != value["real_export"]["export_sha256"]:
        raise ContractError("real LeRobot export manifest changed after indexing")
    if checked(value["real_export"]["metadata_path"])["sha256"] != value["real_export"]["metadata_sha256"]:
        raise ContractError("real wrapper metadata changed after indexing")
    sim_root = Path(value["sim_campaign_root"]).resolve()
    if file_hash(sim_root / "manifests/catalog.json") != value["sim_catalog_sha256"]:
        raise ContractError("sim campaign catalog changed after indexing")
    for record in value["records"]:
        if record["data_ref"]["kind"] == "anchor_tar":
            shard = (sim_root / record["data_ref"]["shard_relative_path"]).resolve()
            if not shard.is_relative_to(sim_root) or file_hash(shard) != record["data_ref"]["shard_sha256"]:
                raise ContractError("prefetched sim shard failed SHA-256")
    return value


def _augmented_macro_slots(model, rng):
    mix = _augmented_model_mix(model)
    source_counts = _largest_remainder(mix["source"], 100)
    tasks = (
        "cylinder_table_to_tray",
        "cylinder_tray_to_table",
        "cylinder_language_zone_sort",
        "cylinder_axis_alignment",
    )
    task_remaining = {task: 25 for task in tasks}
    slots = []
    sources = list(source_counts)
    for source_index, source in enumerate(sources):
        count = source_counts[source]
        if source_index == len(sources) - 1:
            allocation = dict(task_remaining)
        else:
            total_remaining = sum(task_remaining.values())
            allocation = _largest_remainder(
                {task: remaining / total_remaining for task, remaining in task_remaining.items()}, count
            )
        for task, amount in allocation.items():
            if amount > task_remaining[task]:
                raise ContractError("source/task quota cannot be allocated")
            task_remaining[task] -= amount
            slots.extend(dict(source_domain=source, task_id=task) for _ in range(amount))
    if any(task_remaining.values()) or len(slots) != 100:
        raise ContractError("source/task macrocycle allocation failed")

    phase_counts = {
        "reach": 20,
        "pre_grasp": 15,
        "grasp": 20,
        "lift": 15,
        "transport": 10,
        "place": 10,
        "release": 10,
    }
    kinematic_slots = [slot for slot in slots if slot["source_domain"] == "sim_kinematic"]
    kinematic_phase_counts = _largest_remainder(
        {"reach": 20 / 45, "pre_grasp": 15 / 45, "transport": 10 / 45}, len(kinematic_slots)
    )
    rng.shuffle(kinematic_slots)
    cursor = 0
    for phase, count in kinematic_phase_counts.items():
        for slot in kinematic_slots[cursor : cursor + count]:
            slot["phase"] = phase
        phase_counts[phase] -= count
        cursor += count
    remaining_slots = [slot for slot in slots if "phase" not in slot]
    remaining_phases = [phase for phase, count in phase_counts.items() for _ in range(count)]
    if len(remaining_slots) != len(remaining_phases) or any(count < 0 for count in phase_counts.values()):
        raise ContractError("kinematic phase quota conflicts with campaign phase mix")
    rng.shuffle(remaining_phases)
    for slot, phase in zip(remaining_slots, remaining_phases, strict=True):
        slot["phase"] = phase

    simulated = [slot for slot in slots if slot["source_domain"] != "real"]
    render_counts = _largest_remainder(mix["render"], len(simulated)) if simulated else {}
    render_styles = [style for style, count in render_counts.items() for _ in range(count)]
    rng.shuffle(render_styles)
    for slot in slots:
        if slot["source_domain"] == "real":
            slot["render_style"] = "real"
        else:
            slot["render_style"] = render_styles.pop()
    pair_count = round(100 * mix["pair_fraction"])
    if pair_count > len(simulated):
        raise ContractError("pair fraction exceeds simulated slots")
    rng.shuffle(simulated)
    for slot in slots:
        slot["paired"] = False
    for slot in simulated[:pair_count]:
        slot["paired"] = True
    rng.shuffle(slots)
    return slots


def augmented_schedule(index_path, model, samples, seed, *, allow_synthetic=False):
    if samples <= 0 or samples % 100:
        raise ContractError("augmented schedule samples must be a positive multiple of 100")
    index = load_augmented_index(index_path, allow_synthetic=allow_synthetic)
    pools = {}
    for record_index, record in enumerate(index["records"]):
        metadata = record["metadata"]
        key = (
            metadata["source_domain"],
            metadata["task_id"],
            metadata["phase"],
            metadata["render_style"],
            bool(metadata.get("pair_group_id")),
        )
        pools.setdefault(key, []).append((record_index, float(metadata.get("quality_weight", 1.0))))
    rng = random.Random(seed)
    records = []
    for _ in range(samples // 100):
        for slot in _augmented_macro_slots(model, rng):
            key = (
                slot["source_domain"],
                slot["task_id"],
                slot["phase"],
                slot["render_style"],
                slot["paired"],
            )
            pool = pools.get(key, [])
            if not pool:
                raise ContractError(f"augmented sampling stratum is empty: {key!r}")
            indices, weights = zip(*pool, strict=True)
            if sum(weights) <= 0:
                raise ContractError(f"augmented sampling stratum has no positive quality weight: {key!r}")
            selected = rng.choices(indices, weights=weights, k=1)[0]
            records.append(dict(record_index=selected, **slot))
    return sealed(
        dict(
            schema=AUGMENTED_SCHEMA,
            model=model,
            samples=samples,
            seed=seed,
            index_sha256=index["sha256"],
            records=records,
            robot_motion_authorized=False,
        )
    )


def augmented_recipe(model, stage, seed):
    if model not in AUGMENTED_MODELS:
        raise ContractError("unknown augmented model")
    if stage == "screen":
        if seed != 42:
            raise ContractError("screen stage uses the fixed seed 42")
        steps = 500
    elif stage == "confirm":
        if model not in AUGMENTED_CONFIRM_MODELS or seed not in AUGMENTED_CONFIRM_SEEDS:
            raise ContractError("confirm stage is limited to M0/M3/M4 and seeds 42/43/44")
        steps = 2000
    else:
        raise ContractError("unknown augmented training stage")
    return {
        "name": f"{model}_{stage}_s{seed}",
        "model": model,
        "stage": stage,
        "seed": seed,
        "steps": steps,
        "batch_size": 2,
        "snapshots": [steps],
        "peak_lr": 1e-5,
        "decay_lr": 1e-6,
        "warmup_steps": min(100, steps // 5),
        "decay_steps": max(steps, 5000),
        "mix": _augmented_model_mix(model),
        "initialization": "official_pi05_base_new_optimizer",
        "robot_motion_authorized": False,
    }


def augmented_coverage(schedule, consumed):
    rows = schedule["records"][:consumed]
    return {
        "consumed_samples": len(rows),
        "unique_record_indices": len({row["record_index"] for row in rows}),
        "source_counts": dict(Counter(row["source_domain"] for row in rows)),
        "task_counts": dict(Counter(row["task_id"] for row in rows)),
        "phase_counts": dict(Counter(row["phase"] for row in rows)),
        "render_counts": dict(Counter(row["render_style"] for row in rows)),
        "paired_count": sum(bool(row["paired"]) for row in rows),
    }


def write_augmented_schedules(index_path, output_dir, stage, *, allow_synthetic=False):
    index = load_augmented_index(index_path, allow_synthetic=allow_synthetic)
    output_dir = Path(output_dir).resolve()
    plans = (
        [(model, 42) for model in AUGMENTED_MODELS]
        if stage == "screen"
        else [(model, seed) for model in AUGMENTED_CONFIRM_MODELS for seed in AUGMENTED_CONFIRM_SEEDS]
        if stage == "confirm"
        else None
    )
    if plans is None:
        raise ContractError("unknown augmented training stage")
    result = {}
    for model, seed in plans:
        recipe_value = augmented_recipe(model, stage, seed)
        value = augmented_schedule(
            index_path,
            model,
            recipe_value["steps"] * recipe_value["batch_size"],
            seed,
            allow_synthetic=True,
        )
        value = sealed(
            {
                key: item for key, item in value.items() if key != "sha256"
            }
            | {"recipe": recipe_value, "index_sha256": index["sha256"]}
        )
        path = output_dir / f"{recipe_value['name']}.json"
        write_new_json(path, value)
        result[recipe_value["name"]] = {"samples": value["samples"], "sha256": value["sha256"]}
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
    command = sub.add_parser("augmented-index")
    command.add_argument("--real-export", required=True)
    command.add_argument("--real-metadata", required=True)
    command.add_argument("--sim-campaign", required=True)
    command.add_argument("--output", required=True)
    command.add_argument("--allow-synthetic", action="store_true")
    command = sub.add_parser("augmented-wrap-real")
    command.add_argument("--real-export", required=True)
    command.add_argument("--annotations", required=True)
    command.add_argument("--output", required=True)
    command = sub.add_parser("augmented-schedule")
    command.add_argument("--index", required=True)
    command.add_argument("--model", choices=AUGMENTED_MODELS, required=True)
    command.add_argument("--samples", type=int, required=True)
    command.add_argument("--seed", type=int, default=42)
    command.add_argument("--output", required=True)
    command.add_argument("--allow-synthetic", action="store_true")
    command = sub.add_parser("augmented-status")
    command.add_argument("--index", required=True)
    command.add_argument("--allow-synthetic", action="store_true")
    command = sub.add_parser("augmented-schedules")
    command.add_argument("--index", required=True)
    command.add_argument("--output-dir", required=True)
    command.add_argument("--stage", choices=("screen", "confirm"), required=True)
    command.add_argument("--allow-synthetic", action="store_true")
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
    elif args.command == "augmented-index":
        result = build_augmented_index(
            args.real_export,
            args.real_metadata,
            args.sim_campaign,
            args.output,
            allow_synthetic=args.allow_synthetic,
        )
    elif args.command == "augmented-wrap-real":
        result = build_real_wrapper_metadata(args.real_export, args.annotations, args.output)
    elif args.command == "augmented-schedule":
        result = augmented_schedule(
            args.index,
            args.model,
            args.samples,
            args.seed,
            allow_synthetic=args.allow_synthetic,
        )
        write_new_json(args.output, result)
    elif args.command == "augmented-status":
        value = load_augmented_index(args.index, allow_synthetic=args.allow_synthetic)
        result = dict(
            schema=value["schema"],
            sha256=value["sha256"],
            samples=len(value["records"]),
            sources=dict(Counter(record["metadata"]["source_domain"] for record in value["records"])),
            tasks=dict(Counter(record["metadata"]["task_id"] for record in value["records"])),
            robot_motion_authorized=False,
        )
    elif args.command == "augmented-schedules":
        result = write_augmented_schedules(
            args.index,
            args.output_dir,
            args.stage,
            allow_synthetic=args.allow_synthetic,
        )
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
