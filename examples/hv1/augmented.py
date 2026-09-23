"""`hv1_augmented_v1`: real rows plus simulated anchors under one sealed index.

This module is deliberately separate from `pipeline.py`. That module derives the
recipes `pipeline_config.configure` compares against a deployed snapshot, so an
edit there can make `deploy_server` refuse the checkpoint the field is running.
Nothing here is on that path: the augmented models are experiments, they are
never registered, and they carry their own schema string.

Campaign scope - the task list, the source/render mixture of each model, the
macrocycle size and the update counts - is read from the sim campaign's sealed
`manifests/campaign_config.json`, not hardcoded here. Adding a task or a render
style is a change to that profile, and both repositories then agree by
construction rather than by two copies that drift.
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
from .artifacts import checked
from .artifacts import digest
from .artifacts import file_hash
from .artifacts import read_json
from .artifacts import sealed
from .artifacts import write_new_json

AUGMENTED_SCHEMA = "hv1_augmented_v1"
REAL_METADATA_SCHEMA = "hv1_augmented_real_metadata_v1"
REAL_ANNOTATION_SCHEMA = "hv1_augmented_real_annotations_v1"
SEALED_CONFIG_RELATIVE = "manifests/campaign_config.json"

# The names an ablation may take. Which of them a campaign actually runs is the
# profile's decision, checked in `augmented_recipe`; this list only keeps the
# CLI from accepting a track name where a model belongs.
MODEL_NAMESPACE = ("M0", "M1", "M2", "M3", "M4")


# --------------------------------------------------------------------------
# campaign profile
# --------------------------------------------------------------------------


def load_campaign_profile(sim_campaign):
    """Read the profile the sim campaign sealed at initialisation."""

    path = Path(sim_campaign).resolve() / SEALED_CONFIG_RELATIVE
    if not path.is_file():
        raise ContractError(
            f"sim campaign has no sealed profile at {SEALED_CONFIG_RELATIVE}; "
            "initialise the campaign with a current hv1_campaign.py"
        )
    profile = read_json(path)
    if profile.get("interface") != AUGMENTED_SCHEMA:
        raise ContractError("sealed campaign profile is not hv1_augmented_v1")
    return profile


def profile_task_ids(profile):
    return tuple(str(row["id"]) for row in profile["tasks"])


def profile_embodiment_revision(profile):
    revision = profile.get("embodiment", {}).get("revision_id")
    if not isinstance(revision, str) or not revision:
        raise ContractError("sealed campaign profile does not name an embodiment revision")
    return revision


def profile_models(profile):
    return profile["model_ablation"]["models"]


def _largest_remainder(weights, total):
    raw = {key: value * total for key, value in weights.items()}
    counts = {key: int(value) for key, value in raw.items()}
    order = sorted(weights, key=lambda key: (raw[key] - counts[key], key), reverse=True)
    for key in order[: total - sum(counts.values())]:
        counts[key] += 1
    return counts


# --------------------------------------------------------------------------
# real wrapper and index
# --------------------------------------------------------------------------


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
    if annotations.get("schema") != REAL_ANNOTATION_SCHEMA:
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
            "schema": REAL_METADATA_SCHEMA,
            "real_export_sha256": file_hash(real_export_path),
            "profile_sha256": export["profile_sha256"],
            "annotation_sha256": annotations["sha256"],
            "records": records,
        }
    )
    write_new_json(output, value)
    return value


def build_augmented_index(
    real_export_path,
    real_metadata_path,
    sim_campaign,
    output,
    *,
    allow_synthetic=False,
    allow_mixed_embodiment=False,
):
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
    profile_path = sim_campaign / SEALED_CONFIG_RELATIVE
    profile = load_campaign_profile(sim_campaign)
    if real_metadata.get("schema") != REAL_METADATA_SCHEMA:
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

    # One index, one arm. A mechanical change - the planned elbow rework, for
    # instance - retires the derived assets, the reachable workspace and the
    # pose hull together, so rows from either side of it describe different
    # robots. Mixing them is a deliberate experiment, never a default.
    campaign_revision = profile_embodiment_revision(profile)
    revisions = Counter(
        str(record["metadata"].get("embodiment_revision", "")) for record in records
    )
    if "" in revisions:
        raise ContractError("some indexed rows do not name an embodiment revision")
    if not allow_mixed_embodiment and set(revisions) != {campaign_revision}:
        raise ContractError(
            "embodiment revisions disagree with the sim campaign profile "
            f"({campaign_revision!r}): {dict(revisions)!r}. Re-record or re-generate "
            "the odd rows, or pass --allow-mixed-embodiment to study the mixture "
            "on purpose."
        )

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
            campaign_profile=profile.get("campaign_profile"),
            campaign_profile_sha256=file_hash(profile_path),
            embodiment_revision=campaign_revision,
            embodiment_revision_counts=dict(revisions),
            embodiment_mixed=len(revisions) > 1,
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
    if value.get("campaign_profile_sha256") and file_hash(
        sim_root / SEALED_CONFIG_RELATIVE
    ) != value["campaign_profile_sha256"]:
        raise ContractError("sim campaign profile changed after indexing")
    # Hash each distinct shard once. Every record names one of a few shards, so
    # hashing per record would re-read gigabytes per record and the load would
    # never finish at campaign scale.
    verified = {}
    for record in value["records"]:
        data_ref = record["data_ref"]
        if data_ref["kind"] != "anchor_tar":
            continue
        relative = data_ref["shard_relative_path"]
        expected = data_ref["shard_sha256"]
        seen = verified.get(relative)
        if seen is None:
            shard = (sim_root / relative).resolve()
            if not shard.is_relative_to(sim_root):
                raise ContractError("prefetched sim shard escapes the campaign root")
            seen = verified[relative] = file_hash(shard)
        if seen != expected:
            raise ContractError("prefetched sim shard failed SHA-256")
    return value


# --------------------------------------------------------------------------
# schedules
# --------------------------------------------------------------------------


def _augmented_macro_slots(model, rng, profile):
    """One macrocycle of slots for `model`, in the profile's exact proportions.

    Source and render shares are integer counts per macrocycle in the profile,
    so the mixture is exact rather than a rounded fraction.
    """

    ablation = profile["model_ablation"]
    macrocycle = int(ablation["macrocycle_samples"])
    models = profile_models(profile)
    if model not in models:
        raise ContractError(f"unknown augmented model {model!r} for this campaign profile")
    definition = models[model]
    source_counts = {key: int(value) for key, value in definition["source_counts"].items()}
    tasks = profile_task_ids(profile)
    per_task, remainder = divmod(macrocycle, len(tasks))
    if remainder:
        raise ContractError("macrocycle_samples must divide evenly across the profile tasks")

    task_remaining = {task: per_task for task in tasks}
    slots = []
    sources = list(source_counts)
    for source_index, source in enumerate(sources):
        count = source_counts[source]
        if source_index == len(sources) - 1:
            allocation = dict(task_remaining)
        else:
            total_remaining = sum(task_remaining.values())
            allocation = _largest_remainder(
                {task: remaining / total_remaining for task, remaining in task_remaining.items()},
                count,
            )
        for task, amount in allocation.items():
            if amount > task_remaining[task]:
                raise ContractError("source/task quota cannot be allocated")
            task_remaining[task] -= amount
            slots.extend(dict(source_domain=source, task_id=task) for _ in range(amount))
    if any(task_remaining.values()) or len(slots) != macrocycle:
        raise ContractError("source/task macrocycle allocation failed")

    phase_counts = _largest_remainder(dict(profile["phases"]), macrocycle)
    kinematic_phases = tuple(profile["kinematic"]["allowed_phases"])
    kinematic_slots = [slot for slot in slots if slot["source_domain"] == "sim_kinematic"]
    if kinematic_slots:
        kinematic_total = sum(phase_counts[phase] for phase in kinematic_phases)
        kinematic_phase_counts = _largest_remainder(
            {phase: phase_counts[phase] / kinematic_total for phase in kinematic_phases},
            len(kinematic_slots),
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
    render_counts = {key: int(value) for key, value in definition["render_counts"].items()}
    simulated_styles = [
        style for style, count in render_counts.items() if style != "real" for _ in range(count)
    ]
    if len(simulated_styles) != len(simulated):
        raise ContractError("render_counts do not cover the simulated slots")
    rng.shuffle(simulated_styles)
    cursor = 0
    for slot in slots:
        if slot["source_domain"] == "real":
            slot["render_style"] = "real"
        else:
            slot["render_style"] = simulated_styles[cursor]
            cursor += 1
    pair_count = int(definition["pair_count"])
    if pair_count > len(simulated):
        raise ContractError("pair_count exceeds simulated slots")
    rng.shuffle(simulated)
    for slot in slots:
        slot["paired"] = False
    for slot in simulated[:pair_count]:
        slot["paired"] = True
    rng.shuffle(slots)
    return slots


def augmented_schedule(index_path, model, samples, seed, *, allow_synthetic=False):
    index = load_augmented_index(index_path, allow_synthetic=allow_synthetic)
    profile = load_campaign_profile(index["sim_campaign_root"])
    macrocycle = int(profile["model_ablation"]["macrocycle_samples"])
    if samples <= 0 or samples % macrocycle:
        raise ContractError(
            f"augmented schedule samples must be a positive multiple of {macrocycle}"
        )
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
    for _ in range(samples // macrocycle):
        for slot in _augmented_macro_slots(model, rng, profile):
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


def augmented_recipe(model, stage, seed, profile):
    """Training recipe for one ablation run, sized by the campaign profile."""

    ablation = profile["model_ablation"]
    models = profile_models(profile)
    if model not in models:
        raise ContractError("unknown augmented model")
    seeds = tuple(ablation["final_seed_values"])[: int(ablation["final_seeds"])]
    if stage == "screen":
        if model not in ablation["screening_models"]:
            raise ContractError("screen stage is limited to the profile screening models")
        if seed != seeds[0]:
            raise ContractError(f"screen stage uses the fixed seed {seeds[0]}")
        steps = int(ablation["screening_updates"])
    elif stage == "confirm":
        if model not in ablation["finalists"] or seed not in seeds:
            raise ContractError("confirm stage is limited to the profile finalists and seeds")
        steps = int(ablation["final_updates"])
    else:
        raise ContractError("unknown augmented training stage")
    return {
        "name": f"{model}_{stage}_s{seed}",
        "model": model,
        "stage": stage,
        "seed": seed,
        "steps": steps,
        "batch_size": int(ablation["batch_size"]),
        "snapshots": [steps],
        "peak_lr": 1e-5,
        "decay_lr": 1e-6,
        "warmup_steps": min(100, steps // 5),
        "decay_steps": max(steps, 5000),
        "mix": models[model],
        "campaign_profile": profile.get("campaign_profile"),
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
    profile = load_campaign_profile(index["sim_campaign_root"])
    ablation = profile["model_ablation"]
    seeds = tuple(ablation["final_seed_values"])[: int(ablation["final_seeds"])]
    output_dir = Path(output_dir).resolve()
    if stage == "screen":
        plans = [(model, seeds[0]) for model in ablation["screening_models"]]
    elif stage == "confirm":
        plans = [(model, seed) for model in ablation["finalists"] for seed in seeds]
    else:
        raise ContractError("unknown augmented training stage")
    result = {}
    for model, seed in plans:
        recipe_value = augmented_recipe(model, stage, seed, profile)
        value = augmented_schedule(
            index_path,
            model,
            recipe_value["steps"] * recipe_value["batch_size"],
            seed,
            allow_synthetic=True,
        )
        value = sealed(
            {key: item for key, item in value.items() if key != "sha256"}
            | {"recipe": recipe_value, "index_sha256": index["sha256"]}
        )
        path = output_dir / f"{recipe_value['name']}.json"
        write_new_json(path, value)
        result[recipe_value["name"]] = {"samples": value["samples"], "sha256": value["sha256"]}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("index")
    command.add_argument("--real-export", required=True)
    command.add_argument("--real-metadata", required=True)
    command.add_argument("--sim-campaign", required=True)
    command.add_argument("--output", required=True)
    command.add_argument("--allow-synthetic", action="store_true")
    command.add_argument(
        "--allow-mixed-embodiment",
        action="store_true",
        help="index rows from more than one hardware revision on purpose",
    )
    command = sub.add_parser("wrap-real")
    command.add_argument("--real-export", required=True)
    command.add_argument("--annotations", required=True)
    command.add_argument("--output", required=True)
    command = sub.add_parser("schedule")
    command.add_argument("--index", required=True)
    command.add_argument("--model", required=True)
    command.add_argument("--samples", type=int, required=True)
    command.add_argument("--seed", type=int, default=42)
    command.add_argument("--output", required=True)
    command.add_argument("--allow-synthetic", action="store_true")
    command = sub.add_parser("status")
    command.add_argument("--index", required=True)
    command.add_argument("--allow-synthetic", action="store_true")
    command = sub.add_parser("schedules")
    command.add_argument("--index", required=True)
    command.add_argument("--output-dir", required=True)
    command.add_argument("--stage", choices=("screen", "confirm"), required=True)
    command.add_argument("--allow-synthetic", action="store_true")
    args = parser.parse_args()

    if args.command == "index":
        result = build_augmented_index(
            args.real_export,
            args.real_metadata,
            args.sim_campaign,
            args.output,
            allow_synthetic=args.allow_synthetic,
            allow_mixed_embodiment=args.allow_mixed_embodiment,
        )
    elif args.command == "wrap-real":
        result = build_real_wrapper_metadata(args.real_export, args.annotations, args.output)
    elif args.command == "schedule":
        result = augmented_schedule(
            args.index,
            args.model,
            args.samples,
            args.seed,
            allow_synthetic=args.allow_synthetic,
        )
        write_new_json(args.output, result)
    elif args.command == "status":
        value = load_augmented_index(args.index, allow_synthetic=args.allow_synthetic)
        result = dict(
            schema=value["schema"],
            sha256=value["sha256"],
            campaign_profile=value.get("campaign_profile"),
            embodiment_revision=value.get("embodiment_revision"),
            embodiment_revision_counts=value.get("embodiment_revision_counts"),
            embodiment_mixed=value.get("embodiment_mixed"),
            samples=len(value["records"]),
            sources=dict(Counter(record["metadata"]["source_domain"] for record in value["records"])),
            tasks=dict(Counter(record["metadata"]["task_id"] for record in value["records"])),
            robot_motion_authorized=False,
        )
    else:
        result = write_augmented_schedules(
            args.index,
            args.output_dir,
            args.stage,
            allow_synthetic=args.allow_synthetic,
        )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
