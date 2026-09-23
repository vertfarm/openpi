from collections import Counter
import io
import json
from pathlib import Path
import random
import tarfile
from types import SimpleNamespace

import pytest

from examples.hv1 import augmented
from examples.hv1 import native
from examples.hv1 import pipeline
from examples.hv1 import pipeline_train
from examples.hv1.artifacts import ContractError
from examples.hv1.artifacts import digest
from examples.hv1.artifacts import file_hash
from examples.hv1.artifacts import sealed
from examples.hv1.artifacts import write_new_json


TASKS = (
    "cylinder_table_to_tray",
    "cylinder_tray_to_table",
    "cylinder_language_zone_sort",
    "cylinder_axis_alignment",
)
PHASES = ("reach", "pre_grasp", "grasp", "lift", "transport", "place", "release")
KINEMATIC_PHASES = ("reach", "pre_grasp", "transport")
STYLES = ("rtx", "3dgs", "cosmos")
REVISION = "hv1_upper_body_r1"
NEXT_REVISION = "hv1_upper_body_r2_elbow_extended"
PHASE_MIX = {
    "reach": 0.2,
    "pre_grasp": 0.15,
    "grasp": 0.2,
    "lift": 0.15,
    "transport": 0.1,
    "place": 0.1,
    "release": 0.1,
}

# A four-task profile with all three render styles. It exists to prove the
# schedule builder is driven by the profile, not by constants in the code.
WIDE_MODELS = {
    "M0": {
        "label": "real_only",
        "source_counts": {"real": 100},
        "render_counts": {"real": 100},
        "pair_count": 0,
    },
    "M4": {
        "label": "all_plus_cosmos",
        "source_counts": {"real": 40, "sim_physics": 40, "sim_kinematic": 20},
        "render_counts": {"real": 40, "rtx": 30, "3dgs": 18, "cosmos": 12},
        "pair_count": 20,
    },
}

# The shipped v1 profile: one task, RTX only, M0 against M2.
V1_MODELS = {
    "M0": {
        "label": "real_only",
        "source_counts": {"real": 100},
        "render_counts": {"real": 100},
        "pair_count": 0,
    },
    "M2": {
        "label": "real_plus_physics_rtx",
        "source_counts": {"real": 50, "sim_physics": 50},
        "render_counts": {"real": 50, "rtx": 50},
        "pair_count": 20,
    },
}


def _profile(tasks, models, *, finalists, screening, name="test_profile"):
    return {
        "interface": augmented.AUGMENTED_SCHEMA,
        "campaign_profile": name,
        "tasks": [{"id": task, "weight": 1.0 / len(tasks)} for task in tasks],
        "phases": dict(PHASE_MIX),
        "kinematic": {"allowed_phases": list(KINEMATIC_PHASES)},
        "embodiment": {"revision_id": REVISION},
        "model_ablation": {
            "macrocycle_samples": 100,
            "batch_size": 2,
            "screening_updates": 500,
            "screening_models": list(screening),
            "finalists": list(finalists),
            "final_seeds": 3,
            "final_seed_values": [42, 43, 44],
            "final_updates": 28000,
            "models": models,
        },
    }


def _metadata(source, task, phase, style, paired=False, revision=REVISION):
    anchor_ns = 1_000_000_000
    return {
        "schema_version": augmented.AUGMENTED_SCHEMA,
        "task_id": task,
        "phase": phase,
        "embodiment_revision": revision,
        "source_domain": source,
        "render_style": style,
        "synthetic": source != "real",
        "sampleable": True,
        "success": True,
        "quality_weight": 1.0,
        "pair_group_id": "pair" if paired else None,
        "language": "실린더를 목표 위치에 놓아라",
        "seeds": {"trajectory": 1, "physics": 2, "render": 3, "generative": 4},
        "hashes": {
            "robot": "1" * 64,
            "scene": "2" * 64,
            "asset": "3" * 64,
            "policy_checkpoint": "4" * 64,
        },
        "camera_calibration": {camera: {} for camera in native.CAMERAS},
        "randomization_vector": {},
        "timestamps": {
            "anchor_ns": anchor_ns,
            "image_ns": {camera: anchor_ns for camera in native.CAMERAS},
            "actions_ns": [anchor_ns + step for step in range(15)],
        },
    }


def _fixture(tmp_path: Path, *, valid_images=False, profile=None):
    profile = profile or _profile(
        TASKS, WIDE_MODELS, finalists=("M0", "M4"), screening=("M0", "M4")
    )
    real_root = tmp_path / "real"
    (real_root / "meta").mkdir(parents=True)
    (real_root / "meta/info.json").write_text("{}\n", encoding="utf-8")
    export_path = tmp_path / "export.json"
    export = {
        "complete": True,
        "profile": native.profile(),
        "profile_sha256": digest(native.profile()),
        "splits": {
            "train": {"root": str(real_root), "repo_id": "hv1/test", "frames": len(TASKS) * len(PHASES)}
        },
    }
    write_new_json(export_path, export)
    real_records = []
    index = 0
    for task in TASKS:
        for phase in PHASES:
            real_records.append(dict(_metadata("real", task, phase, "real"), dataset_index=index))
            index += 1
    real_metadata_path = tmp_path / "real_metadata.json"
    write_new_json(
        real_metadata_path,
        sealed(
            {
                "schema": augmented.REAL_METADATA_SCHEMA,
                "real_export_sha256": file_hash(export_path),
                "records": real_records,
            }
        ),
    )

    sim = tmp_path / "sim"
    shard = sim / "anchors/rtx/all.tar"
    shard.parent.mkdir(parents=True)
    jpeg = b"\xff\xd8\xff\xd9"
    if valid_images:
        from PIL import Image

        image_stream = io.BytesIO()
        Image.new("RGB", (224, 224), color=(17, 34, 51)).save(image_stream, format="JPEG")
        jpeg = image_stream.getvalue()
    with tarfile.open(shard, "w") as archive:
        sample = 0
        for source, phases in (("sim_physics", PHASES), ("sim_kinematic", KINEMATIC_PHASES)):
            for task in TASKS:
                for phase in phases:
                    for style in STYLES:
                        for paired in (False, True):
                            sample_id = f"sample-{sample:05d}"
                            metadata = _metadata(source, task, phase, style, paired)
                            metadata["state"] = [0.0] * 15
                            metadata["actions"] = [[0.0] * 8 for _ in range(15)]
                            metadata["image_refs"] = {
                                camera: f"{sample_id}.{camera}.jpg" for camera in native.CAMERAS
                            }
                            payload = json.dumps(metadata, separators=(",", ":")).encode()
                            info = tarfile.TarInfo(f"{sample_id}.json")
                            info.size = len(payload)
                            archive.addfile(info, io.BytesIO(payload))
                            for camera in native.CAMERAS:
                                info = tarfile.TarInfo(f"{sample_id}.{camera}.jpg")
                                info.size = len(jpeg)
                                archive.addfile(info, io.BytesIO(jpeg))
                            sample += 1
    catalog_path = sim / "manifests/catalog.json"
    catalog_path.parent.mkdir(parents=True)
    write_new_json(
        catalog_path,
        {
            "shards": [
                {
                    "kind": "anchors/rtx",
                    "relative_path": "anchors/rtx/all.tar",
                    "sha256": file_hash(shard),
                }
            ]
        },
    )
    write_new_json(sim / augmented.SEALED_CONFIG_RELATIVE, profile)
    return export_path, real_metadata_path, sim


def test_deployment_pipeline_carries_no_augmented_surface():
    """The deployed recipe path and the augmented experiments stay separate.

    `pipeline_config.configure` re-derives a snapshot's recipe from
    `pipeline.recipe`, so an augmented edit landing in `pipeline.py` could make
    `deploy_server` refuse the checkpoint the field runs.
    """
    source = Path(pipeline.__file__).read_text(encoding="utf-8")
    assert "augmented" not in source.lower()


def test_augmented_index_and_schedule_follow_the_sealed_profile(tmp_path):
    export, metadata, sim = _fixture(tmp_path)
    index_path = tmp_path / "augmented/index.json"
    value = augmented.build_augmented_index(export, metadata, sim, index_path, allow_synthetic=True)
    assert value["schema"] == augmented.AUGMENTED_SCHEMA
    assert value["profile_sha256"] == digest(value["profile"])
    assert value["campaign_profile"] == "test_profile"

    schedule = augmented.augmented_schedule(index_path, "M4", 100, 20260915, allow_synthetic=True)
    assert Counter(row["source_domain"] for row in schedule["records"]) == {
        "real": 40,
        "sim_physics": 40,
        "sim_kinematic": 20,
    }
    assert Counter(row["task_id"] for row in schedule["records"]) == {task: 25 for task in TASKS}
    assert Counter(row["phase"] for row in schedule["records"]) == {
        "reach": 20,
        "pre_grasp": 15,
        "grasp": 20,
        "lift": 15,
        "transport": 10,
        "place": 10,
        "release": 10,
    }
    assert Counter(
        row["render_style"] for row in schedule["records"] if row["source_domain"] != "real"
    ) == {"rtx": 30, "3dgs": 18, "cosmos": 12}
    assert sum(row["paired"] for row in schedule["records"]) == 20
    assert all(
        row["phase"] in KINEMATIC_PHASES
        for row in schedule["records"]
        if row["source_domain"] == "sim_kinematic"
    )


def test_single_task_rtx_profile_needs_no_code_change(tmp_path):
    """The v1 scope - one task, RTX only, M0 vs M2 - is a profile edit."""
    profile = _profile(
        TASKS[:1], V1_MODELS, finalists=("M0", "M2"), screening=("M0", "M2"), name="v1_single_task_rtx"
    )
    export, metadata, sim = _fixture(tmp_path, profile=profile)
    index_path = tmp_path / "index.json"
    augmented.build_augmented_index(export, metadata, sim, index_path, allow_synthetic=True)

    schedule = augmented.augmented_schedule(index_path, "M2", 100, 42, allow_synthetic=True)
    assert Counter(row["source_domain"] for row in schedule["records"]) == {
        "real": 50,
        "sim_physics": 50,
    }
    assert Counter(row["task_id"] for row in schedule["records"]) == {"cylinder_table_to_tray": 100}
    assert {row["render_style"] for row in schedule["records"]} == {"real", "rtx"}

    recipe = augmented.augmented_recipe("M2", "confirm", 43, profile)
    assert recipe["steps"] == 28000
    assert recipe["batch_size"] == 2
    assert recipe["campaign_profile"] == "v1_single_task_rtx"
    # A model the profile does not define at all.
    with pytest.raises(ContractError, match="unknown augmented model"):
        augmented.augmented_recipe("M4", "confirm", 43, profile)
    # A model the profile defines but did not promote to the confirm stage.
    screened_only = _profile(
        TASKS[:1], V1_MODELS, finalists=("M2",), screening=("M0", "M2")
    )
    with pytest.raises(ContractError, match="finalists"):
        augmented.augmented_recipe("M0", "confirm", 43, screened_only)


def test_confirm_budget_consumes_the_generated_anchors(tmp_path):
    """The reason `final_updates` is 28,000 and not 2,000.

    A 2,000-update confirm draws 4,000 samples per run. Sizing a 50,000-anchor
    campaign against that would leave most of what was rendered unread, which
    is the imbalance the sim profile's epoch guard exists to catch.
    """
    profile = _profile(
        TASKS[:1], V1_MODELS, finalists=("M0", "M2"), screening=("M0", "M2")
    )
    ablation = profile["model_ablation"]
    sim_share = ablation["models"]["M2"]["source_counts"]["sim_physics"] / 100
    sim_draws = ablation["final_updates"] * ablation["batch_size"] * ablation["final_seeds"] * sim_share
    assert sim_draws / 45_000 > 1.0


def test_real_annotations_are_sealed_without_rewriting_the_export(tmp_path):
    export, _, _ = _fixture(tmp_path)
    records = []
    for index, (task, phase) in enumerate((task, phase) for task in TASKS for phase in PHASES):
        records.append(dict(_metadata("real", task, phase, "real"), dataset_index=index))
    annotations_path = tmp_path / "annotations.json"
    write_new_json(
        annotations_path,
        sealed({"schema": augmented.REAL_ANNOTATION_SCHEMA, "records": records}),
    )
    output = tmp_path / "wrapped.json"
    value = augmented.build_real_wrapper_metadata(export, annotations_path, output)
    assert value["real_export_sha256"] == file_hash(export)
    assert value["annotation_sha256"] == pipeline.checked(annotations_path)["sha256"]
    assert len(value["records"]) == len(TASKS) * len(PHASES)


def test_sim_reader_uses_sealed_tar_offsets_and_returns_the_policy_contract(tmp_path):
    pytest.importorskip("PIL.Image")
    export, metadata, sim = _fixture(tmp_path, valid_images=True)
    index_path = tmp_path / "index.json"
    value = augmented.build_augmented_index(export, metadata, sim, index_path, allow_synthetic=True)
    record_index = next(
        index
        for index, record in enumerate(value["records"])
        if record["data_ref"]["kind"] == "anchor_tar"
    )
    dataset = pipeline_train.AugmentedDataset(
        index_path,
        SimpleNamespace(action_horizon=15),
        allow_synthetic=True,
        real_dataset=[],
    )
    sample = dataset[record_index]
    assert sample["observation.state"].shape == (15,)
    assert sample["action"].shape == (15, 8)
    assert all(
        sample[f"observation.images.{camera}"].shape == (224, 224, 3)
        for camera in native.CAMERAS
    )


def test_augmented_index_and_schedule_fail_closed_without_synthetic_gate(tmp_path):
    export, metadata, sim = _fixture(tmp_path)
    index_path = tmp_path / "index.json"
    with pytest.raises(ContractError, match="allow-synthetic"):
        augmented.build_augmented_index(export, metadata, sim, index_path)
    augmented.build_augmented_index(export, metadata, sim, index_path, allow_synthetic=True)
    with pytest.raises(ContractError, match="allow-synthetic"):
        augmented.augmented_schedule(index_path, "M4", 100, 42)


def test_prefetched_shard_hash_is_rechecked_at_schedule_time(tmp_path):
    export, metadata, sim = _fixture(tmp_path)
    index_path = tmp_path / "index.json"
    augmented.build_augmented_index(export, metadata, sim, index_path, allow_synthetic=True)
    with (sim / "anchors/rtx/all.tar").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ContractError, match="SHA-256"):
        augmented.load_augmented_index(index_path, allow_synthetic=True)


def test_shard_is_hashed_once_per_file_not_once_per_record(tmp_path, monkeypatch):
    """Guards the load against O(records x shard bytes) re-hashing.

    Every record names one of a few shards. Hashing inside the record loop made
    a campaign-scale index unloadable: 50,000 records over 2 GiB shards is
    ~100 TB of reads for one call, and training calls it three times.
    """
    export, metadata, sim = _fixture(tmp_path)
    index_path = tmp_path / "index.json"
    value = augmented.build_augmented_index(export, metadata, sim, index_path, allow_synthetic=True)
    anchor_records = sum(
        record["data_ref"]["kind"] == "anchor_tar" for record in value["records"]
    )
    assert anchor_records > 10

    hashed = []
    real_file_hash = augmented.file_hash

    def counting_file_hash(path):
        hashed.append(str(path))
        return real_file_hash(path)

    monkeypatch.setattr(augmented, "file_hash", counting_file_hash)
    augmented.load_augmented_index(index_path, allow_synthetic=True)
    shard_hashes = [path for path in hashed if path.endswith("all.tar")]
    assert len(shard_hashes) == 1


def test_data_from_two_embodiments_does_not_mix_silently(tmp_path):
    """The planned elbow rework retires the workspace its data was recorded in.

    A wider elbow range moves the reachable set, the IK branches and the pose
    hull together, so rows from before and after the swap describe different
    robots. Mixing them has to be a decision, not a default.
    """
    export, metadata_path, sim = _fixture(tmp_path)
    index_path = tmp_path / "index.json"

    # Re-label the real rows as if they had been recorded on the new arm while
    # the sim campaign still generates against the old one.
    real_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for record in real_metadata["records"]:
        record["embodiment_revision"] = NEXT_REVISION
    # `write_new_json` refuses to replace a sealed file, which is the point of
    # it; re-seal and write directly instead of weakening that guard.
    resealed = sealed({key: value for key, value in real_metadata.items() if key != "sha256"})
    metadata_path.write_text(json.dumps(resealed, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ContractError, match="embodiment revisions disagree"):
        augmented.build_augmented_index(
            export, metadata_path, sim, index_path, allow_synthetic=True
        )

    value = augmented.build_augmented_index(
        export,
        metadata_path,
        sim,
        index_path,
        allow_synthetic=True,
        allow_mixed_embodiment=True,
    )
    assert value["embodiment_mixed"] is True
    assert set(value["embodiment_revision_counts"]) == {REVISION, NEXT_REVISION}


def test_profile_change_after_indexing_is_rejected(tmp_path):
    export, metadata, sim = _fixture(tmp_path)
    index_path = tmp_path / "index.json"
    augmented.build_augmented_index(export, metadata, sim, index_path, allow_synthetic=True)
    profile_path = sim / augmented.SEALED_CONFIG_RELATIVE
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile["model_ablation"]["final_updates"] = 1
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    with pytest.raises(ContractError, match="profile changed"):
        augmented.load_augmented_index(index_path, allow_synthetic=True)


@pytest.mark.parametrize(
    ("models", "tasks"),
    [(WIDE_MODELS, TASKS), (V1_MODELS, TASKS[:1])],
)
def test_every_ablation_macrocycle_is_exact(models, tasks):
    profile = _profile(
        tasks, models, finalists=tuple(models), screening=tuple(models)
    )
    per_task = 100 // len(tasks)
    for model in models:
        slots = augmented._augmented_macro_slots(model, random.Random(42), profile)
        assert len(slots) == 100
        assert Counter(slot["task_id"] for slot in slots) == {task: per_task for task in tasks}
        assert Counter(slot["source_domain"] for slot in slots) == models[model]["source_counts"]
