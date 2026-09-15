from collections import Counter
import io
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

import pytest

from examples.hv1 import native
from examples.hv1 import pipeline
from examples.hv1 import pipeline_train
from examples.hv1.artifacts import ContractError
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


def _metadata(source, task, phase, style, paired=False):
    anchor_ns = 1_000_000_000
    return {
        "schema_version": pipeline.AUGMENTED_SCHEMA,
        "task_id": task,
        "phase": phase,
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


def _fixture(tmp_path: Path, *, valid_images=False):
    real_root = tmp_path / "real"
    (real_root / "meta").mkdir(parents=True)
    (real_root / "meta/info.json").write_text("{}\n", encoding="utf-8")
    export_path = tmp_path / "export.json"
    export = {
        "complete": True,
        "profile": native.profile(),
        "profile_sha256": pipeline.digest(native.profile()),
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
                "schema": "hv1_augmented_real_metadata_v1",
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
                            payload = json.dumps(
                                metadata,
                                separators=(",", ":"),
                            ).encode()
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
    return export_path, real_metadata_path, sim


def test_augmented_index_and_m4_schedule_enforce_the_campaign_mix(tmp_path):
    export, metadata, sim = _fixture(tmp_path)
    index_path = tmp_path / "augmented/index.json"
    value = pipeline.build_augmented_index(export, metadata, sim, index_path, allow_synthetic=True)
    assert value["schema"] == pipeline.AUGMENTED_SCHEMA
    assert value["profile_sha256"] == pipeline.digest(value["profile"])

    schedule = pipeline.augmented_schedule(index_path, "M4", 100, 20260915, allow_synthetic=True)
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


def test_real_annotations_are_sealed_without_rewriting_the_export(tmp_path):
    export, _, _ = _fixture(tmp_path)
    records = []
    for index, (task, phase) in enumerate((task, phase) for task in TASKS for phase in PHASES):
        records.append(dict(_metadata("real", task, phase, "real"), dataset_index=index))
    annotations_path = tmp_path / "annotations.json"
    write_new_json(
        annotations_path,
        sealed({"schema": "hv1_augmented_real_annotations_v1", "records": records}),
    )
    output = tmp_path / "wrapped.json"
    value = pipeline.build_real_wrapper_metadata(export, annotations_path, output)
    assert value["real_export_sha256"] == file_hash(export)
    assert value["annotation_sha256"] == pipeline.checked(annotations_path)["sha256"]
    assert len(value["records"]) == len(TASKS) * len(PHASES)


def test_sim_reader_uses_sealed_tar_offsets_and_returns_the_policy_contract(tmp_path):
    pytest.importorskip("PIL.Image")
    export, metadata, sim = _fixture(tmp_path, valid_images=True)
    index_path = tmp_path / "index.json"
    value = pipeline.build_augmented_index(export, metadata, sim, index_path, allow_synthetic=True)
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
        pipeline.build_augmented_index(export, metadata, sim, index_path)
    pipeline.build_augmented_index(export, metadata, sim, index_path, allow_synthetic=True)
    with pytest.raises(ContractError, match="allow-synthetic"):
        pipeline.augmented_schedule(index_path, "M4", 100, 42)


def test_prefetched_shard_hash_is_rechecked_at_schedule_time(tmp_path):
    export, metadata, sim = _fixture(tmp_path)
    index_path = tmp_path / "index.json"
    pipeline.build_augmented_index(export, metadata, sim, index_path, allow_synthetic=True)
    with (sim / "anchors/rtx/all.tar").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ContractError, match="SHA-256"):
        pipeline.load_augmented_index(index_path, allow_synthetic=True)


@pytest.mark.parametrize("model", pipeline.AUGMENTED_MODELS)
def test_every_ablation_macrocycle_is_exact(model):
    slots = pipeline._augmented_macro_slots(model, pipeline.random.Random(42))
    assert len(slots) == 100
    assert Counter(slot["task_id"] for slot in slots) == {task: 25 for task in TASKS}
    expected = {key: round(value * 100) for key, value in pipeline._augmented_model_mix(model)["source"].items()}
    assert Counter(slot["source_domain"] for slot in slots) == expected
