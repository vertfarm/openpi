"""Source recordings -> LeRobot export -> official loader -> registered config.

This is the only test that runs the real openpi loader, and it replaced the
readapt integration test on 2026-09-11 when that campaign was retired. It
exists for one question the offline metrics cannot answer: does the frame the
sampler asked for reach the trainer? Every episode's video carries a flat
brightness equal to `25 + its index in the common export`, so a batch can be
read back and matched against the schedule that produced it.

All fixtures are synthetic and tiny; nothing here loads model weights or opens
a robot connection.
"""

import json

import numpy as np
import pytest

from examples.hv1 import pipeline
from examples.hv1 import pipeline_eval
from examples.hv1 import pipeline_train
from examples.hv1.artifacts import ContractError
from examples.hv1.artifacts import file_hash
from examples.hv1.artifacts import write_new_json
from examples.hv1.transforms import CAMERA_KEYS

FRAMES = 3
EPISODES = 59
BASE_LEVEL = 25


def read_level(image):
    """Recover the source brightness from a loaded, padded, normalized frame."""
    array = np.asarray(image, dtype=np.float32)
    if array.min() < -0.5:  # openpi hands the model [-1, 1]; the export is [0, 255].
        array = (array + 1) / 2 * 255
    elif array.max() <= 1.5:
        array = array * 255
    return array.max()


def test_the_pipeline_from_source_recordings_to_a_registered_shadow_config(tmp_path, monkeypatch, source_session):
    jax = pytest.importorskip("jax")
    pytest.importorskip("av")
    pytest.importorskip("lerobot")
    from huggingface_hub import HfApi

    def no_remote_dataset(*args, **kwargs):
        raise AssertionError("the local export must never fall back to the Hub")

    monkeypatch.setattr(HfApi, "list_repo_refs", no_remote_dataset)

    shape = dict(frames=FRAMES, cycles=((1, 2),), video=True, hand=0.35, arm_step=0.02)
    old = source_session(
        tmp_path / "old",
        pipeline.OLD_SESSION,
        pipeline.EXPECTED_OLD,
        declared=pipeline.EXPECTED_OLD | pipeline.OLD_EXCLUDED,
        first_tint=0,
        **shape,
    )
    today = source_session(
        tmp_path / "today",
        pipeline.TODAY_SESSION,
        pipeline.EXPECTED_TODAY,
        first_tint=len(pipeline.EXPECTED_OLD) * FRAMES,
        **shape,
    )

    campaign = tmp_path / "campaign"
    manifest = pipeline.prepare(old, today, campaign)
    assert len(manifest["episodes"]) == EPISODES

    export = pipeline.export(campaign)
    assert export["complete"] and export["splits"]["train"]["frames"] == EPISODES * FRAMES
    assert export["manifest_sha256"] == manifest["sha256"]

    provenance = pipeline.compute_statistics(campaign, "TODAY30")
    assert provenance["episode_count"] == 30 and provenance["validation_used"] is False
    assert provenance["training_frames"] == 30 * FRAMES

    recipe = pipeline.recipe("TODAY30", manifest["sha256"])
    config, _ = pipeline_train.configure(campaign, recipe)
    assert config.batch_size == 2 and config.ema_decay is None
    assert config.policy_metadata["robot_motion_authorized"] is False
    assert config.policy_metadata["norm_stats_sha256"] == provenance["norm_stats_sha256"]
    assert config.policy_metadata["train_episode_ids"] == sorted(manifest["tracks"]["TODAY30"]["episode_ids"])

    schedule = pipeline.sample_schedule(manifest, "TODAY30", 4000)
    assert max(row["index"] for row in schedule["records"]) < EPISODES * FRAMES
    mesh = jax.sharding.Mesh(np.array(jax.devices("cpu")), ("batch",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("batch"))
    observation, actions = next(iter(pipeline_train.make_loader(config, schedule, sharding)))
    assert actions.shape == (2, 15, 32)
    assert np.isfinite(np.asarray(actions)).all()
    assert set(observation.images) == set(CAMERA_KEYS.values())
    assert all(np.asarray(mask).all() for mask in observation.image_masks.values())

    # The sampler names a row of the common export; this is that row arriving.
    expected = [BASE_LEVEL + row["index"] for row in schedule["records"][: config.batch_size]]
    seen = [read_level(image) for image in np.asarray(observation.images["base_0_rgb"])]
    assert all(abs(a - b) <= 8 for a, b in zip(seen, expected, strict=True)), (seen, expected)
    # A swapped camera key would put the head's channel on a wrist.
    assert read_level(np.asarray(observation.images["left_wrist_0_rgb"])[0]) > 0

    # The registry is the only door to a shadow rollout, and it stays shut until
    # the diagnostic evaluation for this exact snapshot is on disk.
    snapshot = campaign / "snapshots/TODAY30/step_002000"
    snapshot.mkdir(parents=True)
    (snapshot / "weights.inert").write_bytes(b"TEST ONLY, NOT A MODEL")
    record = dict(
        complete=True,
        cpu_roundtrip_pass=True,
        step=2000,
        manifest_sha256=manifest["sha256"],
        recipe=recipe,
        norm_stats_sha256=provenance["norm_stats_sha256"],
        files_sha256={"weights.inert": file_hash(snapshot / "weights.inert")},
    )
    write_new_json(snapshot / "snapshot.json", record)
    with pytest.raises(FileNotFoundError):
        pipeline_eval.register(campaign, snapshot, "synthetic-test")

    evaluation = campaign / "evaluations/TODAY30_002000.json"
    evidence = dict(
        complete=True,
        gpu_reload_pass=True,
        snapshot_sha256=file_hash(snapshot / "snapshot.json"),
        manifest_sha256=manifest["sha256"],
        groups={
            group: [{"episode": episode} for episode in manifest["diagnostics"][group]]
            for group in ("old_fixed6", "today_fixed6")
        },
    )
    write_new_json(evaluation, evidence)
    with pytest.raises(ContractError, match="reviewer and matching"):
        pipeline_eval.register(campaign, snapshot, "   ")
    entry = pipeline_eval.register(campaign, snapshot, "synthetic-test")
    assert entry["status"] == "SHADOW_ONLY" and entry["robot_motion_authorized"] is False

    registry = campaign / "checkpoint_registry.json"
    shadow_config, shadow_record = pipeline_eval.registered_config(campaign, snapshot, registry)
    assert shadow_record["step"] == 2000
    assert shadow_config.policy_metadata["norm_stats_sha256"] == provenance["norm_stats_sha256"]

    evaluation.write_text(json.dumps(dict(evidence, gpu_reload_pass=False)), encoding="utf-8")
    with pytest.raises(ContractError):
        pipeline_eval.registered_config(campaign, snapshot, registry)
