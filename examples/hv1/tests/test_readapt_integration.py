"""Synthetic real HDF5/video -> LeRobot -> official CPU loader integration."""

import json

import numpy as np
import pytest


def test_real_export_fixed_sampler_and_registry(tmp_path, monkeypatch):
    jax = pytest.importorskip("jax")
    av = pytest.importorskip("av")
    pytest.importorskip("lerobot")
    from huggingface_hub import HfApi

    def no_remote_dataset(*args, **kwargs):
        raise AssertionError("local dataset must not fall back to the Hub")

    monkeypatch.setattr(HfApi, "list_repo_refs", no_remote_dataset)
    import h5py

    from examples.hv1 import native
    from examples.hv1 import readapt
    from examples.hv1 import readapt_eval
    from examples.hv1 import readapt_train
    from examples.hv1.workflow import ContractError
    from examples.hv1.workflow import digest
    from examples.hv1.workflow import file_hash
    from examples.hv1.workflow import write_new_json
    from openpi.shared import normalize

    parent = tmp_path / "parent"
    parent_snapshot = parent / "snapshots/C/step_005000"
    write_new_json(parent_snapshot / "snapshot.json", {"synthetic_fixture_not_a_model": True})
    stats_root = parent / "shared_assets/hv1_common_clean19"
    normalize.save(
        stats_root,
        {
            k: normalize.NormStats(mean=np.zeros(n), std=np.ones(n), q01=-np.ones(n), q99=np.ones(n))
            for k, n in (("state", 15), ("actions", 8))
        },
    )
    episodes = []
    for cohort, validation in (("new", False), ("old", False), ("new", True), ("old", True)):
        eid = f"episode_{int(validation):06d}"
        directory = tmp_path / "input" / cohort / eid
        directory.mkdir(parents=True)
        labels = {
            "observation.state.upper_body.joint": native.ARM_STATE,
            "observation.state.hand.joint_r": native.HAND_STATE,
            "action.upper_body.joint": native.ARM_ACTION,
            "action.hand.command_r": ["mode", "open"],
        }
        with h5py.File(directory / "data.hdf5", "w") as f:
            meta = f.create_group("meta")
            meta.attrs.update(labels=json.dumps(labels), num_frames=6, created_at="2026-09-10T03:00:00Z")
            for key, names in labels.items():
                f.create_dataset(key, data=np.zeros((6, len(names)), np.float32))
            f["action.upper_body.joint"][:] = 0.1
            f["action.hand.command_r"][:] = [[2, 0.6], [2, 0.6], [2, 0], [2, 0], [2, 0.6], [2, 0.6]]
            f.create_dataset("timestamp", data=np.arange(6) / 30 + 1)
            f.create_dataset("stamp_ns", data=np.arange(6, dtype=np.int64) * 33333333)
            f.create_dataset("stale", data=np.zeros(6, np.uint32))
        write_new_json(directory / "tasks.json", dict(episode_id=eid, num_frames=6, main_prompt="fixture"))
        for ci, camera in enumerate(native.CAMERAS):
            with av.open(str(directory / f"{camera}.mp4"), mode="w") as out:
                stream = out.add_stream("mpeg4", rate=30)
                stream.width, stream.height, stream.pix_fmt = 640, 480, "yuv420p"
                for i in range(6):
                    image = np.zeros((480, 640, 3), np.uint8)
                    image[..., ci] = 80 + i
                    frame = av.VideoFrame.from_ndarray(image, format="rgb24")
                    for packet in stream.encode(frame):
                        out.mux(packet)
                for packet in stream.encode():
                    out.mux(packet)
        _, _, info = native.read_numeric(directory / "data.hdf5")
        episodes.append(
            dict(
                info,
                id=readapt.uid(cohort, eid),
                episode_id=eid,
                session_id=cohort,
                path=str(directory),
                cohort=cohort,
                validation=validation,
                suspect=False,
                source_hashes={p.name: file_hash(p) for p in native.source_files(directory)},
            )
        )
    campaign = tmp_path / "campaign"
    initial = dict(
        campaign=str(parent),
        snapshot=str(parent_snapshot),
        snapshot_sha256=file_hash(parent_snapshot / "snapshot.json"),
        norm_stats=str(stats_root / "norm_stats.json"),
        norm_stats_sha256=file_hash(stats_root / "norm_stats.json"),
        prior_updates=5000,
    )
    m = readapt.sealed(
        dict(schema=readapt.SCHEMA, parent=initial, profile=native.profile(), episodes=episodes, synthetic_fixture=True)
    )
    write_new_json(campaign / "manifest.json", m)
    scan = dict(profile=native.profile(), raw_root=str(tmp_path / "input"), episodes=episodes)
    scan["manifest_sha256"] = digest(scan)
    write_new_json(campaign / "export_scan.json", scan)
    exported = native.export(campaign / "export_scan.json", campaign / "export", "inclusive")
    assert exported["splits"]["train"]["frames"] == 12
    assert exported["splits"]["validation"]["frames"] == 12
    config, _ = readapt_train.configure(campaign, readapt.recipe("M", m["sha256"]))
    assert config.weight_loader.params_path == str(parent_snapshot / "params")
    assert config.ema_decay is None and config.batch_size == 2
    assert config.policy_metadata["parent"]["snapshot_sha256"] == initial["snapshot_sha256"]
    schedule = readapt.sample_schedule(episodes, "M", 100)
    mesh = jax.sharding.Mesh(np.array(jax.devices("cpu")), ("batch",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("batch"))
    obs, actions = next(iter(readapt_train.make_loader(config, schedule, sharding)))
    assert actions.shape == (2, 15, 32)
    assert set(obs.images) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    assert all(np.asarray(v).all() for v in obs.image_masks.values())
    np.testing.assert_allclose(np.asarray(actions)[..., :7], 0.1, atol=2e-6)
    report = readapt.normalization_report(campaign)
    assert report["statistics_changed"] is False
    assert file_hash(stats_root / "norm_stats.json") == initial["norm_stats_sha256"]
    for e in episodes:
        readapt.verify_files(e)

    # Registry contracts use tiny inert bytes, never a model or hardware endpoint.
    snap = campaign / "snapshots/M/step_002000"
    snap.mkdir(parents=True)
    (snap / "weights.inert").write_bytes(b"TEST ONLY")
    record = dict(
        complete=True,
        cpu_roundtrip_pass=True,
        step=2000,
        manifest_sha256=m["sha256"],
        parent=initial,
        recipe=readapt.recipe("M", m["sha256"]),
        files_sha256={"weights.inert": file_hash(snap / "weights.inert")},
    )
    write_new_json(snap / "snapshot.json", record)
    with pytest.raises(FileNotFoundError):
        readapt_eval.register(campaign, snap, "fixture")
    ep = campaign / "evaluations/M_002000.json"
    evidence = dict(
        complete=True,
        gpu_reload_pass=True,
        snapshot_sha256=file_hash(snap / "snapshot.json"),
        manifest_sha256=m["sha256"],
        groups={
            c: [{"episode": e["id"]} for e in episodes if e["cohort"] == c and e["validation"]] for c in ("new", "old")
        },
    )
    write_new_json(ep, evidence)
    entry = readapt_eval.register(campaign, snap, "synthetic-test")
    assert entry["status"] == "SHADOW_ONLY" and entry["robot_motion_authorized"] is False
    cfg, _ = readapt_eval.registered_config(campaign, snap, campaign / "checkpoint_registry.json")
    assert cfg.policy_metadata["norm_stats_sha256"] == initial["norm_stats_sha256"]
    ep.write_text(json.dumps(dict(evidence, complete=False)))
    with pytest.raises(ContractError):
        readapt_eval.registered_config(campaign, snap, campaign / "checkpoint_registry.json")
