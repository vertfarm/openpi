"""The TODAY30/ALL59 campaign contracts.

The campaign-neutral half of the retired readapt suite landed here on
2026-09-11: session-namespaced identity, deterministic sampling, sampler
resume, generated-state cleanup and the registry gate are properties of the
pipeline, not of any one experiment, and readapt was the only thing testing
them.
"""

from collections import Counter
import json
from pathlib import Path

import numpy as np
import pytest

from examples.hv1 import native
from examples.hv1 import pipeline
from examples.hv1 import pipeline_eval
from examples.hv1.artifacts import ContractError
from examples.hv1.artifacts import file_hash
from examples.hv1.artifacts import write_new_json
from examples.hv1.native import CAMERAS
from examples.hv1.pipeline_run import _safe_generated_cleanup
from examples.hv1.pipeline_train import FixedSampler


def manifest():
    episodes = []
    for cohort, session, ids in (
        ("old", pipeline.OLD_SESSION, sorted(pipeline.EXPECTED_OLD)),
        ("today", pipeline.TODAY_SESSION, sorted(pipeline.EXPECTED_TODAY)),
    ):
        for episode_id in ids:
            recovery = cohort == "today" and episode_id in {"episode_000002", "episode_000016"}
            # Same rule `pipeline._scan` applies, so the fixture has the six-and-six
            # diagnostic groups a real manifest carries.
            diagnostic = (
                "old_fixed6"
                if cohort == "old" and episode_id in pipeline.OLD_DIAGNOSTIC
                else "today_fixed6"
                if cohort == "today" and episode_id in pipeline.TODAY_DIAGNOSTIC
                else None
            )
            episodes.append(
                {
                    "id": pipeline.uid(session, episode_id),
                    "episode_id": episode_id,
                    "cohort": cohort,
                    "frames": 1000,
                    "grasp_frames": [200, 500] if recovery else [200],
                    "release_frames": [350, 800] if recovery else [800],
                    "training_tracks": ["ALL59"] + (["TODAY30"] if cohort == "today" else []),
                    "diagnostic_group": diagnostic,
                    "suspect": cohort == "old" and episode_id in pipeline.OLD_SUSPECT,
                }
            )
    return pipeline.sealed(
        {
            "schema": pipeline.SCHEMA,
            "episodes": episodes,
            "tracks": {
                "TODAY30": {"episode_ids": [e["id"] for e in episodes if e["cohort"] == "today"]},
                "ALL59": {"episode_ids": [e["id"] for e in episodes]},
            },
        }
    )


def test_recipes_start_independently_from_official_base():
    value = manifest()
    for track in pipeline.TRACKS:
        recipe = pipeline.recipe(track, value["sha256"])
        assert recipe["initialization"] == "official_pi05_base_new_optimizer"
        assert recipe["snapshots"] == [250, 500, 1000, 2000]
        assert recipe["batch_size"] == 2
        assert recipe["phase_fractions"] == {"uniform": 0.70, "close": 0.15, "release": 0.15}
        assert recipe["robot_motion_authorized"] is False


def test_track_membership_and_balanced_phase_schedule():
    value = manifest()
    today = pipeline.sample_schedule(value, "TODAY30", 4000)
    all_data = pipeline.sample_schedule(value, "ALL59", 4000)
    assert len(today["train_episode_ids"]) == 30
    assert all(pipeline.TODAY_SESSION in row["episode"] for row in today["records"])
    assert len(all_data["train_episode_ids"]) == 59
    assert Counter(row["phase"] for row in today["records"]) == {
        "uniform": 2800,
        "close": 600,
        "release": 600,
    }
    for schedule, episode_count in ((today, 30), (all_data, 59)):
        for phase in ("uniform", "close", "release"):
            counts = Counter(row["episode"] for row in schedule["records"] if row["phase"] == phase)
            assert len(counts) == episode_count
            assert max(counts.values()) - min(counts.values()) <= 1
        assert all(
            abs(row["frame"] - row["event_frame"]) <= 30
            for row in schedule["records"]
            if row["event_frame"] is not None
        )
        json.dumps(pipeline.coverage(schedule, len(schedule["records"])), allow_nan=False)


def test_recovery_episode_events_are_preserved_in_sampling():
    value = manifest()
    schedule = pipeline.sample_schedule(value, "TODAY30", 4000)
    for episode_id in ("episode_000002", "episode_000016"):
        uid = pipeline.uid(pipeline.TODAY_SESSION, episode_id)
        close = {row["event_frame"] for row in schedule["records"] if row["episode"] == uid and row["phase"] == "close"}
        release = {
            row["event_frame"] for row in schedule["records"] if row["episode"] == uid and row["phase"] == "release"
        }
        assert close == {200, 500}
        assert release == {350, 800}


def test_transition_metrics_match_multiple_cycles_without_reordering():
    truth = np.zeros(1000, dtype=np.float32)
    truth[200:350] = 1
    truth[500:800] = 1
    predicted = np.zeros_like(truth)
    predicted[205:345] = 1
    predicted[490:805] = 1
    result = pipeline_eval.transition_metrics(predicted, truth, [200, 500], [350, 800])
    assert result["matched_close"] == [[205, 200], [490, 500]]
    assert result["matched_release"] == [[345, 350], [805, 800]]
    assert result["missed_close"] == result["missed_release"] == 0
    assert result["extra_close"] == result["extra_release"] == 0


def test_condition_intents_filters_pulse_and_preserves_sustained_transition():
    raw = np.zeros(30, dtype=np.float32)
    raw[3:5] = 1  # 67 ms pulse at 30 Hz.
    raw[10:20] = 1
    filtered, events = pipeline_eval.condition_intents(raw, min_hold_s=0.2)
    assert events == [
        {"event": "close", "time_s": 16 / 30},
        {"event": "open", "time_s": 26 / 30},
    ]
    assert not filtered[:16].any()
    assert filtered[16:26].all()


def test_the_same_episode_id_in_two_sessions_never_collides():
    """Both sessions number from `episode_000000`, so an id that dropped the
    session would silently merge two different recordings."""
    shared = sorted(pipeline.EXPECTED_OLD & pipeline.EXPECTED_TODAY)
    assert len(shared) >= 20
    old = {pipeline.uid(pipeline.OLD_SESSION, e) for e in shared}
    today = {pipeline.uid(pipeline.TODAY_SESSION, e) for e in shared}
    assert not old & today
    value = manifest()
    ids = [episode["id"] for episode in value["episodes"]]
    assert len(ids) == len(set(ids)) == 59
    assert len({episode["episode_id"] for episode in value["episodes"]}) < 59
    for session, episode in (("", "episode_000000"), ("a::b", "episode_000000"), ("s", "000000")):
        with pytest.raises(ContractError):
            pipeline.uid(session, episode)


def test_the_same_manifest_always_yields_the_same_schedule():
    value = manifest()
    first = pipeline.sample_schedule(value, "TODAY30", 4000)
    assert first == pipeline.sample_schedule(value, "TODAY30", 4000)
    today = set(first["train_episode_ids"])
    everything = set(pipeline.sample_schedule(value, "ALL59", 4000)["train_episode_ids"])
    assert today < everything and len(everything) == 59


def test_sampler_offsets_follow_the_common_export_and_resume_at_the_cursor():
    """The sampler indexes the single common export, so an offset has to be the
    running frame total in manifest order - not the track's own order."""
    value = manifest()
    schedule = pipeline.sample_schedule(value, "TODAY30", 4000)
    offsets, running = {}, 0
    for episode in value["episodes"]:
        offsets[episode["id"]] = running
        running += episode["frames"]
    assert schedule["common_export_episode_order"] == [e["id"] for e in value["episodes"]]
    for row in schedule["records"]:
        assert row["index"] == offsets[row["episode"]] + row["frame"]
    complete = list(FixedSampler(schedule))
    assert len(complete) == 4000
    for cursor in (0, 100, 2000, 4000):
        assert list(FixedSampler(schedule, cursor)) == complete[cursor:]
    for outside in (-1, 4001):
        with pytest.raises(ContractError, match="sampler cursor"):
            FixedSampler(schedule, outside)
    consumed = pipeline.coverage(schedule, 100)
    assert consumed["consumed_samples"] == 100
    assert consumed["phase_counts"] == {"uniform": 70, "close": 15, "release": 15}
    assert sum(consumed["episode_counts"].values()) == 100


def test_the_pipelines_differ_only_in_the_data_they_are_given():
    value = manifest()
    today, everything = (pipeline.recipe(track, value["sha256"]) for track in pipeline.TRACKS)
    # Only the identity and the episode pool differ - the seed included, so the
    # two runs draw the same phase order and differ by data alone.
    assert {key for key in today if today[key] != everything[key]} == {"name", "track"}
    assert today["seed"] == everything["seed"]
    assert today["peak_lr"] == everything["peak_lr"]
    assert today["initialization"] == everything["initialization"]


@pytest.mark.parametrize(
    "name,samples,message",
    [
        ("TODAY30", 0, "multiple of 100"),
        ("TODAY30", 4050, "multiple of 100"),
        ("TOMORROW", 4000, "unknown two-track experiment"),
        ("ALL59", -100, "multiple of 100"),
    ],
)
def test_no_schedule_for_an_unknown_track_or_unaligned_sample_count(name, samples, message):
    with pytest.raises(ContractError, match=message):
        pipeline.sample_schedule(manifest(), name, samples)


def sessions(tmp_path, source_session, **damage):
    old = source_session(
        tmp_path / "old",
        pipeline.OLD_SESSION,
        pipeline.EXPECTED_OLD,
        declared=pipeline.EXPECTED_OLD | pipeline.OLD_EXCLUDED,
    )
    today = source_session(tmp_path / "today", pipeline.TODAY_SESSION, pipeline.EXPECTED_TODAY, **damage)
    return old, today


def test_prepare_seals_the_pipeline_inventory(tmp_path, source_session):
    old, today = sessions(tmp_path, source_session)
    value = pipeline.prepare(old, today, tmp_path / "campaign")
    assert value["schema"] == pipeline.SCHEMA and value["robot_motion_authorized"] is False
    assert len(value["episodes"]) == 59
    assert len(value["tracks"]["TODAY30"]["episode_ids"]) == 30
    assert len(value["tracks"]["ALL59"]["episode_ids"]) == 59
    assert len(value["diagnostics"]["old_fixed6"]) == 6
    assert len(value["diagnostics"]["today_fixed6"]) == 6
    assert sum(episode["suspect"] for episode in value["episodes"]) == len(pipeline.OLD_SUSPECT)
    assert pipeline.verify_campaign(tmp_path / "campaign", raw=True)["sha256"] == value["sha256"]
    with pytest.raises(ContractError, match="campaign must be a new directory"):
        pipeline.prepare(old, today, tmp_path / "campaign")
    scanned = Path(value["episodes"][0]["path"]) / "tasks.json"
    scanned.write_text(scanned.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ContractError, match="raw changed"):
        pipeline.verify_campaign(tmp_path / "campaign", raw=True)
    pipeline.verify_campaign(tmp_path / "campaign")


@pytest.mark.parametrize(
    "damage,message",
    [
        (dict(hz=25), "session identity/rate"),
        (dict(cameras=("head", "hand_r")), "three required cameras"),
        (dict(resolution=(1280, 720)), "camera geometry"),
        (dict(rotate=90), "camera geometry"),
    ],
)
def test_a_session_that_differs_from_the_training_contract_is_refused(tmp_path, source_session, damage, message):
    """The 2026-09-11 field session wasted an afternoon on a camera-geometry
    theory; this is the gate that would have to have failed for it to be true."""
    old, today = sessions(tmp_path, source_session, **damage)
    with pytest.raises(ContractError, match=message):
        pipeline.prepare(old, today, tmp_path / "campaign")


def test_a_session_under_a_different_name_is_refused(tmp_path, source_session):
    old = source_session(tmp_path / "old", "some_other_session", pipeline.EXPECTED_OLD)
    today = source_session(tmp_path / "today", pipeline.TODAY_SESSION, pipeline.EXPECTED_TODAY)
    with pytest.raises(ContractError, match="session identity/rate"):
        pipeline.prepare(old, today, tmp_path / "campaign")


def test_the_source_inventory_must_match_the_approved_plan(tmp_path, source_session):
    old = source_session(
        tmp_path / "old",
        pipeline.OLD_SESSION,
        pipeline.EXPECTED_OLD,
        declared=(pipeline.EXPECTED_OLD | pipeline.OLD_EXCLUDED) - {"episode_000005"},
    )
    today = source_session(tmp_path / "today", pipeline.TODAY_SESSION, pipeline.EXPECTED_TODAY)
    with pytest.raises(ContractError, match="approved 29\\+30 plan"):
        pipeline.prepare(old, today, tmp_path / "campaign")


@pytest.mark.parametrize("destination", ["inside_source", "datasets/campaign", "raw/campaign"])
def test_the_campaign_never_lands_on_immutable_source_custody(tmp_path, source_session, destination):
    old, today = sessions(tmp_path, source_session)
    output = today / "campaign" if destination == "inside_source" else tmp_path / destination
    with pytest.raises(ContractError, match="campaign"):
        pipeline.prepare(old, today, output)


@pytest.mark.parametrize("damage", ["tampered", "short_inventory", "track_count", "duplicate_ids"])
def test_verify_campaign_refuses_a_manifest_it_cannot_trust(tmp_path, damage):
    value = dict(manifest())
    if damage == "short_inventory":
        value["episodes"] = value["episodes"][:58]
    elif damage == "track_count":
        value["tracks"]["TODAY30"]["episode_ids"] = value["tracks"]["TODAY30"]["episode_ids"][:29]
    elif damage == "duplicate_ids":
        value["episodes"][1] = dict(value["episodes"][0])
    write_new_json(tmp_path / "manifest.json", value)
    if damage == "tampered":
        (tmp_path / "manifest.json").write_text(json.dumps(dict(value, target_steps=1)), encoding="utf-8")
    with pytest.raises(ContractError):
        pipeline.verify_campaign(tmp_path)


def fake_dataset(value):
    """The LeRobot row shape `_observation` reads, keyed by global frame index."""
    return {
        "observation.state": np.full(15, value, dtype=np.float32),
        **{f"observation.images.{camera}": np.full((4, 4, 3), value, dtype=np.uint8) for camera in CAMERAS},
    }


def test_cross_modal_anchors_on_each_episode_grasp():
    value = manifest()
    offsets, running = {}, 0
    for episode in value["episodes"]:
        offsets[episode["id"]] = running
        running += episode["frames"]
    dataset = {index: fake_dataset(index) for index in range(running)}

    cases = pipeline_eval._cross_modal_cases(value, dataset, offsets, "today_fixed6", 0)
    assert len(cases) == 6
    for case in cases:
        episode = next(e for e in value["episodes"] if e["id"] == case["name"])
        assert episode["diagnostic_group"] == "today_fixed6"
        # Anchored on the grasp, and the row actually fetched is that frame.
        assert case["frame"] == episode["grasp_frames"][0]
        assert case["state"][0] == offsets[case["name"]] + case["frame"]
        assert set(case["images"]) == set(CAMERAS)


def test_cross_modal_offset_shifts_the_anchor_and_stays_inside_the_episode():
    value = manifest()
    offsets = {}
    running = 0
    for episode in value["episodes"]:
        offsets[episode["id"]] = running
        running += episode["frames"]
    dataset = {index: fake_dataset(index) for index in range(running)}

    shifted = pipeline_eval._cross_modal_cases(value, dataset, offsets, "today_fixed6", 5)
    base = pipeline_eval._cross_modal_cases(value, dataset, offsets, "today_fixed6", 0)
    assert [c["frame"] for c in shifted] == [c["frame"] + 5 for c in base]
    # A wild offset must clamp rather than index another episode's frames.
    for offset, expected in ((10**6, 999), (-(10**6), 0)):
        clamped = pipeline_eval._cross_modal_cases(value, dataset, offsets, "today_fixed6", offset)
        assert {c["frame"] for c in clamped} == {expected}


def test_an_ablation_changes_exactly_one_thing_about_its_track():
    """The point of an ablation is the comparison. If it drifts from its track in
    any field but the one under test, the comparison stops meaning anything."""
    value = manifest()
    for name, spec in pipeline.ABLATIONS.items():
        base = pipeline.recipe(spec["track"], value["sha256"])
        ablation = pipeline.recipe(name, value["sha256"])
        changed = {key for key in base if base[key] != ablation.get(key)}
        # name and snapshots are identity and disk, not the variable under test.
        assert changed <= {"name", "snapshots", *spec} - {"track"}, (name, changed)
        assert ablation["track"] == base["track"] == spec["track"]
        assert ablation["ablation_of"] == spec["track"]
        assert ablation["seed"] == base["seed"] and ablation["peak_lr"] == base["peak_lr"]
        assert ablation["snapshots"] == [pipeline.TARGET_STEPS]
        assert ablation["robot_motion_authorized"] is False


def test_state_noise_moves_the_input_and_the_target_together():
    """It runs before HV1Inputs, which computes the action delta against the
    state. If the two were anchored differently the perturbation would become
    label noise the model cannot undo."""
    from examples.hv1.pipeline_train import StateNoise
    from examples.hv1.transforms import HV1Inputs

    profile = native.profile()
    state = np.arange(15, dtype=np.float32) / 10
    action = np.tile(np.arange(8, dtype=np.float32) / 10, (15, 1))
    images = {camera: np.zeros((224, 224, 3), np.uint8) for camera in CAMERAS}
    sample = {"state": state, "actions": action, "images": images}

    clean = HV1Inputs(profile)(dict(sample))
    noise = StateNoise(sigma=tuple([0.05] * 15), seed=42)
    noised = HV1Inputs(profile)(noise(dict(sample)))

    moved = np.asarray(noised["state"]) - np.asarray(clean["state"])
    assert np.any(moved != 0)
    # The arm delta absorbed exactly the shift its anchor took.
    np.testing.assert_allclose(
        np.asarray(noised["actions"])[:, :7],
        np.asarray(clean["actions"])[:, :7] - moved[:7],
        atol=1e-5,
    )
    # The grasp intent is absolute (delta index -1), so it must not move at all.
    np.testing.assert_allclose(np.asarray(noised["actions"])[:, 7], np.asarray(clean["actions"])[:, 7])


def test_state_noise_is_the_same_draw_for_the_same_frame():
    """A resume must not quietly train on a different dataset."""
    from examples.hv1.pipeline_train import StateNoise

    noise = StateNoise(sigma=tuple([0.05] * 15), seed=42)
    sample = {"state": np.arange(15, dtype=np.float32)}
    first, second = noise(dict(sample))["state"], noise(dict(sample))["state"]
    np.testing.assert_array_equal(first, second)
    other = noise({"state": np.arange(15, dtype=np.float32) + 1})["state"]
    assert not np.allclose(first, other - 1)
    assert StateNoise(sigma=tuple([0.05] * 15), seed=7)(dict(sample))["state"].tolist() != first.tolist()


def test_only_an_ablation_that_asked_for_noise_gets_it():
    from examples.hv1.pipeline_train import _noisy_data_config

    class Stats:
        std = np.full(15, 0.2)

    class Data:
        norm_stats = {"state": Stats()}
        data_transforms = None

    value = manifest()
    for name in ("TODAY30", "ALL59", "TODAY30_CLOSE45"):
        data = Data()
        plain, applied = _noisy_data_config(data, pipeline.recipe(name, value["sha256"]))
        assert applied is None, f"{name} did not ask for noise"
        assert plain is data, f"{name}'s data config must be handed back untouched"


def test_the_deployment_config_never_carries_state_noise():
    """`configure` builds the config deployment shares. Noise belongs only to the
    copy `make_loader` makes, or live inference would run on corrupted state."""
    from examples.hv1 import pipeline_config
    from examples.hv1 import pipeline_train

    source = Path(pipeline_config.__file__).read_text(encoding="utf-8")
    assert "StateNoise" not in source and "state_noise" not in source
    trainer = Path(pipeline_train.__file__).read_text(encoding="utf-8")
    assert "StateNoise" in trainer and "_noisy_data_config" in trainer
    # deploy_server reaches the config through registered_config, never make_loader.
    server = Path(Path(pipeline_config.__file__).with_name("deploy_server.py")).read_text(encoding="utf-8")
    assert "make_loader" not in server and "StateNoise" not in server


def test_an_ablation_cannot_invent_a_recipe_field(monkeypatch):
    monkeypatch.setitem(pipeline.ABLATIONS, "BAD", dict(track="TODAY30", typo_lr=1.0))
    with pytest.raises(ContractError, match="unknown recipe field"):
        pipeline.recipe("BAD", manifest()["sha256"])


def test_the_deployed_tracks_keep_the_recipe_their_snapshots_were_written_with():
    """A snapshot stores its recipe and `configure` re-derives and compares it, so
    editing a track's recipe makes deploy_server refuse the running checkpoint."""
    value = manifest()
    for track in pipeline.TRACKS:
        assert pipeline.recipe(track, value["sha256"])["phase_fractions"] == {
            "uniform": 0.70,
            "close": 0.15,
            "release": 0.15,
        }
        assert pipeline.recipe(track, value["sha256"])["snapshots"] == [250, 500, 1000, 2000]


def test_an_ablation_schedule_follows_its_own_phase_weights():
    value = manifest()
    schedule = pipeline.sample_schedule(value, "TODAY30_CLOSE45", 4000)
    assert Counter(row["phase"] for row in schedule["records"]) == {
        "close": 1800,
        "uniform": 1600,
        "release": 600,
    }
    assert schedule["track"] == "TODAY30"
    assert len(schedule["train_episode_ids"]) == 30


def test_the_default_anchor_clears_the_measured_intent_latency():
    """TODAY30-1000 raises intent +1..+7 frames after the recorded grasp, so an
    anchor at +0 compares two near-zero numbers. The default must be past that."""
    assert pipeline_eval.GRASP_ANCHOR_OFFSET > 7


def test_cross_modal_refuses_a_group_it_cannot_cross():
    value = manifest()
    with pytest.raises(ContractError, match="at least two diagnostic episodes"):
        pipeline_eval._cross_modal_cases(value, {}, {}, "no_such_group", 0)


def test_the_registry_path_is_checked_before_any_model_is_loaded(tmp_path):
    """`registered_config` builds a training config; the path gate has to reject
    first, or a wrong argument reaches the weight loader before it is refused."""
    with pytest.raises(ContractError, match="registry/snapshot outside campaign"):
        pipeline_eval.registered_config(tmp_path, tmp_path / "snapshots/TODAY30/step_001000", tmp_path / "wrong.json")
    with pytest.raises(ContractError, match="registry/snapshot outside campaign"):
        pipeline_eval.registered_config(
            tmp_path, tmp_path / "elsewhere/step_001000", tmp_path / "checkpoint_registry.json"
        )


def test_cleanup_removes_only_generated_state_and_journals_it(tmp_path):
    campaign = tmp_path / "campaign"
    evidence = campaign / "evaluations/TODAY30_001000.json"
    write_new_json(evidence, dict(complete=True, gpu_reload_pass=True))
    target = campaign / "restarts/pi05_hv1/TODAY30/1000"
    target.mkdir(parents=True)
    (target / "inert").write_bytes(b"generated-optimizer-state")
    keep = tmp_path / "original.bin"
    keep.write_bytes(b"preserve")

    relative = Path("restarts/pi05_hv1/TODAY30/1000")
    record = _safe_generated_cleanup(campaign, relative, evidence, "snapshot evaluated")
    assert record["complete"] and record["recoverable"] is False
    assert record["evidence_sha256"] == file_hash(evidence)
    assert not target.exists() and keep.read_bytes() == b"preserve"
    # Idempotent: the journal is the evidence that the removal already happened.
    assert _safe_generated_cleanup(campaign, relative, evidence, "again")["complete"]

    with pytest.raises(ContractError, match="escapes the generated campaign subtree"):
        _safe_generated_cleanup(campaign, Path("../parent"), evidence, "escape")
    with pytest.raises(ContractError, match="cleanup target missing without an audit journal"):
        _safe_generated_cleanup(campaign, Path("restarts/never"), evidence, "absent")
    linked = campaign / "restarts/linked"
    linked.mkdir(parents=True)
    (linked / "outside").symlink_to(keep)
    with pytest.raises(ContractError, match="filesystem link"):
        _safe_generated_cleanup(campaign, Path("restarts/linked"), evidence, "link")
    assert keep.read_bytes() == b"preserve"


def test_filter_finetune_recipe_is_parented_and_low_rate():
    recipe = pipeline.recipe("TODAY30_FT", "manifest", 1000)
    assert recipe["track"] == "TODAY30"
    assert recipe["parent"] == {
        "experiment": "TODAY30",
        "step": 1000,
        "snapshot": "snapshots/TODAY30/step_001000",
    }
    assert recipe["peak_lr"] == 2.5e-6
    assert recipe["warmup_steps"] == 50
    assert recipe["snapshots"] == [500, 1000]
    assert recipe["initialization"] == "BF16_parent_weights_FP32_training_new_optimizer"
