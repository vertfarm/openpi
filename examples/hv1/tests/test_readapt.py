"""CPU-only contracts; all robot/data fixtures are synthetic."""

from collections import Counter
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from examples.hv1 import native
from examples.hv1 import readapt
from examples.hv1.readapt_eval import registered_config
from examples.hv1.readapt_eval import snapshot_identity
from examples.hv1.readapt_eval import transition_metrics
from examples.hv1.readapt_train import FixedSampler
from examples.hv1.workflow import ContractError
from examples.hv1.workflow import write_new_json


def dataset():
    records = []
    for p in readapt.PLACEMENTS:
        for i in range(8):
            eid = f"episode_{len(records):06d}"
            records.append(
                dict(
                    id=readapt.uid("today", eid),
                    episode_id=eid,
                    session_id="today",
                    frames=180 + i * 3,
                    grasp_frame=35,
                    release_frame=100,
                    numeric_pass=True,
                    release_tail_s=(80 + i * 3) / 30,
                    created_at="2026-09-10T03:00:00+00:00",
                    cohort="new",
                    placement=p,
                )
            )
    for i in range(3):
        eid = f"episode_{40 + i:06d}"
        records.append(dict(id=readapt.uid("today", eid), episode_id=eid, numeric_pass=False))
    scan = readapt.sealed(dict(episodes=records))
    review = dict(
        scan_sha256=scan["sha256"],
        reviewer="fixture-only",
        configuration_applied_at="2026-09-10T02:00:00+00:00",
        collection_started_at="2026-09-10T02:10:00+00:00",
        runtime_parameters_evidence="fake",
        physical_contract_unchanged=True,
        owner_confirmed_unchanged_during_collection=True,
        episodes={
            e["id"]: dict(
                kind="success" if i < 40 else "diagnostic",
                placement=e.get("placement", ""),
                visual_success=True,
                no_intervention=True,
                no_jitter_or_collision=True,
                transitions_verified=True,
            )
            for i, e in enumerate(records)
        },
    )
    return scan, review


def mixed_episodes():
    scan, review = dataset()
    new = readapt.review_selection(scan, review)
    old = [
        dict(
            id=readapt.uid("yesterday", f"episode_{i:06d}"),
            cohort="old",
            validation=False,
            frames=100 + i * 4,
            grasp_frame=0,
            release_frame=99 + i * 4,
        )
        for i in range(19)
    ]
    old.append(
        dict(
            id="yesterday::episode_000027", cohort="old", validation=True, frames=100, grasp_frame=20, release_frame=70
        )
    )
    return new + old


def test_collection_schedule_balanced_numbered():
    rows = readapt.collection_schedule()
    assert [r["trial"] for r in rows] == list(range(1, 41))
    assert Counter(r["placement"] for r in rows) == Counter({p: 8 for p in readapt.PLACEMENTS})
    assert rows == readapt.collection_schedule()


def test_qualified_ids_do_not_drop_today_first_two():
    scan, review = dataset()
    selected = readapt.review_selection(scan, review)
    assert {"today::episode_000000", "today::episode_000001"} <= {e["id"] for e in selected}
    assert readapt.uid("today", "episode_000000") != readapt.uid("yesterday", "episode_000000")


def test_stratified_split_stable_and_disjoint():
    scan, review = dataset()
    rows = readapt.review_selection(scan, review)
    val = [e for e in rows if e["validation"]]
    assert len(val) == 8 and len(rows) - len(val) == 32
    assert set(e["placement"] for e in val) == set(readapt.PLACEMENTS)
    other = copy.deepcopy(scan)
    other["episodes"].reverse()
    assert rows == readapt.review_selection(other, review)


@pytest.mark.parametrize(
    "field",
    [
        "physical_contract_unchanged",
        "owner_confirmed_unchanged_during_collection",
        "runtime_parameters_evidence",
        "reviewer",
    ],
)
def test_missing_owner_evidence_blocks(field):
    scan, review = dataset()
    review[field] = False
    with pytest.raises(ContractError):
        readapt.review_selection(scan, review)


@pytest.mark.parametrize(
    "field", ["visual_success", "no_intervention", "no_jitter_or_collision", "transitions_verified"]
)
def test_missing_episode_review_blocks(field):
    scan, review = dataset()
    review["episodes"][scan["episodes"][0]["id"]][field] = False
    with pytest.raises(ContractError):
        readapt.review_selection(scan, review)


@pytest.mark.parametrize(
    "change",
    ["short_tail", "bad_numeric", "before_config", "wrong_placement", "not40", "not3diagnostics", "mismatched_scan"],
)
def test_bad_new_collection_blocks(change):
    scan, review = dataset()
    first = scan["episodes"][0]
    if change == "short_tail":
        first["release_tail_s"] = 1.0
    elif change == "bad_numeric":
        first["numeric_pass"] = False
    elif change == "before_config":
        first["created_at"] = "2026-09-09T02:00:00+00:00"
    elif change == "wrong_placement":
        review["episodes"][first["id"]]["placement"] = "unmarked"
    elif change == "not40":
        review["episodes"][first["id"]]["kind"] = "exclude"
    elif change == "not3diagnostics":
        review["episodes"][scan["episodes"][-1]["id"]]["kind"] = "exclude"
    else:
        review["scan_sha256"] = "bad"
    with pytest.raises(ContractError):
        readapt.review_selection(scan, review)


@pytest.mark.parametrize("name,new_count,old_count", [("N", 10000, 0), ("M", 8000, 2000)])
def test_sampling_exact_mixture_phase_no_holdout(name, new_count, old_count):
    episodes = mixed_episodes()
    s = readapt.sample_schedule(episodes, name)
    assert s == readapt.sample_schedule(episodes, name)
    assert Counter(e["cohort"] for e in s["records"]) == Counter(new=new_count, old=old_count)
    assert Counter(e["phase"] for e in s["records"]) == Counter(uniform=7000, close=1500, release=1500)
    holdout = {e["id"] for e in episodes if e["validation"]}
    byid = {e["id"]: e for e in episodes}
    assert not holdout & {e["episode"] for e in s["records"]}
    for r in s["records"]:
        e = byid[r["episode"]]
        assert 0 <= r["frame"] < e["frames"]
        if r["phase"] != "uniform":
            assert abs(r["frame"] - e["grasp_frame" if r["phase"] == "close" else "release_frame"]) <= 30
    for cohort in ("new", "old"):
        counts = Counter(e["episode"] for e in s["records"] if e["cohort"] == cohort)
        if counts:
            assert max(counts.values()) - min(counts.values()) <= 1


def test_sampler_global_offsets_and_resume_tail():
    episodes = mixed_episodes()
    s = readapt.sample_schedule(episodes, "M")
    lookup, offset = {}, 0
    for e in episodes:
        if e["validation"]:
            continue
        lookup[e["id"]] = offset
        offset += e["frames"]
    for r in s["records"]:
        assert r["index"] == lookup[r["episode"]] + r["frame"]
    complete = list(FixedSampler(s))
    for cursor in (0, 100, 4000, 10000):
        assert list(FixedSampler(s, cursor)) == complete[cursor:]
    with pytest.raises(ContractError):
        FixedSampler(s, 10001)
    assert readapt.coverage(s, 100)["cohort_counts"] == {"new": 80, "old": 20}


def test_N_M_same_parent_recipe_one_variable():
    n, m = readapt.recipe("N", "frozen"), readapt.recipe("M", "frozen")
    assert {k for k in n if n[k] != m[k]} == {"name", "new_fraction"}
    assert n["snapshots"] == [2000, 5000] and n["initialization"] == "C5000_new_optimizer"
    assert n["peak_lr"] == 1e-5 and n["ema_decay"] is None


def test_no_schedule_for_empty_old_pool_or_bad_count():
    rows = [e for e in mixed_episodes() if e["cohort"] == "new"]
    with pytest.raises(ContractError):
        readapt.sample_schedule(rows, "M")
    with pytest.raises(ContractError):
        readapt.sample_schedule(rows, "N", 99)


def test_manifest_integrity_and_source_protection(tmp_path):
    p = tmp_path / "manifest.json"
    write_new_json(p, readapt.sealed({"a": 1}))
    assert readapt.checked(p)["a"] == 1
    value = json.loads(p.read_text())
    value["a"] = 2
    p.write_text(json.dumps(value))
    with pytest.raises(ContractError):
        readapt.checked(p)
    with pytest.raises(ContractError):
        readapt.external_output(tmp_path / "datasets/x")
    with pytest.raises(ContractError):
        readapt.external_output(tmp_path / "source/x", tmp_path / "source")


def test_no_normalization_approval_by_default(tmp_path):
    report = readapt.sealed(dict(groups={}, statistics_changed=False))
    write_new_json(tmp_path / "normalization_report.json", report)
    review = dict(
        report_sha256=report["sha256"],
        reviewer="test",
        approved=False,
        units_sign_zero_mapping_confirmed=True,
        distribution_shift_reviewed=True,
    )
    write_new_json(tmp_path / "normalization_review.json", review)
    with pytest.raises(ContractError):
        readapt.require_normalization_review(tmp_path)
    review["approved"] = True
    (tmp_path / "normalization_review.json").write_text(json.dumps(review))
    readapt.require_normalization_review(tmp_path)


def test_transition_missing_and_repeated_release():
    truth = np.r_[np.zeros(30), np.ones(90), np.zeros(60)]
    good = transition_metrics(truth, truth, 30, 120)
    assert good["error_rate"] == 0 and good["close_time_error_s"] == 0 and good["extra_toggles"] == 0
    closed = transition_metrics(np.ones(180), truth, 30, 120)
    assert closed["missed_release"] and closed["initial_closed"]
    repeated = truth.copy()
    repeated[60:70] = 0
    bad = transition_metrics(repeated, truth, 30, 120)
    assert bad["extra_toggles"] == 2 and bad["early_release_frames"] == 10
    delayed = np.r_[np.zeros(36), np.ones(90), np.zeros(54)]
    assert transition_metrics(delayed, truth, 30, 120)["release_time_error_s"] == 0.2


def test_snapshot_cannot_use_incomplete_or_escaped_files(tmp_path):
    write_new_json(tmp_path / "snapshot.json", dict(complete=False))
    with pytest.raises(ContractError):
        snapshot_identity(tmp_path)
    (tmp_path / "snapshot.json").write_text(
        json.dumps(dict(complete=True, cpu_roundtrip_pass=True, files_sha256={"../escape": "bad"}))
    )
    with pytest.raises(ContractError):
        snapshot_identity(tmp_path)


def test_registry_path_gate_before_model_import(tmp_path):
    with pytest.raises(ContractError):
        registered_config(tmp_path, tmp_path / "snapshots/N/step_002000", tmp_path / "wrong.json")


def test_runtime_modules_have_no_robot_io():
    root = Path(readapt.__file__).parent
    for name in ("readapt.py", "readapt_train.py", "readapt_eval.py", "readapt_run.py"):
        text = (root / name).read_text()
        assert "import rclpy" not in text and "import paho" not in text
        assert "create_publisher(" not in text and "ActionClient(" not in text


def test_existing_parent_contract_unchanged():
    p = native.profile()
    assert len(p["state"]["names"]) == 15 and len(p["action"]["names"]) == 8
    assert p["action_horizon"] == 15 and p["gripper"]["mode"] == 2
    assert p["gripper"]["release_open"] == 0.6 and p["camera_dropout"] is False


def test_cleanup_only_completed_generated_states(tmp_path, monkeypatch):
    from examples.hv1.readapt_run import prune_completed
    from examples.hv1.workflow import file_hash

    campaign = tmp_path / "campaign"
    original = tmp_path / "original.bin"
    original.write_bytes(b"preserve")
    monkeypatch.setattr(readapt, "verify_campaign", lambda *args, **kwargs: {"sha256": "fixture"})
    write_new_json(
        campaign / "runs/N/result.json", dict(complete=True, step=5000, identity={"manifest_sha256": "fixture"})
    )
    target = campaign / "restarts/pi05_hv1/N/5000"
    target.mkdir(parents=True)
    (target / "inert").write_bytes(b"generated-test-state")
    with pytest.raises(FileNotFoundError):
        prune_completed(campaign, "N")
    assert target.exists()
    for step in (2000, 5000):
        snap = campaign / f"snapshots/N/step_{step:06d}/snapshot.json"
        write_new_json(snap, {"synthetic": True})
        write_new_json(
            campaign / f"evaluations/N_{step:06d}.json",
            dict(complete=True, gpu_reload_pass=True, snapshot_sha256=file_hash(snap)),
        )
    result = prune_completed(campaign, "N")
    assert result["complete"] and not target.parent.exists()
    assert original.read_bytes() == b"preserve"
    assert (campaign / "snapshots/N/step_005000/snapshot.json").exists()
    assert prune_completed(campaign, "N")["complete"]
    with pytest.raises(ContractError):
        prune_completed(campaign, "../parent")
    with pytest.raises(ContractError):
        prune_completed(campaign, "N", smoke_snapshot=True)
