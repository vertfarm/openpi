"""No ROS or GPU needed; fault-injection tests for the deployment boundary."""

import base64
from http.server import ThreadingHTTPServer
from pathlib import Path
import threading

import numpy as np
import pytest

from examples.hv1.artifacts import ContractError
from examples.hv1.deploy_server import REGISTRY_SCHEMA
from examples.hv1.deploy_server import RecordedEpisode
from examples.hv1.deploy_server import make_handler
from examples.hv1.deploy_server import training_hand_range
from examples.hv1.ros.keti_humanoid_inference.keti_humanoid_inference.core import ARM_STATE
from examples.hv1.ros.keti_humanoid_inference.keti_humanoid_inference.core import CAMERAS
from examples.hv1.ros.keti_humanoid_inference.keti_humanoid_inference.core import CONTRACT
from examples.hv1.ros.keti_humanoid_inference.keti_humanoid_inference.core import CONTRACT_SHA
from examples.hv1.ros.keti_humanoid_inference.keti_humanoid_inference.core import CORE_SOURCE_SHA256
from examples.hv1.ros.keti_humanoid_inference.keti_humanoid_inference.core import HAND_STATE
from examples.hv1.ros.keti_humanoid_inference.keti_humanoid_inference.core import PROMPT
from examples.hv1.ros.keti_humanoid_inference.keti_humanoid_inference.core import ChunkQueue
from examples.hv1.ros.keti_humanoid_inference.keti_humanoid_inference.core import GripEdges
from examples.hv1.ros.keti_humanoid_inference.keti_humanoid_inference.core import IntentConditioner
from examples.hv1.ros.keti_humanoid_inference.keti_humanoid_inference.core import LiveGate
from examples.hv1.ros.keti_humanoid_inference.keti_humanoid_inference.core import Observations
from examples.hv1.ros.keti_humanoid_inference.keti_humanoid_inference.core import PolicyHTTP
from examples.hv1.ros.keti_humanoid_inference.keti_humanoid_inference.core import Rejected
from examples.hv1.ros.keti_humanoid_inference.keti_humanoid_inference.core import make_request
from examples.hv1.ros.keti_humanoid_inference.keti_humanoid_inference.core import named_positions
from examples.hv1.ros.keti_humanoid_inference.keti_humanoid_inference.core import validate_hand_envelope
from examples.hv1.ros.keti_humanoid_inference.keti_humanoid_inference.core import validate_metadata


def metadata():
    return dict(
        ready=True,
        contract=CONTRACT,
        contract_sha256=CONTRACT_SHA,
        action_horizon=15,
        denoise=10,
        snapshot_sha256="test-snapshot",
        adapter_core_sha256=CORE_SOURCE_SHA256,
        hand_state_envelope={
            "names": list(HAND_STATE),
            "basis": "training_min_max",
            "low": [-1.0] * 8,
            "high": [1.0] * 8,
            "margin_rad": 0.05,
        },
    )


def field_profile():
    # Synthetic test values only. Never published as a hardware profile.
    return dict(
        approved=True,
        review_id="TEST-ONLY",
        contract_sha256=CONTRACT_SHA,
        joint_min=[-2.0] * 7,
        joint_max=[2.0] * 7,
        max_step=[0.1] * 7,
        max_tracking_error=[0.2] * 7,
        max_velocity=[2.0] * 7,
        max_acceleration=[100.0] * 7,
        start_q=[0.0] * 7,
        start_tolerance=[0.1] * 7,
        stop_service="/fake/stop",
        guardian_node="/fake",
        snapshot_sha256="test",
        qualification_evidence="synthetic",
    )


def status():
    return dict(
        exclusive=True,
        workspace_clear=True,
        stop_ready=True,
        hardware_watchdog_ready=True,
        review_id="TEST-ONLY",
        grasp_mode=2,
        mqtt_age_s=0.01,
        start_pose_verified=True,
        narrow_open_verified=True,
        approved_proposals=["test"],
    )


def test_the_registry_gate_names_the_same_schema_the_pipeline_writes():
    """deploy_server holds the literal so it can refuse a registry before the
    training stack is imported. That is the only copy, and this is its leash."""
    from examples.hv1 import pipeline

    assert REGISTRY_SCHEMA == pipeline.SCHEMA


def test_named_state_roundtrip():
    names = list(ARM_STATE)
    values = np.arange(7)
    np.testing.assert_array_equal(named_positions(names[::-1], values[::-1], names), values)


@pytest.mark.parametrize("names,values", [(["x", "x"], [1, 2]), (["x"], [float("nan")]), (["x"], [])])
def test_reject_bad_joint_names(names, values):
    with pytest.raises(Rejected):
        named_positions(names, values, ARM_STATE)


def test_metadata_and_three_cameras():
    validate_metadata(metadata())
    bad = metadata()
    bad["contract_sha256"] = "wrong"
    with pytest.raises(Rejected):
        validate_metadata(bad)
    with pytest.raises(Rejected):
        make_request(0, np.zeros(15), {"head": b"x"})


def test_metadata_rejects_runtime_code_mismatch():
    bad = metadata()
    bad["adapter_core_sha256"] = "different"
    with pytest.raises(Rejected, match="source mismatch"):
        validate_metadata(bad)


def test_hand_envelope_allows_prepared_pose_but_rejects_home():
    value = metadata()
    value["hand_state_envelope"] = {
        "names": list(HAND_STATE),
        "basis": "training_min_max",
        "low": [1.549, -0.618, 0.828, -0.618, 0.828, 1.550, -0.617, 0.829],
        "high": [1.577, -0.607, 0.835, -0.605, 0.835, 1.554, -0.609, 0.832],
        "margin_rad": 0.05,
    }
    prepared = [1.560, -0.612, 0.831, -0.612, 0.831, 1.543, -0.612, 0.831]
    np.testing.assert_allclose(validate_hand_envelope(prepared, value), prepared)
    with pytest.raises(Rejected, match="joint_10.*joint_30"):
        validate_hand_envelope(np.zeros(8), value)
    with pytest.raises(Rejected):
        validate_hand_envelope([float("nan")] * 8, value)


def observation():
    obs = Observations()
    for key in ("arm", "hand") + CAMERAS:
        value = b"jpg" if key in CAMERAS else np.zeros(7 if key == "arm" else 8)
        obs.put(key, value, 10.0, 0.01)
    return obs


def test_freshness_both_header_and_receipt():
    obs = observation()
    state, images, anchor, ages = obs.snapshot(10.02)
    assert state.shape == (15,) and len(images) == 3 and anchor == 9.99
    with pytest.raises(Rejected):
        obs.snapshot(10.2)
    obs.put("head", b"x", 10.02, 2.0)
    with pytest.raises(Rejected):
        obs.snapshot(10.02)


def test_missing_and_future_frames():
    with pytest.raises(Rejected):
        Observations().snapshot(1)
    with pytest.raises(Rejected):
        Observations().put("arm", np.zeros(7), 1, -0.2)


def test_align_asynchronous_fresh_sources_without_relaxing_limits():
    obs = Observations()
    for key in ("arm", "hand", "head", "hand_l"):
        value = b"old" if key in CAMERAS else np.zeros(7 if key == "arm" else 8)
        obs.put(key, value, 9.94, 0.0)
        value = b"new" if key in CAMERAS else np.ones(7 if key == "arm" else 8)
        obs.put(key, value, 9.99, 0.0)
    obs.put("hand_r", b"slow", 9.88, 0.0)
    state, images, anchor, ages = obs.snapshot(10.0)
    assert anchor == 9.94
    assert np.all(state == 0) and images["head"] == b"old"
    assert max(ages.values()) - min(ages.values()) <= 0.1
    assert ages["arm"] <= 0.1 and ages["hand_r"] <= 0.15
    with pytest.raises(Rejected):
        obs.snapshot(10.04)  # a stale camera is NOT rescued by buffering


def test_chunk_skips_expired_prefix():
    queue = ChunkQueue()
    a = np.arange(120).reshape(15, 8)
    info = queue.offer(a, 10.0, 10.065)
    assert info["first_model_index"] == 3
    assert queue.pop(10.09) is None
    _, target = queue.pop(10.100001)
    np.testing.assert_array_equal(target, a[3])
    assert len(queue.queue) == 2


def test_prefetch_and_no_catchup():
    queue = ChunkQueue()
    a = np.zeros((15, 8))
    queue.offer(a, 10.0, 10.06)
    with pytest.raises(Rejected):
        queue.pop(10.2)
    assert not queue.queue
    with pytest.raises(Rejected):
        queue.offer(a, 10.0, 10.3)
    with pytest.raises(Rejected):
        queue.offer(np.zeros((15, 7)), 10.0, 10.06)


def test_no_command_burst():
    queue = ChunkQueue()
    queue.queue.extend([(1.0, np.zeros(8)), (1.001, np.zeros(8))])
    queue.pop(1.0)
    with pytest.raises(Rejected):
        queue.pop(1.001)


def test_gripper_edges_not_joint_angles_or_torque():
    edges = GripEdges(min_hold_s=0)
    assert edges.update(-0.1) is None
    assert edges.update(1.2) == "close"
    assert edges.update(0.8) is None
    assert edges.update(0.1) == "open"
    assert edges.update(0.0) is None
    with pytest.raises(Rejected):
        edges.update(0.9)
    with pytest.raises(Rejected):
        GripEdges(min_hold_s=0).update(float("nan"))


def test_gripper_hysteresis_and_dwell_reject_short_pulses():
    conditioned = IntentConditioner(close_threshold=0.7, open_threshold=0.3, min_hold_s=0.2)
    assert conditioned.update(0.8, 0.0) is None
    assert conditioned.update(0.1, 0.1) is None
    assert conditioned.update(0.8, 0.2) is None
    assert conditioned.update(0.9, 0.39) is None
    assert conditioned.update(0.9, 0.4) == "close"
    assert conditioned.update(0.5, 0.5) is None
    assert conditioned.update(0.2, 0.6) is None
    assert conditioned.update(0.2, 0.8) == "open"


def test_unverified_live_profile_cannot_arm():
    with pytest.raises(Rejected):
        LiveGate({"approved": False})
    p = field_profile()
    p["stop_service"] = ""
    with pytest.raises(Rejected):
        LiveGate(p)


@pytest.mark.parametrize("key", ["exclusive", "workspace_clear", "stop_ready", "hardware_watchdog_ready"])
def test_guardian_denial(key):
    gate = LiveGate(field_profile())
    s = status()
    s[key] = False
    with pytest.raises(Rejected):
        gate.arm(np.zeros(7), s, 1.0, 1.01)


def test_expired_lease_or_mqtt():
    gate = LiveGate(field_profile())
    with pytest.raises(Rejected):
        gate.arm(np.zeros(7), status(), 1.0, 1.2)
    s = status()
    s["mqtt_age_s"] = 1.0
    with pytest.raises(Rejected):
        gate.arm(np.zeros(7), s, 1.0, 1.01)


def test_target_and_tracking_limits():
    gate = LiveGate(field_profile())
    gate.arm(np.zeros(7), status(), 1.0, 1.01)
    gate.check(np.ones(7) * 0.01, np.zeros(7), status(), 1.04, 1.05, proposal_id="test")
    with pytest.raises(Rejected):
        gate.check(np.ones(7), np.zeros(7), status(), 1.07, 1.08, proposal_id="test")
    gate.disarm()
    with pytest.raises(Rejected):
        gate.check(np.zeros(7), np.zeros(7), status(), 1.07, 1.08, proposal_id="test")


def test_velocity_acceleration_not_just_joint_range():
    p = field_profile()
    p["max_velocity"] = [0.01] * 7
    gate = LiveGate(p)
    gate.arm(np.zeros(7), status(), 1.0, 1.01)
    with pytest.raises(Rejected):
        gate.check(np.ones(7) * 0.01, np.zeros(7), status(), 1.04, 1.05, proposal_id="test")


def test_each_proposed_trajectory_needs_approval():
    gate = LiveGate(field_profile())
    gate.arm(np.zeros(7), status(), 1.0, 1.01)
    with pytest.raises(Rejected, match="proposed trajectory"):
        gate.check(np.zeros(7), np.zeros(7), status(), 1.04, 1.05, proposal_id="different")


def test_target_trace_retains_request_and_model_index():
    q = ChunkQueue()
    entry = q.offer(np.zeros((15, 8)), 10.0, 10.065, sequence=7)
    q.pop(entry["apply_at"])
    assert q.last_meta["sequence"] == 7
    assert q.last_meta["model_index"] == 3
    assert q.last_meta["proposal_id"] == entry["proposal_id"]


def test_loopback_http_actual_wire(monkeypatch):
    # Mock only image decode/model; exercise real HTTP serialization, identities,
    # bounded client and action dimensionality.
    import examples.hv1.deploy_server as server_module

    monkeypatch.setattr(
        server_module, "prepare_images", lambda p: {c: np.zeros((224, 224, 3), np.uint8) for c in CAMERAS}
    )

    class Model:
        def infer(self, obs):
            assert obs["state"].shape == (15,) and obs["prompt"] == PROMPT
            return {"actions": np.tile(np.r_[obs["state"][:7], 1.0], (15, 1))}

    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(Model(), metadata()))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = PolicyHTTP(server.server_port)
        assert client.metadata()["ready"]
        state = np.r_[np.arange(7), np.zeros(8)]
        request = make_request(9, state, {c: b"test" for c in CAMERAS})
        actions, _ = client.infer(request, "test-snapshot")
        np.testing.assert_array_equal(actions[0], np.r_[np.arange(7), 1.0])
        request["contract_sha256"] = "wrong"
        with pytest.raises(Exception):
            client.infer(request, "test-snapshot")
    finally:
        server.shutdown()
        server.server_close()


def test_live_image_matches_recorder_geometry_and_rgb():
    cv2 = pytest.importorskip("cv2")
    tools = pytest.importorskip("openpi_client.image_tools")
    from examples.hv1.deploy_server import prepare_images

    bgr = np.zeros((720, 1280, 3), dtype=np.uint8)
    bgr[..., 2] = 200
    bgr[200:500, 400:900, 0] = 91
    ok, encoded = cv2.imencode(".png", bgr)
    assert ok
    payload = {
        "image_encoding": "ros_compressed",
        "images": {c: base64.b64encode(encoded.tobytes()).decode() for c in CAMERAS},
    }
    actual = prepare_images(payload)
    recorder = cv2.resize(bgr, (640, 480), interpolation=cv2.INTER_AREA)
    expected = tools.resize_with_pad(cv2.cvtColor(recorder, cv2.COLOR_BGR2RGB), 224, 224)
    for image in actual.values():
        np.testing.assert_array_equal(image, expected)
        assert image.shape == (224, 224, 3) and image.dtype == np.uint8
        assert image[28, 0, 0] == 200 and image[0].max() == 0


def test_grasp_intent_is_the_newest_prediction_not_a_blend():
    """G3: arm joints are averaged over every prediction of an instant, but the
    binary grasp intent is not - a blend of an old "open" and a new "close" sat
    near 0.5 and delayed the close past the dwell."""
    queue = ChunkQueue(ensemble_decay=0.0)
    old, new = np.zeros((15, 8)), np.ones((15, 8))
    old[:, 7], new[:, 7] = 0.0, 1.0
    queue.offer(old, 10.0, 10.0)
    queue.offer(new, 10.1, 10.1)
    for _ in range(3):
        queue.pop(queue.queue[0][0])
    _, action = queue.pop(queue.queue[0][0])
    np.testing.assert_allclose(action[:7], 0.5)
    assert action[7] == 1.0


def test_training_hand_range_spans_every_training_frame():
    """G4: the gate admits everything the policy was trained on. Percentiles cut
    the grasp plateau's tails and the gate refused normal grasps."""
    states = [np.zeros((3, 15)), np.ones((2, 15))]
    states[0][1, 7:] = -0.3
    states[1][0, 14] = 1.7
    low, high = training_hand_range(states)
    np.testing.assert_allclose(low, [-0.3] * 8)
    np.testing.assert_allclose(high, [1.0] * 7 + [1.7])
    envelope = {
        "names": list(HAND_STATE),
        "basis": "training_min_max",
        "low": low.tolist(),
        "high": high.tolist(),
        "margin_rad": 0.05,
    }
    value = {**metadata(), "hand_state_envelope": envelope}
    for frame in np.concatenate(states):
        validate_hand_envelope(frame[7:], value)
    with pytest.raises(Rejected, match="joint_32"):
        validate_hand_envelope([1.0] * 7 + [1.76], value)
    with pytest.raises(ContractError):
        training_hand_range([])


def test_metadata_requires_the_min_max_envelope_basis():
    bad = metadata()
    bad["hand_state_envelope"] = {**bad["hand_state_envelope"], "basis": "q01_q99"}
    with pytest.raises(Rejected, match="training envelope"):
        validate_metadata(bad)


def test_recorded_episode_aligns_rows_to_the_observation_anchor():
    """Open-loop replay serves the recording's actions through the same server,
    queue and gates as a model. Sample i of a chunk is the recording row at
    anchor + i/30 on the recording's own clock, as the queue reads a model chunk."""
    actions = np.arange(40 * 8, dtype=np.float32).reshape(40, 8)
    times = np.arange(40) / 30 + 100.0
    times[20:] += 0.05  # a recorder gap: rows after it are 0.05 s later
    episode = RecordedEpisode(actions, times)
    episode.RESTART_GAP_S = 10.0  # this test jumps ahead; restarts are tested below
    np.testing.assert_array_equal(episode.infer({"anchor": 7.0})["actions"], actions[:15])
    np.testing.assert_array_equal(episode.infer({"anchor": 7.1})["actions"][0], actions[3])
    late = episode.infer({"anchor": 7.0 + 25 / 30 + 0.05})["actions"]
    np.testing.assert_array_equal(late[0], actions[25])
    tail = episode.infer({"anchor": 8.4})["actions"]
    assert np.all(tail == actions[39])


@pytest.mark.parametrize(
    "second",
    [{"anchor": 9.0, "run": None}, {"anchor": 7.2, "run": 7.15}, {"anchor": 6.0, "run": None}],
    ids=["pause", "armed run", "backwards"],
)
def test_recorded_episode_restarts_from_row_zero_on_a_new_run(second):
    """Shadow requests must not use up the recording before ARM: a new armed
    run, a pause or a restarted node starts again at row 0."""
    actions = np.arange(40 * 8, dtype=np.float32).reshape(40, 8)
    episode = RecordedEpisode(actions, np.arange(40) / 30)
    episode.infer({"anchor": 7.0, "run": None})
    episode.infer({"anchor": 7.1, "run": None})
    np.testing.assert_array_equal(episode.infer(second)["actions"], actions[:15])


def test_recorded_episode_needs_the_anchor():
    episode = RecordedEpisode(np.zeros((5, 8)), np.arange(5) / 30)
    with pytest.raises(Rejected, match="anchor"):
        episode.infer({})
    with pytest.raises(Rejected, match="timestamps"):
        RecordedEpisode(np.zeros((5, 8)), np.zeros(5))


def test_request_carries_anchor_and_run():
    images = {c: b"jpg" for c in CAMERAS}
    request = make_request(3, np.zeros(15), images, 12.5, 11.0)
    assert request["anchor"] == 12.5 and request["run"] == 11.0
    assert make_request(3, np.zeros(15), images)["anchor"] is None
    with pytest.raises(Rejected):
        make_request(3, np.zeros(15), images, float("nan"))


def test_grasp_intent_takes_the_newest_of_three_predictions():
    queue = ChunkQueue(ensemble_decay=0.0)
    for anchor, intent in ((10.0, 0.0), (10.1, 1.0), (10.2, 0.0)):
        chunk = np.zeros((15, 8))
        chunk[:, 7] = intent
        queue.offer(chunk, anchor, anchor)
    while len(queue.queue) > 3:
        queue.pop(queue.queue[0][0])
    _, action = queue.pop(queue.queue[0][0])
    assert action[7] == 0.0


def fake_campaign(monkeypatch, tmp_path, hands, *, tamper=None):
    """A campaign whose episodes are tiny recordings with the given hand states."""
    from examples.hv1.artifacts import file_hash
    import examples.hv1.native as native
    import examples.hv1.pipeline as pipeline

    episodes = []
    for index, hand in enumerate(hands):
        directory = tmp_path / f"ep{index}"
        directory.mkdir()
        (directory / "data.hdf5").write_bytes(f"recording {index}".encode())
        episodes.append(
            {
                "id": f"ep{index}",
                "path": str(directory),
                "source_hashes": {"data.hdf5": file_hash(directory / "data.hdf5")},
            }
        )
    if tamper is not None:
        (Path(episodes[tamper]["path"]) / "data.hdf5").write_bytes(b"edited")
    by_path = {str(Path(e["path"]) / "data.hdf5"): np.asarray(h, dtype=np.float64) for e, h in zip(episodes, hands)}

    def read_numeric(path, allow_multiple_cycles=False):
        hand = by_path[str(path)]
        state = np.concatenate((np.zeros((len(hand), 7)), hand), axis=1)
        action = np.zeros((len(hand), 8))
        return state, action, {"row_timestamp_seconds": (np.arange(len(hand)) / 30).tolist()}

    monkeypatch.setattr(pipeline, "verify_campaign", lambda campaign, raw=False: {"episodes": episodes})
    monkeypatch.setattr(native, "read_numeric", read_numeric)
    return episodes


def test_hand_envelope_reads_only_the_named_verified_recordings(monkeypatch, tmp_path):
    from examples.hv1.deploy_server import hand_envelope

    fake_campaign(monkeypatch, tmp_path, [np.zeros((2, 8)), np.ones((2, 8)), np.full((2, 8), 5.0)])
    envelope = hand_envelope(tmp_path, {"ep0", "ep1"})
    assert envelope["basis"] == "training_min_max"
    assert envelope["low"] == [0.0] * 8 and envelope["high"] == [1.0] * 8
    with pytest.raises(Rejected, match="not all in the campaign"):
        hand_envelope(tmp_path, {"ep0", "missing"})


def test_hand_envelope_refuses_an_edited_recording(monkeypatch, tmp_path):
    from examples.hv1.deploy_server import hand_envelope

    fake_campaign(monkeypatch, tmp_path, [np.zeros((2, 8)), np.ones((2, 8))], tamper=1)
    with pytest.raises(Rejected, match="differs from the campaign record"):
        hand_envelope(tmp_path, {"ep0", "ep1"})


def test_load_replay_binds_the_served_rows_to_the_recording_hash(monkeypatch, tmp_path):
    from examples.hv1.deploy_server import load_replay

    episodes = fake_campaign(monkeypatch, tmp_path, [np.zeros((3, 8)), np.ones((3, 8))])
    policy, meta = load_replay(tmp_path, "ep1")
    assert meta["source"] == "recorded_episode" and meta["denoise"] is None
    assert meta["snapshot_sha256"] == episodes[1]["source_hashes"]["data.hdf5"]
    validate_metadata(meta)
    assert policy.infer({"anchor": 1.0})["actions"].shape == (15, 8)
    with pytest.raises(Rejected, match="not in the campaign"):
        load_replay(tmp_path, "ep9")


def test_replay_and_registry_are_exclusive(monkeypatch):
    import sys

    import examples.hv1.deploy_server as server_module

    monkeypatch.setattr(sys, "argv", ["deploy_server", "--campaign", "c", "--replay-episode", "e", "--registry", "r"])
    with pytest.raises(SystemExit):
        server_module.main()
    monkeypatch.setattr(sys, "argv", ["deploy_server", "--campaign", "c", "--snapshot", "s"])
    with pytest.raises(SystemExit):
        server_module.main()


def test_metadata_accepts_a_recorded_episode_source_only_as_such():
    replay = {**metadata(), "denoise": None, "source": "recorded_episode"}
    validate_metadata(replay)
    with pytest.raises(Rejected, match="sampling"):
        validate_metadata({**metadata(), "denoise": None})


GRASP_Q = np.array([0.4, -0.1, 0.2, 0.9, 0.05, 0.3, 0.2])
WIDE = ([-1.5] * 7, [1.5] * 7, "profile-sha")


def test_arm_remap_serves_an_old_arm_policy_on_the_new_arm(arm_urdf):
    """The policy sees the old-arm angles for the robot's tool pose, and its
    old-arm targets go out as new-arm angles for the same tool pose."""
    from examples.hv1.deploy_server import ArmRemap
    from examples.hv1.kinematics import Chain

    old, new = arm_urdf("old"), arm_urdf("new", elbow_x=0.03, wrist_x=-0.03)
    remap = ArmRemap(old, new, WIDE)
    old_chain, new_chain = Chain.from_urdf(old, "Right_tool0"), Chain.from_urdf(new, "Right_tool0")
    q_policy = remap.state(GRASP_Q)
    np.testing.assert_allclose(old_chain.fk(q_policy)[1], new_chain.fk(GRASP_Q)[1], atol=1e-5)
    chunk = np.c_[q_policy + np.linspace(0, 0.05, 15)[:, None], np.r_[np.zeros(7), np.ones(8)]]
    sent = remap.actions(chunk, GRASP_Q, q_policy)
    for target, commanded in zip(chunk, sent):
        np.testing.assert_allclose(new_chain.fk(commanded[:7])[1], old_chain.fk(target[:7])[1], atol=1e-5)
    np.testing.assert_array_equal(sent[:, 7], chunk[:, 7])
    assert remap.identity["robot_limits_profile_sha256"] == "profile-sha"
    assert remap.identity["trained_urdf_sha256"] != remap.identity["robot_urdf_sha256"]


def test_arm_remap_refuses_targets_the_robot_cannot_reach(arm_urdf):
    from examples.hv1.deploy_server import ArmRemap

    remap = ArmRemap(
        arm_urdf("old", limit=3.0), arm_urdf("new", elbow_x=0.03, wrist_x=-0.03), ([-1.0] * 7, [1.0] * 7, "p")
    )
    chunk = np.c_[np.tile([0, 0, 0, 2.5, 0, 0, 0], (15, 1)), np.zeros(15)]
    with pytest.raises(Rejected, match="remap of policy targets: IK"):
        remap.actions(chunk, np.zeros(7), np.zeros(7))


def test_arm_remap_refuses_to_show_the_policy_a_pose_its_arm_never_had(arm_urdf):
    """The policy state is solved inside the trained arm's limits: a robot pose
    that arm could not take is refused, not shown out of distribution."""
    from examples.hv1.deploy_server import ArmRemap

    remap = ArmRemap(arm_urdf("old", limit=1.0), arm_urdf("new", elbow_x=0.03, wrist_x=-0.03, limit=3.0), WIDE)
    with pytest.raises(Rejected, match="remap of robot state"):
        remap.state(np.array([0.0, 0.0, 0.0, 2.5, 0.0, 0.0, 0.0]))


def test_policy_step_applies_the_remap_around_the_model(arm_urdf):
    from examples.hv1.deploy_server import ArmRemap
    from examples.hv1.deploy_server import policy_step

    seen = {}

    class Model:
        def infer(self, obs):
            seen["state"] = obs["state"].copy()
            return {"actions": np.tile(np.r_[obs["state"][:7], 1.0], (15, 1))}

    remap = ArmRemap(arm_urdf("old"), arm_urdf("new", elbow_x=0.03, wrist_x=-0.03), WIDE)
    state = np.r_[GRASP_Q, np.zeros(8)].astype(np.float32)
    images = {c: np.zeros((224, 224, 3), np.uint8) for c in CAMERAS}
    actions = policy_step(Model(), metadata(), {}, state, images, remap)
    np.testing.assert_allclose(seen["state"][:7], remap.state(GRASP_Q), atol=1e-6)
    np.testing.assert_allclose(actions[:, :7], np.tile(GRASP_Q, (15, 1)), atol=1e-4)
    plain = policy_step(Model(), metadata(), {}, state, images, None)
    np.testing.assert_allclose(plain[:, :7], np.tile(GRASP_Q, (15, 1)), atol=1e-6)


def test_remap_needs_both_urdfs_and_the_field_limits(monkeypatch):
    import sys

    import examples.hv1.deploy_server as server_module

    for remap in (["--remap-robot-urdf", "x"], ["--remap-trained-urdf", "x", "--remap-robot-urdf", "y"]):
        monkeypatch.setattr(sys, "argv", ["deploy_server", "--campaign", "c", "--replay-episode", "e", *remap])
        with pytest.raises(SystemExit):
            server_module.main()
