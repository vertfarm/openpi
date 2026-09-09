import json
import sys
import threading
from urllib.error import HTTPError
from urllib.request import Request
from urllib.request import urlopen

import h5py
import numpy as np
import pytest

from examples.hv1.adapter import ShadowAdapter
from examples.hv1.demo import make_demo
from examples.hv1.review import make_server
from examples.hv1.transforms import HV1Inputs
from examples.hv1.transforms import HV1Outputs
from examples.hv1.workflow import Catalog
from examples.hv1.workflow import ContractError
from examples.hv1.workflow import causal_indices
from examples.hv1.workflow import digest
from examples.hv1.workflow import file_hash
from examples.hv1.workflow import inspect_episode
from examples.hv1.workflow import load_manifest
from examples.hv1.workflow import read_frame
from examples.hv1.workflow import read_json
from examples.hv1.workflow import validate_profile


@pytest.fixture
def fixture(tmp_path):
    info = make_demo(tmp_path / "demo")
    p = read_json(info["profile"])
    catalog = Catalog(tmp_path / "runtime", info["raw"], p)
    return p, catalog, tmp_path


def approve_all(catalog):
    reports = catalog.scan()
    for report in reports:
        catalog.review(report["id"], "success", use=True, reason="synthetic verification")
    return reports


def test_scan_is_read_only_and_review_survives_rescan(fixture):
    p, catalog, root = fixture
    before = {f.name: file_hash(f) for f in catalog.raw_root.glob("*.hdf5")}
    reports = approve_all(catalog)
    assert len(reports) == 3
    assert all(r["quality_pass"] and r["frames"] == 12 for r in reports)
    catalog.scan()
    assert all(r["review"]["use"] for r in catalog.list())
    assert before == {f.name: file_hash(f) for f in catalog.raw_root.glob("*.hdf5")}


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda p: p["action"].update(path=p["state"]["path"]), "same dataset"),
        (lambda p: p["action"].update(semantics="measured_state"), "commanded"),
        (lambda p: p["state"].update(clock_domain="device"), "clock"),
        (lambda p: p.update(fps=0), "fps"),
        (lambda p: p["action"].update(delta_state_indices=[0, 99, -1]), "index"),
        (lambda p: p["state"].update(units=["deg", "rad", "fraction"]), "units"),
        (lambda p: p["images"]["head"].update(color_order="UNKNOWN"), "RGB"),
    ],
)
def test_profile_contract_rejections(fixture, mutation, match):
    p, _, _ = fixture
    mutation(p)
    with pytest.raises(ContractError, match=match):
        validate_profile(p)


@pytest.mark.parametrize(
    ("damage", "expected"),
    [
        ("incomplete", "complete"),
        ("timestamp", "increase"),
        ("nan", "NaN"),
        ("missing_camera", "missing dataset"),
        ("image_dtype", "uint8"),
        ("missing_task", "task"),
    ],
)
def test_bad_episode_rejected(fixture, damage, expected):
    p, catalog, _ = fixture
    path = next(catalog.raw_root.glob("*.hdf5"))
    with h5py.File(path, "r+") as handle:
        if damage == "incomplete":
            handle.attrs["complete"] = False
        if damage == "timestamp":
            handle["timestamps_ns"][3] = handle["timestamps_ns"][2]
        if damage == "nan":
            handle["observation/state"][2, 0] = np.nan
        if damage == "missing_camera":
            del handle["images/head"]
        if damage == "image_dtype":
            data = handle["images/head"][:].astype(np.float32)
            del handle["images/head"]
            handle.create_dataset("images/head", data=data)
        if damage == "missing_task":
            del handle.attrs["task"]
    report = inspect_episode(path, p)
    assert not report["quality_pass"]
    assert expected in report["errors"][0]


def test_integer_timestamps_and_hdf5_links(fixture):
    p, catalog, _ = fixture
    path = next(catalog.raw_root.glob("*.hdf5"))
    with h5py.File(path, "r+") as handle:
        del handle["command/target"]
        handle["command/target"] = h5py.SoftLink("/observation/state")
    assert "linked HDF5" in inspect_episode(path, p)["errors"][0]


def test_sampling_is_causal_not_future_filled():
    source = np.array([100, 200, 300], dtype=np.int64)
    np.testing.assert_array_equal(causal_indices(source, np.array([100, 250]), 1), [0, 1])
    with pytest.raises(ContractError, match="future"):
        causal_indices(source, np.array([90]), 1)
    with pytest.raises(ContractError, match="stale"):
        causal_indices(source, np.array([500]), 0.00001)


def test_manifest_session_split_integrity_and_no_overwrite(fixture):
    p, catalog, root = fixture
    approve_all(catalog)
    path = root / "manifest.json"
    with pytest.raises(ContractError, match="synthetic"):
        catalog.manifest(path)
    value = catalog.manifest(path, allow_synthetic=True)
    train = {e["session_id"] for e in value["episodes"] if e["split"] == "train"}
    val = {e["session_id"] for e in value["episodes"] if e["split"] == "validation"}
    assert train
    assert val
    assert not train & val
    assert load_manifest(path)["manifest_sha256"] == value["manifest_sha256"]
    with pytest.raises(FileExistsError):
        catalog.manifest(path, allow_synthetic=True)
    value["split_seed"] = 99
    path.write_text(json.dumps(value))
    with pytest.raises(ContractError, match="integrity"):
        load_manifest(path)


def test_no_outputs_inside_raw_root(fixture):
    p, catalog, _ = fixture
    with pytest.raises(ContractError):
        Catalog(catalog.raw_root / "runtime", catalog.raw_root, p)
    approve_all(catalog)
    with pytest.raises(ContractError, match="immutable"):
        catalog.manifest(catalog.raw_root / "manifest.json", allow_synthetic=True)


def test_single_session_is_not_claimed_as_validation(fixture):
    p, catalog, root = fixture
    report = catalog.scan()[0]
    catalog.review(report["id"], "success", use=True, reason="one session")
    assert not catalog.manifest(root / "single.json", allow_synthetic=True)["validation_available"]


def test_changed_source_cannot_reuse_review(fixture):
    _, catalog, root = fixture
    reports = approve_all(catalog)
    with h5py.File(reports[0]["path"], "r+") as handle:
        handle["command/target"][0, 0] = 0.9
    with pytest.raises(ContractError, match="changed"):
        catalog.manifest(root / "bad.json", allow_synthetic=True)
    catalog.scan()
    assert catalog.get(reports[0]["id"])["superseded"]
    changed = next(r for r in catalog.list() if r["path"] == reports[0]["path"] and not r["superseded"])
    assert not changed["review"]


def test_quality_and_task_success_are_separate(fixture):
    _, catalog, _ = fixture
    report = catalog.scan()[0]
    catalog.review(report["id"], "failure", use=False, reason="object dropped")
    assert catalog.get(report["id"])["quality_pass"]
    with pytest.raises(ContractError):
        catalog.review(report["id"], "failure", use=True, reason="")


def test_delta_roundtrip_and_color_mapping(fixture):
    p, catalog, _ = fixture
    report = catalog.scan()[0]
    frame = read_frame(report["path"], p, 0)
    actions = np.repeat(frame["action"][None], p["action_horizon"], axis=0)
    mapped = HV1Inputs(p)({"state": frame["state"], "images": frame["images"], "actions": actions})
    assert set(mapped["image"]) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    restored = HV1Outputs(p)({"state": mapped["state"], "actions": np.pad(mapped["actions"], ((0, 0), (0, 29)))})
    np.testing.assert_allclose(restored["actions"], actions, atol=1e-7)
    np.testing.assert_array_equal(mapped["actions"][:, 2], actions[:, 2])
    assert "rclpy" not in sys.modules


def metadata(p):
    return {
        "profile_sha256": digest(p),
        "action_names": p["action"]["names"],
        "action_units": p["action"]["units"],
        "output_action_space": "commanded_target",
    }


def test_shadow_returns_first_only_and_never_sends(fixture):
    p, _, _ = fixture
    adapter = ShadowAdapter(p, metadata(p), max_observation_age_s=0.2, response_timeout_s=0.1)
    out = adapter.validate(np.zeros((4, 3)), observation_monotonic=10, request_monotonic=10, now=10.05)
    assert out["sent"] is False
    assert out["candidate_first_action"] == [0, 0, 0]


@pytest.mark.parametrize(
    ("actions", "obs", "request_at", "now"),
    [
        (np.zeros((4, 3)), 9, 10, 10.05),
        (np.zeros((4, 3)), 10, 9, 10.05),
        (np.zeros((4, 3)), 11, 10, 10.05),
        (np.zeros((1, 3)), 10, 10, 10.05),
        (np.full((4, 3), np.nan), 10, 10, 10.05),
    ],
)
def test_shadow_latches_fault(fixture, actions, obs, request_at, now):
    p, _, _ = fixture
    adapter = ShadowAdapter(p, metadata(p), max_observation_age_s=0.2, response_timeout_s=0.1)
    with pytest.raises(ContractError):
        adapter.validate(actions, observation_monotonic=obs, request_monotonic=request_at, now=now)
    with pytest.raises(ContractError, match="latched"):
        adapter.validate(np.zeros((4, 3)), observation_monotonic=10, request_monotonic=10, now=10.05)


def test_shadow_rejects_droid_metadata(fixture):
    p, _, _ = fixture
    with pytest.raises(ContractError, match="metadata"):
        ShadowAdapter(p, {"action_dim": 8}, max_observation_age_s=0.2, response_timeout_s=0.1)


def test_review_http_has_no_control_api_and_protects_writes(fixture):
    _, catalog, _ = fixture
    report = catalog.scan()[0]
    server = make_server(catalog, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base) as response:
            html = response.read().decode()
        assert "로봇 명령 없음" in html
        token = html.split("const csrf='")[1].split("'")[0]
        with urlopen(base + "/api/episodes") as response:
            assert len(json.load(response)["episodes"]) == 3
        with urlopen(base + f"/api/image?id={report['id']}&frame=0&slot=head") as response:
            assert response.read().startswith(b"\x89PNG")
        body = json.dumps({"id": report["id"], "outcome": "success", "use": True}).encode()
        with pytest.raises(HTTPError) as error:
            urlopen(Request(base + "/api/review", data=body))
        assert error.value.code == 403
        req = Request(base + "/api/review", data=body, headers={"X-HV1-CSRF": token})
        with urlopen(req) as response:
            assert json.load(response)["use"] is True
        with pytest.raises(HTTPError) as error:
            urlopen(base + "/api/enable")
        assert error.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
