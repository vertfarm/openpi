import json

import h5py
import numpy as np
import pytest

from examples.hv1.native import ARM_ACTION
from examples.hv1.native import ARM_STATE
from examples.hv1.native import DUMMIES
from examples.hv1.native import HAND_STATE
from examples.hv1.native import SUSPECT
from examples.hv1.native import VALIDATION
from examples.hv1.native import profile
from examples.hv1.native import read_numeric
from examples.hv1.overnight_common import atomic_json
from examples.hv1.overnight_common import recipe
from examples.hv1.overnight_common import select_target
from examples.hv1.overnight_common import snapshot_steps
from examples.hv1.workflow import ContractError


def fixture_native(path):
    specs = {
        "observation.state.upper_body.joint": ARM_STATE,
        "observation.state.hand.joint_r": HAND_STATE,
        "action.upper_body.joint": ARM_ACTION,
        "action.hand.command_r": ["mode", "open"],
    }
    with h5py.File(path, "w") as f:
        meta = f.create_group("meta")
        meta.attrs.update(labels=json.dumps(specs), num_frames=6, created_at="2026-09-09T10:10:00Z")
        for key, names in specs.items():
            f.create_dataset(key, data=np.zeros((6, len(names)), dtype="float32"))
        f["action.upper_body.joint"][:] = 0.1
        f["action.hand.command_r"][:] = [[2, 0.6], [2, 0.6], [2, 0], [2, 0], [2, 0.6], [2, 0.6]]
        f.create_dataset("timestamp", data=np.arange(6) / 30 + 1)
        f.create_dataset("stamp_ns", data=np.arange(6, dtype=np.int64) * 33333333)
        f.create_dataset("stale", data=np.zeros(6, dtype="uint32"))


def test_native_action_is_command_not_state(tmp_path):
    path = tmp_path / "data.hdf5"
    fixture_native(path)
    state, action, info = read_numeric(path)
    assert state.shape == (6, 15) and action.shape == (6, 8)
    np.testing.assert_allclose(action[:, :7], 0.1)
    assert action[:, -1].tolist() == [0, 0, 1, 1, 0, 0]
    assert info["grasp_frame"] == 2 and info["release_frame"] == 4


@pytest.mark.parametrize("damage", ["mode", "open", "stale", "timestamp", "nan", "cycle"])
def test_native_rejects_invalid_fields(tmp_path, damage):
    path = tmp_path / "data.hdf5"
    fixture_native(path)
    with h5py.File(path, "r+") as f:
        if damage == "mode":
            f["action.hand.command_r"][0, 0] = 1
        if damage == "open":
            f["action.hand.command_r"][0, 1] = 1
        if damage == "stale":
            f["stale"][1] = 1
        if damage == "timestamp":
            f["timestamp"][1] = 0
        if damage == "nan":
            f["observation.state.hand.joint_r"][0, 0] = np.nan
        if damage == "cycle":
            f["action.hand.command_r"][:, 1] = 0.6
    with pytest.raises(ContractError):
        read_numeric(path)


def test_all_cameras_and_delta_mapping_fixed():
    p = profile()
    assert set(p["images"]) == {"head", "hand_l", "hand_r"}
    assert p["camera_dropout"] is False
    assert p["action"]["delta_state_indices"] == [0, 1, 2, 3, 4, 5, 6, -1]
    assert p["gripper"]["release_open"] == 0.6
    assert not DUMMIES & (SUSPECT | VALIDATION)
    assert not SUSPECT & VALIDATION


def test_controlled_recipes():
    a = recipe("A", 1000)
    for name, key, value in [
        ("B", "cohort", "inclusive"),
        ("D", "action_horizon", 30),
        ("E", "seed", 7),
        ("F", "lora", True),
    ]:
        r = recipe(name, 1000)
        assert r[key] == value and r["from_base"] and r["seed"] == (7 if name == "E" else 42)
    assert recipe("C", 1000)["peak_lr"] == 1e-5
    assert not a["robot_motion_authorized"]


def test_snapshot_dedup_and_budget():
    assert snapshot_steps("A", 500) == [500]
    assert snapshot_steps("A", 3000) == [500, 1000, 2000, 3000]
    assert snapshot_steps("C", 1000) == [250, 1000]
    assert select_target(4000, 1, 100, pair=True) == 1000
    assert select_target(500, 1, 100, pair=True) == 0
    with pytest.raises(ContractError):
        select_target(1, float("nan"), 1)


def test_atomic_status(tmp_path):
    p = tmp_path / "status.json"
    atomic_json(p, {"phase": "start"})
    atomic_json(p, {"phase": "done"})
    assert json.loads(p.read_text()) == {"phase": "done"}
    assert len(list(tmp_path.iterdir())) == 1


def test_real_orbax_bf16_roundtrip(tmp_path, monkeypatch):
    jax = pytest.importorskip("jax")
    nnx = pytest.importorskip("flax.nnx")
    from types import SimpleNamespace

    from examples.hv1 import overnight_train
    from openpi.shared.normalize import NormStats

    monkeypatch.setattr(overnight_train, "storage_gate", lambda *args: None)
    params = nnx.State({"weight": nnx.Param(jax.numpy.asarray([[1.2345, 2.1], [3.2, 4.3]]))})
    config = SimpleNamespace(exp_name="test", policy_metadata={"recipe": recipe("A", 50)})
    norm = NormStats(mean=np.zeros(2), std=np.ones(2), q01=np.zeros(2), q99=np.ones(2))
    data = SimpleNamespace(asset_id="test", norm_stats={"state": norm})
    loader = SimpleNamespace(data_config=lambda: data)
    result = overnight_train.save_snapshot(tmp_path, config, SimpleNamespace(params=params), loader, 50, {"loss": 0.1})
    report = json.loads((result / "snapshot.json").read_text())
    assert report["cpu_roundtrip_pass"] and report["complete"]
    assert report["checkpoint_dtype"] == "bfloat16"
    assert (result / "assets/test/norm_stats.json").exists()
