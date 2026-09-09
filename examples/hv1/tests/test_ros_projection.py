import numpy as np
import pytest

from examples.hv1.ros_projection import accepted_open
from examples.hv1.ros_projection import named_positions
from examples.hv1.ros_projection import project_arm_state
from examples.hv1.ros_projection import project_single_target
from examples.hv1.workflow import ContractError
from examples.hv1.workflow import validate_profile


def mapping():
    return [{"index": i, "driver": f"wire{i}", "urdf": None if i < 2 else f"arm{i}"} for i in range(16)]


def test_state_names_define_order_neck_not_learned():
    names = [f"arm{i}" for i in range(15, 1, -1)]
    out = project_arm_state(names, list(range(15, 1, -1)), mapping())
    np.testing.assert_array_equal(out, np.arange(2, 16))


@pytest.mark.parametrize(("names", "values"), [(["a", "a"], [0, 0]), (["b"], [0]), (["a"], [float("nan")])])
def test_no_identity_or_numeric_guess(names, values):
    with pytest.raises(ContractError):
        named_positions(names, values, ["a"])


def test_command_reorders_aliases_and_rejects_chunks_partial_targets():
    names = [f"wire{i}" for i in range(16)]
    point = {"positions": list(range(16))}
    np.testing.assert_array_equal(project_single_target(names, [point], mapping()), np.arange(2, 16))
    with pytest.raises(ContractError, match="one point"):
        project_single_target(names, [point, point], mapping())
    with pytest.raises(ContractError, match="missing"):
        project_single_target(names[:9], [{"positions": list(range(9))}], mapping())
    with pytest.raises(ContractError, match="duplicate"):
        project_single_target(["wire2", "arm2"], [{"positions": [0, 0]}], mapping())


def test_accepted_open_does_not_use_requested_value():
    assert accepted_open({"mode": 1, "open": 0}, {"success": True, "open": 0.25}, fixed_mode=1) == 0.25
    with pytest.raises(ContractError, match="accepted"):
        accepted_open({"mode": 1}, {"success": False, "open": 0}, fixed_mode=1)
    with pytest.raises(ContractError, match="mode"):
        accepted_open({"mode": 2}, {"success": True, "open": 0}, fixed_mode=1)


def test_source_draft_cannot_be_used_as_data_profile():
    with pytest.raises(ContractError):
        validate_profile({"schema_version": 1, "status": "draft_source_inferred"})
