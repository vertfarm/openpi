"""Synthetic arms only: no robot geometry is kept in this public repository."""

import json

import numpy as np
import pytest

from examples.hv1.artifacts import ContractError
from examples.hv1.kinematics import Chain
from examples.hv1.kinematics import Remap
from examples.hv1.kinematics import check_step
from examples.hv1.kinematics import field_limits
from examples.hv1.tests.conftest import ARM_JOINTS

POSE = np.array([0.4, -0.1, 0.2, 0.9, 0.05, 0.3, 0.2])


def test_chain_follows_urdf_from_base_to_tool(arm_urdf):
    chain = Chain.from_urdf(arm_urdf("arm"), "Right_tool0")
    assert chain.names == ARM_JOINTS
    rotation, position = chain.fk(np.zeros(7))
    np.testing.assert_allclose(rotation, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(position, [0.0, -0.25, 0.3 - 0.1 - 0.25 - 0.2 - 0.1 - 0.05 - 0.1], atol=1e-12)


@pytest.mark.parametrize("tilt", ["0 0 0", "0.3 -0.2 0.5"])
def test_jacobian_matches_finite_differences(arm_urdf, tilt):
    """Pins the frame convention: origin xyz, then rpy, then the joint about its
    axis in the child frame; a wrong order shows up once rpy is not zero."""
    chain = Chain.from_urdf(arm_urdf("arm", tilt=tilt), "Right_tool0")
    j = chain.jacobian(POSE)
    r0, p0 = chain.fk(POSE)
    for index in range(7):
        dq = np.zeros(7)
        dq[index] = 1e-6
        r1, p1 = chain.fk(POSE + dq)
        np.testing.assert_allclose(j[:3, index], (p1 - p0) / 1e-6, atol=1e-5)
        w = r1 @ r0.T
        np.testing.assert_allclose(j[3:, index], np.array([w[2, 1], w[0, 2], w[1, 0]]) / 1e-6, atol=1e-5)


def test_ik_reaches_a_pose_and_stays_near_the_seed(arm_urdf):
    chain = Chain.from_urdf(arm_urdf("arm", tilt="0.3 -0.2 0.5"), "Right_tool0")
    target = chain.fk(POSE)
    q = chain.ik(*target, seed=POSE + 0.05)
    rotation, position = chain.fk(q)
    assert np.linalg.norm(position - target[1]) < 1e-6
    assert np.linalg.norm(rotation - target[0]) < 1e-5
    assert np.max(np.abs(q - POSE)) < 0.2


def test_ik_solves_a_pose_with_a_joint_at_its_limit(arm_urdf):
    """The recorded start pose sits near the elbow limit on the new arm; the
    solver must not stall on a joint the step keeps pushing into the limit."""
    chain = Chain.from_urdf(arm_urdf("arm", limit=1.0), "Right_tool0")
    at_limit = np.array([0.2, -0.1, 0.1, 1.0, 0.0, 0.2, 0.1])
    target = chain.fk(at_limit)
    for seed in (at_limit + np.array([-0.15, 0.1, -0.1, -0.2, 0.1, -0.1, 0.1]), np.zeros(7)):
        q = chain.ik(*target, seed=seed)
        np.testing.assert_allclose(chain.fk(q)[1], target[1], atol=1e-6)
        assert np.all(q <= 1.0 + 1e-12)


def test_remap_keeps_the_tool_pose_across_the_arm_change(arm_urdf):
    """The recorded angles put the old arm's tool somewhere; the remapped angles
    put the new arm's tool in the same place, which the raw angles do not."""
    old = Chain.from_urdf(arm_urdf("old"), "Right_tool0")
    new = Chain.from_urdf(arm_urdf("new", elbow_x=0.03, wrist_x=-0.03), "Right_tool0")
    q_new = Remap(old, new).convert(POSE, seed=POSE)
    (r_old, p_old), (_, p_raw), (r_new, p_new) = old.fk(POSE), new.fk(POSE), new.fk(q_new)
    assert np.linalg.norm(p_raw - p_old) > 0.02
    assert np.linalg.norm(p_new - p_old) < 1e-6 and np.linalg.norm(r_new - r_old) < 1e-5
    back = Remap(new, old).convert(q_new, seed=q_new)
    np.testing.assert_allclose(old.fk(back)[1], p_old, atol=1e-6)


def test_remapped_trajectory_is_continuous(arm_urdf):
    old = Chain.from_urdf(arm_urdf("old"), "Right_tool0")
    new = Chain.from_urdf(arm_urdf("new", elbow_x=0.03, wrist_x=-0.03), "Right_tool0")
    rows = POSE + np.linspace(0, 0.1, 15)[:, None]
    remap = Remap(old, new)
    converted = remap.trajectory(rows, seed=remap.convert(rows[0], rows[0]), source_seed=rows[0])
    assert converted.shape == (15, 7)
    assert np.max(np.abs(np.diff(converted, axis=0))) < 0.05


def test_step_check_refuses_a_jump_larger_than_its_source():
    check_step(np.full(7, 0.01), np.full(7, 0.02))
    with pytest.raises(ContractError, match="discontinuous"):
        check_step(np.full(7, 0.01), np.r_[0.3, np.zeros(6)])


def test_ik_refuses_a_pose_only_reachable_outside_the_limits(arm_urdf):
    loose = Chain.from_urdf(arm_urdf("loose", limit=3.0), "Right_tool0")
    tight = Chain.from_urdf(arm_urdf("tight", limit=0.5), "Right_tool0")
    target = loose.fk(np.array([0.0, 0.0, 0.0, 2.5, 0.0, 0.0, 0.0]))
    with pytest.raises(ContractError, match="IK"):
        tight.ik(*target, seed=np.zeros(7))


def test_field_limits_replace_the_urdf_placeholders(arm_urdf):
    tight = Chain.from_urdf(arm_urdf("tight", limit=0.5), "Right_tool0")
    wide = tight.with_limits([-3.0] * 7, [3.0] * 7)
    q = wide.ik(*wide.fk(np.array([0.0, 0.0, 0.0, 2.5, 0.0, 0.0, 0.0])), seed=np.r_[0, 0, 0, 2.0, 0, 0, 0])
    assert abs(q[3] - 2.5) < 1e-4
    assert tight.upper[3] == 0.5  # the original chain is unchanged
    with pytest.raises(ContractError, match="lower < upper"):
        tight.with_limits([1.0] * 7, [0.0] * 7)
    with pytest.raises(ContractError, match="finite"):
        tight.with_limits([float("nan")] * 7, [1.0] * 7)


def test_field_limits_come_only_from_an_approved_profile(tmp_path):
    def profile(**values):
        path = tmp_path / "profile.json"
        path.write_text(json.dumps(values), encoding="utf-8")
        return path

    low, high, sha = field_limits(profile(approved=True, review_id="R", joint_min=[-1] * 7, joint_max=[1] * 7))
    assert list(low) == [-1] * 7 and list(high) == [1] * 7 and len(sha) == 64
    for bad in (
        dict(approved=False, review_id="R", joint_min=[-1] * 7, joint_max=[1] * 7),
        dict(approved=True, review_id="R", joint_min=None, joint_max=None),
        dict(approved=True, review_id="R", joint_min=[1] * 7, joint_max=[-1] * 7),
    ):
        with pytest.raises(ContractError):
            field_limits(profile(**bad))


def test_chain_needs_the_tool_link_and_supported_joints(arm_urdf, tmp_path):
    with pytest.raises(ContractError, match="tool"):
        Chain.from_urdf(arm_urdf("arm"), "Left_tool0")
    text = arm_urdf("arm").read_text(encoding="utf-8")
    text = text.replace(
        'name="Right_Wrist_Yaw_Joint" type="revolute"', 'name="Right_Wrist_Yaw_Joint" type="continuous"'
    )
    odd = tmp_path / "odd.urdf"
    odd.write_text(text, encoding="utf-8")
    with pytest.raises(ContractError, match="unsupported"):
        Chain.from_urdf(odd, "Right_tool0")
