"""The guardian must refuse a proposal it cannot re-derive or bound."""

import hashlib
import json

import numpy as np
import pytest

from examples.hv1.ros.keti_humanoid_inference.keti_humanoid_inference.core import qualify_proposal

LIMIT = np.full(7, 1.5708)
STEP = np.full(7, 0.02)


def payload(actions, *, tamper=False):
    proposal = {
        "sequence": 7,
        "anchor": 100.0,
        "targets": [{"due": 100.0 + i * (1 / 30), "action": list(a) + [0.0]} for i, a in enumerate(actions)],
    }
    encoded = json.dumps(proposal, sort_keys=True, separators=(",", ":")).encode()
    identifier = hashlib.sha256(encoded).hexdigest()
    if tamper:
        proposal["targets"][0]["action"][0] += 0.5
    return {"proposal_id": identifier, "proposal": proposal}


def test_accepts_a_bounded_proposal_and_returns_its_own_hash():
    actions = [[0.10] * 7, [0.11] * 7, [0.12] * 7]
    result = qualify_proposal(payload(actions), -LIMIT, LIMIT, STEP)
    assert result == payload(actions)["proposal_id"]


def test_refuses_contents_that_do_not_hash_to_the_supplied_id():
    """Trusting the executor's id would approve whatever it substitutes later."""
    actions = [[0.10] * 7, [0.11] * 7]
    assert qualify_proposal(payload(actions, tamper=True), -LIMIT, LIMIT, STEP) is None


def test_refuses_a_target_outside_the_joint_envelope():
    assert qualify_proposal(payload([[0.1] * 7, [2.0] * 7]), -LIMIT, LIMIT, STEP) is None


def test_refuses_a_step_larger_than_max_step():
    assert qualify_proposal(payload([[0.10] * 7, [0.30] * 7]), -LIMIT, LIMIT, STEP) is None


@pytest.mark.parametrize(
    "broken",
    [{}, {"proposal_id": "x"}, {"proposal": {"targets": []}, "proposal_id": "x"},
     {"proposal": {"targets": [{"action": [0.0] * 3}]}, "proposal_id": "x"}],
)
def test_refuses_malformed_payloads(broken):
    assert qualify_proposal(broken, -LIMIT, LIMIT, STEP) is None


def test_a_rejection_names_the_joint_and_the_margin():
    """The rejected target is never published, so the fault line is the only
    place its magnitude can appear."""
    from examples.hv1.ros.keti_humanoid_inference.keti_humanoid_inference.core import worst

    step = np.array([0.001, 0.002, 0.0500, 0.003, 0.004, 0.005, 0.006])
    message = "target step" + worst(step, np.full(7, 0.041))
    assert message == "target step: joint 3 at 0.0500 over 0.0410"
