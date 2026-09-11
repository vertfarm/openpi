"""The cross-modal matrix has to separate a policy that reads the scene from one
that answers from the arm. Teacher-forced replay cannot: it hands the policy the
state that already implies the answer."""

import numpy as np
import pytest

from examples.hv1.artifacts import ContractError
from examples.hv1.metrics import cross_modal_matrix

HORIZON = 15


def chunk(intent, move):
    out = np.zeros((HORIZON, 8))
    out[:, :7] = np.linspace(0, 1, HORIZON)[:, None] * np.asarray(move)
    out[:, 7] = intent
    return out


def cases(count=3):
    return [{"name": f"ep{i}", "state": np.full(15, i * 0.1), "images": {"head": bytes([i])}} for i in range(count)]


def test_a_state_only_policy_shows_no_scene_gap():
    """What TODAY30-1000 did on 2026-09-11: same answer whatever it was shown."""
    result = cross_modal_matrix(lambda state, images: chunk(1.0, [0.1] * 7), cases())
    assert result["intent_scene_gap"] == pytest.approx(0.0)
    assert result["direction_cosine_median"] == pytest.approx(1.0)


def test_a_scene_reading_policy_shows_a_gap():
    scenes = cases()
    order = {id(case["images"]): i for i, case in enumerate(scenes)}

    def predict(state, images):
        # Fires only for the scene whose state matches: state i carries 0.1 * i.
        wanted = order[id(images)]
        matched = np.isclose(state[0], wanted * 0.1)
        return chunk(1.0 if matched else 0.0, [0.1 if matched else -0.1] * 7)

    result = cross_modal_matrix(predict, scenes)
    assert result["diagonal_intent_median"] == pytest.approx(1.0)
    assert result["off_diagonal_intent_median"] == pytest.approx(0.0)
    assert result["intent_scene_gap"] == pytest.approx(1.0)
    assert result["direction_cosine_median"] == pytest.approx(-1.0)


def test_matrix_is_square_and_labelled():
    result = cross_modal_matrix(lambda state, images: chunk(0.5, [0.01] * 7), cases(4))
    assert np.asarray(result["intent_matrix"]).shape == (4, 4)
    assert result["cases"] == ["ep0", "ep1", "ep2", "ep3"]


def test_one_case_cannot_be_crossed():
    with pytest.raises(ContractError):
        cross_modal_matrix(lambda state, images: chunk(1.0, [0.1] * 7), cases(1))


def test_a_malformed_chunk_is_refused():
    with pytest.raises(ContractError):
        cross_modal_matrix(lambda state, images: np.zeros((HORIZON, 3)), cases())
