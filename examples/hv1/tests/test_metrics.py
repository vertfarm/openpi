"""The cross-modal matrix has to separate a policy that reads the scene from one
that answers from the arm. Teacher-forced replay cannot: it hands the policy the
state that already implies the answer."""

import numpy as np
import pytest

from examples.hv1.artifacts import ContractError
from examples.hv1.metrics import condition_intents
from examples.hv1.metrics import cross_modal_matrix
from examples.hv1.metrics import transition_metrics

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


def truth_one_cycle():
    return np.r_[np.zeros(30), np.ones(90), np.zeros(60)]


def test_a_perfect_replay_scores_no_error():
    truth = truth_one_cycle()
    result = transition_metrics(truth, truth, [30], [120])
    assert result["error_rate"] == 0 and result["transition_error_rate"] == 0
    assert result["close_time_error_s"] == [0] and result["release_time_error_s"] == [0]
    assert result["extra_close"] == result["extra_release"] == 0
    assert result["missed_close"] == result["missed_release"] == 0
    assert result["initial_closed"] is False


def test_a_hand_that_never_opens_is_a_missed_release_not_a_late_one():
    result = transition_metrics(np.ones(180), truth_one_cycle(), [30], [120])
    assert result["missed_release"] == 1 and result["initial_closed"] is True
    assert result["predicted_release_frames"] == []


def test_a_grasp_that_lets_go_and_regrabs_shows_as_extra_transitions():
    """The 2026-09-10 `extra_close` finding: flow-matching sampling on a bimodal
    channel toggles mid-grasp, which teacher-forced error rate barely registers."""
    predicted = truth_one_cycle()
    predicted[60:70] = 0
    result = transition_metrics(predicted, truth_one_cycle(), [30], [120])
    assert result["predicted_close_frames"] == [30, 70]
    assert result["predicted_release_frames"] == [60, 120]
    assert result["extra_close"] == 1 and result["extra_release"] == 1
    assert result["missed_close"] == result["missed_release"] == 0
    assert result["error_rate"] == pytest.approx(10 / 180)


def test_a_late_but_single_cycle_is_reported_as_a_delay():
    delayed = np.r_[np.zeros(36), np.ones(90), np.zeros(54)]
    result = transition_metrics(delayed, truth_one_cycle(), [30], [120])
    assert result["close_time_error_s"] == [0.2] and result["release_time_error_s"] == [0.2]
    assert result["extra_close"] == result["extra_release"] == 0


@pytest.mark.parametrize(
    "predicted,truth",
    [(np.zeros(10), np.zeros(11)), (np.zeros((10, 2)), np.zeros((10, 2))), (np.zeros(0), np.zeros(0))],
)
def test_unaligned_or_empty_series_are_refused(predicted, truth):
    with pytest.raises(ContractError, match="aligned nonempty"):
        transition_metrics(predicted, truth, [0], [1])


@pytest.mark.parametrize("values", [np.zeros((3, 2)), np.zeros(0), np.array([0.0, np.nan, 1.0])])
def test_conditioning_refuses_a_series_it_cannot_filter(values):
    with pytest.raises(ContractError, match="finite nonempty"):
        condition_intents(values)


def test_conditioning_refuses_a_timeline_that_runs_backwards():
    with pytest.raises(ContractError, match="monotonic"):
        condition_intents(np.zeros(3), times=np.array([0.0, 0.2, 0.1]))
