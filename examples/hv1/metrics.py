"""Policy metrics that outlive any one campaign. No trainer, no ROS, no robot.

These moved out of `pipeline_eval` because they describe a policy, not the
2026-09-10 TODAY30/ALL59 experiment whose name that module carries. Every
retraining cycle wants the same numbers, and reaching for them through a
campaign module means each new campaign copies them again.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .artifacts import ContractError
from .ros.keti_humanoid_inference.keti_humanoid_inference.core import IntentConditioner


def _edges(values):
    closed = np.asarray(values) >= 0.5
    changes = np.flatnonzero(np.diff(np.r_[False, closed]) != 0)
    return [int(i) for i in changes if closed[i]], [int(i) for i in changes if not closed[i]]


def _ordered_match(predicted, expected, tolerance):
    predicted, expected = tuple(predicted), tuple(expected)

    @lru_cache(maxsize=None)
    def solve(i, j):
        if i == len(predicted) or j == len(expected):
            return 0, 0, ()
        options = [solve(i + 1, j), solve(i, j + 1)]
        if abs(predicted[i] - expected[j]) <= tolerance:
            matched, cost, pairs = solve(i + 1, j + 1)
            options.append((matched + 1, cost + abs(predicted[i] - expected[j]), ((i, j), *pairs)))
        return max(options, key=lambda value: (value[0], -value[1]))

    _, _, pairs = solve(0, 0)
    return [(predicted[i], expected[j]) for i, j in pairs]


def transition_metrics(predicted, truth, close_frames, release_frames, fps=30):
    predicted = np.asarray(predicted) >= 0.5
    truth = np.asarray(truth) >= 0.5
    if predicted.shape != truth.shape or predicted.ndim != 1 or not len(predicted):
        raise ContractError("aligned nonempty intent series required")
    predicted_close, predicted_release = _edges(predicted)
    close_pairs = _ordered_match(predicted_close, close_frames, fps)
    release_pairs = _ordered_match(predicted_release, release_frames, fps)
    around = np.zeros(len(predicted), dtype=bool)
    for frame in [*close_frames, *release_frames]:
        around[max(0, frame - fps) : min(len(predicted), frame + fps + 1)] = True
    return dict(
        error_rate=float(np.mean(predicted != truth)),
        transition_error_rate=float(np.mean((predicted != truth)[around])),
        expected_close_frames=close_frames,
        expected_release_frames=release_frames,
        predicted_close_frames=predicted_close,
        predicted_release_frames=predicted_release,
        matched_close=[[p, t] for p, t in close_pairs],
        matched_release=[[p, t] for p, t in release_pairs],
        close_time_error_s=[(p - t) / fps for p, t in close_pairs],
        release_time_error_s=[(p - t) / fps for p, t in release_pairs],
        missed_close=len(close_frames) - len(close_pairs),
        missed_release=len(release_frames) - len(release_pairs),
        extra_close=len(predicted_close) - len(close_pairs),
        extra_release=len(predicted_release) - len(release_pairs),
        initial_closed=bool(predicted[0]),
        note="Teacher-forced recorded observations; not measured physical grasp success.",
    )


def condition_intents(values, *, times=None, fps=30, close_threshold=0.7, open_threshold=0.3, min_hold_s=0.2):
    """Apply the same deploy-time hysteresis/dwell conditioner to an intent timeline."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ContractError("finite nonempty scalar intent series required")
    times = np.arange(len(values), dtype=np.float64) / fps if times is None else np.asarray(times, dtype=np.float64)
    if times.shape != values.shape or not np.isfinite(times).all() or np.any(np.diff(times) < 0):
        raise ContractError("monotonic intent times required")
    conditioner = IntentConditioner(
        close_threshold=close_threshold,
        open_threshold=open_threshold,
        min_hold_s=min_hold_s,
    )
    filtered, events = [], []
    for value, now in zip(values, times, strict=True):
        event = conditioner.update(value, now)
        if event:
            events.append({"event": event, "time_s": float(now)})
        filtered.append(float(conditioner.closed))
    return np.asarray(filtered, dtype=np.float32), events


def cross_modal_matrix(predict, cases):
    """How much of a prediction comes from the images rather than the state.

    `cases` pairs a state with the images recorded alongside it; `predict(state,
    images)` returns one action chunk. Every state is then run against every
    case's images, so the diagonal is the pairing that actually occurred and
    everything off it is a state seeing a scene it never came from.

    A policy that reads the scene must answer differently off the diagonal. On
    2026-09-11 the TODAY30-1000 checkpoint scored a median 1.021 on both,
    meaning it decided from the arm alone - which teacher-forced replay cannot
    show, because replay hands the policy the very state that already implies
    the answer.
    """
    if len(cases) < 2:
        raise ContractError("need at least two cases to cross")
    size = len(cases)
    intents = np.zeros((size, size))
    moves = np.zeros((size, size, 7))
    for row, source in enumerate(cases):
        for column, scene in enumerate(cases):
            chunk = np.asarray(predict(source["state"], scene["images"]), dtype=np.float64)
            if chunk.ndim != 2 or chunk.shape[1] < 8:
                raise ContractError("predict must return a (horizon, >=8) chunk")
            intents[row, column] = float(chunk[:, 7].max())
            moves[row, column] = chunk[-1, :7] - chunk[0, :7]

    def unit(vector):
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector

    # How far the arm's intended direction swings when only the scene changes.
    swings = [
        float(unit(moves[row, row]) @ unit(moves[row, column]))
        for row in range(size)
        for column in range(size)
        if row != column
    ]
    mask = ~np.eye(size, dtype=bool)
    diagonal, off = np.diag(intents), intents[mask]
    return {
        "cases": [case.get("name", str(index)) for index, case in enumerate(cases)],
        "intent_matrix": intents.tolist(),
        "diagonal_intent_median": float(np.median(diagonal)),
        "off_diagonal_intent_median": float(np.median(off)),
        "intent_scene_gap": float(np.median(diagonal) - np.median(off)),
        "direction_cosine_median": float(np.median(swings)),
        "note": (
            "Rows are states, columns are the images they were shown. An "
            "intent_scene_gap near zero, or a direction_cosine_median near one, "
            "means the policy answered from the state and ignored the scene."
        ),
    }


def demo_geometry(starts, grasps):
    """How much of a demonstration set a policy can solve without looking.

    `starts` and `grasps` are (episodes, 3) end-effector positions at the first
    frame and at the recorded grasp. The grasp position stands in for the object.
    If the start barely moves and the objects sit close together compared with
    the reach, the mean trajectory already lands near every object, and the
    images are only worth that residual. On 2026-09-30 the 59 HV1 episodes had
    a 1.0 cm start spread, a 16.6 cm reach and a 4.2 cm leave-one-out error for
    the mean grasp - a shortcut ratio of about 0.25.

    `shortcut_ratio` is that mean-only error over the median reach. Small means
    the arm alone explains the task. No threshold is applied.
    """
    starts = np.asarray(starts, dtype=np.float64)
    grasps = np.asarray(grasps, dtype=np.float64)
    if starts.ndim != 2 or starts.shape != grasps.shape or starts.shape[1] != 3 or len(starts) < 3:
        raise ContractError("need matching (episodes>=3, 3) start and grasp positions")
    size = len(starts)
    pairs = [(a, b) for a in range(size) for b in range(a + 1, size)]
    reach = grasps - starts
    lengths = np.linalg.norm(reach, axis=1)
    if np.any(lengths == 0):
        raise ContractError("an episode grasps where it starts")
    directions = reach / lengths[:, None]
    design = np.c_[starts, np.ones(size)]
    mean_errors, regression_errors = [], []
    for held in range(size):
        rest = [index for index in range(size) if index != held]
        mean_errors.append(float(np.linalg.norm(grasps[rest].mean(axis=0) - grasps[held])))
        weights, *_ = np.linalg.lstsq(design[rest], grasps[rest], rcond=None)
        regression_errors.append(float(np.linalg.norm(design[held] @ weights - grasps[held])))
    median_reach = float(np.median(lengths))
    loo_mean = float(np.median(mean_errors))
    return {
        "episodes": size,
        "start_pairwise_median": float(np.median([np.linalg.norm(starts[a] - starts[b]) for a, b in pairs])),
        "grasp_std": grasps.std(axis=0).tolist(),
        "grasp_range": (grasps.max(axis=0) - grasps.min(axis=0)).tolist(),
        "reach_median": median_reach,
        "loo_mean_grasp_error_median": loo_mean,
        "loo_start_regression_error_median": float(np.median(regression_errors)),
        "direction_cosine_median": float(np.median([directions[a] @ directions[b] for a, b in pairs])),
        "shortcut_ratio": loo_mean / median_reach,
        "threshold_applied": False,
    }
