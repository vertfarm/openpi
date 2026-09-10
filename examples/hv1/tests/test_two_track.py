from collections import Counter
import json

import numpy as np

from examples.hv1 import two_track
from examples.hv1 import two_track_eval


def manifest():
    episodes = []
    for cohort, session, ids in (
        ("old", two_track.OLD_SESSION, sorted(two_track.EXPECTED_OLD)),
        ("today", two_track.TODAY_SESSION, sorted(two_track.EXPECTED_TODAY)),
    ):
        for episode_id in ids:
            recovery = cohort == "today" and episode_id in {"episode_000002", "episode_000016"}
            episodes.append(
                {
                    "id": two_track.uid(session, episode_id),
                    "episode_id": episode_id,
                    "cohort": cohort,
                    "frames": 1000,
                    "grasp_frames": [200, 500] if recovery else [200],
                    "release_frames": [350, 800] if recovery else [800],
                    "training_tracks": ["ALL59"] + (["TODAY30"] if cohort == "today" else []),
                    "diagnostic_group": None,
                    "suspect": cohort == "old" and episode_id in two_track.OLD_SUSPECT,
                }
            )
    return two_track.sealed(
        {
            "schema": two_track.SCHEMA,
            "episodes": episodes,
            "tracks": {
                "TODAY30": {"episode_ids": [e["id"] for e in episodes if e["cohort"] == "today"]},
                "ALL59": {"episode_ids": [e["id"] for e in episodes]},
            },
        }
    )


def test_recipes_start_independently_from_official_base():
    value = manifest()
    for track in two_track.TRACKS:
        recipe = two_track.recipe(track, value["sha256"])
        assert recipe["initialization"] == "official_pi05_base_new_optimizer"
        assert recipe["snapshots"] == [250, 500, 1000, 2000]
        assert recipe["batch_size"] == 2
        assert recipe["phase_fractions"] == {"uniform": 0.70, "close": 0.15, "release": 0.15}
        assert recipe["robot_motion_authorized"] is False


def test_track_membership_and_balanced_phase_schedule():
    value = manifest()
    today = two_track.sample_schedule(value, "TODAY30", 4000)
    all_data = two_track.sample_schedule(value, "ALL59", 4000)
    assert len(today["train_episode_ids"]) == 30
    assert all(two_track.TODAY_SESSION in row["episode"] for row in today["records"])
    assert len(all_data["train_episode_ids"]) == 59
    assert Counter(row["phase"] for row in today["records"]) == {
        "uniform": 2800,
        "close": 600,
        "release": 600,
    }
    for schedule, episode_count in ((today, 30), (all_data, 59)):
        for phase in ("uniform", "close", "release"):
            counts = Counter(row["episode"] for row in schedule["records"] if row["phase"] == phase)
            assert len(counts) == episode_count
            assert max(counts.values()) - min(counts.values()) <= 1
        assert all(
            abs(row["frame"] - row["event_frame"]) <= 30
            for row in schedule["records"]
            if row["event_frame"] is not None
        )
        json.dumps(two_track.coverage(schedule, len(schedule["records"])), allow_nan=False)


def test_recovery_episode_events_are_preserved_in_sampling():
    value = manifest()
    schedule = two_track.sample_schedule(value, "TODAY30", 4000)
    for episode_id in ("episode_000002", "episode_000016"):
        uid = two_track.uid(two_track.TODAY_SESSION, episode_id)
        close = {row["event_frame"] for row in schedule["records"] if row["episode"] == uid and row["phase"] == "close"}
        release = {
            row["event_frame"] for row in schedule["records"] if row["episode"] == uid and row["phase"] == "release"
        }
        assert close == {200, 500}
        assert release == {350, 800}


def test_transition_metrics_match_multiple_cycles_without_reordering():
    truth = np.zeros(1000, dtype=np.float32)
    truth[200:350] = 1
    truth[500:800] = 1
    predicted = np.zeros_like(truth)
    predicted[205:345] = 1
    predicted[490:805] = 1
    result = two_track_eval.transition_metrics(predicted, truth, [200, 500], [350, 800])
    assert result["matched_close"] == [[205, 200], [490, 500]]
    assert result["matched_release"] == [[345, 350], [805, 800]]
    assert result["missed_close"] == result["missed_release"] == 0
    assert result["extra_close"] == result["extra_release"] == 0


def test_condition_intents_filters_pulse_and_preserves_sustained_transition():
    raw = np.zeros(30, dtype=np.float32)
    raw[3:5] = 1  # 67 ms pulse at 30 Hz.
    raw[10:20] = 1
    filtered, events = two_track_eval.condition_intents(raw, min_hold_s=0.2)
    assert events == [
        {"event": "close", "time_s": 16 / 30},
        {"event": "open", "time_s": 26 / 30},
    ]
    assert not filtered[:16].any()
    assert filtered[16:26].all()
