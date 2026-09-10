from examples.hv1 import shadow_eval


def test_shadow_summary_reports_no_robot_output_and_motion_metrics():
    events = [
        {"event": "startup", "monotonic": 10.0},
        {
            "event": "request",
            "monotonic": 10.1,
            "source_ages_s": {"arm": 0.01, "hand": 0.02, "head": 0.03, "hand_l": 0.04, "hand_r": 0.05},
        },
        {
            "event": "response",
            "monotonic": 10.2,
            "client_ms": 70.0,
            "server_ms": 60.0,
            "preprocess_ms": 7.0,
        },
        {"event": "schedule", "monotonic": 10.3, "skipped": 4, "first_model_index": 4},
        {
            "event": "target",
            "monotonic": 10.4,
            "action": [0.0] * 7 + [0.0],
            "measured": [0.0] * 7,
            "gripper_event": "open",
            "lateness_ms": 1.0,
            "sent": False,
        },
        {
            "event": "target",
            "monotonic": 10.5,
            "action": [0.1] + [0.0] * 6 + [1.0],
            "measured": [0.0] * 7,
            "gripper_event": "close",
            "lateness_ms": 2.0,
            "sent": False,
        },
        {"event": "shutdown", "monotonic": 11.0, "robot_commands_sent": 0, "inferences": 1},
    ]

    summary = shadow_eval.summarize(events)

    assert summary["shadow_safe"] is True
    assert summary["final_inferences"] == 1
    assert summary["targets"]["sent_true"] == 0
    assert summary["targets"]["max_abs_step_rad"]["max"] == 0.1
    assert summary["targets"]["gripper_events"] == {"close": 1, "open": 1}
    assert summary["targets"]["gripper_intent"]["initial_closed"] is False
    assert summary["targets"]["gripper_intent"]["closed_target_fraction"] == 0.5
    assert summary["targets"]["gripper_intent"]["threshold_crossings"] == {
        "open_to_close": 1,
        "close_to_open": 0,
    }
    assert summary["targets"]["gripper_event_offsets_s"] == [
        {"event": "open", "offset_s": 0.40000000000000036},
        {"event": "close", "offset_s": 0.5},
    ]
    assert summary["targets"]["first"]["max_abs_delta_rad"] == 0.0
    assert summary["source_age_ms"]["hand_l"]["max"] == 40.0


def test_shadow_summary_marks_fault_or_sent_target_unsafe():
    events = [
        {"event": "startup", "monotonic": 1.0},
        {
            "event": "target",
            "monotonic": 1.1,
            "action": [0.0] * 8,
            "measured": [0.0] * 7,
            "sent": True,
        },
        {"event": "fault", "monotonic": 1.2, "reason": "test"},
        {"event": "shutdown", "monotonic": 1.3, "robot_commands_sent": 1, "inferences": 0},
    ]

    assert shadow_eval.summarize(events)["shadow_safe"] is False
