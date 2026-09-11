# HV1 training pipeline

Source recordings to a registered `SHADOW_ONLY` checkpoint. This is the live
procedure; [DEPLOYMENT.md](DEPLOYMENT.md) takes over from the checkpoint.

The campaign it describes is the two-track campaign of 2026-09-10, which is the
one the field is running. Its definition lives in `pipeline.py` constants
(`TRACKS`, `OLD_SESSION`, `TODAY_SESSION`, the excluded and diagnostic sets) -
the next campaign edits those and this document, not a new module family. The
`hv1_two_track_v1` schema string stays as it is, because every manifest,
snapshot and registry on disk carries it.

Retired campaigns keep their own records under [docs/](docs/); their modules are
deleted and those commands no longer run.

## The two tracks

This campaign trains two independent policies from the official `pi05_base`:

- `TODAY30`: the 30 task episodes recorded on September 10.
- `ALL59`: the 29 task episodes recorded on September 9 plus `TODAY30`.

The September 9 dummy episodes `000000`, `000001` and the separate incomplete
`000033` are excluded. Multi-cycle recovery demonstrations are preserved. Both
tracks use their own training-only normalization statistics and the same fixed,
episode-balanced 70/15/15 uniform/close/release sampling schedule.

## Safety boundary

Training, evaluation, registration, and server startup never authorize robot
motion. Registered models remain `SHADOW_ONLY`. The policy publishes only
`/kh/upper_body/action/joint`; it must never publish `joint_command` or a
`_mirror` topic. The current direct-joint controller path does not call IK,
joint-limit, or self-collision checks, so the deployment bridge's verified
limits, step/velocity/acceleration/freshness checks, guardian, ownership checks,
watchdog, and qualified stop service are mandatory before supervised live use.

## Reproducible run

Use a new campaign directory outside the source dataset and repository. The
commands below are CPU-only until `pipeline_run`; `--allow-gpu-run` never grants
robot-control authority.

```bash
python -B -m examples.hv1.pipeline prepare \
  --old /home/keti/workspace/keti_humanoid_ros2/datasets/keti_humanoid_data_260909 \
  --today /home/keti/workspace/keti_humanoid_ros2/datasets/keti_humanoid_data_260910 \
  --campaign "$CAMPAIGN"
python -B -m examples.hv1.pipeline export --campaign "$CAMPAIGN"
python -B -m examples.hv1.pipeline stats --campaign "$CAMPAIGN" --track TODAY30
python -B -m examples.hv1.pipeline stats --campaign "$CAMPAIGN" --track ALL59
python -B -m examples.hv1.pipeline schedules --campaign "$CAMPAIGN"
python -B -m examples.hv1.pipeline_run \
  --campaign "$CAMPAIGN" \
  --deadline 2026-09-10T23:00:00+09:00 \
  --reviewer codex_offline_pipeline_20260910 \
  --allow-gpu-run
```

The approved target is 2,000 updates per track, with inference snapshots at
250, 500, 1,000, and 2,000 updates. `campaign_result.json`,
`candidate_comparison.json`, and `checkpoint_registry.json` are the completion
records. A checkpoint enters the registry only after BF16 CPU round-trip, file
hash verification, GPU reload, and fixed diagnostic replay.

## Does the checkpoint use its cameras?

Diagnostic replay is teacher-forced: it hands the policy the state that already
implies the answer, so a policy deciding from the arm alone passes it. On
2026-09-11 one did - and only failed in the field, after the arm had moved.

Run this on every candidate before trusting a replay score. It crosses each
diagnostic episode's state against every diagnostic episode's images, holding
the sampling noise fixed so the only thing that moves is the input:

```bash
python -B -m examples.hv1.pipeline_eval evaluate-cross-modal \
  --campaign "$CAMPAIGN" --snapshot "$CAMPAIGN/snapshots/TODAY30/step_001000" \
  --allow-gpu-run
```

It writes `evaluations/cross_modal/<experiment>_<step>_<group>_p<offset>.json`
and reports two numbers. `intent_scene_gap` is the diagonal intent median minus
the off-diagonal one: near zero means swapping in a completely different scene
did not change the grasp decision. `direction_cosine_median` near 1.0 means the
intended arm direction did not turn either.

**Read `diagonal_intent_median` first.** The gap only means something while that
is high. The policy raises intent a few frames *after* the recorded grasp - +1 to
+7 across the twelve diagnostic episodes - so anchoring on the grasp frame itself
compares two near-zero numbers and reports a small gap for the wrong reason.
`--frame-offset` defaults to 8 to clear that, and the offset is part of the
filename because a different anchor is a different measurement.

Measured 2026-09-11: both deployment candidates, both diagnostic groups, gap
within +-0.0005 and direction cosine 0.996-0.999. Neither uses its cameras.

**No threshold is applied and none is stored.** The command records the numbers;
a supervisor reads them. It needs the GPU lock, so stop `deploy_server` first.

## Ablations

Experiments that ask *why* a checkpoint behaves as it does live in
`pipeline.ABLATIONS`. Each is a separate experiment name whose entry lists the
only fields it changes, so the comparison against its track stays one-variable,
and a test enforces that. **Never answer such a question by editing a track's
recipe**: `pipeline_config.configure` re-derives a snapshot's recipe and compares
it, so changing `recipe("TODAY30")` makes `deploy_server` refuse TODAY30-1000 -
the checkpoint the field runs.

```bash
python -B -m examples.hv1.pipeline ablation-schedules --campaign "$CAMPAIGN"
python -B -m examples.hv1.pipeline_train --campaign "$CAMPAIGN" \
  --experiment TODAY30_CLOSE45 --deadline <future ISO8601 with offset> --allow-gpu-run
python -B -m examples.hv1.pipeline_eval evaluate-cross-modal \
  --campaign "$CAMPAIGN" --snapshot "$CAMPAIGN/snapshots/TODAY30_CLOSE45/step_002000" \
  --allow-gpu-run
```

Ablations keep one snapshot rather than four. 2,000 updates measured 1,006 s, so
time is not the constraint; a snapshot is 4.9 GiB and disk is.

Some ablations add a knob no track recipe has - `ABLATION_ONLY_FIELDS` lists
which, and nothing else may be added, because a track's recipe has to keep the
exact fields its snapshots were written with.

`state_noise_sigma` is one of those. It is applied inside `make_loader` on a copy
of the data config, prepended **before** `HV1Inputs`: that transform computes the
action delta against the state, so noising afterwards would anchor input and
target differently and become label noise the model cannot undo. The config
`configure` returns - the one deployment shares - never carries it, and
`deploy_server` reaches the policy through `registered_config`, never
`make_loader`. Do not move this into `transforms.py`; that would noise live
inference.

## Latest ROS interface

- observation: `/kh/upper_body/observation/state/joint_states`
- right-arm absolute target: `/kh/upper_body/action/joint`
- right-hand observation: `/kdex_3f/right/rel_angle/joint_state`
- close: `/kdex_3f/right/grasp`
- open: `/kdex_3f/right/set_open`

The bridge maps names explicitly and refuses missing, duplicate, or foreign
publishers. Start the inference server on loopback, run recorded-input smoke,
then run ROS shadow. Model replacement must leave the bridge disarmed.
