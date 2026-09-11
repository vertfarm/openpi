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

It writes `evaluations/cross_modal/<experiment>_<step>_<group>.json` and reports
two numbers. `intent_scene_gap` is the diagonal intent median minus the
off-diagonal one: near zero means swapping in a completely different scene did
not change the grasp decision. `direction_cosine_median` near 1.0 means the
intended arm direction did not turn either. TODAY30-1000 scored a gap of 0.0
with both medians at 1.021.

**No threshold is applied and none is stored.** The command records the numbers;
a supervisor reads them. It needs the GPU lock, so stop `deploy_server` first.

## Latest ROS interface

- observation: `/kh/upper_body/observation/state/joint_states`
- right-arm absolute target: `/kh/upper_body/action/joint`
- right-hand observation: `/kdex_3f/right/rel_angle/joint_state`
- close: `/kdex_3f/right/grasp`
- open: `/kdex_3f/right/set_open`

The bridge maps names explicitly and refuses missing, duplicate, or foreign
publishers. Start the inference server on loopback, run recorded-input smoke,
then run ROS shadow. Model replacement must leave the bridge disarmed.
