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

## `hv1_augmented_v1` Real2Sim2Real 경로

기존 `hv1_two_track_v1` schema와 실물 LeRobot export는 그대로 둔다.

**이 경로는 `examples/hv1/augmented.py`에 있고 `pipeline.py`에는 없다.**
`pipeline_config.configure`가 배포된 snapshot의 recipe를 `pipeline.recipe`에서 다시
유도해 비교하므로, 증강 실험 코드가 그 파일에 있으면 실험 중 편집 하나가
`deploy_server`로 하여금 현장이 돌리는 checkpoint를 거부하게 만들 수 있다. 두
경로는 분리돼 있고, 그 분리는 테스트로 강제된다
(`test_deployment_pipeline_carries_no_augmented_surface`).

새 경로는 실물 row의 metadata wrapper와 sim campaign의 local-prefetched tar shard를
하나의 hash-sealed index로 묶는다. NAS path를 trainer가 직접 읽는 구성은 허용하지
않는다. Index는 각 tar member의 byte offset/size도 봉인하므로 2 GiB shard를 sample마다
재스캔하지 않고 local file seek로 JSON과 JPEG 세 장을 읽는다.

### 캠페인 범위는 sim profile이 정한다

공정 목록, 모델별 source/render 혼합, macrocycle 크기, update 수는 **sim campaign이
`manifests/campaign_config.json`에 봉인한 profile**에서 읽는다. 여기에 사본을 두지
않는다. 공정이나 render style을 추가하는 것은 그 profile의 변경이며 이 코드는 바뀌지
않는다. Index는 profile의 SHA-256을 봉인하므로 indexing 이후 profile이 바뀌면 load가
거부된다.

v1 profile은 **1공정, RTX 단독, M0 대 M2, confirm 28,000 updates**다. 이전 초안의
2,000 updates는 59-episode 실물 캠페인에서 복사된 값이었고, 그대로 두면 50,000개를
만들어 1,500개만 읽게 된다. sim repo의 `validate_campaign_config`가 생성량과 학습
소비량의 비율(epoch)을 검사해 이 불균형을 거부한다.

### 임바디먼트 리비전은 섞이지 않는다

모든 row가 `embodiment_revision`을 들고 다니고, index는 sim profile이 선언한
리비전과 다른 row가 섞이면 거부한다. 기계 변경은 도달 workspace, IK branch,
pose hull을 한꺼번에 바꾸므로 변경 전후 데이터는 서로 다른 로봇을 기술한다.
혼합이 필요하면 `--allow-mixed-embodiment`로 **의도적으로만** 한다. index는
`embodiment_revision_counts`와 `embodiment_mixed`를 봉인한다.

팔꿈치 기구 교체가 예정돼 있으므로(도면 약 152.59°, 현재 URDF placeholder
±90°를 초과) 교체 전에 수집·생성한 데이터는 교체 후 데이터와 자동으로 섞이지
않는다.

Real wrapper는 `hv1_augmented_real_metadata_v1` schema, `real_export_sha256`과
record 목록을 가진 sealed JSON이다. 각 record에는 `dataset_index`, `task_id`,
`phase`, `language`, `source_domain=real`, `render_style=real`, `synthetic=false`,
`sampleable`, `success`/`corrected_recovery`, `quality_weight`, camera calibration,
timestamp와 provenance hash가 있어야 한다.

> **선행조건.** schedule은 profile의 모든 `(source, task, phase, render_style, paired)`
> 조합마다 비어 있지 않은 stratum을 요구하고, 하나라도 비면 fail-closed한다.
> v1이 1공정인 이유가 이것이다. 실물 59 episode에 frame 단위 7-phase 라벨이 아직
> 없으므로, **`wrap-real` 전에 phase 라벨을 붙여야 한다.** `native.py`가 grasp/release
> frame을 이미 뽑으므로 그 경계에서 7-phase를 유도하는 규칙이 다음 작업 단위다.

```bash
python -B -m examples.hv1.augmented wrap-real \
  --real-export "$REAL_CAMPAIGN/export/export.json" \
  --annotations "$AUGMENTED_ROOT/real_annotations.json" \
  --output "$AUGMENTED_ROOT/real_metadata.json"

python -B -m examples.hv1.augmented index \
  --real-export "$REAL_CAMPAIGN/export/export.json" \
  --real-metadata "$AUGMENTED_ROOT/real_metadata.json" \
  --sim-campaign "$SIM_CAMPAIGN" \
  --output "$AUGMENTED_ROOT/index.json" \
  --allow-synthetic

python -B -m examples.hv1.augmented status \
  --index "$AUGMENTED_ROOT/index.json" --allow-synthetic

python -B -m examples.hv1.augmented schedules \
  --index "$AUGMENTED_ROOT/index.json" \
  --output-dir "$AUGMENTED_ROOT/schedules" \
  --stage screen --allow-synthetic
python -B -m examples.hv1.augmented schedules \
  --index "$AUGMENTED_ROOT/index.json" \
  --output-dir "$AUGMENTED_ROOT/schedules" \
  --stage confirm --allow-synthetic
```

Screening은 profile의 `screening_models` 각각 seed 42, 500 updates다. 확정 단계는
`finalists` 각각 seed 42/43/44다. Batch size는 2이며 각 run은 자기 schedule의
정확한 mixture에서 normalization을 먼저 계산한다.

```bash
python -B -m examples.hv1.pipeline_train \
  --augmented-root "$AUGMENTED_ROOT" --experiment M2 \
  --stage screen --seed 42 --stats-only --allow-synthetic

python -B -m examples.hv1.pipeline_train \
  --augmented-root "$AUGMENTED_ROOT" --experiment M2 \
  --stage screen --seed 42 \
  --deadline 2026-09-21T12:00:00+09:00 \
  --allow-synthetic --allow-gpu-run
```

`--allow-synthetic`은 index, schedule, statistics와 training 네 단계 모두에 필요하다.
Schedule은 매 macrocycle(100 sample)에서 profile의 공정에 균등 배정하고, source와
render 몫은 profile의 **정수 count**라 반올림 오차가 없다. v1의 M2는 real 50 /
sim_physics 50, render는 real 50 / rtx 50, pair-group 20개다. Kinematic은 reach,
pre-grasp, transport에만 들어간다. Loader는 schedule cursor를 optimizer checkpoint와
함께 검증하므로 `--resume`이 다른 index, schedule 또는 normalization에 붙을 수 없다.

grasp intent(action 채널 7)는 `[0, 1]`의 이진 계단이다. 실물이 operator protocol에서
그렇게 유도되기 때문이며, sim이 연속값이나 음수를 내면 그 채널만으로 도메인을 알 수
있게 된다. `validate_augmented_sample`이 값역을 검사한다.

M0/M2 snapshot은 자동으로 실기 registry에 들어가지 않는다. 고정 sim nominal,
geometry holdout, 공정별 성공률, 그리고 **실물 진단 episode에서 측정한**
cross-modal 민감도가 별도로 구현·통과된 뒤 기존 `SHADOW_ONLY` 등록 절차로 넘겨야
한다. sim 렌더에서 영상을 쓰는 정책이 실제 카메라는 여전히 무시할 수 있으므로,
데이터 쪽 counterfactual gate가 이 측정을 대신하지 않는다.

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

Run the whole queue with one command. It writes any missing schedule, trains,
evaluates teacher-forced, measures the cross-modal gap on both diagnostic
groups, and reclaims the optimizer state, one experiment at a time because there
is a single GPU lock. The comparison table lands in `ablation_result.json`.

```bash
python -B -m examples.hv1.pipeline_run --campaign "$CAMPAIGN" \
  --ablations --deadline <future ISO8601 with offset> --allow-gpu-run
```

It resumes: an experiment is skipped once its result, teacher-forced evaluation,
both cross-modal records and its restart cleanup are on disk, so an interrupted
run continues by being started again. Ablations are never registered - they are
not deployment candidates and there is no `register` step here.

Single steps, when you want one experiment rather than the queue:

```bash
python -B -m examples.hv1.pipeline ablation-schedules --campaign "$CAMPAIGN"
python -B -m examples.hv1.pipeline_train --campaign "$CAMPAIGN" \
  --experiment TODAY30_CLOSE45 --deadline <future ISO8601 with offset> --allow-gpu-run
python -B -m examples.hv1.pipeline_eval evaluate-cross-modal \
  --campaign "$CAMPAIGN" --snapshot "$CAMPAIGN/snapshots/TODAY30_CLOSE45/step_002000" \
  --allow-gpu-run
```

Running it standalone skips the cleanup that `--ablations` does, so about 30 GiB
of optimizer state stays under `restarts/`.

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
