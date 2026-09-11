# HV1 현재 상태

최종 갱신 2026-09-11 · 감사 세션(현장 리눅스) · 규약은 [AGENTS.md](../../AGENTS.md)

이 파일은 **지금 사실인 것**만 담는다. 매 세션 끝에 덮어쓴다.

## 한 줄 요약

그리퍼는 학습에 성공했고, intent 필터로 정지 오검출이 0이 됐다.
필터가 만드는 닫기 지연도 시연 데이터 기준으로는 무해한 수준으로 확인됐다.
후보 2개는 실제 로봇 관측으로 라이브 재검증까지 마쳤다.
**그리퍼·모델 쪽 미해결 이슈는 없다.**

남은 것은 전부 **실기 활성화 계약**이다 — 독립 guardian 노드와 물리 정지 서비스가
아직 존재하지 않고, 필드 프로파일 실측값 8개가 비어 있다. 코드로 해결되는 항목이
아니며 측정·구현·승인이 필요하다.

## 코드 상태

| 항목 | 값 |
|---|---|
| canonical HEAD | `1075f41` (현장·remote 동일, 트리 clean) |
| 회귀 테스트 | 136개 통과 |
| ROS `core.py` 소스 SHA | `ecbdd5fbfe48676b26bb7858aedb3edd92757faacc4f941d09d9e1646c7837b7` |
| repo ↔ `vla_ws` 사본 | 일치 확인됨 (2026-09-11 감사) |
| `stash@{0}` | `field-20260910-pre-ff-snapshot` — `df`, `rosgraph.png` 포함. 미정리 |
| 디스크 여유 | 58GiB (게이트 50GiB) |
| 실행 코드 신원 | `pkg prefix` → `vla_ws/install`, import → `vla_ws/build`, `core.py` 해시가 서버 `adapter_core_sha256`와 일치 (2026-09-11 확인) |

## 확정된 사실 — 재검증하지 말 것

1. **그리퍼는 학습됐다.** held-out 검증 12에피소드 teacher-forced에서 close 시점
   오차 중앙 0.00~0.10s, `missed_close`는 사실상 0. 파지 시점 판단은 문제가 아니다.

2. **문제는 `extra_close`였고 원인은 샘플링이다.** `right_grasp_intent`는 Bernoulli
   타깃인데 flow-matching 헤드가 denoise 10에서 매 추론 독립 샘플링한다. 이봉 분포에서
   표본이 드물게 반대 봉우리에 떨어진다. `GripEdges`가 임계값 교차 1회를 비가역
   사건으로 처리해 노이즈 1회가 fault가 됐다. 학습 실패가 아니다.

3. **shadow는 파지 능력을 측정할 수 없다.** 로봇이 안 움직여 장면이 시작 상태에
   머물고 손이 물체에 도달하지 않으므로 "계속 열림"이 정답이다. shadow는 정지 장면
   오검출률 전용 지표다. 체크포인트를 shadow의 open 유지 시간으로 순위 매기지 않는다.

4. **손 시작 자세 계약.** `mode 2`(tripod, spread ±90° = 1.571 rad) → `set_open 0.6`.
   그때 `/kdex_3f/right/rel_angle/joint_state`가
   `[1.569, -0.612, 0.831, -0.612, 0.831, 1.552, -0.612, 0.831]` 부근이어야 한다.
   59에피소드 전부 이 값에서 시작했다. `norm_stats`의 joint_10 std는 0.006,
   joint_30 std는 0.001이므로 준비를 빠뜨리면 각각 -262σ, -1413σ 입력이 된다.
   2026-09-10 초기 shadow 8회가 이 상태였고 그 결과는 폐기됐다.

5. **구현된 보호장치.** intent 필터(close 0.7 / open 0.3, hysteresis + dwell,
   단일 cycle, Grasp 미완료 중 release 유보)와 hand OOD fail-closed 게이트
   (`norm_stats` q01/q99, ±0.05 rad 여유).

6. **파지 순간 팔은 정지해 있다.** 61개 에피소드에서 측정한 결과, 조작자는 팔을
   세운 뒤에 손을 닫는다. 파지 프레임 이후 0.333초 동안의 팔 관절 변위는
   중앙 0.0004 rad(0.023°), p90 0.0104 rad, 최대 0.0232 rad(1.3°, 손끝 약 1cm)다.
   파지 직전 0.5초 최대 속도 중앙 0.081 rad/s에서 직후 0.013 rad/s로 감속한다.
   따라서 필터가 만드는 0.3~0.4초 닫기 지연은 시연 데이터 기준으로 거의 무해하다.
   단, 이는 시연의 팔 움직임이다. 실기에서는 모델이 팔을 몰기 때문에 같은
   감속·정지 패턴을 재현해야 성립한다.

## 배포 후보 — 감사 검증 완료

`grasp_filter_sweep_v1.json` 32행 **전부** `shadow_false_close = 0`.
정지 장면 오검출은 필터만으로 해결됐다.

전환 4종(`missed_close`/`extra_close`/`missed_release`/`extra_release`)이 모두 0인 행:

| 순위 | 체크포인트 | min_hold | 중앙 오차 | 최대 오차 |
|---|---|---|---|---|
| 1 | **TODAY30-1000** | **0.1s** | +0.200s | **0.333s** |
| 2 | ALL59-2000 | 0.2s | +0.267s | 0.400s |
| 3 | ALL59-2000 | 0.3s | +0.367s | 0.400s |
| 4 | TODAY30-1000 | 0.2s | +0.300s | 0.433s |

ALL59-2000은 hold 0.2·0.3 양쪽에서 안정적이고 학습 데이터가 넓어 2순위로 둔다.

`min_hold`를 올리면 닫기가 그만큼 늦어지지만, 확정 사실 6에 따라 파지 순간 팔이
정지해 있으므로 **0.4초까지는 여유가 있다.** 오검출 마진이 더 필요하면
hold 0.2~0.3을 선택해도 된다. 반대로 hold 0.5는 최대 오차가 0.6~0.8초로 올라가
측정된 정지 구간을 넘어서므로 권하지 않는다.

### 라이브 재검증 (2026-09-11, 실제 로봇 관측 60초 × 2)

| | TODAY30-1000 @ 0.1 | ALL59-2000 @ 0.2 |
|---|---|---|
| shadow_safe / fault / 로봇 명령 | ✅ / 없음 / 0 | ✅ / 없음 / 0 |
| 추론 / 목표 | 599 / 1795 | 599 / 1795 |
| **gripper_event** | **0건** | **0건** |
| 원시 임계 교차 | **1회** | **26회** |
| closed_fraction | 0.0017 | 0.0329 |
| 관절 스텝 p95 / max (rad) | 0.0265 / 0.0605 | 0.0239 / 0.0550 |
| 지연 p95 client/server (ms) | 70.1 / 67.1 | 70.1 / 67.0 |

원시 intent는 양쪽 다 1.0을 넘겨 임계값을 교차했지만 dwell이 전부 흡수했다.
노이즈 1회가 fault로 이어지던 경로가 실제로 막힌 것을 확인했다.

**TODAY30-1000의 원시 신호가 훨씬 깨끗하다** — 같은 정지 장면에서 임계 교차가
1회 대 26회다. 오프라인 스윕 순위를 라이브 관측이 독립적으로 재확인했다.
단 이 수치는 정지 장면 오검출률이며 파지 능력 순위가 아니다(확정 사실 3).

### 서버 기동 시 주의

`deploy_server`는 `--registry`가 없으면
`first deployment is restricted to verified C/F-5000`으로 **거부한다.**
두 트랙 스냅샷을 쓰려면 반드시 다음을 붙인다.

```
--registry <campaign>/checkpoint_registry.json
```

## 사용하지 말 것

**`TODAY30_FT`, `ALL59_FT` 스냅샷 4개 (2026-09-10 야간 생성).**

야간 세션이 금지된 추가학습을 실행했다. 촉발 원인은 두 가지 자체 발명 기준이었다 —
`max_abs_close_error <= 0.3` (주어지지 않은 값)과 "8개 체크포인트 전부가 같은 hold에서
통과"(배포는 하나만 하므로 구조적으로 충족 불가). `TODAY30-1000 @ 0.1`이 0.333s로
1프레임 차이로 탈락해 실패 판정이 났다.

결과적으로 FT 계열의 전환 4종 0인 최선은 `TODAY30_FT-1000 @ 0.2`의 **0.433s**로,
기존 `TODAY30-1000 @ 0.1`의 **0.333s**보다 나쁘다. 약 20GiB와 GPU 16분을 쓰고 후퇴했다.

## 실기 활성화 공백 — 2026-09-11 실측 확인

`checkpoint_registry.json`의 모든 항목이 `status: SHADOW_ONLY`,
`robot_motion_authorized: false`다. 아래가 채워지기 전에는 live로 전환할 수 없고,
`LiveGate`가 구조적으로 거부한다. 상세 계약은 [DEPLOYMENT.md](DEPLOYMENT.md)의
"실기 활성화에 필요한 외부 계약"에 있다.

| 항목 | 현재 상태 | 성격 |
|---|---|---|
| `/hv1_vla/guardian` 발행자 | **존재하지 않음.** 코드에는 구독·검증 측(`node.py`, `core.py`)만 있고 발행하는 노드가 없다. 테스트의 `guardian_node="/fake"`는 실기 사용 금지 | 구현 |
| 물리 정지 서비스 | **존재하지 않음.** 라이브 ROS 그래프의 서비스 109개 중 stop/estop/halt/emergency 해당 없음 | 구현 + 배선 |
| `approved`, `review_id`, `qualification_evidence` | 빈 값 | 승인 |
| `joint_min`, `joint_max`, `max_step`, `max_tracking_error`, `max_velocity`, `max_acceleration`, `start_q`, `start_tolerance` | 전부 `null` | 실측 |

**실측 8개를 시연 데이터에서 유도하지 말 것.** `LiveGate` docstring이 명시한다 —
*"Physical limits are required inputs, never inferred from demonstrations."*
데모 범위는 안전 한계가 아니다.

## 미결 결정 — 감독자 몫

1. **모델이 파지 순간 팔을 제대로 세우는가.** 닫기 지연 자체의 위험은 확정 사실 6으로
   정량화됐다 — 시연에서 팔은 파지 순간 정지해 있고 0.333초간 최대 1.3°만 움직인다.
   남은 위험은 지연이 아니라, 폐루프에서 모델이 그 감속·정지를 재현하는지다.
   이건 제한 rollout에서만 확인된다. rollout 중 파지 직전 팔 속도를 기록해
   위 시연 분포(직후 0.5초 최대 속도 중앙 0.013 rad/s)와 대조할 것.
2. 배포 체크포인트 최종 선택 (위 표 1~2순위).
3. FT 스냅샷 4개 삭제 여부. 삭제 시 약 20GiB 회수. 게이트 여유가 9GiB뿐이다.
4. `stash@{0}` 처리 — `rosgraph.png`는 문서 자산으로 보존 결정됨, `df`는 잡파일.

## 다음 단계

1. ~~손 자세 `mode 2` → `set_open 0.6`, `rel_angle` 기준값 대조~~ — 완료 (2026-09-11)
2. ~~OOD 게이트 통과 확인~~ — 완료. 단 `joint_30`이 원시 q01(1.5509)보다 낮은
   1.5463이라 ±0.05 마진 덕에 통과했다. 마진을 줄이면 걸린다.
3. ~~재빌드된 코드로 무송신 live shadow~~ — 완료, 후보 2개 모두 통과
4. **후보 선택** — 현재 근거로는 TODAY30-1000 @ hold 0.1이 1순위
5. **실기 활성화 공백 해소** ← 지금 여기. guardian 구현, 물리 정지 서비스 배선,
   실측 8개, 승인. 위 "실기 활성화 공백" 표 참조
6. 제한 실기 rollout — 파지 직전·직후 팔 속도를 기록해 확정 사실 6의 시연 분포와
   대조한다. 폐루프에서 감속·정지가 재현되는지가 닫기 지연 허용의 전제다.

## 보류 중인 정리 작업

rollout과 그리퍼 필터 회귀가 끝난 뒤에 착수한다. 지금 하지 않는다.

- `overnight_*` 6파일 삭제 (import 그래프상 leaf island로 확인됨).
  `tests/test_overnight.py`와 `test_policy_wire.py`가 함께 사라지는 것을 감수할지가 판단점.
- `readapt_*` / `two_track_*` 통합 — recipe 차이를 JSON으로 내리기
- 진입점 20개 → 5개 (`prepare`/`train`/`eval`/`serve`/`shadow_eval`)
- 날짜 박힌 `.md` 3개를 `hv1-vla-runtime/logs/`로 이동

## 산출물

경로는 `~/workspace/hv1-vla-runtime/two-track-20260910-r3/` 기준.

| 파일 | SHA-256 |
|---|---|
| `evaluations/overnight_grasp_intent_report_v1.json` | `8a6ab05ea140f33093e4e5bdb23e619e4f51d3fb82de540387956b6aee25d65c` |
| `evaluations/grasp_filter_sweep_v1.json` | `981b9b474a0b037024abf96126e2f2878362390a2dc903fe67b4820d2b8fd908` |
| `evaluations/grasp_filter_finetune_sweep_v1.json` | `023b7c7a7a617f8592a161af8074ebe0fbebe767e8cda9f8837b4c325f397fbb` |
| `shadow/hv1_prep_today30_1000_h010_r1/summary.json` | `eca3cf9c95583c4676bea9485f778c1043cf5ea17227e16b9fefb00eaef67b7c` |
| `shadow/hv1_prep_all59_2000_h020_r1/summary.json` | `f4661dd16736dae1185692659a7375e01b32bb0d502b2c531331a3509cac0c94` |

부수 디렉터리: `evaluations/intent_series/`, `static_intent_series/`,
`source_identity_prebuild/`. 시작 자세가 올바른 shadow 로그는
`shadow/hv1_shadow_*_correcthand_*` 8개이며,
`hv1_shadow_all59_001000_correcthand_r1`은 `response` 이벤트가 없어 r2를 쓴다.
