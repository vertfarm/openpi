# HV1 현재 상태

최종 갱신 2026-09-14 · 규약은 [AGENTS.md](../../AGENTS.md)

이 파일은 **지금 사실인 것**만 담는다. 매 세션 끝에 덮어쓴다.

## 한 줄 요약

**첫 실기 rollout이 ARM 후 45.0초를 제어 fault 없이 돌고 pilot 상한에서 멈췄다.**
`events.jsonl` 실측: target 1,347개 @ 30.0Hz, 추론 451회 @ 10.02Hz, 기록된 fault는
`pilot duration reached` 하나 — 즉 **보호 중단 그 자체**이고 그 외 제어 fault는 0이다.
파이프라인·안전층·정지 경로·앙상블 전부 작동한다.

> 이전 판에 있던 "420초 무결점 완주"는 틀렸다. 348.5초는 **프로세스 수명**이고
> (정지 후 status 283건이 더 찍힌다 — 터미널이 안 끝나 보이던 이유), 실제 무장 구간은
> 45.0초다. `DEPLOYMENT.md`대로 **45초 상한은 보호 중단이지 작업 성공이 아니다.**

그런데 **정책이 시각을 쓰지 않는다** — 검증 6에피소드의 state와
이미지를 6x6으로 교차해도 grasp intent가 36조합 중 34개에서 1.0이다.

이유는 학습 신호에 있다. 시연들의 **시작 직후 접근 방향이 서로 코사인 0.89~0.99**로
거의 동일하다. 물체 위치는 흩어져 있었지만(파지 자세끼리 중앙 0.355 rad) 초기
접근 방향은 같았으므로 state만 보는 것이 학습 목표상 최적이었다. 물체 위치는
**언제 멈추는지**만 바꾸고 그 정보는 접근 후반부 이미지에만 있는데, 30에피소드로는
그것을 배우지 못했다.

결과: 평균 거리만큼 가고 목표를 **1.5배 지나쳐** 멈춘다(이동 1.060 rad 대
파지 거리 중앙 0.699 rad). 파지 자세에 없으니 intent가 오르지 않고 그리퍼도
안 닫힌다. **intent 헤드 자체는 정상이다** — 파지 자세를 주면 1.0을 낸다.

코드로 고치려는 시도 **셋 다 실패했다** — V1(손실 가중), V2(state 노이즈),
V3(둘의 결합). V2는 그룹별로 gap 부호가 갈려 한때 "일부 반응"으로 읽혔지만,
지표의 노이즈 바닥을 재보니 **자기 자신의 샘플링 흔들림 안**이었고 정책만
25배 불안정해졌다. 측정 방법과 전체 수치는 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

**코드 레버는 소진됐다. 남은 길은 수집 재설계다** — 처음부터 근본 해법으로
지목돼 있던 것이다. 목표치: `|d15|` 대 거리 상관 `|r| > 0.6`,
초기 접근 방향 코사인 중앙 **≤ 0.3** (현재 0.89~0.99).

## 코드 상태

| 항목 | 값 |
|---|---|
| canonical 브랜치 | `codex/hv1-vla-workflow-20260909` (remote `origin`) |
| 회귀 테스트 | 182개 (`.venv/bin/python -B -m pytest examples/hv1/tests -q`, 약 15초) |
| lint | `ruff check --select F,E4,E7,E9,I` + `ruff format --check` 통과 (`ros/**` 제외) |
| 진입점 | 10모듈 / 28명령 (캠페인 5 + 운영 4 + 합성 1). `test_artifacts.py`가 고정 |
| ROS `core.py` 소스 SHA | `59775ee4d0826b09f9b296014269d6ce55271644513dddc17f64e99b13c9566c` |
| repo ↔ `vla_ws` 사본 | `core.py`·`node.py`·`guardian.py` byte 일치. `operator.py`·`__init__.py`는 **빈 줄 1개 차이, AST 동일** — 추적하지 말 것 |
| `stash@{0}` | `field-20260910-pre-ff-snapshot` — `df`, `rosgraph.png` 포함. 미정리 |

숫자는 낡는다. **지금 값은 아래로 직접 확인한다.**

```bash
git log --oneline -1 && git status --short          # HEAD, 미커밋
.venv/bin/python -B -m pytest examples/hv1/tests -q  # 테스트
df -h ~/workspace | tail -1                          # 디스크 (게이트 15GiB)
ps -eo etime,cmd | grep -E "[d]eploy_server|[g]uardian|[v]la_client"   # 떠 있는 것
curl -s http://127.0.0.1:8000/health                 # 서버가 있다면 해시 3종
sha256sum examples/hv1/ros/keti_humanoid_inference/keti_humanoid_inference/core.py
```

`core.py` 해시는 **repo · `vla_ws` 실행 사본 · 서버 `/health`의 `adapter_core_sha256`
세 곳이 일치**해야 한다. 어긋나면 재빌드 없이 실기를 시작하지 않는다.

### 디스크 정책

게이트는 `artifacts.py`의 `MIN_FREE` = **15 GiB**(2026-09-11 감독자 승인으로 50에서 낮춤).
학습 1회가 그 위에 40 GiB를 예약하므로 **여유 55 GiB 미만이면 학습이 거부된다.**

정리는 반드시 `pipeline_run._safe_generated_cleanup`으로 한다 — 각 캠페인 `cleanup/`에
무엇을·왜·어떤 증거로 지웠는지 해시와 함께 남는다. 2026-09-11에 이 방식으로
138.6 GiB를 회수했다(은퇴 캠페인 가중치 88.9 + FT 스냅샷 19.6 + optimizer 상태 30.1).

- **지우지 말 것**: `cache/openpi`·`cache/huggingface` — 모든 학습이 여기서 pi05_base를 읽는다.
- **회수량을 `du`로 재지 말 것.** `cache/uv`는 `du` 7.9G인데 `df`는 0.3G만 늘었다 —
  uv가 venv로 하드링크를 건다. `df`나 `st_nlink`로 확인한다.
- 원본 녹화(`keti_humanoid_ros2/datasets/`, 302 MB)는 **유일하게 대체 불가능한 자산**이다.

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

6. **시작 자세는 오른손만이 아니다.** 수집 시 아래가 모두 갖춰져 있었다.

   | 대상 | 학습 첫 프레임 기준값 |
   |---|---|
   | 오른손 8축 | `[1.569, -0.612, 0.831, -0.612, 0.831, 1.552, -0.612, 0.831]` |
   | **왼손 8축** | `[-1.572, -0.612, 0.831, -0.611, 0.831, -1.551, -0.611, 0.831]` (오른손의 거울상) |
   | 왼팔 7축 | `[-0.367, -0.822, -0.448, -1.490, -0.439, -0.851, 0.005]` |
   | 머리 2축 | `[0.0, 0.0]` |

   **왼손도 `mode 2` → `set_open 0.6`을 해야 한다.** 왼손은 모델 state에 들어가지
   않지만 `hand_l` 카메라가 왼손바닥에 장착돼 있고 이는 필수 입력 3개 중 하나다.
   손가락이 벌어진 상태와 편 상태는 화면 내용이 다르다.

7. **손목 카메라 프레임률은 조명에 묶여 있다.** `auto_exposure=3`(Aperture Priority)
   에서 장면이 어두우면 노출이 길어지고 프레임률이 떨어진다. 지원 모드는 Manual과
   Aperture Priority 둘뿐이라 shutter priority 같은 절충안이 없다.

   | 설정 | fps | mean | **std** | p05 | p95 |
   |---|---|---|---|---|---|
   | 학습 데이터 (hand_r) | ~29 | 126.0 | **50.9** | 32.5 | 198.8 |
   | 자동 노출 (현재) | 8~14 | 105.3 | **50.5** | 28.0 | 180.0 |
   | manual + brightness 200 | 30.0 | 120.7 | **21.5** | 95.0 | 153.0 |

   **수동으로 30fps를 강제하지 말 것.** 평균은 맞출 수 있지만 표준편차가 절반 이하로
   떨어지고 어두운 영역이 32 → 95로 들려 검은색이 날아간다. digital `brightness`는
   게인이 아니라 오프셋이라 대비를 만들지 못한다. 시간 이격을 줄이려다 훨씬 큰
   광학 이격을 만든다. **해법은 작업 영역 조명이다** — 밝아지면 자동 노출이 짧은
   노출을 골라 30fps와 진짜 대비를 동시에 회복한다. 필요 증광량은 대략 3~4배.

8. **카메라 기하 이격은 없다. 저장 형상으로 판단하지 말 것.**
   `frame_builder._image()`가 수집 때 모든 카메라를 640×480으로 리사이즈하고,
   `deploy_server.prepare_images`가 배포 때 동일하게 한다. **두 경로가 같다.**
   손목 카메라의 가로 1.78배 왜곡은 학습 데이터에도 그대로 들어 있고 모델은 그
   상태로 학습했다.

   따라서 학습 mp4가 640×480이고 라이브 토픽이 480×640인 것은 **정상이며**,
   저장 형상은 원본 형상을 말해주지 않는다. 2026-09-11에 이것을 기하 이격으로
   오판해 카메라를 물리적으로 돌릴 뻔했다. 판단은 반드시 **파이프라인 통과 후
   224×224**로 하라. `metadata.json`의 `"rotate"`도 카메라 노드 설정이 아니라
   레코더 자신의 추가 회전값(기본 0)이다.

   점검 도구: `examples/hv1/tools/check_camera_input.py` — 학습/라이브/겹침을
   224×224로 나란히 내고 빨간 매트 경계를 수치로 비교한다.

9. **정책의 접근 방향은 시연과 일치한다.** shadow에서 로봇은 안 움직이지만 정책이
   제안한 목표는 기록된다. 제안 목표에서 실측을 뺀 값을 학습 데이터의 시작 직후
   조작자 의도(command − state)와 비교한 결과 **코사인 유사도 +0.880, 7축 전부
   부호 일치**다. 크기는 정책이 약 2배(0.0146 대 0.0075 rad) 적극적이다.
   정지 장면에서 얻을 수 있는 가장 강한 긍정 신호이며, "그리퍼를 안 닫는다"는
   소극적 확인을 넘어선다. 단 제안의 시간적 표준편차가 평균 크기의 136%로,
   같은 입력을 반복해 보는 정지 장면 특유의 흔들림이 있다.

10. **파지 순간 팔은 정지해 있다.** 61개 에피소드에서 측정한 결과, 조작자는 팔을
   세운 뒤에 손을 닫는다. 파지 프레임 이후 0.333초 동안의 팔 관절 변위는
   중앙 0.0004 rad(0.023°), p90 0.0104 rad, 최대 0.0232 rad(1.3°, 손끝 약 1cm)다.
   파지 직전 0.5초 최대 속도 중앙 0.081 rad/s에서 직후 0.013 rad/s로 감속한다.
   따라서 필터가 만드는 0.3~0.4초 닫기 지연은 시연 데이터 기준으로 거의 무해하다.
   단, 이는 시연의 팔 움직임이다. 실기에서는 모델이 팔을 몰기 때문에 같은
   감속·정지 패턴을 재현해야 성립한다.

11. **정책은 시각을 쓰지 않는다. 학습 신호가 그렇게 만들었다.**
   검증 6에피소드(today_fixed6)의 파지 직후 state와 이미지를 6x6 교차한 결과
   grasp intent가 36조합 중 34개에서 1.0이다. 대각(일치) 중앙 1.021,
   비대각(이미지 완전 불일치) 중앙 1.021로 **차이가 없다.** 라이브에서도
   전혀 다른 학습 프레임을 넣은 출력이 같은 입력 재요청과 구별되지 않았고
   (코사인 +0.94 대 +0.90), state를 0.15 rad 바꾼 영향이 이미지 교체의 5~10배였다.

   원인은 데이터 다양성 부족이 아니다. 물체 위치는 흩어져 있었다 — 파지 순간
   팔 자세끼리 중앙 0.355 rad, 최대 0.940 rad. 그런데 **시작 직후 접근 방향이
   서로 코사인 0.89~0.99로 거의 동일**하다. 초기 구간에서는 이미지를 봐도 얻을
   정보가 없으므로 state만 보는 것이 최적이다. 물체 위치는 **언제 멈추는지**만
   바꾸고 그 정보는 접근 후반부 이미지에만 있다.

12. **그리퍼가 안 닫히는 것은 별개 문제가 아니다.** `intent` 헤드는 정상이다 —
   `ep012` 파지 프레임 222부터 1.01~1.04를 낸다(오프라인 사이드카와 일치).
   24회 반복 추론에서도 편차가 없다. 다만 intent는 **팔이 파지 자세에 있을 때**
   오르고, 실기에서는 그 자세를 지나쳤으므로 0.02에 머물렀다.

   2026-09-11에 파지 프레임 219만 찍어보고 "intent 헤드가 고장"이라고 오판했다.
   3프레임 뒤에 오른다. 시계열로 보라.

13. **실기 fault는 전부 계측·경합 문제였고 정책 문제는 없었다.** 2026-09-11 기록:
   `enable_command_output:=false`로 6회 실행(팔이 물리적으로 연결 안 됨),
   `target step` 3회(정지 장면 통계로 잡은 한계), MQTT 2회(age가 -0.0006초인
   서브틱 경합). 전부 고쳤다. 마지막 실행(`/tmp/hv1_rollout_013`)은 무장 45.0초 동안
   제어 fault 0건이고, 기록된 fault 1건은 pilot 상한에 의한 보호 중단이다.

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

`min_hold`를 올리면 닫기가 그만큼 늦어지지만, 확정 사실 10에 따라 파지 순간 팔이
정지해 있으므로 **0.4초까지는 여유가 있다.** 오검출 마진이 더 필요하면
hold 0.2~0.3을 선택해도 된다. 반대로 hold 0.5는 최대 오차가 0.6~0.8초로 올라가
측정된 정지 구간을 넘어서므로 권하지 않는다.

### 라이브 재검증 (2026-09-11, 실제 로봇 관측 60초 × 5 — 독립 3세션)

| 실행 | safe | gripper_event | 원시 임계교차 | closed_frac | hand_r age p50/p95 | 스톨 |
|---|---|---|---|---|---|---|
| 오전 TODAY30-1000 @0.1 | ✅ | **0** | **1** | 0.0017 | 70.9 / 71.3 ms | 9 |
| 오후 TODAY30-1000 @0.1 | ✅ | **0** | **1** | 0.0017 | 51.0 / 101.1 ms | 7 |
| 오전 ALL59-2000 @0.2 | ✅ | **0** | 26 | 0.0329 | 79.0 / 79.4 ms | 8 |
| 오후 ALL59-2000 @0.2 | ✅ | **0** | 24 | 0.0345 | 68.1 / 118.2 ms | 1 |
| **장면 정렬 후 TODAY30-1000 @0.1** | ✅ | **0** | **1** | **0.0017** | **62.4 / 62.9 ms** | **4** |

다섯 번 모두 `gripper_event` 0건, fault 없음, 로봇 명령 0건, 추론률 9.98Hz.
원시 intent는 1.0을 넘겨 임계값을 교차했지만 dwell이 전부 흡수했다.
노이즈 1회가 fault로 이어지던 경로가 실제로 막힌 것을 확인했다.

**TODAY30-1000의 원시 신호가 압도적으로 깨끗하다** — 임계교차 1회 대 24~26회,
`closed_frac` 20배 차이. 카메라 조건이 다른 두 세션에서 소수점까지 재현됐다.
오프라인 스윕 1위와 일치하므로 **1순위를 TODAY30-1000 @ hold 0.1로 확정한다.**
단 이 수치는 정지 장면 오검출률이며 파지 능력 순위가 아니다(확정 사실 3).

오후 실행에서 오른손 카메라가 8.3Hz까지 떨어져 `hand_r` age p95가 101~118ms로
악화됐다(skew 한도 100ms를 넘나든다). 순위는 바뀌지 않았지만 실기 전에
조명으로 회복해야 한다 — 확정 사실 7 참조.

**장면을 학습 조건에 맞춘 뒤(마지막 행) 그리퍼 거동은 완전히 동일했고
관측 품질만 좋아졌다.** `closed_fraction`이 세 번 모두 0.0017(1795개 중 3개)로
소수점 넷째 자리까지 재현됐다. 개선된 것은 `hand_r` age p95 101 → 62.9ms와
스톨 7 → 4다. 장면 정렬은 그리퍼 판단이 아니라 관측 안정성에 기여한다.

### 서버 기동 시 주의

`--registry`는 2026-09-11부터 argparse 단계에서 **필수**다. C/F-5000만 허용했던
legacy 분기는 삭제했으므로, 예외 없이 다음을 붙인다.

```
--registry <campaign>/checkpoint_registry.json
```

registry의 `schema`가 `hv1_two_track_v1`이 아니면 스냅샷을 열기 전에 거부한다.

## 사용하지 말 것

**`TODAY30_FT`, `ALL59_FT` 스냅샷 4개 (2026-09-10 야간 생성).**

야간 세션이 금지된 추가학습을 실행했다. 촉발 원인은 두 가지 자체 발명 기준이었다 —
`max_abs_close_error <= 0.3` (주어지지 않은 값)과 "8개 체크포인트 전부가 같은 hold에서
통과"(배포는 하나만 하므로 구조적으로 충족 불가). `TODAY30-1000 @ 0.1`이 0.333s로
1프레임 차이로 탈락해 실패 판정이 났다.

결과적으로 FT 계열의 전환 4종 0인 최선은 `TODAY30_FT-1000 @ 0.2`의 **0.433s**로,
기존 `TODAY30-1000 @ 0.1`의 **0.333s**보다 나쁘다. 약 20GiB와 GPU 16분을 쓰고 후퇴했다.

## 실기 활성화 — 2026-09-11 구축 완료

`checkpoint_registry.json`은 여전히 `SHADOW_ONLY`이지만, live 전환을 막던
공백은 채워졌다. 계약은 [DEPLOYMENT.md](DEPLOYMENT.md) 참조.

| 항목 | 상태 |
|---|---|
| `/hv1_vla/guardian` 발행자 | ✅ `guardian.py`. 실기에서 **20.000Hz** 발행 확인 |
| 소프트웨어 정지 | ✅ `node.py`의 `hold_here` + `/hv1_vla/stop` |
| 필드 프로파일 | ✅ `hv1-vla-runtime/two-track-20260910-r3/deploy_field_profile_20260911.json` |
| 물리 정지 | 감독자가 파지하는 전원 차단 E-stop. **소프트웨어에서 호출 불가** |

**안전 층 순서**

```
자동:  LiveGate 7종 + live_health 6종 + 스케줄러 3종 → hold_here (제자리 정지, 토크 유지)
수동1: guardian 터미널 Ctrl-C → 100ms 내 같은 정지
수동2: /hv1_vla/stop 호출 → 같은 정지
최후:  E-stop 전원 차단 → 낙하 (브레이크 없는 QDD. 이 장비는 낙하 허용)
```

**정지가 토크를 빼지 않는 이유.** 팔이 브레이크 없는 QDD라 전원 차단은 정지가
아니라 낙하다. 그래서 fault 대응은 현재 실측 위치를 목표로 발행해 토크를 유지한다.
명령만 끊으면 마지막 setpoint가 한 스텝 앞서 있어 그만큼 더 가는데, `hold_here`가
그 잔여 이동을 없앤다.

**guardian이 정직한 이유.** 로봇 토픽에 퍼블리셔를 하나도 만들지 않는다 — 팔에
영향을 줄 수 있는 유일한 경로가 조용해지는 것이다. 소프트웨어로 증명할 수 없는
`workspace_clear`, `hardware_watchdog_ready`, `release_allowed`는 **기본 false**이고
명령행에서 명시해야 켜진다. 상수 true를 박으면 검증되지 않은 조건이 기록된 보증으로
바뀌어 guardian이 없느니만 못해진다.

`qualify_proposal`은 제안 id를 **내용에서 재계산**한다. 실행기가 준 id를 믿으면
나중에 그 이름으로 무엇을 실행하든 승인하는 셈이다.

### 필드 프로파일 값과 근거

| 항목 | 값 | 근거 |
|---|---|---|
| `max_step` | **0.041** rad | 정책 스텝의 **p99**. 실질 제한 |
| `max_velocity` | 1.4 rad/s | `max_step ÷ 30.2ms` + 여유. 스텝이 통과한 걸 속도가 재거부하지 않게 |
| `max_acceleration` | 50 rad/s² | 한 주기에 `max_velocity` 도달 시 46.4. 백스톱 (정책 p99 35.5) |
| `joint_min/max` | ±1.5708 | URDF 하드웨어 한계 |
| `start_q` | `[0.0711, 0.3023, 0.1041, 1.4013, 0.2629, 0.4531, 0.1898]` | 30에피소드 첫 프레임 중앙값 |
| `start_tolerance` | `[0.06, 0.06, 0.12, 0.08, 0.08, 0.10, 0.07]` | 같은 프레임들의 산포 |
| `max_tracking_error` | 0.15 rad | **잠정.** 움직여야 측정됨 |

**`max_step` p99는 상위 1%에서 fault를 낸다.** 30Hz에서 약 3.3초마다 한 번이다.
첫 시도에서는 의도된 동작이지만 과제를 끝까지 보기 어려우면 관측 최대 0.065로 올린다.

**실측값을 시연 데이터에서 유도하지 말 것.** `LiveGate` docstring —
*"Physical limits are required inputs, never inferred from demonstrations."*
`start_q`·`start_tolerance`는 안전 한계가 아니라 자격 있는 시작 자세라 예외다.

## 미결 결정 — 감독자 몫

1. **모델이 파지 순간 팔을 제대로 세우는가.** 닫기 지연 자체의 위험은 확정 사실 10으로
   정량화됐다 — 시연에서 팔은 파지 순간 정지해 있고 0.333초간 최대 1.3°만 움직인다.
   남은 위험은 지연이 아니라, 폐루프에서 모델이 그 감속·정지를 재현하는지다.
   이건 제한 rollout에서만 확인된다. rollout 중 파지 직전 팔 속도를 기록해
   위 시연 분포(직후 0.5초 최대 속도 중앙 0.013 rad/s)와 대조할 것.
2. ~~배포 체크포인트 최종 선택~~ — TODAY30-1000 @ hold 0.1 확정.
3. FT 스냅샷 4개 삭제 여부. 삭제 시 약 20GiB 회수. 게이트 여유가 9GiB뿐이다.
4. `stash@{0}` 처리 — `rosgraph.png`는 문서 자산으로 보존 결정됨, `df`는 잡파일.
5. **state 지름길의 다음 대응** — V2/V3 결과는 아래 표와 같다. 추가 sigma sweep을
   자동 실행하지 않았으며, 수집 재설계와 함께 감독자가 다음 실험을 선택해야 한다.

### 실행 순서

세 터미널이 필요하다. guardian 터미널의 Ctrl-C가 정지 손잡이다.

```
1  deploy_server  --snapshot snapshots/TODAY30/step_001000 --registry checkpoint_registry.json
2  python3 -m keti_humanoid_inference.guardian --profile <프로파일> --mqtt-host 192.168.0.142
                  --workspace-clear --hardware-watchdog [--release-allowed]
3  ros2 run keti_humanoid_inference vla_client --mode live --port 8000 --grip-min-hold 0.1
                  --profile <프로파일> --mqtt-host 192.168.0.142 --output <새 디렉터리>
```

**재빌드는 하지 않는다.** `--symlink-install`이라 `vla_ws/src`에 복사하면 즉시
반영되며, guardian은 진입점 없이 `python3 -m`으로 실행한다. 2026-09-11 기준
`build/`·`install/`은 무변경으로 유지되고 있다.
백업: `hv1-vla-runtime/vla_ws-backup-20260911-135931.tgz`.

`--release-allowed`는 트레이 조건을 정한 뒤에만 붙인다. 없으면 release 시점에
fault가 난다.

## 다음 단계

파이프라인은 끝났다. 남은 것은 **정책이 이미지를 보게 만드는 것**이고,
코드로 하는 시도는 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)에서 셋 다 실패했다.

1. **데이터 수집 설계를 바꾼다.** ← 여기다. 물체 위치를 흩는 것만으로는 부족하다.
   지금도 파지 자세는 중앙 0.355 rad 떨어져 있는데 **시작 접근 방향이 코사인
   0.89~0.99로 같다.** 물체가 *어디로 갈지*를 바꿔야 이미지가 초기부터 쓸모가 생긴다.

   | 지표 | 현재 | 목표 |
   |---|---|---|
   | 초기 접근 방향 코사인 중앙 | 0.89~0.99 | **≤ 0.3** |
   | `|d15|` 대 파지 거리 상관 `|r|` | +0.14 | **> 0.6** |

   **수집 전에 기존 59에피소드로 검증할 수 있다** — 배치 후보를 정하면 그 배치가
   위 두 수치를 만족하는지 시뮬레이션해볼 것. 찍고 나서 알면 늦다.

2. **손목 카메라 조명 회복** — 작업 영역을 3~4배 밝게. `auto_exposure`를 유지한 채
   `/dev/video8`이 30fps 근처인지 확인한다(확정 사실 7). 수동 노출로 fps를
   맞추지 말 것 — 대비가 무너진다.

3. **재수집 후 재학습** — 같은 명령으로 돌아간다. 판정은 teacher-forced 점수가
   아니라 `evaluate-cross-modal`의 gap과 방향 코사인이다.

## 머신 이전 — 새 PC로 옮길 때

**최소 이전과 오프라인 검증은 완료됐다.** 현장과 새 PC가 서로 다른 서브넷이라
Windows를 신뢰 경계가 아닌 전송 브리지로만 사용했고, 각 홉에서 SHA-256을 다시
검사했다. 접속 정보와 비밀번호는 repo에 넣지 않았다.

**옮겨야 하는 것은 생각보다 작다.** 92 GB 중 대체 불가능한 것은 302 MB뿐이다.

| 대상 | 크기 | 방법 |
|---|---|---|
| **원본 녹화** `keti_humanoid_ros2/datasets/` | **302 MB** | **반드시 복사. 유일한 대체 불가 자산** |
| 코드 | 5.6 MB | `git clone` — repo에 다 있다 |
| 배포 스냅샷 `snapshots/TODAY30/step_001000` | 4.9 GB | 복사(재학습보다 싸다) |
| 캠페인 메타 `manifest/registry/assets/export.json/evaluations` | ~20 MB | 복사. 스냅샷 신뢰 체인이 여기 걸려 있다 |
| 컨테이너 이미지 `keti-humanoid:jazzy` | 6.1 GB | `docker save`/`load` 또는 재빌드 |
| ROS overlay `vla_ws` | 235 MB | **재빌드.** 소스는 `examples/hv1/ros/`에 있다 |
| `.venv` | 7.8 GB | **재생성.** `uv.lock` 고정 |
| `cache/` | 24 GB | 재다운로드 가능하나 pi05_base 11.6 GB는 복사가 빠르다 |
| 나머지 스냅샷·로그 | ~45 GB | 선택. 없어도 현재 배포는 된다 |

**최소 이전 ≈ 12 GB**, 전체 충실 이전 ≈ 92 GB.

현재 검증 결과:

- 원본 59 episode가 manifest와 전수 일치했다. manifest SHA-256은
  `8064bb4c017a224b2f7ce7965e16119ee5efd70809b65439f273d996af7938e5`다.
- OpenPI는 원격 canonical 브랜치에서 재생성했고 JAX가 새 GPU를 `CudaDevice(id=0)`으로
  잡는다. CPU 회귀 182개와 ROS 제외 lint/format 검사가 통과했다.
- 새 PC를 구성한 저자의 `keti_humanoid_ros2` checkout은 현장 기준·GitHub `main`과
  같은 `bfb84cb`다. `.entrypoint.sh`의 설명 주석 8줄만 빠졌던 로컬 변경은 원본으로
  복원했다. 재생성 가능한 `ros2/vla_ws/`는 삭제하지 않고 이 PC의 Git local exclude로
  분류했으며, ROS2와 OpenPI 작업트리는 모두 clean이다.
- 저자가 만든 9/14 Docker 이미지를 기본 `keti-humanoid:jazzy`로 유지한다. 현장에서
  가져온 9/9 이미지는 `keti-humanoid:jazzy-field-20260909`로 별도 보존했다. 새 이미지는
  현장 이미지보다 RealSense 관련 ROS 패키지 3종과 broadcaster 2종이 더 있다.
- 기존 `hand_ws`·`kh_ws` 위에 `vla_ws`를 symlink build했다. repo source, `vla_ws`
  source, 실제 import된 `core.py`가 모두
  `59775ee4d0826b09f9b296014269d6ce55271644513dddc17f64e99b13c9566c`다.
- TODAY30-1000 deploy server의 `/health`가 snapshot/stat/contract/core 해시를 모두
  통과했다. 저장 관측 4 frame의 loopback smoke는 62~75 ms였고 로봇 명령은 0건이다.
  결과는 `~/workspace/hv1-new-pc-validation-20260914/deploy-smoke-260910-000003.json`,
  SHA-256은 `1a6e3c9fd2487558c327970d3fc908dfb810ecd4805f1fe33ed54957d7e2f19c`다.
  검증 후 deploy server를 정상 종료해 GPU lock과 port 8000을 비웠다.
- 원격 정리 후 같은 검증을 다시 실행했다. 회귀 182개와 원본 59 episode 전수 해시가
  다시 통과했고, JAX는 RTX 5090을 `CudaDevice(id=0)`으로 잡았다. 저장 관측 4 frame의
  왕복은 62~65 ms, 로봇 명령은 0건이었다. 결과는
  `~/workspace/hv1-new-pc-validation-20260914/recheck-20260914-154722.json`, SHA-256은
  `0d89cd6e3582e2b6d8bade02f878d754f8b9012438079d1e16224e8c654e5351`다. 종료 후
  GPU lock을 실제로 다시 획득할 수 있었고 port 8000과 관련 프로세스가 비어 있음을
  확인했다.

남은 게이트:

- 새 PC에서 로봇망 MQTT가 아직 도달하지 않는다. ROS domain 10 노드, guardian,
  `vla_client`는 실행하지 않았다. 로봇 실험실 네트워크에 연결하고 감독자가 있을 때만
  아래 순서를 계속한다.
- 격리 domain 213의 ROS 경계 test는 기존 test fixture가 새 필수 인자
  `ensemble_decay`를 만들지 않아 2건 실패했다. 빌드와 실행 import는 성공했으며,
  금지된 `ros/**`를 이번 이전에서 고치지 않았다.

실기 전 확인 순서:

1. `git clone` 후 `uv sync` → `pytest examples/hv1/tests -q`가 통과하는가
2. 원본 302 MB의 파일 해시가 이전과 같은가 (`pipeline verify`로 manifest 대조)
3. `vla_ws` 재빌드 후 `core.py` 해시가 repo와 **byte 일치**하는가
4. `deploy_server` 기동 → `/health`의 `adapter_core_sha256`가 위 둘과 일치하는가
5. ROS domain·MQTT 도달 확인 후에야 **guardian → vla_client** 순으로 실기

**접속 정보는 이 저장소에 넣지 않는다.** GitHub로 푸시되는 공개 이력이다.
호스트·계정·비밀번호는 git 밖(`~/workspace/`의 로컬 노트나 사내 채널)에 둔다.

### 실기 기준선 (2026-09-11 측정)

| 실행 | 앙상블 | 지속 | measured 누적 | 순변위 | 스텝 p99 | 종료 |
|---|---|---|---|---|---|---|
| 007 | OFF | 4.1s | 4.178 | 0.210 | 0.0779 | target step |
| 010 | 0.3 | 5.5s | 0.920 | 0.299 | 0.0138 | MQTT 서브틱 경합 |
| 013 | 0.3 | 45.0s | 5.876 | 1.060 | — | **pilot 상한(보호 중단)** |

지속은 ARM부터 정지까지다. 013은 상한에 걸려 멈춘 것이지 작업을 끝낸 것이 아니다.

앙상블은 진동을 5.6배 줄이고 순변위를 1.4배 늘렸다. 007에서 진동이
0.0095 -> 0.0171로 증폭되던 것이 010에서 0.0055 -> 0.0047로 안정됐다.

## 산출물

경로는 `~/workspace/hv1-vla-runtime/two-track-20260910-r3/` 기준.

| 파일 | SHA-256 |
|---|---|
| `ablation_result.json` | `cf6556ae5b40c36dc46d090fd5a38f415c85344829f4bd847621a108a909c80e` |
| `evaluations/overnight_grasp_intent_report_v1.json` | `8a6ab05ea140f33093e4e5bdb23e619e4f51d3fb82de540387956b6aee25d65c` |
| `evaluations/grasp_filter_sweep_v1.json` | `981b9b474a0b037024abf96126e2f2878362390a2dc903fe67b4820d2b8fd908` |
| `evaluations/grasp_filter_finetune_sweep_v1.json` | `023b7c7a7a617f8592a161af8074ebe0fbebe767e8cda9f8837b4c325f397fbb` |
| `shadow/hv1_prep_today30_1000_h010_r1/summary.json` | `eca3cf9c95583c4676bea9485f778c1043cf5ea17227e16b9fefb00eaef67b7c` |
| `shadow/hv1_prep_all59_2000_h020_r1/summary.json` | `f4661dd16736dae1185692659a7375e01b32bb0d502b2c531331a3509cac0c94` |

부수 디렉터리: `evaluations/intent_series/`, `static_intent_series/`,
`source_identity_prebuild/`. 시작 자세가 올바른 shadow 로그는
`shadow/hv1_shadow_*_correcthand_*` 8개이며,
`hv1_shadow_all59_001000_correcthand_r1`은 `response` 이벤트가 없어 r2를 쓴다.
