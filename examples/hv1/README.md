# HV1 — 오른팔 VLA 파일럿

KETI 휴머노이드 HV1의 오른팔로 은색 실린더를 집어 트레이에 놓는다.
문종술 선임의 ROS2·recorder·3D 마우스는 그대로 두고, **저장된 에피소드 이후**를 담당한다.

```text
원본 HDF5 + 3개 영상  →  검수·고정 manifest  →  LeRobot export
                            ↓
                    OpenPI π0.5 학습  →  BF16 snapshot · 평가 · registry
                            ↓ loopback HTTP
                    ROS shadow  /  감독하 실기 executor
```

## 처음 오셨다면 — 이 순서로 15분

1. **[STATUS.md](STATUS.md)의 「한 줄 요약」과 「확정된 사실」** — 지금 무엇이 사실이고
   무엇을 다시 재지 말아야 하는지. 이 프로젝트에서 가장 값진 문서다.
2. **[../../AGENTS.md](../../AGENTS.md)** — 건드리면 안 되는 것들. 짧다.
3. 아래 「지금 상태를 한 문단으로」와 「실행 가능한 것 전부」.
4. 실제로 뭔가 돌릴 때 [TRAINING.md](TRAINING.md)(데이터→모델) 또는
   [DEPLOYMENT.md](DEPLOYMENT.md)(모델→로봇).

`docs/`에는 [EXPERIMENTS.md](docs/EXPERIMENTS.md)(시각 미사용을 어떻게 쟀고 무엇이
실패했는지 — 같은 시도를 반복하지 않으려면 읽을 것)와 은퇴한 절차의 기록들이 있다.

## 지금 상태를 한 문단으로

파이프라인·안전층·정지 경로는 전부 작동한다. 첫 실기 rollout이 무장 45초를 제어 fault
없이 돌았다. **그런데 정책이 카메라를 쓰지 않는다** — 장면을 통째로 바꿔도 판단이
0.1% 미만 움직인다. 원인은 학습 신호다: 시연들의 초기 접근 방향이 서로 코사인
0.89~0.99로 같아서, 팔 자세만 보는 것이 학습 목표상 최적이었다. 코드로 고치려는
시도 셋(손실 가중, state 노이즈, 둘의 결합)은 실측으로 실패 확정됐다. **남은 길은 수집
재설계다** — 물체 위치가 *언제 멈추는지*만이 아니라 *어디로 가는지*를 바꾸도록.

## 실행 가능한 것 전부

실행 가능한 모듈은 **10개뿐이며 아래가 전부다.** 나머지 파일은 라이브러리다
(`test_artifacts.py`가 이 목록을 고정한다 — `main()`이 생기면 테스트가 깨진다).

**캠페인 경로 5개** — 데이터에서 배포까지의 본선:

| 할 일 | 실행 모듈 | 상세 |
|---|---|---|
| 세션 검수·고정 manifest·export·통계 | `examples.hv1.pipeline` | [TRAINING.md](TRAINING.md) |
| smoke → 트랙 학습·평가 순차 실행 / 절제 실험 | `examples.hv1.pipeline_run` | 같은 문서 |
| 개별 학습 | `examples.hv1.pipeline_train` | 복구·개별 검증용 |
| 평가·후보 등록 / **교차 모달 점검** | `examples.hv1.pipeline_eval` | 같은 문서의 「카메라를 쓰는가」 |
| 검수된 모델의 loopback HTTP 추론 | `examples.hv1.deploy_server` | [DEPLOYMENT.md](DEPLOYMENT.md) |

**운영 도구 4개** — 본선에 끼지 않는 단독 도구:

| 할 일 | 실행 모듈 |
|---|---|
| 기록 관측으로 배포 서버 smoke (ROS 없음) | `examples.hv1.deploy_smoke` |
| shadow/실기 rollout 로그 요약 (영상 미적재) | `examples.hv1.shadow_eval` |
| ROS/URDF 증거 감사 → DRAFT 프로파일 | `examples.hv1.source_contract` |
| 원본 대 export 전수 비교 | `examples.hv1.verify_export` |

**합성 검수 1개** — `examples.hv1.cli` (`demo`/`export`/`scan`/`list`/`review`/
`manifest`/`serve-review`). profile 기반 일반 검수용이며 캠페인 경로가 아니다.

**ROS 컨테이너 전용** — `ros/keti_humanoid_inference`(shadow·실기 경계)와
`tools/`(카메라 입력 비교, 시작자세 복귀). 이 둘만 로봇에 접근한다.

## 바꾸려는 내용별 수정 위치

| 변경 대상 | 관리 원본 |
|---|---|
| 원본 HDF5·영상 구조 / 학습 입출력 계약 | `native.py` / `transforms.py` |
| 시연 선정·트랙 분할·샘플링 / 레시피 / 절제 실험 | `pipeline.py` — `recipe()`가 학습 설정의 기준 |
| OpenPI 설정 / 명시적 로컬 dataset 로딩 | `pipeline_config.py` |
| optimizer 실행·재시작 cursor / 학습 전용 변형 | `pipeline_train.py` |
| 오프라인 지표·SHADOW_ONLY 등록 | `pipeline_eval.py` |
| 캠페인과 무관한 정책 지표 | `metrics.py` — 교차 모달·전이 지표 |
| 순차 실행·완료 상태 보관 정책 | `pipeline_run.py` |
| JSON·해시·디스크 게이트 | `artifacts.py` — 표준 라이브러리만 사용 |
| BF16 저장·재로딩·파일 검증 | `checkpoints.py` — 모든 학습/배포 경로가 공유 |
| 추론 전처리·HTTP / 로봇 안전 경계 | `deploy_server.py` / ROS 패키지의 `core.py`, `node.py` |

## 세 가지 함정 — 고치기 전에 읽을 것

1. **모듈 이름에 캠페인 이름을 넣지 않는다.** `overnight_*` → `readapt_*` →
   `two_track_*` 3세대가 쌓였다가 2026-09-11에 정리됐다. 다음 캠페인은 새 계열이
   아니라 `pipeline.py`의 상수를 고치는 것이다.
2. **디스크의 데이터 이름은 바꾸지 않는다.** `SCHEMA`는 `hv1_two_track_v1`로 남는다.
   기존 manifest·snapshot·registry 전부에 박혀 있고 `deploy_server`가 이것으로
   registry를 검사한다. **모듈 이름은 코드, 스키마는 데이터다.**
3. **트랙 레시피를 수정하지 않는다.** `configure`가 스냅샷에 저장된 레시피와
   비교하므로, `recipe("TODAY30")`을 바꾸면 `deploy_server`가 **배포 중인 체크포인트를
   거부한다.** 실험은 `pipeline.ABLATIONS`에 새 이름으로 추가한다.

## 유지할 계약

- state 15 = 오른팔 실측 7 + 오른손 실측 8, action 8 = 오른팔 관절 목표 7 + 그리퍼 의도.
- 팔 7축은 state에 대한 delta, grasp intent는 절대값(`delta_state_indices` 끝이 −1).
- head/hand_l/hand_r 모두 유지. 카메라 dropout 없음.
- mode 2, wrap 닫기, narrow 0.6 열기. 왼팔 pose 3 조망 자세. **왼손도 준비 자세가 필요하다.**
- 그리퍼는 손가락 8축이 아니라 **스칼라 1개**로 제어한다. 모델은 타이밍만 정하고
  손 모양은 고정 스크립트다.
- 트랙은 공식 `pi05_base`에서 독립 초기화하고, 트랙별 학습 데이터로만 통계를 재계산한다.
- 미검수 데이터·변경된 원본/통계·불완전 snapshot은 차단. 기존 원본과 모델은 자동 정리하지 않는다.
- 모델 등록/교체는 ARM이 아니다. shadow 지표는 실물 성공률이 아니며,
  물리 E-stop만으로 guardian·freshness·소유권 검사를 대신하지 않는다.

계약을 바꿀 때는 이전 세션·manifest를 덮어쓰지 말고 새 버전으로 검수한다.

## 환경·검증

기존 현장 ML `.venv`와 고정 `uv.lock`을 사용한다. ROS 환경과 섞거나 다시 설치하지 않는다.
ROS 패키지는 컨테이너에서 독립 설치되므로 ML 공통 모듈에 의존시키지 않는다.

```bash
cd ~/workspace/openpi-hv1
JAX_PLATFORMS=cpu .venv/bin/python -B -m pytest examples/hv1/tests -q -p no:cacheprovider
.venv/bin/python -B -m ruff check examples/hv1 --select F,E4,E7,E9,I
.venv/bin/python -B -m ruff format --check examples/hv1
```

CPU 테스트 통과는 새 데이터 GPU 학습이나 실기 성공의 증거가 아니다.
`ruff format`은 `ros/**`에 적용하지 않는다 — `core.py` 바이트가 바뀌면
서버·클라이언트 해시 합의가 깨져 양쪽 재시작이 강제된다.
