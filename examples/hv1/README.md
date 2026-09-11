# HV1 데이터 → OpenPI π0.5 → 배포

문종술 선임의 ROS2·recorder·3D 마우스는 유지하고 **저장된 에피소드 이후**를 담당한다.
실물 원본 HDF5/영상은 읽기 전용이며, LeRobot은 학습 export/loader에서만 사용한다.

## 지금 사용할 진입점

실행할 수 있는 모듈은 **10개뿐이며 아래가 전부다.** 나머지 파일은 라이브러리다
(`test_artifacts.py`가 이 목록을 고정한다 — `main()`이 생기면 테스트가 깨진다).

**캠페인 경로 5개** — 데이터에서 배포까지의 본선:

| 할 일 | 실행 모듈 | 상세 절차 |
|---|---|---|
| 세션 검수·고정 manifest·export·통계 | `examples.hv1.pipeline` | [학습 파이프라인·최신 ROS 계약](TRAINING.md) |
| smoke → TODAY30·ALL59 학습·평가 순차 실행 | `examples.hv1.pipeline_run` | 같은 문서의 순차 실행 |
| 개별 학습 / 절제 실험 | `examples.hv1.pipeline_train` | 복구·개별 검증용 |
| 평가·후보 등록 / **교차 모달 점검** | `examples.hv1.pipeline_eval` | 같은 문서의 「카메라를 쓰는가」 |
| 검수된 모델의 loopback HTTP 추론 | `examples.hv1.deploy_server` | [배포 운영](DEPLOYMENT.md) |
| ROS shadow·감독하 실행 경계 | `ros/keti_humanoid_inference` | 같은 배포 문서의 안전 gate |

**운영 도구 4개** — 본선에 끼지 않는 단독 도구:

| 할 일 | 실행 모듈 |
|---|---|
| 기록 관측으로 배포 서버 smoke (ROS 없음) | `examples.hv1.deploy_smoke` |
| shadow/실기 rollout 로그 요약 (영상 미적재) | `examples.hv1.shadow_eval` |
| ROS/URDF 증거 감사 → DRAFT 프로파일 | `examples.hv1.source_contract` |
| 원본 대 export 전수 비교 | `examples.hv1.verify_export` |

**합성 검수 1개** — `examples.hv1.cli` (`demo`/`export`/`scan`/`list`/`review`/
`manifest`/`serve-review`). profile 기반 일반 검수·합성 회귀용이며 캠페인 경로가 아니다.

```text
ROS recorder → 원본 HDF5 + 3개 영상
                 ↓ 읽기 전용 검수·고정 manifest
               학습 export → OpenPI → BF16 snapshot·평가·registry
                                          ↓ loopback HTTP
                                     ROS shadow / 승인된 executor
```

코드는 현장 `~/workspace/openpi-hv1`, 데이터·로그·모델·캐시는
`~/workspace/hv1-vla-runtime`에 둔다. 기존 `keti_humanoid_ros2`의 제어 작업과
OpenPI upstream/DROID 설정은 별도로 유지한다. Windows 터널은 관리용이며 실시간 경로에 넣지 않는다.

## 바꾸려는 내용별 수정 위치

| 변경 대상 | 관리 원본 |
|---|---|
| 원본 HDF5·영상 구조 / 학습 입출력 계약 | `native.py` / `transforms.py` |
| 시연 선정·트랙 분할·샘플링 / 트랙 레시피 | `pipeline.py` — `recipe()`가 학습 설정의 기준 |
| OpenPI 설정 / 명시적 로컬 dataset 로딩 | `pipeline_config.py` |
| optimizer 실행·재시작 cursor | `pipeline_train.py` |
| 오프라인 지표·SHADOW_ONLY 등록 | `pipeline_eval.py` |
| 캠페인과 무관한 정책 지표 | `metrics.py` — 교차 모달·전이 지표 |
| 순차 실행·완료 상태 보관 정책 | `pipeline_run.py` |
| JSON·해시·50GiB 여유 공간 | `artifacts.py` — 표준 라이브러리만 사용 |
| BF16 저장·재로딩·파일 검증 | `checkpoints.py` — 모든 학습/배포 경로가 공유 |
| 추론 전처리·HTTP / 로봇 안전 경계 | `deploy_server.py` / ROS 패키지의 `core.py`, `node.py` |

학습 설정·평가는 trainer를 import하지 않는다. 은퇴한 `overnight_*`(A~F)와
`readapt_*`(N/M)은 2026-09-11에 삭제했고, 어느 경로도 이들을 import하지 않는다.
같은 날 `two_track_*`를 `pipeline*`으로 바꿨다 — **모듈 이름에 캠페인 이름을 넣지
않는다.** 다만 `SCHEMA`는 `hv1_two_track_v1`로 남는다. 디스크에 있는 데이터의
이름이며 `deploy_server`가 registry를 이것으로 검사한다.
기존 public import/CLI는 호환 별칭으로 유지하며, 별칭에 구현을 복제하지 않는다.
ROS 패키지는 컨테이너에서 독립 설치되므로 ML 공통 모듈에 의존시키지 않는다.

## 유지할 계약

- state 15 = 오른팔 실측 7 + 오른손 실측 8, action 8 = 오른팔 관절 목표 7 + 그리퍼 의도.
- head/hand_l/hand_r 모두 유지. 카메라 dropout 없음.
- mode 2, wrap 닫기, narrow 0.6 열기. 왼팔 pose 3 조망 자세.
- 은색 실린더를 오른손으로 집어 트레이에 놓는 고정 prompt.
- TODAY30·ALL59는 공식 `pi05_base`에서 각각 독립 초기화하고, 트랙별 학습 데이터로만
  정규화 통계를 재계산하며, optimizer를 새로 만든다.
- 미검수 데이터·변경된 원본/통계·불완전 snapshot은 차단. 기존 원본과 모델은 자동 정리하지 않는다.
- 모델 등록/교체는 ARM이 아니다. shadow 지표는 실물 성공률이 아니며,
  물리 E-stop만으로 guardian·freshness·소유권 검사를 대신하지 않는다.

계약을 바꿀 때는 이전 세션·manifest를 덮어쓰지 말고 새 버전으로 검수한다.

## 환경·검증

기존 현장 ML `.venv`와 고정 `uv.lock`을 사용한다. ROS 환경과 섞거나 다시 설치하지 않는다.
가벼운 검수 전용 환경은 [requirements-ops.txt](requirements-ops.txt)를 사용하며,
ML 통합 테스트는 전체 현장 환경에서 실행한다.

```bash
cd ~/workspace/openpi-hv1
JAX_PLATFORMS=cpu .venv/bin/python -B -m pytest examples/hv1/tests -q -p no:cacheprovider
.venv/bin/python -B -m ruff check examples/hv1 --select F,E4,E7,E9,I
.venv/bin/python -B -m ruff format --check examples/hv1
.venv/bin/python -B -m examples.hv1.pipeline --help
```

ROS 경계 테스트는 [배포 문서](DEPLOYMENT.md)의 격리 ROS domain·가짜 하드웨어에서만 수행한다.
CPU 테스트 통과는 새 데이터 GPU 학습이나 실기 성공의 증거가 아니다.
Windows 테스트는 OneDrive 밖의 새 `--basetemp`를 지정한다.

## 현재 운영과 구분해 보존하는 코드

- [A~F 올나이트 실험 기록](docs/OVERNIGHT_20260909.md) / [재수집·N/M 운영 기록](docs/READAPT_20260910.md):
  실행 모듈은 삭제했고 두 문서는 무엇을 돌렸는지에 대한 기록으로만 남는다.
- `cli/review/export/demo/adapter`: profile 기반 일반 검수·합성 회귀 테스트.
  **캠페인 경로에서 도달하지 않는 섬이다** — `cli`만 진입점이고 나머지는 그 아래에
  있다. 실제 KETI HDF5+외부 영상 수집은 `pipeline/native` 경로를 쓴다.
  `workflow.py`는 예외로 본선이다 — `native`와 `transforms`가 `validate_profile`을 쓴다.
- `ros_projection`: 오프라인 매핑 확인. 진입점 없는 라이브러리이며 로봇 송신 기능 없음.
- `native.py` / `openpi_run.py`는 2026-09-11에 CLI를 잃고 라이브러리가 됐다.
  각각의 `scan`/`export`, `check-data`/`stats`/`train`/`serve`는 은퇴한 세대의
  진입점이었고 호출자가 없었다.
- [초기 합성 워크플로 기록](docs/INITIAL_WORKFLOW_20260909.md): 과거 상태와 명령을 보존한 참고 문서.
  현재 운영 절차로 사용하지 않는다.

과거 실험의 절대 날짜·검증 수치는 해당 기록에만 두고, 이 README는 진입점과 수정 위치를 안내한다.
