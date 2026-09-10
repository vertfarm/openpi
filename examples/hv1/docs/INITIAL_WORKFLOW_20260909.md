# 초기 합성 데이터 워크플로 — 2026-09-09 기록

이 문서는 초기 구현 재현용입니다. 현재 실물 수집·재학습·배포에는 [운영 README](../README.md)를 먼저 따르세요.

2026-09-09 초기 구현. **실제 에피소드와 로봇 동작은 아직 검증하지 않았다.**
문종술 선임 ROS2/recorder/3D 마우스/핸드캠은 그대로 두고, 완료 파일 이후를 담당한다.
LeRobot은 학습 export/loader에만 사용한다. DROID 관절/속도 배율·로봇 인터페이스는 재사용하지 않는다.

## 배치와 책임

```text
Windows: 개발·검토·명령 관리 ─ RustDesk SSH 터널 ─┐
현장 Mac: LAN SSH / USB·영상·현장 대응 ──────────┤
                                               ▼
현장 HV1 PC
  문종술 ROS2 recorder → 원본 HDF5 + 영상 (미완성/실물 파일 없음)
                         ↓ 읽기 전용
  검수 UI → 판정 sidecar → 고정 manifest → LeRobot export
                         ↓
                 공식 OpenPI π0.5 (격리 환경)
                         ↓ loopback WebSocket
                 shadow 검증, sent=false
  [미구현·별도 승인] ROS executor → 로컬 안전 제어 → 하드웨어
```

- 신규 코드: `~/workspace/openpi-hv1`, branch `codex/hv1-vla-workflow-20260909`.
- 기존 DROID 작업을 포함한 `vertfarm/openpi`의 `d17f10c1dcd78bb18ea2b721abfcb1bf4aab8a59`에서 분기했다.
- 운영 데이터/검수 DB/캐시/체크포인트: `~/workspace/hv1-vla-runtime`.
- 기존 `~/openpi`와 `~/workspace/keti_humanoid_ros2`는 수정하지 않는다.
- 새 모듈은 전부 `examples/hv1/` 아래다. 중앙 OpenPI registry나 기존 DROID config를 수정하지 않는다.
- Windows SSH 터널은 개발용이다. 카메라·추론·제어의 실시간 경로를 Windows/WAN으로 돌리지 않는다.

## 구현과 검증 범위

| 모듈 | 역할 | 현재 경계 |
|---|---|---|
| `source_contract` | 현재 ROS 설정·URDF·인터페이스의 commit/WIP SHA-256 근거 수집 | 실행 불가능한 draft만 생성; 첫 실제 파일 전 schema 확정 금지 |
| `ros_projection` | 관절 이름 기반 재배열, 14축/목 예약 슬롯 구분, 한 점 target·수락된 hand open 검사 | offline payload helper. 구독/명령 발행 없음 |
| `workflow` / `review` | 원본 검수, 3-view 프레임 재생, 성공/실패·선택·메모 | 이미지가 HDF5 uint8 HWC인 fixture/profile 지원. HEVC sidecar decoder 미구현 |
| `export` / `verify_export` | 로컬 LeRobot v2.1 export, 전 프레임 원본 대조 | 고정 dependency API 사용. Hub 업로드 없음 |
| `openpi_run` | HV1 입출력, 공식 loader/chunk, train-only 통계, train/serve 진입점 | loader·전처리 검증 완료. 실제 모델 학습/저장/추론은 미시험 |
| `adapter` | 제한시간 있는 OpenPI wire client, metadata/time/shape 검사 | loopback 전용, 오류 latch, no retry, 항상 `sent=false` |

합성 3 episodes / 36 frames로 검증했다. 세션 단위 train 2개(24 frames), validation 1개(12 frames)를 물리적으로 분리한다.
RGB·state·action·task·timestamp 전 프레임 대조, 두 train episode 끝의 chunk padding, prompt tokenization,
32차원 model padding 및 normalize→unnormalize→절대 target 복원을 확인했다. 모델 weight는 로드하지 않았다.
UI는 선택·이미지·프레임 이동·검수 저장·새로고침 유지를 브라우저에서 확인했다.

## 환경

현재 현장 신규 ML 환경은 Python 3.11 + 기존 `uv.lock`으로 `uv sync --frozen`했다.
기본 dev group을 생략하면 이 고정 버전의 모델 import가 요구하는 `pytest`가 빠지므로 생략하지 않는다.
운영/검수 도구는 별도 Python 3.11 이상 환경과 `requirements-ops.txt`를 쓴다.

```bash
cd ~/workspace/openpi-hv1
export HV_RUNTIME="$HOME/workspace/hv1-vla-runtime"
export UV_CACHE_DIR="$HV_RUNTIME/cache/uv"
export HF_HOME="$HV_RUNTIME/cache/huggingface"
export OPENPI_DATA_HOME="$HV_RUNTIME/cache/openpi"
export PYTHONDONTWRITEBYTECODE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_MODE=disabled

# 신규 ML 복제 환경에서만: 기존 ROS/사용 중인 다른 OpenPI 환경에는 적용하지 않는다.
uv sync --frozen --python 3.11
# 운영 환경이 이미 준비되어 있으므로 재생성할 필요 없음.
# 처음 구축할 때만 별도 venv를 만들고 설치한다:
# uv venv --python 3.11 "$HV_RUNTIME/.venv-ops"
# uv pip install --python "$HV_RUNTIME/.venv-ops/bin/python" -r examples/hv1/requirements-ops.txt
```

RTX PRO 6000 Blackwell 96GB에서 작은 JAX GPU JIT 계산을 확인했다.
이는 π0.5 전체 모델의 메모리·학습 호환성 증명이 아니다. 기존 torch cu126의 sm_120 경고도 해소했다고 보지 않는다.
설치 후 작업 볼륨은 약 256G 여유였다. 미마운트 디스크는 사용/포맷하지 않았다.

## 1. 샘플 없이 지금 수행할 근거 갱신

```bash
.venv/bin/python -m examples.hv1.source_contract \
  --ros-root "$HOME/workspace/keti_humanoid_ros2" \
  --output "$HV_RUNTIME/ros-contract-draft-v2.json"
```

결과를 `status=confirmed`로 이름만 바꿔 사용할 수 없다. 데이터 profile과 별도 형식이며 검사에서 거부된다.
현재 후보는 **state 30(팔 14 + 손 관절 8×2), action 16(팔 target 14 + open 2)**다.
손 관절 feedback 가용성·고정 파지 mode·수락 target 로깅을 확인한 뒤 결정한다. 30/16은 확정 사양이 아니다.
목 예약 2슬롯은 원시 wire에서 보존하되 학습 target에 넣지 않는다.

현재 확인할 불일치:

- recorder action의 `JointState`는 측정값이다. IK 후 실제 commanded target과 원시 입력 twist를 별도로 남긴다.
- 실 코드의 `/joint_states`, `/kh/joint_command`, `/kh/state/...`와 YAML의 잔존 `/upper_body/...`를 대조한다. live namespace/remap은 아직 미확인이다.
- `Float64`에 `mode, open` 두 필드를 기대할 수 없다. 실제 KDEX 메시지/service의 request, success, accepted open, grasp 결과를 구분한다.
- `RecordCommand.srv`가 추가됐다. START/STOP/SAVE/DELETE state machine은 recorder 소유다. 우리 UI는 호출하지 않는다.
- `/kh/joint_command` consumer는 `points[-1]`만 소비한다. 모델 chunk를 한 trajectory로 보내면 안 된다.
- 한 팔씩 들어오는 부분 command는 검증된 commanded-setpoint cache/시각/소유권 없이 다른 팔의 state로 메우지 않는다. 현재 helper는 부분 target을 거부한다.
- URDF limit은 실물 승인 한계가 아니다. 손 장착·TCP·팔꿈치 범위·sign/zero·하중·gain revision은 별도 실기 gate다.

## 2. 합성 검수 연습

이미 있는 `synthetic-smoke-v1`은 통과 기록으로 남긴다. 아래 `-v2` 경로도 존재하면 새 버전을 고른다.
CLI는 원본/manifest/export를 자동 삭제하거나 덮어쓰지 않는다.

```bash
OPS="$HV_RUNTIME/.venv-ops/bin/python"
$OPS -m examples.hv1.cli demo "$HV_RUNTIME/practice-v2"
$OPS -m examples.hv1.cli scan \
  --runtime "$HV_RUNTIME/practice-v2/review" \
  --raw-root "$HV_RUNTIME/practice-v2/raw" --profile "$HV_RUNTIME/practice-v2/profile.json"
$OPS -m examples.hv1.cli serve-review \
  --runtime "$HV_RUNTIME/practice-v2/review" \
  --raw-root "$HV_RUNTIME/practice-v2/raw" --profile "$HV_RUNTIME/practice-v2/profile.json"
```

현장 PC 브라우저에서 `http://127.0.0.1:8766`. loopback만 listen한다.
다른 PC에서 보려면 승인된 별도 UI 터널이 필요하다. 현재 restrict SSH 키로 SSH port forwarding이 된다고 가정하지 않는다.
서버 종료는 실행 터미널의 Ctrl+C. 재시작 후 검수 DB가 유지된다. 자동 부팅/상시 감시 서비스는 설치하지 않았다.
성공+품질 통과에만 첫 baseline 학습 체크가 가능하다. 실패/중단 원본도 삭제하지 않고 남긴다.

## 3. 선택 → export → 학습 전 확인

```bash
$OPS -m examples.hv1.cli manifest \
  --runtime "$HV_RUNTIME/practice-v2/review" \
  --raw-root "$HV_RUNTIME/practice-v2/raw" --profile "$HV_RUNTIME/practice-v2/profile.json" \
  --output "$HV_RUNTIME/practice-v2/manifest.json" --allow-synthetic
.venv/bin/python -m examples.hv1.cli export \
  --manifest "$HV_RUNTIME/practice-v2/manifest.json" \
  --destination "$HV_RUNTIME/practice-export-v2" --allow-synthetic
.venv/bin/python -m examples.hv1.verify_export \
  --manifest "$HV_RUNTIME/practice-v2/manifest.json" --export "$HV_RUNTIME/practice-export-v2/export.json"
.venv/bin/python -m examples.hv1.openpi_run stats \
  --export "$HV_RUNTIME/practice-export-v2/export.json" --runtime "$HV_RUNTIME" --allow-synthetic
.venv/bin/python -m examples.hv1.openpi_run check-data \
  --export "$HV_RUNTIME/practice-export-v2/export.json" --runtime "$HV_RUNTIME" --allow-synthetic
```

실제 데이터에는 synthetic flag를 쓰지 않는다. raw/manifest/source hash 변경은 재검수를 요구한다.
통계는 train repo만 읽는다. validation은 checkpoint 평가용으로 남기며 현재 자동 validation-loss/model-selection 루프는 없다.
프레임 무작위 split이 아니라 session split이며, 1 session뿐이면 validation이 없다고 표시한다.

## 4. GPU 학습·배포 gate (아직 미실행)

첫 실물 episode가 없어도 별도 승인한 synthetic GPU smoke는 가능하다. 그러나 현재는 weight 다운로드/학습을 시작하지 않았다.
진행 전 GPU 작업자 충돌, 캐시/원본/체크포인트 저장 예산과 운영 시간을 확인한다.
`openpi_run train`은 `--allow-gpu-run`, 새로운 `--experiment`, 통계 파일을 요구한다.
합성 smoke면 추가 `--allow-synthetic`이 필요하며 현장용 checkpoint로 사용할 수 없다.
기본 20 steps, batch 1은 연결 smoke 설정이지 성능 학습 recipe가 아니다. GPU lock은 이 HV1 도구끼리만 조정한다.

순서: **tiny GPU train → checkpoint 저장/재로딩 → loopback serve → recorded replay/shadow → 안전·소유권 gate → 현장 감독하 단기 실기 평가**.
`serve`는 선택 experiment 내부 checkpoint와 동일 manifest/profile metadata만 받으며 loopback 8765로 제공한다.
실제 ROS subscriber/executor, 속도/가속/충돌/추종 오차 제한, watchdog/통신 단절 정지, 재무장 기능은 별도 구현·확인이 필요하다.
`sent=false`나 송신 중단은 물리적 정지를 뜻하지 않는다.

## 첫 실제 파일에서 할 일

1. 완료된 HDF5와 연결된 영상의 실제 디렉터리/최종 저장 상태를 확인한다. `output_dir: /workspace/datasets`의 호스트 mount도 확인한다.
2. dataset 이름/차원/단위/clock domain, 실제 command source, 손 accepted event, camera serial·sequence를 mapping profile에 반영한다.
3. HEVC sidecar reader는 실제 파일명·PTS·capture timestamp 대응에 맞춰 추가한다. `frame_index / nominal_fps`로 동기화를 만들어내지 않는다.
4. raw는 그대로 두고 derived 검수/학습용 출력만 생성한다. 실물 원본 대조 → 품질/태스크 판정 → export를 반복한다.
5. 공정·성공기준은 사용자가 제공한 작업을 사용한다. YAML의 `pick object`를 실제 태스크로 확정하지 않는다.

## 테스트

```bash
PYTHONDONTWRITEBYTECODE=1 "$HV_RUNTIME/.venv-ops/bin/python" -m pytest examples/hv1/tests -q -p no:cacheprovider
.venv/bin/python -m ruff check examples/hv1
```

Windows에서는 권한 충돌을 피하기 위해 `--basetemp`에 OneDrive 밖의 **새 고유 디렉터리**를 지정한다.
테스트/캐시/venv/Git clone은 Obsidian/OneDrive에 만들지 않는다.
