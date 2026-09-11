# HV1 작업 규약 — 모든 에이전트 공통

이 파일은 세션마다 바뀌지 않는 규약이다. **지금 무엇이 사실인가**는
[examples/hv1/STATUS.md](examples/hv1/STATUS.md)에 있다. 작업 시작 전 둘 다 읽는다.

## 저장소와 작업트리

- canonical 브랜치: `codex/hv1-vla-workflow-20260909` (remote `origin`)
- 두 작업트리가 같은 브랜치를 쓴다. 현장 리눅스 `~/workspace/openpi-hv1`,
  Windows Codex 워크스페이스(SSH 터널). **git이 두 머신이 공유하는 유일한 채널이다.**
- OneDrive/Obsidian wiki는 사람의 기록이다. 현장 머신에서 읽을 수 없으므로
  에이전트 간 조율 채널로 쓰지 않는다.
- 트리가 갈라져 보이면 먼저 `git fetch` 후 실제 차이를 확인한다.
  파일을 손으로 복사해 "동기화"하지 않는다. 과거에 이것이 분기를 만들었다.
- 코드는 repo, 데이터·로그·모델·캐시는 `~/workspace/hv1-vla-runtime`.
  런타임 산출물을 git에 넣지 않는다.

## 읽기 전용 경계

다음은 수정·삭제하지 않는다.

- 원본 데이터셋 `~/workspace/keti_humanoid_ros2/datasets/`
- 기존 `evaluations/`, 기존 `shadow/` 로그
- 기존 스냅샷

## ROS 코드의 실행 사본

- 실행되는 사본은 `/workspace/ros2/vla_ws` (컨테이너) =
  `~/workspace/keti_humanoid_ros2/ros2/vla_ws` (호스트) 하나뿐이다.
- `/tmp/hv1_mode2_r2` 같은 임시 overlay를 실행 경로로 쓰지 않는다.
- `examples/hv1/ros/**`를 고쳤으면 반드시:
  1. `vla_ws/src`에 반영
  2. `colcon build --symlink-install`
  3. 아래 셋을 기록
     ```
     ros2 pkg prefix keti_humanoid_inference
     python3 -c 'import inspect, keti_humanoid_inference; print(inspect.getfile(keti_humanoid_inference))'
     sha256sum <실행 사본의 core.py>
     ```
- **`CONTRACT_SHA`는 데이터 계약만 보증한다. `GripEdges`와 `LiveGate`는 보증하지 않는다.**
  재빌드를 빠뜨리면 health hash는 통과하고 shadow도 정상 동작하는데 결과만 바뀌지 않는다.
  안전 로직을 고쳤으면 소스 해시로 직접 확인한다.

## 로봇 안전

- shadow 모드에는 command publisher와 gripper client가 없다. 구조적으로 명령이 불가능하다.
- live는 `approved=true` 필드 프로파일과 guardian flag가 있어야 하며 현재 template은
  `approved=false`다. **감독자 없이 live를 시도하지 않는다.**
- 감독자 부재(야간·원격) 시에는 ROS 노드를 어떤 모드로도 실행하지 않는다.
  검증은 저장된 로그 재생으로만 한다.
- 실기 시작 전 손 자세는 `mode 2` → `set_open 0.6`이다. STATUS.md의 기준값과 대조한다.

## 자원

- 디스크 게이트 50GiB (`artifacts.py`의 `MIN_FREE`). 여유를 확인하고 시작한다.
- GPU는 1개다. `hv1-vla-runtime/hv1-ml-gpu.lock`으로 직렬화한다.
- 장시간 작업 후 lock 해제와 프로세스 종료를 확인한다.

## 판단 규칙

- **내가 받지 않은 임계값을 발명하지 않는다.** 기준이 주어지지 않았으면 값을 고르지 말고
  곡선이나 표로 보고한다.
- **자체 도출한 실패 판정은 금지된 작업의 근거가 될 수 없다.** 판정이 실패로 나오면
  거기서 멈추고 보고한다. 범위를 넓혀 해결하려 하지 않는다.
- 배포 체크포인트를 단독으로 선정하지 않는다. 표를 보고하고 선택은 감독자에게 맡긴다.
- 연결·서비스·관측이 없으면 재시도 루프에 들어가지 않는다. 해당 항목을 미완으로
  남기고 사유를 기록한 뒤 나머지를 진행한다.

## 코드 위생

- 새 모듈 계열을 만들지 않는다. 기존 파일을 제자리에서 수정한다.
  이 저장소는 이미 `overnight_*` → `readapt_*` → `two_track_*` 3세대가 쌓여 있다.
- 날짜 박힌 새 `.md`를 만들지 않는다. `DEPLOYMENT.md`와 `STATUS.md`를 갱신한다.
- git history를 재작성하지 않는다. force push, 세대별 WIP 커밋을 만들지 않는다.
- 구조 정리(`overnight_*` 삭제, `readapt`/`two_track` 통합)는 STATUS.md가
  허용할 때까지 보류한다.

## 세션 종료 시

`examples/hv1/STATUS.md`를 **덮어쓴다.** 날짜 섹션을 append하지 않는다.
이력은 git log가 갖고 있다. 기계가 읽을 산출물은 JSON으로 남기고
STATUS.md에는 경로와 SHA만 적는다.
