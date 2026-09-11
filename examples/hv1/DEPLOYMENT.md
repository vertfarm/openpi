# HV1 π0.5 배포 — 오프라인 intent 조건화 검증, 실기 interlock 연동 대기

목표는 학습 모델의 실시간 배포다. 수집 레시피는 관측·명령·시작 자세의
계약이며, 전체 시연 재생은 배포 선행조건이 아니다.

## 구성과 검증 상태

- 모델: 공식 OpenPI + 검수된 C-5000 BF16 snapshot. 비교 후보 F-5000.
- host ML 환경: Python 3.11. ROS Jazzy container: Python 3.12.
- 의존성을 섞지 않도록 loopback HTTP로 연결한다. 서버는 robot API를 import하지 않는다.
- ROS 구현의 관리 원본은 이 폴더의
  [ros/keti_humanoid_inference](ros/keti_humanoid_inference)다.
  현장에는 기존 kh_ws를 수정하지 않는 별도 ros2/vla_ws overlay로 설치했다.
- 서버 C-5000 실제 기록 관측 4개 HTTP 검수 통과: 왕복 약 60~66 ms.
- 실제 ROS domain 10에서 60초 shadow: 446회 추론, 1,022개 무송신 목표,
  callback 기준 왕복 중앙값 약 70 ms. 로봇 명령 0건.
- 최초 DDS discovery 중 관측 누락, 일부 camera/state skew, 모델의 재파지 전환을
  발견했다. 이 기록을 실기 readiness 또는 물리 작업 성공으로 해석하지 않는다.
- 격리 domain 213의 가짜 로봇에서 shadow 무송신, local ARM, 오른팔 명령,
  Grasp(wrap=true), SetOpen(0.6), guardian 상실 시 stop 호출을 시험한다.
  이 fake guardian은 실제 하드웨어에 배포하면 안 된다.

## 입력·출력 계약

- state 15: 오른팔 실측 7축 + 오른손 실측 8축.
- action 8: 오른팔 absolute joint target 7축(rad) + grasp intent.
- 모델의 delta 복원은 기존 HV1Outputs에서 관측 anchor 기준으로 한 번만 한다.
  ROS client에서 IK, sign flip, degree 변환, 마우스 매핑을 추가하지 않는다.
- 세 카메라 head/hand_l/hand_r 모두 필수. RGB, dropout 없음.
- live image: compressed decode → 회전 없음 → recorder와 동일한
  cv2 INTER_AREA 640×480 → BGR→RGB → OpenPI PIL bilinear pad 224.
  원본 HEVC의 손실 압축까지 재현하지는 않는다.
- mode 2는 현장 준비에서 확정한다. close는 Grasp(mode=KEEP, wrap=true,
  tip_distance=0), release는 SetOpen(open=0.6). speed=0으로 기존 server 기본값을
  유지한다. 손 속도를 역추정하지 않고, 손 관절 기록을 on/off 출력 대신 보내지 않는다.
- denoise 10, horizon 15, 한 번에 미래 3개를 30Hz 실행. 추론 1개만 in-flight,
  만료 prefix는 시간 정렬로 제외하며 catch-up burst를 금지한다.
- grasp intent는 손가락 8축 목표가 아니라 close/open 사건을 결정하는 스칼라다.
  실행기는 close 0.7/open 0.3 히스테리시스와 연속 유지시간을 적용한다. close가
  시작된 뒤 Grasp action 결과를 받기 전에는 open을 보내지 않고, guardian의
  release 허가가 올 때까지 해제를 보류한다. 유지시간은 저장된 시계열 스윕 결과를
  감독자가 검토해 선택하며 기본 0.2초를 실기 승인값으로 간주하지 않는다.
- 서버는 학습 통계의 hand state q01/q99에 축별 0.05 rad 절대 여유를 둔 준비자세
  envelope를 `/health`로 제공한다. client와 server가 모두 이를 검사하며, 학습 때의
  mode 2 + open 0.6 준비자세와 크게 다른 raw-home 입력은 clipping 없이 거부한다.
- `/health`의 `adapter_core_sha256`은 서버가 읽은 `core.py`와 ROS 실행 사본을
  byte 단위로 묶는다. 불일치하면 시작을 거부한다. 데이터 계약 해시만 일치하는 것은
  실행 코드가 같다는 증거가 아니다.
- 실제 입력 주기 측정: 관절 약 100Hz, head 30Hz, hand_l/hand_r 약 12Hz.
  최신 메시지만 조합하면 skew가 주기적으로 커져 후보 큐가 끊기는 것을 확인했다.
  bounded history에서 시간적으로 가까운 fresh 관측을 선택하며 state 100ms,
  image 150ms, skew 100ms 한도는 완화하지 않는다. 모델에는 정렬 관측을,
  실기 tracking 검사는 최신 측정값을 사용한다. 수신 주기를 센서 노출 동기로 주장하지 않는다.
- 각 제안/목표에 sequence, model index, observation anchor, proposal hash를 기록한다.
  request마다 선택된 source age도 남긴다.

## 추론 서버와 shadow 실행

현장 host에서 기존 서버가 없다면:

```bash
cd ~/workspace/openpi-hv1
CAMPAIGN=~/workspace/hv1-vla-runtime/two-track-20260910-r3
.venv/bin/python -B -m examples.hv1.deploy_server \
  --campaign "$CAMPAIGN" \
  --snapshot "$CAMPAIGN/snapshots/TODAY30/step_001000" \
  --registry "$CAMPAIGN/checkpoint_registry.json"
```

`--registry`는 필수다(2026-09-11). C/F-5000만 허용했던 legacy 분기는 삭제했다.

서버는 127.0.0.1:8000에만 bind한다. /health에 snapshot/stat/contract 해시가 있다.
GPU lock을 유지해 다른 학습/평가와 충돌하지 않는다. 이미 실행 중이면 중복 시작하지 않는다.
F 비교는 C 서버와 실기 client를 정지·확인한 뒤 snapshot 경로만 바꾸어 시작한다.

현장 ROS container에서, upper를 추가로 실행하거나 기존 스택을 재시작하지 않는다:

```bash
source /opt/ros/jazzy/setup.bash
source /workspace/ros2/hand_ws/install/setup.bash
source /workspace/ros2/kh_ws/install/setup.bash
source /workspace/ros2/vla_ws/install/setup.bash
export ROS_DOMAIN_ID=10
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ros2 run keti_humanoid_inference vla_client \
  --mode shadow --mqtt-host 192.168.0.142 --seconds 60 \
  --output /workspace/ros2/vla_ws/log/NEW_UNIQUE_SHADOW_SESSION
```

output은 매번 새 디렉터리여야 한다. raw dataset을 output으로 사용하지 않는다.
shadow에는 robot command publisher, gripper action/service client가 없다.
상태는 /hv1_vla/status, 무송신 후보는 /hv1_vla/shadow/candidate로 나온다.
events.jsonl과 observation_*.npz에 원시 관측·응답·실행 후보가 저장된다.

## 실기 활성화에 필요한 외부 계약 — 현재 미완료

이 구현은 현재 존재하지 않는 물리 정지 API를 만들어 냈다고 가정하지 않는다.
따라서 template은 approved=false이며 바로 live로 전환할 수 없다.

현장 담당자와 아래 항목을 검증한 뒤 별도 프로파일로 제공해야 한다.

1. 7축별 실측 범위, target step, tracking error, 속도·가속도, 시작 자세 허용 오차.
2. 기존 teleop/replay/MoveL/trajectory action 및 다른 MQTT writer를 배제하는
   독립 제어권 관리. VLA node는 twist publisher 잔존·중복 joint publisher도 거부한다.
3. 프로세스 사망·통신 단절에도 동작하는 독립 hardware watchdog과 실제 정지 동작.
   std_srvs/Trigger stop 서비스는 검증된 물리 정지에 연결되어야 한다.
   “MQTT publish를 그만함” 또는 상수 success 응답은 정지 검증이 아니다.
4. 보정된 기구/TCP, 왼팔·헤드 자세, self/table collision 및 트레이 release 조건.

독립 guardian 인터페이스:

- 입력: /hv1_vla/proposal (std_msgs/String JSON). snapshot hash, proposal_id,
  observation anchor, 앞으로 보낼 3개 target와 예정 시각을 포함한다.
- 출력: /hv1_vla/guardian (std_msgs/String JSON), 같은 Linux monotonic clock.
  sequence는 증가해야 하고 lease는 100 ms 이내다.
- 필수: review_id, sequence, monotonic, exclusive, workspace_clear, stop_ready,
  hardware_watchdog_ready, grasp_mode=2, mqtt_age_s, start_pose_verified,
  narrow_open_verified, release_allowed, approved_proposals.
- approved_proposals에는 **실제로 충돌/작업 공간/소유권 검수한 proposal hash만**
  넣는다. 상태 flag를 상수 true로 발행하거나 fake test guardian을 재사용하지 않는다.
- 실행기는 승인 없는 proposal, lease 만료, raw MQTT age >100 ms, 관측 stale/skew,
  이름/차원/NaN 오류, 큰 목표 변화, 손 요청 실패에서 명령을 거부한다.
- stop 서비스 unavailable/failed는 물리 정지 성공이 아니다. 독립 watchdog과 현장
  E-stop가 이를 담당해야 하며, 시험 전 확인이 필요하다.

현장 프로파일을 채운 뒤 live client를 시작해도 DISARMED 상태다.
ARM은 같은 container의 소유자 전용 UNIX socket(0600)으로만 받는다.
DDS 또는 HTTP에 ARM endpoint를 노출하지 않는다.

```bash
ros2 run keti_humanoid_inference vla_client \
  --mode live --mqtt-host 192.168.0.142 --profile REVIEWED_FIELD_PROFILE.json \
  --seconds 0 --output /tmp/NEW_UNIQUE_LIVE_SESSION

# 현장 담당자가 로봇과 E-stop를 보면서 다른 terminal에서 수행
ros2 run keti_humanoid_inference vla_operator \
  --socket /tmp/NEW_UNIQUE_LIVE_SESSION/operator.sock ARM --review-id REVIEW_ID

ros2 run keti_humanoid_inference vla_operator \
  --socket /tmp/NEW_UNIQUE_LIVE_SESSION/operator.sock STOP
```

모델에 episode 종료 신호가 없으므로 현장 STOP으로 종료/성공을 판정한다.
45초 pilot 상한은 보호 중단이며 작업 성공으로 기록하지 않는다.
FAULT 후 자동 재시작·자동 복귀·손 자동 열기·토크 자동 해제를 하지 않는다.

## 회귀 시험

```bash
python -B -m pytest examples/hv1/tests -q -p no:cacheprovider
```

ROS 통합 시험은 별도 domain 213 + ROS_LOCALHOST_ONLY=1일 때만 실행된다.
MQTT는 모의 객체로 대체한다. 실제 domain 10에서 이 테스트의 skip을 해제하지 않는다.

결과 해석: client/transport/command semantics 검수는 모델 성공률과 다르다.
C-5000은 기록된 release 관측에서도 close를 출력한 사례가 있다. 실제 실기 전
shadow 결과와 safety interlock을 확인하며, 안전 제한 완화로 해결하지 않는다.

## 감독자 부재 시 오프라인 검증

E-stop 체결·팔 torque 해제 상태에서는 `vla_client`를 shadow 포함 어떤 모드로도
실행하지 않는다. 저장된 correct-hand `events.jsonl`의 target action과 GPU로 한 번
생성한 teacher-forced intent sidecar만 재생한다. 관측이나 서비스가 없으면 재시도,
ROS stack 재시작 또는 우회를 하지 않는다.

```bash
# 각 8개 snapshot에 대해 한 번씩, GPU lock을 지키며 직렬 실행
python -B -m examples.hv1.two_track_eval evaluate-intents \
  --campaign CAMPAIGN --snapshot SNAPSHOT --allow-gpu-run

# 이후 스윕은 CPU-only이며 기존 evaluation/shadow 파일을 수정하지 않는다.
python -B -m examples.hv1.two_track_eval sweep-filter \
  --campaign CAMPAIGN --shadow-root CAMPAIGN/shadow \
  --output CAMPAIGN/evaluations/grasp_filter_sweep.json
```

스윕은 유지시간 0.1/0.2/0.3/0.5초별 missed/extra close, close 시간오차,
정적 correct-hand 재생의 false close를 함께 낸다. 기존 평가를 새 점수로 덮어쓰지
않고, 필터 결과만으로 체크포인트를 자동 선정하거나 ARM하지 않는다. 야간 build 뒤에는
ROS 노드를 띄우는 대신 `ros2 pkg prefix`, Python import 경로, `core.py` SHA-256만
기록한다. live shadow와 실기는 감독자가 있는 시간에 다시 수행한다.

사전에 승인된 판정 기준을 만족하는 기존 조합이 하나도 없을 때만 조건부 저학습률
fine-tune을 허용한다. `TODAY30_FT`는 TODAY30-1000, `ALL59_FT`는 ALL59-2000의
BF16 가중치에서 각각 독립적으로 시작하며 optimizer는 새로 만든다. 데이터·정규화
통계·70/15/15 sampler는 부모 트랙과 동일하고 peak LR 2.5e-6, warmup 50,
최대 1,000 update다. 현장 디스크의 50 GiB 하한을 지키기 위해 +500/+1000 추론본만
보존하고 optimizer 재시작 상태는 저장하지 않는다. 새 모델도 teacher-forced intent와
저장된 correct-hand 관측 재생을 모두 통과하기 전에는 registry나 배포 후보로 올리지 않는다.
