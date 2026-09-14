# 0914_check — 결론

cosmos 정책을 KETI 실물 Franka 에 얹기 위한 작업의 결론.
전체 기록은 [0914_test.md](0914_test.md).

---

## 한 줄

**cosmos 는 된다. 코드도 만들었고 시뮬에서 검증했다. 남은 위험은 하나 — 로봇이 속도 명령을
어떻게 해석하는가 — 이고, 어느 쪽이든 대응책이 준비돼 있다.**

---

# 1. openpi(A6000) 코드가 어떻게 작동하나

지금 실물에 돌아가는 것은 `real-robot/openpi` 의
`codex/pi05-droid-jointpos-velocity-20260907` 브랜치다. 기계 세 대로 나뉜다.

| 기계 | 역할 |
|---|---|
| **A6000** (SNU) | 정책 추론. 물리 GPU 2/3, 포트 8001 |
| **3090** (KETI) | 카메라 3대, `RobotEnv`, 제어 루프 |
| **G15** | 실제 팔 |

## 제어 스텝 한 번

```
① 3090   카메라 3장 + 관절각/그리퍼 읽기        (_extract_observation)
② 3090   요청: 외부 1장 + 손목 1장 (각 224x224) + 관절각(7) + 그리퍼(1) + 지시문
             │ websocket -> 8001
③ A6000  모델이 관절각 "델타" 를 냄. 출력 변환 4단:
             Unnormalize                        정규화 해제
             AbsoluteActions                    델타 + 현재상태 = 절대 목표각
             JointPositionToDroidVelocity  ★    절대각 -> 속도, ±0.5 클립
             DroidOutputs                       앞 8차원만
④          응답: actions [15, 8]  = 정규화 속도 7 + 그리퍼 1
⑤ 3090   청크에서 한 줄씩 env.step(). open_loop_horizon 만큼 쓰고 다시 물음
⑥ 로봇   RobotEnv("joint_velocity") 가 x0.2 해서 절대 목표로 -> G15
```

★ 의 산술 (`src/openpi/policies/droid_policy.py`, 30줄이 계약의 전부):

```
v_0 = (목표_0 - 현재관절각) / 0.2      <- 첫 줄만 "측정값" 기준
v_k = (목표_k - 목표_{k-1}) / 0.2      <- 나머지는 "직전 목표" 기준
앞 7차원만 ±0.5 로 클립 (그리퍼는 안 건드림), 포화율 로그
```

## 파일

| 파일 | 하는 일 |
|---|---|
| `scripts/start_pi05_droid_jointpos_velocity.sh` | 기동 껍데기. 계정/GPU/포트/디스크 검사, 체크포인트 SHA-256. **KETI 전용** |
| `scripts/serve_pi05_droid_jointpos_velocity.py` | 모델 로드 → **워밍업 2회 통과 후에야 포트 개방** |
| `src/openpi/training/config.py` | 설정 `pi05_droid_jointpos_velocity`. 위 4단 사슬이 여기 정의됨 |
| `src/openpi/policies/policy.py` | `infer()` 한 번 = 요청 하나. `state` 를 출력 사슬로 넘겨준다 |
| `src/openpi/policies/droid_policy.py` | ★ 속도 변환 |
| `scripts/check_..._velocity.py` | 로봇 없이 웹소켓만으로 `(15,8)` 확인. 움직이기 전 게이트 |
| `examples/droid/main.py` | **3090 코드.** 요청 만들고 `env.step()` 호출 |

**`RobotEnv` 자체는 이 저장소에 없다** — `droid` 패키지 안이고, 그게 §4 의 미해결 항목 원인이다.

---

# 2. cosmos 는 뭐가 다른가

| 항목 | pi | cosmos | 대응 |
|---|---|---|---|
| **모델 출력** | 관절각 **델타** | **이미 절대각** | `AbsoluteActions` 를 **빼야 함** |
| **청크 길이** | 15 | **32** (체크포인트가 정함) | 32 를 그대로 씀 |
| **요청 이미지** | 224x224 **2장** | **합성 1장** 540x640 (카메라 3대) | 요청 형식 교체 |
| **응답 키** | `actions` | `action` | 개명 |
| **서버 구조** | openpi 정책 + 변환 사슬 | **독자 서버**, 사슬 개념 없음 | 앞에 **프록시**를 둠 |
| **체크포인트** | `model.safetensors` | HF 캐시 + `.pt` 오버레이 | 서버 인자 |
| **GPU** | (pi0.5) | **~33 GiB** | §6-1, §6-3 |
| 그리퍼 | 절대 [0,1] | **동일** (서버가 두 번 뒤집어 상쇄) | 건드리지 않음 |
| 프롬프트 | 지시문 | **동일** (서버가 카메라 설명을 자기가 덧붙임) | 건드리지 않음 |

**가장 위험한 차이는 첫 줄이다.** pi 는 델타라서 `AbsoluteActions` 가 필요하지만 cosmos 는
이미 절대각이라, 사슬을 그대로 베끼면 **상태를 두 번 더해 목표가 두 배가 된다.** 예외가 안 난다.

---

# 3. cosmos 용으로 무엇을 만들었나

**`cosmos3-droid/` 에 새 파일 5개. 기존 파일은 하나도 안 고쳤다.**

| 파일 | 내용 |
|---|---|
| `src/openpi/policies/cosmos_droid_policy.py` | 합성 프레임 생성 + `action`→`actions` + 출력 사슬 |
| `src/openpi/policies/cosmos_droid_policy_test.py` | 테스트 **19개** |
| `scripts/serve_cosmos_droid_velocity.py` | cosmos 서버 앞 **프록시** (모델 없음, GPU 없음) |
| `scripts/check_cosmos_droid_velocity.py` | 로봇 없이 도는 게이트 |
| `docs/cosmos_droid_velocity.md` | 계약·수치·미해결 질문 |

## 출력 사슬 — pi 것을 그대로 재사용한다

```
pi:      Unnormalize -> AbsoluteActions -> JointPositionToDroidVelocity -> DroidOutputs
cosmos:                 CosmosActionsToJointPositions -> JointPositionToDroidVelocity -> DroidOutputs
                        └ 키 개명 + float32 + 검증          └────── pi 것 그대로, horizon 만 32 ──────┘
```

- `Unnormalize` 제거 — cosmos 서버가 자기 프로세스에서 함
- `AbsoluteActions` 제거 — **cosmos 는 이미 절대각** (실측: `|a0-q|=0.023` vs `|a0|=0.721` rad)
- 뒤 두 단은 **import 해서 그대로 씀** → G15 가 받는 명령은 pi 와 **문자 그대로 같은 코드**가 만든다

## 3090 이 바꿔야 하는 것 — 3개

| # | 무엇 | 안 바꾸면 |
|---|---|---|
| 1 | 요청에 **카메라 3장** (현재 외부 1장 + 손목) | **0/24.** 정책이 눈이 멂 |
| 2 | `action_horizon` 15 → **32** | assert 에서 즉사 |
| 3 | `open_loop_horizon` 8 → **32** | 성공률 반토막 + **931 ms > 533 ms 라 실시간 불가** |

**1번의 재료는 이미 3090 에 있다** — `_extract_observation` 이 left/right/wrist 를 다 뽑고
요청 만들 때 하나를 버린다. 카메라를 늘리는 게 아니라 요청 형식만 바꾸면 된다.

**로봇 `RobotEnv("joint_velocity")` 와 G15 는 한 글자도 안 바뀐다.**

---

# 4. 어디까지 검증됐나

## 통과한 것

| 검증 | 결과 |
|---|---|
| 오프라인 테스트 19개 | **통과** — 왕복 폐쇄성, 2배 사고 음성대조군, 클립·포화, 그리퍼 불변, 잘못된 입력 |
| 합성 프레임이 cosmos 참조 클라이언트와 | **바이트 동일** |
| 와이어(msgpack 2회 왕복) | **비트 동일**, `max\|diff\|=0.0` |
| 살아있는 cosmos 서버 왕복 | `(32,8)` 속도, 포화 0, **931 ms** |
| pi 원본 테스트 13개 | **그대로 통과** (기존 파일 무수정 확인) |

## 시뮬 실측 (RoboLab, BowlStacking, 24 env)

| 설정 | 실행별 성공 | 합계 |
|---|---|---|
| cosmos 원래대로 (청크 32) | 17, 17, 12 | **63.9%** |
| **출력만 pi 형식** (청크 32) | 12 | 위와 **같은 풀** — 형식 변환은 무손실 |
| 출력 pi 형식 + **청크 15** | 9, 8 | **35.4%** ← 청크를 줄이면 안 되는 이유 |
| **입력까지 pi 형식** | 0 | **0%** ← 카메라 3장이 필요한 이유 |

스텝 0 액션 차이로 원인이 갈린다:

| 비교 | 차이 | 잡음 대비 |
|---|---|---|
| 같은 설정 재실행 (**잡음 눈금**) | 0.01596 rad | 1.00x |
| **출력 형식만** 다름 | 0.01593 rad | **1.00x** — 구분 안 됨 |
| **입력까지** pi 형식 | 0.15926 rad | **9.98x** |

## 못 한 것

| | 왜 |
|---|---|
| 실물 동작 | 시뮬만 |
| `RobotEnv` 의 속도 해석 규칙 | `droid` 패키지가 이 박스에 없음 (전체 검색 확인) |
| openpi 로 pi 체크포인트 실제 로드 | 코드만 읽음 |
| 실물 카메라/장면이 cosmos 학습 분포와 맞는지 | 시뮬 씬은 다른 장면 |
| A6000 에서의 지연/메모리 | 이 박스 숫자 |

**시뮬 성공률은 시뮬 안에서의 상대 비교다.** 실물 성능 예측이 아니다.

---

# 5. 위험 요소 한눈에

| # | 위험 | 증상 | 어디서 다루나 |
|---|---|---|---|
| **1** | **로봇의 속도 해석 규칙** | 팔이 느리고 계획보다 점점 뒤처짐 | §7 **0단계** (최우선) |
| **2** | **A6000 이 cosmos 를 제때 돌리나** | 청크 사이에 팔이 멈춤 | §6-3, §7 2단계 |
| 3 | `cosmos_framework` 설치 | 서버가 안 뜸 | §6-2 |
| 4 | `AbsoluteActions` 를 실수로 넣음 | 목표 2배. **예외 안 남** | 테스트로 고정됨 |
| 5 | 그리퍼 추가 반전 | 잡을 때 펴고 펼 때 잡음. 로그는 정상 | 테스트로 고정됨 |
| 6 | 청크 15 로 자름 | 성공률 63.9% → 35.4% | 32 고정 |
| 7 | 카메라 2장만 보냄 | 0/24. 정책이 눈이 멂 | §7 3단계가 거부 |
| 8 | LoRA 플래그를 따로 줌 | `lora_enabled=False` → 서버 사망 | §8 명령 참조 |
| 9 | pi horizon 15 vs 16 | 조용히 다른 길이 | §7 1단계 |
| 10 | 어댑터 쓸 때 팔이 막힘 | 최대 속도로 밀어붙임 | §7 5단계 경고 |

**1번과 2번만 실제 미지수다.** 나머지는 원인과 대응이 이미 확정돼 있다.

---

## 5-1. ★ 로봇의 속도 해석 규칙 — 유일한 실질 위험

로봇이 속도 `v` 를 받아 목표를 만들 때 무엇에 더하는가:

```
(A)  목표 = 지금 측정된 관절각 + v x 0.2
(B)  목표 = 직전에 보낸 목표   + v x 0.2
```

**시뮬 실측 (8 env, 기준선 63.9% → 8개면 기대 5):**

| 규칙 | 성공 | **의도 궤적 오차** | 실행 속도 |
|---|---|---|---|
| **(A)** 그대로 | **0/8** | **2.61 rad** | 0.0082 (헛돎) |
| **(B)** | 4/8, 2/8 | **0.0000** | 0.0073 |
| **(가)** 클라이언트 어댑터 | 3/8 | **0.176** | **0.0075** (기준선 0.00745) |

⚠ **성공률로는 (B)와 (가)를 구분하지 못한다.** (B)를 두 번 돌려 4/8, 2/8 이 나왔고 그 폭이
규칙 간 차이보다 크다. **판정 근거는 "의도 궤적 오차" 열이다** — 이건 잡음이 아니라 구조다.

**왜 cosmos 만 걸리나:** (A)의 오차는 **개방루프 길이에 비례**한다. pi 는 1~8스텝마다 다시
물어 안 쌓이고, cosmos 는 32스텝 연속이라 쌓인다. **"pi 가 실물에서 됐다" 는 cosmos 가 32에서
될지에 대해 아무것도 말해주지 않는다.**

**(A) 일 가능성이 높다.** 브랜치 문서가 *"the server-side conversion cannot correct mid-chunk
tracking error"* 라고 경고하는데, **(B) 라면 그 걱정을 할 이유가 없다.**

**단, 시뮬이 과장일 수 있다.** RoboLab 팔의 추종 지연이 0.0145 rad/스텝으로 **의도한 움직임
0.0074 의 2배**다. 실제 Franka 가 더 잘 따라가면 (A) 여도 버틴다. → §7 4단계에서 실측한다.

---

# 6. 배포 구성

## 6-1. 어디에 무엇을 두나

```
[A6000]  cosmos 정책 서버  (GPU, ~33 GiB)
         속도 프록시        (GPU 불필요, 같은 기계에 얹어도 공짜)
              │  포트 8001
[3090]   RobotEnv + 카메라 + 제어 루프   ← 요청 형식과 horizon 만 수정
[G15]    팔                              ← 무변경
```

**pi 서버와 cosmos 서버는 동시에 못 띄운다** (33 GiB + pi). 실험할 때 번갈아 띄운다.

## 6-2. A6000 에 cosmos_framework 설치 — **venv 를 복사하면 안 된다**

이 박스 venv(12 G)는 **B300 전용 휠**로 채워져 있다:

```
NATTEN-0.21.6.dev6+cu130.torch210.gb300    ← GB300/B300(sm_103a) 전용 커널
flash_attn / flash_attn_3_nv / transformer_engine  ...+cu130.torch210
torch 2.10.0+cu130
```

**A6000 은 Ampere(sm_86)** 이라 이 휠들이 안 돈다. 다행히 `pyproject.toml` 에 **`cu128` 그룹이
이미 있다** (`natten` 에 `gb300` 접미사가 없는 일반 빌드).

### 설치 절차

```bash
git clone https://github.com/NVIDIA/cosmos-framework.git
cd cosmos-framework && git checkout 2f603cb
# 아래 표의 파일 4개를 덮어쓴다
uv sync --group cu128        # ← 복사가 아니라 새로 만든다. Ampere 니까 cu128
```

### 가져갈 코드 — 파일 4개뿐

업스트림과 우리가 돌린 트리의 차이를 재보니 6개이고, 추론에 필요한 건 4개다:

| 경로 | 차이 | |
|---|---|---|
| `cosmos_framework/scripts/action_policy_server_robolab.py` | **923줄** | ✅ 서버 본체 |
| `cosmos_framework/data/generator/action/datasets/action_sft_dataset.py` | 수정 | ✅ |
| `cosmos_framework/data/generator/action/domain_utils.py` | 수정 | ✅ |
| `cosmos_framework/rl/` (폴더) | 신규 | ✅ 서버가 `sde_sampler` 를 import |
| `*.bak_nlsched`, `*.bak_rawdim` | 백업 | ❌ |

출처: `/dataset/personal/euniejeon/real-robot/cosmos-framework-rc365/`

### 설치 전에 확인

| | 왜 |
|---|---|
| **네트워크로 PyPI / NVIDIA 인덱스 접근되나** | `uv sync` 가 수 GB 를 받는다. **폐쇄망이면 여기서 막힌다** |
| 디스크 | venv 12 G + 체크포인트 31 G ≈ **45 G** |
| `uv` 설치 여부 | 없으면 먼저 |
| CUDA 드라이버 | cu128 휠 요구 버전 |

Python 3.13 은 `uv` 가 알아서 받는다.

## 6-3. ★ A6000 이 cosmos 를 제때 돌리나 — **반드시 먼저 재야 한다**

**cosmos 는 한 번 추론으로 32스텝 = 15 Hz 기준 2.13초 분량을 만든다.
추론이 2.13초보다 오래 걸리면 청크 사이에 팔이 멈춘다.**

| | |
|---|---|
| 이 박스(B300) 실측 | **931 ms** — 여유 있음 |
| A6000 | **미측정.** Ampere 는 B300보다 몇 배 느리다 |

**`check_cosmos_droid_velocity.py` 가 이미 이 판정을 출력한다**:

```
round_trip_latency_ms=931.4
min_open_loop_horizon_for_realtime=14      ← 32 이하여야 한다
```

이 값이 **32 를 넘으면 A6000 으로는 실시간이 안 된다.** 그때 선택지:
- 더 빠른 GPU
- `--num-steps` 를 8 미만으로 (**정책이 바뀐다. 시뮬 점수와 비교 불가**)
- cosmos 를 다른 기계에

**VRAM 은 여유가 있다** (33 GiB / 48 GiB). 병목은 속도다.

---
# 7. KETI 실행 절차

각 단계에 **합격 기준**이 있고, 통과해야 다음으로 간다.
**0단계와 2단계가 이 문서에서 계속 "미지수" 로 남아 있던 둘이다 — 거기서 막히면 그 위는 의미가 없다.**

```
0단계  속도 해석 규칙 판별        5분      ← 이후 전부가 여기 달림
1단계  A6000 설치 + pi 로드 확인  반나절
2단계  cosmos 서버 + 게이트       30분     ← 속도가 여기서 판정됨
3단계  3090 클라이언트 수정       1시간
4단계  no-send 프로브 (안 움직임)  30분
5단계  추종 오차 실측 (살짝 움직임) 30분    ← (A) 의 위험을 실제로 잼
6단계  첫 실주행 → 확대
```

## 0단계. 로봇의 속도 해석 규칙 판별 — 제일 먼저

**방법 A — 소스 읽기 (확실)**
`droid` 패키지 `robot_env.py` 의 `RobotEnv.step()`,
`action_space == "joint_velocity"` 분기가 무엇에 더하는지 본다.

**방법 B — 실물에서 가리기 (소스를 못 볼 때, 안전)**
팔을 조금 움직인 직후 **속도 0 을 5~10 스텝 보낸다.**

| 팔이 | 규칙 |
|---|---|
| **그 자리에 선다** | **(A)** — 목표가 측정각을 따라오므로 |
| **마지막 목표까지 계속 간다** | **(B)** — 목표가 안 바뀌므로 |

속도 0 이라 팔이 새로 가속하지 않는다.

**✔ 합격:** (A)인지 (B)인지 **글로 적어 남긴다.** 3단계에서 쓴다.

## 1단계. A6000 설치 + pi 로드 확인

§6-2 대로 `cosmos_framework` 설치. 같은 김에 **아직 한 번도 안 해본 것**을 확인한다:

```bash
python scripts/serve_pi05_droid_jointpos_velocity.py --checkpoint-dir <khy 경로> --port 8001
```

**✔ 합격:** 워밍업 2회 통과 후 포트가 열린다. (GPU 없이도 되지만 여기서 같이 한다.)

**같이 확정할 것 — pi 서빙 horizon 15 vs 16.** 브랜치가 같은 체크포인트를
`pi05_droid_jointpos`=16, `..._velocity`=15 로 올린다. `config.json` 은 셋 다 15지만
**로더가 그 파일을 안 읽고**, 가중치에도 단서가 없다(15/16 이 shape 에 들어간 텐서 0개).
**틀려도 예외가 안 난다.** ours 는 RLinf 에서 15로 학습된 것이 확실하므로 15.
base 는 브랜치 작성자에게 확인.

## 2단계. cosmos 서버 + 로봇 없이 게이트

```bash
# ① cosmos 정책 서버   ② 속도 프록시   ③ 게이트   (§8 명령 그대로)
python scripts/check_cosmos_droid_velocity.py --port 8001
```

**✔ 합격 기준**

| 출력 | 기대 |
|---|---|
| `action_shape` | **(32, 8)** |
| `actions_finite` | true |
| `joint_limit_fraction` | **< 0.01** (우리 측정 0.000~0.0004) |
| **`min_open_loop_horizon_for_realtime`** | **≤ 32** ← **여기서 A6000 속도가 판정된다** (§6-3) |
| `reconstructed_target_first` | 현재 관절각 근처 |

**✘ 실패하면**

| 증상 | 원인 후보 |
|---|---|
| `min_open_loop_horizon_for_realtime` > 32 | A6000 이 너무 느리다 → §6-3 선택지 |
| `applied < len(state)` 로 서버 사망 | LoRA 플래그를 따로 줬다 → 한 플래그에 4개 |
| shape 이 (15,8) | 프록시 `--action-horizon` 이 32 가 아님 |
| `joint_limit_fraction` 이 크다 | 정책이 장면을 못 본다. 카메라 요청부터 의심 |
| `strict` 로드 예외 | 체크포인트 레이아웃 (§8) |

## 3단계. 3090 클라이언트 수정

**반드시 바꿀 것 3개:**

1. **요청에 카메라 3장** — `_extract_observation` 이 이미 left/right/wrist 를 다 뽑는다.
   `observation/exterior_image_1_left`(왼쪽) / `observation/exterior_image_2_left`(오른쪽) /
   `observation/wrist_image_left` 셋 다 넣는다. **프록시가 합성한다.**
2. `action_horizon` 15 → **32**
3. `open_loop_horizon` 8 → **32**

**로봇 `RobotEnv("joint_velocity")` 와 G15 는 건드리지 않는다.**

### 0단계가 **(A)** 였다면 — 어댑터 15줄 추가

```python
# 청크를 새로 받으면 q_star = 그 시점의 측정 관절각
# 매 스텝:
q_star = q_star + v[k] * 0.2                             # 서버가 의도한 절대 목표
v_cmd  = np.clip((q_star - q_measured) / 0.2, -0.5, 0.5)  # 지금 측정각 기준으로 다시 계산
env.step(concat([v_cmd, gripper]))
```

뒤처지면 명령이 커져 **오차가 쌓이는 게 아니라 교정된다.**
시뮬 검증: 0/8 → 3/8, 실행 속도가 기준선과 일치(0.00754 vs 0.00745).

⚠ **(B) 였다면 넣으면 안 된다.** 오히려 틀려진다.

**✔ 합격:** 요청 한 번에 프록시 로그가 `image=(540, 640, 3)` 을 찍는다.

## 4단계. no-send 프로브 — 팔을 움직이지 않는다

브랜치 기존 절차 그대로. `RobotEnv(do_reset=False)` 만 만들고
`reset` / `step` / `update_robot` 을 **한 번도 부르지 않는다.**

**✔ 합격**
- 카메라 **3뷰 전부** 육안 정상 (pi 는 2뷰였다. **오른쪽 외부 카메라가 새로 들어간다**)
- `(32, 8)` 유한값
- 합성 프레임 한 장 저장해 눈으로 확인 — **위 손목, 아래 왼쪽|오른쪽**

**✘ 좌우가 바뀌었거나 패딩이 다르면 학습 분포를 벗어난다.**

## 5단계. 추종 오차 실측 — 아주 작은 움직임

**(A) 의 위험을 실제로 재는 단계.** 작은 움직임을 주며 **명령한 목표 vs 실제 관절각**을
매 스텝 로그로 남긴다.

| 추종 오차 (중위) | 판정 |
|---|---|
| **< 0.002 rad** | (A) 여도 32스텝 개방루프가 안전. **어댑터 없이도 된다** |
| 0.002 ~ 0.007 rad | (A) 면 어댑터를 쓴다 |
| **> 0.007 rad** | 시뮬(0.0145)과 같은 영역. (A) 면 **어댑터 필수**, 그래도 클립이 자주 걸림 |

기준: cosmos 가 의도하는 움직임이 **스텝당 0.0074 rad**. 추종 오차가 이것과 같은 자리수면
(A) 에서 누적이 심각하다.

## 6단계. 첫 실주행 → 확대

⚠ **pi 의 안전 절차를 그대로 쓸 수 없다.** pi 는 `open_loop_horizon=1` → 8 로 시작하는데
**cosmos 는 추론이 ~1초라 1도 8도 실시간으로 못 준다** (15 Hz 에서 8스텝 = 533 ms).

**개방루프를 줄이는 대신 총 스텝 수를 제한한다. 이건 KETI 와 합의가 필요한 변경이다.**

| | pi 절차 | cosmos 제안 |
|---|---|---|
| 개방루프 | 1 → 8 | **32 고정** (줄일 수 없음) |
| 첫 주행 | 30 스텝 | **32 스텝 = 청크 하나**, 그 뒤 정지 |
| 확대 | 300 스텝 | 64 → 128 → 300 |

- E-stop / 리셋 경로 먼저 확인
- 청크 하나(2.1초) 실행 후 멈춰 팔 위치 확인
- 프록시 로그의 `saturation_fraction` 을 본다. **크면 정책이 장면을 못 본다는 신호**
  (시뮬: 정상 0.0004, 장면을 못 보는 팔 0.0045 — 10배)

### ⚠ (A)+어댑터 를 쓸 때

팔이 뭔가에 걸리면 **간격이 벌어질수록 명령이 커져 최대 속도로 밀어붙인다.**
(A) 를 그대로 쓰면 목표가 팔 근처에 머물러 안 밀어붙인다 — 성능은 최악이지만 가장 순하다.

**권고:** 첫 주행은 **(A) 그대로, 어댑터 없이** 한 청크만. 거동을 보고 어댑터를 붙인다.
어댑터를 상시로 쓸 거면 **"같은 관절이 N스텝 연속 포화하면 정지"** 안전 레이어를 위에 둔다.

### 확대하며 볼 것

| | 정상 |
|---|---|
| `saturation_fraction` | 작고 안정적 |
| 추종 오차 | 5단계 값에서 안 커짐 |
| 팔 궤적 | 계획대로. **점점 뒤처지면 (A) 누적을 의심** |

---

# 8. 부록

## 8-1. 가져갈 체크포인트 — 4개, 약 45 G

| | 경로 | 크기 |
|---|---|---|
| pi 베이스 | `khy5630/.../pi05_droid_jointpos_pytorch` | 6.8 G |
| pi ours | `jeeit18/.../q24_gae099off50_step6_openpi_pytorch` | 6.8 G |
| cosmos 베이스 | `transfer_20260907/cosmos3_base` | 31 G |
| cosmos ours | `transfer_20260907/cosmos_fixedscene_r3/overlay.pt` | 62 M |

pi 폴더는 **통째로** — `assets/droid/norm_stats.json` 이 빠지면 **에러 없이 역정규화가 틀린다.**
`transfer_20260907/pi_base` 는 필요 없다 (khy 것과 같은 가중치, RLinf 전용 형식).

## 8-2. 띄우기

```bash
# pi (베이스/ours 둘 다 동일, --checkpoint-dir 만 교체)
python scripts/serve_pi05_droid_jointpos_velocity.py \
    --config pi05_droid_jointpos_velocity --checkpoint-dir <경로> --port 8001

# cosmos: 베이스를 HF 캐시 자리에 놓고
cp -a <...>/cosmos3_base $HF_HOME/hub/models--nvidia--Cosmos3-Nano-Policy-DROID

# ① cosmos 정책 서버 (업스트림 코드, 수정 없음)
python -m cosmos_framework.scripts.action_policy_server_robolab \
    --checkpoint-path nvidia/Cosmos3-Nano-Policy-DROID --port 8010 \
    --num-steps 8 --guidance 3.0 --shift 3.0 --stochastic-sampler --sampler-deterministic \
    # +ours 일 때만
    --param-overlay <...>/cosmos_fixedscene_r3/overlay.pt \
    --experiment-overrides model.config.lora_enabled=True model.config.lora_rank=16 \
        model.config.lora_alpha=32 \
        model.config.lora_target_modules=q_proj_moe_gen,k_proj_moe_gen,v_proj_moe_gen,o_proj_moe_gen

# ② 속도 프록시 (모델 없음, GPU 없음)
python scripts/serve_cosmos_droid_velocity.py --upstream-port 8010 --port 8001

# ③ 로봇 없이 도는 게이트
python scripts/check_cosmos_droid_velocity.py --port 8001
```

⚠ **LoRA 오버라이드 4개는 반드시 한 플래그 안에.** 따로 주면 마지막 것만 남아
`lora_enabled=False` 로 뜨고, 서버가 `applied < len(state)` 가드로 죽는다.

**네 경우 모두 코드 수정은 없다.** 바뀌는 건 인자와 환경변수뿐이다.

## 8-3. 로그 어디를 보나

| 로그 | 무엇이 보이나 |
|---|---|
| 프록시 stdout | `First request translated: image=... -> cosmos chunk (32,8) -> velocity (32,8)` |
| `JointPositionToDroidVelocity` | `raw_min= raw_max= saturation_fraction= limit= horizon=` **매 요청** |
| cosmos 서버 | 부팅 시 `action_space=joint_pos action_dim=8 chunk=32`, 매 요청 `prompt= seed=` |
| `check_...py` | `min_open_loop_horizon_for_realtime`, `reconstructed_target_*` |

## 8-4. 증상 → 먼저 의심할 것

| 증상 | |
|---|---|
| 팔이 엉뚱한 데로 크게 감 | `AbsoluteActions` 가 사슬에 남음 (목표 2배) |
| 잡을 때 펴고 펼 때 잡음 | 그리퍼를 어딘가에서 한 번 더 뒤집음 |
| **팔이 느리고 점점 뒤처짐** | **(A) 누적.** 0단계 결과와 5단계 추종오차 확인 |
| 움직임이 뚝뚝 끊김 | 개방루프 32 미만이거나 A6000 이 못 따라감 (§6-3) |
| 성공률 낮고 포화율 높음 | 정책이 장면을 못 봄 → 카메라 3장 요청 확인 |
| 서버가 부팅에서 죽음 | LoRA 플래그 / 체크포인트 레이아웃 / `HF_HOME` |

## 8-5. 옛 pi 숫자를 인용할 때

`pi05_droid_jointpos` 는 릴리스가 둘이고(**polaris** / **simeval**) 가중치가 다르다
(`action_in_proj.weight` 상대평균 0.53%, 반올림으로 설명 안 됨).
**우리 base(khy)와 ours 는 둘 다 simeval 혈통** — 비트 단위로 검증했으므로 **우리끼리 비교는
안전하다.** 위험한 건 "예전에 pi 가 몇 점이었다" 를 우리 숫자와 나란히 놓을 때다.

---

# 9. 3090 변경 사항 — KETI A6000 실측(2026-09-14) 후 정리

A6000 쪽은 끝났다. `scripts/start_cosmos_droid_velocity.sh 3` → `status` 가 `READY` 면
프록시가 **포트 8000** 에서 pi 와 같은 속도 계약으로 응답한다 (`(32, 8)`, 실측 왕복 **8.0초**).
3090 은 아래 표대로만 바꾼다. **로봇·G15·카메라 하드웨어는 무변경.**

## 9-1. 바꿀 것

| # | 구분 | 파일 / 위치 | pi | cosmos | 안 바꾸면 |
|---|---|---|---|---|---|
| 1 | 필수 | 클라이언트 `Args.action_horizon` | 15 (또는 16) | **32** | `assert pred_action_chunk.shape == (action_horizon, 8)` 즉사 |
| 2 | 필수 | 클라이언트 `Args.open_loop_horizon` | 8 | **32** | 8스텝마다 8초 대기, 성공률 반토막 (시뮬 63.9% → 35.4%) |
| 3 | 필수 | 클라이언트 `request_data` 이미지 | 외부 1장(`external_camera`) + 손목, 각 224×224 | **3장 원본**: `observation/exterior_image_1_left` = `curr_obs["left_image"]`, `observation/exterior_image_2_left` = `curr_obs["right_image"]`, `observation/wrist_image_left` = `curr_obs["wrist_image"]`. **224 리사이즈 제거** (프록시가 360×640 기준으로 합성) | 정책이 장면을 못 봄 (시뮬 0/24) |
| 4 | 필수 | 클라이언트 `external_camera` assert | left/right 필수 | 제거 (3장 다 보내므로 선택이 없음. 영상 저장용으로만 남겨도 됨) | assert 로 시작 불가 |
| 5 | 필수 | `no_send_openpi_snapshot.py` 통과 조건 | `action_shape=(10, 8)` | **`action_shape=(32, 8)`** + 요청 3장 (3번과 동일) | no-send 게이트 오판 |
| 6 | 필수 | no-send 저장 이미지 육안 확인 | left·wrist 2뷰 | **right 포함 3뷰**. 오른쪽 외부 카메라(`31194385`)가 이번엔 정책 입력이다 | 오른쪽 카메라 이상을 못 잡음 |
| 7 | 필수 | `run-once-openpi.sh` 안전 절차 | `open_loop_horizon` 1→8 늘려감, 30→300스텝 | 개방루프 **32 고정**, `max_timesteps` **32 → 64 → 128 → 300** | 1 이나 8 은 실시간 불가 (§6) |
| 8 | 조건부 | `env.step` 직전 어댑터 15줄 (§7 3단계) | 없음 | 0단계가 **(A)** 일 때만 추가. **(B) 면 넣지 않는다** | (A) 에서 32스텝 누적 오차 |
| 9 | 확인 | `droid/robot_env.py` `RobotEnv.step()` 의 `joint_velocity` 분기 | — | **(A)/(B) 판별해 글로 기록** (§7 0단계) | 8번 결정 불가 |
| 10 | 인지 | 실주행 체감 | 연속 동작 | 요청마다 **~8초 정지 → 2.1초 이동** 반복. 정상 동작이다 (9-3) | 멈춤을 고장으로 오해 |
| — | 무변경 | `remote_host` / `remote_port` | 10.252.205.103 / **8000** | 동일 | |
| — | 무변경 | `RobotEnv(action_space="joint_velocity", gripper_action_space="position")` | | 동일 | |
| — | 무변경 | 관절 속도 클립 ±0.5, 그리퍼 이진화(>0.5) 와 [0,1] 클립, 15Hz sleep, `prevent_keyboard_interrupt` | | 동일 | |
| — | 무변경 | G15 컨트롤러, 카메라 ID 3개, `droid-client-gate.sh`, `droid-workstation-audit.sh` | | 동일 | |

3번의 재료는 이미 있다. `_extract_observation` 이 left/right/wrist 를 다 뽑아 돌려주고,
지금은 요청을 만들 때 하나를 버린다. 카메라를 늘리는 게 아니라 요청 형식만 바꾸면 된다.

## 9-2. 선결 문제 — 3090 → A6000 8000 포트

2026-09-14 pi05 서버(8000)로 시도했을 때 **3090 의 연결이 A6000 에 한 번도 도착하지 않았다.**
A6000 의 ufw 가 켜져 있다. 3090 에서 먼저 확인한다:

```bash
curl -sv --max-time 3 http://10.252.205.103:8000/healthz     # 기대: OK
```

timeout 이면 방화벽이다. A6000 에서 `sudo ufw allow 8000` (snu 계정은 sudo 비번 필요).
"connection refused" 면 서버가 안 떠 있는 것이니 A6000 에서 `scripts/status_cosmos_droid_velocity.sh`.

## 9-3. 왜 8초 정지가 문제가 아닌가

A6000 실측: `--num-steps 8` 에서 왕복 8.0초, `--num-steps 4` 에서 4.1초 (스텝당 ~1초, 선형).
예산 2.13초(32스텝 ÷ 15Hz)에 못 미쳐 **연속 실행은 불가**하다. 그러나:

- 3090 은 32스텝을 다 쓴 뒤 **그 시점의 관측**으로 요청하고, 응답까지 `env.step()` 을 부르지 않으므로
  팔은 마지막 목표에 서 있다. 관측은 요청 시점 것이고 로봇 상태는 그 동안 안 바뀐다.
- 정책 입력은 `history=1`, 상태는 관절각 7 + 그리퍼 1 (속도 없음). **멈춰서 찍은 관측과
  움직이다 찍은 관측을 정책은 구분하지 못한다.**
- 시뮬(63.9%)도 스텝 기반이라 추론 중 세계가 멈춘다. 즉 시뮬은 애초에 "추론 시간 0초의 stop-and-go" 였다.

남는 것은 (1) 청크 경계에서 마지막 목표에 못 미친 채 멈추는 추종 오차 — §7 5단계가 그대로 잰다,
(2) 밀기·붓기 같은 동적 동작은 정지 중 물체가 움직일 수 있다 — 집기-놓기는 준정적이라 영향 적다,
(3) 작업 시간이 약 4배 — 타임아웃만 늘린다. 정책 조건은 시뮬과 같으므로 `--num-steps` 는 **8 유지.**

## 9-4. A6000 운용 명령 (참고)

```bash
cd /data/keti/snu/workspace/cosmos3-droid
scripts/start_cosmos_droid_velocity.sh 3            # base. ours: VARIANT=ours scripts/start_... 3
scripts/status_cosmos_droid_velocity.sh             # SERVER_STATUS=READY 까지 반복 (1~2분)
.venv/bin/python scripts/check_cosmos_droid_velocity.py --port 8000    # 로봇 없이 게이트
scripts/stop_cosmos_droid_velocity.sh
# 로그: /data/keti/snu/home/runtime/cosmos-droid-velocity/logs/{cosmos,proxy}-*.log
```

pi 와 cosmos 는 **같은 GPU 3 에 동시에 못 뜬다** (33.6 GiB + 8.9 GiB > 48 GiB 여유 아님).
pi 를 쓰려면 `stop_cosmos_droid_velocity.sh` 먼저.

## 9-5. 이 문서와 실측이 어긋난 곳 (A6000 설치 중 발견)

| 문서 | 실제 |
|---|---|
| §6-2 덮어쓸 파일 4개 | **9개.** `robocasa365_lerobot_dataset.py` 가 빠지면 `action_sft_dataset` import 에서 즉사. 나머지(`configs/base/config.py` 2줄, robocasa365 config 2개, robocasa365 서버)는 트리 일치용 |
| §6-2 `uv sync --group cu128` | 서버가 쓰는 `webdataset`(train extra), `openpi-server`(policy-server 그룹)가 빠짐. `uv sync --all-extras --group=cu128-train --group=policy-server` |
| §6-2 "uv 설치 여부" | 버전. 저장소가 uv ≥0.11.3 요구. A6000 기본 uv 0.9.16 → `/data/keti/snu/tools/uv-latest/uv` (0.12.13) 별도 설치 |
| §6-2 설치 전 확인 | **`nvidia/Cosmos-Guardrail1` (gated HF 저장소)** 승인 + 토큰이 빠져 있음. 없으면 서버가 부팅 중 죽는다. B300 은 캐시가 있어 안 보였음 |
| §8-2 `--checkpoint-path nvidia/Cosmos3-Nano-Policy-DROID` | 서버가 `uvx hf download` 를 호출하므로 PATH 에 uvx 필요. start 스크립트는 스냅샷 **로컬 경로**를 직접 넘긴다 |
| §6-3 "check 스크립트가 지연을 판정" | openpi 클라이언트의 keepalive ping(20초)이 첫 추론(17~40초) 중 끊겨 판정 전에 죽었음. 프록시의 upstream 연결만 ping 을 끄는 수정으로 해결 (`serve_cosmos_droid_velocity.py`) |
| 전체 포트 8001 | 3090 절차는 **8000.** 프록시 기본값을 8000 으로 |
| §6-3 실시간 판정 | `min_open_loop_horizon_for_realtime=121` (>32). 그러나 9-3 의 이유로 stop-and-go 로 진행 |

## 9-6. 0단계 결과 — (A) 확정, 어댑터 적용 (3090, 2026-09-14)

3090 측 소스 확인: `droid/franka/robot.py:233` 이 매 step `목표 = 현재 측정각 + v×0.2` 로 계산한다 → **(A).**
스케일 0.2 의 출처는 `droid/robot_ik/robot_ik_solver.py:88`. 어댑터(§7 3단계)를 적용했고,
reference 는 청크 시작 시 측정 관절각으로 초기화, 보정 명령도 ±0.5 안에서만, 마지막 목표까지 강제 이동 없음.

**"0.2 가 시뮬과 같은가"** 에 대한 답: 시뮬은 속도를 쓴 적이 없다. cosmos 는 절대 관절각을 내고 시뮬은
그걸 그대로 목표로 넣었다. 0.2 는 프록시↔3090 사이의 **전송 스케일**일 뿐이며, 조건은 세 곳이 같은 값이라는 것 하나다:
프록시 `joint_delta_scale=0.2` (metadata 로 노출), 어댑터 0.2, 로봇 `robot_ik_solver.py:88` 의 0.2.
시뮬의 "출력만 pi 형식" 실행(§4: ÷0.2 → 클립 → ×0.2 누적 적분)이 baseline 과 같은 성공 풀,
스텝 0 액션 차이 잡음 수준(1.00x)이었던 것이 이 왕복이 무손실이라는 실측이다.
0.2 가 실제로 개입하는 곳은 **클립**뿐: 한 step 에 0.1 rad(=0.5×0.2) 넘게 움직이려 하면 잘린다.
시뮬 통계 median 0.037 / p99 0.21 / 포화 0.01~0.04% (속도 단위) 라 무시 가능. 실물에선 프록시 로그
`saturation_fraction` 으로 감시한다.

권고: 3090 어댑터가 0.2 와 0.5 를 하드코딩하지 말고 서버 metadata 의
`joint_delta_scale`, `max_abs_joint_velocity` 와 시작 시 일치 확인(assert)할 것.
