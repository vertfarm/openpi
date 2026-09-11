# HV1 은색 실린더: 올나이트 학습 / 2026-09-09

> **기록 전용 (2026-09-11).** `overnight_*` 모듈은 삭제했으므로 아래 명령은 더 이상
> 실행되지 않는다. 이 문서는 A~F 캠페인 기록이며, 현재 운영 절차는
> [TRAINING.md](../TRAINING.md)를 따른다.
> 코드가 필요하면 `git show 3676ccb^:examples/hv1/overnight.py`로 꺼낸다.

이 기능은 학습·기록 영상 추론 전용이다. ROS import, ROS publisher, MQTT 송신,
로봇 enable, 자동 실기 rollout은 포함하지 않는다. 문종술 선임의 ROS2 WIP는 수정하지 않는다.

## 데이터 계약

- 원본: 현장 ROS2 저장소의 `datasets/keti_humanoid_data_260909`.
- 원본 31개 중 `000000`, `000001`은 사용자 지정 더미로 제외. 원본은 삭제/수정하지 않는다.
- 29개 / 23,267 rows / HEVC 영상 87개를 전부 디코딩하고 HDF5·tasks·영상 해시를 보관한다.
- 오른팔 관절 명령 7개와 측정 7개는 각각 이름으로 대응한다. 측정 손 관절 위치 8개를 추가해 state 15차원이다. effort는 원본에 보존하되 이번 모델 입력에서 제외한다.
- action은 오른팔 목표 7개 + 닫기 의도 1개(총 8차원). 팔에만 state 기준 delta 변환을 적용하고 추론 후 절대 목표로 복원한다.
- mode=2, raw open `0.6 → 0 → 0.6`을 grasp intent `0 → 1 → 0`으로 변환한다. torque on/off나 파지 성공 신호가 아니다. 실기 매핑은 wrap 닫기 / narrow 0.6 열기다.
- head/hand_l/hand_r 모두 RGB 224×224로 종횡비 유지·padding 후 저장한다. 원본 640×480은 보존한다. 이미지별 source camera identity를 유지하며 camera dropout은 없다.
- 영상 N번째 프레임은 recorder N번째 행과 대응한다. 원본 timestamp/stamp_ns를 provenance에 보존하며, 이 대응을 센서 노출 시각 동기화 검증으로 주장하지 않는다.

학습·추론 prompt:

> Pick up the silver cylindrical part from the table with the right hand and place it on the tray.

원본 `pick object`는 그대로 두고 학습 export에만 override한다.

## 공통 분할과 실험

공통 검증은 `000027`~`000032`의 6개 / 4,895 rows다. 세 카메라의 파지·운반·놓기·끝 장면표를 사람이 아닌 assistant가 육안 검토한 증거를 별도 JSON에 남겼다. 이는 전 구간의 하드웨어 안전 검증이 아니다. `000029`는 release 직후 녹화가 끝나 retraction이 짧다.

| 실험 | 학습 데이터 | 변경점 |
|---|---|---|
| A | 의심 4개 제외 19개 / 14,377 rows | full, seed 42, lr 2.5e-5, horizon 15 |
| B | 의심 4개 포함 23개 / 18,372 rows | A 대비 데이터만 변경 |
| C | A와 동일 | lr 1e-5, 최종 lr 1e-6 |
| D | A와 동일 | horizon 30 |
| E | A와 동일 | seed 7 |
| F | A와 동일 | 공식 OpenPI LoRA 모델 variant / freeze mask |

의심 episode는 `000004`, `000008`, `000013`, `000020`. 이 표시는 관절 목표 급변 후보이며 확진된 기구 불량이라는 뜻이 아니다.

모든 실험은 공식 pi05_base에서 독립 초기화한다. 공통 normalization은 A의 19개,
horizon 15의 episode-boundary-clamped chunk에서만 계산한다. D도 같은 통계를 사용한다.
같은 수집 세션의 episode holdout이므로 새로운 날짜·배치·물체에 대한 일반화 점수로 해석하지 않는다.

## 실행

현장 repo는 `/home/keti/workspace/openpi-hv1`, 캠페인은
`/home/keti/workspace/hv1-vla-runtime/overnight-20260909`다. 원본·runtime 경계를 검사한다.

```bash
# 이미 수행된 준비 단계. 존재하는 출력에는 덮어쓰지 않는다.
.venv/bin/python -m examples.hv1.native scan --source RAW_DATASET --output CAMPAIGN/scan
.venv/bin/python -m examples.hv1.native export --source CAMPAIGN/scan/scan.json --output CAMPAIGN/export_clean --cohort clean
.venv/bin/python -m examples.hv1.native export --source CAMPAIGN/scan/scan.json --output CAMPAIGN/export_inclusive --cohort inclusive
# validation_review.json은 실제 장면 검토 후 scan hash와 묶는다.
JAX_PLATFORMS=cpu .venv/bin/python -m examples.hv1.overnight_train stats --campaign CAMPAIGN
JAX_PLATFORMS=cpu .venv/bin/python -m examples.hv1.overnight_check --campaign CAMPAIGN --experiment A
JAX_PLATFORMS=cpu .venv/bin/python -m examples.hv1.overnight_check --campaign CAMPAIGN --experiment B
JAX_PLATFORMS=cpu .venv/bin/python -m examples.hv1.overnight_check --campaign CAMPAIGN --experiment D

# 한 번만 실행. supervisor.lock과 GPU lock이 중복 실행을 막는다.
.venv/bin/python -u -m examples.hv1.overnight --campaign CAMPAIGN
```

실행 순서는 50-step smoke → 전체 checkpoint/BF16 추론 동일성 확인 → A/B 공통 step 목표 결정 → C/D/E/F → 기록 영상 inference sweep이다. 09:30 KST 신규 학습 중단, 10:00 KST 검증 종료가 고정돼 있다. 물리 로봇은 켜지지 않는다.

모델 OOM인 경우 smoke에서만 batch 1을 별도 experiment로 재시험하고 이후 실험 모두 같은 batch를 사용한다. 다른 오류는 자동으로 무시하지 않는다.

## 체크포인트와 검증

- `snapshots/EXPERIMENT/step_NNNNNN`: BF16 inference params와 assets, 원본/코드/recipe/통계 hash. 파일 hash 및 실제 배열 복원이 끝나야 `snapshot.json`과 최종 폴더가 공개된다.
- `restarts/pi05_hv1/EXPERIMENT`: optimizer 포함 최신 학습 상태. 완료된 실험의 최종 BF16 GPU 재로딩을 확인한 뒤 이 캠페인 소유의 restart만 정리한다. `runs/EXPERIMENT/restart_pruned.json`에 경로·용량·사유를 남긴다. 추론용 snapshot과 원본은 삭제하지 않는다.
- `evaluations/`: 기본 10-step denoising의 loss 및 5/10/20-step 추론 비교. 고정 noise로 비교한다. 실행 prefix 1/3/5의 목표 변화와 모델 지연 예산을 기록한다. 실제 네트워크 freshness 및 closed-loop 성공률은 측정하지 않는다.
- `checkpoint_index.json`, `CHECKPOINTS.md`: 재로딩 상태를 합친 색인. smoke는 진단용이며 본학습 A~F와 구분한다.
- `status.json`, `logs/`, `runs/`, `allocation.json`: 상태, 실제 처리속도, 학습량 및 시간 배정 근거.

오프라인 loss가 작다는 이유만으로 실제 로봇 투입을 승인하지 않는다. 실기 executor의 명령 ownership·관절 한계·속도 제한·freshness·비상정지 확인은 별도다.

재시작은 동일 recipe에 `overnight_train train --resume`을 명시한다. 완료 후 restart를 정리한 실험은 optimizer resume가 불가능하고 BF16 inference만 가능하다. recipe/출력/로그가 이미 있으면 자동 덮어쓰거나 새로운 실험으로 오인하지 않는다.

이미 smoke를 통과한 실행기의 명시적 재연결은 `overnight --after-smoke --resume-campaign`이다.
이 경우 살아 있는 기존 학습 자식의 정확한 command line을 확인하고 기다린 뒤 이어서 실행한다.
기존 자식의 PID만 믿고 다른 프로세스를 조작하거나 같은 학습을 중복 시작하지 않는다.
사용자 승인 없이 임의로 학습 프로세스를 재시작하지 말고 상태·로그·완료 결과를 먼저 확인한다.

## 테스트

```bash
JAX_PLATFORMS=cpu .venv/bin/python -m pytest examples/hv1/tests -q
```

기존 회귀와 함께 native 명령/상태 분리, mode/open/stale/timestamp/NaN 거부,
분할/seed/시간 예산/저장 시점, 실제 Orbax BF16 roundtrip을 검증한다.
실제 데이터의 공식 loader에서는 A 76개, B 92개, D 76개 경계·파지/해제 지점을 검사했다.
