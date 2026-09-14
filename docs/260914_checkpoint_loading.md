# base pi0.5 대신 khy5630 체크포인트를 싣는 경로 — 코드 위치와 패치

작성 2026-09-14 16:50 · 대상 트리 `/dataset/personal/bhlee/RL_VLA/0827_opsd/opsd_260831_eval_setup`
관련 — 평가 프로토콜 `2026-09-09_1730_PI05_450STEP_EVAL_PROTOCOL.md` · 감사 `2026-09-09_1700_EVAL_SANITY_AUDIT.md` ·
캠페인 `opsd_vague_eval/README.md` · `ppo_vague_eval/README.md`

---

## 0. 한 줄 결론

**openpi 의 모델 로딩 코드는 한 줄도 고치지 않았다.** 체크포인트 교체는 openpi 바깥 세 지점에서만 일어난다 —
① 심링크, ② 정책 이름 → 디렉터리 규칙 1줄, ③ 서버 기동 분기 3줄. 우리가 **추가**한 파일은
`serve_policy_seeded.py`(RNG 고정 래퍼) 하나뿐이고, 이것도 공유 트리를 수정하지 않고 감싸기만 한다.

---

## 1. 전체 체인

```
run_opsd_epoch_eval.sh:50          POL="hd5opsdb_s80"
        │                          (PPO 는 run_ppo_seed5_balanced.sh:40  POL="hd5ppo80_s80")
        ▼  POLICIES_OVERRIDE
eval_multitask.sh:158,160-170      policy_dir()  hd5* → $CKPT/hard5_${이름#hd5}_pytorch
        ▼
output/rl/ckpt/hard5_opsdb_s80_pytorch          ← 우리 트리 (심링크)
        ▼  심링크
/dataset/personal/khy5630/robot/opsd/output/rl/ckpt/hard5_opsd_base_s80_pytorch
        ▼  dir 인자
lib_serve.sh:81-96                 ensure_server → serve_policy_seeded.py
                                   --policy.config=pi05_droid_jointpos --policy.dir=<dir>
        ▼
libero/openpi/scripts/serve_policy.py           ← 업스트림 그대로
        ▼
src/openpi/policies/policy_config.py:48-55      model.safetensors 있으면 pytorch 분기
        ▼
src/openpi/models/model.py:243-246              safetensors.torch.load_model(model, weight_path)
        ▼
src/openpi/models_pytorch/pi0_pytorch.py        PI0Pytorch (수정 없음)
```

---

## 2. 단계별 코드 (절대경로 · 줄번호)

### 2.1 정책 이름 주입

`/dataset/personal/bhlee/RL_VLA/0827_opsd/opsd_260831_eval_setup/train/run_opsd_epoch_eval.sh:50`

```bash
POL="hd5opsdb_s${ep}"
```

`/dataset/personal/bhlee/RL_VLA/0827_opsd/opsd_260831_eval_setup/train/run_ppo_seed5_balanced.sh:40,57-58`

```bash
POL="hd5ppo80_s${ep}"
...
PERTURB=pose10_perenv PERTURB_SEED="$SEED" OPSD_SERVE_SEED="$SEED" \
POLICIES_OVERRIDE="$POL" TASKS_OVERRIDE="$TASKS" \
```

유닛 러너 경로로 돌 때는 `train/openpi/eval_units_par.sh:104` · `eval_units_par2.sh:107` 이 `OPSD_SERVE_SEED="$seed"` 를 세운다.

### 2.2 이름 → 디렉터리 규칙

`/dataset/personal/bhlee/RL_VLA/0827_opsd/opsd_260831_eval_setup/train/openpi/eval_multitask.sh`

```bash
:34    CKPT="$OPSD/output/rl/ckpt"
:158   POLICIES="${POLICIES_OVERRIDE:-$POLICIES}"
:160   policy_dir(){
:162       base) echo "$CKPT/pi05_droid_jointpos_pytorch" ;;      # ← base pi0.5 (다른 경로)
:170       hd5*) echo "$CKPT/hard5_${1#hd5}_pytorch" ;;           # ← 우리 경로
:196   local dir; dir="$(policy_dir "$pol")"
:215   local port; port="$(SRV_SHARD=$shard ensure_server "$pol" "$dir" "$gpu" $((8800 + shard * 40)))"
```

`hd5opsdb_s80` 에서 접두사 `hd5` 를 떼고 `hard5_…_pytorch` 로 감싼다 → `hard5_opsdb_s80_pytorch`.
**base 는 이 규칙을 타지 않는다** — `base` 라는 이름은 `:162` 에서 전혀 다른 디렉터리로 간다.
그래서 두 모델이 섞일 수 없다.

### 2.3 서버 기동 (venv·진입점 분기)

`/dataset/personal/bhlee/RL_VLA/0827_opsd/opsd_260831_eval_setup/train/openpi/lib_serve.sh:81-96`

```bash
:81    local SERVE_ENTRY="scripts/serve_policy.py"
:82    if [ -n "${OPSD_SERVE_SEED:-}" ]; then
:83      SERVE_ENTRY="$HERE/serve_policy_seeded.py --seed ${OPSD_SERVE_SEED}"
:84    fi
:85    local pybin="$OPENPI/.venv/bin/python" pypath=""
:86    if [ -f "$dir/model.safetensors" ]; then        # ← pytorch 체크포인트 판정
:87      pybin="$HERE/.venv/bin/python"; pypath="$OPENPI/src"     # ← 오버레이 venv
:88    fi
...
:95    setsid "$pybin" $SERVE_ENTRY --port "$port" policy:checkpoint \
:96      --policy.config="${POLICY_CFG:-pi05_droid_jointpos}" --policy.dir="$dir" \
```

변수 정의는 `eval_multitask.sh:30-33`:

```bash
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # …/opsd_260831_eval_setup/train/openpi
OPSD=/dataset/personal/bhlee/RL_VLA/0827_opsd/opsd_260831_eval_setup
OPENPI=/dataset/personal/bhlee/RL_VLA/0827_opsd/libero/openpi
```

**오버레이 venv 를 쓰는 이유 (B300)**: `$OPENPI/.venv` 의 torch 는 sm_103(B300)을 모르고 triton ptxas 도
`sm_103a` 를 모른다(2026-08-12 실측). 그래서 pytorch 체크포인트만 `train/openpi/.venv`(torch 2.7.0+cu128)로
띄우고 `PYTHONPATH=$OPENPI/src` 로 openpi 소스를 붙인다. 같은 줄에서 `TORCH_COMPILE_DISABLE=1
TORCHDYNAMO_DISABLE=1 PYTORCH_JIT=0` 도 함께 건다.

### 2.4 업스트림 로더 (수정 없음)

`/dataset/personal/bhlee/RL_VLA/0827_opsd/libero/openpi/src/openpi/policies/policy_config.py:48-55`

```python
# Check if this is a PyTorch model by looking for model.safetensors
weight_path = os.path.join(checkpoint_dir, "model.safetensors")
is_pytorch = os.path.exists(weight_path)

logging.info("Loading model...")
if is_pytorch:
    model = train_config.model.load_pytorch(train_config, weight_path)
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
```

`/dataset/personal/bhlee/RL_VLA/0827_opsd/libero/openpi/src/openpi/models/model.py:243-246`

```python
def load_pytorch(self, train_config, weight_path: str):
    logger.info(f"train_config: {train_config}")
    model = pi0_pytorch.PI0Pytorch(config=train_config.model)
    safetensors.torch.load_model(model, weight_path)
    return model
```

즉 **디렉터리에 `model.safetensors` 가 있다는 사실만으로** pytorch 경로가 선택된다. 우리가 넘기는 것은
`--policy.dir` 하나이고, 나머지는 업스트림 로직이다.

### 2.5 모델 설정

`/dataset/personal/bhlee/RL_VLA/0827_opsd/libero/openpi/src/openpi/training/config.py:911-913`

```python
name="pi05_droid_jointpos",
model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
```

`pi05_droid` (관절 **속도** 판) 가 아니라 `pi05_droid_jointpos` (관절 **위치 델타** 판) 여야 한다.
config.py 주석에 2026-08-11 사고 기록이 있다 — 속도판을 쓰면 RoboLab `Pi0DroidJointposClient` 가 그 값을
위치 목표로 적용해 7셀 × 24 에피소드가 전부 0% 로 나온다. 체크포인트의 `config.json` 도 같은 값을 선언한다.

---

## 3. 패치 — 무엇을 고쳤고 무엇을 안 고쳤나

### 3.1 우리가 추가한 파일 (1개)

`/dataset/personal/bhlee/RL_VLA/0827_opsd/opsd_260831_eval_setup/train/openpi/serve_policy_seeded.py` (68줄, 08-31 작성)

```python
:28   _SEED = int(os.environ.get("OPSD_SERVE_SEED", "0") or 0)
:29-39  # --seed 는 tyro 가 모르는 인자라 여기서 직접 떼어낸다
:42-54  def _seed_everything(seed):   # random / numpy / torch(+cuda) 전역 RNG 고정
:57   def main() -> None:
:58       _seed_everything(_SEED)
:60-62    sys.path.insert(0, ".../libero/openpi/scripts")
:63       import serve_policy as _sp          # 공유 트리의 원본을 그대로 쓴다
:64       _sp.main(tyro.cli(_sp.Args))
```

**왜 필요한가** — pi0.5 는 flow matching 초기 노이즈를 전역 RNG 에서 뽑는다
(`src/openpi/models_pytorch/pi0_pytorch.py:174` 의 `torch.normal`, generator 인자 없음).
원본 `serve_policy.py` 는 이 스트림을 고정하지 않아 **서버를 새로 띄울 때마다 다른 행동**이 나온다.
이 래퍼는 기동 직전에만 시드를 박고 나머지는 원본에 위임한다 — 공유 코드를 건드리지 않는다.

**한계** (파일 주석에 명시): 전역 RNG 는 요청 순서에 따라 전진하므로, 한 서버에 여러 셀이 붙으면
인터리브에 따라 각 셀이 받는 노이즈가 달라진다. 완전 재현에는 `PARALLEL=1` 이나 셀별 서버 분리가 필요하다.

### 3.2 공유 openpi 트리의 로컬 수정 — **우리 경로에 닿지 않는다**

`/dataset/personal/bhlee/RL_VLA/0827_opsd/libero/openpi` (HEAD `0b33306`, 2026-08-25)

| 파일 | 수정 | 내용 | pytorch 평가 영향 |
|---|---|---|---|
| `src/openpi/models/pi0.py` | +127 | `sample_actions_mid` (mid-teacher s*) | **없음** — JAX 전용 |
| `src/openpi/models/gemma.py` | +55 | `__call__(..., mix_w=None)` | **없음** — JAX 전용 |
| `src/openpi/policies/policy.py` | +57 | mid-teacher 분기 · target_xyz 헤드 | **없음** — 가드 뒤 |
| `src/openpi/policies/policy_config.py` | — | | 수정 없음 |
| `src/openpi/models/model.py` | — | | 수정 없음 |
| `scripts/serve_policy.py` | — | | 수정 없음 |
| `src/openpi/models_pytorch/` (전체) | — | | 수정 없음 |

`policy.py:84` 의 진입 조건이 가드다:

```python
if (not self._is_pytorch_model) and getattr(self, "_sample_actions_mid", None) is not None \
        and isinstance(obs.get("prompt"), str) and MID_SEP in obs["prompt"]:
    return self._infer_mid(obs)
```

`policy.py` 안에서 `_is_pytorch_model` 가 쓰이는 자리는 `:58,61,84,90,102,112,132,143` 이고,
추가된 로직은 전부 `not self._is_pytorch_model` 뒤에 있다. safetensors 체크포인트는 `policy_config.py:50`
에서 `is_pytorch=True` 로 판정되므로 **항상 업스트림 경로**로 간다.

→ **s80 평가에 쓰인 추론 코드는 pi0.5 원본과 동일하다.**

### 3.3 정리: 체크포인트 교체에 든 코드 변경량

| 항목 | 변경 |
|---|---|
| 심링크 | 16개 (OPSD 8 + PPO 8) |
| `eval_multitask.sh:170` | 이름 규칙 **1줄** |
| `lib_serve.sh:83,86-87` | 기동 분기 **3줄** |
| `serve_policy_seeded.py` | 신규 파일 1개 (공유 트리 비침습) |
| openpi 로딩·추론 코드 | **0줄** |

---

## 4. 체크포인트 실체

### 4.1 심링크 (복사 아님)

`/dataset/personal/bhlee/RL_VLA/0827_opsd/opsd_260831_eval_setup/output/rl/ckpt/` 아래:

```
hard5_opsdb_s{10..80}_pytorch -> /dataset/personal/khy5630/robot/opsd/output/rl/ckpt/hard5_opsd_base_s{10..80}_pytorch   (09-09 18:52 생성)
hard5_ppo80_s{10..80}_pytorch -> /dataset/personal/khy5630/robot/opsd/output/rl/ckpt/hard5_base_ppo_s{10..80}_pytorch    (09-10 19:36 생성)
```

복사가 아니라 심링크이므로 7.4 GB × 16 을 중복 저장하지 않고 khy5630 원본을 직접 읽는다.

### 4.2 config.json

`hard5_opsdb_s80_pytorch/config.json`:

```json
{
  "action_dim": 32,
  "action_horizon": 15,
  "exported_from": "/dataset/personal/khy5630/robot/opsd/output/rl/logs/20260909-135145-robolab_spec4_ppo_pi05/robolab_spec4_ppo_opsd_base/checkpoints/global_step_80",
  "openpi_config": "pi05_droid_jointpos"
}
```

### 4.3 출처 — 학습 런 디렉터리

`exported_from` 을 16개 전부 확인한 결과:

| 팔 | global_step | 학습 런 디렉터리 |
|---|---|---|
| OPSD (`robolab_spec4_ppo_opsd_base`) | 10, 20 | `20260908-031742-robolab_spec4_ppo_pi05` |
| | 30, 40 | `20260908-140641-…` |
| | 50, 60 | `20260908-212126-…` |
| | 70 | `20260909-053602-…` |
| | 80 | `20260909-135145-…` |
| PPO (`robolab_spec4_ppo_base_ppo`) | 10 ~ 80 | `20260909-153129-…` (단일 런) |

⚠ **OPSD 8개는 5개의 서로 다른 런 디렉터리에서 나왔다** — khy5630 쪽 학습이 재시작을 거치며 이어진 결과로
보인다(`global_step` 번호는 10~80 으로 일관). PPO 8개는 단일 런에서 나왔다. 두 팔의 회차 간 연속성 전제가
다르므로, epoch 곡선을 해석할 때 참고한다. (재시작 지점에서 옵티마이저 상태가 어떻게 이어졌는지는
khy5630 쪽 기록을 봐야 하며 이 트리에서는 확인할 수 없다.)

### 4.4 base 와의 차이 — dtype 뿐

| | base `pi05_droid_jointpos_pytorch` | OPSD `hard5_opsdb_s80_pytorch` |
|---|---|---|
| 실체 | `/dataset/personal/bhlee/RL_VLA/0827_opsd/libero/openpi/checkpoints/pi05_droid_jointpos_pytorch` | khy5630 트리 (심링크) |
| 크기 | 7,233,650,408 B (7.23 GB) | 7,473,091,464 B (7.47 GB) |
| 텐서 수 | 812 | 812 |
| 키 차이 | — | **0개** (양쪽 동일) |
| shape 차이 | — | **0개** |
| dtype | BF16 812 | F32 122 + BF16 690 |
| mtime | 08-11 19:04 | 09-09 14:33 |

키·shape 가 완전히 같고 critic 등 추가 구조가 없다. 크기 차이 0.24 GB 는 **122개 텐서가 F32 로 저장된**
데서만 나온다. 즉 같은 아키텍처에 학습된 가중치만 바뀐 것이다.
(로드 직후 `policy_config.py:55` 의 `to_bfloat16_for_selected_params("bfloat16")` 가 선택 파라미터를 bf16 으로 돌린다.)

---

## 5. 어떤 체크포인트가 실렸는지 확인하는 법

| 증거 | 위치 | 값 |
|---|---|---|
| 집계 표 | `output/eval_seeds/opsdb_s80_vague/seed1/results.tsv` 1열 | `hd5opsdb_s80` |
| 셀 로그 파일명 | 같은 폴더 | `hd5opsdb_s80_<Task>.log` |
| 서버 로그 | `/tmp/opsd_servers_<RUN_TAG>/serve_<label>_gpu<N>.log` | `[serve-seeded] RNG 고정 seed=<시드>` |
| 타이밍 검증 | 셀 로그 | `[OPSD] VERIFY env.cfg  decimation=4 dt=0.016667 (60Hz) render_interval=60` |
| 감사 | `cd train/openpi && ./.venv/bin/python check_sim.py --mode run` | 실패 0 |

⚠ **원자료 jsonl 의 `policy` 필드는 체크포인트를 구분하지 못한다.** `seed*/episodes/*.jsonl` 에는
`policy = "pi05"` 로만 적히는데 이는 RoboLab 이 붙이는 **아키텍처 이름**이다. 어떤 체크포인트였는지는
파일명 · 디렉터리 · `results.tsv` 로 판별해야 한다.

---

## 6. 리스크

1. **남의 트리 의존** — 심링크라서 khy5630 이 원본을 다시 내보내거나 지우면 우리 평가 대상이 조용히 바뀐다.
   현재 s80 원본 mtime 은 09-09 14:33 이고 캠페인(09-09 ~ 09-11) 내내 변하지 않았다.
   고정이 필요하면 실제 복사본을 떠야 한다 — 16개면 약 120 GB.
2. **`POLICY_CFG` 기본값 의존** — `lib_serve.sh:96` 이 `${POLICY_CFG:-pi05_droid_jointpos}` 라, 호출부에서
   실수로 다른 값을 export 하면 조용히 다른 규약(관절 속도)으로 돌 수 있다. 체크포인트 `config.json` 의
   `openpi_config` 와 대조하는 것이 안전하다 (지금은 16개 전부 `pi05_droid_jointpos`).
3. **오버레이 venv 의 torch 버전** — `train/openpi/.venv` 의 torch 가 바뀌면 B300 에서 조용히 느려지거나
   죽을 수 있다. 현재 2.7.0+cu128.
