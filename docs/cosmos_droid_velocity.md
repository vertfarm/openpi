# Cosmos3 DROID policy on the KETI velocity client

Serves `nvidia/Cosmos3-Nano-Policy-DROID` while preserving the deployed KETI contract:

- 3090 workstation: `RobotEnv(action_space="joint_velocity")` — **unchanged**
- G15 controller — **unchanged**
- Response: N actions × 8 values (7 normalized joint velocities, 1 gripper position)

The output path is:

`CosmosActionsToJointPositions -> JointPositionToDroidVelocity -> DroidOutputs`

The last two stages are `droid_policy`'s own, unmodified, so the command the G15 receives is
produced by the same code the deployed pi05 server uses. For joint targets $q_k^*$ and the
current measured joint state $q_{obs}$:

$$ v_0 = \frac{q_0^* - q_{obs}}{0.2}, \qquad v_k = \frac{q_k^* - q_{k-1}^*}{0.2}. $$

Only the seven joint commands are clipped, to ±0.5; the gripper is untouched.

## Three differences from the pi05 path

### 1. No `AbsoluteActions` — and adding it is silent

pi05 predicts joint-position **deltas**, so `AbsoluteActions` adds the state back. **Cosmos3
predicts absolute joint angles directly.** Measured against a live server: feeding the current
pose `q` and reading the first row `a0` gives `|a0 - q| = 0.023 rad` against `|a0| = 0.721 rad`
— `a0` sits on `q`, not on zero. Confirmed in simulation, where the first executed action
matched RoboLab's Franka reset pose to within 1e-3 rad per joint.

Keeping that stage doubles every joint target and raises nothing.
`cosmos_droid_policy_test.py::test_adding_absolute_actions_would_double_the_targets` pins it,
including the case where the ±0.5 clip caps the command and thereby *hides* the magnitude.

### 2. Action horizon 32, not 15

Set by the checkpoint. Truncating to pi05's 15 is not free — paired simulation A/B over 24
environments:

| chunk | per-run successes | total |
|---|---|---|
| 32 | 17, 17, 12 | 46/72 = 63.9% |
| 15 | 9, 8 | 17/48 = 35.4% |

The per-run ranges do not overlap. Cosmos is trained to emit a complete 32-step motion;
restarting it early leaves the arm repeating the approach.

### 3. The request carries both exterior cameras

pi05 sends one exterior view plus the wrist, each 224×224. Cosmos needs a single 540×640 frame:
the wrist at 360×640 on top, the two exterior views at 180×320 side by side beneath it.

**Sending the pi05 request shape to Cosmos scored 0/24 in simulation** — one camera missing and
the rest already downscaled. The first executed action differed from the reference by 0.159 rad,
ten times the run-to-run noise floor of 0.016 rad.

The robot laptop already has all three views: `examples/droid/main.py::_extract_observation`
extracts `left_image`, `right_image` and `wrist_image` and only the request drops one. So this
is a change to the request, not to the hardware.

## What the 3090 must change

| | from | to | if not |
|---|---|---|---|
| request | 1 exterior + wrist, 224×224 | both exteriors + wrist | policy is blind (0/24) |
| `action_horizon` | 15 | 32 | client assertion fails immediately |
| `open_loop_horizon` | 8 | 32 | see below |

Nothing else moves: `RobotEnv(action_space="joint_velocity")`, the ×0.2 interpretation and the
G15 path are all untouched. Reference client: `examples/droid/main_cosmos.py`.

### Why `open_loop_horizon` cannot stay at 8

Measured end-to-end through this proxy: **931 ms** per chunk. At 15 Hz an 8-step chunk lasts
533 ms, so the server cannot keep up regardless of success rate. `check_cosmos_droid_velocity.py`
reports `min_open_loop_horizon_for_realtime` from the observed latency; it printed 14 on the
first run. 32 leaves a 2.13 s budget against a 0.93 s round trip.

## Clipping

The ±0.5 limit is effectively never reached by Cosmos. Per-step joint deltas from five simulated
runs (24 environments × 300 steps each), converted to velocity units:

| run | median | p99 | max | saturation |
|---|---|---|---|---|
| baseline | 0.037 | 0.21 | 0.70 | 0.041% |
| baseline (repeat) | 0.037 | 0.20 | 0.60 | 0.013% |
| pi-format output | 0.037 | 0.21 | 0.76 | 0.020% |
| **degraded input (0/24)** | 0.044 | 0.33 | **2.56** | **0.446%** |

The last row is useful in itself: a policy that has lost the scene saturates ten times more
often, so `joint_limit_fraction` doubles as a health signal on the real robot.

## Running it

```bash
# 1. the Cosmos policy server (its own process, unmodified upstream code)
python -m cosmos_framework.scripts.action_policy_server_robolab \
  --checkpoint-path nvidia/Cosmos3-Nano-Policy-DROID --port 8010 \
  --num-steps 8 --guidance 3.0 --shift 3.0 --stochastic-sampler --sampler-deterministic

# 2. the velocity proxy (no model, no GPU)
python scripts/serve_cosmos_droid_velocity.py --upstream-port 8010 --port 8001

# 3. the gate, with no RobotEnv
python scripts/check_cosmos_droid_velocity.py --port 8001
```

## Open questions before motion

1. **How does `RobotEnv` apply the command?** `target = q_measured + v·0.2` or
   `target = q_prev_target + v·0.2`. The two diverge once the arm lags, and the conversion above
   assumes the former. This has not been verified against the `droid` package source.
2. **Does Cosmos3 (~33 GiB) fit the A6000 alongside the launcher's GPU-idle checks?** The pi05
   launcher refuses to start if the card already holds more than 2 GiB.
3. **The safety procedure starts at `open_loop_horizon=1`, then 8.** Neither is servable at
   931 ms. A Cosmos-specific first-motion procedure has to be agreed with KETI.
