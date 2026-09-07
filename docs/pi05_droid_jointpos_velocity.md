# π0.5 DROID joint-position checkpoint on the KETI velocity client

This path serves the official JAX checkpoint
`gs://openpi-assets/checkpoints/pi05_droid_jointpos` while preserving the deployed KETI contract:

- A6000 server: physical GPU 3 by default, port 8001
- 3090 workstation: `RobotEnv(action_space="joint_velocity")`
- G15 controller: unchanged
- Response: 15 actions × 8 values (7 normalized joint velocities and 1 gripper position)

The output path is:

`Unnormalize -> AbsoluteActions -> JointPositionToDroidVelocity -> DroidOutputs`

For joint targets $q_k^*$ and the current measured joint state $q_{obs}$:

$$
v_0 = \frac{q_0^* - q_{obs}}{0.2}, \qquad
v_k = \frac{q_k^* - q_{k-1}^*}{0.2}.
$$

The server clips only the seven joint commands to ±0.5 for the first live diagnostic and logs
the pre-clip range and saturation fraction. It does not modify the gripper command.

## SNU server

Run as the `snu` account from this repository:

```bash
scripts/start_pi05_droid_jointpos_velocity.sh 3
scripts/status_pi05_droid_jointpos_velocity.sh
```

The first start caches the official checkpoint under `/data/keti/snu/home`, records SHA-256
checksums, loads the model, and performs two offline warmup inferences before opening port 8001.

Check the running server without constructing `RobotEnv`:

```bash
.venv/bin/python scripts/check_pi05_droid_jointpos_velocity.py --host 127.0.0.1 --port 8001
```

Stop only this server process:

```bash
scripts/stop_pi05_droid_jointpos_velocity.sh
```

## KETI real-robot gate

Before motion, run the 3090 no-send probe with port 8001 and horizon 15. It must report a finite
`(15, 8)` chunk and the three camera views must be visually valid. This probe constructs
`RobotEnv(do_reset=False)` and never calls `reset`, `step`, or `update_robot`.

The first live run remains a KETI-controlled operation: E-stop and reset path confirmed,
`open_loop_horizon=1`, no more than 30 steps. Expand to horizon 8 and 300 steps only after the
diagnostic motion and tracking logs are normal. The server-side conversion cannot correct
mid-chunk tracking error; repeated clipping or tracking error requires a measured-state client adapter.
