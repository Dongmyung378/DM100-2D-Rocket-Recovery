# 2D Powered Landing Guidance

[한국어 README](README.ko.md)

A reproducible simulation and control project for the terminal descent of a reusable rocket in a vertical plane. It combines custom variable-mass dynamics, a Gymnasium environment, classical landing controllers, optimal-control trajectories, learned policies, and paired robustness evaluation.

This is a **landing** project, not a launch-to-landing or 3D flight simulator. Simulated success is not evidence of hardware safety.

## System at a glance

| Component | Implementation |
|---|---|
| State | `x, z, vx, vz, theta, omega, mass` |
| Control | Throttle and gimbal angle |
| Physics | Planar rigid body, variable mass, 0.02 s RK4 integration |
| Control interval | 0.1 s |
| Baselines | Suicide-burn, vertical PID, integrated landing PID |
| Learned controllers | Behavior cloning, DAgger, PID-residual SAC |
| Runtime | Python 3.12.7 |

The environment supports wind and gusts, thrust and inertia mismatch, sensor noise, observation and action delays, and engine lag. The same landing limits are used to classify safe touchdowns and failures.

## Current evidence

The integrated PID remains the recommended controller. The offline optimal-control dataset contains 1,007 accepted training trajectories, 100 IID validation trajectories, and 191 accepted hard out-of-distribution test trajectories. On that **Teacher-accepted** hard test, the PID landed 191/191, behavior cloning 0/191, and DAgger 3/191. This comparison does not imply that PID succeeds on every possible hard initial condition; nine planned test trajectories were rejected by Teacher replay. See the [dataset card](docs/dataset-card.md) and [model card](docs/model-card.md) for the cohort definitions and limitations.

A separate disturbance protocol uses root seed `378` and distinct training, development, and sealed final-holdout streams. On 12 development cases per condition, the PID safe-landing counts are:

| Nominal | Initial-state OOD | Wind | Model mismatch | Sensor/delay | Combined disturbances |
|---:|---:|---:|---:|---:|---:|
| 12/12 | 12/12 | 6/12 | 8/12 | 6/12 | 0/12 |

Residual SAC has occasionally landed one combined-disturbance case, but tested variants also lost successes in other groups. None met the protocol's no-regression requirement, so no learned controller replaces the PID baseline. These are small development samples, not a final robustness estimate; the final holdout has not been opened. The [technical specification](docs/spec.md) describes the exact scenarios, evaluation protocol, and failure analysis.

## Run a landing

From the repository root, use Python 3.12.7:

```powershell
python -m pip install -e ".[dev,rl,video]"
python scripts/run_episode.py --seed 378 --initial-state 10 100 0 -20 0 0 1000 --controller integrated-pid --output artifacts/landing.json --plot artifacts/landing.png --animation artifacts/landing.mp4
```

The initial state is `x z vx vz theta omega mass` in SI units, with angles in radians. The command prints the touchdown outcome and final state, then saves a replayable episode, diagnostic plot, and MP4 animation. Output files are not overwritten. To animate a saved episode without rerunning the simulation:

```powershell
python scripts/run_episode.py --replay artifacts/landing.json --animation artifacts/landing-replay.mp4
```

For a smaller installation without SAC or MP4 support, install `.[dev]` and omit `--animation`.

## Reproduce checks

```powershell
python -m pytest -p no:cacheprovider
python -m ruff check --no-cache .
python -m ruff format --check --no-cache .
python scripts/prepare_rl_protocol.py --output-dir artifacts/rl-protocol
```

The final command evaluates the frozen PID on development cases only. It writes a CSV and protocol report under the Git-ignored `artifacts/` directory and refuses to overwrite existing outputs. The exact condition ranges, root seed, and selection rules are in [`configs/rl-evaluation-v1.yaml`](configs/rl-evaluation-v1.yaml). Training and evaluation entry points are in [`scripts/`](scripts/); the implementation is in [`src/powered_landing_guidance/`](src/powered_landing_guidance/).

## Scope and limitations

- The optimal-control Teacher solves a nominal model; it does not certify performance under combined disturbances or actuator delays.
- The current residual-action filter is a heuristic, not a formal safety guarantee. A filtered policy may still turn a PID success into a failure.
- The simulator omits launch, 3D dynamics, structural impact damage, and flight-qualified hardware behavior.
- Generated outputs and local-only notes are excluded from Git.

Further details: [technical specification](docs/spec.md) · [dataset card](docs/dataset-card.md) · [model card](docs/model-card.md).
