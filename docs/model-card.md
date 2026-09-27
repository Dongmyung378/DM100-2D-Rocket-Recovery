# Model card: planar BC and DAgger landing policies (Day 35)

[한국어 모델 카드](model-card.ko.md)

Status: research-only simulation artifacts. Neither model is suitable for hardware control or safety-critical deployment.

## System and provenance

The simulator has seven states `(x, z, vx, vz, theta, omega, mass)` and two actions `(throttle, gimbal)`. Python 3.12.7 and the project-configured 0.02 s RK4 physics were used. The learned policy issues a new command every 0.1 s. Its inputs and targets use statistics fitted only on the Day 24 train split. The accepted offline dataset contains 1,007 train trajectories (100,700 state-action pairs), 100 IID validation trajectories, and 191 hard OOD test trajectories. Nine other planned hard-test initial states failed Teacher replay and are not part of the 191-case comparison.

The BC checkpoint is the Day 25 `7-64-64-2` ReLU MLP. Day 26 comparisons of tanh and wider ReLU variants did not beat its `0.03098` IID validation normalized MSE. DAgger v4 initializes from BC and uses 125 additional, genuinely re-solved optimal-control labels from student-visited **train-split** states. The three iterations accepted 43/48, 35/48, and 47/48 Teacher queries. Failed queries did not become training labels. The final DAgger evaluation applies command-rate limits of 2 throttle units/s and 60 deg/s gimbal; the BC comparison does not apply that postprocessor. The checkpoint metadata stores normalization, architecture, training settings, data provenance, and parent digest. Rate-limit selection is recorded separately in the Day 33-34 report.

## Evaluation

| Cohort and policy | Safe landings | Main interpretation |
|---|---:|---|
| IID validation, BC | 14/100 | Closed-loop success is far below its one-step action fit. |
| IID validation, DAgger with rate limit | 11/100 | DAgger did not preserve nominal performance. |
| Hard OOD test, BC | 0/191 | No safe landings. |
| Hard OOD test, DAgger with rate limit | 3/191 (1.6%) | A small absolute gain, not robust control. |
| Hard OOD test, frozen PID | 191/191 | Strong classical-control reference on this accepted subset. |

BC and DAgger checkpoints were copied and SHA-256 checked before the hard-test initial states were loaded. The hard test was not used to choose architecture, DAgger iteration, rate limit, or residual-policy mix. The three improved DAgger cases were indices 71, 98, and 125. Separate failure and improved-case MP4/JSON replays are in ignored `artifacts/day35-final/`.

On hard OOD, DAgger failed by fuel depletion 110 times, position error 41 times, attitude error 32 times, and excessive touchdown speed 5 times. The BC counts were 124, 38, 26, and 3 respectively. Among DAgger fuel-depletion cases, median episode duration was 48.96 s and median absolute vertical speed at depletion was 0.255 m/s, consistent with excessive hovering. Inference-only median latencies were 9.4 microseconds (BC) and 9.5 microseconds (DAgger) on the development machine. These exclude sensing, simulation, and actuator dynamics; the offline Teacher's accepted solve attempt averaged 1.236 s on an earlier run, so the two timings are not end-to-end comparisons.

## Known limitations and failure analysis

- All 191 hard-test initial states are outside the train-initial-state min/max range in **x, z, vz, and mass**. For example, train initial altitude was 70.01-107.98 m and hard test was 110.87-139.95 m; train initial mass was 962.07-999.98 kg and hard test was 900.28-959.56 kg. This is deliberate OOD stress, not an IID estimate.
- The 125 DAgger labels cover only a small part of the student's reachable state space. DAgger iteration 3 improved the 32-case development subset from BC 4/32 to 5/32, but the full IID validation result fell from 14/100 to 11/100. The Day 32 nominal-nondegradation target is therefore **not met**.
- Teacher feasibility is not perfect: 19/144 student-state queries failed, and 9/200 planned hard-test Teacher trajectories failed replay. The accepted hard-test subset has a selection bias toward solvable cases. These failures cannot alone explain the large BC/PID gap.
- The model is a two-layer feed-forward policy without memory, uncertainty estimate, safety shield, or flight-phase logic. Its action-error objective does not directly penalize landing failure or fuel-depletion hovering.
- Rate limiting reduced high-frequency commands on one 32-case validation subset without improving its 3/32 success count. Added sensor noise or combined 0.8 m/s wind and 5% thrust loss each reduced it to 2/32. A separate 20,000-row train-only sensor-noise augmentation candidate landed 5/32 clean and 2/32 noisy, so noisy robustness did not improve; it was not included in the already-frozen hard-test comparison. Three PID/DAgger action blends all landed 0/32 and were rejected. PPO was optional and was not trained.
- The environment models planar terminal descent and point contact. It does not model launch, 3D rotation, landing legs, structural impact damage, or a hardware-grade actuator. A simulated success threshold of 2 m/s does not establish safe hardware landing.

## Reproduce and next decision

From the repository root, with the verified Day 24 and Day 25 ignored artifacts present:

~~~bash
python -m pip install -e ".[optimization,video,dev]"
python scripts/advance_imitation_learning.py --through 35
python -m pytest -q
~~~

The output reports are under ignored `artifacts/day26-bc-experiments/`, `artifacts/day27-28-closed-loop/`, `artifacts/day29-32-dagger/`, `artifacts/day33-34-stabilization/`, and `artifacts/day35-final/`. Later residual-SAC work is documented separately in the [project overview](../README.md) and [specification](spec.md); it has not passed the six-group robustness gate. The 191-case hard test documented here was not reused to tune that work and must not be recycled as a tuning set.
