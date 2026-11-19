# DM100 Validation Results Report

[한국어](results.ko.md) | [Project](../README.md) | [Model specification](spec.md) | [Validation protocol](validation.md) | [Roadmap (KO)](roadmap.ko.md)

This report records the **complete 487-case validation campaign** of the unified
DM100 2D SpaceX-style RTLS simulator and documents every guidance, control, and
physics improvement implemented to guarantee stable recovery across worst-case
boundary conditions.

- Raw validation record: [validation-487.json](results/validation-487.json)
- 50 Hz 3D replay exports (ENU + unit quaternion `wxyz`):
  [calm_replay_50hz.json](results/replays/calm_replay_50hz.json),
  [example_weather_replay_50hz.json](results/replays/example_weather_replay_50hz.json)
- **Scope note**: All metrics are computed within a 2D concept simulator under
  declared synthetic disturbance and sensor envelopes; they do not represent
  flight-hardware certification.

## 1. Full 487-Case Campaign Summary

Executed in a single parallel sweep (`--suite full-487 --workers 14`, wall time
`1,718.7 s`) covering **32 adversarial extreme edge cases**, all **169 cases**
from `configs/validation-study-v1.json`, and all **286 cases** from
`configs/apr-regression-v1.json`.

| Evaluation split | Passed / total | Main remaining (kg) | Max `\|x\|` error (m) | Max `q` (kPa) | Max angular accel (deg/s²) | Unpowered glide (km) | 5 s standing |
|---|---:|---:|---:|---:|---:|---:|---:|
| Extreme edge cases (`edge-cases`) | **32 / 32 (100%)** | 9,343 – 10,287 | 1.766 | 29.61 | 4.32 | 9.6 – 14.3 | 32 / 32 |
| Development (`development`) | **16 / 16 (100%)** | 9,505 – 10,020 | 0.313 | 28.37 | 3.61 | 11.4 – 13.6 | 16 / 16 |
| Coupled holdout (`holdout`) | **96 / 96 (100%)** | 9,294 – 10,172 | 0.610 | 29.44 | 4.21 | 11.2 – 13.6 | 96 / 96 |
| Extended-sensor holdout (`sensor-holdout`) | **57 / 57 (100%)** | 9,406 – 10,113 | 0.436 | 28.99 | 4.14 | 11.2 – 13.6 | 57 / 57 |
| Regression development (`regression-development`) | **205 / 205 (100%)** | 9,385 – 10,301 | 1.975 | 28.96 | 4.36 | 10.6 – 14.5 | 205 / 205 |
| Regression previous holdout (`regression-previous-holdout`) | **72 / 72 (100%)** | 9,434 – 9,997 | 0.357 | 28.69 | 4.24 | 10.5 – 13.0 | 72 / 72 |
| Regression supplemental (`regression-supplemental`) | **9 / 9 (100%)** | 9,687 – 10,011 | 0.352 | 27.02 | 3.50 | 11.4 – 12.6 | 9 / 9 |
| **Total** | **487 / 487 (100.0%)** | **9,294 – 10,301** | **1.975** | **29.61** | **4.36** | **9.6 – 14.5** | **487 / 487** |

## 2. Aggregate Distribution and Safety Margins (All 487 Cases)

| Metric | Min | Median | Max | Limit | Status |
|---|---:|---:|---:|---:|---|
| Touchdown position error `\|x\|` (m) | 0.000 | **0.024** | 1.975 | ≤ 2.000 | Pass (487/487) |
| Vertical foot speed (m/s) | 0.159 | 0.787 | 1.430 | ≤ 1.500 | Pass (487/487) |
| Horizontal foot speed (m/s) | 0.000 | 0.027 | 0.481 | ≤ 0.500 | Pass (487/487) |
| Touchdown tilt (deg) | 0.000 | 0.000 | **0.078** | ≤ 3.000 | Pass (487/487) |
| Touchdown angular rate (deg/s) | 0.000 | 0.005 | **0.147** | ≤ 0.500 | Pass (487/487) |
| Peak dynamic pressure `q` (kPa) | 24.21 | 25.64 | 29.61 | ≤ 30.00 | Pass (487/487) |
| Peak recovery load (g) | 4.065 | 4.424 | 4.859 | ≤ 5.000 | Pass (487/487) |
| Peak angular rate (deg/s) | 13.50 | 13.50 | 13.59 | ≤ 15.00 | Pass (487/487) |
| Peak angular acceleration (deg/s²) | 2.694 | 2.875 | **4.355** | ≤ 5.000 | Pass (487/487) |
| Main propellant remaining (kg) | 9,293.9 | **9,772.0** | 10,301.3 | > 0 | Pass (487/487) |
| RCS propellant remaining (kg) | 167.5 | **252.2** | 263.6 | ≥ 0 | Pass (487/487) |
| Engine restarts used | 3 | 3 | 3 | ≤ 3 | Pass (487/487) |
| Unpowered grid-fin glide distance (km) | 9.61 | **12.39** | 14.46 | — | 0% commanded throttle |
| Unpowered glide duration (s) | 34.26 | 46.94 | 61.54 | — | MECO-2 tail-off 5.7–15.9 kg |
| Post-contact max leg stroke (m) | 0.230 | 0.242 | 0.278 | ≤ 0.900 | Pass (487/487) |
| Post-contact max load (g) | 1.109 | 1.124 | 1.982 | ≤ 4.000 | Pass (487/487) |
| 5 s engine-off upright standing | — | — | — | 487 / 487 | **100.0% Pass** |

## 3. Extreme Adversarial Edge-Case Highlights (32 / 32)

In addition to `edge-01`–`edge-24` (maximum ±8/±20 m/s winds, wind shear, ±3 m/s
vertical winds, 5 m/s ascent/glide/entry/landing gusts, max-q and heavy
low-thrust corners, polar/equatorial Coriolis, and maximum sensor/delay corners),
`edge-25`–`edge-32` stack **100% simultaneous operating limits** across every
atmosphere, vehicle, delay, noise, bias, and packet-dropout channel.

| Representative edge case | Stress profile | `\|x\|` (m) | `v_vert` (m/s) | `v_horiz` (m/s) | Max accel (deg/s²) | Main rem (kg) | Status |
|---|---|---:|---:|---:|---:|---:|---|
| `edge-11-worst-maxq-corner` | Max density, min drag, max mass, tailwind Max-Q corner | 0.002 | 0.192 | 0.004 | 3.00 | 9,532 | Pass |
| `edge-23-combined-nightmare-low` | Combined low-drag, headwind, gust, delay, sensor corner | 0.015 | 1.032 | 0.071 | 3.42 | 9,689 | Pass |
| `edge-24-combined-nightmare-high` | Combined high-drag, tailwind, gust, delay, sensor corner | 0.968 | 0.673 | 0.211 | 3.62 | 9,923 | Pass |
| `edge-25-max-everything-low` | 100% simultaneous low corner (mass↑, thrust↓, drag↓, -8/-20 wind, max delays/IMU/dropout) | 1.120 | 0.591 | 0.190 | 3.58 | 9,343 | Pass |
| `edge-26-max-everything-high` | 100% simultaneous high corner (density↑, drag↑, +8/+20 wind, 5 m/s gust, max delays/IMU/dropout) | 1.340 | 0.188 | 0.014 | 4.19 | 10,287 | Pass |
| `edge-27-max-shear-heavy-nightmare` | Extreme wind shear + max mass/inertia/delays/dropouts | 0.176 | 0.950 | 0.099 | 4.32 | 9,418 | Pass |
| `edge-28-reverse-shear-light-nightmare` | Reverse shear (+8/-20 m/s) + min mass/max thrust/max noise/delays | 1.766 | 0.727 | 0.329 | 3.85 | 10,089 | Pass |
| `edge-30-equator-coriolis-heavy` | Equatorial Coriolis + max mass/min thrust/delays/noise | 1.548 | 0.956 | 0.175 | 3.58 | 9,498 | Pass |
| `edge-31-south-pole-cold-dense` | South-pole dense atmosphere + high drag + 5 m/s gust + max noise/dropout | 0.016 | 0.180 | 0.123 | 3.80 | 10,165 | Pass |
| `edge-32-ground-gust-touchdown-shock` | 5 m/s touchdown ground gust shock + 100% sensors/delays/dropouts | 1.732 | 0.678 | 0.107 | 3.94 | 9,857 | Pass |

## 4. Reference Flight Profiles

Representative flights generated with `scripts/run_flight.py`:

| Scenario | Status | Main remaining (kg) | RCS remaining (kg) | `\|x\|` error (m) | `v_vert` / `v_horiz` (m/s) | Unpowered glide | 5 s standing |
|---|---|---:|---:|---:|---:|---:|---|
| `calm` | Pass | **9,771.7** | 257.7 | 0.003 | 0.786 / 0.007 | 46.94 s / 12.41 km | Pass (`0.000°`) |
| `example-weather` | Pass | **9,951.8** | 243.6 | 0.123 | 0.995 / 0.067 | 51.02 s / 12.39 km | Pass (`0.018°`) |

## 5. Key GNC and Physics Improvements Recorded

1. **Unified SpaceX 7-Phase RTLS Engine (`Plant ↔ Forces ↔ GNC`)**
   - Consolidated the entire simulator onto the clean `rtls/` architecture
     (`ascent → flip → boostback → coast → entry_burn → aero_glide → landing_burn → touchdown_settle`)
     and removed all legacy dual-mode paths.
   - Models tail-first blunt-base descent drag ($C_{D,\text{descent}} = 1.65$),
     grid-fin effective angle of attack and hinge torque limits, a 3-engine
     cluster (44% per-engine throttle floor, 3-restart budget) with non-hovering
     3→1 Hoverslam, and 5 s post-contact spring-damper/friction landing legs.
2. **Causal Online Drag/Density Scale Estimation (`_predict_rtls_iip`)**
   - Estimates effective atmospheric drag/density scaling (`[0.75, 1.35]`) online
     during unpowered `aero_glide` strictly from causal `accel_bias_z`, removing
     hidden +26.5% drag/wind IIP drift before landing ignition.
3. **Analytical Committed-Slew Zero-Tilt Predictor (`_predict_zero_tilt_state`)**
   - Integrates exact lateral displacement and velocity committed while unwinding
     `(θ, ω) → (0, 0)` under finite angular deceleration (`2.15 deg/s²`),
     eliminating Stage B limit-cycle oscillation and excessive banking.
4. **Stage A Transition Lookahead & Stage B Touchdown-Time No-Overshoot Cap**
   - Uses lookahead position `x_stage_a` during 3-engine Stage A braking to kill
     strong crosswind/glide-exit velocities early while thrust authority is high.
   - Adds a remaining-time no-overshoot cap (`v_no_overshoot`) in 1-engine
     Stage B to prevent late re-acceleration and pad overshoot right before
     `vertical_body` alignment.
5. **Noise-Immune Vertical-Body Latch & Interval-Matched TVC/RCS Allocation**
   - Uses a deterministic one-way latch (`h ≤ 21.5 m`) so ±2 m nav noise cannot
     chatter or prematurely trigger `vertical_body`.
   - Slew-limits `vertical_body` gimbal commands (`≤ 0.85°/step`) and matches
     RCS counter-torque to the actual interval TVC+fin torque trajectory in
     `PriorityControlAllocator`, holding peak angular acceleration `≤ 4.36 deg/s²`
     across all 487 cases even under 1-step action delay and ground gust shocks.
6. **Continuous Touchdown Taper & Asymmetric Hoverslam Vertical Braking**
   - Replaces discrete touchdown thresholds with continuous smoothstep tapering
     (`always_td_taper`), combines asymmetric vertical braking (`k_v = 2.7`) with
     an anti-bounce MECO-3 cutoff guard under `TWR_min > 1`, achieving soft
     `0.159–1.430 m/s` touchdowns and 487/487 5 s upright standing stability.

## 6. Reproducing the Results

```bash
# Fast 32-case extreme edge-case sweep (~1.5 min)
python scripts/evaluate_validation.py --suite edge-cases --workers 8

# Full 487-case validation campaign
python scripts/evaluate_validation.py --suite full-487 --workers 14 --out-json artifacts/validation-487.json

# Single representative flight with 50 Hz 3D replay JSON export
python scripts/run_flight.py --scenario example-weather --export-replay artifacts/example_weather_replay_50hz.json
```
