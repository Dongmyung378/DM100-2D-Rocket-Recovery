# DM100 Validation Results Report

[한국어](results.ko.md) | [README](../README.md) | [Model Spec](spec.md) | [Validation Protocol](validation.md) | [Roadmap (KO)](roadmap.ko.md)

This report documents the **full 487-case validation campaign** for the DM100 100 t Methalox 700 km orbital launch and first-stage RTLS recovery simulator, certifying **487 / 487 (100.0%) passes** with **every single one of 7,792 constraint checks strictly ≤ 79.3% of its limit** (zero violations of the 80% engineering margin target).

- Raw certification record: [validation-487.json](results/validation-487.json)
- 50 Hz 3D replay exports (ENU + unit quaternion `wxyz`): [calm_replay_50hz.json](results/replays/calm_replay_50hz.json), [example_weather_replay_50hz.json](results/replays/example_weather_replay_50hz.json)
- **Scope notice**: All values come from a 2D conceptual physics simulator over declared synthetic disturbance and sensor envelopes and do not represent hardware flight qualification.

## 1. Full 487-Case Summary by Split

Evaluated in a single parallel batch (`--suite full-487`, wall time `3,388.9 s`) across 32 extreme boundary & 100% simultaneous worst-case edge scenarios, 169 validation study cases, and 286 regression cases.

| Split | Passed / Run | Stage 1 Main Rem (kg) | Max Pos Error (m) | Max q (kPa) | Max Ang Accel (deg/s²) | Unpowered Glide (km) | 700 km Orbit / 5 s Standing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `edge-cases` | **32 / 32 (100%)** | 1,749~3,692 | 0.389 | 33.31 | 3.81 | 25.0~32.3 | 32 / 32 · 32 / 32 |
| `development` | **16 / 16 (100%)** | 2,021~3,102 | 0.202 | 30.61 | 3.18 | 27.7~32.9 | 16 / 16 · 16 / 16 |
| `holdout` | **96 / 96 (100%)** | 1,822~3,146 | 0.901 | 32.24 | 3.15 | 27.3~35.7 | 96 / 96 · 96 / 96 |
| `sensor-holdout` | **57 / 57 (100%)** | 1,817~3,099 | 0.279 | 31.21 | 3.19 | 26.7~35.3 | 57 / 57 · 57 / 57 |
| `regression-development` | **205 / 205 (100%)** | 932~3,895 | 1.500 | 33.08 | 3.32 | 26.6~43.0 | 205 / 205 · 205 / 205 |
| `regression-previous-holdout` | **72 / 72 (100%)** | 2,099~2,884 | 0.349 | 31.42 | 3.07 | 27.9~33.5 | 72 / 72 · 72 / 72 |
| `regression-supplemental` | **9 / 9 (100%)** | 1,882~2,909 | 0.323 | 29.66 | 2.82 | 28.3~34.6 | 9 / 9 · 9 / 9 |
| **Total** | **487 / 487 (100.0%)** | **932~3,895** | **1.500** | **33.31** | **3.81** | **25.0~43.0** | **487 / 487 · 487 / 487** |

## 2. Global Constraint Distribution & ≤ 80% Safety Margin Certification

| Metric | Median (p50) | p95 | Maximum | Limit | Peak % of Limit | Status |
|---|---:|---:|---:|---:|---:|---|
| Touchdown position error `\|x\|` (m) | **0.020** | 0.271 | 1.500 | ≤ 2.000 | **75.0%** | PASS (487/487) |
| Vertical foot speed (m/s) | 0.565 | 0.699 | 1.070 | ≤ 1.500 | **71.3%** | PASS (487/487) |
| Horizontal foot speed (m/s) | 0.018 | 0.118 | 0.360 | ≤ 0.500 | **72.0%** | PASS (487/487) |
| Touchdown tilt (deg) | 0.135 | 1.406 | 2.056 | ≤ 3.000 | **68.5%** | PASS (487/487) |
| Touchdown angular rate (deg/s) | 0.032 | 0.223 | 0.301 | ≤ 0.500 | **60.1%** | PASS (487/487) |
| Peak dynamic pressure `q` (kPa) | 29.40 | 31.98 | 33.31 | ≤ 42.00 | **79.3%** | PASS (487/487) |
| Peak stagnation heat flux (kW/m²) | 14.23 | 16.35 | 21.05 | ≤ 80.00 | **26.3%** | PASS (487/487) |
| Peak ascent `q·α` (kPa·deg) | 76.18 | 86.34 | 102.74 | ≤ 150.0 | **68.5%** | PASS (487/487) |
| Peak ascent high-q `α` (deg) | 2.639 | 2.972 | 3.392 | ≤ 5.000 | **67.8%** | PASS (487/487) |
| Peak ascent axial load (g) | 3.370 | 3.504 | 3.554 | ≤ 4.500 | **79.0%** | PASS (487/487) |
| Peak recovery load (g) | 3.840 | 4.186 | 4.334 | ≤ 5.500 | **78.8%** | PASS (487/487) |
| Peak angular rate (deg/s) | 11.00 | 11.02 | 11.11 | ≤ 15.00 | **74.1%** | PASS (487/487) |
| Peak angular acceleration (deg/s²) | 2.549 | 3.121 | 3.807 | ≤ 5.000 | **76.1%** | PASS (487/487) |
| Peak fin hinge moment (N·m) | 737 | 1,347 | 1,397 | ≤ 7,500 | **18.6%** | PASS (487/487) |
| Post-contact max leg stroke (m) | 0.288 | 0.350 | 0.389 | ≤ 0.900 | **43.3%** | PASS (487/487) |
| Post-contact max load (g) | 1.162 | 1.477 | 1.656 | ≤ 4.000 | **41.4%** | PASS (487/487) |

Global orbital & reserve summary:
- **Stage 2 700 km Circular Orbit (1,800 kg)**: `487/487` inserted (`700.01~703.25 km`, max altitude error `3.25 km` = `21.7%` of `±15 km`), Stage 2 propellant reserve `182.2~918.7 kg` (median `595.6 kg`).
- **Stage 1 Main / RCS Reserves**: `932.0~3,894.6 kg` (median **`2,612.8 kg`**) / `168.5~279.3 kg` (median **`271.9 kg`**, `3/3` restarts).
- **Unpowered Grid-Fin Glide**: `25.0~43.0 km` (median **`29.42 km`**, `72.6~128.8 s`, `0%` throttle command).
- **5 s Engine-Off Upright Standing**: **487 / 487 (100.0%)**.

## 3. Representative Extreme Edge Cases (32/32 Passed)

| Edge Case | Conditions | Pos (m) | v_vert / v_horiz (m/s) | Max q (kPa) | Max Accel (deg/s²) | Orbit (km) | S1 / S2 Rem (kg) |
|---|---|---:|---:|---:|---:|---:|---:|
| `edge-11-worst-maxq-corner` | High density, low drag, max thrust Max-q corner | 0.047 | 0.467 / 0.054 | 32.55 | 2.81 | 702.93 | 3,090 / 602.7 |
| `edge-12-heavy-low-thrust` | Heavy mass, min thrust, dense atmosphere | 0.005 | 0.537 / 0.006 | 33.31 | 2.55 | 701.59 | 1,749 / 700.3 |
| `edge-25-max-everything-low` | 100% simultaneous worst low-side corner | 0.091 | 0.702 / 0.161 | 33.00 | 3.28 | 701.14 | 2,067 / 605.9 |
| `edge-26-max-everything-high` | 100% simultaneous worst high-side corner | 0.389 | 0.187 / 0.230 | 31.91 | 3.71 | 701.02 | 3,338 / 568.6 |
| `edge-27-max-shear-heavy-nightmare` | Extreme wind shear + heavy + max delays/dropouts | 0.138 | 0.563 / 0.061 | 32.67 | 3.59 | 701.26 | 2,022 / 603.3 |
| `edge-28-reverse-shear-light-nightmare` | Reverse shear (+8/-20 m/s) + light/high thrust | 0.136 | 0.440 / 0.340 | 32.12 | 3.81 | 702.84 | 3,318 / 585.0 |
| `edge-29-maxq-nightmare` | 100% simultaneous Max-q high-energy reentry | 0.041 | 0.292 / 0.098 | 32.56 | 3.59 | 702.05 | 3,692 / 373.9 |
| `edge-30-equator-coriolis-heavy` | Equator max Coriolis + heavy / low thrust / delays | 0.228 | 0.637 / 0.023 | 33.15 | 3.16 | 702.66 | 1,933 / 750.9 |
| `edge-31-south-pole-cold-dense` | South pole (0 rotation boost) + dense + 5 m/s gust | 0.287 | 0.204 / 0.062 | 30.64 | 3.70 | 700.15 | 3,335 / 182.2 |
| `edge-32-ground-gust-touchdown-shock` | Terminal 5 m/s gust + 100% sensor/delay/dropouts | 0.081 | 0.272 / 0.112 | 31.80 | 3.79 | 702.15 | 3,270 / 586.1 |

## 4. Reference CLI Scenarios

| Scenario | Status | Stage 2 Orbit | S1 Main / RCS Rem | Pos Error (m) | v_vert / v_horiz (m/s) | Unpowered Glide | 5 s Standing |
|---|---|---:|---:|---:|---:|---:|---|
| `calm` | PASS | **700.05 km** (`595.6 kg` rem) | **2,612.8 kg** / 275.0 kg | 0.002 | 0.567 / 0.006 | 92.24 s / 29.42 km | PASS (`0.000°`) |
| `example-weather` | PASS | **701.18 km** (`682.3 kg` rem) | **2,161.7 kg** / 263.4 kg | 0.010 | 0.533 / 0.133 | 106.48 s / 32.91 km | PASS (`0.095°`) |

## 5. Key Engineering Improvements Behind the ≤ 80% Certification

1. **100 t Methalox 7-Engine Redesign + Real 700 km Stage 2 Orbital Integrator (`rtls/orbit.py`)**: Two-burn spherical Earth orbital propagation with latitude-dependent Earth rotation passes all 487 cases including zero-rotation polar launch (`edge-31`).
2. **Entry Burn Ignition Throttle Ramp (`0.85 s`) & Dual Aerothermal Trigger**: Eliminates vacuum slosh torque shocks at MECO-2 restart while capping peak dynamic pressure at `33.31 kPa` (`79.3%`) and stagnation heat flux at `21.05 kW/m²` (`26.3%`).
3. **Supersonic High-Altitude Grid-Fin AoA Cap (`z ≥ 27 km`, `≤ 5°`) & Low-Altitude High-Authority Glide**: Removes supersonic fin+RCS saturation spikes and eliminates low-altitude proportional wind droop (`k_glide ≤ 0.28 deg/m`).
4. **Low-Pass Filtered Kalman Horizontal Bias (`bias_x`) + Analytical Zero-Tilt Predictor**: Suppresses sensor-noise-excited Stage B attitude ringing.
5. **Smooth `0.28 s` Vertical-Body Accel Ramp & Distance-Aware Terminal Horizontal Braking**: Removes mode-transition acceleration steps (`≤ 3.81 deg/s²`) and balances worst-corner position (`1.500 m`) and horizontal speed (`0.360 m/s`) cleanly inside 75% of limits.
