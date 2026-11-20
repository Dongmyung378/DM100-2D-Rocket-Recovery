# DM100 - 2D SpaceX-Style Rocket Launch and First-Stage Recovery

[한국어](README.ko.md) | [Recorded results](docs/results.md) | [Model specification](docs/spec.md) | [Validation protocol](docs/validation.md) | [Roadmap (KO)](docs/roadmap.ko.md)

A **2D concept simulator** for launching a 100 t two-stage rocket, cold-separating
near 37 km, and returning the first-stage booster to its launch pad using a
**SpaceX (Falcon 9)-style RTLS profile**. Built on a single unified engine with
decoupled `Plant ↔ Forces ↔ GNC` interfaces (`rtls/`), DM100 simulates low-q
flip, boostback, short entry burn (MECO-2), **unpowered grid-fin aerodynamic
glide (9.6–14.5 km, 0% commanded throttle)**, **3→1 min-throttle vertical
Hoverslam**, and **5 s engine-off landing-leg contact and upright-standing
dynamics**.

The offline local browser interface defaults to English (with Korean toggle),
providing **interactive 2D live flight playback**, **50 Hz 3D replay JSON
export (ENU meters + unit quaternion `wxyz`)**, and optional H.264 MP4 rendering.

![Local experiment interface](docs/assets/ui.png)

## Physics, Guidance, and Control Features

- **7-Phase SpaceX RTLS Mission**: `ascent → flip → boostback → coast → entry_burn → aero_glide → landing_burn → touchdown_settle`
- **Vehicle & Contact Physics**: variable mass and inertia, tail-first blunt-base
  descent drag ($C_{D,\text{descent}} = 1.65$), grid-fin effective angle of attack
  and hinge torque limits, **3-engine cluster (44% per-engine throttle floor,
  3-restart budget)**, reduced-order feed/pressurization/slosh/separation models,
  and **5 s post-contact spring-damper & Coulomb-friction landing legs**
- **Causal Estimation**: causal navigation and attitude estimation under bounded
  noise, bias, observation/action delays, engine lag, and scheduled packet
  dropouts (never uses future observations or hidden plant state)
- **Robust RTLS Guidance & Priority Allocation**:
  - Online causal drag/density scale estimation during unpowered `aero_glide`
    (`_predict_rtls_iip`)
  - Analytical committed-slew zero-tilt predictor (`_predict_zero_tilt_state`)
    accounting for finite `2.15 deg/s²` attitude unwind
  - 3-engine Stage A transition drift lookahead + 1-engine Stage B
    touchdown-time no-overshoot speed cap (`v_no_overshoot`)
  - Noise-immune deterministic `vertical_body` latch (`h ≤ 21.5 m`), gimbal step
    limiting, interval-matched priority control allocation (grid fins → TVC → RCS),
    and a convex-LP terminal reference planner
- **Stack**: Python 3.12, NumPy/SciPy, FastAPI/Uvicorn, HTML/CSS/JS, and
  Matplotlib/FFmpeg. Guidance and estimation use explicit physics/control models;
  **this project does not train a machine-learning policy**.

## Full 487-Case Validation Results

Evaluated in a single parallel sweep covering **32 adversarial extreme edge
cases** (including 100% simultaneous operating-limit nightmare corners), all
**169 cases** in `configs/validation-study-v1.json`, and all **286 cases** in
`configs/apr-regression-v1.json` ([validation-487.json](docs/results/validation-487.json);
full analysis in [docs/results.md](docs/results.md)):

| Suite | Passed / total | Main remaining (kg) | Max `\|x\|` error (m) | Max `q` (kPa) | Max angular accel (deg/s²) | Unpowered glide (km) | 5 s standing |
|---|---:|---:|---:|---:|---:|---:|---:|
| Extreme edge cases (`edge-cases`) | **32 / 32 (100%)** | 9,343 – 10,287 | 1.766 | 29.61 | 4.32 | 9.6 – 14.3 | 32 / 32 |
| Development (`development`) | **16 / 16 (100%)** | 9,505 – 10,020 | 0.313 | 28.37 | 3.61 | 11.4 – 13.6 | 16 / 16 |
| Coupled holdout (`holdout`) | **96 / 96 (100%)** | 9,294 – 10,172 | 0.610 | 29.44 | 4.21 | 11.2 – 13.6 | 96 / 96 |
| Extended-sensor holdout (`sensor-holdout`) | **57 / 57 (100%)** | 9,406 – 10,113 | 0.436 | 28.99 | 4.14 | 11.2 – 13.6 | 57 / 57 |
| Regression suite (`regression`, 3 splits) | **286 / 286 (100%)** | 9,385 – 10,301 | 1.975 | 28.96 | 4.36 | 10.5 – 14.5 | 286 / 286 |
| **Total** | **487 / 487 (100.0%)** | **9,294 – 10,301** | **1.975** | **29.61** | **4.36** | **9.6 – 14.5** | **487 / 487** |

Aggregate highlights across all 487 flights:
- **Touchdown position error**: `0.000–1.975 m` (median **`0.024 m`**, limit `≤ 2.0 m`)
- **Foot vertical / horizontal speed**: `0.159–1.430 m/s` (`≤ 1.5`) / `0.000–0.481 m/s` (`≤ 0.5`)
- **Touchdown tilt / angular rate**: `≤ 0.078°` (`≤ 3.0°`) / `≤ 0.147 deg/s` (`≤ 0.5 deg/s`)
- **Peak angular accel / `q` / recovery load**: `≤ 4.36 deg/s²` (`≤ 5.0`) / `≤ 29.61 kPa` (`≤ 30.0`) / `≤ 4.86 g` (`≤ 5.0`)
- **Median propellant remaining**: **`9,772.0 kg`** main / **`252.2 kg`** RCS (`3/3` restarts)
- **5 s engine-off landing-leg standing**: **487 / 487 (100.0%)** (`leg stroke ≤ 0.278 m`, `contact load ≤ 1.98 g`)

![487-case validation and safety margin distribution](docs/assets/dm100-validation.svg)

![SpaceX RTLS phase propellant budget](docs/assets/dm100-propellant.svg)

![SpaceX RTLS trajectory and unpowered glide profile](docs/assets/dm100-trajectory.svg)

Canonical 50 Hz 3D replay exports (ENU meters + unit quaternion `wxyz`, ready for
UE5 / MuJoCo) are provided in
[calm_replay_50hz.json](docs/results/replays/calm_replay_50hz.json) and
[example_weather_replay_50hz.json](docs/results/replays/example_weather_replay_50hz.json).

## Run Locally (Browser UI)

On Windows x86-64, clone or extract the repository and run
[launch-dm100.bat](launch-dm100.bat). The launcher reuses Python 3.12 or
prepares a project-local Python 3.12.15 + `.venv` automatically, and reuses a
healthy environment offline. The UI opens at `127.0.0.1`; stop with `Ctrl+C`.
See [docs/runtime.md](docs/runtime.md) for environment compatibility details.

```powershell
# Prepare the environment only, or verify bundled font & H.264 encoder.
.\launch-dm100.bat --setup-only
.\launch-dm100.bat --doctor
```

Manual installation:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -c requirements/runtime-constraints.txt -e ".[ui,video]"
.\.venv\Scripts\python.exe -m powered_landing_guidance.local_app
```

Linux x86-64 (glibc 2.28+):

```bash
sh launch-dm100.sh
sh launch-dm100.sh --setup-only
sh launch-dm100.sh --doctor
```

All calculations, saved artifacts, and rendering run locally offline with zero
external APIs or CDNs:

1. Select one of 7 built-in presets (calm, coupled weather, extreme shear/gust, extreme Max-Q, heavy low-thrust, etc.) or custom eligible parameters.
2. Run a flight and inspect the **interactive 2D live flight replay** (play/pause, timeline scrubber, 0.5×–20× speed, phase jump buttons, live grid-fin/plume/RCS/leg-stroke HUD), `Glide & Control` plots, SpaceX phase fuel table, and mission/safety/touchdown/5 s standing checks.
3. Optionally render an H.264 MP4 replay or download the **50 Hz 3D replay JSON (`replay_3d.json`)**.

## CLI Flights and 487-Case Reproduction

```bash
# Run single flights and export 50 Hz ENU+quaternion 3D replay JSON
python scripts/run_flight.py --scenario calm --export-replay artifacts/calm_replay_50hz.json
python scripts/run_flight.py --scenario example-weather --export-replay artifacts/example_weather_replay_50hz.json

# Run the 32 extreme edge cases or the complete 487-case validation campaign
python scripts/evaluate_validation.py --suite edge-cases --workers 8
python scripts/evaluate_validation.py --suite full-487 --workers 14 --out-json artifacts/validation-487.json
```

Calculations use CPU `float64`. When running multi-worker sweeps, set
`OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1`.

## Vehicle Parameters and Acceptance Criteria

| Item | Model value |
|---|---:|
| Launch / first-stage wet mass | 100,000 / 82,500 kg |
| First-stage dry / main / RCS mass | 12,000 / 70,200 / 300 kg |
| Upper dry / propellant / payload mass | 1,500 / 15,500 / 500 kg |
| Sea-level / vacuum thrust (3-engine cluster) | 1,600 / ~1,760 kN |
| Sea-level / vacuum Isp | 300 / ~330 s |
| Throttle / restarts / gimbal | 44% per-engine floor (3→1 Hoverslam) / max 3 restarts / ±6 deg |
| Cold separation gate | 37 ±1 km; at least 20,000 kg main propellant |
| First-contact criteria | `\|x\| ≤ 2 m`, `v_vert ≤ 1.5 m/s`, `v_horiz ≤ 0.5 m/s`, `tilt ≤ 3°`, `rate ≤ 0.5°/s` |
| Post-contact criteria | 5 s engine-off upright leg standing, stroke `≤ 0.90 m`, load `≤ 4.0 g` |

See [docs/spec.md](docs/spec.md) and [docs/validation.md](docs/validation.md)
for coordinates, plant states, operating ranges, and conservation checks.

## Software Checks

```bash
python -m pip install -c requirements/runtime-constraints.txt -e ".[ui,video,dev]"
python -m pytest -q -p no:cacheprovider
python -m ruff check --no-cache .
python -m ruff format --check --no-cache .
```

The [Windows/Linux CI workflow](.github/workflows/checks.yml) automatically checks
linting, formatting, unit/integration/UI tests, the recorded 487-case validation
JSON, launchers, and wheel assets.

## Repository Map

```text
configs/                          Fixed operating, regression (286), and study (169) inputs
docs/spec.md, spec.ko.md          Bilingual physical-model specification
docs/validation.md, validation.ko.md  Validation protocol and reproduction commands
docs/results.md, results.ko.md    487-case validation & engineering improvements report
docs/roadmap.ko.md                2D RTLS -> 3D MuJoCo -> UE5 expansion roadmap
docs/results/                     validation-487.json and 50 Hz 3D replay JSONs
docs/assets/                      Local UI screenshot and visual assets
requirements/                     Tested runtime & dev dependency constraints
scripts/                          Flight CLI (run_flight.py) and 487-case evaluator (evaluate_validation.py)
src/powered_landing_guidance/     Shared physics, SpaceX RTLS engine (rtls/), local API & UI
tests/                            Physics, RTLS GNC, estimator, and local UI test suites
launch-dm100.bat / .sh            Windows & Linux auto-setup and app launchers
```

## Scope

This is a theoretical 2D software concept model covering planar translation and
pitch rotation. Coefficients are assumed rather than hardware-calibrated, and
upper-stage orbital insertion is not simulated. Flexible structures, detailed
fluid/thermal/plume physics, leg/pad material strength, 3D flight, and real-vehicle
certification are outside scope. Full 3D 6-DOF physics (MuJoCo) and cinematic
visualization (Unreal Engine 5) will proceed in a separate 3D repository
([roadmap](docs/roadmap.ko.md)).
