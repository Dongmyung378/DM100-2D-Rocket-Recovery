# 100 t 2D Powered Landing Guidance

[한국어](README.ko.md) · [Model specification](docs/spec.md)

A planar simulator for **launch, cold staging and first-stage return to the
launch pad**. It combines variable-mass rigid-body dynamics, model-based guidance,
feedback attitude control, engine gimbal, RCS and aerodynamic fins.

The current vehicle launches at **100,000 kg**, with **70,200 kg main propellant**
and **300 kg RCS propellant** in the first stage. The upper assembly carries a
**500 kg payload** with a theoretical **700 km circular-orbit** objective.
Upper-stage orbital insertion is assumed, not simulated or verified.

## Vehicle and model

| Item | Current design |
|---|---:|
| Launch / first-stage wet mass | 100,000 / 82,500 kg |
| First-stage dry mass | 12,000 kg, including fins and landing gear |
| First-stage main / RCS propellant | 70,200 / 300 kg |
| Upper dry mass / propellant / payload | 1,500 / 15,500 / 500 kg |
| First-stage sea-level / vacuum thrust | 1,600 / approximately 1,760 kN |
| First-stage sea-level / vacuum Isp | 300 / approximately 330 s |
| Initial thrust-to-weight ratio | Approximately 1.63 |
| Throttle / gimbal | Ideal continuous 0-100% / ±6° |
| Separation gate | 37 ±1 km; at least 20,000 kg main propellant |
| Recovery target | Original launch pad |

The 19-state model includes altitude-dependent gravity, a layered atmosphere,
wind and gusts, aerodynamic forces, changing mass properties, actuator response,
delayed observations, liquid acquisition/feed, closed helium pressurization,
nonlinear pendulum slosh, restart readiness and separation impulses.
Four installed fins are represented by two equivalent planar surfaces.

The project evaluates first-foot touchdown, not post-contact settling. Legs are
assumed to withstand foot vertical speed ≤1.5 m/s; contact loads and material
strength are not assessed. Components and coefficients are theoretical design
assumptions, not calibrated hardware data. See the [model contract](docs/spec.md)
for supported inputs, physics and limitations.

## Setup and a single flight

Python 3.12.x is required; the working environment uses Python 3.12.7.

~~~powershell
python -m pip install -e ".[dev,video]"

# Validate the operating input without flying.
python -m scripts.run_nominal_flight --conditions configs/apr-operating-v1.yaml --check-inputs

# Calm wind, standard atmosphere and no vehicle/sensor deviations.
python -m scripts.run_nominal_flight --output-dir artifacts/apr100t/nominal-current --progress

# Full flight with the configured weather and healthy deviations.
python -m scripts.run_nominal_flight --conditions configs/apr-operating-v1.yaml --output-dir artifacts/apr100t/configured-current --progress
~~~

Use a new output directory for each flight. The runner saves `result.json` and
`trajectory.png`, including true states, events, acceptance, fuel ledgers and
source/input hashes. Failed acceptance returns a nonzero exit code.
The fixed root seed is **378**. Input eligibility is not flight success.

## Verified results

The current design passed the following fixed-input full-flight checks, requiring
mission completion, flight safety, landing and fuel conservation together:

| Test set | Passed / tested |
|---|---:|
| Known development conditions | 205 / 205 |
| Previously observed evaluation conditions, tested as regression | 72 / 72 |
| Supplementary actual-exposure and half-step checks | 9 / 9 |

The 277 primary checks and nine supplementary checks have separate denominators.
**All 286 checks passed**, without changing controller or acceptance limits to
obtain a pass. These are known-condition regression results, not a new independent
holdout or a proof for every continuous input combination.

| Quantity across all 286 checks | Worst observed value | Limit |
|---|---:|---:|
| Touchdown COM position error | 0.839 m | 2 m |
| Foot vertical / horizontal speed | 1.2551 / 0.202 m/s | 1.5 / 0.5 m/s |
| Touchdown tilt / angular rate | 0.0212° / 0.0226°/s | 3° / 0.5°/s |
| Maximum dynamic pressure | 28.949 kPa | 30 kPa |
| Maximum pre-contact angular acceleration | 3.780°/s² | 5°/s² |
| Minimum main / RCS propellant remaining | 2,958 / 95.875 kg | >0 / ≥0 kg |

Maxima are rounded upward and remaining propellant downward. The largest RCS use
was 204.124806 kg, leaving 95.875194 kg, or 31.96% of its initial inventory.
The calm check also passed, with approximately 1.005 m/s foot vertical speed and
5,796 kg main / 262.546 kg RCS remaining. Execution used eight CPU processes with
the NumPy/SciPy float64 backend; GPU acceleration was not implemented or benchmarked.

Raw current results are saved locally under `artifacts/apr100t/rcs300-v1`.
Generated reports, media and private notes are Git-ignored; they are not bundled
with a fresh clone. The tracked regression input manifest supports rerunning the
same declared conditions without needing local historical records.

## Repeat the regression suite

The tracked `configs/apr-regression-v1.json` declares the same 286 known inputs,
observation seed modes and integration steps. Preparation validates and hashes
them without flying; execution retains every failed case and does not overwrite
existing results.

~~~powershell
python -m scripts.evaluate_regression_suite --prepare --output-dir artifacts/apr100t/regression-v1
python -m scripts.evaluate_regression_suite --run all --workers 8 --output-dir artifacts/apr100t/regression-v1
~~~

These commands need no private evidence bundle. Use a new output directory for
another campaign. Running all conditions is substantially slower than input
validation; `--workers` controls independent CPU flight processes. Repeating a
known condition does not make it an independent evaluation.

## Results reports

The report tool reads completed current-design results without executing flights.
English is the default; `--language ko` selects Korean and `--language both`
generates linked English/Korean versions.

~~~powershell
# Requires the completed current results saved locally.
python -m scripts.present_validation_results --language both

# After the single flight and regression commands above, including on a fresh clone.
python -m scripts.present_validation_results --campaign-dir artifacts/apr100t/regression-v1 --nominal-result artifacts/apr100t/nominal-current/result.json --out-dir artifacts/apr100t/presentation-current --language both
~~~

Default output is `artifacts/apr100t/presentation-current`, with
`validation-report.html`, `validation-report.ko.html`, overview/telemetry graphs
and `presentation-data.json`. A fresh clone has no generated results: run the
flights first and point `--campaign-dir` and `--nominal-result` at those outputs.
Use `--out-dir` for a new report directory. Reports and graphs do not create
an independent test or a new flight result.

## Animation

Render a completed current-design flight without running another simulation:

~~~powershell
python -m powered_landing_guidance.visualization --replay artifacts/apr100t/nominal-current/result.json --animation artifacts/apr100t/nominal-current/flight.mp4 --preview artifacts/apr100t/nominal-current/preview.png --fps 24 --width 1920 --end-hold 4
~~~

The replay has an altitude-dependent sky, rocket and ground facilities, engine/RCS
plumes and a telemetry dashboard. Playback is normal-speed Full HD at 24 fps.
At first contact, exhaust is hidden and the recorded pose is held for four seconds.
This is a presentation effect, not simulated engine shutdown or stable standing.
MP4 generation is a separate command; running a flight alone does not create a video.

## Structure and checks

~~~text
configs/                         Operating input and fixed regression manifest
docs/                            English/Korean model specification
scripts/                         Flight, regression and results-report tools
src/powered_landing_guidance/     Dynamics, control, systems and visualization
tests/                           Physics, conservation, navigation and protocol checks
~~~

~~~powershell
python -m pytest -p no:cacheprovider
python -m ruff check --no-cache .
python -m ruff format --check --no-cache .
~~~

Software checks and finite synthetic flights do not certify real-vehicle safety.
