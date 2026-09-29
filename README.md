# 2D Powered Landing Guidance

[한국어](README.ko.md)

A **3,000 kg planar rocket simulator** for ascent, 10 km separation and PID-guided
first-stage return to the launch pad. Aero-PID Return (APR) combines variable-mass
rigid-body dynamics, aerodynamic control fins and compliant landing-leg contact.

The retained implementation is the current 3 t APR model. Upper-stage flight and
insertion into the assumed 700 km orbit are **not simulated or verified**.
This is a provisional software model, not a hardware-qualified launch vehicle.

## Vehicle and mission

| Item | Current design |
|---|---|
| Launch mass | 3,000 kg |
| First-stage dry mass | 520 kg, including 24 kg control fins |
| Main / RCS propellant | 1,450 kg / 15 kg |
| Upper assembly | 1,015 kg, including a 30 kg payload |
| Main engine | 42 kN sea-level thrust; nominal Isp 260 s |
| Initial thrust-to-weight | Approximately 1.43 |
| Throttle / gimbal | Continuous 0-100% / ±6° |
| Separation gate | 10,000 ±100 m; target tilt 9°; after observed Max-Q |
| Recovery target | Original launch pad |

The controller coordinates return braking and landing ignition, horizontal
alignment and descent. Its final descent reference is 0.9 m/s; touchdown must
also meet position, horizontal speed, attitude, angular-rate and contact limits.

## Setup

Python 3.12.x is required; the project target is 3.12.7. The existing 3.12.14
fallback is also used for implementation checks.

~~~powershell
python -m pip install -e ".[dev,video]"
~~~

Core dependencies are NumPy, SciPy, PyYAML and Matplotlib. The optional `video`
extra supplies FFmpeg for MP4 export; `dev` supplies pytest and Ruff.
There is no Gymnasium, CasADi or reinforcement-learning dependency in the current
model. An existing environment can be reused.

## Run a flight

Run these commands from the repository root. Every output directory must be new.

~~~powershell
# Check eligible inputs without simulating a flight.
python -m scripts.run_nominal_flight --conditions configs/apr-operating-v1.yaml --check-inputs

# Calm air, standard atmosphere, no equipment uncertainty.
python -m scripts.run_nominal_flight --output-dir artifacts/apr-calm

# Altitude wind, gusts and healthy vehicle/sensor deviations.
python -m scripts.run_nominal_flight --conditions configs/apr-operating-v1.yaml --output-dir artifacts/apr-weather

# Start at the recorded first-stage separation state.
python -m scripts.run_nominal_flight --separation-record artifacts/apr-calm/result.json --output-dir artifacts/apr-return
~~~

Each flight saves `result.json` and `trajectory.png`. Results include the state
history, event and acceptance checks, separate fuel inventories and source/input
hashes. Failed flights remain failures; the runner returns a nonzero exit code.

Recovery-only runs preserve the supplied position, velocity, fuel and actual
throttle, but initialize controller memory. They do not revalidate upstream
ascent or unchanged separation feasibility under a new weather profile.

## Replay a saved flight

Rendering uses saved states and does **not** run another simulation. Playback is
at normal speed, with altitude-dependent sky colors, visible fins and legs, and
a thrust plume. Contact uses the feet rather than the rocket center.

~~~powershell
python -m powered_landing_guidance.visualization --replay artifacts/apr-calm/result.json --animation artifacts/apr-calm/flight.mp4 --preview artifacts/apr-calm/preview.png --fps 20
~~~

Existing MP4 and preview files are never overwritten.

## Conditions, acceptance and evidence

Inputs use root seed **378** and a fixed vehicle design. The YAML validator
rejects known faults, failed weather/health reviews and inputs outside the
provisional normal operating envelope. These bounds are project assumptions,
**not certified launch limits**. The example YAML is eligible input, not evidence
that the corresponding weather flight succeeds.

Touchdown requires COM position error ≤2 m, foot vertical speed ≤1 m/s, foot
horizontal speed ≤0.5 m/s, tilt ≤3° and angular rate ≤0.5°/s. Gear must be locked
and feet on the pad. Main propellant must remain positive and RCS nonnegative;
no extra residual-fuel reserve is required. Success also requires engine-off
standing, contact/load limits and the applicable ascent and recovery checks.

Retained flight evidence covers a calm full-flight development case. Success
throughout the operating envelope and independent success rates have **not**
been established. Supply/pressurization, coupled slosh, engine ignition/restart,
elastic structure, physical separation shock, upper-stage ignition/plume,
detailed thermal/ground-damage analysis and material-level leg/pad strength are
not connected to APR. See the [technical specification](docs/spec.md).

## Project layout and checks

~~~text
configs/apr-operating-v1.yaml             Eligible synthetic operating inputs
docs/spec.md, docs/spec.ko.md             English and Korean model contracts
scripts/run_nominal_flight.py            Flight runner and diagnostic plots
src/powered_landing_guidance/
  apr_recovery.py                        Current vehicle, environment and PID
  recovery_physics.py                    Shared physical laws and foot contact
  integrators.py                        Fixed-step integration
  visualization.py                      Saved-flight MP4 replay
  __init__.py                           Public package exports
tests/test_recovery_physics.py           Model, input, event and replay checks
~~~

~~~powershell
python -m pytest -p no:cacheprovider
python -m ruff check --no-cache .
python -m ruff format --check --no-cache .
~~~

Generated results live in Git-ignored `artifacts/`. Private notes and schedules
remain local under `.local/experiment_logs/`; neither belongs in the public
portfolio. Obsolete vehicle models and their learning pipelines are not part
of this implementation.
