# 100 t first-stage recovery model

[한국어](spec.ko.md) | [README](../README.md)

This is a theoretical **2D software model**, not a flight-qualified vehicle.
Root seed: **378**. The mission is launch, cold separation near 37 km and recovery
of the first stage at the original pad. The upper assembly's 500 kg payload and
700 km circular-orbit insertion are assumed; upper-stage flight is not integrated.

Optional angular sensor errors, packet omissions, the attitude estimator and frozen evaluation assumptions are documented in [Validation design](validation.md). The default observation assumptions here continue to describe the existing operating interface and regression inputs.

## Vehicle inventory

| Inventory | Mass |
|---|---:|
| First-stage engine | 1,200 kg |
| First-stage airframe, tanks and systems | 8,700 kg |
| Landing gear | 1,500 kg |
| Fins and associated equipment | 600 kg |
| First-stage dry total | 12,000 kg |
| First-stage main / RCS propellant | 70,200 / 300 kg |
| First-stage wet total | 82,500 kg |
| Upper dry / propellant / payload | 1,500 / 15,500 / 500 kg |
| Upper assembly / launch total | 17,500 / 100,000 kg |

The first-stage hull is 20 m long and 3 m in diameter. The upper assembly is 8 m
long with lumped COM at station 24 m above the first-stage hull base.

Four installed fins have 6 m² total area, represented by two planar equivalent
surfaces with 3 m² combined effective area. The 600 kg allocation includes
360 kg surfaces, 160 kg actuators, 60 kg deployment and 20 kg electrical equipment.
Four conceptual legs are represented by two feet at a 4 m half-span;
the pad half-width is 10 m.

Feed/pressurization, baffles and separation equipment occupy 1,015 kg **inside**
the airframe/systems budget: 65 kg closed helium, 450 kg bottle/regulation,
150 kg collector hardware, 300 kg baffles and 50 kg pushers.
The 1,500 kg retained liquid buffer is part of main propellant, not added mass.
These are concept allocations, not measured parts or qualified component designs.

| Propulsion | Assumption |
|---|---:|
| Sea-level thrust / Isp | 1.6 MN / 300 s |
| Vacuum thrust / Isp | Approximately 1.76 MN / 330 s |
| Throttle / gimbal travel | Ideal continuous 0-100% / ±6° |
| RCS Isp / maximum torque | 220 s / 100,000 N m |
| RCS type | Assumed chemical propulsion, not cold-gas nitrogen |
| Main descent speed reference | 1.0 m/s |

Nozzle pressure correction uses local pressure. Thrust and propellant use share
the same actual delivered mass flow. RCS depletion removes available RCS torque.

## Coordinates and integrated physics

The launch pad is the local origin: +x east/right, +z up.
Position is instantaneous total COM, not the hull base or feet.
Pitch theta is clockwise from vertical; positive gimbal rotates thrust clockwise.
There are two translations and one rotation, not six spatial degrees of freedom.

~~~text
thrust_x = thrust * sin(theta + gimbal)
thrust_z = thrust * cos(theta + gimbal)
clockwise_torque = arm_z * force_x - arm_x * force_z
~~~

Mass, COM and inertia change with propellant consumption, deployment and staging.
Thrust, atmosphere-relative aerodynamic forces, altitude-dependent gravity,
local Coriolis projection, RCS/fin moments and internal slosh are coupled.
Slosh changes body geometry relative to total COM; it is not a fictitious external
force. States are not overwritten at touchdown to manufacture success.

| State indices | Quantity |
|---|---|
| 0-5 | COM x/z, vx/vz, pitch and angular rate |
| 6-7 | Main and RCS propellant |
| 8-12 | Gear, fin deflection/extension, actual gimbal and throttle |
| 13-14 | Retained buffer mass and liquid acquisition fraction |
| 15-16 | Tank-ullage and bottle helium inventories |
| 17-18 | Slosh pendulum arc length and rate |

The plant integrates all 19 states. Nominal RK4 step is 0.010 s and control period
is 0.020 s. Foot contact uses event detection and smaller near-contact steps.
Four supplementary cases use 0.005 s integration steps; this is not a convergence
proof over the whole continuous envelope.

## Supply, slosh and staging

The reduced supply model uses one equivalent liquid at 826 kg/m³, a 1,500 kg
retained collector and a 650 kg/s supply limit. Liquid acquisition depends on
proper acceleration. Full nominal demand is approximately 543.8 kg/s; the buffer
provides approximately 2.76 s at full flow, with a 0.35 s settling time constant.

~~~text
main_dot = -delivered
buffer_dot = pickup - delivered
bulk_dot = -pickup
~~~

Buffer refill does not consume propellant twice; an inactive engine does not
drain retained liquid during coast. Supply shortage reduces mass flow and thrust
together and remains an explicit failure check.

Closed isothermal helium follows `p = m*R*T/V` with two conserved inventories.
Bottle volume is 1.3 m³ and total helium is 65 kg. Tank regulation targets 350 kPa,
not combustion chamber pressure; inlet supply declines toward 220 kPa.
Restart requires at least 350 kg buffer and 280 kPa tank pressure, then an assumed
0.2 s ignition delay. Vessel, volume and supply budgets are theoretical checks.

Slosh uses a nonlinear planar pendulum: 3% of remaining main liquid in a 0.8 m
mode, damping ratio 0.12. The coupled mass matrix retains body/modal momentum,
Coriolis and centrifugal terms. Restoring action uses proper axial acceleration,
not gravity artificially imposed during free fall. A 100 kN modal constraint-force
requirement and the tank geometry bound model validity; exceeding the domain fails.

Staging applies equal/opposite impulses with 0.5 m/s relative axial speed,
2 mm lateral offset and 0.15 s equivalent pulse duration. Linear/angular momentum
and inherited internal motion are retained. The separated booster is rebased
to its own COM; velocity and angular rate are not reset to zero. Recorded pulse-
average angular acceleration is not a resolved peak shock.

## Guidance and observations

Ascent starts its flow-limited pitch program at 15 s, with a 25° pitch cap.
It anticipates cutoff and finite attitude slew, aligns with airflow before engine
shutdown and checks the actual separation state. Recovery uses a low-pressure RCS
flip, finite-burn return prediction, advance entry braking, horizontal alignment
and vertical terminal landing. Prediction is reduced-order guidance, not the
acceptance simulation.

The controller uses delayed telemetry, delivered-command history, nominal plant
parameters and forecast mean wind. A planar navigation estimator tracks position,
velocity and acceleration bias. Raw position/velocity errors are bounded 2D
uniform-disk errors; a bound on raw noise is not a bound on filtered error.
The controller does not read hidden true wind, thrust or inertia deviations.

Attitude, gyro, propellant and actuator channels are ideal delayed telemetry.
Staging safety and ignition interlocks use actual modeled state/supply.
These ideal instrument assumptions are not a complete sensor implementation.
Acceptance uses the independently recorded true simulated state.

## Eligible environmental and healthy deviations

The atmosphere is a layered hydrostatic approximation over 0-100 km.
Latitude defaults to 37° for local Coriolis projection; this is not a full
3D geodetic Earth or a measured launch-site weather profile.

| Input | Allowed range |
|---|---|
| Pad horizontal mean wind, 0-50 m | Magnitude ≤8 m/s |
| Upper horizontal mean wind | Magnitude ≤20 m/s |
| Finite horizontal gust | Amplitude 0-5 m/s |
| Instantaneous horizontal wind | Pad ≤13 m/s, upper ≤25 m/s |
| Mean horizontal wind shear | ≤4 m/s/km |
| Vertical wind / shear | ±3 m/s / ≤3 m/s/km |
| Actual versus forecast mean wind | Difference ≤2 m/s |
| Final density | ±10% of the fixed nominal atmosphere |
| Sea-level temperature / pressure | 273.15-308.15 K / 95-105 kPa |
| Latitude | -90° to 90° |
| Drag / thrust / first-stage inertia scale | 0.85-1.15 / 0.97-1.03 / 0.97-1.03 |
| Initial total mass scale | 0.99-1.01, equivalent to ±1,000 kg |
| Raw 2D position / velocity error radius | ≤2 m / ≤1 m/s |
| Observation / action delay | 0-2 / 0-1 control periods |
| Engine lag | 0-0.05 s |

Wind profiles must span the full altitude domain. Optional temperature/pressure
profiles must also span it. Final density is checked against the **fixed** nominal
atmosphere, so incompatible temperature/pressure/density combinations are rejected.
Initial mass deviations belong to equivalent dry structure, not additional fuel.

Gusts have finite time and altitude windows; configuring amplitude does not prove
that the flight actually encounters it. Forecast guidance uses mean horizontal
wind only. The retained forecast-gust amplitude field is validated/recorded but
does not create a forecast gust waveform. Actual vertical wind is also withheld.

Known faults, failed weather/health reviews, precipitation and thunderstorm/hail
inputs are excluded. Review booleans do not model lightning or authorize a real
launch. Eligible inputs can still fail flight acceptance.

## Separation and acceptance

| Cold separation gate | Required state |
|---|---:|
| COM altitude | 37 ±1 km |
| Main propellant | ≥20,000 kg |
| Actual engine throttle | ≤0.001 |
| Dynamic pressure | ≤25 kPa |
| Angular rate / angle of attack | ≤5°/s / ≤5° |
| Ascent history | Max-Q already passed |

Mission completion, pre-contact flight safety and landing are separate groups;
overall success requires every applicable criterion.

| Pre-contact flight requirement | Limit |
|---|---:|
| Maximum dynamic pressure | 30 kPa |
| Ascent Q-alpha | 150 kPa deg |
| Ascent angle of attack at q ≥10 kPa | 5° |
| Ascent axial / recovery load | 3 / 5 g |
| Declared absolute acceleration ceiling | 6 g |
| Angular rate / angular acceleration | 15°/s / 5°/s² |
| Equivalent-fin hinge moment | 5,000 N m each |
| Fin model domain | Mach ≤8, angle of attack ≤25° |
| Gimbal | ±6° |

These are provisional concept requirements, not certified limits derived from
100 t launch mass. Simple strength and hinge estimates do not resolve shell
buckling, fatigue, vibration, thermal damage or a calibrated supersonic fin map.

| First-foot touchdown requirement | Limit |
|---|---:|
| COM position error | ≤2 m |
| Foot vertical / horizontal speed | ≤1.5 / ≤0.5 m/s |
| Tilt / angular rate | ≤3° / ≤0.5°/s |
| Main / RCS remaining | >0 kg / ≥0 kg |
| Contact geometry | Gear deployed, feet on pad, no hull collision |

The `kinematic_touchdown` assessment ends at first foot contact. Foot speed
includes body rotation; position refers to COM. No fixed extra residual reserve
is required, but exhausting fuel before completing the mission is a failure.
Legs are assumed adequate at ≤1.5 m/s. Contact forces, strut stroke, material
strength, engine-off settling and stable standing are **not evaluated**, not passed.

Recovery-only runs preserve the supplied separation state but reset controller
memory and mark upstream ascent checks as not evaluated. They do not establish
full-flight success. Replay is presentation only: hiding exhaust at contact
does not simulate engine shutdown or post-contact dynamics.

For a full-flight record with upper-stage staging states, replay carries the
recorded upper COM position, velocity and angular rate into a separate visual
departure. Its illustrative engine starts three seconds after separation, ramps
to a prescribed 18 m/s² thrust acceleration with a 1.2 s time constant, and uses
4 s angular-rate damping. A closed-form display curve includes constant gravity
at separation; it is not a variable-mass upper-stage dynamics or orbit solution.
The dashed overview ends at 100 km altitude or the end of the recording.
These visual assumptions do not consume simulated fuel or change first-stage
states, acceptance or telemetry. Missing staging states and recovery-only
records do not fabricate an upper-stage departure. Clip bounds select recorded
seconds only; full-flight rendering remains the default.

## Local UI contract

The optional `ui` extra exposes the packaged HTML interface through a loopback-only
FastAPI application. Flight calculations run in a separate local process. Only
one calculation or video-rendering job runs at a time; the interface polls its
status without starting another flight. No external API, analytics or CDN is used.
Mutation requests require the application's per-session token and same-origin
checks. Download routes accept only the app's generated result and media filenames.

The UI keeps the current 100 t design, seed 378, active fins, first-contact landing
assessment and 0.01 s integration step fixed. Environment and uncertainty inputs
are validated by `AprConditions` and `RecoveryDisturbances`, including combined
density and altitude-profile constraints. Completed flight failures retain their
results and are distinct from execution errors. Missing contact measurements are
not reported as zero or as passing landing criteria.

The result view uses recorded values and the calculation's input snapshot. Graphs
may use a bounded sample set for display; the downloadable JSON retains the full
record. First-contact speed values use landing-gear contact-point kinematics.
Only a successful full mission enables video generation from its existing record.
User-selected video intervals and resolutions affect rendering, not acceptance.
MP4 generation renders individual frames on the CPU and can take considerably
longer than flight calculation. Longer clips and higher resolutions add work;
use a short 640 × 360 interval for a quick preview. The UI explains this before
generation and while rendering. Keep the application open; recorded graphs and
tables remain available. No fixed wait time or GPU-rendering speedup is promised.
The Windows `launch-dm100.bat` launcher locates a working Python 3.12 interpreter,
creates an isolated `.venv` and installs the project with the `ui` and `video`
extras. It reuses a healthy environment and preserves/rebuilds an unusable or
relocated environment instead of trusting the presence of `python.exe`. First
setup and repairs require internet access. `--setup-only` prepares the environment
without opening the app; other arguments are forwarded to the local application.
Python itself is a prerequisite, not an automatic download. The local UI remains
a source-installed application. Fresh Windows installation and Linux-container
checks are recorded in [environment reproduction](results/reproduction.json).
Korean replay labels on Linux require a Hangul font. Standalone executable
distribution is outside the current release.

## Evidence and limitations

The original design recorded 205 development, 72 previously observed regression and
nine supplementary full-flight checks. The 277 primary and nine supplementary
denominators remain separate. Minimum main/RCS remaining were 2,958.185391 /
95.875194 kg; maximum RCS use was 204.124806 kg. Maximum foot vertical speed over
all 286 checks, including a half-step case, was 1.2550572 m/s.

The current repeated regression and newly frozen evaluation measurements are in
[the numerical report](results.md). Original records and new evaluation
populations have separate denominators.

The tracked `configs/apr-regression-v1.json` preserves the 286 declared inputs,
seed modes and integration steps for portable reruns. They are known regression
conditions, **not new independent holdout**. Results record all failures, fuel
ledgers, actual gust exposure, source/input hashes and execution integrity.
Finite sampled success does not prove the continuous envelope.

Outside scope: separate cryogenic fuel/oxidizer flow, boiling and pump cavitation,
multi-mode fluid impacts, flexible structure, connector/pyroshock transients,
combustion/restart reliability, upper-stage ignition/plume and orbital flight,
detailed heating/ground damage and leg/pad material strength. Helium inventory
transfer conserves mass, but its redistribution does not separately update COM.
No measured parameter calibration or real-vehicle safety certification is claimed.
