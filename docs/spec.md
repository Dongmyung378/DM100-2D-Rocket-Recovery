# 3 t APR Model Specification

[한국어](spec.ko.md) · [README](../README.md)

This contract describes the retained Aero-PID Return (APR) simulator, its
coordinates, inputs, events and acceptance rules. It is a provisional planar
model, not a calibrated vehicle or launch-safety certification.

## Vehicle and scope

The fixed launch mass is 3,000 kg: first-stage dry mass 520 kg, main propellant
1,450 kg, RCS propellant 15 kg and upper assembly 1,015 kg including a 30 kg
payload. The dry allocation is engine 110 kg, airframe/systems 326 kg, gear
60 kg and control fins 24 kg. Dimensions and component properties are lumped
design assumptions, not an actual component inventory.

Main sea-level thrust is 42 kN, nominal Isp 260 s and nozzle exit area 0.02 m².
Throttle is continuous 0-100% and gimbal is limited to ±6°. RCS provides up to
6,000 N·m at assumed Isp 220 s. Main and RCS propellant are consumed separately.

`AprFlight` owns one implementation for full-flight and recovery-only runs.
Full flight includes ascent, 10 km separation, first-stage return, compliant
foot contact, engine shutdown and settled standing. The assumed upper-stage
700 km orbit is neither propagated nor verified. There is no 3D flight model.

## Coordinates and state

The launch pad is the origin, +x is right/east in the modeled plane and +z is up.
Position refers to the **instantaneous mass center**, not the hull base or feet.
Body stations run upward from the hull base. Theta is clockwise from vertical;
positive gimbal rotates the force vector further clockwise.

~~~text
thrust_x = thrust * sin(theta + gimbal)
thrust_z = thrust * cos(theta + gimbal)
clockwise_torque = arm_z * force_x - arm_x * force_z
~~~

The planar model has x/z translation and one pitch rotation.

| State index | Quantity | Unit |
|---|---|---|
| 0, 1 | COM x, z | m |
| 2, 3 | vx, vz | m/s |
| 4, 5 | theta, omega | rad, rad/s |
| 6, 7 | Main and RCS propellant | kg |
| 8 | Gear extension | 0-1 |
| 9, 10 | Left/right tangential contact displacement | m |
| 11 | Actual fin deflection | rad |
| 12 | Fin extension | 0-1 |

Control is `[throttle, gimbal_rad, rcs_torque_nm, fin_target_rad]`.
Geometry, foot velocities and replay account for moving COM and body rotation.
The 520 kg dry mass is not an extra state or a residual-fuel target.

## Physical laws and control

| Part | Retained implementation |
|---|---|
| Translation/rotation | Variable mass, parallel-axis inertia, moving COM and inertia derivatives |
| Environment | Altitude gravity, ISA-style troposphere/isothermal layer and local Coriolis |
| Engine | Sea-level pressure correction, mass flow, optional first-order response |
| Body aerodynamics | Air-relative axial/cross-flow drag, CP moment, rotational damping and transonic correction |
| Fins | Bounded equivalent lift/drag, torque and hinge load; deployment and servo dynamics |
| Guidance | Forecast-based return braking/ignition; synchronized lateral alignment/descent; PID-style feedback and actuator allocation |
| Gear/contact | Deployment, unilateral spring/damper normal force, elastic stick/Coulomb slip, stroke/force limits and body-ground collision |
| Standing | Engine-off observation and continuous settled-state dwell |

The fixed control clock is 0.01 s. RK4 uses 0.01 s flight steps and 0.001 s
near-ground steps; this does not change the control interval. `advance()`
preserves queues, engine response, controller memory and propellant between
calls. Wind and air properties are evaluated at RK4 stage positions/times.

Fin coefficients are assumed, not a Falcon 9 grid-fin map. Deployed-fin
acceptance is limited to Mach ≤0.8 and |angle of attack| ≤25°, with hinge
load ≤400 N·m per fin. The current model is not qualified outside that envelope.

## Operating inputs

`configs/apr-operating-v1.yaml` requires schema 1, root seed **378**, every
declared field and unchanged vehicle values. Missing/unknown fields and
ineligible conditions are rejected before an output directory is created.

| Input | Provisional project range |
|---|---|
| Altitude wind | Ordered linear actual/forecast profiles spanning 0-20 km |
| Pad wind | Worst sustained wind plus gust amplitude ≤13.4 m/s throughout 0-50 m |
| Upper wind / shear | Worst wind including gust ≤28 m/s; profile shear ≤4 m/s/km |
| Gust | Sinusoidal amplitude 0-2 m/s; period 8-60 s; phase within ±pi |
| Forecast error | Worst actual/forecast difference including gust ≤2 m/s |
| Sea-level temperature / pressure | 273.15-308.15 K / 95-105 kPa |
| Healthy thrust / inertia scale | 0.98-1.02 / 0.97-1.03 |
| Sensor noise scale | 0-1 |
| Observation / action delay | 0-2 / 0-1 control steps |
| Engine lag | 0-0.05 s |
| Manual reviews | Health, lightning/cloud, temperature and recovery checks passed |
| Exclusions | Known faults, precipitation, thunderstorm or hail |

The 13.4 m/s pad reference originated from the Falcon 9 Crew Dragon
[30 mph sustained-wind criterion at 162 ft](https://www.nasa.gov/wp-content/uploads/2021/07/falcon9_crewdragon_launch_weather_criteria_fact_sheet.pdf).
Applying it to sustained wind **plus gust** over 0-50 m is a separate conservative
project rule. Other bounds are synthetic project assumptions, not certified
limits for this vehicle. Review booleans do not model lightning/cloud physics
or supply live weather/launch clearance. Eligibility is not flight feasibility.

Actual wind and thrust/inertia deviations affect the plant. Controller
prediction uses forecast wind and nominal parameters, with causal estimates
from noisy/delayed observations. At noise scale 1, position sigma is 0.5 m,
velocity sigma 0.1 m/s, attitude sigma 0.25° and rate sigma 0.25°/s.
Fuel, actuator and Max-Q telemetry remain idealized. Success uses true states;
permitted inputs that fail flight criteria are not removed from evaluation.
Leaving the 20 km profile domain fails domain acceptance.

## Events and acceptance

Separation targets 10,000 ±100 m and 9±0.5° tilt after observed Max-Q, with
dynamic pressure ≤25 kPa, |angle of attack| ≤5° and rate ≤5°/s.
Preparation uses tighter 2° angle-of-attack and 2°/s rate gates with a 0.3 s
dwell. Full-flight separation must pass true-state gates, not only noisy ones.
COM coordinates are rebased to the separated first stage while the physical
body position and local velocities remain consistent; no zero-velocity reset
is used.

| Phase | Required limits |
|---|---|
| Ascent | q ≤30 kPa; q-alpha ≤150 kPa·deg; |alpha| ≤5° when q ≥10 kPa |
| Ascent motion | Rate ≤15°/s; angular acceleration ≤5°/s²; axial acceleration ≤3 g |
| Return motion | Normal load ≤5 g and absolute limit 6 g; rate ≤15°/s; angular acceleration ≤5°/s² |
| Approach | Tilt ≤5°; prior alignment/dwell complete above 200 m |
| Touchdown | COM position error ≤2 m; foot vertical speed ≤1 m/s; foot horizontal speed ≤0.5 m/s |
| Touchdown attitude | Tilt ≤3°; rate ≤0.5°/s |
| Propellant | Main >0 kg; RCS ≥0 kg; no extra minimum reserve |
| Gear and pad | Gear locked; feet within ±10 m; stroke ≤0.3 m; normal force ≤18 kN per foot |
| After contact | Engine off; observe ≥3 s; continuous settled dwell ≥1 s |

The final descent reference is 0.9 m/s. Contact/load limits, body collision,
mass balance and applicable fin limits also remain required. State variables
are not overwritten to manufacture a successful touchdown.

`from_separation()` accepts the 13-state first-stage COM vector, event time,
prior Max-Q assertion, pre-rebase gate altitude and actual engine throttle.
It preserves supplied motion and inventories but starts new controller memory.
Recovery-only output sets upstream ascent criteria to `null`, lists them in
`not_evaluated_criteria` and does not independently verify the Max-Q assertion.

## Results and limits

`result.json` contains `design`, `physics`, `state_names`, `steps`,
`separation`, `first_contact`, `criteria`, `outcome`,
`simulated_acceptance_passed`, `simulation_scope`, `not_evaluated_criteria`,
input conditions, seed and inventories. The runner adds source/input hashes.
Existing results are not overwritten. Saved hashes describe the source version
that produced that file; later code refactoring does not update archived hashes.

Retained evidence is one calm full-flight development case and implementation
checks, not a success-rate estimate across weather conditions or an independent
qualification campaign. Rendering replays stored poses at normal speed and
does not add a new flight experiment or success claim.

**Unconnected:** liquid acquisition/retention, tank pressurization/feed,
coupled slosh, ignition/restart dynamics, elastic structure, physical staging
impulse, upper-stage ignition/plume, detailed heating/ground damage and material
leg/pad strength. Force/stroke/contact checks are implemented but do not establish
material strength or hardware safety. Retained dimensional/material fields are
assumptions; unused subsystem parameters are not evidence of a connected solver.
