# DM100 2D Orbital Reusable RTLS Model Specification

[한국어](spec.ko.md) | [README](../README.md) | [Validation Protocol](validation.md) | [Validation Results](results.md)

DM100 is a theoretical **2D conceptual software model**, not a flight-qualified vehicle specification. The deterministic baseline seed is **378**. A 100-ton two-stage Methalox (LCH₄/LOX) launch vehicle lifts off on a 7-engine cluster, executes high-altitude cold stage separation (`53–71 km`), propagates **Stage 2 into a 700 km circular orbit (1,800 kg payload)** via a two-burn spherical-Earth transfer, and recovers **Stage 1 via RTLS** through unpowered grid-fin glide, 3→1 hoverslam, and 5 s post-touchdown upright landing-leg dynamics.

---

## 1. Vehicle Mass & Geometry Allocation

| Component | Model Value |
|---|---:|
| Stage 1 engines (7-engine Methalox cluster) | 1,200 kg |
| Stage 1 airframe, tanks, feed, pressurization, baffles & separation | 3,100 kg |
| Landing gear (planar left/right struts) | 1,300 kg |
| 4 grid fins, actuators & deployment hardware | 600 kg |
| **Stage 1 dry mass total** | **6,200 kg** |
| Stage 1 main propellant (LCH₄/LOX) / RCS | 69,500 / 300 kg |
| **Stage 1 initial mass** | **76,000 kg** |
| Stage 2 dry mass / propellant / payload | 1,550 / 20,650 / 1,800 kg |
| **Gross liftoff mass** | **100,000 kg** |

- Stage 1 airframe: length `25.5 m`, diameter `3.1 m` (`r = 1.55 m`; upper stack length `9.5 m`, lumped upper CG at `30.0 m`).
- Grid fins: station `22.4 m`, installed area `6.0 m²` (`3.0 m²` planar effective), supersonic Prandtl–Meyer lift-slope scaling, `±20°` deflection limit, `7,500 N·m` per-fin hinge-moment limit.
- Landing gear: `4.0 m` half-span struts, `10.0 m` pad half-width, spring-damper + Coulomb ground friction (`max stroke ≤ 0.90 m`, `contact load ≤ 4.0 g`).
- Coupled subsystems: `1,500 kg` retained liquid feed buffer (subset of main propellant), `75 kg` closed helium pressurization, nonlinear pendulum slosh, and momentum-conserving cold stage separation impulse.

---

## 2. Propulsion, Aerothermal & 8-Phase Flight Sequence

| Parameter | Model Value |
|---|---|
| Stage 1 7-engine cluster SL / vacuum thrust | 1,550 kN (7 × 221.4 kN) / 1,693 kN |
| Stage 1 SL / vacuum Isp (LCH₄/LOX) | 325 s / 355 s |
| Per-engine minimum throttle / max restarts | **36%–100%** / **max 3 restarts** (single center engine min `79.7 kN`, `TWR_min > 1.0`) |
| TVC gimbal range / slew rate / lag | ±6° / 15°/s / 0.05 s |
| RCS Isp / max torque | 260 s / 100,000 N·m |
| Stage 2 vacuum thrust / Isp | 240 kN / 380 s (2-burn 700 km circularization with Earth rotation) |
| Ascent / tail-first descent axial drag | $C_{D,\text{ascent}} = 0.38$ / $C_{D,\text{descent}} = 1.65$ (transonic rise included) |
| Stagnation heat flux model | Sutton–Graves ($\dot{q} \le 80\ \text{kW/m}^2$) |

### Flight Phases (`rtls/guidance.py`, `rtls/orbit.py`)
1. **7-Engine Ascent & Cold Separation (`ascent`)**: Gravity-turn ascent to the MECO-1 propellant reserve gate (`53–71 km` altitude) followed by momentum-conserving cold separation.
2. **Stage 2 700 km Circular Orbit (`rtls/orbit.py`)**: Propagates the detached upper stage from the exact MECO-1 state plus latitude-dependent Earth rotational velocity through Burn 1 (transfer apogee raise) and Burn 2 (apogee circularization), verifying `700 ± 15 km` circular orbit with positive propellant margin.
3. **Low-q Flip (`flip`)**: Rapid RCS attitude reversal in near-vacuum immediately after separation.
4. **3-Engine Boostback (`boostback`, Restart #1)**: Finite-burn predictor-corrector targeting the launch-site aerodynamic glide entry corridor.
5. **Ballistic Coast (`coast`)**: Apogee arc (`120–170 km`) and grid-fin deployment prior to atmospheric interface.
6. **Short Entry Burn (`entry_burn`, Restart #2)**: 3-engine burn triggered only when unpowered prediction exceeds `26.5 kPa` or `62 kW/m²`, featuring a `0.85 s` ignition throttle ramp to suppress vacuum slosh shocks before MECO-2 cutoff.
7. **Unpowered Grid-Fin Glide (`aero_glide`)**: `0%` throttle command across `25.0–43.0 km` altitude drop. Supersonic high-altitude (`z ≥ 27 km`) `5°` AoA cap and causal online drag/density-scale IIP estimation (`_predict_rtls_iip`) guide the booster cleanly over the pad.
8. **3→1 Minimum-Throttle Hoverslam (`landing_burn`, Restart #3) & 5 s Upright Standing (`touchdown_settle`)**:
   - High-thrust 3-engine deceleration (Stage A) stepping down to single-engine hoverslam (Stage B, non-hoverable).
   - Analytical zero-tilt unwind predictor (`_predict_zero_tilt_state`), low-pass filtered `bias_x`, smooth `0.28 s` vertical-body angular-acceleration ramp, distance-aware horizontal braking, convex LP terminal reference, and priority control allocation (`grid fins → TVC → RCS`).
   - Engine cutoff at touchdown followed by **5 s engine-off landing-leg contact & upright stability** verification.

---

## 3. Operating Envelope & Acceptance Criteria

### Allowed Input Envelope (`AprConditions`, `RecoveryDisturbances`, `SensorSettings`)
- Wind & gusts: ground wind `±8 m/s`, upper wind `±20 m/s`, vertical wind `±3 m/s`, forecast error `±2 m/s`, gust amplitude `0–5 m/s` (period `8–60 s`).
- Atmosphere: sea-level temperature `0–35 °C`, pressure `95–105 kPa`, hidden density scale `0.90–1.10` (final density within ±10% of reference), latitude `-90° to +90°` (Coriolis + Earth rotation).
- Perturbations, delays & sensors: thrust `0.97–1.03`, drag `0.85–1.15`, initial mass `0.99–1.01`, inertia `0.97–1.03`, position error `0–2 m`, velocity error `0–1 m/s`, observation delay `0–2 steps`, action delay `0–1 step`, engine lag `0–0.05 s`, extended IMU noise/bias (`≤ 0.1°`, `≤ 0.1°/s`, `±0.05`) and packet dropouts (`up to 2/20`).

### Acceptance Criteria (All Required Simultaneously)
- **Mission & Orbit**: Cold stage separation and **Stage 2 700 ± 15 km circular orbit (1,800 kg payload, positive propellant margin)**.
- **Flight Safety**: `q ≤ 42 kPa`, stagnation heat flux `≤ 80 kW/m²`, ascent `q·α ≤ 150 kPa·deg`, high-q `α ≤ 5.0°`, ascent axial load `≤ 4.5 g`, recovery load `≤ 5.5 g`, rate `≤ 15°/s`, acceleration `≤ 5.0°/s²`, gimbal `≤ 6°`, fin hinge `≤ 7,500 N·m`, mass/helium conservation, feed & slosh bounds.
- **First Touchdown**: `|x| ≤ 2.0 m`, vertical foot speed `≤ 1.5 m/s`, horizontal foot speed `≤ 0.5 m/s`, tilt `≤ 3.0°`, rate `≤ 0.5°/s`, main prop `> 0 kg`, RCS `≥ 0 kg`, gear deployed, feet on pad.
- **Post-Contact Settling (5 s)**: Leg stroke `≤ 0.90 m`, leg force `≤ 450 kN`, contact load `≤ 4.0 g`, zero hull ground collision, upright rest (`final_tilt ≤ 3.0°`, `final_speed ≤ 0.25 m/s`).
