# DM100 2D SpaceX RTLS Model Specification

[한국어](spec.ko.md) | [README](../README.md) | [Validation design](validation.md) | [Recorded results](results.md)

A theoretical **2D software concept model**, not a flight-certified vehicle.
The reference seed is **378**. After liftoff, the vehicle performs cold stage
separation near 37 km and returns the first stage to its launch pad. Upper-stage
delivery of a 500 kg payload to a 700 km circular orbit is assumed and not
integrated.

---

## 1. Vehicle Mass & Geometry

| Component | Value |
|---|---:|
| First-stage engines (3-engine cluster) | 1,200 kg |
| First-stage structure, tanks, feed/pressurization/baffle/staging | 8,700 kg |
| Landing gear (left/right planar struts) | 1,500 kg |
| 4 grid fins, actuators, and deployment hardware | 600 kg |
| **First-stage dry mass** | **12,000 kg** |
| First-stage main / RCS propellant | 70,200 / 300 kg |
| **First-stage wet mass** | **82,500 kg** |
| Upper-stage dry / propellant / payload | 1,500 / 15,500 / 500 kg |
| **Total launch mass** | **100,000 kg** |

- First-stage airframe: length `20 m`, diameter `3 m` (upper stage length `8 m`, lumped COM `24 m` above booster base).
- Grid fins: total area `6 m²` (planar effective area `3 m²`), effective angle of attack $\alpha_{\text{eff}} = \alpha_{\text{body}} + \delta_{\text{fin}}$, deflection limit `±20°`, hinge-moment limit `5,000 N·m`, base drag coefficient `0.35`.
- Landing gear: two planar struts at `±4 m` half-span, pad half-width `10 m`, spring-damper + Coulomb ground-friction contact dynamics.
- Coupled subsystems: `1,500 kg` retention buffer (within main propellant), `65 kg` closed helium pressurization, reduced-order slosh, and momentum-conserving cold stage separation.

---

## 2. Propulsion, Aerodynamics & 7-Phase RTLS Sequence

| Parameter | Value |
|---|---|
| 3-engine sea-level / vacuum thrust | 1,600 kN (3 × 533.3 kN) / ~1,760 kN |
| Sea-level / vacuum Isp | 300 s / ~330 s |
| Per-engine throttle floor / restart budget | **44%–100%** / **max 3 restarts** (1-engine floor `234.7 kN`, `TWR > 1.0` near dry mass) |
| TVC gimbal limit / slew rate / lag | ±6° / 15°/s / 0.05 s |
| RCS Isp / torque limit | 220 s / 100,000 N·m |
| Ascent / tail-first descent axial drag | $C_{D,\text{ascent}} = 0.40$ / $C_{D,\text{descent}} = 1.65$ (with transonic drag rise) |

### Flight Phases (`rtls/guidance.py`)
1. **`ascent`**: Gravity-turn ascent to cold separation gate (`37 ± 1 km`, `≥ 20,000 kg` main reserve, `q ≤ 25 kPa`).
2. **`flip`**: Rapid low-q (`q ≤ 250 Pa`) RCS flip immediately after separation.
3. **`boostback` (Restart #1)**: 3-engine finite-burn predictor-corrector targeting the return glide window.
4. **`coast`**: Ballistic return with grid-fin deployment below `36 km`.
5. **`entry_burn` (Restart #2)**: Short supersonic braking burn only while predicted unpowered peak `q` exceeds the safety trigger (`25 kPa`), followed by full MECO-2 cutoff.
6. **`aero_glide`**: **Zero-fuel unpowered grid-fin aerodynamic glide** (`9.6–14.3 km`), using causal online drag-scale estimation from `est.accel_bias_z_m_s2` for IIP steering.
7. **`landing_burn` (Restart #3) → `touchdown_settle`**:
   - 3→1 engine non-hovering vertical Hoverslam.
   - **Analytical zero-tilt committed-slew predictor (`_predict_zero_tilt_state`)** compensating finite `2.15°/s²` attitude unwind lag without overshoot.
   - Convex-LP terminal reference planner + priority control allocation (`Fins → TVC → RCS`) with interval-matched TVC torque (`sup_mid`) during upright terminal translation.
   - Immediate engine cutoff at touchdown followed by **5 s engine-off landing-leg spring-damper/friction standing stability** verification.

---

## 3. Operating Envelope & Acceptance Criteria

- **Environment**: ground wind `±8 m/s`, upper wind `±20 m/s`, vertical wind `±3 m/s`, forecast error `±2 m/s`, gust `0–5 m/s` (`8–60 s` period), temperature `0–35 °C`, pressure `95–105 kPa`, density scale `0.90–1.10` (hydrostatic density within ±10% of standard), latitude `-90°` to `+90°` (Coriolis enabled).
- **Vehicle & sensors**: thrust `0.97–1.03`, drag `0.85–1.15`, initial mass `0.99–1.01`, inertia `0.97–1.03`, nav noise `0–2 m` / `0–1 m/s`, obs delay `0–2 steps`, action delay `0–1 step`, engine lag `0–0.05 s`, IMU noise/bias (`≤ 0.1°`, `≤ 0.1°/s`, `±0.05`) and packet dropouts (`up to 2/20`).
- **Acceptance**: flight safety (`q ≤ 30 kPa`, `Q-alpha ≤ 150 kPa·deg`, recovery load `≤ 5 g`, rate `≤ 15°/s`, accel `≤ 5°/s²`, fin hinge `≤ 5,000 N·m`, mass/helium/feed/slosh valid), first touchdown (`|x| ≤ 2.0 m`, `v_vert ≤ 1.5 m/s`, `v_horiz ≤ 0.5 m/s`, `tilt ≤ 3.0°`, `rate ≤ 0.5°/s`, `main > 0`, `RCS ≥ 0`), and 5 s post-contact standing (`leg_stroke ≤ 0.75 m`, `leg_force ≤ 450 kN`, `contact_load ≤ 4.0 g`, no hull collision, upright settled).
