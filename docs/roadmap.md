# DM100 Development Roadmap (2D 700 km Orbital RTLS → 3D 6DOF MuJoCo + Unreal Engine 5)

[한국어](roadmap.ko.md) | [README](../README.md) | [Model Spec](spec.md) | [Validation Protocol](validation.md) | [Validation Results](results.md)

This document summarizes the completed 2D architecture of the **DM100 100 t Methalox 700 km circular-orbit (1,800 kg) + first-stage reusable RTLS simulator** and the expansion plan for a dedicated **3D 6-DOF physics (MuJoCo) + cinematic visualization (Unreal Engine 5)** repository.

---

## 1. End-to-End Architecture Strategy

```text
[Stage 1: This Repo · 2D Orbital RTLS]
  · 100 t 7x Methalox · 700 km circular orbit (1,800 kg payload)
  · Strict Plant <-> Forces <-> GNC boundary isolation
  · 25.0~43.0 km unpowered grid-fin glide · 3->1 hoverslam · 5 s standing
  · 487 / 487 passed · all 16 limits certified <= 79.3% (<= 80% margin)
        |
        v  50 Hz 3D Replay JSON (ENU meters + unit quaternion wxyz)
[Stage 2: Dedicated 3D Repo · MuJoCo 6DOF]
  · MJCF Stage 1 rigid body + 4 independent grid fins + 4 landing legs
  · 2D Forces/GNC -> 6DOF 3D extension (MEKF + 3D SOCP + roll/yaw mix)
  · Headless 3D crosswind & cross-range Monte Carlo certification
        |
        v  Validated offline 50 Hz replay stream
[Stage 3: Unreal Engine 5 Cinematic Rendering]
  · UE5 Python Level Sequence import (ENU m -> UE5 cm)
  · Niagara Methalox plume, RCS thrusters & reentry heating
  · Multi-camera ground / onboard / landing-pad Movie Render Queue
```

### Separation of Responsibilities
- **Physics & Verification Engine (2D RK4 → 3D MuJoCo)**: Owns `float64` numerical integration, landing-leg contact and tip-over dynamics, high-throughput headless batch runs, and deterministic reproducibility.
- **Visualization Renderer (Unreal Engine 5)**: Never intervenes in physics; consumes only verified **50 Hz replay JSON (`dm100_replay_3d_v1`, right-handed ENU meters + unit quaternion `wxyz`)** to render offline cinematics.

---

## 2. Stage 1 (Completed): 2D DM100 700 km Orbital Reusable RTLS

- **Clean Interface Isolation (`rtls/interfaces.py`, `rtls/forces.py`)**: Isolates controller and plant via `TelemetryPacket → ActuatorCommand` while keeping external aerodynamics/propulsion modular for direct porting to MuJoCo `xfrc_applied` callbacks.
- **700 km Circular Orbit + 8-Phase Flight Sequence (`rtls/guidance.py`, `rtls/orbit.py`)**:
  1. `ascent` + `rtls/orbit.py`: 7-engine Methalox ascent, high-altitude (`53~71 km`) cold separation, and two-burn spherical Earth Stage 2 insertion into `700 ± 15 km` circular orbit (`1,800 kg`).
  2. `flip` → `boostback` → `coast`: Low-q RCS flip, 3-engine boostback, `120~170 km` ballistic apogee return, and grid-fin deployment.
  3. `entry_burn`: `0.85 s` ignition throttle ramp and dual dynamic-pressure/heat-flux triggers holding peak dynamic pressure `≤ 33.31 kPa` and stagnation heat flux `≤ 21.05 kW/m²`.
  4. `aero_glide`: **0% throttle unpowered grid-fin glide (`25.0~43.0 km`)**, supersonic `5°` high-altitude AoA cap, and causal online drag/density-scale IIP guidance.
  5. `landing_burn` → `touchdown_settle`: 3→1 minimum-throttle (`36%`) hoverslam (analytical zero-tilt predictor `_predict_zero_tilt_state` + low-pass filtered `bias_x` + convex LP planner + priority control allocation) and **5 s engine-off upright landing-leg stability**.
- **Full Certification (`scripts/evaluate_validation.py`)**: **487 / 487 (100.0%)** passed including 100% simultaneous worst-case corners, with peak usage **`79.3%`** across all 16 limits (`7,792` individual checks).
- **Offline Local Browser Workbench (`local_app.py`, `ui/index.html`)**: Oxford Blue (`#002147`) 2D Canvas live replay, glide/control plots, 8-phase propellant & 700 km Stage 2 orbit breakdown, 50 Hz 3D replay JSON export, and optional H.264 MP4 rendering.

---

## 3. Stage 2 & Stage 3 Plan (Dedicated 3D Repository)

This repository (`DM100-2D-Rocket-Recovery`) freezes and maintains the verified 2D orbital RTLS baseline, while Stage 2 (3D MuJoCo 6DOF) and Stage 3 (Unreal Engine 5) will launch in a dedicated **new 3D repository**.

| Stage | Core Components | 2D → 3D Mapping |
| --- | --- | --- |
| **Stage 2: 3D MuJoCo 6DOF** (New 3D Repo) | MJCF Stage 1 body + 4 independent grid fins + 4 landing-leg struts, `float64` RK4, `xfrc_applied` force callback | `theta → quaternion`, 1D fin → 4 fins (roll/pitch/yaw), 1-axis gimbal → 2-axis TVC gimbal, 2D LP → 3D SOCP, 3D crosswind & cross-range (Y-axis) |
| **Stage 3: UE5 Cinematics** (New 3D Repo) | 50 Hz replay JSON → UE5 Python Level Sequence auto-import (`ENU m → UE5 cm`), Niagara Methalox plume/RCS/reentry heating, Movie Render Queue | Offline visualization pipeline completely decoupled from physics verification |
