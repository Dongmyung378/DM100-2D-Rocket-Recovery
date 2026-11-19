# DM100 Validation Design & Reproduction

[한국어](validation.ko.md) | [Model specification](spec.md) | [Recorded results](results.md) | [README](../README.md)

This document describes the **487-case full-envelope validation design** and
reproduction commands for the DM100 2D SpaceX-style RTLS simulator.

---

## 1. Causality & Plant/GNC Isolation

- `SpaceXRtlsController` receives only delayed, noisy, dropout-affected `TelemetryPacket` measurements together with forecast atmosphere and forecast wind.
- Hidden plant perturbations (`density_scale`, `drag_coefficient_scale`, `thrust_scale`, `initial_mass_scale`, `inertia_scale`, true gusts, and true IMU biases) remain strictly encapsulated inside `Plant2D`.
- All adaptive compensations (navigation acceleration bias estimation, causal online glide drag-scale estimation, torque-bias estimation, and gyro-bias estimation) operate strictly causally from past telemetry.

---

## 2. 487-Case Evaluation Suite

| Suite | Cases | Source & Purpose |
|---|---:|---|
| Extreme edge cases (`edge-cases`) | **32** | Built into [evaluate_validation.py](../scripts/evaluate_validation.py): max East/West wind, extreme shear, 5 m/s ascent/entry/glide/landing gusts, Max-Q & heavy low-thrust corners, polar/equatorial Coriolis, max nav/IMU biases & 2/20 dropouts, plus **100% simultaneous limit-corner nightmares (`edge-23 ~ edge-32`)** |
| Development (`development`) | **16** | `configs/validation-study-v1.json` development split |
| Coupled holdout (`holdout`) | **96** | `configs/validation-study-v1.json` coupled-disturbance holdout |
| Extended-sensor holdout (`sensor-holdout`) | **57** | `configs/validation-study-v1.json` IMU noise/bias/dropout holdout |
| Operating regression (`regression-*`) | **286** | `configs/apr-regression-v1.json` (`development` 205 + `previous-holdout` 72 + `supplemental` 9) |
| **Total** | **487** | **Joint mission, flight safety, first touchdown, conservation, and 5 s engine-off upright standing acceptance** |

---

## 3. Reproduction Commands

```bash
# Single flight with 50 Hz ENU+quaternion 3D replay export
python scripts/run_flight.py --scenario calm --export-replay artifacts/calm_replay_50hz.json
python scripts/run_flight.py --scenario example-weather --export-replay artifacts/weather_replay_50hz.json

# 32 extreme adversarial edge cases
python scripts/evaluate_validation.py --suite edge-cases --workers 12

# Full 487-case validation campaign
python scripts/evaluate_validation.py --suite full-487 --workers 14 --out-json artifacts/validation-487.json

# Software, physics, guidance, and local UI checks
python -m pytest -q -p no:cacheprovider
python -m ruff check --no-cache .
python -m ruff format --check --no-cache .
```

> **Scope note**: Passing 487 synthetic 2D cases verifies software robustness across the declared operating envelope; it does not certify real-vehicle flight safety.
