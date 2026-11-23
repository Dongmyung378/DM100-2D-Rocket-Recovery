# DM100 Validation Protocol & Reproduction

[한국어](validation.ko.md) | [Model Spec](spec.md) | [Validation Results](results.md) | [README](../README.md)

This document defines the **487 case validation protocol** and exact reproduction commands for the DM100 100 t Methalox 700 km orbital launch and first stage RTLS simulator.

---

## 1. Strict Causality & Boundary Isolation

- `RtlsController` receives only noisy, biased, delayed, dropout affected `TelemetryPacket` frames plus forecast atmosphere/wind profiles.
- Hidden true plant parameters (`density_scale`, `drag_coefficient_scale`, `thrust_scale`, `initial_mass_scale`, `inertia_scale`, true wind gusts, true IMU biases) exist solely inside `Plant2D`.
- Every compensation term (low pass filtered navigation acceleration bias `bias_x/z`, online glide drag scale estimation, disturbance torque bias, gyro bias) is strictly **causal** using past measurements only.

---

## 2. 487 Case Evaluation Suite

| Split | Cases | Input Source & Purpose |
|---|---:|---|
| `edge-cases` | **32** | Built into [evaluate_validation.py](../scripts/evaluate_validation.py): max east/west wind, extreme shear, 5 m/s ascent/reentry/glide/touchdown gusts, Max q & heavy / low thrust corners, polar (zero Earth rotation boost) & equatorial Coriolis, max navigation/IMU bias & 2/20 packet dropouts, plus **100% simultaneous worst case corners (`edge-23`~`edge-32`)** |
| `development` | **16** | `configs/validation-study-v1.json` development split |
| `holdout` | **96** | `configs/validation-study-v1.json` combined disturbance holdout |
| `sensor-holdout` | **57** | `configs/validation-study-v1.json` extended IMU noise/bias/dropout holdout |
| `regression-*` | **286** | `configs/apr-regression-v1.json` (`development` 205 + `previous-holdout` 72 + `supplemental` 9) |
| **Total** | **487** | **Simultaneous Stage 2 700 km orbit, flight safety (q, heat flux, loads, accel), first touchdown, propellant conservation & 5 s upright standing (all 16 limits certified ≤ 79.3%)** |

---

## 3. Reproduction Commands

```bash
# Single flight & 50 Hz 3D replay JSON export (ENU + unit quaternion)
python scripts/run_flight.py --scenario calm --export-replay artifacts/calm_replay_50hz.json
python scripts/run_flight.py --scenario example-weather --export-replay artifacts/weather_replay_50hz.json

# Fast 32 case extreme edge check
python scripts/evaluate_validation.py --suite edge-cases --workers 12

# Full 487 case parallel certification
python scripts/evaluate_validation.py --suite full-487 --workers 14 --out-json artifacts/validation-487.json

# Automated unit/integration tests and linting
python -m pytest -q -p no:cacheprovider
python -m ruff check --no-cache .
python -m ruff format --check --no-cache .
```

> **Interpretation limit**: Passing 487/487 synthetic cases within ≤ 80% of all limits verifies the software model inside its declared 2D envelope and does not constitute real vehicle flight qualification.
