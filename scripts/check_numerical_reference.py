"""Compare held-control coupled plant integration with SciPy DOP853.

This checks numerical implementation, not the correctness of assumed physics.
There are no guidance updates, staging events or touchdown events in these short
smooth segments; the separate full-flight study exercises the hybrid system.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from powered_landing_guidance.apr_recovery import STATE_NAMES, AprConditions, AprFlight
from powered_landing_guidance.integrators import rk4_step


def reference_checks():
    rows = []
    scales = np.array(
        [1000, 1000, 100, 100, 1, 1, 10000, 100, 1, 1, 1, 1, 1, 100, 1, 100, 100, 1, 1]
    )
    for name, altitude, velocity, throttle in (
        ("return-high", 15000, (80, -250), 0.2),
        ("entry", 5000, (15, -100), 0.5),
        ("terminal", 100, (0.1, -3), 0.3),
    ):
        f = AprFlight(fin_mode="stowed", conditions=AprConditions())
        f.upper_attached = False
        f.on_pad = False
        f.phase = "entry"
        initial = f.state.copy()
        initial[1] = altitude
        initial[2:4] = velocity
        initial[4] = 0.015
        initial[5] = 0.001
        initial[6] = 12000
        initial[12] = throttle
        control = (throttle, 0.002, 100.0, 0.0)

        def rhs(t, y, flight=f, command=control):
            return flight.derivative(y, command, time_s=t, available_propellant=(True, True))

        reference = solve_ivp(
            rhs, (0, 1), initial, method="DOP853", rtol=1e-11, atol=1e-11, max_step=0.01
        )
        if not reference.success:
            raise RuntimeError(reference.message)
        errors = {}
        for dt in (0.01, 0.005, 0.0025):
            state = initial.copy()
            for index in range(round(1 / dt)):
                state = rk4_step(rhs, index * dt, state, dt)
            absolute = abs(state - reference.y[:, -1])
            errors[str(dt)] = {
                "normalized_error_linf": float(np.max(absolute / scales)),
                "absolute_state_errors": dict(zip(STATE_NAMES, absolute.tolist(), strict=True)),
            }
        rows.append(
            {
                "case": name,
                "duration_s": 1.0,
                "reference_solver": "DOP853",
                "reference_rtol": 1e-11,
                "reference_atol": 1e-11,
                "normalization_scales": dict(zip(STATE_NAMES, scales.tolist(), strict=True)),
                "errors": errors,
                "declared_normalized_tolerance": 1e-7,
                "passed": all(row["normalized_error_linf"] <= 1e-7 for row in errors.values()),
            }
        )
    root = Path(__file__).resolve().parents[1]
    names = (
        "scripts/check_numerical_reference.py",
        "src/powered_landing_guidance/apr_recovery.py",
        "src/powered_landing_guidance/recovery_physics.py",
        "src/powered_landing_guidance/integrators.py",
        "src/powered_landing_guidance/vehicle_systems.py",
    )
    return {
        "scope": "Short smooth held-control segments with fins stowed; not real-physics validation",
        "source_hashes": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names
        },
        "checks": rows,
        "passed": all(row["passed"] for row in rows),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = reference_checks()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"passed": result["passed"], "segments": len(result["checks"])}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
