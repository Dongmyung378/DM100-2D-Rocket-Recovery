"""Fixed-step numerical integrators used by the simulator."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

type FloatVector = NDArray[np.float64]
type DerivativeFunction = Callable[[float, FloatVector], ArrayLike]


def _vector(values: ArrayLike, label: str) -> FloatVector:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional, got shape {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} must contain only finite values")
    return vector


def _step_size(dt_s: float) -> float:
    if not np.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("dt_s must be a positive finite number")
    return float(dt_s)


def _evaluate(rhs: DerivativeFunction, time_s: float, state: FloatVector) -> FloatVector:
    derivative = _vector(rhs(time_s, state), "derivative")
    if derivative.shape != state.shape:
        raise ValueError(
            f"derivative shape {derivative.shape} does not match state shape {state.shape}"
        )
    return derivative


def rk4_step(
    rhs: DerivativeFunction,
    time_s: float,
    state: ArrayLike,
    dt_s: float,
) -> FloatVector:
    """Advance one classical fourth-order Runge-Kutta step."""
    current = _vector(state, "state")
    step = _step_size(dt_s)

    k1 = _evaluate(rhs, time_s, current)
    k2 = _evaluate(rhs, time_s + 0.5 * step, current + 0.5 * step * k1)
    k3 = _evaluate(rhs, time_s + 0.5 * step, current + 0.5 * step * k2)
    k4 = _evaluate(rhs, time_s + step, current + step * k3)
    return current + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
