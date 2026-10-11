"""Bounded synthetic instrumentation and causal estimation, not hardware calibration."""

from dataclasses import dataclass
from math import atan2, cos, sin

import numpy as np


@dataclass(frozen=True, slots=True)
class SensorSettings:
    attitude_error_deg: float = 0.0
    gyro_error_deg_s: float = 0.0
    attitude_bias_deg: float = 0.0
    gyro_bias_deg_s: float = 0.0
    dropout_period_steps: int = 0
    dropout_length_steps: int = 0

    def __post_init__(self):
        for name, limit in (
            ("attitude_error_deg", 0.1),
            ("gyro_error_deg_s", 0.1),
            ("attitude_bias_deg", 0.05),
            ("gyro_bias_deg_s", 0.05),
        ):
            value = getattr(self, name)
            lower = -limit if "bias" in name else 0.0
            if isinstance(value, bool) or not np.isfinite(value) or not lower <= value <= limit:
                raise ValueError(f"{name} is outside the declared synthetic sensor range")
        period, length = self.dropout_period_steps, self.dropout_length_steps
        if type(period) is not int or type(length) is not int:
            raise ValueError("dropout periods and lengths must be integers")
        if not ((period == length == 0) or (20 <= period <= 1000 and 1 <= length <= 2)):
            raise ValueError("dropout must be disabled or 1-2 samples per 20-1000 control samples")

    @property
    def angular_noise_enabled(self):
        return any(
            (
                self.attitude_error_deg,
                self.gyro_error_deg_s,
                self.attitude_bias_deg,
                self.gyro_bias_deg_s,
            )
        )


@dataclass(frozen=True, slots=True)
class ControllerOptions:
    translation_bias_compensation: bool = True
    torque_bias_compensation: bool = True
    boost_replanning: bool = True
    attitude_filter: bool = True

    def __post_init__(self):
        if any(type(getattr(self, name)) is not bool for name in self.__dataclass_fields__):
            raise ValueError("controller switches must be booleans")


class PlanarAttitudeEstimator:
    """Estimate angle, rate and gyro bias at received sample timestamps.

    Circular angle innovations and a Joseph covariance update preserve attitude
    continuity and positive covariance. Bias comes only from received angle/rate
    measurements. Process noise is conceptual. Bounded uniform measurement
    second moments do not guarantee bounded filtered errors.
    """

    def __init__(self, attitude_error_deg, gyro_error_deg_s):
        radii = np.deg2rad([attitude_error_deg, gyro_error_deg_s])
        if not np.isfinite(radii).all() or (radii < 0).any():
            raise ValueError("measurement radii must be finite and nonnegative")
        self.measurement_covariance = np.diag(np.maximum(radii**2 / 3, 1e-16))
        self.covariance = np.diag(
            [max(radii[0] ** 2, 1e-12), max(radii[1] ** 2, 1e-12), np.deg2rad(0.05) ** 2]
        )
        self.estimate = None
        self.measurement_time_s = None
        self.angular_acceleration_intensity = np.deg2rad(4.0) ** 2
        self.gyro_bias_walk_intensity = np.deg2rad(0.002) ** 2

    def observe(self, raw, measurement_time_s):
        result = np.asarray(raw, dtype=float).copy()
        if result.ndim != 1 or len(result) < 6 or not np.isfinite(result).all():
            raise ValueError("attitude input must be a finite state vector")
        if not np.isfinite(measurement_time_s):
            raise ValueError("measurement timestamp must be finite")
        if self.estimate is None:
            self.estimate = np.array([result[4], result[5], 0.0])
            self.measurement_time_s = float(measurement_time_s)
        elif measurement_time_s > self.measurement_time_s + 1e-12:
            dt = measurement_time_s - self.measurement_time_s
            transition = np.array([[1, dt, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
            process = self.angular_acceleration_intensity * np.array(
                [[dt**3 / 3, dt**2 / 2, 0], [dt**2 / 2, dt, 0], [0, 0, 0]]
            )
            process[2, 2] = self.gyro_bias_walk_intensity * dt
            predicted = transition @ self.estimate
            covariance = transition @ self.covariance @ transition.T + process
            measurement = np.array([[1, 0, 0], [0, 1, 1]], dtype=float)
            innovation = result[4:6] - measurement @ predicted
            innovation[0] = atan2(sin(innovation[0]), cos(innovation[0]))
            innovation_covariance = (
                measurement @ covariance @ measurement.T + self.measurement_covariance
            )
            gain = np.linalg.solve(innovation_covariance, measurement @ covariance).T
            self.estimate = predicted + gain @ innovation
            residual = np.eye(3) - gain @ measurement
            self.covariance = (
                residual @ covariance @ residual.T + gain @ self.measurement_covariance @ gain.T
            )
            self.covariance = (self.covariance + self.covariance.T) / 2
            self.measurement_time_s = float(measurement_time_s)
        result[4:6] = self.estimate[:2]
        return result
