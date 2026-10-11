"""Independent synthetic motion checks; not full-flight performance evidence."""

from dataclasses import replace

import numpy as np
import pytest

from powered_landing_guidance.apr_recovery import AprFlight
from powered_landing_guidance.observation_models import PlanarAttitudeEstimator, SensorSettings


def test_gyro_bias_is_identified_without_access_to_true_state_or_configured_bias():
    rng = np.random.default_rng(469101)
    estimator = PlanarAttitudeEstimator(0.03, 0.015)
    raw_errors, filtered_errors = [], []
    bias = np.deg2rad(0.025)
    for t in np.arange(0, 40, 0.02):
        truth = np.zeros(19)
        truth[4] = 0.04 * t + 0.003 * np.sin(0.5 * t)
        truth[5] = 0.04 + 0.0015 * np.cos(0.5 * t)
        raw = truth.copy()
        raw[4] += np.deg2rad(rng.uniform(-0.03, 0.03))
        raw[5] += bias + np.deg2rad(rng.uniform(-0.015, 0.015))
        output = estimator.observe(raw, t)
        if t > 20:
            raw_errors.append(raw[4:6] - truth[4:6])
            filtered_errors.append(output[4:6] - truth[4:6])
    rms_raw = np.sqrt(np.mean(np.square(raw_errors), axis=0))
    rms_filtered = np.sqrt(np.mean(np.square(filtered_errors), axis=0))
    assert np.all(rms_filtered < rms_raw)
    assert estimator.estimate[2] == pytest.approx(bias, abs=np.deg2rad(0.008))
    assert np.linalg.eigvalsh(estimator.covariance).min() >= -1e-14


def test_repeated_or_old_measurement_is_not_assimilated_again():
    estimator = PlanarAttitudeEstimator(0.03, 0.015)
    raw = np.zeros(19)
    estimator.observe(raw, 0)
    raw[4] = 0.01
    estimator.observe(raw, 0.02)
    estimate, covariance = estimator.estimate.copy(), estimator.covariance.copy()
    raw[4] = 100
    estimator.observe(raw, 0.02)
    estimator.observe(raw, 0)
    assert np.array_equal(estimate, estimator.estimate)
    assert np.array_equal(covariance, estimator.covariance)


def test_wrapped_angle_measurements_preserve_continuous_rotation():
    estimator = PlanarAttitudeEstimator(0.02, 0.01)
    raw = np.zeros(19)
    raw[5] = 0.1
    for t in np.arange(0, 70, 0.02):
        angle = 0.1 * t
        raw[4] = np.arctan2(np.sin(angle), np.cos(angle))
        output = estimator.observe(raw, t)
        assert output[4] == pytest.approx(angle, abs=0.002)


def test_dropped_packet_retains_timestamp_and_does_not_read_current_plant_state():
    flight = AprFlight(
        sensor_settings=SensorSettings(dropout_period_steps=20, dropout_length_steps=2)
    )
    for index in range(20):
        flight.time_s = index * 0.02
        flight.observe_control_state()
    timestamp = flight.observation_times[-1]
    packet = flight.observations[-1].copy()
    earlier_drops = flight.sensor_dropout_count
    flight.state[4] = 0.2
    for index in (20, 21):
        flight.time_s = index * 0.02
        flight.observe_control_state()
        assert flight.observation_times[-1] == timestamp
        assert np.array_equal(flight.observations[-1], packet)
    assert flight.sensor_dropout_count - earlier_drops == 2
    flight.time_s = 0.44
    flight.observe_control_state()
    assert flight.observation_times[-1] == pytest.approx(0.44)
    assert flight.observations[-1][4] == pytest.approx(0.2)


@pytest.mark.parametrize(
    "changes",
    [
        {"attitude_error_deg": True},
        {"gyro_bias_deg_s": np.inf},
        {"dropout_period_steps": 20},
        {"dropout_period_steps": True, "dropout_length_steps": 1},
        {"dropout_period_steps": 19, "dropout_length_steps": 1},
    ],
)
def test_sensor_contract_rejects_unsupported_settings(changes):
    with pytest.raises(ValueError):
        replace(SensorSettings(), **changes)
