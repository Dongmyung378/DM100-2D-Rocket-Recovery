"""Navigation unit checks; these are not full-flight success evidence."""

from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from powered_landing_guidance.apr_recovery import (
    FINS,
    SAFETY,
    AprConditions,
    AprFlight,
    RecoveryDisturbances,
    load_apr_operating,
)
from powered_landing_guidance.recovery_physics import G0, PlanarNavigationEstimator, gravity


def disk_error(rng, radius):
    angle = rng.uniform(-np.pi, np.pi)
    amplitude = radius * np.sqrt(rng.random())
    return amplitude * np.array((np.cos(angle), np.sin(angle)))


def state_at(time_s, acceleration):
    state = np.arange(19, dtype=float) / 100
    state[:2] = np.array((30.0, 3000.0)) + np.array((5.0, -20.0)) * time_s
    state[:2] += np.asarray(acceleration) * time_s**2 / 2
    state[2:4] = np.array((5.0, -20.0)) + np.asarray(acceleration) * time_s
    return state


def test_disk_component_variance_and_hard_raw_radius():
    rng = np.random.default_rng(378)
    angles = rng.uniform(-np.pi, np.pi, 100_000)
    amplitude = 2.0 * np.sqrt(rng.random(100_000))
    samples = np.column_stack((np.cos(angles), np.sin(angles))) * amplitude[:, None]
    estimator = PlanarNavigationEstimator(2.0, 1.0)
    assert np.max(np.linalg.norm(samples, axis=1)) <= 2.0
    assert np.var(samples, axis=0) == pytest.approx([1.0, 1.0], rel=0.015)
    assert np.diag(estimator.measurement_covariance) == pytest.approx([1.0, 0.25])


def test_zero_noise_is_exact_identity_without_delay_and_does_not_mutate_input():
    estimator = PlanarNavigationEstimator(0, 0)
    acceleration = np.array((0.9, -0.3))
    for time_s in np.arange(0, 1, 0.02):
        raw = state_at(time_s, acceleration)
        preserved = raw.copy()
        output = estimator.observe(raw, time_s, acceleration)
        assert np.array_equal(output, raw)
        assert np.array_equal(raw, preserved)
        assert not np.shares_memory(output, raw)


def test_constant_acceleration_reduces_position_and_velocity_noise():
    estimator = PlanarNavigationEstimator(2, 1)
    rng = np.random.default_rng(378)
    acceleration = np.array((0.7, -0.2))
    raw_errors, filtered_errors = [], []
    for time_s in np.arange(0, 20, 0.02):
        truth = state_at(time_s, acceleration)
        raw = truth.copy()
        raw[:2] += disk_error(rng, 2)
        raw[2:4] += disk_error(rng, 1)
        output = estimator.observe(raw, time_s, acceleration)
        assert np.array_equal(output[4:], raw[4:])
        if time_s > 3:
            raw_errors.append(raw[:4] - truth[:4])
            filtered_errors.append(output[:4] - truth[:4])
    raw_rms = np.sqrt(np.mean(np.square(raw_errors), axis=0))
    filtered_rms = np.sqrt(np.mean(np.square(filtered_errors), axis=0))
    assert np.all(filtered_rms < 0.5 * raw_rms)


def test_acceleration_bias_is_learned_from_measurements_not_hidden_state():
    estimator = PlanarNavigationEstimator(2, 1)
    rng = np.random.default_rng(378)
    actual = np.array((1.2, -0.7))
    nominal = np.array((0.3, -0.4))
    biases = []
    for time_s in np.arange(0, 40, 0.02):
        raw = state_at(time_s, actual)
        raw[:2] += disk_error(rng, 2)
        raw[2:4] += disk_error(rng, 1)
        estimator.observe(raw, time_s, nominal)
        if time_s > 30:
            biases.append(estimator.estimate[:, 2].copy())
    assert np.mean(biases, axis=0) == pytest.approx(actual - nominal, abs=0.05)


def test_delayed_constant_acceleration_is_extrapolated_once():
    estimator = PlanarNavigationEstimator(0, 0)
    acceleration = np.array((0.7, -0.2))
    for time_s in np.arange(0, 2, 0.02):
        sample_time = max(0, time_s - 0.06)
        raw = state_at(sample_time, acceleration)
        output = estimator.observe(raw, time_s, acceleration, time_s - sample_time)
        assert output[:4] == pytest.approx(state_at(time_s, acceleration)[:4], abs=1e-10)
        assert np.array_equal(output[4:], raw[4:])
        assert estimator.measurement_time_s == pytest.approx(sample_time)


def test_repeated_measurement_does_not_gain_spurious_confidence():
    estimator = PlanarNavigationEstimator(2, 1)
    raw = state_at(0, (0, 0))
    estimator.observe(raw, 0, (0, 0))
    initial_covariance = estimator.covariance.copy()
    initial_estimate = estimator.estimate.copy()
    for time_s in (0.02, 0.04, 0.06):
        output = estimator.observe(raw, time_s, (0, 0), delay_s=time_s)
        assert np.array_equal(estimator.covariance, initial_covariance)
        assert np.array_equal(estimator.estimate, initial_estimate)
        assert output[:2] == pytest.approx(raw[:2] + raw[2:4] * time_s)


def test_out_of_order_measurement_is_not_reassimilated():
    estimator = PlanarNavigationEstimator(2, 1)
    estimator.observe(state_at(1, (0, 0)), 1, (0, 0))
    prior = estimator.estimate.copy()
    output = estimator.observe(state_at(0.8, (0, 0)), 1.02, (0, 0), delay_s=0.22)
    assert np.array_equal(estimator.estimate, prior)
    assert output[:4] == pytest.approx(state_at(1.02, (0, 0))[:4])


def test_covariance_remains_finite_symmetric_positive_semidefinite():
    estimator = PlanarNavigationEstimator(2, 1)
    rng = np.random.default_rng(378)
    for step in range(500):
        time_s = step * 0.02
        delay = min(time_s, 0.06)
        raw = state_at(time_s - delay, (0.4, -0.2))
        raw[:2] += disk_error(rng, 2)
        raw[2:4] += disk_error(rng, 1)
        output = estimator.observe(raw, time_s, (0.3, -0.1), delay)
        assert np.all(np.isfinite(output))
        for covariance in (estimator.covariance, estimator.current_covariance):
            assert np.all(np.isfinite(covariance))
            assert np.allclose(covariance, covariance.T, atol=1e-12)
            assert np.min(np.linalg.eigvalsh(covariance)) >= -1e-12


def test_estimator_is_causal_and_does_not_read_unmeasured_fluid_state():
    first, second = PlanarNavigationEstimator(2, 1), PlanarNavigationEstimator(2, 1)
    rng = np.random.default_rng(378)
    prefix = []
    for time_s in np.arange(0, 1, 0.02):
        raw = state_at(time_s, (0.4, -0.2))
        raw[:2] += disk_error(rng, 2)
        raw[2:4] += disk_error(rng, 1)
        changed = raw.copy()
        changed[6:] += 1000
        expected = first.observe(raw, time_s, (0.3, -0.1))
        actual = second.observe(changed, time_s, (0.3, -0.1))
        assert np.array_equal(expected[:4], actual[:4])
        prefix.append(expected.copy())
    first.observe(state_at(1.0, (20, -20)), 1.0, (0.3, -0.1))
    assert not np.shares_memory(prefix[-1], first.estimate)


@pytest.mark.parametrize("radii", [(-1, 0), (0, -1), (np.nan, 1), (1, np.inf)])
def test_invalid_noise_radii_are_rejected(radii):
    with pytest.raises(ValueError, match="radii"):
        PlanarNavigationEstimator(*radii)


@pytest.mark.parametrize("bad_delay", [-0.02, np.nan, np.inf])
def test_invalid_delay_is_rejected(bad_delay):
    with pytest.raises(ValueError, match="delay"):
        PlanarNavigationEstimator(2, 1).observe(np.zeros(19), 0, (0, 0), bad_delay)


def test_nonfinite_or_wrong_shape_input_and_backwards_time_are_rejected():
    estimator = PlanarNavigationEstimator(2, 1)
    with pytest.raises(ValueError, match="state"):
        estimator.observe(np.zeros((19, 1)), 0, (0, 0))
    with pytest.raises(ValueError, match="state"):
        estimator.observe(np.full(19, np.nan), 0, (0, 0))
    with pytest.raises(ValueError, match="acceleration"):
        estimator.observe(np.zeros(19), 0, (np.nan, 0))
    estimator.observe(np.zeros(19), 1, (0, 0))
    with pytest.raises(ValueError, match="backwards"):
        estimator.observe(np.zeros(19), 0.9, (0, 0))


def test_flight_observation_queue_uses_actual_startup_age_and_extrapolates_attitude(monkeypatch):
    flight = AprFlight(disturbances=RecoveryDisturbances(observation_delay_steps=2))
    acceleration = np.array((0.7, -0.2))
    monkeypatch.setattr(flight, "navigation_acceleration", lambda *args: acceleration)
    initial = flight.state.copy()
    for step in range(10):
        flight.time_s = step * 0.02
        kinematic = state_at(flight.time_s, acceleration)
        flight.state[:4] = kinematic[:4]
        flight.state[4:6] = (0.3 + 0.1 * flight.time_s, 0.1)
        observed = flight.observe_control_state()
        expected_sample_time = max(0.0, flight.time_s - 0.04)
        assert flight.observation_times[0] == pytest.approx(expected_sample_time)
        assert flight.navigation_estimator.measurement_time_s == pytest.approx(expected_sample_time)
        assert observed[:4] == pytest.approx(kinematic[:4], abs=1e-10)
        assert observed[4:6] == pytest.approx(flight.state[4:6], abs=1e-12)
        assert np.array_equal(observed[6:11], initial[6:11])


def test_flight_raw_measurement_contract_is_preserved_after_filter_integration():
    flight = AprFlight(
        disturbances=RecoveryDisturbances(
            sensor_position_error_m=2, sensor_velocity_error_m_s=1, observation_delay_steps=2
        )
    )
    preserved = flight.state.copy()
    for step in range(100):
        flight.time_s = step * 0.02
        observed = flight.observe_control_state()
        raw = flight.observations[-1]
        assert np.linalg.norm(raw[:2] - preserved[:2]) <= 2.0
        assert np.linalg.norm(raw[2:4] - preserved[2:4]) <= 1.0
        assert np.array_equal(flight.state, preserved)
        assert np.all(np.isfinite(observed))
    assert flight.navigation_estimator.time_s == flight.time_s


def test_nominal_navigation_prediction_does_not_use_hidden_uncertainties_or_slosh():
    forecast = ((0.0, 0.0), (100_000.0, 0.0))
    flights = [
        AprFlight(
            conditions=AprConditions(
                wind_profile=((0.0, sign), (100_000.0, sign)),
                forecast_wind_profile=forecast,
                density_scale=1 + 0.1 * sign,
            ),
            disturbances=RecoveryDisturbances(
                initial_mass_scale=1 + 0.01 * sign,
                thrust_scale=1 + 0.03 * sign,
                drag_coefficient_scale=1 + 0.15 * sign,
                inertia_scale=1 + 0.03 * sign,
                engine_lag_s=0.05 if sign > 0 else 0.04,
            ),
        )
        for sign in (-1, 1)
    ]
    measured = flights[0].state.copy()
    measured[:6] = (500, 2000, 15, -240, 0.1, 0.02)
    measured[9:13] = (0.1, 1, 0.01, 0.3)
    for flight in flights:
        flight.on_pad = flight.upper_attached = False
    first = flights[0].navigation_acceleration(measured, 100)
    measured[17:19] = (123, -321)
    second = flights[1].navigation_acceleration(measured, 100)
    assert first == pytest.approx(second, abs=1e-12)


def test_delayed_actuator_telemetry_uses_known_interval_commands_not_current_truth():
    flight = AprFlight(disturbances=RecoveryDisturbances(observation_delay_steps=2))
    flight.observe_control_state()
    flight.time_s = 0.02
    flight.control = (1.0, np.deg2rad(6.0), 0.0, 0.0)
    flight.state[11:13] = (0.123, 0.987)  # These new readings are still in the delay queue.
    first = flight.observe_control_state()
    assert first[12] == pytest.approx(1 - np.exp(-0.02 / 0.04))
    assert first[11] == pytest.approx(np.deg2rad(60) * 0.02)
    flight.time_s = 0.04
    flight.control = (0.0, -np.deg2rad(6.0), 0.0, 0.0)
    second = flight.observe_control_state()
    assert second[12] == pytest.approx((1 - np.exp(-0.02 / 0.04)) * np.exp(-0.02 / 0.04))
    assert second[11] == pytest.approx(0.0, abs=1e-12)
    assert second[11:13] != pytest.approx(flight.state[11:13])


def test_flight_perfect_current_observation_matches_raw_except_unmeasured_fluid_mode():
    flight = AprFlight()
    flight.state[17:19] = (0.001, 0.002)
    expected = flight.state.copy()
    expected[17:19] = 0
    assert np.array_equal(flight.observe_control_state(), expected)


def delayed_allocation_case(monkeypatch, *, throttle=0.019673467, ready_offset=-1.0):
    flight = AprFlight(disturbances=RecoveryDisturbances(action_delay_steps=1))
    flight.upper_attached = flight.on_pad = False
    flight.time_s = 300.0
    measured = flight.state.copy()
    measured[:6] = (4100, 36000, -180, -970, 0, 0)
    measured[6:8] = (15844, 468)
    measured[11:13] = (-0.102, throttle)
    pending = (0.05, -np.deg2rad(6), 12345.0, 0.0)
    flight.action_queue.clear()
    flight.action_queue.extend([(0.7, 0.08, 999.0, 0.0), pending])
    flight.ignition_requested = True
    flight.ignition_ready_s = flight.time_s + ready_offset
    monkeypatch.setattr(
        flight, "body_loads", lambda *args, **kwargs: (np.zeros(2), 0.0, 1000.0, 0.0)
    )
    monkeypatch.setattr(flight, "fin_loads", lambda *args, **kwargs: (np.zeros(2), 0.0, 0.0))
    monkeypatch.setattr(flight, "fin_command", lambda *args: 0.0)
    return flight, measured, pending


def pending_tvc_moment_samples(flight, measured, pending, new_throttle):
    dt = flight.return_settings.control_dt_s
    at_application = pending[1] + (measured[11] - pending[1]) * np.exp(-dt / 0.06)
    props = flight.properties(measured, predicted=True)
    lever = props.cg_m - flight.physics.geometry.nozzle_station_m
    pending_level = pending[0] if flight.time_s >= flight.ignition_ready_s else 0.0
    effective_level = new_throttle if flight.time_s + dt >= flight.ignition_ready_s - 1e-9 else 0.0
    samples = []
    for tau in (0.04, 0.05):
        application_level = pending_level + (measured[12] - pending_level) * np.exp(-dt / tau)
        for elapsed in (0.0, dt / 2.0, dt):
            level = effective_level + (application_level - effective_level) * np.exp(-elapsed / tau)
            angle = at_application + np.deg2rad(60.0) * elapsed
            samples.append(-lever * flight.engine(level, measured[1])[0] * np.sin(angle))
    return np.asarray(samples)


def test_delayed_torque_allocation_predicts_pending_command_then_bounded_held_interval(monkeypatch):
    flight, measured, pending = delayed_allocation_case(monkeypatch)
    queue_before = list(flight.action_queue)
    flags_before = (flight.ignition_requested, flight.ignition_ready_s)
    gimbal, rcs, _ = flight.attitude_control(0, 0.05, measured)
    assert gimbal == 0
    dt = flight.return_settings.control_dt_s
    at_application = pending[1] + (measured[11] - pending[1]) * np.exp(-dt / 0.06)
    # The return toward zero is rate-limited throughout this short interval.
    props = flight.properties(measured, predicted=True)
    lever = props.cg_m - flight.physics.geometry.nozzle_station_m
    samples = pending_tvc_moment_samples(flight, measured, pending, 0.05)
    assert rcs == pytest.approx(-(samples.min() + samples.max()) / 2.0, abs=1e-8)
    assert list(flight.action_queue) == queue_before
    assert (flight.ignition_requested, flight.ignition_ready_s) == flags_before
    # This is a local analytic actuator interval, not a complete flight.
    peak_residual = 0.0
    for tau in (0.04, 0.05):
        application_level = 0.05 + (measured[12] - 0.05) * np.exp(-dt / tau)
        for elapsed in np.linspace(0, dt, 31):
            level = 0.05 + (application_level - 0.05) * np.exp(-elapsed / tau)
            angle = at_application + np.deg2rad(60) * elapsed
            moment = rcs - lever * flight.engine(level, measured[1])[0] * np.sin(angle)
            peak_residual = max(peak_residual, abs(np.rad2deg(moment / props.inertia_kg_m2)))
    assert peak_residual < 0.5


@pytest.mark.parametrize("ready_offset", [0.02, 0.021])
def test_delayed_allocation_honors_ignition_at_command_application_boundary(
    monkeypatch, ready_offset
):
    flight, measured, pending = delayed_allocation_case(
        monkeypatch, throttle=0.0, ready_offset=ready_offset
    )
    _, rcs, _ = flight.attitude_control(0, 0.05, measured)
    dt = flight.return_settings.control_dt_s
    if ready_offset > dt:
        # Readiness halfway through a held command cannot turn that command on.
        assert rcs == 0.0
    else:
        samples = pending_tvc_moment_samples(flight, measured, pending, 0.05)
        expected = -(samples.min() + samples.max()) / 2.0
        assert rcs == pytest.approx(expected, abs=1e-8)


def test_delayed_allocation_ignores_queue_head_that_will_be_discarded(monkeypatch):
    flight, measured, pending = delayed_allocation_case(monkeypatch)
    before = flight.attitude_control(0, 0.05, measured)
    flight.action_queue.clear()
    flight.action_queue.extend([(0.0, -0.1, -100000.0, 0.0), pending])
    after = flight.attitude_control(0, 0.05, measured)
    assert after == pytest.approx(before, abs=1e-12)


@pytest.mark.parametrize("height", [-1.0, 0.0, 0.1, 1.0, 2.9, 3.0])
def test_terminal_plateau_has_constant_speed_zero_gradient_and_physical_time(height):
    flight = AprFlight()
    speed, gradient, remaining = flight.descent_speed(height)
    assert speed == flight.design.descent_reference_m_s == 1.0
    assert gradient == 0.0
    assert remaining == pytest.approx(max(height, 0.0))
    assert flight.physics.acceptance.max_vertical_foot_speed_m_s == 1.5


def test_plateau_reference_maps_to_same_braking_curve_and_is_continuous():
    shifted, original = AprFlight(), AprFlight()
    original.return_settings = replace(original.return_settings, terminal_speed_plateau_height_m=0)
    for height in (3.25, 10, 500, 2000, 37000, 100000):
        expected = original.descent_speed(height - 3)
        actual = shifted.descent_speed(height)
        assert actual[:2] == pytest.approx(expected[:2], abs=1e-12)
        assert actual[2] == pytest.approx(expected[2] + 3, abs=1e-12)
    below, above = shifted.descent_speed(3 - 1e-7), shifted.descent_speed(3 + 1e-7)
    assert below[0] == pytest.approx(above[0], abs=1e-6)
    assert below[2] == pytest.approx(above[2], abs=1e-6)


def inferred_force_acceleration(monkeypatch, height, bias, *, vertical_body=True):
    flight = AprFlight()
    measured = flight.state.copy()
    measured[2:6] = 0
    if bias is not None:
        flight.navigation_estimator.estimate = np.zeros((2, 3))
        flight.navigation_estimator.estimate[:, 2] = bias
    captured = {}
    monkeypatch.setattr(flight, "height", lambda *args, **kwargs: height)
    monkeypatch.setattr(flight, "mass", lambda *args, **kwargs: 1000.0)
    monkeypatch.setattr(flight, "properties", lambda *args, **kwargs: SimpleNamespace(cg_m=2))
    monkeypatch.setattr(flight, "loads", lambda *args, **kwargs: (np.zeros(2), 0.0, 0.0, 0.0))
    monkeypatch.setattr(flight, "engine", lambda throttle, altitude: (throttle * 1e6, 0.0))

    def capture(target, throttle, observed, *, thrust_direction=None):
        captured["direction"] = thrust_direction if thrust_direction is not None else target
        return 0.0, 0.0, 0.0

    monkeypatch.setattr(flight, "attitude_control", capture)
    throttle = flight.force_control((0, 0), measured, vertical_body=vertical_body)[0]
    inferred = (
        throttle
        * 1e6
        / 1000
        * np.array((np.sin(captured["direction"]), np.cos(captured["direction"])))
    )
    inferred[1] -= gravity(measured[1])
    return inferred, flight


@pytest.mark.parametrize("height,weight", [(600, 0), (500, 0), (300, 0.5), (100, 1), (0, 1)])
def test_terminal_bias_compensation_is_smooth_and_vector_norm_limited(monkeypatch, height, weight):
    inferred, flight = inferred_force_acceleration(monkeypatch, height, (3, 4))
    assert inferred == pytest.approx(-weight * np.array((1.2, 1.6)), abs=1e-12)
    assert np.linalg.norm(inferred) <= 2 + 1e-12
    assert flight.navigation_estimator.estimate[:, 2] == pytest.approx((3, 4))


def test_terminal_bias_compensation_uses_no_bias_without_estimator_or_outside_vertical_mode(
    monkeypatch,
):
    absent, _ = inferred_force_acceleration(monkeypatch, 20, None)
    nonterminal, _ = inferred_force_acceleration(monkeypatch, 20, (0, -1), vertical_body=False)
    assert absent == pytest.approx((0, 0), abs=1e-12)
    assert nonterminal == pytest.approx((0, 0), abs=1e-12)


def test_terminal_bias_correction_direction_compensates_measured_acceleration_shortfall(
    monkeypatch,
):
    inferred, _ = inferred_force_acceleration(monkeypatch, 20, (0.1, -1.2))
    assert inferred == pytest.approx((-0.1, 1.2), abs=1e-12)


def ascent_alignment_case(monkeypatch, *, altitude=31_000.0, main_propellant=25_000.0):
    flight = AprFlight()
    flight.time_s = 90.0
    measured = flight.state.copy()
    measured[:4] = (3000.0, altitude, 200.0, 1000.0)
    flow = np.arctan2(measured[2], measured[3])
    measured[4:6] = (flow + np.deg2rad(10.0), np.deg2rad(0.4))
    measured[6] = main_propellant
    measured[12] = 0.85
    targets = []
    monkeypatch.setattr(flight, "loads", lambda *args, **kwargs: (np.zeros(2), 0.0, 1000.0, 0.0))

    def capture(target, throttle, observed):
        targets.append(target)
        return 0.0, 0.0, 0.0

    monkeypatch.setattr(flight, "attitude_control", capture)
    return flight, measured, flow, targets


def test_powered_ascent_prepares_flow_alignment_before_separation_altitude(monkeypatch):
    flight, measured, flow, targets = ascent_alignment_case(monkeypatch)
    control = flight.guidance(measured)
    assert measured[1] < flight.ascent_settings.separation_altitude_m
    assert flight.separation_alignment_started_s == flight.time_s
    assert control[0] > 0.0
    assert not flight.cutoff_pending
    initial_pitch = targets[-1]
    flight.time_s += flight.ascent_settings.separation_alignment_blend_s / 2
    assert flight.guidance(measured)[0] > 0.0
    assert targets[-1] == pytest.approx((initial_pitch + flow) / 2)
    flight.time_s += flight.ascent_settings.separation_alignment_blend_s / 2
    assert flight.guidance(measured)[0] > 0.0
    assert targets[-1] == pytest.approx(flow)
    assert not flight.cutoff_pending


def test_ascent_keeps_pitch_program_when_there_is_still_time_to_align(monkeypatch):
    flight, measured, flow, targets = ascent_alignment_case(monkeypatch, altitude=20_000.0)
    control = flight.guidance(measured)
    assert targets[-1] > flow + np.deg2rad(1.0)
    assert flight.separation_alignment_started_s is None
    assert control[0] > 0.0
    assert not flight.cutoff_pending


def test_ascent_alignment_remains_latched_after_pointing_error_disappears(monkeypatch):
    flight, measured, flow, targets = ascent_alignment_case(monkeypatch)
    flight.guidance(measured)
    started_s = flight.separation_alignment_started_s
    flight.time_s += flight.ascent_settings.separation_alignment_blend_s
    measured[1] = 20_000.0
    measured[4:6] = (flow, 0.0)
    control = flight.guidance(measured)
    assert targets[-1] == pytest.approx(flow)
    assert flight.separation_alignment_started_s == started_s
    assert control[0] > 0.0
    assert not flight.cutoff_pending


@pytest.mark.parametrize("offset,engine_off", [(-1.0, False), (0.0, True), (1.0, True)])
def test_powered_alignment_preserves_the_existing_altitude_cutoff_gate(
    monkeypatch, offset, engine_off
):
    flight, measured, flow, targets = ascent_alignment_case(monkeypatch, altitude=37_000 + offset)
    flight.separation_alignment_started_s = (
        flight.time_s - flight.ascent_settings.separation_alignment_blend_s
    )
    throttle = flight.guidance(measured)[0]
    assert flight.cutoff_pending is engine_off
    assert bool(throttle == 0.0) is engine_off
    assert targets[-1] == pytest.approx(flow)


def test_ascent_prepares_alignment_for_propellant_cutoff_before_altitude_gate(monkeypatch):
    flight, measured, flow, targets = ascent_alignment_case(
        monkeypatch, altitude=20_000.0, main_propellant=21_000.0
    )
    control = flight.guidance(measured)
    assert flight.separation_alignment_started_s == flight.time_s
    flight.time_s += flight.ascent_settings.separation_alignment_blend_s
    control = flight.guidance(measured)
    assert targets[-1] == pytest.approx(flow)
    assert control[0] > 0.0
    assert not flight.cutoff_pending


def entry_force_case(monkeypatch, *, q=20_000.0, body_torque=0.0):
    flight = AprFlight()
    flight.upper_attached = flight.on_pad = False
    flight.physics = replace(
        flight.physics, atmosphere=replace(flight.physics.atmosphere, coriolis_enabled=False)
    )
    measured = flight.state.copy()
    measured[:6] = (100.0, 3_000.0, 0.0, -180.0, 0.0, 0.0)
    measured[6:8] = (15_000.0, 400.0)
    targets = []

    def forecast_loads(*args, **kwargs):
        assert kwargs.get("predicted") is True
        return np.zeros(2), body_torque, q, 0.0

    def capture(target, throttle, observed, *, thrust_direction=None):
        targets.append((target, throttle, thrust_direction))
        return 0.0, 0.0, 0.0

    monkeypatch.setattr(flight, "loads", forecast_loads)
    monkeypatch.setattr(flight, "body_loads", forecast_loads)
    monkeypatch.setattr(flight, "attitude_control", capture)
    return flight, measured, targets


@pytest.mark.parametrize("lateral_acceleration", [-20.0, 20.0])
@pytest.mark.parametrize("q", [5_000.0, 20_000.0])
def test_high_q_entry_cone_defers_lateral_force_but_preserves_vertical_braking(
    monkeypatch, lateral_acceleration, q
):
    flight, measured, targets = entry_force_case(monkeypatch, q=q)
    throttle = flight.force_control((lateral_acceleration, 0.0), measured)[0]
    target, captured_throttle, thrust_direction = targets[-1]
    cone = np.deg2rad(flight.return_settings.entry_attitude_cone_deg)
    assert target == pytest.approx(np.sign(lateral_acceleration) * cone)
    assert thrust_direction is None
    assert captured_throttle == throttle
    thrust = flight.engine(throttle, measured[1])[0]
    requested_vertical_force = flight.mass(measured, predicted=True) * gravity(measured[1])
    assert thrust * np.cos(target) == pytest.approx(requested_vertical_force)
    assert abs(thrust * np.sin(target)) < (
        flight.mass(measured, predicted=True) * abs(lateral_acceleration)
    )


@pytest.mark.parametrize(
    "q,vertical_speed,attached",
    [(4_999.0, -180.0, False), (20_000.0, 180.0, False), (20_000.0, -180.0, True)],
)
def test_entry_cone_does_not_clamp_low_q_ascent_or_attached_vehicle(
    monkeypatch, q, vertical_speed, attached
):
    flight, measured, targets = entry_force_case(monkeypatch, q=q)
    flight.upper_attached = attached
    measured[3] = vertical_speed
    expected = np.arctan2(20.0, gravity(measured[1]))
    measured[4] = expected
    flight.force_control((20.0, 0.0), measured)
    assert targets[-1][0] == pytest.approx(expected)


def test_entry_cone_retains_vertical_body_lateral_tvc_allocation(monkeypatch):
    flight, measured, targets = entry_force_case(monkeypatch, body_torque=500_000.0)
    throttle = flight.force_control((20.0, 0.0), measured, vertical_body=True)[0]
    target, _, direction = targets[-1]
    lever = flight.properties(measured, predicted=True).cg_m
    lever -= flight.physics.geometry.nozzle_station_m
    lateral_force_limit = 0.55 * flight.design.rcs_torque_limit_nm / lever
    vertical_force = flight.mass(measured, predicted=True) * gravity(measured[1])
    assert target == 0.0
    assert direction == pytest.approx(np.arctan2(lateral_force_limit, vertical_force))
    thrust = flight.engine(throttle, measured[1])[0]
    assert thrust * np.sin(direction) == pytest.approx(lateral_force_limit)
    assert thrust * np.cos(direction) == pytest.approx(vertical_force)


def test_entry_misalignment_keeps_thrust_needed_for_forecast_torque_authority(monkeypatch):
    body_torque = 337_185.0
    flight, measured, targets = entry_force_case(monkeypatch, body_torque=body_torque)
    measured[4] = -np.deg2rad(20.0)
    throttle = flight.force_control((20.0, 0.0), measured)[0]
    properties = flight.properties(measured, predicted=True)
    lever = properties.cg_m - flight.physics.geometry.nozzle_station_m
    torque_budget = 1.1 * 1.15 * body_torque + properties.inertia_kg_m2 * np.deg2rad(
        flight.return_settings.attitude_acceleration_limit_deg_s2
    )
    required_tvc = torque_budget - 0.8 * flight.design.rcs_torque_limit_nm
    guaranteed_full_tvc = (
        0.97
        * lever
        * flight.engine(1.0, measured[1])[0]
        * np.sin(np.deg2rad(flight.design.gimbal_limit_deg))
    )
    assert abs(targets[-1][0] - measured[4]) > np.deg2rad(15.0)
    assert throttle > 0.05
    assert throttle == pytest.approx(required_tvc / guaranteed_full_tvc)


def test_failed_high_q_peak_still_violates_physical_angular_acceleration_limit():
    # A saved local plant state proves guidance changes do not hide or clip the
    # earlier physical failure. This is a derivative check, not a flight replay.
    config = Path(__file__).resolve().parents[1] / "configs" / "apr-operating-v1.yaml"
    conditions, disturbances = load_apr_operating(config)
    flight = AprFlight(conditions=conditions, disturbances=disturbances)
    flight.upper_attached = flight.on_pad = False
    flight.time_s = 379.47
    flight.phase = "entry"
    flight.state = np.array(
        [
            1.7850384298640398,
            2629.6398568874397,
            5.813539564513628,
            -197.4546233130782,
            -0.33221411026528025,
            0.006045453930135411,
            8300.386632042919,
            439.9564782727205,
            0.0,
            -0.03868372981479898,
            0.32916666666666605,
            0.10471971928431131,
            0.05000000002797186,
            1499.9999999999943,
            0.9999999999999981,
            47.554600846748045,
            17.44539915325823,
            -0.5482691672551138,
            -0.37342894343262745,
        ]
    )
    angular_acceleration = np.rad2deg(
        flight.derivative(flight.state, (0.05, np.deg2rad(6.0), -100_000.0, 0.0))[5]
    )
    assert angular_acceleration == pytest.approx(18.413879829265156, abs=1e-9)


@pytest.mark.parametrize("initial_mass_scale", [0.99, 1.0, 1.01])
def test_ascent_thrust_cap_respects_three_g_at_allowed_upper_thrust_and_lower_mass(
    monkeypatch, initial_mass_scale
):
    nominal = AprFlight()
    uncertain = AprFlight(
        disturbances=RecoveryDisturbances(initial_mass_scale=initial_mass_scale, thrust_scale=1.03)
    )
    measured = nominal.state.copy()
    measured[:6] = (0.0, 25_000.0, 0.0, 1_000.0, 0.0, 0.0)
    measured[6] = 24_000.0
    controls = []
    for flight in (nominal, uncertain):
        flight.time_s = 80.0
        monkeypatch.setattr(
            flight, "loads", lambda *args, **kwargs: (np.zeros(2), 0.0, 1000.0, 0.0)
        )
        monkeypatch.setattr(flight, "attitude_control", lambda *args, **kwargs: (0.0, 0.0, 0.0))
        controls.append(flight.guidance(measured))
    # Identical sensor/forecast inputs produce identical commands; the bound
    # does not read the actual mass/thrust perturbations used by the plant.
    assert controls[0] == pytest.approx(controls[1])
    actual_thrust = uncertain.engine(controls[1][0], measured[1], actual=True)[0]
    actual_g = actual_thrust / (uncertain.mass(measured) * G0)
    assert actual_g <= 2.95 + 1e-12
    assert controls[1][0] > 0.0


def terminal_feedforward_case(monkeypatch, *, wind, height, speed, gradient, vertical_speed):
    profile = ((0.0, wind), (100_000.0, wind))
    flight = AprFlight(
        conditions=AprConditions(wind_profile=profile, forecast_wind_profile=profile)
    )
    flight.upper_attached = flight.on_pad = False
    flight.phase = "landing"
    flight.entry_started_s = 0.0
    measured = flight.state.copy()
    measured[:6] = (0.0354, height + 10.0, 0.1, vertical_speed, 0.0, 0.0)
    measured[6:8] = (15_000.0, 400.0)
    accelerations = []
    monkeypatch.setattr(flight, "height", lambda *args, **kwargs: height)
    monkeypatch.setattr(flight, "descent_speed", lambda *args: (speed, gradient, 10.0))
    monkeypatch.setattr(flight, "loads", lambda *args, **kwargs: (np.zeros(2), 0.0, 1000.0, 0.0))

    def capture(acceleration, observed, *, vertical_body=False):
        accelerations.append(np.asarray(acceleration))
        return 0.0, 0.0, 0.0, 0.0

    monkeypatch.setattr(flight, "force_control", capture)
    return flight, measured, accelerations


@pytest.mark.parametrize("wind", [-8.0, 8.0])
def test_saturated_terminal_vertical_reference_has_no_lateral_feedforward(monkeypatch, wind):
    flight, measured, accelerations = terminal_feedforward_case(
        monkeypatch, wind=wind, height=1.0, speed=1.0, gradient=0.0, vertical_speed=-1.0
    )
    flight.guidance(measured)
    ax, az = accelerations[-1]
    assert ax != 0.0
    assert abs(flight.air_velocity(measured, predicted=True)[0]) > 1.0
    # The lower-bounded vertical reference is constant here, so differentiating
    # its unclipped radicand would incorrectly couple wind and horizontal action
    # into an additional vertical acceleration command.
    assert az == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("wind", [-8.0, 8.0])
def test_unsaturated_terminal_reference_keeps_its_geometric_feedforward(monkeypatch, wind):
    speed, gradient = 20.0, 0.1
    flight, measured, accelerations = terminal_feedforward_case(
        monkeypatch, wind=wind, height=4_000.0, speed=speed, gradient=gradient, vertical_speed=-20.0
    )
    flight.guidance(measured)
    ax, az = accelerations[-1]
    lateral_speed = flight.air_velocity(measured, predicted=True)[0]
    vertical_reference = np.sqrt(speed**2 - lateral_speed**2)
    assert vertical_reference > flight.design.descent_reference_m_s
    feedforward = (-speed * gradient * measured[3] + lateral_speed * ax) / vertical_reference
    expected = feedforward + 1.2 * (-vertical_reference - measured[3])
    assert az == pytest.approx(expected)


@pytest.mark.parametrize("wind", [-8.0, 8.0])
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_terminal_alignment_acceleration_is_continuous_at_horizontal_speed_threshold(
    monkeypatch, wind, sign
):
    flight, measured, accelerations = terminal_feedforward_case(
        monkeypatch, wind=wind, height=280.0, speed=65.0, gradient=0.12, vertical_speed=-65.0
    )
    measured[0] = 0.0
    commands = []
    for horizontal_speed in (0.499, 0.5, 0.501):
        measured[2] = sign * horizontal_speed
        flight.guidance(measured)
        commands.append(accelerations[-1][1])
    # A 0.002 m/s measurement fluctuation must not switch the desired descent
    # law abruptly. The former hard branch produced an O(10 m/s²) step here.
    assert np.ptp(commands) < 0.01
    assert flight.design.descent_reference_m_s == 1.0
    assert flight.physics.acceptance.max_horizontal_contact_speed_m_s == 0.5
    assert flight.physics.acceptance.max_vertical_foot_speed_m_s == 1.5


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_terminal_alignment_acceleration_is_continuous_at_position_threshold(monkeypatch, sign):
    flight, measured, accelerations = terminal_feedforward_case(
        monkeypatch, wind=-8.0, height=280.0, speed=65.0, gradient=0.12, vertical_speed=-65.0
    )
    commands = []
    for position_error in (1.999, 2.0, 2.001):
        measured[0] = sign * position_error
        flight.guidance(measured)
        commands.append(accelerations[-1][1])
    assert np.ptp(commands) < 0.01
    assert flight.physics.acceptance.max_position_error_m == 2.0


@pytest.mark.parametrize("horizontal_speed", [-1.5, -1.0, 1.0, 1.5])
def test_large_terminal_errors_retain_the_full_alignment_descent_law(monkeypatch, horizontal_speed):
    height = 280.0
    flight, measured, accelerations = terminal_feedforward_case(
        monkeypatch, wind=-8.0, height=height, speed=65.0, gradient=0.12, vertical_speed=-40.0
    )
    measured[2] = horizontal_speed
    assert abs(horizontal_speed) / 0.5 >= 2.0
    properties = flight.properties(measured, predicted=True)
    lever = properties.cg_m - flight.physics.geometry.nozzle_station_m
    ax_limit = (
        0.5 * flight.design.rcs_torque_limit_nm / (lever * flight.mass(measured, predicted=True))
    )
    alignment_duration = (
        abs(horizontal_speed) / ax_limit + np.sqrt(2.0 * abs(measured[0]) / ax_limit) + 4.0
    )
    allowed = max(flight.design.descent_reference_m_s, height / alignment_duration)
    expected = -measured[3] / alignment_duration + 1.2 * (-allowed - measured[3])
    expected = np.clip(expected, -gravity(measured[1]), 28.0)
    flight.guidance(measured)
    assert accelerations[-1][1] == pytest.approx(expected)


def moving_fin_allocation_case(monkeypatch, *, action_delay_steps=0):
    flight = AprFlight(disturbances=RecoveryDisturbances(action_delay_steps=action_delay_steps))
    flight.upper_attached = flight.on_pad = False
    flight.phase = "entry"
    flight.time_s = 300.0
    measured = flight.state.copy()
    measured[:6] = (0.0, 4_000.0, 0.0, -200.0, 0.0, 0.0)
    measured[6:8] = (15_844.0, 468.0)
    measured[9:13] = (0.02, 1.0, 0.0, 0.0)
    monkeypatch.setattr(
        flight, "body_loads", lambda *args, **kwargs: (np.zeros(2), 0.0, 1000.0, 0.0)
    )

    def synthetic_fin_loads(state, deflection=None, **kwargs):
        assert kwargs.get("predicted") is True
        angle = state[9] if deflection is None else deflection
        return np.zeros(2), 1_000_000.0 * angle * state[10], 0.0

    monkeypatch.setattr(flight, "fin_loads", synthetic_fin_loads)
    monkeypatch.setattr(flight, "fin_command", lambda *args: 0.0)
    return flight, measured


def test_held_rcs_accounts_for_fin_motion_over_the_control_interval(monkeypatch):
    flight, measured = moving_fin_allocation_case(monkeypatch)
    gimbal, rcs, fin_target = flight.attitude_control(0.0, 0.0, measured)
    dt = flight.return_settings.control_dt_s
    moment_start = 1_000_000.0 * measured[9]
    moment_end = moment_start * np.exp(-dt / FINS.actuator_time_constant_s)
    expected_minimax_rcs = -(moment_start + moment_end) / 2.0
    assert gimbal == fin_target == 0.0
    assert rcs == pytest.approx(expected_minimax_rcs, abs=1e-8)
    moments = moment_start * np.exp(-np.linspace(0.0, dt, 101) / FINS.actuator_time_constant_s)
    peak_error = np.max(np.abs(moments + rcs))
    assert peak_error == pytest.approx((moment_start - moment_end) / 2.0)
    assert peak_error < np.max(np.abs(moments - moment_start))


def test_pending_fin_command_is_propagated_before_new_held_rcs_interval(monkeypatch):
    flight, measured = moving_fin_allocation_case(monkeypatch, action_delay_steps=1)
    flight.action_queue.clear()
    flight.action_queue.extend([(0.0, 0.0, 111.0, -0.02), (0.0, 0.0, 222.0, 0.0)])
    before = list(flight.action_queue)
    _, rcs, _ = flight.attitude_control(0.0, 0.0, measured)
    dt = flight.return_settings.control_dt_s
    initial_at_application = measured[9] * np.exp(-dt / FINS.actuator_time_constant_s)
    moment_start = 1_000_000.0 * initial_at_application
    moment_end = moment_start * np.exp(-dt / FINS.actuator_time_constant_s)
    assert rcs == pytest.approx(-(moment_start + moment_end) / 2.0, abs=1e-8)
    assert list(flight.action_queue) == before


def test_delayed_fin_telemetry_uses_known_commands_not_current_hidden_angle(monkeypatch):
    flight, measured = moving_fin_allocation_case(monkeypatch)
    flight.observation_times.append(flight.time_s - 0.04)
    command = (0.0, 0.0, 0.0, 0.0)
    flight.navigation_command_history.extend(
        [(flight.time_s - 0.02, command), (flight.time_s, command)]
    )
    flight.state[9] = -0.2
    _, rcs, _ = flight.attitude_control(0.0, 0.0, measured)
    dt = flight.return_settings.control_dt_s
    current_angle = measured[9] * np.exp(-0.04 / FINS.actuator_time_constant_s)
    moment_start = 1_000_000.0 * current_angle
    moment_end = moment_start * np.exp(-dt / FINS.actuator_time_constant_s)
    assert rcs == pytest.approx(-(moment_start + moment_end) / 2.0, abs=1e-8)
    assert flight.state[9] == -0.2


def test_fin_interval_prediction_respects_the_physical_deflection_rate(monkeypatch):
    flight, measured = moving_fin_allocation_case(monkeypatch)
    measured[9] = np.deg2rad(20.0)
    target = -np.deg2rad(20.0)
    slope = 100_000.0

    def synthetic_fin_loads(state, deflection=None, **kwargs):
        assert kwargs.get("predicted") is True
        return np.zeros(2), slope * (state[9] if deflection is None else deflection), 0.0

    monkeypatch.setattr(flight, "fin_loads", synthetic_fin_loads)
    monkeypatch.setattr(flight, "fin_command", lambda *args: target)
    gimbal, rcs, fin_target = flight.attitude_control(0.0, 0.0, measured)
    dt = flight.return_settings.control_dt_s
    end_angle = measured[9] - np.deg2rad(FINS.deflection_rate_deg_s) * dt
    assert gimbal == 0.0
    assert fin_target == target
    assert rcs == pytest.approx(-slope * (measured[9] + end_angle) / 2.0, abs=1e-8)


def test_attitude_allocation_does_not_read_hidden_plant_uncertainties(monkeypatch):
    forecast = ((0.0, 0.0), (100_000.0, 0.0))
    flights = [
        AprFlight(
            conditions=AprConditions(
                wind_profile=((0.0, sign), (100_000.0, sign)),
                forecast_wind_profile=forecast,
                vertical_wind_profile=((0.0, 3.0 * sign), (100_000.0, 3.0 * sign)),
                density_scale=1.0 + 0.1 * sign,
            ),
            disturbances=RecoveryDisturbances(
                initial_mass_scale=1.0 + 0.01 * sign,
                thrust_scale=1.0 + 0.03 * sign,
                drag_coefficient_scale=1.0 + 0.15 * sign,
                inertia_scale=1.0 + 0.03 * sign,
                engine_lag_s=0.05 if sign > 0 else 0.04,
                action_delay_steps=1,
            ),
        )
        for sign in (-1, 1)
    ]
    measured = AprFlight().state.copy()
    measured[:6] = (500.0, 4_000.0, 15.0, -240.0, 0.1, 0.02)
    measured[6:8] = (15_844.0, 468.0)
    measured[9:13] = (0.02, 1.0, 0.01, 0.3)
    commands = []
    for index, flight in enumerate(flights):
        flight.upper_attached = flight.on_pad = False
        flight.phase = "entry"
        flight.time_s = 300.0
        flight.control = (0.3, 0.01, 2000.0, 0.02)
        flight.ignition_requested = True
        flight.ignition_ready_s = 299.0
        flight.action_queue.clear()
        flight.action_queue.extend([flight.control, (0.35, 0.02, 1000.0, -0.01)])
        flight.state[17:19] = (123.0 + index, -321.0 - index)

        def forbidden_plant_read(*args, **kwargs):
            raise AssertionError("attitude allocation must not call hidden plant dynamics")

        monkeypatch.setattr(flight, "actual_engine", forbidden_plant_read)
        monkeypatch.setattr(flight, "derivative", forbidden_plant_read)
        monkeypatch.setattr(flight, "feed_status", forbidden_plant_read)
        commands.append(flight.attitude_control(0.2, 0.35, measured))
    assert commands[0] == pytest.approx(commands[1], abs=1e-12)


@pytest.mark.parametrize("main_propellant", [5_000.0, 15_000.0, 25_000.0])
def test_attitude_torque_reference_uses_allowed_lower_inertia_not_hidden_actual_value(
    monkeypatch, main_propellant
):
    nominal = AprFlight()
    lower = AprFlight(
        disturbances=RecoveryDisturbances(initial_mass_scale=0.99, inertia_scale=0.97)
    )
    measured = nominal.state.copy()
    measured[:6] = (0.0, 4_000.0, 0.0, -200.0, 0.0, 0.0)
    measured[6:8] = (main_propellant, 468.0)
    measured[9:13] = 0.0
    commands = []
    for flight in (nominal, lower):
        flight.upper_attached = flight.on_pad = False
        flight.time_s = 300.0
        monkeypatch.setattr(
            flight, "body_loads", lambda *args, **kwargs: (np.zeros(2), 0.0, 1000.0, 0.0)
        )
        monkeypatch.setattr(flight, "fin_loads", lambda *args, **kwargs: (np.zeros(2), 0.0, 0.0))
        monkeypatch.setattr(flight, "fin_command", lambda *args: 0.0)
        commands.append(flight.attitude_control(np.deg2rad(90.0), 0.0, measured))
    assert commands[0] == pytest.approx(commands[1], abs=1e-12)
    inertia_lower = lower.properties(measured).inertia_kg_m2
    requested_torque = commands[0][1]
    alpha_limit = np.deg2rad(nominal.return_settings.attitude_acceleration_limit_deg_s2)
    assert 0.0 < requested_torque <= inertia_lower * alpha_limit + 1e-8


@pytest.mark.parametrize(
    "q,reference_alpha_deg_s2", [(4_999.0, 2.5), (5_000.0, 1.0), (20_000.0, 1.0)]
)
def test_high_q_attitude_command_leaves_margin_without_relaxing_physical_limit(
    monkeypatch, q, reference_alpha_deg_s2
):
    nominal = AprFlight()
    lower = AprFlight(
        disturbances=RecoveryDisturbances(initial_mass_scale=0.99, inertia_scale=0.97)
    )
    measured = nominal.state.copy()
    measured[:6] = (0.0, 4_000.0, 0.0, -200.0, 0.0, 0.0)
    measured[6:8] = (15_000.0, 468.0)
    measured[9:13] = 0.0
    for flight in (nominal, lower):
        flight.upper_attached = flight.on_pad = False
    monkeypatch.setattr(nominal, "body_loads", lambda *args, **kwargs: (np.zeros(2), 0.0, q, 0.0))
    monkeypatch.setattr(nominal, "fin_loads", lambda *args, **kwargs: (np.zeros(2), 0.0, 0.0))
    monkeypatch.setattr(nominal, "fin_command", lambda *args: 0.0)
    _, rcs, _ = nominal.attitude_control(np.deg2rad(90.0), 0.0, measured)
    inertia_lower = lower.properties(measured).inertia_kg_m2
    assert np.rad2deg(rcs / inertia_lower) == pytest.approx(reference_alpha_deg_s2)
    assert SAFETY.max_angular_acceleration_deg_s2 == 5.0


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_terminal_horizontal_control_brakes_before_overshooting_the_pad(monkeypatch, sign):
    flight, measured, accelerations = terminal_feedforward_case(
        monkeypatch, wind=0.0, height=60.0, speed=20.0, gradient=0.1, vertical_speed=-2.0
    )
    measured[0], measured[2] = sign * 37.33, -sign * 6.55
    flight.guidance(measured)
    properties = flight.properties(measured, predicted=True)
    lever = properties.cg_m - flight.physics.geometry.nozzle_station_m
    limit = (
        0.5 * flight.design.rcs_torque_limit_nm / (lever * flight.mass(measured, predicted=True))
    )
    assert accelerations[-1][0] == pytest.approx(sign * limit)
    assert accelerations[-1][0] * measured[2] < 0.0


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_terminal_horizontal_control_keeps_authority_for_large_position_error(monkeypatch, sign):
    flight, measured, accelerations = terminal_feedforward_case(
        monkeypatch, wind=0.0, height=280.0, speed=65.0, gradient=0.12, vertical_speed=-40.0
    )
    measured[0], measured[2] = sign * 500.0, 0.0
    flight.guidance(measured)
    properties = flight.properties(measured, predicted=True)
    lever = properties.cg_m - flight.physics.geometry.nozzle_station_m
    limit = (
        0.5 * flight.design.rcs_torque_limit_nm / (lever * flight.mass(measured, predicted=True))
    )
    assert accelerations[-1][0] == pytest.approx(-sign * limit)


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_terminal_horizontal_velocity_reference_is_linear_near_the_pad(monkeypatch, sign):
    flight, measured, accelerations = terminal_feedforward_case(
        monkeypatch, wind=0.0, height=1.0, speed=1.0, gradient=0.0, vertical_speed=-1.0
    )
    measured[0], measured[2] = sign * 0.1, 0.0
    flight.guidance(measured)
    assert accelerations[-1][0] == pytest.approx(-1.2 * 0.4 * measured[0])
    assert accelerations[-1][1] == pytest.approx(0.0)


@pytest.mark.parametrize("warm_slow_engine", [False, True])
def test_delayed_residual_throttle_keeps_engine_ignition_scenarios_independent(
    monkeypatch, warm_slow_engine
):
    observed_level = 0.000864326270972666 if warm_slow_engine else 0.0005
    flight, measured, _ = delayed_allocation_case(monkeypatch, throttle=observed_level)
    measured[11] = -0.07862628
    flight.ignition_requested, flight.ignition_ready_s = False, float("inf")
    raw = measured.copy()
    raw[12] = observed_level * np.exp(0.04 / 0.04)
    flight.observations.append(raw)
    flight.observation_times.append(flight.time_s - 0.04)
    shutdown = (0.0, 0.0, 0.0, 0.0)
    flight.navigation_command_history.extend(
        [(flight.time_s - 0.02, shutdown), (flight.time_s, shutdown)]
    )
    pending = (0.3838669734733, 0.0335083, 0.0, 0.0)
    flight.action_queue.clear()
    flight.action_queue.extend([shutdown, pending])
    new_throttle = 0.00519601048
    gimbal, rcs, _ = flight.attitude_control(0.0, new_throttle, measured)
    slow_current = raw[12] * np.exp(-0.04 / 0.05)
    assert observed_level < 1e-3
    assert bool(slow_current > 1e-3) is warm_slow_engine
    assert gimbal == 0.0
    if warm_slow_engine:
        assert abs(rcs) > 10_000.0
    else:
        assert abs(rcs) < 1000.0
    lower = AprFlight(
        disturbances=RecoveryDisturbances(initial_mass_scale=0.99, inertia_scale=0.97)
    )
    lower.upper_attached = False
    inertia_lower = lower.properties(measured).inertia_kg_m2
    lever = flight.properties(measured, predicted=True).cg_m
    lever -= flight.physics.geometry.nozzle_station_m
    dt = flight.return_settings.control_dt_s
    at_application = measured[11] + np.deg2rad(60.0) * dt
    supplied = []
    for tau, current_level in ((0.04, observed_level), (0.05, slow_current)):
        fires = current_level > 1e-3
        pending_target = pending[0] if fires else 0.0
        application_level = pending_target + (current_level - pending_target) * np.exp(-dt / tau)
        future_target = new_throttle if fires else 0.0
        for elapsed in (0.0, dt / 2.0, dt):
            level = future_target + (application_level - future_target) * np.exp(-elapsed / tau)
            angle = at_application * np.exp(-elapsed / 0.06)
            supplied.append(-lever * flight.engine(level, measured[1])[0] * np.sin(angle))
    assert rcs == pytest.approx(-(min(supplied) + max(supplied)) / 2.0, abs=1e-8)
    residual_alpha = np.rad2deg((np.asarray(supplied) + rcs) / inertia_lower)
    guard_span = 2.0 * inertia_lower * np.deg2rad(4.0)
    if np.ptp(supplied) <= guard_span:
        assert np.max(np.abs(residual_alpha)) <= 4.0 + 1e-10
    else:
        # This deliberately difficult synthetic pair has no common 4-degree
        # guard solution; the allocator retains minimax within physical limits.
        assert np.max(np.abs(residual_alpha)) < SAFETY.max_angular_acceleration_deg_s2 == 5.0
    assert flight.ignition_requested is False
    assert flight.ignition_ready_s == float("inf")


def test_failed_case_adapter_retains_decimated_approach_diagnostics_without_flying(monkeypatch):
    from scripts import evaluate_operating_envelope as campaign

    fixture = AprFlight()
    fixture.deadline_s = 0.0
    result = fixture.run()  # Zero deadline means no integration steps execute.
    assert result["time_s"] == fixture.time_s == 0.0
    assert all(step["time_s"] == 0.0 for step in result["steps"])
    result["steps"] = []
    for index in range(25):
        state = fixture.state.copy()
        state[1] = 3000.0
        result["steps"].append(
            {
                "time_s": 100.0 + index * 0.2,
                "phase": "landing",
                "state": state.tolist(),
                "control": [0.1, 0.0, 0.0, 0.0],
                "observed_control_state": state[:6].tolist(),
                "navigation_acceleration_bias_m_s2": [0.1, -0.2],
            }
        )
    monkeypatch.setattr(AprFlight, "run", lambda self: result)
    case = {
        "id": "synthetic-adapter-only",
        "group": "boundary",
        "conditions": asdict(AprConditions()),
        "disturbances": asdict(RecoveryDisturbances()),
        "index": 0,
        "stream": 0,
    }
    entry = campaign._evaluate_case(case)
    trace = entry["failure_approach_trace"]
    assert entry["accepted"] is False
    assert [step["time_s"] for step in trace] == pytest.approx([100.0, 102.0, 104.0, 104.8])
    assert all(len(step["state"]) == 19 for step in trace)
    assert trace[-1]["navigation_acceleration_bias_m_s2"] == [0.1, -0.2]
    assert "not causal proof" in entry["failure_approach_trace_sampling"]


def upright_handoff_case(monkeypatch, flight, *, height, sign):
    flight.upper_attached = flight.on_pad = False
    flight.phase = "entry"
    flight.time_s = 375.0
    flight.entry_started_s = 300.0
    flight.ignition_requested = True
    flight.ignition_ready_s = 300.0
    flight.control = (0.15, 0.0, 0.0, 0.0)
    flight.action_queue.clear()
    flight.action_queue.extend([flight.control, flight.control])
    measured = AprFlight().state.copy()
    measured[:6] = (sign * 20.0, height, sign * 0.5, -190.0, sign * np.deg2rad(6.0), 0.0)
    measured[6:8] = (8_000.0, 350.0)
    measured[8:12] = 0.0
    measured[12] = 0.15
    measured[1] += height - flight.height(measured, predicted=True)
    measured_before = measured.copy()
    targets = []

    def forecast_loads(*args, **kwargs):
        assert kwargs.get("predicted") is True
        return np.zeros(2), 0.0, 10_000.0, 0.0

    def forecast_fin(*args, **kwargs):
        assert kwargs.get("predicted") is True
        return np.zeros(2), 0.0, 0.0

    def forbidden_plant_read(*args, **kwargs):
        raise AssertionError("upright handoff must use observed state and forecast dynamics")

    original_attitude = flight.attitude_control

    def capture(target, throttle, observed, *, thrust_direction=None):
        assert observed is measured
        targets.append((target, throttle, thrust_direction))
        return original_attitude(target, throttle, observed, thrust_direction=thrust_direction)

    monkeypatch.setattr(flight, "loads", forecast_loads)
    monkeypatch.setattr(flight, "body_loads", forecast_loads)
    monkeypatch.setattr(flight, "fin_loads", forecast_fin)
    monkeypatch.setattr(flight, "descent_speed", lambda *args: (200.0, 0.02, 20.0))
    monkeypatch.setattr(flight, "actual_engine", forbidden_plant_read)
    monkeypatch.setattr(flight, "derivative", forbidden_plant_read)
    monkeypatch.setattr(flight, "feed_status", forbidden_plant_read)
    monkeypatch.setattr(flight, "attitude_control", capture)
    command = flight.guidance(measured)
    assert np.array_equal(measured, measured_before)
    assert 0.0 < command[0] <= 1.0
    assert abs(command[1]) <= np.deg2rad(flight.design.gimbal_limit_deg)
    assert abs(command[2]) <= flight.design.rcs_torque_limit_nm
    assert abs(command[3]) <= np.deg2rad(FINS.deflection_limit_deg)
    return command, targets[-1], measured


@pytest.mark.parametrize("sign", [-1.0, 1.0])
@pytest.mark.parametrize("height", [2_999.0, 3_001.0])
def test_earlier_upright_handoff_preserves_actuator_limits_and_old_policy_above_gate(
    monkeypatch, sign, height
):
    earlier, previous = AprFlight(), AprFlight()
    previous.return_settings = replace(previous.return_settings, vertical_body_height_m=1_500.0)
    earlier_command, earlier_target, _ = upright_handoff_case(
        monkeypatch, earlier, height=height, sign=sign
    )
    previous_command, previous_target, _ = upright_handoff_case(
        monkeypatch, previous, height=height, sign=sign
    )
    assert earlier.return_settings.vertical_body_height_m == 3_000.0
    assert earlier.physics.acceptance == previous.physics.acceptance
    assert SAFETY.max_angular_acceleration_deg_s2 == 5.0
    assert previous_target[0] != pytest.approx(0.0, abs=1e-5)
    if height < 3_000.0:
        assert earlier_target[0] == 0.0
        assert earlier_target[2] is not None
        assert earlier_command != pytest.approx(previous_command)
    else:
        assert earlier_target[2] is None
        assert earlier_target == pytest.approx(previous_target)
        assert earlier_command == pytest.approx(previous_command)


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_upright_handoff_uses_observed_height_without_hidden_wind_mass_or_thrust(monkeypatch, sign):
    commands, targets = [], []
    forecast = ((0.0, 0.0), (100_000.0, 0.0))
    for deviation in (-1.0, 1.0):
        flight = AprFlight(
            conditions=AprConditions(
                wind_profile=((0.0, 2.0 * deviation), (100_000.0, 2.0 * deviation)),
                forecast_wind_profile=forecast,
                density_scale=1.0 + 0.1 * deviation,
            ),
            disturbances=RecoveryDisturbances(
                initial_mass_scale=1.0 + 0.01 * deviation,
                thrust_scale=1.0 + 0.03 * deviation,
                drag_coefficient_scale=1.0 + 0.15 * deviation,
                inertia_scale=1.0 + 0.03 * deviation,
                engine_lag_s=0.05 if deviation > 0 else 0.04,
                observation_delay_steps=2,
                action_delay_steps=1,
            ),
        )
        # The plant is deliberately above the old and new transition altitudes;
        # guidance receives only the same, independently supplied observation.
        flight.state[:6] = (deviation * 800.0, 40_000.0, 100.0, 200.0, 0.5, 0.1)
        flight.state[17:19] = (deviation * 123.0, deviation * 321.0)
        command, target, measured = upright_handoff_case(
            monkeypatch, flight, height=2_999.0, sign=sign
        )
        assert flight.height(flight.state) > 3_000.0
        assert flight.height(measured, predicted=True) == pytest.approx(2_999.0)
        assert target[0] == 0.0
        commands.append(command)
        targets.append(target)
    assert commands[0] == pytest.approx(commands[1], abs=1e-12)
    assert targets[0] == pytest.approx(targets[1], abs=1e-12)


def angular_observer_case(monkeypatch, *, body_torque=0.0):
    flight = AprFlight(disturbances=RecoveryDisturbances(observation_delay_steps=2))
    flight.upper_attached = flight.on_pad = False
    flight.phase = "entry"
    flight.navigation_command_history.clear()
    measured = flight.state.copy()
    measured[:6] = (0.0, 4_000.0, 0.0, -200.0, 0.0, 0.0)
    measured[6:8] = (8_000.0, 350.0)
    measured[8:13] = 0.0
    inertia, cg = 500_000.0, 2.0

    def properties(state, *, predicted=False):
        assert predicted is True
        return SimpleNamespace(inertia_kg_m2=inertia, cg_m=cg)

    def engine(throttle, altitude, *, actual=False):
        assert actual is False
        return 0.0, 0.0

    def loads(state, *, predicted=False, time_s=None):
        assert predicted is True
        assert np.array_equal(state[17:19], [0.0, 0.0])
        return np.zeros(2), body_torque, 1_000.0, 0.0

    def forbidden_plant_read(*args, **kwargs):
        raise AssertionError("angular observer must not read hidden plant dynamics")

    monkeypatch.setattr(flight, "properties", properties)
    monkeypatch.setattr(flight, "engine", engine)
    monkeypatch.setattr(flight, "loads", loads)
    monkeypatch.setattr(flight, "actual_engine", forbidden_plant_read)
    monkeypatch.setattr(flight, "derivative", forbidden_plant_read)
    monkeypatch.setattr(flight, "feed_status", forbidden_plant_read)
    return flight, measured, inertia, cg


@pytest.mark.parametrize("bias", [-12_000.0, 12_000.0])
def test_angular_torque_observer_learns_constant_bias_from_gyro_history(monkeypatch, bias):
    flight, measured, inertia, _ = angular_observer_case(monkeypatch, body_torque=5_000.0)
    dt = flight.return_settings.control_dt_s
    original = measured.copy()
    for step in range(301):
        timestamp = step * dt
        sample = original.copy()
        sample[5] = (bias + 5_000.0) * timestamp / inertia
        sample[4] = (bias + 5_000.0) * timestamp**2 / (2.0 * inertia)
        flight.navigation_command_history.append((timestamp, (0.0, 0.0, 0.0, 0.0)))
        preserved = sample.copy()
        flight.observe_angular_torque(sample, timestamp)
        assert np.array_equal(sample, preserved)
    assert flight.angular_torque_bias_nm == pytest.approx(bias, abs=1e-6)
    assert flight.angular_observation[0] == pytest.approx(6.0)
    assert np.array_equal(measured, original)


def test_angular_observer_aligns_delayed_gyro_with_executed_not_current_rcs(monkeypatch):
    flight, old, inertia, cg = angular_observer_case(monkeypatch, body_torque=5_000.0)
    old[5] = 0.3
    begin, duration, bias = 100.0, 0.04, 9_000.0
    flight.time_s = 100.08
    flight.observe_angular_torque(old, begin)
    flight.navigation_command_history.extend(
        [
            (100.02, (0.0, 0.0, 20_000.0, 0.0)),
            (100.04, (0.0, 0.0, -10_000.0, 0.0)),
            (100.06, (0.0, 0.0, 100_000.0, 0.0)),
            (100.08, (0.0, 0.0, -100_000.0, 0.0)),
        ]
    )
    flight.control = (0.0, 0.0, -100_000.0, 0.0)
    geometry = flight.physics.geometry
    average_rcs, average_absolute_rcs = 5_000.0, 15_000.0
    flux_coefficient = (
        average_absolute_rcs
        * (geometry.rcs_arm_m**2 + (geometry.rcs_station_m - cg) ** 2)
        / (geometry.rcs_arm_m * flight.design.rcs_isp_s * G0)
    )
    measured = old.copy()
    measured[5] = (
        old[5] * (inertia - duration * flux_coefficient / 2.0)
        + duration * (5_000.0 + average_rcs + bias)
    ) / (inertia + duration * flux_coefficient / 2.0)
    flight.state[5] = 999.0
    flight.observe_angular_torque(measured, begin + duration)
    expected = bias * (1.0 - np.exp(-duration / 0.12))
    assert flight.angular_torque_bias_nm == pytest.approx(expected, abs=1e-6)
    assert flight.angular_observation[0] == pytest.approx(100.04)


@pytest.mark.parametrize("timestamp", [0.0, 0.02, 0.04])
def test_angular_observer_does_not_relearn_repeated_or_older_measurements(monkeypatch, timestamp):
    flight, measured, _, _ = angular_observer_case(monkeypatch)
    flight.angular_observation = (0.04, measured.copy())
    flight.angular_torque_bias_nm = 7_000.0
    previous = flight.angular_observation
    changed = measured.copy()
    changed[5] = 999.0
    flight.observe_angular_torque(changed, timestamp)
    assert flight.angular_torque_bias_nm == 7_000.0
    assert flight.angular_observation is previous


@pytest.mark.parametrize("reset_reason", ["attached", "pad", "preseparation"])
def test_angular_observer_resets_across_pad_support_and_staging(monkeypatch, reset_reason):
    flight, measured, _, _ = angular_observer_case(monkeypatch)
    flight.angular_observation = (99.0, measured.copy())
    flight.angular_torque_bias_nm = 7_000.0
    if reset_reason == "attached":
        flight.upper_attached = True
    elif reset_reason == "pad":
        flight.on_pad = True
    else:
        flight.separation = {"time_s": 100.02}
    flight.observe_angular_torque(measured, 100.0)
    assert flight.angular_torque_bias_nm == 0.0
    assert flight.angular_observation is None


@pytest.mark.parametrize("history_end", [0.02, 0.04])
def test_angular_observer_rejects_incomplete_front_or_tail_command_history(
    monkeypatch, history_end
):
    flight, old, inertia, _ = angular_observer_case(monkeypatch)
    flight.observe_angular_torque(old, 0.0)
    flight.angular_torque_bias_nm = 7_000.0
    flight.navigation_command_history.append((history_end, (0.0, 0.0, 0.0, 0.0)))
    measured = old.copy()
    measured[5] = 12_000.0 * 0.04 / inertia
    flight.observe_angular_torque(measured, 0.04)
    assert flight.angular_torque_bias_nm == 0.0
    assert flight.angular_observation[0] == pytest.approx(0.04)


def test_angular_observer_unwraps_midpoint_angle_and_ignores_unmeasured_slosh(monkeypatch):
    flight, old, _, _ = angular_observer_case(monkeypatch)
    old[4:6] = (np.pi - 0.01, 1.0)
    old[17:19] = (123.0, -321.0)
    flight.observe_angular_torque(old, 0.0)
    flight.navigation_command_history.append((0.02, (0.0, 0.0, 0.0, 0.0)))
    measured = old.copy()
    measured[4] = -np.pi + 0.01
    measured[17:19] = (-123.0, 321.0)
    midpoints = []

    def loads(state, *, predicted=False, time_s=None):
        assert predicted is True
        assert np.array_equal(state[17:19], [0.0, 0.0])
        midpoints.append((state.copy(), time_s))
        return np.zeros(2), 10_000.0 * np.cos(state[4]), 1_000.0, 0.0

    monkeypatch.setattr(flight, "loads", loads)
    flight.observe_angular_torque(measured, 0.02)
    assert midpoints[-1][0][4] == pytest.approx(np.pi)
    assert midpoints[-1][1] == pytest.approx(0.01)
    assert flight.angular_torque_bias_nm == pytest.approx(
        10_000.0 * (1.0 - np.exp(-0.02 / 0.12)), abs=1e-6
    )


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_angular_torque_bias_is_bounded_in_nominal_acceleration_units(monkeypatch, sign):
    flight, old, inertia, _ = angular_observer_case(monkeypatch)
    flight.observe_angular_torque(old, 0.0)
    dt = flight.return_settings.control_dt_s
    for step in range(1, 151):
        timestamp = step * dt
        measured = old.copy()
        measured[5] = sign * 100_000.0 * timestamp / inertia
        flight.navigation_command_history.append((timestamp, (0.0, 0.0, 0.0, 0.0)))
        flight.observe_angular_torque(measured, timestamp)
        assert abs(flight.angular_torque_bias_nm) <= inertia * np.deg2rad(4.0)
    assert flight.angular_torque_bias_nm == pytest.approx(
        sign * inertia * np.deg2rad(4.0), abs=1e-5
    )
    assert SAFETY.max_angular_acceleration_deg_s2 == 5.0


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_attitude_allocator_uses_learned_bias_once_in_both_target_and_guard(monkeypatch, sign):
    flight = AprFlight()
    flight.upper_attached = flight.on_pad = False
    measured = flight.state.copy()
    measured[:6] = (0.0, 4_000.0, 0.0, -200.0, 0.0, 0.0)
    measured[6:8] = (15_000.0, 350.0)
    measured[8:13] = 0.0
    nominal_inertia = flight.properties(measured, predicted=True).inertia_kg_m2
    bias = sign * 0.99 * nominal_inertia * np.deg2rad(4.0)
    flight.angular_torque_bias_nm = bias
    lower = AprFlight(
        disturbances=RecoveryDisturbances(initial_mass_scale=0.99, inertia_scale=0.97)
    )
    lower.upper_attached = False
    lower_inertia = lower.properties(measured).inertia_kg_m2
    assert abs(bias) > lower_inertia * np.deg2rad(4.0)
    monkeypatch.setattr(
        flight, "body_loads", lambda *args, **kwargs: (np.zeros(2), 0.0, 1_000.0, 0.0)
    )
    monkeypatch.setattr(flight, "fin_loads", lambda *args, **kwargs: (np.zeros(2), 0.0, 0.0))
    monkeypatch.setattr(flight, "fin_command", lambda *args: 0.0)
    gimbal, rcs, fin = flight.attitude_control(0.0, 0.0, measured)
    assert gimbal == fin == 0.0
    assert rcs == pytest.approx(-bias, abs=1e-8)
    assert abs(rcs) < flight.design.rcs_torque_limit_nm
    assert np.rad2deg((rcs + bias) / lower_inertia) == pytest.approx(0.0, abs=1e-10)
