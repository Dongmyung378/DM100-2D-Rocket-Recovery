"""100 t APR mass, physics, operating-input and evidence contracts.

These unit tests are not successful-flight evidence. Full baseline flight
results are recorded separately and must satisfy the unchanged landing gates.
"""

import json
from copy import deepcopy
from dataclasses import asdict, astuple, replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from powered_landing_guidance.apr_recovery import (
    FINS,
    OPERATING_SCOPE,
    PHYSICS,
    PROFILE_TOP_M,
    SAFETY,
    STATE_NAMES,
    UNCONNECTED_SYSTEMS,
    AprConditions,
    AprFlight,
    Design,
    RecoveryDisturbances,
    build_descent_reference,
    load_apr_operating,
)
from powered_landing_guidance.integrators import rk4_step
from powered_landing_guidance.recovery_physics import (
    G0,
    R_AIR,
    AtmosphereSettings,
    atmosphere,
    clock_torque,
    engine_performance,
    foot_kinematics,
    gravity,
    mass_gradients,
    mass_properties,
    rotation,
)
from powered_landing_guidance.vehicle_systems import SloshSettings, initial_feed_state
from powered_landing_guidance.visualization import (
    _nominal_flight_frame_times,
    _nominal_flight_geometry,
    _NominalFlightAnimationScene,
    _NominalFlightFrameRenderer,
    _NominalFlightReplay,
    save_nominal_flight_replay,
)
from scripts import run_nominal_flight as apr_cli
from scripts.evaluate_operating_envelope import (
    MANIFEST_SCHEMA_VERSION,
    ROOT_SEED,
    VEHICLE_ID,
    _encoded,
    _load_manifest,
    _sha,
    _source_hashes,
    _stage,
    freeze,
    make_manifest,
    observation_seed_sequence,
    prepare,
)

ROOT = Path(__file__).resolve().parents[1]
ISA_HEIGHTS = (0.0, 11000.0, 20000.0, 32000.0, 47000.0, 51000.0, 71000.0, 84852.0, PROFILE_TOP_M)


def apr_separation_state():
    """Explicit 19-state pose, including a consistent feed inventory."""
    state = AprFlight().state.copy()
    state[:8] = (
        10712.8,
        37113.9,
        773.5,
        613.7,
        np.deg2rad(80.0),
        0.0,
        20000.0,
        Design().rcs_propellant_kg,
    )
    state[13:17] = astuple(initial_feed_state(state[6]))
    return state


def test_complete_separation_state_preserves_explicit_system_history():
    complete = AprFlight.from_separation(apr_separation_state(), time_s=98.0).state.copy()
    complete[13] *= 0.9
    complete[14] = 0.8
    complete[17:19] = (0.01, -0.02)
    flight = AprFlight.from_separation(complete, time_s=98.0)
    np.testing.assert_array_equal(flight.state, complete)


@pytest.mark.parametrize("wrong_size", (13, 15, 21))
def test_separation_input_rejects_incompatible_state_formats(wrong_size):
    with pytest.raises(ValueError, match="19"):
        AprFlight.from_separation(np.zeros(wrong_size), time_s=98.0)


def _write_config(tmp_path, mutate):
    data = yaml.safe_load((ROOT / "configs/apr-operating-v1.yaml").read_text(encoding="utf-8"))
    mutate(data)
    path = tmp_path / "conditions.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_one_public_model_is_the_100_t_design():
    assert apr_cli.AprFlight is AprFlight
    design = Design()
    assert design.launch_mass_kg == 100_000.0
    assert design.dry_mass_kg == 12_000.0
    assert design.main_propellant_kg == 70_200.0
    assert design.rcs_propellant_kg == 300.0
    assert design.upper_mass_kg == 17_500.0
    assert design.payload_mass_kg == 500.0
    assert design.upper_stage_wet_mass_kg == 17_000.0
    assert "100t" in OPERATING_SCOPE
    assert PROFILE_TOP_M == 100_000.0


def test_rcs_redistribution_keeps_first_stage_mass_and_actuator_authority():
    design = Design()
    flight = AprFlight()
    assert design.dry_mass_kg + design.main_propellant_kg + design.rcs_propellant_kg == 82_500.0
    assert design.main_propellant_kg + design.rcs_propellant_kg == 70_500.0
    assert design.rcs_torque_limit_nm == 100_000.0
    assert design.rcs_isp_s == 220.0
    assert flight.state[6:8] == pytest.approx((70_200.0, 300.0))
    assert flight.mass(flight.state) == 100_000.0


def test_100_t_dry_budget_is_not_double_counted_as_fuel():
    geometry = PHYSICS.geometry
    dry = geometry.engine_mass_kg + geometry.airframe_mass_kg + geometry.gear_mass_kg
    fixed = sum(component[0] for component in PHYSICS.fixed_mass_components)
    assert geometry.engine_mass_kg == 1200.0
    assert geometry.airframe_mass_kg == 8700.0
    assert geometry.gear_mass_kg == 1500.0
    assert fixed == FINS.mass_kg == 600.0
    assert dry + fixed == Design().dry_mass_kg
    assert geometry.body_length_m == 20.0
    assert geometry.body_radius_m == 1.5


def test_full_first_stage_liquid_inventory_fits_the_equivalent_tank():
    design = Design()
    properties = mass_properties(
        design.dry_mass_kg + design.main_propellant_kg + design.rcs_propellant_kg,
        0.0,
        PHYSICS,
    )
    assert 0.0 < properties.fuel_height_m <= PHYSICS.geometry.tank_length_m
    assert (
        PHYSICS.geometry.tank_bottom_m + properties.fuel_height_m < PHYSICS.geometry.body_length_m
    )
    assert properties.inertia_kg_m2 > 0.0


@pytest.mark.parametrize("mass", (12_500.0, 32_500.0, 82_500.0))
@pytest.mark.parametrize("deployment", (0.0, 0.4, 1.0))
def test_100_t_mass_and_deployment_gradients_match_finite_differences(mass, deployment):
    props = mass_properties(mass, deployment, PHYSICS)
    cm, cg, im, ig = mass_gradients(mass, deployment, PHYSICS, props)
    mass_step, gear_step = 0.01, 1e-5
    before = mass_properties(mass - mass_step, deployment, PHYSICS)
    after = mass_properties(mass + mass_step, deployment, PHYSICS)
    assert cm == pytest.approx((after.cg_m - before.cg_m) / (2 * mass_step), abs=1e-9)
    assert im == pytest.approx(
        (after.inertia_kg_m2 - before.inertia_kg_m2) / (2 * mass_step), rel=1e-6
    )
    before = mass_properties(mass, deployment - gear_step, PHYSICS)
    after = mass_properties(mass, deployment + gear_step, PHYSICS)
    assert cg == pytest.approx((after.cg_m - before.cg_m) / (2 * gear_step), abs=1e-9)
    assert ig == pytest.approx(
        (after.inertia_kg_m2 - before.inertia_kg_m2) / (2 * gear_step), rel=1e-6
    )


def test_landing_limits_are_preserved_without_positive_reserve_threshold():
    acceptance = PHYSICS.acceptance
    assert acceptance.max_position_error_m == 2.0
    assert acceptance.max_horizontal_contact_speed_m_s == 0.5
    assert acceptance.max_vertical_foot_speed_m_s == 1.5
    assert acceptance.max_contact_tilt_deg == 3.0
    assert acceptance.max_contact_rate_deg_s == 0.5


def test_fin_policy_changes_preserve_physical_and_acceptance_limits():
    flight = AprFlight()
    assert flight.return_settings.vertical_body_height_m == 3_000.0
    assert flight.return_settings.fin_min_descent_speed_m_s == 200.0
    assert flight.design.launch_mass_kg == 100_000.0
    assert flight.design.gimbal_limit_deg == 6.0
    assert FINS.mass_kg == 600.0
    assert FINS.deployment_time_s == FINS.retraction_time_s == 2.0
    assert FINS.deflection_limit_deg == 20.0
    assert FINS.deflection_rate_deg_s == 60.0
    assert FINS.control_max_alpha_deg == 25.0
    assert FINS.per_fin_hinge_limit_nm == 5_000.0
    assert SAFETY.max_q_pa == 30_000.0
    assert SAFETY.max_angular_acceleration_deg_s2 == 5.0
    assert PHYSICS.acceptance.max_vertical_foot_speed_m_s == 1.5


def _fin_policy_bench(monkeypatch, *, height_m=5_000.0, alpha_deg=0.0):
    """Exercise fin state derivatives without integrating a flight trajectory."""
    flight = AprFlight()
    flight.upper_attached = False
    flight.on_pad = False
    flight.phase = "entry"
    state = flight.state.copy()
    state[:8] = (0.0, 5_000.0, 0.0, -201.0, 0.0, 0.0, 20_000.0, Design().rcs_propellant_kg)
    state[10] = 0.5
    state[13:17] = astuple(initial_feed_state(state[6]))
    monkeypatch.setattr(flight, "height", lambda *_args, **_kwargs: height_m)
    monkeypatch.setattr(flight, "alpha_deg", lambda *_args, **_kwargs: alpha_deg)
    return flight, state


@pytest.mark.parametrize(
    "vertical_speed,expected_rate", ((-201.0, 0.5), (-200.0, -0.5), (-199.0, -0.5))
)
def test_fin_policy_retracts_at_the_200_m_s_boundary(monkeypatch, vertical_speed, expected_rate):
    flight, state = _fin_policy_bench(monkeypatch)
    state[3] = vertical_speed
    rate = flight.derivative(state, (0.0, 0.0, 0.0, 0.0))
    assert rate[10] == expected_rate


@pytest.mark.parametrize(
    "phase,expected_rate",
    (
        ("coast", 0.5),
        ("entry", 0.5),
        ("landing", 0.5),
        ("ascent", -0.5),
        ("flip", -0.5),
        ("boostback", -0.5),
    ),
)
def test_fin_policy_preserves_the_active_phase_gate(monkeypatch, phase, expected_rate):
    flight, state = _fin_policy_bench(monkeypatch)
    flight.phase = phase
    assert flight.derivative(state, (0.0, 0.0, 0.0, 0.0))[10] == expected_rate


@pytest.mark.parametrize(
    "height_m,expected_rate",
    (
        (499.0, -0.5),
        (500.0, -0.5),
        (501.0, 0.5),
        (19_999.0, 0.5),
        (20_000.0, -0.5),
        (20_001.0, -0.5),
    ),
)
def test_fin_policy_preserves_the_strict_altitude_window(monkeypatch, height_m, expected_rate):
    flight, state = _fin_policy_bench(monkeypatch, height_m=height_m)
    assert flight.derivative(state, (0.0, 0.0, 0.0, 0.0))[10] == expected_rate


@pytest.mark.parametrize(
    "alpha_deg,expected_rate", ((-15.0, -0.5), (-14.99, 0.5), (14.99, 0.5), (15.0, -0.5))
)
def test_fin_policy_preserves_the_deployment_angle_gate(monkeypatch, alpha_deg, expected_rate):
    flight, state = _fin_policy_bench(monkeypatch, alpha_deg=alpha_deg)
    assert flight.derivative(state, (0.0, 0.0, 0.0, 0.0))[10] == expected_rate


@pytest.mark.parametrize("upper_attached,fin_mode", ((True, "active"), (False, "stowed")))
def test_fin_policy_keeps_attached_and_stowed_fins_retracted(monkeypatch, upper_attached, fin_mode):
    flight, state = _fin_policy_bench(monkeypatch)
    flight.upper_attached = upper_attached
    flight.fin_mode = fin_mode
    assert flight.derivative(state, (0.0, 0.0, 0.0, 0.0))[10] == -0.5


@pytest.mark.parametrize("vertical_speed,extension", ((-201.0, 1.0), (-200.0, 0.0)))
def test_fin_policy_stops_extension_at_the_physical_endpoints(
    monkeypatch, vertical_speed, extension
):
    flight, state = _fin_policy_bench(monkeypatch)
    state[3] = vertical_speed
    state[10] = extension
    assert flight.derivative(state, (0.0, 0.0, 0.0, 0.0))[10] == 0.0


def test_fin_policy_retraction_takes_two_seconds_not_an_instantaneous_switch(monkeypatch):
    flight, state = _fin_policy_bench(monkeypatch)
    state[3] = -200.0
    state[10] = 1.0
    original = state.copy()
    control = (0.0, 0.0, 0.0, 0.0)
    dt_s = 0.01
    for step in range(199):
        rate = flight.derivative(state, control)[10]
        assert rate == -0.5
        state[10] += dt_s * rate
        assert state[10] == pytest.approx(1.0 - (step + 1) * dt_s / 2.0, abs=1e-12)
    assert state[10] == pytest.approx(0.005, abs=1e-12)
    assert state[10] > 0.0
    state[10] = max(0.0, state[10] + dt_s * flight.derivative(state, control)[10])
    assert state[10] == pytest.approx(0.0, abs=1e-12)
    assert flight.derivative(state, control)[10] == 0.0
    np.testing.assert_array_equal(state[:10], original[:10])
    np.testing.assert_array_equal(state[11:], original[11:])


@pytest.mark.parametrize("target_deg", (-20.0, 20.0))
def test_fin_policy_retains_the_finite_deflection_rate_limit(monkeypatch, target_deg):
    flight, state = _fin_policy_bench(monkeypatch)
    rate = flight.derivative(state, (0.0, 0.0, 0.0, np.deg2rad(target_deg)))
    assert rate[9] == pytest.approx(np.sign(target_deg) * np.deg2rad(60.0))


def _angular_torque_observer_bench(monkeypatch, *, body_torque_nm=0.0):
    """Constant-inertia telemetry bench, not a simulated flight trajectory."""
    flight = AprFlight()
    flight.upper_attached = False
    flight.on_pad = False
    flight.time_s = 20.0
    properties = replace(
        flight.properties(flight.state, predicted=True),
        inertia_kg_m2=100_000.0,
        cg_m=7.0,
    )
    load_times = []

    def nominal_properties(_state, *, predicted=False):
        assert predicted is True
        return properties

    def nominal_engine(_throttle, _altitude, *, actual=False):
        assert actual is False
        return 0.0, 0.0

    def nominal_loads(state, *, predicted=False, time_s=None):
        assert predicted is True
        np.testing.assert_array_equal(state[17:19], 0.0)
        load_times.append(time_s)
        return np.zeros(2), body_torque_nm, 0.0, 0.0

    monkeypatch.setattr(flight, "properties", nominal_properties)
    monkeypatch.setattr(flight, "engine", nominal_engine)
    monkeypatch.setattr(flight, "loads", nominal_loads)
    measured = flight.state.copy()
    measured[1] = 5_000.0
    measured[4:6] = 0.0
    return flight, measured, properties, load_times


def _observer_rate_after_rcs_interval(
    flight, properties, rate, duration, mean_rcs, mean_abs_rcs, bias
):
    geometry = flight.physics.geometry
    flux_coefficient = (geometry.rcs_arm_m**2 + (geometry.rcs_station_m - properties.cg_m) ** 2) / (
        geometry.rcs_arm_m * flight.design.rcs_isp_s * G0
    )
    damping = mean_abs_rcs * flux_coefficient
    return (
        properties.inertia_kg_m2 * rate
        + duration * (mean_rcs + bias)
        - duration * damping * rate / 2.0
    ) / (properties.inertia_kg_m2 + duration * damping / 2.0)


def test_angular_torque_observer_recovers_the_sign_of_a_nominal_model_bias(monkeypatch):
    flight, old, props, load_times = _angular_torque_observer_bench(
        monkeypatch, body_torque_nm=2_000.0
    )
    flight.navigation_command_history.append((10.02, (0.0, 0.0, 0.0, 0.0)))
    flight.observe_angular_torque(old, 10.0)
    measured = old.copy()
    measured[5] = 0.02 * (2_000.0 - 1_000.0) / props.inertia_kg_m2
    before = measured.copy()
    flight.observe_angular_torque(measured, 10.02)
    blend = 1.0 - np.exp(-0.02 / 0.12)
    assert flight.angular_torque_bias_nm == pytest.approx(-1_000.0 * blend, abs=1e-8)
    assert load_times == pytest.approx([10.01])
    np.testing.assert_array_equal(measured, before)


def test_angular_torque_observer_integrates_executed_not_requested_or_future_rcs(monkeypatch):
    flight, old, props, _ = _angular_torque_observer_bench(monkeypatch)
    old[5] = 0.03
    flight.control = (0.0, 0.0, -90_000.0, 0.0)
    flight.requested_control = (0.0, 0.0, 100_000.0, 0.0)
    flight.navigation_command_history.extend(
        ((10.01, (0.0, 0.0, 5_000.0, 0.0)), (10.02, (0.0, 0.0, -2_000.0, 0.0)))
    )
    flight.observe_angular_torque(old, 10.0)
    measured = old.copy()
    measured[5] = _observer_rate_after_rcs_interval(
        flight, props, old[5], 0.02, 1_500.0, 3_500.0, 500.0
    )
    flight.observe_angular_torque(measured, 10.02)
    assert flight.angular_torque_bias_nm == pytest.approx(
        500.0 * (1.0 - np.exp(-0.02 / 0.12)), abs=1e-8
    )


def test_angular_torque_observer_keeps_absolute_rcs_outflow_when_signed_torque_cancels(monkeypatch):
    flight, old, props, _ = _angular_torque_observer_bench(monkeypatch)
    old[5] = 0.05
    flight.navigation_command_history.extend(
        ((10.01, (0.0, 0.0, 10_000.0, 0.0)), (10.02, (0.0, 0.0, -10_000.0, 0.0)))
    )
    flight.observe_angular_torque(old, 10.0)
    measured = old.copy()
    measured[5] = _observer_rate_after_rcs_interval(flight, props, old[5], 0.02, 0.0, 10_000.0, 0.0)
    flight.observe_angular_torque(measured, 10.02)
    assert flight.angular_torque_bias_nm == pytest.approx(0.0, abs=1e-8)


@pytest.mark.parametrize("timestamp", (9.99, 10.02))
def test_angular_torque_observer_ignores_older_or_duplicate_measurements(monkeypatch, timestamp):
    flight, old, _, _ = _angular_torque_observer_bench(monkeypatch)
    flight.observe_angular_torque(old, 10.02)
    flight.angular_torque_bias_nm = 123.0
    measured = old.copy()
    measured[5] = 10.0
    flight.observe_angular_torque(measured, timestamp)
    assert flight.angular_torque_bias_nm == 123.0
    assert flight.angular_observation[0] == 10.02
    np.testing.assert_array_equal(flight.angular_observation[1], old)


@pytest.mark.parametrize("mode", ("attached", "pad", "before_separation"))
def test_angular_torque_observer_resets_support_and_stage_impulse_windows(monkeypatch, mode):
    flight, measured, _, _ = _angular_torque_observer_bench(monkeypatch)
    flight.angular_observation = (10.0, measured.copy())
    flight.angular_torque_bias_nm = 500.0
    if mode == "attached":
        flight.upper_attached = True
    elif mode == "pad":
        flight.on_pad = True
    else:
        flight.separation = {"time_s": 10.03}
    flight.observe_angular_torque(measured, 10.02)
    assert flight.angular_observation is None
    assert flight.angular_torque_bias_nm == 0.0


def test_angular_torque_observer_initializes_after_separation_without_differencing_across_impulse(
    monkeypatch,
):
    flight, measured, _, _ = _angular_torque_observer_bench(monkeypatch)
    flight.separation = {"time_s": 10.0}
    flight.angular_torque_bias_nm = 500.0
    measured[5] = 1.0
    flight.observe_angular_torque(measured, 9.99)
    measured[5] = -1.0
    flight.observe_angular_torque(measured, 10.0)
    assert flight.angular_observation[0] == 10.0
    assert flight.angular_torque_bias_nm == 0.0


@pytest.mark.parametrize("missing_part", ("prefix", "suffix"))
def test_angular_torque_observer_rejects_incomplete_command_windows(monkeypatch, missing_part):
    flight, old, _, load_times = _angular_torque_observer_bench(monkeypatch)
    flight.observe_angular_torque(old, 10.0)
    flight.angular_torque_bias_nm = 500.0
    if missing_part == "prefix":
        flight.navigation_command_history.extend(
            ((10.06, (0.0, 0.0, 1_000.0, 0.0)), (10.08, (0.0, 0.0, 1_000.0, 0.0)))
        )
        timestamp = 10.08
    else:
        flight.navigation_command_history.append((10.01, (0.0, 0.0, 1_000.0, 0.0)))
        timestamp = 10.02
    measured = old.copy()
    measured[5] = 10.0
    flight.observe_angular_torque(measured, timestamp)
    assert flight.angular_torque_bias_nm == 0.0
    assert load_times == []


@pytest.mark.parametrize("sign", (-1.0, 1.0))
def test_angular_torque_observer_clips_extreme_residuals_to_a_finite_equivalent_torque(
    monkeypatch, sign
):
    flight, old, props, _ = _angular_torque_observer_bench(monkeypatch)
    flight.navigation_command_history.append((10.02, (0.0, 0.0, 0.0, 0.0)))
    flight.observe_angular_torque(old, 10.0)
    measured = old.copy()
    measured[5] = sign * 0.02 * 100_000.0 / props.inertia_kg_m2
    flight.observe_angular_torque(measured, 10.02)
    bound = props.inertia_kg_m2 * np.deg2rad(4.0)
    assert flight.angular_torque_bias_nm == pytest.approx(
        sign * bound * (1.0 - np.exp(-0.02 / 0.12)), abs=1e-8
    )
    assert np.isfinite(flight.angular_torque_bias_nm)
    assert abs(flight.angular_torque_bias_nm) <= bound


def test_angular_torque_observer_uses_momentum_difference_when_inertia_changes(monkeypatch):
    flight, old, props, _ = _angular_torque_observer_bench(monkeypatch)

    def moving_inertia(state, *, predicted=False):
        assert predicted is True
        return replace(props, inertia_kg_m2=props.inertia_kg_m2 + 10_000.0 * state[8])

    monkeypatch.setattr(flight, "properties", moving_inertia)
    flight.navigation_command_history.append((10.02, (0.0, 0.0, 0.0, 0.0)))
    old[5] = 0.1
    old[8] = 0.0
    flight.observe_angular_torque(old, 10.0)
    measured = old.copy()
    measured[8] = 0.01
    measured[5] = (
        old[5]
        * moving_inertia(old, predicted=True).inertia_kg_m2
        / moving_inertia(measured, predicted=True).inertia_kg_m2
    )
    flight.observe_angular_torque(measured, 10.02)
    assert flight.angular_torque_bias_nm == pytest.approx(0.0, abs=1e-8)


def test_angular_torque_observer_does_not_consult_hidden_plant_or_fluid_coordinates(monkeypatch):
    first, second, _ = _hidden_uncertainty_pair()
    old = apr_separation_state()
    old[5] = 0.02
    old[11:13] = (0.01, 0.1)
    measured = old.copy()
    measured[5] = 0.0201

    def hidden_access(*_args, **_kwargs):
        raise AssertionError("Angular observer must not call the actual plant.")

    for index, flight in enumerate((first, second)):
        flight.time_s = 98.04
        flight.state[5] = 100.0 if index == 0 else -100.0
        flight.state[17:19] = (1.0, -1.0) if index == 0 else (-1.0, 1.0)
        flight.navigation_command_history.append((98.02, (0.1, 0.01, 1_000.0, 0.0)))
        monkeypatch.setattr(flight, "derivative", hidden_access)
        monkeypatch.setattr(flight, "actual_engine", hidden_access)
        monkeypatch.setattr(flight, "feed_status", hidden_access)
        delayed = old.copy()
        delayed[17:19] = flight.state[17:19]
        flight.observe_angular_torque(delayed, 98.0)
        delayed = measured.copy()
        delayed[17:19] = -flight.state[17:19]
        flight.observe_angular_torque(delayed, 98.02)
    assert first.angular_torque_bias_nm == pytest.approx(second.angular_torque_bias_nm, abs=1e-12)


def _contact_gate_bench(**options):
    """Construct one contact pose without integrating or claiming a flight."""
    flight = AprFlight.from_separation(apr_separation_state(), time_s=98.0, **options)
    flight.state[8] = 1.0
    flight.state[4:6] = 0.0
    flight.state[11:13] = 0.0
    flight.state[:4] = (
        0.0,
        flight.properties(flight.state).cg_m - flight.physics.gear.foot_station_m,
        0.0,
        -1.0,
    )
    return flight


@pytest.mark.parametrize(
    "index,value,criterion",
    (
        (0, 2.001, "position"),
        (0, -2.001, "position"),
        (2, 0.501, "horizontal_foot_speed"),
        (2, -0.501, "horizontal_foot_speed"),
        (3, -1.501, "vertical_foot_speed"),
        (4, np.deg2rad(3.001), "tilt"),
        (4, np.deg2rad(-3.001), "tilt"),
        (5, np.deg2rad(0.501), "angular_rate"),
        (5, np.deg2rad(-0.501), "angular_rate"),
        (6, 0.0, "main_propellant"),
        (8, 0.99, "gear_deployed"),
    ),
)
def test_touchdown_gates_reject_each_outside_limit_without_relaxation(index, value, criterion):
    flight = _contact_gate_bench()
    flight.state[index] = value
    flight.first_contact(flight.state, (0.0, 0.0, 0.0, 0.0))
    assert flight.contact["criteria"][criterion] is False


@pytest.mark.parametrize("speed,accepted", ((1.499, True), (1.5, True), (1.501, False)))
def test_touchdown_vertical_limit_remains_one_point_five(speed, accepted):
    flight = _contact_gate_bench()
    flight.state[3] = -speed
    flight.first_contact(flight.state, (0.0, 0.0, 0.0, 0.0))
    assert flight.contact["criteria"]["vertical_foot_speed"] is accepted


def test_zero_rcs_inventory_is_allowed_but_positive_main_inventory_is_required():
    flight = _contact_gate_bench()
    flight.state[7] = 0.0
    flight.first_contact(flight.state, (0.0, 0.0, 0.0, 0.0))
    assert flight.contact["criteria"]["rcs_propellant"] is True
    assert flight.contact["criteria"]["main_propellant"] is True
    flight.state[6] = 0.0
    flight.first_contact(flight.state, (0.0, 0.0, 0.0, 0.0))
    assert flight.contact["criteria"]["main_propellant"] is False


def test_default_landing_assumes_gear_capacity_without_claiming_structural_verification():
    flight = _contact_gate_bench()
    flight.state[3] = -1.5
    before = flight.state.copy()
    flight.first_contact(flight.state, (0.0, 0.0, 0.0, 0.0))
    result = flight.run()
    assert result["landing_assessment"] == "kinematic_touchdown"
    assert result["acceptance_version"] == "apr_100t_kinematic_touchdown_v1"
    assert result["landing_success"] is True
    assert result["simulated_acceptance_passed"] is True
    assert result["assessment_scope"]["contact_structural_dynamics"] is False
    assert result["assessment_scope"]["post_contact_settling"] is False
    assert result["time_s"] == result["contact"]["time_s"] == 98.0
    np.testing.assert_array_equal(flight.state, before)


def test_simple_touchdown_still_rejects_excessive_descent_and_prior_flight_loads():
    flight = _contact_gate_bench()
    flight.state[3] = -1.501
    flight.first_contact(flight.state, (0.0, 0.0, 0.0, 0.0))
    result = flight.run()
    assert result["outcome"] == "touchdown_limits_exceeded"
    assert result["landing_success"] is False
    assert result["simulated_acceptance_passed"] is False
    flight = _contact_gate_bench()
    flight.max_acceleration = 5.001
    flight.first_contact(flight.state, (0.0, 0.0, 0.0, 0.0))
    result = flight.run()
    assert result["landing_success"] is True
    assert result["criteria"]["angular_acceleration"] is False
    assert result["simulated_acceptance_passed"] is False


def test_contact_event_stops_at_first_touchdown_without_fabricating_stationary_state():
    flight = _contact_gate_bench()
    flight.state[1] += 0.005
    flight.state[12] = 0.2
    flight.control = (0.2, 0.0, 0.0, 0.0)
    flight.next_control_s = flight.time_s + 1.0
    flight.next_record_s = flight.time_s + 1.0
    before_time = flight.time_s

    for _ in range(20):
        flight.advance()
        if flight.outcome != "running":
            break
    assert flight.outcome == "landed"
    assert 0.0 < flight.time_s - before_time < 0.02
    assert flight.time_s == flight.contact["time_s"]
    assert abs(flight.height(flight.state)) < 1e-8
    assert flight.state[12] == pytest.approx(0.2)
    assert flight.state[3] < -0.9  # State was not fabricated as stationary.
    final = flight.state.copy()
    flight.advance()
    np.testing.assert_array_equal(flight.state, final)


def test_recovery_only_scope_cannot_certify_an_unobserved_ascent_or_separation():
    # Synthetic successful-contact output checks scope reporting only.
    flight = _contact_gate_bench()
    flight.first_contact(flight.state, (0.0, 0.0, 0.0, 0.0))
    assert all(flight.contact["criteria"].values())
    flight.outcome = "landed"
    result = flight.run()
    assert result["landing_success"] is True
    assert result["ascent_simulated"] is False
    assert result["criteria"]["separation"] is None
    assert result["mission_success"] is False
    assert result["simulation_scope"] == "37km_separation_to_touchdown"
    assert result["simulated_acceptance_passed"] is True


def test_contact_requires_both_actual_feet_inside_the_pad():
    flight = _contact_gate_bench()
    flight.physics = replace(
        flight.physics, gear=replace(flight.physics.gear, pad_half_width_m=3.0)
    )
    flight.first_contact(flight.state, (0.0, 0.0, 0.0, 0.0))
    assert flight.contact["criteria"]["position"] is True
    assert flight.contact["criteria"]["feet_on_pad"] is False
    points, _ = foot_kinematics(flight.vector(flight.state), flight.state[8], flight.settings())
    assert np.max(np.abs(points[:, 0])) == pytest.approx(4.0)


def test_touchdown_foot_velocity_includes_rigid_body_rotation():
    mass = 32_500.0
    props = mass_properties(mass, 1.0, PHYSICS)
    state = np.asarray(
        (
            0.0,
            props.cg_m - PHYSICS.gear.foot_station_m,
            0.0,
            -1.0,
            0.0,
            0.02,
            mass,
            20_000.0,
            0.0,
            0.0,
        )
    )
    points, velocities = foot_kinematics(state, 1.0, PHYSICS)
    np.testing.assert_allclose(points[:, 1], 0.0, atol=1e-10)
    assert velocities[0, 1] > -1.0
    assert velocities[1, 1] < -1.0
    assert np.ptp(velocities[:, 1]) == pytest.approx(2.0 * PHYSICS.gear.half_span_m * state[5])


def test_engine_pressure_performance_matches_the_100_t_design():
    design = Design()
    sea_thrust, sea_flow = engine_performance(
        1.0,
        0.0,
        PHYSICS,
        sea_level_thrust_n=design.sea_level_thrust_n,
        sea_level_isp_s=design.sea_level_isp_s,
    )
    vac_thrust, vac_flow = engine_performance(
        1.0,
        PROFILE_TOP_M,
        PHYSICS,
        sea_level_thrust_n=design.sea_level_thrust_n,
        sea_level_isp_s=design.sea_level_isp_s,
    )
    assert sea_thrust == pytest.approx(1_600_000.0)
    assert sea_flow == pytest.approx(1_600_000.0 / (300.0 * G0))
    assert vac_thrust == pytest.approx(1_760_000.0, abs=5.0)
    assert vac_thrust / (vac_flow * G0) == pytest.approx(330.0, abs=0.001)
    assert vac_flow == pytest.approx(sea_flow)


def test_engine_zero_throttle_has_no_thrust_or_propellant_flow():
    design = Design()
    for height in (0.0, 37_100.0, PROFILE_TOP_M):
        thrust, flow = engine_performance(
            0.0,
            height,
            PHYSICS,
            sea_level_thrust_n=design.sea_level_thrust_n,
            sea_level_isp_s=design.sea_level_isp_s,
        )
        assert thrust == flow == 0.0


def test_derivative_keeps_main_and_rcs_consumption_in_separate_inventory_channels():
    flight = AprFlight.from_separation(apr_separation_state(), time_s=98.0)
    flight.state[12] = 0.5
    command = (0.5, 0.0, flight.design.rcs_torque_limit_nm, 0.0)
    original = flight.state.copy()
    derivative = flight.derivative(flight.state, command)
    assert derivative.shape == flight.state.shape == (len(STATE_NAMES),)
    assert np.isfinite(derivative).all()
    np.testing.assert_array_equal(flight.state, original)
    assert derivative[6] == pytest.approx(-flight.engine(0.5, flight.state[1], actual=True)[1])
    expected_rcs_flow = command[2] / (
        flight.physics.geometry.rcs_arm_m * flight.design.rcs_isp_s * G0
    )
    assert derivative[7] == pytest.approx(-expected_rcs_flow)


def test_empty_rcs_inventory_removes_rcs_torque_and_consumption():
    flight = AprFlight.from_separation(apr_separation_state(), time_s=98.0)
    flight.state[7] = 0.0
    nominal = flight.derivative(flight.state, (0.0, 0.0, 0.0, 0.0))
    requested = flight.derivative(flight.state, (0.0, 0.0, flight.design.rcs_torque_limit_nm, 0.0))
    np.testing.assert_array_equal(requested, nominal)
    assert requested[7] == 0.0


def _inventory_event_bench():
    """Short event fixture with the complete current feed and slosh states."""
    flight = AprFlight.from_separation(apr_separation_state(), time_s=98.0)
    flight.next_control_s = flight.time_s + 0.05
    flight.next_record_s = flight.time_s + 100.0
    return flight


def test_rcs_exhaustion_is_an_exact_zero_stock_event_and_does_not_end_the_flight():
    flight = _inventory_event_bench()
    flight.state[7] = 0.0002
    flight.state[12] = 0.0
    flight.control = (0.0, 0.0, flight.design.rcs_torque_limit_nm, 0.0)
    flight.rcs_consumed = Design().rcs_propellant_kg - flight.state[7]
    started = flight.time_s
    initial_stock = flight.state[7]
    flow = flight.control[2] / (flight.physics.geometry.rcs_arm_m * flight.design.rcs_isp_s * G0)
    flight.advance()
    assert flight.outcome == "running"
    assert flight.state[7] == 0.0
    assert flight.time_s - started == pytest.approx(initial_stock / flow, abs=1e-8)
    assert flight.rcs_consumed == pytest.approx(Design().rcs_propellant_kg, abs=1e-8)
    assert flight.max_mass_balance_error < 1e-6
    consumed = flight.rcs_consumed
    flight.advance()
    assert flight.outcome == "running"
    assert flight.state[7] == 0.0
    assert flight.rcs_consumed == consumed


def test_first_stage_initial_thrust_to_weight_matches_the_mass_budget():
    design = Design()
    thrust_to_weight = design.sea_level_thrust_n / (design.launch_mass_kg * G0)
    assert thrust_to_weight == pytest.approx(1.631546, rel=1e-5)
    assert thrust_to_weight > 1.0


def _separation_event_regression_state():
    """Current propellant split at a stress pose for stale-control regression."""
    state = apr_separation_state()
    state[:8] = (
        10579.400581043896,
        37271.10055257472,
        445.1982537936092,
        964.1566240568388,
        0.4634664488795171,
        -0.0033839639399277388,
        22269.0810901671,
        297.71041647653897,
    )
    state[11:13] = (0.10083304193118577, 0.0009865513294684658)
    state[13:17] = astuple(initial_feed_state(state[6]))
    return state


def test_separation_event_invalidates_the_combined_vehicle_controller_schedule():
    flight = AprFlight()
    separated_state = _separation_event_regression_state()
    old_cg = flight.properties(separated_state).cg_m
    flight.upper_attached = False
    booster_properties = flight.properties(separated_state)
    new_cg = booster_properties.cg_m
    booster_mass = flight.mass(separated_state)
    flight.upper_attached = True
    arm = rotation(separated_state[4]) @ np.array((0.0, new_cg - old_cg))
    flight.state = separated_state.copy()
    flight.state[:2] -= arm
    flight.state[2:4] -= flight.state[5] * np.array((arm[1], -arm[0]))
    flight.time_s = 95.91
    flight.next_control_s = flight.time_s + 0.01
    flight.on_pad = False
    flight.cutoff_pending = True
    flight.ascent_max_q_passed = True
    before_inventory = flight.state[6:8].copy()
    assert all(flight.separation_checks(flight.state).values())
    flight.consider_separation()
    assert flight.upper_attached is False
    assert flight.phase == "flip"
    assert flight.separation["time_s"] == flight.time_s == 95.91
    assert flight.next_control_s == flight.time_s
    # Keep the original COM-frame transform check and include the newly
    # modelled pusher impulse; inventories still cannot jump at separation.
    separator = flight.separation_settings
    basis = rotation(separated_state[4])
    reduced_mass = (
        booster_mass * flight.design.upper_mass_kg / (booster_mass + flight.design.upper_mass_kg)
    )
    impulse = (
        reduced_mass
        * separator.relative_axial_speed_m_s
        * (basis[:, 1] + basis[:, 0] * np.tan(np.deg2rad(separator.impulse_misalignment_deg)))
    )
    expected = separated_state.copy()
    expected[2:4] -= impulse / booster_mass
    event = flight.separation["impulse_model"]
    expected[5] = event["booster_angular_rate_rad_s"]
    expected[18] = event["booster_slosh_velocity_m_s"]
    np.testing.assert_allclose(flight.state, expected, atol=1e-9)
    np.testing.assert_array_equal(flight.state[6:8], before_inventory)
    assert flight.steps == []


def test_post_separation_recomputed_control_removes_the_stale_stack_torque_spike():
    flight = AprFlight.from_separation(_separation_event_regression_state(), time_s=95.91)
    # Suppress modal participation only for this local actuator allocation bench.
    flight.slosh_settings = SloshSettings(modal_mass_fraction=1e-12)
    measured = flight.state.copy()
    stale_stack_control = (0.0, np.deg2rad(6.0), -100000.0, 0.0)
    flight.control = stale_stack_control
    stale_alpha = np.rad2deg(flight.derivative(measured, stale_stack_control)[5])
    updated_control = flight.guidance(measured)
    updated_alpha = np.rad2deg(flight.derivative(measured, updated_control)[5])
    assert abs(stale_alpha) > SAFETY.max_angular_acceleration_deg_s2
    assert updated_control[0] == 0.0
    assert abs(updated_control[2]) < abs(stale_stack_control[2])
    # Allocation balances the extrema of the held-interval TVC prediction,
    # not a single midpoint. Derive that actuator response independently and
    # account for the old/new RCS mass-flux feedforward difference explicitly.
    properties = flight.properties(measured)
    lever = properties.cg_m - flight.physics.geometry.nozzle_station_m
    thrust, main_flow = flight.engine(measured[12], measured[1])
    dt = flight.return_settings.control_dt_s
    supplied_tvc = []
    for instant in (0.0, dt / 2.0, dt):
        angle = measured[11] + np.clip(
            (updated_control[1] - measured[11]) * (1 - np.exp(-instant / 0.06)),
            -np.deg2rad(60) * instant,
            np.deg2rad(60) * instant,
        )
        for tau in (0.04, 0.05):
            supplied_tvc.append(-lever * thrust * np.exp(-instant / tau) * np.sin(angle))
    tvc_midrange = (min(supplied_tvc) + max(supplied_tvc)) / 2.0
    current_tvc = -lever * thrust * np.sin(measured[11])
    _, _, inertia_mass_gradient, _ = mass_gradients(
        flight.mass(measured), measured[8], flight.settings(), properties
    )
    rcs_flow_before = abs(stale_stack_control[2]) / (
        flight.physics.geometry.rcs_arm_m * flight.design.rcs_isp_s * G0
    )
    rcs_flow_after = abs(updated_control[2]) / (
        flight.physics.geometry.rcs_arm_m * flight.design.rcs_isp_s * G0
    )
    rcs_exit_radius_squared = (
        flight.physics.geometry.rcs_arm_m**2
        + (flight.physics.geometry.rcs_station_m - properties.cg_m) ** 2
    )
    old_inertia_rate = -inertia_mass_gradient * (main_flow + rcs_flow_before)
    old_flux = -(main_flow * lever**2 + rcs_flow_before * rcs_exit_radius_squared) * measured[5]
    balanced_alpha = (
        updated_control[2]
        + tvc_midrange
        + flight.loads(measured)[1]
        + old_flux
        - old_inertia_rate * measured[5]
    ) / properties.inertia_kg_m2
    # Initial-mass uncertainty changes fixed airframe mass, not the liquid
    # inventory. The acceleration reference uses this permitted lower inertia.
    lower_settings = replace(
        flight.settings(),
        geometry=replace(
            flight.settings().geometry,
            airframe_mass_kg=flight.settings().geometry.airframe_mass_kg - 1000.0,
        ),
    )
    inertia_lower = (
        0.97
        * mass_properties(flight.mass(measured) - 1000.0, measured[8], lower_settings).inertia_kg_m2
    )
    expected_reference_alpha = -2.5 * inertia_lower / properties.inertia_kg_m2
    assert np.rad2deg(balanced_alpha) == pytest.approx(expected_reference_alpha, abs=1e-9)
    flow_correction = (
        (rcs_flow_after - rcs_flow_before)
        * (inertia_mass_gradient - rcs_exit_radius_squared)
        * measured[5]
    )
    expected_initial_alpha = (
        balanced_alpha + (current_tvc - tvc_midrange + flow_correction) / properties.inertia_kg_m2
    )
    assert updated_alpha == pytest.approx(np.rad2deg(expected_initial_alpha), abs=1e-9)
    assert abs(updated_alpha) < SAFETY.max_angular_acceleration_deg_s2 == 5.0
    np.testing.assert_array_equal(flight.state, measured)
    assert flight.time_s == 95.91
    assert flight.steps == []


def test_missing_subsystems_are_disclosed_instead_of_claimed_as_certified():
    assert "tank_pressurization_and_feed" not in UNCONNECTED_SYSTEMS
    assert "physical_staging_impulse" not in UNCONNECTED_SYSTEMS
    assert "elastic_structure" in UNCONNECTED_SYSTEMS
    assert "detailed_heating_and_ground_damage" in UNCONNECTED_SYSTEMS
    assert "leg_material_strength_and_pad_bearing" in UNCONNECTED_SYSTEMS


@pytest.mark.parametrize("height", ISA_HEIGHTS[1:-1])
def test_layered_atmosphere_is_continuous_at_every_layer_boundary(height):
    lower = np.asarray(atmosphere(height - 1e-3, PHYSICS.atmosphere))
    upper = np.asarray(atmosphere(height + 1e-3, PHYSICS.atmosphere))
    np.testing.assert_allclose(lower, upper, rtol=1e-6)


@pytest.mark.parametrize(
    "height,temperature,pressure",
    (
        (0.0, 288.15, 101325.0),
        (11000.0, 216.65, 22632.06),
        (20000.0, 216.65, 5474.889),
        (32000.0, 228.65, 868.019),
        (47000.0, 270.65, 110.906),
    ),
)
def test_layered_atmosphere_matches_reference_values(height, temperature, pressure):
    actual_temperature, actual_pressure, density = atmosphere(height, PHYSICS.atmosphere)
    assert actual_temperature == pytest.approx(temperature, abs=0.01)
    assert actual_pressure == pytest.approx(pressure, rel=0.001)
    assert density == pytest.approx(actual_pressure / (R_AIR * actual_temperature))


def test_100_km_atmosphere_and_gravity_are_positive_and_decrease():
    heights = np.linspace(0.0, PROFILE_TOP_M, 501)
    values = np.asarray([atmosphere(height, PHYSICS.atmosphere) for height in heights])
    assert np.all(np.isfinite(values))
    assert np.all(values > 0.0)
    assert np.all(np.diff(values[:, 1]) < 0.0)
    assert np.all(np.diff(values[:, 2]) < 0.0)
    assert gravity(PROFILE_TOP_M) < gravity(0.0)


def test_baseline_operating_input_is_calm_without_hidden_uncertainties():
    conditions = AprConditions()
    disturbances = RecoveryDisturbances()
    for height in (0.0, 5000.0, 37100.0, PROFILE_TOP_M):
        for time_s in (0.0, 100.0, 300.0):
            assert conditions.wind(height, time_s) == 0.0
            assert conditions.wind(height, time_s, forecast=True) == 0.0
    assert conditions.density_scale == 1.0
    assert conditions.latitude_deg == 37.0
    assert (
        disturbances.thrust_scale
        == disturbances.drag_coefficient_scale
        == disturbances.initial_mass_scale
        == 1.0
    )
    assert disturbances.sensor_position_error_m == disturbances.sensor_velocity_error_m_s == 0.0


def test_public_example_config_uses_the_current_design_and_full_altitude_domain():
    conditions, disturbances = load_apr_operating(ROOT / "configs/apr-operating-v1.yaml")
    assert conditions.wind_profile[0][0] == 0.0
    assert conditions.wind_profile[-1][0] == PROFILE_TOP_M
    assert conditions.vertical_wind_profile[-1][0] == PROFILE_TOP_M
    assert 0.97 <= disturbances.thrust_scale <= 1.03


@pytest.mark.parametrize(
    "overrides",
    (
        {"health_checks_passed": False},
        {"known_fault": True},
        {"lightning_cloud_review_passed": False},
        {"temperature_review_passed": False},
        {"recovery_weather_review_passed": False},
        {"precipitation_mm_h": 0.01},
        {"thunderstorm_or_hail_present": True},
        {"health_checks_passed": 1},
        {"known_fault": 0},
        {"gust_amplitude_m_s": 5.01},
        {"density_scale": 0.899},
        {"density_scale": 1.101},
        {"gust_period_s": 7.99},
        {"gust_phase_rad": np.inf},
        {"latitude_deg": 91.0},
        {"vertical_wind_profile": ((0.0, 3.01), (PROFILE_TOP_M, 3.01))},
        {"vertical_wind_profile": ((0.0, -3.0), (1.0, 3.0), (PROFILE_TOP_M, 3.0))},
        {"wind_profile": ((0.0, 8.01), (PROFILE_TOP_M, 8.01))},
        {"wind_profile": ((0.0, 0.0), (10000.0, 20.01), (PROFILE_TOP_M, 0.0))},
        {"wind_profile": ((0.0, 0.0), (100.0, 0.0), (100.0, 0.0), (PROFILE_TOP_M, 0.0))},
        {"wind_profile": ((0.0, 0.0), (1000.0, 4.01), (PROFILE_TOP_M, 0.0))},
        {"wind_profile": ((0.0, 0.0), (20000.0, 0.0))},
        {"wind_profile": ((0.0, 0.0), (PROFILE_TOP_M, np.nan))},
    ),
)
def test_disallowed_weather_or_faults_are_rejected(overrides):
    with pytest.raises(ValueError):
        AprConditions(**overrides)


@pytest.mark.parametrize(
    "overrides",
    (
        {"thrust_scale": 0.969},
        {"thrust_scale": 1.031},
        {"drag_coefficient_scale": 0.849},
        {"drag_coefficient_scale": 1.151},
        {"initial_mass_scale": 0.989},
        {"initial_mass_scale": 1.011},
        {"inertia_scale": 0.969},
        {"inertia_scale": 1.031},
        {"sensor_position_error_m": 2.01},
        {"sensor_velocity_error_m_s": 1.01},
        {"observation_delay_steps": 3},
        {"observation_delay_steps": True},
        {"action_delay_steps": 2},
        {"engine_lag_s": 0.051},
    ),
)
def test_nonhealthy_vehicle_uncertainty_is_rejected(overrides):
    with pytest.raises(ValueError):
        RecoveryDisturbances(**overrides)


@pytest.mark.parametrize("scale", (0.99, 1.01))
def test_initial_mass_uncertainty_is_accounted_for_without_changing_propellant_inventory(scale):
    flight = AprFlight(disturbances=RecoveryDisturbances(initial_mass_scale=scale))
    assert flight.mass(flight.state) == pytest.approx(100_000.0 * scale)
    assert flight.design.launch_mass_kg == pytest.approx(100_000.0 * scale)
    assert flight.state[6] == Design().main_propellant_kg
    assert flight.state[7] == Design().rcs_propellant_kg
    assert flight.physics.geometry.airframe_mass_kg == pytest.approx(
        PHYSICS.geometry.airframe_mass_kg + 100_000.0 * (scale - 1.0)
    )


def test_sensor_measurement_error_vectors_are_seeded_and_respect_norm_limits():
    settings = RecoveryDisturbances(sensor_position_error_m=2.0, sensor_velocity_error_m_s=1.0)
    first, second = AprFlight(disturbances=settings), AprFlight(disturbances=settings)
    true_state = first.state.copy()
    for _ in range(100):
        measured = first.observe_control_state()
        repeated = second.observe_control_state()
        np.testing.assert_array_equal(measured, repeated)
        assert np.linalg.norm(measured[:2] - true_state[:2]) <= 2.0 + 1e-9
        assert np.linalg.norm(measured[2:4] - true_state[2:4]) <= 1.0 + 1e-9
        np.testing.assert_array_equal(measured[6:], true_state[6:])
        np.testing.assert_array_equal(first.state, true_state)


def _hidden_uncertainty_pair():
    """Same forecast and measured state; different unobserved plant parameters."""
    forecast = ((0.0, 0.0), (PROFILE_TOP_M, 0.0))
    first = AprFlight.from_separation(
        apr_separation_state(),
        time_s=98.0,
        conditions=AprConditions(
            wind_profile=((0.0, 1.0), (PROFILE_TOP_M, 1.0)),
            forecast_wind_profile=forecast,
            density_scale=1.1,
        ),
        disturbances=RecoveryDisturbances(
            initial_mass_scale=1.01,
            thrust_scale=1.03,
            drag_coefficient_scale=1.15,
            inertia_scale=1.03,
        ),
    )
    second = AprFlight.from_separation(
        apr_separation_state(),
        time_s=98.0,
        conditions=AprConditions(
            wind_profile=((0.0, -1.0), (PROFILE_TOP_M, -1.0)),
            forecast_wind_profile=forecast,
            density_scale=0.9,
        ),
        disturbances=RecoveryDisturbances(
            initial_mass_scale=0.99,
            thrust_scale=0.97,
            drag_coefficient_scale=0.85,
            inertia_scale=0.97,
        ),
    )
    measured = first.state.copy()
    measured[:6] = (500.0, 2000.0, 15.0, -240.0, 0.0, 0.0)
    measured[9:13] = (0.1, 1.0, 0.0, 0.15)
    return first, second, measured


def test_predicted_aerodynamic_and_fin_loads_do_not_read_hidden_plant_parameters():
    first, second, measured = _hidden_uncertainty_pair()
    for method in ("body_loads", "fin_loads", "loads"):
        left = getattr(first, method)(measured, predicted=True)
        right = getattr(second, method)(measured, predicted=True)
        for value, repeated in zip(left, right, strict=True):
            np.testing.assert_allclose(value, repeated, atol=1e-10)
    assert first.mass(measured) != second.mass(measured)
    assert first.mass(measured, predicted=True) == second.mass(measured, predicted=True)


def test_return_forecast_uses_the_same_observation_and_forecast_not_actual_density():
    first, second, measured = _hidden_uncertainty_pair()
    measured[:6] = (10_000.0, 37_000.0, 700.0, 600.0, 0.0, 0.0)
    left = first.predict_return(20.0, measured, direction=-1.0)
    right = second.predict_return(20.0, measured, direction=-1.0)
    assert left == right


@pytest.mark.parametrize("phase", ("coast", "landing"))
def test_guidance_uses_nominal_predictions_when_observations_and_forecasts_are_identical(phase):
    first, second, measured = _hidden_uncertainty_pair()
    first.phase = second.phase = phase
    if phase == "landing":
        measured[:6] = (1.0, 30.0, 0.3, -5.0, 0.0, 0.0)
    else:
        measured[:6] = (500.0, 5000.0, 15.0, -300.0, 0.0, 0.0)
    left = first.guidance(measured)
    right = second.guidance(measured)
    np.testing.assert_allclose(left, right, atol=1e-10)
    if phase == "coast":
        assert first.phase == second.phase == "entry"


@pytest.mark.parametrize("density_scale", (0.9, 1.0, 1.1))
def test_descent_reference_obeys_grid_pressure_and_backward_braking_bounds(density_scale):
    settings = AtmosphereSettings(density_scale=density_scale)
    q_budget = 0.6 * SAFETY.max_q_pa
    heights, squared, times = build_descent_reference(settings, q_budget, 20.0, 8.0)
    assert heights[0] == times[0] == 0.0
    assert heights[-1] == PROFILE_TOP_M
    assert squared[0] == Design().descent_reference_m_s ** 2 == 1.0
    assert np.all(np.diff(heights) > 0.0)
    assert np.all(np.diff(squared) >= 0.0)
    assert np.all(np.diff(times) > 0.0)
    density = np.array([atmosphere(height, settings)[2] for height in heights])
    assert np.all(0.5 * density * squared <= q_budget + 1e-8)
    blending = np.clip((heights - 1500.0) / 2000.0, 0.0, 1.0)
    braking = 8.0 + 12.0 * blending**2 * (3.0 - 2.0 * blending)
    slope = np.diff(squared) / np.diff(heights)
    assert np.all(slope <= braking[:-1] + braking[1:] + 1e-8)
    assert np.all(slope[heights[1:] <= 1500.0] <= 16.0 + 1e-8)
    assert np.all(slope <= 40.0 + 1e-8)
    np.testing.assert_allclose(
        np.diff(times), 2.0 * np.diff(heights) / (np.sqrt(squared[1:]) + np.sqrt(squared[:-1]))
    )
    assert not heights.flags.writeable
    assert not squared.flags.writeable
    assert not times.flags.writeable
    # These are guidance limits, not relaxed flight-safety acceptance criteria.
    assert SAFETY.max_q_pa == 30000.0
    assert SAFETY.max_recovery_flight_g == 5.0


def test_interpolated_descent_reference_keeps_a_physical_stopping_distance():
    flight = AprFlight()
    heights = np.r_[0.0, np.geomspace(0.001, PROFILE_TOP_M, 501)]
    previous_time = -1.0
    for height in heights:
        speed, gradient, remaining_time = flight.descent_speed(height)
        assert np.isfinite((speed, gradient, remaining_time)).all()
        assert speed >= Design().descent_reference_m_s
        assert 0.0 <= speed * gradient <= 20.0 + 1e-8
        stopping_distance = (speed**2 - Design().descent_reference_m_s ** 2) / (2.0 * 20.0)
        assert stopping_distance <= height + 1e-8
        if height <= 1500.0:
            assert speed * gradient <= 8.0 + 1e-8
            assert (speed**2 - 1.0) / 16.0 <= height + 1e-8
        density = atmosphere(height, flight.settings(predicted=True).atmosphere)[2]
        # Linear interpolation has small curvature error in its 18 kPa target;
        # the unchanged 30 kPa physical limit must still be respected between knots.
        assert 0.5 * density * speed**2 < SAFETY.max_q_pa
        assert remaining_time > previous_time
        previous_time = remaining_time
    assert flight.descent_speed(-1.0) == flight.descent_speed(0.0)
    assert flight.descent_speed(PROFILE_TOP_M + 1.0) == flight.descent_speed(PROFILE_TOP_M)


@pytest.mark.parametrize("height", (0.75, 101.0, 1703.25, 4444.0, 37123.0, 81234.0))
def test_descent_reference_speed_and_time_derivatives_match_the_interpolated_curve(height):
    flight = AprFlight()
    speed, gradient, _ = flight.descent_speed(height)
    dh = 0.0001
    lower_speed, _, lower_time = flight.descent_speed(height - dh)
    upper_speed, _, upper_time = flight.descent_speed(height + dh)
    assert (upper_speed - lower_speed) / (2.0 * dh) == pytest.approx(gradient, rel=1e-6)
    assert (upper_time - lower_time) / (2.0 * dh) == pytest.approx(1.0 / speed, rel=1e-6)


def _mock_return_prediction(landing_x, *, feasible=True):
    return dict(
        landing_x_m=landing_x,
        terminal_state=[landing_x, 2000.0, 0.0, -100.0],
        elapsed_s=200.0,
        remaining_main_kg=5000.0,
        peak_dynamic_pressure_pa=10000.0,
        feasible=feasible,
    )


@pytest.mark.parametrize("remaining_main,feasible", ((500.0, False), (20000.0, True)))
def test_return_predictor_budgets_the_terminal_burn_before_calling_a_plan_feasible(
    remaining_main, feasible
):
    flight = AprFlight.from_separation(apr_separation_state(), time_s=98.0)
    measured = flight.state.copy()
    measured[:6] = (1.0, 2000.0, 10.0, -180.0, 0.0, 0.0)
    measured[6] = remaining_main
    measured[12] = 0.0
    before = measured.copy()
    # At the predictor's terminal altitude no candidate propagation is needed.
    # This checks its conservative analytic budget, not a completed landing.
    prediction = flight.predict_return(0.0, measured)
    terminal_time = flight.descent_speed(measured[1])[2]
    delta_v = np.linalg.norm(measured[2:4]) + G0 * terminal_time
    expected_budget = flight.mass(measured, predicted=True) * (
        1.0 - np.exp(-delta_v / (Design().sea_level_isp_s * G0))
    )
    assert prediction["remaining_main_kg"] == remaining_main
    assert prediction["estimated_landing_propellant_kg"] == pytest.approx(expected_budget)
    assert prediction["estimated_main_after_landing_kg"] == pytest.approx(
        remaining_main - expected_budget
    )
    assert prediction["feasible"] is feasible
    np.testing.assert_array_equal(measured, before)


def test_finite_burn_prediction_does_not_read_hidden_plant_values_or_modify_real_state():
    first, second, measured = _hidden_uncertainty_pair()
    measured[:6] = (500.0, 12000.0, 35.0, -300.0, 0.0, 0.0)
    before_measurement = measured.copy()
    before_first, before_second = first.state.copy(), second.state.copy()
    left = first.predict_return(0.25, measured)
    right = second.predict_return(0.25, measured)
    assert left == right
    np.testing.assert_array_equal(measured, before_measurement)
    np.testing.assert_array_equal(first.state, before_first)
    np.testing.assert_array_equal(second.state, before_second)
    assert first.time_s == second.time_s == 98.0


def test_return_planner_rejects_unbracketed_miss_instead_of_executing_the_least_bad_burn(
    monkeypatch,
):
    flight = AprFlight.from_separation(apr_separation_state(), time_s=98.0)
    before = flight.state.copy()
    inventory_before = (flight.main_consumed, flight.rcs_consumed)
    evaluations = []

    def prediction(duration, observed):
        evaluations.append(duration)
        np.testing.assert_array_equal(observed, before)
        return _mock_return_prediction(1000.0 - 10.0 * duration)

    monkeypatch.setattr(flight, "predict_return", prediction)
    flight.plan_return(before)
    assert evaluations == [0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0]
    assert flight.return_plan["feasible"] is False
    assert flight.return_plan["reason"] == "no_bracketed_return_solution"
    assert flight.outcome == "return_plan_infeasible"
    assert flight.boost_end_s is None
    assert flight.return_plan_history == []
    np.testing.assert_array_equal(flight.state, before)
    assert (flight.main_consumed, flight.rcs_consumed) == inventory_before


@pytest.mark.parametrize("feasible", (False, True))
def test_return_planner_executes_only_a_bracketed_physically_feasible_prediction(
    monkeypatch, feasible
):
    flight = AprFlight.from_separation(apr_separation_state(), time_s=98.0)
    before = flight.state.copy()
    inventory_before = (flight.main_consumed, flight.rcs_consumed)
    monkeypatch.setattr(
        flight,
        "predict_return",
        lambda duration, observed: _mock_return_prediction(
            100.0 - 50.0 * duration, feasible=feasible
        ),
    )
    flight.plan_return(before)
    assert flight.return_plan["remaining_burn_s"] == pytest.approx(2.0, abs=0.005)
    assert flight.return_plan["predicted_ground_range_m"] == pytest.approx(0.0, abs=0.25)
    assert flight.return_plan["feasible"] is feasible
    assert flight.return_plan["touchdown_prediction_is_not_acceptance"] is True
    assert flight.return_plan_history == [flight.return_plan]
    if feasible:
        assert flight.boost_end_s == pytest.approx(100.0, abs=0.005)
        assert flight.next_plan_s == pytest.approx(99.0)
        assert flight.outcome == "running"
    else:
        assert flight.boost_end_s is None
        assert flight.outcome == "return_plan_infeasible"
    np.testing.assert_array_equal(flight.state, before)
    assert (flight.main_consumed, flight.rcs_consumed) == inventory_before


def test_return_planner_does_not_confuse_small_position_miss_with_physical_feasibility(monkeypatch):
    flight = AprFlight.from_separation(apr_separation_state(), time_s=98.0)
    monkeypatch.setattr(
        flight,
        "predict_return",
        lambda duration, observed: _mock_return_prediction(1.0, feasible=False),
    )
    flight.plan_return(flight.state.copy())
    assert flight.return_plan["remaining_burn_s"] == 0.0
    assert flight.return_plan["feasible"] is False
    assert flight.outcome == "return_plan_infeasible"
    assert flight.boost_end_s is None


def test_attitude_feedforward_compensates_the_actual_fin_moment_exactly_once(monkeypatch):
    flight = AprFlight.from_separation(apr_separation_state(), time_s=98.0)
    # Isolate the torque allocator while retaining all current subsystem states.
    flight.slosh_settings = SloshSettings(modal_mass_fraction=1e-12)
    measured = flight.state.copy()
    measured[:6] = (0.0, 5000.0, 0.0, -200.0, 0.0, 0.0)
    measured[9:13] = (0.1, 1.0, 0.0, 0.5)
    fin_calls, demands = [], []
    monkeypatch.setattr(
        flight, "body_loads", lambda *args, **kwargs: (np.zeros(2), 50.0, 1000.0, 0.0)
    )

    def fin_loads(*args, **kwargs):
        fin_calls.append(kwargs)
        return np.zeros(2), 100.0, 0.0

    def fin_command(observed, requested):
        demands.append(requested)
        return 0.2

    monkeypatch.setattr(flight, "fin_loads", fin_loads)
    monkeypatch.setattr(flight, "fin_command", fin_command)
    gimbal, rcs, target = flight.attitude_control(0.0, 0.5, measured)
    assert demands == [-50.0]
    # One sizing midpoint and three held-interval samples evaluate the same
    # physical fin; these are alternative instants, not four added torques.
    assert fin_calls == [{"predicted": True}] * 4
    assert target == 0.2
    properties = flight.properties(measured)
    lever = properties.cg_m - flight.physics.geometry.nozzle_station_m
    thrust = flight.engine(0.5, measured[1])[0]
    midpoint = flight.return_settings.control_dt_s / 2
    expected_gimbal = np.arcsin(150.0 / (lever * thrust))
    mid_gimbal = np.clip(
        expected_gimbal * (1 - np.exp(-midpoint / 0.06)),
        -np.deg2rad(60) * midpoint,
        np.deg2rad(60) * midpoint,
    )
    mid_tvc = -lever * thrust * np.sin(mid_gimbal)
    dt = flight.return_settings.control_dt_s
    end_gimbal = expected_gimbal * (1 - np.exp(-dt / 0.06))
    end_tvc = -lever * thrust * np.sin(end_gimbal)
    tvc_midrange = end_tvc / 2.0  # Initial gimbal/TVC torque is zero.
    assert gimbal == pytest.approx(expected_gimbal, abs=1e-12)
    assert rcs == pytest.approx(-150.0 - tvc_midrange, abs=1e-9)
    assert gimbal != 0.0
    # The held RCS cancels body + one fin + the TVC interval midrange.
    assert 50.0 + 100.0 + rcs + tvc_midrange == pytest.approx(0.0, abs=1e-9)
    assert 150.0 + rcs == pytest.approx(-(150.0 + rcs + end_tvc), abs=1e-9)
    derivative = flight.derivative(measured, (0.5, gimbal, rcs, target))
    assert derivative[5] == pytest.approx((150.0 + rcs) / properties.inertia_kg_m2, abs=1e-12)
    assert abs(np.rad2deg(derivative[5])) < SAFETY.max_angular_acceleration_deg_s2 == 5.0
    predicted_midpoint = measured.copy()
    predicted_midpoint[11] = mid_gimbal
    midpoint_derivative = flight.derivative(predicted_midpoint, (0.5, gimbal, rcs, target))
    assert midpoint_derivative[5] == pytest.approx(
        (mid_tvc - tvc_midrange) / properties.inertia_kg_m2, abs=1e-12
    )


def test_wind_and_forecast_are_validated_between_different_knots():
    with pytest.raises(ValueError):
        AprConditions(
            wind_profile=((0.0, 0.0), (PROFILE_TOP_M, 0.0)),
            forecast_wind_profile=((0.0, 0.0), (1000.0, 2.01), (PROFILE_TOP_M, 0.0)),
        )


def test_surface_mean_wind_is_not_misused_as_a_continuous_13_m_s_wind():
    profile = ((0.0, 8.0), (50.0, 8.0), (10000.0, 20.0), (PROFILE_TOP_M, 20.0))
    conditions = AprConditions(
        wind_profile=profile,
        forecast_wind_profile=profile,
        gust_amplitude_m_s=5.0,
        gust_phase_rad=np.pi / 2,
        gust_start_s=-60.0,
        gust_duration_s=120.0,
        gust_altitude_center_m=0.0,
        gust_altitude_half_width_m=5000.0,
    )
    assert conditions.mean_wind(0.0) == 8.0
    assert conditions.wind(0.0, 0.0) == pytest.approx(13.0)
    assert conditions.wind(0.0, 121.0) == 8.0
    assert conditions.mean_wind(37100.0) == 20.0


@pytest.mark.parametrize("scale", (0.9, 1.1))
def test_density_band_is_relative_to_fixed_reference_across_the_full_domain(scale):
    conditions = AprConditions(density_scale=scale)
    perturbed = AtmosphereSettings(density_scale=conditions.density_scale)
    for height in np.linspace(0.0, PROFILE_TOP_M, 101):
        reference = atmosphere(height, AtmosphereSettings())[2]
        assert atmosphere(height, perturbed)[2] / reference == pytest.approx(scale)


def test_combined_temperature_pressure_and_density_cannot_exceed_the_final_band():
    with pytest.raises(ValueError):
        AprConditions(
            sea_level_temperature_k=285.0, sea_level_pressure_pa=103000.0, density_scale=1.1
        )


def test_vehicle_configuration_must_match_the_fixed_100_t_design(tmp_path):
    path = _write_config(tmp_path, lambda data: data["vehicle"].update(dry_mass_kg=12_100.0))
    with pytest.raises(ValueError):
        load_apr_operating(path)


def test_operating_manifest_is_versioned_for_the_100_t_model_and_eligible():
    manifest = make_manifest()
    assert manifest == make_manifest()
    assert manifest["schema_version"] == MANIFEST_SCHEMA_VERSION == 13
    assert manifest["root_seed"] == ROOT_SEED == 378
    assert manifest["vehicle"] == VEHICLE_ID
    assert manifest["operating_scope"] == OPERATING_SCOPE
    assert manifest["nominal_vehicle"] == asdict(Design())
    assert len(manifest["development"]) == 133
    assert len(manifest["holdout"]) == 72
    cases = manifest["development"] + manifest["holdout"]
    assert len({case["id"] for case in cases}) == len(cases)
    assert all(case["stream"] == 0 for case in manifest["development"])
    assert all(case["stream"] == 1 for case in manifest["holdout"][:64])
    assert all(case["stream"] == 2 for case in manifest["holdout"][64:])
    for case in cases:
        conditions = AprConditions(**case["conditions"])
        RecoveryDisturbances(**case["disturbances"])
        assert conditions.wind_profile[-1][0] == PROFILE_TOP_M
        assert conditions.vertical_wind_profile[-1][0] == PROFILE_TOP_M
        assert abs(conditions.mean_wind(0.0)) <= 8.0
        assert abs(conditions.mean_wind(37100.0)) <= 20.0


def test_expanded_manifest_preserves_every_original_input_and_observation_identity():
    manifest = make_manifest()
    assert _sha(manifest["development"][:85]) == (
        "2dae66a1d08f34e5307d8e96b0a24a7604e3a58a96e006364a3a4c5f793f6d6e"
    )
    assert _sha(manifest["holdout"][:32]) == (
        "8fa79f2b028c10436818ce89a27e876506c4deeee81ff02a0c0f40eaa932e59d"
    )
    assert len(manifest["development"][85:117]) == 32
    groups = {case["group"] for case in manifest["holdout"][:64]}
    assert groups == {
        "surface",
        "upper",
        "shear",
        "gust",
        "vertical",
        "density",
        "vehicle",
        "combined",
    }
    for group in groups:
        assert sorted(
            case["index"] for case in manifest["holdout"][:64] if case["group"] == group
        ) == (list(range(8)))


def test_terminal_extension_preserves_all_prior_cases_before_adding_new_inputs():
    manifest = make_manifest()
    assert _sha(manifest["development"][:117]) == (
        "b31e46392b8ff6287d4eb1ddb5ba6528fe518454193fe5d517206b1cce8f54dd"
    )
    assert _sha(manifest["holdout"][:64]) == (
        "98378cf330358fff660a13eec855777135a632d8cdde69c1cf7732ca421dfbc2"
    )
    assert len(manifest["development"][117:]) == 16
    assert len(manifest["holdout"][64:]) == 8


@pytest.mark.parametrize("label", ("plus", "minus"))
@pytest.mark.parametrize("event", ("short-entry", "short-landing", "thin-entry", "thin-landing"))
def test_delayed_gust_cases_keep_the_original_weather_and_add_maximum_healthy_delay(label, event):
    cases = {case["id"]: case for case in make_manifest()["development"]}
    original = cases[f"gust-{event}-{label}"]
    delayed = cases[f"gust-{event}-delayed-{label}"]
    assert delayed["conditions"] == original["conditions"]
    assert delayed["disturbances"] == cases["late-corner-reserve-fast-plus"]["disturbances"]
    assert delayed["disturbances"]["observation_delay_steps"] == 2
    assert delayed["disturbances"]["action_delay_steps"] == 1
    assert delayed["disturbances"]["engine_lag_s"] == 0.05
    assert observation_seed_sequence(delayed).entropy != observation_seed_sequence(original).entropy


@pytest.mark.parametrize("label", ("plus", "minus"))
@pytest.mark.parametrize("corner", ("reserve", "load"))
@pytest.mark.parametrize("speed", ("fast", "slow"))
def test_terminal_gust_windows_extend_coverage_without_changing_the_earlier_corner(
    label, corner, speed
):
    cases = {case["id"]: case for case in make_manifest()["development"]}
    original = cases[f"late-corner-{corner}-{speed}-{label}"]
    terminal = cases[f"touchdown-corner-{corner}-{speed}-{label}"]
    expected = original["conditions"].copy()
    expected.update(gust_start_s=370.0, gust_duration_s=120.0)
    assert terminal["conditions"] == expected
    assert terminal["disturbances"] == original["disturbances"]
    assert original["conditions"]["gust_start_s"] == 310.0
    assert original["conditions"]["gust_duration_s"] == 120.0
    assert (
        observation_seed_sequence(terminal).entropy != observation_seed_sequence(original).entropy
    )


def test_new_terminal_holdout_inputs_have_an_independent_stream_and_fixed_distribution():
    from scripts import evaluate_operating_envelope as campaign_cli

    manifest = make_manifest()
    cases = manifest["holdout"][64:]
    assert [case["index"] for case in cases] == list(range(8))
    starts = []
    for case in cases:
        assert case["stream"] == 2
        assert case["group"] == "combined"
        assert case["id"] == f"late-combined-2-{case['index']:02d}"
        original = campaign_cli._random_case("combined", 2, case["index"])
        expected_conditions = original["conditions"].copy()
        expected_conditions.update(
            gust_start_s=case["conditions"]["gust_start_s"],
            gust_duration_s=120.0,
            gust_altitude_center_m=0.0,
            gust_altitude_half_width_m=5000.0,
        )
        assert case["conditions"] == expected_conditions
        assert 370.0 <= case["conditions"]["gust_start_s"] <= 390.0
        assert 8.0 <= case["conditions"]["gust_period_s"] <= 60.0
        assert 0.2 <= case["conditions"]["gust_amplitude_m_s"] <= 5.0
        expected_disturbances = original["disturbances"].copy()
        expected_disturbances.update(
            observation_delay_steps=2, action_delay_steps=1, engine_lag_s=0.05
        )
        assert case["disturbances"] == expected_disturbances
        starts.append(case["conditions"]["gust_start_s"])
    assert len(set(starts)) == 8
    assert cases == make_manifest()["holdout"][64:]


@pytest.mark.parametrize("label,sign", (("plus", 1.0), ("minus", -1.0)))
@pytest.mark.parametrize(
    "phase_name,start,center,duration",
    (("boostback", 125.0, 70000.0, 30.0), ("coast-high", 180.0, 85000.0, 120.0)),
)
def test_high_altitude_gusts_are_predeclared_without_claiming_actual_phase_exposure(
    label, sign, phase_name, start, center, duration
):
    cases = {case["id"]: case for case in make_manifest()["development"]}
    conditions = AprConditions(**cases[f"gust-{phase_name}-{label}"]["conditions"])
    assert conditions.gust_start_s == start
    assert conditions.gust_altitude_center_m == center
    assert conditions.gust_duration_s == duration
    assert conditions.gust_altitude_half_width_m == 5000.0
    assert conditions.gust_amplitude_m_s == 5.0
    assert conditions.gust_phase_rad == sign * np.pi / 2
    assert conditions.forecast_gust_amplitude_m_s == 0.0
    assert conditions.wind(center + 5000.0, start + 5.0) == 0.0


@pytest.mark.parametrize("label,sign", (("plus", 1.0), ("minus", -1.0)))
@pytest.mark.parametrize(
    "event,start,center,duration,width,peak_time",
    (
        ("short-entry", 319.0, 20000.0, 8.0, 5000.0, 324.8),
        ("short-landing", 398.0, 0.0, 8.0, 5000.0, 400.0),
        ("thin-entry", 318.0, 20000.0, 30.0, 100.0, 324.8),
        ("thin-landing", 390.0, 125.0, 60.0, 100.0, 400.0),
    ),
)
def test_short_and_thin_gust_inputs_have_the_intended_geometry_not_a_flight_success(
    label, sign, event, start, center, duration, width, peak_time
):
    cases = {case["id"]: case for case in make_manifest()["development"]}
    conditions = AprConditions(**cases[f"gust-{event}-{label}"]["conditions"])
    assert conditions.gust_start_s == start
    assert conditions.gust_duration_s == duration
    assert conditions.gust_altitude_half_width_m == width
    assert conditions.gust_period_s == 8.0
    assert conditions.wind(center, peak_time) == pytest.approx(sign * 5.0)
    assert conditions.wind(center + width, peak_time) == 0.0
    assert conditions.wind(center, start) == 0.0
    assert conditions.wind(center, start + duration) == 0.0


@pytest.mark.parametrize("label,sign", (("plus", 1.0), ("minus", -1.0)))
@pytest.mark.parametrize(
    "corner,thrust,mass,inertia", (("reserve", 0.97, 1.01, 1.03), ("load", 1.03, 0.99, 0.97))
)
@pytest.mark.parametrize("speed_name,period", (("fast", 8.0), ("slow", 60.0)))
def test_late_gust_corners_keep_wind_and_all_healthy_uncertainties_at_the_boundary(
    label, sign, corner, thrust, mass, inertia, speed_name, period
):
    cases = {case["id"]: case for case in make_manifest()["development"]}
    case = cases[f"late-corner-{corner}-{speed_name}-{label}"]
    conditions = AprConditions(**case["conditions"])
    assert conditions.mean_wind(0.0) == sign * 8.0
    assert conditions.mean_wind(10000.0) == sign * 20.0
    assert conditions.gust_start_s == 310.0
    assert conditions.gust_altitude_center_m == 0.0
    assert conditions.gust_altitude_half_width_m == 5000.0
    assert conditions.gust_amplitude_m_s == 5.0
    assert conditions.gust_period_s == period
    assert conditions.density_scale == 1.1
    assert RecoveryDisturbances(**case["disturbances"]) == RecoveryDisturbances(
        thrust_scale=thrust,
        drag_coefficient_scale=1.15,
        initial_mass_scale=mass,
        inertia_scale=inertia,
        sensor_position_error_m=2.0,
        sensor_velocity_error_m_s=1.0,
        observation_delay_steps=2,
        action_delay_steps=1,
        engine_lag_s=0.05,
    )


@pytest.mark.parametrize("label", ("plus", "minus"))
@pytest.mark.parametrize(
    "thermal,temperature,pressure,target",
    (
        ("cold-low", 273.15, 95000.0, 0.9),
        ("cold-high", 273.15, 105000.0, 1.1),
        ("warm-low", 308.15, 95000.0, 0.9),
        ("warm-high", 308.15, 105000.0, 1.1),
    ),
)
def test_thermal_corners_respect_the_coupled_final_density_band(
    label, thermal, temperature, pressure, target
):
    from powered_landing_guidance.apr_recovery import _density_ratio_extrema

    cases = {case["id"]: case for case in make_manifest()["development"]}
    case = cases[f"thermal-corner-{thermal}-{label}"]
    conditions = AprConditions(**case["conditions"])
    assert conditions.sea_level_temperature_k == temperature
    assert conditions.sea_level_pressure_pa == pressure
    assert 0.9 <= conditions.density_scale <= 1.1
    minimum, maximum = _density_ratio_extrema(
        AtmosphereSettings(
            sea_level_temperature_k=temperature,
            sea_level_pressure_pa=pressure,
            temperature_profile=conditions.temperature_profile,
            density_scale=conditions.density_scale,
        )
    )
    assert minimum >= 0.9 - 1e-10
    assert maximum <= 1.1 + 1e-10
    assert (minimum if target < 1.0 else maximum) == pytest.approx(target)
    deviations = RecoveryDisturbances(**case["disturbances"])
    assert deviations.observation_delay_steps == 2
    assert deviations.action_delay_steps == 1
    assert deviations.engine_lag_s == 0.05


@pytest.mark.parametrize("label,sign", (("plus", 1.0), ("minus", -1.0)))
@pytest.mark.parametrize("layer", ("low", "upper"))
def test_reversal_cases_include_horizontal_and_vertical_shear_boundaries(label, sign, layer):
    cases = {case["id"]: case for case in make_manifest()["development"]}
    conditions = AprConditions(**cases[f"shear-reversal-{layer}-{label}"]["conditions"])
    horizontal = np.array(conditions.wind_profile)
    gradients = np.diff(horizontal[:, 1]) / np.diff(horizontal[:, 0]) * 1000
    assert np.max(abs(gradients)) == pytest.approx(4.0)
    assert conditions.mean_wind(20000.0) == -sign * 20.0
    assert conditions.vertical_wind(0.0) == -sign * 3.0
    assert conditions.vertical_wind(2000.0) == sign * 3.0
    assert (conditions.vertical_wind(2000.0) - conditions.vertical_wind(0.0)) / 2 == sign * 3.0


def test_manifest_with_valid_hash_but_mismatched_vehicle_identity_is_rejected(tmp_path):
    campaign = tmp_path / "campaign"
    prepare(campaign)
    path = campaign / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.pop("manifest_sha256")
    manifest["vehicle"] = "different conceptual vehicle"
    manifest["manifest_sha256"] = _sha(manifest)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="incompatible"):
        _load_manifest(campaign)


def test_evaluator_rejects_contradictory_success_without_running_a_flight(monkeypatch):
    from scripts import evaluate_operating_envelope as campaign_cli

    class ContradictoryFlight:
        def __init__(self, **kwargs):
            self.rng = None

        def run(self):
            return {
                "mission_success": True,
                "flight_safety": False,
                "landing_success": True,
                "simulated_acceptance_passed": True,
            }

    monkeypatch.setattr(campaign_cli, "AprFlight", ContradictoryFlight)
    case = make_manifest()["development"][0]
    with pytest.raises(ValueError, match="contradicts"):
        campaign_cli._evaluate_case(case)


def test_evaluator_cannot_accept_full_flight_without_separation_and_contact(monkeypatch):
    from scripts import evaluate_operating_envelope as campaign_cli

    class MissingEvidenceFlight:
        def __init__(self, **kwargs):
            self.rng = None

        def run(self):
            return {
                "mission_success": True,
                "flight_safety": True,
                "landing_success": True,
                "simulated_acceptance_passed": True,
                "separation": None,
                "contact": None,
            }

    monkeypatch.setattr(campaign_cli, "AprFlight", MissingEvidenceFlight)
    case = make_manifest()["development"][0]
    with pytest.raises(ValueError, match="requires recorded"):
        campaign_cli._evaluate_case(case)


def test_campaign_adapter_records_a_zero_duration_failure_without_running_a_flight(monkeypatch):
    from scripts import evaluate_operating_envelope as campaign_cli

    # Only exercise output construction: the integration loop must not execute.
    flight = AprFlight()
    flight.deadline_s = 0.0
    result = flight.run()
    monkeypatch.setattr(AprFlight, "run", lambda self: result)
    case = make_manifest()["development"][0]
    entry = campaign_cli._evaluate_case(case)
    assert entry["accepted"] is False
    assert entry["nominal_launch_mass_kg"] == entry["actual_launch_mass_kg"] == 100_000.0
    assert entry["flight_time_s"] == 0.0
    assert entry["separation"] is entry["contact"] is None
    assert {"separation", "touchdown"}.issubset(entry["failed_criteria"])
    assert entry["remaining_main_propellant_kg"] == 70_200.0
    assert entry["remaining_rcs_propellant_kg"] == 300.0
    assert entry["unconnected_systems"] == list(UNCONNECTED_SYSTEMS)
    assert entry["landing_assessment"] == "kinematic_touchdown"
    assert entry["acceptance_version"] == "apr_100t_kinematic_touchdown_v1"
    assert entry["assessment_scope"]["contact_structural_dynamics"] is False
    assert json.loads(_encoded(entry))["accepted"] is False


def test_campaign_gust_evidence_prefers_integration_endpoints_over_sparse_records(monkeypatch):
    from scripts import evaluate_operating_envelope as campaign_cli

    flight = AprFlight()
    flight.deadline_s = 0.0
    result = flight.run()
    result["integration_gust_exposure"] = {
        "maximum_contribution_m_s": 4.9,
        "maximum_by_phase_m_s": {"entry": 4.9},
        "endpoints_above_0_1_m_s": 24,
        "first_endpoint_above_0_1_m_s": 324.66,
        "last_endpoint_above_0_1_m_s": 324.9,
        "gust_encounter_observed": True,
        "sampling_note": "Synthetic adapter fixture, not a simulated gust encounter",
    }
    monkeypatch.setattr(AprFlight, "run", lambda self: result)
    case = next(
        case for case in make_manifest()["development"] if case["id"] == "gust-thin-entry-plus"
    )
    entry = campaign_cli._evaluate_case(case)
    assert entry["gust_exposure"]["maximum_sampled_contribution_m_s"] == 0.0
    assert entry["gust_exposure"]["gust_encounter_observed"] is True
    assert entry["gust_exposure"]["integration_endpoints"] == result["integration_gust_exposure"]
    assert entry["accepted"] is False


def test_campaign_gust_evidence_retains_sparse_fallback_for_older_fixtures(monkeypatch):
    from scripts import evaluate_operating_envelope as campaign_cli

    flight = AprFlight()
    flight.deadline_s = 0.0
    result = flight.run()
    result.pop("integration_gust_exposure", None)
    monkeypatch.setattr(AprFlight, "run", lambda self: result)
    entry = campaign_cli._evaluate_case(make_manifest()["development"][0])
    assert entry["gust_exposure"]["gust_encounter_observed"] is False
    assert "integration_endpoints" not in entry["gust_exposure"]


def test_shared_atmosphere_is_continuous_at_tropopause():
    sea = atmosphere(0.0, PHYSICS.atmosphere)
    assert sea[0] == PHYSICS.atmosphere.sea_level_temperature_k
    assert sea[1] == PHYSICS.atmosphere.sea_level_pressure_pa
    lower = atmosphere(11000.0 - 1e-3, PHYSICS.atmosphere)
    upper = atmosphere(11000.0 + 1e-3, PHYSICS.atmosphere)
    np.testing.assert_allclose(lower, upper, rtol=1e-6)
    assert upper[2] < sea[2]


def test_shared_clockwise_force_and_torque_convention():
    np.testing.assert_allclose(rotation(np.pi / 2) @ [0.0, 1.0], [1.0, 0.0], atol=1e-12)
    assert clock_torque(np.array([0.0, 2.0]), np.array([3.0, 0.0])) == 6.0
    assert clock_torque(np.array([2.0, 0.0]), np.array([0.0, 3.0])) == -6.0


def test_rk4_freefall_matches_the_analytic_solution_and_does_not_mutate_input():
    initial = np.array([100.0, -2.0])
    final = rk4_step(lambda time, state: [state[1], -G0], 0.0, initial, 0.3)
    np.testing.assert_allclose(final, [100.0 - 2.0 * 0.3 - 0.5 * G0 * 0.3**2, -2.0 - G0 * 0.3])
    np.testing.assert_array_equal(initial, [100.0, -2.0])


@pytest.mark.parametrize("dt", [0.0, -1.0, np.nan, np.inf])
def test_rk4_rejects_invalid_time_steps(dt):
    with pytest.raises(ValueError, match="dt_s"):
        rk4_step(lambda time, state: state, 0.0, [1.0], dt)


@pytest.mark.parametrize("derivative", [[np.nan], [1.0, 2.0], [[1.0]]])
def test_rk4_rejects_invalid_derivatives(derivative):
    with pytest.raises(ValueError, match="derivative"):
        rk4_step(lambda time, state: derivative, 0.0, [1.0], 0.1)


def test_operating_cases_have_distinct_deterministic_observation_streams():
    manifest = make_manifest()
    cases = manifest["development"] + manifest["holdout"]
    assert len(cases) == 205
    seed_words = [tuple(observation_seed_sequence(case).entropy) for case in cases]
    first_draws = [
        int(np.random.default_rng(observation_seed_sequence(case)).integers(0, 2**63))
        for case in cases
    ]
    assert len(set(seed_words)) == len(cases)
    assert len(set(first_draws)) == len(cases)
    assert seed_words == [tuple(observation_seed_sequence(case).entropy) for case in cases]


@pytest.mark.parametrize("label,sign", (("plus", 1.0), ("minus", -1.0)))
def test_operating_load_corners_cover_light_vehicle_and_strong_engine(label, sign):
    cases = {case["id"]: case for case in make_manifest()["development"]}
    case = cases[f"corner-load-{label}"]
    conditions = AprConditions(**case["conditions"])
    disturbances = RecoveryDisturbances(**case["disturbances"])
    assert conditions.mean_wind(0.0) == sign * 8.0
    assert conditions.mean_wind(10_000.0) == sign * 20.0
    assert conditions.density_scale == 1.1
    assert disturbances.drag_coefficient_scale == 1.15
    assert disturbances.initial_mass_scale == 0.99
    assert disturbances.thrust_scale == 1.03
    assert disturbances.sensor_position_error_m == 2.0
    assert disturbances.sensor_velocity_error_m_s == 1.0
    assert observation_seed_sequence(case).generate_state(4).tolist() != (
        observation_seed_sequence(cases[f"corner-{label}"]).generate_state(4).tolist()
    )


def test_operating_manifest_retains_exact_hidden_gust_regression_profiles():
    cases = {case["id"]: case for case in make_manifest()["development"]}
    original = AprConditions(**cases["upper-minus"]["conditions"])
    for actual_upper, forecast_upper in (
        (-19.0, -17.0),
        (-18.0, -16.0),
        (-14.0, -12.0),
        (-10.0, -8.0),
        (-6.0, -4.0),
        (0.0, 0.0),
    ):
        case = cases[f"hidden-gust-upper-{abs(int(actual_upper)):02d}"]
        conditions = AprConditions(**case["conditions"])
        assert conditions.wind_profile[:-2] == original.wind_profile[:-2]
        assert conditions.forecast_wind_profile[:-2] == original.wind_profile[:-2]
        assert conditions.wind_profile[-2:] == (
            (10000.0, actual_upper),
            (PROFILE_TOP_M, actual_upper),
        )
        assert conditions.forecast_wind_profile[-2:] == (
            (10000.0, forecast_upper),
            (PROFILE_TOP_M, forecast_upper),
        )
        assert conditions.gust_amplitude_m_s == 5.0
        assert conditions.forecast_gust_amplitude_m_s == 0.0
        assert conditions.gust_phase_rad == original.gust_phase_rad
        assert conditions.gust_start_s == original.gust_start_s
        assert conditions.gust_duration_s == original.gust_duration_s
        assert conditions.gust_altitude_center_m == original.gust_altitude_center_m
        assert conditions.gust_altitude_half_width_m == original.gust_altitude_half_width_m


def test_operating_forecast_rise_regressions_keep_actual_weather_and_distinct_streams():
    cases = {case["id"]: case for case in make_manifest()["development"]}
    original_case = cases["hidden-gust-upper-06"]
    original = AprConditions(**original_case["conditions"])
    for label, rise, middle, upper in (
        ("00", 0.0, -8.0, -8.0),
        ("05", 0.5, -8.8333333333, -7.1666666667),
        ("10", 1.0, -9.6666666667, -6.3333333333),
        ("15", 1.5, -10.5, -5.5),
    ):
        case = cases[f"forecast-rise-{label}"]
        conditions = AprConditions(**case["conditions"])
        actual_fields = asdict(conditions)
        original_fields = asdict(original)
        actual_fields.pop("forecast_wind_profile")
        original_fields.pop("forecast_wind_profile")
        assert actual_fields == original_fields
        assert conditions.forecast_wind_profile[:3] == original.forecast_wind_profile[:3]
        assert conditions.forecast_wind_profile[-3:] == (
            (5000.0, middle),
            (10000.0, upper),
            (PROFILE_TOP_M, upper),
        )
        assert conditions.mean_wind(10000.0, forecast=True) - conditions.mean_wind(
            8500.0, forecast=True
        ) == pytest.approx(rise)
        assert (
            observation_seed_sequence(case).entropy
            != observation_seed_sequence(original_case).entropy
        )


def test_operating_corner_regressions_combine_allowed_environment_and_vehicle_extremes():
    cases = {case["id"]: case for case in make_manifest()["development"]}
    for index, (name, sign) in enumerate((("corner-plus", 1.0), ("corner-minus", -1.0))):
        case = cases[name]
        conditions = AprConditions(**case["conditions"])
        disturbances = RecoveryDisturbances(**case["disturbances"])
        assert case["group"] == "boundary"
        assert case["stream"] == 0
        assert case["index"] == 900 + index
        assert conditions.mean_wind(0.0) == sign * 8.0
        assert conditions.mean_wind(5000.0) == sign * 14.0
        assert conditions.mean_wind(10000.0) == sign * 20.0
        assert conditions.forecast_wind_profile == conditions.wind_profile
        assert conditions.gust_amplitude_m_s == 5.0
        assert conditions.forecast_gust_amplitude_m_s == 0.0
        assert conditions.gust_phase_rad == sign * np.pi / 2
        assert conditions.gust_start_s == 0.0
        assert conditions.gust_duration_s == 120.0
        assert conditions.gust_altitude_center_m == 10000.0
        assert conditions.gust_altitude_half_width_m == 2500.0
        assert conditions.vertical_wind_profile == ((0.0, sign * 3.0), (PROFILE_TOP_M, sign * 3.0))
        assert conditions.density_scale == 1.1
        assert disturbances == RecoveryDisturbances(
            thrust_scale=0.97,
            drag_coefficient_scale=1.15,
            initial_mass_scale=1.01,
            inertia_scale=1.03,
            sensor_position_error_m=2.0,
            sensor_velocity_error_m_s=1.0,
            observation_delay_steps=2,
            action_delay_steps=1,
            engine_lag_s=0.05,
        )


def test_operating_failure_stage_classification_does_not_call_input_eligibility_success():
    assert _stage({"separation": None, "contact": None}) == "ascent_or_separation"
    assert _stage({"separation": {}, "contact": None}) == "return_or_approach"
    assert (
        _stage({"separation": {}, "contact": {}, "landing_assessment": "kinematic_touchdown"})
        == "touchdown"
    )


def test_campaign_json_encodes_numpy_diagnostics_without_nonfinite_values():
    assert json.loads(_encoded({"passed": np.bool_(True), "speed": np.float64(1.2)})) == {
        "passed": True,
        "speed": 1.2,
    }
    with pytest.raises(ValueError):
        _encoded({"speed": float("nan")})


def test_operating_campaign_refuses_missing_development_and_tampered_manifest(tmp_path):
    campaign = tmp_path / "campaign"
    prepare(campaign)
    with pytest.raises(ValueError, match="development summary"):
        freeze(campaign, "development")
    manifest_path = campaign / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["development"][0]["conditions"]["gust_amplitude_m_s"] = 1.0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest has changed"):
        _load_manifest(campaign)


def test_operating_freeze_requires_matching_manifest_and_complete_flight_records(tmp_path):
    campaign = tmp_path / "campaign"
    prepare(campaign)
    manifest = _load_manifest(campaign)
    summary_path = campaign / "development-summary.json"
    flights_path = campaign / "development.jsonl"
    summary = {
        "split": "development",
        "total": len(manifest["development"]),
        "manifest_sha256": "wrong",
        "source_hashes": _source_hashes(),
        "accepted": 0,
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    flights_path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="different manifest"):
        freeze(campaign, "development")
    summary["manifest_sha256"] = manifest["manifest_sha256"]
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="flight protocol"):
        freeze(campaign, "development")
    header = {
        "record_type": "protocol",
        "split": "development",
        "root_seed": ROOT_SEED,
        "manifest_sha256": manifest["manifest_sha256"],
        "source_hashes": _source_hashes(),
        "case_count": len(manifest["development"]),
    }
    records = [
        {"record_type": "flight", "id": case["id"], "input_sha256": _sha(case), "accepted": False}
        for case in manifest["development"]
    ]
    flights_path.write_text(
        "\n".join(_encoded(row) for row in (header, *records)), encoding="utf-8"
    )
    freeze(campaign, "development")
    frozen = json.loads((campaign / "freeze.json").read_text(encoding="utf-8"))
    assert frozen["manifest_sha256"] == manifest["manifest_sha256"]
    assert frozen["development_report_sha256"]


def _test_campaign_records(cases, workers):
    for case in cases:
        yield {
            "record_type": "flight",
            "id": case["id"],
            "group": case["group"],
            "input_sha256": _sha(case),
            "outcome": "landed",
            "accepted": True,
            "failed_criteria": [],
            "failure_stage": None,
            "remaining_main_propellant_kg": 10.0,
        }


def test_operating_parallel_records_keep_order_and_remain_freezable(tmp_path, monkeypatch):
    from scripts import evaluate_operating_envelope as campaign_cli

    calls = []

    def records(cases, workers):
        calls.append(workers)
        yield from _test_campaign_records(cases, workers)

    monkeypatch.setattr(campaign_cli, "_evaluate_cases", records)
    campaign = tmp_path / "campaign"
    prepare(campaign)
    campaign_cli.evaluate(campaign, "development", "development", workers=3)
    manifest = _load_manifest(campaign)
    flights = [
        json.loads(line) for line in (campaign / "development.jsonl").read_text().splitlines()
    ]
    assert [flight["id"] for flight in flights[1:]] == [
        case["id"] for case in manifest["development"]
    ]
    freeze(campaign, "development")
    campaign_cli.evaluate(campaign, "holdout", "holdout", workers=2)
    assert calls == [3, 2]
    summary = json.loads((campaign / "holdout-summary.json").read_text())
    assert summary["total"] == summary["accepted"] == len(manifest["holdout"]) == 72
    assert summary["workers"] == 2


def test_operating_failed_worker_retains_partial_evidence_without_summary(tmp_path, monkeypatch):
    from scripts import evaluate_operating_envelope as campaign_cli

    def interrupted_records(cases, workers):
        yield next(_test_campaign_records(cases, workers))
        raise RuntimeError("worker failed")

    monkeypatch.setattr(campaign_cli, "_evaluate_cases", interrupted_records)
    campaign = tmp_path / "campaign"
    prepare(campaign)
    with pytest.raises(RuntimeError, match="worker failed"):
        campaign_cli.evaluate(campaign, "development", "development", workers=2)
    assert len((campaign / "development.jsonl").read_text().splitlines()) == 2
    assert not (campaign / "development-summary.json").exists()
    with pytest.raises(ValueError, match="development summary"):
        freeze(campaign, "development")


def test_operating_source_change_during_sweep_refuses_summary(tmp_path, monkeypatch):
    from scripts import evaluate_operating_envelope as campaign_cli

    hashes = iter(({"source": "before"}, {"source": "after"}))
    monkeypatch.setattr(campaign_cli, "_source_hashes", lambda: next(hashes))
    monkeypatch.setattr(campaign_cli, "_evaluate_cases", _test_campaign_records)
    campaign = tmp_path / "campaign"
    prepare(campaign)
    with pytest.raises(ValueError, match="source changed during"):
        campaign_cli.evaluate(campaign, "development", "development", workers=2)
    assert not (campaign / "development-summary.json").exists()


def test_operating_process_executor_propagates_invalid_input():
    from scripts import evaluate_operating_envelope as campaign_cli

    invalid_case = {"conditions": {"gust_amplitude_m_s": 6.0}, "disturbances": {}}
    with pytest.raises(ValueError):
        list(campaign_cli._evaluate_cases([invalid_case], workers=2))


@pytest.mark.parametrize("workers", (0, -1, True, 1.5))
def test_operating_campaign_requires_positive_integer_workers(tmp_path, workers):
    from scripts import evaluate_operating_envelope as campaign_cli

    with pytest.raises(ValueError, match="positive integer"):
        campaign_cli.evaluate(tmp_path, "development", "development", workers=workers)


def _synthetic_full_flight_record(base_height=20.0):
    """Handwritten current 100 t poses for rendering, not a simulated mission."""
    design = Design()
    booster_mass = design.dry_mass_kg + 20_000.0 + design.rcs_propellant_kg
    properties = mass_properties(booster_mass, 1.0, PHYSICS)
    combined = (booster_mass * properties.cg_m + design.upper_mass_kg * PHYSICS.upper_station_m) / (
        booster_mass + design.upper_mass_kg
    )
    states = []
    for position in (combined, properties.cg_m, properties.cg_m):
        state = apr_separation_state()
        state[:6] = (0, base_height + position, 0, -1, 0, 0)
        state[8:13] = (1, 0.1, 1, 0, 0)
        states.append(state.tolist())
    return {
        "environment_version": OPERATING_SCOPE,
        "systems_model_version": "apr_100t_coupled_systems_v2",
        "landing_assessment": "kinematic_touchdown",
        "state_names": list(STATE_NAMES),
        "steps": [
            {"time_s": float(index), "state": state, "control": [1, 0, 0, 0], "phase": phase}
            for index, (state, phase) in enumerate(
                zip(states, ("ascent", "boostback", "touchdown"), strict=True)
            )
        ],
        "design": {**asdict(Design()), "launch_mass_kg": Design().launch_mass_kg},
        "physics": asdict(PHYSICS),
        "separation": {"time_s": 1.0},
        "contact": {
            "time_s": 2.0,
            "first_feet": [0],
            "foot_velocities_m_s": [[0.0, -0.9], [0.0, -0.9]],
        },
        "fin_settings": asdict(FINS),
        "outcome": "unverified",
        "simulated_acceptance_passed": False,
    }


def _replay_test_font_family():
    from matplotlib import font_manager

    fonts = font_manager.get_font_names()
    return next(
        (name for name in ("Malgun Gothic", "Noto Sans CJK KR", "NanumGothic") if name in fonts),
        "DejaVu Sans",
    )


def _synthetic_upper_departure_record():
    """Recorded staging poses with an upper COM state, not orbit evidence."""
    record = _synthetic_full_flight_record(base_height=40_000.0)
    theta = 0.2
    base = np.asarray((200.0, 40_000.0))
    axis = rotation(theta) @ np.asarray((0.0, 1.0))
    booster_mass = Design().dry_mass_kg + 20_000.0 + Design().rcs_propellant_kg
    cg = mass_properties(booster_mass, 1.0, PHYSICS).cg_m
    combined_cg = (booster_mass * cg + Design().upper_mass_kg * PHYSICS.upper_station_m) / (
        booster_mass + Design().upper_mass_kg
    )
    for index, row in enumerate(record["steps"]):
        row["state"][:2] = (base + axis * (combined_cg if index == 0 else cg)).tolist()
        row["state"][2:6] = (80.0, 600.0, theta, 0.013)
        row["state"][17:19] = (0.0, 0.0)
    record["steps"][-1]["time_s"] = record["contact"]["time_s"] = 8.0
    record["ascent_simulated"] = True
    record["simulation_scope"] = "launch_to_separation_to_touchdown"
    record["separation"]["state"] = deepcopy(record["steps"][1]["state"])
    record["separation"]["impulse_model"] = {
        "upper_position_m": (base + axis * PHYSICS.upper_station_m).tolist(),
        "upper_velocity_m_s": [81.0, 601.0],
        "upper_angular_rate_rad_s": -0.007,
        "upper_impulse_n_s": [2500.0, 5000.0],
    }
    return record


def test_upper_departure_uses_recorded_post_impulse_state_without_reapplying_impulse():
    record = _synthetic_upper_departure_record()
    original = deepcopy(record)
    replay = _NominalFlightReplay(record)
    departure = replay.upper_departure
    assert departure is not None
    assert departure.sample(0.999) is None
    assert departure.base(0.999) is None
    position, velocity, theta, omega, ignition = departure.sample(1.0)
    impulse = record["separation"]["impulse_model"]
    np.testing.assert_array_equal(position, impulse["upper_position_m"])
    np.testing.assert_array_equal(velocity, impulse["upper_velocity_m_s"])
    assert theta == record["separation"]["state"][4]
    assert omega == impulse["upper_angular_rate_rad_s"]
    assert ignition == 0.0
    assert record == original


def test_upper_departure_base_is_continuous_with_the_attached_shape_at_separation():
    replay = _NominalFlightReplay(_synthetic_upper_departure_record())
    upper_base, theta = replay.upper_departure.base(1.0)
    _, booster_base, *_ = replay.sample(1.0)
    interface = booster_base + rotation(theta) @ np.asarray(
        (0.0, replay.settings.geometry.body_length_m)
    )
    np.testing.assert_allclose(upper_base, interface, atol=1e-10)
    upper_position = replay.upper_departure.sample(1.0)[0]
    np.testing.assert_allclose(
        upper_position,
        upper_base
        + rotation(theta)
        @ np.asarray((0.0, replay.upper_station_m - replay.settings.geometry.body_length_m)),
        atol=1e-10,
    )


def test_upper_departure_retains_initial_motion_and_delays_visual_ignition():
    replay = _NominalFlightReplay(_synthetic_upper_departure_record())
    departure = replay.upper_departure
    initial = departure.sample(1.0)
    epsilon = 1e-5
    nearby = departure.sample(1.0 + epsilon)
    np.testing.assert_allclose((nearby[0] - initial[0]) / epsilon, initial[1], atol=1e-3, rtol=0)
    assert (nearby[2] - initial[2]) / epsilon == pytest.approx(initial[3], abs=1e-6)
    for time_s in (1.0, 2.0, 3.999, 4.0):
        assert departure.sample(time_s)[4] == 0.0
    assert 0.0 < departure.sample(6.0)[4] <= 1.0
    for time_s in (1.0, 2.0, 4.0, 6.0):
        _, _, theta, omega, _ = departure.sample(time_s)
        elapsed = time_s - 1.0
        assert theta == pytest.approx(initial[2] + initial[3] * 4 * (1 - np.exp(-elapsed / 4)))
        assert omega == pytest.approx(initial[3] * np.exp(-elapsed / 4))


def test_upper_departure_is_stateless_and_does_not_change_stage_one_replay():
    record = _synthetic_upper_departure_record()
    original = deepcopy(record)
    without_upper = deepcopy(record)
    without_upper["separation"].pop("impulse_model")
    replay = _NominalFlightReplay(record)
    stage_one_only = _NominalFlightReplay(without_upper)
    expected = {time_s: replay.upper_departure.sample(time_s) for time_s in (1.0, 2.0, 5.0, 7.0)}
    for time_s in (7.0, 1.0, 5.0, 2.0, 7.0):
        actual = replay.upper_departure.sample(time_s)
        for result, saved in zip(actual, expected[time_s], strict=True):
            np.testing.assert_array_equal(result, saved)
        with_upper = replay.sample(time_s)
        without = stage_one_only.sample(time_s)
        for result, saved in zip(with_upper, without, strict=True):
            np.testing.assert_array_equal(result, saved)
    assert record == original


@pytest.mark.parametrize("time_s", (1.5, 4.0, 5.5))
def test_upper_departure_visual_position_and_velocity_remain_kinematically_consistent(time_s):
    departure = _NominalFlightReplay(_synthetic_upper_departure_record()).upper_departure
    epsilon = 1e-3
    before = departure.sample(time_s - epsilon)
    current = departure.sample(time_s)
    after = departure.sample(time_s + epsilon)
    np.testing.assert_allclose(
        (after[0] - before[0]) / (2 * epsilon), current[1], atol=1e-5, rtol=0
    )


@pytest.mark.parametrize(
    "missing", ("upper_position_m", "upper_velocity_m_s", "upper_angular_rate_rad_s")
)
def test_upper_departure_rejects_incomplete_impulse_state(missing):
    record = _synthetic_upper_departure_record()
    record["separation"]["impulse_model"].pop(missing)
    with pytest.raises(ValueError, match="upper-stage"):
        _NominalFlightReplay(record)


@pytest.mark.parametrize(
    "field,invalid",
    (
        ("upper_position_m", [float("nan"), 30.0]),
        ("upper_position_m", [3.0]),
        ("upper_velocity_m_s", [4.0, float("inf")]),
        ("upper_velocity_m_s", [1.0, 2.0, 3.0]),
        ("upper_angular_rate_rad_s", float("nan")),
    ),
)
def test_upper_departure_rejects_invalid_impulse_state(field, invalid):
    record = _synthetic_upper_departure_record()
    record["separation"]["impulse_model"][field] = invalid
    with pytest.raises(ValueError, match="upper-stage"):
        _NominalFlightReplay(record)


@pytest.mark.parametrize("scope", ("missing-impulse", "recovery-only", "no-separation"))
def test_upper_departure_is_not_fabricated_without_an_entire_recorded_ascent(scope):
    record = _synthetic_upper_departure_record()
    if scope == "missing-impulse":
        record["separation"].pop("impulse_model")
    elif scope == "recovery-only":
        record["ascent_simulated"] = False
        record["simulation_scope"] = "37km_separation_to_touchdown"
    else:
        record["separation"] = None
    assert _NominalFlightReplay(record).upper_departure is None


def test_nominal_flight_replay_preserves_body_position_across_separation():
    replay = _NominalFlightReplay(_synthetic_full_flight_record())
    np.testing.assert_allclose(replay.bases, [[0, 20]] * 3)
    for time_s in (0, 0.5, 0.999, 1, 1.001, 1.5, 2):
        state, base, mass, *_ = replay.sample(time_s)
        np.testing.assert_allclose(base, [0, 20])
        assert state[6] == 20_000 and state[7] == 300
        assert mass == replay.design["dry_mass_kg"] + 20_300


def test_replay_includes_exact_separation_event_without_rewriting_saved_steps():
    record = _synthetic_full_flight_record()
    separation_state = deepcopy(record["steps"][1]["state"])
    separation_state[2:4] = (7.0, -4.0)
    separation_state[11:13] = (0.012, 0.17)
    record["separation"] = {"time_s": 1.1, "state": separation_state}
    record["steps"] = [record["steps"][0], record["steps"][2]]
    original = deepcopy(record)
    replay = _NominalFlightReplay(record)
    state, base, _, control, phase, _ = replay.sample(1.1)
    np.testing.assert_allclose(state, separation_state, atol=1e-10)
    np.testing.assert_allclose(base, [0.0, 20.0], atol=1e-10)
    assert control[:2] == pytest.approx((0.17, 0.012))
    assert phase != "ascent"
    assert record == original
    assert len(record["steps"]) == 2


def test_replay_contact_event_preserves_the_exact_grounded_pose_and_raw_throttle():
    record = _synthetic_full_flight_record(base_height=-PHYSICS.gear.foot_station_m)
    contact_state = deepcopy(record["steps"][-1]["state"])
    contact_state[11:13] = (0.01, 0.2)
    record["contact"]["state"] = contact_state
    record["steps"][-1]["state"][1] += 0.1
    original = deepcopy(record)
    replay = _NominalFlightReplay(record)
    parts, _, sample = _nominal_flight_geometry(replay, 2.0)
    np.testing.assert_allclose(sample[0], contact_state, atol=1e-10)
    assert sample[3][:2] == pytest.approx((0.2, 0.01))
    assert parts["foot_left"][:, 1].min() == pytest.approx(0.0, abs=1e-10)
    assert parts["foot_right"][:, 1].min() == pytest.approx(0.0, abs=1e-10)
    assert record == original


@pytest.mark.parametrize("invalid", ("outside_interval", "wrong_size", "nan"))
def test_replay_rejects_invalid_exact_event_states(invalid):
    record = _synthetic_full_flight_record()
    record["separation"]["state"] = deepcopy(record["steps"][1]["state"])
    if invalid == "outside_interval":
        record["separation"]["time_s"] = 2.1
    elif invalid == "wrong_size":
        record["separation"]["state"].pop()
    else:
        record["separation"]["state"][0] = float("nan")
    with pytest.raises(ValueError, match="replay event"):
        _NominalFlightReplay(record)


def test_nominal_flight_replay_displays_actual_actuator_states_at_touchdown():
    record = _synthetic_full_flight_record()
    for row in record["steps"]:
        row["state"][11:13] = (0.01, 0.2)
    record["steps"][-1]["control"] = [0, 0, 0, 0]
    replay = _NominalFlightReplay(record)
    assert replay.sample(2.0)[4] == "touchdown"
    assert replay.sample(2.0)[3][:2] == pytest.approx([0.2, 0.01])
    assert replay.sample(2.0)[0][3] == -1.0


def test_kinematic_replay_does_not_claim_post_contact_stability():
    import matplotlib.pyplot as plt

    record = _synthetic_full_flight_record()
    record["landing_assessment"] = "kinematic_touchdown"
    record["simulated_acceptance_passed"] = True
    scene = _NominalFlightAnimationScene(record)
    try:
        scene.update(2.0)
        assert scene.replay.sample(2.0)[4] == "touchdown"
        assert "접지 조건 통과" in scene.event.get_text()
        assert "미평가" in scene.event.get_text()
        assert "안정화 완료" not in scene.event.get_text()
        assert "엔진 OFF" in scene.event.get_text()
        assert "연출" in scene.event.get_text()
        assert _stage(record) == "touchdown"
    finally:
        plt.close(scene.fig)


def test_replay_contact_switches_display_engine_off_without_rewriting_telemetry():
    import matplotlib.pyplot as plt

    record = _synthetic_full_flight_record()
    for row in record["steps"]:
        row["state"][11:13] = (0.01, 0.2)
    record["steps"][-1]["control"] = [0, 0, 0, 0]
    original = deepcopy(record)
    scene = _NominalFlightAnimationScene(record)
    try:
        scene.update(1.999)
        assert scene.display_engine_on is True
        assert scene.displayed_thrust_n > 0
        assert any(plume.get_visible() for plume in scene.plumes)

        for time_s in (2.0, 2.5, 6.0):
            scene.update(time_s)
            assert scene.display_engine_on is False
            assert scene.displayed_thrust_n == 0.0
            assert not any(plume.get_visible() for plume in scene.plumes)
            state, _, _, control, phase, _ = scene.replay.sample(time_s)
            assert phase == "touchdown"
            assert state[3] == -1.0
            assert state[11:13] == pytest.approx((0.01, 0.2))
            assert control[:2] == pytest.approx((0.2, 0.01))
        assert record == original
    finally:
        plt.close(scene.fig)


@pytest.mark.parametrize(
    "torque_nm,expected_kn_m,visible",
    (
        (100_000.0, 100.0, True),
        (-100_000.0, -100.0, True),
        (25_000.0, 25.0, True),
        (100.0, 0.1, False),
        (-100.0, -0.1, False),
        (2_000.0, 2.0, False),
        (2_001.0, 2.001, True),
        (0.0, 0.0, False),
    ),
)
def test_replay_rcs_torque_uses_newton_meters_and_bounded_visual_jet_length(
    torque_nm, expected_kn_m, visible
):
    import matplotlib.pyplot as plt

    record = _synthetic_full_flight_record()
    for row in record["steps"]:
        row["control"][2] = torque_nm
        row["state"][4] = np.deg2rad(37.0)
    original = deepcopy(record)
    scene = _NominalFlightAnimationScene(record)
    try:
        scene.update(1.5)
        assert scene.rcs_torque_kn_m == pytest.approx(expected_kn_m)
        assert f"RCS 토크 {expected_kn_m:+.1f} kN·m" in scene.status.get_text()
        ratio = np.clip(abs(torque_nm) / record["design"]["rcs_torque_limit_nm"], 0, 1)
        assert scene.rcs_jet_length_m == pytest.approx(2 * np.sqrt(ratio))
        assert 0.0 <= scene.rcs_jet_length_m <= 2.0
        state, base, _, control, _, _ = scene.replay.sample(1.5)
        assert control[2] == torque_nm
        for jet in scene.rcs_jets:
            assert bool(jet.get_visible()) is visible
            local = (jet.get_xy()[:3] - base) @ rotation(state[4])
            jet_tip_distance = np.linalg.norm(local[2] - (local[0] + local[1]) / 2)
            assert jet_tip_distance == pytest.approx(scene.rcs_jet_length_m, abs=1e-10)
            assert jet_tip_distance <= 2.0 + 1e-10
        for time_s in (2.0, 2.5, 6.0):
            scene.update(time_s)
            assert not any(jet.get_visible() for jet in scene.rcs_jets)
            assert scene.replay.sample(time_s)[3][2] == torque_nm
        assert record == original
    finally:
        plt.close(scene.fig)


def test_replay_hides_upper_stage_only_after_recorded_separation():
    import matplotlib.pyplot as plt

    scene = _NominalFlightAnimationScene(_synthetic_full_flight_record())
    try:
        scene.update(0.999)
        assert scene.parts["upper_stage"].get_visible()
        assert scene.parts["separation_band"].get_visible()
        for time_s in (1.0, 1.001, 2.0):
            scene.update(time_s)
            assert not scene.parts["upper_stage"].get_visible()
            assert not scene.parts["separation_band"].get_visible()
    finally:
        plt.close(scene.fig)


def test_replay_keeps_a_recorded_upper_stage_after_separation_and_then_ignites_it():
    import matplotlib.pyplot as plt

    record = _synthetic_upper_departure_record()
    original = deepcopy(record)
    scene = _NominalFlightAnimationScene(record, width=640)
    try:
        assert scene.body_label.get_text() == "DM100"
        assert scene.title.get_text() == "DM100"
        for time_s in (0.999, 1.0, 3.999, 4.0, 6.0, 8.0):
            scene.update(time_s)
            for name in ("upper_stage", "upper_highlight", "upper_band_detail"):
                assert scene.parts[name].get_visible()
            assert scene.parts["upper_nozzle"].get_visible() == (time_s >= 1.0)
            assert scene.parts["separation_band"].get_visible() == (time_s < 1.0)
            assert any(plume.get_visible() for plume in scene.upper_plumes) == (time_s > 4.0)
        assert scene.display_engine_on is False
        assert scene.displayed_thrust_n == 0.0
        assert any(plume.get_visible() for plume in scene.upper_plumes)
        assert "2단" in scene.footer.get_text()
        assert "미계산" in scene.footer.get_text()
        assert record == original
    finally:
        plt.close(scene.fig)


@pytest.mark.parametrize("gust,expected", ((0.0, "무풍 기준"), (5.0, "설정 외란")))
def test_replay_record_note_only_shows_the_condition_title(gust, expected):
    import matplotlib.pyplot as plt

    record = _synthetic_full_flight_record()
    record["conditions"] = {"gust_amplitude_m_s": gust}
    original = deepcopy(record)
    scene = _NominalFlightAnimationScene(record, width=640)
    try:
        assert scene.record_note.get_text() == expected
        assert record == original
    finally:
        plt.close(scene.fig)


def test_replay_upper_stage_geometry_and_overview_are_independent_of_frame_order():
    import matplotlib.pyplot as plt

    record = _synthetic_upper_departure_record()
    original = deepcopy(record)
    scene = _NominalFlightAnimationScene(record, width=640)
    try:
        scene.update(1.0)
        upper_initial = scene.parts["upper_stage"].get_xy().copy()
        upper_base, _ = scene.replay.upper_departure.base(1.0)
        midpoint = (upper_initial[0] + upper_initial[1]) / 2
        np.testing.assert_allclose(midpoint, upper_base, atol=1e-10)
        scene.update(6.0)
        upper_later = scene.parts["upper_stage"].get_xy().copy()
        first_later = scene.parts["body"].get_xy().copy()
        history = tuple(np.asarray(values).copy() for values in scene.upper_history.get_data())
        marker = tuple(np.asarray(values).copy() for values in scene.upper_marker.get_data())
        assert len(history[0]) > 0
        assert np.linalg.norm(upper_later.mean(axis=0) - upper_initial.mean(axis=0)) > 100.0
        position = scene.replay.upper_departure.sample(6.0)[0]
        np.testing.assert_allclose([marker[0][0], marker[1][0]], position / 1000)
        scene.update(0.0)
        assert len(scene.upper_history.get_xdata()) == 0
        assert len(scene.upper_marker.get_xdata()) == 0
        scene.update(6.0)
        np.testing.assert_array_equal(scene.parts["upper_stage"].get_xy(), upper_later)
        np.testing.assert_array_equal(scene.parts["body"].get_xy(), first_later)
        for actual, expected in zip(scene.upper_history.get_data(), history, strict=True):
            np.testing.assert_array_equal(actual, expected)
        for actual, expected in zip(scene.upper_marker.get_data(), marker, strict=True):
            np.testing.assert_array_equal(actual, expected)
        assert record == original
    finally:
        plt.close(scene.fig)


@pytest.mark.parametrize("time_s", (1.0, 6.0, 8.0))
def test_cached_replay_frame_matches_full_canvas_with_detached_upper_stage(time_s):
    import matplotlib
    import matplotlib.pyplot as plt

    record = _synthetic_upper_departure_record()
    original = deepcopy(record)
    with matplotlib.rc_context(
        {"font.family": _replay_test_font_family(), "axes.unicode_minus": False}
    ):
        scene = _NominalFlightAnimationScene(record, width=640)
        renderer = _NominalFlightFrameRenderer(scene)
        try:
            scene.update(time_s)
            cached = np.asarray(renderer.frame()).copy()
            renderer.close()
            scene.fig.canvas.draw()
            np.testing.assert_array_equal(cached, np.asarray(scene.fig.canvas.buffer_rgba()))
            assert record == original
        finally:
            renderer.close()
            plt.close(scene.fig)


@pytest.mark.parametrize("width", (640, 1920))
@pytest.mark.parametrize("time_s", (0.0, 8.0))
def test_replay_telemetry_cards_keep_long_values_clear_of_labels_and_bottom_strip(width, time_s):
    import matplotlib
    import matplotlib.pyplot as plt

    record = _synthetic_upper_departure_record()
    original = deepcopy(record)
    long_values = (
        "100.000 km",
        "+100,000.00 m",
        "-1,234.567 m/s",
        "+1,234.567 m/s",
        "+180.000\N{DEGREE SIGN}",
        "+15.000\N{DEGREE SIGN}/s",
        "70.200 t",
        "300.00 kg",
        "100.000 t",
        "30.000 kPa",
        "1,600.0 kN",
        "100.00 %",
        "100 %",
        "100 %",
    )
    with matplotlib.rc_context(
        {"font.family": _replay_test_font_family(), "axes.unicode_minus": False}
    ):
        scene = _NominalFlightAnimationScene(record, width=width)
        try:
            scene.update(time_s)
            for artist, value in zip(scene.telemetry, long_values, strict=True):
                artist.set_text(value)
            scene.fig.canvas.draw()
            renderer = scene.fig.canvas.get_renderer()
            assert scene.title.get_text() == "DM100"
            assert len(scene.telemetry) == len(scene.telemetry_labels) == 14
            assert len(scene.info.patches) == 14
            text_bounds = []
            card_bounds = [card.get_window_extent(renderer) for card in scene.info.patches]
            for index, (label, value, card) in enumerate(
                zip(scene.telemetry_labels, scene.telemetry, card_bounds, strict=True)
            ):
                for artist in (label, value):
                    bounds = artist.get_window_extent(renderer)
                    assert bounds.x0 >= card.x0 - 0.1, (index, artist.get_text())
                    assert bounds.x1 <= card.x1 + 0.1, (index, artist.get_text())
                    assert bounds.y0 >= card.y0 - 0.1, (index, artist.get_text())
                    assert bounds.y1 <= card.y1 + 0.1, (index, artist.get_text())
                    text_bounds.append(bounds)
            for index, bounds in enumerate(card_bounds):
                assert not any(bounds.overlaps(other) for other in card_bounds[index + 1 :])
            for index, bounds in enumerate(text_bounds):
                assert not any(bounds.overlaps(other) for other in text_bounds[index + 1 :])
            for artist in (scene.status, scene.environment_text, scene.event):
                bottom_bounds = artist.get_window_extent(renderer)
                assert not any(bottom_bounds.overlaps(bounds) for bounds in text_bounds)
            assert record == original
        finally:
            plt.close(scene.fig)


def _rendered_patch_bounds(patch, renderer, *, include_stroke=True):
    """Include rounded path padding and the actual rendered border width."""
    from matplotlib.transforms import Bbox

    bounds = patch.get_path().get_extents(patch.get_transform())
    border = 0.0
    if patch.get_edgecolor()[3] > 0:
        border = renderer.points_to_pixels(patch.get_linewidth()) / 2
    adjustment = border if include_stroke else -border
    return Bbox.from_extents(
        bounds.x0 - adjustment,
        bounds.y0 - adjustment,
        bounds.x1 + adjustment,
        bounds.y1 + adjustment,
    )


def _assert_positive_bounds_gap(left, right, description):
    horizontal = max(right.x0 - left.x1, left.x0 - right.x1)
    vertical = max(right.y0 - left.y1, left.y0 - right.y1)
    assert max(horizontal, vertical) > 0.0, description


def _assert_bounds_inside(inner, outer, description):
    assert inner.x0 >= outer.x0, description
    assert inner.x1 <= outer.x1, description
    assert inner.y0 >= outer.y0, description
    assert inner.y1 <= outer.y1, description


@pytest.mark.parametrize("width", (640, 1280, 1920))
@pytest.mark.parametrize("pose", ("launch", "separation", "high-speed", "touchdown"))
def test_replay_dashboard_all_text_and_rounded_cards_have_positive_clearance(width, pose):
    import matplotlib
    import matplotlib.pyplot as plt

    if pose in ("launch", "touchdown"):
        record = _synthetic_full_flight_record(base_height=1.0)
        time_s = 0.0 if pose == "launch" else 2.0
    else:
        record = _synthetic_upper_departure_record()
        time_s = 1.0 if pose == "separation" else 6.0
    if pose == "high-speed":
        record["conditions"] = {"gust_amplitude_m_s": 5.0, "wind_profile": [[0, 8], [10000, 20]]}
        for row in record["steps"]:
            row["state"][2:4] = [1234.567, -1234.567]
            row["control"][2] = 100_000.0
    original = deepcopy(record)
    with matplotlib.rc_context(
        {"font.family": _replay_test_font_family(), "axes.unicode_minus": False}
    ):
        scene = _NominalFlightAnimationScene(record, width=width)
        try:
            scene.update(time_s)
            for index, artist in enumerate(scene.telemetry):
                if index in (2, 3):
                    artist.set_text("-1,234.567 m/s")
                elif index == 1:
                    artist.set_text("+100,000.00 m")
            scene.fig.canvas.draw()
            renderer = scene.fig.canvas.get_renderer()
            canvas = scene.fig.bbox
            rounded_cards = [*scene.info.patches, *(card for card, _ in scene.timeline)]
            card_bounds = [_rendered_patch_bounds(card, renderer) for card in rounded_cards]
            for index, bounds in enumerate(card_bounds):
                _assert_bounds_inside(bounds, canvas, f"card {index} leaves the video canvas")
                for other_index, other in enumerate(card_bounds[index + 1 :], start=index + 1):
                    _assert_positive_bounds_gap(bounds, other, f"cards {index}/{other_index} touch")
            for index, (card, text) in enumerate(scene.timeline):
                _assert_bounds_inside(
                    text.get_window_extent(renderer),
                    _rendered_patch_bounds(card, renderer, include_stroke=False),
                    f"timeline label {index} escapes its rounded card",
                )
            texts = [
                scene.title,
                scene.subtitle,
                scene.mission_label,
                scene.record_note,
                scene.clock,
                scene.footer,
                scene.telemetry_heading,
                *scene.telemetry_labels,
                *scene.telemetry,
                scene.status,
                scene.environment_text,
                scene.event,
                scene.phase,
                scene.camera_note,
                scene.scale_label,
                scene.overview.title,
                scene.overview.xaxis.label,
                scene.overview.yaxis.label,
                scene.overview.xaxis.get_offset_text(),
                scene.overview.yaxis.get_offset_text(),
                *(text for _, text in scene.timeline),
            ]
            if scene.overview_legend is not None:
                texts.extend(scene.overview_legend.get_texts())
            for axis, limits in (
                (scene.overview.xaxis, scene.overview.get_xlim()),
                (scene.overview.yaxis, scene.overview.get_ylim()),
            ):
                for tick in axis.get_major_ticks():
                    if min(limits) <= tick.get_loc() <= max(limits):
                        texts.append(tick.label1)
            rendered_texts = []
            for artist in texts:
                if not artist.get_visible() or not artist.get_text():
                    continue
                bounds = artist.get_window_extent(renderer)
                if artist.get_bbox_patch() is not None:
                    bounds = _rendered_patch_bounds(artist.get_bbox_patch(), renderer)
                rendered_texts.append((artist, bounds))
                _assert_bounds_inside(bounds, canvas, f"text leaves canvas: {artist.get_text()}")
            for index, (artist, bounds) in enumerate(rendered_texts):
                for other, other_bounds in rendered_texts[index + 1 :]:
                    _assert_positive_bounds_gap(
                        bounds,
                        other_bounds,
                        f"text collision: {artist.get_text()} / {other.get_text()}",
                    )
                if artist not in [text for _, text in scene.timeline]:
                    for card, _ in scene.timeline:
                        _assert_positive_bounds_gap(
                            bounds,
                            _rendered_patch_bounds(card, renderer),
                            f"text crosses a timeline card: {artist.get_text()}",
                        )
            assert record == original
        finally:
            plt.close(scene.fig)


def test_nominal_flight_replay_draws_grounded_feet_without_shifting_hull():
    base_height = -PHYSICS.gear.foot_station_m
    replay = _NominalFlightReplay(_synthetic_full_flight_record(base_height=base_height))
    parts, nozzle, sample = _nominal_flight_geometry(replay, 2)
    state, base, mass, *_ = sample
    vector = np.r_[state[:6], mass, state[6], state[17:19]]
    points, _ = foot_kinematics(vector, state[8], replay.settings)
    np.testing.assert_allclose(points[:, 1], 0, atol=1e-10)
    assert base[1] == pytest.approx(base_height)
    assert parts["foot_left"][:, 1].min() == pytest.approx(0)
    assert parts["foot_right"][:, 1].min() == pytest.approx(0)
    assert parts["body"][:, 1].min() == pytest.approx(base_height)
    assert nozzle[1] == pytest.approx(base_height + replay.settings.geometry.nozzle_station_m)


def test_nominal_flight_replay_uses_recorded_fin_extension():
    deployed = _synthetic_full_flight_record()
    stowed = deepcopy(deployed)
    for row in stowed["steps"]:
        row["state"][10] = 0
    extended, _, _ = _nominal_flight_geometry(_NominalFlightReplay(deployed), 1)
    retracted, _, _ = _nominal_flight_geometry(_NominalFlightReplay(stowed), 1)
    assert np.ptp(extended["fin_right"][:, 0]) > np.ptp(retracted["fin_right"][:, 0])


def test_nominal_flight_replay_real_time_has_no_terminal_hold():
    frames = _nominal_flight_frame_times(np.asarray([0.0, 240.253]), 20)
    np.testing.assert_allclose(np.diff(frames[:-1]), 0.05)
    assert frames[-1] == 240.253
    assert 0 < frames[-1] - frames[-2] <= 0.05
    assert len(frames) / 20 - 240.253 < 0.1


@pytest.mark.parametrize("invalid", ["duplicate_time", "nan_state", "wrong_state_size"])
def test_nominal_flight_replay_rejects_invalid_records(invalid):
    flight = _synthetic_full_flight_record()
    if invalid == "duplicate_time":
        flight["steps"][1]["time_s"] = 0
    elif invalid == "nan_state":
        flight["steps"][0]["state"][0] = float("nan")
    else:
        for row in flight["steps"]:
            row["state"].pop()
    with pytest.raises(ValueError, match="strictly ordered"):
        _NominalFlightReplay(flight)


@pytest.mark.parametrize("invalid", ["systems_version", "environment_version", "state_order"])
def test_replay_rejects_mismatched_model_contract(invalid):
    record = _synthetic_full_flight_record()
    if invalid == "systems_version":
        record["systems_model_version"] = "obsolete"
    elif invalid == "environment_version":
        record["environment_version"] = "different_environment"
    else:
        record["state_names"][9], record["state_names"][10] = (
            record["state_names"][10],
            record["state_names"][9],
        )
    with pytest.raises(ValueError, match="100 t"):
        _NominalFlightReplay(record)


def test_nominal_flight_replay_never_overwrites_output(tmp_path):
    output = tmp_path / "flight.mp4"
    output.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        save_nominal_flight_replay({}, output)
    assert output.read_bytes() == b"existing"


@pytest.mark.parametrize("fps", (0, 61, -1, True, 1.5, float("nan")))
def test_nominal_flight_replay_rejects_invalid_fps_before_rendering(tmp_path, fps):
    with pytest.raises(ValueError, match="fps"):
        save_nominal_flight_replay({}, tmp_path / "flight.mp4", fps=fps)
    assert not (tmp_path / "flight.mp4").exists()


@pytest.mark.parametrize("width", (0, -640, 638, 641, 642, 3842, True, 640.5, float("nan")))
def test_nominal_flight_replay_rejects_invalid_video_width(tmp_path, width):
    with pytest.raises(ValueError, match="width"):
        save_nominal_flight_replay({}, tmp_path / "flight.mp4", width=width)
    assert not (tmp_path / "flight.mp4").exists()


@pytest.mark.parametrize("duration", (-0.01, 30.01, float("nan"), float("inf"), True))
def test_nominal_flight_replay_rejects_invalid_end_hold_duration(tmp_path, duration):
    with pytest.raises(ValueError, match="end_hold"):
        save_nominal_flight_replay({}, tmp_path / "flight.mp4", end_hold_s=duration)
    assert not (tmp_path / "flight.mp4").exists()


@pytest.mark.parametrize("field", ("start_s", "end_s"))
@pytest.mark.parametrize("value", (-0.01, float("nan"), float("inf"), True, "1.0"))
def test_replay_rejects_invalid_interval_bounds_before_loading_a_flight(tmp_path, field, value):
    output = tmp_path / "flight.mp4"
    with pytest.raises(ValueError, match="interval bounds"):
        save_nominal_flight_replay({}, output, **{field: value})
    assert not output.exists()


@pytest.mark.parametrize("start_s,end_s", ((4.0, 6.0), (5.0, 8.0), (6.0, 5.0), (7.0, 7.0)))
def test_replay_refuses_intervals_outside_the_recording_without_creating_output(
    tmp_path, monkeypatch, start_s, end_s
):
    from powered_landing_guidance import visualization

    record = _synthetic_full_flight_record()
    for row in record["steps"]:
        row["time_s"] += 5.0
    record["separation"]["time_s"] += 5.0
    record["contact"]["time_s"] += 5.0
    original = deepcopy(record)
    output = tmp_path / "unused" / "flight.mp4"
    monkeypatch.setattr(visualization, "_ffmpeg_available", lambda: True)
    with pytest.raises(ValueError, match="inside the recorded flight"):
        save_nominal_flight_replay(record, output, start_s=start_s, end_s=end_s, width=640)
    assert not output.parent.exists()
    assert record == original


def test_replay_clip_keeps_normal_time_and_does_not_turn_off_an_airborne_engine(
    tmp_path, monkeypatch
):
    from contextlib import contextmanager

    from powered_landing_guidance import visualization

    record = _synthetic_full_flight_record()
    for row in record["steps"]:
        row["state"][12] = 0.2
    original = deepcopy(record)
    scene_factory = visualization._NominalFlightAnimationScene
    scenes, times, frames = [], [], []

    def make_scene(*args, **kwargs):
        scene = scene_factory(*args, **kwargs)
        original_update = scene.update

        def update(time_s):
            original_update(time_s)
            times.append(time_s)

        scene.update = update
        scenes.append(scene)
        return scene

    class CaptureWriter:
        def __init__(self, *, fps, **_kwargs):
            assert fps == 4

        @contextmanager
        def saving(self, _figure, filename, dpi):
            assert dpi == 40
            Path(filename).write_bytes(b"synthetic interval writer")
            yield

        def grab_frame(self, **_kwargs):
            scene = scenes[-1]
            frames.append(
                (
                    scene.parts["body"].get_xy().copy(),
                    scene.display_engine_on,
                    scene.displayed_thrust_n,
                    scene.event.get_text(),
                )
            )

    def no_integration(*_args, **_kwargs):
        pytest.fail("a recorded interval replay must not advance flight physics")

    monkeypatch.setattr(visualization, "_NominalFlightAnimationScene", make_scene)
    monkeypatch.setattr(visualization, "FFMpegWriter", CaptureWriter)
    monkeypatch.setattr(visualization, "_ffmpeg_available", lambda: True)
    monkeypatch.setattr(AprFlight, "advance", no_integration)
    output = tmp_path / "clip.mp4"
    save_nominal_flight_replay(
        record, output, start_s=0.25, end_s=1.25, fps=4, width=640, end_hold_s=1.0
    )
    np.testing.assert_array_equal(times, (0.25, 0.5, 0.75, 1.0, 1.25, 1.25, 1.25, 1.25, 1.25))
    assert len(frames) == 9
    for _body, engine_on, thrust, event in frames:
        assert engine_on is True
        assert thrust > 0
        assert "엔진 OFF" not in event
    for body, *_ in frames[5:]:
        np.testing.assert_array_equal(body, frames[4][0])
    assert output.read_bytes() == b"synthetic interval writer"
    assert record == original


def test_failed_ascent_replay_does_not_fabricate_separation_or_contact():
    import matplotlib.pyplot as plt

    record = _synthetic_full_flight_record()
    record["separation"] = record["contact"] = None
    for row in record["steps"]:
        row["phase"] = "ascent"
    scene = _NominalFlightAnimationScene(record)
    try:
        scene.update(2.0)
        assert np.isinf(scene.replay.separation_s)
        assert np.isinf(scene.replay.contact_s)
        assert "착륙 성공" not in scene.event.get_text()
        assert scene.parts["upper_stage"].get_visible()
    finally:
        plt.close(scene.fig)


@pytest.mark.parametrize("passed", [False, True])
def test_replay_success_label_matches_saved_acceptance(passed):
    import matplotlib.pyplot as plt

    record = _synthetic_full_flight_record()
    record["simulated_acceptance_passed"] = passed
    scene = _NominalFlightAnimationScene(record)
    try:
        scene.update(2.0)
        assert ("접지 조건 통과" in scene.event.get_text()) is passed
        if not passed:
            assert "착륙 성공" not in scene.event.get_text()
            assert "안정화 완료" not in scene.event.get_text()
    finally:
        plt.close(scene.fig)


def test_short_synthetic_mp4_and_preview_can_be_saved_without_a_flight(tmp_path):
    from powered_landing_guidance.visualization import _ffmpeg_available

    if not _ffmpeg_available():
        pytest.skip("FFmpeg or the optional video extra is unavailable")
    output, preview = tmp_path / "poses.mp4", tmp_path / "poses.png"
    save_nominal_flight_replay(
        _synthetic_full_flight_record(),
        output,
        fps=1,
        width=640,
        end_hold_s=0,
        preview=preview,
    )
    assert b"ftyp" in output.read_bytes()[:32]
    assert preview.read_bytes().startswith(b"\x89PNG")


@pytest.mark.parametrize("end_hold_s", (0.0, 1.25, 1.3))
def test_replay_writer_holds_only_the_final_display_without_adding_simulation(
    tmp_path, monkeypatch, end_hold_s
):
    from contextlib import contextmanager

    from powered_landing_guidance import visualization

    record = _synthetic_full_flight_record()
    for row in record["steps"]:
        row["state"][12] = 0.2
    original = deepcopy(record)
    scene_factory = visualization._NominalFlightAnimationScene
    scenes, rendered, dimensions = [], [], []

    def make_scene(*args, **kwargs):
        scene = scene_factory(*args, **kwargs)
        scenes.append(scene)
        return scene

    class CaptureWriter:
        def __init__(self, *, fps, **_kwargs):
            assert fps == 4

        @contextmanager
        def saving(self, figure, filename, dpi):
            dimensions.append(figure.get_size_inches() * dpi)
            Path(filename).write_bytes(b"synthetic writer, not a flight video")
            yield

        def grab_frame(self, **_kwargs):
            scene = scenes[-1]
            rendered.append(
                (
                    scene.parts["body"].get_xy().copy(),
                    scene.display_engine_on,
                    scene.displayed_thrust_n,
                    scene.status.get_text(),
                )
            )

    def no_integration(*_args, **_kwargs):
        pytest.fail("replaying and holding a stored frame must not advance physics")

    monkeypatch.setattr(visualization, "_NominalFlightAnimationScene", make_scene)
    monkeypatch.setattr(visualization, "FFMpegWriter", CaptureWriter)
    monkeypatch.setattr(visualization, "_ffmpeg_available", lambda: True)
    monkeypatch.setattr(AprFlight, "advance", no_integration)
    output = tmp_path / "captured.mp4"
    save_nominal_flight_replay(record, output, fps=4, width=640, end_hold_s=end_hold_s)
    flight_frames = len(_nominal_flight_frame_times(np.asarray([0.0, 2.0]), 4))
    assert len(rendered) == flight_frames + int(np.ceil(end_hold_s * 4))
    np.testing.assert_allclose(dimensions, [[640.0, 360.0]])
    for body, engine_on, thrust, status in rendered[flight_frames - 1 :]:
        np.testing.assert_array_equal(body, rendered[flight_frames - 1][0])
        assert engine_on is False
        assert thrust == 0.0
        assert status == rendered[flight_frames - 1][3]
    assert record == original
    assert output.exists()


@pytest.mark.parametrize("width", (640, 1920))
@pytest.mark.parametrize("time_s", (0.0, 1.25, 2.0))
def test_cached_replay_frame_matches_full_canvas_at_launch_flip_and_contact(width, time_s):
    import matplotlib
    import matplotlib.pyplot as plt

    record = _synthetic_full_flight_record()
    record["steps"][1]["phase"] = "flip"
    record["steps"][1]["state"][4] = np.deg2rad(37.0)
    for row in record["steps"]:
        row["state"][12] = 0.2
        row["control"][2] = 100_000.0
    original = deepcopy(record)
    with matplotlib.rc_context(
        {"font.family": _replay_test_font_family(), "axes.unicode_minus": False}
    ):
        scene = _NominalFlightAnimationScene(record, width=width)
        renderer = _NominalFlightFrameRenderer(scene)
        try:
            scene.update(time_s)
            frame = renderer.frame()
            assert isinstance(frame, memoryview)
            assert frame.shape == (width * 9 // 16, width, 4)
            assert frame.nbytes == width * (width * 9 // 16) * 4
            cached = np.asarray(frame).copy()
            renderer.close()
            scene.fig.canvas.draw()
            np.testing.assert_array_equal(cached, np.asarray(scene.fig.canvas.buffer_rgba()))
            assert record == original
        finally:
            renderer.close()
            plt.close(scene.fig)


def test_cached_replay_renderer_restores_every_original_animated_flag(monkeypatch):
    import matplotlib
    import matplotlib.pyplot as plt

    monkeypatch.setitem(matplotlib.rcParams, "font.family", [_replay_test_font_family()])
    monkeypatch.setitem(matplotlib.rcParams, "axes.unicode_minus", False)
    scene = _NominalFlightAnimationScene(_synthetic_full_flight_record(), width=640)
    scene.camera.set_animated(True)
    scene.event.set_animated(True)
    flags = [(artist, artist.get_animated()) for artist in scene.fig.findobj()]
    renderer = _NominalFlightFrameRenderer(scene)
    try:
        scene.update(2.0)
        renderer.frame()
        renderer.close()
        for artist, flag in flags:
            assert artist.get_animated() == flag
        renderer.close()
        for artist, flag in flags:
            assert artist.get_animated() == flag
    finally:
        renderer.close()
        plt.close(scene.fig)


@pytest.mark.parametrize("width", (640, 1920))
def test_replay_fast_writer_streams_exact_rgba_frames_without_changing_record(
    tmp_path, monkeypatch, width
):
    from contextlib import contextmanager
    from types import SimpleNamespace

    from powered_landing_guidance import visualization

    record = _synthetic_full_flight_record()
    original = deepcopy(record)
    shapes, sizes, scenes, original_flags = [], [], [], []
    scene_factory = visualization._NominalFlightAnimationScene

    def make_scene(*args, **kwargs):
        scene = scene_factory(*args, **kwargs)
        scenes.append(scene)
        original_flags.append([(artist, artist.get_animated()) for artist in scene.fig.findobj()])
        return scene

    class CapturePipe:
        def write(self, frame):
            assert isinstance(frame, memoryview)
            shapes.append(frame.shape)
            sizes.append(frame.nbytes)
            return frame.nbytes

    class FastWriter:
        frame_format = "rgba"

        def __init__(self, *, fps, **_kwargs):
            assert fps == 1
            self._proc = SimpleNamespace(stdin=CapturePipe())

        @contextmanager
        def saving(self, figure, filename, dpi):
            self.frame_size = tuple(int(value) for value in figure.get_size_inches() * dpi)
            Path(filename).write_bytes(b"synthetic writer, not a flight video")
            yield

        def grab_frame(self, **_kwargs):
            pytest.fail("an available RGBA process must use the tested fast writer path")

    def no_integration(*_args, **_kwargs):
        pytest.fail("cached rendering of stored samples must not advance physics")

    monkeypatch.setattr(visualization, "_NominalFlightAnimationScene", make_scene)
    monkeypatch.setattr(visualization, "FFMpegWriter", FastWriter)
    monkeypatch.setattr(visualization, "_ffmpeg_available", lambda: True)
    monkeypatch.setattr(AprFlight, "advance", no_integration)
    output = tmp_path / "fast.mp4"
    save_nominal_flight_replay(record, output, fps=1, width=width, end_hold_s=0)
    assert shapes == [(width * 9 // 16, width, 4)] * 3
    assert sizes == [width * (width * 9 // 16) * 4] * 3
    assert len(scenes) == 1
    for artist, flag in original_flags[0]:
        assert artist.get_animated() == flag
    assert record == original
    assert output.exists()


@pytest.mark.parametrize("east_wind_m_s", (-8.0, 8.0))
def test_windy_ascent_starts_vertical_before_following_air_relative_flow(east_wind_m_s):
    profile = ((0.0, east_wind_m_s), (PROFILE_TOP_M, east_wind_m_s))
    flight = AprFlight(
        conditions=AprConditions(wind_profile=profile, forecast_wind_profile=profile)
    )
    departure = flight.state.copy()
    departure[3] = 25.0
    flight.time_s = 5.0
    flow = np.rad2deg(np.arctan2(*flight.air_velocity(departure, predicted=True)))
    assert flight.loads(departure, predicted=True)[2] > 100.0
    assert abs(flow) > 10.0

    flight.guidance(departure)
    assert abs(np.rad2deg(flight.attitude_target)) < 0.1

    departure[3] = 200.0
    flight.time_s = 30.0
    flight.guidance(departure)
    flow = np.rad2deg(np.arctan2(*flight.air_velocity(departure, predicted=True)))
    assert abs(np.rad2deg(flight.attitude_target) - flow) <= 3.5 + 1e-9


@pytest.mark.parametrize("east_wind_m_s", (-8.0, 8.0))
def test_surface_wind_departure_keeps_climbing_without_early_windward_tip(east_wind_m_s):
    profile = ((0.0, east_wind_m_s), (PROFILE_TOP_M, east_wind_m_s))
    flight = AprFlight(
        conditions=AprConditions(
            wind_profile=profile,
            forecast_wind_profile=profile,
            gust_amplitude_m_s=5.0,
            gust_phase_rad=np.copysign(np.pi / 2, east_wind_m_s),
            gust_start_s=-60.0,
            gust_duration_s=120.0,
            gust_altitude_center_m=0.0,
            forecast_gust_amplitude_m_s=5.0,
        )
    )
    while flight.time_s < 10.0 and flight.outcome == "running":
        flight.advance()

    assert flight.outcome == "running"
    assert flight.upper_attached
    assert flight.height(flight.state) > 200.0
    assert flight.state[3] > 40.0
    assert abs(flight.state[2]) < 3.0
    assert abs(np.rad2deg(flight.state[4])) < 3.0
    assert flight.max_q_alpha < SAFETY.max_q_alpha_kpa_deg


def test_ascent_pitch_program_preserves_vertical_climb_until_high_altitude():
    flight = AprFlight()
    observed = flight.state.copy()
    observed[1] = 15_000.0
    flight.time_s = 50.0

    flight.guidance(observed)

    assert 0.0 < np.rad2deg(flight.attitude_target) < 10.0


def test_ascent_shutdown_keeps_main_reserve_through_allowed_command_delays():
    delayed = AprFlight(
        disturbances=RecoveryDisturbances(
            observation_delay_steps=2,
            action_delay_steps=1,
            engine_lag_s=0.0,
        )
    )
    observed = delayed.state.copy()
    observed[6] = delayed.ascent_settings.separation_min_main_propellant_kg + 75.0

    shutdown = delayed.guidance(observed)
    assert delayed.cutoff_pending
    assert shutdown[0] == 0.0

    immediate = AprFlight()
    immediate.guidance(observed)
    assert not immediate.cutoff_pending
    assert immediate.ascent_settings.separation_min_main_propellant_kg == 20_000.0
