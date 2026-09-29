"""3 t APR physics, PID, input, event and replay contracts."""

import json
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from powered_landing_guidance.apr_recovery import (
    OPERATING_SCOPE,
    UNCONNECTED_SYSTEMS,
    AprConditions,
    AprFlight,
    RecoveryDisturbances,
    load_apr_operating,
)
from powered_landing_guidance.integrators import integrate_fixed_step, rk4_step
from powered_landing_guidance.recovery_physics import (
    G0,
    RecoverySensorEstimator,
    atmosphere,
    clock_torque,
    engine_performance,
    foot_kinematics,
    gear_contacts,
    gravity,
    mass_gradients,
    mass_properties,
    rotation,
)
from powered_landing_guidance.visualization import (
    _nominal_flight_frame_times,
    _nominal_flight_geometry,
    _NominalFlightAnimationScene,
    _NominalFlightReplay,
    save_nominal_flight_replay,
)
from scripts import run_nominal_flight as apr_cli
from scripts.run_nominal_flight import (
    FINS,
    PHYSICS,
    AlignmentDescent,
    CoastEstimate,
    Design,
    HorizontalTransfer,
    NominalFlight,
    ReturnGuidanceSettings,
    fin_authority_table,
)

ROOT = Path(__file__).resolve().parents[1]


def apr_separation_state():
    return np.asarray(
        (307.0, 10000.0, 18.0, 124.0, np.deg2rad(9.0), 0.0, 461.5, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    )


def test_apr_formal_environment_and_old_cli_use_one_model():
    assert NominalFlight is AprFlight
    assert (
        sum(
            (
                PHYSICS.geometry.engine_mass_kg,
                PHYSICS.geometry.airframe_mass_kg,
                PHYSICS.geometry.gear_mass_kg,
                FINS.mass_kg,
            )
        )
        == Design().dry_mass_kg
    )
    assert PHYSICS.geometry.engine_mass_kg == 110.0
    assert PHYSICS.acceptance.max_position_error_m == 2.0


def test_apr_operating_config_loads_without_running_a_flight():
    conditions, disturbances = load_apr_operating(ROOT / "configs/apr-operating-v1.yaml")
    assert conditions.wind(10000.0, 3.0) == pytest.approx(14.5)
    assert conditions.wind(10000.0, 3.0, forecast=True) == 14.0
    assert disturbances.thrust_scale == 0.99


@pytest.mark.parametrize(
    "overrides",
    [
        {"health_checks_passed": False},
        {"known_fault": True},
        {"lightning_cloud_review_passed": False},
        {"temperature_review_passed": False},
        {"recovery_weather_review_passed": False},
        {"precipitation_mm_h": 0.01},
        {"thunderstorm_or_hail_present": True},
        {"health_checks_passed": 1},
        {"known_fault": 0},
        {"gust_amplitude_m_s": 2.01},
        {"gust_period_s": 7.99},
        {"gust_phase_rad": np.inf},
        {"sea_level_temperature_k": 270.0},
        {"sea_level_pressure_pa": 94000.0},
        {"gust_amplitude_m_s": True},
        {"wind_profile": ((0.0, 0.0), (10000.0, 0.0))},
        {"wind_profile": ((0.0, 0.0), (100.0, 0.0), (100.0, 0.0), (20000.0, 0.0))},
        {"wind_profile": ((0.0, 0.0), (1000.0, 4.01), (20000.0, 0.0))},
        {"wind_profile": ((0.0, 0.0), (10000.0, np.nan), (20000.0, 0.0))},
        {"wind_profile": ((0.0, 13.41), (20000.0, 13.41))},
        {"wind_profile": ((0.0, 0.0), (10000.0, 28.01), (20000.0, 0.0))},
        {"wind_profile": ((0.0, 0.0), (20000.0, 2.01))},
    ],
)
def test_apr_disallowed_weather_or_fault_is_rejected_not_a_success(overrides):
    with pytest.raises(ValueError):
        AprConditions(**overrides)


def test_apr_combined_wind_bounds_are_checked_between_different_profile_knots():
    with pytest.raises(ValueError, match="예보"):
        AprConditions(
            wind_profile=((0.0, 0.0), (20000.0, 0.0)),
            forecast_wind_profile=((0.0, 0.0), (1000.0, 2.0), (20000.0, 0.0)),
            gust_amplitude_m_s=0.1,
        )
    with pytest.raises(ValueError, match="지상"):
        AprConditions(
            wind_profile=((0.0, 13.0), (20000.0, 13.0)),
            forecast_wind_profile=((0.0, 13.0), (20000.0, 13.0)),
            gust_amplitude_m_s=0.5,
        )


def test_apr_boundary_inputs_and_above_separation_wind_are_explicit():
    profile = ((0.0, 11.4), (50.0, 11.4), (10000.0, 26.0), (20000.0, 26.0))
    conditions = AprConditions(
        wind_profile=profile, forecast_wind_profile=profile, gust_amplitude_m_s=2.0
    )
    assert conditions.wind(15000.0, 3.0) == pytest.approx(28.0)
    assert conditions.wind(50.0, 3.0) == pytest.approx(13.4)


@pytest.mark.parametrize(
    "field,value",
    [
        ("root_seed", 379),
        ("schema_version", True),
        ("extra", 1),
        ("vehicle", {"dry_mass_kg": 520.0}),
    ],
)
def test_apr_config_rejects_incompatible_or_incomplete_contract(tmp_path, field, value):
    data = yaml.safe_load((ROOT / "configs/apr-operating-v1.yaml").read_text(encoding="utf-8"))
    data[field] = value
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError):
        load_apr_operating(path)


@pytest.mark.parametrize(
    "group,field,value",
    [
        ("vehicle", "dry_mass_kg", 521.0),
        ("vehicle", "sea_level_thrust_n", 1000000.0),
        ("environment", "unknown", 0),
        ("disturbances", "thrust_scale", 0.8),
        ("disturbances", "inertia_scale", 0.8),
        ("disturbances", "sensor_noise_scale", 1.01),
        ("disturbances", "engine_lag_s", 0.051),
        ("disturbances", "action_delay_steps", 2),
        ("disturbances", "observation_delay_steps", 3),
    ],
)
def test_apr_config_disallows_unrealistic_vehicle_and_fault_ranges(tmp_path, group, field, value):
    data = yaml.safe_load((ROOT / "configs/apr-operating-v1.yaml").read_text(encoding="utf-8"))
    data[group][field] = value
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError):
        load_apr_operating(path)


def test_apr_cli_rejects_bad_inputs_before_creating_output(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("schema_version: [", encoding="utf-8")
    output = tmp_path / "must-not-exist"
    with pytest.raises(SystemExit) as error:
        apr_cli.main(["--conditions", str(path), "--output-dir", str(output)])
    assert error.value.code == 2
    assert not output.exists()


def test_apr_cli_input_check_has_no_integration_or_output(monkeypatch, tmp_path):
    monkeypatch.setattr(AprFlight, "run", lambda _: pytest.fail("input check ran a flight"))
    output = tmp_path / "must-not-exist"
    assert (
        apr_cli.main(
            [
                "--conditions",
                str(ROOT / "configs/apr-operating-v1.yaml"),
                "--output-dir",
                str(output),
                "--check-inputs",
            ]
        )
        == 0
    )
    assert not output.exists()


def test_apr_cli_accepts_current_separation_record_without_running(monkeypatch, tmp_path):
    monkeypatch.setattr(AprFlight, "run", lambda _: pytest.fail("input check ran a flight"))
    case = AprFlight.from_separation(apr_separation_state(), time_s=78.0, after_max_q=True)
    data = {"seed": 378, "design": asdict(Design()), "separation": case.separation}
    path = tmp_path / "record.json"
    path.write_text(json.dumps(data, default=lambda value: value.item()), encoding="utf-8")
    assert apr_cli.main(["--separation-record", str(path), "--check-inputs"]) == 0


def test_apr_changed_weather_does_not_silently_accept_an_invalid_separation(monkeypatch, tmp_path):
    monkeypatch.setattr(AprFlight, "run", lambda _: pytest.fail("invalid separation ran a flight"))
    case = AprFlight.from_separation(apr_separation_state(), time_s=78.0, after_max_q=True)
    data = {"seed": 378, "design": asdict(Design()), "separation": case.separation}
    path = tmp_path / "record.json"
    path.write_text(json.dumps(data, default=lambda value: value.item()), encoding="utf-8")
    output = tmp_path / "must-not-exist"
    with pytest.raises(SystemExit) as error:
        apr_cli.main(
            [
                "--separation-record",
                str(path),
                "--conditions",
                str(ROOT / "configs/apr-operating-v1.yaml"),
                "--output-dir",
                str(output),
            ]
        )
    assert error.value.code == 2
    assert not output.exists()


def test_apr_atmosphere_is_case_specific_and_shared_engine_has_compatible_defaults():
    case = AprFlight(
        conditions=AprConditions(sea_level_temperature_k=300.0, sea_level_pressure_pa=100000.0)
    )
    assert case.physics.atmosphere.sea_level_temperature_k == 300.0
    assert PHYSICS.atmosphere.sea_level_temperature_k == 288.15
    actual, _ = case.engine(0.5, 10000.0)
    shared, _ = engine_performance(
        0.5, 10000.0, case.physics, sea_level_thrust_n=42000.0, sea_level_isp_s=260.0
    )
    assert actual == shared
    assert engine_performance(1.0, 0.0, PHYSICS)[0] == 42000.0


def test_apr_true_wind_and_hidden_thrust_error_do_not_leak_into_guidance():
    forecast = ((0.0, 0.0), (20000.0, 0.0))
    positive = AprConditions(
        wind_profile=((0.0, 1.0), (20000.0, 1.0)), forecast_wind_profile=forecast
    )
    negative = AprConditions(
        wind_profile=((0.0, -1.0), (20000.0, -1.0)), forecast_wind_profile=forecast
    )
    initial = apr_separation_state()
    first = AprFlight.from_separation(initial, time_s=78.0, after_max_q=True, conditions=positive)
    second = AprFlight.from_separation(
        initial,
        time_s=78.0,
        after_max_q=True,
        conditions=negative,
        disturbances=RecoveryDisturbances(thrust_scale=0.98, inertia_scale=1.03),
    )
    np.testing.assert_allclose(first.guidance(initial), second.guidance(initial), atol=1e-12)
    assert first.loads(initial)[0][0] != second.loads(initial)[0][0]
    assert first.engine(0.2, 10000.0) == second.engine(0.2, 10000.0)
    assert second.engine(0.2, 10000.0, actual=True)[0] == pytest.approx(
        0.98 * first.engine(0.2, 10000.0, actual=True)[0]
    )


def test_apr_gust_uses_integration_stage_time_and_air_relative_fin_forces():
    conditions = AprConditions(gust_amplitude_m_s=1.0)
    case = AprFlight.from_separation(
        apr_separation_state(), time_s=0.0, after_max_q=True, conditions=conditions
    )
    state = case.state.copy()
    state[3] = -100.0
    state[4] = 0.0
    state[12] = 1.0
    assert case.wind_field(time_s=0.0).at(15000.0) == 0.0
    assert case.wind_field(time_s=3.0).at(15000.0) == 1.0
    assert case.fin_loads(state, time_s=0.0)[1] != case.fin_loads(state, time_s=3.0)[1]
    assert case.alpha_deg(state, time_s=0.0) != case.alpha_deg(state, time_s=3.0)
    assert not np.array_equal(
        case.derivative(state, (0.0, 0.0, 0.0, 0.0), time_s=0.0),
        case.derivative(state, (0.0, 0.0, 0.0, 0.0), time_s=3.0),
    )


def test_apr_noise_is_seeded_causal_and_does_not_change_the_true_state():
    options = dict(
        time_s=78.0,
        after_max_q=True,
        disturbances=RecoveryDisturbances(sensor_noise_scale=0.5, observation_delay_steps=2),
    )
    first = AprFlight.from_separation(apr_separation_state(), **options)
    second = AprFlight.from_separation(apr_separation_state(), **options)
    before = first.state.copy()
    for index in range(3):
        first.time_s = second.time_s = 78.0 + index * 0.01
        np.testing.assert_array_equal(first.observe_control_state(), second.observe_control_state())
    np.testing.assert_array_equal(first.state, before)
    assert first.observations[0][0] == 78.0
    assert not np.array_equal(first.observed_state[:6], before[:6])


def test_apr_first_order_engine_lag_is_integrated_without_minimum_throttle():
    case = AprFlight(disturbances=RecoveryDisturbances(engine_lag_s=0.02))
    command = (0.1, 0.0, 0.0, 0.0)
    assert case.applied_control(0.0, command, 0.0, 0.0)[0] == 0.0
    assert case.applied_control(0.02, command, 0.0, 0.0)[0] == pytest.approx(
        0.1 * (1.0 - np.exp(-1.0))
    )
    assert AprFlight().applied_control(0.0, command, 0.0, 0.0)[0] == 0.1
    assert case.applied_control(0.02, (0.0, 0.0, 0.0, 0.0), 0.0, 0.1)[0] > 0.0


def test_apr_action_delay_and_incremental_steps_preserve_fuel_balance():
    case = AprFlight.from_separation(
        apr_separation_state(),
        time_s=78.0,
        after_max_q=True,
        disturbances=RecoveryDisturbances(action_delay_steps=1, engine_lag_s=0.02),
    )
    case.guidance = lambda observed: (0.1, 0.0, 0.0, 0.0)
    case.advance()
    assert case.control == (0.0, 0.0, 0.0, 0.0)
    assert case.time_s == pytest.approx(78.01)
    for _ in range(19):
        case.advance()
    assert np.all(np.isfinite(case.state))
    assert case.main_consumed > 0.0
    assert case.max_mass_balance_error < 1e-8
    assert sum(case.initial_propellant) == pytest.approx(
        case.state[6] + case.state[7] + case.main_consumed + case.rcs_consumed
    )


def test_apr_recovery_entry_preserves_momentum_and_marks_ascent_unverified():
    initial = apr_separation_state()
    case = AprFlight.from_separation(initial, time_s=78.0, after_max_q=True)
    np.testing.assert_array_equal(case.state, initial)
    assert not case.upper_attached
    assert case.mass(case.state) == pytest.approx(520.0 + 461.5 + 15.0)
    case.outcome = "timeout"
    result = case.run()
    assert result["environment_version"] == OPERATING_SCOPE
    assert result["ascent_simulated"] is False
    assert result["legacy_evidence_transferred"] is False
    assert result["criteria"]["ascent_q"] is None
    assert result["simulated_acceptance_passed"] is False
    assert result["unconnected_systems"] == list(UNCONNECTED_SYSTEMS)
    assert result["external_disturbances"]["environment"] == asdict(AprConditions())


def test_apr_engine_lag_cannot_burn_unavailable_propellant():
    initial = apr_separation_state()
    initial[6] = 0.0001
    case = AprFlight.from_separation(
        initial,
        time_s=78.0,
        after_max_q=True,
        disturbances=RecoveryDisturbances(engine_lag_s=0.02),
    )
    case.guidance = lambda observed: (1.0, 0.0, 0.0, 0.0)
    for _ in range(3):
        case.advance()
    assert case.state[6] == 0.0
    assert case.main_consumed == pytest.approx(0.0001, abs=1e-8)
    assert case.max_mass_balance_error < 1e-8


@pytest.mark.parametrize(
    "index,value", [(1, 12000.0), (2, 61.0), (3, 0.0), (4, 0.0), (6, 0.0), (7, -0.1), (12, 1.0)]
)
def test_apr_recovery_entry_rejects_impossible_or_out_of_contract_states(index, value):
    initial = apr_separation_state()
    initial[index] = value
    with pytest.raises(ValueError):
        AprFlight.from_separation(initial, time_s=78.0, after_max_q=True)


def test_apr_profile_domain_and_noisy_observation_cannot_fabricate_success():
    case = AprFlight()
    case.outcome = "timeout"
    case.max_altitude = 20001.0
    case.observed_state[0] = 0.0
    case.state[0] = 3.0
    case.upper_attached = False
    case.state[8] = 1.0
    case.state[1] = case.properties(case.state).cg_m - case.physics.gear.foot_station_m
    case.first_contact(case.state, (0.0, 0.0, 0.0, 0.0))
    assert case.contact["criteria"]["position"] is False
    result = case.run()
    assert result["criteria"]["weather_profile_domain"] is False
    assert result["simulated_acceptance_passed"] is False


def test_apr_payload_30_kg_is_included_in_the_existing_upper_mass_budget():
    design = Design()
    assert design.payload_mass_kg == 30.0
    assert design.upper_stage_wet_mass_kg == 985.0
    assert design.upper_stage_wet_mass_kg + design.payload_mass_kg == design.upper_mass_kg
    assert design.launch_mass_kg == 3000.0


@pytest.mark.parametrize("payload", [-1.0, 1015.0, 1016.0, np.nan, np.inf])
def test_apr_payload_rejects_invalid_upper_mass_allocations(payload):
    with pytest.raises(ValueError, match="payload_mass_kg"):
        Design(payload_mass_kg=payload)


def test_apr_payload_allocation_does_not_change_lumped_first_stage_dynamics():
    reference, selected = NominalFlight(), NominalFlight()
    reference.design = replace(reference.design, payload_mass_kg=0.0)
    assert reference.mass(reference.state) == selected.mass(selected.state)
    assert reference.properties(reference.state) == selected.properties(selected.state)
    assert np.array_equal(
        reference.derivative(reference.state, reference.control),
        selected.derivative(selected.state, selected.control),
    )


def test_apr_orbit_mission_is_an_assumption_not_a_simulated_success():
    flight = NominalFlight()
    flight.outcome = "timeout"  # Inspect output without integrating a flight.
    result = flight.run()
    assert flight.time_s == 0.0
    assert result["design"]["payload_mass_kg"] == 30.0
    assert result["upper_stage_mission"] == {
        "payload_mass_kg": 30.0,
        "stage_wet_mass_kg": 985.0,
        "target_orbit_altitude_m": 700_000.0,
        "orbit_insertion_assumed": True,
        "orbit_insertion_simulated": False,
        "orbit_insertion_verified": False,
    }
    assert result["first_stage_mission"] == {
        "separation_altitude_m": 10_000.0,
        "separation_attitude_from_vertical_deg": 9.0,
        "separation_attitude_tolerance_deg": 0.5,
        "landing_target_x_m": 0.0,
        "horizontal_distance_prescribed": False,
        "horizontal_velocity_prescribed": False,
    }


def test_apr_ascent_gravity_turn_reference_reaches_nine_deg_without_state_clamps():
    flight = NominalFlight()
    y = flight.state.copy()
    y[1:4] = (10_000.0, 120.0 * np.tan(np.deg2rad(9.0)), 120.0)
    y[4] = np.deg2rad(9.0)
    before = y.copy()
    assert flight.ascent_attitude_target(y, 10_000.0) == pytest.approx(np.deg2rad(9.0))
    assert flight.ascent_command["flight_path_reference_deg"] == pytest.approx(9.0)
    assert flight.ascent_command["target_alpha_deg"] == pytest.approx(0.0, abs=1e-12)
    assert np.array_equal(y, before)


def test_apr_ascent_reference_turns_gradually_and_respects_steering_budgets():
    flight = NominalFlight()
    y = flight.state.copy()
    y[3] = 120.0
    references = []
    for height in (500.0, 1500.0, 5000.0, 8500.0, 10_000.0):
        y[1] = height
        target = flight.ascent_attitude_target(y, height)
        references.append(flight.ascent_command["flight_path_reference_deg"])
        alpha_limit = (
            flight.ascent_settings.preparation_alpha_reference_limit_deg
            if height >= 9000.0
            else flight.ascent_settings.alpha_reference_limit_deg
        )
        assert abs(np.rad2deg(target)) <= alpha_limit + 1e-12
    assert np.all(np.diff(references) > 0.0)
    assert references[-1] == pytest.approx(9.0)
    y[3] = 500.0
    flight.ascent_attitude_target(y, 10_000.0)
    assert flight.ascent_command["horizontal_speed_reference_m_s"] == 25.0
    y[0] = 900.0
    flight.ascent_attitude_target(y, 10_000.0)
    assert flight.ascent_command["horizontal_speed_reference_m_s"] <= 25.0


def test_apr_separation_requires_actual_nine_deg_and_keeps_existing_gates():
    flight = NominalFlight()
    flight.max_q_passed = True
    y = flight.state.copy()
    y[1:6] = (10_000.0, 120.0 * np.tan(np.deg2rad(9.0)), 120.0, np.deg2rad(9.0), 0.0)
    assert all(flight.separation_checks(y).values())
    y[4] = np.deg2rad(9.6)
    assert not flight.separation_checks(y)["mission_attitude"]
    y[4] = np.deg2rad(9.4)
    assert flight.separation_checks(y)["mission_attitude"]
    flight.max_q_passed = False
    assert flight.separation_checks(y)["after_observed_max_q"] is False
    y[5] = np.deg2rad(6.0)
    assert not flight.separation_checks(y)["angular_rate"]
    y[1] = 10_101.0
    assert not flight.separation_checks(y)["altitude"]


def test_apr_separation_failure_does_not_force_the_nine_deg_mission_state():
    flight = NominalFlight()
    flight.max_q_passed = True
    flight.state[1:6] = (10_101.0, 0.0, 120.0, 0.0, 0.0)
    before = flight.state.copy()
    flight.consider_separation(flight.state)
    assert flight.outcome == "separation_gate_failed"
    assert "separation_mission_attitude" in flight.failures
    assert flight.upper_attached
    assert flight.separation is None
    assert np.array_equal(flight.state, before)


def test_apr_fin_budget_is_inside_520_kg_and_fixed_masses_are_not_fuel():
    flight = NominalFlight()
    assert flight.mass(flight.state) == 3000.0
    assert FINS.mass_kg == 24.0
    g = PHYSICS.geometry
    assert g.engine_mass_kg + g.airframe_mass_kg + g.gear_mass_kg + FINS.mass_kg == 520.0
    flight.upper_attached = False
    y = flight.state.copy()
    y[6:8] = 0.0
    props = flight.properties(y)
    assert props.fuel_height_m == 0.0
    expected_cg = (
        110.0 * g.engine_station_m
        + 326.0 * g.airframe_station_m
        + 60.0 * 1.3
        + 24.0 * FINS.station_m
    ) / 520.0
    assert props.cg_m == pytest.approx(expected_cg)


def fin_bench():
    flight = NominalFlight(fin_mode="active")
    flight.upper_attached = False
    flight.phase = "coast"
    flight.state[1:4] = (2000.0, 0.0, -120.0)
    flight.state[6:8] = (462.0, 15.0)
    flight.state[12] = 1.0
    return flight


def test_apr_nine_deg_boostback_brakes_outbound_velocity_with_physical_controls():
    flight = fin_bench()
    flight.phase = "boostback"
    flight.return_settings = ReturnGuidanceSettings.for_stage("sync")
    flight.state[:6] = (350.0, 10_000.0, 19.0, 124.0, 0.0, 0.0)
    flight.state[12] = 0.0
    before = flight.state.copy()
    throttle, gimbal, _, _ = flight.guidance()
    assert throttle == pytest.approx(0.15)
    assert flight.attitude_target < 0.0
    assert abs(gimbal) <= np.deg2rad(flight.design.gimbal_limit_deg)
    assert flight.phase == "boostback"
    assert np.array_equal(flight.state, before)


def test_apr_boostback_records_the_actual_state_before_coasting():
    flight = fin_bench()
    flight.phase = "boostback"
    flight.return_settings = ReturnGuidanceSettings.for_stage("sync")
    flight.state[:6] = (350.0, 10_000.0, -20.0, 80.0, 0.0, 0.0)
    flight.state[12] = 0.0
    before = flight.state.copy()
    throttle, _, _, _ = flight.guidance()
    assert throttle == 0.0
    assert flight.phase == "coast"
    assert flight.boostback_completed["x_m"] == 350.0
    assert flight.boostback_completed["vx_m_s"] == -20.0
    assert flight.boostback_completed["main_propellant_kg"] == 462.0
    assert flight.boostback_completed["return_vx_reference_m_s"] < 0.0
    assert flight.boostback_completed["reference_model"] == "ballistic"
    assert np.array_equal(flight.state, before)


@pytest.mark.parametrize("vz", [124.0, 0.0, -100.0, -300.0])
def test_apr_coast_surrogate_obeys_its_analytic_drag_energy_and_velocity_decay(vz):
    estimate = CoastEstimate.between(10_000.0, vz, 2000.0, 1000.0, 0.668)
    k = 0.5 * estimate.mean_density_kg_m3 * 0.668 / 1000.0
    terminal_squared = gravity(10_000.0) / k
    rise = np.log1p(vz**2 / terminal_squared) / (2.0 * k) if vz > 0.0 else 0.0
    if vz >= 0.0:
        expected_downspeed_squared = terminal_squared * (1.0 - np.exp(-2 * k * (8000.0 + rise)))
    else:
        expected_downspeed_squared = terminal_squared + (vz**2 - terminal_squared) * np.exp(
            -2 * k * 8000.0
        )
    assert estimate.downspeed_m_s**2 == pytest.approx(expected_downspeed_squared)
    assert estimate.horizontal_velocity_ratio == pytest.approx(np.exp(-k * (8000.0 + 2 * rise)))
    assert 0.0 < estimate.horizontal_gain_s < estimate.duration_s
    assert 0.0 < estimate.horizontal_velocity_ratio < 1.0


def test_apr_coast_surrogate_handles_terminal_speed_and_zero_coast_distance():
    reference = CoastEstimate.between(10_000.0, -100.0, 2000.0, 1000.0, 0.668)
    k = 0.5 * reference.mean_density_kg_m3 * 0.668 / 1000.0
    terminal = np.sqrt(gravity(10_000.0) / k)
    estimate = CoastEstimate.between(10_000.0, -terminal, 2000.0, 1000.0, 0.668)
    assert estimate.duration_s == pytest.approx(8000.0 / terminal)
    assert estimate.downspeed_m_s == pytest.approx(terminal)
    stopped = CoastEstimate.between(2000.0, -100.0, 2000.0, 1000.0, 0.668)
    assert stopped.duration_s == 0.0
    assert stopped.horizontal_gain_s == 0.0
    assert stopped.horizontal_velocity_ratio == 1.0
    assert stopped.downspeed_m_s == 100.0


@pytest.mark.parametrize(
    "mass, area", [(0.0, 0.668), (-1.0, 0.668), (1000.0, 0.0), (np.nan, 0.668)]
)
def test_apr_coast_surrogate_rejects_invalid_physical_inputs(mass, area):
    with pytest.raises(ValueError, match="coast estimate"):
        CoastEstimate.between(10_000.0, -100.0, 2000.0, mass, area)


def test_apr_cumulative_guidance_stages_keep_the_original_limits_and_can_be_selected():
    approach = NominalFlight(guidance_stage="approach").return_settings
    sync = NominalFlight(guidance_stage="sync").return_settings
    joint = NominalFlight().return_settings
    assert not approach.synchronized_alignment and not approach.joint_return_planning
    assert sync.synchronized_alignment and not sync.joint_return_planning
    assert joint.synchronized_alignment and joint.joint_return_planning
    for settings in (approach, sync, joint):
        assert settings.approach_speed_limit_m_s == 20.0
        assert settings.load_governor_g == 4.75
        assert settings.alignment_vertical_speed_gate_m_s == 1.5
        assert settings.alignment_position_gate_m == 0.35
        assert settings.alignment_dwell_s == 1.0
        assert settings.boostback_tilt_limit_deg == 20.0
    with pytest.raises(ValueError, match="guidance stage"):
        NominalFlight(guidance_stage="other")


def test_apr_joint_boostback_waits_for_physical_orientation_without_replacing_the_state():
    flight = fin_bench()
    flight.phase = "boostback"
    flight.state[:6] = (350.0, 10_000.0, 19.0, 124.0, 0.0, 0.0)
    flight.state[12] = 0.0
    before = flight.state.copy()
    throttle, gimbal, torque, _ = flight.guidance()
    assert throttle == 0.0
    assert flight.attitude_target < 0.0
    assert abs(gimbal) <= np.deg2rad(flight.design.gimbal_limit_deg)
    assert abs(torque) <= flight.design.rcs_torque_limit_nm
    assert flight.return_plan["return_vx_reference_m_s"] < 0.0
    assert flight.return_plan["candidate_count"] == 3
    assert flight.return_plan["estimated_objective_propellant_kg"] >= 0.0
    assert np.array_equal(flight.state, before)
    flight.state[4] = np.deg2rad(-20.0)
    throttle, _, _, _ = flight.guidance()
    assert 0.0 < throttle <= flight.return_settings.boostback_throttle


def test_apr_joint_plan_is_mirrored_and_refreshes_only_on_its_control_interval():
    flights = []
    for sign in (1.0, -1.0):
        flight = fin_bench()
        flight.phase = "boostback"
        flight.state[:6] = (350.0 * sign, 10_000.0, 19.0 * sign, 124.0, 0.0, 0.0)
        flight.state[12] = 0.0
        flight.update_return_plan(flight.state)
        flights.append(flight)
    a, b = flights
    assert a.return_plan["return_vx_reference_m_s"] == pytest.approx(
        -b.return_plan["return_vx_reference_m_s"]
    )
    assert a.return_plan["estimated_objective_propellant_kg"] == pytest.approx(
        b.return_plan["estimated_objective_propellant_kg"]
    )
    original_plan = a.return_plan
    a.time_s = 0.4
    a.update_return_plan(a.state)
    assert a.return_plan is original_plan
    a.time_s = 0.5
    a.update_return_plan(a.state)
    assert a.return_plan is not original_plan


def test_apr_joint_ignition_includes_both_braking_and_horizontal_time_requirements():
    flight = fin_bench()
    y = flight.state
    vertical = flight.joint_ignition_height(y, 180.0, 0.0, 0.0)
    lateral = flight.joint_ignition_height(y, 180.0, 180.0, -0.7)
    faster = flight.joint_ignition_height(y, 230.0, 0.0, 0.0)
    assert lateral["horizontal_extra_time_s"] > 0.0
    assert lateral["ignition_height_m"] > vertical["ignition_height_m"]
    assert faster["ignition_height_m"] > vertical["ignition_height_m"]
    assert vertical["reaction_time_s"] > flight.return_settings.return_plan_interval_s
    assert vertical["braking_deceleration_m_s2"] <= 14.0


def test_apr_joint_alignment_keeps_a_vertical_braking_horizon_after_horizontal_completion():
    flight = fin_bench()
    flight.phase = "landing"
    flight.state[:6] = (0.1, 600.0, 0.01, -49.0, 0.0, 0.0)
    flight.horizontal_transfer = HorizontalTransfer.plan(0.0, 1.0, -0.1, flight.return_settings)
    flight.time_s = flight.horizontal_transfer.duration_s + 10.0
    flight.update_alignment(flight.state, 600.0)
    assert flight.alignment_descent is not None
    assert flight.alignment_descent.duration_s == pytest.approx(12.0)
    assert flight.alignment_descent.peak_deceleration_m_s2 <= 8.0
    assert flight.phase == "alignment"
    assert flight.alignment_completed is None


def test_apr_lift_and_torque_are_coupled_and_drag_dissipates_energy():
    flight = fin_bench()
    force, torque, hinge = flight.fin_loads(flight.state, np.deg2rad(10.0))
    assert torque > 0.0
    arm = FINS.station_m - flight.properties(flight.state).cg_m
    assert torque == pytest.approx(arm * force[0])
    assert force @ flight.state[2:4] < 0.0
    assert hinge > 0.0
    opposite = flight.fin_loads(flight.state, np.deg2rad(-10.0))
    assert opposite[1] == pytest.approx(-torque)


@pytest.mark.parametrize("case", ["stowed", "still", "attached"])
def test_apr_has_no_magic_aerodynamic_authority(case):
    flight = fin_bench()
    if case == "stowed":
        flight.state[12] = 0.0
    elif case == "still":
        flight.state[2:4] = 0.0
    else:
        flight.upper_attached = True
    force, torque, hinge = flight.fin_loads(flight.state, 0.1)
    assert np.array_equal(force, np.zeros(2))
    assert torque == hinge == 0.0


def test_apr_servo_rate_and_deflection_limit_and_actual_torque_allocation():
    flight = fin_bench()
    target = flight.fin_command(flight.state, 1e9)
    assert target == pytest.approx(np.deg2rad(FINS.deflection_limit_deg))
    derivative = flight.derivative(flight.state, (0.0, 0.0, 0.0, target))
    assert derivative[11] == pytest.approx(np.deg2rad(FINS.deflection_rate_deg_s))
    requested_target = np.deg2rad(2.0)
    _, unassisted_rcs, fin_target = flight.attitude(requested_target, 0.0)
    assert unassisted_rcs > 0.0  # The servo has not moved yet.
    flight.state[11] = fin_target
    _, assisted_rcs, _ = flight.attitude(requested_target, 0.0)
    assert abs(assisted_rcs) < abs(unassisted_rcs)


def test_apr_same_hardware_and_initial_state_in_both_comparison_modes():
    stowed, active = NominalFlight(fin_mode="stowed"), NominalFlight(fin_mode="active")
    assert np.array_equal(stowed.state, active.state)
    assert stowed.design == active.design
    assert np.array_equal(
        stowed.derivative(stowed.state, stowed.control),
        active.derivative(active.state, active.control),
    )


def test_apr_fin_authority_scales_with_q_not_a_guaranteed_rcs_saving():
    rows = fin_authority_table()
    assert rows[2]["max_fin_torque_nm"] / rows[1]["max_fin_torque_nm"] == pytest.approx(5.0)
    assert rows[-1]["max_fin_torque_nm"] > 6000.0
    assert rows[0]["max_fin_torque_nm"] < 6000.0


@pytest.mark.parametrize("fuel", [50.0, 462.0, 1450.0])
def test_apr_fixed_fin_mass_preserves_analytic_cg_and_inertia_gradients(fuel):
    mass, deployment, step = 520.0 + fuel, 0.5, 1e-3
    props = mass_properties(mass, deployment, PHYSICS)
    analytic = mass_gradients(mass, deployment, PHYSICS, props)
    plus = mass_properties(mass + step, deployment, PHYSICS)
    minus = mass_properties(mass - step, deployment, PHYSICS)
    assert analytic[0] == pytest.approx((plus.cg_m - minus.cg_m) / (2 * step), rel=1e-6)
    assert analytic[2] == pytest.approx(
        (plus.inertia_kg_m2 - minus.inertia_kg_m2) / (2 * step), rel=1e-6
    )


def test_apr_passive_fins_restore_a_small_descent_attitude_error():
    flight = fin_bench()
    flight.state[4] = np.deg2rad(5.0)
    assert flight.fin_loads(flight.state, 0.0)[1] < 0.0


def test_apr_feet_use_fin_inclusive_cg_and_not_a_second_fuel_allocation():
    flight = fin_bench()
    flight.state[8] = 1.0
    props = flight.properties(flight.state)
    flight.state[1] = props.cg_m - PHYSICS.gear.foot_station_m
    points, _ = foot_kinematics(flight.vector(flight.state), 1.0, PHYSICS)
    assert np.allclose(points[:, 1], 0.0, atol=1e-12)


def test_apr_envelope_failure_is_not_silently_accepted():
    flight = fin_bench()
    flight.outcome = "timeout"
    flight.max_deployed_fin_alpha_deg = 30.0
    result = flight.run()
    assert result["criteria"]["fin_model_envelope"] is False
    assert "fin_model_envelope" in result["failed_criteria"]
    assert result["simulated_acceptance_passed"] is False


def test_apr_powered_allocation_uses_tvc_and_only_residual_rcs():
    flight = fin_bench()
    flight.state[12] = 0.0
    flight.fin_share = 0.0
    gimbal, torque, _ = flight.attitude(np.deg2rad(1.0), 0.5)
    assert abs(gimbal) < np.deg2rad(flight.design.gimbal_limit_deg)
    assert abs(torque) < 1e-9
    derivative = flight.derivative(flight.state, (0.5, gimbal, torque, 0.0))
    assert np.rad2deg(derivative[5]) == pytest.approx(
        flight.last_allocation["desired_angular_acceleration_deg_s2"], abs=1e-9
    )


def test_apr_blended_fin_request_fades_to_assistance_not_duplicate_tvc():
    flight = fin_bench()
    flight.fin_share = 1.0
    flight.attitude(np.deg2rad(1.0), 0.5)
    unpowered_role = flight.last_allocation["fin_requested_torque_nm"]
    assert unpowered_role > 0.0
    flight.fin_share = 0.0
    flight.attitude(np.deg2rad(1.0), 0.5)
    assert flight.last_allocation["fin_requested_torque_nm"] == 0.0


def test_apr_fin_plan_does_not_deploy_at_apogee_and_retracts_before_alignment():
    flight = fin_bench()
    flight.state[12] = 0.0
    flight.state[3] = 0.0
    flight.update_fin_plan(flight.state, 10_000.0)
    assert not flight.fin_deployment_started
    flight.state[3] = -40.0
    flight.update_fin_plan(flight.state, 9000.0)
    assert flight.fin_deployment_started
    assert flight.fin_extension_rate(flight.state) == 0.5
    flight.state[12] = 1.0
    flight.phase = "landing"
    flight.update_fin_plan(flight.state, 1700.0)
    assert flight.fin_retraction_started
    assert flight.fin_extension_rate(flight.state) == -0.5
    flight.state[12] = 0.0
    assert flight.fin_extension_rate(flight.state) == 0.0


def test_apr_reference_has_slew_limit_and_low_q_fins_feather():
    flight = fin_bench()
    target = flight.filter_attitude_reference(np.deg2rad(10.0))
    assert (
        abs(target)
        <= np.deg2rad(flight.return_settings.attitude_reference_slew_deg_s)
        * flight.return_settings.control_dt_s
    )
    flight.state[3] = -5.0
    flight.state[4] = np.deg2rad(4.0)
    assert flight.fin_command(flight.state, 1e6) == pytest.approx(np.deg2rad(4.0))


@pytest.mark.parametrize("gimbal_deg", [-6.0, 0.0, 6.0])
def test_apr_total_load_governor_includes_aero_for_every_gimbal_direction(gimbal_deg):
    flight = fin_bench()
    flight.state[6] = 50.0
    flight.state[3] = -190.0
    throttle = flight.governed_throttle(1.0, flight.state)
    assert throttle < 1.0
    aero = flight.loads(flight.state)[0]
    thrust = flight.engine(throttle, flight.state[1])[0]
    angle = flight.state[4] + np.deg2rad(gimbal_deg)
    total = aero + thrust * np.array([np.sin(angle), np.cos(angle)])
    assert np.linalg.norm(total) / (flight.mass(flight.state) * G0) <= 4.75 + 1e-12


def test_apr_horizontal_pd_is_continuous_and_brakes_before_crossing_target():
    flight = fin_bench()
    flight.state[0], flight.state[2] = 10.0, -2.0
    assert flight.horizontal_acceleration(flight.state) > 0.0
    flight.state[0], flight.state[2] = -1e-6, 0.0
    left = flight.horizontal_acceleration(flight.state)
    flight.state[0] = 1e-6
    right = flight.horizontal_acceleration(flight.state)
    assert abs(left - right) < 1e-6
    s = flight.return_settings
    assert s.horizontal_kd_s_inv**2 == pytest.approx(4.0 * s.horizontal_kp_s2)


@pytest.mark.parametrize(
    ("position", "velocity"),
    [(0.0, 0.0), (180.0, -0.7), (-180.0, 0.7), (180.0, 5.0), (20.0, -8.0), (0.0, 4.0)],
)
def test_apr_horizontal_transfer_matches_endpoints_and_bounds_reference(position, velocity):
    settings = NominalFlight().return_settings
    transfer = HorizontalTransfer.plan(50.0, position, velocity, settings)
    assert transfer.duration_s >= settings.horizontal_transfer_min_duration_s
    assert transfer.reference(50.0) == pytest.approx((position, velocity, 0.0))
    end = 50.0 + transfer.duration_s
    assert transfer.reference(end) == (0.0, 0.0, 0.0)
    assert transfer.reference(end + 5.0) == (0.0, 0.0, 0.0)
    assert transfer.reference(end - 1e-6) == pytest.approx((0.0, 0.0, 0.0), abs=1e-5)
    assert (
        transfer.peak_reference_acceleration_m_s2 <= settings.horizontal_transfer_acceleration_m_s2
    )
    assert transfer.peak_reference_jerk_m_s3 <= settings.horizontal_transfer_jerk_m_s3
    assert (
        transfer.peak_compensated_acceleration_m_s2
        <= settings.horizontal_transfer_acceleration_m_s2
    )
    samples = [transfer.reference(t) for t in np.linspace(50.0, end, 101)]
    assert max(abs(row[2]) for row in samples) <= transfer.peak_reference_acceleration_m_s2 + 1e-10
    for t in np.linspace(50.0, end, 101):
        compensated = transfer.reference(t)[
            2
        ] + transfer.attitude_lag_compensation_s * transfer.jerk(t)
        assert abs(compensated) <= transfer.peak_compensated_acceleration_m_s2 + 1e-10
    assert transfer.jerk(end + 1.0) == 0.0


@pytest.mark.parametrize(
    "overrides",
    [
        {"horizontal_transfer_acceleration_m_s2": 0.0},
        {"horizontal_transfer_acceleration_m_s2": 1.0},
        {"horizontal_transfer_acceleration_m_s2": np.nan},
        {"horizontal_transfer_jerk_m_s3": -1.0},
        {"horizontal_transfer_jerk_m_s3": np.inf},
        {"horizontal_transfer_min_duration_s": 0.0},
        {"attitude_position_gain_s_inv": 0.0},
        {"attitude_reference_tau_s": -1.0},
    ],
)
def test_apr_horizontal_transfer_rejects_invalid_bounds(overrides):
    settings = replace(NominalFlight().return_settings, **overrides)
    with pytest.raises(ValueError, match="horizontal transfer"):
        HorizontalTransfer.plan(0.0, 180.0, -0.7, settings)


def test_apr_horizontal_transfer_is_symmetric_and_starts_from_measurements():
    flight = fin_bench()
    flight.state[:4] = (180.0, 2200.0, -0.7, -180.0)
    before = flight.state.copy()
    right = HorizontalTransfer.plan(10.0, 180.0, -0.7, flight.return_settings)
    left = HorizontalTransfer.plan(10.0, -180.0, 0.7, flight.return_settings)
    assert right.duration_s == left.duration_s
    for elapsed in np.linspace(0.0, right.duration_s, 11):
        assert right.reference(10.0 + elapsed) == pytest.approx(
            tuple(-value for value in left.reference(10.0 + elapsed))
        )
    flight.time_s = 10.0
    flight.horizontal_transfer = right
    assert flight.horizontal_acceleration(flight.state) == pytest.approx(
        right.attitude_lag_compensation_s * right.jerk(10.0)
    )
    assert np.array_equal(flight.state, before)


def test_apr_horizontal_transfer_adds_feed_forward_and_preserves_feedback_after_endpoint():
    flight = fin_bench()
    transfer = HorizontalTransfer.plan(0.0, 180.0, -0.7, flight.return_settings)
    flight.horizontal_transfer = transfer
    flight.time_s = 0.8 * transfer.duration_s
    x, vx, ax = transfer.reference(flight.time_s)
    flight.state[0], flight.state[2] = x, vx
    assert ax > 0.0
    compensated = ax + transfer.attitude_lag_compensation_s * transfer.jerk(flight.time_s)
    assert flight.horizontal_acceleration(flight.state) == pytest.approx(compensated)
    flight.state[0] += 1.0
    assert flight.horizontal_acceleration(flight.state) == pytest.approx(
        compensated - flight.return_settings.horizontal_kp_s2
    )
    flight.time_s = transfer.duration_s + 1.0
    flight.state[0], flight.state[2] = 10.0, -2.0
    expected = -flight.return_settings.horizontal_kp_s2 * 10.0
    expected += flight.return_settings.horizontal_kd_s_inv * 2.0
    assert flight.horizontal_acceleration(flight.state) == pytest.approx(expected)


def test_apr_finished_reference_cannot_replace_actual_alignment_requirements():
    flight = fin_bench()
    flight.state[:6] = (3.0, 350.0, 0.6, -0.5, 0.0, 0.0)
    flight.state[12] = 0.0
    flight.phase = "landing"
    flight.horizontal_transfer = HorizontalTransfer.plan(0.0, 180.0, -0.7, flight.return_settings)
    flight.update_alignment(flight.state, 350.0)
    flight.time_s = flight.horizontal_transfer.duration_s + 5.0
    assert flight.horizontal_transfer.reference(flight.time_s) == (0.0, 0.0, 0.0)
    flight.update_alignment(flight.state, 350.0)
    assert flight.phase == "alignment"
    assert flight.alignment_completed is None
    assert flight.state[0] == 3.0


def test_apr_alignment_fuel_accounting_uses_actual_event_inventories():
    flight = fin_bench()
    flight.phase = "landing"
    flight.state[:6] = (0.1, 350.0, 0.02, -0.5, np.deg2rad(0.2), 0.0)
    flight.state[12] = 0.0
    flight.time_s = 10.0
    flight.update_alignment(flight.state, 350.0)
    flight.state[6] -= 1.5
    flight.time_s = 11.01
    flight.update_alignment(flight.state, 350.0)
    assert flight.alignment_completed["duration_s"] == pytest.approx(1.01)
    assert flight.alignment_completed["main_propellant_used_kg"] == 1.5


def test_apr_alignment_descent_matches_measured_start_and_horizontal_deadline():
    settings = NominalFlight().return_settings
    plan = AlignmentDescent.plan(185.0, 600.0, -49.0, 17.0, settings)
    assert plan is not None
    assert plan.reference(185.0)[:2] == pytest.approx((600.0, -49.0))
    height, vz, acceleration = plan.reference(202.0)
    assert height == pytest.approx(plan.end_height_m)
    assert vz == -1.0
    assert acceleration == 0.0
    assert height > settings.alignment_min_height_m
    for elapsed in np.linspace(0.0, 17.0, 101):
        h, v, a = plan.reference(185.0 + elapsed)
        assert plan.end_height_m <= h <= 600.0
        assert -49.0 <= v <= -1.0
        assert 0.0 <= a <= settings.alignment_reference_deceleration_m_s2
    assert plan.reference(1000.0) == (250.0, 0.0, 0.0)


def test_apr_alignment_descent_height_derivative_matches_velocity_and_acceleration():
    plan = AlignmentDescent.plan(0.0, 600.0, -49.0, 35.0, NominalFlight().return_settings)
    assert plan is not None
    assert plan.speed_power > 2.0
    assert plan.end_height_m == pytest.approx(250.0)
    for t in (1.0, 10.0, 34.0):
        dt = 1e-4
        before, after = plan.reference(t - dt), plan.reference(t + dt)
        h, vz, az = plan.reference(t)
        assert h >= 250.0
        assert (after[0] - before[0]) / (2 * dt) == pytest.approx(vz, abs=1e-6)
        assert (after[1] - before[1]) / (2 * dt) == pytest.approx(az, abs=1e-6)


@pytest.mark.parametrize(
    "height, vz, duration",
    [(240.0, -49.0, 17.0), (600.0, 1.0, 17.0), (600.0, -49.0, 1.0), (600.0, -200.0, 10.0)],
)
def test_apr_alignment_descent_uses_existing_hold_when_timing_is_not_feasible(height, vz, duration):
    flight = fin_bench()
    assert AlignmentDescent.plan(0.0, height, vz, duration, flight.return_settings) is None
    expected = flight.return_settings.alignment_vertical_kp_s2 * (350.0 - height)
    expected -= flight.return_settings.alignment_vertical_kd_s_inv * vz
    flight.state[3] = vz
    assert flight.alignment_acceleration(flight.state, height) == pytest.approx(expected)


def test_apr_alignment_descent_does_not_overwrite_or_accept_the_measured_state():
    flight = fin_bench()
    flight.phase = "landing"
    flight.state[:6] = (30.0, 600.0, -4.0, -49.0, 0.0, 0.0)
    before = flight.state.copy()
    flight.horizontal_transfer = HorizontalTransfer.plan(0.0, 180.0, -0.7, flight.return_settings)
    flight.time_s = flight.horizontal_transfer.duration_s - 17.0
    flight.update_alignment(flight.state, 600.0)
    assert flight.alignment_descent is not None
    assert flight.alignment_completed is None
    assert flight.alignment_acceleration(flight.state, 600.0) > 0.0
    assert np.array_equal(before, flight.state)
    flight.time_s += 30.0
    flight.update_alignment(flight.state, 300.0)
    assert flight.phase == "alignment"
    assert flight.alignment_completed is None


@pytest.mark.parametrize("thrust_acceleration", [9.8, 20.0, 40.0])
def test_apr_linearized_lateral_attitude_filter_system_has_stable_poles(thrust_acceleration):
    # A small-signal algebraic check, not another simulated trajectory.
    flight = fin_bench()
    s = flight.return_settings
    props = flight.properties(flight.state)
    lever = props.cg_m - PHYSICS.geometry.nozzle_station_m
    b = props.inertia_kg_m2 / (flight.mass(flight.state) * lever)
    kp = s.attitude_rate_gain_s_inv * s.attitude_position_gain_s_inv
    kd, tau = s.attitude_rate_gain_s_inv, s.attitude_reference_tau_s
    a = thrust_acceleration
    matrix = np.array(
        [
            [0, 1, 0, 0, 0],
            [0, 0, a + b * kp, b * kd, -b * kp],
            [0, 0, 0, 1, 0],
            [0, 0, -kp, -kd, kp],
            [-s.horizontal_kp_s2 / (a * tau), -s.horizontal_kd_s_inv / (a * tau), 0, 0, -1 / tau],
        ]
    )
    assert np.max(np.linalg.eigvals(matrix).real) < 0.0


def test_apr_alignment_requires_dwell_and_stowed_fins_before_approach():
    flight = fin_bench()
    flight.phase = "landing"
    flight.state[:6] = (0.1, 350.0, 0.02, -0.5, np.deg2rad(0.2), 0.0)
    flight.state[12] = 0.0
    flight.update_alignment(flight.state, 350.0)
    assert flight.phase == "alignment"
    assert flight.alignment_completed is None
    flight.time_s = 0.5
    flight.state[12] = 0.1
    flight.update_alignment(flight.state, 350.0)
    assert flight.alignment_ready_since_s is None
    flight.state[12] = 0.0
    flight.update_alignment(flight.state, 350.0)
    flight.time_s = 1.51
    flight.update_alignment(flight.state, 350.0)
    assert flight.phase == "approach"
    assert flight.alignment_completed["foot_height_m"] > 200.0


def test_apr_approach_speed_reference_has_no_post_alignment_jump():
    flight = fin_bench()
    flight.phase = "approach"
    flight.approach_speed_reference = 0.5
    flight.state[3] = -0.5
    az = flight.descent_command(flight.state, 350.0)
    assert flight.approach_speed_reference == pytest.approx(0.52)
    assert az == pytest.approx(-2.018)
    assert flight.landing_tilt_limit(200.0) == pytest.approx(np.deg2rad(1.5))


def test_apr_faster_middle_approach_still_brakes_to_the_terminal_speed():
    flight = fin_bench()
    flight.phase = "approach"
    flight.approach_speed_reference = 20.0
    flight.state[3] = -20.0
    assert flight.return_settings.approach_speed_limit_m_s == 20.0
    assert flight.descent_command(flight.state, 200.0) == pytest.approx(0.0)
    assert flight.descent_command(flight.state, 50.0) > 0.0
    assert flight.approach_speed_reference == pytest.approx(19.98)
    flight.approach_speed_reference = 0.9
    flight.state[3] = -0.9
    assert flight.descent_command(flight.state, 2.0) == pytest.approx(0.0)


def test_apr_terminal_profile_keeps_09_speed_and_acceptance_not_relaxed():
    flight = fin_bench()
    flight.phase = "approach"
    flight.approach_speed_reference = 0.9
    flight.state[3] = -0.9
    assert flight.descent_command(flight.state, 2.0) == pytest.approx(0.0)
    flight.state[:6] = (0.0, 300.0, 0.0, -50.0, 0.0, 0.0)
    assert flight.alignment_ready(flight.state) is False


def test_apr_no_horizontal_only_burn_when_vertical_thrust_is_not_needed():
    flight = fin_bench()
    flight.phase = "landing"
    flight.state[0] = -30.0
    flight.state[3] = 0.0
    flight.state[12] = 0.0
    throttle, _, _, _ = flight.guidance()
    assert throttle == 0.0


def test_apr_strict_touchdown_limits_still_reject_lateral_error_on_static_state():
    # Only evaluate a contact state; do not integrate a second flight.
    flight = fin_bench()
    flight.state[8] = 1.0
    props = flight.properties(flight.state)
    flight.state[:6] = (3.0, props.cg_m - PHYSICS.gear.foot_station_m, 0.6, -0.9, 0.0, 0.0)
    flight.first_contact(flight.state, (0.0, 0.0, 0.0, 0.0))
    assert flight.contact["criteria"]["position"] is False
    assert flight.contact["criteria"]["horizontal_speed"] is False
    assert flight.contact["criteria"]["vertical_speed"] is True


def _synthetic_full_flight_record(base_height=20.0):
    """Handwritten poses for rendering tests, not an integrated flight."""
    from types import SimpleNamespace

    from powered_landing_guidance.recovery_physics import (
        AtmosphereSettings,
        Geometry,
        LandingGear,
        mass_properties,
    )

    geometry, gear = Geometry(), LandingGear()
    fixed = ((24.0, 4.65, 13.5),)
    settings = SimpleNamespace(geometry=geometry, gear=gear, fixed_mass_components=fixed)
    dry = geometry.engine_mass_kg + geometry.airframe_mass_kg + geometry.gear_mass_kg + 24
    mass, upper = dry + 110, 1015.0
    cg = mass_properties(mass, 1, settings).cg_m
    combined = (mass * cg + upper * 6) / (mass + upper)
    states = [
        [0, base_height + position, 0, -1, 0, 0, 100, 10, 1, 0, 0, 0.1, 1]
        for position in (combined, cg, cg)
    ]
    return {
        "state_names": [
            "x_m",
            "z_m",
            "vx_m_s",
            "vz_m_s",
            "theta_rad",
            "omega_rad_s",
            "main_propellant_kg",
            "rcs_propellant_kg",
            "gear_extension",
            "left_friction_m",
            "right_friction_m",
            "fin_deflection_rad",
            "fin_extension",
        ],
        "steps": [
            {"time_s": float(index), "state": state, "control": [1, 0, 0, 0], "phase": phase}
            for index, (state, phase) in enumerate(
                zip(states, ("ascent", "boostback", "ground_settling"), strict=True)
            )
        ],
        "design": asdict(Design()),
        "physics": {
            "geometry": asdict(geometry),
            "gear": asdict(gear),
            "atmosphere": asdict(AtmosphereSettings()),
            "fixed_mass_components": fixed,
        },
        "separation": {"time_s": 1.0},
        "first_contact": {
            "time_s": 1.5,
            "first_feet": [0],
            "foot_velocities_m_s": [[0.0, -0.9], [0.0, -0.9]],
        },
        "fin_settings": {"station_m": 4.65},
        "outcome": "unverified",
        "simulated_acceptance_passed": False,
    }


def test_nominal_flight_replay_preserves_body_position_across_separation():
    replay = _NominalFlightReplay(_synthetic_full_flight_record())
    np.testing.assert_allclose(replay.bases, [[0, 20]] * 3)
    for time_s in (0, 0.5, 0.999, 1, 1.001, 1.5, 2):
        state, base, mass, *_ = replay.sample(time_s)
        np.testing.assert_allclose(base, [0, 20])
        assert state[6] == 100 and state[7] == 10
        assert mass == replay.design["dry_mass_kg"] + 110


def test_nominal_flight_replay_stops_plume_at_recorded_contact():
    replay = _NominalFlightReplay(_synthetic_full_flight_record())
    assert replay.sample(1.49)[3][0] == 1
    assert replay.sample(1.5)[4] == "ground_settling"
    np.testing.assert_array_equal(replay.sample(1.5)[3], np.zeros(4))


def test_nominal_flight_replay_draws_grounded_feet_without_shifting_hull():
    replay = _NominalFlightReplay(_synthetic_full_flight_record(base_height=0.80))
    parts, nozzle, _ = _nominal_flight_geometry(replay, 2)
    assert parts["foot_left"][:, 1].min() == pytest.approx(0)
    assert parts["foot_right"][:, 1].min() == pytest.approx(0)
    assert parts["body"][:, 1].min() == pytest.approx(0.80)
    assert nozzle[1] == pytest.approx(0.80 + replay.settings.geometry.nozzle_station_m)


def test_nominal_flight_replay_uses_recorded_fin_extension():
    deployed = _synthetic_full_flight_record()
    stowed = deepcopy(deployed)
    for row in stowed["steps"]:
        row["state"][12] = 0
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


def test_nominal_flight_replay_never_overwrites_output(tmp_path):
    output = tmp_path / "flight.mp4"
    output.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        save_nominal_flight_replay({}, output)
    assert output.read_bytes() == b"existing"


def test_shared_3t_mass_gradients_match_finite_differences():
    mass, deployment = 800.0, 0.4
    props = mass_properties(mass, deployment, PHYSICS)
    cm, cg, im, ig = mass_gradients(mass, deployment, PHYSICS, props)
    epsilon = 1e-4
    before = mass_properties(mass - epsilon, deployment, PHYSICS)
    after = mass_properties(mass + epsilon, deployment, PHYSICS)
    assert cm == pytest.approx((after.cg_m - before.cg_m) / (2 * epsilon), abs=1e-8)
    assert im == pytest.approx((after.inertia_kg_m2 - before.inertia_kg_m2) / (2 * epsilon))
    before = mass_properties(mass, deployment - epsilon, PHYSICS)
    after = mass_properties(mass, deployment + epsilon, PHYSICS)
    assert cg == pytest.approx((after.cg_m - before.cg_m) / (2 * epsilon), abs=1e-8)
    assert ig == pytest.approx((after.inertia_kg_m2 - before.inertia_kg_m2) / (2 * epsilon))


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


def test_shared_contact_never_pulls_on_the_pad_or_exceeds_coulomb_friction():
    mass = 700.0
    props = mass_properties(mass, 1.0, PHYSICS)
    state = np.array(
        [0.0, props.cg_m - PHYSICS.gear.foot_station_m, 2.0, -0.5, 0.0, 0.0, mass, 1.0, 0.0, 0.0]
    )
    contact = gear_contacts(state, PHYSICS)
    assert np.all(contact["normal"] > 0.0)
    assert np.all(np.abs(contact["tangent"]) <= PHYSICS.gear.static_friction * contact["normal"])
    assert np.all(contact["tangent"] < 0.0)
    state[1] += 0.1
    airborne = gear_contacts(state, PHYSICS)
    np.testing.assert_array_equal(airborne["forces"], np.zeros((2, 2)))


def test_shared_estimator_preserves_propellant_and_actuator_telemetry():
    raw = apr_separation_state()
    raw[6:8] = (173.0, 14.7)
    estimator = RecoverySensorEstimator(0.5)
    estimator.observe(raw, 0.0, (0.0, 0.0), 0.0)
    observed = estimator.observe(raw, 0.01, (0.0, 0.0), 0.01)
    np.testing.assert_array_equal(observed[6:], raw[6:])
    assert observed[0] == pytest.approx(raw[0] + raw[2] * 0.01, abs=0.1)


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


@pytest.mark.parametrize("method", ["euler", "rk4"])
def test_integrator_shortens_the_final_step(method):
    times, states = integrate_fixed_step(
        lambda time, state: [2.0], [1.0], (0.0, 1.0), 0.3, method=method
    )
    np.testing.assert_allclose(times, [0.0, 0.3, 0.6, 0.9, 1.0])
    np.testing.assert_allclose(states[:, 0], 1.0 + 2.0 * times)


def test_failed_ascent_replay_does_not_fabricate_separation_or_contact():
    import matplotlib.pyplot as plt

    record = _synthetic_full_flight_record()
    record["separation"] = record["first_contact"] = None
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
        assert ("착륙 성공" in scene.event.get_text()) is passed
    finally:
        plt.close(scene.fig)


def test_short_synthetic_mp4_and_preview_can_be_saved_without_a_flight(tmp_path):
    from powered_landing_guidance.visualization import _ffmpeg_available

    if not _ffmpeg_available():
        pytest.skip("FFmpeg or the optional video extra is unavailable")
    output, preview = tmp_path / "poses.mp4", tmp_path / "poses.png"
    save_nominal_flight_replay(_synthetic_full_flight_record(), output, fps=1, preview=preview)
    assert b"ftyp" in output.read_bytes()[:32]
    assert preview.read_bytes().startswith(b"\x89PNG")
