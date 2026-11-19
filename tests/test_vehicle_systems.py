"""Inventory, conservation and domain tests for conceptual vehicle systems."""

from dataclasses import replace

import numpy as np
import pytest

from powered_landing_guidance.vehicle_systems import (
    FeedSettings,
    FeedState,
    IgnitionSettings,
    SeparationSettings,
    SloshSettings,
    SloshState,
    design_budget_check,
    feed_rates,
    ignition_ready,
    initial_feed_state,
    separation_event,
    slosh_displacement,
    slosh_response,
)


def test_initial_feed_inventory_is_not_extra_propellant():
    settings = FeedSettings()
    state = initial_feed_state(70_000.0, settings)
    result = feed_rates(70_000.0, 0.0, 9.80665, state, settings)
    assert state.buffer_kg == 1500.0
    assert state.tank_gas_kg + state.bottle_gas_kg == pytest.approx(65.0)
    assert result.tank_pressure_pa == pytest.approx(350_000.0)
    assert 20e6 < result.bottle_pressure_pa < 35e6
    assert result.inventory_valid
    assert result.delivered_flow_kg_s == 0
    assert np.allclose(result.state_derivatives, 0)


def test_free_fall_retains_buffer_without_fictitious_refill():
    settings = FeedSettings()
    state = initial_feed_state(22_000.0, settings)
    for _ in range(10_000):
        result = feed_rates(22_000.0, 0.0, 0.0, state, settings)
        state = FeedState(*(np.array(tuple_state(state)) + 0.02 * result.state_derivatives))
    assert state.buffer_kg == 1500
    assert state.acquisition < 1e-9
    assert result.inventory_valid
    assert result.pickup_flow_kg_s == 0


def tuple_state(state):
    return state.buffer_kg, state.acquisition, state.tank_gas_kg, state.bottle_gas_kg


def test_retained_buffer_bridges_restart_settling_without_mass_creation():
    settings = FeedSettings()
    main = 22_000.0
    state = replace(initial_feed_state(main, settings), acquisition=0.0)
    initial_main = main
    minimum_buffer = state.buffer_kg
    consumed = 0.0
    total_pickup = 0.0
    dt = 0.002
    for _ in range(4000):
        result = feed_rates(main, 543.85, 15.0, state, settings)
        assert result.inventory_valid
        assert result.delivered_flow_kg_s == pytest.approx(543.85)
        main -= result.delivered_flow_kg_s * dt
        consumed += result.delivered_flow_kg_s * dt
        total_pickup += result.pickup_flow_kg_s * dt
        state = FeedState(*(np.array(tuple_state(state)) + dt * result.state_derivatives))
        minimum_buffer = min(minimum_buffer, state.buffer_kg)
    assert minimum_buffer > 1250.0
    assert main + consumed == pytest.approx(initial_main, abs=1e-7)
    initial_bulk = initial_main - settings.buffer_capacity_kg
    assert main - state.buffer_kg == pytest.approx(initial_bulk - total_pickup, abs=1e-7)
    assert state.tank_gas_kg + state.bottle_gas_kg == pytest.approx(65.0, abs=1e-10)
    assert result.tank_pressure_pa == pytest.approx(350_000.0, abs=1.0)


@pytest.mark.parametrize("remaining", [0.0, 100.0, 1500.0, 70_000.0, 70_200.0])
def test_feed_initial_state_respects_remaining_main(remaining):
    state = initial_feed_state(remaining)
    result = feed_rates(remaining, 500, 0.0, state)
    assert result.inventory_valid
    assert 0 <= state.buffer_kg <= remaining
    assert result.delivered_flow_kg_s <= 500
    if remaining == 0:
        assert result.delivered_flow_kg_s == 0


def test_buffer_cannot_feed_engine_without_inlet_pressure():
    state = initial_feed_state(22_000.0)
    state = replace(state, tank_gas_kg=0.0, bottle_gas_kg=65.0)
    result = feed_rates(22_000.0, 500, 15, state)
    assert result.delivered_flow_kg_s == 0
    assert result.demand_shortfall_kg_s == 500
    assert result.tank_gas_derivative_kg_s > 0
    assert result.tank_gas_derivative_kg_s == -result.bottle_gas_derivative_kg_s


def test_empty_buffer_limits_requested_chamber_flow():
    state = replace(initial_feed_state(22_000.0), acquisition=0.0, buffer_kg=0.0)
    result = feed_rates(22_000.0, 500, 0.0, state)
    assert result.delivered_flow_kg_s == 0
    assert result.demand_shortfall_kg_s == 500
    assert result.buffer_derivative_kg_s == 0
    assert not ignition_ready(state, result.tank_pressure_pa)


@pytest.mark.parametrize(
    "mutation",
    [dict(buffer_kg=-1), dict(buffer_kg=1501), dict(acquisition=1.1), dict(bottle_gas_kg=-1)],
)
def test_invalid_feed_inventory_is_reported(mutation):
    state = replace(initial_feed_state(22_000.0), **mutation)
    assert not feed_rates(22_000.0, 0, 0, state).inventory_valid


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(main_tank_volume_m3=0),
        dict(minimum_pressure_pa=400_000),
        dict(buffer_capacity_kg=100_000),
        dict(helium_total_mass_kg=float("nan")),
        dict(acquisition_time_s=True),
    ],
)
def test_feed_settings_reject_invalid_design(kwargs):
    with pytest.raises(ValueError):
        FeedSettings(**kwargs)


def test_tank_overfill_and_insufficient_pressurant_rejected():
    with pytest.raises(ValueError, match="ullage"):
        initial_feed_state(100_000)
    with pytest.raises(ValueError, match="helium"):
        initial_feed_state(1000, FeedSettings(helium_total_mass_kg=2))


def test_restart_readiness_requires_both_pressure_and_buffer():
    state = initial_feed_state(22_000)
    assert ignition_ready(state, 350_000)
    assert not ignition_ready(state, 200_000)
    assert not ignition_ready(replace(state, buffer_kg=300), 350_000)
    with pytest.raises(ValueError):
        IgnitionSettings(delay_s=-0.1)


def _slosh(state=None, **kwargs):
    state = SloshState() if state is None else state
    values = dict(
        total_mass_kg=35_000.0,
        rigid_inertia_kg_m2=1_000_000.0,
        main_propellant_kg=22_000.0,
        station_offset_m=2.0,
        axial_proper_acceleration_m_s2=9.80665,
        lateral_proper_acceleration_m_s2=0.0,
        external_torque_nm=0.0,
        angular_rate_rad_s=0.0,
        state=state,
        settings=SloshSettings(),
    )
    values.update(kwargs)
    return slosh_response(**values)


def test_slosh_equilibrium_and_positive_definite_mass_matrix():
    result = _slosh()
    assert result.valid
    assert result.angular_acceleration_rad_s2 == 0
    assert result.displacement_acceleration_m_s2 == 0
    assert result.mass_matrix_determinant > 0
    assert result.stiffness_n_m > 0
    assert result.modal_mass_kg == 660.0


def test_slosh_reacts_back_on_body_without_external_torque():
    result = _slosh(SloshState(0.1, 0.0))
    assert result.displacement_acceleration_m_s2 < 0
    assert result.angular_acceleration_rad_s2 > 0
    assert (
        result.effective_inertia_kg_m2 * result.angular_acceleration_rad_s2
        + result.rotational_slosh_coupling_kg_m * result.displacement_acceleration_m_s2
    ) == pytest.approx(0.0, abs=1e-9)


def test_slosh_free_fall_has_capillary_not_terrestrial_gravity_stiffness():
    zero_g = _slosh(SloshState(0.1), axial_proper_acceleration_m_s2=0)
    one_g = _slosh(SloshState(0.1))
    assert 0 < zero_g.stiffness_n_m < 1e-4 * one_g.stiffness_n_m
    no_capillary = _slosh(
        SloshState(0.1),
        axial_proper_acceleration_m_s2=0,
        settings=SloshSettings(surface_tension_n_m=0),
    )
    assert no_capillary.stiffness_n_m == 0


def test_incompatible_slosh_allocation_is_rejected():
    with pytest.raises(ValueError, match="inertia"):
        _slosh(rigid_inertia_kg_m2=1000)
    with pytest.raises(ValueError):
        SloshSettings(modal_mass_fraction=0.7)
    with pytest.raises(ValueError):
        SloshSettings(pendulum_length_m=2)


@pytest.mark.parametrize("arc", [-6.0, -2.0, 0.0, 1.0, 6.0])
def test_nonlinear_mode_remains_in_physical_tank_without_arc_clipping(arc):
    settings = SloshSettings()
    u, tangent = slosh_displacement(SloshState(arc, 0.0), settings)
    result = _slosh(SloshState(arc, 0.0), settings=settings)
    assert abs(u[0]) <= settings.pendulum_length_m
    assert 0 <= u[1] <= 2 * settings.pendulum_length_m
    assert np.linalg.norm(tangent) == pytest.approx(1.0)
    assert result.valid
    assert result.mass_matrix_determinant > 0


@pytest.mark.parametrize("arc,qd,omega", [(0.4, -0.3, 0.2), (1.8, 1.1, -0.15), (-3.0, 0.6, 0.1)])
def test_nonlinear_energy_and_angular_momentum_balance(arc, qd, omega):
    settings = SloshSettings()
    torque, lateral, axial = 4000.0, -0.2, 3.0
    state = SloshState(arc, qd)
    result = _slosh(
        state,
        settings=settings,
        angular_rate_rad_s=omega,
        external_torque_nm=torque,
        lateral_proper_acceleration_m_s2=lateral,
        axial_proper_acceleration_m_s2=axial,
    )
    alpha, qdd = result.angular_acceleration_rad_s2, result.displacement_acceleration_m_s2
    m, mu, length = result.modal_mass_kg, result.reduced_modal_mass_kg, settings.pendulum_length_m
    angle = arc / length
    ai = 2 * (m * 2 + mu * length) * np.sin(angle)
    bi = -(m * 2 / length + mu) * np.sin(angle)
    b, a = result.rotational_slosh_coupling_kg_m, result.effective_inertia_kg_m2
    angular_momentum_rate = ai * qd * omega + a * alpha + bi * qd**2 + b * qdd
    assert angular_momentum_rate == pytest.approx(torque, abs=1e-8)
    # Finite-difference the complete mechanical energy, independently of RHS algebra.
    epsilon = 1e-6
    energies = []
    for direction in (-1, 1):
        other = _slosh(
            SloshState(arc + direction * epsilon * qd, qd + direction * epsilon * qdd),
            settings=settings,
            angular_rate_rad_s=omega + direction * epsilon * alpha,
            external_torque_nm=torque,
            lateral_proper_acceleration_m_s2=lateral,
            axial_proper_acceleration_m_s2=axial,
        )
        energies.append(other.kinetic_energy_j + other.potential_energy_j)
    energy_rate = (energies[1] - energies[0]) / (2 * epsilon)
    expected_rate = torque * omega - m * lateral * np.cos(angle) * qd - result.dissipated_power_w
    assert energy_rate == pytest.approx(expected_rate, abs=1e-5)


def test_slosh_mode_reports_excessive_constraint_force():
    state = SloshState(2.0, 1.0)
    assert _slosh(state, settings=SloshSettings()).valid
    assert not _slosh(state, settings=SloshSettings(maximum_mode_force_n=1.0)).valid


def test_nonlinear_free_fall_has_no_terrestrial_gravity_floor():
    result = _slosh(
        SloshState(1.0),
        settings=SloshSettings(surface_tension_n_m=0),
        axial_proper_acceleration_m_s2=0,
    )
    assert result.displacement_acceleration_m_s2 == 0
    assert result.angular_acceleration_rad_s2 == 0
    assert result.potential_energy_j == 0


def _separate(**kwargs):
    args = dict(
        position_m=[10_000, 37_000],
        velocity_m_s=[400, 950],
        theta_rad=0.3,
        angular_rate_rad_s=0.02,
        booster_mass_kg=35_000,
        upper_mass_kg=17_500,
        booster_cg_station_m=8,
        upper_cg_station_m=24,
        booster_inertia_kg_m2=1_000_000,
        upper_inertia_kg_m2=120_000,
    )
    args.update(kwargs)
    return separation_event(**args)


@pytest.mark.parametrize("theta", [-2.0, 0.0, 0.3, 1.5])
@pytest.mark.parametrize("offset", [-0.01, 0.0, 0.01])
def test_separation_conserves_both_momenta(theta, offset):
    event = _separate(theta_rad=theta, settings=SeparationSettings(lateral_offset_m=offset))
    assert np.linalg.norm(event["linear_momentum_residual_kg_m_s"]) < 1e-7
    assert abs(event["angular_momentum_residual_kg_m2_s"]) < 1e-6
    assert np.allclose(event["booster_impulse_n_s"], -np.array(event["upper_impulse_n_s"]))
    assert event["stored_energy_released_j"] > 0


def test_separation_energy_matches_reduced_mass_and_rotational_increments():
    event = _separate(
        velocity_m_s=[0, 0],
        angular_rate_rad_s=0,
        settings=SeparationSettings(lateral_offset_m=0),
    )
    reduced_mass = 35_000 * 17_500 / 52_500
    assert event["stored_energy_released_j"] == pytest.approx(0.5 * reduced_mass * 0.5**2)
    assert event["booster_angular_rate_rad_s"] == pytest.approx(0, abs=1e-14)
    assert event["upper_angular_rate_rad_s"] == pytest.approx(0, abs=1e-14)


def test_separation_misalignment_conserves_angular_momentum():
    event = _separate(settings=SeparationSettings(impulse_misalignment_deg=0.5))
    assert abs(event["angular_momentum_residual_kg_m2_s"]) < 1e-6
    assert event["booster_angular_rate_rad_s"] != 0.02
    assert event["pyroshock_resolved"] is False


@pytest.mark.parametrize("arc,qd", [(0.2, 0.7), (1.8, -1.2), (-3.0, 0.3)])
def test_nonlinear_separation_preserves_full_stack_internal_momentum(arc, qd):
    settings = SloshSettings()
    state = SloshState(arc, qd)
    event = _separate(
        main_propellant_kg=22_000,
        slosh_state=state,
        slosh_station_m=10,
        slosh_settings=settings,
        settings=SeparationSettings(impulse_misalignment_deg=0.5),
    )
    mass, m, omega = 52_500, 660, 0.02
    cg = (35_000 * 8 + 17_500 * 24) / mass
    rigid_i = 1_000_000 + 120_000 + 35_000 * (8 - cg) ** 2 + 17_500 * (24 - cg) ** 2
    u, tangent = slosh_displacement(state, settings)
    mu = m * (1 - m / mass)
    effective_i = rigid_i + 2 * m * (10 - cg) * u[1] + mu * float(u @ u)
    b = m * (10 - cg) * tangent[0] + mu * (u[1] * tangent[0] - u[0] * tangent[1])
    expected_h = effective_i * omega + b * qd
    assert event["pre_separation_angular_momentum_kg_m2_s"] == pytest.approx(expected_h, abs=1e-8)
    assert abs(event["angular_momentum_residual_kg_m2_s"]) < 1e-8
    assert np.linalg.norm(event["linear_momentum_residual_kg_m_s"]) < 1e-7


def test_separation_rejects_invalid_mass_and_normal_range():
    with pytest.raises(ValueError):
        _separate(booster_mass_kg=0)
    with pytest.raises(ValueError):
        SeparationSettings(impulse_misalignment_deg=10)


def test_conceptual_hardware_allocations_stay_within_existing_dry_mass():
    budget = design_budget_check()
    assert budget["theoretical_budget_consistent"]
    assert budget["total_systems_allocation_kg"] == 1015
    assert budget["other_airframe_systems_mass_kg"] == 7685
    assert budget["ideal_spherical_bottle_wall_lower_bound_kg"] < 160
    assert budget["buffer_full_flow_endurance_s"] > 2.7
    assert budget["hardware_certified"] is False
    assert not design_budget_check(airframe_systems_mass_kg=1000)["theoretical_budget_consistent"]


def test_new_main_inventory_fits_existing_theoretical_feed_budget():
    assert design_budget_check() == design_budget_check(initial_main_kg=70_200.0)
    budget = design_budget_check(initial_main_kg=70_200.0)
    assert budget["theoretical_budget_consistent"]
    assert budget["hardware_certified"] is False
    state = initial_feed_state(70_200.0)
    result = feed_rates(70_200.0, 0.0, 9.80665, state)
    assert state.buffer_kg == 1500.0
    assert state.tank_gas_kg + state.bottle_gas_kg == pytest.approx(65.0)
    assert result.inventory_valid
    assert result.tank_pressure_pa == pytest.approx(350_000.0)
