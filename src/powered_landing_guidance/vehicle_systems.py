"""Reduced vehicle-system models with explicit inventories and validity limits.

These are conceptual single-fluid engineering analogues, not component test
correlations. The gas is closed (bottle to ullage, no venting), the feed buffer
is a subset of main propellant, and slosh cannot accelerate the total COM.
See NASA NTRS 20100022074 (pressurization), 20210000785 (mechanical slosh),
and 20250000297 (spring separation and tip-off) for the modelling principles.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from math import cos, isfinite, pi, sin, sqrt, tanh

import numpy as np


def _positive_settings(value, zero_allowed=()):
    for field in fields(value):
        item = getattr(value, field.name)
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not isfinite(item):
            raise ValueError(f"{field.name} must be a finite number")
        if item < 0 or (item == 0 and field.name not in zero_allowed):
            raise ValueError(f"{field.name} must be positive")


@dataclass(frozen=True, slots=True)
class FeedSettings:
    """Pump-fed tank inlet model; tank pressure is NOT chamber pressure.

    The 1,500 kg retained buffer bridges 2.76 s at the nominal 543.8 kg/s
    full-flow demand, versus assumed 0.35 s bulk settling time constant.
    PMD/check-valve retention and regulator performance require hardware tests.
    The 65 kg helium inventory is allocated inside existing dry systems mass,
    not added to either launch mass or propellant. Isothermal gas and one
    equivalent liquid omit cryogenic collapse, two-species mixing and boiling.
    """

    main_tank_volume_m3: float = 13.0 * pi * 1.5**2
    liquid_density_kg_m3: float = 826.0
    buffer_capacity_kg: float = 1500.0
    maximum_flow_kg_s: float = 650.0
    buffer_fill_time_s: float = 0.5
    flow_emptying_time_s: float = 0.05
    acquisition_time_s: float = 0.35
    acquisition_loss_time_s: float = 8.0
    settled_acceleration_m_s2: float = 0.3
    regulated_pressure_pa: float = 350_000.0
    minimum_pressure_pa: float = 220_000.0
    bottle_volume_m3: float = 1.3
    helium_total_mass_kg: float = 65.0
    gas_temperature_k: float = 290.0
    helium_specific_gas_constant_j_kg_k: float = 2077.1
    regulator_time_s: float = 0.1
    regulator_maximum_flow_kg_s: float = 0.6
    regulator_head_pa: float = 50_000.0

    def __post_init__(self):
        _positive_settings(self)
        if self.minimum_pressure_pa >= self.regulated_pressure_pa:
            raise ValueError("minimum pressure must be below regulated pressure")
        if self.buffer_capacity_kg >= self.main_tank_volume_m3 * self.liquid_density_kg_m3:
            raise ValueError("buffer capacity must be within the main tank inventory")


@dataclass(frozen=True, slots=True)
class FeedState:
    buffer_kg: float
    acquisition: float
    tank_gas_kg: float
    bottle_gas_kg: float


@dataclass(frozen=True, slots=True)
class FeedRates:
    delivered_flow_kg_s: float
    pickup_flow_kg_s: float
    buffer_derivative_kg_s: float
    acquisition_derivative_s_inv: float
    tank_gas_derivative_kg_s: float
    bottle_gas_derivative_kg_s: float
    tank_pressure_pa: float
    bottle_pressure_pa: float
    demand_shortfall_kg_s: float
    inventory_valid: bool

    @property
    def state_derivatives(self):
        return np.array(
            [
                self.buffer_derivative_kg_s,
                self.acquisition_derivative_s_inv,
                self.tank_gas_derivative_kg_s,
                self.bottle_gas_derivative_kg_s,
            ]
        )


DEFAULT_FEED = FeedSettings()


def initial_feed_state(main_propellant_kg: float, settings: FeedSettings = DEFAULT_FEED):
    if not isfinite(main_propellant_kg) or main_propellant_kg < 0:
        raise ValueError("main propellant must be finite and nonnegative")
    ullage = settings.main_tank_volume_m3 - main_propellant_kg / settings.liquid_density_kg_m3
    if ullage <= 0:
        raise ValueError("main tank requires positive ullage volume")
    gas_rt = settings.helium_specific_gas_constant_j_kg_k * settings.gas_temperature_k
    tank_gas = settings.regulated_pressure_pa * ullage / gas_rt
    bottle_gas = settings.helium_total_mass_kg - tank_gas
    if bottle_gas <= 0 or bottle_gas * gas_rt / settings.bottle_volume_m3 <= (
        settings.regulated_pressure_pa + settings.regulator_head_pa
    ):
        raise ValueError("insufficient helium for initial tank pressure and regulator head")
    return FeedState(
        min(main_propellant_kg, settings.buffer_capacity_kg), 1.0, tank_gas, bottle_gas
    )


def feed_rates(
    main_propellant_kg: float,
    demanded_flow_kg_s: float,
    axial_proper_acceleration_m_s2: float,
    state: FeedState,
    settings: FeedSettings = DEFAULT_FEED,
) -> FeedRates:
    """Continuous rates; callers integrate all four states with main mass.

    Main_dot=-delivered, buffer_dot=pickup-delivered and bulk_dot=-pickup.
    Consequently buffer refill never consumes propellant a second time. A
    retained buffer does not drain in free fall while the engine is off.
    Supply limiting acts on chamber mass flow AND thrust in the plant.
    """
    main = max(0.0, main_propellant_kg)
    buffer = min(main, max(0.0, state.buffer_kg))
    acquisition = min(1.0, max(0.0, state.acquisition))
    ullage = max(1e-9, settings.main_tank_volume_m3 - main / settings.liquid_density_kg_m3)
    gas_rt = settings.helium_specific_gas_constant_j_kg_k * settings.gas_temperature_k
    tank_pressure = max(0.0, state.tank_gas_kg) * gas_rt / ullage
    bottle_pressure = max(0.0, state.bottle_gas_kg) * gas_rt / settings.bottle_volume_m3
    pressure_margin = min(
        1.0,
        max(
            0.0,
            (tank_pressure - settings.minimum_pressure_pa)
            / (settings.regulated_pressure_pa - settings.minimum_pressure_pa),
        ),
    )
    pressure_flow_limit = settings.maximum_flow_kg_s * sqrt(pressure_margin)
    demand = max(0.0, demanded_flow_kg_s)
    # Required pickup includes chamber demand plus restoration of buffer fill.
    desired_pickup = (
        demand + max(0.0, settings.buffer_capacity_kg - buffer) / settings.buffer_fill_time_s
    )
    pickup = min(
        acquisition * pressure_flow_limit,
        desired_pickup,
        max(0.0, main - buffer) / settings.flow_emptying_time_s,
    )
    delivered = min(demand, pressure_flow_limit, pickup + buffer / settings.flow_emptying_time_s)
    settled_fraction = min(
        1.0, max(0.0, axial_proper_acceleration_m_s2 / settings.settled_acceleration_m_s2)
    )
    acquisition_tau = (
        settings.acquisition_time_s
        if settled_fraction > acquisition
        else settings.acquisition_loss_time_s
    )
    target_gas = settings.regulated_pressure_pa * ullage / gas_rt
    gas_feedforward = (
        settings.regulated_pressure_pa * delivered / (settings.liquid_density_kg_m3 * gas_rt)
    )
    requested_gas = max(
        0.0, (target_gas - state.tank_gas_kg) / settings.regulator_time_s + gas_feedforward
    )
    regulator_fraction = min(
        1.0,
        max(0.0, (bottle_pressure - tank_pressure) / settings.regulator_head_pa),
    )
    transferred_gas = min(
        requested_gas,
        regulator_fraction * settings.regulator_maximum_flow_kg_s,
        max(0.0, state.bottle_gas_kg) / settings.regulator_time_s,
    )
    valid = (
        isfinite(main_propellant_kg)
        and main_propellant_kg >= -1e-6
        and -1e-6 <= state.buffer_kg <= min(main, settings.buffer_capacity_kg) + 1e-6
        and -1e-8 <= state.acquisition <= 1 + 1e-8
        and state.tank_gas_kg >= -1e-8
        and state.bottle_gas_kg >= -1e-8
        and abs(state.tank_gas_kg + state.bottle_gas_kg - settings.helium_total_mass_kg) <= 1e-6
        and main / settings.liquid_density_kg_m3 < settings.main_tank_volume_m3
    )
    return FeedRates(
        delivered,
        pickup,
        pickup - delivered,
        (settled_fraction - acquisition) / acquisition_tau,
        transferred_gas,
        -transferred_gas,
        tank_pressure,
        bottle_pressure,
        demand - delivered,
        bool(valid),
    )


@dataclass(frozen=True, slots=True)
class IgnitionSettings:
    """Assumed restart readiness and delay, not ignition/reliability chemistry."""

    delay_s: float = 0.2
    minimum_buffer_kg: float = 350.0
    minimum_pressure_pa: float = 280_000.0

    def __post_init__(self):
        _positive_settings(self, ("delay_s", "minimum_buffer_kg"))


DEFAULT_IGNITION = IgnitionSettings()


def ignition_ready(state: FeedState, tank_pressure_pa: float, settings=DEFAULT_IGNITION):
    return bool(
        state.buffer_kg >= settings.minimum_buffer_kg
        and tank_pressure_pa >= settings.minimum_pressure_pa
    )


@dataclass(frozen=True, slots=True)
class SloshSettings:
    """One baffled lateral mode, with provisional coefficients.

    Modal participation 3% and damping 12% are design assumptions needing tank
    correlation, not universal liquid properties. Capillary stiffness remains
    at zero proper acceleration; terrestrial gravity is never added in free
    fall. Default q is nonlinear pendulum arc length, not lateral displacement.
    """

    modal_mass_fraction: float = 0.03
    tank_radius_m: float = 1.5
    tank_length_m: float = 13.0
    pendulum_length_m: float = 0.8
    liquid_density_kg_m3: float = 826.0
    surface_tension_n_m: float = 0.025
    damping_ratio: float = 0.12
    viscous_decay_rate_s_inv: float = 0.08
    maximum_mode_force_n: float = 100_000.0

    def __post_init__(self):
        _positive_settings(
            self, ("surface_tension_n_m", "damping_ratio", "viscous_decay_rate_s_inv")
        )
        if not 0 < self.modal_mass_fraction < 0.5:
            raise ValueError("slosh modal fraction must be in (0, 0.5)")
        if (
            self.pendulum_length_m >= self.tank_radius_m
            or 2 * self.pendulum_length_m > self.tank_length_m
        ):
            raise ValueError("pendulum excursion must fit inside the equivalent tank")


@dataclass(frozen=True, slots=True)
class SloshState:
    displacement_m: float = 0.0
    velocity_m_s: float = 0.0


@dataclass(frozen=True, slots=True)
class SloshResponse:
    angular_acceleration_rad_s2: float
    displacement_acceleration_m_s2: float
    modal_mass_kg: float
    stiffness_n_m: float
    damping_n_s_m: float
    effective_inertia_kg_m2: float
    mass_matrix_determinant: float
    kinetic_energy_j: float
    potential_energy_j: float
    dissipated_power_w: float
    valid: bool
    modal_constraint_force_n: float = 0.0
    rotational_slosh_coupling_kg_m: float = 0.0
    reduced_modal_mass_kg: float = 0.0


DEFAULT_SLOSH = SloshSettings()


def slosh_displacement(state: SloshState, settings: SloshSettings = DEFAULT_SLOSH):
    """Body-frame displacement from settled mode and derivative per arc length."""
    length = settings.pendulum_length_m
    angle = state.displacement_m / length
    return np.array([length * sin(angle), length * (1 - cos(angle))]), np.array(
        [cos(angle), sin(angle)]
    )


def _mode_mass_coefficients(mass, inertia, modal_mass, station, state, settings):
    u, tangent = slosh_displacement(state, settings)
    mu = modal_mass * (1.0 - modal_mass / mass)
    effective_i = inertia + 2 * modal_mass * station * u[1] + mu * float(u @ u)
    coupling = modal_mass * station * tangent[0] + mu * _clock_moment(u, tangent)
    inertia_gradient = 2 * (modal_mass * station + mu * settings.pendulum_length_m) * tangent[1]
    coupling_gradient = -(modal_mass * station / settings.pendulum_length_m + mu) * tangent[1]
    return u, tangent, mu, effective_i, coupling, inertia_gradient, coupling_gradient


def slosh_response(
    total_mass_kg: float,
    rigid_inertia_kg_m2: float,
    main_propellant_kg: float,
    station_offset_m: float,
    axial_proper_acceleration_m_s2: float,
    lateral_proper_acceleration_m_s2: float,
    external_torque_nm: float,
    angular_rate_rad_s: float,
    state: SloshState,
    settings: SloshSettings = DEFAULT_SLOSH,
) -> SloshResponse:
    """Solve the coupled rotational/lateral mass matrix about total COM.

    The nonlinear pendulum uses its exact COM-reduced kinetic metric and full
    Coriolis/centrifugal terms; proper axial acceleration supplies restoring
    force through the curved trajectory, not an artificial gravity floor.
    Eliminating rigid lateral translation leaves its total-COM acceleration
    unchanged; the slosh coordinate is driven by proper lateral acceleration.
    q points along body +x and theta is clockwise from +z. The passed torque
    may already include the rigid inertia-rate and exhaust-momentum terms.
    Modal coefficients are frozen at each RHS evaluation; distributed outflow,
    nonlinear waves, baffle impact and changing modal-shape momentum are not
    resolved. This is a coupled engineering analogue, not full fluid dynamics.
    """
    if total_mass_kg <= 0 or rigid_inertia_kg_m2 <= 0:
        raise ValueError("positive vehicle mass and inertia are required")
    if main_propellant_kg < 0 or main_propellant_kg > total_mass_kg:
        raise ValueError("main propellant must lie within total vehicle mass")
    m = settings.modal_mass_fraction * main_propellant_kg
    q, qd, w = state.displacement_m, state.velocity_m_s, angular_rate_rad_s
    if m <= 1e-9:
        return SloshResponse(
            external_torque_nm / rigid_inertia_kg_m2,
            0.0,
            m,
            0.0,
            0.0,
            rigid_inertia_kg_m2,
            0.0,
            0.5 * rigid_inertia_kg_m2 * w**2,
            0.0,
            0.0,
            bool(np.all(np.isfinite((q, qd, w, external_torque_nm)))),
        )
    return _pendulum_response(
        total_mass_kg,
        rigid_inertia_kg_m2,
        main_propellant_kg,
        station_offset_m,
        axial_proper_acceleration_m_s2,
        lateral_proper_acceleration_m_s2,
        external_torque_nm,
        angular_rate_rad_s,
        state,
        settings,
    )


def _pendulum_response(
    mass, inertia, main, station, axial, lateral, torque, omega, state, settings
):
    m = settings.modal_mass_fraction * main
    u, tangent, mu, effective_i, b, ai, bi = _mode_mass_coefficients(
        mass, inertia, m, station, state, settings
    )
    determinant = effective_i * mu - b**2
    if determinant <= 0 or effective_i <= 0:
        raise ValueError("pendulum mass allocation is incompatible with rigid inertia")
    length, qd = settings.pendulum_length_m, state.velocity_m_s
    depth = main / (settings.liquid_density_kg_m3 * pi * settings.tank_radius_m**2)
    wave_number = 1.841183781 / settings.tank_radius_m
    capillary_frequency_squared = (
        settings.surface_tension_n_m
        / settings.liquid_density_kg_m3
        * wave_number**3
        * tanh(wave_number * depth)
    )
    frequency_squared = axial / length + capillary_frequency_squared
    damping = m * (
        2 * settings.damping_ratio * sqrt(max(0.0, frequency_squared))
        + settings.viscous_decay_rate_s_inv
    )
    rotational_rhs = torque - ai * qd * omega - bi * qd**2
    modal_rhs = (
        0.5 * ai * omega**2
        - m * (lateral * tangent[0] + axial * tangent[1])
        - m * length * capillary_frequency_squared * tangent[1]
        - damping * qd
    )
    alpha = (mu * rotational_rhs - b * modal_rhs) / determinant
    qdd = (effective_i * modal_rhs - b * rotational_rhs) / determinant
    kinetic = 0.5 * effective_i * omega**2 + b * omega * qd + 0.5 * mu * qd**2
    # Includes the frozen-axial-acceleration potential for the energy audit.
    potential = m * (axial + length * capillary_frequency_squared) * u[1]
    eta = m / mass
    arm = np.array([0.0, station]) + (1 - eta) * u
    arm_rate = (1 - eta) * tangent * qd
    curvature = np.array([-tangent[1], tangent[0]]) / length
    arm_acceleration = (1 - eta) * (tangent * qdd + curvature * qd**2)
    proper_mode_acceleration = (
        np.array([lateral, axial])
        + alpha * np.array([arm[1], -arm[0]])
        - omega**2 * arm
        + 2 * omega * np.array([arm_rate[1], -arm_rate[0]])
        + arm_acceleration
    )
    constraint_force = m * float(np.linalg.norm(proper_mode_acceleration))
    valid = (
        bool(np.all(np.isfinite([alpha, qdd, kinetic, potential, constraint_force])))
        and abs(u[0]) <= settings.tank_radius_m
        and -1e-12 <= u[1] <= settings.tank_length_m
        and constraint_force <= settings.maximum_mode_force_n
    )
    return SloshResponse(
        alpha,
        qdd,
        m,
        m * frequency_squared,
        damping,
        effective_i,
        determinant,
        kinetic,
        potential,
        damping * qd**2,
        valid,
        constraint_force,
        b,
        mu,
    )


@dataclass(frozen=True, slots=True)
class SeparationSettings:
    """Finite-energy pusher impulse; not a pyroshock/structural transient solver."""

    relative_axial_speed_m_s: float = 0.5
    application_station_m: float = 20.0
    lateral_offset_m: float = 0.002
    impulse_misalignment_deg: float = 0.0
    effective_duration_s: float = 0.15

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isfinite(value):
                raise ValueError(f"{field.name} must be finite")
        if self.relative_axial_speed_m_s <= 0 or self.effective_duration_s <= 0:
            raise ValueError("separation speed and pulse duration must be positive")
        if abs(self.lateral_offset_m) > 0.1 or abs(self.impulse_misalignment_deg) > 1:
            raise ValueError("separation offset/misalignment exceeds the conceptual normal range")


def _clock_moment(arm, force):
    return float(arm[1] * force[0] - arm[0] * force[1])


DEFAULT_SEPARATION = SeparationSettings()


def separation_event(
    position_m,
    velocity_m_s,
    theta_rad: float,
    angular_rate_rad_s: float,
    booster_mass_kg: float,
    upper_mass_kg: float,
    booster_cg_station_m: float,
    upper_cg_station_m: float,
    booster_inertia_kg_m2: float,
    upper_inertia_kg_m2: float,
    settings: SeparationSettings = DEFAULT_SEPARATION,
    *,
    main_propellant_kg: float = 0.0,
    slosh_state: SloshState | None = None,
    slosh_station_m: float | None = None,
    slosh_settings: SloshSettings = DEFAULT_SLOSH,
):
    """Apply equal/opposite impulses at the same interface point.

    Both stages inherit V+omega-cross-r from the attached stack. Returning the
    upper-stage state permits auditing linear/angular momentum even when that
    stage is not propagated. Instantaneous impulse and estimated pulse-average
    acceleration are reported separately; delta-omega/dt is not a resolved
    pyroshock peak. Stored pusher energy accounts for the kinetic-energy gain.
    If slosh is supplied, both actual COM offsets and internal momentum are
    retained and the booster impulse is solved through the coupled mass matrix.
    """
    values = (booster_mass_kg, upper_mass_kg, booster_inertia_kg_m2, upper_inertia_kg_m2)
    if any(not isfinite(value) or value <= 0 for value in values):
        raise ValueError("separation masses and inertias must be positive")
    position = np.asarray(position_m, dtype=float)
    velocity = np.asarray(velocity_m_s, dtype=float)
    if position.shape != (2,) or velocity.shape != (2,):
        raise ValueError("separation position and velocity must be planar vectors")
    if not np.all(np.isfinite(np.r_[position, velocity, theta_rad, angular_rate_rad_s])):
        raise ValueError("separation state must be finite")
    total_mass = booster_mass_kg + upper_mass_kg
    if slosh_state is not None and (
        not 0 <= main_propellant_kg <= booster_mass_kg
        or slosh_station_m is None
        or not np.all(
            np.isfinite(
                [
                    main_propellant_kg,
                    slosh_station_m,
                    slosh_state.displacement_m,
                    slosh_state.velocity_m_s,
                ]
            )
        )
    ):
        raise ValueError("valid main inventory, slosh station and state are required")
    modal_mass = slosh_settings.modal_mass_fraction * main_propellant_kg if slosh_state else 0.0
    eta_b, eta_stack = modal_mass / booster_mass_kg, modal_mass / total_mass
    qd = slosh_state.velocity_m_s if slosh_state else 0.0
    u, tangent, mu, effective_i, coupling, _, _ = _mode_mass_coefficients(
        booster_mass_kg,
        booster_inertia_kg_m2,
        modal_mass,
        (slosh_station_m - booster_cg_station_m) if modal_mass else 0.0,
        slosh_state or SloshState(),
        slosh_settings,
    )
    cg = (booster_mass_kg * booster_cg_station_m + upper_mass_kg * upper_cg_station_m) / total_mass
    axis = np.array([sin(theta_rad), cos(theta_rad)])
    normal = np.array([cos(theta_rad), -sin(theta_rad)])
    displacement_world = normal * u[0] + axis * u[1]
    tangent_world = normal * tangent[0] + axis * tangent[1]
    rb = axis * (booster_cg_station_m - cg) + (eta_b - eta_stack) * displacement_world
    ru = axis * (upper_cg_station_m - cg) - eta_stack * displacement_world
    pb, pu = position + rb, position + ru
    vb_relative = (
        angular_rate_rad_s * np.array([rb[1], -rb[0]]) + tangent_world * (eta_b - eta_stack) * qd
    )
    vu_relative = angular_rate_rad_s * np.array([ru[1], -ru[0]]) - tangent_world * eta_stack * qd
    vb, vu = velocity + vb_relative, velocity + vu_relative
    interface = (
        position
        + axis * (settings.application_station_m - cg)
        + normal * settings.lateral_offset_m
        - eta_stack * displacement_world
    )
    reduced_mass = booster_mass_kg * upper_mass_kg / total_mass
    angle = settings.impulse_misalignment_deg * pi / 180.0
    impulse = reduced_mass * settings.relative_axial_speed_m_s * (axis + normal * np.tan(angle))
    booster_application_arm = (
        axis * (settings.application_station_m - booster_cg_station_m)
        + normal * settings.lateral_offset_m
        - eta_b * displacement_world
    )
    upper_application_arm = (
        axis * (settings.application_station_m - upper_cg_station_m)
        + normal * settings.lateral_offset_m
    )
    jb = _clock_moment(booster_application_arm, -impulse)
    ju = _clock_moment(upper_application_arm, impulse)
    vba, vua = vb - impulse / booster_mass_kg, vu + impulse / upper_mass_kg
    delta_rate, delta_qd = jb / booster_inertia_kg_m2, 0.0
    if modal_mass > 1e-9:
        determinant = effective_i * mu - coupling**2
        if determinant <= 0:
            raise ValueError("slosh separation mass matrix must be positive definite")
        modal_impulse = eta_b * float(impulse @ tangent_world)
        delta_rate = (mu * jb - coupling * modal_impulse) / determinant
        delta_qd = (effective_i * modal_impulse - coupling * jb) / determinant
    wb = angular_rate_rad_s + delta_rate
    qda = qd + delta_qd
    wu = angular_rate_rad_s + ju / upper_inertia_kg_m2
    p_before = booster_mass_kg * vb + upper_mass_kg * vu
    p_after = booster_mass_kg * vba + upper_mass_kg * vua
    h_before = (
        (effective_i + upper_inertia_kg_m2) * angular_rate_rad_s
        + coupling * qd
        + _clock_moment(rb, booster_mass_kg * vb_relative)
        + _clock_moment(ru, upper_mass_kg * vu_relative)
    )
    h_after = (
        effective_i * wb
        + coupling * qda
        + upper_inertia_kg_m2 * wu
        + _clock_moment(rb, booster_mass_kg * (vb_relative - impulse / booster_mass_kg))
        + _clock_moment(ru, upper_mass_kg * (vu_relative + impulse / upper_mass_kg))
    )
    kinetic_before = (
        0.5
        * (
            booster_mass_kg * float(vb_relative @ vb_relative)
            + upper_mass_kg * float(vu_relative @ vu_relative)
            + (effective_i + upper_inertia_kg_m2) * angular_rate_rad_s**2
            + mu * qd**2
        )
        + coupling * angular_rate_rad_s * qd
    )
    vba_relative = vb_relative - impulse / booster_mass_kg
    vua_relative = vu_relative + impulse / upper_mass_kg
    kinetic_after = (
        0.5
        * (
            booster_mass_kg * float(vba_relative @ vba_relative)
            + upper_mass_kg * float(vua_relative @ vua_relative)
            + effective_i * wb**2
            + upper_inertia_kg_m2 * wu**2
            + mu * qda**2
        )
        + coupling * wb * qda
    )
    return dict(
        booster_position_m=pb.tolist(),
        booster_velocity_m_s=vba.tolist(),
        booster_angular_rate_rad_s=float(wb),
        booster_slosh_velocity_m_s=float(qda),
        coupled_slosh_impulse_modelled=slosh_state is not None,
        pre_separation_angular_momentum_kg_m2_s=float(h_before),
        post_separation_angular_momentum_kg_m2_s=float(h_after),
        upper_position_m=pu.tolist(),
        upper_velocity_m_s=vua.tolist(),
        upper_angular_rate_rad_s=float(wu),
        upper_impulse_n_s=impulse.tolist(),
        booster_impulse_n_s=(-impulse).tolist(),
        booster_angular_impulse_nm_s=jb,
        upper_angular_impulse_nm_s=ju,
        impulse_application_position_m=interface.tolist(),
        linear_momentum_residual_kg_m_s=(p_after - p_before).tolist(),
        angular_momentum_residual_kg_m2_s=float(h_after - h_before),
        stored_energy_released_j=float(kinetic_after - kinetic_before),
        booster_pulse_average_angular_acceleration_rad_s2=float(
            delta_rate / settings.effective_duration_s
        ),
        effective_duration_s=settings.effective_duration_s,
        pyroshock_resolved=False,
    )


def design_budget_check(
    settings: FeedSettings = DEFAULT_FEED,
    initial_main_kg: float = 70_200.0,
    airframe_systems_mass_kg: float = 8700.0,
    full_engine_flow_kg_s: float = 1_600_000.0 / (300.0 * 9.80665),
    slosh_settings: SloshSettings = DEFAULT_SLOSH,
):
    """Audit an allocation inside dry mass, not a qualified hardware design.

    The ideal thin spherical vessel lower bound uses sigma=p*r/(2*t), giving
    m_wall=3*p*V*rho_material/(2*sigma_allowable). It omits liners, joints,
    fatigue, thermal margins and real-gas corrections. A larger 450 kg combined
    bottle/regulator allocation is an assumed requirement, not proof of buildability.
    """
    if airframe_systems_mass_kg <= 0 or full_engine_flow_kg_s <= 0:
        raise ValueError("positive systems mass and full engine flow are required")
    state = initial_feed_state(initial_main_kg, settings)
    status = feed_rates(initial_main_kg, 0.0, 9.80665, state, settings)
    composite_density_kg_m3 = 1600.0
    allowable_membrane_stress_pa = 600_000_000.0
    wall_lower_bound = (
        3.0
        * status.bottle_pressure_pa
        * settings.bottle_volume_m3
        * composite_density_kg_m3
        / (2.0 * allowable_membrane_stress_pa)
    )
    allocations = dict(
        helium_inventory_kg=settings.helium_total_mass_kg,
        bottle_and_regulation_hardware_kg=450.0,
        retained_buffer_collector_hardware_kg=150.0,
        tank_baffles_kg=300.0,
        separation_pushers_kg=50.0,
    )
    allocated = sum(allocations.values())
    gas_rt = settings.helium_specific_gas_constant_j_kg_k * settings.gas_temperature_k
    gas_required_at_empty = settings.regulated_pressure_pa * settings.main_tank_volume_m3 / gas_rt
    final_bottle_gas = settings.helium_total_mass_kg - gas_required_at_empty
    final_bottle_pressure = final_bottle_gas * gas_rt / settings.bottle_volume_m3
    endurance = settings.buffer_capacity_kg / full_engine_flow_kg_s
    support_area = slosh_settings.maximum_mode_force_n / (250_000_000.0 / 1.5)
    # Ideal steel tension-member bound only, not plate bending/buckling design.
    support_mass_lower_bound = support_area * 2 * slosh_settings.pendulum_length_m * 7850.0
    checks = dict(
        allocations_inside_existing_dry_mass=allocated <= airframe_systems_mass_kg,
        bottle_allocation_exceeds_ideal_shell_lower_bound=wall_lower_bound <= 450.0,
        helium_covers_full_tank_drain=(
            final_bottle_pressure > settings.regulated_pressure_pa + settings.regulator_head_pa
        ),
        feed_capacity_exceeds_nominal_engine_flow=settings.maximum_flow_kg_s
        >= full_engine_flow_kg_s,
        buffer_covers_three_settling_time_constants=endurance >= 3 * settings.acquisition_time_s,
        modal_support_allocation_exceeds_tension_member_bound=support_mass_lower_bound <= 300.0,
    )
    return dict(
        assumptions_are_measured=False,
        hardware_certified=False,
        allocated_within_existing_airframe_systems_mass_kg=allocations,
        total_systems_allocation_kg=allocated,
        other_airframe_systems_mass_kg=airframe_systems_mass_kg - allocated,
        ideal_spherical_bottle_wall_lower_bound_kg=wall_lower_bound,
        initial_bottle_pressure_pa=status.bottle_pressure_pa,
        final_empty_tank_bottle_pressure_pa=final_bottle_pressure,
        buffer_full_flow_endurance_s=endurance,
        settling_three_time_constants_s=3 * settings.acquisition_time_s,
        modal_force_design_limit_n=slosh_settings.maximum_mode_force_n,
        ideal_modal_support_section_area_m2=support_area,
        ideal_modal_support_tension_member_mass_kg=support_mass_lower_bound,
        checks=checks,
        theoretical_budget_consistent=all(checks.values()),
    )
