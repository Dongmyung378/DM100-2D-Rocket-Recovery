"""Physical laws and contact geometry for the 3 t APR vehicle.

Positions use the instantaneous COM; body stations run upwards from the hull
base, and theta is clockwise from vertical. Parameters are provisional, not
measured hardware or certification limits.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, exp, hypot, pi, sin, sqrt
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

G0 = 9.80665
R_AIR = 287.05287
EARTH_RADIUS_M = 6_371_000.0


@dataclass(frozen=True, slots=True)
class Geometry:
    body_length_m: float = 5.0
    body_radius_m: float = 0.45
    engine_mass_kg: float = 110.0
    engine_station_m: float = 0.45
    airframe_mass_kg: float = 326.0
    airframe_station_m: float = 2.85
    gear_mass_kg: float = 60.0
    tank_bottom_m: float = 1.8
    tank_length_m: float = 2.7
    propellant_density_kg_m3: float = 900.0
    nozzle_station_m: float = -0.15
    nozzle_radius_m: float = 0.18
    rcs_station_m: float = 2.2
    rcs_arm_m: float = 2.0


@dataclass(frozen=True, slots=True)
class Aerodynamics:
    axial_cd: float = 0.95
    normal_cd: float = 0.6
    pressure_center_m: float = 2.5
    rotation_damping: float = 0.3
    transonic_drag_rise: float = 0.2
    max_dynamic_pressure_pa: float = 30_000.0
    max_bending_moment_nm: float = 60_000.0
    max_lateral_force_n: float = 30_000.0


@dataclass(frozen=True, slots=True)
class Propulsion:
    nozzle_exit_area_m2: float = 0.02
    shutdown_time_constant_s: float = 0.04
    ignition_propellant_flow_kg_s: float = 0.04
    unusable_propellant_kg: float = 5.0
    rcs_specific_impulse_s: float = 220.0
    max_rcs_torque_nm: float = 6000.0


@dataclass(frozen=True, slots=True)
class LandingGear:
    half_span_m: float = 1.6
    foot_station_m: float = -0.85
    anchor_station_m: float = 0.6
    deployment_height_m: float = 200.0
    deployment_time_s: float = 2.0
    stiffness_n_m: float = 90_000.0
    damping_n_s_m: float = 6500.0
    max_stroke_m: float = 0.30
    max_normal_force_n: float = 18_000.0
    max_tangential_force_n: float = 9000.0
    tangential_stiffness_n_m: float = 200_000.0
    tangential_damping_n_s_m: float = 1200.0
    static_friction: float = 0.8
    kinetic_friction: float = 0.6
    pad_half_width_m: float = 10.0
    foot_area_m2: float = 0.12
    max_pad_pressure_pa: float = 200_000.0
    tube_outer_diameter_m: float = 0.06
    tube_wall_m: float = 0.004
    strut_length_m: float = 2.0
    bending_arm_m: float = 0.15
    young_modulus_pa: float = 70e9
    yield_stress_pa: float = 250e6
    safety_factor: float = 1.5


@dataclass(frozen=True, slots=True)
class AtmosphereSettings:
    sea_level_temperature_k: float = 288.15
    sea_level_pressure_pa: float = 101_325.0
    latitude_deg: float = 37.0
    coriolis_enabled: bool = True


@dataclass(frozen=True, slots=True)
class LandingAcceptance:
    max_position_error_m: float = 2.0
    max_horizontal_contact_speed_m_s: float = 0.5
    max_vertical_foot_speed_m_s: float = 1.0
    max_contact_tilt_deg: float = 3.0
    max_contact_rate_deg_s: float = 0.5
    fuel_reserve_kg: float = 0.0
    observation_after_contact_s: float = 3.0
    settled_dwell_s: float = 1.0
    settled_speed_m_s: float = 0.05
    settled_rate_deg_s: float = 0.25
    max_post_contact_s: float = 8.0
    max_body_load_g: float = 5.0


class WindField(Protocol):
    def at(self, altitude_m: float) -> float: ...


class PhysicsSettings(Protocol):
    geometry: Geometry
    aerodynamics: Aerodynamics
    propulsion: Propulsion
    gear: LandingGear
    atmosphere: AtmosphereSettings


@dataclass(frozen=True, slots=True)
class MassProperties:
    cg_m: float
    inertia_kg_m2: float
    fuel_height_m: float


def mass_properties(mass_kg: float, deployment: float, settings: PhysicsSettings) -> MassProperties:
    """Settled liquid in one equivalent cylindrical tank, with parallel-axis inertia.

    The liquid is constrained to follow the body; this is not a slosh solver.
    Gear mass moves symmetrically as the massless actuator deploys the struts.
    """
    g, gear = settings.geometry, settings.gear
    # Optional fixed components are (mass, body station, local pitch inertia).
    fixed = getattr(settings, "fixed_mass_components", ())
    dry_mass = g.engine_mass_kg + g.airframe_mass_kg + g.gear_mass_kg
    dry_mass += sum(component[0] for component in fixed)
    fuel = max(0.0, mass_kg - dry_mass)
    fuel_height = fuel / (g.propellant_density_kg_m3 * pi * g.body_radius_m**2)
    fuel_station = g.tank_bottom_m + fuel_height / 2.0
    gear_station = (1.0 - deployment) * 1.3 + deployment * 0.15
    gear_radius = (1.0 - deployment) * g.body_radius_m + deployment * gear.half_span_m / 2.0
    components = (
        (g.engine_mass_kg, g.engine_station_m, 20.0),
        (
            g.airframe_mass_kg,
            g.airframe_station_m,
            g.airframe_mass_kg * (g.body_length_m**2 / 12.0 + g.body_radius_m**2 / 4.0),
        ),
        (g.gear_mass_kg, gear_station, g.gear_mass_kg * gear_radius**2),
        (fuel, fuel_station, fuel * (fuel_height**2 / 12.0 + g.body_radius_m**2 / 4.0)),
    ) + fixed
    total = sum(component[0] for component in components)
    cg = sum(m * station for m, station, _ in components) / total
    inertia = sum(local + m * (station - cg) ** 2 for m, station, local in components)
    return MassProperties(cg, inertia, fuel_height)


def mass_gradients(
    mass_kg: float, deployment: float, settings: PhysicsSettings, properties: MassProperties
) -> tuple[float, float, float, float]:
    """Analytic mass/deployment derivatives for a bottom-filled cylindrical tank."""
    g, gear = settings.geometry, settings.gear
    tank_top = g.tank_bottom_m + properties.fuel_height_m
    cg_mass = (tank_top - properties.cg_m) / mass_kg
    cg_gear = -1.15 * g.gear_mass_kg / mass_kg
    inertia_mass = g.body_radius_m**2 / 4.0 + (tank_top - properties.cg_m) ** 2
    gear_station = 1.3 - 1.15 * deployment
    gear_radius = g.body_radius_m + deployment * (gear.half_span_m / 2.0 - g.body_radius_m)
    inertia_gear = (
        2.0
        * g.gear_mass_kg
        * (
            (gear_station - properties.cg_m) * -1.15
            + gear_radius * (gear.half_span_m / 2.0 - g.body_radius_m)
        )
    )
    return cg_mass, cg_gear, inertia_mass, inertia_gear


def atmosphere(altitude_m: float, settings: AtmosphereSettings) -> tuple[float, float, float]:
    height = max(0.0, altitude_m)
    t0, p0 = settings.sea_level_temperature_k, settings.sea_level_pressure_pa
    temperature = t0 - 0.0065 * min(height, 11_000.0)
    pressure = p0 * (temperature / t0) ** (G0 / (R_AIR * 0.0065))
    if height > 11_000.0:
        pressure *= exp(-G0 * (height - 11_000.0) / (R_AIR * temperature))
    return temperature, pressure, pressure / (R_AIR * temperature)


def gravity(altitude_m: float) -> float:
    """Local effective sea-level gravity, with inverse-square altitude correction."""
    return G0 * (EARTH_RADIUS_M / (EARTH_RADIUS_M + max(0.0, altitude_m))) ** 2


def rotation(theta: float) -> NDArray[np.float64]:
    return np.asarray(((cos(theta), sin(theta)), (-sin(theta), cos(theta))))


def clock_torque(arm: NDArray[np.float64], force: NDArray[np.float64]) -> float:
    return float(arm[1] * force[0] - arm[0] * force[1])


def wind_at(wind: WindField, height: float) -> float:
    """Evaluate the current environment's validated altitude/time wind field."""
    return wind.at(height)


def aerodynamic_loads(
    vector: NDArray[np.float64],
    deployment: float,
    wind: WindField,
    settings: PhysicsSettings,
    properties: MassProperties | None = None,
) -> tuple[NDArray[np.float64], float, float, float]:
    g, a = settings.geometry, settings.aerodynamics
    properties = properties or mass_properties(vector[6], deployment, settings)
    temperature, _, density = atmosphere(vector[1], settings.atmosphere)
    relative_x, relative_z = vector[2] - wind_at(wind, vector[1]), vector[3]
    speed = hypot(relative_x, relative_z)
    cosine, sine = cos(vector[4]), sin(vector[4])
    normal_velocity = cosine * relative_x - sine * relative_z
    axial_velocity = sine * relative_x + cosine * relative_z
    mach = speed / sqrt(1.4 * R_AIR * temperature)
    correction = 1.0 + a.transonic_drag_rise * exp(-(((mach - 1.0) / 0.2) ** 2))
    axial_area = pi * g.body_radius_m**2
    normal_area = 2.0 * g.body_radius_m * g.body_length_m
    # Cylinder cross-flow depends on NORMAL velocity, not total descent speed.
    # Using total speed here would invent a large linear side force at tiny AoA.
    scale = -0.5 * density * correction
    normal_force = scale * a.normal_cd * normal_area * abs(normal_velocity) * normal_velocity
    axial_force = scale * a.axial_cd * axial_area * abs(axial_velocity) * axial_velocity
    force = np.asarray(
        (cosine * normal_force + sine * axial_force, -sine * normal_force + cosine * axial_force)
    )
    torque = (a.pressure_center_m - properties.cg_m) * normal_force
    torque -= (
        0.5
        * density
        * normal_area
        * g.body_length_m**2
        * a.rotation_damping
        * (speed + g.body_length_m * abs(vector[5]))
        * vector[5]
    )
    return force, float(torque), 0.5 * density * speed**2, mach


def engine_performance(
    throttle: float,
    height: float,
    settings: PhysicsSettings,
    thrust_scale: float = 1.0,
    *,
    sea_level_thrust_n: float = 42_000.0,
    sea_level_isp_s: float = 260.0,
) -> tuple[float, float]:
    """Versioned sea-level design values; pressure thrust is added exactly once."""
    _, pressure, _ = atmosphere(height, settings.atmosphere)
    thrust = (
        throttle
        * (sea_level_thrust_n + settings.propulsion.nozzle_exit_area_m2 * (101_325.0 - pressure))
        * thrust_scale
    )
    flow = throttle * sea_level_thrust_n * thrust_scale / (sea_level_isp_s * G0)
    return thrust, flow


def foot_kinematics(
    vector: NDArray[np.float64],
    deployment: float,
    settings: PhysicsSettings,
    *,
    mass_rate: float = 0.0,
    deployment_rate: float = 0.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    g, gear = settings.geometry, settings.gear
    properties = mass_properties(vector[6], deployment, settings)
    span = g.body_radius_m + deployment * (gear.half_span_m - g.body_radius_m)
    station = gear.anchor_station_m + deployment * (gear.foot_station_m - gear.anchor_station_m)
    offset = offset_rate = 0.0
    if hasattr(settings, "lateral_body_offset"):
        offset, offset_rate = settings.lateral_body_offset(vector, mass_rate)
    body_arms = np.asarray(
        ((-span + offset, station - properties.cg_m), (span + offset, station - properties.cg_m))
    )
    arms = body_arms @ rotation(vector[4]).T
    points = vector[:2] + arms
    velocities = vector[2:4] + vector[5] * np.column_stack((arms[:, 1], -arms[:, 0]))
    cg_mass, cg_gear, _, _ = mass_gradients(vector[6], deployment, settings, properties)
    cg_rate = cg_mass * mass_rate + cg_gear * deployment_rate
    relative_rates = np.asarray(
        (
            (
                -(gear.half_span_m - g.body_radius_m) * deployment_rate + offset_rate,
                (gear.foot_station_m - gear.anchor_station_m) * deployment_rate - cg_rate,
            ),
            (
                (gear.half_span_m - g.body_radius_m) * deployment_rate + offset_rate,
                (gear.foot_station_m - gear.anchor_station_m) * deployment_rate - cg_rate,
            ),
        )
    )
    velocities += relative_rates @ rotation(vector[4]).T
    return points, velocities


def gear_contacts(
    vector: NDArray[np.float64],
    settings: PhysicsSettings,
    *,
    mass_rate: float = 0.0,
    deployment_rate: float = 0.0,
) -> dict[str, Any]:
    """Unilateral vertical compliance and elastic stick / Coulomb slip per foot.

    Penetration of the UNCOMPRESSED foot is equivalent gear stroke, not ground
    penetration of the actual foot. The physical pad remains at ground level.
    """
    deployment = float(vector[7])
    points, velocities = foot_kinematics(
        vector, deployment, settings, mass_rate=mass_rate, deployment_rate=deployment_rate
    )
    gear = settings.gear
    stroke = np.maximum(0.0, -points[:, 1])
    active = points[:, 1] <= 0.0
    normal = np.where(
        active,
        np.maximum(0.0, gear.stiffness_n_m * stroke - gear.damping_n_s_m * velocities[:, 1]),
        0.0,
    )
    trial = (
        -gear.tangential_stiffness_n_m * vector[8:10]
        - gear.tangential_damping_n_s_m * velocities[:, 0]
    )
    slipping = np.abs(trial) > gear.static_friction * normal
    direction = np.where(
        np.abs(velocities[:, 0]) > 1e-9, np.sign(velocities[:, 0]), -np.sign(trial)
    )
    tangent = np.where(slipping, -gear.kinetic_friction * normal * direction, trial)
    tangent = np.where(active & (normal > 0.0), tangent, 0.0)
    force = np.column_stack((tangent, normal))
    arms = points - vector[:2]
    torque = float(np.sum(arms[:, 1] * tangent - arms[:, 0] * normal))
    return dict(
        points=points,
        velocities=velocities,
        stroke=stroke,
        active=active,
        normal=normal,
        tangent=tangent,
        forces=force,
        torque=torque,
        slipping=slipping,
    )


class RecoverySensorEstimator:
    """Causal linear Kalman estimate of x/vx and theta/omega on the control clock.

    Prediction uses only previous estimates, actuator telemetry and FORECAST
    wind. The actual plant state and actual wind are never estimator inputs.
    """

    def __init__(self, noise_scale: float) -> None:
        self.noise_scale = noise_scale
        deviations = noise_scale * np.asarray((0.5, 0.1, np.deg2rad(0.25), np.deg2rad(0.25)))
        self.measurement_covariance = np.diag(np.maximum(deviations**2, 1e-12))
        self.covariance = self.measurement_covariance.copy()
        self.estimate: NDArray[np.float64] | None = None
        self.time_s: float | None = None

    def observe(
        self,
        raw: NDArray[np.float64],
        time_s: float,
        acceleration: tuple[float, float],
        delay_s: float,
    ) -> NDArray[np.float64]:
        measured = raw[[0, 2, 4, 5]]
        if self.estimate is None or self.noise_scale == 0.0:
            self.estimate = measured.copy()
        else:
            dt = time_s - self.time_s
            transition = np.eye(4)
            transition[0, 1] = transition[2, 3] = dt
            input_matrix = np.asarray(((dt**2 / 2, 0), (dt, 0), (0, dt**2 / 2), (0, dt)))
            predicted = transition @ self.estimate + input_matrix @ acceleration
            process = input_matrix @ np.diag((0.8**2, 0.5**2)) @ input_matrix.T
            covariance = transition @ self.covariance @ transition.T + process
            gain = np.linalg.solve(covariance + self.measurement_covariance, covariance.T).T
            innovation = measured - predicted
            innovation[2] = atan2(sin(innovation[2]), cos(innovation[2]))
            self.estimate = predicted + gain @ innovation
            residual = np.eye(4) - gain
            self.covariance = (
                residual @ covariance @ residual.T + gain @ self.measurement_covariance @ gain.T
            )
        self.time_s = time_s
        result = raw.copy()
        result[[0, 2, 4, 5]] = self.estimate
        result[0] += delay_s * result[2]
        result[4] += delay_s * result[5]
        return result
