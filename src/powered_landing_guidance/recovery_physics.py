"""Physical laws and contact geometry for the 100 t APR vehicle.

Positions use the instantaneous COM; body stations run upwards from the hull
base, and theta is clockwise from vertical. Parameters are provisional, not
measured hardware or certification limits.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, exp, hypot, log, pi, sin, sqrt
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

G0 = 9.80665
R_AIR = 287.05287
EARTH_RADIUS_M = 6_371_000.0


@dataclass(frozen=True, slots=True)
class Geometry:
    body_length_m: float = 20.0
    body_radius_m: float = 1.5
    engine_mass_kg: float = 1200.0
    engine_station_m: float = 0.8
    airframe_mass_kg: float = 8700.0
    airframe_station_m: float = 10.0
    gear_mass_kg: float = 1500.0
    gear_retracted_station_m: float = 2.0
    gear_deployed_station_m: float = 0.3
    tank_bottom_m: float = 3.0
    tank_length_m: float = 13.0
    propellant_density_kg_m3: float = 826.0
    nozzle_station_m: float = -0.2
    nozzle_radius_m: float = 0.71
    rcs_station_m: float = 18.5
    rcs_arm_m: float = 9.0


@dataclass(frozen=True, slots=True)
class Aerodynamics:
    axial_cd: float = 0.4
    normal_cd: float = 0.8
    pressure_center_m: float = 10.0
    rotation_damping: float = 0.3
    transonic_drag_rise: float = 0.2


@dataclass(frozen=True, slots=True)
class Propulsion:
    # Vacuum thrust 1.76 MN at the unchanged sea-level mass flow (Isp 330 s).
    nozzle_exit_area_m2: float = 160_000.0 / 101_325.0


@dataclass(frozen=True, slots=True)
class LandingGear:
    """Four conceptual legs represented by two planar foot locations.

    The 2D acceptance assumes adequate leg capacity within the 1.5 m/s
    vertical touchdown limit and stops at first contact.
    """

    half_span_m: float = 4.0
    foot_station_m: float = -1.0
    anchor_station_m: float = 1.2
    deployment_time_s: float = 2.0
    pad_half_width_m: float = 10.0

    def __post_init__(self) -> None:
        for name in (
            "half_span_m",
            "deployment_time_s",
            "pad_half_width_m",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"landing-gear {name} must be finite and positive")
        if not np.isfinite(self.foot_station_m) or not np.isfinite(self.anchor_station_m):
            raise ValueError("landing-gear attachment stations must be finite")


@dataclass(frozen=True, slots=True)
class AtmosphereSettings:
    sea_level_temperature_k: float = 288.15
    sea_level_pressure_pa: float = 101_325.0
    latitude_deg: float = 37.0
    coriolis_enabled: bool = True
    temperature_profile: tuple[tuple[float, float], ...] = ()
    pressure_profile: tuple[tuple[float, float], ...] = ()
    density_scale: float = 1.0


@dataclass(frozen=True, slots=True)
class LandingAcceptance:
    max_position_error_m: float = 2.0
    max_horizontal_contact_speed_m_s: float = 0.5
    max_vertical_foot_speed_m_s: float = 1.5
    max_contact_tilt_deg: float = 3.0
    max_contact_rate_deg_s: float = 0.5


class WindField(Protocol):
    def at(self, altitude_m: float) -> float: ...

    def vertical_at(self, altitude_m: float) -> float: ...


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

    Main and RCS inventories occupy one equivalent liquid column in this lumped
    inertia model; their real tank allocation and slosh are not resolved.
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
    gear_station = (
        1.0 - deployment
    ) * g.gear_retracted_station_m + deployment * g.gear_deployed_station_m
    gear_radius = (1.0 - deployment) * g.body_radius_m + deployment * gear.half_span_m / 2.0
    components = (
        # Equivalent engine cylinder, length twice its body station and radius
        # equal to the nozzle radius; keeps local inertia dimensionally scaled.
        (
            g.engine_mass_kg,
            g.engine_station_m,
            g.engine_mass_kg
            * ((2.0 * g.engine_station_m) ** 2 / 12.0 + g.nozzle_radius_m**2 / 4.0),
        ),
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
    station_gradient = g.gear_deployed_station_m - g.gear_retracted_station_m
    cg_gear = station_gradient * g.gear_mass_kg / mass_kg
    inertia_mass = g.body_radius_m**2 / 4.0 + (tank_top - properties.cg_m) ** 2
    gear_station = g.gear_retracted_station_m + station_gradient * deployment
    gear_radius = g.body_radius_m + deployment * (gear.half_span_m / 2.0 - g.body_radius_m)
    inertia_gear = (
        2.0
        * g.gear_mass_kg
        * (
            (gear_station - properties.cg_m) * station_gradient
            + gear_radius * (gear.half_span_m / 2.0 - g.body_radius_m)
        )
    )
    return cg_mass, cg_gear, inertia_mass, inertia_gear


def standard_temperature_profile(
    sea_level_temperature_k: float = 288.15,
) -> tuple[tuple[float, float], ...]:
    """ISA lapse-rate layers, applied to local altitude as an engineering model.

    Layer values follow NASA/TM-2005-213659. Geopotential conversion, molecular
    composition changes and thermospheric heating are not resolved. The last
    layer is held isothermal above 84.852 km; this is not a 700 km atmosphere.
    """
    offset = sea_level_temperature_k - 288.15
    return tuple(
        (height, temperature + offset)
        for height, temperature in (
            (0.0, 288.15),
            (11_000.0, 216.65),
            (20_000.0, 216.65),
            (32_000.0, 228.65),
            (47_000.0, 270.65),
            (51_000.0, 270.65),
            (71_000.0, 214.65),
            (84_852.0, 186.946),
            (100_000.0, 186.946),
        )
    )


def _extended_temperature_profile(settings: AtmosphereSettings):
    standard = standard_temperature_profile(settings.sea_level_temperature_k)
    if not settings.temperature_profile:
        return standard
    supplied = settings.temperature_profile
    top_height, top_temperature = supplied[-1]
    offset = top_temperature - _profile_value(standard, top_height)
    return supplied + tuple(
        (height, temperature + offset) for height, temperature in standard if height > top_height
    )


def atmosphere(altitude_m: float, settings: AtmosphereSettings) -> tuple[float, float, float]:
    height = max(0.0, altitude_m)
    p0 = settings.sea_level_pressure_pa
    temperatures = _extended_temperature_profile(settings)
    temperature = _profile_value(temperatures, height)
    pressure = _hydrostatic_pressure(height, temperatures, p0)
    if settings.pressure_profile:
        # Preserve the hydrostatic shape between observations. Interpolate only
        # each observation's multiplicative correction to that reference.
        log_corrections = tuple(
            (z, log(observed / _hydrostatic_pressure(z, temperatures, p0)))
            for z, observed in settings.pressure_profile
        )
        pressure *= exp(_profile_value(log_corrections, height))
    return temperature, pressure, settings.density_scale * pressure / (R_AIR * temperature)


def _profile_value(profile, height: float) -> float:
    if height <= profile[0][0]:
        return profile[0][1]
    if height >= profile[-1][0]:
        return profile[-1][1]
    for (z0, v0), (z1, v1) in zip(profile, profile[1:], strict=False):
        if height <= z1:
            fraction = (height - z0) / (z1 - z0)
            return v0 + fraction * (v1 - v0)
    raise AssertionError("profile must cover the queried altitude")


def _hydrostatic_pressure(height, temperatures, sea_level_pressure):
    pressure = sea_level_pressure
    for (z0, t_start), (z1, t_end) in zip(temperatures, temperatures[1:], strict=False):
        if height <= z0:
            break
        end = min(height, z1)
        slope = (t_end - t_start) / (z1 - z0)
        pressure *= exp(-G0 / R_AIR * _inverse_temperature_integral(t_start, slope, end - z0))
        if height <= z1:
            break
    if height > temperatures[-1][0]:
        pressure *= exp(-G0 * (height - temperatures[-1][0]) / (R_AIR * temperatures[-1][1]))
    return pressure


def _inverse_temperature_integral(start_k: float, slope_k_m: float, distance_m: float) -> float:
    if abs(slope_k_m) < 1e-12:
        return distance_m / start_k
    return log((start_k + slope_k_m * distance_m) / start_k) / slope_k_m


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
    relative_x = vector[2] - wind_at(wind, vector[1])
    relative_z = vector[3] - wind.vertical_at(vector[1])
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
    sea_level_thrust_n: float = 1_600_000.0,
    sea_level_isp_s: float = 300.0,
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
    offset = np.zeros(2)
    offset_rate = np.zeros(2)
    if hasattr(settings, "body_frame_offset"):
        offset, offset_rate = settings.body_frame_offset(vector, mass_rate)
    body_arms = np.asarray(
        (
            (-span + offset[0], station - properties.cg_m + offset[1]),
            (span + offset[0], station - properties.cg_m + offset[1]),
        )
    )
    arms = body_arms @ rotation(vector[4]).T
    points = vector[:2] + arms
    velocities = vector[2:4] + vector[5] * np.column_stack((arms[:, 1], -arms[:, 0]))
    cg_mass, cg_gear, _, _ = mass_gradients(vector[6], deployment, settings, properties)
    cg_rate = cg_mass * mass_rate + cg_gear * deployment_rate
    relative_rates = np.asarray(
        (
            (
                -(gear.half_span_m - g.body_radius_m) * deployment_rate + offset_rate[0],
                (gear.foot_station_m - gear.anchor_station_m) * deployment_rate
                - cg_rate
                + offset_rate[1],
            ),
            (
                (gear.half_span_m - g.body_radius_m) * deployment_rate + offset_rate[0],
                (gear.foot_station_m - gear.anchor_station_m) * deployment_rate
                - cg_rate
                + offset_rate[1],
            ),
        )
    )
    velocities += relative_rates @ rotation(vector[4]).T
    return points, velocities


class PlanarNavigationEstimator:
    """Causal position/velocity estimate with a slowly varying acceleration bias.

    Each planar axis uses [position, velocity, acceleration bias]. The axes
    share a covariance because their measurement radii and dynamics match.
    Uniform-disk measurement errors have component variance radius**2 / 4;
    a Kalman update uses these second moments without claiming Gaussian noise.

    ``acceleration_xy`` is a nominal inertial acceleration, held over the
    prediction interval. The caller must obtain it from previous estimates,
    known actuator commands and forecast conditions, not hidden plant state.
    ``time_s - delay_s`` is the measurement timestamp. Only the returned copy
    is extrapolated to current time; the posterior remains at measurement time.
    Attitude, actuator telemetry and all other raw components are untouched.

    Process noise is a conceptual tuning assumption, not measured IMU data.
    Bounded raw sensor errors do not imply a hard bound on filtered errors.
    """

    def __init__(self, position_error_m: float, velocity_error_m_s: float) -> None:
        for value in (position_error_m, velocity_error_m_s):
            if not np.isfinite(value) or value < 0:
                raise ValueError("navigation measurement radii must be finite and nonnegative")
        self.position_error_m = float(position_error_m)
        self.velocity_error_m_s = float(velocity_error_m_s)
        self.measurement_covariance = np.diag(
            [(position_error_m / 2.0) ** 2, (velocity_error_m_s / 2.0) ** 2]
        )
        self.covariance = np.diag(
            [self.measurement_covariance[0, 0], self.measurement_covariance[1, 1], 2.0**2]
        )
        self.current_covariance = self.covariance.copy()
        self.estimate: NDArray[np.float64] | None = None
        self.measurement_time_s: float | None = None
        self.time_s: float | None = None
        # White acceleration intensity [m²/s³] and bias random walk [m²/s⁵].
        self.acceleration_noise_intensity = 0.3**2
        self.bias_walk_intensity = 0.5**2

    def _prediction_matrices(
        self, dt: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        transition = np.array(((1.0, dt, dt**2 / 2), (0.0, 1.0, dt), (0.0, 0.0, 1.0)))
        acceleration_input = np.array((dt**2 / 2, dt, 0.0))
        process = self.acceleration_noise_intensity * np.array(
            ((dt**3 / 3, dt**2 / 2, 0), (dt**2 / 2, dt, 0), (0, 0, 0))
        ) + self.bias_walk_intensity * np.array(
            (
                (dt**5 / 20, dt**4 / 8, dt**3 / 6),
                (dt**4 / 8, dt**3 / 3, dt**2 / 2),
                (dt**3 / 6, dt**2 / 2, dt),
            )
        )
        return transition, acceleration_input, process

    def observe(
        self,
        raw: NDArray[np.float64],
        time_s: float,
        acceleration_xy: tuple[float, float] | NDArray[np.float64],
        delay_s: float = 0.0,
    ) -> NDArray[np.float64]:
        """Assimilate a new measurement once, then extrapolate to current time.

        Repeated/older measurement timestamps are ignored, which also handles
        startup delay queues. The supplied delay must be the actual sample age,
        not the configured maximum age while that queue is still filling.
        """
        result = np.asarray(raw, dtype=float).copy()
        acceleration = np.asarray(acceleration_xy, dtype=float)
        if result.ndim != 1 or len(result) < 6 or not np.all(np.isfinite(result)):
            raise ValueError("navigation input must be a finite one-dimensional state")
        if acceleration.shape != (2,) or not np.all(np.isfinite(acceleration)):
            raise ValueError("navigation acceleration must contain two finite components")
        if not np.isfinite(time_s) or not np.isfinite(delay_s) or delay_s < 0.0:
            raise ValueError("navigation time and nonnegative delay must be finite")
        if self.time_s is not None and time_s < self.time_s - 1e-12:
            raise ValueError("navigation control time cannot move backwards")
        measured = result[[[0, 2], [1, 3]]]
        measured_time = float(time_s - delay_s)
        if self.estimate is None:
            self.estimate = np.column_stack((measured, np.zeros(2)))
            self.measurement_time_s = measured_time
        elif measured_time > self.measurement_time_s + 1e-12:
            dt = measured_time - self.measurement_time_s
            transition, acceleration_input, process = self._prediction_matrices(dt)
            predicted = self.estimate @ transition.T + np.outer(acceleration, acceleration_input)
            covariance = transition @ self.covariance @ transition.T + process
            innovation_covariance = covariance[:2, :2] + self.measurement_covariance
            gain = np.linalg.solve(innovation_covariance, covariance[:, :2].T).T
            self.estimate = predicted + (measured - predicted[:, :2]) @ gain.T
            residual = np.eye(3)
            residual[:, :2] -= gain
            # Joseph form preserves positive semidefiniteness through rounding.
            self.covariance = (
                residual @ covariance @ residual.T + gain @ self.measurement_covariance @ gain.T
            )
            self.covariance = (self.covariance + self.covariance.T) / 2
            # An exact channel needs no numerical approximation to its reading.
            if self.position_error_m == 0:
                self.estimate[:, 0] = measured[:, 0]
            if self.velocity_error_m_s == 0:
                self.estimate[:, 1] = measured[:, 1]
            self.measurement_time_s = measured_time
        age = max(0.0, float(time_s - self.measurement_time_s))
        transition, acceleration_input, process = self._prediction_matrices(age)
        current = self.estimate @ transition.T + np.outer(acceleration, acceleration_input)
        self.current_covariance = transition @ self.covariance @ transition.T + process
        result[:2] = current[:, 0]
        result[2:4] = current[:, 1]
        self.time_s = float(time_s)
        return result
