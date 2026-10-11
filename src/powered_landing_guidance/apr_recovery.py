"""100 t APR ascent, separation and rigid-body recovery.

The ideal engine admits continuous 0-100% throttle. Main and RCS inventories
are separate. Feed, closed pressurization, reduced slosh and staging impulses
are coupled concept models, not calibrated hardware or a certification model.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, fields, replace
from functools import lru_cache
from math import atan2, cos, exp, hypot, pi, sin, sqrt
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import brentq

from powered_landing_guidance.integrators import rk4_step
from powered_landing_guidance.observation_models import (
    ControllerOptions,
    PlanarAttitudeEstimator,
    SensorSettings,
)
from powered_landing_guidance.recovery_physics import (
    G0,
    R_AIR,
    Aerodynamics,
    AtmosphereSettings,
    Geometry,
    LandingAcceptance,
    LandingGear,
    PlanarNavigationEstimator,
    Propulsion,
    aerodynamic_loads,
    atmosphere,
    clock_torque,
    engine_performance,
    foot_kinematics,
    gravity,
    mass_gradients,
    mass_properties,
    rotation,
    standard_temperature_profile,
)
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

RETURN_METHOD = "Aero-PID Return (APR)"
ASSUMED_UPPER_STAGE_ORBIT_ALTITUDE_M = 700_000.0


@dataclass(frozen=True, slots=True)
class RecoveryDisturbances:
    """Small healthy-model uncertainties, not user-selectable equipment faults."""

    thrust_scale: float = 1.0
    drag_coefficient_scale: float = 1.0
    initial_mass_scale: float = 1.0
    inertia_scale: float = 1.0
    sensor_position_error_m: float = 0.0
    sensor_velocity_error_m_s: float = 0.0
    observation_delay_steps: int = 0
    action_delay_steps: int = 0
    engine_lag_s: float = 0.0

    def __post_init__(self) -> None:
        for key, value, low, high in (
            ("thrust_scale", self.thrust_scale, 0.97, 1.03),
            ("drag_coefficient_scale", self.drag_coefficient_scale, 0.85, 1.15),
            ("initial_mass_scale", self.initial_mass_scale, 0.99, 1.01),
            ("inertia_scale", self.inertia_scale, 0.97, 1.03),
            ("sensor_position_error_m", self.sensor_position_error_m, 0.0, 2.0),
            ("sensor_velocity_error_m_s", self.sensor_velocity_error_m_s, 0.0, 1.0),
            ("engine_lag_s", self.engine_lag_s, 0.0, 0.05),
        ):
            if isinstance(value, bool) or not np.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{key} is outside the healthy-model uncertainty range")
        for key, value, maximum in (
            ("observation_delay_steps", self.observation_delay_steps, 2),
            ("action_delay_steps", self.action_delay_steps, 1),
        ):
            if type(value) is not int or not 0 <= value <= maximum:
                raise ValueError(f"{key} is outside the operating delay range")


@dataclass(frozen=True)
class FinSettings:
    """Assumed equivalent control surfaces, not a measured Falcon 9 grid-fin map."""

    surface_mass_kg: float = 360.0
    actuator_mass_kg: float = 160.0
    deployment_mass_kg: float = 60.0
    electrical_mass_kg: float = 20.0
    station_m: float = 18.6
    mean_mass_radius_m: float = 1.5
    installed_area_m2: float = 6.0
    planar_effective_area_m2: float = 3.0
    lift_slope_rad_inv: float = 2.4
    max_lift_coefficient: float = 0.8
    base_drag_coefficient: float = 0.08
    induced_drag_factor: float = 0.6
    deflection_limit_deg: float = 20.0
    deflection_rate_deg_s: float = 60.0
    actuator_time_constant_s: float = 0.08
    deployment_time_s: float = 2.0
    retraction_time_s: float = 2.0
    control_min_q_pa: float = 100.0
    control_max_alpha_deg: float = 25.0
    model_max_mach: float = 8.0
    hinge_arm_m: float = 0.12
    per_fin_hinge_limit_nm: float = 5000.0

    @property
    def mass_kg(self):
        return (
            self.surface_mass_kg
            + self.actuator_mass_kg
            + self.deployment_mass_kg
            + self.electrical_mass_kg
        )


FINS = FinSettings()


@dataclass(frozen=True)
class Design:
    dry_mass_kg: float = 12_000.0
    main_propellant_kg: float = 70_200.0
    rcs_propellant_kg: float = 300.0
    upper_mass_kg: float = 17_500.0
    sea_level_thrust_n: float = 1_600_000.0
    sea_level_isp_s: float = 300.0
    nozzle_exit_area_m2: float = 160_000.0 / 101_325.0
    gimbal_limit_deg: float = 6.0
    rcs_isp_s: float = 220.0
    rcs_torque_limit_nm: float = 100_000.0
    high_q_threshold_pa: float = 10_000.0
    descent_reference_m_s: float = 1.0
    payload_mass_kg: float = 500.0  # Included in upper_mass_kg, not added to it.

    def __post_init__(self):
        if (
            not np.isfinite(self.upper_mass_kg)
            or not np.isfinite(self.payload_mass_kg)
            or not 0.0 <= self.payload_mass_kg < self.upper_mass_kg
        ):
            raise ValueError("payload_mass_kg must be finite and within [0, upper_mass_kg)")

    @property
    def upper_stage_wet_mass_kg(self):
        return self.upper_mass_kg - self.payload_mass_kg

    @property
    def launch_mass_kg(self):
        return (
            self.dry_mass_kg + self.main_propellant_kg + self.rcs_propellant_kg + self.upper_mass_kg
        )


@dataclass(frozen=True)
class AprPhysics:
    # Provisional lumped mass allocation, not a selected component bill of materials.
    geometry: Geometry = Geometry()
    aerodynamics: Aerodynamics = Aerodynamics()
    atmosphere: AtmosphereSettings = AtmosphereSettings()
    gear: LandingGear = LandingGear()
    propulsion: Propulsion = Propulsion()
    acceptance: LandingAcceptance = LandingAcceptance(
        max_position_error_m=2.0,
        max_horizontal_contact_speed_m_s=0.5,
        max_vertical_foot_speed_m_s=1.5,
        max_contact_tilt_deg=3.0,
        max_contact_rate_deg_s=0.5,
    )
    upper_station_m: float = 24.0
    upper_length_m: float = 8.0
    slosh_mass_fraction: float = 0.03
    slosh_settings: SloshSettings = SloshSettings()
    fixed_mass_components: tuple[tuple[float, float, float], ...] = (
        (FINS.mass_kg, FINS.station_m, FINS.mass_kg * FINS.mean_mass_radius_m**2),
    )

    def body_frame_offset(self, vector, mass_rate=0.0):
        if len(vector) < 10:
            return np.zeros(2), np.zeros(2)
        main, q, qd = vector[7:10]
        mass = vector[6]
        ratio = self.slosh_mass_fraction * max(main, 0.0) / mass
        # Full mass rate approximates modal outflow in this single-liquid model.
        ratio_rate = self.slosh_mass_fraction * mass_rate * (mass - main) / mass**2
        displacement, direction = slosh_displacement(SloshState(q, qd), self.slosh_settings)
        return -ratio * displacement, -ratio * direction * qd - ratio_rate * displacement


PHYSICS = AprPhysics()


PROFILE_TOP_M = 100_000.0
MAX_PAD_WIND_M_S = 8.0
MAX_ALOFT_WIND_M_S = 20.0
MAX_GUST_M_S = 5.0
MAX_VERTICAL_WIND_M_S = 3.0
MAX_DENSITY_SCALE_DEVIATION = 0.1
MAX_WIND_SHEAR_M_S_PER_KM = 4.0
MAX_VERTICAL_WIND_SHEAR_M_S_PER_KM = 3.0
MAX_FORECAST_ERROR_M_S = 2.0
OPERATING_SCOPE = "apr_100t_operating_v1"
UNCONNECTED_SYSTEMS = (
    "elastic_structure",
    "upper_stage_ignition_and_plume",
    "detailed_heating_and_ground_damage",
    "leg_material_strength_and_pad_bearing",
)

SYSTEMS_MODEL_VERSION = "apr_100t_coupled_systems_v2"
STATE_NAMES = (
    "x_m",
    "z_m",
    "vx_m_s",
    "vz_m_s",
    "theta_rad",
    "omega_rad_s",
    "main_propellant_kg",
    "rcs_propellant_kg",
    "gear_extension",
    "fin_deflection_rad",
    "fin_extension",
    "gimbal_rad",
    "engine_throttle",
    "feed_buffer_kg",
    "liquid_acquisition",
    "ullage_helium_kg",
    "bottle_helium_kg",
    "slosh_displacement_m",
    "slosh_velocity_m_s",
)


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, int | float) or not np.isfinite(value):
        raise ValueError(f"{name}: 유한한 숫자가 필요합니다.")
    return float(value)


def _wind_profile(value, name):
    if not isinstance(value, list | tuple) or not 2 <= len(value) <= 100:
        raise ValueError(f"{name}: 고도·바람 쌍을 2~100개 입력해야 합니다.")
    levels = []
    for pair in value:
        if not isinstance(pair, list | tuple) or len(pair) != 2:
            raise ValueError(f"{name}: [고도_m, 동쪽바람_m_s] 쌍이 필요합니다.")
        levels.append((_finite(pair[0], name), _finite(pair[1], name)))
    if levels[0][0] != 0.0 or levels[-1][0] != PROFILE_TOP_M:
        raise ValueError(f"{name}: 0~100000 m 전체 고도 구간이 필요합니다.")
    for left, right in zip(levels, levels[1:], strict=False):
        if right[0] <= left[0]:
            raise ValueError(f"{name}: 고도는 중복 없이 증가해야 합니다.")
        if (
            1000.0 * abs(right[1] - left[1]) / (right[0] - left[0])
            > MAX_WIND_SHEAR_M_S_PER_KM + 1e-9
        ):
            raise ValueError(f"{name}: 잠정 윈드시어 한계 4 m/s/km를 초과했습니다.")
    if any(abs(w) > MAX_ALOFT_WIND_M_S for _, w in levels):
        raise ValueError(f"{name}: 상층 기본풍 한계 ±20 m/s를 초과했습니다.")
    pad = [w for z, w in levels if z <= 50.0]
    pad.append(float(np.interp(50.0, [p[0] for p in levels], [p[1] for p in levels])))
    if max(abs(w) for w in pad) > MAX_PAD_WIND_M_S:
        raise ValueError(f"{name}: 지상 지속풍 입력 상한 8 m/s를 초과했습니다.")
    return tuple(levels)


def _vertical_wind_profile(value):
    if not isinstance(value, list | tuple) or not 2 <= len(value) <= 100:
        raise ValueError("vertical_wind_profile: 고도·수직풍 쌍을 2~100개 입력해야 합니다.")
    levels = []
    for pair in value:
        if not isinstance(pair, list | tuple) or len(pair) != 2:
            raise ValueError("vertical_wind_profile: [고도_m, 상승풍_m_s] 쌍이 필요합니다.")
        levels.append(
            (_finite(pair[0], "vertical_wind_profile"), _finite(pair[1], "vertical_wind_profile"))
        )
    if levels[0][0] != 0.0 or levels[-1][0] != PROFILE_TOP_M:
        raise ValueError("vertical_wind_profile: 0~100000 m 전체 고도 구간이 필요합니다.")
    for left, right in zip(levels, levels[1:], strict=False):
        if right[0] <= left[0]:
            raise ValueError("vertical_wind_profile: 고도는 중복 없이 증가해야 합니다.")
        if (
            1000.0 * abs(right[1] - left[1]) / (right[0] - left[0])
            > MAX_VERTICAL_WIND_SHEAR_M_S_PER_KM + 1e-9
        ):
            raise ValueError("vertical_wind_profile: 수직풍 기울기 3 m/s/km를 초과했습니다.")
    if any(abs(w) > MAX_VERTICAL_WIND_M_S for _, w in levels):
        raise ValueError("vertical_wind_profile: 수직풍 ±3 m/s를 초과했습니다.")
    return tuple(levels)


def _atmosphere_profile(value, name, sea_level_value):
    if not isinstance(value, list | tuple):
        raise ValueError(f"{name}: 고도·값 쌍이 필요합니다.")
    if not value:
        return ()
    if not 2 <= len(value) <= 100:
        raise ValueError(f"{name}: 고도·값 쌍을 2~100개 입력해야 합니다.")
    levels = []
    for pair in value:
        if not isinstance(pair, list | tuple) or len(pair) != 2:
            raise ValueError(f"{name}: [고도_m, 값] 쌍이 필요합니다.")
        levels.append((_finite(pair[0], name), _finite(pair[1], name)))
    if levels[0][0] != 0.0 or levels[-1][0] != PROFILE_TOP_M:
        raise ValueError(f"{name}: 0~100000 m 전체 고도 구간이 필요합니다.")
    if not np.isclose(levels[0][1], sea_level_value, rtol=0.0, atol=1e-6):
        raise ValueError(f"{name}: 지상값은 sea_level 입력과 일치해야 합니다.")
    if any(right[0] <= left[0] for left, right in zip(levels, levels[1:], strict=False)):
        raise ValueError(f"{name}: 고도는 중복 없이 증가해야 합니다.")
    return tuple(levels)


def _smoothstep(unit):
    """C1 taper for one deterministic, compact gust event."""
    unit = min(max(unit, 0.0), 1.0)
    return unit * unit * (3.0 - 2.0 * unit)


def _density_ratio_extrema(settings: AtmosphereSettings) -> tuple[float, float]:
    """Find final density extrema relative to the fixed nominal atmosphere.

    Both temperatures are linear on each knot interval and the pressure-profile
    correction is log-linear. The derivative of log(density ratio) therefore has
    a quadratic numerator, so checking knots and its real roots covers the
    entire supported 0-100 km domain, including extrema between supplied knots.
    """
    nominal = AtmosphereSettings()
    hydrostatic = replace(settings, pressure_profile=(), density_scale=1.0)
    knots = sorted(
        {z for z, _ in standard_temperature_profile()}
        | {0.0, PROFILE_TOP_M}
        | {z for z, _ in settings.temperature_profile}
        | {z for z, _ in settings.pressure_profile}
    )

    def ratio(height):
        return atmosphere(height, settings)[2] / atmosphere(height, nominal)[2]

    ratios = [ratio(z) for z in knots]
    for z0, z1 in zip(knots, knots[1:], strict=False):
        length = z1 - z0
        t0 = atmosphere(z0, settings)[0]
        tn0 = atmosphere(z0, nominal)[0]
        t_slope = (atmosphere(z1, settings)[0] - t0) / length
        tn_slope = (atmosphere(z1, nominal)[0] - tn0) / length
        correction0 = atmosphere(z0, settings)[1] / atmosphere(z0, hydrostatic)[1]
        correction1 = atmosphere(z1, settings)[1] / atmosphere(z1, hydrostatic)[1]
        log_correction_slope = np.log(correction1 / correction0) / length
        actual_factor = -G0 / R_AIR - t_slope
        nominal_factor = G0 / R_AIR + tn_slope
        coefficients = (
            log_correction_slope * t_slope * tn_slope,
            actual_factor * tn_slope
            + nominal_factor * t_slope
            + log_correction_slope * (t0 * tn_slope + t_slope * tn0),
            actual_factor * tn0 + nominal_factor * t0 + log_correction_slope * t0 * tn0,
        )
        if abs(coefficients[0]) > 1e-24:
            roots = np.roots(coefficients)
        elif abs(coefficients[1]) > 1e-18:
            roots = (-coefficients[2] / coefficients[1],)
        else:
            roots = ()
        for root in roots:
            if abs(np.imag(root)) < 1e-9 and 0.0 < float(np.real(root)) < length:
                ratios.append(ratio(z0 + float(np.real(root))))
    return min(ratios), max(ratios)


@dataclass(frozen=True)
class AprConditions:
    """Synthetic input eligibility, not proof of flight clearance or landing feasibility."""

    wind_profile: tuple[tuple[float, float], ...] = ((0.0, 0.0), (PROFILE_TOP_M, 0.0))
    forecast_wind_profile: tuple[tuple[float, float], ...] = ((0.0, 0.0), (PROFILE_TOP_M, 0.0))
    vertical_wind_profile: tuple[tuple[float, float], ...] = ((0.0, 0.0), (PROFILE_TOP_M, 0.0))
    gust_amplitude_m_s: float = 0.0
    gust_period_s: float = 12.0
    gust_phase_rad: float = 0.0
    gust_start_s: float = 0.0
    gust_duration_s: float = 60.0
    gust_altitude_center_m: float = 10_000.0
    gust_altitude_half_width_m: float = 2_500.0
    forecast_gust_amplitude_m_s: float = 0.0
    sea_level_temperature_k: float = 288.15
    sea_level_pressure_pa: float = 101325.0
    density_scale: float = 1.0
    health_checks_passed: bool = True
    known_fault: bool = False
    lightning_cloud_review_passed: bool = True
    temperature_review_passed: bool = True
    recovery_weather_review_passed: bool = True
    precipitation_mm_h: float = 0.0
    thunderstorm_or_hail_present: bool = False
    temperature_profile: tuple[tuple[float, float], ...] = ()
    pressure_profile: tuple[tuple[float, float], ...] = ()
    latitude_deg: float = 37.0

    def __post_init__(self):
        for name in (
            "health_checks_passed",
            "lightning_cloud_review_passed",
            "temperature_review_passed",
            "recovery_weather_review_passed",
        ):
            if getattr(self, name) is not True:
                raise ValueError(f"{name}: 기체·기상 검토 미통과 입력은 실행하지 않습니다.")
        if self.known_fault is not False or self.thunderstorm_or_hail_present is not False:
            raise ValueError("알려진 고장·뇌우·우박 입력은 정상 운용 실험에서 제외합니다.")
        limits = {
            "gust_amplitude_m_s": (0.0, MAX_GUST_M_S),
            "gust_period_s": (8.0, 60.0),
            "gust_phase_rad": (-pi, pi),
            "gust_start_s": (-120.0, 600.0),
            "gust_duration_s": (8.0, 120.0),
            "gust_altitude_center_m": (0.0, PROFILE_TOP_M),
            "gust_altitude_half_width_m": (100.0, 5_000.0),
            "forecast_gust_amplitude_m_s": (0.0, MAX_GUST_M_S),
            "sea_level_temperature_k": (273.15, 308.15),
            "sea_level_pressure_pa": (95000.0, 105000.0),
            "density_scale": (1.0 - MAX_DENSITY_SCALE_DEVIATION, 1.0 + MAX_DENSITY_SCALE_DEVIATION),
            "latitude_deg": (-90.0, 90.0),
            "precipitation_mm_h": (0.0, 0.0),
        }
        for name, (low, high) in limits.items():
            value = _finite(getattr(self, name), name)
            if not low <= value <= high:
                raise ValueError(f"{name}: 잠정 정상 운용 범위 [{low}, {high}]를 벗어났습니다.")
        if self.gust_amplitude_m_s > 0.0 and not (
            self.gust_start_s < 600.0 and self.gust_start_s + self.gust_duration_s > 0.0
        ):
            raise ValueError("gust window: 돌풍은 0~600 s 비행 시간과 겹쳐야 합니다.")
        temperatures = _atmosphere_profile(
            self.temperature_profile, "temperature_profile", self.sea_level_temperature_k
        )
        pressures = _atmosphere_profile(
            self.pressure_profile, "pressure_profile", self.sea_level_pressure_pa
        )
        temperature_checks = {z for z, _ in temperatures} | ({11_000.0} if temperatures else set())
        for z in temperature_checks:
            temperature = float(
                np.interp(z, [p[0] for p in temperatures], [p[1] for p in temperatures])
            )
            standard = atmosphere(
                z, AtmosphereSettings(sea_level_temperature_k=self.sea_level_temperature_k)
            )[0]
            if not 150.0 <= temperature <= 330.0 or abs(temperature - standard) > 25.0:
                raise ValueError("temperature_profile: 고도별 정상 대기 범위를 벗어났습니다.")
        for (z0, t0), (z1, t1) in zip(temperatures, temperatures[1:], strict=False):
            if abs(t1 - t0) / (z1 - z0) > 0.02:
                raise ValueError("temperature_profile: 고도별 온도 기울기가 너무 큽니다.")
        reference = AtmosphereSettings(
            sea_level_temperature_k=self.sea_level_temperature_k,
            sea_level_pressure_pa=self.sea_level_pressure_pa,
            temperature_profile=temperatures,
        )
        for z, pressure in pressures:
            reference_pressure = atmosphere(z, reference)[1]
            if not 0.9 <= pressure / reference_pressure <= 1.1:
                raise ValueError("pressure_profile: 고도별 기압이 정역학 기준에서 벗어났습니다.")
        for (z0, p0), (z1, p1) in zip(pressures, pressures[1:], strict=False):
            if not 0.0 < p0 <= 105000.0 or not 0.0 < p1 < p0:
                raise ValueError("pressure_profile: 기압은 양수이고 고도에 따라 감소해야 합니다.")
            expected = atmosphere(z1, reference)[1] / atmosphere(z0, reference)[1]
            measured_log_drop = np.log(p0 / p1)
            expected_log_drop = -np.log(expected)
            if not 0.9 <= measured_log_drop / expected_log_drop <= 1.1:
                raise ValueError("pressure_profile: 온도와 맞지 않는 비정상 기압 기울기입니다.")
        object.__setattr__(self, "temperature_profile", temperatures)
        object.__setattr__(self, "pressure_profile", pressures)
        physical_atmosphere = AtmosphereSettings(
            sea_level_temperature_k=self.sea_level_temperature_k,
            sea_level_pressure_pa=self.sea_level_pressure_pa,
            temperature_profile=temperatures,
            pressure_profile=pressures,
            density_scale=self.density_scale,
        )
        lowest_ratio, highest_ratio = _density_ratio_extrema(physical_atmosphere)
        if (
            lowest_ratio < 1.0 - MAX_DENSITY_SCALE_DEVIATION - 1e-9
            or highest_ratio > 1.0 + MAX_DENSITY_SCALE_DEVIATION + 1e-9
        ):
            raise ValueError(
                "최종 대기밀도가 고정 기준 대기모델 대비 ±10%를 초과했습니다. "
                f"전 고도 비율 범위: {lowest_ratio:.4f}~{highest_ratio:.4f}"
            )
        actual = _wind_profile(self.wind_profile, "wind_profile")
        forecast = _wind_profile(self.forecast_wind_profile, "forecast_wind_profile")
        vertical = _vertical_wind_profile(self.vertical_wind_profile)
        object.__setattr__(self, "wind_profile", actual)
        object.__setattr__(self, "forecast_wind_profile", forecast)
        object.__setattr__(self, "vertical_wind_profile", vertical)
        altitudes = {z for z, _ in actual} | {z for z, _ in forecast} | {50.0}
        for z in altitudes:
            aw, fw = self.mean_wind(z), self.mean_wind(z, forecast=True)
            if abs(aw) + self.gust_amplitude_m_s > MAX_ALOFT_WIND_M_S + MAX_GUST_M_S + 1e-9:
                raise ValueError("상층 순간풍이 잠정 한계를 초과했습니다.")
            if (
                z <= 50.0
                and abs(aw) + self.gust_amplitude_m_s > MAX_PAD_WIND_M_S + MAX_GUST_M_S + 1e-9
            ):
                raise ValueError("지상 순간풍이 잠정 13 m/s 한계를 초과했습니다.")
            if abs(aw - fw) > MAX_FORECAST_ERROR_M_S + 1e-9:
                raise ValueError("기본풍 예보 오차가 잠정 ±2 m/s 범위를 초과했습니다.")

    def mean_wind(self, altitude_m, *, forecast=False):
        levels = self.forecast_wind_profile if forecast else self.wind_profile
        return float(np.interp(altitude_m, [p[0] for p in levels], [p[1] for p in levels]))

    def wind(self, altitude_m, time_s, *, forecast=False):
        wind = self.mean_wind(altitude_m, forecast=forecast)
        if not forecast and self.gust_amplitude_m_s > 0.0:
            elapsed = time_s - self.gust_start_s
            offset = abs(altitude_m - self.gust_altitude_center_m)
            if 0.0 < elapsed < self.gust_duration_s and offset < self.gust_altitude_half_width_m:
                # A single finite event, with smooth 2 s onset/recovery and a
                # compact altitude layer. No random draws occur during calls.
                ramp_s = min(2.0, self.gust_duration_s / 4.0)
                time_envelope = _smoothstep(elapsed / ramp_s) * _smoothstep(
                    (self.gust_duration_s - elapsed) / ramp_s
                )
                altitude_envelope = cos(0.5 * pi * offset / self.gust_altitude_half_width_m) ** 2
                wind += (
                    self.gust_amplitude_m_s
                    * time_envelope
                    * altitude_envelope
                    * sin(2.0 * pi * time_s / self.gust_period_s + self.gust_phase_rad)
                )
        return wind

    def vertical_wind(self, altitude_m, *, forecast=False):
        if forecast:
            return 0.0  # The tested updraft/downdraft is not supplied to the controller.
        levels = self.vertical_wind_profile
        return float(np.interp(altitude_m, [p[0] for p in levels], [p[1] for p in levels]))


@dataclass(frozen=True)
class AprWindField:
    conditions: AprConditions
    time_s: float = 0.0
    forecast: bool = False

    def at(self, altitude_m):
        return self.conditions.wind(altitude_m, self.time_s, forecast=self.forecast)

    def vertical_at(self, altitude_m):
        return self.conditions.vertical_wind(altitude_m, forecast=self.forecast)


def load_apr_operating(path):
    try:
        value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError("APR 설정 YAML 구문을 확인하세요.") from error
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "root_seed",
        "vehicle",
        "environment",
        "disturbances",
    }:
        raise ValueError("APR 설정의 누락 또는 알 수 없는 항목을 확인하세요.")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("APR 설정은 schema_version 1을 사용합니다.")
    if type(value["root_seed"]) is not int or value["root_seed"] != 378:
        raise ValueError("root_seed는 378로 고정합니다.")
    design = asdict(Design())
    if not isinstance(value["vehicle"], dict) or set(value["vehicle"]) != set(design):
        raise ValueError("vehicle에는 현재 100톤 기체의 모든 제원만 입력할 수 있습니다.")
    for name, expected in design.items():
        if _finite(value["vehicle"][name], name) != expected:
            raise ValueError(f"vehicle.{name}: 현재 기준값 {expected}으로 고정됩니다.")
    for name, group in (("environment", AprConditions), ("disturbances", RecoveryDisturbances)):
        section = value[name]
        permitted = {f.name for f in fields(group)}
        optional = (
            {
                "temperature_profile",
                "pressure_profile",
                "latitude_deg",
                "vertical_wind_profile",
                "density_scale",
                "gust_start_s",
                "gust_duration_s",
                "gust_altitude_center_m",
                "gust_altitude_half_width_m",
                "forecast_gust_amplitude_m_s",
            }
            if name == "environment"
            else {
                "drag_coefficient_scale",
                "initial_mass_scale",
                "sensor_position_error_m",
                "sensor_velocity_error_m_s",
            }
        )
        if not isinstance(section, dict) or not permitted - optional <= set(section) <= permitted:
            raise ValueError(f"{name}: 누락 또는 알 수 없는 설정 항목입니다.")
    return AprConditions(**value["environment"]), RecoveryDisturbances(**value["disturbances"])


@dataclass(frozen=True)
class FlightSafety:
    """100톤 개념 구조의 계산 한계와 보수적인 운용 요구값."""

    shell_wall_m: float = 0.004
    shell_yield_pa: float = 250_000_000.0
    material_safety_factor: float = 1.5
    max_q_pa: float = 30_000.0
    max_q_alpha_kpa_deg: float = 150.0
    max_high_q_alpha_deg: float = 5.0
    max_angular_rate_deg_s: float = 15.0
    max_angular_acceleration_deg_s2: float = 5.0
    max_ascent_axial_g: float = 3.0
    max_recovery_flight_g: float = 5.0
    absolute_g: float = 6.0

    def derivation(self):
        radius = PHYSICS.geometry.body_radius_m
        stress = self.shell_yield_pa / self.material_safety_factor
        axial = 2 * pi * radius * self.shell_wall_m * stress
        bending = pi * radius**2 * self.shell_wall_m * stress
        area = 2 * radius * PHYSICS.geometry.body_length_m
        fin_q = FINS.per_fin_hinge_limit_nm / (
            FINS.planar_effective_area_m2 / 2 * FINS.max_lift_coefficient * FINS.hinge_arm_m
        )
        return dict(
            shell_axial_allowable_n=axial,
            shell_bending_allowable_nm=bending,
            full_deflection_fin_q_allowable_pa=fin_q,
            conservative_body_q_alpha_allowable_kpa_deg=(
                bending / (area * 0.8 * PHYSICS.geometry.body_length_m / 2) * 180 / pi / 1000
            ),
            buckling_fatigue_thermal_and_real_hardware_verified=False,
            operational_limits_are_design_requirements=True,
        )


SAFETY = FlightSafety()


@dataclass(frozen=True)
class AscentGuidanceSettings:
    separation_altitude_m: float = 37_000.0
    separation_tolerance_m: float = 1_000.0
    separation_min_main_propellant_kg: float = 20_000.0
    separation_max_q_pa: float = 25_000.0
    separation_max_rate_deg_s: float = 5.0
    separation_max_alpha_deg: float = 5.0
    pitch_start_s: float = 15.0
    pitch_duration_s: float = 30.0
    final_pitch_deg: float = 80.0
    separation_alignment_preview_s: float = 8.0
    separation_alignment_blend_s: float = 4.0


@dataclass(frozen=True)
class ReturnGuidanceSettings:
    control_dt_s: float = 0.02
    attitude_rate_limit_deg_s: float = 8.0
    attitude_acceleration_limit_deg_s2: float = 2.5
    landing_entry_height_m: float = 2_000.0
    gear_deployment_height_m: float = 300.0
    forecast_step_s: float = 0.5
    main_reserve_requirement_kg: float = 0.0
    flip_max_q_pa: float = 250.0
    boost_acceleration_m_s2: float = 3.0 * G0
    boost_replan_period_s: float = 1.0
    predicted_miss_limit_m: float = 75.0
    entry_q_fraction: float = 0.60
    entry_braking_m_s2: float = 20.0
    terminal_braking_m_s2: float = 8.0
    terminal_speed_plateau_height_m: float = 3.0
    acceleration_bias_start_height_m: float = 500.0
    acceleration_bias_full_height_m: float = 100.0
    acceleration_bias_limit_m_s2: float = 2.0
    horizontal_alignment_height_m: float = 3_000.0
    # Finish the finite-time upright turn before strong low-altitude braking.
    vertical_body_height_m: float = 3_000.0
    # A two-second retraction must precede the rapid flow rotation during braking.
    fin_min_descent_speed_m_s: float = 200.0
    entry_attitude_cone_deg: float = 8.0
    entry_attitude_cone_min_q_pa: float = 5_000.0


@lru_cache(maxsize=16)
def build_descent_reference(atmosphere_settings, q_budget, entry_braking, terminal_braking):
    """접지에서 거꾸로 적분한 속도 한계. v²(h+dh) ≤ v²(h)+2*a*dh.

    현재 동압이 한계에 도달하기 전에, 더 낮은 고도의 밀도와 남은 제동
    거리를 반영한다. 속도·동압 여유는 유도 참조이며 실제 성공 기준은 별도다.
    """
    heights = np.unique(
        np.r_[
            np.linspace(0, 500, 251),
            np.linspace(500, 10_000, 381),
            np.linspace(10_000, PROFILE_TOP_M, 901),
        ]
    )
    density = np.array([atmosphere(h, atmosphere_settings)[2] for h in heights])
    q_speed_squared = 2.0 * q_budget / density
    blending = np.clip((heights - 1500) / 2000, 0, 1)
    blending = blending**2 * (3 - 2 * blending)
    braking = terminal_braking + (entry_braking - terminal_braking) * blending
    squared = np.empty_like(heights)
    squared[0] = Design().descent_reference_m_s ** 2
    for index in range(1, len(heights)):
        reachable = squared[index - 1] + (braking[index - 1] + braking[index]) * (
            heights[index] - heights[index - 1]
        )
        squared[index] = min(q_speed_squared[index], reachable)
    speed = np.sqrt(squared)
    time_to_ground = np.r_[0, np.cumsum(2 * np.diff(heights) / (speed[1:] + speed[:-1]))]
    for array in (heights, squared, time_to_ground):
        array.flags.writeable = False
    return heights, squared, time_to_ground


class AprFlight:
    """실제 유한 자세·짐벌·핀·엔진 응답과 별도 추진제 장부를 적분한다.

    예측기는 참조 궤적용 근사이며, 판정은 예측값이 아닌 적분된 강체·발 접지
    상태만 사용한다. 기본 평가는 첫 접지에서 끝내고 다리 내구성은 설계 가정으로
    둔다. 2단은 전체 질량에 포함하되 분리 후 비행을 적분하지 않는다.
    """

    def __init__(
        self,
        *,
        fin_mode="active",
        conditions=None,
        disturbances=None,
        integration_dt_s=0.01,
        feed_settings=None,
        slosh_settings=None,
        separation_settings=None,
        ignition_settings=None,
        landing_gear=None,
        sensor_settings=None,
        controller_options=None,
    ):
        if fin_mode not in ("stowed", "active"):
            raise ValueError("fin_mode must be stowed or active")
        self.landing_assessment = "kinematic_touchdown"
        if not np.isfinite(integration_dt_s) or not 0 < integration_dt_s <= 0.02:
            raise ValueError("integration_dt_s must be within (0, 0.02]")
        self.nominal_design = Design()
        self.sensor_settings = SensorSettings() if sensor_settings is None else sensor_settings
        self.controller_options = (
            ControllerOptions() if controller_options is None else controller_options
        )
        if not isinstance(self.sensor_settings, SensorSettings):
            raise TypeError("Expected SensorSettings")
        if not isinstance(self.controller_options, ControllerOptions):
            raise TypeError("Expected ControllerOptions")
        self.feed_settings = FeedSettings() if feed_settings is None else feed_settings
        self.slosh_settings = SloshSettings() if slosh_settings is None else slosh_settings
        self.separation_settings = (
            SeparationSettings() if separation_settings is None else separation_settings
        )
        self.ignition_settings = (
            IgnitionSettings() if ignition_settings is None else ignition_settings
        )
        for value, expected in (
            (self.feed_settings, FeedSettings),
            (self.slosh_settings, SloshSettings),
            (self.separation_settings, SeparationSettings),
            (self.ignition_settings, IgnitionSettings),
        ):
            if not isinstance(value, expected):
                raise TypeError(f"Expected {expected.__name__}")
        if landing_gear is not None and not isinstance(landing_gear, LandingGear):
            raise TypeError("landing_gear must be LandingGear")
        self.conditions = (
            AprConditions() if conditions is None else AprConditions(**asdict(conditions))
        )
        self.disturbances = (
            RecoveryDisturbances()
            if disturbances is None
            else RecoveryDisturbances(**asdict(disturbances))
        )
        extra = self.nominal_design.launch_mass_kg * (self.disturbances.initial_mass_scale - 1)
        self.design = replace(
            self.nominal_design, dry_mass_kg=self.nominal_design.dry_mass_kg + extra
        )
        self.physics = replace(
            PHYSICS,
            slosh_mass_fraction=self.slosh_settings.modal_mass_fraction,
            slosh_settings=self.slosh_settings,
            gear=PHYSICS.gear if landing_gear is None else landing_gear,
            geometry=replace(
                PHYSICS.geometry, airframe_mass_kg=PHYSICS.geometry.airframe_mass_kg + extra
            ),
            atmosphere=replace(
                PHYSICS.atmosphere,
                sea_level_temperature_k=self.conditions.sea_level_temperature_k,
                sea_level_pressure_pa=self.conditions.sea_level_pressure_pa,
                density_scale=self.conditions.density_scale,
                temperature_profile=self.conditions.temperature_profile,
                pressure_profile=self.conditions.pressure_profile,
                latitude_deg=self.conditions.latitude_deg,
            ),
            propulsion=replace(
                PHYSICS.propulsion, nozzle_exit_area_m2=self.design.nozzle_exit_area_m2
            ),
        )
        budget = (
            self.physics.geometry.engine_mass_kg
            + self.physics.geometry.airframe_mass_kg
            + self.physics.geometry.gear_mass_kg
            + FINS.mass_kg
        )
        if not np.isclose(budget, self.design.dry_mass_kg, atol=1e-8):
            raise ValueError("100톤 건조질량과 부품 배분이 일치하지 않습니다.")
        g = self.physics.geometry
        if pi * g.body_radius_m**2 * g.tank_length_m * g.propellant_density_kg_m3 <= (
            self.design.main_propellant_kg + self.design.rcs_propellant_kg
        ):
            raise ValueError("추진제 탱크에 양의 기상 공간이 필요합니다.")
        self.fin_mode = fin_mode
        self.fin_selection_reason = "installed_control_surfaces"
        self.ascent_settings = AscentGuidanceSettings()
        self.return_settings = ReturnGuidanceSettings()
        self.integration_dt_s = integration_dt_s
        self.upper_attached = True
        self.phase = "ascent"
        self.scope = "launch_to_separation_to_touchdown"
        self.state = np.zeros(len(STATE_NAMES))
        self.state[6:8] = (self.design.main_propellant_kg, self.design.rcs_propellant_kg)
        feed = initial_feed_state(self.design.main_propellant_kg, self.feed_settings)
        self.state[13:17] = (feed.buffer_kg, feed.acquisition, feed.tank_gas_kg, feed.bottle_gas_kg)
        self.ignition_requested = False
        self.ignition_ready_s = float("inf")
        self.ignition_events = []
        self.minimum_feed_buffer_kg = float(feed.buffer_kg)
        self.minimum_tank_pressure_pa = float("inf")
        self.max_feed_shortage_kg_s = 0.0
        self.max_helium_balance_error_kg = 0.0
        self.max_slosh_displacement_m = 0.0
        self.max_slosh_mode_force_n = 0.0
        self.slosh_model_valid = True
        self.first_subsystem_failure = None
        self.feed_inventory_valid = True
        self.feed_supply_valid = True
        self.state[1] = self.properties(self.state).cg_m - self.physics.gear.foot_station_m
        self.on_pad = True
        self.time_s = 0.0
        self.started_s = 0.0
        self.deadline_s = 600.0
        self.next_control_s = 0.0
        self.next_record_s = 0.0
        self.control = (0.0, 0.0, 0.0, 0.0)
        self.requested_control = self.control
        self.engine_throttle = 0.0
        self.attitude_target = 0.0
        self.separation = None
        self.contact = None
        self.outcome = "running"
        self.failures = []
        self.steps = []
        self.cutoff_pending = False
        self.separation_alignment_started_s = None
        self.ascent_max_q_passed = False
        self.max_ascent_q = 0.0
        self.max_q = 0.0
        self.max_return_q = 0.0
        self.max_q_alpha = 0.0
        self.max_ascent_alpha = 0.0
        self.max_ascent_axial_g = 0.0
        self.max_flight_g = 0.0
        self.max_rate = 0.0
        self.max_acceleration = 0.0
        self.peak_angular_acceleration = None
        self.max_gimbal = 0.0
        self.max_fin_hinge = 0.0
        self.max_fin_mach = 0.0
        self.max_fin_alpha = 0.0
        self.max_altitude = 0.0
        self.max_mass_balance_error = 0.0
        self.integration_gust_exposure = dict(
            maximum_contribution_m_s=0.0,
            maximum_by_phase_m_s={},
            endpoints_above_0_1_m_s=0,
            first_endpoint_above_0_1_m_s=None,
            last_endpoint_above_0_1_m_s=None,
            gust_encounter_observed=False,
            closest_descending_altitude_sample=None,
            sampling_note="Accepted integration endpoints, not every RK intermediate stage",
        )
        self.main_consumed = 0.0
        self.rcs_consumed = 0.0
        self.boost_target_vx = None
        self.boostback_completed = None
        self.return_plan = None
        self.return_plan_history = []
        self.boost_direction = -1.0
        self.boost_end_s = None
        self.next_plan_s = 0.0
        self.flip_started_s = None
        self.entry_started_s = None
        self.minimum_control_torque_margin_nm = 1e12
        self.rng = np.random.default_rng(378)
        self.sensor_rng = np.random.default_rng(9378)
        self.sensor_sample_count = 0
        self.sensor_dropout_count = 0
        self.last_sensor_packet = None
        self.observations = deque(maxlen=self.disturbances.observation_delay_steps + 1)
        self.observation_times = deque(maxlen=self.observations.maxlen)
        self.navigation_command_history = deque(
            maxlen=self.observations.maxlen + self.sensor_settings.dropout_length_steps + 1
        )
        self.action_queue = deque(
            [self.control] * (self.disturbances.action_delay_steps + 1),
            maxlen=self.disturbances.action_delay_steps + 1,
        )
        self.navigation_estimator = PlanarNavigationEstimator(
            self.disturbances.sensor_position_error_m,
            self.disturbances.sensor_velocity_error_m_s,
        )
        self.observed_state = self.state.copy()
        self.angular_observation = None
        self.angular_torque_bias_nm = 0.0
        self.attitude_estimator = PlanarAttitudeEstimator(
            self.sensor_settings.attitude_error_deg, self.sensor_settings.gyro_error_deg_s
        )

    @classmethod
    def from_separation(cls, state, *, time_s, **kwargs):
        flight = cls(**kwargs)
        value = np.asarray(state, dtype=float)
        if value.ndim != 1 or len(value) != len(STATE_NAMES) or not np.isfinite(value).all():
            raise ValueError("현재 100톤 모델의 19개 유한 상태값이 필요합니다.")
        if not 0 < value[6] <= flight.design.main_propellant_kg or not (
            0 <= value[7] <= flight.design.rcs_propellant_kg
        ):
            raise ValueError("분리 추진제 장부가 제원을 벗어났습니다.")
        flight.upper_attached = False
        flight.on_pad = False
        flight.state[:] = value
        flight.time_s = flight.started_s = float(time_s)
        flight.next_control_s = flight.next_record_s = flight.time_s
        flight.deadline_s = flight.time_s + 500
        flight.phase = "flip"
        flight.scope = "37km_separation_to_touchdown"
        flight.main_consumed = flight.design.main_propellant_kg - value[6]
        flight.rcs_consumed = flight.design.rcs_propellant_kg - value[7]
        flight.separation = dict(
            time_s=time_s, state=flight.state.tolist(), checks_before_separation={}
        )
        return flight

    def mass(self, state, *, predicted=False):
        return (
            (self.nominal_design.dry_mass_kg if predicted else self.design.dry_mass_kg)
            + state[6]
            + state[7]
            + (self.design.upper_mass_kg if self.upper_attached else 0)
        )

    def settings(self, *, attached=None, predicted=False):
        attached = self.upper_attached if attached is None else attached
        # 설정은 불변이다. 기체·제원이 교체되면 이전 캐시를 버린다.
        upper = self.design.upper_mass_kg
        cache = getattr(self, "_settings_cache", None)
        if (
            cache is None
            or cache[0] is not self.physics
            or cache[1] != upper
            or cache[2] is not PHYSICS.geometry
        ):
            cache = (self.physics, upper, PHYSICS.geometry, {})
            self._settings_cache = cache
        key = (attached, predicted)
        cached = cache[3].get(key)
        if cached is not None:
            return cached
        settings = self.physics
        if predicted:
            settings = replace(
                settings,
                atmosphere=replace(settings.atmosphere, density_scale=1.0),
                geometry=PHYSICS.geometry,
            )
        if attached:
            settings = replace(
                settings,
                fixed_mass_components=settings.fixed_mass_components
                + (
                    (
                        upper,
                        settings.upper_station_m,
                        upper
                        * (
                            settings.upper_length_m**2 / 12 + settings.geometry.body_radius_m**2 / 4
                        ),
                    ),
                ),
            )
        cache[3][key] = settings
        return settings

    def properties(self, state, *, predicted=False):
        properties = mass_properties(
            self.mass(state, predicted=predicted), state[8], self.settings(predicted=predicted)
        )
        if predicted or self.disturbances.inertia_scale == 1.0:
            return properties
        # Uncertainty belongs to the first-stage central inertia, not the
        # exactly known stage-to-stage parallel-axis terms. This allocation
        # remains the same on both sides of the separation event.
        booster = (
            mass_properties(
                self.mass(state) - self.design.upper_mass_kg,
                state[8],
                self.settings(attached=False),
            )
            if self.upper_attached
            else properties
        )
        return replace(
            properties,
            inertia_kg_m2=properties.inertia_kg_m2
            + (self.disturbances.inertia_scale - 1.0) * booster.inertia_kg_m2,
        )

    def inertia_rate(self, state, mass_rate, deployment_rate, properties=None):
        properties = self.properties(state) if properties is None else properties
        mass = self.mass(state)
        _, _, im, idp = mass_gradients(mass, state[8], self.settings(), properties)
        nominal_rate = im * mass_rate + idp * deployment_rate
        scale = self.disturbances.inertia_scale
        if scale == 1.0:
            return nominal_rate
        if not self.upper_attached:
            return scale * nominal_rate
        booster_mass = mass - self.design.upper_mass_kg
        booster_settings = self.settings(attached=False)
        booster = mass_properties(booster_mass, state[8], booster_settings)
        _, _, bm, bg = mass_gradients(booster_mass, state[8], booster_settings, booster)
        return nominal_rate + (scale - 1.0) * (bm * mass_rate + bg * deployment_rate)

    def vector(self, state, *, predicted=False):
        vector = np.empty(10)
        vector[:6] = state[:6]
        vector[6] = self.mass(state, predicted=predicted)
        vector[7] = state[6]
        vector[8:] = state[17:19]
        return vector

    def height(self, state, *, predicted=False, properties=None):
        vector = self.vector(state, predicted=predicted)
        settings = self.settings(predicted=predicted)
        g, gear = settings.geometry, settings.gear
        if properties is None:
            properties = mass_properties(vector[6], state[8], settings)
        span = g.body_radius_m + state[8] * (gear.half_span_m - g.body_radius_m)
        station = gear.anchor_station_m + state[8] * (gear.foot_station_m - gear.anchor_station_m)
        offset = settings.body_frame_offset(vector)[0]
        body_arms = np.asarray(
            (
                (-span + offset[0], station - properties.cg_m + offset[1]),
                (span + offset[0], station - properties.cg_m + offset[1]),
            )
        )
        points = vector[:2] + body_arms @ rotation(vector[4]).T
        return float(points[:, 1].min())

    def wind_field(self, *, forecast=False, time_s=None):
        return AprWindField(self.conditions, self.time_s if time_s is None else time_s, forecast)

    def engine(self, throttle, altitude, *, actual=False):
        thrust, flow = engine_performance(
            throttle,
            altitude,
            self.physics,
            sea_level_thrust_n=self.design.sea_level_thrust_n,
            sea_level_isp_s=self.design.sea_level_isp_s,
        )
        if actual:
            # 추력 편차는 같은 유량에서의 성능 편차로 처리한다.
            thrust *= self.disturbances.thrust_scale
        return thrust, flow

    def air_velocity(self, state, *, predicted=False, time_s=None):
        field = self.wind_field(forecast=predicted, time_s=time_s)
        return np.array([state[2] - field.at(state[1]), state[3] - field.vertical_at(state[1])])

    def feed_status(self, state, demand=0.0, axial_acceleration=0.0):
        return feed_rates(
            state[6], demand, axial_acceleration, FeedState(*state[13:17]), self.feed_settings
        )

    def actual_engine(self, state):
        thrust, flow = self.engine(float(np.clip(state[12], 0, 1)), state[1], actual=True)
        feed = self.feed_status(state, flow)
        if flow > 0:
            thrust *= feed.delivered_flow_kg_s / flow
            flow = feed.delivered_flow_kg_s
        return (thrust, flow) if state[6] > 0 else (0.0, 0.0)

    def ignition_control(self, command):
        throttle, *other = command
        if throttle <= 1e-8:
            self.ignition_requested = False
            self.ignition_ready_s = float("inf")
            return command
        if not self.ignition_requested:
            self.ignition_requested = True
            self.ignition_ready_s = self.time_s if self.state[12] > 1e-3 else float("inf")
        if not np.isfinite(self.ignition_ready_s):
            feed = self.feed_status(self.state)
            ready = ignition_ready(
                FeedState(*self.state[13:17]), feed.tank_pressure_pa, self.ignition_settings
            )
            if ready:
                self.ignition_ready_s = self.time_s + self.ignition_settings.delay_s
                self.ignition_events.append(
                    dict(
                        requested_s=self.time_s,
                        ready_s=self.ignition_ready_s,
                        phase=self.phase,
                        buffer_kg=float(self.state[13]),
                        tank_pressure_pa=feed.tank_pressure_pa,
                    )
                )
        if self.time_s < self.ignition_ready_s - 1e-9:
            return (0.0, *other)
        return command

    def alpha_deg(self, state, *, predicted=False):
        vx, vz = self.air_velocity(state, predicted=predicted)
        angle = atan2(vx, vz) if hypot(vx, vz) > 1e-6 else state[4]
        # 재진입은 엔진 쪽이 선두인 축방향 유동이다. 축의 양방향을 구분한다.
        error = atan2(sin(state[4] - angle), cos(state[4] - angle))
        if error > pi / 2:
            error -= pi
        elif error < -pi / 2:
            error += pi
        return float(np.rad2deg(error))

    def body_loads(self, state, *, predicted=False, time_s=None):
        settings = self.settings(predicted=predicted)
        if self.upper_attached:
            length = self.physics.geometry.body_length_m + self.physics.upper_length_m
            settings = replace(
                settings,
                geometry=replace(settings.geometry, body_length_m=length),
                aerodynamics=replace(settings.aerodynamics, pressure_center_m=length / 2),
            )
        if not predicted:
            scale = self.disturbances.drag_coefficient_scale
            settings = replace(
                settings,
                aerodynamics=replace(
                    settings.aerodynamics,
                    axial_cd=settings.aerodynamics.axial_cd * scale,
                    normal_cd=settings.aerodynamics.normal_cd * scale,
                ),
            )
        force, moment, q, mach = aerodynamic_loads(
            self.vector(state, predicted=predicted),
            state[8],
            self.wind_field(forecast=predicted, time_s=time_s),
            settings,
            self.properties(state, predicted=predicted),
        )
        offset = settings.body_frame_offset(self.vector(state, predicted=predicted))[0]
        moment += clock_torque(rotation(state[4]) @ offset, force)
        return force, moment, q, mach

    def fin_loads(self, state, deflection=None, *, predicted=False, time_s=None):
        if self.upper_attached or self.fin_mode == "stowed":
            return np.zeros(2), 0.0, 0.0
        velocity = self.air_velocity(state, predicted=predicted, time_s=time_s)
        speed = hypot(*velocity)
        if speed < 1e-6:
            return np.zeros(2), 0.0, 0.0
        rho = atmosphere(state[1], self.settings(predicted=predicted).atmosphere)[2]
        q = 0.5 * rho * speed**2
        angle = state[9] if deflection is None else deflection
        cl = float(
            np.clip(
                FINS.lift_slope_rad_inv * angle,
                -FINS.max_lift_coefficient,
                FINS.max_lift_coefficient,
            )
        )
        extension = state[10]
        lift = q * FINS.planar_effective_area_m2 * cl * extension
        drag = (
            q
            * FINS.planar_effective_area_m2
            * (FINS.base_drag_coefficient + FINS.induced_drag_factor * cl**2)
            * extension
        )
        normal = np.array([-velocity[1], velocity[0]]) / speed
        force = lift * normal - drag * velocity / speed
        offset = self.settings(predicted=predicted).body_frame_offset(
            self.vector(state, predicted=predicted)
        )[0]
        arm = rotation(state[4]) @ np.array(
            [
                offset[0],
                FINS.station_m - self.properties(state, predicted=predicted).cg_m + offset[1],
            ]
        )
        hinge = abs(lift) / 2 * FINS.hinge_arm_m
        return force, clock_torque(arm, force), hinge

    def loads(self, state, *, predicted=False, time_s=None):
        body, torque, q, alpha = self.body_loads(state, predicted=predicted, time_s=time_s)
        fin, moment, _ = self.fin_loads(state, predicted=predicted, time_s=time_s)
        return body + fin, torque + moment, q, alpha

    def fin_command(self, state, requested_torque):
        if self.upper_attached or self.fin_mode == "stowed" or state[10] < 0.5:
            return 0.0
        unit = np.deg2rad(1.0)
        slope = (
            self.fin_loads(state, unit, predicted=True)[1]
            - self.fin_loads(state, -unit, predicted=True)[1]
        ) / (2 * unit)
        if abs(slope) < 1:
            return 0.0
        target = float(np.clip(requested_torque / slope, -np.deg2rad(20), np.deg2rad(20)))
        q = self.loads(state, predicted=True)[2]
        cl_limit = min(
            FINS.max_lift_coefficient,
            FINS.per_fin_hinge_limit_nm
            / max(q * FINS.planar_effective_area_m2 / 2 * FINS.hinge_arm_m, 1),
        )
        return float(
            np.clip(target, -cl_limit / FINS.lift_slope_rad_inv, cl_limit / FINS.lift_slope_rad_inv)
        )

    def derivative(
        self,
        state,
        control,
        *,
        time_s=None,
        available_propellant=None,
        systems_diagnostics=None,
    ):
        time_s = self.time_s if time_s is None else time_s
        throttle, gimbal, rcs, fin_target = control
        main_available, rcs_available = (
            (state[6] > 0, state[7] > 0) if available_propellant is None else available_propellant
        )
        throttle = throttle if main_available else 0
        rcs = rcs if rcs_available else 0
        actual_throttle = float(np.clip(state[12], 0, 1)) if main_available else 0
        thrust, demanded_flow = self.engine(actual_throttle, state[1], actual=True)
        feed = self.feed_status(state, demanded_flow)
        main_flow = demanded_flow
        if demanded_flow > 0:
            main_flow = feed.delivered_flow_kg_s
            thrust *= main_flow / demanded_flow
        rcs_flow = abs(rcs) / (self.physics.geometry.rcs_arm_m * self.design.rcs_isp_s * G0)
        mass_rate = -main_flow - rcs_flow
        props = self.properties(state)
        height = self.height(state, properties=props) if not self.upper_attached else None
        deployment_rate = (
            1 / self.physics.gear.deployment_time_s
            if not self.upper_attached
            and height < self.return_settings.gear_deployment_height_m
            and state[8] < 1
            else 0
        )
        force, aero_torque, _, _ = self.loads(state, time_s=time_s)
        force += thrust * np.array([sin(state[4] + state[11]), cos(state[4] + state[11])])
        mass = self.mass(state)
        acceleration = force / mass - np.array([0.0, gravity(state[1])])
        if self.physics.atmosphere.coriolis_enabled:
            omega = 7.292115e-5 * cos(np.deg2rad(self.conditions.latitude_deg))
            acceleration += 2 * omega * np.array([-state[3], state[2]])
        inertia_rate = self.inertia_rate(state, mass_rate, deployment_rate, props)
        lever = props.cg_m - self.physics.geometry.nozzle_station_m
        angular_flux = -main_flow * lever**2 * state[5]
        angular_flux -= (
            rcs_flow
            * (
                self.physics.geometry.rcs_arm_m**2
                + (self.physics.geometry.rcs_station_m - props.cg_m) ** 2
            )
            * state[5]
        )
        offset = self.settings().body_frame_offset(self.vector(state))[0]
        torque = (-lever + offset[1]) * thrust * sin(state[11]) - offset[0] * thrust * cos(
            state[11]
        )
        torque += rcs + aero_torque
        angular_acceleration = (
            torque + angular_flux - inertia_rate * state[5]
        ) / props.inertia_kg_m2
        proper_body = rotation(state[4]).T @ (force / mass)
        supported_on_pad = (
            self.on_pad and self.upper_attached and force[1] < mass * gravity(state[1])
        )
        if supported_on_pad:
            proper_body = rotation(state[4]).T @ np.array([0.0, gravity(state[1])])
        feed = self.feed_status(state, demanded_flow, float(proper_body[1]))
        station = self.physics.geometry.tank_bottom_m + props.fuel_height_m / 2.0
        slosh = slosh_response(
            mass,
            props.inertia_kg_m2,
            max(state[6], 0.0),
            station - props.cg_m,
            float(proper_body[1]),
            float(proper_body[0]),
            torque + angular_flux - inertia_rate * state[5],
            state[5],
            SloshState(*state[17:19]),
            self.slosh_settings,
        )
        angular_acceleration = slosh.angular_acceleration_rad_s2
        if supported_on_pad:
            acceleration[:] = 0
            angular_acceleration = 0
        fin_rate = np.clip(
            (fin_target - state[9]) / FINS.actuator_time_constant_s,
            -np.deg2rad(FINS.deflection_rate_deg_s),
            np.deg2rad(FINS.deflection_rate_deg_s),
        )
        fin_allowed = (
            not self.upper_attached
            and self.fin_mode == "active"
            and self.phase in ("coast", "entry", "landing")
            and 500 < height < 20_000
            and state[3] < -self.return_settings.fin_min_descent_speed_m_s
            and abs(self.alpha_deg(state)) < 15
        )
        fin_extension_rate = (
            1 / FINS.deployment_time_s
            if fin_allowed and state[10] < 1
            else -1 / FINS.retraction_time_s
            if not fin_allowed and state[10] > 0
            else 0
        )
        engine_tau = max(0.04, self.disturbances.engine_lag_s)
        mode_acceleration = slosh.displacement_acceleration_m_s2
        if supported_on_pad and slosh.modal_mass_kg > 1e-9:
            mode_acceleration += (
                slosh.rotational_slosh_coupling_kg_m
                / slosh.reduced_modal_mass_kg
                * slosh.angular_acceleration_rad_s2
            )
        if systems_diagnostics is not None:
            systems_diagnostics.update(feed=feed, slosh=slosh)
        rate = np.empty(len(STATE_NAMES))
        rate[:2] = state[2:4]
        rate[2:4] = acceleration
        rate[4] = state[5]
        rate[5] = angular_acceleration
        rate[6:11] = (-main_flow, -rcs_flow, deployment_rate, fin_rate, fin_extension_rate)
        rate[11] = np.clip((gimbal - state[11]) / 0.06, -np.deg2rad(60), np.deg2rad(60))
        rate[12] = (throttle - state[12]) / engine_tau
        rate[13:17] = feed.state_derivatives
        rate[17] = state[18]
        rate[18] = mode_acceleration
        return rate

    def attitude_control(self, target, throttle, observed, *, thrust_direction=None):
        self.attitude_target = target
        limits = self.return_settings
        error = atan2(sin(target - observed[4]), cos(target - observed[4]))
        rate_limit = np.deg2rad(limits.attitude_rate_limit_deg_s)
        alpha_limit = np.deg2rad(limits.attitude_acceleration_limit_deg_s2)
        body_load = self.body_loads(observed, predicted=True)
        if not self.upper_attached and body_load[2] >= limits.entry_attitude_cone_min_q_pa:
            alpha_limit = min(alpha_limit, np.deg2rad(1.0))
        stopping_rate = sqrt(2 * 0.8 * alpha_limit * abs(error))
        desired_rate = np.sign(error) * min(rate_limit, 1.5 * abs(error), stopping_rate)
        desired_alpha = float(np.clip(4 * (desired_rate - observed[5]), -alpha_limit, alpha_limit))
        props = self.properties(observed, predicted=True)
        # 총 발사질량 ±1%는 추진제가 아니라 기체 고정질량 ±1000 kg이다.
        # 같은 액체 장부를 유지한 가벼운 기체의 관성으로 가속 토크를 제한한다.
        lower_settings = self.settings(predicted=True)
        mass_deviation = 0.01 * self.nominal_design.launch_mass_kg
        lower_geometry = replace(
            lower_settings.geometry,
            airframe_mass_kg=lower_settings.geometry.airframe_mass_kg - mass_deviation,
        )
        lower_settings = replace(lower_settings, geometry=lower_geometry)
        lower_mass = self.mass(observed, predicted=True) - mass_deviation
        lower_props = mass_properties(lower_mass, observed[8], lower_settings)
        lower_booster = (
            mass_properties(
                lower_mass - self.design.upper_mass_kg,
                observed[8],
                replace(self.settings(attached=False, predicted=True), geometry=lower_geometry),
            )
            if self.upper_attached
            else lower_props
        )
        acceleration_inertia = lower_props.inertia_kg_m2 - 0.03 * lower_booster.inertia_kg_m2
        _, main_flow = self.engine(observed[12], observed[1])
        rcs_flow = abs(self.control[2]) / (
            self.physics.geometry.rcs_arm_m * self.design.rcs_isp_s * G0
        )
        lever = props.cg_m - self.physics.geometry.nozzle_station_m
        _, _, im, ideploy = mass_gradients(
            self.mass(observed, predicted=True), observed[8], self.settings(predicted=True), props
        )
        gear_rate = (
            1 / self.physics.gear.deployment_time_s
            if not self.upper_attached
            and observed[8] < 1
            and self.height(observed, predicted=True) < limits.gear_deployment_height_m
            else 0.0
        )
        inertia_rate = -im * (main_flow + rcs_flow) + ideploy * gear_rate
        angular_flux = -main_flow * lever**2 * observed[5]
        angular_flux -= (
            rcs_flow
            * (
                self.physics.geometry.rcs_arm_m**2
                + (self.physics.geometry.rcs_station_m - props.cg_m) ** 2
            )
            * observed[5]
        )
        # 몸체 토크와 핀 토크를 분리하여 실제 핀 모멘트는 한 번만 상쇄한다.
        requested_raw = (
            acceleration_inertia * desired_alpha
            + inertia_rate * observed[5]
            - angular_flux
            - body_load[1]
            - self.angular_torque_bias_nm
        )
        fin_target = self.fin_command(observed, requested_raw)
        limit = np.deg2rad(self.design.gimbal_limit_deg)
        # A new command executes only after the pending queue. Predict its
        # actuator starting state from observed telemetry and known commands;
        # do not flush the queue or read hidden actuator/model deviations.
        allocation_gimbal = float(observed[11])
        allocation_fin = float(observed[9])
        allocation_time = self.time_s
        pending_level = float(observed[12])
        pending_slow_level = pending_level
        pending_commands = list(self.action_queue)[1:]

        def eligible_throttle(scenario, command_level, start_time):
            if command_level <= 1e-8:
                scenario["ready"], scenario["requested"] = float("inf"), False
                return 0.0
            if not scenario["requested"]:
                scenario["requested"] = True
                scenario["ready"] = start_time if scenario["level"] > 1e-3 else float("inf")
            # Unknown supply/ignition readiness is not assumed to be true.
            return command_level if start_time >= scenario["ready"] - 1e-9 else 0.0

        def advance_actuator(angle, target_angle, duration, tau=0.06):
            rate = np.deg2rad(60.0)
            difference = target_angle - angle
            linear_time = min(duration, max(0.0, (abs(difference) - rate * tau) / rate))
            angle += np.sign(difference) * rate * linear_time
            return target_angle + (angle - target_angle) * exp(-(duration - linear_time) / tau)

        # 핀도 지연 측정이다. 측정 이후 실제로 전달했던 명령만 전파한다.
        if self.observation_times:
            fin_time = self.observation_times[0]
            if self.observations:
                pending_slow_level = float(self.observations[0][12])
            for end, command in self.navigation_command_history:
                if end <= fin_time + 1e-12:
                    continue
                duration = min(end, self.time_s) - fin_time
                if duration <= 0:
                    continue
                allocation_fin = advance_actuator(
                    allocation_fin, command[3], duration, FINS.actuator_time_constant_s
                )
                pending_slow_level = command[0] + (pending_slow_level - command[0]) * exp(
                    -duration / 0.05
                )
                fin_time += duration
        engine_scenarios = [
            dict(
                tau=tau,
                level=level,
                ready=self.ignition_ready_s,
                requested=self.ignition_requested,
            )
            for tau, level in ((0.04, pending_level), (0.05, pending_slow_level))
        ]
        for pending in pending_commands:
            for scenario in engine_scenarios:
                level = eligible_throttle(scenario, pending[0], allocation_time)
                scenario["level"] = level + (scenario["level"] - level) * exp(
                    -limits.control_dt_s / scenario["tau"]
                )
            allocation_gimbal = advance_actuator(allocation_gimbal, pending[1], limits.control_dt_s)
            allocation_fin = advance_actuator(
                allocation_fin, pending[3], limits.control_dt_s, FINS.actuator_time_constant_s
            )
            allocation_time += limits.control_dt_s
        # Apply ignition eligibility before sizing TVC as well as RCS. A
        # fictitious future flame prepositions the gimbal during the wait,
        # then adds its torque to the held RCS command when ignition begins.
        for scenario in engine_scenarios:
            scenario["thrust"] = self.engine(scenario["level"], observed[1])[0]
            effective_level = eligible_throttle(scenario, throttle, allocation_time)
            scenario["command_thrust"] = self.engine(effective_level, observed[1])[0]
            scenario["future_thrust"] = scenario["command_thrust"] + (
                scenario["thrust"] - scenario["command_thrust"]
            ) * exp(-limits.control_dt_s / scenario["tau"])
        midpoint = limits.control_dt_s / 2.0
        mid_fin = advance_actuator(
            allocation_fin, fin_target, midpoint, FINS.actuator_time_constant_s
        )
        requested = requested_raw - self.fin_loads(observed, mid_fin, predicted=True)[1]
        # 점화 직후 작은 현재 추력으로 큰 짐벌각을 요구하지 않는다.
        sizing_thrust = max(
            max(scenario["thrust"], scenario["future_thrust"]) for scenario in engine_scenarios
        )
        gimbal = (
            float(
                np.clip(
                    np.arcsin(np.clip(-requested / max(lever * sizing_thrust, 1), -1, 1)),
                    -limit,
                    limit,
                )
            )
            if sizing_thrust > 1
            else 0.0
        )
        if thrust_direction is not None and sizing_thrust > 1:
            lower = max(
                -limit,
                np.arcsin(
                    np.clip(
                        (-self.design.rcs_torque_limit_nm - requested) / (lever * sizing_thrust),
                        -1,
                        1,
                    )
                ),
            )
            upper = min(
                limit,
                np.arcsin(
                    np.clip(
                        (self.design.rcs_torque_limit_nm - requested) / (lever * sizing_thrust),
                        -1,
                        1,
                    )
                ),
            )
            if lower <= upper:
                gimbal = float(np.clip(thrust_direction - observed[4], lower, upper))
        # 한 명령을 유지하는 구간의 시작·중간·끝에서 핀과 TVC를 함께 배분한다.
        # 40/50 ms 엔진 응답 후보의 극값 중심은 고정 RCS의 최대 토크 오차를 줄인다.
        supplied = []
        for instant in (0.0, midpoint, limits.control_dt_s):
            fin_angle = advance_actuator(
                allocation_fin, fin_target, instant, FINS.actuator_time_constant_s
            )
            fin_moment = self.fin_loads(observed, fin_angle, predicted=True)[1]
            gimbal_angle = advance_actuator(allocation_gimbal, gimbal, instant)
            for scenario in engine_scenarios:
                interval_thrust = scenario["command_thrust"] + (
                    scenario["thrust"] - scenario["command_thrust"]
                ) * exp(-instant / scenario["tau"])
                supplied.append(-lever * interval_thrust * sin(gimbal_angle) + fin_moment)
        supplied_midrange = (min(supplied) + max(supplied)) / 2.0
        rcs_limit = self.design.rcs_torque_limit_nm
        background = (
            body_load[1] + angular_flux - inertia_rate * observed[5] + self.angular_torque_bias_nm
        )
        # 잔여 화염의 점화 경계가 불확실해도 두 후보 모두 안전한 고정 RCS를 찾는다.
        # 4°/s²는 제어 여유이며 실제 비행 판정 5°/s²를 바꾸지 않는다.
        guard_torque = acceleration_inertia * np.deg2rad(4.0)
        lower_rcs = max(-rcs_limit, -guard_torque - background - min(supplied))
        upper_rcs = min(rcs_limit, guard_torque - background - max(supplied))
        if lower_rcs > upper_rcs:
            lower_rcs, upper_rcs = -rcs_limit, rcs_limit
        rcs = (
            float(
                np.clip(
                    requested_raw - supplied_midrange,
                    lower_rcs,
                    upper_rcs,
                )
            )
            if observed[7] > 0
            else 0.0
        )
        if self.upper_attached and self.cutoff_pending:
            # A queued pre-separation command must also be safe for the lighter
            # booster. This preserves physical action delay instead of flushing it.
            lower_mass = self.mass(observed, predicted=True) - self.design.upper_mass_kg
            lower_props = mass_properties(
                lower_mass, observed[8], self.settings(attached=False, predicted=True)
            )
            rcs_limit = lower_props.inertia_kg_m2 * np.deg2rad(2.0)
            rcs = float(np.clip(rcs, -rcs_limit, rcs_limit))
        capacity = lever * sizing_thrust * sin(limit) + self.design.rcs_torque_limit_nm
        self.minimum_control_torque_margin_nm = min(
            self.minimum_control_torque_margin_nm, capacity - abs(requested)
        )
        return gimbal, rcs, fin_target

    def force_control(self, acceleration, observed, *, vertical_body=False):
        aero, _, q, _ = self.loads(observed, predicted=True)
        mass = self.mass(observed, predicted=True)
        proper = np.asarray(acceleration, dtype=float) + [0, gravity(observed[1])]
        if (
            vertical_body
            and self.controller_options.translation_bias_compensation
            and self.navigation_estimator.estimate is not None
        ):
            settings = self.return_settings
            height = self.height(observed, predicted=True)
            blend = _smoothstep(
                (settings.acceleration_bias_start_height_m - height)
                / (
                    settings.acceleration_bias_start_height_m
                    - settings.acceleration_bias_full_height_m
                )
            )
            # The observer learns force/model mismatch from past measurements.
            # It does not receive true mass, thrust error, gust or fluid state.
            bias = self.navigation_estimator.estimate[:, 2].copy()
            norm = hypot(*bias)
            if norm > settings.acceleration_bias_limit_m_s2:
                bias *= settings.acceleration_bias_limit_m_s2 / norm
            proper -= blend * bias
        if self.physics.atmosphere.coriolis_enabled:
            omega = 7.292115e-5 * cos(np.deg2rad(self.conditions.latitude_deg))
            proper -= 2 * omega * np.array([-observed[3], observed[2]])
        magnitude = hypot(*proper)
        if magnitude > 4.3 * G0:
            proper *= 4.3 * G0 / magnitude
        force = mass * proper - aero
        force[1] = max(0.0, force[1])
        if vertical_body:
            lever = self.properties(observed, predicted=True).cg_m - (
                self.physics.geometry.nozzle_station_m
            )
            # 수직 몸체를 유지할 때 수평 추력의 토크를 RCS로 상쇄할 여유를 둔다.
            lateral_limit = 0.55 * self.design.rcs_torque_limit_nm / lever
            force[0] = float(np.clip(force[0], -lateral_limit, lateral_limit))
        direction = atan2(force[0], force[1])
        limits = self.return_settings
        if (
            not self.upper_attached
            and not vertical_body
            and observed[3] < 0
            and q >= limits.entry_attitude_cone_min_q_pa
        ):
            # High-q horizontal correction must not demand a broadside body
            # attitude whose aerodynamic torque exceeds TVC/RCS authority.
            # Preserve the vertical braking force; defer the unattainable
            # lateral part instead of reducing thrust until control is lost.
            air = self.air_velocity(observed, predicted=True)
            flow_axis = atan2(-air[0], -air[1])
            error = atan2(sin(direction - flow_axis), cos(direction - flow_axis))
            cone = np.deg2rad(limits.entry_attitude_cone_deg)
            direction = flow_axis + float(np.clip(error, -cone, cone))
            force[0] = force[1] * np.tan(direction)
        magnitude = hypot(*force)
        full = self.engine(1, observed[1])[0]
        throttle = float(np.clip(magnitude / full, 0, 1))
        target = 0.0 if vertical_body else direction
        alignment = abs(atan2(sin(direction - observed[4]), cos(direction - observed[4])))
        if not vertical_body and alignment > np.deg2rad(15):
            throttle = min(throttle, 0.05)
        if not self.upper_attached and not vertical_body and q > 100.0:
            properties = self.properties(observed, predicted=True)
            lever = properties.cg_m - self.physics.geometry.nozzle_station_m
            torque_budget = 1.1 * 1.15 * abs(
                self.body_loads(observed, predicted=True)[1]
            ) + properties.inertia_kg_m2 * np.deg2rad(limits.attitude_acceleration_limit_deg_s2)
            required_tvc = max(0.0, torque_budget - 0.8 * self.design.rcs_torque_limit_nm)
            minimum_throttle = required_tvc / (
                0.97 * lever * full * sin(np.deg2rad(self.design.gimbal_limit_deg))
            )
            # Misalignment is not a reason to turn off the actuator needed
            # to recover attitude. This is guidance reserve, not a relaxed
            # physical angular-acceleration or flight-load criterion.
            throttle = float(np.clip(max(throttle, minimum_throttle), 0, 1))
        gimbal, rcs, fin = self.attitude_control(
            target, throttle, observed, thrust_direction=direction if vertical_body else None
        )
        return throttle, gimbal, rcs, fin

    def descent_speed(self, height):
        s = self.return_settings
        heights, squared, times = build_descent_reference(
            self.settings(predicted=True).atmosphere,
            s.entry_q_fraction * SAFETY.max_q_pa,
            s.entry_braking_m_s2,
            s.terminal_braking_m_s2,
        )
        physical_height = float(np.clip(height, 0, PROFILE_TOP_M))
        plateau = s.terminal_speed_plateau_height_m
        height = max(0.0, physical_height - plateau)
        index = min(max(int(np.searchsorted(heights, height)) - 1, 0), len(heights) - 2)
        slope = (squared[index + 1] - squared[index]) / (heights[index + 1] - heights[index])
        value = squared[index] + slope * (height - heights[index])
        speed = sqrt(value)
        time = times[index] + 2 * (height - heights[index]) / (speed + sqrt(squared[index]))
        # Complete braking before touchdown so CG/altimetry error does not
        # turn a small positive height error into an excessive contact speed.
        gradient = 0.0 if physical_height <= plateau else slope / (2 * speed)
        time += min(physical_height, plateau) / self.design.descent_reference_m_s
        return speed, gradient, float(time)

    def coast_attitude(self, observed):
        velocity = self.air_velocity(observed, predicted=True)
        # 원지점의 vz=0에서 참조 자세가 180도 뒤집히는 것을 피한다.
        return atan2(-velocity[0], max(-velocity[1], 300.0))

    def predict_return(self, burn_duration_s, observed=None, *, direction=None):
        """유한 연소·엔진 응답·회전·공력·밀도 제동을 포함한 유도용 예측.

        회수 참조를 계산하는 질점+유한 회전 근사다. 실제 강체 검증이나
        접지 성공을 대신하지 않으며, 이 예측의 상태를 실제 상태에 복사하지 않는다.
        """
        observed = self.observed_state if observed is None else observed
        direction = self.boost_direction if direction is None else direction
        y = observed[:4].copy()
        mass = self.mass(observed, predicted=True)
        theta, rate, throttle = float(observed[4]), float(observed[5]), float(observed[12])
        atmosphere_settings = self.settings(predicted=True).atmosphere
        s = self.return_settings
        elapsed, peak_q, peak_height = 0.0, 0.0, float(y[1])
        q = 0.0
        omega = 7.292115e-5 * cos(np.deg2rad(self.conditions.latitude_deg))
        axis_area = pi * self.physics.geometry.body_radius_m**2
        normal_area = 2 * self.physics.geometry.body_radius_m * self.physics.geometry.body_length_m
        while y[1] > s.landing_entry_height_m and elapsed < 500:
            temperature, pressure, rho = atmosphere(y[1], atmosphere_settings)
            vx = y[2] - self.conditions.wind(y[1], self.time_s + elapsed, forecast=True)
            vz = y[3]
            speed = hypot(vx, vz)
            q = 0.5 * rho * speed**2
            peak_q = max(peak_q, q)
            peak_height = max(peak_height, float(y[1]))
            mach = speed / sqrt(1.4 * R_AIR * temperature)
            correction = 1 + PHYSICS.aerodynamics.transonic_drag_rise * exp(
                -(((mach - 1) / 0.2) ** 2)
            )
            axial = sin(theta) * vx + cos(theta) * vz
            normal = cos(theta) * vx - sin(theta) * vz
            aero = (
                -0.5
                * rho
                * correction
                * (
                    PHYSICS.aerodynamics.axial_cd
                    * axis_area
                    * abs(axial)
                    * axial
                    * np.array([sin(theta), cos(theta)])
                    + PHYSICS.aerodynamics.normal_cd
                    * normal_area
                    * abs(normal)
                    * normal
                    * np.array([cos(theta), -sin(theta)])
                )
            )
            if (
                self.fin_mode == "active"
                and 500 < y[1] < 20_000
                and vz < -s.fin_min_descent_speed_m_s
            ):
                aero -= (
                    q
                    * FINS.planar_effective_area_m2
                    * FINS.base_drag_coefficient
                    * (np.array([vx, vz]) / max(speed, 1))
                )
            coriolis = (
                2 * omega * np.array([-y[3], y[2]])
                if (self.physics.atmosphere.coriolis_enabled)
                else np.zeros(2)
            )
            full = self.design.sea_level_thrust_n + self.design.nozzle_exit_area_m2 * (
                101325 - pressure
            )
            boosting = elapsed < burn_duration_s - 1e-9
            if boosting:
                desired_force = (
                    mass * (np.array([direction * s.boost_acceleration_m_s2, 0.0]) - coriolis)
                    - aero
                )
                desired_force[1] = max(0, desired_force[1])
                target = atan2(desired_force[0], desired_force[1])
                command = min(1, hypot(*desired_force) / full)
            else:
                target = atan2(-vx, max(-vz, 300.0))
                command = 0.0
                if vz < 0 and speed > 1:
                    reference, gradient, _ = self.descent_speed(y[1])
                    desired_rate = gradient * vz + 0.9 * (reference - speed)
                    free = (aero / mass + coriolis - [0, gravity(y[1])]) @ [vx, vz] / speed
                    proper = max(0.0, free - desired_rate)
                    if proper > 0:
                        target = atan2(-vx, -vz)
                        proper = min(proper, max(0, 4.2 * G0 - hypot(*aero) / mass))
                        command = min(1, mass * proper / full)
            dt = min(s.forecast_step_s, 500 - elapsed)
            if boosting:
                dt = min(dt, burn_duration_s - elapsed)
            error = atan2(sin(target - theta), cos(target - theta))
            alpha_limit = np.deg2rad(s.attitude_acceleration_limit_deg_s2)
            desired_rate = np.sign(error) * min(
                np.deg2rad(s.attitude_rate_limit_deg_s),
                1.5 * abs(error),
                sqrt(2 * 0.8 * alpha_limit * abs(error)),
            )
            alpha = float(np.clip(4 * (desired_rate - rate), -alpha_limit, alpha_limit))
            if abs(error) > np.deg2rad(15):
                command = min(command, 0.05)
            decay = exp(-dt / 0.04)
            average_throttle = command + (throttle - command) * 0.04 / dt * (1 - decay)
            throttle = command + (throttle - command) * decay
            flow = (
                average_throttle
                * self.design.sea_level_thrust_n
                / (self.design.sea_level_isp_s * G0)
            )
            middle_mass = mass - 0.5 * flow * dt
            middle_theta = theta + rate * dt / 2 + alpha * dt**2 / 8
            force = aero + average_throttle * full * np.array(
                [sin(middle_theta), cos(middle_theta)]
            )
            acceleration = force / max(middle_mass, self.nominal_design.dry_mass_kg) + coriolis
            acceleration[1] -= gravity(y[1])
            y[:2] += y[2:] * dt + acceleration * dt**2 / 2
            y[2:] += acceleration * dt
            theta += rate * dt + alpha * dt**2 / 2
            rate += alpha * dt
            mass -= flow * dt
            elapsed += dt
            if mass <= self.nominal_design.dry_mass_kg + observed[7]:
                break
        height = max(y[1], 0.0)
        # 마지막 제동 구간의 수평 이동도 포함한다. 실제 접지는 별도로 적분한다.
        landing_x = float(y[0] + y[2] * height / max(-y[3], 10.0))
        remaining = mass - self.nominal_design.dry_mass_kg - observed[7]
        landing_time = self.descent_speed(height)[2]
        # 공력의 감속 이득을 제외한 마지막 연소 예산이다.
        landing_delta_v = hypot(y[2], y[3]) + G0 * landing_time
        landing_propellant = mass * (1 - exp(-landing_delta_v / (self.design.sea_level_isp_s * G0)))
        return dict(
            landing_x_m=landing_x,
            terminal_state=y.tolist(),
            elapsed_s=elapsed,
            remaining_main_kg=float(remaining),
            peak_dynamic_pressure_pa=float(peak_q),
            peak_altitude_m=peak_height,
            estimated_landing_propellant_kg=float(landing_propellant),
            estimated_main_after_landing_kg=float(remaining - landing_propellant),
            feasible=bool(
                y[1] <= s.landing_entry_height_m
                and remaining > landing_propellant + 200
                and peak_q <= SAFETY.max_q_pa
                and peak_height <= PROFILE_TOP_M
            ),
        )

    def plan_return(self, observed=None):
        observed = self.observed_state if observed is None else observed
        cache = {}

        def evaluate(duration):
            key = float(duration)
            if key not in cache:
                cache[key] = self.predict_return(key, observed)
            return cache[key]

        free = evaluate(0.0)
        miss = free["landing_x_m"]
        if abs(miss) <= self.return_settings.predicted_miss_limit_m:
            duration = 0.0
        else:
            root = None
            for duration_limit in (8.0, 16.0, 24.0, 32.0, 40.0, 48.0):
                endpoint = evaluate(duration_limit)
                if miss * endpoint["landing_x_m"] <= 0:
                    root = brentq(
                        lambda time: evaluate(time)["landing_x_m"],
                        0,
                        duration_limit,
                        xtol=0.005,
                        maxiter=32,
                    )
                    break
            if root is None:
                self.return_plan = dict(
                    time_s=self.time_s,
                    feasible=False,
                    reason="no_bracketed_return_solution",
                    predicted_ground_range_m=miss,
                )
                self.outcome = "return_plan_infeasible"
                return
            duration = float(root)
        prediction = evaluate(duration)
        plan = dict(
            time_s=self.time_s,
            remaining_burn_s=duration,
            burn_end_time_s=self.time_s + duration,
            predicted_ground_range_m=prediction["landing_x_m"],
            predicted_main_at_terminal_kg=prediction["remaining_main_kg"],
            predicted_peak_q_pa=prediction["peak_dynamic_pressure_pa"],
            prediction_evaluations=len(cache),
            feasible=prediction["feasible"],
            source="finite_burn_predictor_corrector",
            touchdown_prediction_is_not_acceptance=True,
        )
        self.return_plan = plan
        self.return_plan_history.append(plan.copy())
        if not prediction["feasible"]:
            self.outcome = "return_plan_infeasible"
            return
        self.boost_end_s = self.time_s + duration
        self.next_plan_s = self.time_s + self.return_settings.boost_replan_period_s

    def navigation_acceleration(self, measured, measurement_time_s):
        """Nominal inertial acceleration from delayed measurements and forecast.

        No plant derivative, actual wind, thrust scale or fluid-mode state is
        consulted. Unmodelled forces are estimated through the filter's bias.
        Pad support is a known discrete flight mode, not a hidden force input.
        """
        nominal = measured.copy()
        nominal[17:19] = 0.0
        force = self.loads(nominal, predicted=True, time_s=measurement_time_s)[0]
        thrust = self.engine(float(np.clip(nominal[12], 0, 1)), nominal[1])[0]
        force += thrust * np.array([sin(nominal[4] + nominal[11]), cos(nominal[4] + nominal[11])])
        mass = self.mass(nominal, predicted=True)
        local_g = gravity(nominal[1])
        if self.on_pad and self.upper_attached and force[1] < mass * local_g:
            return np.zeros(2)
        acceleration = force / mass - np.array([0.0, local_g])
        if self.physics.atmosphere.coriolis_enabled:
            omega = 7.292115e-5 * cos(np.deg2rad(self.conditions.latitude_deg))
            acceleration += 2 * omega * np.array([-nominal[3], nominal[2]])
        return acceleration

    def predict_navigation_actuators(self, measured, measurement_time_s):
        """Propagate delayed throttle/gimbal telemetry through known commands.

        Command entries describe the interval ending at their timestamp. The
        engine uses its nominal 40 ms response, not the hidden lag deviation.
        The gimbal prediction respects the nominal 60 degree/s actuator rate.
        """
        throttle, gimbal = float(measured[12]), float(measured[11])
        begin = measurement_time_s
        rate, tau = np.deg2rad(60.0), 0.06
        for end, command in self.navigation_command_history:
            if end <= begin + 1e-12:
                continue
            duration = min(end, self.time_s) - begin
            if duration <= 0:
                continue
            throttle = command[0] + (throttle - command[0]) * exp(-duration / 0.04)
            difference = command[1] - gimbal
            linear_time = min(duration, max(0.0, (abs(difference) - rate * tau) / rate))
            gimbal += np.sign(difference) * rate * linear_time
            gimbal = command[1] + (gimbal - command[1]) * exp(-(duration - linear_time) / tau)
            begin += duration
        return float(gimbal), float(np.clip(throttle, 0, 1))

    def observe_angular_torque(self, measured, measurement_time_s):
        """Estimate torque mismatch from delayed angular momentum telemetry.

        RCS history entries describe executed intervals ending at their timestamp.
        Only nominal properties, forecast loads and measured actuator states are
        used. Staging impulses and pad support are not continuous disturbances.
        """
        if not self.controller_options.torque_bias_compensation:
            self.angular_torque_bias_nm = 0.0
            return
        previous = self.angular_observation
        if (
            self.upper_attached
            or self.on_pad
            or (
                self.separation is not None
                and measurement_time_s < self.separation["time_s"] - 1e-9
            )
        ):
            self.angular_observation = None
            self.angular_torque_bias_nm = 0.0
            return
        if previous is not None and measurement_time_s <= previous[0] + 1e-9:
            return
        self.angular_observation = (measurement_time_s, measured.copy())
        if previous is None:
            return
        begin, old = previous
        duration = measurement_time_s - begin
        midpoint = (old + measured) / 2.0
        midpoint[4] = old[4] + 0.5 * atan2(sin(measured[4] - old[4]), cos(measured[4] - old[4]))
        midpoint[17:19] = 0.0
        rcs_integral = rcs_absolute_integral = 0.0
        interval_begin = begin
        for end, command in self.navigation_command_history:
            if end <= interval_begin + 1e-9:
                continue
            if end - self.return_settings.control_dt_s > interval_begin + 1e-9:
                break
            overlap = min(end, measurement_time_s) - interval_begin
            if overlap <= 0.0:
                continue
            rcs_integral += overlap * command[2]
            rcs_absolute_integral += overlap * abs(command[2])
            interval_begin += overlap
            if interval_begin >= measurement_time_s - 1e-9:
                break
        if interval_begin < measurement_time_s - 1e-9:
            # An incomplete command window cannot identify the torque residual.
            self.angular_torque_bias_nm = 0.0
            return
        props = self.properties(midpoint, predicted=True)
        lever = props.cg_m - self.physics.geometry.nozzle_station_m
        thrust, main_flow = self.engine(midpoint[12], midpoint[1])
        rcs_flow = rcs_absolute_integral / (
            duration * self.physics.geometry.rcs_arm_m * self.design.rcs_isp_s * G0
        )
        flux = -main_flow * lever**2 * midpoint[5]
        flux -= (
            rcs_flow
            * (
                self.physics.geometry.rcs_arm_m**2
                + (self.physics.geometry.rcs_station_m - props.cg_m) ** 2
            )
            * midpoint[5]
        )
        nominal_torque = self.loads(
            midpoint, predicted=True, time_s=(begin + measurement_time_s) / 2.0
        )[1]
        nominal_torque += -lever * thrust * sin(midpoint[11]) + rcs_integral / duration + flux
        old_momentum = self.properties(old, predicted=True).inertia_kg_m2 * old[5]
        momentum = self.properties(measured, predicted=True).inertia_kg_m2 * measured[5]
        residual = (momentum - old_momentum) / duration - nominal_torque
        bound = props.inertia_kg_m2 * np.deg2rad(4.0)
        residual = float(np.clip(residual, -bound, bound))
        blend = 1.0 - exp(-duration / 0.12)
        self.angular_torque_bias_nm += blend * (residual - self.angular_torque_bias_nm)
        self.angular_torque_bias_nm = float(np.clip(self.angular_torque_bias_nm, -bound, bound))

    def observe_control_state(self):
        raw = self.state.copy()
        # Fluid mode coordinates are plant states, not ideal measurements
        # secretly supplied to the guidance law.
        raw[17:19] = 0.0
        # 위치/속도 오차는 매 제어 시각에만 발생한다. 적분 호출 수와 무관하다.
        errors = self.disturbances
        if errors.sensor_position_error_m > 0:
            error = self.rng.normal(size=2)
            error /= max(hypot(*error), 1e-12)
            raw[:2] += error * sqrt(self.rng.random()) * errors.sensor_position_error_m
        if errors.sensor_velocity_error_m_s > 0:
            error = self.rng.normal(size=2)
            error /= max(hypot(*error), 1e-12)
            raw[2:4] += error * sqrt(self.rng.random()) * errors.sensor_velocity_error_m_s
        sensors = self.sensor_settings
        if sensors.angular_noise_enabled:
            raw[4] += np.deg2rad(
                sensors.attitude_bias_deg
                + self.sensor_rng.uniform(-sensors.attitude_error_deg, sensors.attitude_error_deg)
            )
            raw[5] += np.deg2rad(
                sensors.gyro_bias_deg_s
                + self.sensor_rng.uniform(-sensors.gyro_error_deg_s, sensors.gyro_error_deg_s)
            )
        drop = (
            sensors.dropout_period_steps > 0
            and self.sensor_sample_count > 0
            and self.sensor_sample_count % sensors.dropout_period_steps
            < sensors.dropout_length_steps
        )
        self.sensor_sample_count += 1
        if drop:
            raw, sample_time = self.last_sensor_packet
            self.sensor_dropout_count += 1
        else:
            sample_time = self.time_s
            self.last_sensor_packet = (raw.copy(), sample_time)
        self.observations.append(raw.copy())
        self.observation_times.append(sample_time)
        self.navigation_command_history.append((self.time_s, self.control))
        measured = self.observations[0]
        measurement_time = self.observation_times[0]
        if sensors.angular_noise_enabled and self.controller_options.attitude_filter:
            measured = self.attitude_estimator.observe(measured, measurement_time)
        self.observe_angular_torque(measured, measurement_time)
        delay = max(0.0, self.time_s - measurement_time)
        acceleration = self.navigation_acceleration(measured, measurement_time)
        observed = self.navigation_estimator.observe(measured, self.time_s, acceleration, delay)
        observed[4] += observed[5] * delay
        observed[11], observed[12] = self.predict_navigation_actuators(measured, measurement_time)
        self.observed_state = observed
        return observed

    def guidance(self, observed=None):
        y = self.state if observed is None else observed
        h = self.height(y, predicted=True)
        forecast_atmosphere = self.settings(predicted=True).atmosphere
        q = self.loads(y, predicted=True)[2]
        mass = self.mass(y, predicted=True)
        aero = self.loads(y, predicted=True)[0]
        velocity = self.air_velocity(y, predicted=True)
        if self.contact is not None:
            return 0.0, 0.0, 0.0, 0.0
        if self.upper_attached:
            s = self.ascent_settings
            # The observed tank mass can be old, a shutdown command can queue,
            # and the engine keeps flowing while its throttle decays. Budget
            # full flow over those known delays and the allowed 0.05 s lag.
            shutdown_delay_s = (
                self.disturbances.observation_delay_steps + self.disturbances.action_delay_steps + 1
            ) * self.return_settings.control_dt_s + 0.05
            shutdown_reserve_kg = self.engine(1.0, y[1])[1] * shutdown_delay_s + 10.0
            if (
                y[6] <= s.separation_min_main_propellant_kg + shutdown_reserve_kg
                or y[1] >= s.separation_altitude_m
            ):
                self.cutoff_pending = True
            programmed_pitch = np.deg2rad(s.final_pitch_deg) * _smoothstep(
                (self.time_s - s.pitch_start_s) / s.pitch_duration_s
            )
            # Build altitude before turning over. A large time-only pitch
            # program can consume the vertical velocity needed for 37 km.
            altitude_pitch_cap = np.deg2rad(25.0) * _smoothstep(
                max(h, 0.0) / s.separation_altitude_m
            )
            pitch = min(programmed_pitch, altitude_pitch_cap)
            flow = atan2(velocity[0], velocity[1]) if hypot(*velocity) > 20 else 0.0
            if q > 100:
                # At liftoff, the horizontal wind dominates the air-relative
                # direction before the rocket has a meaningful upward speed.
                # Bring the flow reference in gradually during vertical departure.
                departure = _smoothstep((y[3] - 40.0) / 60.0)
                protected_flow = departure * flow
                # Low q permits more steering to recover a vertical path.
                # Near max q, leave margin under both the 5-degree and
                # Q-alpha acceptance limits for finite attitude tracking.
                angle_budget_deg = 10.0 - 6.5 * _smoothstep((q - 5_000.0) / 5_000.0)
                allowance = np.deg2rad(
                    min(angle_budget_deg, 0.6 * SAFETY.max_q_alpha_kpa_deg * 1000 / q)
                )
                pitch = float(
                    np.clip(pitch, protected_flow - allowance, protected_flow + allowance)
                )
            # Align while TVC is still available, not after shutting down at
            # the gate. At roughly 1 km/s, the altitude window is shorter
            # than the finite-time turn from the ascent pitch to the flow.
            time_to_altitude = (s.separation_altitude_m - y[1]) / max(y[3], 1.0)
            full_flow = self.engine(1.0, y[1])[1]
            time_to_reserve = (
                y[6] - s.separation_min_main_propellant_kg - shutdown_reserve_kg
            ) / full_flow
            angle_error = abs(atan2(sin(y[4] - flow), cos(y[4] - flow)))
            alpha_limit = np.deg2rad(self.return_settings.attitude_acceleration_limit_deg_s2)
            rate_limit = np.deg2rad(self.return_settings.attitude_rate_limit_deg_s)
            slew_time = (
                max(
                    2 * sqrt(angle_error / alpha_limit),
                    angle_error / rate_limit + rate_limit / alpha_limit,
                )
                + abs(y[5]) / alpha_limit
            )
            preview = max(s.separation_alignment_preview_s, slew_time + 2.0)
            if (
                self.separation_alignment_started_s is None
                and y[3] > 100.0
                and min(time_to_altitude, time_to_reserve) <= preview
            ):
                self.separation_alignment_started_s = self.time_s
            if self.separation_alignment_started_s is not None:
                blend = _smoothstep(
                    (self.time_s - self.separation_alignment_started_s)
                    / s.separation_alignment_blend_s
                )
                difference = atan2(sin(flow - pitch), cos(flow - pitch))
                pitch += blend * difference
            if self.cutoff_pending:
                pitch = flow
                throttle = 0.0
            else:
                full = self.engine(1, y[1])[0]
                minimum_mass = mass - 0.01 * self.nominal_design.launch_mass_kg
                thrust = min(full, 2.95 * G0 * minimum_mass / 1.03)
                speed = hypot(*velocity)
                if q > 0.75 * SAFETY.max_q_pa and speed > 10:
                    rho = atmosphere(y[1], forecast_atmosphere)[2]
                    gradient = (
                        np.log(atmosphere(y[1] + 20, forecast_atmosphere)[2]) - np.log(rho)
                    ) / 20
                    # Density scale is hidden from guidance (up to +10%). Keep
                    # additional headroom for an unforecast gust and throttle lag.
                    robust_q = 0.94 * SAFETY.max_q_pa / (1.0 + MAX_DENSITY_SCALE_DEVIATION)
                    cap = sqrt(2 * robust_q / rho)
                    desired = -0.5 * cap * gradient * y[3] + 0.7 * (cap - speed)
                    free = float((aero / mass - [0, gravity(y[1])]) @ velocity) / speed
                    projection = (sin(pitch) * velocity[0] + cos(pitch) * velocity[1]) / speed
                    if projection > 0.05:
                        thrust = min(thrust, max(0, mass * (desired - free) / projection))
                throttle = thrust / full
            gimbal, rcs, fin = self.attitude_control(pitch, throttle, y)
            return throttle, gimbal, rcs, fin
        limits = self.return_settings
        if self.phase == "flip":
            if self.flip_started_s is None:
                if q > limits.flip_max_q_pa:
                    target = atan2(velocity[0], velocity[1]) if y[3] > 0 else self.coast_attitude(y)
                    gimbal, rcs, fin = self.attitude_control(target, 0.0, y)
                    return 0.0, gimbal, rcs, fin
                self.flip_started_s = self.time_s
                ground_time = (max(y[3], 0) + sqrt(y[3] ** 2 + 2 * gravity(y[1]) * y[1])) / gravity(
                    y[1]
                )
                self.boost_direction = -1.0 if y[0] + y[2] * ground_time >= 0 else 1.0
            target = self.boost_direction * pi / 2
            error = atan2(sin(target - y[4]), cos(target - y[4]))
            if abs(error) < np.deg2rad(0.25) and abs(y[5]) < np.deg2rad(0.1):
                self.phase = "boostback"
                self.plan_return(y)
            else:
                # 낮은 동압에서 RCS로 회전한다. 자세 전환 중 불필요한 병진 추력을 피한다.
                gimbal, rcs, fin = self.attitude_control(target, 0.0, y)
                return 0.0, gimbal, rcs, fin
        if self.outcome != "running":
            return 0.0, 0.0, 0.0, 0.0
        if self.phase == "boostback":
            if self.time_s >= self.boost_end_s - 1e-8:
                self.phase = "coast"
                self.boostback_completed = dict(time_s=self.time_s, state=self.state.tolist())
            else:
                if (
                    self.controller_options.boost_replanning
                    and self.time_s >= self.next_plan_s
                    and self.boost_end_s - self.time_s > 0.5
                ):
                    self.plan_return(y)
                if self.outcome != "running":
                    return 0.0, 0.0, 0.0, 0.0
                if self.time_s < self.boost_end_s:
                    acceleration = np.array(
                        [self.boost_direction * limits.boost_acceleration_m_s2, -gravity(y[1])]
                    )
                    return self.force_control(acceleration, y)
                self.phase = "coast"
                self.boostback_completed = dict(time_s=self.time_s, state=self.state.tolist())
        if y[3] >= 0:
            gimbal, rcs, fin = self.attitude_control(self.coast_attitude(y), 0.0, y)
            return 0.0, gimbal, rcs, fin

        speed, gradient, time_to_ground = self.descent_speed(h)
        alignment_time = self.descent_speed(limits.horizontal_alignment_height_m)[2]
        horizontal_time = max(4.0, time_to_ground - alignment_time + 4.0)
        # ZEM/ZEV 형태의 종말 위치·속도 피드백. 수평 정렬 목표는 접지보다 앞선다.
        ax = float(np.clip(-6 * y[0] / horizontal_time**2 - 4 * y[2] / horizontal_time, -8.0, 8.0))
        vertical_body = h < limits.vertical_body_height_m
        if vertical_body:
            lever = self.properties(y, predicted=True).cg_m - self.physics.geometry.nozzle_station_m
            ax_limit = 0.5 * self.design.rcs_torque_limit_nm / (lever * mass)
            # A fixed four-second ZEM horizon can demand acceleration toward
            # the pad even when the limited lateral actuator must already
            # brake. Shape the velocity reference by reachable stopping
            # distance, with 30% braking margin and a linear near-pad limit.
            closing_speed = min(0.4 * abs(y[0]), sqrt(2.0 * 0.7 * ax_limit * abs(y[0])))
            desired_vx = -np.sign(y[0]) * closing_speed
            ax = float(np.clip(1.2 * (desired_vx - y[2]), -ax_limit, ax_limit))
        lateral_speed = velocity[0]
        vertical_speed_squared = speed**2 - lateral_speed**2
        reference_squared = self.design.descent_reference_m_s**2
        vertical_speed = sqrt(max(reference_squared, vertical_speed_squared))
        desired_vz = -vertical_speed
        # The lower speed bound is a constant reference, so its derivative
        # is zero. Differentiating the unclipped expression here couples
        # horizontal wind/correction into an unintended touchdown acceleration.
        feedforward = (
            (-speed * gradient * y[3] + lateral_speed * ax) / vertical_speed
            if vertical_speed_squared > reference_squared
            else 0.0
        )
        alignment_blend = (
            _smoothstep(max(abs(y[0]) / 2.0, abs(y[2]) / 0.5) - 1.0) if vertical_body else 0.0
        )
        if alignment_blend > 0.0:
            # 진입 분산이 남으면 고도를 소진하기 전에 정렬 시간을 확보한다.
            alignment_duration = abs(y[2]) / ax_limit + sqrt(2 * abs(y[0]) / ax_limit) + 4
            allowed = max(self.design.descent_reference_m_s, h / alignment_duration)
            if allowed < vertical_speed:
                desired_vz += alignment_blend * (-allowed - desired_vz)
                alignment_feedforward = max(0, -y[3] / alignment_duration)
                feedforward += alignment_blend * (alignment_feedforward - feedforward)
        az = float(np.clip(feedforward + 1.2 * (desired_vz - y[3]), -gravity(y[1]), 28.0))
        upward_force = mass * (az + gravity(y[1])) - aero[1]
        actual_speed = hypot(*velocity)
        if self.entry_started_s is None and (upward_force <= 0 or actual_speed < 0.95 * speed):
            self.phase = "coast"
            gimbal, rcs, fin = self.attitude_control(self.coast_attitude(y), 0.0, y)
            return 0.0, gimbal, rcs, fin
        if self.entry_started_s is None:
            self.entry_started_s = self.time_s
        self.phase = "landing" if h <= limits.landing_entry_height_m else "entry"
        if upward_force <= 0:
            gimbal, rcs, fin = self.attitude_control(self.coast_attitude(y), 0.0, y)
            return 0.0, gimbal, rcs, fin
        return self.force_control((ax, az), y, vertical_body=vertical_body)

    def separation_checks(self, state):
        s = self.ascent_settings
        return dict(
            altitude=abs(state[1] - s.separation_altitude_m) <= s.separation_tolerance_m,
            dynamic_pressure=self.loads(state)[2] <= s.separation_max_q_pa,
            angular_rate=abs(np.rad2deg(state[5])) <= s.separation_max_rate_deg_s,
            alpha=abs(self.alpha_deg(state)) <= s.separation_max_alpha_deg,
            main_reserve=state[6] >= s.separation_min_main_propellant_kg,
            main_engine_off=state[12] <= 1e-3,
            after_observed_max_q=self.ascent_max_q_passed,
        )

    def consider_separation(self):
        if not self.upper_attached or not self.cutoff_pending:
            return
        checks = self.separation_checks(self.state)
        if not all(checks.values()):
            if self.state[1] > self.ascent_settings.separation_altitude_m + 5_000:
                self.outcome = "separation_gate_missed"
                self.failures.append(self.outcome)
            return
        gate_altitude = float(self.state[1])
        pre_separation = self.state.copy()
        self.upper_attached = False
        props = self.properties(self.state)
        upper_inertia = self.design.upper_mass_kg * (
            self.physics.upper_length_m**2 / 12 + self.physics.geometry.body_radius_m**2 / 4
        )
        staging = separation_event(
            pre_separation[:2],
            pre_separation[2:4],
            pre_separation[4],
            pre_separation[5],
            self.mass(self.state),
            self.design.upper_mass_kg,
            props.cg_m,
            self.physics.upper_station_m,
            props.inertia_kg_m2,
            upper_inertia,
            self.separation_settings,
            main_propellant_kg=max(float(self.state[6]), 0.0),
            slosh_state=SloshState(*self.state[17:19]),
            slosh_station_m=self.physics.geometry.tank_bottom_m + props.fuel_height_m / 2.0,
            slosh_settings=self.slosh_settings,
        )
        self.state[:2] = staging["booster_position_m"]
        self.state[2:4] = staging["booster_velocity_m_s"]
        self.state[5] = staging["booster_angular_rate_rad_s"]
        self.state[18] = staging["booster_slosh_velocity_m_s"]
        self.on_pad = False
        self.phase = "flip"
        # 분리는 관성이 불연속으로 바뀌는 사건이다. 다음 적분 전에 새 기체의
        # 제어를 계산하여 결합체용 RCS 명령을 한 제어 주기 더 유지하지 않는다.
        self.next_control_s = self.time_s
        self.separation = dict(
            time_s=self.time_s,
            state=self.state.tolist(),
            checks_before_separation=checks,
            engine_throttle=float(self.state[12]),
            gate_altitude_before_cg_change_m=gate_altitude,
            first_stage_mass_kg=self.mass(self.state),
            actual_separation_impulse_modelled=True,
            impulse_model=staging,
        )

    def first_contact(self, state, control):
        rate = self.derivative(state, control)
        mass_rate = rate[6] + rate[7]
        points, velocities = foot_kinematics(
            self.vector(state),
            state[8],
            self.settings(),
            mass_rate=mass_rate,
            deployment_rate=rate[8],
        )
        gate = self.physics.acceptance
        criteria = dict(
            position=abs(state[0]) <= gate.max_position_error_m,
            vertical_foot_speed=np.max(abs(velocities[:, 1])) <= gate.max_vertical_foot_speed_m_s,
            horizontal_foot_speed=np.max(abs(velocities[:, 0]))
            <= gate.max_horizontal_contact_speed_m_s,
            tilt=abs(np.rad2deg(state[4])) <= gate.max_contact_tilt_deg,
            angular_rate=abs(np.rad2deg(state[5])) <= gate.max_contact_rate_deg_s,
            main_propellant=state[6] > 0,
            rcs_propellant=state[7] >= 0,
            gear_deployed=state[8] >= 1 - 1e-8,
            feet_on_pad=np.max(abs(points[:, 0])) <= self.physics.gear.pad_half_width_m,
        )
        self.contact = dict(
            time_s=self.time_s,
            angular_torque_bias_nm=self.angular_torque_bias_nm,
            state=state.tolist(),
            observed_control_state=self.observed_state[:6].tolist(),
            navigation_acceleration_bias_m_s2=(
                self.navigation_estimator.estimate[:, 2].tolist()
                if self.navigation_estimator.estimate is not None
                else None
            ),
            foot_positions_m=points.tolist(),
            foot_velocities_m_s=velocities.tolist(),
            criteria={key: bool(value) for key, value in criteria.items()},
        )
        if self.outcome == "running":
            self.outcome = "landed" if all(criteria.values()) else "touchdown_limits_exceeded"

    def monitor(self, state, control):
        if self.conditions.gust_amplitude_m_s > 0.0:
            gust = abs(
                self.conditions.wind(state[1], self.time_s) - self.conditions.mean_wind(state[1])
            )
            exposure = self.integration_gust_exposure
            exposure["maximum_contribution_m_s"] = max(exposure["maximum_contribution_m_s"], gust)
            phases = exposure["maximum_by_phase_m_s"]
            phases[self.phase] = max(phases.get(self.phase, 0.0), gust)
            altitude_error = abs(state[1] - self.conditions.gust_altitude_center_m)
            closest = exposure["closest_descending_altitude_sample"]
            if (
                state[3] < 0.0
                and altitude_error <= self.conditions.gust_altitude_half_width_m
                and (closest is None or altitude_error < closest["altitude_error_m"])
            ):
                exposure["closest_descending_altitude_sample"] = dict(
                    time_s=self.time_s,
                    phase=self.phase,
                    altitude_m=float(state[1]),
                    altitude_error_m=float(altitude_error),
                    gust_contribution_m_s=gust,
                )
            if gust >= 0.1:
                exposure["endpoints_above_0_1_m_s"] += 1
                if exposure["first_endpoint_above_0_1_m_s"] is None:
                    exposure["first_endpoint_above_0_1_m_s"] = self.time_s
                exposure["last_endpoint_above_0_1_m_s"] = self.time_s
                exposure["gust_encounter_observed"] = True
        if not self.on_pad:
            geometry = self.physics.geometry
            cg = self.properties(state).cg_m
            length = geometry.body_length_m + (
                self.physics.upper_length_m if self.upper_attached else 0.0
            )
            body_corners = np.array(
                [
                    [-geometry.body_radius_m, -cg],
                    [geometry.body_radius_m, -cg],
                    [-geometry.body_radius_m, length - cg],
                    [geometry.body_radius_m, length - cg],
                ]
            )
            body_corners += self.settings().body_frame_offset(self.vector(state))[0]
            corners = body_corners @ rotation(state[4]).T + state[:2]
            if np.min(corners[:, 1]) < -0.001:
                self.outcome = "body_ground_collision"
        force, _, q, _ = self.loads(state)
        alpha = abs(self.alpha_deg(state))
        self.max_q = max(self.max_q, q)
        self.max_altitude = max(self.max_altitude, state[1])
        if self.upper_attached:
            self.max_ascent_q = max(self.max_ascent_q, q)
            self.ascent_max_q_passed |= self.max_ascent_q > 5000 and q < 0.9 * self.max_ascent_q
            self.max_q_alpha = max(self.max_q_alpha, q * alpha / 1000)
            if q > self.design.high_q_threshold_pa:
                self.max_ascent_alpha = max(self.max_ascent_alpha, alpha)
        else:
            self.max_return_q = max(self.max_return_q, q)
            self.max_fin_hinge = max(self.max_fin_hinge, self.fin_loads(state)[2])
            if state[10] > 0.01 and self.fin_mode == "active":
                temperature = atmosphere(state[1], self.physics.atmosphere)[0]
                self.max_fin_mach = max(
                    self.max_fin_mach,
                    hypot(*self.air_velocity(state)) / sqrt(1.4 * R_AIR * temperature),
                )
                self.max_fin_alpha = max(self.max_fin_alpha, alpha)
        thrust = self.actual_engine(state)[0]
        supported = force + thrust * np.array(
            [sin(state[4] + state[11]), cos(state[4] + state[11])]
        )
        flight_g = hypot(*supported) / (self.mass(state) * G0)
        self.max_flight_g = max(self.max_flight_g, flight_g)
        if self.upper_attached:
            self.max_ascent_axial_g = max(
                self.max_ascent_axial_g,
                abs(float((rotation(state[4]).T @ supported)[1])) / (self.mass(state) * G0),
            )
        systems = {}
        derivative = self.derivative(state, control, systems_diagnostics=systems)
        demand = self.engine(float(np.clip(state[12], 0, 1)), state[1], actual=True)[1]
        feed = self.feed_status(state, demand)
        self.minimum_feed_buffer_kg = min(self.minimum_feed_buffer_kg, float(state[13]))
        self.minimum_tank_pressure_pa = min(self.minimum_tank_pressure_pa, feed.tank_pressure_pa)
        self.max_feed_shortage_kg_s = max(self.max_feed_shortage_kg_s, feed.demand_shortfall_kg_s)
        self.feed_inventory_valid &= feed.inventory_valid
        # A numerical micro-flow at shutdown is not an engine firing.
        self.feed_supply_valid &= demand < 1e-3 or feed.demand_shortfall_kg_s < 0.01 * demand
        if not self.feed_inventory_valid:
            self.outcome = "feed_inventory_violation"
        elif not self.feed_supply_valid:
            # Emptying regularisation approaches zero inventory smoothly.
            # Stop at loss of usable supply instead of waiting for tiny
            # positive liquid masses to break the modal mass matrix later.
            self.outcome = "propellant_feed_starvation"
        self.max_helium_balance_error_kg = max(
            self.max_helium_balance_error_kg,
            abs(float(state[15] + state[16]) - self.feed_settings.helium_total_mass_kg),
        )
        displacement = slosh_displacement(SloshState(*state[17:19]), self.slosh_settings)[0]
        self.max_slosh_displacement_m = max(
            self.max_slosh_displacement_m, float(np.linalg.norm(displacement))
        )
        self.max_slosh_mode_force_n = max(
            self.max_slosh_mode_force_n, systems["slosh"].modal_constraint_force_n
        )
        self.slosh_model_valid &= systems["slosh"].valid
        liquid_station = (
            self.physics.geometry.tank_bottom_m + self.properties(state).fuel_height_m / 2
        )
        self.slosh_model_valid &= (
            self.physics.geometry.tank_bottom_m - 1e-8
            <= liquid_station + displacement[1]
            <= self.physics.geometry.tank_bottom_m + self.physics.geometry.tank_length_m + 1e-8
            and abs(displacement[0]) <= self.physics.geometry.body_radius_m + 1e-8
        )
        if not self.slosh_model_valid:
            self.outcome = "slosh_model_out_of_domain"
        if self.first_subsystem_failure is None and not (
            self.feed_inventory_valid and self.feed_supply_valid and self.slosh_model_valid
        ):
            self.first_subsystem_failure = dict(
                time_s=self.time_s,
                phase=self.phase,
                feed_inventory_valid=bool(self.feed_inventory_valid),
                feed_supply_valid=bool(self.feed_supply_valid),
                slosh_model_valid=bool(self.slosh_model_valid),
                state=state.tolist(),
            )
        self.max_rate = max(self.max_rate, abs(np.rad2deg(state[5])))
        angular_acceleration = float(np.rad2deg(derivative[5]))
        if abs(angular_acceleration) > self.max_acceleration:
            self.max_acceleration = abs(angular_acceleration)
            self.peak_angular_acceleration = dict(
                time_s=self.time_s,
                angular_torque_bias_nm=self.angular_torque_bias_nm,
                phase=self.phase,
                upper_attached=self.upper_attached,
                acceleration_deg_s2=angular_acceleration,
                state=state.tolist(),
                control=list(control),
                observed_control_state=self.observed_state.tolist(),
                requested_control=list(self.requested_control),
                navigation_acceleration_bias_m_s2=(
                    self.navigation_estimator.estimate[:, 2].tolist()
                    if self.navigation_estimator.estimate is not None
                    else None
                ),
                attitude_target_deg=float(np.rad2deg(self.attitude_target)),
            )
        self.max_gimbal = max(self.max_gimbal, abs(np.rad2deg(state[11])))
        balance = self.design.main_propellant_kg + self.design.rcs_propellant_kg
        balance -= state[6] + state[7] + self.main_consumed + self.rcs_consumed
        self.max_mass_balance_error = max(self.max_mass_balance_error, abs(balance))

    def record(self):
        self.steps.append(
            dict(
                time_s=self.time_s,
                angular_torque_bias_nm=self.angular_torque_bias_nm,
                phase=self.phase,
                state=self.state.tolist(),
                control=list(self.control),
                requested_control=list(self.requested_control),
                observed_control_state=self.observed_state[:6].tolist(),
                navigation_acceleration_bias_m_s2=(
                    self.navigation_estimator.estimate[:, 2].tolist()
                    if self.navigation_estimator.estimate is not None
                    else None
                ),
                engine_throttle=float(self.state[12]),
                mass_kg=self.mass(self.state),
                dynamic_pressure_pa=self.loads(self.state)[2],
                attitude_target_deg=float(np.rad2deg(self.attitude_target)),
                fin_torque_nm=self.fin_loads(self.state)[1],
            )
        )

    def advance(self):
        if self.outcome != "running":
            return
        if self.time_s >= self.next_control_s - 1e-9:
            self.requested_control = self.guidance(self.observe_control_state())
            self.action_queue.append(self.requested_control)
            self.control = self.ignition_control(self.action_queue[0])
            self.next_control_s += self.return_settings.control_dt_s
        if self.time_s >= self.next_record_s - 1e-9:
            self.record()
            self.next_record_s += 0.2
        step = min(
            self.integration_dt_s
            if self.height(self.state) > 2 or self.upper_attached
            else min(self.integration_dt_s, 0.001),
            self.next_control_s - self.time_s,
        )
        if step < 1e-9:
            self.time_s = self.next_control_s
            return
        before = self.state.copy()
        available_propellant = (before[6] > 0, before[7] > 0)

        def rhs(time, value):
            # 소진 직전의 매끄러운 유량으로 사건 시각을 찾는다.
            # 중간 RK 단계에서 유량을 끄면 실제 소진 시각이 왜곡된다.
            return self.derivative(
                value, self.control, time_s=time, available_propellant=available_propellant
            )

        advanced = rk4_step(rhs, self.time_s, before, step)
        if not np.isfinite(advanced).all():
            self.outcome = "numerical_failure"
            return
        # 추진제 소진은 실제 장부로 판정한다. 음의 질량으로 비행을 계속하지 않는다.
        if advanced[6] <= 0 or advanced[7] < 0:
            event_steps = [
                brentq(
                    lambda dt, index=index: (
                        rk4_step(rhs, self.time_s, before, dt)[index] if dt else before[index]
                    ),
                    0,
                    step,
                    xtol=1e-10,
                )
                for index in (6, 7)
                if before[index] > 0 and advanced[index] <= 0
            ]
            step = min(event_steps)
            advanced = rk4_step(rhs, self.time_s, before, step)
            advanced[6:8] = np.maximum(advanced[6:8], 0)
            if advanced[6] < 1e-7:
                advanced[6] = 0
                self.outcome = "main_propellant_depleted"
            if advanced[7] < 1e-7:
                advanced[7] = 0
            self.time_s += step
            self.main_consumed += before[6] - advanced[6]
            self.rcs_consumed += before[7] - advanced[7]
            self.state = advanced
            self.monitor(advanced, self.control)
            return
        if (
            not self.upper_attached
            and self.contact is None
            and self.height(before) > 0 >= self.height(advanced)
        ):

            def free_rhs(time, value):
                return self.derivative(
                    value,
                    self.control,
                    time_s=time,
                    available_propellant=available_propellant,
                )

            fraction = brentq(
                lambda dt: (
                    self.height(rk4_step(free_rhs, self.time_s, before, dt))
                    if dt
                    else self.height(before)
                ),
                0,
                step,
                xtol=1e-10,
            )
            advanced = rk4_step(free_rhs, self.time_s, before, fraction)
            step = fraction
            self.time_s += step
            self.state = advanced
            self.main_consumed += before[6] - advanced[6]
            self.rcs_consumed += before[7] - advanced[7]
            self.first_contact(advanced, self.control)
            self.monitor(advanced, self.control)
            return
        self.time_s += step
        advanced[8] = np.clip(advanced[8], 0, 1)
        advanced[9] = np.clip(advanced[9], -np.deg2rad(20), np.deg2rad(20))
        advanced[10] = np.clip(advanced[10], 0, 1)
        advanced[11] = np.clip(
            advanced[11],
            -np.deg2rad(self.design.gimbal_limit_deg),
            np.deg2rad(self.design.gimbal_limit_deg),
        )
        advanced[12] = np.clip(advanced[12], 0, 1)
        self.main_consumed += before[6] - advanced[6]
        self.rcs_consumed += before[7] - advanced[7]
        self.state = advanced
        if self.upper_attached and self.actual_engine(advanced)[0] > self.mass(advanced) * gravity(
            advanced[1]
        ):
            self.on_pad = False
        self.monitor(advanced, self.control)
        self.consider_separation()

    def run(self, progress=None):
        next_progress_s = self.time_s
        previous_phase = None
        while self.outcome == "running" and self.time_s < self.deadline_s:
            if progress is not None and (
                self.time_s >= next_progress_s or self.phase != previous_phase
            ):
                progress(self)
                previous_phase = self.phase
                next_progress_s = self.time_s + 20.0
            self.advance()
        if self.outcome == "running":
            self.outcome = "timeout"
        if self.outcome != "landed" and self.outcome not in self.failures:
            self.failures.append(self.outcome)
        self.record()
        separation_passed = self.separation is not None and all(
            self.separation["checks_before_separation"].values()
        )
        contact_passed = self.contact is not None and all(self.contact["criteria"].values())
        s = SAFETY
        eps = 1e-7  # 부동소수점 비교만 허용한다. 물리 허용값을 늘리지 않는다.
        safety = dict(
            dynamic_pressure=self.max_q <= s.max_q_pa + eps,
            ascent_q_alpha=self.max_q_alpha <= s.max_q_alpha_kpa_deg + eps,
            ascent_high_q_alpha=self.max_ascent_alpha <= s.max_high_q_alpha_deg + eps,
            ascent_axial_g=self.max_ascent_axial_g <= s.max_ascent_axial_g + eps,
            flight_load_g=self.max_flight_g <= s.max_recovery_flight_g + eps,
            angular_rate=self.max_rate <= s.max_angular_rate_deg_s + eps,
            angular_acceleration=self.max_acceleration <= s.max_angular_acceleration_deg_s2 + eps,
            gimbal=self.max_gimbal <= self.design.gimbal_limit_deg + eps,
            fin_hinge=self.max_fin_hinge <= FINS.per_fin_hinge_limit_nm + eps,
            fin_model_envelope=self.max_fin_mach <= FINS.model_max_mach + eps
            and self.max_fin_alpha <= FINS.control_max_alpha_deg + eps,
            contact_load_g=None,
            leg_force=None,
            leg_stroke=None,
            mass_balance=self.max_mass_balance_error < 1e-6,
            weather_profile_domain=self.max_altitude <= PROFILE_TOP_M,
            feed_inventory=bool(self.feed_inventory_valid),
            feed_supply=bool(self.feed_supply_valid),
            helium_mass_balance=self.max_helium_balance_error_kg < 1e-6,
            slosh_model_domain=bool(self.slosh_model_valid),
        )
        staging = self.separation.get("impulse_model") if self.separation else None
        safety["separation_momentum"] = staging is None or (
            np.linalg.norm(staging["linear_momentum_residual_kg_m_s"]) < 1e-6
            and abs(staging["angular_momentum_residual_kg_m2_s"]) < 1e-5
        )
        safety["separation_pulse_estimate"] = (
            staging is None
            or abs(np.rad2deg(staging["booster_pulse_average_angular_acceleration_rad_s2"]))
            <= s.max_angular_acceleration_deg_s2
        )
        full_mission = self.scope == "launch_to_separation_to_touchdown"
        mission = bool(
            full_mission
            and separation_passed
            and self.contact is not None
            and abs(self.contact["state"][0]) <= 2
        )
        landing = bool(contact_passed and self.outcome == "landed")
        if not full_mission:
            for name in ("ascent_q_alpha", "ascent_high_q_alpha", "ascent_axial_g"):
                safety[name] = None
        safe = bool(all(value for value in safety.values() if value is not None))
        criteria = {name: None if value is None else bool(value) for name, value in safety.items()}
        criteria.update(
            separation=bool(separation_passed),
            touchdown=bool(contact_passed),
            engine_off_standing=None,
        )
        if not full_mission:
            criteria["separation"] = None
        return dict(
            description=(
                "100톤·500 kg 탑재체의 2D 강체 개념 모델. 실기체 인증·전체 2단 비행 검증 아님."
            ),
            environment_version=OPERATING_SCOPE,
            systems_model_version=SYSTEMS_MODEL_VERSION,
            acceptance_version=f"apr_100t_{self.landing_assessment}_v1",
            landing_assessment=self.landing_assessment,
            assessment_scope=dict(
                flight_safety="through_first_contact",
                contact_structural_dynamics=False,
                post_contact_settling=False,
                leg_load_adequacy="assumed_within_vertical_touchdown_speed_limit",
                vertical_touchdown_speed_limit_m_s=self.physics.acceptance.max_vertical_foot_speed_m_s,
            ),
            state_names=list(STATE_NAMES),
            simulation_scope=self.scope,
            ascent_simulated=full_mission,
            operating_input_eligible=True,
            seed=378,
            simulation_count=1,
            return_method=RETURN_METHOD,
            fin_mode=self.fin_mode,
            fin_selection_reason=self.fin_selection_reason,
            fin_settings=asdict(FINS),
            design=asdict(self.design) | {"launch_mass_kg": self.design.launch_mass_kg},
            physics=asdict(self.physics),
            safety_limits=asdict(s),
            safety_derivation=s.derivation(),
            ascent_settings=asdict(self.ascent_settings),
            conditions=asdict(self.conditions),
            disturbances=asdict(self.disturbances),
            sensor_settings=asdict(self.sensor_settings),
            controller_options=asdict(self.controller_options),
            sensor_diagnostics=dict(
                generated_samples=self.sensor_sample_count,
                dropped_packets=self.sensor_dropout_count,
                estimated_gyro_bias_deg_s=(
                    float(np.rad2deg(self.attitude_estimator.estimate[2]))
                    if self.attitude_estimator.estimate is not None
                    else None
                ),
                configured_bias_is_not_supplied_to_estimator=True,
            ),
            integration_dt_s=self.integration_dt_s,
            criteria=criteria,
            mission_success=mission,
            flight_safety=safe,
            landing_success=landing,
            simulated_acceptance_passed=bool(
                (mission if full_mission else True) and safe and landing
            ),
            outcome=self.outcome,
            failures=self.failures,
            separation=self.separation,
            contact=self.contact,
            boostback_completed=self.boostback_completed,
            return_plan=self.return_plan,
            return_plan_history=self.return_plan_history,
            return_guidance_settings=asdict(self.return_settings),
            flip_started_s=self.flip_started_s,
            entry_started_s=self.entry_started_s,
            separation_alignment_started_s=self.separation_alignment_started_s,
            minimum_control_torque_margin_nm=self.minimum_control_torque_margin_nm,
            final_state=self.state.tolist(),
            time_s=self.time_s,
            main_propellant_used_kg=self.main_consumed,
            rcs_propellant_used_kg=self.rcs_consumed,
            main_propellant_remaining_kg=float(self.state[6]),
            rcs_propellant_remaining_kg=float(self.state[7]),
            max_dynamic_pressure_pa=self.max_q,
            max_recovery_dynamic_pressure_pa=self.max_return_q,
            max_ascent_q_alpha_kpa_deg=self.max_q_alpha,
            max_angular_rate_deg_s=self.max_rate,
            max_angular_acceleration_deg_s2=self.max_acceleration,
            peak_angular_acceleration=self.peak_angular_acceleration,
            max_flight_load_g=self.max_flight_g,
            max_contact_load_g=None,
            max_gimbal_deg=self.max_gimbal,
            max_fin_hinge_nm=self.max_fin_hinge,
            max_mass_balance_error_kg=self.max_mass_balance_error,
            max_altitude_m=self.max_altitude,
            integration_gust_exposure=self.integration_gust_exposure,
            peak_leg_force_n=None,
            peak_leg_stroke_m=None,
            upper_stage_mission=dict(
                payload_mass_kg=self.design.payload_mass_kg,
                stage_wet_mass_kg=self.design.upper_stage_wet_mass_kg,
                total_with_payload_kg=self.design.upper_mass_kg,
                dry_mass_kg=1500,
                main_propellant_kg=15500,
                target_orbit_altitude_m=700000,
                orbit_insertion_assumed=True,
                orbit_insertion_simulated=False,
                orbit_insertion_verified=False,
            ),
            first_stage_mission=dict(
                separation_altitude_m=self.ascent_settings.separation_altitude_m,
                landing_target_x_m=0.0,
                return_main_budget_kg=20000,
                horizontal_distance_prescribed=False,
                horizontal_velocity_prescribed=False,
            ),
            nominal_dry_mass_budget_kg=dict(
                engine=1200, other_airframe_systems=8700, landing_gear=1500, fins=600
            ),
            unconnected_systems=list(UNCONNECTED_SYSTEMS),
            systems_settings=dict(
                feed=asdict(self.feed_settings),
                slosh=asdict(self.slosh_settings),
                separation=asdict(self.separation_settings),
                ignition=asdict(self.ignition_settings),
            ),
            subsystem_design_budget=design_budget_check(
                self.feed_settings, initial_main_kg=self.design.main_propellant_kg
            ),
            systems_diagnostics=dict(
                minimum_feed_buffer_kg=self.minimum_feed_buffer_kg,
                minimum_tank_pressure_pa=(
                    self.minimum_tank_pressure_pa
                    if np.isfinite(self.minimum_tank_pressure_pa)
                    else None
                ),
                maximum_feed_shortfall_kg_s=self.max_feed_shortage_kg_s,
                maximum_helium_balance_error_kg=self.max_helium_balance_error_kg,
                maximum_slosh_displacement_m=self.max_slosh_displacement_m,
                maximum_slosh_mode_force_n=self.max_slosh_mode_force_n,
                first_failure=self.first_subsystem_failure,
                ignition_events=self.ignition_events,
            ),
            validation_status=dict(
                basis="theoretical_concept_design",
                measured_parameter_calibration=False,
                real_vehicle_safety_certified=False,
                continuous_operating_envelope_proven=False,
                fluid_model="single_equivalent_fluid_nonlinear_pendulum",
                slosh_coordinate="pendulum_arc_length",
                pressure_model="closed_isothermal_helium_pump_inlet",
                separation_model="conserved_rigid_impulse_not_pyroshock",
            ),
            steps=self.steps,
        )
