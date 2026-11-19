"""DM100 100 t vehicle design, operating environment, disturbance, and safety definitions.

Shared physical configuration dataclasses for the DM100 2D SpaceX-style RTLS
launch and first-stage recovery simulator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from math import cos, pi, sin
from pathlib import Path

import numpy as np
import yaml

from powered_landing_guidance.recovery_physics import (
    G0,
    R_AIR,
    Aerodynamics,
    AtmosphereSettings,
    Geometry,
    LandingAcceptance,
    LandingGear,
    Propulsion,
    atmosphere,
    standard_temperature_profile,
)
from powered_landing_guidance.vehicle_systems import (
    SloshSettings,
    SloshState,
    slosh_displacement,
)

RETURN_METHOD = "SpaceX RTLS"
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
