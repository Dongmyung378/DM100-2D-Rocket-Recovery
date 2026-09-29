"""Versioned 3 t APR recovery environment using the shared rigid-body laws.

The ideal engine admits continuous 0-100% throttle. Main and RCS inventories
are separate. Shared aerodynamic and compliant-foot laws are retained, but
feed, slosh, second-stage ignition and separation impulses are not modelled.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, fields, replace
from math import atan2, cos, exp, hypot, pi, sin, sqrt
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import brentq

from powered_landing_guidance.integrators import rk4_step
from powered_landing_guidance.recovery_physics import (
    G0,
    R_AIR,
    Aerodynamics,
    AtmosphereSettings,
    Geometry,
    LandingAcceptance,
    LandingGear,
    MassProperties,
    Propulsion,
    RecoverySensorEstimator,
    aerodynamic_loads,
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

RETURN_METHOD = "Aero-PID Return (APR)"
ASSUMED_UPPER_STAGE_ORBIT_ALTITUDE_M = 700_000.0


@dataclass(frozen=True, slots=True)
class RecoveryDisturbances:
    """Small healthy-model uncertainties, not user-selectable equipment faults."""

    thrust_scale: float = 1.0
    inertia_scale: float = 1.0
    sensor_noise_scale: float = 0.0
    observation_delay_steps: int = 0
    action_delay_steps: int = 0
    engine_lag_s: float = 0.0

    def __post_init__(self) -> None:
        for key, value, low, high in (
            ("thrust_scale", self.thrust_scale, 0.98, 1.02),
            ("inertia_scale", self.inertia_scale, 0.97, 1.03),
            ("sensor_noise_scale", self.sensor_noise_scale, 0.0, 1.0),
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

    surface_mass_kg: float = 12.0
    actuator_mass_kg: float = 8.0
    deployment_mass_kg: float = 3.0
    electrical_mass_kg: float = 1.0
    station_m: float = 4.65
    mean_mass_radius_m: float = 0.75
    installed_area_m2: float = 0.8
    planar_effective_area_m2: float = 0.4
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
    model_max_mach: float = 0.8
    hinge_arm_m: float = 0.08
    per_fin_hinge_limit_nm: float = 400.0

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
    dry_mass_kg: float = 520.0
    main_propellant_kg: float = 1450.0
    rcs_propellant_kg: float = 15.0
    upper_mass_kg: float = 1015.0
    sea_level_thrust_n: float = 42_000.0
    sea_level_isp_s: float = 260.0
    nozzle_exit_area_m2: float = 0.02
    gimbal_limit_deg: float = 6.0
    rcs_isp_s: float = 220.0
    rcs_torque_limit_nm: float = 6000.0
    high_q_threshold_pa: float = 10_000.0
    descent_reference_m_s: float = 0.9
    payload_mass_kg: float = 30.0  # Included in upper_mass_kg, not added to it.

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
    geometry: Geometry = Geometry(
        engine_mass_kg=110.0,
        airframe_mass_kg=326.0,
        gear_mass_kg=60.0,
        tank_bottom_m=1.8,
        tank_length_m=2.7,
    )
    aerodynamics: Aerodynamics = Aerodynamics(max_dynamic_pressure_pa=30_000.0)
    atmosphere: AtmosphereSettings = AtmosphereSettings()
    gear: LandingGear = LandingGear()
    propulsion: Propulsion = Propulsion(
        unusable_propellant_kg=0.0,
        ignition_propellant_flow_kg_s=0.0,
    )
    acceptance: LandingAcceptance = LandingAcceptance(
        max_position_error_m=2.0,
        max_horizontal_contact_speed_m_s=0.5,
        max_vertical_foot_speed_m_s=1.0,
        max_contact_tilt_deg=3.0,
        max_contact_rate_deg_s=0.5,
        fuel_reserve_kg=0.0,
        max_body_load_g=5.0,
    )
    fixed_mass_components: tuple[tuple[float, float, float], ...] = (
        (FINS.mass_kg, FINS.station_m, FINS.mass_kg * FINS.mean_mass_radius_m**2),
    )


PHYSICS = AprPhysics()


PROFILE_TOP_M = 20_000.0
MAX_PAD_WIND_M_S = 13.4
MAX_ALOFT_WIND_M_S = 28.0
MAX_WIND_SHEAR_M_S_PER_KM = 4.0
MAX_FORECAST_ERROR_M_S = 2.0
OPERATING_SCOPE = "apr_3t_operating_v1"
UNCONNECTED_SYSTEMS = (
    "liquid_acquisition_and_retention",
    "tank_pressurization_and_feed",
    "engine_ignition_and_restart_dynamics",
    "coupled_liquid_slosh",
    "elastic_structure",
    "physical_staging_impulse",
    "upper_stage_ignition_and_plume",
    "detailed_heating_and_ground_damage",
    "leg_material_strength_and_pad_bearing",
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
        raise ValueError(f"{name}: 0~20000 m 전체 고도 구간이 필요합니다.")
    for left, right in zip(levels, levels[1:], strict=False):
        if right[0] <= left[0]:
            raise ValueError(f"{name}: 고도는 중복 없이 증가해야 합니다.")
        if (
            1000.0 * abs(right[1] - left[1]) / (right[0] - left[0])
            > MAX_WIND_SHEAR_M_S_PER_KM + 1e-9
        ):
            raise ValueError(f"{name}: 잠정 윈드시어 한계 4 m/s/km를 초과했습니다.")
    if any(abs(w) > MAX_ALOFT_WIND_M_S for _, w in levels):
        raise ValueError(f"{name}: 잠정 상층풍 한계 ±28 m/s를 초과했습니다.")
    pad = [w for z, w in levels if z <= 50.0]
    pad.append(float(np.interp(50.0, [p[0] for p in levels], [p[1] for p in levels])))
    if max(abs(w) for w in pad) > MAX_PAD_WIND_M_S:
        raise ValueError(f"{name}: 지상풍 입력 상한 13.4 m/s를 초과했습니다.")
    return tuple(levels)


@dataclass(frozen=True)
class AprConditions:
    """Synthetic input eligibility, not proof of flight clearance or landing feasibility."""

    wind_profile: tuple[tuple[float, float], ...] = ((0.0, 0.0), (PROFILE_TOP_M, 0.0))
    forecast_wind_profile: tuple[tuple[float, float], ...] = ((0.0, 0.0), (PROFILE_TOP_M, 0.0))
    gust_amplitude_m_s: float = 0.0
    gust_period_s: float = 12.0
    gust_phase_rad: float = 0.0
    sea_level_temperature_k: float = 288.15
    sea_level_pressure_pa: float = 101325.0
    health_checks_passed: bool = True
    known_fault: bool = False
    lightning_cloud_review_passed: bool = True
    temperature_review_passed: bool = True
    recovery_weather_review_passed: bool = True
    precipitation_mm_h: float = 0.0
    thunderstorm_or_hail_present: bool = False

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
            "gust_amplitude_m_s": (0.0, 2.0),
            "gust_period_s": (8.0, 60.0),
            "gust_phase_rad": (-pi, pi),
            "sea_level_temperature_k": (273.15, 308.15),
            "sea_level_pressure_pa": (95000.0, 105000.0),
            "precipitation_mm_h": (0.0, 0.0),
        }
        for name, (low, high) in limits.items():
            value = _finite(getattr(self, name), name)
            if not low <= value <= high:
                raise ValueError(f"{name}: 잠정 정상 운용 범위 [{low}, {high}]를 벗어났습니다.")
        actual = _wind_profile(self.wind_profile, "wind_profile")
        forecast = _wind_profile(self.forecast_wind_profile, "forecast_wind_profile")
        object.__setattr__(self, "wind_profile", actual)
        object.__setattr__(self, "forecast_wind_profile", forecast)
        altitudes = {z for z, _ in actual} | {z for z, _ in forecast} | {50.0}
        for z in altitudes:
            aw, fw = self.mean_wind(z), self.mean_wind(z, forecast=True)
            if abs(aw) + self.gust_amplitude_m_s > MAX_ALOFT_WIND_M_S + 1e-9:
                raise ValueError("지속풍과 돌풍을 합친 상층풍이 잠정 한계를 초과했습니다.")
            if z <= 50.0 and abs(aw) + self.gust_amplitude_m_s > MAX_PAD_WIND_M_S + 1e-9:
                raise ValueError("지상 지속풍과 돌풍을 합친 입력이 13.4 m/s를 초과했습니다.")
            if abs(aw - fw) + self.gust_amplitude_m_s > MAX_FORECAST_ERROR_M_S + 1e-9:
                raise ValueError("돌풍을 포함한 예보 오차가 잠정 ±2 m/s 범위를 초과했습니다.")

    def mean_wind(self, altitude_m, *, forecast=False):
        levels = self.forecast_wind_profile if forecast else self.wind_profile
        return float(np.interp(altitude_m, [p[0] for p in levels], [p[1] for p in levels]))

    def wind(self, altitude_m, time_s, *, forecast=False):
        wind = self.mean_wind(altitude_m, forecast=forecast)
        if not forecast:
            wind += self.gust_amplitude_m_s * sin(
                2.0 * pi * time_s / self.gust_period_s + self.gust_phase_rad
            )
        return wind


@dataclass(frozen=True)
class AprWindField:
    conditions: AprConditions
    time_s: float = 0.0
    forecast: bool = False

    def at(self, altitude_m):
        return self.conditions.wind(altitude_m, self.time_s, forecast=self.forecast)


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
        raise ValueError("vehicle에는 현재 3톤 기체의 모든 제원만 입력할 수 있습니다.")
    for name, expected in design.items():
        if _finite(value["vehicle"][name], name) != expected:
            raise ValueError(f"vehicle.{name}: 현재 기준값 {expected}으로 고정됩니다.")
    for name, group in (("environment", AprConditions), ("disturbances", RecoveryDisturbances)):
        section = value[name]
        if not isinstance(section, dict) or set(section) != {f.name for f in fields(group)}:
            raise ValueError(f"{name}: 누락 또는 알 수 없는 설정 항목입니다.")
    return AprConditions(**value["environment"]), RecoveryDisturbances(**value["disturbances"])


@dataclass(frozen=True)
class AscentGuidanceSettings:
    steering_start_height_m: float = 500.0
    steering_ramp_height_m: float = 1000.0
    separation_altitude_m: float = 10_000.0
    separation_attitude_deg: float = 9.0
    separation_attitude_tolerance_deg: float = 0.5
    separation_vertical_speed_m_s: float = 120.0
    vertical_profile_deceleration_m_s2: float = 2.0
    vertical_speed_limit_m_s: float = 300.0
    horizontal_speed_budget_m_s: float = 25.0
    horizontal_distance_budget_m: float = 600.0
    horizontal_velocity_gain_deg_s_m: float = 2.0
    horizontal_position_gain_s_inv: float = 0.02
    alpha_reference_limit_deg: float = 2.0
    preparation_alpha_reference_limit_deg: float = 1.5
    preparation_start_altitude_m: float = 8500.0
    preparation_full_altitude_m: float = 9000.0
    preparation_rate_limit_deg_s: float = 2.0
    separation_alpha_target_deg: float = 2.0
    separation_rate_target_deg_s: float = 2.0
    separation_dwell_s: float = 0.3
    max_q_fall_dwell_s: float = 0.5
    max_q_fall_fraction: float = 0.99
    max_q_min_observed_peak_pa: float = 1000.0


@dataclass(frozen=True)
class ReturnGuidanceSettings:
    """Conservative controller design; acceptance limits are unchanged."""

    version: str = "apr-fuel-joint-v1"
    synchronized_alignment: bool = True
    joint_return_planning: bool = True
    control_dt_s: float = 0.01
    boostback_throttle: float = 0.15
    boostback_velocity_gain_s_inv: float = 0.8
    boostback_acceleration_limit_m_s2: float = 2.0
    boostback_tilt_limit_deg: float = 20.0
    boostback_min_burn_tilt_deg: float = 8.0
    return_plan_interval_s: float = 0.5
    ignition_range_candidates_m: tuple[float, ...] = (0.0, 75.0, 150.0)
    horizontal_kp_s2: float = 0.0256
    horizontal_kd_s_inv: float = 0.32
    horizontal_acceleration_limit_m_s2: float = 1.0
    horizontal_transfer_acceleration_m_s2: float = 0.8
    horizontal_transfer_jerk_m_s3: float = 0.2
    horizontal_transfer_min_duration_s: float = 10.0
    attitude_reference_tau_s: float = 0.5
    attitude_reference_slew_deg_s: float = 2.0
    attitude_position_gain_s_inv: float = 1.2
    attitude_rate_gain_s_inv: float = 3.0
    attitude_rate_limit_deg_s: float = 4.0
    attitude_acceleration_limit_deg_s2: float = 3.5
    fin_share_tau_s: float = 0.5
    gimbal_capacity_fraction: float = 0.8
    fin_deploy_min_downspeed_m_s: float = 30.0
    fin_deploy_max_alpha_deg: float = 10.0
    fin_deploy_max_rate_deg_s: float = 1.0
    fin_retract_height_m: float = 1800.0
    load_governor_g: float = 4.75
    ignition_deceleration_m_s2: float = 14.0
    ignition_margin_m: float = 120.0
    descent_deceleration_m_s2: float = 2.0
    vertical_velocity_gain_s_inv: float = 0.9
    terminal_flare_height_m: float = 3.0
    alignment_trigger_height_m: float = 600.0
    alignment_hold_height_m: float = 350.0
    alignment_min_height_m: float = 250.0
    alignment_reference_deceleration_m_s2: float = 8.0
    alignment_end_speed_m_s: float = 1.0
    alignment_vertical_kp_s2: float = 0.12
    alignment_vertical_kd_s_inv: float = 0.8
    alignment_position_gate_m: float = 0.35
    alignment_horizontal_speed_gate_m_s: float = 0.15
    alignment_tilt_gate_deg: float = 1.5
    alignment_rate_gate_deg_s: float = 0.15
    alignment_vertical_speed_gate_m_s: float = 1.5
    alignment_dwell_s: float = 1.0
    approach_speed_limit_m_s: float = 20.0
    approach_reference_acceleration_m_s2: float = 2.0

    @classmethod
    def for_stage(cls, stage):
        if stage == "joint":
            return cls()
        if stage not in ("approach", "sync"):
            raise ValueError("guidance stage must be approach, sync or joint")
        return replace(
            cls(),
            version=f"apr-fuel-{stage}-v1",
            synchronized_alignment=stage == "sync",
            joint_return_planning=False,
            ignition_deceleration_m_s2=10.0,
            ignition_margin_m=600.0,
        )


@dataclass(frozen=True)
class CoastEstimate:
    """Closed-form constant-density, flow-aligned quadratic-drag surrogate.

    The vertical drag law is solved analytically. Horizontal velocity follows
    dvx/dt = -k*abs(vz)*vx, ignoring cross-flow, fin lift and Earth rotation.
    No vehicle trajectory is integrated during planning.
    """

    duration_s: float
    horizontal_gain_s: float
    horizontal_velocity_ratio: float
    downspeed_m_s: float
    mean_density_kg_m3: float

    @classmethod
    def between(
        cls,
        height_m,
        vz_m_s,
        ignition_height_m,
        mass_kg,
        drag_area_m2,
        *,
        atmosphere_settings=PHYSICS.atmosphere,
    ):
        if (
            not all(
                np.isfinite(v) for v in (height_m, vz_m_s, ignition_height_m, mass_kg, drag_area_m2)
            )
            or mass_kg <= 0.0
            or drag_area_m2 <= 0.0
            or ignition_height_m < 0.0
        ):
            raise ValueError("coast estimate requires finite heights and positive mass and drag")
        if height_m <= ignition_height_m and vz_m_s <= 0.0:
            return cls(0.0, 0.0, 1.0, -vz_m_s, atmosphere(height_m, atmosphere_settings)[2])
        densities = [
            atmosphere(z, atmosphere_settings)[2]
            for z in (height_m, (height_m + ignition_height_m) / 2.0, ignition_height_m)
        ]
        density = (densities[0] + 4.0 * densities[1] + densities[2]) / 6.0
        k = 0.5 * density * drag_area_m2 / mass_kg
        g = gravity(height_m)
        terminal = sqrt(g / k)
        frequency = sqrt(g * k)
        if vz_m_s >= 0.0:
            beta = float(np.arctan(vz_m_s / terminal))
            rise = float(np.log1p((vz_m_s / terminal) ** 2)) / (2.0 * k)
            distance = max(0.0, height_m + rise - ignition_height_m)
            u = float(np.arccosh(np.exp(k * distance)))
            duration = (beta + u) / frequency
            gain = (
                cos(beta)
                * (float(np.arcsinh(np.tan(beta))) + float(np.arctan(np.sinh(u))))
                / frequency
            )
            ratio = cos(beta) / float(np.cosh(u))
            downspeed = terminal * float(np.tanh(u))
        else:
            distance = height_m - ignition_height_m
            speed = -vz_m_s
            if abs(speed / terminal - 1.0) < 1e-7:
                duration = distance / terminal
                gain = float(-np.expm1(-k * distance)) / frequency
                ratio = float(np.exp(-k * distance))
                downspeed = terminal
            elif speed < terminal:
                start = float(np.arctanh(speed / terminal))
                end = float(np.arccosh(np.cosh(start) * np.exp(k * distance)))
                duration = (end - start) / frequency
                gain = (
                    float(np.cosh(start))
                    * (float(np.arctan(np.sinh(end))) - float(np.arctan(np.sinh(start))))
                    / frequency
                )
                ratio = float(np.cosh(start) / np.cosh(end))
                downspeed = terminal * float(np.tanh(end))
            else:
                start = float(np.arctanh(terminal / speed))
                end = float(np.arcsinh(np.sinh(start) * np.exp(k * distance)))
                duration = (end - start) / frequency
                gain = (
                    float(np.sinh(start))
                    * float(np.log(np.tanh(end / 2.0) / np.tanh(start / 2.0)))
                    / frequency
                )
                ratio = float(np.sinh(start) / np.sinh(end))
                downspeed = terminal / float(np.tanh(end))
        return cls(duration, gain, ratio, downspeed, density)


@dataclass(frozen=True)
class HorizontalTransfer:
    """Quintic reference from measured position/velocity to a stationary origin."""

    started_s: float
    duration_s: float
    coefficients_m: tuple[float, ...]
    peak_reference_acceleration_m_s2: float
    peak_reference_jerk_m_s3: float
    attitude_lag_compensation_s: float
    peak_compensated_acceleration_m_s2: float

    @classmethod
    def plan(cls, started_s, x_m, vx_m_s, settings):
        acceleration = settings.horizontal_transfer_acceleration_m_s2
        jerk = settings.horizontal_transfer_jerk_m_s3
        position_gain = settings.attitude_position_gain_s_inv
        filter_tau = settings.attitude_reference_tau_s
        if (
            not all(
                np.isfinite(value)
                for value in (started_s, x_m, vx_m_s, acceleration, jerk, position_gain, filter_tau)
            )
            or acceleration <= 0.0
            or jerk <= 0.0
            or position_gain <= 0.0
            or filter_tau < 0.0
            or not np.isfinite(settings.horizontal_transfer_min_duration_s)
            or settings.horizontal_transfer_min_duration_s <= 0.0
            or acceleration >= settings.horizontal_acceleration_limit_m_s2
        ):
            raise ValueError("horizontal transfer requires finite inputs and positive margins")
        # Low-frequency delay of the reference filter and cascaded attitude PD.
        # This is a bounded first-order lead, not an exact actuator inverse.
        lag = filter_tau + 1.0 / position_gain
        duration = max(
            settings.horizontal_transfer_min_duration_s,
            sqrt(6.0 * abs(x_m) / acceleration),
            float(np.cbrt(60.0 * abs(x_m) / jerk)),
            4.0 * abs(vx_m_s) / acceleration,
        )
        for _ in range(12):
            velocity = vx_m_s * duration
            coefficients = (
                x_m,
                velocity,
                0.0,
                -10.0 * x_m - 6.0 * velocity,
                15.0 * x_m + 8.0 * velocity,
                -6.0 * x_m - 3.0 * velocity,
            )
            acceleration_poly = np.polynomial.polynomial.polyder(coefficients, 2) / duration**2
            jerk_poly = np.polynomial.polynomial.polyder(coefficients, 3) / duration**3
            compensated_poly = np.polynomial.polynomial.polyadd(acceleration_poly, lag * jerk_poly)
            peaks = []
            for derivative in (acceleration_poly, jerk_poly, compensated_poly):
                roots = np.polynomial.polynomial.polyroots(
                    np.polynomial.polynomial.polyder(derivative)
                )
                points = [0.0, 1.0]
                points.extend(
                    float(root.real)
                    for root in roots
                    if abs(root.imag) < 1e-9 and 0.0 < root.real < 1.0
                )
                peaks.append(
                    max(
                        abs(np.polynomial.polynomial.polyval(point, derivative)) for point in points
                    )
                )
            if peaks[0] <= acceleration and peaks[1] <= jerk and peaks[2] <= acceleration:
                return cls(
                    started_s,
                    duration,
                    coefficients,
                    float(peaks[0]),
                    float(peaks[1]),
                    lag,
                    float(peaks[2]),
                )
            # Widen one reference's horizon, not a search over simulated flights.
            duration *= 1.25
        raise ValueError("could not construct a bounded horizontal transfer")

    def reference(self, time_s):
        elapsed = time_s - self.started_s
        if elapsed >= self.duration_s:
            return 0.0, 0.0, 0.0
        fraction = max(0.0, elapsed) / self.duration_s
        return tuple(
            float(
                np.polynomial.polynomial.polyval(
                    fraction, np.polynomial.polynomial.polyder(self.coefficients_m, order)
                )
                / self.duration_s**order
            )
            for order in (0, 1, 2)
        )

    def jerk(self, time_s):
        elapsed = time_s - self.started_s
        if elapsed < 0.0 or elapsed >= self.duration_s:
            return 0.0
        return float(
            np.polynomial.polynomial.polyval(
                elapsed / self.duration_s,
                np.polynomial.polynomial.polyder(self.coefficients_m, 3),
            )
            / self.duration_s**3
        )


@dataclass(frozen=True)
class AlignmentDescent:
    """Monotone descent ending at the horizontal reference's deadline.

    A power-law speed reference has analytic height and acceleration. Planning
    it does not propagate the vehicle or alter its measured state.
    """

    started_s: float
    duration_s: float
    started_height_m: float
    started_downspeed_m_s: float
    end_downspeed_m_s: float
    speed_power: float
    end_height_m: float
    min_height_m: float
    peak_deceleration_m_s2: float

    @classmethod
    def plan(cls, time_s, height_m, vz_m_s, duration_s, settings):
        final_speed = settings.alignment_end_speed_m_s
        floor = settings.alignment_min_height_m
        limit = settings.alignment_reference_deceleration_m_s2
        if (
            not all(
                np.isfinite(value)
                for value in (time_s, height_m, vz_m_s, duration_s, final_speed, floor, limit)
            )
            or final_speed <= 0.0
            or final_speed >= settings.alignment_vertical_speed_gate_m_s
            or floor <= 200.0
            or limit <= 0.0
        ):
            raise ValueError("alignment descent requires finite inputs and positive safety margins")
        downspeed = -vz_m_s
        available_distance = height_m - floor - final_speed * duration_s
        if duration_s < 2.0 or downspeed <= final_speed or available_distance <= 0.0:
            return None
        power = max(2.0, (downspeed - final_speed) * duration_s / available_distance - 1.0)
        peak = (downspeed - final_speed) * power / duration_s
        if peak > limit:
            return None  # Use the existing altitude hold when timing is infeasible.
        distance = duration_s * (final_speed + (downspeed - final_speed) / (power + 1.0))
        return cls(
            time_s,
            duration_s,
            height_m,
            downspeed,
            final_speed,
            power,
            height_m - distance,
            floor,
            peak,
        )

    def reference(self, time_s):
        elapsed = max(0.0, time_s - self.started_s)
        if elapsed >= self.duration_s:
            height = self.end_height_m - self.end_downspeed_m_s * (elapsed - self.duration_s)
            if height <= self.min_height_m:
                return self.min_height_m, 0.0, 0.0
            return height, -self.end_downspeed_m_s, 0.0
        remaining = 1.0 - elapsed / self.duration_s
        difference = self.started_downspeed_m_s - self.end_downspeed_m_s
        distance = self.end_downspeed_m_s * elapsed
        distance += (
            difference
            * self.duration_s
            * (1.0 - remaining ** (self.speed_power + 1.0))
            / (self.speed_power + 1.0)
        )
        speed = self.end_downspeed_m_s + difference * remaining**self.speed_power
        acceleration = self.peak_deceleration_m_s2 * remaining ** (self.speed_power - 1.0)
        return self.started_height_m - distance, -speed, acceleration


class AprFlight:
    """Cascaded position/velocity and attitude PD with nominal force feed-forward."""

    def __init__(
        self, *, fin_mode="active", guidance_stage="joint", conditions=None, disturbances=None
    ):
        if fin_mode not in ("stowed", "active"):
            raise ValueError("fin_mode must be stowed or active")
        self.fin_mode = fin_mode
        self.design = Design()
        self.conditions = (
            AprConditions() if conditions is None else AprConditions(**asdict(conditions))
        )
        self.disturbances = (
            RecoveryDisturbances()
            if disturbances is None
            else RecoveryDisturbances(**asdict(disturbances))
        )
        self.physics = replace(
            PHYSICS,
            atmosphere=replace(
                PHYSICS.atmosphere,
                sea_level_temperature_k=self.conditions.sea_level_temperature_k,
                sea_level_pressure_pa=self.conditions.sea_level_pressure_pa,
            ),
        )
        self.ascent_settings = AscentGuidanceSettings()
        self.return_settings = ReturnGuidanceSettings.for_stage(guidance_stage)
        g = self.physics.geometry
        dry_mass = g.engine_mass_kg + g.airframe_mass_kg + g.gear_mass_kg + FINS.mass_kg
        if not np.isclose(dry_mass, self.design.dry_mass_kg):
            raise ValueError("dry mass allocation must sum to 520 kg")
        capacity = pi * g.body_radius_m**2 * g.tank_length_m * g.propellant_density_kg_m3
        if capacity <= self.design.main_propellant_kg + self.design.rcs_propellant_kg:
            raise ValueError("equivalent liquid volume must leave positive ullage")
        self.upper_attached = True
        self.phase = "ascent"
        # Append actual fin deflection and deployment to the original flight state.
        self.state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1450.0, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.state[1] = self.properties(self.state).cg_m - self.physics.gear.foot_station_m
        self.time_s = 0.0
        self.started_s = 0.0
        self.deadline_s = 300.0
        self.next_control_s = 0.0
        self.next_record_s = 0.0
        self.scope = "launch_to_separation_to_settled_landing"
        self.initial_propellant = (self.design.main_propellant_kg, self.design.rcs_propellant_kg)
        self.rng = np.random.default_rng(378)
        self.sensor_estimator = RecoverySensorEstimator(self.disturbances.sensor_noise_scale)
        self.observations = deque(maxlen=self.disturbances.observation_delay_steps + 1)
        self.action_queue = deque(
            ((0.0, 0.0, 0.0, 0.0) for _ in range(self.disturbances.action_delay_steps)),
            maxlen=self.disturbances.action_delay_steps + 1,
        )
        self.observed_state = self.state.copy()
        self.vertical_estimate = None
        self.observation_time_s = None
        self.engine_throttle = 0.0
        self.requested_control = (0.0, 0.0, 0.0, 0.0)
        self.contact = None
        self.separation = None
        self.boostback_completed = None
        self.return_plan = None
        self.next_return_plan_s = 0.0
        self.landing_ignition = None
        self.settled_since = None
        self.outcome = "running"
        self.failures = []
        self.steps = []
        self.max_q = 0.0
        self.max_q_before_separation = 0.0
        self.max_q_alpha = 0.0
        self.max_ascent_alpha = 0.0
        self.max_ascent_rate = 0.0
        self.max_ascent_acceleration = 0.0
        self.max_axial_g = 0.0
        self.max_recovery_g = 0.0
        self.max_recovery_rate = 0.0
        self.max_recovery_acceleration = 0.0
        self.max_approach_tilt = 0.0
        self.max_mass_balance_error = 0.0
        self.peak_leg_force = np.zeros(2)
        self.peak_leg_stroke = np.zeros(2)
        self.main_consumed = self.rcs_consumed = 0.0
        self.max_x = 0.0
        self.max_altitude = 0.0
        self.control = (1.0, 0.0, 0.0, 0.0)
        self.approach_started_s = None
        self.ascent_preparation_started_s = None
        self.max_q_fall_since_s = None
        self.max_q_passed = False
        self.previous_ascent_q_pa = 0.0
        self.separation_ready_since_s = None
        self.last_separation_checks = None
        self.last_separation_check_state = None
        self.ascent_command = None
        self.attitude_target = 0.0
        self.max_fin_force_n = 0.0
        self.max_fin_torque_nm = 0.0
        self.max_fin_hinge_nm = 0.0
        self.max_deployed_fin_mach = 0.0
        self.max_deployed_fin_alpha_deg = 0.0
        self.fin_deployment_started = False
        self.fin_retraction_started = False
        self.fin_deployment_started_s = None
        self.fin_retraction_started_s = None
        self.fin_share = 0.0
        self.filtered_attitude_target = 0.0
        self.alignment_started_s = None
        self.alignment_started_main_propellant_kg = None
        self.alignment_ready_since_s = None
        self.alignment_actual_ready_since_s = None
        self.alignment_completed = None
        self.alignment_descent = None
        self.vertical_command = None
        self.approach_speed_reference = None
        self.horizontal_transfer = None
        self.horizontal_command = None
        self.load_governor_activations = 0
        self.last_allocation = {}

    @classmethod
    def from_separation(
        cls, state, *, time_s, after_max_q, gate_altitude_m=None, engine_throttle=0.0, **kwargs
    ):
        """Start at the first-stage COM without resetting separation momentum.

        ``after_max_q`` is an explicit upstream assertion, not an ascent result
        verified by this recovery-only episode. The state has the same 13 entries
        as a full-flight record; fins and gear must be stowed at separation.
        """
        initial = np.asarray(state, dtype=float)
        if initial.shape != (13,) or not np.all(np.isfinite(initial)):
            raise ValueError("분리 상태는 유한한 숫자 13개여야 합니다.")
        started = _finite(time_s, "time_s")
        throttle = _finite(engine_throttle, "engine_throttle")
        if started < 0.0 or not 0.0 <= throttle <= 1.0 or after_max_q is not True:
            raise ValueError("분리 시간과 Max-Q 통과 확인이 필요합니다.")
        flight = cls(**kwargs)
        design = flight.design
        if not (
            abs(initial[0]) <= 1000.0
            and 9890.0 <= initial[1] <= 10100.0
            and abs(initial[2]) <= 60.0
            and 0.0 < initial[3] <= 300.0
            and 0.0 < initial[6] <= design.main_propellant_kg
            and 0.0 <= initial[7] <= design.rcs_propellant_kg
            and np.all(initial[8:] == 0.0)
        ):
            raise ValueError("분리 상태가 현재 3톤 기체의 잠정 진입 범위를 벗어났습니다.")
        flight.upper_attached = False
        flight.state = initial.copy()
        flight.observed_state = initial.copy()
        flight.time_s = flight.started_s = started
        flight.deadline_s = started + 300.0
        flight.next_control_s = flight.next_record_s = started
        flight.scope = "10km_separation_to_settled_landing"
        flight.initial_propellant = (float(initial[6]), float(initial[7]))
        flight.engine_throttle = throttle
        flight.max_q_passed = True
        gate_height = (
            initial[1] if gate_altitude_m is None else _finite(gate_altitude_m, "gate_altitude_m")
        )
        checks = flight.separation_checks(initial)
        checks["altitude"] = abs(gate_height - 10000.0) <= 100.0
        if not all(checks.values()):
            failed = ", ".join(name for name, passed in checks.items() if not passed)
            raise ValueError(f"분리 진입 조건 미통과: {failed}")
        flight.separation = dict(
            time_s=started,
            state=initial.tolist(),
            altitude_m=float(initial[1]),
            x_m=float(initial[0]),
            vx_m_s=float(initial[2]),
            vz_m_s=float(initial[3]),
            main_propellant_kg=float(initial[6]),
            rcs_propellant_kg=float(initial[7]),
            first_stage_mass_kg=flight.mass(initial),
            gate_altitude_before_cg_change_m=gate_height,
            gate_altitude_reference="upstream_separation_record",
            checks_before_separation=checks,
            after_observed_max_q=True,
            upstream_max_q_assertion=True,
            ascent_simulated=False,
            engine_throttle=throttle,
            controller_memory_initialized=True,
        )
        flight.phase = "boostback" if abs(initial[0]) > 2.0 else "coast"
        flight.filtered_attitude_target = float(initial[4])
        return flight

    def wind_field(self, *, predicted=False, time_s=None):
        return AprWindField(self.conditions, self.time_s if time_s is None else time_s, predicted)

    def observe_control_state(self):
        """Measured states and actuator telemetry only; no true wind or thrust error."""
        self.observations.append((self.time_s, self.state.copy()))
        sample_time, raw = self.observations[0]
        raw = raw.copy()
        scale = self.disturbances.sensor_noise_scale
        delay = self.time_s - sample_time
        if scale == 0.0 and delay == 0.0:
            self.observed_state = raw
            return raw
        if scale:
            deviations = scale * np.asarray(
                (0.5, 0.5, 0.1, 0.1, np.deg2rad(0.25), np.deg2rad(0.25))
            )
            raw[:6] += self.rng.normal(0.0, deviations)
        prediction = self.observed_state.copy()
        force, aero_torque, _, _ = self.loads(prediction, predicted=True)
        thrust, _ = self.engine(self.engine_throttle, prediction[1])
        ax = (force[0] + thrust * sin(prediction[4] + self.control[1])) / self.mass(prediction)
        properties = self.properties(prediction)
        torque = (
            aero_torque
            + self.control[2]
            - (properties.cg_m - self.physics.geometry.nozzle_station_m)
            * thrust
            * sin(self.control[1])
        )
        raw = self.sensor_estimator.observe(
            raw,
            self.time_s,
            (ax, torque / properties.inertia_kg_m2),
            delay,
        )
        if self.vertical_estimate is None:
            self.vertical_estimate = np.asarray((raw[1], raw[3]))
        else:
            dt = self.time_s - self.observation_time_s
            if dt > 0.0:
                self.vertical_estimate[0] += dt * self.vertical_estimate[1]
                self.vertical_estimate[0] += dt / (0.5 + dt) * (raw[1] - self.vertical_estimate[0])
                self.vertical_estimate[1] += dt / (0.15 + dt) * (raw[3] - self.vertical_estimate[1])
        raw[1] = self.vertical_estimate[0] + delay * self.vertical_estimate[1]
        raw[3] = self.vertical_estimate[1]
        self.observation_time_s = self.time_s
        self.observed_state = raw
        return raw

    def applied_control(self, time_s, command, started_s, initial_throttle):
        lag = self.disturbances.engine_lag_s
        throttle = (
            command[0]
            if lag == 0.0
            else (command[0] + (initial_throttle - command[0]) * exp(-(time_s - started_s) / lag))
        )
        return (throttle, *command[1:])

    def vector(self, y):
        return np.r_[y[:6], self.design.dry_mass_kg + y[6] + y[7], y[8:11]]

    def properties(self, y):
        first = mass_properties(self.design.dry_mass_kg + y[6] + y[7], y[8], self.physics)
        if not self.upper_attached:
            return first
        mass = self.design.dry_mass_kg + y[6] + y[7]
        upper = self.design.upper_mass_kg
        upper_station = 6.0
        cg = (mass * first.cg_m + upper * upper_station) / (mass + upper)
        inertia = first.inertia_kg_m2 + mass * (first.cg_m - cg) ** 2
        inertia += upper * ((upper_station - cg) ** 2 + 2.0**2 / 12.0 + 0.45**2 / 4.0)
        return MassProperties(cg, inertia, first.fuel_height_m)

    def body_loads(self, y, *, predicted=False, time_s=None):
        vector = self.vector(y)
        settings = self.physics
        if self.upper_attached:
            vector[6] += self.design.upper_mass_kg
            settings = replace(
                self.physics,
                geometry=replace(self.physics.geometry, body_length_m=7.0),
                aerodynamics=replace(self.physics.aerodynamics, pressure_center_m=3.5),
            )
        return aerodynamic_loads(
            vector,
            y[8],
            self.wind_field(predicted=predicted, time_s=time_s),
            settings,
            self.properties(y),
        )

    def fin_loads(self, y, deflection=None, *, predicted=False, time_s=None):
        """Lift is perpendicular to flow; drag opposes it, with a finite lever arm.

        Four installed surfaces are represented by one effective in-plane pair.
        Saturated lift at large AoA is only a screening approximation. Every run
        reports excursions outside the declared subsonic/small-AoA envelope.
        """
        if self.upper_attached or y[12] <= 0.0:
            return np.zeros(2), 0.0, 0.0
        velocity = np.array(
            [y[2] - self.wind_field(predicted=predicted, time_s=time_s).at(y[1]), y[3]]
        )
        speed = hypot(*velocity)
        if speed < 1e-8:
            return np.zeros(2), 0.0, 0.0
        density = atmosphere(y[1], self.physics.atmosphere)[2]
        q = 0.5 * density * speed**2
        normal_axis = rotation(y[4])[:, 0]
        lift_direction = np.array([-velocity[1], velocity[0]]) / speed
        if float(lift_direction @ normal_axis) < 0.0:
            lift_direction *= -1.0
        angle = y[11] if deflection is None else deflection
        cl = float(
            np.clip(
                FINS.lift_slope_rad_inv
                * (angle - np.deg2rad(self.alpha_deg(y, predicted=predicted, time_s=time_s))),
                -FINS.max_lift_coefficient,
                FINS.max_lift_coefficient,
            )
        )
        lift = q * y[12] * FINS.planar_effective_area_m2 * cl
        drag_area = (
            FINS.installed_area_m2 * FINS.base_drag_coefficient
            + FINS.planar_effective_area_m2 * FINS.induced_drag_factor * cl**2
        )
        force = lift * lift_direction - q * y[12] * drag_area * velocity / speed
        arm = rotation(y[4]) @ np.array([0.0, FINS.station_m - self.properties(y).cg_m])
        torque = clock_torque(arm, force)
        # The planar lift pair shares the aerodynamic hinge load equally.
        hinge = abs(lift) * FINS.hinge_arm_m / 2.0
        return force, torque, hinge

    def loads(self, y, *, predicted=False, time_s=None):
        force, torque, q, mach = self.body_loads(y, predicted=predicted, time_s=time_s)
        fin_force, fin_torque, _ = self.fin_loads(y, predicted=predicted, time_s=time_s)
        return force + fin_force, torque + fin_torque, q, mach

    def fin_command(self, y, requested_torque):
        if (
            self.fin_mode != "active"
            or self.upper_attached
            or y[12] <= 0.0
            or self.contact is not None
            or abs(self.alpha_deg(y, predicted=True)) > FINS.control_max_alpha_deg
        ):
            return 0.0
        if self.body_loads(y, predicted=True)[2] < FINS.control_min_q_pa:
            # Feather rather than snap to body-neutral at the low-q transition.
            return float(
                np.deg2rad(
                    np.clip(
                        self.alpha_deg(y, predicted=True),
                        -FINS.deflection_limit_deg,
                        FINS.deflection_limit_deg,
                    )
                )
            )
        limit = np.deg2rad(FINS.deflection_limit_deg)
        low = self.fin_loads(y, -limit, predicted=True)[1]
        high = self.fin_loads(y, limit, predicted=True)[1]
        if requested_torque <= low:
            return -limit
        if requested_torque >= high:
            return limit
        return float(
            brentq(
                lambda angle: self.fin_loads(y, angle, predicted=True)[1] - requested_torque,
                -limit,
                limit,
                xtol=1e-9,
            )
        )

    def mass(self, y):
        return (
            self.design.dry_mass_kg
            + y[6]
            + y[7]
            + (self.design.upper_mass_kg if self.upper_attached else 0.0)
        )

    def height(self, y):
        if self.upper_attached:
            return y[1] - self.properties(y).cg_m + self.physics.gear.foot_station_m
        return float(foot_kinematics(self.vector(y), y[8], self.physics)[0][:, 1].min())

    def engine(self, throttle, height, *, actual=False):
        return engine_performance(
            throttle,
            height,
            self.physics,
            self.disturbances.thrust_scale if actual else 1.0,
            sea_level_thrust_n=self.design.sea_level_thrust_n,
            sea_level_isp_s=self.design.sea_level_isp_s,
        )

    def alpha_deg(self, y, *, predicted=False, time_s=None):
        relative_velocity = np.array(
            [y[2] - self.wind_field(predicted=predicted, time_s=time_s).at(y[1]), y[3]]
        )
        speed = hypot(*relative_velocity)
        if speed < 1e-5:
            return 0.0
        # AoA describes flow relative to the nearest direction of the body axis.
        normal, axial = rotation(y[4]).T @ relative_velocity
        return float(np.rad2deg(atan2(normal, abs(axial))))

    def preparation_fraction(self, y):
        settings = self.ascent_settings
        return float(
            np.clip(
                (y[1] - settings.preparation_start_altitude_m)
                / (settings.preparation_full_altitude_m - settings.preparation_start_altitude_m),
                0.0,
                1.0,
            )
        )

    def ascent_attitude_target(self, y, height):
        """Air-relative flight-path tracking with bounded lateral steering bias."""
        settings = self.ascent_settings
        blend = self.preparation_fraction(y)
        steering = float(
            np.clip(
                (height - settings.steering_start_height_m) / settings.steering_ramp_height_m,
                0.0,
                1.0,
            )
        )
        # Small-angle gravity-turn schedule for the existing vertical speed
        # profile: d(log(tan(gamma)))/dz ~ g/vz**2. Feedback handles departures
        # from that reference; neither velocity nor attitude state is overwritten.
        vertical_reference = self.ascent_vertical_speed_reference(height)
        speed_ratio = settings.separation_vertical_speed_m_s / vertical_reference
        flight_path_reference = atan2(
            np.tan(np.deg2rad(settings.separation_attitude_deg))
            * speed_ratio ** (G0 / settings.vertical_profile_deceleration_m_s2),
            1.0,
        )
        reference = steering * max(y[3], 1.0) * np.tan(flight_path_reference)
        # These are guidance budgets, not new acceptance limits or state clamps.
        overshoot = max(abs(y[0]) - settings.horizontal_distance_budget_m, 0.0)
        reference -= np.sign(y[0]) * settings.horizontal_position_gain_s_inv * overshoot
        reference = float(
            np.clip(
                reference,
                -settings.horizontal_speed_budget_m_s,
                settings.horizontal_speed_budget_m_s,
            )
        )
        velocity_x = y[2] - self.wind_field(predicted=True).at(y[1])
        flight_path = atan2(velocity_x, max(y[3], 1.0))
        alpha_limit = (
            (1.0 - blend) * settings.alpha_reference_limit_deg
            + blend * settings.preparation_alpha_reference_limit_deg
        )
        bias_deg = float(
            np.clip(
                settings.horizontal_velocity_gain_deg_s_m * (reference - velocity_x),
                -alpha_limit,
                alpha_limit,
            )
        )
        target = steering * (flight_path + np.deg2rad(bias_deg))
        self.ascent_command = dict(
            flight_path_angle_deg=float(np.rad2deg(flight_path)),
            flight_path_reference_deg=float(np.rad2deg(flight_path_reference)),
            target_attitude_deg=float(np.rad2deg(target)),
            target_alpha_deg=float(np.rad2deg(flight_path - target)),
            horizontal_speed_reference_m_s=reference,
            preparation_fraction=blend,
        )
        return target

    def ascent_vertical_speed_reference(self, height):
        settings = self.ascent_settings
        return min(
            settings.vertical_speed_limit_m_s,
            sqrt(
                settings.separation_vertical_speed_m_s**2
                + 2.0
                * settings.vertical_profile_deceleration_m_s2
                * max(settings.separation_altitude_m - height, 0.0)
            ),
        )

    def update_max_q_passed(self, q):
        """Causal descending-q evidence, not advance knowledge of a future maximum."""
        settings = self.ascent_settings
        descending = (
            self.max_q_before_separation >= settings.max_q_min_observed_peak_pa
            and q < settings.max_q_fall_fraction * self.max_q_before_separation
            and q < self.previous_ascent_q_pa
        )
        if not descending:
            self.max_q_fall_since_s = None
            self.max_q_passed = False
        else:
            if self.max_q_fall_since_s is None:
                self.max_q_fall_since_s = self.time_s
            self.max_q_passed = self.time_s - self.max_q_fall_since_s >= settings.max_q_fall_dwell_s
        self.previous_ascent_q_pa = q

    def separation_checks(self, y, *, predicted=False):
        settings = self.ascent_settings
        return dict(
            altitude=abs(y[1] - settings.separation_altitude_m) <= 100.0,
            mission_attitude=abs(np.rad2deg(y[4]) - settings.separation_attitude_deg)
            <= settings.separation_attitude_tolerance_deg,
            dynamic_pressure=self.loads(y, predicted=predicted)[2] <= 25_000.0,
            after_observed_max_q=self.max_q_passed,
            alpha=abs(self.alpha_deg(y, predicted=predicted)) <= 5.0,
            angular_rate=abs(np.rad2deg(y[5])) <= 5.0,
            preparation_alpha=abs(self.alpha_deg(y, predicted=predicted))
            <= self.ascent_settings.separation_alpha_target_deg,
            preparation_rate=abs(np.rad2deg(y[5]))
            <= self.ascent_settings.separation_rate_target_deg_s,
            ascending=y[3] > 0.0,
        )

    def consider_separation(self, y, *, predicted=False):
        if y[1] < self.ascent_settings.preparation_start_altitude_m:
            return
        checks = self.separation_checks(y, predicted=predicted)
        self.last_separation_checks = checks
        self.last_separation_check_state = dict(
            time_s=self.time_s,
            altitude_m=y[1],
            dynamic_pressure_pa=self.loads(y, predicted=predicted)[2],
            alpha_deg=self.alpha_deg(y, predicted=predicted),
            attitude_deg=float(np.rad2deg(y[4])),
            angular_rate_deg_s=float(np.rad2deg(y[5])),
        )
        if all(checks.values()):
            if self.separation_ready_since_s is None:
                self.separation_ready_since_s = self.time_s
            if (
                self.time_s - self.separation_ready_since_s
                >= self.ascent_settings.separation_dwell_s
            ):
                self.separate()
                return
        else:
            self.separation_ready_since_s = None
        if y[1] > self.ascent_settings.separation_altitude_m + 100.0:
            self.outcome = "separation_gate_failed"
            self.failures.append(self.outcome)
            self.failures.extend(
                f"separation_{name}" for name, passed in checks.items() if not passed
            )

    def inertia_rate(self, y, mass_rate, deploy_rate):
        properties = self.properties(y)
        if not self.upper_attached:
            _, _, dm, dd = mass_gradients(self.mass(y), y[8], self.physics, properties)
            return dm * mass_rate + dd * deploy_rate
        # Analytic tank-top derivative including the passive upper-stage point mass.
        top = self.physics.geometry.tank_bottom_m + properties.fuel_height_m
        return mass_rate * (
            self.physics.geometry.body_radius_m**2 / 4.0 + (top - properties.cg_m) ** 2
        )

    def update_fin_plan(self, y, height):
        if self.fin_mode != "active" or self.upper_attached:
            return
        s = self.return_settings
        if self.contact is not None or (
            self.phase in ("landing", "alignment", "approach") and height <= s.fin_retract_height_m
        ):
            if not self.fin_retraction_started:
                self.fin_retraction_started_s = self.time_s
            self.fin_retraction_started = True
        if (
            not self.fin_deployment_started
            and not self.fin_retraction_started
            and self.phase == "coast"
            and -y[3] >= s.fin_deploy_min_downspeed_m_s
            and abs(self.alpha_deg(y, predicted=True)) <= s.fin_deploy_max_alpha_deg
            and abs(np.rad2deg(y[5])) <= s.fin_deploy_max_rate_deg_s
        ):
            self.fin_deployment_started = True
            self.fin_deployment_started_s = self.time_s

    def fin_extension_rate(self, y):
        if self.fin_retraction_started:
            return -1.0 / FINS.retraction_time_s if y[12] > 0.0 else 0.0
        if self.fin_deployment_started and y[12] < 1.0:
            return 1.0 / FINS.deployment_time_s
        return 0.0

    def derivative(self, y, control, *, contact_enabled=True, time_s=None):
        throttle, gimbal, torque, fin_target = control
        if y[6] <= 0.0:
            throttle = 0.0
        if y[7] <= 0.0:
            torque = 0.0
        thrust, main_flow = self.engine(throttle, y[1], actual=True)
        rcs_flow = abs(torque) / (self.physics.geometry.rcs_arm_m * self.design.rcs_isp_s * G0)
        mass_rate = -main_flow - rcs_flow
        deployment_rate = (
            1.0 / self.physics.gear.deployment_time_s
            if not self.upper_attached
            and y[1] < self.physics.gear.deployment_height_m
            and y[8] < 1.0
            else 0.0
        )
        force, aero_torque, _, _ = self.loads(y, time_s=time_s)
        force += thrust * np.array([sin(y[4] + gimbal), cos(y[4] + gimbal)])
        contacts = None
        if not self.upper_attached and contact_enabled and self.height(y) < 2.0:
            contacts = gear_contacts(
                self.vector(y), self.physics, mass_rate=mass_rate, deployment_rate=deployment_rate
            )
            force += contacts["forces"].sum(axis=0)
        mass = self.mass(y)
        acceleration = force / mass - np.array([0.0, gravity(y[1])])
        omega_n = 7.292115e-5 * cos(np.deg2rad(self.physics.atmosphere.latitude_deg))
        acceleration += 2.0 * omega_n * np.array([-y[3], y[2]])
        properties = self.properties(y)
        arm = properties.cg_m - self.physics.geometry.nozzle_station_m
        angular_flux = -main_flow * arm**2 * y[5]
        angular_flux -= (
            rcs_flow
            * (
                self.physics.geometry.rcs_arm_m**2
                + (self.physics.geometry.rcs_station_m - properties.cg_m) ** 2
            )
            * y[5]
        )
        net_torque = -arm * thrust * sin(gimbal) + torque + aero_torque
        if contacts is not None:
            net_torque += contacts["torque"]
        scale = self.disturbances.inertia_scale
        angular_acceleration = (
            net_torque
            + angular_flux
            - scale * self.inertia_rate(y, mass_rate, deployment_rate) * y[5]
        ) / (scale * properties.inertia_kg_m2)
        friction_rate = (
            np.where(contacts["normal"] > 0.0, contacts["velocities"][:, 0], 0.0)
            if contacts is not None
            else np.zeros(2)
        )
        return np.r_[
            y[2:4],
            acceleration,
            y[5],
            angular_acceleration,
            -main_flow,
            -rcs_flow,
            deployment_rate,
            friction_rate,
            float(
                np.clip(
                    (fin_target - y[11]) / FINS.actuator_time_constant_s,
                    -np.deg2rad(FINS.deflection_rate_deg_s),
                    np.deg2rad(FINS.deflection_rate_deg_s),
                )
            ),
            self.fin_extension_rate(y),
        ]

    def attitude(self, target, throttle, y=None):
        y = self.state if y is None else y
        error = atan2(sin(target - y[4]), cos(target - y[4]))
        rate_limit = 12.0
        if self.upper_attached:
            blend = self.preparation_fraction(y)
            rate_limit += blend * (self.ascent_settings.preparation_rate_limit_deg_s - rate_limit)
        if self.upper_attached:
            rate = float(np.clip(1.5 * error, -np.deg2rad(rate_limit), np.deg2rad(rate_limit)))
            alpha = float(np.clip(4.0 * (rate - y[5]), -np.deg2rad(4.5), np.deg2rad(4.5)))
        else:
            s = self.return_settings
            rate_limit = np.deg2rad(s.attitude_rate_limit_deg_s)
            rate = float(np.clip(s.attitude_position_gain_s_inv * error, -rate_limit, rate_limit))
            acceleration_limit = np.deg2rad(s.attitude_acceleration_limit_deg_s2)
            alpha = float(
                np.clip(
                    s.attitude_rate_gain_s_inv * (rate - y[5]),
                    -acceleration_limit,
                    acceleration_limit,
                )
            )
        thrust, flow = self.engine(throttle, y[1])
        properties = self.properties(y)
        lever = properties.cg_m - self.physics.geometry.nozzle_station_m
        aero_torque = self.body_loads(y, predicted=True)[1]
        flux = -flow * lever**2 * y[5]
        requested = properties.inertia_kg_m2 * alpha - aero_torque - flux
        requested += self.inertia_rate(y, -flow, 0.0) * y[5]
        limit = np.deg2rad(self.design.gimbal_limit_deg)
        gimbal_capacity = lever * thrust * sin(limit)
        # Fins lead unpowered descent; TVC leads powered flight. Fins only assist
        # beyond reserved TVC capacity, instead of chasing the same torque twice.
        reserve = self.return_settings.gimbal_capacity_fraction * gimbal_capacity
        assistance = requested - float(np.clip(requested, -reserve, reserve))
        fin_requested = self.fin_share * requested + (1.0 - self.fin_share) * assistance
        fin_target = self.fin_command(y, fin_requested)
        # Compensate the ACTUAL delayed fin torque, not an ideal commanded torque.
        requested -= self.fin_loads(y, predicted=True)[1]
        gimbal = float(
            np.clip(
                np.arcsin(np.clip(-requested / max(lever * thrust, 1e-12), -1.0, 1.0)),
                -limit,
                limit,
            )
        )
        supplied = -lever * thrust * sin(gimbal)
        rcs = float(
            np.clip(
                requested - supplied,
                -self.design.rcs_torque_limit_nm,
                self.design.rcs_torque_limit_nm,
            )
        )
        if self.contact is not None:
            return 0.0, 0.0, 0.0
        self.last_allocation = dict(
            desired_angular_acceleration_deg_s2=float(np.rad2deg(alpha)),
            fin_share=self.fin_share,
            fin_requested_torque_nm=fin_requested,
            actual_fin_torque_nm=self.fin_loads(y, predicted=True)[1],
            actual_gimbal_torque_nm=supplied,
            rcs_torque_nm=rcs,
        )
        return gimbal, rcs, fin_target

    def horizontal_acceleration(self, y):
        s = self.return_settings
        reference_x, reference_vx, reference_ax = (
            self.horizontal_transfer.reference(self.time_s)
            if self.horizontal_transfer is not None
            else (0.0, 0.0, 0.0)
        )
        lag_compensation = (
            self.horizontal_transfer.attitude_lag_compensation_s
            * self.horizontal_transfer.jerk(self.time_s)
            if self.horizontal_transfer is not None
            else 0.0
        )
        feed_forward = reference_ax + lag_compensation
        requested = (
            feed_forward
            - s.horizontal_kp_s2 * (y[0] - reference_x)
            - s.horizontal_kd_s_inv * (y[2] - reference_vx)
        )
        acceleration = float(
            np.clip(
                requested,
                -s.horizontal_acceleration_limit_m_s2,
                s.horizontal_acceleration_limit_m_s2,
            )
        )
        self.horizontal_command = dict(
            position_reference_m=reference_x,
            velocity_reference_m_s=reference_vx,
            acceleration_reference_m_s2=reference_ax,
            attitude_lag_compensation_m_s2=lag_compensation,
            acceleration_feed_forward_m_s2=feed_forward,
            requested_acceleration_m_s2=requested,
            limited_acceleration_m_s2=acceleration,
        )
        return acceleration

    def landing_tilt_limit(self, height):
        return float(np.deg2rad(np.interp(height, (200.0, 400.0, 600.0), (1.5, 4.0, 8.0))))

    def filter_attitude_reference(self, target):
        s = self.return_settings
        error = atan2(
            sin(target - self.filtered_attitude_target), cos(target - self.filtered_attitude_target)
        )
        change = -np.expm1(-s.control_dt_s / s.attitude_reference_tau_s) * error
        bound = np.deg2rad(s.attitude_reference_slew_deg_s) * s.control_dt_s
        self.filtered_attitude_target += float(np.clip(change, -bound, bound))
        return self.filtered_attitude_target

    def alignment_ready(self, y):
        s = self.return_settings
        return bool(
            abs(y[0]) <= s.alignment_position_gate_m
            and abs(y[2]) <= s.alignment_horizontal_speed_gate_m_s
            and abs(np.rad2deg(y[4])) <= s.alignment_tilt_gate_deg
            and abs(np.rad2deg(y[5])) <= s.alignment_rate_gate_deg_s
            and abs(y[3]) <= s.alignment_vertical_speed_gate_m_s
            and y[12] <= 1e-8
        )

    def update_alignment(self, y, height):
        s = self.return_settings
        if self.phase == "landing" and height <= s.alignment_trigger_height_m:
            self.phase = "alignment"
            self.alignment_started_s = self.time_s
            self.alignment_started_main_propellant_kg = float(y[6])
            if s.synchronized_alignment and self.horizontal_transfer is not None:
                remaining = (
                    self.horizontal_transfer.started_s
                    + self.horizontal_transfer.duration_s
                    - self.time_s
                )
                if s.joint_return_planning:
                    remaining = max(
                        remaining,
                        2.0
                        * max(-y[3] - s.alignment_end_speed_m_s, 0.0)
                        / s.alignment_reference_deceleration_m_s2,
                    )
                self.alignment_descent = AlignmentDescent.plan(
                    self.time_s, height, y[3], remaining, s
                )
        if self.phase != "alignment":
            return
        if not self.alignment_ready(y):
            self.alignment_ready_since_s = None
            return
        if self.alignment_ready_since_s is None:
            self.alignment_ready_since_s = self.time_s
        if self.time_s - self.alignment_ready_since_s >= s.alignment_dwell_s:
            actual = self.state
            actual_since = self.alignment_actual_ready_since_s
            if (
                self.disturbances.sensor_noise_scale == 0.0
                and self.disturbances.observation_delay_steps == 0
            ):
                actual_since = self.alignment_ready_since_s
            self.alignment_completed = dict(
                time_s=self.time_s,
                duration_s=self.time_s - self.alignment_started_s,
                main_propellant_used_kg=self.alignment_started_main_propellant_kg - float(y[6]),
                foot_height_m=self.height(actual),
                x_m=actual[0],
                vx_m_s=actual[2],
                vz_m_s=actual[3],
                attitude_deg=float(np.rad2deg(actual[4])),
                angular_rate_deg_s=float(np.rad2deg(actual[5])),
                dwell_s=self.time_s - self.alignment_ready_since_s,
                actual_gates_passed=self.alignment_ready(actual),
                actual_dwell_s=0.0 if actual_since is None else self.time_s - actual_since,
                observed_state=y.tolist(),
            )
            self.phase = "approach"
            self.approach_speed_reference = max(-y[3], 0.0)

    def alignment_acceleration(self, y, height):
        s = self.return_settings
        if self.alignment_descent is None:
            reference_height, reference_vz, feed_forward = s.alignment_hold_height_m, 0.0, 0.0
            mode = "altitude_hold_fallback"
        else:
            reference_height, reference_vz, feed_forward = self.alignment_descent.reference(
                self.time_s
            )
            mode = "synchronized_descent"
        self.vertical_command = dict(
            mode=mode,
            height_reference_m=reference_height,
            velocity_reference_m_s=reference_vz,
            acceleration_reference_m_s2=feed_forward,
        )
        return (
            feed_forward
            + s.alignment_vertical_kp_s2 * (reference_height - height)
            + s.alignment_vertical_kd_s_inv * (reference_vz - y[3])
        )

    def descent_command(self, y, height):
        s = self.return_settings
        speed = sqrt(
            self.design.descent_reference_m_s**2
            + 2.0 * s.descent_deceleration_m_s2 * max(height - s.terminal_flare_height_m, 0.0)
        )
        if self.phase == "approach":
            wanted = min(s.approach_speed_limit_m_s, speed)
            bound = s.approach_reference_acceleration_m_s2 * s.control_dt_s
            change = float(np.clip(wanted - self.approach_speed_reference, -bound, bound))
            self.approach_speed_reference += change
            reference, feed_forward = -self.approach_speed_reference, -change / s.control_dt_s
        else:
            reference = -speed
            feed_forward = (
                -s.descent_deceleration_m_s2 * y[3] / speed
                if height > s.terminal_flare_height_m
                else 0.0
            )
        return feed_forward + s.vertical_velocity_gain_s_inv * (reference - y[3])

    def governed_throttle(self, requested, y):
        # Triangle bound keeps |F_aero + F_engine| below the limit for EVERY
        # gimbal direction. Gravity is not part of the sensed support load.
        maximum, _ = self.engine(1.0, y[1])
        aero = self.loads(y, predicted=True)[0]
        cap = self.return_settings.load_governor_g * self.mass(y) * G0
        allowed = max(0.0, cap - hypot(*aero)) / maximum
        throttle = float(np.clip(requested, 0.0, min(1.0, allowed)))
        return throttle

    def joint_ignition_height(self, y, downspeed, x_m, vx_m_s):
        """Stopping margin plus the height needed by the SAME lateral horizon."""
        s = self.return_settings
        maximum, _ = self.engine(1.0, y[1])
        aero = self.loads(y, predicted=True)[0]
        available = (
            self.governed_throttle(1.0, y) * maximum * max(cos(y[4]), 0.5) / self.mass(y)
            + aero[1] / self.mass(y)
            - gravity(y[1])
        )
        deceleration = min(s.ignition_deceleration_m_s2, max(1.0, 0.6 * available))
        alignment_speed = sqrt(
            self.design.descent_reference_m_s**2
            + 2.0
            * s.descent_deceleration_m_s2
            * (s.alignment_trigger_height_m - s.terminal_flare_height_m)
        )
        brake_time = max(0.0, downspeed - alignment_speed) / deceleration
        brake_distance = max(0.0, downspeed**2 - alignment_speed**2) / (2.0 * deceleration)
        lateral = HorizontalTransfer.plan(self.time_s, x_m, vx_m_s, s)
        reaction_time = s.attitude_reference_tau_s + 1.0 / s.attitude_position_gain_s_inv
        reaction_time += s.return_plan_interval_s
        alignment_time_budget = (s.alignment_trigger_height_m - s.alignment_min_height_m) / (
            s.alignment_end_speed_m_s + (alignment_speed - s.alignment_end_speed_m_s) / 3.0
        )
        extra_time = max(
            0.0, lateral.duration_s + reaction_time - brake_time - alignment_time_budget
        )
        height = s.alignment_trigger_height_m + brake_distance + downspeed * reaction_time
        height += s.ignition_margin_m + 0.5 * gravity(y[1]) * reaction_time**2
        height += alignment_speed * extra_time
        return dict(
            ignition_height_m=height,
            braking_deceleration_m_s2=deceleration,
            braking_time_s=brake_time,
            horizontal_duration_s=lateral.duration_s,
            alignment_time_budget_s=alignment_time_budget,
            reaction_time_s=reaction_time,
            horizontal_extra_time_s=extra_time,
        )

    def update_return_plan(self, y):
        s = self.return_settings
        if self.return_plan is not None and self.time_s < self.next_return_plan_s:
            return
        h, mass = self.height(y), self.mass(y)
        area = self.physics.aerodynamics.axial_cd * pi * self.physics.geometry.body_radius_m**2
        if self.fin_mode == "active":
            area += FINS.installed_area_m2 * FINS.base_drag_coefficient
        candidates = s.ignition_range_candidates_m if self.phase == "boostback" else (None,)
        plans = []
        direction = 1.0 if y[0] >= 0.0 else -1.0
        for range_candidate in candidates:
            ignition_height = 2000.0
            for _ in range(4):
                estimate = CoastEstimate.between(
                    h,
                    y[3],
                    ignition_height,
                    mass,
                    area,
                    atmosphere_settings=self.physics.atmosphere,
                )
                forecast_wind = (
                    self.conditions.mean_wind(h, forecast=True)
                    + 4.0 * self.conditions.mean_wind((h + ignition_height) / 2.0, forecast=True)
                    + self.conditions.mean_wind(ignition_height, forecast=True)
                ) / 6.0
                if range_candidate is None:
                    desired_vx = None
                    ignition_x = (
                        y[0]
                        + forecast_wind * estimate.duration_s
                        + (y[2] - forecast_wind) * estimate.horizontal_gain_s
                    )
                    ignition_vx = (
                        forecast_wind + (y[2] - forecast_wind) * estimate.horizontal_velocity_ratio
                    )
                else:
                    ignition_x = direction * range_candidate
                    desired_vx = forecast_wind + (
                        ignition_x - y[0] - forecast_wind * estimate.duration_s
                    ) / max(estimate.horizontal_gain_s, 1.0)
                    ignition_vx = (
                        forecast_wind
                        + (desired_vx - forecast_wind) * estimate.horizontal_velocity_ratio
                    )
                budget = self.joint_ignition_height(
                    y, estimate.downspeed_m_s, ignition_x, ignition_vx
                )
                ignition_height = budget["ignition_height_m"]
            # The fixed-point iterations are closed-form budgets, not flight rollouts.
            minimum_align_time = (
                2.0
                * (
                    sqrt(
                        self.design.descent_reference_m_s**2
                        + 2.0
                        * s.descent_deceleration_m_s2
                        * (s.alignment_trigger_height_m - s.terminal_flare_height_m)
                    )
                    - s.alignment_end_speed_m_s
                )
                / s.alignment_reference_deceleration_m_s2
                + 2.0 * (s.attitude_reference_tau_s + 1.0 / s.attitude_position_gain_s_inv)
                + s.alignment_dwell_s
            )
            extra_powered_time = max(
                0.0, budget["horizontal_duration_s"] - budget["braking_time_s"] - minimum_align_time
            )
            additional_velocity = 0.0
            if desired_vx is not None:
                additional_velocity = max(0.0, direction * (y[2] - desired_vx))
            boost_fuel = (
                mass
                * additional_velocity
                / (self.design.sea_level_isp_s * G0 * sin(np.deg2rad(s.boostback_tilt_limit_deg)))
            )
            delay_fuel = (
                mass * gravity(y[1]) * extra_powered_time / (self.design.sea_level_isp_s * G0)
            )
            plans.append(
                dict(
                    time_s=self.time_s,
                    phase=self.phase,
                    candidate_count=len(candidates),
                    ignition_target_range_m=range_candidate,
                    return_vx_reference_m_s=desired_vx,
                    predicted_ignition_x_m=float(ignition_x),
                    predicted_ignition_vx_m_s=float(ignition_vx),
                    predicted_ignition_downspeed_m_s=estimate.downspeed_m_s,
                    forecast_mean_wind_m_s=forecast_wind,
                    coast_estimate=asdict(estimate),
                    **budget,
                    estimated_boost_propellant_kg=boost_fuel,
                    estimated_lateral_delay_propellant_kg=delay_fuel,
                    estimated_objective_propellant_kg=boost_fuel + delay_fuel,
                )
            )
        self.return_plan = min(plans, key=lambda plan: plan["estimated_objective_propellant_kg"])
        self.next_return_plan_s = self.time_s + s.return_plan_interval_s

    def guidance(self, observed=None):
        y = self.state if observed is None else observed
        mass = self.mass(y)
        g = gravity(y[1])
        maximum, _ = self.engine(1.0, y[1])
        aero = self.loads(y, predicted=True)[0]
        h = self.height(y)
        target = 0.0
        throttle = 0.0
        if self.phase == "ascent":
            target = self.ascent_attitude_target(y, h)
            # Slow before 10 km rather than forcing separation in high dynamic pressure.
            reference = self.ascent_vertical_speed_reference(h)
            az = 0.8 * (reference - y[3])
            force = max(0.0, mass * (g + az) - aero[1])
            throttle = float(np.clip(force / (maximum * max(cos(y[4]), 0.2)), 0.0, 1.0))
            # Feedback dynamic-pressure governor, with density-gradient feed-forward.
            density = atmosphere(y[1], self.physics.atmosphere)[2]
            v_limit = sqrt(2.0 * 22_000.0 / density)
            density_next = atmosphere(y[1] + 1.0, self.physics.atmosphere)[2]
            dv_limit_dz = sqrt(2.0 * 22_000.0 / density_next) - v_limit
            limit_az = dv_limit_dz * y[3] + 0.8 * (
                v_limit - hypot(y[2] - self.wind_field(predicted=True).at(y[1]), y[3])
            )
            q_throttle = (mass * (g + limit_az) - aero[1]) / (maximum * max(cos(y[4]), 0.2))
            throttle = min(throttle, max(0.0, float(q_throttle)))
        elif self.phase == "boostback":
            s = self.return_settings
            if s.joint_return_planning:
                self.update_return_plan(y)
                desired_vx = self.return_plan["return_vx_reference_m_s"]
            else:
                fall_time = (y[3] + sqrt(y[3] ** 2 + 2 * g * max(h, 0.0))) / g
                desired_vx = -y[0] / max(fall_time, 1.0)
            if (y[0] > 0.0 and y[2] <= desired_vx) or (y[0] < 0.0 and y[2] >= desired_vx):
                self.boostback_completed = dict(
                    time_s=self.time_s,
                    altitude_m=y[1],
                    x_m=y[0],
                    vx_m_s=y[2],
                    vz_m_s=y[3],
                    return_vx_reference_m_s=desired_vx,
                    reference_model="drag_and_ignition_budget"
                    if s.joint_return_planning
                    else "ballistic",
                    return_plan=self.return_plan,
                    main_propellant_kg=y[6],
                    rcs_propellant_kg=y[7],
                )
                self.phase = "coast"
                self.next_return_plan_s = 0.0
                target = 0.0
            else:
                s = self.return_settings
                throttle = s.boostback_throttle
                ax = float(
                    np.clip(
                        s.boostback_velocity_gain_s_inv * (desired_vx - y[2]),
                        -s.boostback_acceleration_limit_m_s2,
                        s.boostback_acceleration_limit_m_s2,
                    )
                )
                target = float(
                    np.clip(
                        np.arcsin(np.clip((mass * ax - aero[0]) / (throttle * maximum), -1, 1)),
                        -np.deg2rad(s.boostback_tilt_limit_deg),
                        np.deg2rad(s.boostback_tilt_limit_deg),
                    )
                )
                if s.joint_return_planning:
                    force_x = mass * ax - aero[0]
                    if abs(force_x) > 1e-8:
                        target = float(
                            np.copysign(
                                max(abs(target), np.deg2rad(s.boostback_min_burn_tilt_deg)), force_x
                            )
                        )
                    # Do not burn nearly upright while turning. Once oriented,
                    # request only the thrust needed for horizontal acceleration.
                    if (
                        abs(y[4]) < np.deg2rad(s.boostback_min_burn_tilt_deg)
                        or force_x * sin(y[4]) <= 0.0
                    ):
                        throttle = 0.0
                    else:
                        throttle = float(
                            np.clip(force_x / (maximum * sin(y[4])), 0.0, s.boostback_throttle)
                        )
        elif self.phase == "coast":
            s = self.return_settings
            if s.joint_return_planning:
                self.update_return_plan(y)
                budget = self.joint_ignition_height(
                    y,
                    max(-y[3], 0.0),
                    self.return_plan["predicted_ignition_x_m"],
                    self.return_plan["predicted_ignition_vx_m_s"],
                )
                ignition_height = max(
                    self.return_plan["ignition_height_m"], budget["ignition_height_m"]
                )
            else:
                available = self.governed_throttle(1.0, y) * maximum / mass + aero[1] / mass - g
                deceleration = min(s.ignition_deceleration_m_s2, max(1.0, 0.6 * available))
                ignition_height = max(0.0, y[3] ** 2 - 26.0**2) / (2 * deceleration)
                ignition_height += s.ignition_margin_m
            if y[3] < -s.fin_deploy_min_downspeed_m_s:
                target = float(
                    np.clip(
                        atan2(self.wind_field(predicted=True).at(y[1]) - y[2], -y[3]),
                        -np.deg2rad(8),
                        np.deg2rad(8),
                    )
                )
            if y[3] < 0.0 and h <= ignition_height:
                self.landing_ignition = dict(
                    time_s=self.time_s,
                    foot_height_m=h,
                    x_m=float(y[0]),
                    vx_m_s=float(y[2]),
                    vz_m_s=float(y[3]),
                    main_propellant_kg=float(y[6]),
                    rcs_propellant_kg=float(y[7]),
                    ignition_height_reference_m=ignition_height,
                    return_plan=self.return_plan,
                )
                self.phase = "landing"
        if self.phase in ("landing", "alignment", "approach"):
            s = self.return_settings
            if self.horizontal_transfer is None:
                self.horizontal_transfer = HorizontalTransfer.plan(self.time_s, y[0], y[2], s)
            self.update_alignment(y, h)
            ax = self.horizontal_acceleration(y)
            if self.phase == "alignment":
                az = self.alignment_acceleration(y, h)
            else:
                az = self.descent_command(y, h)
            fx = mass * ax - aero[0]
            fz = max(0.0, mass * (g + az) - aero[1])
            tilt_limit = self.landing_tilt_limit(h)
            # Respect the available vertical thrust. Do not turn a zero-upward
            # request into a horizontal-only burn with a 25-degree reference.
            fx = float(np.clip(fx, -fz * np.tan(tilt_limit), fz * np.tan(tilt_limit)))
            target = atan2(fx, max(fz, 1e-12))
            throttle = fz / (maximum * max(cos(y[4]), 0.5))
            if h < 200.0 and self.approach_started_s is None:
                self.approach_started_s = self.time_s
        if self.contact is not None:
            throttle, target = 0.0, 0.0
        if y[6] <= 0.0:
            throttle = 0.0
        if not self.upper_attached:
            self.update_fin_plan(y, h)
            target = self.filter_attitude_reference(target)
            wanted_share = 1.0 if self.phase == "coast" else 0.0
            s = self.return_settings
            self.fin_share += -np.expm1(-s.control_dt_s / s.fin_share_tau_s) * (
                wanted_share - self.fin_share
            )
            governed = self.governed_throttle(throttle, y)
            if governed < throttle - 1e-12:
                self.load_governor_activations += 1
            throttle = governed
        self.attitude_target = target
        gimbal, torque, fin_target = self.attitude(target, throttle, y)
        if self.phase in ("landing", "alignment", "approach"):
            # Reconcile vertical thrust with the actual body attitude AND gimbal.
            throttle = fz / (maximum * max(cos(y[4] + gimbal), 0.5))
            throttle = self.governed_throttle(throttle, y)
            if y[6] <= 0.0:
                throttle = 0.0
            gimbal, torque, fin_target = self.attitude(target, throttle, y)
        return throttle, gimbal, torque, fin_target

    def separate(self):
        actual_checks = self.separation_checks(self.state)
        actual_gate_altitude = float(self.state[1])
        before = self.properties(self.state)
        self.upper_attached = False
        after = self.properties(self.state)
        arm = rotation(self.state[4]) @ np.array([0.0, after.cg_m - before.cg_m])
        self.state[:2] += arm
        self.state[2:4] += self.state[5] * np.array([arm[1], -arm[0]])
        self.separation = dict(
            time_s=self.time_s,
            state=self.state.tolist(),
            engine_throttle=self.engine_throttle,
            altitude_m=self.state[1],
            x_m=self.state[0],
            vx_m_s=self.state[2],
            vz_m_s=self.state[3],
            first_stage_mass_kg=self.mass(self.state),
            main_propellant_kg=self.state[6],
            rcs_propellant_kg=self.state[7],
            dynamic_pressure_pa=self.loads(self.state)[2],
            attitude_deg=float(np.rad2deg(self.state[4])),
            angular_rate_deg_s=float(np.rad2deg(self.state[5])),
            alpha_deg=self.alpha_deg(self.state),
            after_observed_max_q=self.max_q_passed,
            gate_altitude_before_cg_change_m=actual_gate_altitude,
            gate_altitude_reference="attached_vehicle_center_of_mass",
            checks_before_separation=actual_checks,
            observed_checks_before_separation=self.last_separation_checks,
            preparation_started_s=self.ascent_preparation_started_s,
            simultaneous_conditions_dwell_s=self.time_s - self.separation_ready_since_s,
        )
        self.phase = "boostback" if abs(self.state[0]) > 2.0 else "coast"
        self.filtered_attitude_target = float(self.state[4])
        self.observations.clear()
        self.vertical_estimate = None
        self.sensor_estimator.estimate = None

    def first_contact(self, y, control):
        derivative = self.derivative(y, control, contact_enabled=False)
        points, velocities = foot_kinematics(
            self.vector(y), y[8], self.physics, mass_rate=derivative[6] + derivative[7]
        )
        active = points[:, 1] <= 1e-6
        self.contact = dict(
            time_s=self.time_s,
            state=y.tolist(),
            foot_positions_m=points.tolist(),
            foot_velocities_m_s=velocities.tolist(),
            first_feet=np.flatnonzero(active).tolist(),
        )
        c = self.physics.acceptance
        tests = dict(
            position=abs(y[0]) <= c.max_position_error_m,
            horizontal_speed=bool(
                np.all(abs(velocities[active, 0]) <= c.max_horizontal_contact_speed_m_s)
            ),
            vertical_speed=bool(
                np.all(abs(velocities[active, 1]) <= c.max_vertical_foot_speed_m_s)
            ),
            attitude=abs(np.rad2deg(atan2(sin(y[4]), cos(y[4])))) <= c.max_contact_tilt_deg,
            angular_rate=abs(np.rad2deg(y[5])) <= c.max_contact_rate_deg_s,
            main_propellant=y[6] > 0.0,
            rcs_propellant=y[7] >= 0.0,
            gear_locked=y[8] >= 1.0 - 1e-8,
            pad=bool(np.all(abs(points[active, 0]) <= self.physics.gear.pad_half_width_m)),
        )
        tests = {name: bool(passed) for name, passed in tests.items()}
        self.contact["criteria"] = tests
        self.failures.extend(f"touchdown_{name}" for name, passed in tests.items() if not passed)
        self.phase = "ground_settling"
        self.control = (0.0, 0.0, 0.0, 0.0)

    def monitor(self, y, control):
        derivative = self.derivative(y, control)
        force, aero_torque, q, _ = self.loads(y)
        thrust, _ = self.engine(control[0] if y[6] > 0.0 else 0.0, y[1], actual=True)
        force += thrust * np.array([sin(y[4] + control[1]), cos(y[4] + control[1])])
        load_g = hypot(*force) / (self.mass(y) * G0)
        self.max_q = max(self.max_q, q)
        self.max_x = max(self.max_x, abs(y[0]))
        self.max_altitude = max(self.max_altitude, y[1])
        alpha = abs(self.alpha_deg(y))
        rate = abs(float(np.rad2deg(y[5])))
        angular_acceleration = abs(float(np.rad2deg(derivative[5])))
        if self.upper_attached:
            self.max_q_before_separation = max(self.max_q_before_separation, q)
            self.update_max_q_passed(q)
            if (
                self.ascent_preparation_started_s is None
                and y[1] >= self.ascent_settings.preparation_start_altitude_m
            ):
                self.ascent_preparation_started_s = self.time_s
            self.max_q_alpha = max(self.max_q_alpha, q / 1000.0 * alpha)
            if q >= self.design.high_q_threshold_pa:
                self.max_ascent_alpha = max(self.max_ascent_alpha, alpha)
            self.max_ascent_rate = max(self.max_ascent_rate, rate)
            self.max_ascent_acceleration = max(self.max_ascent_acceleration, angular_acceleration)
            axial_g = abs(float((rotation(y[4]).T @ force)[1])) / (self.mass(y) * G0)
            self.max_axial_g = max(self.max_axial_g, axial_g)
        else:
            fin_force, fin_torque, hinge = self.fin_loads(y)
            self.max_fin_force_n = max(self.max_fin_force_n, hypot(*fin_force))
            self.max_fin_torque_nm = max(self.max_fin_torque_nm, abs(fin_torque))
            self.max_fin_hinge_nm = max(self.max_fin_hinge_nm, hinge)
            if y[12] > 0.01:
                temperature = atmosphere(y[1], self.physics.atmosphere)[0]
                mach = hypot(y[2] - self.wind_field().at(y[1]), y[3]) / sqrt(
                    1.4 * R_AIR * temperature
                )
                self.max_deployed_fin_mach = max(self.max_deployed_fin_mach, mach)
                self.max_deployed_fin_alpha_deg = max(
                    self.max_deployed_fin_alpha_deg, abs(self.alpha_deg(y))
                )
            self.max_recovery_g = max(self.max_recovery_g, load_g)
            if self.contact is None:
                if self.phase == "alignment":
                    if self.alignment_ready(y):
                        if self.alignment_actual_ready_since_s is None:
                            self.alignment_actual_ready_since_s = self.time_s
                    else:
                        self.alignment_actual_ready_since_s = None
                self.max_recovery_rate = max(self.max_recovery_rate, rate)
                self.max_recovery_acceleration = max(
                    self.max_recovery_acceleration, angular_acceleration
                )
                if self.approach_started_s is not None:
                    self.max_approach_tilt = max(self.max_approach_tilt, abs(np.rad2deg(y[4])))
            else:
                contacts = gear_contacts(self.vector(y), self.physics)
                self.peak_leg_force = np.maximum(self.peak_leg_force, contacts["normal"])
                self.peak_leg_stroke = np.maximum(self.peak_leg_stroke, contacts["stroke"])
                supported = force + contacts["forces"].sum(axis=0)
                contact_g = hypot(*supported) / (self.mass(y) * G0)
                self.max_recovery_g = max(self.max_recovery_g, contact_g)
                if np.any(contacts["stroke"] > self.physics.gear.max_stroke_m):
                    self.failures.append("leg_bottom_out")
                    self.outcome = "leg_bottom_out"
                if np.any(contacts["normal"] > self.physics.gear.max_normal_force_n):
                    self.failures.append("leg_normal_overload")
                    self.outcome = "leg_normal_overload"
                points = contacts["points"]
                g = self.physics.geometry
                props = self.properties(y)
                hull = np.array(
                    [
                        [-g.body_radius_m, -props.cg_m],
                        [g.body_radius_m, -props.cg_m],
                        [-g.nozzle_radius_m, g.nozzle_station_m - props.cg_m],
                        [g.nozzle_radius_m, g.nozzle_station_m - props.cg_m],
                        [-g.body_radius_m, g.body_length_m - props.cg_m],
                        [g.body_radius_m, g.body_length_m - props.cg_m],
                    ]
                )
                if float((hull @ rotation(y[4]).T)[:, 1].min() + y[1]) <= 0.0:
                    self.failures.append("body_ground_collision")
                    self.outcome = "body_ground_collision"
                settled = (
                    np.all(contacts["normal"] > 0.0)
                    and hypot(y[2], y[3]) <= self.physics.acceptance.settled_speed_m_s
                    and abs(y[5]) <= np.deg2rad(self.physics.acceptance.settled_rate_deg_s)
                    and abs(y[4]) <= np.deg2rad(self.physics.acceptance.max_contact_tilt_deg)
                    and abs(y[0]) <= self.physics.acceptance.max_position_error_m
                )
                if settled and self.settled_since is None:
                    self.settled_since = self.time_s
                if not settled:
                    self.settled_since = None
                acceptance = self.physics.acceptance
                if (
                    self.outcome == "running"
                    and self.time_s - self.contact["time_s"]
                    >= acceptance.observation_after_contact_s
                ):
                    if (
                        self.settled_since is not None
                        and self.time_s - self.settled_since >= acceptance.settled_dwell_s
                    ):
                        self.outcome = "landed"
                    elif self.time_s - self.contact["time_s"] >= acceptance.max_post_contact_s:
                        self.outcome = "post_contact_unstable"
                        self.failures.append(self.outcome)
                if np.any(abs(points[contacts["active"], 0]) > self.physics.gear.pad_half_width_m):
                    self.failures.append("contact_outside_pad")
                    self.outcome = "contact_outside_pad"
        balance = sum(self.initial_propellant)
        balance -= y[6] + y[7] + self.main_consumed + self.rcs_consumed
        self.max_mass_balance_error = max(self.max_mass_balance_error, abs(balance))

    def advance(self):
        """Advance one integration interval, preserving sensors, delays and fuel."""
        if self.outcome != "running" or self.time_s >= self.deadline_s:
            return
        if self.time_s >= self.next_control_s - 1e-10:
            self.requested_control = self.guidance(self.observe_control_state())
            self.action_queue.append(self.requested_control)
            self.control = self.action_queue[0]
            if self.contact is not None:
                self.control = (0.0, 0.0, 0.0, 0.0)
            self.next_control_s += self.return_settings.control_dt_s
        if self.time_s >= self.next_record_s - 1e-10:
            self.steps.append(
                dict(
                    time_s=self.time_s,
                    phase=self.phase,
                    state=self.state.tolist(),
                    control=list(self.control),
                    requested_control=list(self.requested_control),
                    engine_throttle=self.engine_throttle,
                    observed_state=self.observed_state.tolist(),
                    wind_m_s=self.wind_field().at(self.state[1]),
                    mass_kg=self.mass(self.state),
                    dynamic_pressure_pa=self.loads(self.state)[2],
                    ascent_command=self.ascent_command if self.upper_attached else None,
                    attitude_target_deg=float(np.rad2deg(self.attitude_target)),
                    horizontal_command=self.horizontal_command,
                    vertical_command=self.vertical_command if self.phase == "alignment" else None,
                    fin_torque_nm=self.fin_loads(self.state)[1],
                    control_allocation=self.last_allocation,
                )
            )
            self.next_record_s += 0.1
        y = self.state
        h = 0.001 if not self.upper_attached and self.height(y) < 2.0 else 0.01
        h = min(h, self.next_control_s - self.time_s, self.deadline_s - self.time_s)
        if h < 1e-10:
            self.time_s = self.next_control_s
            return
        initial_throttle = self.engine_throttle
        started_s = self.time_s
        command = self.control
        flow = self.derivative(
            y,
            self.applied_control(started_s, command, started_s, initial_throttle),
            contact_enabled=False,
        )
        for index in (6, 7):
            if flow[index] < 0.0:
                h = min(h, max(y[index], 0.0) / (-flow[index]))
        if h <= 1e-12:
            self.outcome = "propellant_depletion"
            self.failures.append(self.outcome)
            return

        def rhs(_time, value):
            return self.derivative(
                value,
                self.applied_control(_time, command, started_s, initial_throttle),
                time_s=_time,
            )

        advanced = rk4_step(rhs, self.time_s, y, h)
        depleted = [index for index in (6, 7) if y[index] > 0.0 and advanced[index] < 0.0]
        if depleted:
            h = min(
                brentq(
                    lambda dt, index=index: (
                        rk4_step(rhs, self.time_s, y, dt)[index] if dt else y[index]
                    ),
                    0.0,
                    h,
                    xtol=1e-12,
                )
                for index in depleted
            )
            advanced = rk4_step(rhs, self.time_s, y, h)
            for index in depleted:
                if abs(advanced[index]) < 1e-8:
                    advanced[index] = 0.0
        # Exact first foot contact, using the unforced flight limit from above.
        if (
            not self.upper_attached
            and self.contact is None
            and self.height(y) > 0.0
            and self.height(advanced) <= 0.0
        ):

            def flight_rhs(_time, value):
                return self.derivative(
                    value,
                    self.applied_control(_time, command, started_s, initial_throttle),
                    contact_enabled=False,
                    time_s=_time,
                )

            event = brentq(
                lambda dt, y=y, step=flight_rhs, time=self.time_s: (
                    self.height(rk4_step(step, time, y, dt)) if dt else self.height(y)
                ),
                0.0,
                h,
                xtol=1e-12,
            )
            advanced = rk4_step(flight_rhs, self.time_s, y, event)
            h = event
            self.time_s += h
            actual_control = self.applied_control(self.time_s, command, started_s, initial_throttle)
            self.engine_throttle = actual_control[0]
            self.state = advanced
            self.main_consumed += y[6] - advanced[6]
            self.rcs_consumed += y[7] - advanced[7]
            self.first_contact(advanced, actual_control)
            self.monitor(advanced, actual_control)
            return
        self.time_s += h
        actual_control = self.applied_control(self.time_s, command, started_s, initial_throttle)
        self.engine_throttle = actual_control[0]
        self.main_consumed += y[6] - advanced[6]
        self.rcs_consumed += y[7] - advanced[7]
        advanced[6:8] = np.maximum(advanced[6:8], 0.0)
        advanced[8] = float(np.clip(advanced[8], 0.0, 1.0))
        advanced[11] = float(
            np.clip(
                advanced[11],
                -np.deg2rad(FINS.deflection_limit_deg),
                np.deg2rad(FINS.deflection_limit_deg),
            )
        )
        advanced[12] = float(np.clip(advanced[12], 0.0, 1.0))
        if not self.upper_attached:
            contacts = gear_contacts(self.vector(advanced), self.physics)
            for leg in range(2):
                if contacts["normal"][leg] <= 0.0:
                    advanced[9 + leg] = 0.0
                elif contacts["slipping"][leg]:
                    bound = (
                        abs(contacts["tangent"][leg]) / self.physics.gear.tangential_stiffness_n_m
                    )
                    advanced[9 + leg] = np.clip(advanced[9 + leg], -bound, bound)
        self.state = advanced
        self.monitor(advanced, actual_control)
        if self.upper_attached:
            if (
                self.disturbances.sensor_noise_scale == 0.0
                and self.disturbances.observation_delay_steps == 0
            ):
                self.consider_separation(advanced)
            else:
                self.consider_separation(self.observed_state, predicted=True)

    def run(self):
        while self.outcome == "running" and self.time_s < self.deadline_s:
            self.advance()
        if self.outcome == "running":
            self.outcome = "timeout"
            self.failures.append(self.outcome)
        self.steps.append(
            dict(
                time_s=self.time_s,
                phase=self.phase,
                state=self.state.tolist(),
                control=list(self.control),
                mass_kg=self.mass(self.state),
                dynamic_pressure_pa=self.loads(self.state)[2],
                attitude_target_deg=float(np.rad2deg(self.attitude_target)),
                fin_torque_nm=self.fin_loads(self.state)[1],
            )
        )
        criteria = dict(
            surface_wind=True,
            weather_profile_domain=self.max_altitude <= PROFILE_TOP_M,
            separation=self.separation is not None
            and all(self.separation["checks_before_separation"].values()),
            ascent_q=self.max_q_before_separation <= 30_000.0,
            ascent_q_alpha=self.max_q_alpha <= 150.0,
            ascent_high_q_alpha=self.max_ascent_alpha <= 5.0,
            ascent_angular_rate=self.max_ascent_rate <= 15.0,
            ascent_angular_acceleration=self.max_ascent_acceleration <= 5.0,
            ascent_axial_acceleration=self.max_axial_g <= 3.0,
            recovery_normal_acceleration=self.max_recovery_g <= 5.0,
            recovery_absolute_acceleration=self.max_recovery_g <= 6.0,
            recovery_angular_rate=self.max_recovery_rate <= 15.0,
            recovery_angular_acceleration=self.max_recovery_acceleration <= 5.0,
            approach_attitude=self.max_approach_tilt <= 5.0,
            touchdown=self.contact is not None and all(self.contact["criteria"].values()),
            engine_off_standing=self.outcome == "landed",
            pre_approach_alignment=(
                self.alignment_completed is not None
                and self.alignment_completed["foot_height_m"] > 200.0
                and self.alignment_completed["actual_gates_passed"]
                and self.alignment_completed["actual_dwell_s"]
                >= self.return_settings.alignment_dwell_s
            ),
            mass_balance=self.max_mass_balance_error <= 1e-6,
            fin_hinge_load=self.max_fin_hinge_nm <= FINS.per_fin_hinge_limit_nm,
            fin_model_envelope=(
                self.max_deployed_fin_mach <= FINS.model_max_mach
                and self.max_deployed_fin_alpha_deg <= FINS.control_max_alpha_deg
            ),
        )
        criteria = {name: bool(passed) for name, passed in criteria.items()}
        not_evaluated = []
        if self.scope == "10km_separation_to_settled_landing":
            not_evaluated.extend(name for name in criteria if name.startswith("ascent_"))
        if self.separation is None:
            not_evaluated.extend(
                name
                for name in criteria
                if name.startswith("recovery_") or name == "approach_attitude"
            )
        for name in not_evaluated:
            criteria[name] = None
        return dict(
            description=(
                "3톤 APR 강체 회수 계산. 입력 허용 판정은 실제 발사 승인·실기체 검증이 아닙니다."
            ),
            environment_version=OPERATING_SCOPE,
            simulation_scope=self.scope,
            ascent_simulated=self.scope == "launch_to_separation_to_settled_landing",
            operating_input_eligible=True,
            seed=378,
            simulation_count=1,
            return_method=RETURN_METHOD if self.fin_mode == "active" else "PID, fins stowed",
            fin_mode=self.fin_mode,
            fin_settings=asdict(FINS),
            dry_mass_budget_kg=dict(
                engine=110.0,
                landing_gear=60.0,
                other_airframe_systems=326.0,
                fin_surfaces=FINS.surface_mass_kg,
                fin_actuators=FINS.actuator_mass_kg,
                fin_deployment=FINS.deployment_mass_kg,
                fin_electrical=FINS.electrical_mass_kg,
            ),
            design=asdict(self.design),
            upper_stage_mission=dict(
                payload_mass_kg=self.design.payload_mass_kg,
                stage_wet_mass_kg=self.design.upper_stage_wet_mass_kg,
                target_orbit_altitude_m=ASSUMED_UPPER_STAGE_ORBIT_ALTITUDE_M,
                orbit_insertion_assumed=True,
                orbit_insertion_simulated=False,
                orbit_insertion_verified=False,
            ),
            first_stage_mission=dict(
                separation_altitude_m=self.ascent_settings.separation_altitude_m,
                separation_attitude_from_vertical_deg=self.ascent_settings.separation_attitude_deg,
                separation_attitude_tolerance_deg=self.ascent_settings.separation_attitude_tolerance_deg,
                landing_target_x_m=0.0,
                horizontal_distance_prescribed=False,
                horizontal_velocity_prescribed=False,
            ),
            physics=asdict(self.physics),
            ascent_guidance=asdict(self.ascent_settings),
            return_guidance=asdict(self.return_settings),
            initial_twr=self.design.sea_level_thrust_n / (self.design.launch_mass_kg * G0),
            external_disturbances=dict(
                environment=asdict(self.conditions), healthy_vehicle=asdict(self.disturbances)
            ),
            initial_propellant_kg=dict(
                main=self.initial_propellant[0], rcs=self.initial_propellant[1]
            ),
            unconnected_systems=list(UNCONNECTED_SYSTEMS),
            legacy_evidence_transferred=False,
            assumptions=[
                "주 엔진 비추력 260초, RCS 비추력 220초는 기존 모델에서 가져온 가정입니다.",
                "주 엔진은 지상 최대 42 kN에 고도별 압력 추력을 더하며 0-100% 조절합니다. "
                "입력한 1차 응답 지연은 반영하지만 "
                "최소 20% 스로틀·시동/재점화 제약은 적용하지 않습니다.",
                "실제 물리는 고도별 바람·정현파 돌풍·온도·기압·추력/관성 편차를 사용합니다. "
                "제어기는 예보 바람과 인과적 센서 추정만 사용하며 실제 상태로 성공을 판정합니다.",
                "바람은 평면 내 수평 성분이며 0-20 km의 선형 프로파일입니다. "
                "입력 경계는 잠정 프로젝트 범위이며 NASA의 이 기체 인증 기준이 아닙니다.",
                "잡음은 위치·속도·자세·각속도에 적용합니다. 연료·기구 상태·실제 스로틀·"
                "Max-Q 계측은 이상적인 원격 계측으로 가정합니다. "
                "낙뢰·구름·기체 검토는 입력 게이트입니다.",
                "1단 형상 5 m x 지름 0.9 m, 결합체 7 m, 2단 질량 중심 위치 6 m를 가정합니다.",
                "상부 총질량 1,015 kg 안에 탑재체 30 kg을 포함하며, "
                "추진제를 포함한 2단 자체 질량은 985 kg입니다.",
                "분리 후 2단의 700 km 원궤도 도달은 임무 가정입니다. "
                "2단 비행을 계산하거나 궤도 투입 성공을 검증하지 않습니다.",
                "건조질량 520 kg 안에서 엔진 110 kg·다리 60 kg·"
                "기타 구조와 계통 326 kg·핀과 구동계 24 kg로 잠정 배분했습니다.",
                "핀 질량·면적·계수·힌지 허용치는 설계 가정이며 "
                "실측된 격자핀 또는 Falcon 9의 공력 데이터가 아닙니다.",
                "4개 핀 총 면적 0.8 m² 중 평면 내 한 쌍의 유효 면적은 0.4 m²입니다. "
                "전개와 재접기는 각각 2초, 조종각은 ±20도, 구동 속도는 60도/s입니다.",
                "핀은 분리 후 하강 때 전개합니다. 구동 지연과 항력은 반영하지만 "
                "전개 중 질량 분포 변화는 평균 반경으로 근사합니다.",
                "전개기구가 기존 24 kg 예산 안에서 양방향 구동·재잠금을 할 수 있다고 "
                "추가 가정합니다. 실제 재접기 하중·구동 전력·기구 강도는 미검증입니다.",
                "핀은 무추력 하강을 담당하고 엔진이 켜지면 짐벌 우선으로 전환합니다. "
                "실제 지연된 핀 토크를 보상하며 RCS는 남은 토크만 제공합니다.",
                "추력과 공력 합산 하중을 4.75 g 목표로 제한합니다. "
                "5 g 정상 기준·6 g 절대 기준·접지 기준은 변경하지 않습니다.",
                "수평 PD·자세 참조 필터·점화 제동과 접근 하강 속도 램프를 적용합니다. "
                "중간 접근 상한은 20 m/s이고 최종 접지 참조는 0.9 m/s입니다.",
                "착륙 점화 시 관측한 수평 위치·속도로 5차 목표 궤적을 계획합니다. "
                "가속도 참조 0.8 m/s²·저크 참조 0.2 m/s³ 이내에서 원점과 정지 상태를 "
                "함께 목표로 하며 기존 PID가 오차를 보정합니다.",
                "참조 필터와 자세 PD의 저주파 추종 지연을 1차 선행 보상합니다. "
                "보상된 가속도도 참조 예산 0.8 m/s² 이내로 제한하며 "
                "실제 자세는 기존 속도·가속도 한계를 지켜 적분합니다.",
                "600 m부터 수평 참조 종료 시각에 맞춰 단조 하강·감속 참조를 계획합니다. "
                "참조 순가속도 상한 8 m/s²·최저 유지 높이 250 m를 적용하고 "
                "계획이 불가능하면 기존 350 m 고도 유지로 돌아갑니다.",
                "정렬의 실제 위치·속도·자세·각속도·핀 재접기·1초 유지 조건과 "
                "접지 기준은 완화하지 않습니다. 전역 연료 최소 최적해는 아닙니다.",
                "추가 5도/s² 진단은 이전과 같이 회수 전체에 적용하며, "
                "사용자가 정한 Flip 구간만의 한계보다 보수적인 평가입니다.",
                "RCS와 주 추진제의 소비·잔량은 별도 계산하며 "
                "질량 분포는 동일 위치의 등가 액체로 근사합니다.",
                "500 m 이후 작은 각도 중력 선회 참조와 공기 상대속도 피드백으로 "
                "수직 기준 9도 분리를 목표로 합니다. 수평거리·속도는 비행 결과입니다.",
                "상승 수평속도 유도 예산 25 m/s·수평거리 유도 예산 600 m를 적용하며 "
                "실제 속도·자세·좌표를 강제로 맞추지 않습니다.",
                "분리 자세는 수직 기준 9±0.5도이며 기존 받음각·각속도·동압 조건도 "
                "동시에 만족해야 합니다.",
                "귀환 제동은 최대 스로틀 15%·수평가속도 참조 최대 2 m/s²·자세 참조 최대 "
                "20도를 사용합니다. 공동 계획 모드는 실제 기울기 8도 미만에서 "
                "불필요한 수직 추력을 끄고, 이후 필요한 수평 힘만 요청합니다.",
                "공동 계획 모드는 평균 대기밀도·축방향 항력·핀 기본 항력의 닫힌 근사식으로 "
                "귀환 속도 후보와 점화 고도를 함께 정합니다. 미래 비행을 추가 적분하지 않습니다.",
                "근사 비용은 귀환 제동 추진제와 추가 정렬 체류 추진제의 추정값입니다. "
                "이 계획은 횡공력·핀 양력·밀도 변화·코리올리와 점화 전 질량 변화를 "
                "정확히 예측하지 않으며 실제 비행은 기존 물리식으로 적분합니다.",
                "점화 고도에는 실제 제동 여유·자세 지연·계획 갱신 지연과 "
                "수평 정렬 시간 예산을 반영합니다. 전역 연료 최적해를 주장하지 않습니다.",
                "8.5-9 km에서 분리 준비로 전환하며 "
                "받음각·각속도 목표 2도·2도/s 이내 여유를 둡니다.",
                "분리 허용 범위는 결합체 질량 중심 기준 9900-10100 m이며 "
                "모든 조건을 0.3초 유지해야 합니다.",
                "Max-Q 통과는 관측 피크 대비 1% 이상 낮아지면서 "
                "0.5초 연속 감소하는 인과적 근사입니다.",
                "고동압 기준 10 kPa 이상, 분리 후 부하·착륙 기준과 기체 제원은 "
                "이전 실험과 같습니다.",
                "접지 상한 1.0 m/s를 위한 최종 하강 속도 참조는 0.9 m/s입니다.",
                "분리 후 질량 중심 변화와 회전 운동학을 반영하지만 분리 충격은 0으로 가정합니다.",
                "핫스테이징 상태 조건만 평가하며 "
                "2단 실제 점화·플룸·분리 기구는 구현하지 않았습니다.",
                "공급·가압·액체 슬로싱·탄성·상세 열해석을 제외하므로 "
                "기존 systems-v1 검증과 구별해야 합니다.",
                "기존 스프링·댐퍼·마찰 접지를 사용하되 "
                "다리 재료·좌굴·응력 인증은 이 예비 모델의 범위 밖입니다.",
            ],
            separation=self.separation,
            boostback_completed=self.boostback_completed,
            landing_ignition=self.landing_ignition,
            return_plan=self.return_plan,
            last_separation_checks=self.last_separation_checks,
            last_separation_check_state=self.last_separation_check_state,
            first_contact=self.contact,
            simulated_acceptance_passed=all(
                passed is True for name, passed in criteria.items() if name not in not_evaluated
            ),
            criteria=criteria,
            not_evaluated_criteria=not_evaluated,
            outcome=self.outcome,
            failure_reasons=sorted(set(self.failures)),
            failed_criteria=[name for name, passed in criteria.items() if passed is False],
            flight_time_s=self.time_s,
            elapsed_time_s=self.time_s - self.started_s,
            max_altitude_m=self.max_altitude,
            max_horizontal_distance_m=self.max_x,
            max_ascent_q_pa=self.max_q_before_separation,
            max_ascent_q_alpha_kpa_deg=self.max_q_alpha,
            max_ascent_high_q_alpha_deg=self.max_ascent_alpha,
            max_ascent_rate_deg_s=self.max_ascent_rate,
            max_ascent_angular_acceleration_deg_s2=self.max_ascent_acceleration,
            max_ascent_axial_g=self.max_axial_g,
            max_recovery_g=self.max_recovery_g,
            max_recovery_rate_deg_s=self.max_recovery_rate,
            max_recovery_angular_acceleration_deg_s2=self.max_recovery_acceleration,
            max_approach_tilt_deg=self.max_approach_tilt,
            alignment_started_s=self.alignment_started_s,
            alignment_completed=self.alignment_completed,
            alignment_descent=asdict(self.alignment_descent)
            if self.alignment_descent is not None
            else None,
            horizontal_transfer=asdict(self.horizontal_transfer)
            if self.horizontal_transfer is not None
            else None,
            fin_deployment_started_s=self.fin_deployment_started_s,
            fin_retraction_started_s=self.fin_retraction_started_s,
            load_governor_activations=self.load_governor_activations,
            peak_leg_force_n=self.peak_leg_force.tolist(),
            peak_leg_stroke_m=self.peak_leg_stroke.tolist(),
            main_propellant_used_kg=self.main_consumed,
            rcs_propellant_used_kg=self.rcs_consumed,
            remaining_main_propellant_kg=self.state[6],
            remaining_rcs_propellant_kg=self.state[7],
            max_mass_balance_error_kg=self.max_mass_balance_error,
            max_fin_force_n=self.max_fin_force_n,
            max_fin_torque_nm=self.max_fin_torque_nm,
            max_fin_hinge_torque_nm=self.max_fin_hinge_nm,
            max_deployed_fin_mach=self.max_deployed_fin_mach,
            max_deployed_fin_alpha_deg=self.max_deployed_fin_alpha_deg,
            fin_model_envelope_passed=(
                self.max_deployed_fin_mach <= FINS.model_max_mach
                and self.max_deployed_fin_alpha_deg <= FINS.control_max_alpha_deg
            ),
            development_only=True,
            state_names=[
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
            steps=self.steps,
        )
