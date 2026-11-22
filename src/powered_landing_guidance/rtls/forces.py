"""High-speed scalar float64 aerodynamics, propulsion, and mass-property models.

Designed as a pure force/torque module that:
  1. Eliminates per-step dataclass allocation and Python loop overhead (~20x faster).
  2. Separates nose-first ascent drag from tail-first blunt-base descent drag and
     models realistic grid-fin effective angle of attack (alpha_eff = delta - alpha_retro).
  3. Enforces Reusable multi-engine cluster throttling (7 x 221.4 kN = 1,550 kN SL)
     with realistic per-engine minimum throttle (non-hovering Hoverslam on 1 engine)
     and bounded restart count.
  4. Can be plugged directly into MuJoCo's `xfrc_applied` callback in Stage 2 (3D).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, exp, hypot, pi, sin, sqrt

import numpy as np

from powered_landing_guidance.apr_recovery import (
    FINS,
    PHYSICS,
    PROFILE_TOP_M,
    AprConditions,
    Design,
)
from powered_landing_guidance.recovery_physics import (
    EARTH_RADIUS_M,
    G0,
    R_AIR,
    AtmosphereSettings,
    Geometry,
    LandingGear,
    MassProperties,
    atmosphere,
)
from powered_landing_guidance.rtls.interfaces import ForceTorqueSample

OMEGA_EARTH = 7.292115e-5
SUTTON_GRAVES_EARTH_K = 1.7415e-4


def compute_stagnation_heat_flux_w_m2(
    rho_kg_m3: float,
    speed_m_s: float,
    nose_radius_m: float = 1.55,
) -> float:
    """Sutton-Graves convective stagnation heat flux [W/m^2] for Earth entry."""
    if rho_kg_m3 <= 0.0 or speed_m_s <= 0.0:
        return 0.0
    return SUTTON_GRAVES_EARTH_K * sqrt(rho_kg_m3 / max(nose_radius_m, 0.1)) * (speed_m_s**3)


@dataclass(frozen=True, slots=True)
class RtlsAerodynamics:
    """Planar aerodynamics separating nose-first ascent from tail-first reentry.

    During nose-first ascent (v_axial > 0), the payload fairing yields low
    axial drag (`axial_cd_ascent = 0.4`).
    During tail-first descent (v_axial < 0), the blunt engine skirt, nozzles,
    and open interstage cavity produce substantially higher base drag
    (`axial_cd_descent = 1.4`) plus slender-body potential normal force
    (`slender_lift_slope = 2.0`, NASA Jorgensen cylinder value) for smooth
    zero-deadband aerodynamic gliding.
    """

    axial_cd_ascent: float = 0.4
    axial_cd_descent: float = 1.65
    normal_cd: float = 0.8
    slender_lift_slope: float = 2.0
    pressure_center_ascent_m: float = 11.5
    pressure_center_descent_m: float = 11.0
    rotation_damping: float = 0.3
    transonic_drag_rise: float = 0.2


@dataclass(frozen=True, slots=True)
class RtlsFinSettings:
    """Reusable grid-fin parameters with effective angle-of-attack physics."""

    surface_mass_kg: float = FINS.surface_mass_kg
    actuator_mass_kg: float = FINS.actuator_mass_kg
    deployment_mass_kg: float = FINS.deployment_mass_kg
    electrical_mass_kg: float = FINS.electrical_mass_kg
    station_m: float = FINS.station_m
    mean_mass_radius_m: float = FINS.mean_mass_radius_m
    installed_area_m2: float = FINS.installed_area_m2
    planar_effective_area_m2: float = FINS.planar_effective_area_m2
    lift_slope_rad_inv: float = 2.8
    max_lift_coefficient: float = 1.0
    base_drag_coefficient: float = 0.35
    induced_drag_factor: float = 0.50
    deflection_limit_deg: float = 20.0
    deflection_rate_deg_s: float = 60.0
    actuator_time_constant_s: float = 0.08
    deployment_time_s: float = 2.0
    retraction_time_s: float = 2.0
    control_min_q_pa: float = 80.0
    control_max_alpha_deg: float = 25.0
    model_max_mach: float = 8.0
    hinge_arm_m: float = 0.12
    per_fin_hinge_limit_nm: float = 7_500.0

    @property
    def mass_kg(self) -> float:
        return (
            self.surface_mass_kg
            + self.actuator_mass_kg
            + self.deployment_mass_kg
            + self.electrical_mass_kg
        )


@dataclass(frozen=True, slots=True)
class RtlsPropulsionSettings:
    """Reusable 7-engine hexagonal Methalox cluster (7 x 221.4 kN = 1,550 kN SL).

    Total thrust (1.55 MN SL / 1.693 MN vac), Isp (325 s SL / 355 s vac), and nozzle
    exit area match the 100 t Methalox orbital RTLS Design.
    Each individual engine throttles only within [min_engine_throttle, 1.0] or OFF.
    Landing on the 1 center engine at min_engine_throttle=0.42 yields 93.0 kN,
    giving TWR_min ~ 1.25 on an ~7.6 t booster -> true non-hovering Hoverslam.
    """

    engine_count: int = 7
    sea_level_thrust_n: float = 1_550_000.0
    sea_level_isp_s: float = 325.0
    nozzle_exit_area_m2: float = 1.4113
    min_engine_throttle: float = 0.36
    max_restarts: int = 3
    gimbal_limit_deg: float = 6.0
    gimbal_rate_deg_s: float = 60.0
    gimbal_tau_s: float = 0.06
    nominal_engine_tau_s: float = 0.04
    rcs_isp_s: float = 260.0
    rcs_torque_limit_nm: float = 100_000.0

    def throttle_bounds(self, active_engines: int) -> tuple[float, float]:
        """Allowed total equivalent throttle fraction for `active_engines` lit."""
        if active_engines <= 0:
            return (0.0, 0.0)
        frac = min(active_engines, self.engine_count) / self.engine_count
        return (self.min_engine_throttle * frac, frac)

    def quantize_throttle(self, requested_total: float, active_engines: int) -> float:
        """Enforce physical engine throttle floor [min, max] or 0 when off."""
        if active_engines <= 0 or requested_total <= 1e-6:
            return 0.0
        low, high = self.throttle_bounds(active_engines)
        if requested_total < low:
            return low
        if requested_total > high:
            return high
        return requested_total


class AtmosphereLUT:
    """Fast piecewise-linear/log-pressure LUT matching recovery_physics.atmosphere().

    Evaluates (T, P, rho) in ~0.2 us using pure scalar math instead of
    rebuilding temperature/pressure profiles on every RK4 substep.
    """

    __slots__ = ("_dz", "_inv_dz", "_max_h", "_n", "_t", "_log_p", "_rho")

    def __init__(self, settings: AtmosphereSettings, step_m: float = 100.0) -> None:
        self._dz = float(step_m)
        self._inv_dz = 1.0 / self._dz
        self._max_h = PROFILE_TOP_M
        n = int(round(self._max_h / self._dz)) + 1
        self._n = n
        temps = [0.0] * n
        log_ps = [0.0] * n
        rhos = [0.0] * n
        for i in range(n):
            h = i * self._dz
            t_k, p_pa, rho = atmosphere(h, settings)
            temps[i] = t_k
            log_ps[i] = float(np.log(max(p_pa, 1e-30)))
            rhos[i] = rho
        self._t = temps
        self._log_p = log_ps
        self._rho = rhos

    def at(self, altitude_m: float) -> tuple[float, float, float]:
        if altitude_m <= 0.0:
            return self._t[0], exp(self._log_p[0]), self._rho[0]
        if altitude_m >= self._max_h:
            last = self._n - 1
            t_top = self._t[last]
            extra = altitude_m - self._max_h
            p_top = exp(self._log_p[last] - G0 * extra / (R_AIR * t_top))
            rho_top = self._rho[last] * (p_top / exp(self._log_p[last]))
            return t_top, p_top, rho_top
        pos = altitude_m * self._inv_dz
        idx = int(pos)
        if idx >= self._n - 1:
            last = self._n - 1
            return self._t[last], exp(self._log_p[last]), self._rho[last]
        frac = pos - idx
        t0, t1 = self._t[idx], self._t[idx + 1]
        lp0, lp1 = self._log_p[idx], self._log_p[idx + 1]
        r0, r1 = self._rho[idx], self._rho[idx + 1]
        t_k = t0 + frac * (t1 - t0)
        p_pa = exp(lp0 + frac * (lp1 - lp0))
        rho = r0 * exp(frac * (np.log(max(r1, 1e-30)) - np.log(max(r0, 1e-30))))
        return t_k, p_pa, rho


class WindLUT:
    """Fast scalar evaluator for AprConditions wind + gust profiles."""

    __slots__ = (
        "_z_h",
        "_w_h",
        "_z_v",
        "_w_v",
        "_gust_amp",
        "_gust_omega",
        "_gust_phase",
        "_gust_start",
        "_gust_end",
        "_gust_dur",
        "_gust_z_c",
        "_gust_z_hw",
        "_forecast",
    )

    def __init__(self, conditions: AprConditions, *, forecast: bool = False) -> None:
        prof = conditions.forecast_wind_profile if forecast else conditions.wind_profile
        self._z_h = tuple(p[0] for p in prof)
        self._w_h = tuple(p[1] for p in prof)
        vprof = conditions.vertical_wind_profile
        self._z_v = tuple(p[0] for p in vprof)
        self._w_v = tuple(p[1] for p in vprof)
        self._forecast = forecast
        self._gust_amp = 0.0 if forecast else conditions.gust_amplitude_m_s
        self._gust_omega = 2.0 * pi / conditions.gust_period_s
        self._gust_phase = conditions.gust_phase_rad
        self._gust_start = conditions.gust_start_s
        self._gust_dur = conditions.gust_duration_s
        self._gust_end = conditions.gust_start_s + conditions.gust_duration_s
        self._gust_z_c = conditions.gust_altitude_center_m
        self._gust_z_hw = conditions.gust_altitude_half_width_m

    @staticmethod
    def _interp(zs: tuple[float, ...], vs: tuple[float, ...], z: float) -> float:
        if z <= zs[0]:
            return vs[0]
        if z >= zs[-1]:
            return vs[-1]
        # Profiles typically have 2-10 knots: linear scan is faster than numpy.
        for i in range(len(zs) - 1):
            z1 = zs[i + 1]
            if z <= z1:
                z0 = zs[i]
                return vs[i] + (z - z0) / (z1 - z0) * (vs[i + 1] - vs[i])
        return vs[-1]

    @staticmethod
    def _smoothstep(u: float) -> float:
        if u <= 0.0:
            return 0.0
        if u >= 1.0:
            return 1.0
        return u * u * (3.0 - 2.0 * u)

    def horizontal(self, altitude_m: float, time_s: float) -> float:
        base = self._interp(self._z_h, self._w_h, altitude_m)
        if self._gust_amp == 0.0:
            return base
        if time_s <= self._gust_start or time_s >= self._gust_end:
            return base
        dz = abs(altitude_m - self._gust_z_c)
        if dz >= self._gust_z_hw:
            return base
        z_env = self._smoothstep(1.0 - dz / self._gust_z_hw)
        t_unit = (time_s - self._gust_start) / self._gust_dur
        t_env = self._smoothstep(2.0 * min(t_unit, 1.0 - t_unit))
        return base + self._gust_amp * z_env * t_env * sin(
            self._gust_omega * (time_s - self._gust_start) + self._gust_phase
        )

    def vertical(self, altitude_m: float) -> float:
        if self._forecast:
            return 0.0
        return self._interp(self._z_v, self._w_v, altitude_m)


def fast_gravity(altitude_m: float) -> float:
    h = altitude_m if altitude_m > 0.0 else 0.0
    ratio = EARTH_RADIUS_M / (EARTH_RADIUS_M + h)
    return G0 * ratio * ratio


def fast_mass_properties(
    total_mass_kg: float,
    deployment: float,
    *,
    geometry: Geometry = PHYSICS.geometry,
    gear: LandingGear = PHYSICS.gear,
    fin_mass_kg: float = FINS.mass_kg,
    fin_station_m: float = FINS.station_m,
    fin_radius_m: float = FINS.mean_mass_radius_m,
    airframe_mass_offset_kg: float = 0.0,
    upper_attached: bool = False,
    upper_mass_kg: float = 24_000.0,
    upper_station_m: float = PHYSICS.upper_station_m,
    upper_length_m: float = PHYSICS.upper_length_m,
    inertia_scale: float = 1.0,
) -> tuple[float, float, float]:
    """Scalar (cg_m, inertia_kg_m2, fuel_height_m) matching recovery_physics.mass_properties()."""
    airframe_mass = geometry.airframe_mass_kg + airframe_mass_offset_kg
    fixed_dry = (
        geometry.engine_mass_kg
        + airframe_mass
        + geometry.gear_mass_kg
        + fin_mass_kg
        + (upper_mass_kg if upper_attached else 0.0)
    )
    fuel = total_mass_kg - fixed_dry
    if fuel < 0.0:
        fuel = 0.0
    r = geometry.body_radius_m
    r2 = r * r
    fuel_h = fuel / (geometry.propellant_density_kg_m3 * pi * r2)
    fuel_st = geometry.tank_bottom_m + 0.5 * fuel_h
    gear_st = (
        1.0 - deployment
    ) * geometry.gear_retracted_station_m + deployment * geometry.gear_deployed_station_m
    gear_r = (1.0 - deployment) * r + deployment * (0.5 * gear.half_span_m)

    m_eng = geometry.engine_mass_kg
    s_eng = geometry.engine_station_m
    i_eng = m_eng * ((2.0 * s_eng) ** 2 / 12.0 + geometry.nozzle_radius_m**2 / 4.0)

    s_air = geometry.airframe_station_m
    i_air = airframe_mass * (geometry.body_length_m**2 / 12.0 + r2 / 4.0)

    m_gear = geometry.gear_mass_kg
    i_gear = m_gear * gear_r * gear_r

    i_fuel = fuel * (fuel_h * fuel_h / 12.0 + r2 / 4.0)
    i_fin = fin_mass_kg * fin_radius_m * fin_radius_m

    tot = m_eng + airframe_mass + m_gear + fuel + fin_mass_kg
    mom = (
        m_eng * s_eng
        + airframe_mass * s_air
        + m_gear * gear_st
        + fuel * fuel_st
        + fin_mass_kg * fin_station_m
    )
    if upper_attached:
        tot += upper_mass_kg
        mom += upper_mass_kg * upper_station_m
    cg = mom / tot

    inertia = (
        i_eng
        + m_eng * (s_eng - cg) ** 2
        + i_air
        + airframe_mass * (s_air - cg) ** 2
        + i_gear
        + m_gear * (gear_st - cg) ** 2
        + i_fuel
        + fuel * (fuel_st - cg) ** 2
        + i_fin
        + fin_mass_kg * (fin_station_m - cg) ** 2
    )
    if upper_attached:
        i_up = upper_mass_kg * (upper_length_m**2 / 12.0 + r2 / 4.0)
        # inertia_scale applies only to the first-stage booster inertia.
        booster_tot = tot - upper_mass_kg
        booster_cg = (mom - upper_mass_kg * upper_station_m) / booster_tot
        booster_i = (
            i_eng
            + m_eng * (s_eng - booster_cg) ** 2
            + i_air
            + airframe_mass * (s_air - booster_cg) ** 2
            + i_gear
            + m_gear * (gear_st - booster_cg) ** 2
            + i_fuel
            + fuel * (fuel_st - booster_cg) ** 2
            + i_fin
            + fin_mass_kg * (fin_station_m - booster_cg) ** 2
        )
        inertia = (
            inertia
            + (inertia_scale - 1.0) * booster_i
            + i_up
            + upper_mass_kg * (upper_station_m - cg) ** 2
        )
    else:
        inertia *= inertia_scale
    return cg, inertia, fuel_h


def fast_mass_gradients(
    total_mass_kg: float,
    deployment: float,
    cg_m: float,
    fuel_height_m: float,
    *,
    geometry: Geometry = PHYSICS.geometry,
    gear: LandingGear = PHYSICS.gear,
) -> tuple[float, float, float, float]:
    """Scalar analytic (dcg/dm, dcg/ddeploy, dI/dm, dI/ddeploy)."""
    tank_top = geometry.tank_bottom_m + fuel_height_m
    dcg_dm = (tank_top - cg_m) / total_mass_kg
    s_grad = geometry.gear_deployed_station_m - geometry.gear_retracted_station_m
    dcg_ddep = s_grad * geometry.gear_mass_kg / total_mass_kg
    di_dm = geometry.body_radius_m**2 / 4.0 + (tank_top - cg_m) ** 2
    gear_st = geometry.gear_retracted_station_m + s_grad * deployment
    r = geometry.body_radius_m
    dr = 0.5 * gear.half_span_m - r
    gear_r = r + deployment * dr
    di_ddep = 2.0 * geometry.gear_mass_kg * ((gear_st - cg_m) * s_grad + gear_r * dr)
    return dcg_dm, dcg_ddep, di_dm, di_ddep


def compute_engine_thrust(
    throttle_total: float,
    pressure_pa: float,
    propulsion: RtlsPropulsionSettings,
    *,
    thrust_scale: float = 1.0,
) -> tuple[float, float]:
    """Compute (thrust_n, demanded_flow_kg_s) from total equivalent throttle."""
    if throttle_total <= 0.0:
        return 0.0, 0.0
    thrust = (
        throttle_total
        * (
            propulsion.sea_level_thrust_n
            + propulsion.nozzle_exit_area_m2 * (101_325.0 - pressure_pa)
        )
        * thrust_scale
    )
    flow = (
        throttle_total
        * propulsion.sea_level_thrust_n
        * thrust_scale
        / (propulsion.sea_level_isp_s * G0)
    )
    return thrust, flow


def compute_body_aero(
    z_m: float,
    vx_m_s: float,
    vz_m_s: float,
    theta_rad: float,
    omega_rad_s: float,
    cg_m: float,
    atmo_lut: AtmosphereLUT,
    wind_lut: WindLUT,
    time_s: float,
    aero: RtlsAerodynamics,
    *,
    geometry: Geometry = PHYSICS.geometry,
    drag_scale: float = 1.0,
    upper_attached: bool = False,
    body_offset_x: float = 0.0,
    body_offset_z: float = 0.0,
) -> tuple[float, float, float, float, float, float]:
    """Compute scalar body aero (fx, fz, torque_nm, q_pa, mach, alpha_deg)."""
    t_k, _, rho = atmo_lut.at(z_m)
    rel_x = vx_m_s - wind_lut.horizontal(z_m, time_s)
    rel_z = vz_m_s - wind_lut.vertical(z_m)
    speed = hypot(rel_x, rel_z)
    q = 0.5 * rho * speed * speed
    mach = speed / sqrt(1.4 * R_AIR * t_k) if speed > 1e-9 else 0.0
    if speed <= 1e-9:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    c = cos(theta_rad)
    s = sin(theta_rad)
    v_n = c * rel_x - s * rel_z
    v_a = s * rel_x + c * rel_z
    corr = 1.0 + aero.transonic_drag_rise * exp(-(((mach - 1.0) / 0.2) ** 2))
    r = geometry.body_radius_m
    length = geometry.body_length_m
    axial_area = pi * r * r
    normal_area = 2.0 * r * length

    # Nose-first ascent vs tail-first blunt-base descent axial drag
    ascending_flow = v_a >= 0.0 or upper_attached
    axial_cd = aero.axial_cd_ascent if ascending_flow else aero.axial_cd_descent
    cp_m = aero.pressure_center_ascent_m if ascending_flow else aero.pressure_center_descent_m

    scale = -0.5 * rho * corr * drag_scale
    axial_force = scale * axial_cd * axial_area * abs(v_a) * v_a

    # Crossflow drag + slender-body potential normal force (zero deadband at small AoA)
    normal_force = scale * aero.normal_cd * normal_area * abs(v_n) * v_n
    if not upper_attached and aero.slender_lift_slope > 0.0:
        sin_a = v_n / speed
        cos_a = abs(v_a) / speed
        normal_force += scale * aero.slender_lift_slope * axial_area * speed * speed * sin_a * cos_a

    fx = c * normal_force + s * axial_force
    fz = -s * normal_force + c * axial_force

    torque = (cp_m - cg_m) * normal_force
    torque -= (
        0.5
        * rho
        * drag_scale
        * normal_area
        * length
        * length
        * aero.rotation_damping
        * (speed + length * abs(omega_rad_s))
        * omega_rad_s
    )
    if body_offset_x != 0.0 or body_offset_z != 0.0:
        ox = c * body_offset_x + s * body_offset_z
        oz = -s * body_offset_x + c * body_offset_z
        torque += oz * fx - ox * fz

    # Angle of attack relative to forward (ascent) or retrograde (descent) axis
    alpha_rad = atan2(v_n, max(v_a, 1e-9)) if ascending_flow else atan2(v_n, max(-v_a, 1e-9))
    return fx, fz, torque, q, mach, alpha_rad * (180.0 / pi)


def compute_fin_aero(
    z_m: float,
    vx_m_s: float,
    vz_m_s: float,
    theta_rad: float,
    cg_m: float,
    fin_deflection_rad: float,
    fin_extension: float,
    atmo_lut: AtmosphereLUT,
    wind_lut: WindLUT,
    time_s: float,
    fins: RtlsFinSettings,
    *,
    upper_attached: bool = False,
    drag_scale: float = 1.0,
    body_offset_x: float = 0.0,
    body_offset_z: float = 0.0,
) -> tuple[float, float, float, float, float]:
    """Compute scalar grid-fin aero (fx, fz, torque_nm, hinge_nm, alpha_eff_deg).

    Effective angle of attack includes both fin deflection and tail-first body AoA:
        alpha_eff = fin_deflection_rad - alpha_retro
    so undeflected fins naturally provide weathercock stability and deflected fins
    trim the booster to a commanded gliding angle of attack.
    """
    if upper_attached or fin_extension <= 1e-9:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    rel_x = vx_m_s - wind_lut.horizontal(z_m, time_s)
    rel_z = vz_m_s - wind_lut.vertical(z_m)
    speed = hypot(rel_x, rel_z)
    if speed < 1e-6:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    _, _, rho = atmo_lut.at(z_m)
    q = 0.5 * rho * speed * speed
    c = cos(theta_rad)
    s = sin(theta_rad)
    v_n = c * rel_x - s * rel_z
    v_a = s * rel_x + c * rel_z

    # Tail-first flow incidence: alpha_retro > 0 when nose tilts clockwise into flow
    alpha_retro = atan2(v_n, max(-v_a, 1e-3)) if v_a < 0.0 else 0.0
    max_a = fins.control_max_alpha_deg * (pi / 180.0)
    alpha_eff = fin_deflection_rad - alpha_retro
    if alpha_eff > max_a:
        alpha_eff = max_a
    elif alpha_eff < -max_a:
        alpha_eff = -max_a

    cl = fins.lift_slope_rad_inv * alpha_eff
    if cl > fins.max_lift_coefficient:
        cl = fins.max_lift_coefficient
    elif cl < -fins.max_lift_coefficient:
        cl = -fins.max_lift_coefficient

    area_ext = fins.planar_effective_area_m2 * fin_extension
    lift = q * area_ext * cl * drag_scale
    drag = (
        q
        * area_ext
        * (fins.base_drag_coefficient + fins.induced_drag_factor * cl * cl)
        * drag_scale
    )

    # Normal to air-relative velocity: [-rel_z, rel_x] / speed -> [+1, 0] for straight descent
    inv_spd = 1.0 / speed
    nx = -rel_z * inv_spd
    nz = rel_x * inv_spd
    fx = lift * nx - drag * (rel_x * inv_spd)
    fz = lift * nz - drag * (rel_z * inv_spd)

    arm_bx = body_offset_x
    arm_bz = fins.station_m - cg_m + body_offset_z
    arm_x = c * arm_bx + s * arm_bz
    arm_z = -s * arm_bx + c * arm_bz
    torque = arm_z * fx - arm_x * fz
    hinge = 0.5 * abs(lift) * fins.hinge_arm_m
    return fx, fz, torque, hinge, alpha_eff * (180.0 / pi)


def compute_forces_and_torques(
    *,
    z_m: float,
    vx_m_s: float,
    vz_m_s: float,
    theta_rad: float,
    omega_rad_s: float,
    cg_m: float,
    throttle_actual: float,
    gimbal_rad: float,
    rcs_torque_nm: float,
    fin_deflection_rad: float,
    fin_extension: float,
    atmo_lut: AtmosphereLUT,
    wind_lut: WindLUT,
    time_s: float,
    aero: RtlsAerodynamics,
    fins: RtlsFinSettings,
    propulsion: RtlsPropulsionSettings,
    geometry: Geometry = PHYSICS.geometry,
    upper_attached: bool = False,
    thrust_scale: float = 1.0,
    drag_scale: float = 1.0,
    delivered_ratio: float = 1.0,
    body_offset_x: float = 0.0,
    body_offset_z: float = 0.0,
) -> ForceTorqueSample:
    """Pure external force/torque evaluation (2D Plant and future 3D MuJoCo hook)."""
    _, pressure_pa, _ = atmo_lut.at(z_m)
    thrust, main_flow = compute_engine_thrust(
        throttle_actual, pressure_pa, propulsion, thrust_scale=thrust_scale
    )
    thrust *= delivered_ratio
    main_flow *= delivered_ratio
    rcs_flow = abs(rcs_torque_nm) / (geometry.rcs_arm_m * propulsion.rcs_isp_s * G0)

    bfx, bfz, btau, q, mach, alpha_deg = compute_body_aero(
        z_m,
        vx_m_s,
        vz_m_s,
        theta_rad,
        omega_rad_s,
        cg_m,
        atmo_lut,
        wind_lut,
        time_s,
        aero,
        geometry=geometry,
        drag_scale=drag_scale,
        upper_attached=upper_attached,
        body_offset_x=body_offset_x,
        body_offset_z=body_offset_z,
    )
    ffx, ffz, ftau, hinge, fin_alpha_eff_deg = compute_fin_aero(
        z_m,
        vx_m_s,
        vz_m_s,
        theta_rad,
        cg_m,
        fin_deflection_rad,
        fin_extension,
        atmo_lut,
        wind_lut,
        time_s,
        fins,
        upper_attached=upper_attached,
        drag_scale=drag_scale,
        body_offset_x=body_offset_x,
        body_offset_z=body_offset_z,
    )

    aero_fx = bfx + ffx
    aero_fz = bfz + ffz
    tfx = thrust * sin(theta_rad + gimbal_rad)
    tfz = thrust * cos(theta_rad + gimbal_rad)

    lever = cg_m - geometry.nozzle_station_m
    tvc_torque = (-lever + body_offset_z) * thrust * sin(gimbal_rad) - body_offset_x * thrust * cos(
        gimbal_rad
    )
    total_torque = tvc_torque + rcs_torque_nm + btau + ftau

    return ForceTorqueSample(
        force_x_n=aero_fx + tfx,
        force_z_n=aero_fz + tfz,
        torque_nm=total_torque,
        aero_force_x_n=aero_fx,
        aero_force_z_n=aero_fz,
        body_torque_nm=btau,
        fin_force_x_n=ffx,
        fin_force_z_n=ffz,
        fin_torque_nm=ftau,
        fin_hinge_nm=hinge,
        thrust_n=thrust,
        main_flow_kg_s=main_flow,
        rcs_flow_kg_s=rcs_flow,
        dynamic_pressure_pa=q,
        mach=mach,
        alpha_deg=alpha_deg,
        fin_alpha_eff_deg=fin_alpha_eff_deg,
    )


def foot_kinematics_scalar(
    x_m: float,
    z_m: float,
    vx_m_s: float,
    vz_m_s: float,
    theta_rad: float,
    omega_rad_s: float,
    cg_m: float,
    deployment: float,
    *,
    geometry: Geometry = PHYSICS.geometry,
    gear: LandingGear = PHYSICS.gear,
    offset_x: float = 0.0,
    offset_z: float = 0.0,
    offset_vx: float = 0.0,
    offset_vz: float = 0.0,
    cg_rate: float = 0.0,
    deployment_rate: float = 0.0,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Fast scalar left/right foot positions and world velocities."""
    r = geometry.body_radius_m
    span = r + deployment * (gear.half_span_m - r)
    station = gear.anchor_station_m + deployment * (gear.foot_station_m - gear.anchor_station_m)
    bz = station - cg_m + offset_z
    c = cos(theta_rad)
    s = sin(theta_rad)

    dspan = (gear.half_span_m - r) * deployment_rate
    dbz = (gear.foot_station_m - gear.anchor_station_m) * deployment_rate - cg_rate + offset_vz

    # Left foot: bx = -span + offset_x
    lbx = -span + offset_x
    lax = c * lbx + s * bz
    laz = -s * lbx + c * bz
    lpx, lpz = x_m + lax, z_m + laz
    lrbx = -dspan + offset_vx
    lvx = vx_m_s + omega_rad_s * laz + c * lrbx + s * dbz
    lvz = vz_m_s - omega_rad_s * lax - s * lrbx + c * dbz

    # Right foot: bx = +span + offset_x
    rbx = span + offset_x
    rax = c * rbx + s * bz
    raz = -s * rbx + c * bz
    rpx, rpz = x_m + rax, z_m + raz
    rrbx = dspan + offset_vx
    rvx = vx_m_s + omega_rad_s * raz + c * rrbx + s * dbz
    rvz = vz_m_s - omega_rad_s * rax - s * rrbx + c * dbz

    return (lpx, lpz), (rpx, rpz), (lvx, lvz), (rvx, rvz)


def default_design() -> Design:
    return Design()


def default_mass_properties_obj(cg_m: float, inertia: float, fuel_h: float) -> MassProperties:
    return MassProperties(cg_m, inertia, fuel_h)
