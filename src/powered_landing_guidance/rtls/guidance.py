"""Reusable 2D Orbital RTLS GNC Core (Engine-agnostic, 3D MuJoCo ready).

Mission phases:
  1. ascent           : 7-engine Methalox gravity turn to ~60 km orbital MECO-1
  2. flip             : Fast near-vacuum RCS flip toward launch pad
  3. boostback        : 3-engine predictor-corrector targeting the zero-AoA glide window
  4. coast            : Ballistic coast, retrograde alignment, grid-fin deployment
  5. entry_burn       : 3-engine braking guarding BOTH dynamic pressure and stagnation heat flux
  6. aero_glide       : ZERO main propellant! Active grid-fin AoA guidance steering IIP -> 0
  7. landing_burn     : 3->1 center-engine minimum-throttle Hoverslam + convex reference
  8. touchdown_settle : Immediate engine cutoff at contact + 5 s upright standing check
"""

from __future__ import annotations

from dataclasses import dataclass
from math import asin, atan2, cos, exp, hypot, pi, sin, sqrt

from scipy.optimize import brentq

from powered_landing_guidance.apr_recovery import (
    MAX_DENSITY_SCALE_DEVIATION,
    PHYSICS,
    SAFETY,
    AprConditions,
    AscentGuidanceSettings,
    Design,
)
from powered_landing_guidance.observation_models import ControllerOptions, SensorSettings
from powered_landing_guidance.recovery_physics import EARTH_RADIUS_M, G0, AtmosphereSettings
from powered_landing_guidance.rtls.control_allocation import (
    AllocationInput,
    PriorityControlAllocator,
)
from powered_landing_guidance.rtls.convex_guidance import (
    ConvexLandingProblem,
    ConvexLandingSolution,
    solve_convex_hoverslam,
)
from powered_landing_guidance.rtls.estimator import CausalRtlsEstimator, EstimatedState
from powered_landing_guidance.rtls.forces import (
    OMEGA_EARTH,
    AtmosphereLUT,
    RtlsAerodynamics,
    RtlsFinSettings,
    RtlsPropulsionSettings,
    WindLUT,
    compute_body_aero,
    compute_engine_thrust,
    compute_fin_aero,
    compute_stagnation_heat_flux_w_m2,
    fast_gravity,
    fast_mass_gradients,
    fast_mass_properties,
    foot_kinematics_scalar,
)
from powered_landing_guidance.rtls.interfaces import (
    ActuatorCommand,
    GuidanceDiagnostics,
    TelemetryPacket,
)


def _wrap(angle: float) -> float:
    return atan2(sin(angle), cos(angle))


def _smoothstep(u: float) -> float:
    if u <= 0.0:
        return 0.0
    if u >= 1.0:
        return 1.0
    return u * u * (3.0 - 2.0 * u)


def _predict_zero_tilt_state(
    x: float,
    vx: float,
    theta: float,
    omega: float,
    g_eff: float,
    a_att: float,
    *,
    x_horizon_cap_s: float = 3.2,
    dv_horizon_cap_s: float | None = None,
) -> tuple[float, float]:
    """Predict horizontal (x, vx) once current (theta, omega) unwinds to upright at `a_att`.

    Eliminates finite attitude-slew phase lag and limit-cycle overshoot under large
    horizontal diverts while preserving full tilt authority far from the pad.
    """
    a_eff = max(a_att * 0.88, 1e-4)
    if theta * omega > 0.0:
        t1 = abs(omega) / a_eff
        th_pk = theta + 0.5 * omega * t1
        dv1 = g_eff * sin(0.5 * (theta + th_pk)) * t1
        t2 = 2.0 * sqrt(abs(th_pk) / a_eff)
        dv2 = g_eff * sin(0.5 * th_pk) * t2
        t_tot = t1 + t2
        dv = dv1 + dv2
    else:
        stop_ang = 0.5 * omega * omega / a_eff
        if stop_ang <= abs(theta):
            v_mid = sqrt(0.5 * omega * omega + a_eff * abs(theta))
            t_tot = max(0.0, (2.0 * v_mid - abs(omega)) / a_eff)
            dv = g_eff * sin(0.45 * theta) * t_tot
        else:
            t1 = abs(omega) / a_eff
            th_over = theta + 0.5 * omega * t1
            dv1 = g_eff * sin(0.5 * (theta + th_over)) * t1
            t2 = 2.0 * sqrt(abs(th_over) / a_eff)
            dv2 = g_eff * sin(0.5 * th_over) * t2
            t_tot = t1 + t2
            dv = dv1 + dv2
    if dv_horizon_cap_s is not None and t_tot > dv_horizon_cap_s and t_tot > 1e-6:
        scale_t = dv_horizon_cap_s / t_tot
        dv *= scale_t
        t_tot = dv_horizon_cap_s
    vx_zero = vx + dv
    # Shift x by small transport lag (0.45 s) + tilt-unwind impulse ONLY (no double-counting vx)
    x_zero = x + 0.45 * vx + 0.5 * dv * min(t_tot, x_horizon_cap_s)
    return x_zero, vx_zero


@dataclass(frozen=True, slots=True)
class RtlsGuidanceSettings:
    control_dt_s: float = 0.02
    # Fast near-vacuum flip limits (within 80% of 15 deg/s and 5 deg/s^2 safety bounds)
    flip_max_q_pa: float = 250.0
    flip_rate_limit_deg_s: float = 11.0
    flip_accel_limit_deg_s2: float = 2.8
    # Standard attitude tracking limits
    attitude_rate_limit_deg_s: float = 10.0
    attitude_accel_limit_deg_s2: float = 2.8
    high_q_accel_limit_deg_s2: float = 1.4
    high_q_threshold_pa: float = 5_000.0
    # Boostback parameters (3 engines out of 7-engine hexagonal cluster)
    boostback_throttle_total: float = 3.0 / 7.0
    boostback_max_load_g: float = 3.85
    boostback_replan_period_s: float = 0.40
    boostback_miss_tolerance_m: float = 70.0
    boostback_x_target_m: float = -780.0
    # Entry burn: 3->1 burn keeping unpowered peak_q <= 80% of 42 kPa and qdot << 80 kW/m^2
    # Margin covers hidden +10% density and -15% drag simultaneously
    entry_q_trigger_pa: float = 26_500.0
    entry_q_cutoff_pa: float = 25_500.0
    entry_heat_trigger_w_m2: float = 62_000.0
    entry_heat_cutoff_w_m2: float = 58_000.0
    entry_min_ignite_q_pa: float = 350.0
    entry_min_ignite_heat_w_m2: float = 10_000.0
    entry_max_load_g: float = 4.10
    # Unpowered grid-fin aerodynamic glide
    fin_deploy_altitude_m: float = 58_000.0
    glide_iip_period_s: float = 0.10
    glide_max_alpha_deg: float = 16.0
    glide_handoff_altitude_m: float = 950.0
    # 3->1 center-engine vertical Hoverslam (load-guarded 3-engine braking if fast)
    landing_v_switch_m_s: float = 100.0
    landing_stage3_net_decel_m_s2: float = 28.0
    landing_touchdown_speed_m_s: float = 0.61
    landing_low_zone_height_m: float = 18.0
    landing_divert_max_tilt_deg: float = 22.0
    landing_vertical_align_height_m: float = 48.0
    landing_vertical_body_height_m: float = 21.5
    gear_deploy_height_m: float = 400.0
    use_convex_reference: bool = True
    # Ignition margin for 0.2 s delay + spool lag + obs/action delays
    ignition_delay_budget_s: float = 0.28
    ignition_height_margin_m: float = 75.0


class RtlsController:
    """Strictly causal Reusable RTLS controller implementing ControllerProtocol."""

    def __init__(
        self,
        *,
        design: Design,
        conditions: AprConditions,
        aero: RtlsAerodynamics,
        fins: RtlsFinSettings,
        propulsion: RtlsPropulsionSettings,
        ascent_settings: AscentGuidanceSettings | None = None,
        guidance_settings: RtlsGuidanceSettings | None = None,
        position_error_m: float = 0.0,
        velocity_error_m_s: float = 0.0,
        observation_delay_steps: int = 0,
        action_delay_steps: int = 0,
        sensor_settings: SensorSettings | None = None,
        controller_options: ControllerOptions | None = None,
        coriolis_enabled: bool = True,
    ) -> None:
        self.design = design
        self.conditions = conditions
        self.aero = aero
        self.fins = fins
        self.propulsion = propulsion
        self.ascent_settings = ascent_settings or AscentGuidanceSettings()
        self.settings = guidance_settings or RtlsGuidanceSettings()
        self.sensor_settings = sensor_settings or SensorSettings()
        self.controller_options = controller_options or ControllerOptions()
        self.coriolis_enabled = coriolis_enabled
        self.observation_delay_steps = observation_delay_steps
        self.action_delay_steps = action_delay_steps

        # Forecast atmosphere (NEVER uses hidden density_scale!)
        forecast_atmo_settings = AtmosphereSettings(
            sea_level_temperature_k=conditions.sea_level_temperature_k,
            sea_level_pressure_pa=conditions.sea_level_pressure_pa,
            latitude_deg=conditions.latitude_deg,
            coriolis_enabled=coriolis_enabled,
            temperature_profile=conditions.temperature_profile,
            pressure_profile=conditions.pressure_profile,
            density_scale=1.0,
        )
        self.forecast_atmo = AtmosphereLUT(forecast_atmo_settings, step_m=100.0)
        self.forecast_wind = WindLUT(conditions, forecast=True)
        self.allocator = PriorityControlAllocator(propulsion, fins)
        self.estimator = CausalRtlsEstimator(
            design=design,
            aero=aero,
            fins=fins,
            propulsion=propulsion,
            forecast_atmo=self.forecast_atmo,
            forecast_wind=self.forecast_wind,
            latitude_deg=conditions.latitude_deg,
            coriolis_enabled=coriolis_enabled,
            position_error_m=position_error_m,
            velocity_error_m_s=velocity_error_m_s,
            sensor_settings=self.sensor_settings,
            controller_options=self.controller_options,
            control_dt_s=self.settings.control_dt_s,
            action_delay_steps=self.action_delay_steps,
        )

        # Mission state machine
        self.phase: str = "ascent"
        self.cutoff_pending: bool = False
        self.sep_align_started_s: float | None = None
        self.flip_started_s: float | None = None
        self.boost_started_s: float | None = None
        self.boost_direction: float = -1.0
        self.boost_end_s: float = 0.0
        self.next_replan_s: float = 0.0
        self.replanning_count: int = 0
        self.entry_burn_done: bool = False
        self.entry_single_engine: bool = False
        self._eb_steer_smooth_deg: float = 0.0
        self._glide_alpha_smooth_deg: float = 0.0
        self.entry_burn_started_s: float | None = None
        self.entry_burn_ended_s: float | None = None
        self.glide_started_s: float | None = None
        self.landing_started_s: float | None = None
        self.landing_single_engine: bool = False
        self.in_vertical_body: bool = False
        self._vb_started_s: float | None = None
        self._land_bx_smooth: float = 0.0
        self._land_bz_smooth: float = 0.0
        self._vb_wind_trim_rad: float = 0.0
        self.cached_drag_scale: float = 1.0
        self.last_iip_time_s: float = -1e9
        self.cached_iip_x_m: float = 0.0
        self.cached_unpowered_q_pa: float | None = None
        self.convex_solution: ConvexLandingSolution | None = None
        self.convex_plan_start_s: float = 0.0
        self.last_command: ActuatorCommand = ActuatorCommand()
        self.last_diag: GuidanceDiagnostics = GuidanceDiagnostics()
        self._last_rcs_cmd: float = 0.0
        self._coriolis_om: float = (
            OMEGA_EARTH * cos(conditions.latitude_deg * (pi / 180.0)) if coriolis_enabled else 0.0
        )

    def _nominal_mass(self, est: EstimatedState) -> float:
        return (
            self.design.dry_mass_kg
            + max(est.main_propellant_kg, 0.0)
            + max(est.rcs_propellant_kg, 0.0)
            + (self.design.upper_mass_kg if est.upper_attached else 0.0)
        )

    def _foot_height(self, est: EstimatedState, cg_m: float) -> float:
        (_, lpz), (_, rpz), _, _ = foot_kinematics_scalar(
            est.x_m,
            est.z_m,
            est.vx_m_s,
            est.vz_m_s,
            est.theta_rad,
            est.omega_rad_s,
            cg_m,
            est.gear_extension,
        )
        return min(lpz, rpz)

    def _air_velocity(self, est: EstimatedState) -> tuple[float, float]:
        wx = self.forecast_wind.horizontal(est.z_m, est.time_s)
        return est.vx_m_s - wx, est.vz_m_s

    def _retro_angle(self, vx_air: float, vz_air: float) -> float:
        return atan2(-vx_air, max(-vz_air, 20.0))

    def _coast_attitude(self, est: EstimatedState) -> float:
        vx_air, vz_air = self._air_velocity(est)
        return atan2(-vx_air, max(-vz_air, 260.0))

    def _predict_unpowered_peak_q(
        self,
        z0: float,
        vx0: float,
        vz0: float,
        mass0: float,
        time0: float,
        *,
        dt: float = 0.30,
    ) -> tuple[float, float]:
        """Fast scalar rollout predicting peak (q_pa, heat_flux_w_m2) if engine stays OFF."""
        z, vx, vz, t = z0, vx0, vz0, time0
        peak_q = 0.0
        peak_qdot = 0.0
        r = PHYSICS.geometry.body_radius_m
        axial_area = pi * r * r
        fin_area = self.fins.planar_effective_area_m2
        cd_desc = self.aero.axial_cd_descent
        fin_cd0 = self.fins.base_drag_coefficient
        tdr = self.aero.transonic_drag_rise
        om = self._coriolis_om
        q_declining_steps = 0

        for _ in range(700):
            if z <= 800.0:
                break
            t_k, _, rho = self.forecast_atmo.at(z)
            wx = self.forecast_wind.horizontal(z, t)
            rx = vx - wx
            rz = vz
            spd = hypot(rx, rz)
            q = 0.5 * rho * spd * spd
            qdot = compute_stagnation_heat_flux_w_m2(rho, spd, r)
            if q > peak_q:
                peak_q = q
                q_declining_steps = 0
            elif z < 18_000.0 and q < 0.85 * peak_q:
                q_declining_steps += 1
                if q_declining_steps >= 6:
                    break
            if qdot > peak_qdot:
                peak_qdot = qdot
            mach = spd / sqrt(1.4 * 287.05287 * t_k) if spd > 1.0 else 0.0
            corr = 1.0 + tdr * exp(-(((mach - 1.0) / 0.2) ** 2))
            fin_drag_area = (
                fin_area * fin_cd0
                if (z < self.settings.fin_deploy_altitude_m and vz < 0.0)
                else 0.0
            )
            drag = q * (corr * cd_desc * axial_area + fin_drag_area)
            inv_s = 1.0 / max(spd, 1.0)
            r_orb = EARTH_RADIUS_M + max(z, 0.0)
            ax = -drag * (rx * inv_s) / mass0 - 2.0 * om * vz
            az = -drag * (rz * inv_s) / mass0 - fast_gravity(z) + 2.0 * om * vx + (vx * vx) / r_orb
            z += vz * dt + 0.5 * az * dt * dt
            vx += ax * dt
            vz += az * dt
            t += dt
        return peak_q, peak_qdot

    def _predict_rtls_iip(
        self,
        est: EstimatedState,
        burn_duration_s: float,
        *,
        include_entry_burn: bool = True,
        handoff_z_m: float = 950.0,
        dt: float = 0.40,
        use_nav_bias: bool = False,
    ) -> tuple[float, float, float]:
        """Predict horizontal impact point x_iip at `handoff_z_m`, peak_q, and remaining mass."""
        x, z, vx, vz = est.x_m, est.z_m, est.vx_m_s, est.vz_m_s
        mass = self._nominal_mass(est)
        t = est.time_s
        elapsed = 0.0
        peak_q = 0.0
        r = PHYSICS.geometry.body_radius_m
        axial_area = pi * r * r
        fin_area = self.fins.planar_effective_area_m2
        cd_desc = self.aero.axial_cd_descent
        fin_cd0 = self.fins.base_drag_coefficient
        tdr = self.aero.transonic_drag_rise
        om = self._coriolis_om
        entry_active = False
        entry_done = self.entry_burn_done or (not include_entry_burn)
        bb_thr_cap = self.settings.boostback_throttle_total
        ign_wait = (
            self.settings.ignition_delay_budget_s
            if (burn_duration_s > 0.0 and est.engine_throttle < 0.05)
            else 0.0
        )
        # Only include horizontal nav bias `bx` in glide IIP when body AoA is small AND
        # altitude < 10 km: prevents high-AoA lift-scale residual from creating positive-feedback
        # IIP runaway on thin-air cases while still correcting low-altitude wind forecast error!
        bx = 0.0
        if use_nav_bias and est.engine_throttle < 0.02 and est.z_m < 10_000.0:
            wx_b = self.forecast_wind.horizontal(est.z_m, est.time_s)
            rx_b = est.vx_m_s - wx_b
            rz_b = est.vz_m_s
            c_b, sn_b = cos(est.theta_rad), sin(est.theta_rad)
            vn_b = c_b * rx_b - sn_b * rz_b
            va_b = sn_b * rx_b + c_b * rz_b
            alpha_b_deg = abs(atan2(vn_b, max(-va_b, 1.0)) * (180.0 / pi))
            zero_aoa_w = max(0.0, 1.0 - alpha_b_deg / 7.0)
            bx = zero_aoa_w * max(-1.6, min(1.6, est.accel_bias_x_m_s2))
        drag_scale_est = 1.0
        if use_nav_bias and est.engine_throttle < 0.02 and est.z_m < 22_000.0:
            t0_k, _, rho0 = self.forecast_atmo.at(est.z_m)
            wx0 = self.forecast_wind.horizontal(est.z_m, est.time_s)
            spd0 = hypot(est.vx_m_s - wx0, est.vz_m_s)
            q0 = 0.5 * rho0 * spd0 * spd0
            if q0 > 1_500.0 and est.vz_m_s < -50.0:
                m0 = spd0 / sqrt(1.4 * 287.05287 * t0_k)
                c0 = 1.0 + tdr * exp(-(((m0 - 1.0) / 0.2) ** 2))
                f_nom_z = (
                    q0
                    * (c0 * cd_desc * axial_area + fin_area * fin_cd0)
                    * (-est.vz_m_s / max(spd0, 1.0))
                )
                a_nom_z = f_nom_z / max(mass, 1.0)
                if a_nom_z > 2.5:
                    drag_scale_est = max(0.75, min(1.35, 1.0 + est.accel_bias_z_m_s2 / a_nom_z))

        while z > handoff_z_m and elapsed < 580.0:
            step = dt
            boosting = elapsed < burn_duration_s - 1e-9
            if boosting and elapsed + step > burn_duration_s:
                step = burn_duration_s - elapsed
            if step < 1e-4:
                step = 1e-4

            t_k, p_pa, rho = self.forecast_atmo.at(z)
            wx = self.forecast_wind.horizontal(z, t)
            rx = vx - wx
            rz = vz
            spd = hypot(rx, rz)
            q = 0.5 * rho * spd * spd
            if q > peak_q:
                peak_q = q
            qdot = compute_stagnation_heat_flux_w_m2(rho, spd, r)
            mach = spd / sqrt(1.4 * 287.05287 * t_k) if spd > 1.0 else 0.0
            corr = 1.0 + tdr * exp(-(((mach - 1.0) / 0.2) ** 2))
            fin_drag_area = (
                fin_area * fin_cd0
                if (not boosting and z < self.settings.fin_deploy_altitude_m and vz < 0.0)
                else 0.0
            )
            drag = drag_scale_est * q * (corr * cd_desc * axial_area + fin_drag_area)
            inv_s = 1.0 / max(spd, 1.0)
            fx = -drag * (rx * inv_s)
            fz = -drag * (rz * inv_s)
            flow = 0.0

            if boosting:
                if elapsed >= ign_wait:
                    full_t, _ = compute_engine_thrust(1.0, p_pa, self.propulsion)
                    min_m_bb = max(
                        self.design.dry_mass_kg - 1_000.0,
                        mass - 0.01 * self.design.launch_mass_kg,
                    )
                    max_bb_t = self.settings.boostback_max_load_g * G0 * min_m_bb / 1.03
                    lo3, _ = self.propulsion.throttle_bounds(3)
                    ramp = (
                        _smoothstep((elapsed - ign_wait) / 2.00)
                        if est.engine_throttle < 0.05
                        else 1.0
                    )
                    bb_req = lo3 + ramp * max(0.0, bb_thr_cap - lo3)
                    bb_thr = self.propulsion.quantize_throttle(
                        min(bb_req, max_bb_t / max(full_t, 1.0)), 3
                    )
                    thrust, flow = compute_engine_thrust(bb_thr, p_pa, self.propulsion)
                    fx += self.boost_direction * thrust
            elif not entry_done and vz < 0.0 and z < 68_000.0:
                if not entry_active:
                    if (
                        q >= self.settings.entry_min_ignite_q_pa
                        or qdot >= self.settings.entry_min_ignite_heat_w_m2
                    ):
                        pred_q, pred_qdot = self._predict_unpowered_peak_q(
                            z, vx, vz, mass, t, dt=0.45
                        )
                        if (
                            pred_q > self.settings.entry_q_trigger_pa
                            or pred_qdot > self.settings.entry_heat_trigger_w_m2
                        ):
                            entry_active = True
                else:
                    pred_q, pred_qdot = self._predict_unpowered_peak_q(z, vx, vz, mass, t, dt=0.45)
                    if (
                        pred_q <= self.settings.entry_q_cutoff_pa
                        and pred_qdot <= self.settings.entry_heat_cutoff_w_m2
                    ) or z < 11_000.0:
                        entry_active = False
                        entry_done = True
                if entry_active:
                    full_t, _ = compute_engine_thrust(1.0, p_pa, self.propulsion)
                    min_m_eb = max(
                        self.design.dry_mass_kg - 1_000.0,
                        mass - 0.01 * self.design.launch_mass_kg,
                    )
                    max_t = max(
                        0.0, (self.settings.entry_max_load_g * G0 * min_m_eb - 1.15 * drag) / 1.03
                    )
                    raw_thr = max(0.0, max_t / max(full_t, 1.0))
                    lo3, _ = self.propulsion.throttle_bounds(3)
                    eb_eng = 1 if (q > 12_000.0 and raw_thr < lo3) else 3
                    e_thr = self.propulsion.quantize_throttle(raw_thr, eb_eng)
                    thrust, flow = compute_engine_thrust(e_thr, p_pa, self.propulsion)
                    fx += thrust * (-rx * inv_s)
                    fz += thrust * (-rz * inv_s)

            mid_mass = max(self.design.dry_mass_kg, mass - 0.5 * flow * step)
            r_orb = EARTH_RADIUS_M + max(z, 0.0)
            ax = fx / mid_mass - 2.0 * om * vz + (bx if z < 14_000.0 else 0.0)
            az = fz / mid_mass - fast_gravity(z) + 2.0 * om * vx + (vx * vx) / r_orb
            x += vx * step + 0.5 * ax * step * step
            z += vz * step + 0.5 * az * step * step
            vx += ax * step
            vz += az * step
            mass = max(self.design.dry_mass_kg, mass - flow * step)
            t += step
            elapsed += step

        # Extrapolate remaining drop to ground lightly using terminal velocity ratio
        rem_h = max(0.0, z)
        x_iip = x + vx * (rem_h / max(-vz, 30.0)) * 0.55
        rem_main = mass - self.design.dry_mass_kg - est.rcs_propellant_kg
        return x_iip, peak_q, rem_main

    def _upwind_iip_target_m(self, time_s: float) -> float:
        """Mild westward IIP target keeping all thin/dense corners centered inside glide reach."""
        return -150.0

    def _plan_boostback(self, est: EstimatedState) -> None:
        cache: dict[float, tuple[float, float, float]] = {}
        x_tgt = self._upwind_iip_target_m(est.time_s)

        def eval_x(dur: float) -> float:
            key = round(dur, 4)
            if key not in cache:
                cache[key] = self._predict_rtls_iip(
                    est,
                    dur,
                    include_entry_burn=True,
                    handoff_z_m=self.settings.glide_handoff_altitude_m,
                )
            return cache[key][0] - x_tgt

        miss0 = eval_x(0.0)
        if abs(miss0) <= self.settings.boostback_miss_tolerance_m:
            dur = 0.0
        else:
            root: float | None = None
            prev_t = 0.0
            prev_m = miss0
            for lim in (4.0, 10.0, 18.0, 28.0, 40.0, 54.0, 70.0):
                m_lim = eval_x(lim)
                if prev_m * m_lim <= 0.0:
                    root = float(brentq(eval_x, prev_t, lim, xtol=0.01, maxiter=24))
                    break
                prev_t, prev_m = lim, m_lim
            dur = root if root is not None else 30.0

        self.boost_end_s = est.time_s + dur
        self.next_replan_s = est.time_s + self.settings.boostback_replan_period_s
        self.replanning_count += 1

    def _attitude_allocate(
        self,
        est: EstimatedState,
        target_theta: float,
        throttle_cmd: float,
        *,
        rate_limit_deg_s: float | None = None,
        accel_limit_deg_s2: float | None = None,
        allow_fin: bool = False,
        thrust_direction: float | None = None,
        rcs_deadband_nm: float = 0.0,
        high_q_guard: bool = True,
        accel_guard_deg_s2: float = 3.0,
        use_torque_bias: bool = True,
    ) -> tuple[float, float, float]:
        """Compute (gimbal_rad, rcs_torque_nm, fin_deflection_rad) via PriorityControlAllocator."""
        s = self.settings
        mass = self._nominal_mass(est)
        cg, inertia, fuel_h = fast_mass_properties(
            mass,
            est.gear_extension,
            fin_mass_kg=self.fins.mass_kg,
            upper_attached=est.upper_attached,
            upper_mass_kg=self.design.upper_mass_kg,
        )
        # Conservative lower inertia bound covering hidden -1% mass & -3% inertia
        mass_dev = 0.01 * self.design.launch_mass_kg
        _, lower_i, _ = fast_mass_properties(
            mass - mass_dev,
            est.gear_extension,
            fin_mass_kg=self.fins.mass_kg,
            airframe_mass_offset_kg=-mass_dev,
            upper_attached=est.upper_attached,
            upper_mass_kg=self.design.upper_mass_kg,
            inertia_scale=0.97,
        )
        accel_inertia = lower_i

        _, _, body_tau, q, _, _ = compute_body_aero(
            est.z_m,
            est.vx_m_s,
            est.vz_m_s,
            est.theta_rad,
            est.omega_rad_s,
            cg,
            self.forecast_atmo,
            self.forecast_wind,
            est.time_s,
            self.aero,
            upper_attached=est.upper_attached,
        )

        r_lim = (rate_limit_deg_s or s.attitude_rate_limit_deg_s) * (pi / 180.0)
        a_lim = (accel_limit_deg_s2 or s.attitude_accel_limit_deg_s2) * (pi / 180.0)
        if high_q_guard and not est.upper_attached and q >= s.high_q_threshold_pa:
            a_lim = min(a_lim, s.high_q_accel_limit_deg_s2 * (pi / 180.0))

        err = _wrap(target_theta - est.theta_rad)
        stop_rate = sqrt(2.0 * 0.78 * a_lim * abs(err))
        des_rate = (1.0 if err >= 0.0 else -1.0) * min(r_lim, 1.8 * abs(err), stop_rate)
        des_alpha = max(-a_lim, min(a_lim, 4.2 * (des_rate - est.omega_rad_s)))

        _, p_pa, _ = self.forecast_atmo.at(est.z_m)
        cur_thrust, cur_flow = compute_engine_thrust(est.engine_throttle, p_pa, self.propulsion)
        cmd_thrust, _ = compute_engine_thrust(throttle_cmd, p_pa, self.propulsion)
        future_thrust = cmd_thrust + (cur_thrust - cmd_thrust) * exp(
            -s.control_dt_s / self.propulsion.nominal_engine_tau_s
        )
        # During cold restart ignition wait OR engine cutoff tail-off (throttle_cmd == 0),
        # do not command fresh TVC gimbal deflection (allocator still uses current_thrust_n
        # to cancel residual tail-off TVC torque smoothly via fins/RCS!)
        if throttle_cmd <= 1e-3 or (est.engine_throttle < 1e-3 and throttle_cmd > 0.0):
            sizing_thrust = 0.0
        else:
            sizing_thrust = max(cur_thrust, future_thrust)

        lever = cg - PHYSICS.geometry.nozzle_station_m
        _, _, di_dm, di_ddep = fast_mass_gradients(mass, est.gear_extension, cg, fuel_h)
        gear_rate = (
            0.5
            if (
                not est.upper_attached
                and est.gear_extension < 1.0
                and self._foot_height(est, cg) < s.gear_deploy_height_m
            )
            else 0.0
        )
        rcs_arm = PHYSICS.geometry.rcs_arm_m
        rcs_st = PHYSICS.geometry.rcs_station_m
        rcs_flow = abs(self._last_rcs_cmd) / (rcs_arm * self.propulsion.rcs_isp_s * G0)
        i_rate = -di_dm * (cur_flow + rcs_flow) + di_ddep * gear_rate
        ang_flux = (
            -cur_flow * lever * lever * est.omega_rad_s
            - rcs_flow * (rcs_arm * rcs_arm + (rcs_st - cg) ** 2) * est.omega_rad_s
        )

        torque_bias = est.angular_torque_bias_nm if use_torque_bias else 0.0
        req_torque = (
            accel_inertia * des_alpha + i_rate * est.omega_rad_s - ang_flux - body_tau - torque_bias
        )
        background = body_tau + ang_flux - i_rate * est.omega_rad_s + torque_bias

        # Linearize fin torque around current operating point: tau(delta) ~ tau0 + b_fin * delta
        vx_air, vz_air = self._air_velocity(est)
        c, sn = cos(est.theta_rad), sin(est.theta_rad)
        v_n = c * vx_air - sn * vz_air
        v_a = sn * vx_air + c * vz_air
        alpha_retro = atan2(v_n, max(-v_a, 1e-3)) if v_a < 0.0 else 0.0

        b_fin = 0.0
        fin_tau0 = 0.0
        if not est.upper_attached and est.fin_extension > 0.05 and q >= self.fins.control_min_q_pa:
            d_ref = max(-0.25, min(0.25, alpha_retro))
            unit = pi / 180.0
            _, _, t_plus, _, _ = compute_fin_aero(
                est.z_m,
                est.vx_m_s,
                est.vz_m_s,
                est.theta_rad,
                cg,
                d_ref + unit,
                est.fin_extension,
                self.forecast_atmo,
                self.forecast_wind,
                est.time_s,
                self.fins,
            )
            _, _, t_minus, _, _ = compute_fin_aero(
                est.z_m,
                est.vx_m_s,
                est.vz_m_s,
                est.theta_rad,
                cg,
                d_ref - unit,
                est.fin_extension,
                self.forecast_atmo,
                self.forecast_wind,
                est.time_s,
                self.fins,
            )
            _, _, t_ref, _, _ = compute_fin_aero(
                est.z_m,
                est.vx_m_s,
                est.vz_m_s,
                est.theta_rad,
                cg,
                d_ref,
                est.fin_extension,
                self.forecast_atmo,
                self.forecast_wind,
                est.time_s,
                self.fins,
            )
            b_fin = (t_plus - t_minus) / (2.0 * unit)
            fin_tau0 = t_ref - b_fin * d_ref

        out = self.allocator.allocate(
            AllocationInput(
                requested_torque_nm=req_torque,
                background_torque_nm=background,
                acceleration_inertia_kg_m2=accel_inertia,
                lever_arm_m=lever,
                sizing_thrust_n=sizing_thrust,
                fin_effectiveness_nm_per_rad=b_fin,
                fin_zero_deflection_torque_nm=fin_tau0,
                fin_alpha_retro_rad=alpha_retro,
                dynamic_pressure_pa=q,
                fin_extension=est.fin_extension,
                rcs_available=est.rcs_propellant_kg > 0.0,
                allow_fin_allocation=allow_fin and (est.fin_extension >= 0.05),
                thrust_direction_rad=thrust_direction,
                body_theta_rad=est.theta_rad,
                current_gimbal_rad=est.gimbal_rad,
                current_thrust_n=cur_thrust,
                control_dt_s=s.control_dt_s,
                angular_accel_guard_deg_s2=accel_guard_deg_s2,
                rcs_deadband_nm=rcs_deadband_nm,
            )
        )
        # During pre-separation cutoff, guard against lighter post-sep booster rate jump
        rcs_cmd = out.rcs_torque_nm
        if est.upper_attached and self.cutoff_pending:
            _, low_bi, _ = fast_mass_properties(
                mass - self.design.upper_mass_kg,
                est.gear_extension,
                fin_mass_kg=self.fins.mass_kg,
                upper_attached=False,
            )
            lim_sep = low_bi * (2.0 * pi / 180.0)
            rcs_cmd = max(-lim_sep, min(lim_sep, rcs_cmd))

        self._last_rcs_cmd = rcs_cmd
        return out.gimbal_rad, rcs_cmd, out.fin_deflection_rad

    def _compute_hoverslam_ignition_height(self, est: EstimatedState, mass: float) -> float:
        """Compute exact foot-altitude trigger via fast 1D rollout of the 3->1 Hoverslam."""
        s = self.settings
        vz_down = max(0.0, -est.vz_m_s)
        v_td = s.landing_touchdown_speed_m_s
        if vz_down <= v_td:
            return 0.0

        g_loc = fast_gravity(est.z_m)
        _, p_pa, rho = self.forecast_atmo.at(min(max(est.z_m, 0.0), 3_000.0))
        r = PHYSICS.geometry.body_radius_m
        cda = self.aero.axial_cd_descent * (pi * r * r) + (
            self.fins.base_drag_coefficient * self.fins.planar_effective_area_m2
        )

        # 1) Free-fall with drag during ignition delay + spool lag + control delays
        t_lag_total = (
            s.ignition_delay_budget_s
            + (self.observation_delay_steps + self.action_delay_steps) * s.control_dt_s
        )
        dt_lag = t_lag_total / 10.0
        v = vz_down
        m = mass
        dist = 0.0
        for _ in range(10):
            drag = 0.5 * rho * v * v * cda
            dv = g_loc - drag / m
            dist += v * dt_lag + 0.5 * dv * dt_lag * dt_lag
            v = max(v_td, v + dv * dt_lag)

        # 2) Stage A (3 engines load-capped <= 3.75 g nominal -> <= 4.30 g worst-case) -> Stage B
        full_t, _ = compute_engine_thrust(1.0, p_pa, self.propulsion)
        _, hi3 = self.propulsion.throttle_bounds(3)
        thr_3_nom = self.propulsion.quantize_throttle(min(hi3, 3.75 * G0 * m / max(full_t, 1.0)), 3)
        t_3_nom, f_3_nom = compute_engine_thrust(thr_3_nom, p_pa, self.propulsion)
        lo1, hi1 = self.propulsion.throttle_bounds(1)
        t_min_1, _ = compute_engine_thrust(lo1, p_pa, self.propulsion)
        t_max_1, _ = compute_engine_thrust(hi1, p_pa, self.propulsion)
        m_est_b = max(self.design.dry_mass_kg, m - 700.0)
        m_worst_light = max(self.design.dry_mass_kg, m_est_b - 0.011 * self.design.launch_mass_kg)
        a_min_1 = max(0.5, t_min_1 / m_est_b - g_loc)
        a_min_1_worst = max(a_min_1, 1.04 * t_min_1 / m_worst_light - g_loc)
        a_max_1 = max(a_min_1_worst + 2.5, t_max_1 / m_est_b - g_loc)
        a_lo = min(a_max_1 - 3.0, a_min_1_worst + 0.45)
        a_hi = min(a_max_1 - 2.0, max(a_lo + 2.0, a_min_1 + 0.58 * (a_max_1 - a_min_1)))
        h_shape = 680.0
        v_stage_b_top = sqrt(v_td * v_td + (a_lo + a_hi) * h_shape)
        v_sw = min(s.landing_v_switch_m_s, v_stage_b_top)

        k2 = (a_hi - a_lo) / h_shape
        rhs_v = max(0.0, min(v, v_sw) ** 2 - v_td * v_td)
        if k2 > 1e-6:
            h_stage_b = (-a_lo + sqrt(a_lo * a_lo + k2 * rhs_v)) / k2
        else:
            h_stage_b = rhs_v / max(2.0 * a_lo, 1.0)

        dt = 0.05
        h_stage_a = 0.0
        while v > v_sw:
            drag = 0.5 * rho * v * v * cda
            a_dec = max(2.0, (t_3_nom + 0.7 * drag) / m - g_loc)
            step_t = min(dt, (v - v_sw) / a_dec)
            h_stage_a += v * step_t - 0.5 * a_dec * step_t * step_t
            v -= a_dec * step_t
            m = max(self.design.dry_mass_kg, m - f_3_nom * step_t)

        return dist + h_stage_a + h_stage_b + s.ignition_height_margin_m

    def _ascent_guidance(self, est: EstimatedState) -> ActuatorCommand:
        s = self.ascent_settings
        mass = self._nominal_mass(est)
        cg, _, _ = fast_mass_properties(
            mass,
            est.gear_extension,
            fin_mass_kg=self.fins.mass_kg,
            upper_attached=True,
            upper_mass_kg=self.design.upper_mass_kg,
        )
        bfx, bfz, _, q, _, _ = compute_body_aero(
            est.z_m,
            est.vx_m_s,
            est.vz_m_s,
            est.theta_rad,
            est.omega_rad_s,
            cg,
            self.forecast_atmo,
            self.forecast_wind,
            est.time_s,
            self.aero,
            upper_attached=True,
        )
        vx_air, vz_air = self._air_velocity(est)
        speed = hypot(vx_air, vz_air)

        _, p_pa, rho = self.forecast_atmo.at(est.z_m)
        full_t, full_flow = compute_engine_thrust(1.0, p_pa, self.propulsion)
        shutdown_delay = (
            self.observation_delay_steps + self.action_delay_steps + 1
        ) * self.settings.control_dt_s + 0.05
        shutdown_res = full_flow * shutdown_delay + 18.0
        if (
            est.main_propellant_kg <= s.separation_min_main_propellant_kg + shutdown_res
            or est.z_m >= s.separation_altitude_m + 0.75 * s.separation_tolerance_m
        ):
            self.cutoff_pending = True

        # True orbital Gravity Turn with causal Day-of-Launch forecast wind biasing:
        #   1) Vertical tower clear (t < pitch_start_s)
        #   2) Wind-biased pitch-over kick so +/-20 m/s ascent winds enter the same MECO corridor
        #   3) Zero-AoA gravity turn + gentle upper-atmosphere MECO corridor shaping
        flow_air = atan2(vx_air, max(vz_air, 1.0)) if speed > 20.0 else 0.0
        flow_inert = atan2(est.vx_m_s, max(est.vz_m_s, 1.0)) if est.vz_m_s > 20.0 else 0.0
        w_ascent_fc = 0.65 * self.forecast_wind.horizontal(
            1_000.0, 0.0
        ) + 0.35 * self.forecast_wind.horizontal(4_000.0, 0.0)
        kick_deg = max(3.5, min(9.5, s.final_pitch_deg + 0.16 * w_ascent_fc))
        t_kick_end = s.pitch_start_s + s.pitch_duration_s
        if est.time_s < s.pitch_start_s:
            pitch = 0.0
        elif est.time_s < t_kick_end:
            pitch = (kick_deg * pi / 180.0) * _smoothstep(
                (est.time_s - s.pitch_start_s) / s.pitch_duration_s
            )
        else:
            kick_rad = kick_deg * pi / 180.0
            gt_blend = _smoothstep((est.time_s - t_kick_end) / 4.0)
            pitch = (1.0 - gt_blend) * kick_rad + gt_blend * flow_inert
            # Above z = 10 km, steer pitch gently into the golden 38.5 deg MECO corridor
            # (within +/-2.3 deg at high-q, relaxing to +/-17 deg once q drops below 13.5 kPa)
            # across all +/-20 m/s winds and +/-1% mass / +/-3% thrust dispersions!
            if est.z_m > 10_000.0:
                rem_prop = max(0.0, est.main_propellant_kg - s.separation_min_main_propellant_kg)
                frac_burn = max(
                    0.0,
                    min(
                        1.0,
                        1.0
                        - rem_prop
                        / max(
                            self.design.main_propellant_kg - s.separation_min_main_propellant_kg,
                            1.0,
                        ),
                    ),
                )
                target_gt_rad = (38.5 * pi / 180.0) * frac_burn
                trim_lim = (2.3 + 14.7 * _smoothstep((13_500.0 - q) / 8_500.0)) * (pi / 180.0)
                trim = max(-trim_lim, min(trim_lim, 1.60 * (target_gt_rad - flow_inert)))
                pitch = flow_inert + trim

        if q > 100.0:
            dep = _smoothstep((est.vz_m_s - 40.0) / 60.0)
            p_flow = dep * flow_air
            ang_budget = 16.0 - 13.5 * _smoothstep((q - 2_500.0) / 5_500.0)
            allow = min(ang_budget, 0.55 * SAFETY.max_q_alpha_kpa_deg * 1000.0 / q) * (pi / 180.0)
            pitch = max(p_flow - allow, min(p_flow + allow, pitch))

        time_to_res = (
            est.main_propellant_kg - s.separation_min_main_propellant_kg - shutdown_res
        ) / max(full_flow, 1.0)
        ang_err = abs(_wrap(est.theta_rad - flow_air))
        a_lim = self.settings.attitude_accel_limit_deg_s2 * (pi / 180.0)
        r_lim = self.settings.attitude_rate_limit_deg_s * (pi / 180.0)
        slew = (
            max(2.0 * sqrt(ang_err / a_lim), ang_err / r_lim + r_lim / a_lim)
            + abs(est.omega_rad_s) / a_lim
        )
        preview = max(s.separation_alignment_preview_s, slew + 2.0)
        if self.sep_align_started_s is None and est.vz_m_s > 100.0 and time_to_res <= preview:
            self.sep_align_started_s = est.time_s
        if self.sep_align_started_s is not None:
            blend = _smoothstep(
                (est.time_s - self.sep_align_started_s) / s.separation_alignment_blend_s
            )
            pitch += blend * _wrap(flow_air - pitch)

        if self.cutoff_pending:
            pitch = flow_air
            throttle = 0.0
            active_engines = 0
        else:
            min_mass = mass - 0.01 * self.design.launch_mass_kg
            thrust = min(full_t, 3.55 * G0 * min_mass / 1.03)
            if q > 0.55 * SAFETY.max_q_pa and speed > 10.0:
                _, _, rho_hi = self.forecast_atmo.at(est.z_m + 20.0)
                grad = (log_safe(rho_hi) - log_safe(rho)) / 20.0
                rob_q = 0.77 * SAFETY.max_q_pa / (1.0 + MAX_DENSITY_SCALE_DEVIATION)
                cap = sqrt(2.0 * rob_q / max(rho, 1e-9))
                des = -0.5 * cap * grad * est.vz_m_s + 0.7 * (cap - speed)
                free = (
                    (bfx / mass) * vx_air + (bfz / mass - fast_gravity(est.z_m)) * vz_air
                ) / speed
                proj = (sin(pitch) * vx_air + cos(pitch) * vz_air) / speed
                if proj > 0.05:
                    thrust = min(thrust, max(0.0, mass * (des - free) / proj))
            active_engines = self.propulsion.engine_count
            throttle = self.propulsion.quantize_throttle(thrust / max(full_t, 1.0), active_engines)

        gimbal, rcs, fin = self._attitude_allocate(
            est,
            pitch,
            throttle,
            rate_limit_deg_s=8.0,
            accel_limit_deg_s2=2.5,
            allow_fin=False,
        )
        self.last_diag = GuidanceDiagnostics(
            phase="ascent",
            attitude_target_deg=pitch * (180.0 / pi),
        )
        return ActuatorCommand(
            throttle_cmd=throttle,
            active_engines=active_engines,
            gimbal_rad=gimbal,
            rcs_torque_nm=rcs,
            fin_deflection_rad=fin,
            deploy_fins=False,
            deploy_gear=False,
            phase="ascent",
        )

    def _landing_burn_guidance(
        self, est: EstimatedState, mass: float, cg: float, h: float
    ) -> ActuatorCommand:
        s = self.settings
        _, p_pa, _ = self.forecast_atmo.at(est.z_m)
        g_loc = fast_gravity(est.z_m)
        vz_down = max(0.0, -est.vz_m_s)

        # Causal translation acceleration bias compensation near pad (also adapts
        # minimum-throttle floor and CG height shift causally for +/-1% hidden mass!)
        bias_x = 0.0
        bias_z = 0.0
        if self.controller_options.translation_bias_compensation:
            blend = _smoothstep((950.0 - h) / 250.0)
            bnorm = hypot(est.accel_bias_x_m_s2, est.accel_bias_z_m_s2)
            bscale = min(1.0, 3.5 / max(bnorm, 1e-9))
            bias_x = blend * bscale * est.accel_bias_x_m_s2
            bias_z = blend * bscale * est.accel_bias_z_m_s2
        alpha_bx = 1.0 - exp(-s.control_dt_s / (0.60 if self.in_vertical_body else 1.80))
        self._land_bx_smooth += alpha_bx * (bias_x - self._land_bx_smooth)
        alpha_bz = 1.0 - exp(-s.control_dt_s / 0.50)
        self._land_bz_smooth += alpha_bz * (bias_z - self._land_bz_smooth)

        # Monotone 3 -> 1 engine switch ONLY once 1 center engine can comfortably land with margin!
        t_min_1, _ = compute_engine_thrust(
            self.propulsion.throttle_bounds(1)[0], p_pa, self.propulsion
        )
        t_max_1, _ = compute_engine_thrust(
            self.propulsion.throttle_bounds(1)[1], p_pa, self.propulsion
        )
        m_worst_light = max(self.design.dry_mass_kg, mass - 0.01 * self.design.launch_mass_kg)
        a_min_1_chk = max(0.5, t_min_1 / mass - g_loc)
        # Smoothly adapt lower deceleration floor causally using observed vertical accel bias:
        #   - heavy booster (bias_z < -1.0) lowers floor & accounts for ~0.4 m higher true CG
        #   - nominal/light booster stays above worst-light floor so TWR_min>1 never bounces!
        a_min_1_worst = max(a_min_1_chk, 1.03 * t_min_1 / m_worst_light - g_loc)
        if self.controller_options.translation_bias_compensation and h < 350.0:
            heavy_w = _smoothstep((-self._land_bz_smooth + 0.25) / 1.10)
            light_w = _smoothstep((self._land_bz_smooth - 0.35) / 0.85)
            a_light = max(a_min_1_worst + 0.28, a_min_1_chk + max(0.0, bias_z) + 0.50)
            a_heavy = max(0.85, a_min_1_chk + 0.35)
            a_min_1_eff = (1.0 - heavy_w) * a_light + heavy_w * a_heavy
            # Symmetric causal CG-height shift for +/-1% hidden dry/airframe mass:
            # heavy booster shifts true CG higher (foot hits earlier); light booster shifts
            # true CG lower (foot hits ~0.38 m later, preventing premature cutoff of lateral vb!).
            cg_shift_est = heavy_w * max(0.0, min(0.45, 0.20 - 0.18 * self._land_bz_smooth)) - (
                light_w * max(0.0, min(0.18, 0.15 * self._land_bz_smooth))
            )
        else:
            light_w = 0.0
            a_min_1_eff = a_min_1_worst + 0.28
            cg_shift_est = 0.0
        h_vref = max(0.0, h - cg_shift_est)
        a_max_1_net = max(a_min_1_eff + 2.5, t_max_1 / mass - g_loc)
        a_lo_chk = min(a_max_1_net - 3.0, a_min_1_eff + 0.16)
        a_hi_chk = min(
            a_max_1_net - 2.0, max(a_lo_chk + 2.0, a_min_1_chk + 0.58 * (a_max_1_net - a_min_1_chk))
        )
        h_shape = 680.0
        h_c_chk = min(h_vref, h_shape)
        v_stage_b_cap = sqrt(
            s.landing_touchdown_speed_m_s**2
            + 2.0 * a_lo_chk * h_vref
            + ((a_hi_chk - a_lo_chk) / h_shape) * h_c_chk * h_c_chk
            + 2.0 * a_hi_chk * max(0.0, h_vref - h_shape)
        )
        if not self.landing_single_engine and (
            vz_down <= min(s.landing_v_switch_m_s, v_stage_b_cap) or h <= 200.0
        ):
            self.landing_single_engine = True
            if s.use_convex_reference and h > 30.0:
                prob = ConvexLandingProblem(
                    x0_m=est.x_m,
                    z0_m=max(h, 5.0),
                    vx0_m_s=est.vx_m_s,
                    vz0_m_s=min(est.vz_m_s, -2.0),
                    a_min_m_s2=t_min_1 / mass,
                    a_max_m_s2=t_max_1 / mass,
                    gravity_m_s2=g_loc,
                    v_touchdown_m_s=s.landing_touchdown_speed_m_s,
                    max_tilt_deg=s.landing_divert_max_tilt_deg,
                )
                self.convex_solution = solve_convex_hoverslam(prob)
                self.convex_plan_start_s = est.time_s

        active_engines = 1 if self.landing_single_engine else 3
        t_min, _ = compute_engine_thrust(
            self.propulsion.throttle_bounds(active_engines)[0], p_pa, self.propulsion
        )
        t_max, _ = compute_engine_thrust(
            self.propulsion.throttle_bounds(active_engines)[1], p_pa, self.propulsion
        )
        full_t, _ = compute_engine_thrust(1.0, p_pa, self.propulsion)

        bfx, bfz, btau, q_land, _, _ = compute_body_aero(
            est.z_m,
            est.vx_m_s,
            est.vz_m_s,
            est.theta_rad,
            est.omega_rad_s,
            cg,
            self.forecast_atmo,
            self.forecast_wind,
            est.time_s,
            self.aero,
        )
        ffx, ffz, ftau, _, _ = compute_fin_aero(
            est.z_m,
            est.vx_m_s,
            est.vz_m_s,
            est.theta_rad,
            cg,
            est.fin_deflection_rad,
            est.fin_extension,
            self.forecast_atmo,
            self.forecast_wind,
            est.time_s,
            self.fins,
        )
        aero_fx = bfx + ffx
        aero_fz = bfz + ffz
        lever_trim = max(cg - PHYSICS.geometry.nozzle_station_m, 1.0)
        max_gim_rad = self.propulsion.gimbal_limit_deg * (pi / 180.0)
        tau_aero_now = btau + ftau

        # Compute equilibrium wind-trim tilt where net horizontal accel = 0 so
        # _predict_zero_tilt_state has full finite-slew phase lead with ZERO crosswind static bias!
        bfx0, _, _, _, _, _ = compute_body_aero(
            est.z_m,
            est.vx_m_s,
            est.vz_m_s,
            0.0,
            0.0,
            cg,
            self.forecast_atmo,
            self.forecast_wind,
            est.time_s,
            self.aero,
        )
        v_td = s.landing_touchdown_speed_m_s
        # Accurate remaining time-to-go along the 3->1 Hoverslam deceleration curve
        theta_vb = 0.0
        a_avg_b = a_lo_chk + 0.5 * (a_hi_chk - a_lo_chk) * min(max(h, 0.0) / h_shape, 1.0)
        if not self.landing_single_engine:
            v_sw = min(s.landing_v_switch_m_s, v_stage_b_cap)
            t_go = max(0.0, (vz_down - v_sw) / 28.0) + max(1.0, (v_sw - v_td) / max(a_avg_b, 1.5))
            k2 = (a_hi_chk - a_lo_chk) / h_shape
            rhs_sw = max(0.0, v_sw * v_sw - v_td * v_td)
            h_sw = (-a_lo_chk + sqrt(a_lo_chk * a_lo_chk + k2 * rhs_sw)) / max(k2, 1e-6)
            dh = max(h - h_sw, 25.0)
            az_net = max(
                (t_min / mass - g_loc),
                min(2.85 * G0, (vz_down * vz_down - v_sw * v_sw) / (2.0 * dh)),
            )
            vertical_body = False
            a_att = 2.95 * (pi / 180.0)
            a_t_smooth = max(t_min / mass, min(t_max / mass, az_net + g_loc))
            gim_trim_now = asin(
                max(
                    -0.85 * sin(max_gim_rad),
                    min(
                        0.85 * sin(max_gim_rad),
                        tau_aero_now / max(lever_trim * mass * a_t_smooth, 1.0),
                    ),
                )
            )
            th_trim = max(
                -0.30,
                min(
                    0.30,
                    atan2(
                        -aero_fx / mass
                        - self._land_bx_smooth
                        + 2.0 * self._coriolis_om * est.vz_m_s,
                        a_t_smooth,
                    ),
                ),
            )
            x_lead, vx_lead = _predict_zero_tilt_state(
                est.x_m,
                est.vx_m_s,
                est.theta_rad - th_trim,
                est.omega_rad_s,
                a_t_smooth,
                a_att,
                x_horizon_cap_s=2.4,
            )
            t_align = max(3.2, t_go - 2.8)
            ax_cmd = -6.5 * x_lead / (t_align * t_align) - 5.5 * vx_lead / t_align
            ax_cmd = max(-7.5, min(7.5, ax_cmd))
            stage_b_tilt_lim = s.landing_divert_max_tilt_deg * (pi / 180.0)
        else:
            # Stage B (1 center engine): C1 smooth non-hovering Hoverslam energy profile!
            a_max_1 = a_max_1_net
            a_lo = a_lo_chk
            a_hi = a_hi_chk
            h_c = min(h_vref, h_shape)
            a_ff = a_lo + (a_hi - a_lo) * (h_c / h_shape)
            v_ref = v_stage_b_cap
            if vz_down <= 0.78 * v_td:
                # Never boost upward with -bias_z when already slower than touchdown target!
                az_net = -3.5
            else:
                k_v = (
                    (2.7 if h <= 15.0 else 2.1) if vz_down > v_ref else (1.8 if h > 15.0 else 1.45)
                )
                az_net = a_ff + k_v * (vz_down - v_ref)
            az_net = max(-3.5, min(a_max_1, az_net))

            t_go = max(1.0, (vz_down - v_td) / max(a_avg_b, 1.5))
            # Deterministic vertical-body latch altitude (immune to single-tick nav noise
            # spikes that could prematurely trip a one-way latch).
            # Delay vertical-body latch slightly ONLY when far off-pad AND not already
            # sprinting fast toward x = 0 (preserves full braking time for fast approaches!).
            slow_off_pad = 1.0 - _smoothstep((-est.x_m * est.vx_m_s - 10.0) / 15.0)
            vb_h = (
                s.landing_vertical_body_height_m
                - 5.5
                - 1.5 * slow_off_pad * _smoothstep((abs(est.x_m) - 4.5) / 4.5)
            )
            if h_vref <= vb_h:
                if not self.in_vertical_body:
                    self._vb_started_s = est.time_s
                self.in_vertical_body = True
            vertical_body = self.in_vertical_body
            if vertical_body:
                a_t_smooth = max(t_min / mass, min(t_max / mass, a_ff + g_loc - aero_fz / mass))
                lever = cg - PHYSICS.geometry.nozzle_station_m
                t_rem = 2.0 * max(0.0, h_vref) / max(vz_down + v_td, 1.0)
                # 1) Quasi-steady upwind body wind-trim (<= 1.65 deg << 3.0 deg touchdown limit,
                #    heavily low-pass filtered so contact angular rate stays < 0.15 deg/s):
                #    cancels strong ground crosswinds/gusts directly with thrust tilt so the
                #    full +/-10.6 kN gimbal+RCS budget remains available to center x -> 0!
                a_wind_vb = (
                    bfx0 / mass + self._land_bx_smooth - 2.0 * self._coriolis_om * est.vz_m_s
                )
                raw_wind_trim = max(
                    -2.05 * (pi / 180.0),
                    min(2.05 * (pi / 180.0), atan2(-a_wind_vb, max(a_t_smooth, 5.0))),
                )
                # Do not lean upwind while already sprinting fast upwind and needing to brake!
                if est.vx_m_s * raw_wind_trim > 0.0:
                    raw_wind_trim *= 1.0 - _smoothstep((abs(est.vx_m_s) - 1.4) / 1.8)
                alpha_wt = 1.0 - exp(-s.control_dt_s / 0.85)
                self._vb_wind_trim_rad += alpha_wt * (raw_wind_trim - self._vb_wind_trim_rad)
                # 2) Keep vertical_body strictly inside the 2.40 deg (80%) touchdown tilt envelope:
                #    quasi-steady upwind wind-trim (<= 2.05 deg) plus a gentle off-pad lean
                #    ONLY when predicted stop point is still far off-pad above h = 2.2 m.
                x_pos_trim = est.x_m + 0.95 * est.vx_m_s
                vb_mid_env = (
                    _smoothstep((h_vref - 2.2) / 3.4)
                    * _smoothstep((abs(est.x_m) - 2.2) / 2.0)
                    * _smoothstep((abs(x_pos_trim) - 1.4) / 1.8)
                )
                th_vb_mid = (
                    max(
                        -2.50 * (pi / 180.0),
                        min(2.50 * (pi / 180.0), (-0.75 * (pi / 180.0)) * x_pos_trim),
                    )
                    * vb_mid_env
                )
                theta_vb = max(
                    -2.65 * (pi / 180.0),
                    min(2.65 * (pi / 180.0), self._vb_wind_trim_rad + th_vb_mid),
                )
                ax_lim_rcs = 0.92 * self.propulsion.rcs_torque_limit_nm / max(lever * mass, 1.0)
                # Exact physical horizontal braking capacity against current closing direction:
                # pure gimbal+RCS minus crosswind or kinematically un-unwindable residual tilt
                # at touchdown (average unwind rate ~1.30 deg/s before final 0.55 s rate clamp).
                s_move = (
                    (1.0 if est.vx_m_s >= 0.0 else -1.0)
                    if abs(est.vx_m_s) > 0.25
                    else (-1.0 if est.x_m >= 0.0 else 1.0)
                )
                dth_err = est.theta_rad + 0.45 * est.omega_rad_s - self._vb_wind_trim_rad
                dth_unwind = (1.85 * pi / 180.0) * max(t_rem - 0.55, 0.0)
                if abs(dth_err) <= dth_unwind:
                    th_persist = self._vb_wind_trim_rad
                else:
                    th_persist = self._vb_wind_trim_rad + (
                        (abs(dth_err) - dth_unwind) * (1.0 if dth_err >= 0.0 else -1.0)
                    )
                a_push_with_v = s_move * (a_t_smooth * sin(th_persist) + a_wind_vb)
                ax_brake_avail = max(0.35, min(1.35 * ax_lim_rcs, ax_lim_rcs - a_push_with_v))
                far_x_w = _smoothstep((abs(est.x_m) - 1.6) / 1.2)
                off_pad_floor = 0.12 * _smoothstep((abs(est.x_m) - 1.45) / 0.55)
                brake_frac = 0.86 + 0.14 * far_x_w
                t_mgn = 0.29 - 0.09 * far_x_w
                v_ground_safe = (
                    0.12 + off_pad_floor + brake_frac * ax_brake_avail * max(t_rem - t_mgn, 0.0)
                )
                v_deck_cap = 0.07 + off_pad_floor + 3.4 * _smoothstep((h_vref - 0.52) / 1.65)
                closing = min(
                    0.82 * abs(est.x_m),
                    sqrt(2.0 * 0.90 * ax_brake_avail * abs(est.x_m)),
                    v_ground_safe,
                    v_deck_cap,
                )
                des_vx = (-1.0 if est.x_m >= 0.0 else 1.0) * closing
                k_vb = 3.2 + 0.8 * _smoothstep((h_vref - 1.5) / 4.0)
                ax_cmd = max(-3.5, min(3.5, k_vb * (des_vx - est.vx_m_s)))
                stage_b_tilt_lim = s.landing_divert_max_tilt_deg * (pi / 180.0)
            else:
                theta_vb = 0.0
                # Smooth physically-realizable feedforward thrust accel + TVC trim decoupling:
                # thrust acts along theta + gimbal, so accounting for quasi-static aerodynamic
                # gimbal trim delivers exact thrust vectoring without positive feedback!
                # Scale predicted Stage-B angular acceleration with causal mass-bias estimate:
                # nominal/heavy boosters (true I > lower_i) slew at ~2.44 deg/s^2, whereas
                # light boosters (bz > 0.85 m/s^2, true I -> lower_i) slew at ~2.82 deg/s^2.
                a_att = (2.44 + 0.34 * _smoothstep((self._land_bz_smooth - 0.85) / 0.22)) * (
                    pi / 180.0
                )
                a_t_smooth = max(t_min / mass, min(t_max / mass, a_ff + g_loc - aero_fz / mass))
                gim_trim_now = asin(
                    max(
                        -0.85 * sin(max_gim_rad),
                        min(
                            0.85 * sin(max_gim_rad),
                            tau_aero_now / max(lever_trim * mass * a_t_smooth, 1.0),
                        ),
                    )
                )
                th_trim = max(
                    -0.30,
                    min(
                        0.30,
                        atan2(
                            -aero_fx / mass
                            - self._land_bx_smooth
                            + 2.0 * self._coriolis_om * est.vz_m_s,
                            a_t_smooth,
                        ),
                    ),
                )
                t_kin = 2.0 * max(h_vref, 0.0) / max(vz_down + v_td, 1.0)
                a_dec_eff = max(1.5, 0.5 * (a_lo + a_ff) + max(0.0, 0.55 * self._land_bz_smooth))
                t_dec = max(0.0, vz_down - v_td) / a_dec_eff
                t_rem_b = min(28.0, max(1.0, 0.50 * (t_kin + t_dec)))
                t_align = max(2.75, t_rem_b - 3.85)
                # Gimbal trim opposes body tilt when aerodynamic restoring torque is large,
                # reducing effective unwind impulse; only apply when opposing so gimbal never
                # hallucinates a phantom body-slew lag at near-zero tilt!
                th_rel = est.theta_rad - th_trim
                if th_rel * gim_trim_now < 0.0:
                    th_rel_eff = th_rel + 0.80 * gim_trim_now
                    if th_rel_eff * th_rel < 0.0:
                        th_rel_eff = 0.0
                else:
                    th_rel_eff = th_rel
                x_lead, vx_lead = _predict_zero_tilt_state(
                    est.x_m,
                    est.vx_m_s,
                    th_rel_eff,
                    est.omega_rad_s,
                    a_t_smooth,
                    a_att,
                    x_horizon_cap_s=2.4,
                )
                far_blend = _smoothstep((max(abs(est.x_m), abs(x_lead)) - 1.5) / 7.5)
                speed_blend = _smoothstep((abs(vx_lead) - 0.6) / 1.5)
                tilt_frac = _smoothstep(
                    (h_vref - s.landing_vertical_body_height_m)
                    / max(s.landing_vertical_align_height_m - s.landing_vertical_body_height_m, 1.0)
                )
                tilt_floor = 0.08 + 0.42 * far_blend + 0.24 * speed_blend
                stage_b_tilt_lim = (s.landing_divert_max_tilt_deg * pi / 180.0) * max(
                    tilt_floor, tilt_frac
                )
                max_lat = a_t_smooth * sin(stage_b_tilt_lim)
                ax_cmd = -7.4 * x_lead / (t_align * t_align) - 6.2 * vx_lead / t_align
                ax_cmd = max(-max_lat, min(max_lat, ax_cmd))

        # Required thrust vector compensating gravity, Coriolis, aero, and causal nav bias
        prop_x = ax_cmd - self._land_bx_smooth + 2.0 * self._coriolis_om * est.vz_m_s
        if self.landing_single_engine and vz_down <= 0.78 * v_td:
            fz_req = t_min
        else:
            prop_z = az_net + g_loc - bias_z - 2.0 * self._coriolis_om * est.vx_m_s
            fz_req = max(t_min * 0.9, mass * prop_z - aero_fz)
        fx_req = mass * prop_x - aero_fx

        if vertical_body:
            lever = cg - PHYSICS.geometry.nozzle_station_m
            # Smoothly center gimbal & RCS counter-torque right before ground contact / MECO-3
            # without any discontinuous threshold jump, while keeping full gust rejection down to
            # h = 0.35 m whenever horizontal speed is non-trivial.
            td_need_lat = max(
                0.95 * _smoothstep((abs(est.vx_m_s) - 0.18) / 0.20),
                0.60 * _smoothstep((abs(a_wind_vb) - 0.15) / 0.25),
            )
            always_td_taper = max(td_need_lat, _smoothstep((h_vref - 0.01) / 0.08))
            wind_calm_guard = 1.0 - _smoothstep((abs(a_wind_vb) - 0.10) / 0.15)
            calm_blend = (
                (1.0 - _smoothstep((abs(est.vx_m_s) - 0.14) / 0.16))
                * (1.0 - _smoothstep((abs(est.x_m) - 0.55) / 0.45))
                * wind_calm_guard
            )
            lat_calm_taper = min(
                always_td_taper, 1.0 - calm_blend * (1.0 - _smoothstep((h_vref - 1.5) / 3.0))
            )
            rcs_lat_share = 0.90 - 0.12 * _smoothstep(
                (abs(est.theta_rad) - 1.2 * pi / 180.0) / (2.0 * pi / 180.0)
            )
            lat_lim = (
                rcs_lat_share * self.propulsion.rcs_torque_limit_nm / max(lever, 1.0)
            ) * lat_calm_taper
            fx_body_center = fz_req * sin(est.theta_rad)
            fx_req = max(fx_body_center - lat_lim, min(fx_body_center + lat_lim, fx_req))
            direction = atan2(fx_req, fz_req)
        else:
            direction = atan2(fx_req, fz_req)
            max_tilt = min(s.landing_divert_max_tilt_deg * (pi / 180.0), stage_b_tilt_lim)
            dir_clamped = max(-max_tilt, min(max_tilt, direction))
            _, _, btau_dir, _, _, _ = compute_body_aero(
                est.z_m,
                est.vx_m_s,
                est.vz_m_s,
                dir_clamped,
                0.0,
                cg,
                self.forecast_atmo,
                self.forecast_wind,
                est.time_s,
                self.aero,
                drag_scale=self.cached_drag_scale,
            )
            gim_trim_dir = asin(
                max(
                    -0.85 * sin(max_gim_rad),
                    min(
                        0.85 * sin(max_gim_rad),
                        btau_dir / max(lever_trim * mass * a_t_smooth, 1.0),
                    ),
                )
            )
            body_dir = max(-max_tilt, min(max_tilt, dir_clamped - gim_trim_dir))
            direction = body_dir
            fx_req = fz_req * sin(dir_clamped) / max(cos(dir_clamped), 0.5)

        f_mag = hypot(fx_req, fz_req)
        # Strictly enforce Cluster per-engine throttle bounds [min, max]!
        throttle = self.propulsion.quantize_throttle(f_mag / max(full_t, 1.0), active_engines)

        # Hoverslam anti-bounce MECO-3 guard: cut only on actual upward reversal right above deck
        # so 1 m/s velocity sensor noise never false-triggers cutoff while still descending!
        meco3_cut = self.landing_single_engine and h_vref <= 0.22 and est.vz_m_s >= 0.05
        if meco3_cut:
            throttle = 0.0
            active_engines = 0

        # During cold restart ignition wait (engine not yet lit), hold zero-AoA retrograde
        # so high-q aerodynamic torque never spikes while grid fins finish retracting!
        lit_thr_threshold = 0.5 * self.propulsion.throttle_bounds(1)[0]
        if est.engine_throttle < lit_thr_threshold:
            vx_air_l, vz_air_l = self._air_velocity(est)
            target_theta = self._retro_angle(vx_air_l, vz_air_l)
        else:
            target_theta = theta_vb if vertical_body else direction
        deploy_fins = False
        deploy_gear = h <= s.gear_deploy_height_m

        # Protect the 0.5 deg/s touchdown angular-rate envelope once tilt is already well inside
        # the 3.0 deg touchdown limit near the pad.
        land_rate_lim = 9.0
        if vertical_body and abs(est.theta_rad) < 2.9 * (pi / 180.0):
            land_rate_lim = 0.22 + 5.78 * _smoothstep((h_vref - 0.65) / 3.1)
        vb_blend = (
            _smoothstep((est.time_s - (self._vb_started_s or est.time_s)) / 0.28)
            if vertical_body
            else 1.0
        )
        vb_acc_lim = 2.30 + 0.50 * vb_blend
        gimbal, rcs, fin = self._attitude_allocate(
            est,
            target_theta,
            throttle,
            rate_limit_deg_s=land_rate_lim,
            accel_limit_deg_s2=vb_acc_lim if vertical_body else 3.05,
            allow_fin=(est.fin_extension >= 0.05 and h > 300.0),
            thrust_direction=direction if (vertical_body and not meco3_cut) else None,
            high_q_guard=False,
            accel_guard_deg_s2=vb_acc_lim if vertical_body else 3.0,
        )
        # Keep gimbal centered during cold restart spool-up (allocator still balances tail-off
        # gimbal torque with RCS on MECO-3 so cutting thrust never leaves unbalanced TVC torque!)
        if est.engine_throttle < lit_thr_threshold:
            gimbal = 0.0
        self.last_diag = GuidanceDiagnostics(
            phase="landing_burn",
            attitude_target_deg=target_theta * (180.0 / pi),
            convex_solver_used=(self.convex_solution is not None and self.convex_solution.feasible),
            convex_solver_status=(
                self.convex_solution.status if self.convex_solution is not None else "unused"
            ),
            extra={"active_engines": active_engines, "vertical_body": vertical_body},
        )
        return ActuatorCommand(
            throttle_cmd=throttle,
            active_engines=active_engines,
            gimbal_rad=gimbal,
            rcs_torque_nm=rcs,
            fin_deflection_rad=fin,
            deploy_fins=deploy_fins,
            deploy_gear=deploy_gear,
            phase="landing_burn",
        )

    def step(self, telemetry: TelemetryPacket) -> ActuatorCommand:
        """Main 50 Hz GNC step: causal estimation -> RTLS phase guidance -> allocation."""
        est = self.estimator.observe(telemetry)

        # Phase 7: Post-touchdown immediate cutoff & settle
        if est.touchdown_detected:
            self.phase = "touchdown_settle"
            cmd = ActuatorCommand(
                throttle_cmd=0.0,
                active_engines=0,
                gimbal_rad=0.0,
                rcs_torque_nm=0.0,
                fin_deflection_rad=0.0,
                deploy_fins=True,
                deploy_gear=True,
                phase="touchdown_settle",
            )
            self.estimator.record_command(est.time_s, cmd)
            self.last_command = cmd
            self.last_diag = GuidanceDiagnostics(phase="touchdown_settle")
            return cmd

        # Phase 1: Ascent
        if est.upper_attached:
            self.phase = "ascent"
            cmd = self._ascent_guidance(est)
            self.estimator.record_command(est.time_s, cmd)
            self.last_command = cmd
            return cmd

        # Post-separation booster recovery!
        s = self.settings
        mass = self._nominal_mass(est)
        cg, _, _ = fast_mass_properties(
            mass,
            est.gear_extension,
            fin_mass_kg=self.fins.mass_kg,
            upper_attached=False,
        )
        h = self._foot_height(est, cg)
        _, _, _, q, _, _ = compute_body_aero(
            est.z_m,
            est.vx_m_s,
            est.vz_m_s,
            est.theta_rad,
            est.omega_rad_s,
            cg,
            self.forecast_atmo,
            self.forecast_wind,
            est.time_s,
            self.aero,
        )

        # Transition from ascent -> flip
        if self.phase == "ascent":
            self.phase = "flip"

        # Phase 2: Fast low-q RCS Flip
        if self.phase == "flip":
            if self.flip_started_s is None:
                if q > s.flip_max_q_pa:
                    vx_air, vz_air = self._air_velocity(est)
                    target = (
                        atan2(vx_air, vz_air) if est.vz_m_s > 0.0 else self._coast_attitude(est)
                    )
                    gimbal, rcs, fin = self._attitude_allocate(est, target, 0.0)
                    cmd = ActuatorCommand(
                        throttle_cmd=0.0,
                        active_engines=0,
                        gimbal_rad=gimbal,
                        rcs_torque_nm=rcs,
                        fin_deflection_rad=fin,
                        deploy_fins=False,
                        deploy_gear=False,
                        phase="flip",
                    )
                    self.estimator.record_command(est.time_s, cmd)
                    self.last_command = cmd
                    return cmd
                self.flip_started_s = est.time_s
                g_loc = fast_gravity(est.z_m)
                t_gnd = (
                    max(est.vz_m_s, 0.0) + sqrt(est.vz_m_s**2 + 2.0 * g_loc * max(est.z_m, 1.0))
                ) / g_loc
                self.boost_direction = -1.0 if est.x_m + est.vx_m_s * t_gnd >= 0.0 else 1.0

            target = self.boost_direction * (0.5 * pi)
            err = _wrap(target - est.theta_rad)
            if abs(err) < 1.2 * (pi / 180.0) and abs(est.omega_rad_s) < 0.6 * (pi / 180.0):
                self.phase = "boostback"
                self.boost_started_s = est.time_s
                self._plan_boostback(est)
            else:
                gimbal, rcs, fin = self._attitude_allocate(
                    est,
                    target,
                    0.0,
                    rate_limit_deg_s=s.flip_rate_limit_deg_s,
                    accel_limit_deg_s2=s.flip_accel_limit_deg_s2,
                    high_q_guard=False,
                )
                self.last_diag = GuidanceDiagnostics(
                    phase="flip",
                    attitude_target_deg=target * (180.0 / pi),
                )
                cmd = ActuatorCommand(
                    throttle_cmd=0.0,
                    active_engines=0,
                    gimbal_rad=gimbal,
                    rcs_torque_nm=rcs,
                    fin_deflection_rad=fin,
                    deploy_fins=False,
                    deploy_gear=False,
                    phase="flip",
                )
                self.estimator.record_command(est.time_s, cmd)
                self.last_command = cmd
                return cmd

        # Phase 3: Boostback Burn (Restart #1)
        if self.phase == "boostback":
            if est.time_s >= self.boost_end_s - 1e-8:
                self.phase = "coast"
            else:
                if (
                    self.controller_options.boost_replanning
                    and est.time_s >= self.next_replan_s
                    and self.boost_end_s - est.time_s > 0.12
                ):
                    self._plan_boostback(est)
                if est.time_s >= self.boost_end_s - 1e-8:
                    self.phase = "coast"
                else:
                    target = self.boost_direction * (0.5 * pi)
                    _, p_pa, _ = self.forecast_atmo.at(est.z_m)
                    full_t, _ = compute_engine_thrust(1.0, p_pa, self.propulsion)
                    min_m_bb = max(
                        self.design.dry_mass_kg - 1_000.0,
                        mass - 0.01 * self.design.launch_mass_kg,
                    )
                    max_bb_t = s.boostback_max_load_g * G0 * min_m_bb / 1.03
                    lo3, _ = self.propulsion.throttle_bounds(3)
                    bb_elapsed = est.time_s - (self.boost_started_s or est.time_s)
                    ramp = _smoothstep((bb_elapsed - 0.15) / 2.00)
                    bb_target_thr = lo3 + ramp * max(0.0, s.boostback_throttle_total - lo3)
                    throttle = self.propulsion.quantize_throttle(
                        min(bb_target_thr, max_bb_t / max(full_t, 1.0)), 3
                    )
                    gimbal, rcs, fin = self._attitude_allocate(
                        est,
                        target,
                        throttle,
                        rate_limit_deg_s=6.0,
                        accel_limit_deg_s2=0.75,
                        high_q_guard=False,
                        use_torque_bias=False,
                    )
                    self.last_diag = GuidanceDiagnostics(
                        phase="boostback",
                        attitude_target_deg=target * (180.0 / pi),
                    )
                    cmd = ActuatorCommand(
                        throttle_cmd=throttle,
                        active_engines=3,
                        gimbal_rad=gimbal,
                        rcs_torque_nm=rcs,
                        fin_deflection_rad=fin,
                        deploy_fins=False,
                        deploy_gear=False,
                        phase="boostback",
                    )
                    self.estimator.record_command(est.time_s, cmd)
                    self.last_command = cmd
                    return cmd

        # Phase 4 & 5: Coast -> Short Entry Burn -> Unpowered Grid-Fin Aerodynamic Glide
        if self.phase == "coast":
            deploy_fins = est.vz_m_s < 0.0 and est.z_m <= s.fin_deploy_altitude_m
            if est.vz_m_s < 0.0 and est.z_m < 68_000.0 and not self.entry_burn_done:
                _, _, rho_now = self.forecast_atmo.at(est.z_m)
                vx_air_c, vz_air_c = self._air_velocity(est)
                qdot_now = compute_stagnation_heat_flux_w_m2(
                    rho_now, hypot(vx_air_c, vz_air_c), PHYSICS.geometry.body_radius_m
                )
                if q >= s.entry_min_ignite_q_pa or qdot_now >= s.entry_min_ignite_heat_w_m2:
                    pred_q, pred_qdot = self._predict_unpowered_peak_q(
                        est.z_m, est.vx_m_s, est.vz_m_s, mass, est.time_s
                    )
                    self.cached_unpowered_q_pa = pred_q
                    if pred_q > s.entry_q_trigger_pa or pred_qdot > s.entry_heat_trigger_w_m2:
                        self.phase = "entry_burn"
                        self.entry_burn_started_s = est.time_s
                elif est.z_m < 22_000.0:
                    # Entry burn not needed (gentle reentry); enter aero glide directly!
                    self.entry_burn_done = True
                    self.phase = "aero_glide"
                    self.glide_started_s = est.time_s

            if self.phase == "coast":
                target = self._coast_attitude(est)
                gimbal, rcs, fin = self._attitude_allocate(
                    est,
                    target,
                    0.0,
                    allow_fin=deploy_fins,
                    rcs_deadband_nm=150.0 if q < 100.0 else 0.0,
                )
                self.last_diag = GuidanceDiagnostics(
                    phase="coast",
                    attitude_target_deg=target * (180.0 / pi),
                    predicted_unpowered_peak_q_pa=self.cached_unpowered_q_pa,
                )
                cmd = ActuatorCommand(
                    throttle_cmd=0.0,
                    active_engines=0,
                    gimbal_rad=gimbal,
                    rcs_torque_nm=rcs,
                    fin_deflection_rad=fin,
                    deploy_fins=deploy_fins,
                    deploy_gear=False,
                    phase="coast",
                )
                self.estimator.record_command(est.time_s, cmd)
                self.last_command = cmd
                return cmd

        # Phase 5: Entry Burn (Restart #2) -> MECO-2 once BOTH peak_q and heat_flux are safe!
        if self.phase == "entry_burn":
            pred_q, pred_qdot = self._predict_unpowered_peak_q(
                est.z_m, est.vx_m_s, est.vz_m_s, mass, est.time_s
            )
            self.cached_unpowered_q_pa = pred_q
            if (
                pred_q <= s.entry_q_cutoff_pa and pred_qdot <= s.entry_heat_cutoff_w_m2
            ) or est.z_m <= 11_000.0:
                self.entry_burn_done = True
                self.entry_burn_ended_s = est.time_s
                self.phase = "aero_glide"
                self.glide_started_s = est.time_s
            else:
                vx_air, vz_air = self._air_velocity(est)
                flow_retro = self._retro_angle(vx_air, vz_air)
                # Gentle +/-5 deg RTLS entry-burn steering to zero IIP into glide window:
                # thrust points along (+sin(theta), +cos(theta)), so when x_iip > 0 (East)
                # tilt theta slightly NEGATIVE (pushing West), and vice versa.
                if est.time_s >= self.last_iip_time_s + 0.40 - 1e-9:
                    x_iip_eb, _, _ = self._predict_rtls_iip(
                        est,
                        0.0,
                        include_entry_burn=False,
                        handoff_z_m=100.0,
                        dt=0.35,
                    )
                    self.cached_iip_x_m = x_iip_eb
                    self.last_iip_time_s = est.time_s
                eb_steer_lim_deg = 10.0 if q < 14_000.0 else (6.5 if q < 24_000.0 else 4.0)
                # Smoothly taper entry-burn steering back to 0 deg BEFORE MECO-2 cutoff
                # so TVC gimbal is centered before thrust tail-off (zero angular-accel spike)!
                q_margin_taper = _smoothstep((pred_q - s.entry_q_cutoff_pa) / 6_500.0)
                raw_eb_steer = (
                    max(
                        -eb_steer_lim_deg,
                        min(
                            eb_steer_lim_deg,
                            -0.042 * (self.cached_iip_x_m - self._upwind_iip_target_m(est.time_s)),
                        ),
                    )
                    * q_margin_taper
                )
                alpha_eb = 1.0 - exp(-s.control_dt_s / 0.45)
                self._eb_steer_smooth_deg += alpha_eb * (raw_eb_steer - self._eb_steer_smooth_deg)
                target = flow_retro + self._eb_steer_smooth_deg * (pi / 180.0)
                _, p_pa, _ = self.forecast_atmo.at(est.z_m)
                full_t, _ = compute_engine_thrust(1.0, p_pa, self.propulsion)
                bfx, bfz, _, q_land, _, _ = compute_body_aero(
                    est.z_m,
                    est.vx_m_s,
                    est.vz_m_s,
                    est.theta_rad,
                    est.omega_rad_s,
                    cg,
                    self.forecast_atmo,
                    self.forecast_wind,
                    est.time_s,
                    self.aero,
                )
                ffx, ffz, _, _, _ = compute_fin_aero(
                    est.z_m,
                    est.vx_m_s,
                    est.vz_m_s,
                    est.theta_rad,
                    cg,
                    est.fin_deflection_rad,
                    est.fin_extension,
                    self.forecast_atmo,
                    self.forecast_wind,
                    est.time_s,
                    self.fins,
                )
                aero_mag = hypot(bfx + ffx, bfz + ffz)
                min_m_eb = max(
                    self.design.dry_mass_kg - 1_000.0,
                    mass - 0.01 * self.design.launch_mass_kg,
                )
                max_thrust = max(0.0, (s.entry_max_load_g * G0 * min_m_eb - 1.15 * aero_mag) / 1.03)
                lo3, hi3 = self.propulsion.throttle_bounds(3)
                raw_thr = min(hi3, max_thrust / max(full_t, 1.0))
                if q > 12_000.0 and raw_thr < lo3:
                    self.entry_single_engine = True
                eb_engines = 1 if self.entry_single_engine else 3
                eb_elapsed = est.time_s - (self.entry_burn_started_s or est.time_s)
                eb_ramp = _smoothstep(eb_elapsed / 0.85)
                lo_eb = self.propulsion.throttle_bounds(eb_engines)[0]
                throttle = self.propulsion.quantize_throttle(
                    lo_eb + eb_ramp * max(0.0, raw_thr - lo_eb), eb_engines
                )
                # Ensure angular alignment before high thrust
                if abs(_wrap(target - est.theta_rad)) > 15.0 * (pi / 180.0):
                    throttle = self.propulsion.quantize_throttle(lo_eb, eb_engines)

                gimbal, rcs, fin = self._attitude_allocate(
                    est,
                    target,
                    throttle,
                    rate_limit_deg_s=6.5,
                    accel_limit_deg_s2=1.6,
                    allow_fin=True,
                    use_torque_bias=(eb_elapsed > 1.2),
                )
                self.last_diag = GuidanceDiagnostics(
                    phase="entry_burn",
                    attitude_target_deg=target * (180.0 / pi),
                    predicted_unpowered_peak_q_pa=pred_q,
                )
                cmd = ActuatorCommand(
                    throttle_cmd=throttle,
                    active_engines=eb_engines,
                    gimbal_rad=gimbal,
                    rcs_torque_nm=rcs,
                    fin_deflection_rad=fin,
                    deploy_fins=True,
                    deploy_gear=False,
                    phase="entry_burn",
                )
                self.estimator.record_command(est.time_s, cmd)
                self.last_command = cmd
                return cmd

        # Phase 6: Unpowered Grid-Fin Aerodynamic Glide (0 kg main propellant!)
        if self.phase == "aero_glide":
            h_ignite = self._compute_hoverslam_ignition_height(est, mass)
            if h <= h_ignite:
                self.phase = "landing_burn"
                self.landing_started_s = est.time_s
            else:
                if est.time_s >= self.last_iip_time_s + s.glide_iip_period_s - 1e-9:
                    x_iip, _, _ = self._predict_rtls_iip(
                        est,
                        0.0,
                        include_entry_burn=False,
                        handoff_z_m=100.0,
                        dt=0.20,
                        use_nav_bias=self.controller_options.translation_bias_compensation,
                    )
                    self.cached_iip_x_m = x_iip
                    self.last_iip_time_s = est.time_s

                vx_air, vz_air = self._air_velocity(est)
                flow_retro = self._retro_angle(vx_air, vz_air)

                # Aerodynamic glide law:
                #   x_iip > 0 (East overshoot) -> alpha_cmd > 0 (nose CW) -> body lift West (-x)
                #   x_iip < 0 (West undershoot) -> alpha_cmd < 0 (nose CCW) -> body lift East (+x)
                h_above_ign = max(h - h_ignite, 150.0)
                k_glide_deg_per_m = max(
                    0.07, min(0.28, 0.11 * (5_000.0 / max(h_above_ign, 1_500.0)))
                )

                # Keep glide AoA inside the grid-fin mechanical deflection trim envelope:
                # supersonic high-altitude (z >= 27 km) fin effectiveness drops as 1/sqrt(M^2-1)
                # so cap AoA <= 5 deg there; below z = 19 km use full subsonic/transonic q_cap.
                pre_retract_taper = _smoothstep((h - (h_ignite + 300.0)) / 420.0)
                hi_z_cap = 5.0 + (s.glide_max_alpha_deg - 5.0) * (
                    1.0 - _smoothstep((est.z_m - 19_000.0) / 8_000.0)
                )
                q_cap = min(
                    hi_z_cap,
                    s.glide_max_alpha_deg - 8.5 * _smoothstep((q - 3_500.0) / 6_500.0),
                )
                max_alpha = q_cap * pre_retract_taper

                # Causal upwind landing-burn drift lead: the unpowered freefall IIP spends
                # only ~8 s below h_ignite, whereas the actual 3->1 Hoverslam takes ~24 s
                # exposed to low-altitude crosswind. Aiming upwind in proportion to
                # low-altitude wind and causally estimated air-density/drag scale cancels
                # the extra powered-descent wind drift cleanly before ignition!
                x_glide_target = 0.0
                if self.controller_options.translation_bias_compensation and est.z_m < 14_000.0:
                    t_g, _, rho_g = self.forecast_atmo.at(est.z_m)
                    wx_g = self.forecast_wind.horizontal(est.z_m, est.time_s)
                    spd_g = hypot(est.vx_m_s - wx_g, est.vz_m_s)
                    q_g = 0.5 * rho_g * spd_g * spd_g
                    if q_g > 1_500.0 and est.vz_m_s < -50.0:
                        r_g = PHYSICS.geometry.body_radius_m
                        m_g = spd_g / sqrt(1.4 * 287.05287 * t_g)
                        c_g = 1.0 + self.aero.transonic_drag_rise * exp(-(((m_g - 1.0) / 0.2) ** 2))
                        f_nz = (
                            q_g
                            * (
                                c_g * self.aero.axial_cd_descent * (pi * r_g * r_g)
                                + self.fins.planar_effective_area_m2
                                * self.fins.base_drag_coefficient
                            )
                            * (-est.vz_m_s / max(spd_g, 1.0))
                        )
                        a_nz = f_nz / max(mass, 1.0)
                        if a_nz > 2.5:
                            d_est = max(0.75, min(1.35, 1.0 + est.accel_bias_z_m_s2 / a_nz))
                            self.cached_drag_scale += 0.04 * (
                                max(0.78, min(1.32, d_est)) - self.cached_drag_scale
                            )
                            dense_w = max(0.0, min(1.4, (self.cached_drag_scale - 0.94) / 0.24))
                            w_gnd = self.forecast_wind.horizontal(150.0, est.time_s)
                            w_mid = self.forecast_wind.horizontal(2_000.0, est.time_s)
                            w_hi = self.forecast_wind.horizontal(5_000.0, est.time_s)
                            if w_gnd * w_mid > 0.0 and w_gnd * w_hi > 0.0:
                                w_low = 0.65 * w_gnd + 0.35 * w_mid
                                x_glide_target = max(-65.0, min(65.0, -6.5 * w_low * dense_w))

                glide_start_ramp = _smoothstep(
                    (est.time_s - (self.glide_started_s or est.time_s)) / 2.5
                )
                raw_alpha_cmd_deg = (
                    max(
                        -max_alpha,
                        min(max_alpha, k_glide_deg_per_m * (self.cached_iip_x_m - x_glide_target)),
                    )
                    * glide_start_ramp
                )
                # Hold zero-AoA retrograde while MECO-2 engine tail-off thrust decays to 0
                # so residual TVC gimbal never couples with a sudden grid-fin AoA step!
                if est.engine_throttle > 0.01:
                    raw_alpha_cmd_deg = 0.0
                alpha_gl_filt = 1.0 - exp(-s.control_dt_s / 0.35)
                self._glide_alpha_smooth_deg += alpha_gl_filt * (
                    raw_alpha_cmd_deg - self._glide_alpha_smooth_deg
                )
                alpha_cmd_deg = self._glide_alpha_smooth_deg
                target = flow_retro + alpha_cmd_deg * (pi / 180.0)
                # Check actual body AoA before starting fin retraction to avoid folding under load
                c_e, sn_e = cos(est.theta_rad), sin(est.theta_rad)
                vn_e = c_e * vx_air - sn_e * vz_air
                va_e = sn_e * vx_air + c_e * vz_air
                alpha_now_deg = abs(atan2(vn_e, max(-va_e, 1.0)) * (180.0 / pi))
                deploy_fins_glide = (h > h_ignite + 260.0) or (
                    h > h_ignite + 160.0 and alpha_now_deg > 2.0
                )

                gimbal, rcs, fin = self._attitude_allocate(
                    est,
                    target,
                    0.0,
                    rate_limit_deg_s=9.0,
                    accel_limit_deg_s2=2.2,
                    allow_fin=True,
                    rcs_deadband_nm=5_000.0 if (q > 1_000.0 and est.fin_extension >= 0.4) else 0.0,
                    accel_guard_deg_s2=2.25,
                )
                self.last_diag = GuidanceDiagnostics(
                    phase="aero_glide",
                    attitude_target_deg=target * (180.0 / pi),
                    glide_alpha_cmd_deg=alpha_cmd_deg,
                    predicted_iip_x_m=self.cached_iip_x_m,
                    hoverslam_ignition_height_m=h_ignite,
                )
                cmd = ActuatorCommand(
                    throttle_cmd=0.0,
                    active_engines=0,
                    gimbal_rad=gimbal,
                    rcs_torque_nm=rcs,
                    fin_deflection_rad=fin,
                    deploy_fins=deploy_fins_glide,
                    deploy_gear=(h <= s.gear_deploy_height_m),
                    phase="aero_glide",
                )
                self.estimator.record_command(est.time_s, cmd)
                self.last_command = cmd
                return cmd

        # Phase 7: 3->1 Engine Min-Throttle Vertical Hoverslam
        cmd = self._landing_burn_guidance(est, mass, cg, h)
        self.estimator.record_command(est.time_s, cmd)
        self.last_command = cmd
        return cmd

    def diagnostics(self) -> GuidanceDiagnostics:
        return self.last_diag


def log_safe(val: float) -> float:
    from math import log

    return log(max(val, 1e-30))
