"""Strictly causal state estimator consuming only TelemetryPacket and command history.

Wraps the validated PlanarNavigationEstimator and PlanarAttitudeEstimator while
isolating the GNC core from any plant internals or hidden model parameters.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import atan2, cos, exp, pi, sin

import numpy as np

from powered_landing_guidance.apr_recovery import PHYSICS, Design
from powered_landing_guidance.observation_models import (
    ControllerOptions,
    PlanarAttitudeEstimator,
    SensorSettings,
)
from powered_landing_guidance.recovery_physics import EARTH_RADIUS_M, G0, PlanarNavigationEstimator
from powered_landing_guidance.rtls.forces import (
    OMEGA_EARTH,
    AtmosphereLUT,
    RtlsAerodynamics,
    RtlsFinSettings,
    RtlsPropulsionSettings,
    WindLUT,
    compute_forces_and_torques,
    fast_gravity,
    fast_mass_properties,
)
from powered_landing_guidance.rtls.interfaces import ActuatorCommand, TelemetryPacket


@dataclass(frozen=True, slots=True)
class EstimatedState:
    time_s: float
    x_m: float
    z_m: float
    vx_m_s: float
    vz_m_s: float
    theta_rad: float
    omega_rad_s: float
    main_propellant_kg: float
    rcs_propellant_kg: float
    gear_extension: float
    fin_deflection_rad: float
    fin_extension: float
    gimbal_rad: float
    engine_throttle: float
    upper_attached: bool
    on_pad: bool
    touchdown_detected: bool
    active_engines: int
    restarts_used: int
    accel_bias_x_m_s2: float
    accel_bias_z_m_s2: float
    angular_torque_bias_nm: float
    estimated_gyro_bias_deg_s: float | None


class CausalRtlsEstimator:
    """Causal navigation, attitude, actuator-lag, and torque-bias estimator."""

    def __init__(
        self,
        *,
        design: Design,
        aero: RtlsAerodynamics,
        fins: RtlsFinSettings,
        propulsion: RtlsPropulsionSettings,
        forecast_atmo: AtmosphereLUT,
        forecast_wind: WindLUT,
        latitude_deg: float,
        coriolis_enabled: bool,
        position_error_m: float,
        velocity_error_m_s: float,
        sensor_settings: SensorSettings,
        controller_options: ControllerOptions,
        control_dt_s: float = 0.02,
        action_delay_steps: int = 0,
    ) -> None:
        self.design = design
        self.aero = aero
        self.fins = fins
        self.propulsion = propulsion
        self.forecast_atmo = forecast_atmo
        self.forecast_wind = forecast_wind
        self.latitude_deg = latitude_deg
        self.coriolis_enabled = coriolis_enabled
        self.sensor_settings = sensor_settings
        self.controller_options = controller_options
        self.control_dt_s = control_dt_s
        self.action_delay_steps = action_delay_steps

        self.nav = PlanarNavigationEstimator(position_error_m, velocity_error_m_s)
        self.att = PlanarAttitudeEstimator(
            sensor_settings.attitude_error_deg,
            sensor_settings.gyro_error_deg_s,
        )
        self.command_history: deque[tuple[float, ActuatorCommand]] = deque(maxlen=64)
        self.prev_angular: tuple[float, ...] | None = None
        self.angular_torque_bias_nm: float = 0.0
        self._separation_time_s: float | None = None
        self._last_upper_attached: bool = True

    def record_command(self, time_s: float, command: ActuatorCommand) -> None:
        exec_end_s = time_s + (self.action_delay_steps + 1) * self.control_dt_s
        self.command_history.append((exec_end_s, command))

    def _nominal_mass(self, main_kg: float, rcs_kg: float, upper_attached: bool) -> float:
        return (
            self.design.dry_mass_kg
            + max(main_kg, 0.0)
            + max(rcs_kg, 0.0)
            + (self.design.upper_mass_kg if upper_attached else 0.0)
        )

    def _nominal_acceleration(
        self, pkt: TelemetryPacket, theta: float, omega: float
    ) -> tuple[float, float]:
        mass = self._nominal_mass(pkt.main_propellant_kg, pkt.rcs_propellant_kg, pkt.upper_attached)
        cg, _, _ = fast_mass_properties(
            mass,
            pkt.gear_extension,
            fin_mass_kg=self.fins.mass_kg,
            upper_attached=pkt.upper_attached,
            upper_mass_kg=self.design.upper_mass_kg,
        )
        ft = compute_forces_and_torques(
            z_m=pkt.z_m,
            vx_m_s=pkt.vx_m_s,
            vz_m_s=pkt.vz_m_s,
            theta_rad=theta,
            omega_rad_s=omega,
            cg_m=cg,
            throttle_actual=max(0.0, min(1.0, pkt.engine_throttle)),
            gimbal_rad=pkt.gimbal_rad,
            rcs_torque_nm=0.0,
            fin_deflection_rad=pkt.fin_deflection_rad,
            fin_extension=pkt.fin_extension,
            atmo_lut=self.forecast_atmo,
            wind_lut=self.forecast_wind,
            time_s=pkt.measurement_time_s,
            aero=self.aero,
            fins=self.fins,
            propulsion=self.propulsion,
            upper_attached=pkt.upper_attached,
        )
        g_loc = fast_gravity(pkt.z_m)
        if pkt.on_pad and pkt.upper_attached and ft.force_z_n < mass * g_loc:
            return 0.0, 0.0
        ax = ft.force_x_n / mass
        r_orb = EARTH_RADIUS_M + max(pkt.z_m, 0.0)
        az = ft.force_z_n / mass - g_loc + (pkt.vx_m_s * pkt.vx_m_s) / r_orb
        if self.coriolis_enabled:
            om = OMEGA_EARTH * cos(self.latitude_deg * (pi / 180.0))
            ax += -2.0 * om * pkt.vz_m_s
            az += 2.0 * om * pkt.vx_m_s
        return ax, az

    def _predict_actuators(self, pkt: TelemetryPacket) -> tuple[float, float, float]:
        """Propagate delayed gimbal, throttle, and fin telemetry through issued commands."""
        gimbal = pkt.gimbal_rad
        throttle = pkt.engine_throttle
        fin = pkt.fin_deflection_rad
        begin = pkt.measurement_time_s
        g_rate = self.propulsion.gimbal_rate_deg_s * (pi / 180.0)
        g_tau = self.propulsion.gimbal_tau_s
        f_rate = self.fins.deflection_rate_deg_s * (pi / 180.0)
        f_tau = self.fins.actuator_time_constant_s
        e_tau = self.propulsion.nominal_engine_tau_s

        for end, cmd in self.command_history:
            if end <= begin + 1e-12:
                continue
            dur = min(end, pkt.time_s) - begin
            if dur <= 0.0:
                continue
            throttle = cmd.throttle_cmd + (throttle - cmd.throttle_cmd) * exp(-dur / e_tau)
            # Gimbal
            dg = cmd.gimbal_rad - gimbal
            lin_g = min(dur, max(0.0, (abs(dg) - g_rate * g_tau) / g_rate))
            gimbal += (1.0 if dg >= 0.0 else -1.0) * g_rate * lin_g
            gimbal = cmd.gimbal_rad + (gimbal - cmd.gimbal_rad) * exp(-(dur - lin_g) / g_tau)
            # Fin
            df = cmd.fin_deflection_rad - fin
            lin_f = min(dur, max(0.0, (abs(df) - f_rate * f_tau) / f_rate))
            fin += (1.0 if df >= 0.0 else -1.0) * f_rate * lin_f
            fin = cmd.fin_deflection_rad + (fin - cmd.fin_deflection_rad) * exp(
                -(dur - lin_f) / f_tau
            )
            begin += dur
        return gimbal, max(0.0, min(1.0, throttle)), fin

    def _update_torque_bias(self, pkt: TelemetryPacket, theta: float, omega: float) -> None:
        if not self.controller_options.torque_bias_compensation:
            self.angular_torque_bias_nm = 0.0
            return
        if self._last_upper_attached and not pkt.upper_attached:
            self._separation_time_s = pkt.time_s
        self._last_upper_attached = pkt.upper_attached

        if (
            pkt.upper_attached
            or pkt.on_pad
            or pkt.touchdown_detected
            or (
                self._separation_time_s is not None
                and pkt.measurement_time_s < self._separation_time_s + 0.25
            )
        ):
            self.prev_angular = None
            self.angular_torque_bias_nm = 0.0
            return

        mass = self._nominal_mass(pkt.main_propellant_kg, pkt.rcs_propellant_kg, False)
        cg, inertia, _ = fast_mass_properties(
            mass,
            pkt.gear_extension,
            fin_mass_kg=self.fins.mass_kg,
            upper_attached=False,
        )
        cur = (
            pkt.measurement_time_s,
            theta,
            omega,
            inertia,
            pkt.z_m,
            pkt.vx_m_s,
            pkt.vz_m_s,
            pkt.engine_throttle,
            pkt.gimbal_rad,
            pkt.fin_deflection_rad,
            pkt.fin_extension,
        )
        prev = self.prev_angular
        if prev is not None and pkt.measurement_time_s <= prev[0] + 1e-9:
            return
        self.prev_angular = cur
        if prev is None:
            return

        dt = pkt.measurement_time_s - prev[0]
        if dt <= 1e-6 or dt > 0.2:
            return

        # Integrate executed RCS commands over [prev[0], measurement_time_s]
        rcs_int = 0.0
        rcs_abs_int = 0.0
        t_pos = prev[0]
        for end, cmd in self.command_history:
            if end <= t_pos + 1e-9:
                continue
            if end - self.control_dt_s > t_pos + 1e-9:
                break
            ov = min(end, pkt.measurement_time_s) - t_pos
            if ov > 0.0:
                rcs_val = cmd.rcs_torque_nm if pkt.rcs_propellant_kg > 0.0 else 0.0
                rcs_int += ov * rcs_val
                rcs_abs_int += ov * abs(rcs_val)
                t_pos += ov
            if t_pos >= pkt.measurement_time_s - 1e-9:
                break
        if t_pos < pkt.measurement_time_s - 1e-9:
            return

        mid_theta = prev[1] + 0.5 * atan2(sin(theta - prev[1]), cos(theta - prev[1]))
        mid_omega = 0.5 * (prev[2] + omega)
        mid_z = 0.5 * (prev[4] + pkt.z_m)
        mid_vx = 0.5 * (prev[5] + pkt.vx_m_s)
        mid_vz = 0.5 * (prev[6] + pkt.vz_m_s)
        mid_thr = 0.5 * (prev[7] + pkt.engine_throttle)
        mid_gimbal = 0.5 * (prev[8] + pkt.gimbal_rad)
        mid_fin = 0.5 * (prev[9] + pkt.fin_deflection_rad)
        mid_fin_ext = 0.5 * (prev[10] + pkt.fin_extension)
        ft = compute_forces_and_torques(
            z_m=mid_z,
            vx_m_s=mid_vx,
            vz_m_s=mid_vz,
            theta_rad=mid_theta,
            omega_rad_s=mid_omega,
            cg_m=cg,
            throttle_actual=mid_thr,
            gimbal_rad=mid_gimbal,
            rcs_torque_nm=rcs_int / dt,
            fin_deflection_rad=mid_fin,
            fin_extension=mid_fin_ext,
            atmo_lut=self.forecast_atmo,
            wind_lut=self.forecast_wind,
            time_s=0.5 * (prev[0] + pkt.measurement_time_s),
            aero=self.aero,
            fins=self.fins,
            propulsion=self.propulsion,
            upper_attached=False,
        )
        lever = cg - PHYSICS.geometry.nozzle_station_m
        rcs_flow = rcs_abs_int / (dt * PHYSICS.geometry.rcs_arm_m * self.propulsion.rcs_isp_s * G0)
        flux = (
            -ft.main_flow_kg_s * lever * lever * mid_omega
            - rcs_flow
            * (PHYSICS.geometry.rcs_arm_m**2 + (PHYSICS.geometry.rcs_station_m - cg) ** 2)
            * mid_omega
        )

        old_L = prev[3] * prev[2]
        new_L = inertia * omega
        residual = (new_L - old_L) / dt - (ft.torque_nm + flux)
        bound = max(inertia * (15.0 * pi / 180.0), 150_000.0)
        residual = max(-bound, min(bound, residual))
        alpha = 1.0 - exp(-dt / 0.12)
        self.angular_torque_bias_nm += alpha * (residual - self.angular_torque_bias_nm)
        self.angular_torque_bias_nm = max(-bound, min(bound, self.angular_torque_bias_nm))

    def observe(self, pkt: TelemetryPacket) -> EstimatedState:
        raw6 = np.array(
            [pkt.x_m, pkt.z_m, pkt.vx_m_s, pkt.vz_m_s, pkt.theta_rad, pkt.omega_rad_s],
            dtype=np.float64,
        )
        if self.sensor_settings.angular_noise_enabled and self.controller_options.attitude_filter:
            filtered = self.att.observe(raw6, pkt.measurement_time_s)
            theta_meas = float(filtered[4])
            omega_meas = float(filtered[5])
            gyro_bias_deg_s = (
                float(self.att.estimate[2] * (180.0 / pi))
                if self.att.estimate is not None
                else None
            )
        else:
            theta_meas = pkt.theta_rad
            omega_meas = pkt.omega_rad_s
            gyro_bias_deg_s = None

        self._update_torque_bias(pkt, theta_meas, omega_meas)

        delay = max(0.0, pkt.time_s - pkt.measurement_time_s)
        acc_x, acc_z = self._nominal_acceleration(pkt, theta_meas, omega_meas)
        raw6[4] = theta_meas
        raw6[5] = omega_meas
        nav_out = self.nav.observe(raw6, pkt.time_s, (acc_x, acc_z), delay)

        theta_now = float(theta_meas + omega_meas * delay)
        gimbal_now, throttle_now, fin_now = self._predict_actuators(pkt)
        bx = float(self.nav.estimate[0, 2]) if self.nav.estimate is not None else 0.0
        bz = float(self.nav.estimate[1, 2]) if self.nav.estimate is not None else 0.0

        return EstimatedState(
            time_s=pkt.time_s,
            x_m=float(nav_out[0]),
            z_m=float(nav_out[1]),
            vx_m_s=float(nav_out[2]),
            vz_m_s=float(nav_out[3]),
            theta_rad=theta_now,
            omega_rad_s=omega_meas,
            main_propellant_kg=pkt.main_propellant_kg,
            rcs_propellant_kg=pkt.rcs_propellant_kg,
            gear_extension=pkt.gear_extension,
            fin_deflection_rad=fin_now,
            fin_extension=pkt.fin_extension,
            gimbal_rad=gimbal_now,
            engine_throttle=throttle_now,
            upper_attached=pkt.upper_attached,
            on_pad=pkt.on_pad,
            touchdown_detected=pkt.touchdown_detected,
            active_engines=pkt.active_engines,
            restarts_used=pkt.restarts_used,
            accel_bias_x_m_s2=bx,
            accel_bias_z_m_s2=bz,
            angular_torque_bias_nm=self.angular_torque_bias_nm,
            estimated_gyro_bias_deg_s=gyro_bias_deg_s,
        )
