"""High-speed 2D 19-state rigid-body Plant implementing PlantProtocol.

Reuses DM100's exact coupled vehicle subsystems (propellant feed buffer,
closed isothermal helium pressurization, nonlinear slosh pendulum, and
momentum-conserving cold stage separation) while running ~20-40x faster
with fast LUT-backed float64 RK4 integration, post-touchdown landing-leg contact dynamics, and 50 Hz
ENU/quaternion replay recording.

In Stage 2, this class is swapped directly for a 3D MuJoCo Plant while
reusing the same Force Model concepts, GNC Core, and Replay Schema.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import cos, exp, hypot, pi, sin, sqrt

import numpy as np
from scipy.optimize import brentq

from powered_landing_guidance.apr_recovery import (
    PHYSICS,
    PROFILE_TOP_M,
    SAFETY,
    STATE_NAMES,
    AprConditions,
    AscentGuidanceSettings,
    Design,
    RecoveryDisturbances,
)
from powered_landing_guidance.integrators import rk4_step
from powered_landing_guidance.observation_models import SensorSettings
from powered_landing_guidance.recovery_physics import G0, AtmosphereSettings, LandingAcceptance
from powered_landing_guidance.rtls.contact import ContactSettings, evaluate_ground_contact
from powered_landing_guidance.rtls.forces import (
    OMEGA_EARTH,
    AtmosphereLUT,
    RtlsAerodynamics,
    RtlsFinSettings,
    RtlsPropulsionSettings,
    WindLUT,
    compute_body_aero,
    compute_engine_thrust,
    compute_forces_and_torques,
    fast_gravity,
    fast_mass_gradients,
    fast_mass_properties,
    foot_kinematics_scalar,
)
from powered_landing_guidance.rtls.interfaces import (
    ActuatorCommand,
    ContactSample,
    ReplayFrame,
    TelemetryPacket,
    pitch_to_quat_wxyz,
)
from powered_landing_guidance.vehicle_systems import (
    FeedSettings,
    FeedState,
    IgnitionSettings,
    SeparationSettings,
    SloshSettings,
    SloshState,
    feed_rates,
    ignition_ready,
    initial_feed_state,
    separation_event,
    slosh_displacement,
    slosh_response,
)


@dataclass(frozen=True, slots=True)
class PlantConfig:
    integration_dt_s: float = 0.01
    control_dt_s: float = 0.02
    contact_dt_s: float = 0.002
    deadline_s: float = 520.0
    landing_mode: str = "contact_dynamics"  # "contact_dynamics" or "kinematic_touchdown"
    replay_hz: float = 50.0
    record_replay: bool = True
    seed: int = 378
    sensor_seed: int = 1378


class Plant2D:
    """Fast 2D coupled rigid-body rocket plant with sensors, ignition, separation & contact."""

    def __init__(
        self,
        *,
        design: Design,
        conditions: AprConditions,
        disturbances: RecoveryDisturbances,
        sensor_settings: SensorSettings,
        aero: RtlsAerodynamics,
        fins: RtlsFinSettings,
        propulsion: RtlsPropulsionSettings,
        contact_settings: ContactSettings,
        ascent_settings: AscentGuidanceSettings | None = None,
        config: PlantConfig | None = None,
    ) -> None:
        self.design = design
        self.conditions = conditions
        self.disturbances = disturbances
        self.sensor_settings = sensor_settings
        self.aero = aero
        self.fins = fins
        self.propulsion = propulsion
        self.contact_settings = contact_settings
        self.ascent_settings = ascent_settings or AscentGuidanceSettings()
        self.config = config or PlantConfig()

        # True atmosphere (includes hidden density_scale!) & true wind (includes hidden gusts!)
        actual_atmo_settings = AtmosphereSettings(
            sea_level_temperature_k=conditions.sea_level_temperature_k,
            sea_level_pressure_pa=conditions.sea_level_pressure_pa,
            latitude_deg=conditions.latitude_deg,
            coriolis_enabled=True,
            temperature_profile=conditions.temperature_profile,
            pressure_profile=conditions.pressure_profile,
            density_scale=conditions.density_scale,
        )
        self.atmo_lut = AtmosphereLUT(actual_atmo_settings, step_m=100.0)
        self.wind_lut = WindLUT(conditions, forecast=False)
        self.coriolis_om = OMEGA_EARTH * cos(conditions.latitude_deg * (pi / 180.0))

        self.feed_settings = FeedSettings()
        self.ignition_settings = IgnitionSettings()
        self.slosh_settings = SloshSettings()
        self.separation_settings = SeparationSettings()
        self.acceptance = LandingAcceptance(
            max_position_error_m=2.0,
            max_horizontal_contact_speed_m_s=0.5,
            max_vertical_foot_speed_m_s=1.5,
            max_contact_tilt_deg=3.0,
            max_contact_rate_deg_s=0.5,
        )

        # Hidden mass perturbation applied to equivalent dry airframe structure
        self.airframe_mass_offset_kg = (
            disturbances.initial_mass_scale - 1.0
        ) * design.launch_mass_kg
        self.actual_dry_mass_kg = design.dry_mass_kg + self.airframe_mass_offset_kg
        self.engine_tau_s = max(propulsion.nominal_engine_tau_s, disturbances.engine_lag_s)

        # Initial 19-element state matching DM100 STATE_NAMES identically
        feed0 = initial_feed_state(design.main_propellant_kg, self.feed_settings)
        init_mass = (
            self.actual_dry_mass_kg
            + design.main_propellant_kg
            + design.rcs_propellant_kg
            + design.upper_mass_kg
        )
        cg0, _, _ = fast_mass_properties(
            init_mass,
            0.0,
            fin_mass_kg=fins.mass_kg,
            airframe_mass_offset_kg=self.airframe_mass_offset_kg,
            upper_attached=True,
            upper_mass_kg=design.upper_mass_kg,
            inertia_scale=disturbances.inertia_scale,
        )
        self.state = np.zeros(len(STATE_NAMES), dtype=np.float64)
        self.state[1] = cg0 - PHYSICS.geometry.nozzle_station_m
        self.state[6] = design.main_propellant_kg
        self.state[7] = design.rcs_propellant_kg
        self.state[13] = feed0.buffer_kg
        self.state[14] = feed0.acquisition
        self.state[15] = feed0.tank_gas_kg
        self.state[16] = feed0.bottle_gas_kg

        self.time_s: float = 0.0
        self.upper_attached: bool = True
        self.on_pad: bool = True
        self.phase: str = "ascent"
        self.outcome: str = "running"
        self.failures: list[str] = []

        # Propulsion restart & ignition state
        self.active_engines: int = 0
        self.restarts_used: int = 0
        self._engine_lit_before: bool = False
        self._ignition_requested: bool = False
        self._ignition_ready_s: float = float("inf")

        # Sensor RNGs & delay queues
        self.rng = np.random.default_rng(self.config.seed)
        self.sensor_rng = np.random.default_rng(self.config.sensor_seed)
        self.obs_queue: deque[TelemetryPacket] = deque(
            maxlen=disturbances.observation_delay_steps + 1
        )
        self.action_queue: deque[ActuatorCommand] = deque(
            [ActuatorCommand()] * (disturbances.action_delay_steps + 1),
            maxlen=disturbances.action_delay_steps + 1,
        )
        self.sensor_sample_count: int = 0
        self.sensor_dropout_count: int = 0
        self._last_packet: TelemetryPacket | None = None

        # Monitoring extrema & accounting
        self.main_consumed_kg: float = 0.0
        self.rcs_consumed_kg: float = 0.0
        self.max_q_pa: float = 0.0
        self.max_q_alpha_kpa_deg: float = 0.0
        self.max_ascent_high_q_alpha_deg: float = 0.0
        self.max_ascent_axial_g: float = 0.0
        self.max_recovery_g: float = 0.0
        self.max_rate_deg_s: float = 0.0
        self.max_accel_deg_s2: float = 0.0
        self.max_gimbal_deg: float = 0.0
        self.max_fin_hinge_nm: float = 0.0
        self.max_fin_deflection_deg: float = 0.0
        self.max_mass_balance_err: float = 0.0
        self.max_helium_balance_err: float = 0.0
        self.observed_max_q_passed: bool = False
        self._peak_ascent_q: float = 0.0
        self.fin_model_ok: bool = True
        self.feed_inventory_ok: bool = True
        self.feed_supply_ok: bool = True
        self.slosh_domain_ok: bool = True

        # Separation, first-touchdown, and post-contact stability records
        self.separation_record: dict | None = None
        self.upper_state: dict | None = None
        self.first_contact_record: dict | None = None
        self.touchdown_time_s: float | None = None
        self.max_leg_stroke_m: float = 0.0
        self.max_leg_force_n: float = 0.0
        self.max_contact_load_g: float = 0.0
        self.body_ground_collision: bool = False
        self.post_contact_max_tilt_deg: float = 0.0
        self.standing_stable: bool | None = None

        # Phase-level telemetry summaries
        self.phase_stats: dict[str, dict] = {}
        self.replay_frames: list[ReplayFrame] = []
        self._next_replay_s: float = 0.0
        self._replay_dt_s: float = 1.0 / max(self.config.replay_hz, 1.0)
        self._last_contact_sample = ContactSample()

    @property
    def done(self) -> bool:
        return self.outcome != "running"

    def _total_mass(self, st: np.ndarray) -> float:
        return (
            self.actual_dry_mass_kg
            + max(float(st[6]), 0.0)
            + max(float(st[7]), 0.0)
            + (self.design.upper_mass_kg if self.upper_attached else 0.0)
        )

    def _mass_props(self, st: np.ndarray) -> tuple[float, float, float]:
        return fast_mass_properties(
            self._total_mass(st),
            float(st[8]),
            fin_mass_kg=self.fins.mass_kg,
            airframe_mass_offset_kg=self.airframe_mass_offset_kg,
            upper_attached=self.upper_attached,
            upper_mass_kg=self.design.upper_mass_kg,
            inertia_scale=self.disturbances.inertia_scale,
        )

    def _body_offset(self, st: np.ndarray, mass: float) -> tuple[float, float]:
        main = max(float(st[6]), 0.0)
        ratio = self.slosh_settings.modal_mass_fraction * main / max(mass, 1.0)
        disp, _ = slosh_displacement(SloshState(float(st[17]), float(st[18])), self.slosh_settings)
        return float(-ratio * disp[0]), float(-ratio * disp[1])

    def _foot_height(self, st: np.ndarray) -> float:
        mass = self._total_mass(st)
        cg, _, _ = self._mass_props(st)
        ox, oz = self._body_offset(st, mass)
        (_, lpz), (_, rpz), _, _ = foot_kinematics_scalar(
            float(st[0]),
            float(st[1]),
            float(st[2]),
            float(st[3]),
            float(st[4]),
            float(st[5]),
            cg,
            float(st[8]),
            offset_x=ox,
            offset_z=oz,
        )
        return min(lpz, rpz)

    def _apply_ignition_interlock(self, cmd: ActuatorCommand) -> ActuatorCommand:
        """Enforce physical restart budget, ignition readiness, and 0.2 s ignition delay."""
        if cmd.throttle_cmd <= 1e-6 or cmd.active_engines <= 0 or self.touchdown_time_s is not None:
            self._ignition_requested = False
            self._ignition_ready_s = float("inf")
            self.active_engines = 0
            return ActuatorCommand(
                throttle_cmd=0.0,
                active_engines=0,
                gimbal_rad=cmd.gimbal_rad,
                rcs_torque_nm=cmd.rcs_torque_nm,
                fin_deflection_rad=cmd.fin_deflection_rad,
                deploy_fins=cmd.deploy_fins,
                deploy_gear=cmd.deploy_gear,
                phase=cmd.phase,
            )

        # Quantize strictly to propulsion throttle floor
        eff_throttle = self.propulsion.quantize_throttle(cmd.throttle_cmd, cmd.active_engines)
        currently_burning = float(self.state[12]) > 1e-3

        if currently_burning:
            self._ignition_requested = True
            self._ignition_ready_s = min(self._ignition_ready_s, self.time_s)
            self.active_engines = cmd.active_engines
            return ActuatorCommand(
                throttle_cmd=eff_throttle,
                active_engines=cmd.active_engines,
                gimbal_rad=cmd.gimbal_rad,
                rcs_torque_nm=cmd.rcs_torque_nm,
                fin_deflection_rad=cmd.fin_deflection_rad,
                deploy_fins=cmd.deploy_fins,
                deploy_gear=cmd.deploy_gear,
                phase=cmd.phase,
            )

        # Starting from engine OFF (cold start or restart)
        if (
            not self.upper_attached
            and not self._ignition_requested
            and self.restarts_used >= self.propulsion.max_restarts
        ):
            return ActuatorCommand(
                throttle_cmd=0.0,
                active_engines=0,
                gimbal_rad=cmd.gimbal_rad,
                rcs_torque_nm=cmd.rcs_torque_nm,
                fin_deflection_rad=cmd.fin_deflection_rad,
                deploy_fins=cmd.deploy_fins,
                deploy_gear=cmd.deploy_gear,
                phase=cmd.phase,
            )

        if not self._ignition_requested:
            feed_st = FeedState(
                float(self.state[13]),
                float(self.state[14]),
                float(self.state[15]),
                float(self.state[16]),
            )
            f_rates = feed_rates(float(self.state[6]), 0.0, 0.0, feed_st, self.feed_settings)
            if self.upper_attached or ignition_ready(
                feed_st, f_rates.tank_pressure_pa, self.ignition_settings
            ):
                self._ignition_requested = True
                delay = 0.0 if self.upper_attached else self.ignition_settings.delay_s
                self._ignition_ready_s = self.time_s + delay
                if not self.upper_attached:
                    self.restarts_used += 1

        if self._ignition_requested and self.time_s >= self._ignition_ready_s - 1e-9:
            self.active_engines = cmd.active_engines
            return ActuatorCommand(
                throttle_cmd=eff_throttle,
                active_engines=cmd.active_engines,
                gimbal_rad=cmd.gimbal_rad,
                rcs_torque_nm=cmd.rcs_torque_nm,
                fin_deflection_rad=cmd.fin_deflection_rad,
                deploy_fins=cmd.deploy_fins,
                deploy_gear=cmd.deploy_gear,
                phase=cmd.phase,
            )

        self.active_engines = 0
        return ActuatorCommand(
            throttle_cmd=0.0,
            active_engines=0,
            gimbal_rad=cmd.gimbal_rad,
            rcs_torque_nm=cmd.rcs_torque_nm,
            fin_deflection_rad=cmd.fin_deflection_rad,
            deploy_fins=cmd.deploy_fins,
            deploy_gear=cmd.deploy_gear,
            phase=cmd.phase,
        )

    def derivative(
        self,
        st: np.ndarray,
        cmd: ActuatorCommand,
        time_s: float,
        available: tuple[bool, bool],
        diag: dict | None = None,
    ) -> np.ndarray:
        main_avail, rcs_avail = available
        thr_cmd = cmd.throttle_cmd if main_avail else 0.0
        rcs_cmd = cmd.rcs_torque_nm if rcs_avail else 0.0
        thr_act = max(0.0, min(1.0, float(st[12]))) if main_avail else 0.0

        _, p_pa, _ = self.atmo_lut.at(float(st[1]))
        _, demanded_flow = compute_engine_thrust(
            thr_act, p_pa, self.propulsion, thrust_scale=self.disturbances.thrust_scale
        )
        feed_st = FeedState(float(st[13]), float(st[14]), float(st[15]), float(st[16]))
        feed0 = feed_rates(float(st[6]), demanded_flow, 0.0, feed_st, self.feed_settings)
        delivered_ratio = feed0.delivered_flow_kg_s / demanded_flow if demanded_flow > 1e-9 else 1.0

        mass = self._total_mass(st)
        cg, inertia, fuel_h = self._mass_props(st)
        ox, oz = self._body_offset(st, mass)

        ft = compute_forces_and_torques(
            z_m=float(st[1]),
            vx_m_s=float(st[2]),
            vz_m_s=float(st[3]),
            theta_rad=float(st[4]),
            omega_rad_s=float(st[5]),
            cg_m=cg,
            throttle_actual=thr_act,
            gimbal_rad=float(st[11]),
            rcs_torque_nm=rcs_cmd,
            fin_deflection_rad=float(st[9]),
            fin_extension=float(st[10]),
            atmo_lut=self.atmo_lut,
            wind_lut=self.wind_lut,
            time_s=time_s,
            aero=self.aero,
            fins=self.fins,
            propulsion=self.propulsion,
            upper_attached=self.upper_attached,
            thrust_scale=self.disturbances.thrust_scale,
            drag_scale=self.disturbances.drag_coefficient_scale,
            delivered_ratio=delivered_ratio,
            body_offset_x=ox,
            body_offset_z=oz,
        )

        # Post-touchdown ground contact forces (contact_dynamics mode after booster separation)
        contact = ContactSample()
        if not self.upper_attached and float(st[1]) < 25.0:
            contact = evaluate_ground_contact(
                x_m=float(st[0]),
                z_m=float(st[1]),
                vx_m_s=float(st[2]),
                vz_m_s=float(st[3]),
                theta_rad=float(st[4]),
                omega_rad_s=float(st[5]),
                total_mass_kg=mass,
                cg_m=cg,
                gear_extension=float(st[8]),
                settings=self.contact_settings,
                offset_x=ox,
                offset_z=oz,
            )

        fx = ft.force_x_n + contact.force_x_n
        fz = ft.force_z_n + contact.force_z_n
        tau_ext = ft.torque_nm + contact.torque_nm

        g_loc = fast_gravity(float(st[1]))
        ax = fx / mass - 2.0 * self.coriolis_om * float(st[3])
        az = fz / mass - g_loc + 2.0 * self.coriolis_om * float(st[2])

        gear_rate = (
            1.0 / PHYSICS.gear.deployment_time_s
            if (not self.upper_attached and cmd.deploy_gear and float(st[8]) < 1.0)
            else 0.0
        )
        mass_rate = -ft.main_flow_kg_s - ft.rcs_flow_kg_s
        _, _, di_dm, di_ddep = fast_mass_gradients(mass, float(st[8]), cg, fuel_h)
        i_rate = di_dm * mass_rate + di_ddep * gear_rate

        lever = cg - PHYSICS.geometry.nozzle_station_m
        omega = float(st[5])
        ang_flux = (
            -ft.main_flow_kg_s * lever * lever * omega
            - ft.rcs_flow_kg_s
            * (PHYSICS.geometry.rcs_arm_m**2 + (PHYSICS.geometry.rcs_station_m - cg) ** 2)
            * omega
        )

        supported_on_pad = self.on_pad and self.upper_attached and fz < mass * g_loc
        c = cos(float(st[4]))
        s = sin(float(st[4]))
        if supported_on_pad:
            prop_lat = -s * g_loc
            prop_ax = c * g_loc
        else:
            prop_lat = c * (fx / mass) - s * (fz / mass)
            prop_ax = s * (fx / mass) + c * (fz / mass)

        feed = feed_rates(float(st[6]), demanded_flow, prop_ax, feed_st, self.feed_settings)
        tank_station = PHYSICS.geometry.tank_bottom_m + 0.5 * fuel_h
        slosh = slosh_response(
            mass,
            inertia,
            max(float(st[6]), 0.0),
            tank_station - cg,
            prop_ax,
            prop_lat,
            tau_ext + ang_flux - i_rate * omega,
            omega,
            SloshState(float(st[17]), float(st[18])),
            self.slosh_settings,
        )
        alpha_acc = slosh.angular_acceleration_rad_s2
        mode_acc = slosh.displacement_acceleration_m_s2

        if supported_on_pad:
            ax = 0.0
            az = 0.0
            alpha_acc = 0.0
            if slosh.modal_mass_kg > 1e-9:
                mode_acc += (
                    slosh.rotational_slosh_coupling_kg_m
                    / slosh.reduced_modal_mass_kg
                    * slosh.angular_acceleration_rad_s2
                )

        f_rate_lim = self.fins.deflection_rate_deg_s * (pi / 180.0)
        fin_rate = max(
            -f_rate_lim,
            min(
                f_rate_lim,
                (cmd.fin_deflection_rad - float(st[9])) / self.fins.actuator_time_constant_s,
            ),
        )
        if not self.upper_attached and cmd.deploy_fins and float(st[10]) < 1.0:
            fin_ext_rate = 1.0 / self.fins.deployment_time_s
        elif (self.upper_attached or not cmd.deploy_fins) and float(st[10]) > 0.0:
            fin_ext_rate = -1.0 / self.fins.retraction_time_s
        else:
            fin_ext_rate = 0.0

        g_rate_lim = self.propulsion.gimbal_rate_deg_s * (pi / 180.0)
        gimbal_rate = max(
            -g_rate_lim,
            min(g_rate_lim, (cmd.gimbal_rad - float(st[11])) / self.propulsion.gimbal_tau_s),
        )
        throttle_rate = (thr_cmd - float(st[12])) / self.engine_tau_s

        if diag is not None:
            diag["ft"] = ft
            diag["feed"] = feed
            diag["slosh"] = slosh
            diag["contact"] = contact
            diag["ax"] = ax
            diag["az"] = az
            diag["alpha_acc"] = alpha_acc
            diag["prop_ax"] = prop_ax
            diag["prop_lat"] = prop_lat

        rate = np.empty(len(STATE_NAMES), dtype=np.float64)
        rate[0] = float(st[2])
        rate[1] = float(st[3])
        rate[2] = ax
        rate[3] = az
        rate[4] = omega
        rate[5] = alpha_acc
        rate[6] = -ft.main_flow_kg_s
        rate[7] = -ft.rcs_flow_kg_s
        rate[8] = gear_rate
        rate[9] = fin_rate
        rate[10] = fin_ext_rate
        rate[11] = gimbal_rate
        rate[12] = throttle_rate
        rate[13:17] = feed.state_derivatives
        rate[17] = float(st[18])
        rate[18] = mode_acc
        return rate

    def telemetry(self) -> TelemetryPacket:
        """Generate one causal noisy/delayed TelemetryPacket for the GNC controller."""
        x, z, vx, vz, th, om = (
            float(self.state[0]),
            float(self.state[1]),
            float(self.state[2]),
            float(self.state[3]),
            float(self.state[4]),
            float(self.state[5]),
        )
        err = self.disturbances
        if err.sensor_position_error_m > 0.0:
            v = self.rng.normal(size=2)
            nrm = max(hypot(float(v[0]), float(v[1])), 1e-12)
            rad = sqrt(float(self.rng.random())) * err.sensor_position_error_m
            x += float(v[0]) / nrm * rad
            z += float(v[1]) / nrm * rad
        if err.sensor_velocity_error_m_s > 0.0:
            v = self.rng.normal(size=2)
            nrm = max(hypot(float(v[0]), float(v[1])), 1e-12)
            rad = sqrt(float(self.rng.random())) * err.sensor_velocity_error_m_s
            vx += float(v[0]) / nrm * rad
            vz += float(v[1]) / nrm * rad

        ss = self.sensor_settings
        if ss.angular_noise_enabled:
            th += (
                ss.attitude_bias_deg
                + float(self.sensor_rng.uniform(-ss.attitude_error_deg, ss.attitude_error_deg))
            ) * (pi / 180.0)
            om += (
                ss.gyro_bias_deg_s
                + float(self.sensor_rng.uniform(-ss.gyro_error_deg_s, ss.gyro_error_deg_s))
            ) * (pi / 180.0)

        drop = (
            ss.dropout_period_steps > 0
            and self.sensor_sample_count > 0
            and (self.sensor_sample_count % ss.dropout_period_steps) < ss.dropout_length_steps
        )
        self.sensor_sample_count += 1
        if drop and self._last_packet is not None:
            self.sensor_dropout_count += 1
            base = self._last_packet
            pkt = TelemetryPacket(
                time_s=self.time_s,
                measurement_time_s=base.measurement_time_s,
                x_m=base.x_m,
                z_m=base.z_m,
                vx_m_s=base.vx_m_s,
                vz_m_s=base.vz_m_s,
                theta_rad=base.theta_rad,
                omega_rad_s=base.omega_rad_s,
                main_propellant_kg=base.main_propellant_kg,
                rcs_propellant_kg=base.rcs_propellant_kg,
                gear_extension=base.gear_extension,
                fin_deflection_rad=base.fin_deflection_rad,
                fin_extension=base.fin_extension,
                gimbal_rad=base.gimbal_rad,
                engine_throttle=base.engine_throttle,
                upper_attached=self.upper_attached,
                on_pad=self.on_pad,
                touchdown_detected=self.touchdown_time_s is not None,
                active_engines=self.active_engines,
                restarts_used=self.restarts_used,
                packet_dropped=True,
            )
        else:
            pkt = TelemetryPacket(
                time_s=self.time_s,
                measurement_time_s=self.time_s,
                x_m=x,
                z_m=z,
                vx_m_s=vx,
                vz_m_s=vz,
                theta_rad=th,
                omega_rad_s=om,
                main_propellant_kg=float(self.state[6]),
                rcs_propellant_kg=float(self.state[7]),
                gear_extension=float(self.state[8]),
                fin_deflection_rad=float(self.state[9]),
                fin_extension=float(self.state[10]),
                gimbal_rad=float(self.state[11]),
                engine_throttle=float(self.state[12]),
                upper_attached=self.upper_attached,
                on_pad=self.on_pad,
                touchdown_detected=self.touchdown_time_s is not None,
                active_engines=self.active_engines,
                restarts_used=self.restarts_used,
                packet_dropped=False,
            )
            self._last_packet = pkt

        self.obs_queue.append(pkt)
        delayed = self.obs_queue[0]
        # Deliver with current clock time_s while preserving delayed measurement_time_s
        return TelemetryPacket(
            time_s=self.time_s,
            measurement_time_s=delayed.measurement_time_s,
            x_m=delayed.x_m,
            z_m=delayed.z_m,
            vx_m_s=delayed.vx_m_s,
            vz_m_s=delayed.vz_m_s,
            theta_rad=delayed.theta_rad,
            omega_rad_s=delayed.omega_rad_s,
            main_propellant_kg=delayed.main_propellant_kg,
            rcs_propellant_kg=delayed.rcs_propellant_kg,
            gear_extension=delayed.gear_extension,
            fin_deflection_rad=delayed.fin_deflection_rad,
            fin_extension=delayed.fin_extension,
            gimbal_rad=delayed.gimbal_rad,
            engine_throttle=delayed.engine_throttle,
            upper_attached=self.upper_attached,
            on_pad=self.on_pad,
            touchdown_detected=self.touchdown_time_s is not None,
            active_engines=self.active_engines,
            restarts_used=self.restarts_used,
            packet_dropped=delayed.packet_dropped,
        )

    def _check_separation(self) -> None:
        if not self.upper_attached:
            return
        s = self.ascent_settings
        st = self.state
        mass = self._total_mass(st)
        cg, inertia, fuel_h = self._mass_props(st)
        _, _, _, q, _, alpha_deg = compute_body_aero(
            float(st[1]),
            float(st[2]),
            float(st[3]),
            float(st[4]),
            float(st[5]),
            cg,
            self.atmo_lut,
            self.wind_lut,
            self.time_s,
            self.aero,
            upper_attached=True,
            drag_scale=self.disturbances.drag_coefficient_scale,
        )
        if q > self._peak_ascent_q:
            self._peak_ascent_q = q
        elif self._peak_ascent_q > 10_000.0 and q < 0.85 * self._peak_ascent_q:
            self.observed_max_q_passed = True

        checks = {
            "altitude": abs(float(st[1]) - s.separation_altitude_m) <= s.separation_tolerance_m,
            "dynamic_pressure": q <= s.separation_max_q_pa,
            "angular_rate": abs(float(st[5]) * (180.0 / pi)) <= s.separation_max_rate_deg_s,
            "alpha": abs(alpha_deg) <= s.separation_max_alpha_deg,
            "main_reserve": float(st[6]) >= s.separation_min_main_propellant_kg,
            "main_engine_off": float(st[12]) <= 0.001,
            "after_observed_max_q": self.observed_max_q_passed,
        }
        if not all(checks.values()):
            if float(st[1]) > s.separation_altitude_m + s.separation_tolerance_m:
                self.outcome = "separation_gate_missed"
            return

        # Execute momentum-conserving cold stage separation
        booster_mass = mass - self.design.upper_mass_kg
        b_cg, b_i, b_fh = fast_mass_properties(
            booster_mass,
            float(st[8]),
            fin_mass_kg=self.fins.mass_kg,
            airframe_mass_offset_kg=self.airframe_mass_offset_kg,
            upper_attached=False,
            inertia_scale=self.disturbances.inertia_scale,
        )
        u_mass = self.design.upper_mass_kg
        u_i = u_mass * (PHYSICS.upper_length_m**2 / 12.0 + PHYSICS.geometry.body_radius_m**2 / 4.0)
        ox, oz = self._body_offset(st, mass)
        tank_st = PHYSICS.geometry.tank_bottom_m + 0.5 * b_fh
        imp = separation_event(
            st[:2],
            st[2:4],
            float(st[4]),
            float(st[5]),
            booster_mass,
            u_mass,
            b_cg,
            PHYSICS.upper_station_m,
            b_i,
            u_i,
            self.separation_settings,
            main_propellant_kg=float(st[6]),
            slosh_state=SloshState(float(st[17]), float(st[18])),
            slosh_station_m=tank_st,
            slosh_settings=self.slosh_settings,
        )
        self.upper_attached = False
        self.state[0:2] = np.asarray(imp["booster_position_m"], dtype=np.float64)
        self.state[2:4] = np.asarray(imp["booster_velocity_m_s"], dtype=np.float64)
        self.state[5] = float(imp["booster_angular_rate_rad_s"])
        self.state[18] = float(imp["booster_slosh_velocity_m_s"])
        self.upper_state = {
            "time_s": self.time_s,
            "pos": np.asarray(imp["upper_position_m"], dtype=np.float64),
            "vel": np.asarray(imp["upper_velocity_m_s"], dtype=np.float64),
            "theta": float(st[4]),
            "omega": float(imp["upper_angular_rate_rad_s"]),
        }
        self.separation_record = {
            "time_s": self.time_s,
            "altitude_m": float(self.state[1]),
            "main_propellant_kg": float(self.state[6]),
            "rcs_propellant_kg": float(self.state[7]),
            "checks_before_separation": checks,
            "linear_momentum_residual": imp["linear_momentum_residual_kg_m_s"],
            "angular_momentum_residual": float(imp["angular_momentum_residual_kg_m2_s"]),
            "pulse_accel_deg_s2": float(
                imp["booster_pulse_average_angular_acceleration_rad_s2"] * 180.0 / pi
            ),
        }

    def _record_first_contact(self, st: np.ndarray, cmd: ActuatorCommand) -> None:
        mass = self._total_mass(st)
        cg, _, fuel_h = self._mass_props(st)
        ox, oz = self._body_offset(st, mass)
        (lpx, lpz), (rpx, rpz), (lvx, lvz), (rvx, rvz) = foot_kinematics_scalar(
            float(st[0]),
            float(st[1]),
            float(st[2]),
            float(st[3]),
            float(st[4]),
            float(st[5]),
            cg,
            float(st[8]),
            offset_x=ox,
            offset_z=oz,
        )
        idx = 0 if lpz <= rpz else 1
        foot_v = (lvx, lvz) if idx == 0 else (rvx, rvz)
        pos_err = abs(float(st[0]))
        v_vert = max(0.0, -foot_v[1])
        v_horiz = abs(foot_v[0])
        tilt_deg = abs(float(st[4]) * (180.0 / pi))
        rate_deg_s = abs(float(st[5]) * (180.0 / pi))
        pad_hw = PHYSICS.gear.pad_half_width_m
        feet_on_pad = max(abs(lpx), abs(rpx)) <= pad_hw

        acc = self.acceptance
        criteria = {
            "position": pos_err <= acc.max_position_error_m,
            "vertical_foot_speed": v_vert <= acc.max_vertical_foot_speed_m_s,
            "horizontal_foot_speed": v_horiz <= acc.max_horizontal_contact_speed_m_s,
            "tilt": tilt_deg <= acc.max_contact_tilt_deg,
            "angular_rate": rate_deg_s <= acc.max_contact_rate_deg_s,
            "main_propellant": float(st[6]) > 0.0,
            "rcs_propellant": float(st[7]) >= 0.0,
            "gear_deployed": float(st[8]) >= 0.999,
            "feet_on_pad": feet_on_pad,
        }
        self.touchdown_time_s = self.time_s
        self.first_contact_record = {
            "time_s": self.time_s,
            "position_error_m": pos_err,
            "vertical_foot_speed_m_s": v_vert,
            "horizontal_foot_speed_m_s": v_horiz,
            "tilt_deg": tilt_deg,
            "contact_rate_deg_s": rate_deg_s,
            "main_remaining_kg": float(st[6]),
            "rcs_remaining_kg": float(st[7]),
            "foot_positions_m": [[lpx, lpz], [rpx, rpz]],
            "foot_velocities_m_s": [[lvx, lvz], [rvx, rvz]],
            "criteria": criteria,
        }

    def _monitor(self, st: np.ndarray, cmd: ActuatorCommand) -> None:
        diag: dict = {}
        self.derivative(st, cmd, self.time_s, (st[6] > 0.0, st[7] > 0.0), diag=diag)
        ft = diag["ft"]
        feed = diag["feed"]
        slosh = diag["slosh"]
        contact: ContactSample = diag["contact"]
        self._last_contact_sample = contact

        q = ft.dynamic_pressure_pa
        self.max_q_pa = max(self.max_q_pa, q)
        if self.upper_attached:
            self.max_q_alpha_kpa_deg = max(
                self.max_q_alpha_kpa_deg, (q / 1000.0) * abs(ft.alpha_deg)
            )
            if q >= 10_000.0:
                self.max_ascent_high_q_alpha_deg = max(
                    self.max_ascent_high_q_alpha_deg, abs(ft.alpha_deg)
                )
            self.max_ascent_axial_g = max(self.max_ascent_axial_g, abs(diag["prop_ax"]) / G0)
        elif self.touchdown_time_s is None:
            self.max_recovery_g = max(
                self.max_recovery_g, hypot(diag["prop_ax"], diag["prop_lat"]) / G0
            )

        if self.touchdown_time_s is None:
            self.max_rate_deg_s = max(self.max_rate_deg_s, abs(float(st[5]) * (180.0 / pi)))
            self.max_accel_deg_s2 = max(
                self.max_accel_deg_s2, abs(float(diag["alpha_acc"]) * (180.0 / pi))
            )
            self.max_gimbal_deg = max(self.max_gimbal_deg, abs(float(st[11]) * (180.0 / pi)))
            self.max_fin_hinge_nm = max(self.max_fin_hinge_nm, ft.fin_hinge_nm)
            self.max_fin_deflection_deg = max(
                self.max_fin_deflection_deg, abs(float(st[9]) * (180.0 / pi))
            )
            if float(st[10]) > 0.05 and (
                ft.mach > self.fins.model_max_mach
                or abs(ft.alpha_deg) > self.fins.control_max_alpha_deg + 1e-6
            ):
                self.fin_model_ok = False
        else:
            self.max_leg_stroke_m = max(
                self.max_leg_stroke_m, contact.left_stroke_m, contact.right_stroke_m
            )
            self.max_leg_force_n = max(
                self.max_leg_force_n, contact.left_normal_n, contact.right_normal_n
            )
            self.max_contact_load_g = max(self.max_contact_load_g, contact.contact_load_g)
            self.post_contact_max_tilt_deg = max(
                self.post_contact_max_tilt_deg, abs(float(st[4]) * (180.0 / pi))
            )
            if contact.body_ground_collision:
                self.body_ground_collision = True

        if not feed.inventory_valid:
            self.feed_inventory_ok = False
        if feed.demand_shortfall_kg_s > 1e-3:
            self.feed_supply_ok = False
        if not slosh.valid:
            self.slosh_domain_ok = False

        mb = (
            self.design.main_propellant_kg
            + self.design.rcs_propellant_kg
            - (float(st[6]) + float(st[7]) + self.main_consumed_kg + self.rcs_consumed_kg)
        )
        self.max_mass_balance_err = max(self.max_mass_balance_err, abs(mb))
        hb = float(st[15]) + float(st[16]) - self.feed_settings.helium_total_mass_kg
        self.max_helium_balance_err = max(self.max_helium_balance_err, abs(hb))

        # Update phase statistics
        p = self.phase
        if p not in self.phase_stats:
            self.phase_stats[p] = {
                "start_s": self.time_s,
                "end_s": self.time_s,
                "start_z_m": float(st[1]),
                "end_z_m": float(st[1]),
                "start_x_m": float(st[0]),
                "end_x_m": float(st[0]),
                "start_main_kg": float(st[6]),
                "end_main_kg": float(st[6]),
                "start_rcs_kg": float(st[7]),
                "end_rcs_kg": float(st[7]),
                "max_throttle": float(st[12]),
                "min_active_throttle": 1.0,
                "active_steps": 0,
                "total_steps": 0,
                "max_fin_deg": 0.0,
                "max_alpha_deg": 0.0,
                "max_q_pa": 0.0,
            }
        ps = self.phase_stats[p]
        ps["end_s"] = self.time_s
        ps["end_z_m"] = float(st[1])
        ps["end_x_m"] = float(st[0])
        ps["end_main_kg"] = float(st[6])
        ps["end_rcs_kg"] = float(st[7])
        ps["total_steps"] += 1
        thr = float(st[12])
        if thr > ps["max_throttle"]:
            ps["max_throttle"] = thr
        if thr > 0.01:
            ps["active_steps"] += 1
            if thr < ps["min_active_throttle"]:
                ps["min_active_throttle"] = thr
        ps["max_fin_deg"] = max(ps["max_fin_deg"], abs(float(st[9]) * (180.0 / pi)))
        ps["max_alpha_deg"] = max(ps["max_alpha_deg"], abs(ft.alpha_deg))
        ps["max_q_pa"] = max(ps["max_q_pa"], q)

        # Record 50 Hz UE5/MuJoCo-compatible ReplayFrame
        if self.config.record_replay and self.time_s >= self._next_replay_s - 1e-9:
            self._next_replay_s = self.time_s + self._replay_dt_s
            u_pos = None
            u_quat = None
            if self.upper_state is not None:
                dt_u = self.time_s - self.upper_state["time_s"]
                # Analytic visual coast/burn progression for separated upper stage
                up0 = self.upper_state["pos"]
                uv0 = self.upper_state["vel"]
                uth0 = self.upper_state["theta"]
                uom0 = self.upper_state["omega"]
                u_pos_x = float(up0[0] + uv0[0] * dt_u)
                u_pos_z = float(
                    up0[1] + uv0[1] * dt_u - 0.5 * fast_gravity(float(up0[1])) * dt_u * dt_u
                )
                if dt_u > 3.0:
                    tb = dt_u - 3.0
                    a_up = 18.0
                    u_pos_x += 0.5 * a_up * sin(uth0) * tb * tb
                    u_pos_z += 0.5 * a_up * cos(uth0) * tb * tb
                u_theta = uth0 + uom0 * 4.0 * (1.0 - exp(-dt_u / 4.0))
                if u_pos_z <= PROFILE_TOP_M:
                    u_pos = (u_pos_x, 0.0, u_pos_z)
                    u_quat = pitch_to_quat_wxyz(u_theta)

            fin_d = float(st[9])
            ls = float(contact.left_stroke_m)
            rs = float(contact.right_stroke_m)
            self.replay_frames.append(
                ReplayFrame(
                    time_s=self.time_s,
                    phase=self.phase,
                    pos_enu_m=(float(st[0]), 0.0, float(st[1])),
                    quat_wxyz=pitch_to_quat_wxyz(float(st[4])),
                    vel_enu_m_s=(float(st[2]), 0.0, float(st[3])),
                    omega_body_rad_s=(0.0, float(st[5]), 0.0),
                    throttle=thr,
                    active_engines=self.active_engines,
                    gimbal_rad=(float(st[11]), 0.0),
                    rcs_torque_nm=(0.0, cmd.rcs_torque_nm, 0.0),
                    fin_extension=float(st[10]),
                    fin_deflection_rad=(fin_d, fin_d, 0.0, 0.0),
                    gear_extension=float(st[8]),
                    leg_stroke_m=(ls, rs, ls, rs),
                    dynamic_pressure_pa=q,
                    mach=ft.mach,
                    alpha_deg=ft.alpha_deg,
                    main_propellant_kg=float(st[6]),
                    rcs_propellant_kg=float(st[7]),
                    upper_attached=self.upper_attached,
                    upper_pos_enu_m=u_pos,
                    upper_quat_wxyz=u_quat,
                )
            )

    def step(self, command: ActuatorCommand, duration_s: float) -> None:
        """Advance physics by `duration_s` (one control interval) under delayed `command`."""
        if self.done:
            return
        self.phase = command.phase
        self.action_queue.append(command)
        active_cmd = self._apply_ignition_interlock(self.action_queue[0])

        t_end = self.time_s + duration_s
        while self.time_s < t_end - 1e-12 and not self.done:
            in_contact_zone = (not self.upper_attached) and (
                self.touchdown_time_s is not None or self._foot_height(self.state) <= 2.5
            )
            dt_base = self.config.contact_dt_s if in_contact_zone else self.config.integration_dt_s
            dt = min(dt_base, t_end - self.time_s)
            if dt < 1e-10:
                self.time_s = t_end
                break

            before = self.state.copy()
            avail = (bool(before[6] > 0.0), bool(before[7] > 0.0))

            def rhs(t: float, y: np.ndarray, _avail=avail, _cmd=active_cmd) -> np.ndarray:
                return self.derivative(y, _cmd, t, _avail)

            nxt = rk4_step(rhs, self.time_s, before, dt)
            if not np.isfinite(nxt).all():
                self.outcome = "numerical_failure"
                return

            # Exact propellant depletion event
            if nxt[6] <= 0.0 or nxt[7] < 0.0:
                ev_dts = [
                    float(
                        brentq(
                            lambda s_dt, idx=idx, _b=before, _rhs=rhs: (
                                float(rk4_step(_rhs, self.time_s, _b, s_dt)[idx])
                                if s_dt > 0.0
                                else float(_b[idx])
                            ),
                            0.0,
                            dt,
                            xtol=1e-9,
                        )
                    )
                    for idx in (6, 7)
                    if before[idx] > 0.0 and nxt[idx] <= 0.0
                ]
                if ev_dts:
                    dt = min(ev_dts)
                    nxt = rk4_step(rhs, self.time_s, before, dt)
                nxt[6] = max(0.0, float(nxt[6]))
                nxt[7] = max(0.0, float(nxt[7]))
                if nxt[6] < 1e-7:
                    nxt[6] = 0.0
                    self.outcome = "main_propellant_depleted"
                if nxt[7] < 1e-7:
                    nxt[7] = 0.0

            # Exact first-touchdown event detection
            if (
                not self.upper_attached
                and self.touchdown_time_s is None
                and self._foot_height(before) > 0.0 >= self._foot_height(nxt)
            ):
                frac_dt = float(
                    brentq(
                        lambda s_dt, _b=before, _rhs=rhs: (
                            self._foot_height(rk4_step(_rhs, self.time_s, _b, s_dt))
                            if s_dt > 0.0
                            else self._foot_height(_b)
                        ),
                        0.0,
                        dt,
                        xtol=1e-9,
                    )
                )
                nxt = rk4_step(rhs, self.time_s, before, frac_dt)
                dt = frac_dt
                self.time_s += dt
                self.main_consumed_kg += float(before[6] - nxt[6])
                self.rcs_consumed_kg += float(before[7] - nxt[7])
                self.state = nxt
                self._record_first_contact(nxt, active_cmd)
                self._monitor(nxt, active_cmd)
                if self.config.landing_mode == "kinematic_touchdown":
                    self.outcome = (
                        "landed"
                        if all(self.first_contact_record["criteria"].values())
                        else "touchdown_limits_exceeded"
                    )
                    return
                # In contact_dynamics mode, immediately cut engine command for rest of control tick!
                active_cmd = ActuatorCommand(
                    throttle_cmd=0.0,
                    active_engines=0,
                    gimbal_rad=0.0,
                    rcs_torque_nm=0.0,
                    fin_deflection_rad=0.0,
                    deploy_fins=True,
                    deploy_gear=True,
                    phase="touchdown_settle",
                )
                self.phase = "touchdown_settle"
                continue

            # Clamp actuator states cleanly
            nxt[8] = max(0.0, min(1.0, float(nxt[8])))
            f_lim = self.fins.deflection_limit_deg * (pi / 180.0)
            nxt[9] = max(-f_lim, min(f_lim, float(nxt[9])))
            nxt[10] = max(0.0, min(1.0, float(nxt[10])))
            g_lim = self.propulsion.gimbal_limit_deg * (pi / 180.0)
            nxt[11] = max(-g_lim, min(g_lim, float(nxt[11])))
            nxt[12] = max(0.0, min(1.0, float(nxt[12])))

            self.time_s += dt
            self.main_consumed_kg += float(before[6] - nxt[6])
            self.rcs_consumed_kg += float(before[7] - nxt[7])
            self.state = nxt

            if self.upper_attached and self.on_pad:
                _, p_pa, _ = self.atmo_lut.at(float(nxt[1]))
                t_now, _ = compute_engine_thrust(
                    float(nxt[12]),
                    p_pa,
                    self.propulsion,
                    thrust_scale=self.disturbances.thrust_scale,
                )
                if t_now > self._total_mass(nxt) * fast_gravity(float(nxt[1])):
                    self.on_pad = False

            if in_contact_zone or self.time_s >= t_end - 1e-12:
                self._monitor(nxt, active_cmd)
            self._check_separation()

            # Check post-touchdown settle completion or tip-over
            if self.touchdown_time_s is not None:
                if self.body_ground_collision:
                    self.standing_stable = False
                    self.outcome = "body_ground_collision"
                    return
                if abs(float(nxt[4]) * (180.0 / pi)) > 35.0:
                    self.standing_stable = False
                    self.outcome = "tip_over"
                    return
                if self.time_s - self.touchdown_time_s >= self.contact_settings.settle_duration_s:
                    cs = self.contact_settings
                    spd = hypot(float(nxt[2]), float(nxt[3]))
                    tilt = abs(float(nxt[4]) * (180.0 / pi))
                    rate = abs(float(nxt[5]) * (180.0 / pi))
                    self.standing_stable = (
                        not self.body_ground_collision
                        and float(nxt[12]) <= 1e-3
                        and tilt <= cs.max_standing_tilt_deg
                        and spd <= cs.max_standing_speed_m_s
                        and rate <= cs.max_standing_rate_deg_s
                        and self._last_contact_sample.left_in_contact
                        and self._last_contact_sample.right_in_contact
                    )
                    self.outcome = "landed" if self.standing_stable else "standing_unstable"
                    return

            if self.time_s >= self.config.deadline_s:
                self.outcome = "timeout"
                return

    def summary(self) -> dict:
        """Produce full DM100-compatible mission, safety, touchdown & post-contact report."""
        s = SAFETY
        cs = self.contact_settings
        eps = 1e-7
        sep_ok = self.separation_record is not None and all(
            self.separation_record["checks_before_separation"].values()
        )
        touch_ok = self.first_contact_record is not None and all(
            self.first_contact_record["criteria"].values()
        )
        contact_mode = self.config.landing_mode == "contact_dynamics"
        criteria = {
            "dynamic_pressure": self.max_q_pa <= s.max_q_pa + eps,
            "ascent_q_alpha": self.max_q_alpha_kpa_deg <= s.max_q_alpha_kpa_deg + eps,
            "ascent_high_q_alpha": self.max_ascent_high_q_alpha_deg <= s.max_high_q_alpha_deg + eps,
            "ascent_axial_g": self.max_ascent_axial_g <= s.max_ascent_axial_g + eps,
            "flight_load_g": self.max_recovery_g <= s.max_recovery_flight_g + eps,
            "angular_rate": self.max_rate_deg_s <= s.max_angular_rate_deg_s + eps,
            "angular_acceleration": self.max_accel_deg_s2
            <= s.max_angular_acceleration_deg_s2 + eps,
            "gimbal": self.max_gimbal_deg <= self.propulsion.gimbal_limit_deg + eps,
            "fin_hinge": self.max_fin_hinge_nm <= self.fins.per_fin_hinge_limit_nm + eps,
            "fin_model_envelope": self.fin_model_ok,
            "mass_balance": self.max_mass_balance_err <= 1e-4,
            "helium_mass_balance": self.max_helium_balance_err <= 1e-4,
            "feed_inventory": self.feed_inventory_ok,
            "feed_supply": self.feed_supply_ok,
            "slosh_model_domain": self.slosh_domain_ok,
            "separation": sep_ok,
            "touchdown": touch_ok,
            "leg_stroke": (self.max_leg_stroke_m <= cs.max_leg_stroke_m + eps)
            if contact_mode
            else None,
            "leg_force": (self.max_leg_force_n <= cs.max_leg_force_n + eps)
            if contact_mode
            else None,
            "contact_load_g": (self.max_contact_load_g <= cs.max_contact_load_g + eps)
            if contact_mode
            else None,
            "engine_off_standing": bool(self.standing_stable) if contact_mode else None,
        }
        failed = [k for k, v in criteria.items() if v is False]
        passed = (self.outcome == "landed") and (len(failed) == 0)

        fc = self.first_contact_record or {}
        # Clean phase stats
        phase_summary = {}
        for name, ps in self.phase_stats.items():
            phase_summary[name] = {
                "start_s": ps["start_s"],
                "end_s": ps["end_s"],
                "duration_s": ps["end_s"] - ps["start_s"],
                "start_z_m": ps["start_z_m"],
                "end_z_m": ps["end_z_m"],
                "start_x_m": ps["start_x_m"],
                "end_x_m": ps["end_x_m"],
                "main_used_kg": ps["start_main_kg"] - ps["end_main_kg"],
                "rcs_used_kg": ps["start_rcs_kg"] - ps["end_rcs_kg"],
                "active_fraction": ps["active_steps"] / max(ps["total_steps"], 1),
                "min_active_throttle": ps["min_active_throttle"] if ps["active_steps"] > 0 else 0.0,
                "max_throttle": ps["max_throttle"],
                "max_fin_deg": ps["max_fin_deg"],
                "max_alpha_deg": ps["max_alpha_deg"],
                "max_q_pa": ps["max_q_pa"],
            }

        return {
            "version": "dm100_rtls",
            "landing_mode": self.config.landing_mode,
            "outcome": self.outcome,
            "requirements_passed": passed,
            "criteria": criteria,
            "failed_criteria": failed,
            "flight_time_s": self.time_s,
            "touchdown_time_s": self.touchdown_time_s,
            "restarts_used": self.restarts_used,
            "main_used_kg": self.main_consumed_kg,
            "main_remaining_kg": float(self.state[6]),
            "rcs_used_kg": self.rcs_consumed_kg,
            "rcs_remaining_kg": float(self.state[7]),
            "position_error_m": fc.get("position_error_m"),
            "vertical_foot_speed_m_s": fc.get("vertical_foot_speed_m_s"),
            "horizontal_foot_speed_m_s": fc.get("horizontal_foot_speed_m_s"),
            "tilt_deg": fc.get("tilt_deg"),
            "contact_rate_deg_s": fc.get("contact_rate_deg_s"),
            "peak_dynamic_pressure_pa": self.max_q_pa,
            "peak_q_alpha_kpa_deg": self.max_q_alpha_kpa_deg,
            "peak_ascent_axial_g": self.max_ascent_axial_g,
            "peak_recovery_g": self.max_recovery_g,
            "peak_angular_rate_deg_s": self.max_rate_deg_s,
            "peak_angular_acceleration_deg_s2": self.max_accel_deg_s2,
            "peak_gimbal_deg": self.max_gimbal_deg,
            "peak_fin_hinge_nm": self.max_fin_hinge_nm,
            "peak_fin_deflection_deg": self.max_fin_deflection_deg,
            "post_contact": {
                "max_leg_stroke_m": self.max_leg_stroke_m,
                "max_leg_force_n": self.max_leg_force_n,
                "max_contact_load_g": self.max_contact_load_g,
                "max_tilt_deg": self.post_contact_max_tilt_deg,
                "standing_stable": self.standing_stable,
                "final_speed_m_s": hypot(float(self.state[2]), float(self.state[3])),
                "final_tilt_deg": abs(float(self.state[4]) * (180.0 / pi)),
            },
            "separation": self.separation_record,
            "first_contact": self.first_contact_record,
            "phases": phase_summary,
            "replay_frame_count": len(self.replay_frames),
        }
