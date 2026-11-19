"""Matrix/priority-based control allocation (Fins -> TVC Gimbal -> RCS).

Formulated around the actuator effectiveness vector B = [b_fin, b_tvc, b_rcs]
solving B u = tau_des subject to actuator saturation, fin hinge-moment limits,
and angular-acceleration safety guards.

In Stage 2 (3D MuJoCo), the exact same allocation interface scales from
B in R^{1x3} to B in R^{3x9} (4 independent grid fins + 2-axis TVC + 3-axis RCS).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import asin, pi, sin

from powered_landing_guidance.rtls.forces import RtlsFinSettings, RtlsPropulsionSettings


@dataclass(frozen=True, slots=True)
class AllocationInput:
    requested_torque_nm: float
    background_torque_nm: float
    acceleration_inertia_kg_m2: float
    lever_arm_m: float
    sizing_thrust_n: float
    fin_effectiveness_nm_per_rad: float
    fin_zero_deflection_torque_nm: float
    fin_alpha_retro_rad: float
    dynamic_pressure_pa: float
    fin_extension: float
    rcs_available: bool
    allow_fin_allocation: bool
    thrust_direction_rad: float | None = None
    body_theta_rad: float = 0.0
    current_gimbal_rad: float = 0.0
    current_thrust_n: float = 0.0
    control_dt_s: float = 0.02
    angular_accel_guard_deg_s2: float = 3.55
    rcs_deadband_nm: float = 0.0


@dataclass(frozen=True, slots=True)
class AllocationOutput:
    gimbal_rad: float
    rcs_torque_nm: float
    fin_deflection_rad: float
    effectiveness_vector: tuple[float, float, float]
    allocated_torque_nm: float
    torque_margin_nm: float


class PriorityControlAllocator:
    """Hierarchical allocator minimizing propellant usage: Fins -> TVC -> RCS."""

    def __init__(
        self,
        propulsion: RtlsPropulsionSettings,
        fins: RtlsFinSettings,
    ) -> None:
        self.propulsion = propulsion
        self.fins = fins
        self._gimbal_limit = propulsion.gimbal_limit_deg * (pi / 180.0)
        self._fin_limit = fins.deflection_limit_deg * (pi / 180.0)

    def _fin_deflection_bounds(
        self,
        q_pa: float,
        fin_extension: float,
        alpha_retro_rad: float,
    ) -> tuple[float, float]:
        """Compute deflection bounds respecting both mechanical ±20° and hinge limit."""
        if fin_extension < 0.05 or q_pa < self.fins.control_min_q_pa:
            return (0.0, 0.0)
        max_lift_for_hinge = (
            2.0 * self.fins.per_fin_hinge_limit_nm / max(self.fins.hinge_arm_m, 1e-6)
        )
        denom = max(q_pa * self.fins.planar_effective_area_m2 * fin_extension, 1e-6)
        cl_limit = min(self.fins.max_lift_coefficient, 0.92 * max_lift_for_hinge / denom)
        eff_limit = min(
            self.fins.control_max_alpha_deg * (pi / 180.0),
            cl_limit / max(self.fins.lift_slope_rad_inv, 1e-6),
        )
        # alpha_eff = delta - alpha_retro => delta in [alpha_retro - eff_limit, +eff_limit]
        low = max(-self._fin_limit, alpha_retro_rad - eff_limit)
        high = min(self._fin_limit, alpha_retro_rad + eff_limit)
        if low > high:
            mid = max(-self._fin_limit, min(self._fin_limit, alpha_retro_rad))
            return (mid, mid)
        return (low, high)

    def allocate(self, inp: AllocationInput) -> AllocationOutput:
        b_fin_phys = inp.fin_effectiveness_nm_per_rad
        b_fin = b_fin_phys if inp.allow_fin_allocation else 0.0
        b_tvc = -inp.lever_arm_m * inp.sizing_thrust_n if inp.sizing_thrust_n > 1.0 else 0.0
        b_rcs = 1.0 if inp.rcs_available else 0.0

        # 1. Grid fins first (zero propellant cost!)
        fin_cmd = 0.0
        fin_torque_est = 0.0
        if inp.allow_fin_allocation and abs(b_fin_phys) > 1.0:
            low_f, high_f = self._fin_deflection_bounds(
                inp.dynamic_pressure_pa, inp.fin_extension, inp.fin_alpha_retro_rad
            )
            needed_from_delta = inp.requested_torque_nm - inp.fin_zero_deflection_torque_nm
            raw_fin = needed_from_delta / b_fin_phys
            fin_cmd = max(low_f, min(high_f, raw_fin))
            fin_torque_est = inp.fin_zero_deflection_torque_nm + b_fin_phys * fin_cmd
        elif inp.fin_extension >= 0.05 and inp.dynamic_pressure_pa > self.fins.control_min_q_pa:
            # Feather fins into the airflow and evaluate ACTUAL physical fin torque using b_fin_phys
            low_f, high_f = self._fin_deflection_bounds(
                inp.dynamic_pressure_pa, inp.fin_extension, inp.fin_alpha_retro_rad
            )
            fin_cmd = max(low_f, min(high_f, inp.fin_alpha_retro_rad))
            fin_torque_est = inp.fin_zero_deflection_torque_nm + b_fin_phys * fin_cmd

        residual_after_fin = inp.requested_torque_nm - fin_torque_est

        # 2. TVC gimbal next (zero extra RCS propellant when engine is running!)
        gimbal_cmd = 0.0
        tvc_torque_est = 0.0
        if inp.sizing_thrust_n > 1.0 and inp.lever_arm_m > 0.1:
            max_tvc = inp.lever_arm_m * inp.sizing_thrust_n
            arg = max(-1.0, min(1.0, -residual_after_fin / max_tvc))
            gimbal_cmd = max(-self._gimbal_limit, min(self._gimbal_limit, asin(arg)))
            if inp.thrust_direction_rad is not None:
                # In vertical-body mode, allow gimbal to steer thrust direction within RCS margin
                rcs_lim = self.propulsion.rcs_torque_limit_nm
                lo_arg = max(-1.0, min(1.0, (-rcs_lim - residual_after_fin) / max_tvc))
                hi_arg = max(-1.0, min(1.0, (rcs_lim - residual_after_fin) / max_tvc))
                lo_g = max(-self._gimbal_limit, asin(lo_arg))
                hi_g = min(self._gimbal_limit, asin(hi_arg))
                if lo_g <= hi_g:
                    desired_g = inp.thrust_direction_rad - inp.body_theta_rad
                    # Limit vertical-body gimbal step relative to current gimbal so
                    # TVC lag vs instantaneous RCS counter-torque never spikes angular
                    # acceleration under 1-step action delay.
                    max_g_step = 0.85 * (pi / 180.0)
                    slew_lo = max(lo_g, inp.current_gimbal_rad - max_g_step)
                    slew_hi = min(hi_g, inp.current_gimbal_rad + max_g_step)
                    if slew_lo <= slew_hi:
                        gimbal_cmd = max(slew_lo, min(slew_hi, desired_g))
                    else:
                        gimbal_cmd = max(lo_g, min(hi_g, desired_g))
            tvc_torque_est = -max_tvc * sin(gimbal_cmd)

        # 3. RCS for remaining torque only, guarded over the ENTIRE actuator slew interval
        #    [current_gimbal -> next_gimbal] and [current_thrust -> sizing_thrust]!
        rcs_cmd = 0.0
        if inp.rcs_available:
            g_rate = self.propulsion.gimbal_rate_deg_s * (pi / 180.0)
            g_tau = self.propulsion.gimbal_tau_s
            dg = gimbal_cmd - inp.current_gimbal_rad
            # Predicted gimbal angle at start, mid, and end of control interval
            dt = inp.control_dt_s
            g_mid = inp.current_gimbal_rad + max(
                -g_rate * 0.5 * dt,
                min(g_rate * 0.5 * dt, dg * (1.0 - 2.718281828 ** (-0.5 * dt / g_tau))),
            )
            g_end = inp.current_gimbal_rad + max(
                -g_rate * dt, min(g_rate * dt, dg * (1.0 - 2.718281828 ** (-dt / g_tau)))
            )
            t_cur = inp.current_thrust_n
            t_fut = inp.sizing_thrust_n
            t_mid = 0.5 * (t_cur + t_fut)
            supplied_samples = (
                fin_torque_est - inp.lever_arm_m * t_cur * sin(inp.current_gimbal_rad),
                fin_torque_est - inp.lever_arm_m * t_mid * sin(g_mid),
                fin_torque_est - inp.lever_arm_m * t_fut * sin(g_end),
            )
            sup_min = min(supplied_samples)
            sup_max = max(supplied_samples)
            sup_mid = 0.5 * (sup_min + sup_max)

            if inp.thrust_direction_rad is not None:
                # In vertical-body mode, gimbal steers thrust laterally while RCS counter-torques
                # instantaneously; match RCS to actual interval TVC+fin torque `sup_mid` so gimbal
                # slew lag never injects an unbalanced angular-acceleration spike.
                residual_after_tvc = inp.requested_torque_nm - sup_mid
            else:
                residual_after_tvc = inp.requested_torque_nm - (fin_torque_est + tvc_torque_est)
            if abs(residual_after_tvc) <= inp.rcs_deadband_nm:
                residual_after_tvc = 0.0
            rcs_lim = self.propulsion.rcs_torque_limit_nm
            guard_torque = inp.acceleration_inertia_kg_m2 * (
                inp.angular_accel_guard_deg_s2 * (pi / 180.0)
            )
            lower_rcs = max(-rcs_lim, -guard_torque - inp.background_torque_nm - sup_min)
            upper_rcs = min(rcs_lim, guard_torque - inp.background_torque_nm - sup_max)
            if lower_rcs > upper_rcs:
                # Pick midpoint that minimizes worst-case angular acceleration over the interval
                mid_safe = max(-rcs_lim, min(rcs_lim, -inp.background_torque_nm - sup_mid))
                rcs_cmd = mid_safe
            else:
                rcs_cmd = max(lower_rcs, min(upper_rcs, residual_after_tvc))

        allocated = fin_torque_est + tvc_torque_est + rcs_cmd
        capacity = (
            abs(b_fin) * self._fin_limit
            + abs(b_tvc) * sin(self._gimbal_limit)
            + (self.propulsion.rcs_torque_limit_nm if inp.rcs_available else 0.0)
        )
        return AllocationOutput(
            gimbal_rad=gimbal_cmd,
            rcs_torque_nm=rcs_cmd,
            fin_deflection_rad=fin_cmd,
            effectiveness_vector=(b_fin, b_tvc, b_rcs),
            allocated_torque_nm=allocated,
            torque_margin_nm=capacity - abs(inp.requested_torque_nm),
        )
