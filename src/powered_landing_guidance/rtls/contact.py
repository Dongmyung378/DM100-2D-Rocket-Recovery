"""Post-touchdown 2D landing-leg contact dynamics and standing stability.

Evaluates the four post-contact structural and upright stability criteria:
  - leg_stroke: telescoping strut compression within structural stroke limit
  - leg_force: per-leg ground reaction force within structural limit
  - contact_load_g: peak body deceleration load during touchdown impact
  - engine_off_standing: upright, settled booster on the pad after engine cutoff

This planar spring-damper + Coulomb friction model bridges directly to MuJoCo's
4-leg 3D contact constraints in Stage 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, hypot, sin, tanh

from powered_landing_guidance.apr_recovery import PHYSICS
from powered_landing_guidance.recovery_physics import G0, Geometry, LandingGear
from powered_landing_guidance.rtls.forces import foot_kinematics_scalar
from powered_landing_guidance.rtls.interfaces import ContactSample


@dataclass(frozen=True, slots=True)
class ContactSettings:
    """Conceptual 2-leg planar strut compliance and standing-stability limits.

    Sized for a ~7.5-9.5 t Methalox landing booster (weight ~74-93 kN, 2 planar feet
    representing 4 physical legs, static load ~37-47 kN per planar foot):
      - Static equilibrium deflection at 8 t: ~39 kN / 160 kN/m = 0.24 m
      - Critical damping ratio ~0.90 for 4 t per foot -> c ~ 45 kN*s/m
      - At 1.5 m/s vertical impact: max stroke ~0.36 m << 0.90 m limit,
        peak load ~1.4-2.3 g << 4.0 g limit, zero bounce.
    """

    leg_stiffness_n_m: float = 160_000.0
    leg_damping_n_s_m: float = 45_000.0
    friction_coefficient: float = 0.75
    friction_velocity_scale_m_s: float = 0.05
    max_leg_stroke_m: float = 0.90
    max_leg_force_n: float = 250_000.0
    max_contact_load_g: float = 4.0
    settle_duration_s: float = 5.0
    max_standing_tilt_deg: float = 5.0
    max_standing_speed_m_s: float = 0.08
    max_standing_rate_deg_s: float = 0.5


def evaluate_ground_contact(
    *,
    x_m: float,
    z_m: float,
    vx_m_s: float,
    vz_m_s: float,
    theta_rad: float,
    omega_rad_s: float,
    total_mass_kg: float,
    cg_m: float,
    gear_extension: float,
    settings: ContactSettings,
    geometry: Geometry = PHYSICS.geometry,
    gear: LandingGear = PHYSICS.gear,
    offset_x: float = 0.0,
    offset_z: float = 0.0,
) -> ContactSample:
    """Evaluate planar ground reaction forces, strokes, friction, and hull clearance."""
    (lpx, lpz), (rpx, rpz), (lvx, lvz), (rvx, rvz) = foot_kinematics_scalar(
        x_m,
        z_m,
        vx_m_s,
        vz_m_s,
        theta_rad,
        omega_rad_s,
        cg_m,
        gear_extension,
        geometry=geometry,
        gear=gear,
        offset_x=offset_x,
        offset_z=offset_z,
    )

    # Check hull bottom & nozzle clearance (body-frame corners at bottom of hull)
    c = cos(theta_rad)
    s = sin(theta_rad)
    r = geometry.body_radius_m
    hull_collision = False
    for bx, bz in (
        (-r + offset_x, -cg_m + offset_z),
        (r + offset_x, -cg_m + offset_z),
        (offset_x, geometry.nozzle_station_m - cg_m + offset_z),
        (-r + offset_x, geometry.body_length_m - cg_m + offset_z),
        (r + offset_x, geometry.body_length_m - cg_m + offset_z),
    ):
        wz = z_m - s * bx + c * bz
        if wz <= 0.0:
            hull_collision = True
            break

    def _leg_load(px: float, pz: float, vx: float, vz: float) -> tuple[float, float, float, bool]:
        if pz >= 0.0:
            return 0.0, 0.0, 0.0, False
        stroke = -pz
        # Unilateral viscoelastic contact: pushes upward only (no ground adhesion).
        fn = settings.leg_stiffness_n_m * stroke - settings.leg_damping_n_s_m * vz
        if fn < 0.0:
            fn = 0.0
        # Smooth regularized Coulomb friction opposing tangential foot slip
        ft = -settings.friction_coefficient * fn * tanh(vx / settings.friction_velocity_scale_m_s)
        return fn, ft, stroke, True

    l_fn, l_ft, l_stroke, l_touch = _leg_load(lpx, lpz, lvx, lvz)
    r_fn, r_ft, r_stroke, r_touch = _leg_load(rpx, rpz, rvx, rvz)

    fx = l_ft + r_ft
    fz = l_fn + r_fn

    # Clockwise torque = arm_z * Fx - arm_x * Fz about total COM
    tau = 0.0
    if l_touch:
        tau += (lpz - z_m) * l_ft - (lpx - x_m) * l_fn
    if r_touch:
        tau += (rpz - z_m) * r_ft - (rpx - x_m) * r_fn

    any_contact = l_touch or r_touch
    load_g = hypot(fx, fz) / (max(total_mass_kg, 1.0) * G0) if any_contact else 0.0

    return ContactSample(
        force_x_n=fx,
        force_z_n=fz,
        torque_nm=tau,
        left_normal_n=l_fn,
        right_normal_n=r_fn,
        left_friction_n=l_ft,
        right_friction_n=r_ft,
        left_stroke_m=l_stroke,
        right_stroke_m=r_stroke,
        left_in_contact=l_touch,
        right_in_contact=r_touch,
        any_contact=any_contact,
        body_ground_collision=hull_collision,
        contact_load_g=load_g,
    )
