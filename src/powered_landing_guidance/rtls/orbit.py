"""2D spherical-Earth Stage 2 orbital insertion evaluator (700 km circular LEO/SSO).

Propagates the detached Methalox upper stage from the exact MECO-1 cold-separation
state (altitude, downrange, local horizontal & vertical velocity) using a realistic
two-burn transfer + apogee circularization profile, including Earth rotational boost
at the launch latitude.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, exp, hypot, pi, radians, sin, sqrt

from powered_landing_guidance.recovery_physics import EARTH_RADIUS_M, G0

EARTH_MU_M3_S2 = 3.986004418e14
EARTH_EQUATORIAL_SPEED_M_S = 465.101


@dataclass(frozen=True, slots=True)
class UpperStageSettings:
    """Methalox vacuum upper stage sized for 1,800 kg payload to 700 km circular orbit."""

    dry_mass_kg: float = 1_550.0
    propellant_mass_kg: float = 20_650.0
    payload_mass_kg: float = 1_800.0
    vacuum_thrust_n: float = 240_000.0
    vacuum_isp_s: float = 380.0
    target_altitude_m: float = 700_000.0
    altitude_tolerance_m: float = 15_000.0
    ignition_coast_s: float = 4.0

    @property
    def total_mass_kg(self) -> float:
        return self.dry_mass_kg + self.propellant_mass_kg + self.payload_mass_kg


def evaluate_upper_stage_orbit(
    *,
    z_sep_m: float,
    vx_sep_m_s: float,
    vz_sep_m_s: float,
    latitude_deg: float = 37.0,
    settings: UpperStageSettings | None = None,
) -> dict:
    """Evaluate real 2-burn Stage 2 insertion into a 700 km circular orbit."""
    s = settings or UpperStageSettings()
    r = EARTH_RADIUS_M + max(z_sep_m, 0.0)
    # Inertial tangential velocity includes eastward Earth rotation at launch latitude
    v_earth = EARTH_EQUATORIAL_SPEED_M_S * cos(radians(latitude_deg))
    vt = vx_sep_m_s + v_earth
    vr = vz_sep_m_s

    # Short post-separation coast before Stage 2 engine ignition
    dt_coast = 0.2
    for _ in range(int(round(s.ignition_coast_s / dt_coast))):
        g_loc = EARTH_MU_M3_S2 / (r * r)
        ar = -g_loc + (vt * vt) / r
        at = -(vr * vt) / r
        vr += ar * dt_coast
        vt += at * dt_coast
        r += vr * dt_coast

    m_prop = s.propellant_mass_kg
    m_inert = s.dry_mass_kg + s.payload_mass_kg
    mdot = s.vacuum_thrust_n / (s.vacuum_isp_s * G0)
    r_target = EARTH_RADIUS_M + s.target_altitude_m
    dt = 0.2
    burn1_s = 0.0

    # Burn 1: lift apogee cleanly to 700 km while building tangential velocity
    while m_prop > 0.0 and burn1_s < 460.0:
        v2 = vr * vr + vt * vt
        eps = 0.5 * v2 - EARTH_MU_M3_S2 / r
        h_ang = r * vt
        step_dt = dt
        if eps < 0.0:
            a_orb = -EARTH_MU_M3_S2 / (2.0 * eps)
            e = sqrt(max(0.0, 1.0 + 2.0 * eps * h_ang * h_ang / (EARTH_MU_M3_S2**2)))
            ra = a_orb * (1.0 + e)
            if ra >= r_target:
                break
            if ra > r_target - 120_000.0:
                step_dt = 0.02
        else:
            break

        mass = m_inert + m_prop
        # Gentle pitch above local horizontal early to keep perigee safe, then prograde
        pitch_deg = max(0.0, 14.0 - 0.06 * burn1_s)
        pitch_rad = pitch_deg * (pi / 180.0)
        ar = (s.vacuum_thrust_n * sin(pitch_rad)) / mass - EARTH_MU_M3_S2 / (r * r) + (vt * vt) / r
        at = (s.vacuum_thrust_n * cos(pitch_rad)) / mass - (vr * vt) / r
        vr += ar * step_dt
        vt += at * step_dt
        r += vr * step_dt
        m_prop = max(0.0, m_prop - mdot * step_dt)
        burn1_s += step_dt

    v_end = hypot(vr, vt)
    eps = 0.5 * v_end * v_end - EARTH_MU_M3_S2 / r
    h_ang = r * vt
    if eps >= 0.0 or m_prop <= 0.0:
        return {
            "target_altitude_km": s.target_altitude_m / 1000.0,
            "payload_mass_kg": s.payload_mass_kg,
            "transfer_perigee_km": None,
            "transfer_apogee_km": None,
            "final_perigee_km": None,
            "final_apogee_km": None,
            "circularization_dv_m_s": None,
            "upper_propellant_remaining_kg": 0.0,
            "orbit_inserted": False,
        }

    a_orb = -EARTH_MU_M3_S2 / (2.0 * eps)
    e = sqrt(max(0.0, 1.0 + 2.0 * eps * h_ang * h_ang / (EARTH_MU_M3_S2**2)))
    ra = a_orb * (1.0 + e)
    rp = a_orb * (1.0 - e)

    # Burn 2: apogee circularization into 700 km circular orbit
    v_at_ra = sqrt(max(0.0, EARTH_MU_M3_S2 * (2.0 / ra - 1.0 / a_orb)))
    v_circ = sqrt(EARTH_MU_M3_S2 / ra)
    dv_circ = max(0.0, v_circ - v_at_ra)
    m_before = m_inert + m_prop
    m_after = m_before / exp(dv_circ / (s.vacuum_isp_s * G0))
    prop_rem = m_after - m_inert
    alt_km = (ra - EARTH_RADIUS_M) / 1000.0
    target_km = s.target_altitude_m / 1000.0
    tol_km = s.altitude_tolerance_m / 1000.0

    # Achievable circular orbit if positive propellant remains and apogee is within gate
    ok = (prop_rem > 0.0) and (abs(alt_km - target_km) <= tol_km)
    return {
        "target_altitude_km": target_km,
        "payload_mass_kg": s.payload_mass_kg,
        "burn1_duration_s": burn1_s,
        "transfer_perigee_km": (rp - EARTH_RADIUS_M) / 1000.0,
        "transfer_apogee_km": alt_km,
        "final_perigee_km": alt_km if ok else (rp - EARTH_RADIUS_M) / 1000.0,
        "final_apogee_km": alt_km,
        "circular_speed_m_s": v_circ,
        "circularization_dv_m_s": dv_circ,
        "upper_propellant_remaining_kg": max(0.0, prop_rem),
        "orbit_inserted": bool(ok),
    }
