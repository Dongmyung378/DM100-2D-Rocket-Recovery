"""Engine-agnostic interfaces for DM100 (2D Plant -> 3D MuJoCo -> UE5 Replay).

These data contracts strictly separate:
  1. Plant / Physics Engine (2D RK4 now, 3D MuJoCo later)
  2. External Aerodynamics & Propulsion Force Model (reusable as MuJoCo xfrc callback)
  3. Causal GNC Core (consumes ONLY TelemetryPacket, returns ActuatorCommand)
  4. Offline 50 Hz ENU + Quaternion Replay (consumable by UE5 Level Sequence importer)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, sin
from typing import Protocol


@dataclass(frozen=True, slots=True)
class TelemetryPacket:
    """Strictly causal telemetry delivered from Plant/Sensors to the GNC core.

    Hidden plant truth (true wind gusts, thrust/drag/mass/inertia scales,
    internal slosh coordinates, true sensor biases) MUST NEVER appear here.
    """

    time_s: float
    measurement_time_s: float
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
    packet_dropped: bool = False


@dataclass(frozen=True, slots=True)
class ActuatorCommand:
    """Commands issued by the GNC core to the Plant actuators.

    In 2D, throttle_cmd is total equivalent throttle fraction [0, 1] produced
    by `active_engines` running within [min_engine_throttle, 1.0] (or 0.0).
    In 3D MuJoCo, this extends directly to 4 fins, 2-axis gimbal, 3-axis RCS.
    """

    throttle_cmd: float = 0.0
    active_engines: int = 0
    gimbal_rad: float = 0.0
    rcs_torque_nm: float = 0.0
    fin_deflection_rad: float = 0.0
    deploy_fins: bool = False
    deploy_gear: bool = False
    phase: str = "ascent"


@dataclass(frozen=True, slots=True)
class ForceTorqueSample:
    """World-frame forces and body torque returned by the pure Force Model."""

    force_x_n: float
    force_z_n: float
    torque_nm: float
    aero_force_x_n: float
    aero_force_z_n: float
    body_torque_nm: float
    fin_force_x_n: float
    fin_force_z_n: float
    fin_torque_nm: float
    fin_hinge_nm: float
    thrust_n: float
    main_flow_kg_s: float
    rcs_flow_kg_s: float
    dynamic_pressure_pa: float
    mach: float
    alpha_deg: float
    fin_alpha_eff_deg: float


@dataclass(frozen=True, slots=True)
class ContactSample:
    """Post-touchdown landing-leg contact forces, strokes, and structural loads."""

    force_x_n: float = 0.0
    force_z_n: float = 0.0
    torque_nm: float = 0.0
    left_normal_n: float = 0.0
    right_normal_n: float = 0.0
    left_friction_n: float = 0.0
    right_friction_n: float = 0.0
    left_stroke_m: float = 0.0
    right_stroke_m: float = 0.0
    left_in_contact: bool = False
    right_in_contact: bool = False
    any_contact: bool = False
    body_ground_collision: bool = False
    contact_load_g: float = 0.0


def pitch_to_quat_wxyz(theta_rad: float) -> tuple[float, float, float, float]:
    """Convert planar clockwise-from-+Z pitch into right-handed ENU unit quaternion.

    In right-handed ENU (+X East, +Y North, +Z Up), a clockwise pitch about
    the viewing axis (-Y looking North, i.e. rotating +Z toward +X) is a
    positive rotation by +theta around the +Y axis:
        q = (cos(theta/2), 0, sin(theta/2), 0)
    Verify: rotating (0, 0, 1) by +theta around +Y gives (+sin(theta), 0, cos(theta)).
    """
    half = 0.5 * theta_rad
    return (cos(half), 0.0, sin(half), 0.0)


@dataclass(frozen=True, slots=True)
class ReplayFrame:
    """50 Hz ENU + unit-quaternion replay frame for MuJoCo / Unreal Engine 5.

    Coordinate convention:
      - Right-handed ENU (+X East, +Y North, +Z Up), SI units (meters, radians).
      - Unreal Engine 5 importer converts ENU (m) -> UE5 left-handed (cm) in one step.
    """

    time_s: float
    phase: str
    pos_enu_m: tuple[float, float, float]
    quat_wxyz: tuple[float, float, float, float]
    vel_enu_m_s: tuple[float, float, float]
    omega_body_rad_s: tuple[float, float, float]
    throttle: float
    active_engines: int
    gimbal_rad: tuple[float, float]
    rcs_torque_nm: tuple[float, float, float]
    fin_extension: float
    fin_deflection_rad: tuple[float, float, float, float]
    gear_extension: float
    leg_stroke_m: tuple[float, float, float, float]
    dynamic_pressure_pa: float
    mach: float
    alpha_deg: float
    main_propellant_kg: float
    rcs_propellant_kg: float
    upper_attached: bool
    upper_pos_enu_m: tuple[float, float, float] | None = None
    upper_quat_wxyz: tuple[float, float, float, float] | None = None

    def to_dict(self) -> dict:
        return {
            "time_s": self.time_s,
            "phase": self.phase,
            "pos_enu_m": list(self.pos_enu_m),
            "quat_wxyz": list(self.quat_wxyz),
            "vel_enu_m_s": list(self.vel_enu_m_s),
            "omega_body_rad_s": list(self.omega_body_rad_s),
            "throttle": self.throttle,
            "active_engines": self.active_engines,
            "gimbal_rad": list(self.gimbal_rad),
            "rcs_torque_nm": list(self.rcs_torque_nm),
            "fin_extension": self.fin_extension,
            "fin_deflection_rad": list(self.fin_deflection_rad),
            "gear_extension": self.gear_extension,
            "leg_stroke_m": list(self.leg_stroke_m),
            "dynamic_pressure_pa": self.dynamic_pressure_pa,
            "mach": self.mach,
            "alpha_deg": self.alpha_deg,
            "main_propellant_kg": self.main_propellant_kg,
            "rcs_propellant_kg": self.rcs_propellant_kg,
            "upper_attached": self.upper_attached,
            "upper_pos_enu_m": (
                list(self.upper_pos_enu_m) if self.upper_pos_enu_m is not None else None
            ),
            "upper_quat_wxyz": (
                list(self.upper_quat_wxyz) if self.upper_quat_wxyz is not None else None
            ),
        }


@dataclass(slots=True)
class GuidanceDiagnostics:
    """Optional diagnostic telemetry exported by the GNC core for analysis."""

    phase: str = "ascent"
    attitude_target_deg: float = 0.0
    glide_alpha_cmd_deg: float = 0.0
    predicted_iip_x_m: float | None = None
    predicted_unpowered_peak_q_pa: float | None = None
    hoverslam_ignition_height_m: float | None = None
    convex_solver_used: bool = False
    convex_solver_status: str = "unused"
    extra: dict = field(default_factory=dict)


class ControllerProtocol(Protocol):
    """Engine-agnostic GNC interface (shared between 2D Plant and 3D MuJoCo)."""

    def step(self, telemetry: TelemetryPacket) -> ActuatorCommand: ...

    def diagnostics(self) -> GuidanceDiagnostics: ...


class PlantProtocol(Protocol):
    """Physics backend interface implemented by Plant2D (and later MuJoCoPlant3D)."""

    def telemetry(self) -> TelemetryPacket: ...

    def step(self, command: ActuatorCommand, duration_s: float) -> None: ...

    @property
    def done(self) -> bool: ...
