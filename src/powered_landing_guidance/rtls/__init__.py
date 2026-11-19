"""DM100 SpaceX-style RTLS package (2D Plant -> 3D MuJoCo -> UE5 Replay)."""

from powered_landing_guidance.rtls.contact import ContactSettings, evaluate_ground_contact
from powered_landing_guidance.rtls.control_allocation import (
    AllocationInput,
    AllocationOutput,
    PriorityControlAllocator,
)
from powered_landing_guidance.rtls.convex_guidance import (
    ConvexLandingProblem,
    ConvexLandingSolution,
    solve_convex_hoverslam,
)
from powered_landing_guidance.rtls.flight import RtlsFlight
from powered_landing_guidance.rtls.forces import (
    AtmosphereLUT,
    RtlsAerodynamics,
    RtlsFinSettings,
    RtlsPropulsionSettings,
    WindLUT,
    compute_body_aero,
    compute_fin_aero,
    compute_forces_and_torques,
)
from powered_landing_guidance.rtls.guidance import RtlsGuidanceSettings, SpaceXRtlsController
from powered_landing_guidance.rtls.interfaces import (
    ActuatorCommand,
    ContactSample,
    ControllerProtocol,
    ForceTorqueSample,
    PlantProtocol,
    ReplayFrame,
    TelemetryPacket,
    pitch_to_quat_wxyz,
)
from powered_landing_guidance.rtls.plant_2d import Plant2D, PlantConfig
from powered_landing_guidance.rtls.replay import REPLAY_SCHEMA_VERSION, export_replay_json

__all__ = [
    "ActuatorCommand",
    "AllocationInput",
    "AllocationOutput",
    "AtmosphereLUT",
    "ContactSample",
    "ContactSettings",
    "ControllerProtocol",
    "ConvexLandingProblem",
    "ConvexLandingSolution",
    "ForceTorqueSample",
    "Plant2D",
    "PlantConfig",
    "PlantProtocol",
    "PriorityControlAllocator",
    "REPLAY_SCHEMA_VERSION",
    "ReplayFrame",
    "RtlsAerodynamics",
    "RtlsFinSettings",
    "RtlsFlight",
    "RtlsGuidanceSettings",
    "RtlsPropulsionSettings",
    "SpaceXRtlsController",
    "TelemetryPacket",
    "WindLUT",
    "compute_body_aero",
    "compute_fin_aero",
    "compute_forces_and_torques",
    "evaluate_ground_contact",
    "export_replay_json",
    "pitch_to_quat_wxyz",
    "solve_convex_hoverslam",
]
