"""50 Hz ENU + unit-quaternion replay export for MuJoCo / Unreal Engine 5.

Coordinate contract:
  - Right-handed ENU: +X East, +Y North, +Z Up; SI units (meters, radians).
  - Unit quaternion (w, x, y, z) describing body-to-world rotation.
  - Unreal Engine 5 importer converts ENU (m) -> UE5 left-handed (cm) in one step.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from powered_landing_guidance.rtls.interfaces import ReplayFrame

REPLAY_SCHEMA_VERSION = "dm100_replay_3d_v1"


def export_replay_json(
    frames: list[ReplayFrame],
    output_path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
    indent: int | None = None,
) -> Path:
    """Write a self-describing 3D/UE5-compatible replay JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "coordinate_frame": "ENU_right_handed_SI",
        "units": {"length": "m", "angle": "rad", "quaternion_order": "wxyz"},
        "ue5_conversion_note": (
            "Map ENU (x, y, z) [m] -> UE5 (x*100, -y*100, z*100) [cm]; "
            "pitch +theta around ENU +Y rotates +Z toward +X."
        ),
        "frame_count": len(frames),
        "metadata": metadata or {},
        "frames": [frame.to_dict() for frame in frames],
    }
    if indent is None:
        text = json.dumps(payload, separators=(",", ":"))
    else:
        text = json.dumps(payload, indent=indent)
    path.write_text(text, encoding="utf-8")
    return path
