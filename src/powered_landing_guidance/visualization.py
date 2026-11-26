"""Replay recorded 100 t flights without rerunning physics."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import matplotlib
import numpy as np
from numpy.typing import NDArray

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.animation import FFMpegWriter, writers  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import Ellipse, FancyBboxPatch, Polygon, Rectangle  # noqa: E402


def _ffmpeg_available() -> bool:
    try:
        from powered_landing_guidance.runtime_support import video_encoder_path

        matplotlib.rcParams["animation.ffmpeg_path"] = video_encoder_path()
    except (ImportError, OSError, RuntimeError):
        return False
    return writers.is_available("ffmpeg")


def _recovery_sky(altitudes: NDArray[np.float64]) -> NDArray[np.float64]:
    levels = (0.0, 2200.0, 7000.0, 14_000.0, 30_000.0, 58_000.0, 90_000.0)
    colors = np.asarray(
        (
            (0.68, 0.85, 0.96),
            (0.38, 0.68, 0.89),
            (0.18, 0.41, 0.68),
            (0.075, 0.19, 0.39),
            (0.022, 0.065, 0.16),
            (0.008, 0.022, 0.062),
            (0.004, 0.010, 0.028),
        )
    )
    return np.stack([np.interp(altitudes, levels, colors[:, i]) for i in range(3)], axis=-1)


def _strut(start: tuple, end: tuple, width: float) -> NDArray[np.float64]:
    a, b = np.asarray(start), np.asarray(end)
    direction = b - a
    length = np.linalg.norm(direction)
    if length < 1e-12:
        return np.repeat(a[None, :], 4, axis=0)
    normal = width * np.asarray((-direction[1], direction[0])) / length
    return np.asarray((a - normal, a + normal, b + normal, b - normal))


def _clip_plume_at_ground(vertices: NDArray[np.float64], ground: float) -> NDArray[np.float64]:
    """Clip exhaust only, not solid rocket geometry or simulated states."""
    clipped = []
    previous = vertices[-1]
    for current in vertices:
        if (previous[1] >= ground) != (current[1] >= ground):
            fraction = (ground - previous[1]) / (current[1] - previous[1])
            crossing = previous + fraction * (current - previous)
            crossing[1] = ground
            clipped.append(crossing)
        if current[1] >= ground:
            clipped.append(current)
        previous = current
    return np.asarray(clipped, dtype=np.float64).reshape(-1, 2)


def _physical_recovery_geometry(state, deployment, settings):
    """World-space hull and uncompressed feet at the recorded first contact."""
    from powered_landing_guidance.recovery_physics import foot_kinematics, mass_properties, rotation

    geometry, gear = settings.geometry, settings.gear
    radius, length = geometry.body_radius_m, geometry.body_length_m
    cg = mass_properties(state[6], deployment, settings).cg_m
    parts = {
        "body": np.asarray(
            ((-radius, 0), (radius, 0), (radius, length - 0.3), (-radius, length - 0.3))
        ),
        "adapter": np.asarray(
            (
                (-radius, length - 0.3),
                (radius, length - 0.3),
                (radius * 0.7, length),
                (-radius * 0.7, length),
            )
        ),
        "upper_band": np.asarray(
            (
                (-radius, length - 0.65),
                (radius, length - 0.65),
                (radius, length - 0.35),
                (-radius, length - 0.35),
            )
        ),
        "lower_band": np.asarray(((-radius, 0), (radius, 0), (radius, 0.3), (-radius, 0.3))),
        "highlight": np.asarray(
            (
                (-radius * 0.7, 0.3),
                (-radius * 0.4, 0.3),
                (-radius * 0.4, length - 0.65),
                (-radius * 0.7, length - 0.65),
            )
        ),
        "panel": np.asarray(
            ((radius * 0.4, 1.7), (radius * 0.7, 1.7), (radius * 0.7, 3.0), (radius * 0.4, 3.0))
        ),
        "nozzle": np.asarray(
            (
                (-0.10, 0),
                (0.10, 0),
                (geometry.nozzle_radius_m, geometry.nozzle_station_m),
                (-geometry.nozzle_radius_m, geometry.nozzle_station_m),
            )
        ),
        "fin_left": np.asarray(
            (
                (-radius, length - 0.8),
                (-radius - 0.3, length - 1.0),
                (-radius - 0.3, length - 1.35),
                (-radius, length - 1.2),
            )
        ),
        "fin_right": np.asarray(
            (
                (radius, length - 0.8),
                (radius + 0.3, length - 1.0),
                (radius + 0.3, length - 1.35),
                (radius, length - 1.2),
            )
        ),
    }
    transform = rotation(state[4])
    offset = (
        settings.body_frame_offset(state)[0]
        if hasattr(settings, "body_frame_offset")
        else np.zeros(2)
    )
    origin = state[:2] + transform @ offset
    parts = {name: (vertices - (0, cg)) @ transform.T + origin for name, vertices in parts.items()}
    feet, _ = foot_kinematics(state, deployment, settings)
    for index, (sign, side) in enumerate(((-1, "left"), (1, "right"))):
        foot = feet[index].copy()
        anchor = np.asarray((sign * radius, gear.anchor_station_m - cg)) @ transform.T + origin
        brace = np.asarray((sign * radius, -cg)) @ transform.T + origin
        parts[f"leg_{side}"] = _strut(anchor, foot, 0.035)
        parts[f"brace_{side}"] = _strut(brace, foot, 0.018)
        parts[f"foot_{side}"] = foot + np.asarray(
            ((-0.18, 0), (0.18, 0), (0.18, 0.05), (-0.18, 0.05))
        )
    nozzle = np.asarray((0, geometry.nozzle_station_m - cg)) @ transform.T + origin
    return parts, nozzle


class _NominalFlightReplay:
    """Display interpolation of a saved flight; no controller or integration calls."""

    def __init__(self, flight: dict[str, Any]) -> None:
        from powered_landing_guidance.apr_recovery import OPERATING_SCOPE, STATE_NAMES, AprPhysics
        from powered_landing_guidance.recovery_physics import (
            AtmosphereSettings,
            Geometry,
            LandingGear,
            mass_properties,
        )
        from powered_landing_guidance.vehicle_systems import SloshSettings

        if (
            flight.get("environment_version") != OPERATING_SCOPE
            or flight.get("systems_model_version") != "apr_100t_coupled_systems_v2"
            or flight.get("state_names") != list(STATE_NAMES)
            or flight.get("landing_assessment") not in ("kinematic_touchdown", "contact_dynamics")
        ):
            raise ValueError("unsupported current 100 t flight format or state order")
        steps = flight["steps"]
        self.times = np.asarray([row["time_s"] for row in steps], dtype=np.float64)
        self.states = np.asarray([row["state"] for row in steps], dtype=np.float64)
        self.controls = np.asarray([row["control"] for row in steps], dtype=np.float64)
        if (
            len(self.times) < 2
            or self.states.shape != (len(self.times), len(STATE_NAMES))
            or self.controls.shape != (len(self.times), 4)
            or not np.all(np.isfinite(self.times))
            or np.any(np.diff(self.times) <= 0)
            or not np.all(np.isfinite(self.states))
            or not np.all(np.isfinite(self.controls))
        ):
            raise ValueError("full-flight replay requires finite, strictly ordered samples")
        # Exact event samples take precedence over sparse 0.2 s display samples.
        # In particular, interpolating across shutdown otherwise shows a false
        # staging throttle. The saved input and simulation are never modified.
        replay_steps = [dict(row) for row in steps]
        for event_name, event_phase in (("separation", "flip"), ("contact", "touchdown")):
            event = flight.get(event_name)
            if not event or "state" not in event:
                continue
            event_time = float(event["time_s"])
            event_state = np.asarray(event["state"], dtype=np.float64)
            if event_state.shape != (len(STATE_NAMES),) or not np.all(np.isfinite(event_state)):
                raise ValueError("replay event state must match the current finite state order")
            if not self.times[0] <= event_time <= self.times[-1]:
                raise ValueError("replay event must lie within the recorded interval")
            previous = max(0, int(np.searchsorted(self.times, event_time, side="right") - 1))
            event_row = {
                **replay_steps[previous],
                "time_s": event_time,
                "state": event_state.tolist(),
                "phase": event_phase,
            }
            if all("dynamic_pressure_pa" in row for row in steps):
                event_row["dynamic_pressure_pa"] = float(
                    np.interp(event_time, self.times, [row["dynamic_pressure_pa"] for row in steps])
                )
            existing = next(
                (i for i, row in enumerate(replay_steps) if abs(row["time_s"] - event_time) < 1e-9),
                None,
            )
            if existing is None:
                replay_steps.append(event_row)
            else:
                replay_steps[existing] = event_row
        replay_steps.sort(key=lambda row: row["time_s"])
        self.times = np.asarray([row["time_s"] for row in replay_steps], dtype=np.float64)
        self.states = np.asarray([row["state"] for row in replay_steps], dtype=np.float64)
        self.controls = np.asarray([row["control"] for row in replay_steps], dtype=np.float64)
        self.steps = replay_steps
        self.flight = flight
        self.design = flight["design"]
        self.phases = [row["phase"] for row in replay_steps]
        self.active_engines = [
            int(
                row["active_engines"]
                if row.get("active_engines") is not None
                else (
                    7
                    if row["phase"] == "ascent"
                    else 3
                    if row["phase"] in {"boostback", "entry", "entry_burn", "landing_burn"}
                    else 1
                )
            )
            for row in replay_steps
        ]
        self.separation_s = (
            float(flight["separation"]["time_s"]) if flight.get("separation") else np.inf
        )
        self.contact_record = flight.get("contact")
        self.contact_s = float(self.contact_record["time_s"]) if self.contact_record else np.inf
        physics = flight["physics"]
        self.upper_station_m = float(physics["upper_station_m"])
        self.upper_length_m = float(physics["upper_length_m"])
        self.settings = AprPhysics(
            geometry=Geometry(**physics["geometry"]),
            gear=LandingGear(**physics["gear"]),
            atmosphere=AtmosphereSettings(**physics["atmosphere"]),
            fixed_mass_components=tuple(tuple(item) for item in physics["fixed_mass_components"]),
            slosh_mass_fraction=physics["slosh_mass_fraction"],
            slosh_settings=SloshSettings(**physics["slosh_settings"]),
        )
        self.bases = self.states[:, :2].copy()
        for index, state in enumerate(self.states):
            mass = self.design["dry_mass_kg"] + state[6] + state[7]
            cg = mass_properties(mass, state[8], self.settings).cg_m
            total_mass = mass
            if self.times[index] < self.separation_s:
                upper = self.design["upper_mass_kg"]
                cg = (mass * cg + upper * self.upper_station_m) / (mass + upper)
                total_mass += upper
            offset = self.com_mode_offset(state, total_mass)
            from powered_landing_guidance.recovery_physics import rotation

            self.bases[index] -= rotation(state[4]) @ (np.array([0.0, cg]) + offset)
        self.upper_departure = _UpperStageDeparture.from_replay(self)

    def com_mode_offset(self, state, mass):
        vector = np.r_[state[:6], mass, state[6], state[17:19]]
        return -self.settings.body_frame_offset(vector)[0]

    def sample(self, time_s: float):
        """Interpolate the body base, not the COM discontinuity at separation."""
        from powered_landing_guidance.recovery_physics import mass_properties

        time_s = float(np.clip(time_s, self.times[0], self.times[-1]))
        left = int(np.searchsorted(self.times, time_s, side="right") - 1)
        right = min(left + 1, len(self.times) - 1)
        weight = (
            (time_s - self.times[left]) / (self.times[right] - self.times[left])
            if right != left
            else 0.0
        )
        state = (1 - weight) * self.states[left] + weight * self.states[right]
        base = (1 - weight) * self.bases[left] + weight * self.bases[right]
        mass = self.design["dry_mass_kg"] + state[6] + state[7]
        cg = mass_properties(mass, state[8], self.settings).cg_m
        total_mass = mass
        if time_s < self.separation_s:
            upper = self.design["upper_mass_kg"]
            cg = (mass * cg + upper * self.upper_station_m) / (mass + upper)
            total_mass += upper
        from powered_landing_guidance.recovery_physics import rotation

        state[:2] = base + rotation(state[4]) @ (
            np.array([0.0, cg]) + self.com_mode_offset(state, total_mass)
        )
        control = self.controls[left].copy()
        control[0] = float(state[12])
        control[1] = float(state[11])
        phase = self.phases[left]
        if time_s >= self.contact_s:
            phase = "touchdown"
        elif time_s >= self.separation_s and phase == "ascent":
            phase = "boostback"
        return state, base, mass, control, phase, left


class _UpperStageDeparture:
    """A display-only departure anchored to the recorded staging impulse.

    Ignition delay, acceleration ramp and attitude damping are cinematic
    assumptions. They neither integrate an upper-stage vehicle nor establish
    orbital insertion, propellant use or flight acceptance.
    """

    ignition_delay_s = 3.0
    acceleration_m_s2 = 18.0
    ramp_s = 1.2
    attitude_damping_s = 4.0

    @classmethod
    def from_replay(cls, replay):
        separation = replay.flight.get("separation")
        recovery_only = replay.flight.get("ascent_simulated") is False or (
            replay.flight.get("simulation_scope") == "37km_separation_to_touchdown"
        )
        if not separation or recovery_only or separation.get("impulse_model") is None:
            return None
        return cls(replay)

    def __init__(self, replay):
        from powered_landing_guidance.recovery_physics import gravity

        separation = replay.flight["separation"]
        impulse = separation["impulse_model"]
        try:
            self.position = np.asarray(impulse["upper_position_m"], dtype=float)
            self.velocity = np.asarray(impulse["upper_velocity_m_s"], dtype=float)
            self.theta = float(separation["state"][4])
            self.omega = float(impulse["upper_angular_rate_rad_s"])
            self.separation_s = float(separation["time_s"])
        except (KeyError, TypeError, ValueError, IndexError) as error:
            raise ValueError(
                "upper-stage departure requires recorded finite staging states"
            ) from error
        if (
            self.position.shape != (2,)
            or self.velocity.shape != (2,)
            or not np.all(np.isfinite(self.position))
            or not np.all(np.isfinite(self.velocity))
            or not np.all(np.isfinite((self.theta, self.omega, self.separation_s)))
        ):
            raise ValueError("upper-stage departure requires recorded finite staging states")
        self.cg_station_m = replay.upper_station_m - replay.settings.geometry.body_length_m
        self.gravity = np.asarray((0.0, -gravity(self.position[1])))
        self.axis = np.asarray((np.sin(self.theta), np.cos(self.theta)))

    def sample(self, time_s):
        """Return an illustrative COM pose, never a simulated upper-stage state."""
        if time_s < self.separation_s:
            return None
        elapsed = float(time_s - self.separation_s)
        powered = max(0.0, elapsed - self.ignition_delay_s)
        ratio = float(-np.expm1(-powered / self.ramp_s))
        velocity_ramp = powered - self.ramp_s * ratio
        position_ramp = 0.5 * powered**2 - self.ramp_s * powered + self.ramp_s**2 * ratio
        acceleration = self.acceleration_m_s2 * self.axis
        position = (
            self.position
            + self.velocity * elapsed
            + 0.5 * self.gravity * elapsed**2
            + acceleration * position_ramp
        )
        velocity = self.velocity + self.gravity * elapsed + acceleration * velocity_ramp
        damp = np.exp(-elapsed / self.attitude_damping_s)
        theta = self.theta + self.omega * self.attitude_damping_s * (1 - damp)
        return position, velocity, float(theta), float(self.omega * damp), ratio

    def base(self, time_s):
        from powered_landing_guidance.recovery_physics import rotation

        sample = self.sample(time_s)
        if sample is None:
            return None
        position, _, theta, _, _ = sample
        return position - rotation(theta) @ np.asarray((0.0, self.cg_station_m)), theta


def _upper_stage_geometry(base, theta, radius, length):
    from powered_landing_guidance.recovery_physics import rotation

    parts = {
        "upper_stage": np.asarray(
            (
                (-radius, 0),
                (radius, 0),
                (radius, length * 0.7),
                (radius * 0.65, length * 0.9),
                (0, length),
                (-radius * 0.65, length * 0.9),
                (-radius, length * 0.7),
            )
        ),
        "upper_nozzle": np.asarray(((-0.42, 0), (0.42, 0), (0.20, 0.65), (-0.20, 0.65))),
        "upper_highlight": np.asarray(
            (
                (-radius * 0.65, 0.8),
                (-radius * 0.4, 0.8),
                (-radius * 0.4, length * 0.67),
                (-radius * 0.65, length * 0.67),
            )
        ),
        "upper_band_detail": np.asarray(
            ((-radius, 0.7), (radius, 0.7), (radius, 1.0), (-radius, 1.0))
        ),
    }
    transform = rotation(theta)
    return {name: vertices @ transform.T + base for name, vertices in parts.items()}


def _nominal_flight_geometry(replay: _NominalFlightReplay, time_s: float):
    """Draw the recorded stage-one geometry and its actual ground-contact points."""
    from powered_landing_guidance.recovery_physics import rotation

    state, base, mass, control, phase, index = replay.sample(time_s)
    vector = np.r_[state[:6], mass, state[6], state[17:19]]
    # Geometry utilities describe the first stage. Its own COM is reconstructed
    # from the continuous hull base even while the upper assembly is attached.
    from powered_landing_guidance.recovery_physics import mass_properties

    first_cg = mass_properties(mass, state[8], replay.settings).cg_m
    vector[:2] = base + rotation(state[4]) @ (
        np.array([0.0, first_cg]) + replay.com_mode_offset(state, mass)
    )
    parts, nozzle = _physical_recovery_geometry(vector, state[8], replay.settings)
    transform = rotation(state[4])
    radius = replay.settings.geometry.body_radius_m
    station = replay.flight["fin_settings"]["station_m"]
    fin_span = np.sqrt(replay.flight["fin_settings"]["planar_effective_area_m2"] / 2)
    for sign, side in ((-1, "left"), (1, "right")):
        hinge = np.asarray((sign * radius, station))
        half_chord = fin_span / 2
        panel = np.asarray(
            (
                (0, -half_chord),
                (sign * fin_span, -half_chord),
                (sign * fin_span, half_chord),
                (0, half_chord),
            )
        )
        panel[:, 0] *= state[10]
        panel = panel @ rotation(state[9]).T + hinge
        parts[f"fin_{side}"] = panel @ transform.T + base
    length = replay.settings.geometry.body_length_m
    upper_length = replay.upper_length_m
    upper_base = base + transform @ np.asarray((0, length))
    upper_theta = state[4]
    if time_s >= replay.separation_s and replay.upper_departure is not None:
        upper_base, upper_theta = replay.upper_departure.base(time_s)
    parts.update(_upper_stage_geometry(upper_base, upper_theta, radius, upper_length))
    parts["separation_band"] = (
        np.asarray(
            (
                (-radius, length - 0.05),
                (radius, length - 0.05),
                (radius, length + 0.08),
                (-radius, length + 0.08),
            )
        )
        @ transform.T
        + base
    )
    return parts, nozzle, (state, base, mass, control, phase, index)


def _nominal_flight_frame_times(times: NDArray[np.float64], fps: int) -> NDArray[np.float64]:
    """One recorded second per playback second, with only the final frame rounded."""
    count = int(np.ceil((times[-1] - times[0]) * fps)) + 1
    return np.minimum(times[0] + np.arange(count) / fps, times[-1])


class _NominalFlightAnimationScene:
    """A flight-recorder presentation; scenery and engine-off outro are visual only."""

    def __init__(self, flight: dict[str, Any], *, width: int = 1920) -> None:
        self.replay = _NominalFlightReplay(flight)
        self.fig = plt.figure(figsize=(16, 9), dpi=width / 16, facecolor="#040E1D")
        self.fig.add_artist(
            Rectangle(
                (0.0, 0.884),
                1.0,
                0.116,
                transform=self.fig.transFigure,
                facecolor="#071830",
                edgecolor="none",
                zorder=0,
            )
        )
        self.fig.add_artist(
            FancyBboxPatch(
                (0.036, 0.200),
                0.573,
                0.665,
                transform=self.fig.transFigure,
                boxstyle="round,pad=0.002,rounding_size=0.008",
                facecolor="#051326",
                edgecolor="#244975",
                linewidth=1.1,
                zorder=0,
            )
        )
        self.camera = self.fig.add_axes((0.04, 0.205, 0.565, 0.655))
        self.camera_aspect = (16 * 0.565) / (9 * 0.655)
        self.overview = self.fig.add_axes((0.656, 0.686, 0.300, 0.152), facecolor="#061529")
        self.info = self.fig.add_axes((0.645, 0.205, 0.32, 0.432))
        self.info.axis("off")
        self.display_engine_on = False
        self.displayed_thrust_n = 0.0
        self.rcs_torque_kn_m = 0.0
        self.rcs_jet_length_m = 0.0
        self.q = np.asarray(
            [row.get("dynamic_pressure_pa", np.nan) for row in self.replay.steps], dtype=float
        )
        recovery_only = flight.get("ascent_simulated") is False or (
            flight.get("simulation_scope") == "37km_separation_to_touchdown"
        )
        self.fig.add_artist(
            Rectangle(
                (0.04, 0.935),
                0.006,
                0.042,
                transform=self.fig.transFigure,
                facecolor="#D4AF5A",
                edgecolor="none",
                zorder=2,
            )
        )
        self.title = self.fig.text(
            0.054, 0.943, "DM100", color="#F7F5F0", fontsize=25, weight="bold"
        )
        self.subtitle = self.fig.text(
            0.160,
            0.949,
            "100톤 메탈록스 700 km 원궤도 발사 · 1단 재사용 RTLS",
            color="#C4D4E6",
            fontsize=13.5,
        )
        self.mission_label = self.fig.text(
            0.956,
            0.950,
            "1단 회수" if recovery_only else "700 km LEO · RTLS",
            ha="right",
            color="#E5C573",
            fontsize=12,
            weight="bold",
        )
        conditions = flight.get("conditions", {})
        calm = not conditions.get("gust_amplitude_m_s", 0) and not any(
            speed
            for key in ("wind_profile", "vertical_wind_profile")
            for _, speed in conditions.get(key, ())
        )
        self.record_note = self.fig.text(
            0.054,
            0.903,
            "무풍 기준 비행 · LCH4/LOX 7→3→1 클러스터"
            if calm
            else "외란 조건 비행 · LCH4/LOX 7→3→1 클러스터",
            color="#8DA8C6",
            fontsize=9.8,
        )
        self.clock = self.fig.text(
            0.956,
            0.902,
            "",
            ha="right",
            color="#F5F3EE",
            fontsize=13.5,
            weight="bold",
            family="DejaVu Sans",
        )
        has_orbit = bool(flight.get("upper_stage_orbit"))
        self.footer = self.fig.text(
            0.04,
            0.025,
            (
                "DM100 2D 개념 모델 · 2단 700 km 원궤도(1,800 kg) 별도 적분 판정 · 1단 RTLS 회수"
                if has_orbit
                else "2D 모델 · 1단 RTLS 회수 시뮬레이션"
            ),
            color="#8DA8C6",
            fontsize=9,
        )
        self.fig.add_artist(
            Rectangle(
                (0.04, 0.883),
                0.92,
                0.0022,
                transform=self.fig.transFigure,
                facecolor="#D4AF5A",
                alpha=0.72,
                edgecolor="none",
            )
        )
        self.camera.set_axis_off()
        self.camera.set_aspect("equal", adjustable="box")
        self.sky = self.camera.imshow(
            np.zeros((96, 96, 3), dtype=np.uint8), origin="lower", aspect="auto", zorder=0
        )
        self.earth_limb = Ellipse(
            (0.5, -1.72),
            3.2,
            3.65,
            transform=self.camera.transAxes,
            facecolor="#1E6CA8",
            edgecolor="#6DD5FA",
            linewidth=2.4,
            alpha=0.0,
            zorder=0.6,
        )
        self.camera.add_patch(self.earth_limb)
        scenery_rng = np.random.default_rng(378)
        self.stars = self.camera.scatter(
            scenery_rng.uniform(0.03, 0.97, 92),
            scenery_rng.uniform(0.08, 0.96, 92),
            s=scenery_rng.uniform(0.4, 3.2, 92),
            transform=self.camera.transAxes,
            color="#E4F0FF",
            alpha=0,
            zorder=1,
        )
        self.clouds = []
        for cx, cy, size in ((0.18, 0.70, 0.20), (0.80, 0.49, 0.25), (0.52, 0.86, 0.14)):
            for dx, dy, factor in ((-0.3, 0, 0.65), (0, 0.04, 1), (0.3, -0.01, 0.7)):
                cloud = Ellipse(
                    (cx + dx * size, cy + dy * size),
                    size * factor,
                    size * 0.18 * factor,
                    transform=self.camera.transAxes,
                    facecolor="#EBF4FC",
                    edgecolor="none",
                    alpha=0.16,
                    zorder=1,
                )
                self.camera.add_patch(cloud)
                self.clouds.append(cloud)
        self.hills = []
        for color, peak, opacity in (("#4F6D7E", 3.4, 0.34), ("#35505C", 1.9, 0.56)):
            hill = Polygon(
                np.zeros((7, 2)), facecolor=color, alpha=opacity, edgecolor="none", zorder=1
            )
            self.camera.add_patch(hill)
            self.hills.append((hill, peak))
        self.terrain = Polygon(np.zeros((4, 2)), facecolor="#2D3E36", edgecolor="none", zorder=2)
        self.camera.add_patch(self.terrain)
        self.soil = Polygon(np.zeros((4, 2)), facecolor="#1E2824", edgecolor="none", zorder=2.1)
        self.camera.add_patch(self.soil)
        self.camera.add_patch(
            Rectangle(
                (-22, -0.55),
                44,
                0.55,
                facecolor="#5C6C75",
                edgecolor="#98ABB5",
                linewidth=0.6,
                zorder=3,
            )
        )
        span = self.replay.settings.gear.pad_half_width_m
        self.camera.add_patch(
            Rectangle(
                (-span, -0.16),
                span * 2,
                0.16,
                facecolor="#8F9EA6",
                edgecolor="#CFDCE3",
                linewidth=0.8,
                zorder=4,
            )
        )
        self.camera.plot(
            (-span + 0.8, span - 0.8), (0.02, 0.02), color="#E5C573", linewidth=1.4, zorder=4
        )
        self.camera.plot((-1.3, 1.3), (0.03, 0.03), color="#56CCF2", linewidth=2.3, zorder=5)
        for px in (-14, -11.5, 11.5, 14):
            self.camera.add_patch(
                Rectangle((px, -0.52), 0.32, 0.52, facecolor="#D4AF5A", edgecolor="none", zorder=4)
            )
        # The launch supports are outside the nozzle and do not alter contact geometry.
        self.launch_supports = []
        for sign in (-1, 1):
            support = Polygon(
                (
                    (sign * 1.1 - 0.12, 0),
                    (sign * 1.1 + 0.12, 0),
                    (sign * 1.35 + 0.12, 1),
                    (sign * 1.35 - 0.12, 1),
                ),
                facecolor="#586B78",
                edgecolor="#B9CAD4",
                linewidth=0.6,
                zorder=4,
            )
            self.camera.add_patch(support)
            self.launch_supports.append(support)
        self._draw_ground_facilities()
        (self.trail,) = self.camera.plot(
            [], [], color="#56CCF2", alpha=0.42, linewidth=1.45, zorder=2.2
        )
        self.parts = {}
        geometry, _, _ = _nominal_flight_geometry(self.replay, self.replay.times[0])
        colors = {
            "body": "#DDE6ED",
            "adapter": "#E4ECF2",
            "upper_stage": "#F0F4F8",
            "upper_nozzle": "#1B2C3B",
            "upper_highlight": "#FFFFFF",
            "upper_band_detail": "#0E2A4B",
            "highlight": "#F8FBFE",
            "panel": "#7D92A3",
            "nozzle": "#1E2E3B",
            "upper_band": "#0D2545",
            "lower_band": "#172A3D",
            "separation_band": "#D4AF5A",
            "fin_left": "#1E364D",
            "fin_right": "#1E364D",
        }
        for name in geometry:
            patch = Polygon(
                np.zeros((4, 2)),
                facecolor=colors.get(name, "#4B6478"),
                edgecolor="#1A2E42",
                linewidth=0.55,
                zorder=9 if name == "highlight" else 8,
            )
            self.camera.add_patch(patch)
            self.parts[name] = patch
        self.shell_shading = []
        for color, alpha in (("#738B9E", 0.48), ("#FFFFFF", 0.66), ("#30495E", 0.44)):
            patch = Polygon(
                np.zeros((4, 2)), facecolor=color, alpha=alpha, edgecolor="none", zorder=9
            )
            self.camera.add_patch(patch)
            self.shell_shading.append(patch)
        self.engine_skirt = Polygon(
            np.zeros((4, 2)),
            facecolor="#1E3042",
            edgecolor="#6E8CA3",
            linewidth=0.45,
            zorder=9,
        )
        self.camera.add_patch(self.engine_skirt)
        self.seams = LineCollection([], colors="#5C7588", linewidths=0.55, zorder=10)
        self.fin_mesh = LineCollection([], colors="#C6A65C", linewidths=0.65, zorder=10)
        self.camera.add_collection(self.seams)
        self.camera.add_collection(self.fin_mesh)
        self.body_label = self.camera.text(
            0,
            0,
            "DM100",
            ha="center",
            va="center",
            rotation=90,
            color="#0B2442",
            fontsize=9.2,
            weight="bold",
            zorder=11,
        )
        # Authentic LCH4/LOX Methalox violet-cyan Mach-diamond main plumes
        self.plumes = []
        for color, alpha in (
            ("#5878FF", 0.24),
            ("#4FB4FF", 0.66),
            ("#C4EBFF", 0.93),
            ("#F8FDFF", 1.0),
        ):
            plume = Polygon(
                np.zeros((4, 2)), facecolor=color, edgecolor="none", alpha=alpha, zorder=6
            )
            self.camera.add_patch(plume)
            self.plumes.append(plume)
        self.mach_diamonds = []
        for _ in range(3):
            diamond = Polygon(
                np.zeros((4, 2)), facecolor="#F2FAFF", edgecolor="none", alpha=0.88, zorder=6.4
            )
            self.camera.add_patch(diamond)
            self.mach_diamonds.append(diamond)
        self.upper_plumes = []
        for color, alpha, order in (
            ("#6086FF", 0.26, 6),
            ("#8ED6FF", 0.68, 6.1),
            ("#F4FCFF", 0.96, 6.2),
        ):
            plume = Polygon(
                np.zeros((4, 2)), facecolor=color, edgecolor="none", alpha=alpha, zorder=order
            )
            self.camera.add_patch(plume)
            self.upper_plumes.append(plume)
        self.upper_tag = self.camera.text(
            0,
            0,
            "2단 · 700 km 궤도",
            color="#F5DC93",
            fontsize=8.5,
            weight="bold",
            ha="left",
            va="center",
            bbox={
                "facecolor": "#071830",
                "edgecolor": "#D4AF5A",
                "boxstyle": "round,pad=0.3",
                "alpha": 0.86,
            },
            zorder=13,
        )
        self.upper_tag.set_visible(False)
        self.reentry_glows = []
        for color, max_a, order in (
            ("#FF5722", 0.20, 7.1),
            ("#FF9638", 0.30, 7.2),
            ("#FFE4B5", 0.38, 7.3),
        ):
            glow = Polygon(
                np.zeros((13, 2)), facecolor=color, edgecolor="none", alpha=0.0, zorder=order
            )
            self.camera.add_patch(glow)
            self.reentry_glows.append((glow, max_a))
        self.rcs_jets = []
        for _ in range(2):
            jet = Polygon(
                np.zeros((3, 2)), facecolor="#DDF2FF", edgecolor="none", alpha=0.72, zorder=7
            )
            self.camera.add_patch(jet)
            self.rcs_jets.append(jet)
        self.heat_glow = Ellipse(
            (0, 0), 10, 1, facecolor="#9AD4FF", edgecolor="none", alpha=0, zorder=5
        )
        self.camera.add_patch(self.heat_glow)
        self.dust = []
        for i in range(8):
            puff = Ellipse((0, 0), 1, 1, facecolor="#CFDCE5", edgecolor="none", alpha=0, zorder=7)
            self.camera.add_patch(puff)
            self.dust.append((puff, i))
        self.phase = self.camera.text(
            0.025,
            0.96,
            "",
            transform=self.camera.transAxes,
            va="top",
            color="#F7F5F0",
            fontsize=11.5,
            weight="bold",
            bbox={
                "facecolor": "#071932",
                "edgecolor": "#D4AF5A",
                "boxstyle": "round,pad=0.48",
                "alpha": 0.92,
                "linewidth": 1.1,
            },
            zorder=14,
        )
        self.camera_note = self.camera.text(
            0.025,
            0.032,
            "",
            transform=self.camera.transAxes,
            color="#DCE9F5",
            fontsize=9,
            zorder=14,
            bbox={
                "facecolor": "#071932",
                "edgecolor": "#244975",
                "boxstyle": "round,pad=0.35",
                "alpha": 0.84,
            },
        )
        (self.scale_line,) = self.camera.plot([], [], color="#E8F3FA", linewidth=1.8, zorder=15)
        self.scale_label = self.camera.text(
            0, 0, "", color="#E8F3FA", fontsize=8.2, ha="center", va="bottom", zorder=15
        )
        self.telemetry = []
        self.telemetry_labels = []
        self.telemetry_bars: dict[int, Rectangle] = {}
        self.telemetry_heading = self.fig.text(
            0.654,
            0.638,
            "실시간 비행 텔레메트리",
            color="#E5C573",
            fontsize=9.5,
            weight="bold",
        )
        labels = (
            "중심 고도 (COM)",
            "발사장 수평 위치",
            "수직속도 (COM)",
            "수평속도 (COM)",
            "자세 · 수직 기준",
            "각속도",
            "주 추진제",
            "RCS 추진제",
            "비행 질량",
            "동압",
            "주 엔진 추력",
            "실제 스로틀",
            "핀 전개율",
            "다리 전개율",
        )
        bar_colors = {6: "#56CCF2", 7: "#D4AF5A", 9: "#56CCF2", 11: "#E5C573"}
        # Fixed-width tiles keep changing values aligned, with embedded live gauges.
        for i, label in enumerate(labels):
            col, row = i % 2, i // 2
            xx, yy = 0.015 + col * 0.522, 0.955 - row * 0.140
            self.info.add_patch(
                FancyBboxPatch(
                    (xx, yy - 0.109),
                    0.460,
                    0.118,
                    transform=self.info.transAxes,
                    boxstyle="round,pad=0.002,rounding_size=0.014",
                    facecolor="#091B33",
                    edgecolor="#1E3F66",
                    linewidth=0.75,
                )
            )
            label_artist = self.info.text(
                xx + 0.022, yy + 0.002, label, color="#8DAAC7", fontsize=8.1, va="top"
            )
            self.telemetry_labels.append(label_artist)
            value = self.info.text(
                xx + 0.022,
                yy - 0.036,
                "",
                color="#56CCF2" if i == 6 else "#E5C573" if i in (7, 11) else "#EEF5FB",
                fontsize=12.2,
                weight="bold",
                va="top",
                family="DejaVu Sans",
            )
            self.telemetry.append(value)
            if i in bar_colors:
                self.info.add_patch(
                    Rectangle(
                        (xx + 0.020, yy - 0.101),
                        0.420,
                        0.011,
                        transform=self.info.transAxes,
                        facecolor="#132B49",
                        edgecolor="none",
                    )
                )
                bar = Rectangle(
                    (xx + 0.020, yy - 0.101),
                    0.0,
                    0.011,
                    transform=self.info.transAxes,
                    facecolor=bar_colors[i],
                    edgecolor="none",
                )
                self.info.add_patch(bar)
                self.telemetry_bars[i] = bar
        self.status = self.fig.text(0.654, 0.145, "", color="#A8C4DD", fontsize=8.6, va="bottom")
        self.environment_text = self.fig.text(
            0.654, 0.171, "", color="#86A4C2", fontsize=8.4, va="bottom"
        )
        self.event = self.fig.text(
            0.045,
            0.163,
            "",
            color="#7AE5D2",
            fontsize=10.2,
            weight="bold",
            va="center",
            linespacing=1.50,
        )
        self._draw_overview()
        self.timeline = []
        timeline_labels = ("발사", "단 분리", "귀환", "재진입", "활강", "호버슬램", "직립 착륙")
        for i, label in enumerate(timeline_labels):
            xx = 0.04 + i * 0.1325
            card = FancyBboxPatch(
                (xx, 0.068),
                0.1235,
                0.056,
                transform=self.fig.transFigure,
                boxstyle="round,pad=0.002,rounding_size=0.008",
                facecolor="#081A32",
                edgecolor="#1D3E63",
                linewidth=0.85,
            )
            self.fig.add_artist(card)
            accent = Rectangle(
                (xx + 0.003, 0.120),
                0.1175,
                0.005,
                transform=self.fig.transFigure,
                facecolor="#1D3E63",
                edgecolor="none",
            )
            self.fig.add_artist(accent)
            text = self.fig.text(
                xx + 0.010,
                0.092,
                f"{i + 1:02}  {label}",
                color="#7E9CB8",
                fontsize=9.3,
                weight="bold",
            )
            self.timeline.append((card, accent, text))
        self.update(self.replay.times[0])
        from matplotlib.text import Text

        from powered_landing_guidance.runtime_support import replay_font_family

        family = replay_font_family()
        for text_artist in self.fig.findobj(match=Text):
            text_artist.set_fontfamily(family)

    def _draw_ground_facilities(self) -> None:
        """Illustrative background facilities; not simulated collision obstacles."""
        for sign in (-1, 1):
            x = sign * 23
            self.camera.add_patch(
                Rectangle(
                    (x - 1.1, 0),
                    2.2,
                    23,
                    facecolor="#738992",
                    edgecolor="#b1c0c5",
                    linewidth=0.6,
                    zorder=2.5,
                )
            )
            for z in np.arange(1, 23, 3):
                self.camera.plot(
                    (x - 1, x + 1), (z, z + 2.8), color="#b8c5c6", linewidth=0.55, zorder=2.6
                )
                self.camera.plot(
                    (x + 1, x - 1), (z, z + 2.8), color="#b8c5c6", linewidth=0.55, zorder=2.6
                )
            self.camera.add_patch(
                Rectangle(
                    (x - 1.65, 22.7), 3.3, 0.35, facecolor="#8fa6ae", edgecolor="none", zorder=2.7
                )
            )
            self.camera.add_patch(
                Ellipse((x, 23.35), 0.75, 0.75, facecolor="#FF6B6B", edgecolor="none", zorder=2.8)
            )
        self.camera.add_patch(
            Rectangle(
                (29, 0),
                14,
                4.5,
                facecolor="#768f98",
                edgecolor="#b1c3cb",
                linewidth=0.5,
                zorder=2.5,
            )
        )
        self.camera.add_patch(
            Rectangle(
                (-42, 0),
                10,
                3.5,
                facecolor="#617d83",
                edgecolor="#a3b8bd",
                linewidth=0.5,
                zorder=2.5,
            )
        )
        for x in (31, 34, 37, 40):
            self.camera.add_patch(
                Rectangle(
                    (x, 1.7),
                    1.4,
                    1.5,
                    facecolor="#35535f",
                    edgecolor="#b8c8c8",
                    linewidth=0.3,
                    zorder=2.6,
                )
            )

    def _draw_overview(self) -> None:
        bases = self.replay.bases
        xlow = min(0, bases[:, 0].min()) / 1000 - 0.5
        xhigh = max(0, bases[:, 0].max()) / 1000 + 0.5
        zhigh = max(1, bases[:, 1].max() / 1000 + 2)
        self.upper_path_times = np.empty(0)
        self.upper_path = np.empty((0, 2))
        departure = self.replay.upper_departure
        if departure is not None:
            # Display departure through 100 km, not an invented 700 km orbit.
            times = np.linspace(
                departure.separation_s,
                min(self.replay.times[-1], departure.separation_s + 180),
                361,
            )
            positions = np.asarray([departure.sample(t)[0] for t in times])
            crossed = np.flatnonzero(positions[:, 1] >= 100_000)
            if len(crossed):
                end = int(crossed[0])
                if end > 0:
                    weight = (100_000 - positions[end - 1, 1]) / (
                        positions[end, 1] - positions[end - 1, 1]
                    )
                    times[end] = times[end - 1] + weight * (times[end] - times[end - 1])
                    positions[end] = (1 - weight) * positions[end - 1] + weight * positions[end]
                times, positions = times[: end + 1], positions[: end + 1]
            self.upper_path_times, self.upper_path = times, positions
            xlow = min(xlow, positions[:, 0].min() / 1000 - 0.5)
            xhigh = max(xhigh, positions[:, 0].max() / 1000 + 0.5)
            zhigh = max(zhigh, positions[:, 1].max() / 1000 + 2)
        self.overview.imshow(
            _recovery_sky(np.linspace(0, zhigh * 1000, 64))[:, None, :],
            extent=(xlow, xhigh, 0, zhigh),
            origin="lower",
            aspect="auto",
            alpha=0.72,
        )
        self.overview.plot(
            bases[:, 0] / 1000,
            bases[:, 1] / 1000,
            color="#6b8a9d",
            alpha=0.4,
            linewidth=1,
        )
        self.overview.scatter([0], [0], marker="s", color="#56CCF2", s=26, zorder=5)
        (self.history,) = self.overview.plot(
            [], [], color="#56CCF2", linewidth=1.7, zorder=4, label="1단 계산"
        )
        (self.marker_halo,) = self.overview.plot(
            [], [], "o", color="#E5C573", alpha=0.35, markersize=10, zorder=5.5
        )
        (self.marker,) = self.overview.plot([], [], "o", color="#F7DC88", markersize=5.2, zorder=6)
        (self.upper_history,) = self.overview.plot(
            [], [], "--", color="#E5C573", linewidth=1.5, zorder=4, label="2단 연출"
        )
        (self.upper_marker,) = self.overview.plot(
            [], [], "^", color="#F7DC88", markersize=5.2, zorder=6
        )
        self.upper_marker.set_visible(False)
        self.overview_legend = None
        if departure is not None:
            legend = self.overview.legend(loc="upper left", fontsize=7.2, frameon=False, ncol=2)
            self.overview_legend = legend
            for text in legend.get_texts():
                text.set_color("#D3E4F2")
        self.overview.set(
            xlim=(xlow, xhigh),
            ylim=(-1, zhigh),
            title="비행 경로 · 궤도 전개",
        )
        self.overview.set_xlabel("발사장 거리 (km)", color="#9BB5CC", fontsize=8)
        self.overview.set_ylabel("고도 (km)", color="#9BB5CC", fontsize=8)
        self.overview.title.set_color("#E5C573")
        self.overview.title.set_fontsize(10)
        self.overview.title.set_weight("bold")
        self.overview.tick_params(colors="#9BB5CC", labelsize=7)
        self.overview.grid(alpha=0.14, color="#A8D0F0", linewidth=0.5)
        for spine in self.overview.spines.values():
            spine.set_color("#244975")

    def _update_shell(self, parts, base, state, attached) -> None:
        from powered_landing_guidance.recovery_physics import rotation

        transform = rotation(state[4])
        radius = self.replay.settings.geometry.body_radius_m
        length = self.replay.settings.geometry.body_length_m
        bands = ((-1, -0.7), (-0.65, -0.3), (0.6, 1))
        for patch, (low, high) in zip(self.shell_shading, bands, strict=True):
            local = np.asarray(
                (
                    (low * radius, 1.1),
                    (high * radius, 1.1),
                    (high * radius, length - 0.7),
                    (low * radius, length - 0.7),
                )
            )
            patch.set_xy(local @ transform.T + base)
        skirt = np.asarray(((-radius, 0), (radius, 0), (radius, 1.1), (-radius, 1.1)))
        self.engine_skirt.set_xy(skirt @ transform.T + base)
        seam_segments = [
            np.asarray(((-radius, station * length), (radius, station * length))) @ transform.T
            + base
            for station in (0.15, 0.37, 0.65, 0.88, 0.965)
        ]
        self.seams.set_segments(seam_segments)
        grid = []
        for side in ("left", "right"):
            vertices = parts[f"fin_{side}"]
            for weight in (0.2, 0.4, 0.6, 0.8):
                grid.append(
                    (
                        (1 - weight) * vertices[0] + weight * vertices[3],
                        (1 - weight) * vertices[1] + weight * vertices[2],
                    )
                )
                grid.append(
                    (
                        (1 - weight) * vertices[0] + weight * vertices[1],
                        (1 - weight) * vertices[3] + weight * vertices[2],
                    )
                )
        self.fin_mesh.set_segments(grid)
        self.fin_mesh.set_visible(state[10] > 0.01)
        label = np.asarray((0, 0.55 * length)) @ transform.T + base
        self.body_label.set_position(label)
        self.body_label.set_rotation(90 - np.rad2deg(state[4]))
        for support in self.launch_supports:
            support.set_visible(attached)

    def update(self, time_s: float) -> None:
        from powered_landing_guidance.recovery_physics import atmosphere, gravity, rotation

        shown_time = float(np.clip(time_s, self.replay.times[0], self.replay.times[-1]))
        parts, nozzle, sample = _nominal_flight_geometry(self.replay, shown_time)
        state, base, mass, control, phase, index = sample
        attached = shown_time < self.replay.separation_s
        contacted = shown_time >= self.replay.contact_s
        departure = self.replay.upper_departure
        upper_present = attached or departure is not None
        for name, vertices in parts.items():
            self.parts[name].set_xy(vertices)
            visible = True
            if name in {"upper_stage", "upper_highlight", "upper_band_detail"}:
                visible = upper_present
            elif name == "upper_nozzle":
                visible = departure is not None and not attached
            elif name == "separation_band":
                visible = attached
            self.parts[name].set_visible(visible)
        self._update_shell(parts, base, state, attached)
        x, z, vx, vz, theta, omega = state[:6]
        rocket_length = self.replay.settings.geometry.body_length_m
        if attached:
            rocket_length += self.replay.upper_length_m
        elif np.isfinite(self.replay.separation_s):
            # Stateless zoom easing: arbitrary preview times produce the same shot.
            rocket_length += self.replay.upper_length_m * np.exp(
                -(shown_time - self.replay.separation_s) / 1.5
            )
        height = float(np.clip(rocket_length * 2.05 + max(z, 0) * 0.0007, 43, 105))
        center = base + 0.48 * rocket_length * np.asarray((np.sin(theta), np.cos(theta)))
        if departure is not None and not attached:
            # Show both vehicles briefly, then ease back to first-stage tracking.
            elapsed = shown_time - self.replay.separation_s
            blend = float(1 - np.clip((elapsed - 5) / 5, 0, 1))
            blend = blend**2 * (3 - 2 * blend)
            vertices = np.vstack((parts["body"], parts["upper_stage"]))
            low, high = vertices.min(axis=0), vertices.max(axis=0)
            joint_center = (low + high) / 2
            aspect = self.camera_aspect
            joint_height = max(height, 1.6 * (high[1] - low[1]), 1.6 * (high[0] - low[0]) / aspect)
            center += blend * (joint_center - center)
            height += blend * (joint_height - height)
        bottom = max(-4.5, center[1] - 0.45 * height)
        view_width = height * self.camera_aspect
        left, right = center[0] - view_width / 2, center[0] + view_width / 2
        self.camera.set(xlim=(left, right), ylim=(bottom, bottom + height))
        sky = np.repeat(
            _recovery_sky(np.linspace(max(0, bottom), max(0, bottom + height), 96))[:, None, :],
            96,
            axis=1,
        )
        light = np.linspace(0.025, -0.008, 96)[None, :, None]
        sky = np.clip(sky + light * np.exp(-max(0, z) / 18000), 0, 1)
        self.sky.set_data(np.round(sky * 255).astype(np.uint8))
        self.sky.set_extent((left, right, bottom, bottom + height))
        self.earth_limb.set_alpha(float(np.clip((z - 22_000) / 36_000, 0, 0.24)))
        self.stars.set_alpha(float(np.clip((z - 16_000) / 36_000, 0, 0.72)))
        for cloud in self.clouds:
            cloud.set_alpha(float(0.16 * np.exp(-max(0, z) / 2200)))
        for hill, peak in self.hills:
            points = np.asarray(
                (
                    (left, 0),
                    (left, peak * 0.5),
                    (left + view_width * 0.24, peak),
                    (left + view_width * 0.48, peak * 0.6),
                    (left + view_width * 0.75, peak * 0.9),
                    (right, peak * 0.35),
                    (right, 0),
                )
            )
            hill.set_xy(points)
            hill.set_visible(bottom < peak)
        self.terrain.set_xy(((left, bottom), (right, bottom), (right, 0), (left, 0)))
        self.soil.set_xy(((left, bottom), (right, bottom), (right, -0.9), (left, -0.9)))
        self.terrain.set_visible(bottom < 0)
        self.soil.set_visible(bottom < -0.9)
        history = self.replay.bases[max(0, index - 55) : index + 1]
        self.trail.set_data(history[:, 0], history[:, 1])
        self.history.set_data(
            self.replay.bases[: index + 1, 0] / 1000, self.replay.bases[: index + 1, 1] / 1000
        )
        self.marker.set_data([base[0] / 1000], [base[1] / 1000])
        self.marker_halo.set_data([base[0] / 1000], [base[1] / 1000])
        upper_sample = departure.sample(shown_time) if departure and not attached else None
        self.upper_history.set_visible(upper_sample is not None)
        self.upper_marker.set_visible(False)
        self.upper_tag.set_visible(False)
        self.upper_history.set_data([], [])
        self.upper_marker.set_data([], [])
        for plume in self.upper_plumes:
            plume.set_visible(False)
        if upper_sample is not None:
            position, _, upper_theta, _, ignition = upper_sample
            through = int(np.searchsorted(self.upper_path_times, shown_time, side="right"))
            path = self.upper_path[:through]
            in_domain = shown_time <= self.upper_path_times[-1]
            if in_domain:
                path = np.vstack((path, position))
                self.upper_marker.set_data([position[0] / 1000], [position[1] / 1000])
                self.upper_marker.set_visible(True)
            self.upper_history.set_data(path[:, 0] / 1000, path[:, 1] / 1000)
            upper_base, _ = departure.base(shown_time)
            exhaust = -np.asarray((np.sin(upper_theta), np.cos(upper_theta)))
            side = np.asarray((np.cos(upper_theta), -np.sin(upper_theta)))
            flicker = 0.96 + 0.04 * np.sin(shown_time * 43)
            for plume, scale in zip(self.upper_plumes, (1.35, 0.95, 0.52), strict=True):
                width = 0.72 * np.sqrt(ignition) * scale
                length = 13.5 * np.sqrt(ignition) * scale * flicker
                local = np.asarray(
                    (
                        (-width, 0),
                        (-width * 1.65, length * 0.24),
                        (0, length),
                        (width * 1.65, length * 0.24),
                        (width, 0),
                    )
                )
                plume.set_xy(upper_base + local[:, :1] * side + local[:, 1:] * exhaust)
                plume.set_visible(ignition > 0.01 and in_domain)
            elapsed_sep = shown_time - self.replay.separation_s
            tag_pos = position + np.asarray((3.2, 2.2))
            in_view = (
                left + 2.0 < tag_pos[0] < right - 14.0
                and bottom + 3.0 < tag_pos[1] < bottom + height - 3.0
            )
            if 0.3 <= elapsed_sep <= 12.5 and in_view:
                self.upper_tag.set_position(tag_pos)
                self.upper_tag.set_visible(True)
        _, pressure, density = atmosphere(z, self.replay.settings.atmosphere)
        thrust = (
            control[0]
            * (
                self.replay.design["sea_level_thrust_n"]
                + self.replay.design["nozzle_exit_area_m2"] * (101325 - pressure)
            )
            * self.replay.flight.get("disturbances", {}).get("thrust_scale", 1.0)
        )
        self.display_engine_on = bool(thrust > 1e-6 and not contacted)
        self.displayed_thrust_n = float(thrust if not contacted else 0)
        ratio = float(np.clip(control[0], 0, 1)) if self.display_engine_on else 0
        exhaust = -np.asarray((np.sin(theta + control[1]), np.cos(theta + control[1])))
        side = np.asarray((np.cos(theta + control[1]), -np.sin(theta + control[1])))
        radius = self.replay.settings.geometry.nozzle_radius_m
        active_eng = self.replay.active_engines[index]
        eng_scale = 1.24 if active_eng >= 7 else 0.94 if active_eng >= 3 else 0.64
        vac_bloom = 1.0 + 0.48 * float(np.clip(z / 42_000, 0.0, 1.0))
        flicker = 0.95 + 0.05 * np.sin(shown_time * 57)
        length = (1.1 + 24.5 * radius) * np.sqrt(ratio) * flicker * (0.85 + 0.15 * eng_scale)
        for plume, scale in zip(self.plumes, (1.22, 1.0, 0.72, 0.42), strict=True):
            plume_width = radius * (0.38 + 0.85 * np.sqrt(ratio)) * scale * eng_scale * vac_bloom
            local = np.asarray(
                (
                    (-plume_width, 0),
                    (-plume_width * 1.48, length * 0.20 * scale),
                    (-plume_width * 0.68, length * 0.56 * scale),
                    (0, length * scale),
                    (plume_width * 0.68, length * 0.56 * scale),
                    (plume_width * 1.48, length * 0.20 * scale),
                    (plume_width, 0),
                )
            )
            vertices = _clip_plume_at_ground(
                nozzle + local[:, :1] * side + local[:, 1:] * exhaust, 0
            )
            plume.set_visible(self.display_engine_on and len(vertices) >= 3)
            if len(vertices) >= 3:
                plume.set_xy(vertices)
        for idx_d, diamond in enumerate(self.mach_diamonds):
            if self.display_engine_on and ratio > 0.15:
                d_center = length * (0.14 + idx_d * 0.18)
                d_half_w = radius * 0.36 * eng_scale * (1.0 - idx_d * 0.18) * np.sqrt(ratio)
                d_half_l = length * 0.065 * (1.0 - idx_d * 0.15)
                local_d = np.asarray(
                    (
                        (0.0, d_center - d_half_l),
                        (d_half_w, d_center),
                        (0.0, d_center + d_half_l),
                        (-d_half_w, d_center),
                    )
                )
                d_verts = _clip_plume_at_ground(
                    nozzle + local_d[:, :1] * side + local_d[:, 1:] * exhaust, 0
                )
                diamond.set_visible(len(d_verts) >= 3)
                if len(d_verts) >= 3:
                    diamond.set_xy(d_verts)
            else:
                diamond.set_visible(False)
        q = np.interp(shown_time, self.replay.times, self.q)
        transform = rotation(theta)
        reentry_intensity = (
            float(np.clip((q - 11_000.0) / 22_000.0, 0.0, 1.0))
            if (np.isfinite(q) and not attached and not contacted and vz < -120.0)
            else 0.0
        )
        if reentry_intensity > 0.04:
            br = self.replay.settings.geometry.body_radius_m
            angles = np.linspace(np.pi * 0.08, np.pi * 0.92, 11)
            for (glow, max_a), w_sc, d_sc, tail_y in zip(
                self.reentry_glows,
                (1.52, 1.32, 1.12),
                (1.55, 1.05, 0.62),
                (3.6, 2.2, 1.1),
                strict=True,
            ):
                arc_x = -br * w_sc * np.cos(angles)
                arc_y = -d_sc * np.sin(angles)
                local_re = np.vstack(
                    (
                        np.asarray(((-br * w_sc * 0.92, tail_y),)),
                        np.column_stack((arc_x, arc_y)),
                        np.asarray(((br * w_sc * 0.92, tail_y),)),
                    )
                )
                glow.set_xy(local_re @ transform.T + base)
                glow.set_alpha(reentry_intensity * max_a)
                glow.set_visible(True)
        else:
            for glow, _ in self.reentry_glows:
                glow.set_visible(False)
        sign = float(np.sign(control[2]))
        rcs_ratio = float(
            np.clip(abs(control[2]) / self.replay.design["rcs_torque_limit_nm"], 0, 1)
        )
        self.rcs_torque_kn_m = float(control[2] / 1000)
        self.rcs_jet_length_m = float(2 * np.sqrt(rcs_ratio))
        for jet, station, direction in zip(self.rcs_jets, (18.5, 1.8), (-sign, sign), strict=True):
            local = np.asarray(
                (
                    (direction * self.replay.settings.geometry.body_radius_m, station - 0.1),
                    (direction * self.replay.settings.geometry.body_radius_m, station + 0.1),
                    (
                        direction
                        * (self.replay.settings.geometry.body_radius_m + self.rcs_jet_length_m),
                        station,
                    ),
                )
            )
            jet.set_xy(local @ transform.T + base)
            jet.set_visible(rcs_ratio > 0.02 and not contacted)
        near_ground = max(0, 1 - max(0, nozzle[1]) / 18) * np.sqrt(ratio)
        self.heat_glow.center = (nozzle[0], 0.10)
        self.heat_glow.width = 7 + 12 * near_ground
        self.heat_glow.set_alpha(0.32 * near_ground)
        for puff, i in self.dust:
            sign = -1 if i < 4 else 1
            distance = (i % 4 + 1) * (1.8 + 2.5 * near_ground)
            puff.center = (nozzle[0] + sign * distance, 0.5 + 0.28 * (i % 4))
            puff.width, puff.height = 3 + 2.6 * near_ground, 1.1 + near_ground
            puff.set_alpha(near_ground * 0.23 * (1 - (i % 4) * 0.12))
        scale = 10 if height < 75 else 20
        sx, sy = right - view_width * 0.05 - scale, bottom + height * 0.06
        self.scale_line.set_data((sx, sx + scale), (sy, sy))
        self.scale_label.set_position((sx + scale / 2, sy + height * 0.012))
        self.scale_label.set_text(f"{scale} m")
        phase_names = {
            "ascent": "발사 · 상승",
            "flip": "분리 · RCS 자세 전환",
            "boostback": "귀환 · 역추진",
            "coast": "관성 비행 · 공력핀 보조",
            "entry": "대기 진입 · 감속",
            "entry_burn": "단축 재진입 연소 · MECO-2",
            "aero_glide": "무동력 격자핀 활강 (엔진 OFF)",
            "landing": "착륙 · 주엔진 제동",
            "landing_burn": "호버슬램 착륙 연소 (3→1)",
            "alignment": "수평 · 자세 정렬",
            "approach": "최종 접근",
            "touchdown": "첫 접지 · 비행 기록 종료",
            "touchdown_settle": "엔진 OFF · 접지 후 직립 안정화",
        }
        self.phase.set_text(phase_names.get(phase, phase))
        self.clock.set_text(f"T+ {shown_time:07.2f} s")
        self.camera_note.set_text(
            "엔진 OFF · 접지 이후 고정 화면" if contacted else f"1단 하단 고도 {base[1]:,.1f} m"
        )
        total_mass = mass + (self.replay.design["upper_mass_kg"] if attached else 0)
        q_text = f"{q / 1000:.3f} kPa" if np.isfinite(q) else "-"
        values = (
            f"{z / 1000:,.3f} km",
            f"{x:+,.2f} m",
            f"{vz:+,.3f} m/s",
            f"{vx:+,.3f} m/s",
            f"{np.rad2deg(theta):+,.3f}°",
            f"{np.rad2deg(omega):+,.3f}°/s",
            f"{state[6] / 1000:,.3f} t",
            f"{state[7]:,.2f} kg",
            f"{total_mass / 1000:,.3f} t",
            q_text,
            "OFF / 0 kN" if contacted else f"{thrust / 1000:,.1f} kN ({active_eng}기)",
            f"{state[12] * 100:.2f} %",
            f"{state[10] * 100:.0f} %",
            f"{state[8] * 100:.0f} %",
        )
        for artist, value in zip(self.telemetry, values, strict=True):
            artist.set_text(value)
        main_cap = max(float(self.replay.design.get("main_propellant_kg", 69_500.0)), 1.0)
        rcs_cap = max(float(self.replay.design.get("rcs_propellant_kg", 300.0)), 1.0)
        q_frac = float(np.clip((q / 42_000.0) if np.isfinite(q) else 0.0, 0.0, 1.0))
        self.telemetry_bars[6].set_width(0.420 * float(np.clip(state[6] / main_cap, 0.0, 1.0)))
        self.telemetry_bars[7].set_width(0.420 * float(np.clip(state[7] / rcs_cap, 0.0, 1.0)))
        self.telemetry_bars[9].set_width(0.420 * q_frac)
        self.telemetry_bars[9].set_facecolor(
            "#EF4444" if q_frac >= 0.78 else "#E5C573" if q_frac >= 0.50 else "#56CCF2"
        )
        self.telemetry_bars[11].set_width(
            0.0 if contacted else 0.420 * float(np.clip(state[12], 0.0, 1.0))
        )
        self.telemetry_labels[10].set_text("엔진 · 접지 후 연출" if contacted else "주 엔진 추력")
        self.telemetry_labels[11].set_text("접지 순간 스로틀" if contacted else "실제 스로틀")
        self.environment_text.set_text(
            f"중력 {gravity(z):.3f} m/s² · 기압 {pressure / 1000:.3f} kPa · "
            f"밀도 {density:.4f} kg/m³"
        )
        self.status.set_text(
            "접지 순간 기록값"
            if contacted
            else f"짐벌 {np.rad2deg(control[1]):+.2f}° · RCS 토크 {self.rcs_torque_kn_m:+.1f} kN·m"
        )
        if contacted:
            speed = max(abs(v[1]) for v in self.replay.contact_record["foot_velocities_m_s"])
            passed = self.replay.flight.get("simulated_acceptance_passed") is True
            heading = "접지 조건 통과" if passed else f"판정: {self.replay.flight.get('outcome')}"
            is_contact_dyn = self.replay.flight.get("landing_assessment") == "contact_dynamics"
            orb = self.replay.flight.get("upper_stage_orbit") or {}
            orb_note = (
                f"2단 {orb['final_apogee_km']:.1f} km 원궤도 투입 ({orb['payload_mass_kg']:.0f} kg)"
                if orb.get("orbit_inserted") and orb.get("final_apogee_km") is not None
                else "2단 궤도 판정 미충족"
                if orb
                else "1단 회수 기록"
            )
            if is_contact_dyn:
                pc = self.replay.flight.get("post_contact") or {}
                stable_txt = "직립 안정 통과" if pc.get("standing_stable") else "직립 안정 미충족"
                self.event.set_text(
                    f"{heading} · 발 수직속력 {speed:.3f} m/s\n"
                    f"엔진 OFF · 5초 착륙다리 동역학 ({stable_txt})"
                )
                self.footer.set_text(
                    f"DM100 Reusable RTLS · 접지 후 5초 물리 안정화 평가 · {orb_note}"
                )
            else:
                self.event.set_text(
                    f"{heading} · 발 수직속력 {speed:.3f} m/s\n"
                    "엔진 OFF · 재생 연출 / 접지 이후 거동 미평가"
                )
                self.footer.set_text(
                    f"접지 후 화면 고정·엔진 OFF는 연출 · 기립 미평가 · {orb_note}"
                )
        elif attached:
            self.event.set_text("1·2단 결합 (7엔진 상승)")
        else:
            orb = self.replay.flight.get("upper_stage_orbit") or {}
            orb_line = (
                f"2단 {orb['final_apogee_km']:.1f} km 원궤도 투입 ({orb['payload_mass_kg']:.0f} kg)"
                if orb.get("orbit_inserted") and orb.get("final_apogee_km") is not None
                else "2단 분리 상승(궤도 별도 적분)"
                if departure is not None
                else "2단 초기 기록 없음 · 1단만 재생"
            )
            self.event.set_text(f"분리 T+{self.replay.separation_s:.2f} s · 1단 귀환\n{orb_line}")
        stage = {
            "ascent": 0,
            "flip": 1,
            "boostback": 2,
            "coast": 2,
            "entry": 3,
            "entry_burn": 3,
            "aero_glide": 4,
            "landing": 5,
            "landing_burn": 5,
            "alignment": 5,
            "approach": 5,
            "touchdown": 6,
            "touchdown_settle": 6,
        }.get(phase, 0)
        for i, (card, accent, text) in enumerate(self.timeline):
            card.set_facecolor("#0F3562" if i == stage else "#0B2445" if i < stage else "#07162B")
            card.set_edgecolor("#D4AF5A" if i == stage else "#285385" if i < stage else "#183354")
            accent.set_facecolor("#D4AF5A" if i == stage else "#56CCF2" if i < stage else "#183354")
            text.set_color("#F7F5F0" if i == stage else "#C3D8EC" if i < stage else "#6A86A3")


class _NominalFlightFrameRenderer:
    """Cache the fixed dashboard while redrawing the moving camera and telemetry."""

    def __init__(self, scene: _NominalFlightAnimationScene) -> None:
        self.scene = scene
        self.artists = [
            scene.camera,
            scene.history,
            scene.upper_history,
            *([scene.overview_legend] if scene.overview_legend is not None else []),
            scene.marker_halo,
            scene.marker,
            scene.upper_marker,
            *scene.telemetry,
            *scene.telemetry_bars.values(),
            scene.telemetry_labels[10],
            scene.telemetry_labels[11],
            scene.status,
            scene.environment_text,
            scene.event,
            scene.clock,
            scene.footer,
            *(artist for triple in scene.timeline for artist in triple),
        ]
        self.previous_flags = [artist.get_animated() for artist in self.artists]
        for artist in self.artists:
            artist.set_animated(True)
        scene.fig.canvas.draw()
        self.background = scene.fig.canvas.copy_from_bbox(scene.fig.bbox)

    def frame(self) -> memoryview:
        canvas = self.scene.fig.canvas
        canvas.restore_region(self.background)
        for artist in self.artists:
            if artist.get_visible():
                self.scene.fig.draw_artist(artist)
        return canvas.buffer_rgba()

    def close(self) -> None:
        for artist, flag in zip(self.artists, self.previous_flags, strict=True):
            artist.set_animated(flag)


def save_nominal_flight_replay(
    flight: dict[str, Any],
    path: str | Path,
    *,
    fps: int = 24,
    preview: str | Path | None = None,
    width: int = 1920,
    end_hold_s: float = 4.0,
    start_s: float | None = None,
    end_s: float | None = None,
) -> Path:
    """Render all or a selected interval of a stored flight at normal speed."""
    if type(fps) is not int or not 1 <= fps <= 60:
        raise ValueError("full-flight replay fps must be an integer in [1, 60]")
    if type(width) is not int or not 640 <= width <= 3840 or width % 32:
        raise ValueError("replay width must be a multiple of 32 in [640, 3840]")
    if (
        isinstance(end_hold_s, bool)
        or not isinstance(end_hold_s, int | float)
        or not np.isfinite(end_hold_s)
        or not 0 <= end_hold_s <= 30
    ):
        raise ValueError("replay end_hold_s must be a finite number in [0, 30]")
    for value in (start_s, end_s):
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not np.isfinite(value)
            or value < 0
        ):
            raise ValueError("replay interval bounds must be finite nonnegative seconds")
    destination = Path(path)
    if destination.suffix.lower() != ".mp4":
        raise ValueError("full-flight replay output must end in .mp4")
    if destination.exists():
        raise FileExistsError(destination)
    poster = Path(preview) if preview else None
    if poster and (poster.exists() or poster.resolve() == destination.resolve()):
        raise FileExistsError(poster)
    if poster and poster.suffix.lower() != ".png":
        raise ValueError("full-flight replay preview must end in .png")
    if not _ffmpeg_available():
        raise RuntimeError("full-flight MP4 requires FFmpeg or the video extra")
    from powered_landing_guidance.runtime_support import replay_font_family

    family = replay_font_family()
    temporary = None
    with matplotlib.rc_context({"font.family": family, "axes.unicode_minus": False}):
        scene = _NominalFlightAnimationScene(flight, width=width)
        try:
            start = scene.replay.times[0] if start_s is None else float(start_s)
            end = scene.replay.times[-1] if end_s is None else float(end_s)
            if not scene.replay.times[0] <= start < end <= scene.replay.times[-1]:
                raise ValueError(
                    "replay interval must lie inside the recorded flight with start < end"
                )
            frames = _nominal_flight_frame_times(np.asarray((start, end)), fps)
            hold_frames = int(np.ceil(end_hold_s * fps))
            if hold_frames:
                frames = np.r_[frames, np.full(hold_frames, end)]
            destination.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(
                prefix=f".{destination.stem}-", suffix=".mp4", dir=destination.parent, delete=False
            ) as stream:
                temporary = Path(stream.name)
            writer = FFMpegWriter(
                fps=fps,
                codec="libx264",
                bitrate=-1,
                metadata={"title": "DM100 - 2D first-stage recovery", "artist": "DM100"},
                extra_args=[
                    "-crf",
                    "18",
                    "-preset",
                    "fast",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                ],
            )
            with writer.saving(scene.fig, str(temporary), dpi=width / 16):
                # Matplotlib's writer accepts raw RGBA frames. Avoid repeated savefig
                # layout work, retaining its normal process/error handling and fallback.
                process = getattr(writer, "_proc", None)
                fast_frames = None
                if (
                    process is not None
                    and getattr(process, "stdin", None) is not None
                    and getattr(writer, "frame_format", None) == "rgba"
                    and scene.fig.canvas.get_width_height() == writer.frame_size
                    and scene.fig.canvas.supports_blit
                ):
                    fast_frames = _NominalFlightFrameRenderer(scene)
                try:
                    for index, time_s in enumerate(frames):
                        scene.update(float(time_s))
                        if fast_frames is None:
                            writer.grab_frame(facecolor=scene.fig.get_facecolor())
                        else:
                            process.stdin.write(fast_frames.frame())
                        if index % (fps * 30) == 0:
                            print(
                                f"Rendering saved flight: {time_s:.1f} / {frames[-1]:.1f} s",
                                flush=True,
                            )
                finally:
                    if fast_frames is not None:
                        fast_frames.close()
            with temporary.open("rb") as source, destination.open("xb") as target:
                shutil.copyfileobj(source, target)
            if poster:
                poster.parent.mkdir(parents=True, exist_ok=True)
                scene.fig.savefig(poster, dpi=width / 16, facecolor=scene.fig.get_facecolor())
        finally:
            plt.close(scene.fig)
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return destination


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True, help="저장한 100톤 비행 결과 JSON")
    parser.add_argument("--animation", type=Path, required=True, help="새 MP4 출력 경로")
    parser.add_argument("--preview", type=Path, help="선택적인 PNG 미리보기")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--width", type=int, default=1920, help="640-3840 범위의 32배수")
    parser.add_argument("--end-hold", type=float, default=4.0, help="끝화면 유지 시간(0-30초)")
    parser.add_argument("--start-time", type=float, help="Optional clip start in recorded seconds")
    parser.add_argument("--end-time", type=float, help="Optional clip end in recorded seconds")
    args = parser.parse_args(argv)
    try:
        record = json.loads(args.replay.read_text(encoding="utf-8"))
        path = save_nominal_flight_replay(
            record,
            args.animation,
            fps=args.fps,
            preview=args.preview,
            width=args.width,
            end_hold_s=args.end_hold,
            start_s=args.start_time,
            end_s=args.end_time,
        )
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
        parser.error(str(error))
    print(f"애니메이션 저장: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
