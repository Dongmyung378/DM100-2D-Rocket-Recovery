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
    levels = (0.0, 2500.0, 6500.0, 11_000.0, 30_000.0, 80_000.0)
    colors = np.asarray(
        (
            (0.78, 0.91, 0.98),
            (0.48, 0.75, 0.92),
            (0.24, 0.49, 0.73),
            (0.10, 0.25, 0.45),
            (0.025, 0.07, 0.15),
            (0.005, 0.01, 0.025),
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
            or flight.get("landing_assessment") != "kinematic_touchdown"
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
        self.fig = plt.figure(figsize=(16, 9), dpi=width / 16, facecolor="#07131e")
        self.camera = self.fig.add_axes((0.04, 0.205, 0.565, 0.655))
        self.camera_aspect = (16 * 0.565) / (9 * 0.655)
        self.overview = self.fig.add_axes((0.660, 0.700, 0.295, 0.155), facecolor="#0e2535")
        self.info = self.fig.add_axes((0.645, 0.205, 0.32, 0.43))
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
        self.title = self.fig.text(
            0.04, 0.944, "DM100", color="#edf6fb", fontsize=25, weight="bold"
        )
        self.subtitle = self.fig.text(
            0.205, 0.948, "2D 로켓 · 발사 및 1단 회수", color="#bad0dc", fontsize=15
        )
        self.mission_label = self.fig.text(
            0.956,
            0.95,
            "1단 회수" if recovery_only else "발사장 복귀",
            ha="right",
            color="#63ddce",
            fontsize=12,
        )
        conditions = flight.get("conditions", {})
        calm = not conditions.get("gust_amplitude_m_s", 0) and not any(
            speed
            for key in ("wind_profile", "vertical_wind_profile")
            for _, speed in conditions.get(key, ())
        )
        self.record_note = self.fig.text(
            0.04,
            0.902,
            "무풍 기준" if calm else "설정 외란",
            color="#8da6b6",
            fontsize=10,
        )
        self.clock = self.fig.text(
            0.956, 0.902, "", ha="right", color="#e6f5fc", fontsize=13, family="DejaVu Sans"
        )
        self.footer = self.fig.text(
            0.04,
            0.035,
            "2D 모델 · 2단 상승은 연출(궤도·연료 미계산)",
            color="#8da6b6",
            fontsize=9,
        )
        self.fig.add_artist(
            Rectangle(
                (0.04, 0.883),
                0.92,
                0.0015,
                transform=self.fig.transFigure,
                facecolor="#274454",
                edgecolor="none",
            )
        )
        self.camera.set_axis_off()
        self.camera.set_aspect("equal", adjustable="box")
        self.sky = self.camera.imshow(
            np.zeros((96, 96, 3), dtype=np.uint8), origin="lower", aspect="auto", zorder=0
        )
        scenery_rng = np.random.default_rng(378)
        self.stars = self.camera.scatter(
            scenery_rng.uniform(0.05, 0.95, 45),
            scenery_rng.uniform(0.1, 0.95, 45),
            s=scenery_rng.uniform(0.3, 2.2, 45),
            transform=self.camera.transAxes,
            color="#dceafb",
            alpha=0,
            zorder=1,
        )
        self.clouds = []
        for cx, cy, size in ((0.18, 0.68, 0.18), (0.80, 0.47, 0.24), (0.52, 0.86, 0.13)):
            for dx, dy, factor in ((-0.3, 0, 0.65), (0, 0.04, 1), (0.3, -0.01, 0.7)):
                cloud = Ellipse(
                    (cx + dx * size, cy + dy * size),
                    size * factor,
                    size * 0.19 * factor,
                    transform=self.camera.transAxes,
                    facecolor="#ffffff",
                    edgecolor="none",
                    alpha=0.18,
                    zorder=1,
                )
                self.camera.add_patch(cloud)
                self.clouds.append(cloud)
        self.hills = []
        for color, peak, opacity in (("#849fa5", 3.3, 0.28), ("#667f7b", 1.8, 0.5)):
            hill = Polygon(
                np.zeros((7, 2)), facecolor=color, alpha=opacity, edgecolor="none", zorder=1
            )
            self.camera.add_patch(hill)
            self.hills.append((hill, peak))
        self.terrain = Polygon(np.zeros((4, 2)), facecolor="#526353", edgecolor="none", zorder=2)
        self.camera.add_patch(self.terrain)
        self.soil = Polygon(np.zeros((4, 2)), facecolor="#3d443e", edgecolor="none", zorder=2.1)
        self.camera.add_patch(self.soil)
        self.camera.add_patch(
            Rectangle(
                (-20, -0.5),
                40,
                0.5,
                facecolor="#829091",
                edgecolor="#bac7c7",
                linewidth=0.5,
                zorder=3,
            )
        )
        span = self.replay.settings.gear.pad_half_width_m
        self.camera.add_patch(
            Rectangle(
                (-span, -0.14),
                span * 2,
                0.14,
                facecolor="#afb8b4",
                edgecolor="#d4ddd7",
                linewidth=0.7,
                zorder=4,
            )
        )
        self.camera.plot(
            (-span + 0.8, span - 0.8), (0.02, 0.02), color="#e7dec0", linewidth=1.2, zorder=4
        )
        self.camera.plot((-1.1, 1.1), (0.025, 0.025), color="#59c5b6", linewidth=2, zorder=5)
        for px in (-14, -11.5, 11.5, 14):
            self.camera.add_patch(
                Rectangle((px, -0.48), 0.28, 0.48, facecolor="#d9b967", edgecolor="none", zorder=4)
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
                facecolor="#697b83",
                edgecolor="#c2d1d5",
                linewidth=0.6,
                zorder=4,
            )
            self.camera.add_patch(support)
            self.launch_supports.append(support)
        self._draw_ground_facilities()
        (self.trail,) = self.camera.plot(
            [], [], color="#4b91a3", alpha=0.25, linewidth=0.9, zorder=2
        )
        self.parts = {}
        geometry, _, _ = _nominal_flight_geometry(self.replay, self.replay.times[0])
        colors = {
            "body": "#c8d4db",
            "adapter": "#d5dfe3",
            "upper_stage": "#e4eced",
            "upper_nozzle": "#233541",
            "upper_highlight": "#ffffff",
            "upper_band_detail": "#456775",
            "highlight": "#f0f5f3",
            "panel": "#8b9da7",
            "nozzle": "#283a45",
            "upper_band": "#344b58",
            "lower_band": "#263b47",
            "separation_band": "#37636d",
            "fin_left": "#294b55",
            "fin_right": "#294b55",
        }
        for name in geometry:
            patch = Polygon(
                np.zeros((4, 2)),
                facecolor=colors.get(name, "#506b77"),
                edgecolor="#2b424e",
                linewidth=0.5,
                zorder=9 if name == "highlight" else 8,
            )
            self.camera.add_patch(patch)
            self.parts[name] = patch
        self.shell_shading = []
        for color, alpha in (("#8195a0", 0.48), ("#ffffff", 0.6), ("#46616e", 0.4)):
            patch = Polygon(
                np.zeros((4, 2)), facecolor=color, alpha=alpha, edgecolor="none", zorder=9
            )
            self.camera.add_patch(patch)
            self.shell_shading.append(patch)
        self.engine_skirt = Polygon(
            np.zeros((4, 2)),
            facecolor="#344b57",
            edgecolor="#78939f",
            linewidth=0.4,
            zorder=9,
        )
        self.camera.add_patch(self.engine_skirt)
        self.seams = LineCollection([], colors="#637c86", linewidths=0.55, zorder=10)
        self.fin_mesh = LineCollection([], colors="#91adb5", linewidths=0.55, zorder=10)
        self.camera.add_collection(self.seams)
        self.camera.add_collection(self.fin_mesh)
        self.body_label = self.camera.text(
            0,
            0,
            "DM100",
            ha="center",
            va="center",
            rotation=90,
            color="#325160",
            fontsize=9,
            weight="bold",
            zorder=11,
        )
        self.plumes = []
        for color, alpha in (
            ("#e74b24", 0.23),
            ("#ff7435", 0.65),
            ("#ffc164", 0.93),
            ("#fff5c7", 1),
        ):
            plume = Polygon(
                np.zeros((4, 2)), facecolor=color, edgecolor="none", alpha=alpha, zorder=6
            )
            self.camera.add_patch(plume)
            self.plumes.append(plume)
        self.upper_plumes = []
        for color, alpha, order in (
            ("#69a9f5", 0.22, 6),
            ("#97d5ff", 0.65, 6.1),
            ("#f0faff", 0.95, 6.2),
        ):
            plume = Polygon(
                np.zeros((4, 2)), facecolor=color, edgecolor="none", alpha=alpha, zorder=order
            )
            self.camera.add_patch(plume)
            self.upper_plumes.append(plume)
        self.rcs_jets = []
        for _ in range(2):
            jet = Polygon(
                np.zeros((3, 2)), facecolor="#daeffa", edgecolor="none", alpha=0.6, zorder=7
            )
            self.camera.add_patch(jet)
            self.rcs_jets.append(jet)
        self.heat_glow = Ellipse(
            (0, 0), 10, 1, facecolor="#edb35e", edgecolor="none", alpha=0, zorder=5
        )
        self.camera.add_patch(self.heat_glow)
        self.dust = []
        for i in range(8):
            puff = Ellipse((0, 0), 1, 1, facecolor="#dedcc4", edgecolor="none", alpha=0, zorder=7)
            self.camera.add_patch(puff)
            self.dust.append((puff, i))
        self.phase = self.camera.text(
            0.025,
            0.96,
            "",
            transform=self.camera.transAxes,
            va="top",
            color="#eef8ff",
            fontsize=12,
            weight="bold",
            bbox={
                "facecolor": "#0b2533",
                "edgecolor": "#315260",
                "boxstyle": "round,pad=0.5",
                "alpha": 0.9,
            },
            zorder=14,
        )
        self.camera_note = self.camera.text(
            0.025,
            0.03,
            "",
            transform=self.camera.transAxes,
            color="#eef8ff",
            fontsize=9,
            zorder=14,
            bbox={"facecolor": "#0b2533", "edgecolor": "none", "alpha": 0.78, "pad": 4},
        )
        (self.scale_line,) = self.camera.plot([], [], color="#eff8fc", linewidth=1.7, zorder=15)
        self.scale_label = self.camera.text(
            0, 0, "", color="#eff8fc", fontsize=8, ha="center", va="bottom", zorder=15
        )
        self.telemetry = []
        self.telemetry_labels = []
        self.telemetry_heading = self.fig.text(
            0.654,
            0.634,
            "비행 텔레메트리",
            color="#a6bfcd",
            fontsize=9,
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
        # Fixed-width tiles keep changing values aligned, with a separate
        # status strip below the main panels rather than squeezing extra rows in.
        for i, label in enumerate(labels):
            col, row = i % 2, i // 2
            xx, yy = 0.015 + col * 0.525, 0.952 - row * 0.140
            self.info.add_patch(
                FancyBboxPatch(
                    (xx, yy - 0.107),
                    0.455,
                    0.115,
                    transform=self.info.transAxes,
                    boxstyle="round,pad=0.002,rounding_size=0.012",
                    facecolor="#0d2130",
                    edgecolor="#203c4c",
                    linewidth=0.55,
                )
            )
            label_artist = self.info.text(
                xx + 0.022, yy, label, color="#86a8ba", fontsize=8.5, va="top"
            )
            self.telemetry_labels.append(label_artist)
            value = self.info.text(
                xx + 0.022,
                yy - 0.050,
                "",
                color="#79e0cc" if i in (6, 7) else "#e8f2f8",
                fontsize=13,
                va="top",
                family="DejaVu Sans",
            )
            self.telemetry.append(value)
        self.status = self.fig.text(0.654, 0.142, "", color="#8daebb", fontsize=8.5, va="bottom")
        self.environment_text = self.fig.text(
            0.654, 0.168, "", color="#86a8ba", fontsize=8.5, va="bottom"
        )
        self.event = self.fig.text(
            0.045, 0.164, "", color="#6ee0c6", fontsize=10, va="center", linespacing=1.55
        )
        self._draw_overview()
        self.timeline = []
        for i, label in enumerate(("발사", "분리", "귀환", "진입", "접근", "접지")):
            xx = 0.04 + i * 0.155
            card = FancyBboxPatch(
                (xx, 0.073),
                0.138,
                0.054,
                transform=self.fig.transFigure,
                boxstyle="round,pad=0.002,rounding_size=0.008",
                facecolor="#102636",
                edgecolor="#203d4c",
                linewidth=0.8,
            )
            self.fig.add_artist(card)
            text = self.fig.text(
                xx + 0.012, 0.094, f"{i + 1:02}  {label}", color="#7493a5", fontsize=10
            )
            self.timeline.append((card, text))
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
        self.overview.scatter([0], [0], marker="s", color="#63ddce", s=22, zorder=5)
        (self.history,) = self.overview.plot(
            [], [], color="#63ddce", linewidth=1.5, zorder=4, label="1단 계산"
        )
        (self.marker,) = self.overview.plot([], [], "o", color="#ffcc85", markersize=5, zorder=6)
        (self.upper_history,) = self.overview.plot(
            [], [], "--", color="#f6c56b", linewidth=1.4, zorder=4, label="2단 연출"
        )
        (self.upper_marker,) = self.overview.plot(
            [], [], "^", color="#f6c56b", markersize=5, zorder=6
        )
        self.upper_marker.set_visible(False)
        self.overview_legend = None
        if departure is not None:
            legend = self.overview.legend(loc="upper left", fontsize=7, frameon=False, ncol=2)
            self.overview_legend = legend
            for text in legend.get_texts():
                text.set_color("#c6dce8")
        self.overview.set(
            xlim=(xlow, xhigh),
            ylim=(-1, zhigh),
            title="비행 경로",
        )
        self.overview.set_xlabel("발사장 거리 (km)", color="#a5bfcc", fontsize=8)
        self.overview.set_ylabel("고도 (km)", color="#a5bfcc", fontsize=8)
        self.overview.title.set_color("#c6dce8")
        self.overview.title.set_fontsize(10)
        self.overview.tick_params(colors="#9ab6c6", labelsize=7)
        self.overview.grid(alpha=0.13, color="#d2e7ee", linewidth=0.5)
        for spine in self.overview.spines.values():
            spine.set_color("#335461")

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
            np.asarray(((-radius, station), (radius, station))) @ transform.T + base
            for station in (3, 7.5, 13, 17.6, 19.3)
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
        label = np.asarray((0, 11)) @ transform.T + base
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
        self.stars.set_alpha(float(np.clip((z - 28000) / 40000, 0, 0.55)))
        for cloud in self.clouds:
            cloud.set_alpha(float(0.18 * np.exp(-max(0, z) / 2200)))
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
        history = self.replay.bases[max(0, index - 45) : index + 1]
        self.trail.set_data(history[:, 0], history[:, 1])
        self.history.set_data(
            self.replay.bases[: index + 1, 0] / 1000, self.replay.bases[: index + 1, 1] / 1000
        )
        self.marker.set_data([base[0] / 1000], [base[1] / 1000])
        upper_sample = departure.sample(shown_time) if departure and not attached else None
        self.upper_history.set_visible(upper_sample is not None)
        self.upper_marker.set_visible(False)
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
            for plume, scale in zip(self.upper_plumes, (1.2, 0.9, 0.5), strict=True):
                width = 0.6 * np.sqrt(ignition) * scale
                length = 11 * np.sqrt(ignition) * scale * flicker
                local = np.asarray(
                    (
                        (-width, 0),
                        (-width * 1.4, length * 0.2),
                        (0, length),
                        (width * 1.4, length * 0.2),
                        (width, 0),
                    )
                )
                plume.set_xy(upper_base + local[:, :1] * side + local[:, 1:] * exhaust)
                plume.set_visible(ignition > 0.01 and in_domain)
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
        flicker = 0.95 + 0.05 * np.sin(shown_time * 57)
        length = (0.9 + 22 * radius) * np.sqrt(ratio) * flicker
        for plume, scale in zip(self.plumes, (1.15, 1, 0.73, 0.43), strict=True):
            plume_width = radius * (0.35 + 0.85 * np.sqrt(ratio)) * scale
            local = np.asarray(
                (
                    (-plume_width, 0),
                    (-plume_width * 1.45, length * 0.18 * scale),
                    (-plume_width * 0.7, length * 0.55 * scale),
                    (0, length * scale),
                    (plume_width * 0.7, length * 0.55 * scale),
                    (plume_width * 1.45, length * 0.18 * scale),
                    (plume_width, 0),
                )
            )
            vertices = _clip_plume_at_ground(
                nozzle + local[:, :1] * side + local[:, 1:] * exhaust, 0
            )
            plume.set_visible(self.display_engine_on and len(vertices) >= 3)
            if len(vertices) >= 3:
                plume.set_xy(vertices)
        transform = rotation(theta)
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
            "landing": "착륙 · 주엔진 제동",
            "alignment": "수평 · 자세 정렬",
            "approach": "최종 접근",
            "touchdown": "첫 접지 · 비행 기록 종료",
        }
        self.phase.set_text(phase_names.get(phase, phase))
        self.clock.set_text(f"T+ {shown_time:07.2f} s")
        self.camera_note.set_text(
            "엔진 OFF · 접지 이후 고정 화면" if contacted else f"1단 하단 고도 {base[1]:,.1f} m"
        )
        total_mass = mass + (self.replay.design["upper_mass_kg"] if attached else 0)
        q = np.interp(shown_time, self.replay.times, self.q)
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
            "OFF / 0 kN" if contacted else f"{thrust / 1000:,.1f} kN",
            f"{state[12] * 100:.2f} %",
            f"{state[10] * 100:.0f} %",
            f"{state[8] * 100:.0f} %",
        )
        for artist, value in zip(self.telemetry, values, strict=True):
            artist.set_text(value)
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
            self.event.set_text(
                f"{heading} · 발 수직속력 {speed:.3f} m/s\n"
                "엔진 OFF · 재생 연출 / 접지 이후 거동 미평가"
            )
            self.footer.set_text(
                "접지 후 화면 고정·엔진 OFF는 연출 · 기립 미평가 · 2단 궤도 미계산"
            )
        elif attached:
            self.event.set_text("1·2단 결합")
        else:
            self.event.set_text(
                f"분리 T+{self.replay.separation_s:.2f} s · 1단 귀환\n"
                + (
                    "2단 상승 연출 / 궤도 미계산"
                    if departure is not None
                    else "2단 초기 기록 없음 · 1단만 재생"
                )
            )
        stage = {
            "ascent": 0,
            "flip": 1,
            "boostback": 2,
            "coast": 2,
            "entry": 3,
            "landing": 4,
            "alignment": 4,
            "approach": 4,
            "touchdown": 5,
        }.get(phase, 0)
        for i, (card, text) in enumerate(self.timeline):
            card.set_facecolor("#173e46" if i <= stage else "#102636")
            card.set_edgecolor("#5bd2be" if i == stage else "#254554")
            text.set_color("#d6f5ee" if i <= stage else "#7493a5")


class _NominalFlightFrameRenderer:
    """Cache the fixed dashboard while redrawing the moving camera and telemetry."""

    def __init__(self, scene: _NominalFlightAnimationScene) -> None:
        self.scene = scene
        self.artists = [
            scene.camera,
            scene.history,
            scene.upper_history,
            *([scene.overview_legend] if scene.overview_legend is not None else []),
            scene.marker,
            scene.upper_marker,
            *scene.telemetry,
            scene.telemetry_labels[10],
            scene.telemetry_labels[11],
            scene.status,
            scene.environment_text,
            scene.event,
            scene.clock,
            scene.footer,
            *(artist for pair in scene.timeline for artist in pair),
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
