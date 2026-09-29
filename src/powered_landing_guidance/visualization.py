"""Replay recorded 3 t APR flights without rerunning physics."""

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
from matplotlib import font_manager  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.animation import FFMpegWriter, writers  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402


def _ffmpeg_available() -> bool:
    if writers.is_available("ffmpeg"):
        return True
    try:
        from imageio_ffmpeg import get_ffmpeg_exe

        matplotlib.rcParams["animation.ffmpeg_path"] = get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        return False
    return writers.is_available("ffmpeg")


def _recovery_sky(altitudes: NDArray[np.float64]) -> NDArray[np.float64]:
    levels = (0.0, 2500.0, 6500.0, 11_000.0)
    colors = np.asarray(
        ((0.78, 0.91, 0.98), (0.48, 0.75, 0.92), (0.24, 0.49, 0.73), (0.10, 0.25, 0.45))
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


def _physical_recovery_geometry(state, deployment, strokes, settings, body_offset_m=0.0):
    """World-space hull/feet using recorded COM, deployment and actual gear stroke."""
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
    origin = state[:2] + transform @ np.asarray((body_offset_m, 0.0))
    parts = {name: (vertices - (0, cg)) @ transform.T + origin for name, vertices in parts.items()}
    feet, _ = foot_kinematics(state, deployment, settings)
    feet += transform @ np.asarray((body_offset_m, 0.0))
    for index, (sign, side) in enumerate(((-1, "left"), (1, "right"))):
        foot = feet[index].copy()
        foot[1] += strokes[index]
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
        from types import SimpleNamespace

        from powered_landing_guidance.recovery_physics import (
            AtmosphereSettings,
            Geometry,
            LandingGear,
            mass_properties,
        )

        expected = [
            "x_m",
            "z_m",
            "vx_m_s",
            "vz_m_s",
            "theta_rad",
            "omega_rad_s",
            "main_propellant_kg",
            "rcs_propellant_kg",
            "gear_extension",
            "left_friction_m",
            "right_friction_m",
            "fin_deflection_rad",
            "fin_extension",
        ]
        if flight.get("state_names") != expected:
            raise ValueError("unsupported full-flight state order")
        steps = flight["steps"]
        self.times = np.asarray([row["time_s"] for row in steps], dtype=np.float64)
        self.states = np.asarray([row["state"] for row in steps], dtype=np.float64)
        self.controls = np.asarray([row["control"] for row in steps], dtype=np.float64)
        if (
            len(self.times) < 2
            or self.states.shape != (len(self.times), 13)
            or self.controls.shape != (len(self.times), 4)
            or not np.all(np.isfinite(self.times))
            or np.any(np.diff(self.times) <= 0)
            or not np.all(np.isfinite(self.states))
            or not np.all(np.isfinite(self.controls))
        ):
            raise ValueError("full-flight replay requires finite, strictly ordered samples")
        self.flight = flight
        self.design = flight["design"]
        self.phases = [row["phase"] for row in steps]
        self.separation_s = (
            float(flight["separation"]["time_s"]) if flight.get("separation") else np.inf
        )
        self.contact_s = (
            float(flight["first_contact"]["time_s"]) if flight.get("first_contact") else np.inf
        )
        physics = flight["physics"]
        self.settings = SimpleNamespace(
            geometry=Geometry(**physics["geometry"]),
            gear=LandingGear(**physics["gear"]),
            atmosphere=AtmosphereSettings(**physics["atmosphere"]),
            fixed_mass_components=tuple(tuple(item) for item in physics["fixed_mass_components"]),
        )
        self.bases = self.states[:, :2].copy()
        for index, state in enumerate(self.states):
            mass = self.design["dry_mass_kg"] + state[6] + state[7]
            cg = mass_properties(mass, state[8], self.settings).cg_m
            if self.times[index] < self.separation_s:
                # The recorded plant places the upper-stage lumped COM at station 6 m.
                upper = self.design["upper_mass_kg"]
                cg = (mass * cg + upper * 6.0) / (mass + upper)
            self.bases[index] -= cg * np.asarray((np.sin(state[4]), np.cos(state[4])))

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
        state[:2] = base + cg * np.asarray((np.sin(state[4]), np.cos(state[4])))
        control = self.controls[left].copy()
        phase = self.phases[left]
        if time_s >= self.contact_s:
            control[:] = 0
            phase = "ground_settling"
        elif time_s >= self.separation_s and phase == "ascent":
            phase = "boostback"
        return state, base, mass, control, phase, left


def _nominal_flight_geometry(replay: _NominalFlightReplay, time_s: float):
    """Draw the recorded stage-one geometry and its actual ground-contact points."""
    from powered_landing_guidance.recovery_physics import foot_kinematics, rotation

    state, base, mass, control, phase, index = replay.sample(time_s)
    vector = np.r_[state[:6], mass]
    feet, _ = foot_kinematics(vector, state[8], replay.settings)
    strokes = np.maximum(0, -feet[:, 1])
    parts, nozzle = _physical_recovery_geometry(vector, state[8], strokes, replay.settings)
    transform = rotation(state[4])
    radius = replay.settings.geometry.body_radius_m
    station = replay.flight["fin_settings"]["station_m"]
    for sign, side in ((-1, "left"), (1, "right")):
        hinge = np.asarray((sign * radius, station))
        panel = np.asarray(((0, -0.18), (sign * 0.55, -0.18), (sign * 0.55, 0.18), (0, 0.18)))
        panel[:, 0] *= state[12]
        panel = panel @ rotation(state[11]).T + hinge
        parts[f"fin_{side}"] = panel @ transform.T + base
    length = replay.settings.geometry.body_length_m
    parts["upper_stage"] = (
        np.asarray(
            (
                (-radius, length),
                (radius, length),
                (radius, length + 1.25),
                (radius * 0.65, length + 1.65),
                (0, length + 2),
                (-radius * 0.65, length + 1.65),
                (-radius, length + 1.25),
            )
        )
        @ transform.T
        + base
    )
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
    def __init__(self, flight: dict[str, Any]) -> None:
        self.replay = _NominalFlightReplay(flight)
        self.fig = plt.figure(figsize=(12.8, 7.2), facecolor="#0c1728")
        self.camera = self.fig.add_axes((0.065, 0.16, 0.59, 0.68))
        self.overview = self.fig.add_axes((0.735, 0.66, 0.23, 0.21), facecolor="#15263e")
        self.info = self.fig.add_axes((0.735, 0.13, 0.245, 0.48))
        self.info.axis("off")
        self.fig.text(
            0.055,
            0.94,
            "3톤 로켓 | 발사 - 분리 - 1단 귀환 - 착륙",
            color="white",
            fontsize=19,
            weight="bold",
        )
        self.fig.text(
            0.055,
            0.895,
            "APR / PID · 저장 비행 재생 · 전 구간 정상 속도 (1배속)",
            color="#a9c3da",
            fontsize=10,
        )
        self.fig.text(
            0.055,
            0.045,
            "2D 예비 모델 · 형상은 도식 · 2단 분리 후 비행 및 실제 핫스테이징 점화는 미계산",
            color="#9eb3ca",
            fontsize=9,
        )
        self.camera.set_xlabel("수평 위치 (m)", color="#b6c9dc", fontsize=9)
        self.camera.set_ylabel("고도 (m)", color="#b6c9dc", fontsize=9)
        self.camera.tick_params(colors="#b6c9dc", labelsize=8)
        self.camera.grid(alpha=0.13, color="#223e55", linewidth=0.6)
        for axis in (self.camera, self.overview):
            for spine in axis.spines.values():
                spine.set_color("#47607a")
        self.sky = self.camera.imshow(np.zeros((64, 1, 3)), origin="lower", aspect="auto")
        self.terrain = Polygon(np.zeros((4, 2)), color="#566d59", zorder=2)
        self.camera.add_patch(self.terrain)
        span = self.replay.settings.gear.pad_half_width_m
        self.camera.add_patch(
            Polygon(
                ((-span, -0.3), (span, -0.3), (span, 0), (-span, 0)),
                facecolor="#758591",
                edgecolor="#263748",
                zorder=4,
            )
        )
        self.camera.axhline(0, color="#405344", linewidth=1.4, zorder=3)
        self.camera.plot((-1, 1), (0, 0), color="#52d7a0", linewidth=3, zorder=5)
        (self.trail,) = self.camera.plot([], [], color="#235b81", alpha=0.35, linewidth=1)
        self.parts = {}
        geometry, _, _ = _nominal_flight_geometry(self.replay, self.replay.times[0])
        for name in geometry:
            color = (
                "#eff3f6"
                if name in {"body", "adapter", "upper_stage"}
                else "#ffffff"
                if name == "highlight"
                else "#bdc9d4"
                if name == "panel"
                else "#273546"
            )
            patch = Polygon(
                np.zeros((4, 2)),
                facecolor=color,
                edgecolor="#233244",
                linewidth=0.5,
                zorder=9 if name == "highlight" else 8,
            )
            self.camera.add_patch(patch)
            self.parts[name] = patch
        self.plumes = []
        for color, alpha in (("#ff642e", 0.7), ("#ffc945", 0.95), ("#ffffe9", 1)):
            patch = Polygon(np.zeros((4, 2)), color=color, alpha=alpha, zorder=6)
            self.camera.add_patch(patch)
            self.plumes.append(patch)
        self.phase = self.camera.text(
            0.025,
            0.96,
            "",
            transform=self.camera.transAxes,
            va="top",
            color="white",
            fontsize=11,
            bbox={"facecolor": "#15263e", "edgecolor": "none", "pad": 5},
        )
        self.status = self.info.text(
            0, 1, "", va="top", color="#e3edf7", fontsize=10, linespacing=1.55
        )
        self.event = self.info.text(0, 0.01, "", va="bottom", color="#61e4ad", fontsize=10)
        bases = self.replay.bases
        xlow, xhigh = min(0, bases[:, 0].min()) / 1000 - 0.04, bases[:, 0].max() / 1000 + 0.04
        zhigh = bases[:, 1].max() / 1000 + 0.3
        self.overview.imshow(
            _recovery_sky(np.linspace(0, zhigh * 1000, 64))[:, None, :],
            extent=(xlow, xhigh, 0, zhigh),
            origin="lower",
            aspect="auto",
        )
        self.overview.plot(
            bases[:, 0] / 1000, bases[:, 1] / 1000, color="white", alpha=0.3, linewidth=1
        )
        self.overview.scatter([0], [0], marker="x", color="#60f0b4", s=25)
        (self.history,) = self.overview.plot([], [], color="#ffc945", linewidth=1.5)
        (self.marker,) = self.overview.plot([], [], "o", color="#ffc945", markersize=5)
        self.overview.set(xlim=(xlow, xhigh), ylim=(-0.15, zhigh), title="전체 비행 경로")
        self.overview.set_xlabel("수평 거리 (km)", color="#b6c9dc", fontsize=8)
        self.overview.set_ylabel("고도 (km)", color="#b6c9dc", fontsize=8)
        self.overview.title.set_color("#e3edf7")
        self.overview.title.set_fontsize(10)
        self.overview.tick_params(colors="#b6c9dc", labelsize=7)
        self.overview.grid(alpha=0.12)
        self.update(self.replay.times[0])

    def update(self, time_s: float) -> None:
        from powered_landing_guidance.recovery_physics import atmosphere

        parts, nozzle, sample = _nominal_flight_geometry(self.replay, time_s)
        state, base, mass, control, phase, index = sample
        attached = time_s < self.replay.separation_s
        for name, vertices in parts.items():
            self.parts[name].set_xy(vertices)
            self.parts[name].set_visible(name not in {"upper_stage", "separation_band"} or attached)
        x, z, vx, vz, theta, omega = state[:6]
        height = float(np.clip(18 + max(z, 0) * 0.002, 18, 36))
        bottom = max(-1.5, z - 0.45 * height)
        width = height * (12.8 * 0.59) / (7.2 * 0.68)
        left, right = x - width / 2, x + width / 2
        self.camera.set(xlim=(left, right), ylim=(bottom, bottom + height))
        self.sky.set_data(_recovery_sky(np.linspace(bottom, bottom + height, 64))[:, None, :])
        self.sky.set_extent((left, right, bottom, bottom + height))
        self.terrain.set_xy(((left, bottom), (right, bottom), (right, 0), (left, 0)))
        self.terrain.set_visible(bottom < 0)
        history = self.replay.bases[max(0, index - 60) : index + 1]
        self.trail.set_data(history[:, 0], history[:, 1])
        self.history.set_data(
            self.replay.bases[: index + 1, 0] / 1000, self.replay.bases[: index + 1, 1] / 1000
        )
        self.marker.set_data([base[0] / 1000], [base[1] / 1000])
        pressure = atmosphere(z, self.replay.settings.atmosphere)[1]
        thrust = control[0] * (
            self.replay.design["sea_level_thrust_n"]
            + self.replay.design["nozzle_exit_area_m2"] * (101325 - pressure)
        )
        ratio = float(np.clip(control[0], 0, 1))
        exhaust = -np.asarray((np.sin(theta + control[1]), np.cos(theta + control[1])))
        side = np.asarray((np.cos(theta + control[1]), -np.sin(theta + control[1])))
        length = (1.2 + 8.5 * np.sqrt(ratio)) * (0.96 + 0.04 * np.sin(time_s * 57))
        for plume, scale in zip(self.plumes, (1, 0.78, 0.5), strict=True):
            width = (0.24 + 0.25 * np.sqrt(ratio)) * scale
            local = np.asarray(
                (
                    (-width, 0),
                    (-width * 1.3, length * 0.22 * scale),
                    (0, length * scale),
                    (width * 1.3, length * 0.22 * scale),
                    (width, 0),
                )
            )
            vertices = _clip_plume_at_ground(
                nozzle + local[:, :1] * side + local[:, 1:] * exhaust, 0
            )
            plume.set_visible(thrust > 1e-6 and len(vertices) >= 3)
            if len(vertices) >= 3:
                plume.set_xy(vertices)
        names = {
            "ascent": "발사·상승 / 1·2단 결합",
            "boostback": "분리 후 귀환 궤적 수정",
            "coast": "공력핀 보조 관성 하강",
            "landing": "주엔진 제동",
            "alignment": "수평·자세 정렬",
            "approach": "최종 착륙 접근",
            "ground_settling": "접지·엔진 정지 / 안정화 확인",
        }
        self.phase.set_text(names.get(phase, phase))
        mass += self.replay.design["upper_mass_kg"] if attached else 0
        self.status.set_text(
            f"비행 시간     {time_s:8.2f} s\n"
            f"1단 하단 고도 {base[1]:8.2f} m\n"
            f"수평 위치     {x:+8.3f} m\n"
            f"수직 속도     {vz:+8.3f} m/s\n"
            f"수평 속도     {vx:+8.3f} m/s\n"
            f"자세          {np.rad2deg(theta):+8.3f}°\n"
            f"각속도        {np.rad2deg(omega):+8.3f}°/s\n"
            f"비행 질량     {mass:8.2f} kg\n"
            f"주 추진제     {state[6]:8.2f} kg\n"
            f"RCS 추진제    {state[7]:8.3f} kg\n"
            f"주엔진 추력   {thrust / 1000:8.2f} kN\n"
            f"핀 / 다리 전개  {state[12] * 100:4.0f}% / {state[8] * 100:3.0f}%"
        )
        if time_s >= self.replay.contact_s:
            contact = self.replay.flight["first_contact"]
            foot = contact["first_feet"][0]
            speed = abs(contact["foot_velocities_m_s"][foot][1])
            event = f"접지 수직속력 {speed:.3f} m/s"
            if time_s >= self.replay.times[-1]:
                if self.replay.flight.get("simulated_acceptance_passed") is True:
                    event += "\n착륙 성공 · 안정화 완료"
                else:
                    event += f"\n판정: {self.replay.flight.get('outcome', 'unverified')}"
            else:
                event += "\n무추력 상태 관찰 중"
        elif attached:
            event = "2단·탑재체 1,015 kg 결합 상태"
        else:
            event = f"분리 t={self.replay.separation_s:.2f} s\n2단 이후 궤적은 표시하지 않음"
        self.event.set_text(event)


def save_nominal_flight_replay(
    flight: dict[str, Any], path: str | Path, *, fps: int = 20, preview: str | Path | None = None
) -> Path:
    """Render a stored ascent/separation/recovery flight at normal playback speed."""
    if type(fps) is not int or not 1 <= fps <= 60:
        raise ValueError("full-flight replay fps must be an integer in [1, 60]")
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
    fonts = font_manager.get_font_names()
    family = next(
        (name for name in ("Malgun Gothic", "Noto Sans CJK KR", "NanumGothic") if name in fonts),
        "DejaVu Sans",
    )
    temporary = None
    with matplotlib.rc_context({"font.family": family, "axes.unicode_minus": False}):
        scene = _NominalFlightAnimationScene(flight)
        try:
            frames = _nominal_flight_frame_times(scene.replay.times, fps)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(
                prefix=f".{destination.stem}-", suffix=".mp4", dir=destination.parent, delete=False
            ) as stream:
                temporary = Path(stream.name)
            writer = FFMpegWriter(
                fps=fps,
                codec="libx264",
                bitrate=2500,
                extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
            )
            with writer.saving(scene.fig, str(temporary), dpi=100):
                for index, time_s in enumerate(frames):
                    scene.update(float(time_s))
                    writer.grab_frame(facecolor=scene.fig.get_facecolor())
                    if index % (fps * 30) == 0:
                        print(
                            f"Rendering saved flight: {time_s:.1f} / {frames[-1]:.1f} s", flush=True
                        )
            with temporary.open("rb") as source, destination.open("xb") as target:
                shutil.copyfileobj(source, target)
            if poster:
                poster.parent.mkdir(parents=True, exist_ok=True)
                scene.fig.savefig(poster, dpi=100, facecolor=scene.fig.get_facecolor())
        finally:
            plt.close(scene.fig)
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return destination


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True, help="저장한 3톤 APR 결과 JSON")
    parser.add_argument("--animation", type=Path, required=True, help="새 MP4 출력 경로")
    parser.add_argument("--preview", type=Path, help="선택적인 PNG 미리보기")
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args(argv)
    try:
        record = json.loads(args.replay.read_text(encoding="utf-8"))
        path = save_nominal_flight_replay(
            record, args.animation, fps=args.fps, preview=args.preview
        )
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
        parser.error(str(error))
    print(f"애니메이션 저장: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
