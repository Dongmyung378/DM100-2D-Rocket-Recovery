"""Save, load and render recorded landing episodes without rerunning physics."""

from __future__ import annotations

import io
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import matplotlib
import numpy as np
from numpy.typing import NDArray

matplotlib.use("Agg")

from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter, writers  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Polygon  # noqa: E402

from powered_landing_guidance.model import ACTION_NAMES, STATE_NAMES


@dataclass(frozen=True, slots=True)
class EpisodeTrajectory:
    """Validated arrays extracted from an episode log.

    State-related arrays include the initial frame. Actions have one row per
    recorded step because each command applies over the following interval.
    """

    times_s: NDArray[np.float64]
    states: NDArray[np.float64]
    actions: NDArray[np.float64]
    thrust_n: NDArray[np.float64]
    outcomes: tuple[str, ...]
    ground_z_m: float


def _ffmpeg_available() -> bool:
    if writers.is_available("ffmpeg"):
        return True
    try:
        from imageio_ffmpeg import get_ffmpeg_exe

        matplotlib.rcParams["animation.ffmpeg_path"] = get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        return False
    return writers.is_available("ffmpeg")


def _vector(values: Any, size: int, label: str) -> NDArray[np.float64]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (size,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} must contain {size} finite values")
    return vector


def _number(value: Any, label: str, *, minimum: float | None = None) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a finite number") from error
    if not np.isfinite(number) or minimum is not None and number < minimum:
        raise ValueError(f"{label} must be a finite number")
    return number


def episode_trajectory(episode: dict[str, Any]) -> EpisodeTrajectory:
    """Convert a JSON-compatible episode log into plotting arrays."""
    if not isinstance(episode, dict) or episode.get("schema_version") != 1:
        raise ValueError("episode must use schema_version 1")
    if episode.get("state_names") != list(STATE_NAMES):
        raise ValueError("episode state order does not match this simulator")
    if episode.get("action_names") != list(ACTION_NAMES):
        raise ValueError("episode action order does not match this simulator")

    initial = _vector(episode.get("initial_state"), len(STATE_NAMES), "initial_state")
    steps = episode.get("steps")
    if not isinstance(steps, list):
        raise ValueError("episode steps must be a list")

    times = [0.0]
    states = [initial]
    actions: list[NDArray[np.float64]] = []
    thrust = []
    outcomes = ["running"]
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            raise ValueError(f"step {index} must be a mapping")
        time_s = _number(step.get("time_s"), f"step {index} time")
        if time_s < times[-1]:
            raise ValueError("episode times must be finite and nondecreasing")
        times.append(time_s)
        states.append(_vector(step.get("state"), len(STATE_NAMES), f"step {index} state"))
        actions.append(
            _vector(step.get("clipped_action"), len(ACTION_NAMES), f"step {index} action")
        )
        thrust_n = _number(step.get("thrust_end_n"), f"step {index} thrust", minimum=0.0)
        thrust.append(thrust_n)
        outcome = step.get("outcome")
        if not isinstance(outcome, str):
            raise ValueError(f"step {index} outcome must be a string")
        outcomes.append(outcome)

    if actions:
        step_actions = np.vstack(actions)
        first_thrust = _number(steps[0].get("thrust_start_n"), "initial thrust", minimum=0.0)
        frame_thrust = np.asarray((first_thrust, *thrust), dtype=np.float64)
    else:
        step_actions = np.empty((0, len(ACTION_NAMES)), dtype=np.float64)
        frame_thrust = np.zeros(1, dtype=np.float64)

    declared_outcome = episode.get("outcome")
    if not isinstance(declared_outcome, str) or declared_outcome != outcomes[-1]:
        raise ValueError("episode outcome does not match its final step")

    try:
        ground = _number(episode["config"]["simulation"]["ground_z_m"], "ground_z_m")
    except (KeyError, TypeError) as error:
        raise ValueError("episode config is missing simulation.ground_z_m") from error

    return EpisodeTrajectory(
        times_s=np.asarray(times, dtype=np.float64),
        states=np.vstack(states),
        actions=step_actions,
        thrust_n=frame_thrust,
        outcomes=tuple(outcomes),
        ground_z_m=ground,
    )


def save_episode_data(episode: dict[str, Any], path: str | Path) -> Path:
    """Save an episode as JSON or compressed NPZ without overwriting a file."""
    destination = Path(path)
    suffix = destination.suffix.lower()
    if suffix not in {".json", ".npz"}:
        raise ValueError("episode output must end in .json or .npz")
    trajectory = episode_trajectory(episode)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".json":
        with destination.open("x", encoding="utf-8") as stream:
            json.dump(episode, stream, ensure_ascii=False, indent=2, allow_nan=False)
    else:
        payload = json.dumps(episode, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        with destination.open("xb") as stream:
            np.savez_compressed(
                stream,
                episode_json=np.asarray(payload),
                times_s=trajectory.times_s,
                states=trajectory.states,
                actions=trajectory.actions,
                thrust_n=trajectory.thrust_n,
            )
    return destination


def load_episode_data(path: str | Path) -> dict[str, Any]:
    """Load and validate JSON or NPZ episode data."""
    source = Path(path)
    if source.suffix.lower() == ".json":
        with source.open(encoding="utf-8") as stream:
            episode = json.load(stream)
    elif source.suffix.lower() == ".npz":
        with np.load(source, allow_pickle=False) as archive:
            required = {"episode_json", "times_s", "states", "actions", "thrust_n"}
            if set(archive.files) != required or archive["episode_json"].shape != ():
                raise ValueError("NPZ episode has an unsupported schema")
            episode = json.loads(str(archive["episode_json"].item()))
            trajectory = episode_trajectory(episode)
            for key, expected in (
                ("times_s", trajectory.times_s),
                ("states", trajectory.states),
                ("actions", trajectory.actions),
                ("thrust_n", trajectory.thrust_n),
            ):
                if not np.array_equal(archive[key], expected):
                    raise ValueError(f"NPZ {key} does not match its episode log")
    else:
        raise ValueError("episode input must end in .json or .npz")
    episode_trajectory(episode)
    return episode


def _write_figure(fig, destination: Path, *, dpi: int) -> Path:
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
        with destination.open("xb") as stream:
            stream.write(buffer.getvalue())
    finally:
        plt.close(fig)
    return destination


def save_time_series(episode: dict[str, Any], path: str | Path, *, dpi: int = 160) -> Path:
    """Save position, velocity, attitude, mass and control histories as PNG."""
    destination = Path(path)
    if destination.suffix.lower() != ".png":
        raise ValueError("time-series output must end in .png")
    trajectory = episode_trajectory(episode)
    t = trajectory.times_s
    state = trajectory.states
    if len(trajectory.actions):
        action = np.vstack((trajectory.actions, trajectory.actions[-1]))
    else:
        action = np.zeros((len(t), len(ACTION_NAMES)), dtype=np.float64)

    fig, axes = plt.subplots(3, 2, figsize=(11, 8), sharex=True, layout="constrained")
    panels = (
        (axes[0, 0], (state[:, 0], state[:, 1]), ("x", "z"), "Position (m)"),
        (axes[0, 1], (state[:, 2], state[:, 3]), ("vx", "vz"), "Velocity (m/s)"),
        (
            axes[1, 0],
            (np.rad2deg(state[:, 4]), np.rad2deg(state[:, 5])),
            ("theta", "omega"),
            "Attitude (deg, deg/s)",
        ),
        (axes[1, 1], (state[:, 6],), ("mass",), "Mass (kg)"),
    )
    for axis, values, labels, title in panels:
        for series, label in zip(values, labels, strict=True):
            axis.plot(t, series, label=label)
        axis.set_title(title)
        axis.legend(loc="best")

    axes[2, 0].step(t, action[:, 0], where="post", color="tab:purple")
    axes[2, 0].set_title("Throttle")
    axes[2, 0].set_ylim(-0.05, 1.05)
    axes[2, 1].step(t, np.rad2deg(action[:, 1]), where="post", color="tab:red")
    axes[2, 1].set_title("Gimbal angle (deg)")
    for axis in axes.flat:
        axis.grid(alpha=0.3)
    axes[2, 0].set_xlabel("Time (s)")
    axes[2, 1].set_xlabel("Time (s)")
    fig.suptitle(f"Landing episode - {episode.get('outcome', trajectory.outcomes[-1])}")
    return _write_figure(fig, destination, dpi=dpi)


def _rocket_vertices(x: float, z: float, theta: float, length: float) -> NDArray[np.float64]:
    axis = np.asarray((np.sin(theta), np.cos(theta)))
    side = np.asarray((np.cos(theta), -np.sin(theta)))
    width = 0.18 * length
    base = np.asarray((x, z))
    return np.vstack(
        (
            base - width * side,
            base + width * side,
            base + 0.72 * length * axis + width * side,
            base + length * axis,
            base + 0.72 * length * axis - width * side,
        )
    )


def save_animation(
    episode: dict[str, Any],
    path: str | Path,
    *,
    fps: int = 30,
    max_frames: int = 600,
) -> Path:
    """Render a recorded trajectory as GIF or, when FFmpeg is installed, MP4."""
    if fps <= 0 or max_frames <= 0:
        raise ValueError("fps and max_frames must be positive")
    destination = Path(path)
    suffix = destination.suffix.lower()
    if suffix not in {".gif", ".mp4"}:
        raise ValueError("animation output must end in .gif or .mp4")
    if suffix == ".mp4" and not _ffmpeg_available():
        raise RuntimeError("MP4 output requires FFmpeg or the video extra")
    trajectory = episode_trajectory(episode)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)

    x = trajectory.states[:, 0]
    z = trajectory.states[:, 1]
    ground = trajectory.ground_z_m
    height = max(float(np.max(z) - ground), 1.0)
    rocket_length = max(1.0, min(5.0, height * 0.06))
    all_vertices = np.vstack(
        [
            _rocket_vertices(state[0], state[1], state[4], rocket_length)
            for state in trajectory.states
        ]
    )
    x_low = min(0.0, float(np.min(all_vertices[:, 0])))
    x_high = max(0.0, float(np.max(all_vertices[:, 0])))
    x_padding = max(0.5, 0.08 * max(x_high - x_low, height))
    z_low = min(ground, float(np.min(all_vertices[:, 1])))
    z_high = max(float(np.max(z)), float(np.max(all_vertices[:, 1])))
    z_padding = max(0.2, 0.06 * height)

    fig = plt.figure(figsize=(9, 7), layout="constrained")
    grid = fig.add_gridspec(1, 2, width_ratios=(4.0, 1.5))
    axis = fig.add_subplot(grid[0, 0])
    information = fig.add_subplot(grid[0, 1])
    information.axis("off")
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(x_low - x_padding, x_high + x_padding)
    axis.set_ylim(z_low - z_padding, z_high + z_padding)
    axis.axhline(ground, color="0.25", linewidth=2, label="Ground")
    axis.scatter([0.0], [ground], marker="x", s=80, color="tab:green", label="Landing target")
    (path_line,) = axis.plot([], [], color="tab:blue", linewidth=1.5, label="Trajectory")
    rocket = Polygon(np.zeros((5, 2)), closed=True, facecolor="0.85", edgecolor="black")
    axis.add_patch(rocket)
    thrust = FancyArrowPatch(
        (0, 0),
        (0, 0),
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=2,
        color="tab:red",
        label="Thrust direction",
    )
    axis.add_patch(thrust)
    status = information.text(0.0, 1.0, "", va="top", family="monospace")
    axis.set_xlabel("x (m)")
    axis.set_ylabel("z (m)")
    axis.set_title("Recorded landing episode")
    axis.grid(alpha=0.2)
    handles, labels = axis.get_legend_handles_labels()
    information.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.0, 0.0))
    information.text(
        0.0,
        0.38,
        "+x: right\n+z: up\n+angle: clockwise",
        va="top",
        color="0.3",
    )

    frame_count = len(trajectory.times_s)
    indices = np.unique(np.linspace(0, frame_count - 1, min(frame_count, max_frames), dtype=int))

    def update(frame: int):
        state = trajectory.states[frame]
        if len(trajectory.actions):
            command = trajectory.actions[max(0, frame - 1)]
        else:
            command = np.zeros(len(ACTION_NAMES), dtype=np.float64)
        path_line.set_data(x[: frame + 1], z[: frame + 1])
        rocket.set_xy(_rocket_vertices(state[0], state[1], state[4], rocket_length))
        direction = state[4] + command[1]
        start = np.asarray((state[0], state[1]))
        end = start + 0.85 * rocket_length * np.asarray((np.sin(direction), np.cos(direction)))
        thrust.set_positions(start, end)
        thrust.set_visible(trajectory.thrust_n[frame] > 0)
        status.set_text(
            f"t       {trajectory.times_s[frame]:8.3f} s\n"
            f"theta   {np.rad2deg(state[4]):8.2f} deg\n"
            f"gimbal  {np.rad2deg(command[1]):8.2f} deg\n"
            f"throttle{command[0]:8.3f}\n"
            f"mass    {state[6]:8.2f} kg\n"
            f"outcome {trajectory.outcomes[frame]}"
        )
        return path_line, rocket, thrust, status

    animation = FuncAnimation(
        fig,
        update,
        frames=indices,
        interval=1000 / fps,
        repeat=False,
        blit=False,
    )
    writer = PillowWriter(fps=fps) if suffix == ".gif" else FFMpegWriter(fps=fps)
    temporary = None
    try:
        with NamedTemporaryFile(
            prefix=f".{destination.stem}-",
            suffix=suffix,
            dir=destination.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
        animation.save(temporary, writer=writer, dpi=110)
        with temporary.open("rb") as source, destination.open("xb") as target:
            shutil.copyfileobj(source, target)
    finally:
        plt.close(fig)
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination
