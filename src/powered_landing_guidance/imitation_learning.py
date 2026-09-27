"""Closed-loop evaluation and on-policy data collection for planar imitation learning."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from powered_landing_guidance.behavior_cloning import BCPolicy
from powered_landing_guidance.controllers import IntegratedLandingController
from powered_landing_guidance.envs import RocketLandingEnv


def initial_states_from_shard(path: Path) -> NDArray[np.float64]:
    with np.load(path, allow_pickle=False) as archive:
        offsets = archive["state_offsets"]
        states = archive["states"][offsets[:-1]]
    if states.ndim != 2 or states.shape[1] != 7 or not np.all(np.isfinite(states)):
        raise ValueError("invalid initial states in trajectory shard")
    return np.asarray(states, dtype=np.float64)


@dataclass(frozen=True)
class ActuatorSettings:
    throttle_rate_per_s: float | None = None
    gimbal_rate_rad_s: float | None = None
    noise_std: tuple[float, ...] = (0.0,) * 7
    seed: int = 0


DEFAULT_ACTUATOR = ActuatorSettings()


def run_episode(
    config: dict[str, Any],
    initial_state: NDArray[Any],
    policy: BCPolicy | None,
    normalization: dict[str, Any] | None = None,
    *,
    wind_m_s: float = 0.0,
    thrust_scale: float = 1.0,
    pid_mix: float = 0.0,
    actuator: ActuatorSettings = DEFAULT_ACTUATOR,
    collect: bool = False,
) -> tuple[dict[str, Any], NDArray[np.float64]]:
    """Roll out a learned policy or the PID reference at the simulator's 0.1 s control rate."""
    episode_config = deepcopy(config)
    episode_config["environment"]["horizontal_wind_m_s"] = float(wind_m_s)
    if not 0.5 <= thrust_scale <= 1.1:
        raise ValueError("thrust_scale must be between 0.5 and 1.1")
    episode_config["vehicle"]["max_thrust_n"] *= float(thrust_scale)
    if not 0.0 <= pid_mix <= 1.0:
        raise ValueError("pid_mix must be within [0, 1]")
    env = RocketLandingEnv(episode_config)
    pid = (
        IntegratedLandingController.from_config(episode_config)
        if policy is None or pid_mix > 0.0
        else None
    )
    rng = np.random.default_rng(actuator.seed)
    noise = np.asarray(actuator.noise_std, dtype=np.float64)
    if noise.shape != (7,) or np.any(noise < 0) or not np.all(np.isfinite(noise)):
        raise ValueError("noise_std must contain seven nonnegative finite values")
    states: list[NDArray[np.float64]] = []
    try:
        state, info = env.reset(options={"initial_state": initial_state})
        applied: NDArray[np.float64] | None = None
        interval = float(config["integrated_landing_controller"]["control_interval_s"])
        next_control = 0.0
        target = np.zeros(2, dtype=np.float64)
        while True:
            time_s = float(info["time_s"])
            if time_s + 1e-9 >= next_control:
                if collect:
                    states.append(state.copy())
                observed = state + rng.normal(0.0, noise)
                if policy is None:
                    target = pid.command(observed, time_s=time_s)
                else:
                    if normalization is None:
                        raise ValueError("normalization is required for learned policies")
                    target = policy.predict_action(
                        observed,
                        normalization,
                        (env.parameters.throttle_min, env.parameters.throttle_max),
                        env.parameters.gimbal_limit_rad,
                    )
                    if pid_mix > 0.0:
                        target = (1.0 - pid_mix) * target + pid_mix * pid.command(
                            observed, time_s=time_s
                        )
                next_control += interval
            action = target.copy()
            if applied is not None:
                for index, rate in enumerate(
                    (actuator.throttle_rate_per_s, actuator.gimbal_rate_rad_s)
                ):
                    if rate is not None:
                        action[index] = np.clip(
                            action[index],
                            applied[index] - rate * env.dt,
                            applied[index] + rate * env.dt,
                        )
            applied = action.copy()
            state, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        return env.episode_log, np.asarray(states, dtype=np.float64).reshape(-1, 7)
    finally:
        env.close()


def failure_type(episode: dict[str, Any], config: dict[str, Any]) -> str:
    outcome = episode["outcome"]
    if outcome == "success":
        return "success"
    if outcome in {"fuel_depletion", "timeout"}:
        return outcome
    state = np.asarray(episode["steps"][-1]["state"], dtype=np.float64)
    limits = config["landing_success"]
    if abs(state[0]) > limits["max_abs_x_m"]:
        return "position_error"
    if abs(state[4]) > np.deg2rad(limits["max_abs_theta_deg"]) or abs(state[5]) > np.deg2rad(
        limits["max_abs_omega_deg_s"]
    ):
        return "attitude_error"
    if abs(state[2]) > limits["max_abs_vx_m_s"]:
        return "horizontal_speed"
    if abs(state[3]) > limits["max_abs_vz_m_s"]:
        return "hard_touchdown"
    return "other_failure"


def episode_metrics(episode: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    steps = episode["steps"]
    final = np.asarray(steps[-1]["state"], dtype=np.float64)
    commands = np.asarray([step["clipped_action"] for step in steps], dtype=np.float64)
    times = np.asarray([step["time_s"] for step in steps], dtype=np.float64)
    change = np.diff(commands, axis=0)
    delta_t = np.diff(times)
    # Ground-contact and fuel-cutoff steps can be shorter than the command update.
    # They do not represent a full actuator-rate interval.
    rapid = delta_t >= float(config["simulation"]["dt_s"]) - 1e-7
    rates = np.abs(change[rapid] / delta_t[rapid, None]) if np.any(rapid) else np.zeros((0, 2))
    return {
        "outcome": episode["outcome"],
        "failure_type": failure_type(episode, config),
        "duration_s": float(times[-1]),
        "fuel_used_kg": float(episode["initial_state"][6] - final[6]),
        "touchdown_x_m": float(final[0]),
        "touchdown_vx_m_s": float(final[2]),
        "touchdown_vz_m_s": float(final[3]),
        "touchdown_theta_deg": float(np.rad2deg(final[4])),
        "max_throttle_rate_per_s": float(np.max(rates[:, 0])) if len(rates) else 0.0,
        "max_gimbal_rate_deg_s": float(np.rad2deg(np.max(rates[:, 1]))) if len(rates) else 0.0,
        "action_total_variation": float(
            np.sum(np.abs(change[:, 0])) + np.sum(np.abs(change[:, 1]))
        ),
    }


def evaluate_cohort(
    config: dict[str, Any],
    states: NDArray[Any],
    policy: BCPolicy | None,
    normalization: dict[str, Any] | None = None,
    *,
    wind_m_s: float = 0.0,
    thrust_scale: float = 1.0,
    pid_mix: float = 0.0,
    actuator: ActuatorSettings = DEFAULT_ACTUATOR,
    keep_episodes: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cases = []
    episodes = []
    for index, initial in enumerate(states):
        episode, _ = run_episode(
            config,
            initial,
            policy,
            normalization,
            wind_m_s=wind_m_s,
            thrust_scale=thrust_scale,
            pid_mix=pid_mix,
            actuator=ActuatorSettings(
                actuator.throttle_rate_per_s,
                actuator.gimbal_rate_rad_s,
                actuator.noise_std,
                actuator.seed + index,
            ),
        )
        cases.append({"index": index, **episode_metrics(episode, config)})
        if keep_episodes:
            episodes.append(episode)
    failure_counts = Counter(case["failure_type"] for case in cases)
    summary = {
        "episodes": len(cases),
        "successes": failure_counts["success"],
        "success_rate": failure_counts["success"] / len(cases),
        "failure_counts": dict(failure_counts),
        "mean_fuel_used_kg": float(np.mean([case["fuel_used_kg"] for case in cases])),
        "mean_absolute_touchdown_x_m": float(
            np.mean([abs(case["touchdown_x_m"]) for case in cases])
        ),
        "mean_max_throttle_rate_per_s": float(
            np.mean([case["max_throttle_rate_per_s"] for case in cases])
        ),
        "mean_max_gimbal_rate_deg_s": float(
            np.mean([case["max_gimbal_rate_deg_s"] for case in cases])
        ),
        "cases": cases,
    }
    return summary, episodes


def state_distribution_drift(
    train_states: NDArray[Any],
    rollout_states: NDArray[Any],
    *,
    seed: int = 1,
    reference_size: int = 5000,
) -> dict[str, float]:
    """Nearest-neighbor distance in train-normalized state units."""
    if len(rollout_states) == 0:
        raise ValueError("rollout must contain states")
    rng = np.random.default_rng(seed)
    reference = train_states[
        rng.choice(len(train_states), min(reference_size, len(train_states)), replace=False)
    ]
    distances, _ = cKDTree(reference).query(rollout_states)
    return {
        "median_distance": float(np.median(distances)),
        "p95_distance": float(np.percentile(distances, 95)),
        "fraction_over_2": float(np.mean(distances > 2.0)),
    }


def inference_latency_us(
    policy: BCPolicy, normalized_states: NDArray[Any], *, count: int = 1000
) -> dict[str, float]:
    if count <= 0 or len(normalized_states) == 0:
        raise ValueError("latency requires positive count and nonempty states")
    for state in normalized_states[: min(len(normalized_states), 50)]:
        policy.predict_normalized(state.reshape(1, 7))
    durations = np.empty(count, dtype=np.float64)
    for index in range(count):
        state = normalized_states[index % len(normalized_states)]
        started = perf_counter_ns()
        policy.predict_normalized(state.reshape(1, 7))
        durations[index] = (perf_counter_ns() - started) / 1000.0
    return {"median_us": float(np.median(durations)), "p95_us": float(np.percentile(durations, 95))}
