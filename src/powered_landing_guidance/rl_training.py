"""Landing-first reward and train-only curriculum for residual control."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from powered_landing_guidance.residual_landing import ResidualLandingEnv, ResidualSafetyFilter
from powered_landing_guidance.rl_protocol import ProtocolCase, RLProtocol, sample_cases

CURRICULUM_STAGES: dict[str, tuple[tuple[str, str, float], ...]] = {
    "nominal": (("nominal", "clean", 1.0),),
    "initial_ood": (
        ("nominal", "clean", 0.4),
        ("initial_ood", "expanded", 0.3),
        ("initial_ood", "hard", 0.3),
    ),
    "wind": (
        ("nominal", "clean", 0.3),
        ("initial_ood", "hard", 0.3),
        ("wind", "constant_15", 0.2),
        ("wind", "constant_30", 0.2),
    ),
    "model_mismatch": (
        ("nominal", "clean", 0.2),
        ("initial_ood", "hard", 0.2),
        ("wind", "constant_30", 0.3),
        ("model_mismatch", "light", 0.3),
    ),
    "sensor_delay": (
        ("nominal", "clean", 0.2),
        ("initial_ood", "hard", 0.1),
        ("wind", "constant_30", 0.2),
        ("model_mismatch", "light", 0.15),
        ("sensor_delay", "medium", 0.35),
    ),
    "single_disturbance": (
        ("nominal", "clean", 0.2),
        ("initial_ood", "hard", 0.2),
        ("wind", "constant_30", 0.2),
        ("model_mismatch", "light", 0.2),
        ("sensor_delay", "medium", 0.2),
    ),
    "combined": (
        ("nominal", "clean", 0.15),
        ("initial_ood", "hard", 0.15),
        ("wind", "constant_30", 0.15),
        ("model_mismatch", "light", 0.15),
        ("sensor_delay", "medium", 0.15),
        ("combined", "light", 0.15),
        ("combined", "medium", 0.10),
    ),
}


def landing_potential(state: NDArray[np.float64], ground_z_m: float) -> float:
    """Bounded progress potential; no mass term or fuel-saving incentive."""
    x, z, vx, vz, theta, omega, _mass = state
    terms = (
        3.0 * min(abs(x) / 30.0, 1.0),
        2.0 * min(max(z - ground_z_m, 0.0) / 160.0, 1.0),
        2.0 * min(abs(vx) / 10.0, 1.0),
        2.0 * min(abs(vz + 1.0) / 40.0, 1.0),
        min(abs(theta) / np.deg2rad(20.0), 1.0),
        min(abs(omega) / np.deg2rad(30.0), 1.0),
    )
    return -float(sum(terms))


def landing_reward(
    previous: NDArray[np.float64],
    current: NDArray[np.float64],
    *,
    ground_z_m: float,
    elapsed_s: float,
    outcome: str,
) -> tuple[float, dict[str, float]]:
    """Keep any physical failure below any successful touchdown reward."""
    if elapsed_s < 0 or not np.isfinite(elapsed_s):
        raise ValueError("elapsed_s must be finite and nonnegative")
    progress = landing_potential(current, ground_z_m) - landing_potential(previous, ground_z_m)
    time_cost = -0.1 * elapsed_s
    terminal = {
        "success": 100.0,
        "hard_landing": -120.0,
        "crash": -150.0,
        "fuel_depletion": -130.0,
        "timeout": -130.0,
        "running": 0.0,
    }
    if outcome not in terminal:
        raise ValueError(f"unknown landing outcome: {outcome}")
    parts = {"terminal": terminal[outcome], "progress": progress, "time": time_cost}
    return float(sum(parts.values())), parts


class LandingRewardWrapper(gym.Wrapper):
    """Replace sparse training reward without modifying the physical outcome."""

    def __init__(self, env: ResidualLandingEnv) -> None:
        super().__init__(env)
        self.previous_state: NDArray[np.float64] | None = None
        self.previous_time_s = 0.0

    def reset(self, **kwargs: Any):
        observation, info = self.env.reset(**kwargs)
        self.previous_state = np.asarray(info["true_state"], dtype=np.float64).copy()
        self.previous_time_s = float(info["time_s"])
        return observation, info

    def step(self, action: NDArray[np.float32]):
        if self.previous_state is None:
            raise RuntimeError("reset the reward wrapper before stepping")
        observation, _sparse, terminated, truncated, info = self.env.step(action)
        current = np.asarray(info["true_state"], dtype=np.float64)
        current_time = float(info["time_s"])
        reward, parts = landing_reward(
            self.previous_state,
            current,
            ground_z_m=self.env.plant.ground,
            elapsed_s=current_time - self.previous_time_s,
            outcome=str(info["outcome"]),
        )
        self.previous_state = current.copy()
        self.previous_time_s = current_time
        info["reward_components"] = parts
        return observation, reward, terminated, truncated, info


class CurriculumResidualEnv(gym.Env):
    """Sample only training IDs while broadening conditions by named stages."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        protocol: RLProtocol,
        *,
        stage: str = "nominal",
        safety_filter: ResidualSafetyFilter | None = None,
    ) -> None:
        super().__init__()
        self.protocol = protocol
        self.safety_filter = safety_filter
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(17,), dtype=np.float32)
        self.cases: dict[tuple[str, str], list[ProtocolCase]] = {}
        for entries in CURRICULUM_STAGES.values():
            for group, level, _weight in entries:
                key = (group, level)
                if key not in self.cases:
                    self.cases[key] = sample_cases(protocol, "train", group, level=level)
        self.stage = ""
        self.set_stage(stage)
        self.active_stage = ""
        self.rng = np.random.default_rng(protocol.settings["seed"])
        self.episode_index = 0
        self.current: LandingRewardWrapper | None = None

    def set_stage(self, stage: str) -> None:
        if stage not in CURRICULUM_STAGES:
            raise ValueError(f"unknown curriculum stage: {stage}")
        self.stage = stage

    def sampler_state(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "episode_index": self.episode_index,
            "rng_state": self.rng.bit_generator.state,
        }

    def restore_sampler_state(self, state: dict[str, Any]) -> None:
        self.set_stage(str(state["stage"]))
        self.episode_index = int(state["episode_index"])
        self.rng.bit_generator.state = state["rng_state"]

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)
        if options:
            raise ValueError("curriculum cases are sampled internally")
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.episode_index = 0
        if self.current is not None:
            self.current.close()
        entries = CURRICULUM_STAGES[self.stage]
        weights = np.asarray([entry[2] for entry in entries], dtype=np.float64)
        choice = int(self.rng.choice(len(entries), p=weights))
        group, level, _ = entries[choice]
        candidates = self.cases[(group, level)]
        case = candidates[int(self.rng.integers(len(candidates)))]
        self.current = LandingRewardWrapper(
            ResidualLandingEnv(self.protocol, case, safety_filter=self.safety_filter)
        )
        observation, info = self.current.reset()
        self.episode_index += 1
        self.active_stage = self.stage
        info["curriculum_stage"] = self.active_stage
        info["curriculum_episode"] = self.episode_index
        return observation, info

    def step(self, action: NDArray[np.float32]):
        if self.current is None:
            raise RuntimeError("reset the curriculum environment before stepping")
        observation, reward, terminated, truncated, info = self.current.step(action)
        info["curriculum_stage"] = self.active_stage
        info["curriculum_episode"] = self.episode_index
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        if self.current is not None:
            self.current.close()
            self.current = None
        super().close()
