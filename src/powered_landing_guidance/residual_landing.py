"""PID-residual landing environment with a conservative command fallback."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from time import perf_counter_ns
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from powered_landing_guidance.dynamics import PlanarDynamicsParameters
from powered_landing_guidance.rl_protocol import ProtocolCase, ProtocolCasePlant, RLProtocol


@dataclass(frozen=True, slots=True)
class SafetyDecision:
    command: NDArray[np.float64]
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ResidualSafetyFilter:
    """Reject hazardous-looking corrections; this is not a safety guarantee."""

    throttle_limit: float = 0.08
    gimbal_limit_rad: float = np.deg2rad(2.0)
    fuel_reserve_kg: float = 1.0
    attitude_limit_rad: float = np.deg2rad(20.0)
    ground_margin_m: float = 0.5

    def __post_init__(self) -> None:
        values = (
            self.throttle_limit,
            self.gimbal_limit_rad,
            self.fuel_reserve_kg,
            self.attitude_limit_rad,
            self.ground_margin_m,
        )
        if not all(np.isfinite(value) and value > 0 for value in values):
            raise ValueError("safety filter limits must be finite and positive")

    def _ground_margin(
        self,
        observed: NDArray[np.float64],
        command: NDArray[np.float64],
        parameters: PlanarDynamicsParameters,
        interval_s: float,
        ground_z_m: float,
    ) -> float:
        mass = max(float(observed[6]), parameters.dry_mass_kg)
        thrust = parameters.max_thrust_n * float(command[0])
        acceleration = thrust * np.cos(observed[4] + command[1]) / mass
        acceleration -= parameters.gravity_m_s2
        next_z = observed[1] + observed[3] * interval_s + 0.5 * acceleration * interval_s**2
        next_vz = observed[3] + acceleration * interval_s
        max_deceleration = parameters.max_thrust_n / mass - parameters.gravity_m_s2
        if max_deceleration <= 0:
            return -np.inf
        stopping_distance = max(0.0, (-next_vz) ** 2 - 1.0) / (2.0 * max_deceleration)
        return float(next_z - ground_z_m - stopping_distance)

    def _attitude_projection(
        self,
        observed: NDArray[np.float64],
        command: NDArray[np.float64],
        parameters: PlanarDynamicsParameters,
        interval_s: float,
    ) -> float:
        torque = -parameters.engine_lever_arm_m * parameters.max_thrust_n
        torque *= float(command[0]) * np.sin(command[1])
        angular_acceleration = torque / parameters.moment_of_inertia_kg_m2
        return float(
            observed[4] + observed[5] * interval_s + 0.5 * angular_acceleration * interval_s**2
        )

    def check(
        self,
        residual: Any,
        pid_action: NDArray[np.float64],
        observed: NDArray[np.float64],
        previous_action: NDArray[np.float64] | None,
        parameters: PlanarDynamicsParameters,
        interval_s: float,
        ground_z_m: float,
        slew_rates: NDArray[np.float64],
    ) -> SafetyDecision:
        baseline = pid_action.copy()
        try:
            normalized = np.asarray(residual, dtype=np.float64)
        except (TypeError, ValueError, OverflowError):
            return SafetyDecision(baseline, ("invalid_residual",))
        if normalized.shape != (2,):
            return SafetyDecision(baseline, ("invalid_residual_shape",))
        if not np.all(np.isfinite(normalized)):
            return SafetyDecision(baseline, ("nonfinite_residual",))
        if np.any(np.abs(normalized) > 1.0):
            return SafetyDecision(baseline, ("residual_bounds",))
        if np.all(normalized == 0.0):
            return SafetyDecision(baseline, ())

        candidate = baseline + normalized * np.asarray(
            (self.throttle_limit, self.gimbal_limit_rad), dtype=np.float64
        )
        if not np.all(np.isfinite(candidate)):
            return SafetyDecision(baseline, ("nonfinite_command",))
        if (
            candidate[0] < parameters.throttle_min
            or candidate[0] > parameters.throttle_max
            or abs(candidate[1]) > parameters.gimbal_limit_rad
        ):
            return SafetyDecision(baseline, ("actuator_limit",))
        if previous_action is not None and np.any(
            np.abs(candidate - previous_action) > slew_rates * interval_s + 1e-12
        ):
            return SafetyDecision(baseline, ("slew_limit",))

        fuel_rate = parameters.max_thrust_n / (
            parameters.specific_impulse_s * parameters.standard_gravity_m_s2
        )
        fuel_left = observed[6] - parameters.dry_mass_kg
        if (
            fuel_left - candidate[0] * fuel_rate * interval_s < self.fuel_reserve_kg
            and candidate[0] > baseline[0]
        ):
            return SafetyDecision(baseline, ("fuel_margin",))

        candidate_theta = abs(
            self._attitude_projection(observed, candidate, parameters, interval_s)
        )
        baseline_theta = abs(self._attitude_projection(observed, baseline, parameters, interval_s))
        if candidate_theta > self.attitude_limit_rad and candidate_theta > baseline_theta + 1e-12:
            return SafetyDecision(baseline, ("attitude_margin",))

        candidate_ground = self._ground_margin(
            observed, candidate, parameters, interval_s, ground_z_m
        )
        baseline_ground = self._ground_margin(
            observed, baseline, parameters, interval_s, ground_z_m
        )
        if candidate_ground < self.ground_margin_m and candidate_ground < baseline_ground - 1e-12:
            return SafetyDecision(baseline, ("ground_margin",))
        return SafetyDecision(candidate, ())


class ResidualLandingEnv(gym.Env):
    """One RL action per PID update, five physics steps per action by default.

    Observations contain delayed/noisy state, PID command, PID memory, phase,
    pending actuator command, and engine-lag throttle, scaled to typical ranges.
    The base reward is sparse; training can wrap this environment separately.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        protocol: RLProtocol,
        case: ProtocolCase,
        *,
        allow_final_holdout: bool = False,
        safety_filter: ResidualSafetyFilter | None = None,
    ) -> None:
        super().__init__()
        if case.split == "final_holdout" and not allow_final_holdout:
            raise PermissionError("final holdout is sealed until checkpoint freeze")
        self.protocol = protocol
        self.case = case
        self.plant = ProtocolCasePlant(protocol, case)
        self.safety_filter = safety_filter or ResidualSafetyFilter()
        controller = self.plant.controller
        self.control_interval_s = controller.control_interval_s
        self.physics_steps_per_action = round(self.control_interval_s / self.plant.dt_s)
        self.slew_rates = np.asarray(
            (controller.throttle_slew_rate_per_s, controller.gimbal_slew_rate_rad_s)
        )
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(17,), dtype=np.float32)
        self.previous_action: NDArray[np.float64] | None = None
        self.control_updates = 0
        self.interventions = 0
        self.intervention_reasons: Counter[str] = Counter()
        self.intervention_axes: Counter[str] = Counter()
        self.safety_times_us: list[float] = []

    def _observation(self) -> NDArray[np.float32]:
        plant = self.plant
        assert plant.observed is not None and plant.pid_action is not None
        controller = plant.controller
        observed = plant.observed.copy()
        observed[1] -= plant.ground
        observed[6] -= plant.dry_mass
        state_scale = np.asarray(
            (30.0, 160.0, 10.0, 40.0, np.deg2rad(20.0), np.deg2rad(30.0), 250.0)
        )
        gimbal_limit = plant.env.parameters.gimbal_limit_rad
        command_scale = np.asarray((1.0, gimbal_limit))
        pending = plant.queued_actions[0] if plant.queued_actions else np.zeros(2)
        integral_limit = controller.approach_vertical.integral_limit_m
        memory = np.asarray(
            (
                controller.approach_vertical.integral_error_m / integral_limit,
                controller.terminal_vertical.integral_error_m / integral_limit,
                (controller.approach_vertical.previous_error_m_s or 0.0) / 40.0,
                (controller.terminal_vertical.previous_error_m_s or 0.0) / 40.0,
                float(controller.phase == "terminal"),
            )
        )
        return np.concatenate(
            (
                observed / state_scale,
                plant.pid_action / command_scale,
                memory,
                pending / command_scale,
                np.asarray((plant.applied_throttle,)),
            )
        ).astype(np.float32)

    def _info(self) -> dict[str, Any]:
        plant = self.plant
        return {
            **plant.info,
            "case_id": self.case.case_id,
            "group": self.case.group,
            "level": self.case.level,
            "true_state": None if plant.state is None else plant.state.copy(),
            "observed_state": None if plant.observed is None else plant.observed.copy(),
            "pid_action": None if plant.pid_action is None else plant.pid_action.copy(),
            "control_updates": self.control_updates,
            "interventions": self.interventions,
            "intervention_rate": (
                self.interventions / self.control_updates if self.control_updates else 0.0
            ),
            "intervention_reason_counts": dict(self.intervention_reasons),
            "intervention_axis_counts": dict(self.intervention_axes),
        }

    @property
    def safety_timing_us(self) -> dict[str, float]:
        """Wall-clock diagnostics kept outside deterministic Gymnasium step info."""
        if not self.safety_times_us:
            return {"mean": 0.0, "p95": 0.0, "max": 0.0}
        return {
            "mean": float(np.mean(self.safety_times_us)),
            "p95": float(np.percentile(self.safety_times_us, 95)),
            "max": float(np.max(self.safety_times_us)),
        }

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)
        if options:
            raise ValueError("this environment runs its assigned protocol case only")
        self.plant.reset()
        self.previous_action = None
        self.control_updates = 0
        self.interventions = 0
        self.intervention_reasons.clear()
        self.intervention_axes.clear()
        self.safety_times_us.clear()
        return self._observation(), self._info()

    def step(self, action: Any) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        plant = self.plant
        if plant.state is None or plant.terminated or plant.truncated:
            raise RuntimeError("reset the residual environment before stepping")
        assert plant.observed is not None and plant.pid_action is not None
        baseline_action = plant.pid_action.copy()
        start_ns = perf_counter_ns()
        decision = self.safety_filter.check(
            action,
            plant.pid_action,
            plant.observed,
            self.previous_action,
            plant.controller.parameters,
            self.control_interval_s,
            plant.ground,
            self.slew_rates,
        )
        elapsed_us = (perf_counter_ns() - start_ns) / 1000.0
        self.safety_times_us.append(elapsed_us)
        self.control_updates += 1
        if decision.reasons:
            self.interventions += 1
            self.intervention_reasons.update(decision.reasons)
            try:
                normalized = np.asarray(action, dtype=np.float64)
            except (TypeError, ValueError, OverflowError):
                normalized = np.empty(0)
            if normalized.shape == (2,) and np.all(np.isfinite(normalized)):
                candidate = baseline_action + normalized * np.asarray(
                    (self.safety_filter.throttle_limit, self.safety_filter.gimbal_limit_rad)
                )
                if "slew_limit" in decision.reasons and self.previous_action is not None:
                    exceeded = np.abs(candidate - self.previous_action)
                    exceeded = exceeded > self.slew_rates * self.control_interval_s + 1e-12
                    for axis, outside in zip(("throttle", "gimbal"), exceeded, strict=True):
                        if outside:
                            self.intervention_axes[f"slew_{axis}"] += 1
                if "actuator_limit" in decision.reasons:
                    params = plant.controller.parameters
                    if candidate[0] < params.throttle_min or candidate[0] > params.throttle_max:
                        self.intervention_axes["actuator_throttle"] += 1
                    if abs(candidate[1]) > params.gimbal_limit_rad:
                        self.intervention_axes["actuator_gimbal"] += 1
        self.previous_action = decision.command.copy()
        reward = 0.0
        for _ in range(self.physics_steps_per_action):
            _, step_reward, terminated, truncated, _ = plant.step_physics(decision.command)
            reward += step_reward
            if terminated or truncated:
                break
        info = self._info()
        try:
            proposed_array = np.asarray(action, dtype=np.float64)
        except (TypeError, ValueError, OverflowError):
            proposed = repr(action)
        else:
            proposed = (
                proposed_array.copy()
                if proposed_array.shape == (2,) and np.all(np.isfinite(proposed_array))
                else repr(action)
            )
        info.update(
            proposed_residual=proposed,
            baseline_action=baseline_action,
            chosen_action=decision.command.copy(),
            intervention_reasons=list(decision.reasons),
        )
        return (
            self._observation(),
            reward,
            plant.terminated,
            plant.truncated,
            info,
        )

    def close(self) -> None:
        self.plant.close()
