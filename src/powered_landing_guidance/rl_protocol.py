"""Frozen scenario sampling and PID pilot for residual-RL evaluation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, deque
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from powered_landing_guidance.config import load_config, validate_config
from powered_landing_guidance.controllers import IntegratedLandingController
from powered_landing_guidance.envs import RocketLandingEnv
from powered_landing_guidance.evaluation import integrated_controller_sha256

GROUPS = (
    "nominal",
    "initial_ood",
    "wind",
    "model_mismatch",
    "sensor_delay",
    "combined",
)
SPLITS = ("train", "development", "final_holdout")
RANGE_FIELDS = (
    "abs_x_m",
    "z_m",
    "vx_m_s",
    "vz_m_s",
    "theta_deg",
    "omega_deg_s",
    "mass_kg",
)
DISTURBANCE_FIELDS = frozenset(
    {
        "wind_m_s",
        "gust_m_s",
        "thrust_scale",
        "inertia_scale",
        "sensor_noise_scale",
        "observation_delay_physics_steps",
        "action_delay_physics_steps",
        "engine_lag_s",
    }
)
CSV_FIELDS = (
    "protocol_sha256",
    "split",
    "group",
    "level",
    "case_id",
    "pair_id",
    "case_index",
    "scenario_seed",
    "initial_x_m",
    "initial_z_m",
    "initial_vx_m_s",
    "initial_vz_m_s",
    "initial_theta_deg",
    "initial_omega_deg_s",
    "initial_mass_kg",
    "wind_m_s",
    "gust_m_s",
    "wind_direction",
    "gust_start_s",
    "thrust_scale",
    "inertia_scale",
    "sensor_noise_scale",
    "observation_delay_physics_steps",
    "action_delay_physics_steps",
    "engine_lag_s",
    "outcome",
    "failure_type",
    "safe_landing",
    "duration_s",
    "fuel_used_kg",
    "final_x_m",
    "final_z_m",
    "final_vx_m_s",
    "final_vz_m_s",
    "final_theta_deg",
    "final_omega_deg_s",
    "final_mass_kg",
)
_ROOT_FIELDS = {
    "schema_version",
    "id",
    "base_config",
    "base_config_sha256",
    "controller_sha256",
    "seed",
    "splits",
    "pilot_cases_per_level",
    "initial_ranges",
    "gust_start_s",
    "gust_duration_s",
    "groups",
    "selection",
}
_SELECTION = {
    "primary": "safe_landing_success_by_group",
    "nominal_regression_allowed": False,
    "group_regression_allowed": False,
    "stress_comparison": "paired_success_difference",
    "stress_ci": "stratified_paired_bootstrap_95",
    "stress_group_weighting": "equal_group_weight",
    "stress_ci_lower_bound_gt": 0.0,
    "fuel_priority": "after_robustness_gate",
    "final_holdout_access": "after_checkpoint_freeze_only",
}


@dataclass(frozen=True, slots=True)
class RLProtocol:
    settings: dict[str, Any]
    base_config: dict[str, Any]
    sha256: str


@dataclass(frozen=True, slots=True)
class ProtocolCase:
    case_id: str
    pair_id: str
    split: str
    group: str
    level: str
    index: int
    scenario_seed: int
    initial_state: NDArray[np.float64]
    disturbances: dict[str, float | int]


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _seed(*parts: object) -> int:
    identity = "|".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(identity.encode("utf-8")).digest()[:8], "little")


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not np.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _bounds(value: Any, label: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{label} must contain two bounds")
    low, high = (_finite_number(item, label) for item in value)
    if low >= high:
        raise ValueError(f"{label} must have low < high")
    return low, high


def validate_protocol(settings: dict[str, Any], base_config: dict[str, Any]) -> None:
    """Reject ambiguous scenario definitions before any simulation starts."""
    if not isinstance(settings, dict) or set(settings) != _ROOT_FIELDS:
        raise ValueError("RL protocol fields do not match schema v1")
    if settings["schema_version"] != 1:
        raise ValueError("unsupported RL protocol schema")
    if not isinstance(settings["id"], str) or not settings["id"].strip():
        raise ValueError("protocol id must be nonempty")
    for key in ("base_config_sha256", "controller_sha256"):
        value = settings[key]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
        ):
            raise ValueError(f"{key} must be a lowercase SHA-256 digest")
    if settings["controller_sha256"] != integrated_controller_sha256(base_config):
        raise ValueError("frozen PID controller digest does not match")
    if type(settings["seed"]) is not int or settings["seed"] != 378:
        raise ValueError("RL protocol seed must be 378")
    if settings["selection"] != _SELECTION:
        raise ValueError("checkpoint selection rules do not match protocol v1")

    splits = settings["splits"]
    if not isinstance(splits, dict) or set(splits) != set(SPLITS):
        raise ValueError("train, development, and final_holdout splits are required")
    stream_ids = []
    for name in SPLITS:
        spec = splits[name]
        if not isinstance(spec, dict) or set(spec) != {"stream_id", "episodes_per_group"}:
            raise ValueError(f"{name} split fields are invalid")
        stream_id, count = spec["stream_id"], spec["episodes_per_group"]
        if type(stream_id) is not int or stream_id < 0:
            raise ValueError(f"{name} stream_id must be a nonnegative integer")
        if not isinstance(count, int) or isinstance(count, bool) or count < 200:
            raise ValueError(f"{name} requires at least 200 cases per group")
        stream_ids.append(stream_id)
    if len(set(stream_ids)) != len(stream_ids):
        raise ValueError("split stream IDs must be distinct")
    pilot_count = settings["pilot_cases_per_level"]
    if (
        not isinstance(pilot_count, int)
        or isinstance(pilot_count, bool)
        or not 1 <= pilot_count <= splits["development"]["episodes_per_group"]
    ):
        raise ValueError("pilot_cases_per_level is invalid")

    ranges = settings["initial_ranges"]
    if not isinstance(ranges, dict) or not ranges:
        raise ValueError("initial_ranges must be nonempty")
    ground = float(base_config["simulation"]["ground_z_m"])
    dry_mass = float(base_config["vehicle"]["dry_mass_kg"])
    for name, spec in ranges.items():
        if (
            not isinstance(name, str)
            or not isinstance(spec, dict)
            or set(spec) != set(RANGE_FIELDS)
        ):
            raise ValueError(f"initial range {name} has invalid fields")
        parsed = {key: _bounds(spec[key], f"{name}.{key}") for key in RANGE_FIELDS}
        if parsed["abs_x_m"][0] < 0 or parsed["z_m"][0] < ground:
            raise ValueError(f"{name} starts outside the horizontal/altitude envelope")
        if parsed["mass_kg"][0] <= dry_mass or parsed["vz_m_s"][1] >= 0:
            raise ValueError(f"{name} needs propellant and a descending initial state")
    gust_start = _bounds(settings["gust_start_s"], "gust_start_s")
    if gust_start[0] < 0 or _finite_number(settings["gust_duration_s"], "gust_duration_s") <= 0:
        raise ValueError("gust timing must be nonnegative with positive duration")

    groups = settings["groups"]
    if not isinstance(groups, dict) or set(groups) != set(GROUPS):
        raise ValueError("exactly six named condition groups are required")
    for group in GROUPS:
        spec = groups[group]
        if not isinstance(spec, dict) or set(spec) != {"evaluation_level", "levels"}:
            raise ValueError(f"{group} group fields are invalid")
        levels = spec["levels"]
        if not isinstance(levels, list) or not levels:
            raise ValueError(f"{group} needs at least one level")
        names = []
        for level in levels:
            if not isinstance(level, dict) or set(level) != {
                "name",
                "initial_range",
                "disturbances",
            }:
                raise ValueError(f"{group} level fields are invalid")
            if not isinstance(level["name"], str) or not level["name"]:
                raise ValueError(f"{group} level name is invalid")
            names.append(level["name"])
            if level["initial_range"] not in ranges:
                raise ValueError(f"{group} references an unknown initial range")
            disturbance = level["disturbances"]
            if not isinstance(disturbance, dict) or set(disturbance) - DISTURBANCE_FIELDS:
                raise ValueError(f"{group} disturbances contain unknown fields")
            for key, value in disturbance.items():
                if key.endswith("_physics_steps"):
                    if (
                        not isinstance(value, int)
                        or isinstance(value, bool)
                        or not 0 <= value <= 100
                    ):
                        raise ValueError(f"{group}.{key} must be an integer in [0, 100]")
                elif key in {"thrust_scale", "inertia_scale"}:
                    if not 0.5 <= _finite_number(value, key) <= 2.0:
                        raise ValueError(f"{group}.{key} must be in [0.5, 2.0]")
                elif _finite_number(value, key) < 0:
                    raise ValueError(f"{group}.{key} must be nonnegative")
        if len(names) != len(set(names)) or spec["evaluation_level"] not in names:
            raise ValueError(f"{group} level names or evaluation level are invalid")
    if settings["groups"]["nominal"]["levels"] != [
        {"name": "clean", "initial_range": "easy", "disturbances": {}}
    ]:
        raise ValueError("nominal group must remain an unperturbed easy baseline")


def load_protocol(path: str | Path) -> RLProtocol:
    """Load a checked protocol without sampling or opening holdout cases."""
    protocol_path = Path(path).resolve()
    with protocol_path.open(encoding="utf-8") as stream:
        settings = yaml.safe_load(stream)
    if not isinstance(settings, dict) or not isinstance(settings.get("base_config"), str):
        raise ValueError("protocol must name a base configuration")
    root = protocol_path.parent.parent
    base_path = (root / settings["base_config"]).resolve()
    if not base_path.is_relative_to(root):
        raise ValueError("base_config must stay within the project")
    if hashlib.sha256(base_path.read_bytes()).hexdigest() != settings["base_config_sha256"]:
        raise ValueError("base configuration file digest does not match")
    base_config = load_config(base_path)
    validate_protocol(settings, base_config)
    canonical = json.dumps(settings, sort_keys=True, separators=(",", ":"), allow_nan=False)
    digest = _hash_text("powered-landing-guidance:rl-protocol:v1\n" + canonical)
    return RLProtocol(settings=settings, base_config=base_config, sha256=digest)


def _level_spec(protocol: RLProtocol, group: str, level: str | None) -> dict[str, Any]:
    if group not in GROUPS:
        raise ValueError(f"unknown condition group: {group}")
    spec = protocol.settings["groups"][group]
    requested = spec["evaluation_level"] if level is None else level
    for candidate in spec["levels"]:
        if candidate["name"] == requested:
            return candidate
    raise ValueError(f"unknown level {requested} for {group}")


def sample_cases(
    protocol: RLProtocol,
    split: str,
    group: str,
    *,
    level: str | None = None,
    count: int | None = None,
    unlock_final_holdout: bool = False,
) -> list[ProtocolCase]:
    """Sample stable case IDs and paired quantiles; seal the final split by default."""
    if split not in SPLITS:
        raise ValueError(f"unknown split: {split}")
    if split == "final_holdout" and not unlock_final_holdout:
        raise PermissionError("final holdout is sealed until the checkpoint is frozen")
    level_spec = _level_spec(protocol, group, level)
    split_spec = protocol.settings["splits"][split]
    available = split_spec["episodes_per_group"]
    selected = available if count is None else count
    if (
        not isinstance(selected, int)
        or isinstance(selected, bool)
        or not 1 <= selected <= available
    ):
        raise ValueError("case count must be within the split budget")
    seed = _seed(protocol.settings["seed"], protocol.sha256, split, group, split_spec["stream_id"])
    rng = np.random.default_rng(seed)
    quantiles = rng.random((available, 7))[:selected]
    signs = rng.choice(np.asarray((-1.0, 1.0)), size=available)[:selected]
    bounds = protocol.settings["initial_ranges"][level_spec["initial_range"]]
    lows = np.asarray([bounds[key][0] for key in RANGE_FIELDS], dtype=np.float64)
    highs = np.asarray([bounds[key][1] for key in RANGE_FIELDS], dtype=np.float64)
    states = lows + quantiles * (highs - lows)
    states[:, 0] *= signs
    states[:, 4:6] = np.deg2rad(states[:, 4:6])
    cases = []
    for index, state in enumerate(states):
        identity = f"{protocol.sha256}|{split}|{group}|{index}"
        pair_id = _hash_text("pair|" + identity + "|" + level_spec["initial_range"])[:24]
        case_id = _hash_text("case|" + identity + "|" + level_spec["name"])[:24]
        cases.append(
            ProtocolCase(
                case_id=case_id,
                pair_id=pair_id,
                split=split,
                group=group,
                level=level_spec["name"],
                index=index,
                scenario_seed=_seed("scenario", protocol.settings["seed"], identity),
                initial_state=state.copy(),
                disturbances=dict(level_spec["disturbances"]),
            )
        )
    return cases


def _sensor_deviations(config: dict[str, Any]) -> NDArray[np.float64]:
    settings = config["disturbance_evaluation"]["sensor_noise"]["standard_deviations"]
    return np.asarray(
        [
            settings["x_m"],
            settings["z_m"],
            settings["vx_m_s"],
            settings["vz_m_s"],
            np.deg2rad(settings["theta_deg"]),
            np.deg2rad(settings["omega_deg_s"]),
            settings["mass_kg"],
        ],
        dtype=np.float64,
    )


def _failure_type(outcome: str, final: NDArray[np.float64], config: dict[str, Any]) -> str:
    if outcome in {"success", "fuel_depletion", "timeout"}:
        return outcome
    limits = config["landing_success"]
    theta = np.arctan2(np.sin(final[4]), np.cos(final[4]))
    if abs(final[0]) > limits["max_abs_x_m"]:
        return "position_error"
    if abs(theta) > np.deg2rad(limits["max_abs_theta_deg"]) or abs(final[5]) > np.deg2rad(
        limits["max_abs_omega_deg_s"]
    ):
        return "attitude_error"
    if abs(final[2]) > limits["max_abs_vx_m_s"]:
        return "horizontal_speed"
    if abs(final[3]) > limits["max_abs_vz_m_s"]:
        return "hard_touchdown"
    return "other_failure"


class ProtocolCasePlant:
    """Advance one frozen case at the physics rate with PID sensor/actuator effects."""

    def __init__(self, protocol: RLProtocol, case: ProtocolCase) -> None:
        self.protocol = protocol
        self.case = case
        base = protocol.base_config
        disturbance = case.disturbances
        actual = deepcopy(base)
        self.thrust_scale = float(disturbance.get("thrust_scale", 1.0))
        self.inertia_scale = float(disturbance.get("inertia_scale", 1.0))
        actual["vehicle"]["max_thrust_n"] *= self.thrust_scale
        actual["vehicle"]["moment_of_inertia_kg_m2"] *= self.inertia_scale
        validate_config(actual)
        scenario_rng = np.random.default_rng(case.scenario_seed)
        self.wind_direction = int(scenario_rng.choice(np.asarray((-1, 1))))
        gust_low, gust_high = protocol.settings["gust_start_s"]
        self.gust_start_s = float(scenario_rng.uniform(gust_low, gust_high))
        self.constant_wind = float(disturbance.get("wind_m_s", 0.0))
        self.gust = float(disturbance.get("gust_m_s", 0.0))
        gust_duration = float(protocol.settings["gust_duration_s"])

        def wind(time_s: float) -> NDArray[np.float64]:
            phase = (time_s - self.gust_start_s) / gust_duration
            gust_speed = self.gust * np.sin(np.pi * phase) if 0.0 <= phase <= 1.0 else 0.0
            speed = self.wind_direction * (self.constant_wind + gust_speed)
            return np.asarray((speed, 0.0), dtype=np.float64)

        self.env = RocketLandingEnv(
            actual, wind_velocity_m_s=wind if self.constant_wind or self.gust else None
        )
        self.controller = IntegratedLandingController.from_config(base)
        self.observation_delay = int(disturbance.get("observation_delay_physics_steps", 0))
        self.action_delay = int(disturbance.get("action_delay_physics_steps", 0))
        self.noise_scale = float(disturbance.get("sensor_noise_scale", 0.0))
        self.lag_s = float(disturbance.get("engine_lag_s", 0.0))
        self.deviations = _sensor_deviations(base) * self.noise_scale
        self.ground = float(base["simulation"]["ground_z_m"])
        self.dry_mass = float(base["vehicle"]["dry_mass_kg"])
        self.dt_s = float(base["simulation"]["dt_s"])
        self.state: NDArray[np.float64] | None = None
        self.observed: NDArray[np.float64] | None = None
        self.pid_action: NDArray[np.float64] | None = None
        self.info: dict[str, Any] = {}
        self.terminated = False
        self.truncated = False
        self.applied_throttle = 0.0
        self.noise_rng = np.random.default_rng(_seed(case.scenario_seed, "sensor"))
        self.observations: deque[NDArray[np.float64]] = deque()
        self.queued_actions: deque[NDArray[np.float64]] = deque()

    def reset(self) -> NDArray[np.float64]:
        """Restore the case and all controller, delay, lag, and sensor state."""
        self.controller.reset()
        self.noise_rng = np.random.default_rng(_seed(self.case.scenario_seed, "sensor"))
        self.applied_throttle = 0.0
        self.queued_actions = deque(
            (np.zeros(2, dtype=np.float64) for _ in range(self.action_delay)),
            maxlen=self.action_delay + 1,
        )
        self.state, self.info = self.env.reset(options={"initial_state": self.case.initial_state})
        self.observations = deque(
            (self.state.copy() for _ in range(self.observation_delay)),
            maxlen=self.observation_delay + 1,
        )
        self.terminated = self.truncated = False
        self._prepare_control()
        return self.state.copy()

    def _prepare_control(self) -> None:
        assert self.state is not None
        self.observations.append(self.state.copy())
        observed = self.observations[0].copy()
        if self.noise_scale:
            observed += self.noise_rng.normal(0.0, self.deviations)
            observed[1] = max(observed[1], self.ground)
            observed[6] = max(observed[6], self.dry_mass)
        self.observed = observed
        self.pid_action = self.controller.command(observed, time_s=float(self.info["time_s"]))

    def step_physics(
        self, command: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], float, bool, bool, dict[str, Any]]:
        if self.state is None or self.terminated or self.truncated:
            raise RuntimeError("reset the protocol case before stepping")
        self.queued_actions.append(np.asarray(command, dtype=np.float64).copy())
        applied = self.queued_actions[0].copy()
        if self.lag_s:
            self.applied_throttle += -np.expm1(-self.dt_s / self.lag_s) * (
                float(applied[0]) - self.applied_throttle
            )
            applied[0] = self.applied_throttle
        else:
            self.applied_throttle = float(applied[0])
        self.state, reward, self.terminated, self.truncated, self.info = self.env.step(applied)
        if not (self.terminated or self.truncated):
            self._prepare_control()
        return self.state.copy(), reward, self.terminated, self.truncated, dict(self.info)

    def close(self) -> None:
        self.env.close()


def run_pid_case(protocol: RLProtocol, case: ProtocolCase) -> dict[str, Any]:
    """Run a paired PID case with fixed physical and sensor disturbances."""
    plant = ProtocolCasePlant(protocol, case)
    try:
        plant.reset()
        while not (plant.terminated or plant.truncated):
            assert plant.pid_action is not None
            plant.step_physics(plant.pid_action)
    finally:
        plant.close()

    initial = case.initial_state
    assert plant.state is not None
    state = plant.state
    info = plant.info
    outcome = str(info["outcome"])
    row = {
        "protocol_sha256": protocol.sha256,
        "split": case.split,
        "group": case.group,
        "level": case.level,
        "case_id": case.case_id,
        "pair_id": case.pair_id,
        "case_index": case.index,
        "scenario_seed": case.scenario_seed,
        "initial_x_m": float(initial[0]),
        "initial_z_m": float(initial[1]),
        "initial_vx_m_s": float(initial[2]),
        "initial_vz_m_s": float(initial[3]),
        "initial_theta_deg": float(np.rad2deg(initial[4])),
        "initial_omega_deg_s": float(np.rad2deg(initial[5])),
        "initial_mass_kg": float(initial[6]),
        "wind_m_s": plant.constant_wind,
        "gust_m_s": plant.gust,
        "wind_direction": plant.wind_direction,
        "gust_start_s": plant.gust_start_s,
        "thrust_scale": plant.thrust_scale,
        "inertia_scale": plant.inertia_scale,
        "sensor_noise_scale": plant.noise_scale,
        "observation_delay_physics_steps": plant.observation_delay,
        "action_delay_physics_steps": plant.action_delay,
        "engine_lag_s": plant.lag_s,
        "outcome": outcome,
        "failure_type": _failure_type(outcome, state, protocol.base_config),
        "safe_landing": int(outcome == "success"),
        "duration_s": float(info["time_s"]),
        "fuel_used_kg": float(info["fuel_used_kg"]),
        "final_x_m": float(state[0]),
        "final_z_m": float(state[1]),
        "final_vx_m_s": float(state[2]),
        "final_vz_m_s": float(state[3]),
        "final_theta_deg": float(np.rad2deg(state[4])),
        "final_omega_deg_s": float(np.rad2deg(state[5])),
        "final_mass_kg": float(state[6]),
    }
    if set(row) != set(CSV_FIELDS):
        raise RuntimeError("PID result does not match the frozen CSV schema")
    return row


def pilot_report(protocol: RLProtocol, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize exploratory PID-only levels without claiming final-test performance."""
    levels = []
    for group in GROUPS:
        for level in protocol.settings["groups"][group]["levels"]:
            subset = [
                row for row in rows if row["group"] == group and row["level"] == level["name"]
            ]
            failures = Counter(row["failure_type"] for row in subset)
            levels.append(
                {
                    "group": group,
                    "level": level["name"],
                    "episodes": len(subset),
                    "safe_landings": failures["success"],
                    "success_rate": failures["success"] / len(subset),
                    "failure_counts": dict(sorted(failures.items())),
                    "mean_fuel_used_kg_all_outcomes": float(
                        np.mean([row["fuel_used_kg"] for row in subset])
                    ),
                }
            )
    rows_json = json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {
        "schema_version": 1,
        "protocol_id": protocol.settings["id"],
        "protocol_sha256": protocol.sha256,
        "base_config_sha256": protocol.settings["base_config_sha256"],
        "controller_sha256": protocol.settings["controller_sha256"],
        "root_seed": protocol.settings["seed"],
        "split_stream_ids": {
            name: protocol.settings["splits"][name]["stream_id"] for name in SPLITS
        },
        "evaluation_episodes_per_group": {
            name: protocol.settings["splits"][name]["episodes_per_group"] for name in SPLITS
        },
        "pilot_split": "development",
        "pilot_rows": len(rows),
        "pilot_rows_sha256": _hash_text(rows_json),
        "final_holdout": "sealed_not_materialized",
        "historical_day35_test": "report_only_not_for_model_selection",
        "landing_success_limits": deepcopy(protocol.base_config["landing_success"]),
        "initial_ranges": deepcopy(protocol.settings["initial_ranges"]),
        "condition_groups": deepcopy(protocol.settings["groups"]),
        "gust_start_s": list(protocol.settings["gust_start_s"]),
        "gust_duration_s": protocol.settings["gust_duration_s"],
        "checkpoint_selection": deepcopy(protocol.settings["selection"]),
        "csv_fields": list(CSV_FIELDS),
        "levels": levels,
    }
