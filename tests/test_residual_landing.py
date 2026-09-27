"""Day 37-38 zero-residual equivalence and conservative fallback checks."""

from __future__ import annotations

import json
import sys
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from powered_landing_guidance.dynamics import PlanarDynamicsParameters
from powered_landing_guidance.residual_landing import ResidualLandingEnv, ResidualSafetyFilter
from powered_landing_guidance.rl_protocol import (
    GROUPS,
    ProtocolCasePlant,
    load_protocol,
    sample_cases,
)
from scripts.verify_residual_landing import main, run_probe

PROTOCOL_PATH = Path(__file__).resolve().parents[1] / "configs/rl-evaluation-v1.yaml"


@pytest.fixture(scope="module")
def protocol():
    return load_protocol(PROTOCOL_PATH)


@pytest.mark.parametrize("group", GROUPS)
def test_zero_residual_matches_every_pid_physics_step(protocol, group: str) -> None:
    case = sample_cases(protocol, "development", group, count=1)[0]
    baseline = ProtocolCasePlant(protocol, case)
    residual_env = ResidualLandingEnv(protocol, case)
    try:
        baseline.reset()
        observation, info = residual_env.reset(seed=378)
        assert observation.shape == (17,)
        assert observation.dtype == np.float32
        assert residual_env.observation_space.contains(observation)
        assert info["case_id"] == case.case_id
        assert residual_env.physics_steps_per_action == 5
        while True:
            for _ in range(residual_env.physics_steps_per_action):
                assert baseline.pid_action is not None
                baseline.step_physics(baseline.pid_action)
                if baseline.terminated or baseline.truncated:
                    break
            observation, _, terminated, truncated, info = residual_env.step(
                np.zeros(2, dtype=np.float32)
            )
            np.testing.assert_allclose(residual_env.plant.state, baseline.state, atol=1e-12)
            np.testing.assert_array_equal(info["chosen_action"], info["baseline_action"])
            assert info["intervention_reasons"] == []
            assert residual_env.observation_space.contains(observation)
            assert (terminated, truncated) == (baseline.terminated, baseline.truncated)
            if terminated or truncated:
                break
        actual_steps = residual_env.plant.env.episode_log["steps"]
        baseline_steps = baseline.env.episode_log["steps"]
        assert len(actual_steps) == len(baseline_steps)
        np.testing.assert_allclose(
            [entry["state"] for entry in actual_steps],
            [entry["state"] for entry in baseline_steps],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            [entry["clipped_action"] for entry in actual_steps],
            [entry["clipped_action"] for entry in baseline_steps],
            atol=1e-12,
        )
        assert info["interventions"] == 0
        assert info["outcome"] == baseline.info["outcome"]
    finally:
        baseline.close()
        residual_env.close()


def test_noisy_delayed_case_resets_deterministically(protocol) -> None:
    case = sample_cases(protocol, "train", "sensor_delay", count=1)[0]
    env = ResidualLandingEnv(protocol, case)
    try:
        first, _ = env.reset()
        first_next, _, _, _, first_info = env.step(np.zeros(2))
        for _ in range(8):
            env.step(np.zeros(2))
        second, reset_info = env.reset()
        second_next, _, _, _, second_info = env.step(np.zeros(2))
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first_next, second_next)
        np.testing.assert_array_equal(first_info["true_state"], second_info["true_state"])
        assert reset_info["interventions"] == 0
        assert second_info["control_updates"] == 1
    finally:
        env.close()


def test_final_holdout_case_is_rejected_by_default(protocol) -> None:
    case = sample_cases(protocol, "train", "nominal", count=1)[0]
    with pytest.raises(PermissionError, match="sealed"):
        ResidualLandingEnv(protocol, replace(case, split="final_holdout"))


@pytest.mark.parametrize(
    ("residual", "pid", "state", "previous", "reason"),
    [
        ([0.0, 0.0, 0.0], [0.5, 0.0], [0, 100, 0, -5, 0, 0, 900], None, "invalid_residual_shape"),
        ([np.nan, 0.0], [0.5, 0.0], [0, 100, 0, -5, 0, 0, 900], None, "nonfinite_residual"),
        ([2.0, 0.0], [0.5, 0.0], [0, 100, 0, -5, 0, 0, 900], None, "residual_bounds"),
        ([1.0, 0.0], [0.98, 0.0], [0, 100, 0, -5, 0, 0, 900], None, "actuator_limit"),
        ([1.0, 0.0], [0.5, 0.0], [0, 100, 0, -5, 0, 0, 750.5], None, "fuel_margin"),
        (
            [0.0, -1.0],
            [0.5, 0.0],
            [0, 100, 0, -5, np.deg2rad(21), np.deg2rad(5), 900],
            None,
            "attitude_margin",
        ),
        ([-1.0, 0.0], [0.8, 0.0], [0, 1, 0, -5, 0, 0, 900], None, "ground_margin"),
        (
            [1.0, 0.0],
            [0.55, 0.0],
            [0, 100, 0, -5, 0, 0, 900],
            [0.4, 0.0],
            "slew_limit",
        ),
    ],
)
def test_safety_filter_rejects_and_falls_back(
    protocol, residual, pid, state, previous, reason: str
) -> None:
    params = PlanarDynamicsParameters.from_config(protocol.base_config)
    decision = ResidualSafetyFilter().check(
        residual,
        np.asarray(pid, dtype=np.float64),
        np.asarray(state, dtype=np.float64),
        None if previous is None else np.asarray(previous, dtype=np.float64),
        params,
        0.1,
        0.0,
        np.asarray((2.0, np.deg2rad(60.0))),
    )
    assert decision.reasons == (reason,)
    np.testing.assert_array_equal(decision.command, pid)


def test_small_residual_passes_and_zero_never_intervenes(protocol) -> None:
    params = PlanarDynamicsParameters.from_config(protocol.base_config)
    safety = ResidualSafetyFilter()
    pid = np.asarray((0.5, 0.0))
    observed = np.asarray((0, 100, 0, -5, 0, 0, 900), dtype=np.float64)
    args = (pid, observed, pid, params, 0.1, 0.0, np.asarray((2.0, np.deg2rad(60))))
    accepted = safety.check(np.asarray((0.25, 0.25)), *args)
    assert accepted.reasons == ()
    np.testing.assert_allclose(accepted.command, pid + np.asarray((0.02, np.deg2rad(0.5))))
    zero = safety.check(np.zeros(2), *args)
    assert zero.reasons == ()
    np.testing.assert_array_equal(zero.command, pid)


@pytest.mark.parametrize(
    ("previous_action", "residual", "axis"),
    [
        ((0.2, 0.0), (1.0, 0.0), "slew_throttle"),
        ((0.5, np.deg2rad(-6)), (0.0, 1.0), "slew_gimbal"),
    ],
)
def test_rejected_slew_records_the_affected_axis(
    protocol, previous_action, residual, axis: str
) -> None:
    case = sample_cases(protocol, "train", "nominal", count=1)[0]
    env = ResidualLandingEnv(protocol, case)
    try:
        env.reset()
        env.plant.pid_action = np.asarray((0.5, 0.0), dtype=np.float64)
        env.previous_action = np.asarray(previous_action, dtype=np.float64)
        _, _, _, _, info = env.step(np.asarray(residual, dtype=np.float32))
        assert info["intervention_reasons"] == ["slew_limit"]
        assert info["intervention_axis_counts"][axis] == 1
    finally:
        env.close()


def test_rejected_first_action_recovers_full_pid_trajectory(protocol) -> None:
    case = sample_cases(protocol, "development", "wind", count=1)[0]
    baseline = ProtocolCasePlant(protocol, case)
    env = ResidualLandingEnv(protocol, case)
    try:
        baseline.reset()
        env.reset()
        first = True
        while True:
            action = np.asarray((np.nan, 0.0)) if first else np.zeros(2)
            _, _, terminated, truncated, info = env.step(action)
            first = False
            if terminated or truncated:
                break
        while not (baseline.terminated or baseline.truncated):
            assert baseline.pid_action is not None
            baseline.step_physics(baseline.pid_action)
        assert info["interventions"] == 1
        assert info["intervention_reason_counts"] == {"nonfinite_residual": 1}
        np.testing.assert_allclose(
            [step["state"] for step in env.plant.env.episode_log["steps"]],
            [step["state"] for step in baseline.env.episode_log["steps"]],
            atol=1e-12,
        )
        assert info["outcome"] == baseline.info["outcome"]
    finally:
        baseline.close()
        env.close()


def test_fallback_after_accepted_residual_uses_current_pid(protocol) -> None:
    case = sample_cases(protocol, "train", "nominal", count=1)[0]
    env = ResidualLandingEnv(protocol, case)
    try:
        env.reset()
        _, _, _, _, accepted = env.step(np.asarray((0.1, 0.1)))
        assert accepted["intervention_reasons"] == []
        assert not np.array_equal(accepted["chosen_action"], accepted["baseline_action"])
        _, _, _, _, rejected = env.step(np.asarray((np.nan, 0.0)))
        assert rejected["intervention_reasons"] == ["nonfinite_residual"]
        assert isinstance(rejected["proposed_residual"], str)
        np.testing.assert_array_equal(rejected["chosen_action"], rejected["baseline_action"])
        assert rejected["control_updates"] == 2
        assert rejected["interventions"] == 1
    finally:
        env.close()


def test_intervention_and_latency_are_reported(protocol) -> None:
    case = sample_cases(protocol, "train", "nominal", count=1)[0]
    env = ResidualLandingEnv(protocol, case)
    try:
        env.reset()
        _, _, _, _, info = env.step(np.asarray((np.inf, 0.0)))
        assert info["intervention_reasons"] == ["nonfinite_residual"]
        assert info["intervention_rate"] == 1.0
        assert info["control_updates"] == 1
        assert env.safety_timing_us["mean"] >= 0.0
        assert np.isfinite(env.safety_timing_us["p95"])
        assert np.all(np.isfinite(info["true_state"]))
    finally:
        env.close()


def test_gymnasium_checker_accepts_residual_environment(protocol) -> None:
    case = sample_cases(protocol, "train", "nominal", count=1)[0]
    env = ResidualLandingEnv(protocol, case)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Box observation space.*")
            check_env(env, skip_render_check=True)
    finally:
        env.close()


def test_train_only_probe_report_and_no_overwrite(protocol, tmp_path: Path, monkeypatch) -> None:
    first = run_probe(protocol, 1, 0.25)
    second = run_probe(protocol, 1, 0.25)
    assert first["root_seed"] == 378
    assert first["split"] == "train"
    assert [item["group"] for item in first["groups"]] == list(GROUPS)
    for one, two in zip(first["groups"], second["groups"], strict=True):
        assert one["case_ids"] == two["case_ids"]
        assert one["interventions"] == two["interventions"]
        assert one["intervention_reasons"] == two["intervention_reasons"]
        assert 0 <= one["intervention_rate"] <= 1
        assert one["safety_p95_us"] >= 0
    output = tmp_path / "day38"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_residual_landing.py",
            "--protocol",
            str(PROTOCOL_PATH),
            "--output-dir",
            str(output),
            "--cases-per-group",
            "1",
        ],
    )
    main()
    saved = json.loads((output / "random-residual-probe.json").read_text(encoding="utf-8"))
    assert saved["split"] == "train"
    assert len(saved["groups"]) == len(GROUPS)
    with pytest.raises(FileExistsError):
        main()
