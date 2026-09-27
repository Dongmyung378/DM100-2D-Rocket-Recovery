"""Regression tests for weighted BC and learned-policy rollout helpers."""

from pathlib import Path

import numpy as np
import pytest

from powered_landing_guidance import load_config
from powered_landing_guidance.behavior_cloning import BCPolicy, load_checkpoint, save_checkpoint
from powered_landing_guidance.imitation_learning import (
    ActuatorSettings,
    episode_metrics,
    run_episode,
)

ROOT = Path(__file__).resolve().parents[1]


def test_weighted_gradient_matches_finite_difference() -> None:
    policy = BCPolicy((5,), 5, "tanh")
    states = np.random.default_rng(1).normal(size=(4, 7)).astype(np.float32)
    actions = np.random.default_rng(2).normal(size=(4, 2)).astype(np.float32)
    weights = np.asarray((1.0, 2.0, 3.0, 4.0), dtype=np.float32)
    _, gradients, _ = policy.gradients(states, actions, weights, (1.0, 2.0))
    original = float(policy.weights[0][0, 0])
    epsilon = 1e-3
    policy.weights[0][0, 0] = original + epsilon
    upper = policy.gradients(states, actions, weights, (1.0, 2.0))[0]
    policy.weights[0][0, 0] = original - epsilon
    lower = policy.gradients(states, actions, weights, (1.0, 2.0))[0]
    policy.weights[0][0, 0] = original
    assert gradients[0][0, 0] == pytest.approx((upper - lower) / (2 * epsilon), rel=2e-3)


def test_tanh_checkpoint_roundtrip(tmp_path: Path) -> None:
    policy = BCPolicy((8, 6), 10, "tanh")
    path = tmp_path / "policy.npz"
    save_checkpoint(
        path, policy, {"training_settings": {"hidden_sizes": [8, 6]}, "activation": "tanh"}
    )
    loaded, _ = load_checkpoint(path)
    assert loaded.activation == "tanh"
    states = np.random.default_rng(3).normal(size=(2, 7))
    np.testing.assert_array_equal(
        policy.predict_normalized(states), loaded.predict_normalized(states)
    )


def test_legacy_checkpoint_defaults_to_relu(tmp_path: Path) -> None:
    path = tmp_path / "legacy.npz"
    save_checkpoint(path, BCPolicy((8,), 4), {"training_settings": {"hidden_sizes": [8]}})
    policy, _ = load_checkpoint(path)
    assert policy.activation == "relu"


def test_rollout_and_rate_limit() -> None:
    config = load_config(ROOT / "configs/pid-baseline-v1.yaml")
    policy = BCPolicy((8,), 3)
    normalization = {
        "state": {"mean": [0.0] * 7, "scale": [1.0] * 7},
        "action": {"mean": [0.5, 0.0], "scale": [0.1, 0.1]},
    }
    initial = np.asarray((0.0, 100.0, 0.0, -20.0, 0.0, 0.0, 1000.0))
    actuator = ActuatorSettings(2.0, np.deg2rad(60.0))
    episode, visited = run_episode(
        config, initial, policy, normalization, actuator=actuator, collect=True
    )
    assert len(visited) > 0
    metrics = episode_metrics(episode, config)
    assert metrics["max_throttle_rate_per_s"] <= 2.0 + 1e-8
    assert metrics["max_gimbal_rate_deg_s"] <= 60.0 + 1e-8
    second, _ = run_episode(config, initial, policy, normalization, actuator=actuator)
    assert episode["outcome"] == second["outcome"]
    np.testing.assert_array_equal(episode["steps"][-1]["state"], second["steps"][-1]["state"])


def test_pid_mix_one_matches_frozen_pid() -> None:
    config = load_config(ROOT / "configs/pid-baseline-v1.yaml")
    initial = np.asarray((0.0, 100.0, 0.0, -20.0, 0.0, 0.0, 1000.0))
    normalization = {
        "state": {"mean": [0.0] * 7, "scale": [1.0] * 7},
        "action": {"mean": [0.5, 0.0], "scale": [0.1, 0.1]},
    }
    baseline, _ = run_episode(config, initial, None)
    blended, _ = run_episode(config, initial, BCPolicy((8,), 3), normalization, pid_mix=1.0)
    assert baseline["outcome"] == blended["outcome"]
    np.testing.assert_allclose(
        baseline["steps"][-1]["state"], blended["steps"][-1]["state"], atol=1e-10
    )
