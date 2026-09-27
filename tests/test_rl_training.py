"""Day 39 reward and curriculum split-isolation checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from powered_landing_guidance.residual_landing import ResidualLandingEnv
from powered_landing_guidance.rl_protocol import load_protocol, sample_cases
from powered_landing_guidance.rl_training import (
    CURRICULUM_STAGES,
    CurriculumResidualEnv,
    LandingRewardWrapper,
    landing_potential,
    landing_reward,
)

PROTOCOL_PATH = Path(__file__).resolve().parents[1] / "configs/rl-evaluation-v1.yaml"


@pytest.fixture(scope="module")
def protocol():
    return load_protocol(PROTOCOL_PATH)


def test_reward_orders_success_above_every_failure() -> None:
    start = np.asarray((15.0, 100.0, 2.0, -20.0, 0.1, 0.1, 950.0))
    touchdown = np.asarray((0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 900.0))
    success, parts = landing_reward(
        start, touchdown, ground_z_m=0.0, elapsed_s=10, outcome="success"
    )
    assert parts["terminal"] == 100.0
    assert success > 70.0
    for outcome in ("crash", "hard_landing", "fuel_depletion", "timeout"):
        failure, components = landing_reward(
            start, touchdown, ground_z_m=0.0, elapsed_s=10, outcome=outcome
        )
        assert failure < 0
        assert components["terminal"] <= -120
        assert failure < success


def test_hovering_and_fuel_depletion_cannot_earn_positive_reward() -> None:
    hovering = np.asarray((0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 900.0))
    running, parts = landing_reward(
        hovering, hovering, ground_z_m=0.0, elapsed_s=0.1, outcome="running"
    )
    assert running < 0
    assert parts["progress"] == 0
    assert (
        landing_reward(hovering, hovering, ground_z_m=0.0, elapsed_s=120, outcome="timeout")[0]
        < -130
    )
    assert (
        landing_reward(hovering, hovering, ground_z_m=0.0, elapsed_s=10, outcome="fuel_depletion")[
            0
        ]
        < -130
    )


def test_progress_cannot_be_farmed_by_oscillation() -> None:
    left = np.asarray((-8.0, 100.0, 0.0, -20.0, 0.0, 0.0, 950.0))
    right = np.asarray((8.0, 100.0, 0.0, -20.0, 0.0, 0.0, 950.0))
    center = np.asarray((0.0, 100.0, 0.0, -20.0, 0.0, 0.0, 950.0))
    rewards = [
        landing_reward(a, b, ground_z_m=0.0, elapsed_s=0.1, outcome="running")[0]
        for a, b in ((left, center), (center, right), (right, left))
    ]
    assert sum(rewards) == pytest.approx(-0.03)
    assert -11 <= landing_potential(left, 0.0) <= 0


def test_reward_wrapper_preserves_physical_outcome(protocol) -> None:
    case = sample_cases(protocol, "train", "nominal", count=1)[0]
    env = LandingRewardWrapper(ResidualLandingEnv(protocol, case))
    try:
        observation, _ = env.reset()
        assert observation.shape == (17,)
        _, reward, _, _, info = env.step(np.zeros(2, dtype=np.float32))
        assert reward == pytest.approx(sum(info["reward_components"].values()))
        assert info["outcome"] == "running"
        assert info["reward_components"]["terminal"] == 0
    finally:
        env.close()


def test_curriculum_is_train_only_and_stage_order_is_valid(protocol) -> None:
    assert list(CURRICULUM_STAGES) == [
        "nominal",
        "initial_ood",
        "wind",
        "model_mismatch",
        "sensor_delay",
        "single_disturbance",
        "combined",
    ]
    env = CurriculumResidualEnv(protocol)
    try:
        assert all(case.split == "train" for cases in env.cases.values() for case in cases)
        assert all(
            sum(weight for _, _, weight in entries) == pytest.approx(1.0)
            for entries in CURRICULUM_STAGES.values()
        )
        first, first_info = env.reset(seed=378)
        assert first_info["group"] == "nominal"
        assert first_info["curriculum_stage"] == "nominal"
        env.set_stage("wind")
        _, _, _, _, old_episode = env.step(np.zeros(2, dtype=np.float32))
        assert old_episode["curriculum_stage"] == "nominal"
        second, second_info = env.reset()
        assert second_info["curriculum_stage"] == "wind"
        assert second_info["group"] in {"nominal", "initial_ood", "wind"}
        assert first.shape == second.shape == (17,)
        with pytest.raises(ValueError, match="unknown curriculum"):
            env.set_stage("final_holdout")
    finally:
        env.close()


def test_curriculum_sampler_state_restores_next_case(protocol) -> None:
    left = CurriculumResidualEnv(protocol, stage="wind")
    right = CurriculumResidualEnv(protocol, stage="wind")
    try:
        left.reset(seed=378)
        saved = left.sampler_state()
        right.restore_sampler_state(saved)
        _, expected = left.reset()
        _, actual = right.reset()
        assert expected["case_id"] == actual["case_id"]
        assert expected["curriculum_episode"] == actual["curriculum_episode"]
    finally:
        left.close()
        right.close()
