"""Day 36 protocol, split isolation, and PID pilot regression tests."""

from __future__ import annotations

import csv
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from powered_landing_guidance.controllers import IntegratedLandingController
from powered_landing_guidance.envs import RocketLandingEnv
from powered_landing_guidance.rl_protocol import (
    CSV_FIELDS,
    GROUPS,
    RLProtocol,
    load_protocol,
    run_pid_case,
    sample_cases,
    validate_protocol,
)
from scripts.prepare_rl_protocol import main

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs/rl-evaluation-v1.yaml"


def test_protocol_is_frozen_and_has_all_six_groups() -> None:
    protocol = load_protocol(PROTOCOL_PATH)
    assert set(protocol.settings["groups"]) == set(GROUPS)
    assert len(protocol.sha256) == 64
    assert protocol.settings["seed"] == 378
    assert [
        protocol.settings["splits"][name]["stream_id"]
        for name in ("train", "development", "final_holdout")
    ] == [0, 1, 2]
    assert protocol.settings["splits"]["final_holdout"]["episodes_per_group"] >= 200
    assert protocol.settings["selection"]["fuel_priority"] == "after_robustness_gate"
    assert protocol.settings["selection"]["stress_ci_lower_bound_gt"] == 0.0
    assert protocol.settings["base_config"] == "configs/pid-baseline-v1.yaml"


def test_sampling_is_repeatable_and_split_separated() -> None:
    protocol = load_protocol(PROTOCOL_PATH)
    first = sample_cases(protocol, "development", "wind", level="constant_15", count=8)
    again = sample_cases(protocol, "development", "wind", level="constant_15", count=8)
    stronger = sample_cases(protocol, "development", "wind", level="constant_30", count=8)
    training = sample_cases(protocol, "train", "wind", level="constant_15", count=8)
    longer = sample_cases(protocol, "development", "wind", level="constant_15", count=20)
    assert [case.case_id for case in first] == [case.case_id for case in again]
    assert np.array_equal(
        np.asarray([case.initial_state for case in first]),
        np.asarray([case.initial_state for case in again]),
    )
    assert [case.case_id for case in first] == [case.case_id for case in longer[:8]]
    assert np.array_equal(
        np.asarray([case.initial_state for case in first]),
        np.asarray([case.initial_state for case in longer[:8]]),
    )
    assert [case.pair_id for case in first] == [case.pair_id for case in stronger]
    assert np.array_equal(
        np.asarray([case.initial_state for case in first]),
        np.asarray([case.initial_state for case in stronger]),
    )
    assert not {case.case_id for case in first} & {case.case_id for case in training}
    assert not {case.pair_id for case in first} & {case.pair_id for case in training}
    ood_expanded = sample_cases(protocol, "development", "initial_ood", level="expanded", count=1)
    ood_hard = sample_cases(protocol, "development", "initial_ood", level="hard", count=1)
    assert ood_expanded[0].pair_id != ood_hard[0].pair_id
    revised = RLProtocol(protocol.settings, protocol.base_config, "0" * 64)
    assert sample_cases(revised, "development", "wind", count=1)[0].case_id != first[0].case_id


def test_final_holdout_is_sealed_by_default() -> None:
    protocol = load_protocol(PROTOCOL_PATH)
    with pytest.raises(PermissionError, match="sealed"):
        sample_cases(protocol, "final_holdout", "nominal", count=1)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda settings: settings["splits"]["final_holdout"].update(
                stream_id=settings["splits"]["development"]["stream_id"]
            ),
            "distinct",
        ),
        (
            lambda settings: settings.update(seed=379),
            "must be 378",
        ),
        (
            lambda settings: settings["initial_ranges"]["easy"].update(mass_kg=[700.0, 900.0]),
            "propellant",
        ),
        (
            lambda settings: settings["groups"]["wind"]["levels"][0]["disturbances"].update(
                unknown_disturbance=1.0
            ),
            "unknown",
        ),
        (
            lambda settings: settings["groups"]["nominal"]["levels"][0]["disturbances"].update(
                wind_m_s=1.0
            ),
            "unperturbed",
        ),
        (
            lambda settings: settings.update(controller_sha256="0" * 64),
            "digest",
        ),
    ],
)
def test_invalid_protocol_is_rejected(mutate, message: str) -> None:
    protocol = load_protocol(PROTOCOL_PATH)
    settings = deepcopy(protocol.settings)
    mutate(settings)
    with pytest.raises(ValueError, match=message):
        validate_protocol(settings, protocol.base_config)


def test_zero_disturbance_pid_matches_direct_controller() -> None:
    protocol = load_protocol(PROTOCOL_PATH)
    case = sample_cases(protocol, "development", "nominal", count=1)[0]
    row = run_pid_case(protocol, case)
    env = RocketLandingEnv(protocol.base_config)
    controller = IntegratedLandingController.from_config(protocol.base_config)
    try:
        state, info = env.reset(options={"initial_state": case.initial_state})
        while True:
            command = controller.command(state, time_s=float(info["time_s"]))
            state, _, terminated, truncated, info = env.step(command)
            if terminated or truncated:
                break
    finally:
        env.close()
    assert row["outcome"] == info["outcome"]
    assert row["duration_s"] == pytest.approx(info["time_s"], abs=1e-12)
    assert row["fuel_used_kg"] == pytest.approx(info["fuel_used_kg"], abs=1e-12)
    assert row["final_x_m"] == pytest.approx(state[0], abs=1e-12)
    assert row["final_vz_m_s"] == pytest.approx(state[3], abs=1e-12)


def test_perturbed_case_is_reproducible_and_has_complete_schema() -> None:
    protocol = load_protocol(PROTOCOL_PATH)
    case = sample_cases(protocol, "development", "combined", level="medium", count=1)[0]
    row = run_pid_case(protocol, case)
    assert row == run_pid_case(protocol, case)
    assert tuple(row) == CSV_FIELDS
    assert row["thrust_scale"] == 0.85
    assert row["observation_delay_physics_steps"] == 5
    assert row["safe_landing"] == int(row["outcome"] == "success")


def test_pilot_command_writes_only_development_results(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "day36"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_rl_protocol.py",
            "--protocol",
            str(PROTOCOL_PATH),
            "--output-dir",
            str(output),
            "--pilot-cases",
            "1",
        ],
    )
    main()
    with (output / "pid-development-pilot.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    report = json.loads((output / "protocol-report.json").read_text(encoding="utf-8"))
    assert len(rows) == sum(
        len(group["levels"]) for group in load_protocol(PROTOCOL_PATH).settings["groups"].values()
    )
    assert set(rows[0]) == set(CSV_FIELDS)
    assert all(row["split"] == "development" for row in rows)
    assert report["final_holdout"] == "sealed_not_materialized"
    assert report["root_seed"] == 378
    assert report["split_stream_ids"] == {"train": 0, "development": 1, "final_holdout": 2}
    assert set(report["condition_groups"]) == set(GROUPS)
    assert report["pilot_rows"] == len(rows)
    assert report["canonical_pilot_count"] is False
    with pytest.raises(FileExistsError):
        main()
