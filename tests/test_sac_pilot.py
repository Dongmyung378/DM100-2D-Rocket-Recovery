"""Day 40 SAC pilot: zero-PID start, checkpoint resume, and split isolation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from powered_landing_guidance.rl_protocol import load_protocol
from scripts.train_residual_sac import load_pilot_settings, run_training

ROOT = Path(__file__).resolve().parents[1]


def test_pilot_rejects_final_holdout_evaluation(tmp_path: Path) -> None:
    protocol = load_protocol(ROOT / "configs/rl-evaluation-v1.yaml")
    settings = yaml.safe_load((ROOT / "configs/sac-pilot-v1.yaml").read_text())
    settings["evaluation"]["split"] = "final_holdout"
    settings_path = tmp_path / "invalid.yaml"
    settings_path.write_text(yaml.safe_dump(settings), encoding="utf-8")
    with pytest.raises(ValueError, match="development"):
        load_pilot_settings(settings_path, protocol)


def test_short_pilot_resumes_without_retesting_holdout(tmp_path: Path) -> None:
    protocol = load_protocol(ROOT / "configs/rl-evaluation-v1.yaml")
    settings = yaml.safe_load((ROOT / "configs/sac-pilot-v1.yaml").read_text())
    settings["total_timesteps"] = 12
    settings["stages"] = [
        {"name": "nominal", "end_timestep": 4},
        {"name": "initial_ood", "end_timestep": 8},
        {"name": "wind", "end_timestep": 12},
    ]
    settings["evaluation"]["cases_per_group"] = 1
    settings["sac"]["buffer_size"] = 64
    settings["sac"]["learning_starts"] = 2
    settings["sac"]["batch_size"] = 2
    settings_path = tmp_path / "short-pilot.yaml"
    settings_path.write_text(yaml.safe_dump(settings), encoding="utf-8")
    output = tmp_path / "pilot"

    first = run_training(protocol, settings_path, output, until=4, resume=False)
    assert first["completed_timesteps"] == 4
    initial = json.loads((output / "eval-step-000000.json").read_text(encoding="utf-8"))
    assert all(
        group["pid_safe_landings"] == group["policy_safe_landings"] for group in initial["groups"]
    )
    assert (output / "replay-step-000004.pkl").exists()
    assert (output / "learning-curve-step-000004.png").exists()

    resumed = run_training(protocol, settings_path, output, until=12, resume=True)
    assert resumed["completed_timesteps"] == 12
    final = json.loads((output / "eval-step-000012.json").read_text(encoding="utf-8"))
    assert [group["group"] for group in final["groups"]] == ["nominal", "initial_ood", "wind"]
    assert final["final_holdout"] == "sealed_not_materialized"
    assert not list(output.glob("*holdout*"))
    assert (output / "manifest-step-000012.json").exists()
    with pytest.raises(FileExistsError):
        run_training(protocol, settings_path, output, until=12, resume=False)
