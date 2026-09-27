"""Multi-seed training and controlled disturbance extension checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from powered_landing_guidance.rl_protocol import load_protocol
from scripts.advance_residual_sac import derived_seeds, load_settings, run_experiment

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/rl-evaluation-v1.yaml"


def _short_settings(tmp_path: Path) -> tuple[Path, dict]:
    settings = yaml.safe_load((ROOT / "configs/sac-day41-v1.yaml").read_text())
    settings["seed_count"] = 1
    settings["total_timesteps"] = 6
    settings["stages"] = [
        {"name": "nominal", "end_timestep": 2},
        {"name": "initial_ood", "end_timestep": 4},
        {"name": "model_mismatch", "end_timestep": 6},
    ]
    settings["evaluation"]["cases_per_group"] = 1
    settings["sac"]["buffer_size"] = 64
    settings["sac"]["learning_starts"] = 2
    settings["sac"]["batch_size"] = 2
    path = tmp_path / "day41-test.yaml"
    path.write_text(yaml.safe_dump(settings), encoding="utf-8")
    return path, settings


def test_seed_derivation_and_development_only(tmp_path: Path) -> None:
    protocol = load_protocol(PROTOCOL)
    path, settings = _short_settings(tmp_path)
    assert derived_seeds(378, 3) == [3024638723, 2326541261, 1907174973]
    assert load_settings(path, protocol)["seed"] == 378
    settings["evaluation"]["split"] = "final_holdout"
    path.write_text(yaml.safe_dump(settings), encoding="utf-8")
    with pytest.raises(ValueError, match="development"):
        load_settings(path, protocol)


def test_multiseed_resume_and_single_filter_change(tmp_path: Path) -> None:
    settings_path, settings = _short_settings(tmp_path)
    first_output = tmp_path / "day41"
    run_experiment(PROTOCOL, settings_path, first_output, until=4)
    run_experiment(PROTOCOL, settings_path, first_output, until=6, resume=True)
    first_summary = json.loads((first_output / "summary.json").read_text(encoding="utf-8"))
    assert len(first_summary["runs"]) == 1
    assert len(first_summary["runs"][0]["groups"]) == 6
    assert first_summary["final_holdout"] == "sealed_not_materialized"

    settings["source"] = {
        "root_dir": (first_output / "baseline").relative_to(ROOT).as_posix(),
        "settings": settings_path.relative_to(ROOT).as_posix(),
        "step": 6,
    }
    settings["total_timesteps"] = 10
    settings["stages"] = [
        {"name": "sensor_delay", "end_timestep": 8},
        {"name": "combined", "end_timestep": 10},
    ]
    control = settings["variants"][0]
    tuned = {
        "name": "tuned",
        "safety_filter": {**control["safety_filter"], "throttle_limit": 0.1},
    }
    settings["variants"] = [{**control, "name": "control"}, tuned]
    second_settings = tmp_path / "day42-test.yaml"
    second_settings.write_text(yaml.safe_dump(settings), encoding="utf-8")
    second_output = tmp_path / "day42"
    run_experiment(PROTOCOL, second_settings, second_output)
    second_summary = json.loads((second_output / "summary.json").read_text(encoding="utf-8"))
    assert {row["variant"] for row in second_summary["runs"]} == {"control", "tuned"}
    assert all(len(row["groups"]) == 6 for row in second_summary["runs"])
    assert not list(second_output.glob("*holdout*"))
