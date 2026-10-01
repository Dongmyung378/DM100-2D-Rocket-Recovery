"""Report integrity checks never execute a flight."""

import importlib.util
import json
import math
from copy import deepcopy
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/present_validation_results.py"
SPEC = importlib.util.spec_from_file_location("presentation_results", SCRIPT)
presentation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(presentation)


@pytest.fixture
def flight_record():
    state = [0.2, 7.5, 20.0, -20.0, math.radians(0.1), math.radians(-0.2), 4000.0, 100.0]
    return {
        "id": "synthetic",
        "group": "calm",
        "accepted": True,
        "mission_success": True,
        "flight_safety": True,
        "landing_success": True,
        "final_state": state,
        "contact": {
            "state": state,
            "foot_velocities_m_s": [[-0.15, -1.1], [0.25, -0.9]],
            "criteria": {"position": True, "vertical_foot_speed": True},
        },
        "remaining_main_propellant_kg": 4000.0,
        "remaining_rcs_propellant_kg": 100.0,
        "main_propellant_used_kg": 66200.0,
        "rcs_propellant_used_kg": 200.0,
        "flight_time_s": 400.0,
        "max_dynamic_pressure_pa": 25000.0,
        "max_angular_acceleration_deg_s2": 3.0,
        "max_flight_load_g": 4.0,
        "gust_exposure": {
            "configured_amplitude_m_s": 0.0,
            "integration_endpoints": {"gust_encounter_observed": False},
        },
    }


def test_actual_foot_metrics_not_com_velocity_and_input_is_not_mutated(flight_record):
    preserved = deepcopy(flight_record)
    row = presentation.extract_metrics(flight_record, "development")
    assert row["foot_vertical_speed_m_s"] == 1.1
    assert row["foot_horizontal_speed_m_s"] == 0.25
    assert row["position_error_m"] == 0.2
    assert row["tilt_deg"] == pytest.approx(0.1)
    assert row["contact_angular_rate_deg_s"] == pytest.approx(0.2)
    assert flight_record == preserved


@pytest.mark.parametrize("flag", ["mission_success", "flight_safety", "landing_success"])
def test_inconsistent_acceptance_is_rejected(flight_record, flag):
    flight_record[flag] = False
    with pytest.raises(ValueError, match="combined acceptance"):
        presentation.extract_metrics(flight_record, "development")


def test_false_contact_criterion_cannot_be_presented_as_success(flight_record):
    flight_record["contact"]["criteria"]["position"] = False
    with pytest.raises(ValueError, match="contact acceptance"):
        presentation.extract_metrics(flight_record, "development")


@pytest.mark.parametrize("key", ["main_propellant_used_kg", "rcs_propellant_used_kg"])
def test_fuel_ledger_uses_absolute_one_milligram_tolerance(flight_record, key):
    flight_record[key] += 0.00001
    with pytest.raises(ValueError, match="mass balance"):
        presentation.extract_metrics(flight_record, "development")


@pytest.mark.parametrize("foot", [0, 1])
def test_nonfinite_either_foot_cannot_disappear_in_max(flight_record, foot):
    flight_record["contact"]["foot_velocities_m_s"][foot][1] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite foot"):
        presentation.extract_metrics(flight_record, "development")


def test_failed_no_contact_flight_is_preserved(flight_record):
    flight_record.update(accepted=False, landing_success=False, contact=None)
    row = presentation.extract_metrics(flight_record, "development")
    assert not row["accepted"]
    assert not row["has_contact"]
    assert row["foot_vertical_speed_m_s"] is None


def test_extrema_keeps_job_identifier_and_ignores_missing_contact():
    rows = [
        {"job_id": "a", "speed": 0.9},
        {"job_id": "b", "speed": 1.2},
        {"job_id": "c", "speed": None},
    ]
    assert presentation.extrema(rows, "speed") == {"value": 1.2, "job_id": "b"}
    assert presentation.extrema(rows, "speed", minimum=True)["job_id"] == "a"


def test_html_table_escapes_untrusted_content():
    rendered = presentation.table(["<heading>"], [["<script>alert('x')</script>"]])
    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered
    assert "&lt;heading&gt;" in rendered
    assert "&#x27;" in rendered


@pytest.fixture
def campaign(tmp_path, flight_record):
    case = {"id": "synthetic", "group": "calm", "conditions": {}, "disturbances": {}}
    digest = presentation.semantic_sha(case)
    job = {
        "job_id": "development/synthetic",
        "split": "development",
        "case": case,
        "input_sha256": digest,
        "integration_dt_s": 0.01,
        "observation_seed_sequence": {"mode": "root"},
        "required_exposure_phase": None,
        "required_exposure_peak_m_s": None,
    }
    sources = {"archived-source.py": "source-sha"}
    protocol = {
        "source_hashes": sources,
        "design": dict(presentation.CURRENT_INVENTORY),
        "jobs": [job],
        "total": 1,
        "counts": {"development": 1},
    }
    flight_record.update(input_sha256=digest, input=case)
    record = {
        **{
            key: job[key]
            for key in (
                "job_id",
                "split",
                "input_sha256",
                "integration_dt_s",
                "observation_seed_sequence",
                "required_exposure_phase",
            )
        },
        "result": flight_record,
        "source_hashes_after": sources,
        "fuel_ledger": {"passed": True},
        "required_exposure_passed": True,
        "requirements_passed": True,
    }
    (tmp_path / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    protocol_sha = presentation.sha256(tmp_path / "protocol.json")
    header = {
        "protocol_sha256": protocol_sha,
        "fresh_independent": False,
        "source_hashes": sources,
        "selected_job_ids": [job["job_id"]],
    }
    (tmp_path / "all-results.jsonl").write_text(
        "\n".join(map(json.dumps, (header, record))), encoding="utf-8"
    )
    summary = {
        "protocol_sha256": protocol_sha,
        "previous_holdout_is_fresh_independent": False,
        "source_unchanged": True,
        "protocol_unchanged": True,
        "source_hashes_before": sources,
        "source_hashes_after": sources,
        "ordered_job_ids": [job["job_id"]],
        "total": 1,
        "completed": 1,
        "accepted": 1,
        "requirements_passed": 1,
        "failed_job_ids": [],
        "execution_errors": [],
        "by_split": {"development": {"total": 1, "accepted": 1, "requirements_passed": 1}},
    }
    (tmp_path / "all-summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (tmp_path / "calm").mkdir()
    nominal = {
        "source_unchanged_during_run": True,
        "source_sha256": sources,
        "source_sha256_after_run": sources,
        "design": protocol["design"],
    }
    (tmp_path / "calm/result.json").write_text(json.dumps(nominal), encoding="utf-8")
    return tmp_path


def test_historical_source_bindings_do_not_require_active_or_old_source_files(campaign):
    _, rows, _ = presentation.load_evidence(campaign)
    assert rows[0]["requirements_passed"]
    assert not (campaign / "archived-source.py").exists()


def test_portable_producer_explicit_regression_flag(campaign):
    path = campaign / "all-summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary.pop("previous_holdout_is_fresh_independent")
    summary["fresh_independent"] = False
    path.write_text(json.dumps(summary), encoding="utf-8")
    _, rows, _ = presentation.load_evidence(campaign)
    assert rows[0]["requirements_passed"]


def test_true_flags_cannot_hide_excessive_numeric_foot_speed(flight_record):
    flight_record["contact"]["foot_velocities_m_s"][0][1] = -1.6
    with pytest.raises(ValueError, match="numeric landing limits"):
        presentation.extract_metrics(flight_record, "development")


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda record: record.update(integration_dt_s=0.005), "integration_dt_s"),
        (lambda record: record.update(input_sha256="wrong"), "input_sha256"),
        (
            lambda record: record.update(observation_seed_sequence={"mode": "case"}),
            "observation_seed_sequence",
        ),
        (lambda record: record.update(source_hashes_after={"changed": "changed"}), "Worker source"),
        (lambda record: record.update(requirements_passed=False), "Requirement status"),
    ],
)
def test_execution_bindings_reject_tampering(campaign, mutation, match):
    path = campaign / "all-results.jsonl"
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    mutation(lines[1])
    path.write_text("\n".join(map(json.dumps, lines)), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        presentation.load_evidence(campaign)


def test_missing_result_does_not_shrink_denominator(campaign):
    path = campaign / "all-results.jsonl"
    path.write_text(path.read_text(encoding="utf-8").splitlines()[0], encoding="utf-8")
    with pytest.raises(ValueError, match="Missing, extra or duplicate"):
        presentation.load_evidence(campaign)


def test_current_stored_regression_if_present():
    if not (presentation.CAMPAIGN / "all-summary.json").exists():
        pytest.skip("Local stored evidence is not shipped with the source repository")
    summary, rows, nominal = presentation.load_evidence(presentation.CAMPAIGN)
    assert len(rows) == summary["total"] == 286
    assert summary["requirements_passed"] == 286
    assert nominal["design"]["rcs_propellant_kg"] == 300.0


def test_report_cannot_import_flight_controller():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "from powered_landing_guidance" not in source
    assert "AprFlight" not in source


def test_bilingual_reports_use_current_inventory_and_honest_scope(campaign, tmp_path):
    summary, rows, _ = presentation.load_evidence(campaign)
    for language, suffix in (("en", ""), ("ko", ".ko")):
        path = tmp_path / f"report{suffix}.html"
        presentation.write_report(summary, rows, path, language, suffix, True)
        contents = path.read_text(encoding="utf-8")
        assert f'lang="{language}"' in contents
        assert "70,200 kg main + 300 kg RCS" in contents
        assert "validation-report.ko.html" in contents
        assert "validation-report.html" in contents
        if language == "en":
            assert "not new independent validation" in contents
        else:
            assert "새 독립 검증이 아닙니다" in contents
