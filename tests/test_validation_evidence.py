"""Evidence-integrity checks using synthetic rows, never performance claims."""

from copy import deepcopy

import pytest

from scripts.evaluate_study import make_manifest
from scripts.validate_evidence import digest, validate_rows, wilson_interval


@pytest.fixture
def evidence():
    manifest = make_manifest()
    job = manifest["jobs"][0]
    row = {
        key: job[key]
        for key in (
            "job_id",
            "split",
            "variant",
            "case_id",
            "seed",
            "integration_dt_s",
            "case_sha256",
        )
    }
    row.update(
        control_dt_s=0.02,
        requirements_passed=True,
        execution_error=None,
        source_bundle_sha256="synthetic-unit-test",
        mission_success=True,
        flight_safety=True,
        landing_success=True,
        criteria={"separation": True, "touchdown": True},
        failed_criteria=[],
        main_used_kg=65000.0,
        main_remaining_kg=5200.0,
        rcs_used_kg=200.0,
        rcs_remaining_kg=100.0,
        position_error_m=0.5,
        vertical_foot_speed_m_s=1.0,
        horizontal_foot_speed_m_s=0.1,
        tilt_deg=0.0,
        contact_rate_deg_s=0.0,
        peak_dynamic_pressure_pa=28000.0,
        peak_angular_acceleration_deg_s2=3.0,
        flight_time_s=400.0,
        wall_seconds=100.0,
        contact={
            "state": [0.5, 10, 0, -1, 0, 0],
            "foot_velocities_m_s": [[0.1, -1], [-0.1, -1]],
            "criteria": {"position": True},
        },
    )
    return manifest, job, row


def test_one_declared_synthetic_row_has_consistent_accounting(evidence):
    manifest, job, row = evidence
    result = validate_rows(manifest, [row], [job["job_id"]], "synthetic-unit-test")
    assert result["development/full"]["passed"] == 1


@pytest.mark.parametrize(
    "change",
    [
        {"position_error_m": 3.0},
        {"main_used_kg": 65001.0},
        {"seed": -1},
        {"requirements_passed": False},
        {"execution_error": "crash"},
        {"source_bundle_sha256": "another-source"},
        {"vertical_foot_speed_m_s": float("nan")},
        {"failed_criteria": ["not_recorded"]},
        {"contact": None},
    ],
)
def test_inconsistent_or_tampered_rows_cannot_be_accepted(evidence, change):
    manifest, job, row = evidence
    row.update(change)
    with pytest.raises(ValueError):
        validate_rows(manifest, [row], [job["job_id"]], "synthetic-unit-test")


def test_duplicate_or_missing_result_cannot_shrink_denominator(evidence):
    manifest, job, row = evidence
    with pytest.raises(ValueError):
        validate_rows(manifest, [row, row], [job["job_id"]])
    with pytest.raises(ValueError):
        validate_rows(manifest, [], [job["job_id"]])


def test_rehashed_manifest_cannot_relax_acceptance(evidence):
    manifest, job, row = evidence
    altered = deepcopy(manifest)
    altered["acceptance"]["position_error_m"] = 4.0
    altered["manifest_sha256"] = digest(
        {k: v for k, v in altered.items() if k != "manifest_sha256"}
    )
    with pytest.raises(ValueError, match="thresholds"):
        validate_rows(altered, [row], [job["job_id"]])


def test_rehashed_manifest_cannot_disconnect_case_and_job(evidence):
    manifest, job, row = evidence
    altered = deepcopy(manifest)
    altered["cases"][0]["conditions"]["density_scale"] = 0.99
    altered["manifest_sha256"] = digest(
        {k: v for k, v in altered.items() if k != "manifest_sha256"}
    )
    with pytest.raises(ValueError, match="digest"):
        validate_rows(altered, [row], [job["job_id"]])


def test_finite_perfect_record_still_has_an_interval_below_one():
    lower, upper = wilson_interval(96, 96)
    assert 0.95 < lower < 1
    assert upper == pytest.approx(1)
