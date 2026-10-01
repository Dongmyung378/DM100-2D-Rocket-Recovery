"""Portable regression contract tests; no real full-flight simulations."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import evaluate_regression_suite as replay


@pytest.fixture(scope="module")
def suite():
    return replay.load_suite()


@pytest.fixture
def job(suite):
    return copy.deepcopy(suite["jobs"][0])


def result():
    return {
        "accepted": True,
        "failed_criteria": [],
        "main_propellant_used_kg": 65000.0,
        "remaining_main_propellant_kg": 5200.0,
        "rcs_propellant_used_kg": 200.0,
        "remaining_rcs_propellant_kg": 100.0,
        "gust_exposure": {"integration_endpoints": {"maximum_by_phase_m_s": {"ascent": 0.2}}},
    }


def save_suite(path, value):
    unsigned = {key: item for key, item in value.items() if key != "manifest_sha256"}
    value["manifest_sha256"] = replay.campaign._sha(unsigned)
    replay.write_json_new(path, value)


def test_tracked_suite_is_self_contained_and_complete(suite):
    assert len(suite["jobs"]) == 286
    assert suite["counts"] == {"development": 205, "previous-holdout": 72, "supplemental": 9}
    assert suite["fresh_independent"] is False
    assert suite["design"]["main_propellant_kg"] == 70200
    assert suite["design"]["rcs_propellant_kg"] == 300
    assert all(
        "baseline" not in job and "baseline_record_sha256" not in job for job in suite["jobs"]
    )
    assert "archive" not in json.dumps(suite)


def test_supplemental_original_intervals_rng_modes_and_exposure_preserved(suite):
    jobs = [job for job in suite["jobs"] if job["split"] == "supplemental"]
    assert len(jobs) == 9
    assert sum(job["integration_dt_s"] == 0.005 for job in jobs) == 4
    assert sum(job["observation_seed_sequence"]["mode"] == "root" for job in jobs) == 4
    assert sum(job["required_exposure_phase"] == "ascent" for job in jobs) == 3
    assert sum(job["required_exposure_phase"] == "entry" for job in jobs) == 3


def test_all_payload_and_rng_hashes_match_after_json_roundtrip(suite):
    restored = json.loads(json.dumps(suite))
    for job in restored["jobs"]:
        assert job["input_sha256"] == replay.campaign._sha(job["case"])
        fresh = replay.seed_metadata(job["case"], job["observation_seed_sequence"]["mode"])
        assert replay.campaign._sha(fresh) == replay.campaign._sha(job["observation_seed_sequence"])


@pytest.mark.parametrize("mode", ["root", "case"])
def test_root_rng_preserved_but_case_identity_salts_case_rng(job, mode):
    case = job["case"]
    before = replay.seed_metadata(case, mode)
    changed = copy.deepcopy(case)
    changed["id"] = "another-case"
    assert (before == replay.seed_metadata(changed, mode)) == (mode == "root")
    if mode == "root":
        assert (
            before["generated_state_uint32"]
            == np.random.SeedSequence(378).generate_state(8).tolist()
        )


def test_invalid_rng_mode_rejected(job):
    with pytest.raises(ValueError, match="RNG"):
        replay.seed_metadata(job["case"], "unknown")


@pytest.mark.parametrize("change", ["input_type", "dt", "seed", "fresh", "duplicate", "design"])
def test_tampered_suite_rejected_even_with_recomputed_manifest_hash(tmp_path, suite, change):
    altered = copy.deepcopy(suite)
    job = altered["jobs"][0]
    if change == "input_type":
        # Numerically equal 0 and 0.0 are different canonical input payloads.
        assert type(job["case"]["conditions"]["gust_start_s"]) is float
        job["case"]["conditions"]["gust_start_s"] = 0
    elif change == "dt":
        job["integration_dt_s"] = True
    elif change == "seed":
        job["observation_seed_sequence"]["generated_state_uint32"][0] += 1
    elif change == "fresh":
        altered["fresh_independent"] = True
    elif change == "duplicate":
        altered["jobs"][1]["job_id"] = job["job_id"]
    else:
        altered["design"]["rcs_propellant_kg"] = 301
    path = tmp_path / "suite.json"
    save_suite(path, altered)
    with pytest.raises(ValueError):
        replay.load_suite(path)


def test_manifest_file_tampering_rejected(tmp_path, suite):
    altered = copy.deepcopy(suite)
    altered["scope"] = "modified"
    path = tmp_path / "suite.json"
    replay.write_json_new(path, altered)
    with pytest.raises(ValueError, match="SHA"):
        replay.load_suite(path)


def test_evaluate_uses_existing_evaluator_and_restores_factories(monkeypatch, job):
    called = []
    original_factory = replay.campaign.AprFlight
    original_seed = replay.campaign.observation_seed_sequence
    job["integration_dt_s"] = 0.005
    job["observation_seed_sequence"] = replay.seed_metadata(job["case"], "root")

    def fake_evaluate(actual_case):
        assert actual_case == job["case"]
        assert replay.campaign.AprFlight.func is original_factory
        assert replay.campaign.AprFlight.keywords == {"integration_dt_s": 0.005}
        assert replay.campaign.observation_seed_sequence(actual_case).state["entropy"] == 378
        called.append(actual_case)
        return result()

    monkeypatch.setattr(replay.campaign, "_evaluate_case", fake_evaluate)
    monkeypatch.setattr(replay, "source_hashes", lambda: {"test": "hash"})
    row = replay.evaluate_job(job)
    assert called == [job["case"]]
    assert row["requirements_passed"] is True
    assert row["fuel_ledger"]["main_error_kg"] == 0
    assert row["fuel_ledger"]["rcs_error_kg"] == 0
    assert replay.campaign.AprFlight is original_factory
    assert replay.campaign.observation_seed_sequence is original_seed


def test_evaluate_restores_factories_even_on_failure(monkeypatch, job):
    original_factory = replay.campaign.AprFlight
    original_seed = replay.campaign.observation_seed_sequence
    job["observation_seed_sequence"] = replay.seed_metadata(job["case"], "root")

    def broken(_):
        raise RuntimeError("intentional execution error")

    monkeypatch.setattr(replay.campaign, "_evaluate_case", broken)
    with pytest.raises(RuntimeError):
        replay.evaluate_job(job)
    assert replay.campaign.AprFlight is original_factory
    assert replay.campaign.observation_seed_sequence is original_seed


@pytest.mark.parametrize("field", ["main_propellant_used_kg", "rcs_propellant_used_kg"])
def test_ledger_mismatch_is_not_silently_accepted(monkeypatch, job, field):
    measured = result()
    measured[field] += 0.001
    monkeypatch.setattr(replay.campaign, "_evaluate_case", lambda _: measured)
    monkeypatch.setattr(replay, "source_hashes", lambda: {})
    row = replay.evaluate_job(job)
    assert row["result"]["accepted"] is True
    assert row["fuel_ledger"]["passed"] is False
    assert row["requirements_passed"] is False


@pytest.mark.parametrize("peak,passed", [(0.099, False), (0.1, True), (0.2, True)])
def test_required_phase_exposure_is_separate_from_landing(monkeypatch, job, peak, passed):
    job["required_exposure_phase"] = "ascent"
    job["required_exposure_peak_m_s"] = 0.1
    measured = result()
    measured["gust_exposure"]["integration_endpoints"]["maximum_by_phase_m_s"] = {
        "ascent": peak,
        "entry": 5.0,
    }
    monkeypatch.setattr(replay.campaign, "_evaluate_case", lambda _: measured)
    monkeypatch.setattr(replay, "source_hashes", lambda: {})
    row = replay.evaluate_job(job)
    assert row["result"]["accepted"] is True
    assert row["required_exposure_passed"] is passed
    assert row["requirements_passed"] is passed


def test_serial_execution_error_remains_failed_record(monkeypatch, job):
    def broken(_):
        raise RuntimeError("execution failed")

    monkeypatch.setattr(replay, "evaluate_job", broken)
    rows = list(replay.evaluate_jobs([job], 1))
    assert rows[0][1]["requirements_passed"] is False
    assert rows[0][1]["error_type"] == "RuntimeError"


def test_summary_is_ordered_and_never_calls_seen_holdout_fresh(job):
    other = copy.deepcopy(job)
    other["job_id"] = "previous-holdout/test"
    other["split"] = "previous-holdout"
    rows = [
        {"job_id": other["job_id"], "split": other["split"], "requirements_passed": False},
        {
            "job_id": job["job_id"],
            "split": job["split"],
            "requirements_passed": True,
            "source_hashes_after": {"a": "b"},
            "result": result(),
        },
    ]
    summary = replay.summarize(rows, [job, other], {"a": "b"}, {"a": "b"})
    assert summary["ordered_job_ids"] == [job["job_id"], other["job_id"]]
    assert summary["accepted"] == summary["requirements_passed"] == 1
    assert summary["execution_errors"] == [other["job_id"]]
    assert summary["fresh_independent"] is False
    assert summary["continuous_envelope_proven"] is False


def test_summary_detects_source_change(job):
    row = {
        "job_id": job["job_id"],
        "split": job["split"],
        "requirements_passed": True,
        "source_hashes_after": {"a": "modified"},
        "result": result(),
    }
    assert replay.summarize([row], [job], {"a": "b"}, {"a": "b"})["source_unchanged"] is False


def test_summary_rejects_duplicate_jobs(job):
    row = {"job_id": job["job_id"], "split": job["split"], "requirements_passed": False}
    with pytest.raises(ValueError):
        replay.summarize([row, row], [job], {}, {})


def test_prepare_and_selected_run_need_only_current_suite(monkeypatch, tmp_path, suite):
    monkeypatch.setattr(replay, "load_suite", lambda: copy.deepcopy(suite))
    monkeypatch.setattr(replay, "source_hashes", lambda: {"current": "hash"})
    protocol = replay.prepare(tmp_path)
    assert protocol["fresh_independent"] is False
    assert "baseline_source_archive" not in protocol

    def fake_jobs(jobs, workers):
        assert workers == 2
        for job in reversed(jobs):
            yield (
                job,
                {
                    "job_id": job["job_id"],
                    "split": job["split"],
                    "requirements_passed": True,
                    "source_hashes_after": {"current": "hash"},
                    "result": result(),
                },
            )

    monkeypatch.setattr(replay, "evaluate_jobs", fake_jobs)
    selected = [job["job_id"] for job in suite["jobs"][:2]]
    summary = replay.run(tmp_path, "all", workers=2, case_ids=selected)
    assert summary["accepted"] == 2
    assert summary["source_unchanged"] and summary["protocol_unchanged"]
    assert summary["ordered_job_ids"] == selected
    results = next(tmp_path.glob("*-results.jsonl"))
    rows = [json.loads(line) for line in results.read_text(encoding="utf-8").splitlines()]
    assert [row["job_id"] for row in rows[1:]] == selected
    with pytest.raises(FileExistsError):
        replay.run(tmp_path, "all", workers=2, case_ids=selected)


def test_changed_source_rejected_before_any_flight(monkeypatch, tmp_path, suite):
    monkeypatch.setattr(replay, "load_suite", lambda: copy.deepcopy(suite))
    monkeypatch.setattr(replay, "source_hashes", lambda: {"current": "old"})
    replay.prepare(tmp_path)
    monkeypatch.setattr(replay, "source_hashes", lambda: {"current": "changed"})
    with pytest.raises(ValueError):
        replay.run(tmp_path, "all")


def test_new_json_never_overwrites_previous_evidence(tmp_path):
    path = tmp_path / "protocol.json"
    replay.write_json_new(path, {"x": 0.0})
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        replay.write_json_new(path, {"x": 0})
    assert path.read_bytes() == original
    assert type(json.loads(original)["x"]) is float


@pytest.mark.parametrize("workers", [0, -1, True, 1.5])
def test_invalid_workers_rejected_before_any_flight(tmp_path, workers):
    with pytest.raises(ValueError, match="workers"):
        replay.run(tmp_path, "all", workers)


def test_default_paths_and_source_binding_are_tracked_current_files():
    assert replay.DEFAULT_SUITE.is_relative_to(Path(replay.ROOT))
    assert replay.DEFAULT_OUTPUT.is_relative_to(Path(replay.ROOT))
    assert all("revalidate_propellant" not in path for path in replay.SOURCE_PATHS)
    assert len(replay.source_hashes()) == 10
