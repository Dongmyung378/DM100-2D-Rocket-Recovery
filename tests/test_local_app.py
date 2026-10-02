"""Fast local-UI checks: no complete flight or MP4 render is executed here."""

import json
import queue
import threading
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi", reason="Install the ui extra to test the local application")
pytest.importorskip("httpx", reason="Install the dev extra for TestClient")

from fastapi.testclient import TestClient  # noqa: E402

from powered_landing_guidance import local_app  # noqa: E402
from powered_landing_guidance.apr_recovery import (  # noqa: E402
    AprConditions,
    RecoveryDisturbances,
)

ORIGIN = "http://127.0.0.1:8765"
JOB_ID = "a" * 32


@pytest.fixture
def saved_flight():
    """A compact extract of the saved rcs300-v1 calm flight, not new physics."""
    state = [
        -3.815976289644869e-05,
        7.694232659114542,
        2.4773201349595625e-05,
        -0.9956923136545647,
        -4.801083297627676e-10,
        -2.9128470107070252e-09,
        5796.334709999539,
        262.54688797240703,
        1.0,
        9.412324086940618e-166,
        0.0,
        -1.3010092818432262e-05,
        0.11062688711168646,
        1499.9999999999943,
        0.9999999999999981,
        49.31607357317591,
        15.683926426829109,
        1.3797496077898656e-05,
        2.843616368944636e-06,
    ]
    contact_time = 405.70928261485176
    return {
        "simulated_acceptance_passed": True,
        "mission_success": True,
        "flight_safety": True,
        "landing_success": True,
        "outcome": "landed",
        "failures": [],
        "seed": 378,
        "integration_dt_s": 0.01,
        "simulation_count": 1,
        "conditions": asdict(AprConditions()),
        "disturbances": asdict(RecoveryDisturbances()),
        "design": {"main_propellant_kg": 70_200.0, "rcs_propellant_kg": 300.0},
        "physics": {
            "acceptance": {
                "max_position_error_m": 2.0,
                "max_horizontal_contact_speed_m_s": 0.5,
                "max_vertical_foot_speed_m_s": 1.5,
                "max_contact_tilt_deg": 3.0,
                "max_contact_rate_deg_s": 0.5,
            }
        },
        "criteria": {"separation": True, "touchdown": True, "dynamic_pressure": True},
        "contact": {
            "time_s": contact_time,
            "state": state,
            "foot_velocities_m_s": [
                [2.4769172778908256e-05, -1.004542783320675],
                [2.4769172778897068e-05, -1.004542760017899],
            ],
            "criteria": {
                "position": True,
                "vertical_foot_speed": True,
                "horizontal_foot_speed": True,
                "tilt": True,
                "angular_rate": True,
                "main_propellant": True,
                "rcs_propellant": True,
                "gear_deployed": True,
                "feet_on_pad": True,
            },
        },
        "time_s": contact_time,
        "final_state": state,
        "max_altitude_m": 87551.59055603191,
        "max_dynamic_pressure_pa": 25635.559720201207,
        "main_propellant_remaining_kg": state[6],
        "rcs_propellant_remaining_kg": state[7],
        "main_propellant_used_kg": 70_200.0 - state[6],
        "rcs_propellant_used_kg": 300.0 - state[7],
        "separation": {"time_s": 95.96999999999805},
        "entry_started_s": 301.78999999983364,
        "steps": [
            {
                "time_s": 0.0,
                "phase": "ascent",
                "state": [
                    0.0,
                    12.592534979297781,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    70_200.0,
                    300.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1500.0,
                    1.0,
                    4.011382048348414,
                    60.988617951651584,
                    0.0,
                    0.0,
                ],
                "dynamic_pressure_pa": 0.0,
            },
            {
                "time_s": 95.99999999999805,
                "phase": "flip",
                "state": [
                    3929.6551586553405,
                    37290.987892555735,
                    207.64321426014016,
                    984.6220910801347,
                    0.206626397560052,
                    0.001852286929976491,
                    22891.22713724249,
                    296.9380601716666,
                    0.0,
                    0.0,
                    0.0,
                    0.0013318797294361684,
                    0.0004568920054856278,
                    1500.0,
                    0.9852664911997254,
                    37.290689232409235,
                    27.709310767595827,
                    0.0003021695610989924,
                    0.0011459757394018707,
                ],
                "dynamic_pressure_pa": 2923.2952647137517,
            },
            {
                "time_s": contact_time,
                "phase": "landing",
                "state": state,
                "dynamic_pressure_pa": 0.6067860411725794,
            },
        ],
    }


class SpyManager:
    """Observe API dispatch without launching a process or touching flight state."""

    def __init__(self, directory):
        self.output_dir = directory
        self.submissions = []
        self.video_submissions = []
        self.closed = False
        self.jobs = {}

    def submit(self, settings):
        self.submissions.append(deepcopy(settings))
        job = {"id": JOB_ID, "status": "running"}
        self.jobs[JOB_ID] = job
        return deepcopy(job)

    def get(self, job_id):
        return deepcopy(self.jobs[job_id])

    def start_video(self, job_id, params):
        self.video_submissions.append((job_id, deepcopy(params)))
        return {"id": job_id, "status": "running"}

    def close(self):
        self.closed = True


@pytest.fixture
def api(tmp_path):
    manager = SpyManager(tmp_path)
    app = local_app.create_app(output_dir=tmp_path, manager=manager)
    with TestClient(app, base_url=ORIGIN) as client:
        response = client.get("/api/config")
        assert response.status_code == 200
        config = response.json()
        headers = {"X-DM100-Token": config["token"], "Origin": ORIGIN}
        yield client, manager, headers, config
    assert manager.closed


def test_defaults_map_to_the_existing_fixed_engine():
    conditions, disturbances = local_app.build_conditions({})
    expected = AprConditions()
    assert asdict(disturbances) == asdict(RecoveryDisturbances())
    actual_fields = asdict(conditions)
    expected_fields = asdict(expected)
    for key in ("wind_profile", "forecast_wind_profile", "vertical_wind_profile"):
        actual_fields.pop(key)
        expected_fields.pop(key)
    assert actual_fields == expected_fields
    for altitude in (0.0, 50.0, 1_000.0, 10_000.0, 100_000.0):
        assert conditions.mean_wind(altitude) == expected.mean_wind(altitude)
        assert conditions.mean_wind(altitude, forecast=True) == 0.0
        assert conditions.vertical_wind(altitude) == 0.0


def test_units_and_forecast_are_mapped_without_silent_clipping():
    settings = {
        "ground_wind_m_s": 1.0,
        "upper_wind_m_s": 3.0,
        "vertical_wind_m_s": -1.0,
        "forecast_error_m_s": 2.0,
        "sea_level_temperature_c": 16.0,
        "sea_level_pressure_kpa": 101.0,
        "thrust_scale": 0.98,
        "drag_coefficient_scale": 1.1,
        "observation_delay_steps": 2,
    }
    original = deepcopy(settings)
    conditions, disturbances = local_app.build_conditions(settings)
    assert conditions.sea_level_temperature_k == pytest.approx(289.15)
    assert conditions.sea_level_pressure_pa == pytest.approx(101_000.0)
    assert conditions.mean_wind(0.0) == 1.0
    assert conditions.mean_wind(50.0) == 1.0
    assert conditions.mean_wind(100_000.0) == 3.0
    assert conditions.mean_wind(0.0, forecast=True) == 3.0
    assert conditions.mean_wind(100_000.0, forecast=True) == 5.0
    assert conditions.vertical_wind(0.0) == -1.0
    assert disturbances.thrust_scale == 0.98
    assert disturbances.drag_coefficient_scale == 1.1
    assert disturbances.observation_delay_steps == 2
    assert settings == original


@pytest.mark.parametrize(
    "settings",
    [
        {"ground_wind_m_s": -8.0, "upper_wind_m_s": -20.0},
        {"ground_wind_m_s": 8.0, "upper_wind_m_s": 20.0},
        {"vertical_wind_m_s": -3.0},
        {"vertical_wind_m_s": 3.0},
        {"gust_amplitude_m_s": 5.0, "gust_period_s": 8.0, "gust_duration_s": 8.0},
        {"gust_period_s": 60.0, "gust_duration_s": 120.0},
        {"density_scale": 0.9},
        {"density_scale": 1.1},
        {"latitude_deg": -90.0},
        {"latitude_deg": 90.0},
        {
            "thrust_scale": 0.97,
            "drag_coefficient_scale": 0.85,
            "initial_mass_scale": 0.99,
            "inertia_scale": 0.97,
        },
        {
            "thrust_scale": 1.03,
            "drag_coefficient_scale": 1.15,
            "initial_mass_scale": 1.01,
            "inertia_scale": 1.03,
            "sensor_position_error_m": 2.0,
            "sensor_velocity_error_m_s": 1.0,
            "observation_delay_steps": 2,
            "action_delay_steps": 1,
            "engine_lag_s": 0.05,
        },
    ],
)
def test_eligible_boundary_inputs_use_the_engine_validator(settings):
    validation = local_app.validate_settings(settings)
    assert validation["eligible"] is True, validation["errors"]
    conditions, disturbances = local_app.build_conditions(settings)
    assert isinstance(conditions, AprConditions)
    assert isinstance(disturbances, RecoveryDisturbances)


@pytest.mark.parametrize(
    "settings",
    [
        {"ground_wind_m_s": 8.0001},
        {"upper_wind_m_s": -20.0001},
        {"vertical_wind_m_s": 3.0001},
        {"gust_amplitude_m_s": -0.01},
        {"ground_wind_m_s": 8.0, "forecast_error_m_s": 2.0},
        {"gust_period_s": 7.999},
        {"gust_duration_s": 120.001},
        {"gust_altitude_half_width_m": 99.0},
        {"gust_start_s": 600.0, "gust_amplitude_m_s": 1.0},
        {"density_scale": 1.1001},
        {"thrust_scale": 1.031},
        {"drag_coefficient_scale": 0.849},
        {"initial_mass_scale": 0.989},
        {"inertia_scale": 1.031},
        {"sensor_position_error_m": 2.001},
        {"sensor_velocity_error_m_s": 1.001},
        {"observation_delay_steps": 3},
        {"action_delay_steps": 2},
        {"engine_lag_s": 0.051},
        {"latitude_deg": 90.001},
        {"sea_level_temperature_c": -0.001},
        {"sea_level_pressure_kpa": 105.001},
        {"density_scale": 1.1, "sea_level_pressure_kpa": 105.0},
        {"density_scale": 0.9, "sea_level_pressure_kpa": 95.0},
    ],
)
def test_invalid_inputs_explain_eligibility_and_never_start_a_job(api, settings):
    client, manager, headers, _ = api
    checked = client.post("/api/validate", json=settings, headers=headers)
    assert checked.status_code == 200
    assert checked.json()["eligible"] is False
    assert checked.json()["errors"]
    submitted = client.post("/api/runs", json=settings, headers=headers)
    assert submitted.status_code == 422
    assert manager.submissions == []


def test_forecast_error_is_checked_at_explicit_profile_boundaries(api):
    client, manager, headers, _ = api
    actual = [[0.0, 0.0], [50.0, 0.0], [2_000.0, 4.0], [100_000.0, 4.0]]
    forecast = [[0.0, 0.0], [50.0, 0.0], [2_000.0, 2.0], [100_000.0, 2.0]]
    settings = {"wind_profile": actual, "forecast_wind_profile": forecast}
    assert client.post("/api/validate", json=settings, headers=headers).json()["eligible"]
    forecast[2][1] = 1.999
    response = client.post("/api/runs", json=settings, headers=headers)
    assert response.status_code == 422
    assert manager.submissions == []


@pytest.mark.parametrize(
    "settings",
    [
        {"ground_wind_m_s": True},
        {"sea_level_temperature_c": False},
        {"thrust_scale": "1.0"},
        {"observation_delay_steps": 1.0},
        {"action_delay_steps": True},
        {"ground_wind_m_s": None},
        {"seed": 1},
        {"integration_dt_s": 0.1},
        {"vehicle": {"main_propellant_kg": 1.0}},
        {"health_checks_passed": False},
        {"known_fault": True},
        {"output_dir": "../../outside"},
        {"unknown": 1},
        {"wind_profile": [[False, 0.0], [100_000.0, 0.0]]},
        {"wind_profile": [[0.0, 0.0], [0.0, 1.0], [100_000.0, 0.0]]},
        {"wind_profile": [[50.0, 0.0], [100_000.0, 0.0]]},
        {"wind_profile": [[0.0, 0.0], [1_000.0, 0.0]]},
        {"vertical_wind_profile": [[0.0, 0.0], [100_000.0, 3.001]]},
        {"wind_profile": [[0.0, 0.0], [100_000.0, 0.0]], "ground_wind_m_s": 1.0},
    ],
)
def test_wrong_types_and_out_of_scope_fields_do_not_start_jobs(api, settings):
    client, manager, headers, _ = api
    response = client.post("/api/runs", json=settings, headers=headers)
    assert response.status_code == 422
    assert manager.submissions == []


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_json_numbers_are_rejected_before_dispatch(api, literal):
    client, manager, headers, _ = api
    response = client.post(
        "/api/runs",
        content='{"ground_wind_m_s":' + literal + "}",
        headers={**headers, "Content-Type": "application/json"},
    )
    assert response.status_code == 422
    assert manager.submissions == []


@pytest.mark.parametrize("headers", [{}, {"X-DM100-Token": "wrong"}])
def test_mutations_require_the_session_token(api, headers):
    client, manager, _, _ = api
    assert client.post("/api/runs", json={}, headers=headers).status_code == 403
    assert manager.submissions == []


@pytest.mark.parametrize(
    "origin",
    ["https://example.com", "null", "http://localhost:8765", "http://127.0.0.1:8766"],
)
def test_mutations_reject_every_nonmatching_origin(api, origin):
    client, manager, headers, _ = api
    response = client.post("/api/runs", json={}, headers={**headers, "Origin": origin})
    assert response.status_code == 403
    assert manager.submissions == []


def test_exact_matching_origin_and_token_can_start_a_job(api):
    client, manager, headers, _ = api
    response = client.post("/api/runs", json={}, headers=headers)
    assert response.status_code in (200, 202)
    assert response.json()["id"] == JOB_ID
    assert len(manager.submissions) == 1


@pytest.mark.parametrize("host", ["example.com", "127.0.0.1.evil.test:8765", "0.0.0.0:8765"])
def test_nonloopback_hosts_cannot_read_config_or_dispatch(api, host):
    client, manager, headers, _ = api
    assert client.get("/api/config", headers={"Host": host}).status_code == 403
    assert client.post("/api/runs", json={}, headers={**headers, "Host": host}).status_code == 403
    assert manager.submissions == []


def test_static_ui_is_packaged_and_uses_only_local_assets(api):
    client, _, _, _ = api
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "DM100" in response.text
    asset_text = response.text
    for text in ("착륙 수평속도", "착륙 수직속도", "최초 접지", "결과값"):
        assert text in asset_text
    assert "발 수평속도" not in asset_text
    assert "발 수직속도" not in asset_text
    assert "Run simulation" in asset_text
    assert "Generate video" in asset_text
    # The XML namespace identifies an inline SVG; it is not a fetched asset.
    asset_text = asset_text.replace("http://www.w3.org/2000/svg", "")
    for external in ("https://", "http://", "fonts.googleapis", "cdn."):
        assert external not in asset_text
    for preview in ("DESIGN PREVIEW", "SAVED EXAMPLE", "ILLUSTRATIVE ONLY", "sampleRows"):
        assert preview not in asset_text


def test_video_rendering_notice_is_bilingual_and_shown_before_controls(api):
    client, _, _, _ = api
    html = client.get("/").text
    notice = html.index('id="video-notice"')
    controls = html.index('id="video-width"')
    button = html.index('id="video-button"')
    assert notice < controls < button
    assert 'aria-labelledby="video-notice-title"' in html
    assert 'id="video-rendering-note"' in html
    assert 'data-i18n="videoRenderingNotice" role="status" aria-live="polite"' in html
    for message in (
        "Rendering frames on the CPU can take considerably longer than the flight calculation.",
        "Longer clips and higher resolutions take more time.",
        "For a quick preview, choose 640 × 360 and a short time interval.",
        "You can still view the result table and graphs.",
        "MP4 영상 생성에는 시간이 더 걸립니다",
        "구간이 길거나 해상도가 높을수록 생성 시간이 늘어납니다.",
        "640 × 360 해상도와 짧은 시간 구간을 선택하세요.",
        "결과표와 그래프는 계속 확인할 수 있습니다.",
    ):
        assert message in html
    assert 'byId("video-notice").classList.toggle("hidden",done||rendering)' in html
    assert 'byId("video-rendering-note").classList.toggle("hidden",!rendering)' in html


class MemoryQueue(queue.Queue):
    def __init__(self):
        super().__init__()
        self.closed = False

    def close(self):
        self.closed = True


class OwnedProcess:
    """An owned, cancellable stand-in for one spawned subprocess."""

    exitcode = 0

    def __init__(self):
        self.alive = True
        self.joined = False

    def is_alive(self):
        return self.alive

    def join(self, timeout=None):
        self.joined = True
        self.alive = False


@pytest.fixture
def managed_api(tmp_path, monkeypatch):
    manager = local_app.JobManager(tmp_path)
    spawned = []

    def deferred_spawn(job_id, kind, payload):
        active = {
            "id": job_id,
            "kind": kind,
            "payload": deepcopy(payload),
            "process": OwnedProcess(),
            "messages": MemoryQueue(),
            "cancel": threading.Event(),
        }
        manager._active = active
        spawned.append(active)

    monkeypatch.setattr(manager, "_spawn", deferred_spawn)
    monkeypatch.setattr(local_app, "_video_available", lambda: True)
    app = local_app.create_app(manager=manager)
    with TestClient(app, base_url=ORIGIN) as client:
        config = client.get("/api/config").json()
        headers = {"X-DM100-Token": config["token"], "Origin": ORIGIN}
        yield client, manager, headers, spawned
    assert manager._closed


def complete_flight(manager, spawned, raw):
    """Deliver a persisted worker result to the actual lifecycle monitor."""
    active = spawned[-1]
    directory = manager.output_dir / active["id"]
    (directory / "result.json").write_text(json.dumps(raw), encoding="utf-8")
    (directory / "trajectory.png").write_bytes(b"stored-trajectory")
    active["messages"].put(
        {"kind": "completed", "result": local_app.summarize_result(raw, active["id"])}
    )
    manager._monitor(active)
    return active["id"], directory


@pytest.fixture
def completed_api(managed_api, saved_flight):
    client, manager, headers, spawned = managed_api
    response = client.post("/api/runs", json={}, headers=headers)
    assert response.status_code == 202
    job_id, directory = complete_flight(manager, spawned, saved_flight)
    return client, manager, headers, spawned, job_id, directory


def test_summary_uses_recorded_foot_speed_and_keeps_input_intact(saved_flight):
    preserved = deepcopy(saved_flight)
    summary = local_app.summarize_result(saved_flight, JOB_ID)
    metrics = {row["key"]: row for row in summary["metrics"]}
    assert summary["success"] is True
    assert metrics["landing_vertical_speed_m_s"]["value"] == 1.004542783320675
    assert metrics["landing_horizontal_speed_m_s"]["value"] == 2.4769172778908256e-05
    assert metrics["landing_position_m"]["value"] == abs(saved_flight["contact"]["state"][0])
    assert metrics["landing_vertical_speed_m_s"]["value"] != abs(
        saved_flight["contact"]["state"][3]
    )
    samples = summary["charts"]["samples"]
    assert len(samples) == len(saved_flight["steps"])
    assert samples[1]["x_m"] == saved_flight["steps"][1]["state"][0]
    assert samples[-1]["time_s"] == saved_flight["contact"]["time_s"]
    assert saved_flight == preserved


def test_failed_acceptance_is_a_completed_run_with_true_recorded_graphs(managed_api, saved_flight):
    client, manager, headers, spawned = managed_api
    saved_flight.update(simulated_acceptance_passed=False, landing_success=False)
    saved_flight["contact"]["criteria"]["horizontal_foot_speed"] = False
    saved_flight["contact"]["foot_velocities_m_s"][1][0] = 0.72
    client.post("/api/runs", json={}, headers=headers)
    job_id, directory = complete_flight(manager, spawned, saved_flight)
    response = client.get(f"/api/runs/{job_id}")
    assert response.status_code == 200
    job = response.json()
    assert job["status"] == "completed"
    assert job["error"] is None
    assert job["progress"]["fraction"] == 1
    assert job["result"]["success"] is False
    assert "contact.horizontal_foot_speed" in job["result"]["failed_criteria"]
    horizontal = next(
        row for row in job["result"]["metrics"] if row["key"] == "landing_horizontal_speed_m_s"
    )
    assert horizontal["value"] == 0.72
    assert horizontal["passed"] is False
    assert job["result"]["charts"]["samples"][-1]["time_s"] == saved_flight["time_s"]
    assert client.get(job["result"]["files"]["trajectory_png"]).status_code == 200
    video = client.post(f"/api/runs/{job_id}/video", json={}, headers=headers)
    assert video.status_code == 409
    assert len(spawned) == 1
    assert not (directory / "flight.mp4").exists()


def test_missing_contact_is_reported_as_unavailable_not_zero(saved_flight):
    saved_flight.update(contact=None, landing_success=False, simulated_acceptance_passed=False)
    summary = local_app.summarize_result(saved_flight, JOB_ID)
    assert summary["success"] is False
    assert summary["contact_time_s"] is None
    assert all(row["value"] is None for row in summary["metrics"] if row["group"] == "landing")
    assert len(summary["charts"]["samples"]) == 3


@pytest.mark.parametrize(
    "invalid_record,reason",
    [
        ("source_changed", "source_changed"),
        ("first_foot_nan", "nonfinite_contact"),
        ("second_foot_nan", "nonfinite_contact"),
        ("mission_flag_mismatch", "inconsistent_acceptance"),
    ],
)
def test_invalid_record_cannot_be_presented_as_success_or_rendered_as_a_success_video(
    managed_api, saved_flight, invalid_record, reason
):
    client, manager, headers, spawned = managed_api
    if invalid_record == "source_changed":
        saved_flight["local_app_provenance"] = {"source_unchanged": False}
    elif invalid_record == "mission_flag_mismatch":
        saved_flight["mission_success"] = False
    else:
        foot = 0 if invalid_record == "first_foot_nan" else 1
        saved_flight["contact"]["foot_velocities_m_s"][foot][1] = float("nan")
    preserved = deepcopy(saved_flight)
    summary = local_app.summarize_result(saved_flight, JOB_ID)
    assert summary["success"] is False
    assert summary["result_valid"] is False
    assert reason in {error["key"] for error in summary["integrity_errors"]}
    assert summary["failure_labels"], (
        "Invalid records need a visible reason distinct from a flight failure"
    )
    for error in summary["integrity_errors"]:
        assert error["label"]["ko"]
        assert error["label"]["en"]
    if invalid_record.endswith("foot_nan"):
        vertical = next(
            metric for metric in summary["metrics"] if metric["key"] == "landing_vertical_speed_m_s"
        )
        assert vertical["value"] is None
        assert vertical["passed"] is not True
    # Summarizing must leave the raw engine flags and evidence unmodified.
    assert json.dumps(saved_flight, sort_keys=True) == json.dumps(preserved, sort_keys=True)
    assert saved_flight["simulated_acceptance_passed"] is True
    client.post("/api/runs", json={}, headers=headers)
    job_id, directory = complete_flight(manager, spawned, saved_flight)
    assert manager.get(job_id)["result"]["success"] is False
    assert client.post(f"/api/runs/{job_id}/video", json={}, headers=headers).status_code == 409
    assert len(spawned) == 1
    assert not (directory / "flight.mp4").exists()


def test_duplicate_submission_is_busy_then_completion_allows_a_new_run(managed_api, saved_flight):
    client, manager, headers, spawned = managed_api
    settings = {"ground_wind_m_s": 2.0, "upper_wind_m_s": 4.0}
    expected = deepcopy(settings)
    first = client.post("/api/runs", json=settings, headers=headers)
    first_id = first.json()["id"]
    settings["ground_wind_m_s"] = -2.0
    duplicate = client.post("/api/runs", json=settings, headers=headers)
    assert duplicate.status_code == 409
    assert len(spawned) == 1
    assert len(list(manager.output_dir.iterdir())) == 1
    input_file = manager.output_dir / first_id / "input.json"
    snapshot = json.loads(input_file.read_text(encoding="utf-8"))
    assert snapshot["settings"] == expected
    assert snapshot["seed"] == 378
    assert snapshot["integration_dt_s"] == 0.01
    complete_flight(manager, spawned, saved_flight)
    second = client.post("/api/runs", json={}, headers=headers)
    assert second.status_code == 202
    assert second.json()["id"] != first_id
    assert len(spawned) == 2
    assert json.loads(input_file.read_text(encoding="utf-8")) == snapshot


def test_progress_and_worker_errors_are_observed_and_release_the_busy_slot(managed_api):
    client, manager, headers, spawned = managed_api
    response = client.post("/api/runs", json={}, headers=headers)
    job_id = response.json()["id"]
    active = spawned[-1]
    snapshots = []
    original_get = active["messages"].get

    def observed_get(*args, **kwargs):
        snapshots.append(manager.get(job_id))
        return original_get(*args, **kwargs)

    active["messages"].get = observed_get
    active["messages"].put(
        {
            "kind": "progress",
            "progress": {
                "phase": "entry",
                "time_s": 300.0,
                "deadline_s": 600.0,
                "fraction": 0.5,
            },
        }
    )
    active["messages"].put({"kind": "execution_error", "error": "test worker failure"})
    manager._monitor(active)
    assert snapshots[-1]["progress"]["phase"] == "entry"
    assert snapshots[-1]["progress"]["fraction"] == 0.5
    result = client.get(f"/api/runs/{job_id}").json()
    assert result["status"] == "execution_error"
    assert result["error"] == "test worker failure"
    assert result["result"] is None
    assert manager._active is None
    assert active["messages"].closed
    assert active["process"].joined
    assert client.post("/api/runs", json={}, headers=headers).status_code == 202


def test_close_cancels_only_the_owned_process_and_prevents_new_jobs(managed_api):
    client, manager, headers, spawned = managed_api
    client.post("/api/runs", json={}, headers=headers)
    active = spawned[-1]
    manager.close()
    assert active["cancel"].is_set()
    assert active["process"].joined
    assert not active["process"].is_alive()
    assert client.post("/api/runs", json={}, headers=headers).status_code == 409
    assert len(spawned) == 1


def test_unexpected_worker_exit_becomes_an_error_and_releases_the_slot(managed_api):
    client, manager, headers, spawned = managed_api
    job_id = client.post("/api/runs", json={}, headers=headers).json()["id"]
    active = spawned[-1]
    active["process"].alive = False
    active["process"].exitcode = 23
    manager._monitor(active)
    result = client.get(f"/api/runs/{job_id}").json()
    assert result["status"] == "execution_error"
    assert "23" in result["error"]
    assert manager._active is None
    assert active["messages"].closed
    assert client.post("/api/runs", json={}, headers=headers).status_code == 202


def test_process_launch_error_keeps_the_input_snapshot_and_does_not_stay_busy(
    tmp_path, monkeypatch
):
    manager = local_app.JobManager(tmp_path)
    queues = []

    def new_queue():
        messages = MemoryQueue()
        queues.append(messages)
        return messages

    class CannotStartProcess:
        def __init__(self, **kwargs):
            pass

        def start(self):
            raise OSError("test process launch denied")

    monkeypatch.setattr(
        manager,
        "_context",
        SimpleNamespace(Queue=new_queue, Event=threading.Event, Process=CannotStartProcess),
    )
    job = manager.submit({})
    assert job["status"] == "execution_error"
    assert "launch denied" in manager.get(job["id"])["error"]
    assert (tmp_path / job["id"] / "input.json").is_file()
    assert manager._active is None
    assert queues[0].closed
    assert manager.submit({})["id"] != job["id"]
    manager.close()


def test_allowed_downloads_preserve_raw_result_and_hide_internal_input_snapshot(completed_api):
    client, manager, _, _, job_id, directory = completed_api
    original = (directory / "result.json").read_bytes()
    raw = client.get(f"/api/runs/{job_id}/files/result.json")
    assert raw.status_code == 200
    assert raw.content == original
    assert raw.headers["content-type"].startswith("application/json")
    assert "attachment" in raw.headers["content-disposition"]
    assert client.get(f"/api/runs/{job_id}/files/trajectory.png").content == b"stored-trajectory"
    assert client.get(f"/api/runs/{job_id}/files/input.json").status_code == 404
    assert client.get(f"/api/runs/{job_id}/files/flight.mp4").status_code == 404
    status = manager.get(job_id)
    status["result"]["success"] = False
    assert manager.get(job_id)["result"]["success"] is True
    assert (directory / "result.json").read_bytes() == original


@pytest.mark.parametrize(
    "path",
    [
        "/api/runs/../files/result.json",
        "/api/runs/%2E%2E/files/result.json",
        "/api/runs/%2E%2E%2Foutside/files/result.json",
        "/api/runs/" + "A" * 32 + "/files/result.json",
        "/api/runs/" + "0" * 31 + "/files/result.json",
        "/api/runs/" + "0" * 32 + "/files/result.json",
    ],
)
def test_traversal_or_unknown_job_identifiers_never_serve_files(api, path):
    client, _, _, _ = api
    assert client.get(path).status_code == 404


def test_filename_traversal_and_private_files_are_not_served(completed_api):
    client, _, _, _, job_id, directory = completed_api
    private = directory.parent / "private.json"
    private.write_text("private fixture", encoding="utf-8")
    for filename in ("%2E%2E%2Fprivate.json", "..%5Cprivate.json", "input.json", "private.json"):
        response = client.get(f"/api/runs/{job_id}/files/{filename}")
        assert response.status_code == 404
        assert b"private fixture" not in response.content


@pytest.mark.parametrize(
    "params",
    [
        {"width": True},
        {"width": 640.0},
        {"width": 639},
        {"width": 1921},
        {"width": 800},
        {"start_s": -0.01},
        {"end_s": 500},
        {"start_s": 2, "end_s": 2},
        {"start_s": 3, "end_s": 2},
        {"start_s": True},
        {"end_s": "10"},
        {"fps": 24},
        {"output_path": "../../outside.mp4"},
        {"filename": "../../outside.mp4"},
    ],
)
def test_invalid_video_parameters_do_not_dispatch_or_write(completed_api, params):
    client, _, headers, spawned, job_id, directory = completed_api
    response = client.post(f"/api/runs/{job_id}/video", json=params, headers=headers)
    assert response.status_code == 422
    assert len(spawned) == 1
    assert not (directory / "flight.mp4").exists()
    assert client.get(f"/api/runs/{job_id}/video/status").json()["status"] == "not_requested"


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_video_times_do_not_dispatch(completed_api, literal):
    client, _, headers, spawned, job_id, _ = completed_api
    response = client.post(
        f"/api/runs/{job_id}/video",
        content='{"start_s":' + literal + "}",
        headers={**headers, "Content-Type": "application/json"},
    )
    assert response.status_code == 422
    assert len(spawned) == 1


def test_video_uses_saved_record_without_recalculating_flight(completed_api):
    client, manager, headers, spawned, job_id, directory = completed_api
    original = (directory / "result.json").read_bytes()
    response = client.post(
        f"/api/runs/{job_id}/video",
        json={"start_s": 95, "end_s": 96, "width": 640},
        headers=headers,
    )
    assert response.status_code == 202
    assert response.json()["status"] == "rendering"
    active = spawned[-1]
    assert active["kind"] == "video"
    assert active["payload"] == {"start_s": 95.0, "end_s": 96.0, "width": 640}
    assert client.post(f"/api/runs/{job_id}/video", json={}, headers=headers).status_code == 409
    assert client.post("/api/runs", json={}, headers=headers).status_code == 409
    (directory / "flight.mp4").write_bytes(b"stored-video")
    active["messages"].put({"kind": "completed"})
    manager._monitor(active)
    video = client.get(f"/api/runs/{job_id}/video/status").json()
    assert video["status"] == "completed"
    assert client.get(video["url"]).content == b"stored-video"
    assert manager.get(job_id)["status"] == "completed"
    assert client.post(f"/api/runs/{job_id}/video", json={}, headers=headers).status_code == 409
    assert (directory / "result.json").read_bytes() == original
    assert len(spawned) == 2


def test_unavailable_video_renderer_reports_actionable_error(completed_api, monkeypatch):
    client, _, headers, spawned, job_id, _ = completed_api
    monkeypatch.setattr(local_app, "_video_available", lambda: False)
    response = client.post(f"/api/runs/{job_id}/video", json={}, headers=headers)
    assert response.status_code == 409
    assert "FFmpeg" in response.json()["detail"]
    assert len(spawned) == 1


def test_video_error_does_not_change_the_completed_flight_and_allows_retry(completed_api):
    client, manager, headers, spawned, job_id, directory = completed_api
    original = (directory / "result.json").read_bytes()
    original_result = manager.get(job_id)["result"]
    client.post(f"/api/runs/{job_id}/video", json={}, headers=headers)
    active = spawned[-1]
    active["messages"].put({"kind": "execution_error", "error": "test renderer failure"})
    manager._monitor(active)
    video = client.get(f"/api/runs/{job_id}/video/status").json()
    assert video["status"] == "execution_error"
    assert video["error"] == "test renderer failure"
    assert video["url"] is None
    assert manager.get(job_id)["status"] == "completed"
    assert manager.get(job_id)["result"] == original_result
    assert client.get(f"/api/runs/{job_id}/files/flight.mp4").status_code == 404
    assert (directory / "result.json").read_bytes() == original
    assert client.post(f"/api/runs/{job_id}/video", json={}, headers=headers).status_code == 202


@pytest.mark.parametrize("corruption", ["source_changed", "second_foot_nan"])
def test_video_rechecks_saved_json_even_when_the_cached_result_passed(completed_api, corruption):
    client, manager, headers, spawned, job_id, directory = completed_api
    result_file = directory / "result.json"
    raw = json.loads(result_file.read_text(encoding="utf-8"))
    if corruption == "source_changed":
        raw["local_app_provenance"] = {"source_unchanged": False}
    else:
        raw["contact"]["foot_velocities_m_s"][1][1] = float("nan")
    # Even an old derived validation field must not override current evidence.
    raw["local_app_result_validation"] = {"success": True, "result_valid": True}
    result_file.write_text(json.dumps(raw), encoding="utf-8")
    original = result_file.read_bytes()
    assert manager.get(job_id)["result"]["success"] is True
    response = client.post(f"/api/runs/{job_id}/video", json={}, headers=headers)
    assert response.status_code == 409
    assert len(spawned) == 1
    assert manager.get(job_id)["video"]["status"] == "not_requested"
    assert result_file.read_bytes() == original
    assert not (directory / "flight.mp4").exists()


def test_worker_runs_fixed_engine_once_and_persists_exact_raw_values(
    tmp_path, monkeypatch, saved_flight
):
    captured = []
    preserved = deepcopy(saved_flight)

    class RecordedFlight:
        def __init__(self, **kwargs):
            captured.append(kwargs)

        def run(self, *, progress):
            progress(SimpleNamespace(phase="landing", time_s=400.0, deadline_s=600.0))
            return deepcopy(saved_flight)

    monkeypatch.setattr(local_app, "AprFlight", RecordedFlight)
    monkeypatch.setattr(local_app, "_save_plot", lambda raw, path: path.write_bytes(b"trajectory"))
    directory = tmp_path / JOB_ID
    directory.mkdir()
    messages = MemoryQueue()
    local_app._run_flight_worker({}, str(directory), messages, threading.Event())
    first, final = messages.get_nowait(), messages.get_nowait()
    assert first["kind"] == "progress"
    assert final["kind"] == "completed"
    assert len(captured) == 1
    assert captured[0]["fin_mode"] == "active"
    assert captured[0]["integration_dt_s"] == 0.01
    assert isinstance(captured[0]["conditions"], AprConditions)
    assert isinstance(captured[0]["disturbances"], RecoveryDisturbances)
    persisted = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    provenance = persisted.pop("local_app_provenance")
    validation = persisted.pop("local_app_result_validation")
    assert validation == {"result_valid": True, "success": True, "integrity_errors": []}
    assert provenance["requested_settings"] == {}
    assert "apr_recovery.py" in provenance["source_sha256_before"]
    assert provenance["source_sha256_before"] == provenance["source_sha256_after"]
    assert provenance["source_unchanged"] is True
    assert persisted == json.loads(json.dumps(preserved))
    assert saved_flight == preserved


def test_video_worker_reads_saved_json_and_never_invokes_the_engine(
    tmp_path, monkeypatch, saved_flight
):
    from powered_landing_guidance import visualization

    directory = tmp_path / JOB_ID
    directory.mkdir()
    result_file = directory / "result.json"
    result_file.write_text(json.dumps(saved_flight), encoding="utf-8")
    original = result_file.read_bytes()
    calls = []

    def render(raw, destination, **kwargs):
        calls.append((raw, destination, kwargs))
        destination.write_bytes(b"rendered fixture")

    def no_new_flight(**kwargs):
        pytest.fail("Video rendering attempted a new flight")

    monkeypatch.setattr(local_app, "AprFlight", no_new_flight)
    monkeypatch.setattr(visualization, "save_nominal_flight_replay", render)
    messages = MemoryQueue()
    params = {"start_s": 95.0, "end_s": 96.0, "width": 640}
    local_app._video_worker(params, str(directory), messages, threading.Event())
    assert messages.get_nowait()["kind"] == "completed"
    assert len(calls) == 1
    assert calls[0][0] == json.loads(json.dumps(saved_flight))
    assert calls[0][1] == directory / "flight.mp4"
    assert calls[0][2] == {**params, "fps": 24}
    assert result_file.read_bytes() == original


def test_video_worker_itself_rejects_invalid_records_before_rendering(
    tmp_path, monkeypatch, saved_flight
):
    from powered_landing_guidance import visualization

    directory = tmp_path / JOB_ID
    directory.mkdir()
    saved_flight["mission_success"] = False
    saved_flight["local_app_result_validation"] = {"success": True, "result_valid": True}
    result_file = directory / "result.json"
    result_file.write_text(json.dumps(saved_flight), encoding="utf-8")
    original = result_file.read_bytes()

    def no_render(*args, **kwargs):
        pytest.fail("An invalid saved record reached the video renderer")

    monkeypatch.setattr(visualization, "save_nominal_flight_replay", no_render)
    messages = MemoryQueue()
    local_app._video_worker(
        {"start_s": 95, "end_s": 96, "width": 640},
        str(directory),
        messages,
        threading.Event(),
    )
    result = messages.get_nowait()
    assert result["kind"] == "execution_error"
    assert "valid saved flight" in result["error"]
    assert result_file.read_bytes() == original
    assert not (directory / "flight.mp4").exists()
