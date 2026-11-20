"""Unit and integration tests for DM100 Reusable RTLS recovery."""

from __future__ import annotations

import json
from math import cos, hypot, radians, sin
from pathlib import Path

import pytest

from powered_landing_guidance.apr_recovery import (
    AprConditions,
    RecoveryDisturbances,
    load_apr_operating,
)
from powered_landing_guidance.observation_models import ControllerOptions, SensorSettings
from powered_landing_guidance.recovery_physics import AtmosphereSettings, atmosphere
from powered_landing_guidance.rtls import (
    AllocationInput,
    AtmosphereLUT,
    ContactSettings,
    ConvexLandingProblem,
    PlantConfig,
    PriorityControlAllocator,
    RtlsAerodynamics,
    RtlsController,
    RtlsFinSettings,
    RtlsFlight,
    RtlsPropulsionSettings,
    TelemetryPacket,
    WindLUT,
    compute_body_aero,
    compute_fin_aero,
    evaluate_ground_contact,
    pitch_to_quat_wxyz,
    solve_convex_hoverslam,
)

ROOT = Path(__file__).resolve().parents[1]


def test_atmosphere_lut_matches_reference() -> None:
    settings = AtmosphereSettings(density_scale=1.05, sea_level_temperature_k=290.0)
    lut = AtmosphereLUT(settings, step_m=100.0)
    for h in (0.0, 250.0, 1_550.0, 11_000.0, 25_000.0, 47_500.0, 80_000.0):
        t_ref, p_ref, rho_ref = atmosphere(h, settings)
        t_lut, p_lut, rho_lut = lut.at(h)
        assert abs(t_lut - t_ref) / t_ref < 5e-4
        assert abs(p_lut - p_ref) / p_ref < 1.5e-3
        assert abs(rho_lut - rho_ref) / rho_ref < 1.5e-3


def test_ascent_vs_tail_first_descent_aerodynamics() -> None:
    atmo = AtmosphereLUT(AtmosphereSettings())
    wind = WindLUT(AprConditions(), forecast=False)
    aero = RtlsAerodynamics()

    # Nose-first ascent vs tail-first descent at identical speed and altitude
    _, fz_asc, _, q_asc, _, _ = compute_body_aero(
        5_000.0, 0.0, 200.0, 0.0, 0.0, 8.0, atmo, wind, 0.0, aero, upper_attached=True
    )
    _, fz_desc, _, q_desc, _, _ = compute_body_aero(
        5_000.0, 0.0, -200.0, 0.0, 0.0, 7.0, atmo, wind, 0.0, aero, upper_attached=False
    )
    assert abs(q_asc - q_desc) < 1e-6
    assert abs(fz_desc) > 3.0 * abs(fz_asc)

    # Small clockwise tilt during vertical descent produces westward (-x) lift and restoring torque
    fx_tilt, _, tau_tilt, _, _, alpha_deg = compute_body_aero(
        5_000.0, 0.0, -200.0, radians(6.0), 0.0, 7.0, atmo, wind, 0.0, aero, upper_attached=False
    )
    assert alpha_deg > 5.0
    assert fx_tilt < -5_000.0
    assert tau_tilt < 0.0


def test_grid_fin_effective_aoa_and_weathercock() -> None:
    atmo = AtmosphereLUT(AtmosphereSettings())
    wind = WindLUT(AprConditions(), forecast=False)
    fins = RtlsFinSettings()

    # Feathered fin (delta == alpha_retro) yields ~0 lift and ~0 alpha_eff
    alpha_retro = radians(8.0)
    _, _, tau_feathered, hinge_feathered, a_eff_feathered = compute_fin_aero(
        5_000.0, 0.0, -200.0, alpha_retro, 7.0, alpha_retro, 1.0, atmo, wind, 0.0, fins
    )
    assert abs(a_eff_feathered) < 1e-3
    assert hinge_feathered < 1e-3

    # Deflecting fin past alpha_retro generates positive pitch-up torque and nonzero hinge load
    _, _, tau_pos, hinge_pos, a_eff_pos = compute_fin_aero(
        5_000.0,
        0.0,
        -200.0,
        alpha_retro,
        7.0,
        alpha_retro + radians(6.0),
        1.0,
        atmo,
        wind,
        0.0,
        fins,
    )
    assert a_eff_pos == pytest.approx(6.0, abs=0.05)
    assert tau_pos > tau_feathered + 10_000.0
    assert hinge_pos > 0.0


def test_propulsion_cluster_throttle_floor_and_bounds() -> None:
    prop = RtlsPropulsionSettings()
    lo3, hi3 = prop.throttle_bounds(3)
    lo1, hi1 = prop.throttle_bounds(1)
    assert lo3 == pytest.approx(prop.min_engine_throttle)
    assert hi3 == pytest.approx(1.0)
    assert lo1 == pytest.approx(prop.min_engine_throttle / 3.0)
    assert hi1 == pytest.approx(1.0 / 3.0)
    # Non-zero request below minimum floor is clamped up to physical floor
    assert prop.quantize_throttle(0.05, 1) == pytest.approx(lo1)
    # Above 1-engine max is clamped to 1/3
    assert prop.quantize_throttle(0.9, 1) == pytest.approx(hi1)
    # Cutoff returns exact 0
    assert prop.quantize_throttle(0.0, 1) == 0.0
    assert prop.quantize_throttle(0.5, 0) == 0.0


def test_priority_allocator_fins_tvc_rcs_order() -> None:
    prop = RtlsPropulsionSettings()
    fins = RtlsFinSettings()
    alloc = PriorityControlAllocator(prop, fins)

    # 1. Aero glide (no thrust, fins active): fins handle torque with 0 RCS
    out_fin = alloc.allocate(
        AllocationInput(
            requested_torque_nm=25_000.0,
            background_torque_nm=0.0,
            acceleration_inertia_kg_m2=650_000.0,
            lever_arm_m=7.0,
            sizing_thrust_n=0.0,
            fin_effectiveness_nm_per_rad=400_000.0,
            fin_zero_deflection_torque_nm=0.0,
            fin_alpha_retro_rad=0.0,
            dynamic_pressure_pa=10_000.0,
            fin_extension=1.0,
            rcs_available=True,
            allow_fin_allocation=True,
        )
    )
    assert abs(out_fin.fin_deflection_rad) > 0.05
    assert out_fin.gimbal_rad == 0.0
    assert out_fin.rcs_torque_nm == 0.0

    # 2. Powered burn with unsaturated TVC: TVC handles torque with ZERO RCS chatter
    out_tvc = alloc.allocate(
        AllocationInput(
            requested_torque_nm=40_000.0,
            background_torque_nm=0.0,
            acceleration_inertia_kg_m2=650_000.0,
            lever_arm_m=7.0,
            sizing_thrust_n=400_000.0,
            fin_effectiveness_nm_per_rad=0.0,
            fin_zero_deflection_torque_nm=0.0,
            fin_alpha_retro_rad=0.0,
            dynamic_pressure_pa=500.0,
            fin_extension=0.0,
            rcs_available=True,
            allow_fin_allocation=False,
            current_gimbal_rad=0.0,
            current_thrust_n=400_000.0,
        )
    )
    assert out_tvc.gimbal_rad < 0.0
    assert out_tvc.rcs_torque_nm == 0.0

    # 3. Vacuum flip (no fins, no thrust): RCS provides full requested torque
    out_rcs = alloc.allocate(
        AllocationInput(
            requested_torque_nm=30_000.0,
            background_torque_nm=0.0,
            acceleration_inertia_kg_m2=650_000.0,
            lever_arm_m=7.0,
            sizing_thrust_n=0.0,
            fin_effectiveness_nm_per_rad=0.0,
            fin_zero_deflection_torque_nm=0.0,
            fin_alpha_retro_rad=0.0,
            dynamic_pressure_pa=0.0,
            fin_extension=0.0,
            rcs_available=True,
            allow_fin_allocation=False,
        )
    )
    assert out_rcs.rcs_torque_nm == pytest.approx(30_000.0)


def test_ground_contact_upright_and_hull_collision() -> None:
    cs = ContactSettings()

    # High in air -> zero contact
    air = evaluate_ground_contact(
        x_m=0.0,
        z_m=15.0,
        vx_m_s=0.0,
        vz_m_s=-1.0,
        theta_rad=0.0,
        omega_rad_s=0.0,
        total_mass_kg=22_000.0,
        cg_m=7.0,
        gear_extension=1.0,
        settings=cs,
    )
    assert not air.any_contact
    assert air.force_z_n == 0.0

    # Gentle compression on both feet -> upward normal force, no hull collision
    touch = evaluate_ground_contact(
        x_m=0.0,
        z_m=7.25,
        vx_m_s=0.0,
        vz_m_s=-0.8,
        theta_rad=0.0,
        omega_rad_s=0.0,
        total_mass_kg=22_000.0,
        cg_m=7.0,
        gear_extension=1.0,
        settings=cs,
    )
    assert touch.left_in_contact and touch.right_in_contact
    assert touch.force_z_n > 100_000.0
    assert not touch.body_ground_collision

    # Severe tip-over -> hull ground collision flagged
    tipped = evaluate_ground_contact(
        x_m=0.0,
        z_m=2.0,
        vx_m_s=0.0,
        vz_m_s=-1.0,
        theta_rad=radians(75.0),
        omega_rad_s=0.5,
        total_mass_kg=22_000.0,
        cg_m=7.0,
        gear_extension=1.0,
        settings=cs,
    )
    assert tipped.body_ground_collision


def test_convex_hoverslam_lp_feasibility() -> None:
    prob = ConvexLandingProblem(
        x0_m=15.0,
        z0_m=300.0,
        vx0_m_s=-2.0,
        vz0_m_s=-65.0,
        a_min_m_s2=10.8,
        a_max_m_s2=24.0,
        gravity_m_s2=9.80665,
        v_touchdown_m_s=0.95,
        max_tilt_deg=14.0,
    )
    sol = solve_convex_hoverslam(prob)
    assert sol.feasible
    assert sol.flight_time_s > 5.0
    assert len(sol.times_s) == prob.n_steps + 1
    assert abs(sol.x_m[-1]) < 0.5
    assert abs(sol.z_m[-1]) < 0.5
    assert abs(sol.vz_m_s[-1] + prob.v_touchdown_m_s) < 0.5


def test_replay_quaternion_unit_norm_and_axis() -> None:
    for deg in (-90.0, -15.0, 0.0, 12.5, 45.0):
        th = radians(deg)
        w, qx, qy, qz = pitch_to_quat_wxyz(th)
        assert hypot(w, qx, qy, qz) == pytest.approx(1.0, abs=1e-12)
        assert qx == 0.0 and qz == 0.0
        # Rotating body +Z=(0,0,1) by q=(cos(th/2),0,sin(th/2),0) gives (sin(th),0,cos(th))
        rot_x = 2.0 * (w * qy)
        rot_z = 1.0 - 2.0 * (qy * qy)
        assert rot_x == pytest.approx(sin(th), abs=1e-12)
        assert rot_z == pytest.approx(cos(th), abs=1e-12)


def test_strict_causality_controller_ignores_hidden_density_scale() -> None:
    # Build two controllers with identical forecast conditions except hidden density_scale
    c_nom = AprConditions(density_scale=1.0)
    c_hidden = AprConditions(density_scale=1.10)
    ctrl_a = RtlsController(
        design=RtlsFlight().design,
        conditions=c_nom,
        aero=RtlsAerodynamics(),
        fins=RtlsFinSettings(),
        propulsion=RtlsPropulsionSettings(),
    )
    ctrl_b = RtlsController(
        design=RtlsFlight().design,
        conditions=c_hidden,
        aero=RtlsAerodynamics(),
        fins=RtlsFinSettings(),
        propulsion=RtlsPropulsionSettings(),
    )
    pkt = TelemetryPacket(
        time_s=0.0,
        measurement_time_s=0.0,
        x_m=0.0,
        z_m=11.8,
        vx_m_s=0.0,
        vz_m_s=0.0,
        theta_rad=0.0,
        omega_rad_s=0.0,
        main_propellant_kg=70_200.0,
        rcs_propellant_kg=300.0,
        gear_extension=0.0,
        fin_deflection_rad=0.0,
        fin_extension=0.0,
        gimbal_rad=0.0,
        engine_throttle=0.0,
        upper_attached=True,
        on_pad=True,
        touchdown_detected=False,
        active_engines=0,
        restarts_used=0,
    )
    assert ctrl_a.step(pkt) == ctrl_b.step(pkt)


def test_rtls_calm_end_to_end_and_replay_export(tmp_path: Path) -> None:
    flight = RtlsFlight(
        conditions=AprConditions(),
        disturbances=RecoveryDisturbances(),
        plant_config=PlantConfig(record_replay=True, seed=378, sensor_seed=1378),
    )
    res = flight.run()
    assert res["requirements_passed"], f"Failed criteria: {res['failed_criteria']}"
    assert res["outcome"] == "landed"
    assert res["restarts_used"] <= 3
    assert res["main_remaining_kg"] > 9_000.0
    assert res["position_error_m"] <= 2.0
    assert res["post_contact"]["standing_stable"] is True
    assert res["convex_reference_feasible"] is True

    # Verify true unpowered grid-fin aerodynamic glide (>99% engine-off, tail-off only)!
    ag = res["phases"]["aero_glide"]
    assert ag["active_fraction"] < 0.01
    assert ag["main_used_kg"] < 25.0
    assert ag["start_z_m"] - ag["end_z_m"] > 10_000.0
    assert ag["max_fin_deg"] > 2.0

    # Verify 50 Hz UE5/MuJoCo replay schema export
    replay_path = flight.export_replay(tmp_path / "calm_replay.json", metadata={"case": "calm"})
    payload = json.loads(replay_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "dm100_replay_3d_v1"
    assert payload["coordinate_frame"] == "ENU_right_handed_SI"
    assert payload["units"]["quaternion_order"] == "wxyz"
    assert payload["metadata"]["replay_hz"] == 50.0
    assert payload["metadata"]["case"] == "calm"
    assert payload["frame_count"] == len(payload["frames"]) > 10_000
    f0 = payload["frames"][0]
    assert len(f0["pos_enu_m"]) == 3
    assert len(f0["quat_wxyz"]) == 4
    assert len(f0["fin_deflection_rad"]) == 4
    assert len(f0["leg_stroke_m"]) == 4


def test_rtls_example_weather_extended_sensors_end_to_end() -> None:
    cond, dist = load_apr_operating(ROOT / "configs" / "apr-operating-v1.yaml")
    sens = SensorSettings(
        attitude_error_deg=0.05,
        gyro_error_deg_s=0.03,
        attitude_bias_deg=0.02,
        gyro_bias_deg_s=-0.015,
        dropout_period_steps=80,
        dropout_length_steps=2,
    )
    flight = RtlsFlight(
        conditions=cond,
        disturbances=dist,
        sensor_settings=sens,
        controller_options=ControllerOptions(),
        plant_config=PlantConfig(record_replay=False, seed=378, sensor_seed=1378),
    )
    res = flight.run()
    assert res["requirements_passed"], f"Failed criteria: {res['failed_criteria']}"
    assert res["outcome"] == "landed"
    assert res["main_remaining_kg"] > 9_000.0
    assert res["post_contact"]["standing_stable"] is True
    assert res["phases"]["aero_glide"]["active_fraction"] < 0.01
    assert res["phases"]["aero_glide"]["main_used_kg"] < 25.0
    assert (
        res["phases"]["aero_glide"]["start_z_m"] - res["phases"]["aero_glide"]["end_z_m"] > 10_000.0
    )


def test_rtls_kinematic_touchdown_mode() -> None:
    flight = RtlsFlight(
        conditions=AprConditions(),
        disturbances=RecoveryDisturbances(),
        plant_config=PlantConfig(
            landing_mode="kinematic_touchdown",
            record_replay=False,
            seed=378,
            sensor_seed=1378,
        ),
    )
    res = flight.run()
    assert res["requirements_passed"], f"Failed criteria: {res['failed_criteria']}"
    assert res["criteria"]["engine_off_standing"] is None
    assert res["criteria"]["leg_stroke"] is None


def test_local_app_rtls_worker_and_visualization_replay(tmp_path: Path) -> None:
    import queue
    import threading

    from powered_landing_guidance import local_app
    from powered_landing_guidance.visualization import _NominalFlightReplay

    job_dir = tmp_path / ("b" * 32)
    job_dir.mkdir()
    messages: queue.Queue = queue.Queue()
    local_app._run_flight_worker({}, str(job_dir), messages, threading.Event())
    events = []
    while not messages.empty():
        events.append(messages.get_nowait())
    assert any(e["kind"] == "progress" for e in events)
    final = events[-1]
    assert final["kind"] == "completed", final
    summary = final["result"]
    assert summary["success"] is True
    assert summary["result_valid"] is True
    assert summary["landing_assessment"] == "contact_dynamics"
    assert summary["post_contact"]["standing_stable"] is True
    assert "aero_glide" in summary["phases"]
    assert summary["files"]["replay_3d_json"].endswith("/replay_3d.json")
    assert (job_dir / "result.json").is_file()
    assert (job_dir / "trajectory.png").is_file()
    assert (job_dir / "replay_3d.json").is_file()

    # Verify saved JSON loads cleanly in the MP4 visualization replay model
    raw = json.loads((job_dir / "result.json").read_text(encoding="utf-8"))
    replay = _NominalFlightReplay(raw)
    _, _, _, _, phase_td, _ = replay.sample(raw["time_s"])
    assert phase_td in ("touchdown", "touchdown_settle")


def test_local_app_ui_config_and_interactive_html(tmp_path: Path) -> None:
    pytest.importorskip("fastapi", reason="Install the ui extra to test the local application")
    pytest.importorskip("httpx", reason="Install the dev extra for TestClient")
    from fastapi.testclient import TestClient

    from powered_landing_guidance import local_app

    app = local_app.create_app(tmp_path / "runs")
    try:
        with TestClient(app, base_url="http://127.0.0.1:8765") as client:
            cfg = client.get("/api/config").json()
            preset_ids = {p["id"] for p in cfg["presets"]}
            assert {
                "calm",
                "example_weather",
                "extreme_shear_gust",
                "extreme_maxq",
                "extreme_heavy",
            } <= preset_ids

            html = client.get("/").text
            for required_id in (
                'id="mode-badge"',
                'id="replay-canvas"',
                'id="replay-play"',
                'id="replay-scrub"',
                'id="tab-control"',
                'id="phase-panel"',
                "downloadReplay3d",
            ):
                assert required_id in html
            assert "https://" not in html
            assert "http://" not in html.replace("http://www.w3.org/2000/svg", "")
    finally:
        app.state.manager.close()


def test_zero_tilt_predictor_symmetry_and_unwind_margin() -> None:
    from powered_landing_guidance.rtls.guidance import _predict_zero_tilt_state

    # Positive (East) tilt + positive rate commits positive East delta-v before unwinding to 0
    x_pos, vx_pos = _predict_zero_tilt_state(
        0.0, 0.0, radians(6.0), radians(2.0), 12.0, radians(2.15)
    )
    x_neg, vx_neg = _predict_zero_tilt_state(
        0.0, 0.0, radians(-6.0), radians(-2.0), 12.0, radians(2.15)
    )
    assert vx_pos > 1.5
    assert vx_pos == pytest.approx(-vx_neg, rel=1e-9)
    assert x_pos == pytest.approx(-x_neg, rel=1e-9)
    # Upright stationary rocket predicts zero motion
    x0, vx0 = _predict_zero_tilt_state(10.0, 0.0, 0.0, 0.0, 12.0, radians(2.15))
    assert x0 == pytest.approx(10.0)
    assert vx0 == pytest.approx(0.0)
