"""Run a DM100 2D Reusable RTLS flight (Flip -> Boostback -> Entry -> Glide -> Hoverslam).

Outputs phase-by-phase telemetry, propellant accounting, post-touchdown landing-leg
standing stability, and optionally exports a 50 Hz ENU + unit-quaternion replay file
for 3D MuJoCo / Unreal Engine 5 visualization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from powered_landing_guidance.apr_recovery import (
    AprConditions,
    RecoveryDisturbances,
    load_apr_operating,
)
from powered_landing_guidance.observation_models import ControllerOptions, SensorSettings
from powered_landing_guidance.rtls import PlantConfig, RtlsFlight

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_EXTENDED_SENSORS = SensorSettings(
    attitude_error_deg=0.05,
    gyro_error_deg_s=0.03,
    attitude_bias_deg=0.02,
    gyro_bias_deg_s=-0.015,
    dropout_period_steps=80,
    dropout_length_steps=2,
)


def _load_scenario(
    scenario: str, operating_config: Path | None
) -> tuple[AprConditions, RecoveryDisturbances, SensorSettings]:
    if operating_config is not None:
        cond, dist = load_apr_operating(operating_config)
        return cond, dist, SensorSettings()
    if scenario == "calm":
        return AprConditions(), RecoveryDisturbances(), SensorSettings()
    cond, dist = load_apr_operating(ROOT / "configs" / "apr-operating-v1.yaml")
    if scenario == "example-weather":
        return cond, dist, SensorSettings()
    return cond, dist, DEFAULT_EXTENDED_SENSORS


def _format_report(result: dict, scenario_label: str) -> str:
    lines = [
        f"=== DM100 Reusable RTLS Flight Report ({scenario_label}) ===",
        (
            f"Outcome: {result['outcome']} | Passed: {result['requirements_passed']} | "
            f"Failed: {result['failed_criteria']} | Wall time: {result['wall_seconds']:.2f} s"
        ),
        (
            f"Flight time: {result['flight_time_s']:.2f} s "
            f"(touchdown @ {result['touchdown_time_s']:.2f} s) | "
            f"Restarts used: {result['restarts_used']}/3 | "
            f"Replanning: {result['replanning_count']} | "
            f"Convex LP feasible: {result['convex_reference_feasible']}"
        ),
        (
            f"Propellant remaining: Main = {result['main_remaining_kg']:.1f} kg "
            f"(used {result['main_used_kg']:.1f} kg), "
            f"RCS = {result['rcs_remaining_kg']:.1f} kg (used {result['rcs_used_kg']:.1f} kg)"
        ),
    ]
    if result["position_error_m"] is not None:
        lines.append(
            f"Touchdown: |x| = {result['position_error_m']:.3f} m (<=2.0), "
            f"v_vert = {result['vertical_foot_speed_m_s']:.3f} m/s (<=1.5), "
            f"v_horiz = {result['horizontal_foot_speed_m_s']:.3f} m/s (<=0.5), "
            f"tilt = {result['tilt_deg']:.3f} deg (<=3.0), "
            f"rate = {result['contact_rate_deg_s']:.3f} deg/s (<=0.5)"
        )
    lines.append(
        f"Flight peaks: max_q = {result['peak_dynamic_pressure_pa'] / 1000.0:.2f} kPa (<=30), "
        f"max_g_rec = {result['peak_recovery_g']:.2f} g (<=5.0), "
        f"max_rate = {result['peak_angular_rate_deg_s']:.2f} deg/s (<=15), "
        f"max_accel = {result['peak_angular_acceleration_deg_s2']:.2f} deg/s^2 (<=5), "
        f"max_fin = {result['peak_fin_deflection_deg']:.2f} deg, "
        f"max_hinge = {result['peak_fin_hinge_nm']:.0f} N*m (<=5000)"
    )
    pc = result["post_contact"]
    lines.append(
        f"Post-contact (5 s engine-off): standing_stable = {pc['standing_stable']}, "
        f"leg_stroke = {pc['max_leg_stroke_m']:.3f} m, "
        f"leg_force = {pc['max_leg_force_n'] / 1000.0:.1f} kN, "
        f"contact_g = {pc['max_contact_load_g']:.2f} g, "
        f"final_tilt = {pc['final_tilt_deg']:.3f} deg, "
        f"final_speed = {pc['final_speed_m_s']:.4f} m/s"
    )
    lines.append("")
    lines.append(
        f"{'Phase':16s} {'Dur(s)':>7s} {'z_start->end(km)':>17s} "
        f"{'Main(kg)':>9s} {'RCS(kg)':>8s} {'Thr(min-max)':>13s} "
        f"{'Fin(deg)':>9s} {'q_max(kPa)':>10s}"
    )
    lines.append("-" * 96)
    for phase_name, ps in result["phases"].items():
        z_rng = f"{ps['start_z_m'] / 1000.0:6.2f} -> {ps['end_z_m'] / 1000.0:6.2f}"
        thr_str = f"{ps['min_active_throttle']:.2f}-{ps['max_throttle']:.2f}"
        lines.append(
            f"{phase_name:16s} {ps['duration_s']:7.2f} {z_rng:>17s} "
            f"{ps['main_used_kg']:9.1f} {ps['rcs_used_kg']:8.2f} {thr_str:>13s} "
            f"{ps['max_fin_deg']:9.2f} {ps['max_q_pa'] / 1000.0:10.2f}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        choices=("calm", "example-weather", "extended-sensors"),
        default="example-weather",
        help="Built-in RTLS mission scenario",
    )
    parser.add_argument(
        "--operating-config",
        type=Path,
        default=None,
        help="Optional YAML file matching configs/apr-operating-v1.yaml",
    )
    parser.add_argument(
        "--landing-mode",
        choices=("contact_dynamics", "kinematic_touchdown"),
        default="contact_dynamics",
    )
    parser.add_argument("--seed", type=int, default=378)
    parser.add_argument("--sensor-seed", type=int, default=1378)
    parser.add_argument(
        "--export-replay",
        type=Path,
        default=None,
        help="Optional path to export 50 Hz ENU+quaternion JSON replay",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write full flight summary JSON",
    )
    args = parser.parse_args(argv)

    cond, dist, sens = _load_scenario(args.scenario, args.operating_config)
    label = str(args.operating_config) if args.operating_config else args.scenario
    flight = RtlsFlight(
        conditions=cond,
        disturbances=dist,
        sensor_settings=sens,
        controller_options=ControllerOptions(),
        plant_config=PlantConfig(
            landing_mode=args.landing_mode,
            record_replay=args.export_replay is not None,
            seed=args.seed,
            sensor_seed=args.sensor_seed,
        ),
    )
    result = flight.run()
    print(_format_report(result, label))

    if args.export_replay is not None:
        out_path = flight.export_replay(
            args.export_replay, metadata={"scenario": label, "seed": args.seed}
        )
        print(f"\nExported {result['replay_frame_count']} replay frames (50 Hz) -> {out_path}")
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Wrote flight summary JSON -> {args.json_out}")

    return 0 if result["requirements_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
