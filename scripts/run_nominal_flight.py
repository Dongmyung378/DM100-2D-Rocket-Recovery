"""CLI for the versioned 3 t APR flight and recovery environment."""

from __future__ import annotations

import argparse
import hashlib
import json
from math import atan2, cos, sin, sqrt
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from powered_landing_guidance.apr_recovery import (  # noqa: E402
    FINS,
    PHYSICS,
    RETURN_METHOD,
    AlignmentDescent,
    AprConditions,
    AprFlight,
    CoastEstimate,
    Design,
    HorizontalTransfer,
    RecoveryDisturbances,
    ReturnGuidanceSettings,
    load_apr_operating,
)
from powered_landing_guidance.recovery_physics import atmosphere  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NominalFlight = AprFlight
# The CLI and package use the same flight implementation.
__all__ = [
    "FINS",
    "PHYSICS",
    "AlignmentDescent",
    "CoastEstimate",
    "Design",
    "HorizontalTransfer",
    "NominalFlight",
    "ReturnGuidanceSettings",
    "fin_authority_table",
]


def plot(result, path):
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    states = np.array([row["state"] for row in result["steps"]])
    times = np.array([row["time_s"] for row in result["steps"]])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].plot(states[:, 0], states[:, 1] / 1000.0)
    axes[0, 0].scatter([0], [0], marker="^", color="black", label="발사점 / 착륙 목표")
    axes[0, 0].set(xlabel="수평 거리 (m)", ylabel="고도 (km)", title="입력 조건 1회 비행")
    axes[0, 0].legend()
    axes[0, 1].plot(times, states[:, 1] / 1000.0)
    axes[0, 1].set(xlabel="시간 (s)", ylabel="고도 (km)")
    axes[1, 0].plot(times, states[:, 6], label="주 추진제")
    axes[1, 0].plot(times, states[:, 7], label="RCS 추진제")
    axes[1, 0].set(xlabel="시간 (s)", ylabel="잔여 추진제 (kg)")
    axes[1, 0].legend()
    axes[1, 1].plot(times, states[:, 2], label="수평 속도")
    axes[1, 1].plot(times, states[:, 3], label="수직 속도")
    axes[1, 1].set(xlabel="시간 (s)", ylabel="속도 (m/s)")
    axes[1, 1].legend()
    for ax in axes.flat:
        ax.grid(alpha=0.25)
    if result["separation"]:
        for ax in (axes[0, 1], axes[1, 0], axes[1, 1]):
            ax.axvline(result["separation"]["time_s"], color="black", linestyle=":", alpha=0.6)
    fig.suptitle("3톤 / 42 kN / APR 강체 회수 - 실제 핫스테이징·공급계통 검증 아님")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def descent_tracking_rms(result):
    samples = [
        row
        for row in result["steps"]
        if row["phase"] in ("coast", "landing")
        and row["state"][3] < 0.0
        and row["state"][1] > 200.0
        and row["dynamic_pressure_pa"] >= 1000.0
    ]
    if not samples:
        return None
    errors = [
        atan2(
            sin(row["state"][4] - np.deg2rad(row["attitude_target_deg"])),
            cos(row["state"][4] - np.deg2rad(row["attitude_target_deg"])),
        )
        for row in samples
    ]
    return float(np.rad2deg(np.sqrt(np.mean(np.square(errors)))))


def fin_authority_table():
    flight = NominalFlight(fin_mode="active")
    flight.upper_attached = False
    y = flight.state.copy()
    y[6:8] = (462.0, 15.0)
    y[12] = 1.0
    y[1] = 2000.0
    density = atmosphere(y[1], PHYSICS.atmosphere)[2]
    rows = []
    for q in (100.0, 1000.0, 5000.0, 20_000.0):
        y[3] = -sqrt(2.0 * q / density)
        _, torque, hinge = flight.fin_loads(y, np.deg2rad(FINS.deflection_limit_deg))
        rows.append(
            dict(
                dynamic_pressure_pa=q,
                first_stage_mass_kg=flight.mass(y),
                fin_lever_m=FINS.station_m - flight.properties(y).cg_m,
                max_fin_torque_nm=abs(torque),
                equivalent_rcs_capacity_fraction=abs(torque) / flight.design.rcs_torque_limit_nm,
                angular_acceleration_deg_s2=float(
                    np.rad2deg(
                        abs(torque) / flight.properties(y).inertia_kg_m2,
                    )
                ),
                per_fin_hinge_torque_nm=hinge,
            )
        )
    return rows


def compare_fins():
    # Exactly one full mission per mode; no tuning, retries or discarded failures.
    results = {}
    for mode in ("stowed", "active"):
        print(f"핀 비교: {mode} 1회 시작", flush=True)
        results[mode] = NominalFlight(fin_mode=mode).run()
        print(f"핀 비교: {mode} -> {results[mode]['outcome']}", flush=True)
    baseline, active = results["stowed"], results["active"]
    saved = baseline["rcs_propellant_used_kg"] - active["rcs_propellant_used_kg"]
    return dict(
        description="520 kg 동일 기체의 핀 비전개/활성 무풍 개발 비교. 독립 성공률 검증 아님.",
        seed=378,
        simulation_count=2,
        return_method=RETURN_METHOD,
        experiment_contract=dict(
            equal_initial_state=True,
            equal_dry_mass=True,
            equal_propellant=True,
            equal_environment=True,
            unchanged_return_pid=True,
            controller_difference="핀 토크 분배와 그에 따른 짐벌/RCS 잔차 보상만 변경",
            fixed_mass_distribution_during_deployment=True,
            weather_validation=False,
            tuning_sweep=False,
        ),
        fin_authority_at_zero_alpha=fin_authority_table(),
        comparison=dict(
            rcs_saved_kg=saved,
            rcs_reduction_percent=100.0 * saved / baseline["rcs_propellant_used_kg"]
            if baseline["rcs_propellant_used_kg"] > 0.0
            else None,
            main_propellant_saved_kg=(
                baseline["main_propellant_used_kg"] - active["main_propellant_used_kg"]
            ),
            descent_tracking_rms_deg=dict(
                stowed=descent_tracking_rms(baseline),
                active=descent_tracking_rms(active),
            ),
            both_successful=(
                baseline["simulated_acceptance_passed"] and active["simulated_acceptance_passed"]
            ),
            note="RMS는 동압 1 kPa 이상·고도 200 m 초과 하강의 자세 참조 추종 오차입니다. "
            "종료 시점·궤적이 다르면 소비량과 RMS 차이만으로 효율 우위를 입증하지 않습니다.",
        ),
        runs=results,
    )


def plot_comparison(result, path):
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for mode, flight in result["runs"].items():
        states = np.asarray([row["state"] for row in flight["steps"]])
        times = np.asarray([row["time_s"] for row in flight["steps"]])
        label = "핀 활성" if mode == "active" else "핀 비전개"
        axes[0, 0].plot(states[:, 0], states[:, 1] / 1000.0, label=label)
        axes[0, 1].plot(times, states[:, 1] / 1000.0, label=label)
        axes[0, 2].plot(times, 15.0 - states[:, 7], label=label)
        if flight["separation"]:
            mask = times >= flight["separation"]["time_s"]
            axes[1, 0].plot(times[mask], states[mask, 2], label=label)
            axes[1, 1].plot(times[mask], np.rad2deg(states[mask, 4]), label=label)
            axes[1, 2].plot(times[mask], np.rad2deg(states[mask, 11]), label=label)
    for ax, title, x, y in zip(
        axes.flat,
        ("귀환 경로", "고도", "RCS 소비", "귀환 수평속도", "귀환 기체 자세", "실제 핀 조종각"),
        ("수평거리 (m)", "시간 (s)", "시간 (s)", "시간 (s)", "시간 (s)", "시간 (s)"),
        ("고도 (km)", "고도 (km)", "소비량 (kg)", "속도 (m/s)", "각도 (deg)", "각도 (deg)"),
        strict=True,
    ):
        ax.set(title=title, xlabel=x, ylabel=y)
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("Aero-PID Return - 핀 추가만의 효과 비교 / 개발용 근사 모델")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--conditions", type=Path, help="검토를 통과한 APR 환경·정상 편차 YAML")
    parser.add_argument(
        "--separation-record", type=Path, help="새 APR 결과의 10 km 분리 상태부터 시작"
    )
    parser.add_argument(
        "--check-inputs", action="store_true", help="입력만 확인하며 비행은 실행하지 않음"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fin-mode", choices=("stowed", "active"), default="active")
    mode.add_argument("--compare-fins", action="store_true")
    parser.add_argument("--guidance-stage", choices=("approach", "sync", "joint"), default="joint")
    args = parser.parse_args(argv)
    if args.compare_fins and (
        args.guidance_stage != "joint" or args.conditions or args.separation_record
    ):
        parser.error("fin comparison uses the default joint guidance only")
    try:
        conditions, disturbances = (
            load_apr_operating(args.conditions)
            if args.conditions
            else (AprConditions(), RecoveryDisturbances())
        )
        options = dict(
            fin_mode=args.fin_mode,
            guidance_stage=args.guidance_stage,
            conditions=conditions,
            disturbances=disturbances,
        )
        if args.separation_record:
            record = json.loads(args.separation_record.read_text(encoding="utf-8"))
            expected_design = vars(Design())
            if record.get("seed") != 378 or record.get("design") != expected_design:
                raise ValueError("분리 기록은 기준 시드 378·현재 3톤 기체 제원이어야 합니다.")
            separation = record["separation"]
            if not separation or not all(separation["checks_before_separation"].values()):
                raise ValueError("원본 기록의 실제 분리 조건을 통과해야 합니다.")
            flight = AprFlight.from_separation(
                separation["state"],
                time_s=separation["time_s"],
                after_max_q=separation["after_observed_max_q"],
                gate_altitude_m=separation["gate_altitude_before_cg_change_m"],
                engine_throttle=separation["engine_throttle"],
                **options,
            )
        else:
            flight = AprFlight(**options)
    except (ValueError, TypeError, KeyError, OSError, AttributeError) as error:
        parser.error(f"실행 전 입력 확인 실패: {error}")
    if args.check_inputs:
        print("입력 허용: 기준 시드 378 / 정상 운용 범위. 착륙 가능성 검증은 아닙니다.")
        return 0
    if args.output_dir is None:
        parser.error("--output-dir가 필요합니다.")
    if args.output_dir.exists():
        parser.error("output directory already exists; preserve the first-run evidence")
    args.output_dir.mkdir(parents=True)
    result = compare_fins() if args.compare_fins else flight.run()
    result["source_sha256"] = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            Path(__file__),
            ROOT / "src/powered_landing_guidance/__init__.py",
            ROOT / "src/powered_landing_guidance/apr_recovery.py",
            ROOT / "src/powered_landing_guidance/recovery_physics.py",
            ROOT / "src/powered_landing_guidance/integrators.py",
        )
    }
    for name, path in (
        ("conditions", args.conditions),
        ("separation_record", args.separation_record),
    ):
        if path:
            result[f"{name}_source"] = str(path.resolve())
            result[f"{name}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    # Cast numpy scalars, but keep non-finite values invalid rather than silently accepting them.
    encoded = json.dumps(
        result, ensure_ascii=False, indent=2, default=lambda value: value.item(), allow_nan=False
    )
    (args.output_dir / "result.json").write_text(encoded + "\n", encoding="utf-8")
    if args.compare_fins:
        plot_comparison(result, args.output_dir / "trajectory.png")
    else:
        plot(result, args.output_dir / "trajectory.png")
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in ("steps", "physics", "runs")
            },
            ensure_ascii=False,
            indent=2,
            default=lambda value: value.item(),
            allow_nan=False,
        )
    )
    accepted = (
        result["comparison"]["both_successful"]
        if args.compare_fins
        else result["simulated_acceptance_passed"]
    )
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
