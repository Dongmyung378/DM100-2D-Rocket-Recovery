"""Run and record the 100 t planar first-stage mission."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from powered_landing_guidance.apr_recovery import (  # noqa: E402
    PHYSICS,
    SYSTEMS_MODEL_VERSION,
    AprConditions,
    AprFlight,
    Design,
    RecoveryDisturbances,
    load_apr_operating,
)
from powered_landing_guidance.recovery_physics import atmosphere  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def environment_samples(flight: AprFlight) -> list[dict]:
    rows = []
    for height in (0, 1000, 5000, 10000, 20000, 37000, 50000, 80000, 100000):
        temperature, pressure, density = atmosphere(height, flight.physics.atmosphere)
        rows.append(
            dict(
                altitude_m=height,
                temperature_k=temperature,
                pressure_pa=pressure,
                density_kg_m3=density,
                reference_density_kg_m3=atmosphere(height, PHYSICS.atmosphere)[2],
                mean_east_wind_m_s=flight.conditions.mean_wind(height),
                forecast_east_wind_m_s=flight.conditions.mean_wind(height, forecast=True),
                vertical_wind_m_s=flight.conditions.vertical_wind(height),
            )
        )
    return rows


def plot(result: dict, path: Path) -> None:
    """Plot saved flight states; plotting never reruns the controller."""
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    states = np.asarray([row["state"] for row in result["steps"]])
    times = np.asarray([row["time_s"] for row in result["steps"]])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].plot(states[:, 0] / 1000, states[:, 1] / 1000)
    axes[0, 0].scatter([0], [0], marker="^", color="black", label="발사점 / 착륙 목표")
    axes[0, 0].set(xlabel="수평 거리 (km)", ylabel="고도 (km)")
    axes[0, 0].legend()
    axes[0, 1].plot(times, states[:, 1] / 1000)
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
    if result.get("separation"):
        for ax in (axes[0, 1], axes[1, 0], axes[1, 1]):
            ax.axvline(result["separation"]["time_s"], color="black", linestyle=":", alpha=0.6)
    mass_t = result["design"]["launch_mass_kg"] / 1000
    thrust_kn = result["design"]["sea_level_thrust_n"] / 1000
    fig.suptitle(f"{mass_t:g}톤 / {thrust_kn:g} kN / 1단 회수 - 2단 궤도 비행 미검증")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--conditions", type=Path, help="환경·정상 기체 편차 YAML")
    parser.add_argument("--check-inputs", action="store_true", help="입력만 확인하고 비행하지 않음")
    parser.add_argument("--progress", action="store_true", help="비행 단계와 진행 상태 표시")
    parser.add_argument("--integration-dt", type=float, default=0.01, help="비행 적분 간격(s)")
    parser.add_argument("--fin-mode", choices=("stowed", "active"), default="active")
    parser.add_argument(
        "--separation-record", type=Path, help="현재 100톤 결과의 분리 상태부터 실행"
    )
    args = parser.parse_args(argv)
    try:
        conditions, disturbances = (
            load_apr_operating(args.conditions)
            if args.conditions
            else (AprConditions(), RecoveryDisturbances())
        )
        options = dict(
            fin_mode=args.fin_mode,
            conditions=conditions,
            disturbances=disturbances,
            integration_dt_s=args.integration_dt,
        )
        if args.separation_record:
            record = json.loads(args.separation_record.read_text(encoding="utf-8"))
            recorded_design = record.get("design", {})
            if record.get("systems_model_version") != SYSTEMS_MODEL_VERSION:
                raise ValueError("분리 재실행은 현재 결합 물리 모델의 계통 상태 기록이 필요합니다.")
            if record.get("seed") != 378 or any(
                recorded_design.get(key) != value for key, value in vars(Design()).items()
            ):
                raise ValueError("분리 기록은 시드 378·현재 100톤 제원이어야 합니다.")
            separation = record["separation"]
            if not separation or not all(separation["checks_before_separation"].values()):
                raise ValueError("원본 기록의 분리 조건을 통과해야 합니다.")
            flight = AprFlight.from_separation(
                separation["state"],
                time_s=separation["time_s"],
                **options,
            )
        else:
            flight = AprFlight(**options)
    except (ValueError, TypeError, KeyError, OSError, AttributeError) as error:
        parser.error(f"실행 전 입력 확인 실패: {error}")
    if args.check_inputs:
        print("입력 허용: 시드 378 / 100톤 제원. 임무 성공 검증은 아닙니다.")
        print(f"실제 초기 질량: {flight.design.launch_mass_kg:.1f} kg")
        print("고도(m)  기온(K)  기압(Pa)  밀도 실제/기준(kg/m³)  동풍 실제/예보(m/s)  수직풍(m/s)")
        for sample in environment_samples(flight):
            print(
                f"{sample['altitude_m']:>7.0f}  {sample['temperature_k']:>7.2f}  "
                f"{sample['pressure_pa']:>9.1f}  "
                f"{sample['density_kg_m3']:>8.5f}/{sample['reference_density_kg_m3']:>8.5f}  "
                f"{sample['mean_east_wind_m_s']:>6.2f}/"
                f"{sample['forecast_east_wind_m_s']:>6.2f}  "
                f"{sample['vertical_wind_m_s']:>6.2f}"
            )
        return 0
    if args.output_dir is None:
        parser.error("--output-dir가 필요합니다.")
    if args.output_dir.exists():
        parser.error("출력 폴더가 이미 있습니다. 기존 실험 결과는 덮어쓰지 않습니다.")
    args.output_dir.mkdir(parents=True)

    def progress(current):
        print(
            f"t={current.time_s:.1f}s {current.phase} "
            f"x={current.state[0]:.1f}m z={current.state[1]:.1f}m "
            f"vx={current.state[2]:.2f} vz={current.state[3]:.2f}m/s "
            f"main={current.state[6]:.1f}kg rcs={current.state[7]:.1f}kg",
            file=sys.stderr,
            flush=True,
        )

    source_paths = (
        Path(__file__),
        ROOT / "src/powered_landing_guidance/__init__.py",
        ROOT / "src/powered_landing_guidance/apr_recovery.py",
        ROOT / "src/powered_landing_guidance/recovery_physics.py",
        ROOT / "src/powered_landing_guidance/integrators.py",
        ROOT / "src/powered_landing_guidance/vehicle_systems.py",
    )

    def source_hashes():
        return {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_paths
        }

    before_hashes = source_hashes()
    result = flight.run(progress=progress if args.progress else None)
    result["environment_samples"] = environment_samples(flight)
    result["source_sha256"] = before_hashes
    result["source_sha256_after_run"] = source_hashes()
    result["source_unchanged_during_run"] = before_hashes == result["source_sha256_after_run"]
    for name, path in (
        ("conditions", args.conditions),
        ("separation_record", args.separation_record),
    ):
        if path:
            result[f"{name}_source"] = str(path.resolve())
            result[f"{name}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    encoded = json.dumps(
        result, ensure_ascii=False, indent=2, default=lambda value: value.item(), allow_nan=False
    )
    (args.output_dir / "result.json").write_text(encoded + "\n", encoding="utf-8")
    plot(result, args.output_dir / "trajectory.png")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key not in ("steps", "physics")},
            ensure_ascii=False,
            indent=2,
            default=lambda value: value.item(),
            allow_nan=False,
        )
    )
    return (
        0 if result["simulated_acceptance_passed"] and result["source_unchanged_during_run"] else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
