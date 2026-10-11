"""Render English/Korean reports and figures from recorded evidence only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from scripts.validate_evidence import percentile, validate_rows
from scripts.validate_regression_evidence import verify


def paired_fuel(rows):
    selected = [row for row in rows if row["split"] == "sensor-holdout"]
    lookup = {(row["case_id"], row["variant"]): row for row in selected}
    pairs = [
        (row, lookup[row["case_id"], "raw-attitude"])
        for row in selected
        if row["variant"] == "full"
    ]
    both = [
        (full, raw)
        for full, raw in pairs
        if full["requirements_passed"] and raw["requirements_passed"]
    ]
    full = [300 - row["rcs_remaining_kg"] for row, _ in both]
    raw = [300 - row["rcs_remaining_kg"] for _, row in both]
    full_median, raw_median = percentile(full, 0.5), percentile(raw, 0.5)
    reduction = None if not raw_median else 100 * (raw_median - full_median) / raw_median
    return {
        "all_pairs": len(pairs),
        "both_pass_pairs": len(both),
        "filtered_used_kg": full,
        "raw_used_kg": raw,
        "filtered_median_kg": full_median,
        "raw_median_kg": raw_median,
        "median_consumption_reduction_pct": reduction,
        "lower_consumption_pairs": sum(a < b for a, b in zip(full, raw, strict=True)),
        "higher_consumption_pairs": sum(a > b for a, b in zip(full, raw, strict=True)),
        "median_paired_saving_kg": percentile([b - a for a, b in zip(full, raw, strict=True)], 0.5),
        "comparison_scope": (
            "Identical inputs and seeds; fuel comparison only when both complete acceptance"
        ),
    }


def chart(summary, regression, fuel, path):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    groups = [
        (
            "Known-condition\nregression",
            regression["requirements_passed"],
            regression["total"],
            None,
        ),
        (
            "New coupled\nconditions",
            summary["holdout/full"]["passed"],
            summary["holdout/full"]["total"],
            summary["holdout/full"]["wilson_95"],
        ),
        (
            "Added angular\nsensor errors",
            summary["sensor-holdout/full"]["passed"],
            summary["sensor-holdout/full"]["total"],
            summary["sensor-holdout/full"]["wilson_95"],
        ),
    ]
    for index, (_label, passed, total, interval) in enumerate(groups):
        rate = 100 * passed / total
        axes[0].bar(index, rate, color=("#667085", "#287a99", "#21a59b")[index], width=0.58)
        if interval:
            axes[0].errorbar(
                index,
                rate,
                yerr=[
                    [max(0, rate - 100 * interval[0])],
                    [max(0, 100 * interval[1] - rate)],
                ],
                fmt="none",
                color="#101828",
                capsize=5,
            )
        axes[0].text(index, rate + 2, f"{passed}/{total}", ha="center", fontsize=11)
    axes[0].set(
        xticks=range(3),
        xticklabels=[g[0] for g in groups],
        ylim=(0, 112),
        ylabel="Joint acceptance (%)",
        title="Separate evaluation populations",
    )
    axes[0].grid(axis="y", alpha=0.15)
    if fuel["both_pass_pairs"]:
        boxes = axes[1].boxplot(
            [fuel["raw_used_kg"], fuel["filtered_used_kg"]],
            tick_labels=["Raw angular\nmeasurements", "Attitude\nestimator"],
            patch_artist=True,
        )
        for box, color in zip(boxes["boxes"], ["#e4a45b", "#21a59b"], strict=True):
            box.set_facecolor(color)
        axes[1].set(ylabel="RCS propellant used (kg)", title="Paired sensor experiment")
        axes[1].text(
            0.5,
            0.97,
            f"{fuel['both_pass_pairs']} pairs passing with both methods",
            transform=axes[1].transAxes,
            ha="center",
            va="top",
            fontsize=9,
        )
        axes[1].grid(axis="y", alpha=0.15)
    fig.suptitle("DM100 recorded validation", fontsize=15, fontweight="bold")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def render(evidence, regression_dir, output):
    manifest = json.loads((evidence / "manifest.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (evidence / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    summary = validate_rows(manifest, rows)
    regression = verify(
        regression_dir, Path(__file__).resolve().parents[1] / "configs/apr-regression-v1.json"
    )
    fuel = paired_fuel(rows)
    data = json.loads((evidence / "analysis.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    (output / "assets").mkdir(exist_ok=True)
    (output / "results").mkdir(exist_ok=True)
    chart(summary, regression, fuel, output / "assets" / "validation.png")
    model_summary = {
        "regression": regression,
        "new_conditions": summary["holdout/full"],
        "extended_sensors": summary["sensor-holdout/full"],
        "raw_extended_sensors": summary["sensor-holdout/raw-attitude"],
        "paired_fuel": fuel,
    }
    (output / "results" / "performance.json").write_text(
        json.dumps(model_summary, indent=2) + "\n", encoding="utf-8"
    )
    for language in ("en", "ko"):
        korean = language == "ko"
        header = "# 검증 결과" if korean else "# Recorded validation results"
        link = "[English](results.md)" if korean else "[한국어](results.ko.md)"
        lines = [
            header,
            "",
            link
            + (
                " | [검증 설계](validation.ko.md) | [공개 기록](results/validation-study-v1/)"
                if korean
                else " | [Protocol](validation.md) | [Evidence](results/validation-study-v1/)"
            ),
            "",
            "![Recorded validation](assets/validation.png)",
            "",
            "| 평가 집단 | 통과 / 실행 |" if korean else "| Population | Passed / attempted |",
            "|---|---:|",
            f"| {'기존 조건의 회귀 검증' if korean else 'Known-condition regression'} "
            f"| {regression['requirements_passed']} / 286 |",
            f"| {'새로운 복합 조건' if korean else 'New coupled conditions'} "
            f"| {summary['holdout/full']['passed']} / 96 |",
            f"| {'확장 센서 조건과 자세 필터' if korean else 'Extended sensors with estimator'} "
            f"| {summary['sensor-holdout/full']['passed']} / 48 |",
            f"| {'동일 센서 조건과 기존 계측 처리' if korean else 'Same sensors, raw processing'} "
            f"| {summary['sensor-holdout/raw-attitude']['passed']} / 48 |",
            "",
            (
                "각 집단은 별도 분모입니다. 기존 286개는 독립 평가가 아닙니다. 새 평가의 "
                "성공률도 선언한 합성 조건에 한정합니다. 서로 다른 집단의 최대 오차로 "
                "성능 개선율을 계산하지 않습니다."
            )
            if korean
            else (
                "Each population has its own denominator. The 286 known conditions are not "
                "independent evaluation. New evaluation success rates apply only to the declared "
                "synthetic sampling recipe. Worst errors across different populations do not "
                "measure improvement."
            ),
            "",
        ]
        for key, label in (
            ("holdout/full", "새로운 조건" if korean else "New conditions"),
            ("sensor-holdout/full", "확장 센서 조건" if korean else "Extended sensors"),
        ):
            interval = summary[key]["wilson_95"]
            lines.append(
                f"{label}: Wilson 95% {'구간' if korean else 'interval'} "
                f"{100 * interval[0]:.2f}% {'~' if korean else 'to'} "
                f"{100 * interval[1]:.2f}%."
            )
        lines.extend(
            [
                "",
                "## 착륙과 추진제 지표" if korean else "## Touchdown and propellant measurements",
                "",
                "| 항목 | 기존 286개 | 새로운 96개 | 센서 확장 48개 |"
                if korean
                else "| Metric | Known 286 | New 96 | Extended sensors 48 |",
                "|---|---:|---:|---:|",
            ]
        )
        for metric, label, unit, minimum in (
            ("position_error_m", "Position error", "m", False),
            ("vertical_foot_speed_m_s", "Vertical foot speed", "m/s", False),
            ("horizontal_foot_speed_m_s", "Horizontal foot speed", "m/s", False),
            ("peak_dynamic_pressure_pa", "Peak dynamic pressure", "Pa", False),
            ("main_remaining_kg", "Main propellant remaining", "kg", True),
            ("rcs_remaining_kg", "RCS propellant remaining", "kg", True),
        ):
            aggregate = "min" if minimum else "max"
            old = regression["extrema"][metric][aggregate]
            new = summary["holdout/full"]["metrics"][metric][aggregate]
            sensor = summary["sensor-holdout/full"]["metrics"][metric][aggregate]
            label_ko = {
                "Position error": "최대 위치 오차",
                "Vertical foot speed": "최대 수직 접지 속도",
                "Horizontal foot speed": "최대 수평 접지 속도",
                "Peak dynamic pressure": "최대 동압",
                "Main propellant remaining": "주 추진제 최소 잔량",
                "RCS propellant remaining": "RCS 추진제 최소 잔량",
            }
            lines.append(
                f"| {label_ko[label] if korean else label + ' (' + aggregate + ')'} | "
                f"{old:.4f} {unit} | "
                f"{new:.4f} {unit} | {sensor:.4f} {unit} |"
            )
        lines.extend(
            [
                "",
                (
                    "실패 비행도 측정 가능한 지표에 포함합니다. 접지 지표의 측정 수는 "
                    "공개 summary.json에 별도로 표시합니다."
                )
                if korean
                else (
                    "Failed flights remain in metric populations when the measurement exists. "
                    "Contact-metric counts are recorded separately in summary.json."
                ),
                "",
                "## 같은 조건에서의 센서 처리 비교"
                if korean
                else "## Paired sensor processing comparison",
                "",
            ]
        )
        if fuel["both_pass_pairs"]:
            lines.append(
                (
                    f"두 방식 모두 성공한 동일 조건 {fuel['both_pass_pairs']}쌍에서 RCS 소모량 "
                    f"중앙값은 기존 계측 처리 {fuel['raw_median_kg']:.3f} kg에서 자세 추정기 "
                    f"{fuel['filtered_median_kg']:.3f} kg으로 바뀌었습니다. 감소율은 "
                    f"{fuel['median_consumption_reduction_pct']:.2f}%입니다."
                )
                if korean
                else (
                    f"Across {fuel['both_pass_pairs']} identical-condition pairs passing with both "
                    f"methods, median RCS consumption changed from {fuel['raw_median_kg']:.3f} kg "
                    f"with raw angular measurements to {fuel['filtered_median_kg']:.3f} kg with "
                    f"the attitude estimator, a "
                    f"{fuel['median_consumption_reduction_pct']:.2f}% reduction."
                )
            )
            lines.append(
                (
                    f"개별 쌍에서 소모량이 감소한 조건은 {fuel['lower_consumption_pairs']}개, "
                    f"증가한 조건은 {fuel['higher_consumption_pairs']}개이며, "
                    f"쌍별 절감량 중앙값은 {fuel['median_paired_saving_kg']:.3f} kg입니다."
                )
                if korean
                else (
                    f"Consumption decreased in {fuel['lower_consumption_pairs']} pairs and "
                    f"increased in {fuel['higher_consumption_pairs']} pairs. Median per-pair "
                    f"saving was {fuel['median_paired_saving_kg']:.3f} kg."
                )
            )
        lines.extend(
            [
                "",
                (
                    "성공 여부는 전체 48쌍에서 비교하고, 추진제 효율은 두 방식 모두 기준을 "
                    "충족한 쌍에서 비교합니다. 센서 오차는 모델 가정이며 "
                    "실제 기체의 절감률이 아닙니다."
                )
                if korean
                else (
                    "Success uses all 48 pairs. Fuel efficiency uses only pairs meeting acceptance "
                    "with both methods. Sensor errors are assumed; this is not measured vehicle "
                    "fuel savings."
                ),
                "",
            ]
        )
        lines += [
            "| 같은 48개 조건의 지표 | 기존 계측 처리 | 자세 추정기 |"
            if korean
            else "| Metric on the same 48 conditions | Raw processing | Attitude estimator |",
            "|---|---:|---:|",
        ]
        for metric, aggregate, english, korean_label, unit in (
            (
                "position_error_m",
                "p95",
                "Position error, 95th percentile",
                "위치 오차 95백분위",
                "m",
            ),
            (
                "vertical_foot_speed_m_s",
                "max",
                "Maximum vertical foot speed",
                "최대 수직 접지 속도",
                "m/s",
            ),
            (
                "main_remaining_kg",
                "median",
                "Median main propellant remaining",
                "주 추진제 잔량 중앙값",
                "kg",
            ),
        ):
            raw = summary["sensor-holdout/raw-attitude"]["metrics"][metric][aggregate]
            filtered = summary["sensor-holdout/full"]["metrics"][metric][aggregate]
            lines.append(
                f"| {korean_label if korean else english} | "
                f"{raw:.4f} {unit} | {filtered:.4f} {unit} |"
            )
        lines += [
            "",
            (
                "자세 추정기는 이 조건에서 RCS 효율을 개선했지만 모든 접지 또는 추진제 "
                "지표를 개선하지는 않았습니다. 전체 성공률을 유지하면서 발생한 "
                "지표별 변화로 해석합니다."
            )
            if korean
            else (
                "The estimator improved RCS efficiency here, but not every touchdown or "
                "propellant metric. These are metric-specific tradeoffs with unchanged "
                "joint acceptance in the observed conditions."
            ),
            "",
            "## 개별 기능 제거 실험" if korean else "## Feature ablations",
            "",
            "| 기능 설정 | 통과 / 12개 |" if korean else "| Variant | Passed / 12 |",
            "|---|---:|",
        ]
        for variant in (
            "full",
            "raw-attitude",
            "no-translation-bias",
            "no-torque-bias",
            "no-replanning",
        ):
            lines.append(f"| {variant} | {summary['ablation/' + variant]['passed']} / 12 |")
        consistent = sum(row["acceptance_consistent"] for row in data["convergence"])
        within = sum(
            all(v["within_declared_tolerances"] for v in row["differences"].values())
            for row in data["convergence"]
        )
        lines.extend(
            [
                "",
                "## 수치 계산과 민감도" if korean else "## Numerical and sensitivity checks",
                "",
                (
                    f"적분 간격 세 종류를 비교한 네 조건 중 성공 판정이 일치한 조건은 "
                    f"{consistent}/4개이며, 선언한 수치 허용오차를 모두 만족한 "
                    f"조건은 {within}/4개입니다."
                )
                if korean
                else (
                    f"Across four cases at three integration intervals, acceptance agreed in "
                    f"{consistent}/4 cases and all predeclared numerical tolerances held "
                    f"in {within}/4 cases."
                ),
                "",
                (
                    "가장 작은 적분 간격은 정확해가 아닌 비교 기준입니다. 개별 차이와 민감도 "
                    "분석의 결과는 공개 analysis.json에 남겼습니다."
                )
                if korean
                else (
                    "The finest integration interval is a comparison reference, not an exact "
                    "solution. "
                    "Per-case differences and one-factor sensitivity results are in analysis.json."
                ),
                "",
                "## 기록 검증" if korean else "## Check the recorded evidence",
                "",
                "```bash",
                "python scripts/validate_evidence.py docs/results/validation-study-v1",
                "python -m scripts.validate_regression_evidence docs/results/regression-v1.1",
                "```",
                "",
                (
                    "두 검증기는 Python 표준 라이브러리만 사용하며 비행을 실행하지 않습니다. "
                    "저장된 결과의 일관성과 요약 수치를 확인합니다."
                )
                if korean
                else (
                    "Both validators use only the Python standard library and do not run flights. "
                    "They check recorded evidence consistency and recompute aggregate measurements."
                ),
                "",
            ]
        )
        insertion = lines.index("## 기록 검증" if korean else "## Check the recorded evidence")
        numerical = [
            "| 비교 지표 | 가장 작은 간격과의 최대 차이 | 사전 허용오차 |"
            if korean
            else "| Comparison metric | Maximum difference from finest step | Declared tolerance |",
            "|---|---:|---:|",
        ]
        for metric, english, korean_label, unit in (
            ("position_error_m", "Contact position", "접지 위치", "m"),
            ("vertical_foot_speed_m_s", "Vertical foot speed", "수직 접지 속도", "m/s"),
            ("horizontal_foot_speed_m_s", "Horizontal foot speed", "수평 접지 속도", "m/s"),
            ("flight_time_s", "Flight time", "비행 시간", "s"),
            ("main_remaining_kg", "Main propellant remaining", "주 추진제 잔량", "kg"),
            ("rcs_remaining_kg", "RCS propellant remaining", "RCS 추진제 잔량", "kg"),
            ("peak_dynamic_pressure_pa", "Peak dynamic pressure", "최대 동압", "Pa"),
        ):
            differences = [
                comparison["absolute_difference_from_finest"][metric]
                for case in data["convergence"]
                for comparison in case["differences"].values()
            ]
            valid = [value for value in differences if value is not None]
            maximum = "N/A" if not valid else f"{max(valid):.6g} {unit}"
            numerical.append(
                f"| {korean_label if korean else english} | {maximum} | "
                f"{manifest['convergence_tolerances'][metric]:.6g} {unit} |"
            )
        reference = json.loads((output / "results" / "numerical-reference.json").read_text())
        worst = max(
            value["normalized_error_linf"]
            for check in reference["checks"]
            for value in check["errors"].values()
        )
        numerical += [
            "",
            (
                f"매끄러운 세 구간의 RK4와 DOP853 비교는 "
                f"{sum(check['passed'] for check in reference['checks'])}/3개를 통과했습니다. "
                f"최대 정규화 오차는 {worst:.6g}이며, 선언한 허용오차는 1e-7입니다. "
                "이 값은 실물 기체의 정확도가 아닌 수치 적분기 간 차이입니다."
            )
            if korean
            else (
                f"RK4/DOP853 comparison passed "
                f"{sum(check['passed'] for check in reference['checks'])}/3 smooth segments. "
                f"Maximum normalized error was {worst:.6g}, against the declared 1e-7 tolerance. "
                "This measures solver disagreement, not real-vehicle accuracy."
            ),
            "",
        ]
        sensitivity = data["sensitivity"]
        numerical += [
            "## 개별 매개변수 민감도" if korean else "## One-factor parameter sensitivity",
            "",
            (
                f"기준 조건과 추력 ±3%, 항력 ±15%, 관성 ±3%, 초기 질량 ±1% 조건은 "
                f"{sum(row['accepted'] for row in sensitivity)}/9개를 통과했습니다. "
                "아래 값은 다른 입력을 유지한 상태에서 기준 조건에 대한 차이입니다. "
                "동시에 변하는 매개변수의 모든 조합을 검증한 결과는 아닙니다."
            )
            if korean
            else (
                f"The reference and thrust ±3%, drag ±15%, inertia ±3% and initial mass "
                f"±1% conditions passed in {sum(row['accepted'] for row in sensitivity)}/9 "
                "cases. Differences below use the same other inputs and the reference case. "
                "This does not cover every simultaneous parameter combination."
            ),
            "",
            (
                "| 조건 | 통과 | 위치 오차 차이 (m) | 수직 접지 속도 차이 (m/s) | "
                "주 추진제 잔량 차이 (kg) | RCS 잔량 차이 (kg) |"
            )
            if korean
            else (
                "| Case | Passed | Position error difference (m) | "
                "Vertical foot speed difference (m/s) | Main remaining difference (kg) | "
                "RCS remaining difference (kg) |"
            ),
            "|---|---|---:|---:|---:|---:|",
        ]
        for row in sorted(
            sensitivity, key=lambda item: (item["case_id"] != "calm-reference", item["case_id"])
        ):
            values = row["deltas_from_nominal"]
            formatted = [
                "N/A" if values[metric] is None else f"{values[metric]:+.6g}"
                for metric in (
                    "position_error_m",
                    "vertical_foot_speed_m_s",
                    "main_remaining_kg",
                    "rcs_remaining_kg",
                )
            ]
            numerical.append(
                f"| {row['case_id']} | {row['accepted']} | " + " | ".join(formatted) + " |"
            )
        numerical.append("")
        lines[insertion:insertion] = numerical
        name = "results.ko.md" if korean else "results.md"
        (output / name).write_text("\n".join(lines), encoding="utf-8")
    print(f"Rendered bilingual reports from {len(rows)} study rows and 286 regression rows")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--regression-dir", type=Path, required=True)
    parser.add_argument(
        "--docs-dir", type=Path, default=Path(__file__).resolve().parents[1] / "docs"
    )
    args = parser.parse_args(argv)
    render(args.evidence_dir, args.regression_dir, args.docs_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
