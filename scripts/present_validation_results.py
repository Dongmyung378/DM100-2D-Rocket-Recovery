"""Render current APR 100 t evidence without running or importing the simulator."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import font_manager  # noqa: E402
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "artifacts/apr100t/rcs300-v1"
OUTPUT = ROOT / "artifacts/apr100t/presentation-current"
CURRENT_INVENTORY = {"main_propellant_kg": 70_200.0, "rcs_propellant_kg": 300.0}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def semantic_sha(value: dict) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def extract_metrics(record: dict, cohort: str, inventory: dict | None = None) -> dict:
    inventory = CURRENT_INVENTORY if inventory is None else inventory
    accepted = all(
        record[key] is True for key in ("mission_success", "flight_safety", "landing_success")
    )
    require(record["accepted"] is accepted, "Inconsistent combined acceptance")
    contact = record.get("contact")
    row = {
        "id": record["id"],
        "cohort": cohort,
        "group": record["group"],
        "accepted": accepted,
        "has_contact": contact is not None,
        "position_error_m": None,
        "foot_vertical_speed_m_s": None,
        "foot_horizontal_speed_m_s": None,
        "tilt_deg": None,
        "contact_angular_rate_deg_s": None,
        "failed_criteria": record.get("failed_criteria", []),
    }
    state = record["final_state"] if contact is None else contact["state"]
    require(
        len(state) >= 8 and all(math.isfinite(v) for v in state), "Nonfinite or invalid final state"
    )
    if contact is not None:
        feet = contact["foot_velocities_m_s"]
        require(
            len(feet) == 2 and all(len(foot) == 2 for foot in feet), "Invalid foot velocity shape"
        )
        require(all(math.isfinite(v) for foot in feet for v in foot), "Nonfinite foot velocity")
        require(
            not record["landing_success"] or all(contact["criteria"].values()),
            "Inconsistent contact acceptance",
        )
        row.update(
            position_error_m=abs(state[0]),
            foot_vertical_speed_m_s=max(abs(foot[1]) for foot in feet),
            foot_horizontal_speed_m_s=max(abs(foot[0]) for foot in feet),
            tilt_deg=abs(math.degrees(math.atan2(math.sin(state[4]), math.cos(state[4])))),
            contact_angular_rate_deg_s=abs(math.degrees(state[5])),
        )
    else:
        require(not record["landing_success"], "Successful landing without contact")
    if record["landing_success"]:
        require(
            all(
                row[key] <= limit
                for key, limit in (
                    ("position_error_m", 2.0),
                    ("foot_vertical_speed_m_s", 1.5),
                    ("foot_horizontal_speed_m_s", 0.5),
                    ("tilt_deg", 3.0),
                    ("contact_angular_rate_deg_s", 0.5),
                )
            ),
            "Successful contact exceeds numeric landing limits",
        )
    for index, name in ((6, "main"), (7, "rcs")):
        remaining, used = (
            record[f"remaining_{name}_propellant_kg"],
            record[f"{name}_propellant_used_kg"],
        )
        require(math.isfinite(remaining) and math.isfinite(used), "Nonfinite propellant")
        require(
            math.isclose(state[index], remaining, rel_tol=0, abs_tol=1e-6),
            "Contact inventory mismatch",
        )
        require(
            math.isclose(
                remaining + used, inventory[f"{name}_propellant_kg"], rel_tol=0, abs_tol=1e-6
            ),
            "Propellant mass balance mismatch",
        )
        row[f"remaining_{name}_propellant_kg"], row[f"{name}_propellant_used_kg"] = remaining, used
    for key in (
        "flight_time_s",
        "max_dynamic_pressure_pa",
        "max_angular_acceleration_deg_s2",
        "max_flight_load_g",
    ):
        require(math.isfinite(record[key]), f"Nonfinite metric: {key}")
        row[key] = record[key]
    exposure = record["gust_exposure"]
    row["gust_configured"] = abs(exposure["configured_amplitude_m_s"]) > 0
    row["gust_encountered"] = exposure["integration_endpoints"]["gust_encounter_observed"]
    return row


def load_evidence(
    campaign: Path, nominal_path: Path | None = None
) -> tuple[dict, list[dict], dict]:
    protocol, summary = (
        read_json(campaign / name) for name in ("protocol.json", "all-summary.json")
    )
    lines = [
        json.loads(line)
        for line in (campaign / "all-results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    header, records = lines[0], lines[1:]
    digest = sha256(campaign / "protocol.json")
    require(
        header["protocol_sha256"] == summary["protocol_sha256"] == digest, "Protocol hash mismatch"
    )
    require(
        header.get("fresh_independent") is False
        and summary.get("previous_holdout_is_fresh_independent", summary.get("fresh_independent"))
        is False,
        "Observed regression cannot be called fresh independent validation",
    )
    require(
        summary["source_unchanged"] is True and summary["protocol_unchanged"] is True,
        "Evidence changed during execution",
    )
    sources = protocol["source_hashes"]
    require(
        summary["source_hashes_before"]
        == summary["source_hashes_after"]
        == header["source_hashes"]
        == sources,
        "Execution source hash mismatch",
    )
    require(
        all(protocol["design"][key] == value for key, value in CURRENT_INVENTORY.items()),
        "Not the current propellant inventory",
    )
    jobs = protocol["jobs"]
    ids = [job["job_id"] for job in jobs]
    require(len(set(ids)) == len(ids), "Duplicate declared job")
    require(
        header["selected_job_ids"] == summary["ordered_job_ids"] == ids,
        "Declared job order mismatch",
    )
    by_id = {record["job_id"]: record for record in records}
    require(
        len(by_id) == len(records) == len(ids) and set(by_id) == set(ids),
        "Missing, extra or duplicate result",
    )
    require(protocol["total"] == summary["total"] == len(ids), "Incorrect total denominator")
    require(
        dict(Counter(job["split"] for job in jobs)) == protocol["counts"],
        "Incorrect declared split counts",
    )
    rows = []
    for job in jobs:
        record = by_id[job["job_id"]]
        require(record["split"] == job["split"], "Result split mismatch")
        if record.get("record_type") == "execution_error":
            require(record["requirements_passed"] is False, "Execution error counted as success")
            rows.append(
                {
                    "job_id": job["job_id"],
                    "id": job["case"]["id"],
                    "cohort": job["split"],
                    "group": job["case"]["group"],
                    "accepted": False,
                    "requirements_passed": False,
                    "execution_error": record["message"],
                }
            )
            continue
        for key in (
            "input_sha256",
            "integration_dt_s",
            "observation_seed_sequence",
            "required_exposure_phase",
        ):
            require(record[key] == job[key], f"Result binding mismatch: {key}")
        require(
            semantic_sha(job["case"]) == job["input_sha256"] == record["result"]["input_sha256"],
            "Input hash mismatch",
        )
        require(record["result"]["input"] == job["case"], "Recorded input mismatch")
        require(record["source_hashes_after"] == sources, "Worker source hash mismatch")
        metrics = extract_metrics(record["result"], job["split"], protocol["design"])
        phase = job["required_exposure_phase"]
        peak = (
            record["result"]["gust_exposure"]["integration_endpoints"]
            .get("maximum_by_phase_m_s", {})
            .get(phase, 0.0)
        )
        exposure_passed = phase is None or peak >= job["required_exposure_peak_m_s"]
        require(
            record["required_exposure_passed"] is exposure_passed, "Required gust exposure mismatch"
        )
        require(record["fuel_ledger"]["passed"] is True, "Failed fuel ledger")
        passed = metrics["accepted"] and exposure_passed
        require(record["requirements_passed"] is passed, "Requirement status mismatch")
        metrics.update(
            job_id=job["job_id"],
            requirements_passed=passed,
            integration_dt_s=record["integration_dt_s"],
        )
        rows.append(metrics)
    for split, count in summary["by_split"].items():
        cohort = [row for row in rows if row["cohort"] == split]
        require(
            count
            == {
                "total": len(cohort),
                "accepted": sum(row["accepted"] for row in cohort),
                "requirements_passed": sum(row["requirements_passed"] for row in cohort),
            },
            "Split summary mismatch",
        )
    require(set(summary["by_split"]) == set(protocol["counts"]), "Missing split summary")
    require(
        sum("execution_error" not in row for row in rows) == summary["completed"],
        "Completed count mismatch",
    )
    require(sum(row["accepted"] for row in rows) == summary["accepted"], "Accepted count mismatch")
    require(
        sum(row["requirements_passed"] for row in rows) == summary["requirements_passed"],
        "Requirement count mismatch",
    )
    require(
        [row["job_id"] for row in rows if not row["requirements_passed"]]
        == summary["failed_job_ids"],
        "Failure list mismatch",
    )
    require(
        [row["job_id"] for row in rows if "execution_error" in row] == summary["execution_errors"],
        "Execution error list mismatch",
    )
    nominal_path = campaign / "calm/result.json" if nominal_path is None else nominal_path
    nominal = read_json(nominal_path)
    require(
        nominal["source_unchanged_during_run"] is True
        and nominal["source_sha256"] == nominal["source_sha256_after_run"],
        "Nominal source changed during execution",
    )
    require(
        all(nominal["design"][key] == value for key, value in protocol["design"].items()),
        "Nominal design mismatch",
    )
    return summary, rows, nominal


def extrema(rows: list[dict], key: str, minimum: bool = False) -> dict | None:
    available = [row for row in rows if row.get(key) is not None]
    if not available:
        return None
    selected = (min if minimum else max)(available, key=lambda row: row[key])
    return {"value": selected[key], "job_id": selected["job_id"]}


def text(en: str, ko: str, language: str) -> str:
    return ko if language == "ko" else en


def escaped(value: object) -> str:
    return html.escape(str(value), quote=True)


def table(headers: list[str], rows: list[list]) -> str:
    return (
        "<table><thead><tr>"
        + "".join(f"<th>{escaped(h)}</th>" for h in headers)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>" + "".join(f"<td>{escaped(value)}</td>" for value in row) + "</tr>"
            for row in rows
        )
        + "</tbody></table>"
    )


def set_plot_style(language: str) -> None:
    font = "DejaVu Sans"
    if language == "ko":
        for name in ("Malgun Gothic", "NanumGothic", "Noto Sans CJK KR"):
            if any(item.name == name for item in font_manager.fontManager.ttflist):
                font = name
                break
    plt.rcParams.update(
        {
            "font.family": font,
            "axes.unicode_minus": False,
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "figure.facecolor": "#f7f9fc",
            "axes.facecolor": "white",
        }
    )


def plot_overview(rows: list[dict], path: Path, language: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    groups = sorted({row["group"] for row in rows})
    totals = [sum(row["group"] == group for row in rows) for group in groups]
    passed = [
        sum(row["group"] == group and row["requirements_passed"] for row in rows)
        for group in groups
    ]
    axes[0, 0].barh(groups, totals, color="#df6c6c", label=text("Failed", "실패", language))
    axes[0, 0].barh(groups, passed, color="#188b84", label=text("Passed", "성공", language))
    axes[0, 0].legend()
    axes[0, 0].set_title(text("Results by condition group", "조건군별 결과", language))
    contacts = [row for row in rows if row.get("has_contact")]
    colors = ["#188b84" if row["requirements_passed"] else "#df6c6c" for row in contacts]
    axes[0, 1].scatter(
        [row["foot_horizontal_speed_m_s"] for row in contacts],
        [row["foot_vertical_speed_m_s"] for row in contacts],
        c=colors,
        s=18,
        alpha=0.6,
    )
    axes[0, 1].axhline(1.5, color="#c05a36", linestyle="--")
    axes[0, 1].axvline(0.5, color="#c05a36", linestyle="--")
    axes[0, 1].set(
        xlabel=text("Foot horizontal speed (m/s)", "발 수평속도 (m/s)", language),
        ylabel=text("Foot vertical speed (m/s)", "발 수직속도 (m/s)", language),
    )
    for axis, name in zip(axes[1], ("main", "rcs"), strict=True):
        values = [
            row[f"remaining_{name}_propellant_kg"] for row in rows if "execution_error" not in row
        ]
        axis.hist(values, bins=25, color="#346b9e" if name == "main" else "#188b84")
        axis.set(
            xlabel=text(
                f"Remaining {name.upper()} propellant (kg)",
                f"남은 {name.upper()} 추진제 (kg)",
                language,
            ),
            ylabel=text("Cases", "시험 수", language),
        )
    fig.suptitle(
        text(
            "APR 100 t | 70,200 kg main + 300 kg RCS | finite 2D regression",
            "APR 100톤 | 주 추진제 70,200 kg + RCS 300 kg | 유한 2D 회귀 시험",
            language,
        )
    )
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_telemetry(nominal: dict, path: Path, language: str) -> None:
    steps = nominal["steps"]
    state = np.asarray([step["state"][:8] for step in steps])
    time = [step["time_s"] for step in steps]
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), constrained_layout=True)
    axes[0, 0].plot(state[:, 0] / 1000, state[:, 1] / 1000, color="#188b84")
    axes[0, 0].set(
        xlabel=text("Horizontal position (km)", "수평 위치 (km)", language),
        ylabel=text("COM altitude (km)", "질량중심 고도 (km)", language),
    )
    series = (
        (axes[0, 1], state[:, 1] / 1000, "COM altitude (km)", "질량중심 고도 (km)"),
        (
            axes[2, 0],
            [step["engine_throttle"] for step in steps],
            "Actual engine throttle",
            "실제 엔진 스로틀",
        ),
        (
            axes[2, 1],
            [step["dynamic_pressure_pa"] / 1000 for step in steps],
            "Dynamic pressure (kPa)",
            "동압 (kPa)",
        ),
    )
    for axis, values, en, ko in series:
        axis.plot(time, values, color="#346b9e")
        axis.set(xlabel=text("Time (s)", "시간 (s)", language), ylabel=text(en, ko, language))
    for axis, columns, labels, unit in (
        (axes[1, 0], (2, 3), ("vx", "vz"), "m/s"),
        (axes[1, 1], (6, 7), ("main", "RCS"), "kg"),
    ):
        for column, label in zip(columns, labels, strict=True):
            axis.plot(time, state[:, column], label=label)
        axis.set(xlabel=text("Time (s)", "시간 (s)", language), ylabel=unit)
        axis.legend()
    axes[2, 1].axhline(30, color="#c05a36", linestyle="--")
    fig.suptitle(
        text(
            "Current calm flight | stored telemetry, no new simulation",
            "현재 무풍 비행 | 저장된 기록, 새 시뮬레이션 없음",
            language,
        )
    )
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_report(
    summary: dict, rows: list[dict], path: Path, language: str, suffix: str, paired: bool
) -> None:
    def t(en, ko):
        return text(en, ko, language)

    splits = {
        "development": t("Development regression", "개발 회귀 시험"),
        "previous-holdout": t(
            "Previously observed holdout regression", "이미 관측한 기존 홀드아웃 재시험"
        ),
        "supplemental": t("Supplemental checks", "보충 검사"),
    }
    counts = table(
        [t("Cohort", "시험군"), t("Passed / declared", "성공 / 선언 수")],
        [
            [splits[key], f"{value['requirements_passed']} / {value['total']}"]
            for key, value in summary["by_split"].items()
        ],
    )
    metrics = []
    for key, label, limit, minimum in (
        ("position_error_m", t("Position error (m)", "위치 오차 (m)"), "≤2", False),
        (
            "foot_vertical_speed_m_s",
            t("Foot vertical speed (m/s)", "발 수직속도 (m/s)"),
            "≤1.5",
            False,
        ),
        (
            "foot_horizontal_speed_m_s",
            t("Foot horizontal speed (m/s)", "발 수평속도 (m/s)"),
            "≤0.5",
            False,
        ),
        ("tilt_deg", t("Touchdown tilt (deg)", "접지 자세 오차 (deg)"), "≤3", False),
        (
            "contact_angular_rate_deg_s",
            t("Touchdown angular rate (deg/s)", "접지 각속도 (deg/s)"),
            "≤0.5",
            False,
        ),
        (
            "remaining_main_propellant_kg",
            t("Minimum main propellant (kg)", "최소 남은 주 추진제 (kg)"),
            ">0",
            True,
        ),
        (
            "remaining_rcs_propellant_kg",
            t("Minimum RCS propellant (kg)", "최소 남은 RCS 추진제 (kg)"),
            "≥0",
            True,
        ),
    ):
        extreme = extrema(rows, key, minimum)
        metrics.append(
            [
                label,
                "-" if extreme is None else f"{extreme['value']:.6f}",
                limit,
                "-" if extreme is None else extreme["job_id"],
            ]
        )
    worst = table(
        [
            t("Metric", "항목"),
            t("Worst measured", "최악 기록"),
            t("Limit", "기준"),
            t("Case", "조건"),
        ],
        metrics,
    )
    rendered = []
    for row in rows:
        values = [
            "-" if row.get(key) is None else f"{row[key]:.4f}"
            for key in (
                "position_error_m",
                "foot_vertical_speed_m_s",
                "foot_horizontal_speed_m_s",
                "remaining_main_propellant_kg",
                "remaining_rcs_propellant_kg",
            )
        ]
        failure = row.get("execution_error", ", ".join(row.get("failed_criteria", [])))
        rendered.append(
            [
                row["job_id"],
                t("PASS", "성공") if row["requirements_passed"] else t("FAIL", "실패"),
                *values,
                failure,
            ]
        )
    cases = table(
        [
            t("Case", "조건"),
            t("Result", "결과"),
            "|x| (m)",
            "|vz| (m/s)",
            "|vx| (m/s)",
            "Main (kg)",
            "RCS (kg)",
            t("Failure", "실패 항목"),
        ],
        rendered,
    )
    configured = sum(row.get("gust_configured", False) for row in rows)
    encountered = sum(
        row.get("gust_configured", False) and row.get("gust_encountered", False) for row in rows
    )
    switch = ""
    if paired:
        switch = (
            '<a href="validation-report.html">English</a> | '
            '<a href="validation-report.ko.html">한국어</a>'
        )
    scope = t(
        "Stored full-flight results: launch, cold separation, first-stage return to the launch "
        "pad and first-foot touchdown. This page performs no simulation.",
        "저장된 전체 비행 결과: 발사, 콜드 분리, 1단 발사장 귀환과 첫 발 접지. "
        "이 페이지는 시뮬레이션을 실행하지 않습니다.",
    )
    interpretation = t(
        "All conditions have already been observed. The 72 previous holdout cases are regression "
        "checks, not new independent validation. Finite successful samples do not prove every "
        "combination in a continuous operating range.",
        "모든 조건은 이미 관측한 조건입니다. 기존 홀드아웃 72개는 회귀 검사이며 "
        "새 독립 검증이 아닙니다. 유한한 성공 시험은 연속 범위의 모든 조합에 대한 증명이 아닙니다.",
    )
    metric_note = t(
        "Speeds are the maximum absolute actual velocity across the two feet, not COM velocity. "
        "Position error is COM horizontal position. Displayed values are rounded.",
        "속도는 질량중심 속도가 아니라 실제 두 발 속도의 최대 절댓값입니다. "
        "위치 오차는 질량중심의 수평 위치입니다. 표시 숫자는 반올림했습니다.",
    )
    gust_note = t(
        f"Configured gust cases: {configured}; encountered gust cases: {encountered}. "
        "Encounter is observed at accepted integration endpoints with absolute gust contribution "
        "at least 0.1 m/s; it is not the continuous peak.",
        f"돌풍 설정 조건: {configured}개, 실제 조우 조건: {encountered}개. "
        "조우는 채택된 적분 끝점에서 돌풍 성분 절댓값이 0.1 m/s 이상 관측된 경우이며 "
        "연속 최대값은 아닙니다.",
    )
    calm_note = t(
        "This stored calm reference is shown separately; it is not added again to the declared "
        "cases. Separation altitude and peak altitude differ because upward inertia remains "
        "after separation.",
        "별도 저장된 무풍 기준 기록을 표시했으며 선언된 시험에 중복 합산하지 않습니다. "
        "분리 후 상승 관성이 남으므로 분리 고도와 최고 고도는 다릅니다.",
    )
    limitations = t(
        "The model is a theoretical 2D concept, not calibrated hardware certification. Legs are "
        "assumed adequate within the 1.5 m/s vertical-foot limit; material loads and post-contact "
        "settling are not certified. The upper stage carrying a 500 kg payload to 700 km is an "
        "assumption, not an orbital simulation. Elastic structure, detailed heating, plume and "
        "ground damage remain outside scope.",
        "이 모델은 이론 기반 2D 개념이며 실측 보정이나 실기체 인증이 아닙니다. "
        "다리는 발 수직속도 1.5 m/s 이내에서 충분하다고 가정하며 재료 하중과 접지 후 정착은 "
        "인증하지 않습니다. 500 kg 탑재체의 700 km 도달은 2단에 대한 가정이며 궤도 시뮬레이션이 "
        "아닙니다. 탄성 구조, 상세 열·플룸·지면 손상은 범위 밖입니다.",
    )
    content = f"""<!doctype html><html lang="{language}"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>APR 100 t - {t("Validation results", "검증 결과")}</title>
<style>body{{margin:0;background:#f7f9fc;color:#203044;font:16px/1.6 system-ui,sans-serif}}
main{{max-width:1200px;margin:auto;padding:32px}}
section{{background:white;padding:24px;margin:20px 0;border-radius:12px}}
img{{width:100%;height:auto}}a{{color:#126f85}}
table{{border-collapse:collapse;width:100%;font-size:14px}}
td,th{{padding:8px;text-align:left;border-bottom:1px solid #dce4eb}}details{{overflow:auto}}
.muted{{color:#617185}}code{{overflow-wrap:anywhere}}</style></head><body><main>{switch}
<h1>APR 100 t - {t("Current 2D validation", "현재 2D 검증")}</h1>
<p>70,200 kg main + 300 kg RCS · seed 378 ·
{summary["requirements_passed"]}/{summary["total"]} {t("requirements passed", "전체 기준 통과")}</p>
<section><h2>{t("Scope and interpretation", "범위와 해석")}</h2>
<p>{scope}</p><p>{interpretation}</p>{counts}</section>
<section><h2>{t("Touchdown and propellant extremes", "접지·추진제 최악 기록")}</h2>
{worst}<p class="muted">{metric_note}</p></section>
<section><h2>{t("Condition overview", "조건별 결과")}</h2>
<img src="result-overview{suffix}.png" alt="{t("Condition results", "조건별 결과")}">
<p>{gust_note}</p></section>
<section><h2>{t("Current calm-flight telemetry", "현재 무풍 비행 기록")}</h2>
<img src="calm-telemetry{suffix}.png" alt="{t("Calm-flight telemetry", "무풍 비행 기록")}">
<p class="muted">{calm_note}</p></section>
<section><h2>{t("All declared cases", "선언된 모든 시험")}</h2><details>
<summary>{t("Expand all results, including failures", "실패를 포함한 전체 결과 펼치기")}</summary>
{cases}</details></section>
<section><h2>{t("Model limitations", "모델의 한계")}</h2><p>{limitations}</p>
<p><a href="presentation-data.json">
{t("Derived data and SHA-256 evidence bindings", "재계산 데이터와 SHA-256 근거 연결")}
</a></p></section>
</main></body></html>"""
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create current APR 100 t reports from stored evidence only"
    )
    parser.add_argument("--campaign-dir", type=Path, default=CAMPAIGN)
    parser.add_argument("--nominal-result", type=Path)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT)
    parser.add_argument("--language", choices=("en", "ko", "both"), default="en")
    args = parser.parse_args()
    campaign, output = args.campaign_dir.resolve(), args.out_dir.resolve()
    require(
        campaign != output and campaign not in output.parents,
        "Do not write reports inside raw evidence",
    )
    nominal_path = (args.nominal_result or campaign / "calm/result.json").resolve()
    summary, rows, nominal = load_evidence(campaign, nominal_path)
    output.mkdir(parents=True, exist_ok=True)
    languages = ("en", "ko") if args.language == "both" else (args.language,)
    for language in languages:
        suffix = ".ko" if language == "ko" else ""
        set_plot_style(language)
        plot_overview(rows, output / f"result-overview{suffix}.png", language)
        plot_telemetry(nominal, output / f"calm-telemetry{suffix}.png", language)
        write_report(
            summary,
            rows,
            output / f"validation-report{suffix}.html",
            language,
            suffix,
            len(languages) == 2,
        )
        print(output / f"validation-report{suffix}.html")
    paths = [
        campaign / name for name in ("protocol.json", "all-summary.json", "all-results.jsonl")
    ] + [nominal_path]
    data = {
        "scope": "Derived presentation of stored current 2D regression; no new simulation",
        "previous_holdout_is_fresh_independent": False,
        "continuous_envelope_proven": False,
        "source_files_sha256": {str(path): sha256(path) for path in paths},
        "rendering_script_sha256": sha256(Path(__file__)),
        "counts": summary["by_split"],
        "rows": rows,
    }
    (output / "presentation-data.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
