"""Check the original 286-input regression evidence using only the standard library."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def canonical_sha(value):
    text = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    return hashlib.sha256(text.encode()).hexdigest()


def verify(directory, suite_path):
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    protocol = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (directory / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    unsigned = {key: value for key, value in suite.items() if key != "manifest_sha256"}
    if canonical_sha(unsigned) != suite["manifest_sha256"]:
        raise ValueError("Regression manifest SHA differs")
    if (
        protocol["suite_manifest_sha256"] != suite["manifest_sha256"]
        or protocol["jobs"] != suite["jobs"]
    ):
        raise ValueError("Declared inputs changed")
    jobs = {job["job_id"]: job for job in suite["jobs"]}
    if (
        len(rows) != 286
        or len({row["job_id"] for row in rows}) != 286
        or set(jobs) != {row["job_id"] for row in rows}
    ):
        raise ValueError("Regression results have missing or duplicate rows")
    completed = accepted = passed = 0
    values = {
        key: []
        for key in (
            "position_error_m",
            "vertical_foot_speed_m_s",
            "horizontal_foot_speed_m_s",
            "tilt_deg",
            "contact_rate_deg_s",
            "main_remaining_kg",
            "rcs_remaining_kg",
            "peak_dynamic_pressure_pa",
            "peak_angular_acceleration_deg_s2",
        )
    }
    for row in rows:
        job = jobs[row["job_id"]]
        if row["split"] != job["split"]:
            raise ValueError("Regression split changed")
        if "result" not in row:
            if row["requirements_passed"] is not False:
                raise ValueError("Execution error cannot pass")
            continue
        completed += 1
        result = row["result"]
        if (
            row["input_sha256"] != job["input_sha256"]
            or row["observation_seed_sequence"] != job["observation_seed_sequence"]
        ):
            raise ValueError("Case input or sensor random stream changed")
        if row["source_hashes_after"] != protocol["source_hashes"]:
            raise ValueError("Source changed during regression")
        correct_acceptance = (
            result["mission_success"] is True
            and result["flight_safety"] is True
            and result["landing_success"] is True
        )
        if result["accepted"] != correct_acceptance:
            raise ValueError("Acceptance contradicts recorded sub-assessments")
        accepted += int(correct_acceptance)
        ledger = (
            abs(result["main_propellant_used_kg"] + result["remaining_main_propellant_kg"] - 70200)
            <= 1e-6
            and abs(result["rcs_propellant_used_kg"] + result["remaining_rcs_propellant_kg"] - 300)
            <= 1e-6
        )
        phase = job["required_exposure_phase"]
        actual = (
            result["gust_exposure"]
            .get("integration_endpoints", {})
            .get("maximum_by_phase_m_s", {})
            .get(phase, 0)
        )
        exposure = phase is None or actual >= job["required_exposure_peak_m_s"]
        if row["requirements_passed"] != (correct_acceptance and ledger and exposure):
            raise ValueError("Regression pass contradicts conservation or actual gust exposure")
        passed += int(row["requirements_passed"])
        contact = result["contact"]
        metrics = {
            "main_remaining_kg": result["remaining_main_propellant_kg"],
            "rcs_remaining_kg": result["remaining_rcs_propellant_kg"],
            "peak_dynamic_pressure_pa": result["max_dynamic_pressure_pa"],
            "peak_angular_acceleration_deg_s2": result["max_angular_acceleration_deg_s2"],
        }
        if contact:
            state = contact["state"]
            metrics.update(
                position_error_m=abs(state[0]),
                vertical_foot_speed_m_s=max(abs(v[1]) for v in contact["foot_velocities_m_s"]),
                horizontal_foot_speed_m_s=max(abs(v[0]) for v in contact["foot_velocities_m_s"]),
                tilt_deg=abs(math.degrees(state[4])),
                contact_rate_deg_s=abs(math.degrees(state[5])),
            )
        if row["requirements_passed"]:
            limits = {
                "position_error_m": 2.0,
                "vertical_foot_speed_m_s": 1.5,
                "horizontal_foot_speed_m_s": 0.5,
                "tilt_deg": 3.0,
                "contact_rate_deg_s": 0.5,
                "peak_dynamic_pressure_pa": 30000.0,
                "peak_angular_acceleration_deg_s2": 5.0,
            }
            if not contact or not all(contact["criteria"].values()):
                raise ValueError("Missing passing contact evidence")
            if any(metrics.get(key, float("inf")) > limit + 1e-7 for key, limit in limits.items()):
                raise ValueError("Passing regression exceeds a physical threshold")
        for key, value in metrics.items():
            if not math.isfinite(value):
                raise ValueError("Non-finite measurement")
            values[key].append(value)
    actual = {
        "total": 286,
        "completed": completed,
        "accepted": accepted,
        "requirements_passed": passed,
    }
    if any(summary.get(key) != value for key, value in actual.items()):
        raise ValueError("Regression summary differs from recorded results")
    return actual | {
        "extrema": {
            key: ({"count": len(v), "min": min(v), "max": max(v)} if v else None)
            for key, v in values.items()
        }
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--suite",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/apr-regression-v1.json",
    )
    args = parser.parse_args(argv)
    result = verify(args.directory, args.suite)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
