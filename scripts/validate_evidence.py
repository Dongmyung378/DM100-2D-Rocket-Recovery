"""Validate published experiment evidence without importing the simulator."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def encoded(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def digest(value):
    return hashlib.sha256(encoded(value).encode()).hexdigest()


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * fraction
    left = int(index)
    right = min(left + 1, len(values) - 1)
    return values[left] + (values[right] - values[left]) * (index - left)


def wilson_interval(passed, total):
    if not total:
        return None
    z = 1.959963984540054
    proportion = passed / total
    center = (proportion + z * z / (2 * total)) / (1 + z * z / total)
    radius = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
    radius /= 1 + z * z / total
    return [max(0.0, center - radius), min(1.0, center + radius)]


METRICS = (
    "position_error_m",
    "vertical_foot_speed_m_s",
    "horizontal_foot_speed_m_s",
    "tilt_deg",
    "contact_rate_deg_s",
    "main_remaining_kg",
    "rcs_remaining_kg",
    "peak_dynamic_pressure_pa",
    "peak_angular_acceleration_deg_s2",
    "flight_time_s",
    "wall_seconds",
)


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["split"], row["variant"])].append(row)
    summary = {}
    for (split, variant), selected in sorted(groups.items()):
        passed = sum(row["requirements_passed"] is True for row in selected)
        failures = Counter(name for row in selected for name in row.get("failed_criteria", []))
        metrics = {}
        for metric in METRICS:
            values = [row[metric] for row in selected if row.get(metric) is not None]
            metrics[metric] = (
                None
                if not values
                else {
                    "count": len(values),
                    "min": min(values),
                    "median": percentile(values, 0.5),
                    "p95": percentile(values, 0.95),
                    "max": max(values),
                }
            )
        summary[f"{split}/{variant}"] = {
            "total": len(selected),
            "passed": passed,
            "pass_rate": passed / len(selected),
            "wilson_95": wilson_interval(passed, len(selected))
            if split in ("holdout", "sensor-holdout")
            else None,
            "execution_errors": sum(row.get("execution_error") is not None for row in selected),
            "failed_criteria": dict(sorted(failures.items())),
            "metrics": metrics,
            "metric_population": "Every completed flight with that measurement, including failures",
        }
    return summary


def validate_rows(manifest, rows, selected_jobs=None, source_bundle_sha256=None):
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if digest(unsigned) != manifest["manifest_sha256"]:
        raise ValueError("Manifest SHA does not match")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("study_id") != "DM100-validation-study-v1"
    ):
        raise ValueError("Unsupported study contract")
    jobs = {job["job_id"]: job for job in manifest["jobs"]}
    cases = {case["case_id"]: case for case in manifest["cases"]}
    if len(jobs) != len(manifest["jobs"]) or len(cases) != len(manifest["cases"]):
        raise ValueError("Duplicate input identifiers")
    for job in jobs.values():
        case = cases[job["case_id"]]
        if job["case_sha256"] != digest(case) or job["seed"] != case["seed"]:
            raise ValueError("Job input digest or random seed differs from the declared case")
    expected = set(jobs) if selected_jobs is None else set(selected_jobs)
    ids = [row["job_id"] for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != expected or not expected <= jobs.keys():
        raise ValueError("Missing, duplicate or unexpected result rows")
    thresholds = manifest["acceptance"]
    if thresholds != {
        "position_error_m": 2.0,
        "vertical_foot_speed_m_s": 1.5,
        "horizontal_foot_speed_m_s": 0.5,
        "tilt_deg": 3.0,
        "contact_rate_deg_s": 0.5,
        "peak_dynamic_pressure_pa": 30000.0,
        "peak_angular_acceleration_deg_s2": 5.0,
    }:
        raise ValueError("Original acceptance thresholds changed")
    for row in rows:
        job = jobs[row["job_id"]]
        if source_bundle_sha256 is not None and row["source_bundle_sha256"] != source_bundle_sha256:
            raise ValueError("A result row uses another declared source")
        for field in ("split", "variant", "case_id", "seed", "integration_dt_s", "case_sha256"):
            if row[field] != job[field]:
                raise ValueError(f"{row['job_id']}: mismatched {field}")
        if row["control_dt_s"] != 0.02:
            raise ValueError("Control period changed")
        if row.get("execution_error") is not None:
            if row["requirements_passed"] is not False:
                raise ValueError("An execution error cannot pass")
            continue
        for metric in METRICS:
            value = row.get(metric)
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
                raise ValueError("Invalid metric value")
        criteria = row["criteria"]
        if not criteria or any(
            value is not None and type(value) is not bool for value in criteria.values()
        ):
            raise ValueError("Invalid acceptance criteria")
        failed = sorted(name for name, value in criteria.items() if value is False)
        if sorted(row["failed_criteria"]) != failed:
            raise ValueError("Failed criteria were omitted or relabelled")
        accepted = (
            row["mission_success"] is True
            and row["flight_safety"] is True
            and row["landing_success"] is True
            and not failed
        )
        ledger = (
            abs(row["main_used_kg"] + row["main_remaining_kg"] - 70200.0) <= 1e-6
            and abs(row["rcs_used_kg"] + row["rcs_remaining_kg"] - 300.0) <= 1e-6
        )
        if row["requirements_passed"] != (accepted and ledger):
            raise ValueError("Acceptance or fuel accounting is inconsistent")
        if row["requirements_passed"]:
            contact = row.get("contact")
            if contact is None or not all(contact["criteria"].values()):
                raise ValueError("Passing flight lacks passing contact evidence")
            measured = {
                "position_error_m": abs(contact["state"][0]),
                "vertical_foot_speed_m_s": max(abs(v[1]) for v in contact["foot_velocities_m_s"]),
                "horizontal_foot_speed_m_s": max(abs(v[0]) for v in contact["foot_velocities_m_s"]),
                "tilt_deg": abs(math.degrees(contact["state"][4])),
                "contact_rate_deg_s": abs(math.degrees(contact["state"][5])),
            }
            if any(abs(row[metric] - value) > 1e-9 for metric, value in measured.items()):
                raise ValueError("Reported contact metrics differ from the recorded state")
            for metric, maximum in thresholds.items():
                if row.get(metric) is None or row[metric] > maximum + 1e-7:
                    raise ValueError(f"Passing flight exceeds {metric}")
            if row["main_remaining_kg"] <= 0 or row["rcs_remaining_kg"] < 0:
                raise ValueError("Passing flight has exhausted propellant")
    return summarize(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    directory = args.directory
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (directory / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    protocol = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
    frozen = json.loads((directory / "freeze.json").read_text(encoding="utf-8"))
    if (
        frozen["source_hashes"] != protocol["source_hashes"]
        or frozen["holdout_inspected"] is not False
    ):
        raise ValueError("Source freeze is inconsistent")
    development = directory / "development-results.jsonl"
    if hashlib.sha256(development.read_bytes()).hexdigest() != frozen["development_results_sha256"]:
        raise ValueError("Development evidence differs from its frozen hash")
    checksums = json.loads((directory / "checksums.json").read_text(encoding="utf-8"))
    for name, expected_digest in checksums.items():
        if Path(name).name != name:
            raise ValueError("Checksum filename must stay inside the bundle")
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected_digest:
            raise ValueError(f"Bundle file checksum differs: {name}")
    actual = validate_rows(manifest, rows, source_bundle_sha256=digest(protocol["source_hashes"]))
    expected = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if digest(actual) != digest(expected):
        raise ValueError("Published summary differs from recomputed results")
    print(f"Validated {len(rows)} rows and {len(actual)} experiment groups; physics not rerun.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
