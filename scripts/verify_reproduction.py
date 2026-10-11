"""Verify representative fresh-environment flights against published study rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.validate_evidence import digest, validate_rows

TOLERANCES = {
    "position_error_m": 0.001,
    "vertical_foot_speed_m_s": 0.0001,
    "horizontal_foot_speed_m_s": 0.0001,
    "tilt_deg": 0.0001,
    "contact_rate_deg_s": 0.0001,
    "flight_time_s": 0.0001,
    "main_remaining_kg": 0.01,
    "rcs_remaining_kg": 0.01,
    "peak_dynamic_pressure_pa": 1.0,
}


def compare(evidence_dir, reproduced_paths):
    if not reproduced_paths:
        raise ValueError("At least one reproduced flight is required")
    manifest = json.loads((evidence_dir / "manifest.json").read_text(encoding="utf-8"))
    protocol = json.loads((evidence_dir / "protocol.json").read_text(encoding="utf-8"))
    binding = digest(protocol["source_hashes"])
    official = {
        row["job_id"]: row
        for row in (
            json.loads(line)
            for line in (evidence_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
        )
    }
    checks = []
    for path in reproduced_paths:
        row = json.loads(path.read_text(encoding="utf-8"))
        validate_rows(manifest, [row], [row["job_id"]], binding)
        reference = official[row["job_id"]]
        if any(row[key] != reference[key] for key in ("criteria", "requirements_passed")):
            raise ValueError(f"Acceptance differs for {path.name}")
        differences = {}
        for metric, tolerance in TOLERANCES.items():
            if row.get(metric) is None or reference.get(metric) is None:
                raise ValueError(f"Missing representative measurement: {metric}")
            error = abs(row[metric] - reference[metric])
            if error > tolerance:
                raise ValueError(f"Reproduction exceeds {metric} tolerance: {error}")
            differences[metric] = error
        checks.append(
            {"file": path.name, "job_id": row["job_id"], "absolute_differences": differences}
        )
    return {
        "representative_reproductions": len(checks),
        "source_bundle_sha256": binding,
        "all_acceptance_equal": True,
        "all_within_declared_tolerances": True,
        "declared_tolerances": TOLERANCES,
        "checks": checks,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("reproduced_files", type=Path, nargs="+")
    args = parser.parse_args(argv)
    files = [
        file
        for path in args.reproduced_files
        for file in (sorted(path.glob("*.json")) if path.is_dir() else [path])
    ]
    print(json.dumps(compare(args.evidence_dir, files), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
