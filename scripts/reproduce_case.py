"""Rerun one exact declared study input without a full campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.evaluate_study import evaluate_job, load_manifest, source_hashes
from scripts.validate_evidence import validate_rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = load_manifest()
    try:
        job = next(job for job in manifest["jobs"] if job["job_id"] == args.job_id)
    except StopIteration:
        parser.error("Unknown job identifier")
    if args.output.exists():
        parser.error("Output already exists; choose a new file")
    case = next(case for case in manifest["cases"] if case["case_id"] == job["case_id"])
    row = evaluate_job(job, case, source_hashes())
    validate_rows(manifest, [row], [job["job_id"]])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(row, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "job_id": job["job_id"],
                "passed": row["requirements_passed"],
                "execution_error": row.get("execution_error"),
                "wall_seconds": row["wall_seconds"],
            }
        )
    )
    return 0 if row.get("execution_error") is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
