"""Freeze the residual-RL evaluation protocol and survey PID development difficulty."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from powered_landing_guidance.rl_protocol import (
    CSV_FIELDS,
    GROUPS,
    load_protocol,
    pilot_report,
    run_pid_case,
    sample_cases,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=root / "configs/rl-evaluation-v1.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "artifacts/day36-rl-protocol",
    )
    parser.add_argument(
        "--pilot-cases",
        type=int,
        help="development cases per level; default is the frozen protocol value",
    )
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    pilot_cases = (
        protocol.settings["pilot_cases_per_level"] if args.pilot_cases is None else args.pilot_cases
    )
    if not 1 <= pilot_cases <= protocol.settings["splits"]["development"]["episodes_per_group"]:
        parser.error("--pilot-cases must be within the development split budget")

    output = args.output_dir.resolve()
    csv_path = output / "pid-development-pilot.csv"
    report_path = output / "protocol-report.json"
    if csv_path.exists() or report_path.exists():
        raise FileExistsError(f"Day 36 outputs already exist in {output}")
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for group in GROUPS:
        for level in protocol.settings["groups"][group]["levels"]:
            name = level["name"]
            cases = sample_cases(
                protocol,
                "development",
                group,
                level=name,
                count=pilot_cases,
            )
            results = [run_pid_case(protocol, case) for case in cases]
            rows.extend(results)
            successes = sum(row["safe_landing"] for row in results)
            print(f"{group}/{name}: PID {successes}/{len(results)} safe landings", flush=True)

    report = pilot_report(protocol, rows)
    report["pilot_cases_per_level"] = pilot_cases
    report["canonical_pilot_count"] = pilot_cases == protocol.settings["pilot_cases_per_level"]
    with csv_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"saved {len(rows)} PID pilot rows to {csv_path}")
    print(f"saved frozen protocol report to {report_path}")


if __name__ == "__main__":
    main()
