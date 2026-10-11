"""Curate complete recorded experiments into a reviewable public evidence bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from scripts.validate_evidence import digest, encoded, percentile, validate_rows


def analysis(manifest, rows):
    paired = {}
    for split in ("ablation", "sensor-holdout"):
        selected = [row for row in rows if row["split"] == split]
        lookup = {(row["case_id"], row["variant"]): row for row in selected}
        variants = sorted({row["variant"] for row in selected} - {"full"})
        for variant in variants:
            pairs = [
                (lookup[case, "full"], lookup[case, variant])
                for case, name in lookup
                if name == "full"
            ]
            deltas = {}
            for metric in ("position_error_m", "vertical_foot_speed_m_s", "rcs_remaining_kg"):
                values = [
                    full[metric] - other[metric]
                    for full, other in pairs
                    if full.get(metric) is not None and other.get(metric) is not None
                ]
                deltas[metric] = {
                    "pairs_with_measurement": len(values),
                    "median_full_minus_other": percentile(values, 0.5),
                }
            paired[f"{split}/{variant}"] = {
                "pairs": len(pairs),
                "full_only_pass": sum(
                    full["requirements_passed"] and not other["requirements_passed"]
                    for full, other in pairs
                ),
                "other_only_pass": sum(
                    other["requirements_passed"] and not full["requirements_passed"]
                    for full, other in pairs
                ),
                "both_pass": sum(
                    full["requirements_passed"] and other["requirements_passed"]
                    for full, other in pairs
                ),
                "neither_pass": sum(
                    not full["requirements_passed"] and not other["requirements_passed"]
                    for full, other in pairs
                ),
                "deltas": deltas,
            }
    convergence = []
    selected = [row for row in rows if row["split"] == "convergence"]
    for case in sorted({row["case_id"] for row in selected}):
        values = {row["integration_dt_s"]: row for row in selected if row["case_id"] == case}
        reference = values[0.0025]
        differences = {}
        for dt in (0.01, 0.005):
            errors = {
                metric: None
                if values[dt].get(metric) is None or reference.get(metric) is None
                else abs(values[dt][metric] - reference[metric])
                for metric in manifest["convergence_tolerances"]
            }
            if values[dt].get("contact") and reference.get("contact"):
                errors["position_error_m"] = abs(
                    values[dt]["contact"]["state"][0] - reference["contact"]["state"][0]
                )
            differences[str(dt)] = {
                "absolute_difference_from_finest": errors,
                "within_declared_tolerances": all(
                    error is not None and error <= manifest["convergence_tolerances"][metric]
                    for metric, error in errors.items()
                ),
            }
        convergence.append(
            {
                "case_id": case,
                "reference_integration_dt_s": 0.0025,
                "control_dt_s": 0.02,
                "same_declared_seeds": len({row["seed"] for row in values.values()}) == 1,
                "acceptance_consistent": len(
                    {row["requirements_passed"] for row in values.values()}
                )
                == 1,
                "all_accepted": all(row["requirements_passed"] for row in values.values()),
                "differences": differences,
                "interpretation": (
                    "Finite step-size comparison, not an exact solution "
                    "or continuous-envelope proof"
                ),
            }
        )
    sensitivity = []
    selected = [row for row in rows if row["split"] == "sensitivity"]
    nominal = next(row for row in selected if row["case_id"] == "calm-reference")
    cases = {case["case_id"]: case for case in manifest["cases"]}
    for row in selected:
        sensitivity.append(
            {
                "case_id": row["case_id"],
                "accepted": row["requirements_passed"],
                "disturbances": cases[row["case_id"]]["disturbances"],
                "deltas_from_nominal": {
                    metric: None
                    if row.get(metric) is None or nominal.get(metric) is None
                    else row[metric] - nominal[metric]
                    for metric in (
                        "position_error_m",
                        "vertical_foot_speed_m_s",
                        "main_remaining_kg",
                        "rcs_remaining_kg",
                        "peak_dynamic_pressure_pa",
                    )
                },
            }
        )
    return {"paired_comparisons": paired, "convergence": convergence, "sensitivity": sensitivity}


def export(directory, output):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    protocol = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
    frozen = json.loads((directory / "freeze.json").read_text(encoding="utf-8"))
    rows = []
    for phase in ("development", "final", "ablation", "convergence", "sensitivity"):
        rows.extend(
            json.loads(line)
            for line in (directory / f"{phase}-results.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
    if any(row["source_bundle_sha256"] != digest(protocol["source_hashes"]) for row in rows):
        raise ValueError("Evidence comes from mixed source versions")
    if frozen["source_hashes"] != protocol["source_hashes"]:
        raise ValueError("Freeze does not match source declaration")
    summary = validate_rows(manifest, rows, source_bundle_sha256=digest(protocol["source_hashes"]))
    output.mkdir(parents=True, exist_ok=False)
    for filename, value in (
        ("manifest.json", manifest),
        ("protocol.json", protocol),
        ("freeze.json", frozen),
        ("summary.json", summary),
        ("analysis.json", analysis(manifest, rows)),
    ):
        (output / filename).write_text(
            json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    (output / "results.jsonl").write_text(
        "".join(encoded(row) + "\n" for row in rows), encoding="utf-8"
    )
    shutil.copyfile(directory / "development-results.jsonl", output / "development-results.jsonl")
    files = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in output.iterdir()}
    (output / "checksums.json").write_text(json.dumps(files, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {len(rows)} complete rows to {output}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    export(args.campaign_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
