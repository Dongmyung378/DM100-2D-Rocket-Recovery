"""Evaluate 286 known conditions for the current 100 t rocket; not a fresh holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from functools import partial
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import evaluate_operating_envelope as campaign  # noqa: E402

DEFAULT_SUITE = ROOT / "configs/apr-regression-v1.json"
DEFAULT_OUTPUT = ROOT / "artifacts/apr100t/regression-v1"
SOURCE_PATHS = (
    *campaign.SOURCE_PATHS,
    "scripts/run_nominal_flight.py",
    "src/powered_landing_guidance/__init__.py",
    "scripts/evaluate_regression_suite.py",
    "configs/apr-regression-v1.json",
)
SPLITS = ("development", "previous-holdout", "supplemental")
EXPECTED_COUNTS = dict(zip(SPLITS, (205, 72, 9), strict=True))


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    return {name: file_sha(ROOT / name) for name in SOURCE_PATHS}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(
            value,
            stream,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
            default=lambda item: item.item(),
        )
        stream.write("\n")


def seed_metadata(case, mode):
    if mode not in ("case", "root"):
        raise ValueError("Observation RNG mode must be case or root.")
    sequence = (
        campaign.observation_seed_sequence(case)
        if mode == "case"
        else np.random.SeedSequence(campaign.ROOT_SEED)
    )
    return {
        "mode": mode,
        "state": sequence.state,
        "generated_state_uint32": sequence.generate_state(8).tolist(),
    }


def load_suite(path=DEFAULT_SUITE):
    suite = read_json(path)
    unsigned = {key: value for key, value in suite.items() if key != "manifest_sha256"}
    if suite.get("manifest_sha256") != campaign._sha(unsigned):
        raise ValueError("Regression suite canonical SHA does not match.")
    if (
        suite.get("schema_version") != 1
        or suite.get("suite_id") != "APR100t-regression-v1"
        or suite.get("root_seed") != 378
        or suite.get("fresh_independent") is not False
        or suite.get("counts") != EXPECTED_COUNTS
    ):
        raise ValueError("Regression suite contract or declared counts do not match.")
    design = campaign.Design()
    if (
        suite["design"] != asdict(design)
        or suite["launch_mass_kg"] != design.launch_mass_kg
        or design.launch_mass_kg != 100000.0
        or design.main_propellant_kg != 70200.0
        or design.rcs_propellant_kg != 300.0
    ):
        raise ValueError("Suite must match the current 100 t design with 70,200/300 kg propellant.")
    jobs = suite["jobs"]
    if (
        len(jobs) != 286
        or len({job["job_id"] for job in jobs}) != 286
        or dict(Counter(job["split"] for job in jobs)) != EXPECTED_COUNTS
    ):
        raise ValueError("Suite jobs must be unique and contain exactly 205+72+9 cases.")
    for job in jobs:
        case = job["case"]
        dt = job["integration_dt_s"]
        mode = job["observation_seed_sequence"]["mode"]
        phase = job["required_exposure_phase"]
        if (
            not job["job_id"].startswith(job["split"] + "/")
            or isinstance(dt, bool)
            or dt not in (0.01, 0.005)
            or job["input_sha256"] != campaign._sha(case)
            or phase not in (None, "entry", "ascent")
            or job["required_exposure_peak_m_s"] != (0.1 if phase else None)
        ):
            raise ValueError("Job input, integration interval or exposure contract does not match.")
        if campaign._sha(job["observation_seed_sequence"]) != campaign._sha(
            seed_metadata(case, mode)
        ):
            raise ValueError("Job observation RNG does not match the original stream.")
        if job["split"] != "supplemental" and (dt != 0.01 or mode != "case" or phase is not None):
            raise ValueError("Primary integration interval, observation RNG or criteria changed.")
        campaign.AprConditions(**case["conditions"])
        campaign.RecoveryDisturbances(**case["disturbances"])
    supplemental = [job for job in jobs if job["split"] == "supplemental"]
    if (
        sum(job["integration_dt_s"] == 0.005 for job in supplemental) != 4
        or sum(job["observation_seed_sequence"]["mode"] == "root" for job in supplemental) != 4
        or sum(job["required_exposure_phase"] == "entry" for job in supplemental) != 3
        or sum(job["required_exposure_phase"] == "ascent" for job in supplemental) != 3
    ):
        raise ValueError("The nine supplemental integration, RNG or exposure contracts changed.")
    return suite


def prepare(output_dir):
    suite = load_suite()
    protocol = {
        "record_type": "regression_execution_protocol",
        "schema_version": 1,
        "root_seed": 378,
        "fresh_independent": False,
        "scope": suite["scope"],
        "suite_manifest_sha256": suite["manifest_sha256"],
        "source_hashes": source_hashes(),
        "design": suite["design"],
        "total": len(suite["jobs"]),
        "counts": EXPECTED_COUNTS,
        "jobs": suite["jobs"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_new(output_dir / "protocol.json", protocol)
    return protocol


def evaluate_job(job):
    started, cpu = time.perf_counter(), time.process_time()
    factory, seed_factory = campaign.AprFlight, campaign.observation_seed_sequence
    try:
        campaign.AprFlight = partial(factory, integration_dt_s=job["integration_dt_s"])
        if job["observation_seed_sequence"]["mode"] == "root":
            campaign.observation_seed_sequence = lambda case: np.random.SeedSequence(378)
        result = campaign._evaluate_case(job["case"])
    finally:
        campaign.AprFlight, campaign.observation_seed_sequence = factory, seed_factory
    design = campaign.Design()
    ledger = {
        "main_error_kg": result["main_propellant_used_kg"]
        + result["remaining_main_propellant_kg"]
        - design.main_propellant_kg,
        "rcs_error_kg": result["rcs_propellant_used_kg"]
        + result["remaining_rcs_propellant_kg"]
        - design.rcs_propellant_kg,
    }
    ledger["passed"] = all(math.isfinite(value) and abs(value) <= 1e-6 for value in ledger.values())
    phase = job["required_exposure_phase"]
    exposure = result["gust_exposure"].get("integration_endpoints", {})
    peak = exposure.get("maximum_by_phase_m_s", {}).get(phase, 0.0) if phase else None
    exposure_passed = phase is None or peak >= job["required_exposure_peak_m_s"]
    return {
        "record_type": "regression_flight",
        "job_id": job["job_id"],
        "split": job["split"],
        "integration_dt_s": job["integration_dt_s"],
        "input_sha256": job["input_sha256"],
        "observation_seed_sequence": job["observation_seed_sequence"],
        "result": result,
        "fuel_ledger": ledger,
        "required_exposure_phase": phase,
        "actual_required_phase_peak_m_s": peak,
        "required_exposure_passed": exposure_passed,
        "requirements_passed": result["accepted"] is True and ledger["passed"] and exposure_passed,
        "source_hashes_after": source_hashes(),
        "wall_seconds": time.perf_counter() - started,
        "cpu_seconds": time.process_time() - cpu,
    }


def execution_error(job, error):
    return {
        "record_type": "execution_error",
        "job_id": job["job_id"],
        "split": job["split"],
        "error_type": type(error).__name__,
        "message": str(error),
        "requirements_passed": False,
    }


def evaluate_jobs(jobs, workers):
    if workers == 1:
        for job in jobs:
            try:
                yield job, evaluate_job(job)
            except Exception as error:
                yield job, execution_error(job, error)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pending = {pool.submit(evaluate_job, job): job for job in jobs}
            for future in as_completed(pending):
                job = pending[future]
                try:
                    yield job, future.result()
                except Exception as error:
                    yield job, execution_error(job, error)


def summarize(rows, jobs, before, after):
    by_id = {row["job_id"]: row for row in rows}
    if len(by_id) != len(rows) or set(by_id) != {job["job_id"] for job in jobs}:
        raise ValueError("Result job identifiers are duplicated or missing.")
    ordered = [by_id[job["job_id"]] for job in jobs]
    completed = [row for row in ordered if "result" in row]
    failures = Counter(name for row in completed for name in row["result"]["failed_criteria"])
    extrema = {}
    for metric, reduce in (
        ("remaining_main_propellant_kg", min),
        ("remaining_rcs_propellant_kg", min),
        ("main_propellant_used_kg", max),
        ("rcs_propellant_used_kg", max),
    ):
        chosen = (
            reduce(completed, key=lambda row, metric=metric: row["result"][metric])
            if completed
            else None
        )
        extrema[metric] = (
            None
            if chosen is None
            else {"value": chosen["result"][metric], "job_id": chosen["job_id"]}
        )
    return {
        "record_type": "regression_summary",
        "total": len(jobs),
        "completed": len(completed),
        "accepted": sum(row["result"]["accepted"] is True for row in completed),
        "requirements_passed": sum(row["requirements_passed"] is True for row in ordered),
        "execution_errors": [row["job_id"] for row in ordered if "result" not in row],
        "failed_job_ids": [row["job_id"] for row in ordered if not row["requirements_passed"]],
        "failed_criteria": dict(sorted(failures.items())),
        "fuel_extrema": extrema,
        "by_split": {
            split: {
                "total": sum(row["split"] == split for row in ordered),
                "accepted": sum(
                    row["split"] == split and row.get("result", {}).get("accepted") is True
                    for row in ordered
                ),
                "requirements_passed": sum(
                    row["split"] == split and row["requirements_passed"] is True for row in ordered
                ),
            }
            for split in SPLITS
        },
        "source_hashes_before": before,
        "source_hashes_after": after,
        "source_unchanged": before == after
        and all(row.get("source_hashes_after", before) == before for row in ordered),
        "fresh_independent": False,
        "continuous_envelope_proven": False,
        "real_vehicle_safety_certified": False,
        "ordered_job_ids": [job["job_id"] for job in jobs],
    }


def run(output_dir, split, workers=1, case_ids=None):
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer.")
    suite, protocol = load_suite(), read_json(output_dir / "protocol.json")
    before = source_hashes()
    if (
        before != protocol["source_hashes"]
        or suite["manifest_sha256"] != protocol["suite_manifest_sha256"]
        or suite["design"] != protocol["design"]
        or campaign._sha(suite["jobs"]) != campaign._sha(protocol["jobs"])
        or protocol.get("fresh_independent") is not False
    ):
        raise ValueError("Source, conditions or design changed after declaration.")
    jobs = [job for job in suite["jobs"] if split == "all" or job["split"] == split]
    if case_ids:
        wanted = set(case_ids)
        if wanted - {job["job_id"] for job in jobs}:
            raise ValueError("Unknown job_id selected.")
        jobs = [job for job in jobs if job["job_id"] in wanted]
    if not jobs:
        raise ValueError("No jobs selected.")
    tag = split if not case_ids else f"{split}-selection-{campaign._sha(sorted(case_ids))[:12]}"
    paths = [
        output_dir / f"{tag}{suffix}"
        for suffix in ("-execution.jsonl", "-results.jsonl", "-summary.json")
    ]
    if any(path.exists() for path in paths):
        raise FileExistsError("First-run evidence already exists; outputs are never overwritten.")
    header = {
        "record_type": "execution_binding",
        "protocol_sha256": file_sha(output_dir / "protocol.json"),
        "suite_manifest_sha256": suite["manifest_sha256"],
        "source_hashes": before,
        "workers": workers,
        "selected_job_ids": [job["job_id"] for job in jobs],
        "fresh_independent": False,
    }
    rows = []
    with paths[0].open("x", encoding="utf-8") as stream:
        stream.write(campaign._encoded(header) + "\n")
        stream.flush()
        for _, row in evaluate_jobs(jobs, workers):
            rows.append(row)
            stream.write(campaign._encoded(row) + "\n")
            stream.flush()
            status = "pass" if row["requirements_passed"] else "fail"
            print(f"[{len(rows)}/{len(jobs)}] {row['job_id']}: {status}", flush=True)
    summary = summarize(rows, jobs, before, source_hashes())
    summary.update(
        protocol_sha256=header["protocol_sha256"],
        protocol_unchanged=file_sha(output_dir / "protocol.json") == header["protocol_sha256"],
        suite_manifest_sha256=suite["manifest_sha256"],
        workers=workers,
    )
    by_id = {row["job_id"]: row for row in rows}
    with paths[1].open("x", encoding="utf-8") as stream:
        stream.write(campaign._encoded(header) + "\n")
        for job in jobs:
            stream.write(campaign._encoded(by_id[job["job_id"]]) + "\n")
    write_json_new(paths[2], summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", "--campaign-dir", type=Path, default=DEFAULT_OUTPUT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", choices=("all", *SPLITS))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--case-id", action="append", help="Exact job_id, e.g. development/calm")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("workers must be a positive integer.")
    if args.case_id and args.prepare:
        parser.error("--case-id is only available for execution.")
    try:
        value = (
            prepare(args.output_dir)
            if args.prepare
            else run(args.output_dir, args.run, args.workers, args.case_id)
        )
    except (ValueError, FileNotFoundError, FileExistsError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(
        campaign._encoded(
            {
                key: value[key]
                for key in ("total", "accepted", "requirements_passed", "source_unchanged")
                if key in value
            }
        ),
        flush=True,
    )
    return (
        0
        if args.prepare
        or (
            value["requirements_passed"] == value["total"]
            and value["source_unchanged"]
            and value["protocol_unchanged"]
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
