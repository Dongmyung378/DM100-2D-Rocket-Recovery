"""Predeclared new conditions, paired ablations, sensor and numerical experiments."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from powered_landing_guidance.apr_recovery import AprConditions, AprFlight, RecoveryDisturbances
from powered_landing_guidance.observation_models import ControllerOptions, SensorSettings
from scripts.validate_evidence import digest, encoded, validate_rows

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/validation-study-v1.json"
SOURCE_FILES = (
    "src/powered_landing_guidance/apr_recovery.py",
    "src/powered_landing_guidance/recovery_physics.py",
    "src/powered_landing_guidance/vehicle_systems.py",
    "src/powered_landing_guidance/integrators.py",
    "src/powered_landing_guidance/observation_models.py",
    "scripts/evaluate_study.py",
    "scripts/validate_evidence.py",
    "configs/validation-study-v1.json",
    "requirements/constraints.txt",
    "pyproject.toml",
)
VARIANTS = {
    "full": ControllerOptions(),
    "raw-attitude": ControllerOptions(attitude_filter=False),
    "no-translation-bias": ControllerOptions(translation_bias_compensation=False),
    "no-torque-bias": ControllerOptions(torque_bias_compensation=False),
    "no-replanning": ControllerOptions(boost_replanning=False),
}


def source_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def environment():
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "PyYAML", "matplotlib")
        },
    }


def write_new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                value,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
                default=lambda item: item.item(),
            )
            + "\n"
        )


def random_case(rng, case_id, instrumented):
    pad, upper = rng.uniform(-6, 6), rng.uniform(-18, 18)
    wind = (
        (0.0, float(pad)),
        (4000.0, float(0.65 * pad + 0.35 * upper + rng.uniform(-2, 2))),
        (10000.0, float(0.2 * pad + 0.8 * upper + rng.uniform(-2, 2))),
        (22000.0, float(upper)),
        (100000.0, float(upper)),
    )
    forecast_offset = float(rng.uniform(-1.8, 1.8))
    vertical = float(rng.uniform(-2.5, 2.5))
    gust_event = int(rng.integers(0, 3))
    start, center = ((35, 9000), (270, 20000), (310, 0))[gust_event]
    conditions = AprConditions(
        wind_profile=wind,
        forecast_wind_profile=tuple((z, w + forecast_offset) for z, w in wind),
        vertical_wind_profile=((0.0, vertical), (100000.0, vertical)),
        gust_amplitude_m_s=float(rng.uniform(0, 5)),
        gust_period_s=float(rng.uniform(8, 60)),
        gust_phase_rad=float(rng.uniform(-np.pi, np.pi)),
        gust_start_s=float(start),
        gust_duration_s=120.0,
        gust_altitude_center_m=float(center),
        gust_altitude_half_width_m=5000.0,
        density_scale=float(rng.uniform(0.97, 1.03)),
        sea_level_temperature_k=float(rng.uniform(287.4, 288.9)),
        sea_level_pressure_pa=float(rng.uniform(100925, 101725)),
        latitude_deg=float(rng.uniform(-60, 60)),
    )
    deviations = RecoveryDisturbances(
        thrust_scale=float(rng.uniform(0.97, 1.03)),
        drag_coefficient_scale=float(rng.uniform(0.85, 1.15)),
        initial_mass_scale=float(rng.uniform(0.99, 1.01)),
        inertia_scale=float(rng.uniform(0.97, 1.03)),
        sensor_position_error_m=float(rng.uniform(0, 2)),
        sensor_velocity_error_m_s=float(rng.uniform(0, 1)),
        observation_delay_steps=int(rng.integers(0, 3)),
        action_delay_steps=int(rng.integers(0, 2)),
        engine_lag_s=float(rng.uniform(0, 0.05)),
    )
    sensors = (
        SensorSettings()
        if not instrumented
        else SensorSettings(
            attitude_error_deg=float(rng.uniform(0.01, 0.05)),
            gyro_error_deg_s=float(rng.uniform(0.005, 0.03)),
            attitude_bias_deg=float(rng.uniform(-0.02, 0.02)),
            gyro_bias_deg_s=float(rng.uniform(-0.03, 0.03)),
            dropout_period_steps=int(rng.integers(60, 121)),
            dropout_length_steps=int(rng.integers(1, 3)),
        )
    )
    return {
        "case_id": case_id,
        "conditions": asdict(conditions),
        "disturbances": asdict(deviations),
        "sensors": asdict(sensors),
        "seed": int(rng.integers(1, 2**32 - 1)),
        "gust_event": gust_event,
    }


def make_manifest():
    cases = []
    jobs = []

    def add(case, split, variant="full", dt=0.01):
        if not any(old["case_id"] == case["case_id"] for old in cases):
            cases.append(case)
        job = {
            "job_id": f"{split}/{case['case_id']}/{variant}/dt-{dt:g}",
            "split": split,
            "variant": variant,
            "case_id": case["case_id"],
            "seed": case["seed"],
            "integration_dt_s": dt,
            "case_sha256": digest(case),
        }
        jobs.append(job)

    streams = {
        "development": (16, 12764103, True),
        "holdout": (96, 58392017, False),
        "sensor-holdout": (48, 90732169, True),
    }
    development = []
    for split, (count, seed, instrumented) in streams.items():
        rng = np.random.default_rng(seed)
        for index in range(count):
            case = random_case(rng, f"{split}-{index:03d}", instrumented)
            add(case, split)
            if split == "development":
                development.append(case)
                add(case, split, "raw-attitude")
            elif split == "sensor-holdout":
                add(case, split, "raw-attitude")
    for case in development[:12]:
        for variant in VARIANTS:
            add(case, "ablation", variant)
    calm = {
        "case_id": "calm-reference",
        "conditions": asdict(AprConditions()),
        "disturbances": asdict(RecoveryDisturbances()),
        "sensors": asdict(SensorSettings()),
        "seed": 746103,
        "gust_event": None,
    }
    for case in (calm, *development[:3]):
        for dt in (0.01, 0.005, 0.0025):
            add(case, "convergence", dt=dt)
    add(calm, "sensitivity")
    for name, values in (
        ("thrust_scale", (0.97, 1.03)),
        ("drag_coefficient_scale", (0.85, 1.15)),
        ("inertia_scale", (0.97, 1.03)),
        ("initial_mass_scale", (0.99, 1.01)),
    ):
        for value in values:
            case = calm | {
                "case_id": f"sensitivity-{name}-{value:g}",
                "disturbances": asdict(replace(RecoveryDisturbances(), **{name: value})),
            }
            add(case, "sensitivity")
    manifest = {
        "schema_version": 1,
        "study_id": "DM100-validation-study-v1",
        "scope": "Synthetic 2D conceptual plant; first foot contact; fixed original acceptance",
        "sampling": "Independent uniform draws from the declared coupled-condition recipe",
        "streams": streams,
        "cases": cases,
        "jobs": jobs,
        "acceptance": {
            "position_error_m": 2.0,
            "vertical_foot_speed_m_s": 1.5,
            "horizontal_foot_speed_m_s": 0.5,
            "tilt_deg": 3.0,
            "contact_rate_deg_s": 0.5,
            "peak_dynamic_pressure_pa": 30000.0,
            "peak_angular_acceleration_deg_s2": 5.0,
        },
        "convergence_tolerances": {
            "position_error_m": 0.15,
            "vertical_foot_speed_m_s": 0.08,
            "horizontal_foot_speed_m_s": 0.05,
            "flight_time_s": 0.15,
            "main_remaining_kg": 40.0,
            "rcs_remaining_kg": 3.0,
            "peak_dynamic_pressure_pa": 100.0,
        },
        "no_holdout_outcome_access_before_freeze": True,
        "confidence_scope": (
            "Wilson intervals conditional on this synthetic sampling recipe; not real flights"
        ),
        "controller_variants": {name: asdict(options) for name, options in VARIANTS.items()},
    }
    manifest["manifest_sha256"] = digest(manifest)
    return manifest


def load_manifest():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if digest(unsigned) != manifest["manifest_sha256"]:
        raise ValueError("Manifest SHA does not match")
    if manifest["controller_variants"] != {
        name: asdict(options) for name, options in VARIANTS.items()
    }:
        raise ValueError("Controller options differ from the declared variants")
    for case in manifest["cases"]:
        AprConditions(**case["conditions"])
        RecoveryDisturbances(**case["disturbances"])
        SensorSettings(**case["sensors"])
    return manifest


def evaluate_job(job, case, hashes):
    started = time.perf_counter()
    base = {
        key: job[key]
        for key in (
            "job_id",
            "split",
            "variant",
            "case_id",
            "seed",
            "integration_dt_s",
            "case_sha256",
        )
    }
    base.update(control_dt_s=0.02, source_bundle_sha256=digest(hashes))
    try:
        flight = AprFlight(
            conditions=AprConditions(**case["conditions"]),
            disturbances=RecoveryDisturbances(**case["disturbances"]),
            sensor_settings=SensorSettings(**case["sensors"]),
            controller_options=VARIANTS[job["variant"]],
            integration_dt_s=job["integration_dt_s"],
        )
        flight.rng = np.random.default_rng(job["seed"])
        flight.sensor_rng = np.random.default_rng(np.random.SeedSequence([job["seed"], 1]))
        result = flight.run()
        contact = result["contact"]
        feet = None if contact is None else np.asarray(contact["foot_velocities_m_s"])
        state = None if contact is None else contact["state"]
        main, rcs = result["main_propellant_remaining_kg"], result["rcs_propellant_remaining_kg"]
        ledger = (
            abs(result["main_propellant_used_kg"] + main - 70200.0) <= 1e-6
            and abs(result["rcs_propellant_used_kg"] + rcs - 300.0) <= 1e-6
        )
        base.update(
            requirements_passed=bool(result["simulated_acceptance_passed"] and ledger),
            mission_success=result["mission_success"],
            flight_safety=result["flight_safety"],
            landing_success=result["landing_success"],
            criteria=result["criteria"],
            failed_criteria=sorted(
                name
                for name, value in result["criteria"].items()
                if value is not None and not bool(value)
            ),
            outcome=result["outcome"],
            failure_reasons=result["failures"],
            execution_error=None,
            contact=contact,
            separation=result["separation"],
            position_error_m=None if state is None else abs(state[0]),
            vertical_foot_speed_m_s=None if feet is None else float(np.max(abs(feet[:, 1]))),
            horizontal_foot_speed_m_s=None if feet is None else float(np.max(abs(feet[:, 0]))),
            tilt_deg=None if state is None else abs(float(np.rad2deg(state[4]))),
            contact_rate_deg_s=None if state is None else abs(float(np.rad2deg(state[5]))),
            main_remaining_kg=main,
            rcs_remaining_kg=rcs,
            main_used_kg=result["main_propellant_used_kg"],
            rcs_used_kg=result["rcs_propellant_used_kg"],
            peak_dynamic_pressure_pa=result["max_dynamic_pressure_pa"],
            peak_angular_acceleration_deg_s2=result["max_angular_acceleration_deg_s2"],
            flight_time_s=result["time_s"],
            sensor_diagnostics=result["sensor_diagnostics"],
            replanning_count=len(result["return_plan_history"]),
            trace=[
                {
                    key: step[key]
                    for key in ("time_s", "phase", "state", "control", "observed_control_state")
                }
                for step in result["steps"][::25]
            ],
            wall_seconds=time.perf_counter() - started,
        )
    except Exception as error:
        base.update(
            requirements_passed=False,
            execution_error=type(error).__name__ + ": " + str(error),
            failed_criteria=[],
            wall_seconds=time.perf_counter() - started,
        )
    # Convert NumPy scalars before evidence serialization.
    return json.loads(json.dumps(base, default=lambda value: value.item(), allow_nan=False))


def prepare(directory):
    manifest = load_manifest()
    directory.mkdir(parents=True, exist_ok=True)
    write_new(
        directory / "protocol.json",
        {
            "manifest_sha256": manifest["manifest_sha256"],
            "source_hashes": source_hashes(),
            "environment": environment(),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )
    write_new(directory / "manifest.json", manifest)


def freeze(directory):
    protocol = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
    if protocol["source_hashes"] != source_hashes():
        raise ValueError("Source changed after preparation; prepare a new development campaign")
    development = directory / "development-results.jsonl"
    manifest = load_manifest()
    rows = [json.loads(line) for line in development.read_text(encoding="utf-8").splitlines()]
    expected = [job["job_id"] for job in manifest["jobs"] if job["split"] == "development"]
    validate_rows(manifest, rows, expected, digest(protocol["source_hashes"]))
    write_new(
        directory / "freeze.json",
        protocol
        | {
            "development_results_sha256": hashlib.sha256(development.read_bytes()).hexdigest(),
            "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "holdout_inspected": False,
            "acceptance_relaxed": False,
        },
    )


def run(directory, phase, workers, resume=False):
    manifest = load_manifest()
    protocol = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
    hashes = source_hashes()
    if (
        protocol["source_hashes"] != hashes
        or protocol["manifest_sha256"] != manifest["manifest_sha256"]
    ):
        raise ValueError("Source or manifest changed after preparation")
    selected_splits = ("holdout", "sensor-holdout") if phase == "final" else (phase,)
    if phase == "final":
        frozen = json.loads((directory / "freeze.json").read_text(encoding="utf-8"))
        if frozen["source_hashes"] != hashes or frozen["holdout_inspected"] is not False:
            raise ValueError("Final evaluation requires an unchanged frozen controller")
    jobs = [job for job in manifest["jobs"] if job["split"] in selected_splits]
    cases = {case["case_id"]: case for case in manifest["cases"]}
    path = directory / f"{phase}-results.jsonl"
    rows = []
    if resume:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if any(row["source_bundle_sha256"] != digest(hashes) for row in rows):
            raise ValueError("Resume evidence uses another source")
    completed = {row["job_id"] for row in rows}
    if len(completed) != len(rows) or not completed <= {job["job_id"] for job in jobs}:
        raise ValueError("Invalid partial evidence")
    with (
        path.open("a" if resume else "x", encoding="utf-8") as stream,
        ProcessPoolExecutor(max_workers=workers) as pool,
    ):
        pending = {
            pool.submit(evaluate_job, job, cases[job["case_id"]], hashes): job
            for job in jobs
            if job["job_id"] not in completed
        }
        for future in as_completed(pending):
            row = future.result()
            rows.append(row)
            stream.write(encoded(row) + "\n")
            stream.flush()
            print(
                f"[{len(rows)}/{len(jobs)}] {row['job_id']}: "
                f"{'pass' if row['requirements_passed'] else 'fail'} "
                f"({row['wall_seconds']:.1f}s)",
                flush=True,
            )
    if hashes != source_hashes():
        raise ValueError("Source changed during calculation")
    summary = validate_rows(manifest, rows, [job["job_id"] for job in jobs], digest(hashes))
    write_new(directory / f"{phase}-summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--make-manifest", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument(
        "--run", choices=("development", "final", "ablation", "convergence", "sensitivity")
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/validation-study-v1")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("workers must be positive")
    if args.make_manifest:
        write_new(MANIFEST, make_manifest())
    if args.prepare:
        prepare(args.output_dir)
    if args.freeze:
        freeze(args.output_dir)
    if args.run:
        run(args.output_dir, args.run, args.workers, args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
