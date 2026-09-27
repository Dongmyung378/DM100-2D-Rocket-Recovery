"""Run the Day 26-35 BC, DAgger, stabilization, and held-out evaluation workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from powered_landing_guidance import load_config  # noqa: E402
from powered_landing_guidance.behavior_cloning import (  # noqa: E402
    BCDataset,
    TrainingSettings,
    action_error,
    load_bc_dataset,
    load_checkpoint,
    save_checkpoint,
    train_bc,
)
from powered_landing_guidance.imitation_learning import (  # noqa: E402
    ActuatorSettings,
    episode_metrics,
    evaluate_cohort,
    inference_latency_us,
    initial_states_from_shard,
    run_episode,
    state_distribution_drift,
)
from powered_landing_guidance.teacher_pipeline import solve_teacher_batch  # noqa: E402
from powered_landing_guidance.visualization import save_animation, save_episode_data  # noqa: E402


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(
    states: np.ndarray, actions: np.ndarray, stats: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    state_stats, action_stats = stats["state"], stats["action"]
    x = (states - np.asarray(state_stats["mean"])) / np.asarray(state_stats["scale"])
    y = (actions - np.asarray(action_stats["mean"])) / np.asarray(action_stats["scale"])
    return x.astype(np.float32), y.astype(np.float32)


def checkpoint_metadata(
    config: dict[str, Any],
    dataset: BCDataset,
    settings: TrainingSettings,
    *,
    activation: str,
    name: str,
    parent: Path | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "policy": name,
        "activation": activation,
        "dataset_id": dataset.dataset_id,
        "dataset_configuration_sha256": dataset.dataset_configuration_sha256,
        "shard_sha256": dataset.shard_sha256,
        "normalization": dataset.normalization,
        "training_settings": settings.as_dict(),
        "parent_checkpoint_sha256": None if parent is None else digest(parent),
        "throttle_bounds": [config["vehicle"]["throttle_min"], config["vehicle"]["throttle_max"]],
        "gimbal_limit_rad": float(np.deg2rad(config["vehicle"]["gimbal_limit_deg"])),
    }


def day26(
    root: Path, config: dict[str, Any], dataset: BCDataset, settings: TrainingSettings
) -> Path:
    directory = root / "artifacts/day26-bc-experiments"
    directory.mkdir(parents=True, exist_ok=True)
    baseline = root / "artifacts/day25-bc/bc-checkpoint.npz"
    candidates = [
        ("day25-64-relu", baseline, None, "relu", (1.0, 1.0), 1.0),
        ("96-tanh-weighted", directory / "96-tanh-weighted.npz", (96, 96), "tanh", (1.0, 2.0), 3.0),
        (
            "128-relu-weighted",
            directory / "128-relu-weighted.npz",
            (128, 128),
            "relu",
            (1.0, 2.0),
            3.0,
        ),
    ]
    rows = []
    for name, checkpoint, widths, activation, action_weights, terminal_weight in candidates:
        if widths is None:
            policy, metadata = load_checkpoint(checkpoint)
            training_settings = TrainingSettings.from_dict(metadata["training_settings"])
        else:
            training_settings = replace(
                settings,
                hidden_sizes=widths,
                max_epochs=48,
                patience=8,
                seed=settings.seed + (1 if activation == "tanh" else 2),
            )
            if not checkpoint.exists():
                physical_z = (
                    dataset.train_states[:, 1] * dataset.normalization["state"]["scale"][1]
                    + dataset.normalization["state"]["mean"][1]
                )
                weights = np.where(physical_z < 30.0, terminal_weight, 1.0).astype(np.float32)
                print(f"Day 26 training {name}", flush=True)
                policy, training = train_bc(
                    dataset,
                    training_settings,
                    activation=activation,
                    action_weights=action_weights,
                    sample_weights=weights,
                )
                metadata = checkpoint_metadata(
                    config,
                    dataset,
                    training_settings,
                    activation=activation,
                    name=f"day26-{name}",
                    parent=baseline,
                )
                metadata["action_loss_weights"] = list(action_weights)
                metadata["terminal_sample_weight"] = terminal_weight
                metadata["best_epoch"] = training["best_epoch"]
                save_checkpoint(checkpoint, policy, metadata)
            policy, metadata = load_checkpoint(checkpoint)
        score = action_error(
            policy,
            dataset.validation_states,
            dataset.validation_actions,
            dataset.normalization,
        )
        rows.append(
            {
                "name": name,
                "checkpoint": str(checkpoint.relative_to(root)),
                "sha256": digest(checkpoint),
                "activation": activation,
                "hidden_sizes": list(training_settings.hidden_sizes),
                "action_loss_weights": list(action_weights),
                "terminal_sample_weight": terminal_weight,
                "validation": score,
            }
        )
    selected = min(rows, key=lambda item: item["validation"]["normalized_mse"])
    report = {
        "day": 26,
        "selection_rule": "minimum IID validation normalized MSE",
        "candidates": rows,
        "selected": selected["name"],
        "selected_checkpoint": selected["checkpoint"],
    }
    if not (directory / "report.json").exists():
        write_json(directory / "report.json", report)
    return root / selected["checkpoint"]


def day27_28(
    root: Path,
    config: dict[str, Any],
    dataset: BCDataset,
    checkpoint: Path,
    validation_states: np.ndarray,
) -> None:
    directory = root / "artifacts/day27-28-closed-loop"
    report_path = directory / "report.json"
    if report_path.exists():
        return
    directory.mkdir(parents=True, exist_ok=True)
    policy, metadata = load_checkpoint(checkpoint)
    cohort = validation_states
    bc, episodes = evaluate_cohort(
        config, cohort, policy, metadata["normalization"], keep_episodes=True
    )
    pid, _ = evaluate_cohort(config, cohort, None)
    offline_normal = dataset.train_states
    on_policy = []
    for episode in episodes:
        states = np.asarray(
            [episode["initial_state"], *[step["state"] for step in episode["steps"]]],
            dtype=np.float64,
        )
        x, _ = normalize(states, np.zeros((len(states), 2)), dataset.normalization)
        on_policy.append(x[::5])
    drift = state_distribution_drift(offline_normal, np.concatenate(on_policy))
    latency = inference_latency_us(policy, dataset.validation_states)
    teacher_report = json.loads(
        (root / "artifacts/day21-teacher-validation/teacher-validation.json").read_text(
            encoding="utf-8"
        )
    )
    teacher_attempt_s = teacher_report["teacher"]["mean_accepted_attempt_time_s"]
    representative = next(
        (i for i, row in enumerate(bc["cases"]) if row["outcome"] != "success"), 0
    )
    save_episode_data(episodes[representative], directory / "bc-representative.json")
    save_animation(episodes[representative], directory / "bc-representative.mp4", max_frames=90)
    report = {
        "days": [27, 28],
        "split": "IID validation; held-out hard OOD test untouched",
        "checkpoint": str(checkpoint.relative_to(root)),
        "checkpoint_sha256": digest(checkpoint),
        "bc": bc,
        "pid": pid,
        "teacher_reference": {
            "accepted_trajectories": len(cohort),
            "note": "Frozen optimal-control validation trajectories, not an online policy",
        },
        "state_distribution_drift": drift,
        "failure_modes": bc["failure_counts"],
        "inference_latency": latency,
        "teacher_mean_accepted_attempt_time_s": teacher_attempt_s,
        "checkpoint_roundtrip": metadata["normalization"] == dataset.normalization,
        "student_faster_than_teacher": latency["p95_us"] * 1e-6 < teacher_attempt_s,
        "nominal_closed_loop_success": bc["successes"] > 0,
    }
    write_json(report_path, report)
    print(f"Day 27-28 BC {bc['success_rate']:.1%}; PID {pid['success_rate']:.1%}", flush=True)


def collect_dagger_candidates(
    config: dict[str, Any],
    states: np.ndarray,
    policy: Any,
    normalization: dict[str, Any],
    iteration: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    rng = np.random.default_rng(20260929 + iteration)
    candidates = []
    episodes = []
    for index, initial in enumerate(states[:16]):
        modified = initial.copy()
        if iteration == 2:
            modified[0] += rng.choice([-1.0, 1.0]) * 5.0
            modified[2] += rng.choice([-1.0, 1.0]) * 1.0
        wind = 0.8 if iteration == 3 and index % 2 == 0 else 0.0
        scale = 0.95 if iteration == 3 and index % 2 else 1.0
        episode, visited = run_episode(
            config,
            modified,
            policy,
            normalization,
            wind_m_s=wind,
            thrust_scale=scale,
            collect=True,
        )
        episodes.append(
            {
                "index": index,
                "wind_m_s": wind,
                "thrust_scale": scale,
                **episode_metrics(episode, config),
            }
        )
        possible = visited[
            (visited[:, 1] >= 25.0)
            & (visited[:, 1] <= 110.0)
            & (np.abs(visited[:, 4]) <= np.deg2rad(15.0))
            & (np.abs(visited[:, 5]) <= np.deg2rad(25.0))
            & (visited[:, 6] > config["vehicle"]["dry_mass_kg"] + 5.0)
        ]
        if len(possible):
            # Cover early drift and the last feasible part of failed trajectories.
            locations = np.unique(
                np.linspace(0, len(possible) - 1, min(3, len(possible)), dtype=int)
            )
            for location in locations:
                candidates.append(possible[location])
    return np.asarray(candidates, dtype=np.float64).reshape(-1, 7), episodes


def day29_32(
    root: Path,
    config: dict[str, Any],
    dataset: BCDataset,
    baseline_path: Path,
    train_initial_states: np.ndarray,
    validation_states: np.ndarray,
) -> Path:
    directory = root / "artifacts/day29-32-dagger"
    directory.mkdir(parents=True, exist_ok=True)
    baseline, base_meta = load_checkpoint(baseline_path)
    policy = baseline
    parent = baseline_path
    all_states = []
    all_actions = []
    iteration_summaries = []
    development = validation_states[:32]
    train_indices = np.random.default_rng(20260929).permutation(len(train_initial_states))
    base_evaluation, _ = evaluate_cohort(config, development, baseline, dataset.normalization)
    for iteration in (1, 2, 3):
        file = directory / f"iteration-{iteration}.npz"
        report_path = directory / f"iteration-{iteration}.json"
        checkpoint = directory / f"dagger-v{iteration + 1}.npz"
        if file.exists() and report_path.exists() and checkpoint.exists():
            with np.load(file, allow_pickle=False) as archive:
                accepted_states = archive["states"].copy()
                accepted_actions = archive["actions"].copy()
            summary = json.loads(report_path.read_text(encoding="utf-8"))
        else:
            pool = train_initial_states[train_indices[(iteration - 1) * 24 : iteration * 24]]
            candidates, episodes = collect_dagger_candidates(
                config,
                pool,
                policy,
                dataset.normalization,
                iteration,
            )
            if len(candidates) == 0:
                raise RuntimeError(f"Day {28 + iteration} collected no feasible student states")
            print(
                f"DAgger iteration {iteration}: relabel {len(candidates)} visited states",
                flush=True,
            )
            started = perf_counter()
            solved = solve_teacher_batch(
                config,
                candidates,
                attempt_timeout_s=12.0,
                max_retries=0,
                retry_initial_guess="pid",
                worker_restart_after_attempts=20,
                replay_dt_s=0.02,
                progress=lambda done, total, case: (
                    print(f"  Teacher {done}/{total}: {case['status']}", flush=True)
                    if done % 8 == 0 or done == total
                    else None
                ),
            )
            accepted_states = candidates[list(solved.successful_case_indices)]
            accepted_actions = np.asarray(
                [solution.controls[0] for solution in solved.solutions], dtype=np.float64
            ).reshape(-1, 2)
            if not len(accepted_states):
                raise RuntimeError(
                    f"DAgger iteration {iteration} produced no accepted Teacher labels"
                )
            with file.open("xb") as stream:
                np.savez_compressed(
                    stream,
                    states=accepted_states,
                    actions=accepted_actions,
                    candidate_states=candidates,
                )
            summary = {
                "iteration": iteration,
                "collection": "student-visited states with genuine re-solved OCP Teacher labels",
                "candidate_count": len(candidates),
                "accepted_labels": len(accepted_states),
                "teacher_failures": len(candidates) - len(accepted_states),
                "teacher_wall_time_s": solved.wall_time_s,
                "elapsed_collection_s": perf_counter() - started,
                "rollout_failures": dict(Counter(row["failure_type"] for row in episodes)),
                "rollouts": episodes,
                "teacher_cases": list(solved.cases),
            }
        all_states.append(accepted_states)
        all_actions.append(accepted_actions)
        if not checkpoint.exists():
            new_x, new_y = normalize(
                np.concatenate(all_states),
                np.concatenate(all_actions),
                dataset.normalization,
            )
            # Replay new on-policy rows so they make up about 5% of training batches.
            repeat = max(1, min(100, int(0.05 * len(dataset.train_states) / len(new_x))))
            mixed = replace(
                dataset,
                train_states=np.concatenate((dataset.train_states, np.tile(new_x, (repeat, 1)))),
                train_actions=np.concatenate((dataset.train_actions, np.tile(new_y, (repeat, 1)))),
            )
            settings = TrainingSettings.from_dict(base_meta["training_settings"])
            settings = replace(
                settings,
                max_epochs=30,
                patience=7,
                learning_rate=2e-4,
                seed=settings.seed + 10 + iteration,
            )
            policy, training = train_bc(
                mixed,
                settings,
                activation=policy.activation,
                initial_policy=policy,
            )
            metadata = checkpoint_metadata(
                config,
                dataset,
                settings,
                activation=policy.activation,
                name=f"dagger-v{iteration + 1}",
                parent=parent,
            )
            metadata.update(
                {
                    "iteration": iteration,
                    "accepted_on_policy_labels": len(new_x),
                    "replay_repeat": repeat,
                    "best_epoch": training["best_epoch"],
                }
            )
            save_checkpoint(checkpoint, policy, metadata)
            summary["training"] = {
                "settings": settings.as_dict(),
                "best_epoch": training["best_epoch"],
                "best_validation": training["best_validation"],
                "old_to_new_rows": [len(dataset.train_states), repeat * len(new_x)],
            }
            print(f"DAgger iteration {iteration}: best epoch {training['best_epoch']}", flush=True)
        else:
            policy, _ = load_checkpoint(checkpoint)
        evaluation, _ = evaluate_cohort(config, development, policy, dataset.normalization)
        summary["development_evaluation"] = evaluation
        if not report_path.exists():
            write_json(report_path, summary)
        iteration_summaries.append(
            {
                "iteration": iteration,
                "checkpoint": str(checkpoint.relative_to(root)),
                "accepted_labels": summary["accepted_labels"],
                "success_rate": evaluation["success_rate"],
            }
        )
        parent = checkpoint
    selected = max(
        iteration_summaries,
        key=lambda row: (row["success_rate"], -row["iteration"]),
    )
    overview = {
        "days": [29, 30, 31, 32],
        "development_split": "first 32 IID validation initial states; collection uses train only",
        "bc_development_success_rate": base_evaluation["success_rate"],
        "iterations": iteration_summaries,
        "selected": selected,
        "total_accepted_labels": sum(row["accepted_labels"] for row in iteration_summaries),
        "wind_and_thrust_error_in_iteration_3": True,
    }
    if not (directory / "report.json").exists():
        write_json(directory / "report.json", overview)
    return root / selected["checkpoint"]


def day33_34(
    root: Path,
    config: dict[str, Any],
    dataset: BCDataset,
    dagger_path: Path,
    validation_states: np.ndarray,
) -> tuple[Path, ActuatorSettings]:
    directory = root / "artifacts/day33-34-stabilization"
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / "report.json"
    if report_path.exists():
        saved = json.loads(report_path.read_text(encoding="utf-8"))
        if saved["checkpoint"] != str(dagger_path.relative_to(root)):
            raise ValueError("stabilization report checkpoint does not match")
        selected = (
            ActuatorSettings(2.0, float(np.deg2rad(60.0)))
            if saved["selected_rate_limit"]
            else ActuatorSettings()
        )
        return dagger_path, selected
    policy, metadata = load_checkpoint(dagger_path)
    cohort = validation_states[32:64]
    raw, _ = evaluate_cohort(config, cohort, policy, dataset.normalization)
    actuator = ActuatorSettings(
        throttle_rate_per_s=2.0,
        gimbal_rate_rad_s=float(np.deg2rad(60.0)),
    )
    filtered, _ = evaluate_cohort(
        config,
        cohort,
        policy,
        dataset.normalization,
        actuator=actuator,
    )
    sensor_noise = ActuatorSettings(
        actuator.throttle_rate_per_s,
        actuator.gimbal_rate_rad_s,
        (0.1, 0.1, 0.05, 0.05, np.deg2rad(0.15), np.deg2rad(0.1), 0.1),
        20261003,
    )
    noisy, _ = evaluate_cohort(
        config,
        cohort,
        policy,
        dataset.normalization,
        actuator=sensor_noise,
    )
    stressed, _ = evaluate_cohort(
        config,
        cohort,
        policy,
        dataset.normalization,
        wind_m_s=0.8,
        thrust_scale=0.95,
        actuator=actuator,
    )
    residual_candidates = []
    for pid_mix in (0.25, 0.5, 0.75):
        result, _ = evaluate_cohort(
            config,
            cohort,
            policy,
            dataset.normalization,
            pid_mix=pid_mix,
            actuator=actuator,
        )
        residual_candidates.append(
            {
                "pid_mix": pid_mix,
                "success_rate": result["success_rate"],
                "successes": result["successes"],
                "mean_fuel_used_kg": result["mean_fuel_used_kg"],
            }
        )
    residual_selected = max(
        residual_candidates,
        key=lambda item: (item["success_rate"], -item["mean_fuel_used_kg"]),
    )
    residual_accepted = residual_selected["success_rate"] > filtered["success_rate"]
    use_filter = filtered["success_rate"] >= raw["success_rate"]
    report = {
        "days": [33, 34],
        "checkpoint": str(dagger_path.relative_to(root)),
        "raw": raw,
        "rate_limited": filtered,
        "rate_limited_with_sensor_noise": noisy,
        "rate_limited_with_wind_and_thrust_error": stressed,
        "actuator_limit": {"throttle_per_s": 2.0, "gimbal_deg_s": 60.0},
        "selected_rate_limit": use_filter,
        "residual_policy": {
            "definition": "blend of DAgger and frozen integrated PID actions",
            "validation_candidates": residual_candidates,
            "status": "accepted" if residual_accepted else "rejected_no_success_gain",
            "selected": residual_selected if residual_accepted else None,
            "note": "Hybrid uses PID online; do not count it as standalone learned-policy success",
        },
        "ppo": {
            "status": "not_run",
            "reason": "optional Day 34 branch; residual feasibility test showed no gain",
        },
    }
    if not report_path.exists():
        write_json(report_path, report)
    return dagger_path, actuator if use_filter else ActuatorSettings()


def day33_sensor_augmentation(
    root: Path,
    config: dict[str, Any],
    dataset: BCDataset,
    dagger_path: Path,
    validation_states: np.ndarray,
) -> None:
    """Train a small sensor-noise candidate without changing the frozen DAgger choice."""
    directory = root / "artifacts/day33-34-stabilization"
    checkpoint = directory / "sensor-augmented.npz"
    report_path = directory / "sensor-augmentation.json"
    if report_path.exists():
        if not checkpoint.exists():
            raise ValueError("sensor-augmentation report has no checkpoint")
        return
    policy, parent_metadata = load_checkpoint(dagger_path)
    noise_physical = np.asarray(
        (0.1, 0.1, 0.05, 0.05, np.deg2rad(0.15), np.deg2rad(0.1), 0.1),
        dtype=np.float64,
    )
    noise_normalized = noise_physical / np.asarray(dataset.normalization["state"]["scale"])
    rng = np.random.default_rng(20261003)
    rows = rng.choice(len(dataset.train_states), 20000, replace=False)
    noisy_states = dataset.train_states[rows] + rng.normal(
        0.0, noise_normalized, size=(len(rows), 7)
    ).astype(np.float32)
    mixed = replace(
        dataset,
        train_states=np.concatenate((dataset.train_states, noisy_states)),
        train_actions=np.concatenate((dataset.train_actions, dataset.train_actions[rows])),
    )
    settings = TrainingSettings.from_dict(parent_metadata["training_settings"])
    settings = replace(
        settings,
        learning_rate=1e-4,
        max_epochs=12,
        patience=3,
        seed=20261003,
    )
    if not checkpoint.exists():
        candidate, training = train_bc(
            mixed,
            settings,
            activation=policy.activation,
            initial_policy=policy,
        )
        metadata = checkpoint_metadata(
            config,
            dataset,
            settings,
            activation=policy.activation,
            name="dagger-sensor-augmented-exploratory",
            parent=dagger_path,
        )
        metadata.update(
            {
                "noise_physical_std": noise_physical.tolist(),
                "augmented_train_rows": len(rows),
                "best_epoch": training["best_epoch"],
                "not_selected_for_frozen_hard_test": True,
            }
        )
        save_checkpoint(checkpoint, candidate, metadata)
    candidate, _ = load_checkpoint(checkpoint)
    cohort = validation_states[32:64]
    actuator = ActuatorSettings(2.0, float(np.deg2rad(60.0)))
    noisy_actuator = ActuatorSettings(
        actuator.throttle_rate_per_s,
        actuator.gimbal_rate_rad_s,
        tuple(noise_physical.tolist()),
        20261003,
    )
    candidate_clean, _ = evaluate_cohort(
        config,
        cohort,
        candidate,
        dataset.normalization,
        actuator=actuator,
    )
    candidate_noisy, _ = evaluate_cohort(
        config,
        cohort,
        candidate,
        dataset.normalization,
        actuator=noisy_actuator,
    )
    report = {
        "day": 33,
        "status": "exploratory; not selected for the frozen Day 35 hard test",
        "source_checkpoint_sha256": digest(dagger_path),
        "candidate_checkpoint_sha256": digest(checkpoint),
        "train_only_augmented_rows": len(rows),
        "noise_physical_std": noise_physical.tolist(),
        "validation_cohort": "IID validation indices 32:64",
        "candidate_clean": candidate_clean,
        "candidate_noisy": candidate_noisy,
    }
    write_json(report_path, report)


def day35(
    root: Path,
    config: dict[str, Any],
    dataset: BCDataset,
    bc_path: Path,
    dagger_path: Path,
    actuator: ActuatorSettings,
    test_states: np.ndarray,
) -> None:
    directory = root / "artifacts/day35-final"
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / "report.json"
    if report_path.exists():
        saved = json.loads(report_path.read_text(encoding="utf-8"))
        if saved["tested_initial_conditions"] != len(test_states):
            raise ValueError("cached final report does not match test cohort")
        for name, source in (("bc", bc_path), ("dagger", dagger_path)):
            expected = saved["frozen_checkpoints"][name]["sha256"]
            if digest(source) != expected or digest(directory / f"{name}-final.npz") != expected:
                raise ValueError("cached final checkpoint digest mismatch")
        return
    frozen = {}
    for name, source in (("bc", bc_path), ("dagger", dagger_path)):
        destination = directory / f"{name}-final.npz"
        if not destination.exists():
            with source.open("rb") as incoming, destination.open("xb") as outgoing:
                shutil.copyfileobj(incoming, outgoing)
        if digest(source) != digest(destination):
            raise RuntimeError("frozen checkpoint digest mismatch")
        frozen[name] = {"source": str(source.relative_to(root)), "sha256": digest(destination)}
    bc, _ = load_checkpoint(directory / "bc-final.npz")
    dagger, _ = load_checkpoint(directory / "dagger-final.npz")
    print(f"Day 35: final hard OOD evaluation on {len(test_states)} states", flush=True)
    bc_result, bc_episodes = evaluate_cohort(
        config,
        test_states,
        bc,
        dataset.normalization,
        keep_episodes=True,
    )
    dagger_result, dagger_episodes = evaluate_cohort(
        config,
        test_states,
        dagger,
        dataset.normalization,
        actuator=actuator,
        keep_episodes=True,
    )
    pid_result, _ = evaluate_cohort(config, test_states, None)
    cases = []
    for index, (base, improved) in enumerate(
        zip(bc_result["cases"], dagger_result["cases"], strict=True)
    ):
        if base["outcome"] != "success" and improved["outcome"] == "success":
            cases.append(index)
    improved_index = cases[0] if cases else None
    failure_index = next(
        (
            index
            for index, case in enumerate(dagger_result["cases"])
            if case["outcome"] != "success"
        ),
        None,
    )
    for name, index, episodes in (
        (
            "bc-failure",
            next(
                (i for i, case in enumerate(bc_result["cases"]) if case["outcome"] != "success"),
                None,
            ),
            bc_episodes,
        ),
        ("dagger-failure", failure_index, dagger_episodes),
        ("dagger-improvement", improved_index, dagger_episodes),
    ):
        if index is not None:
            save_episode_data(episodes[index], directory / f"{name}.json")
            save_animation(episodes[index], directory / f"{name}.mp4", max_frames=90)
    report = {
        "day": 35,
        "split": "held-out hard OOD test, evaluated after checkpoint freeze",
        "frozen_checkpoints": frozen,
        "tested_initial_conditions": len(test_states),
        "bc": bc_result,
        "dagger": dagger_result,
        "pid": pid_result,
        "dagger_minus_bc_success_rate": dagger_result["success_rate"] - bc_result["success_rate"],
        "improved_case_indices": cases,
        "inference_latency_us": {
            "bc": inference_latency_us(bc, dataset.validation_states),
            "dagger": inference_latency_us(dagger, dataset.validation_states),
        },
        "selected_actuator_rate_limit": actuator.throttle_rate_per_s is not None,
        "completion_gate": {
            "dagger_improves_bc": dagger_result["successes"] > bc_result["successes"],
            "failure_explanation_required_if_not": dagger_result["successes"]
            <= bc_result["successes"],
        },
    }
    write_json(report_path, report)
    print(
        f"Day 35: BC {bc_result['success_rate']:.1%}, "
        f"DAgger {dagger_result['success_rate']:.1%}, PID {pid_result['success_rate']:.1%}",
        flush=True,
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--through", type=int, choices=(26, 28, 32, 34, 35), default=35)
    args = parser.parse_args()
    config = load_config(root / "configs/pid-baseline-v1.yaml")
    settings = TrainingSettings.from_dict(
        yaml.safe_load((root / "configs/bc-day25.yaml").read_text(encoding="utf-8"))
    )
    data_dir = root / "artifacts/day24-final-dataset"
    dataset = load_bc_dataset(data_dir, config)
    train_initial_states = initial_states_from_shard(data_dir / "train-trajectories.npz")
    validation = initial_states_from_shard(data_dir / "validation-trajectories.npz")
    bc_path = day26(root, config, dataset, settings)
    if args.through == 26:
        return
    day27_28(root, config, dataset, bc_path, validation)
    if args.through == 28:
        return
    dagger_path = day29_32(root, config, dataset, bc_path, train_initial_states, validation)
    if args.through == 32:
        return
    dagger_path, actuator = day33_34(root, config, dataset, dagger_path, validation)
    day33_sensor_augmentation(root, config, dataset, dagger_path, validation)
    if args.through == 34:
        return
    test = initial_states_from_shard(data_dir / "test-trajectories.npz")
    day35(root, config, dataset, bc_path, dagger_path, actuator, test)


if __name__ == "__main__":
    main()
