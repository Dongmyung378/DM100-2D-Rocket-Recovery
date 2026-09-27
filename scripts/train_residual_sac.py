"""Run the fixed Day 40 SAC pilot with development-only paired evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from powered_landing_guidance.residual_landing import ResidualLandingEnv, ResidualSafetyFilter
from powered_landing_guidance.rl_protocol import (
    RLProtocol,
    load_protocol,
    run_pid_case,
    sample_cases,
)
from powered_landing_guidance.rl_training import CURRICULUM_STAGES, CurriculumResidualEnv


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)


def load_pilot_settings(path: Path, protocol: RLProtocol) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        settings = yaml.safe_load(stream)
    required = {
        "schema_version",
        "protocol_sha256",
        "seed",
        "total_timesteps",
        "stages",
        "evaluation",
        "sac",
    }
    if not isinstance(settings, dict) or set(settings) != required:
        raise ValueError("SAC pilot settings do not match schema v1")
    if settings["schema_version"] != 1 or settings["protocol_sha256"] != protocol.sha256:
        raise ValueError("SAC pilot protocol digest does not match")
    if settings["seed"] != protocol.settings["seed"] or settings["seed"] != 378:
        raise ValueError("SAC pilot seed must match the fixed root seed 378")
    total = settings["total_timesteps"]
    if type(total) is not int or total < 3:
        raise ValueError("total_timesteps must be a positive integer")
    stages = settings["stages"]
    if not isinstance(stages, list) or not stages:
        raise ValueError("SAC pilot needs training stages")
    previous = 0
    for item in stages:
        if not isinstance(item, dict) or set(item) != {"name", "end_timestep"}:
            raise ValueError("invalid training stage")
        if item["name"] not in CURRICULUM_STAGES:
            raise ValueError("unknown curriculum stage")
        end = item["end_timestep"]
        if type(end) is not int or end <= previous:
            raise ValueError("stage end_timestep must increase")
        previous = end
    if previous != total:
        raise ValueError("last stage must end at total_timesteps")
    evaluation = settings["evaluation"]
    if not isinstance(evaluation, dict) or set(evaluation) != {
        "split",
        "cases_per_group",
        "groups",
    }:
        raise ValueError("invalid evaluation settings")
    if evaluation["split"] != "development":
        raise ValueError("pilot evaluation must use development cases")
    if evaluation["groups"] != ["nominal", "initial_ood", "wind"]:
        raise ValueError("Day 40 must evaluate nominal, initial OOD, and wind")
    count = evaluation["cases_per_group"]
    maximum = protocol.settings["splits"]["development"]["episodes_per_group"]
    if type(count) is not int or not 1 <= count <= maximum:
        raise ValueError("invalid development evaluation count")
    sac = settings["sac"]
    if not isinstance(sac, dict) or set(sac) != {
        "device",
        "learning_rate",
        "buffer_size",
        "learning_starts",
        "batch_size",
        "gamma",
        "tau",
        "train_freq",
        "gradient_steps",
        "net_arch",
    }:
        raise ValueError("invalid SAC hyperparameter fields")
    if sac["device"] != "cpu" or sac["net_arch"] != [64, 64]:
        raise ValueError("Day 40 uses the fixed CPU 64x64 pilot network")
    if any(
        type(sac[key]) is not int or sac[key] <= 0
        for key in ("buffer_size", "learning_starts", "batch_size", "train_freq", "gradient_steps")
    ):
        raise ValueError("SAC integer hyperparameters must be positive")
    if (
        not all(np.isfinite(sac[key]) and sac[key] > 0 for key in ("learning_rate", "gamma", "tau"))
        or sac["gamma"] >= 1
        or sac["tau"] > 1
    ):
        raise ValueError("SAC floating hyperparameters are invalid")
    return settings


def initialize_zero_residual_actor(model: Any) -> None:
    """Make the initial deterministic SAC policy exactly the PID residual zero."""
    import torch

    with torch.no_grad():
        actor = model.policy.actor
        actor.mu.weight.zero_()
        actor.mu.bias.zero_()
        actor.log_std.weight.zero_()
        actor.log_std.bias.fill_(-3.0)


def _development_cases(protocol: RLProtocol, settings: dict[str, Any]) -> dict[str, list]:
    count = settings["evaluation"]["cases_per_group"]
    return {
        group: sample_cases(protocol, "development", group, count=count)
        for group in settings["evaluation"]["groups"]
    }


def evaluate_model(
    model: Any,
    protocol: RLProtocol,
    cases_by_group: dict[str, list],
    baseline: dict[str, dict[str, Any]],
    *,
    timestep: int,
    safety_filter: ResidualSafetyFilter | None = None,
) -> dict[str, Any]:
    groups = []
    for group, cases in cases_by_group.items():
        records = []
        for case in cases:
            env = ResidualLandingEnv(protocol, case, safety_filter=safety_filter)
            try:
                observation, _ = env.reset()
                while True:
                    action, _ = model.predict(observation, deterministic=True)
                    observation, _, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        break
                reference = baseline[case.case_id]
                records.append(
                    {
                        "case_id": case.case_id,
                        "pid_outcome": reference["outcome"],
                        "policy_outcome": info["outcome"],
                        "pid_safe_landing": reference["safe_landing"],
                        "policy_safe_landing": int(info["outcome"] == "success"),
                        "policy_fuel_used_kg": info["fuel_used_kg"],
                        "interventions": info["interventions"],
                        "control_updates": info["control_updates"],
                        "intervention_reason_counts": info["intervention_reason_counts"],
                        "intervention_axis_counts": info["intervention_axis_counts"],
                    }
                )
            finally:
                env.close()
        pid_success = sum(row["pid_safe_landing"] for row in records)
        policy_success = sum(row["policy_safe_landing"] for row in records)
        intervention_reasons: Counter[str] = Counter()
        intervention_axes: Counter[str] = Counter()
        for row in records:
            intervention_reasons.update(row["intervention_reason_counts"])
            intervention_axes.update(row["intervention_axis_counts"])
        control_updates = sum(row["control_updates"] for row in records)
        interventions = sum(row["interventions"] for row in records)
        groups.append(
            {
                "group": group,
                "episodes": len(records),
                "pid_safe_landings": pid_success,
                "policy_safe_landings": policy_success,
                "rescued_pid_failures": sum(
                    not row["pid_safe_landing"] and row["policy_safe_landing"] for row in records
                ),
                "new_failures_on_pid_success": sum(
                    row["pid_safe_landing"] and not row["policy_safe_landing"] for row in records
                ),
                "severe_crashes": sum(row["policy_outcome"] == "crash" for row in records),
                "interventions": interventions,
                "control_updates": control_updates,
                "intervention_rate": interventions / control_updates if control_updates else 0.0,
                "intervention_reason_counts": dict(sorted(intervention_reasons.items())),
                "intervention_axis_counts": dict(sorted(intervention_axes.items())),
                "failure_counts": dict(
                    sorted(
                        Counter(
                            row["policy_outcome"]
                            for row in records
                            if not row["policy_safe_landing"]
                        ).items()
                    )
                ),
                "cases": records,
            }
        )
    return {
        "protocol_sha256": protocol.sha256,
        "root_seed": protocol.settings["seed"],
        "split": "development",
        "timestep": timestep,
        "groups": groups,
        "final_holdout": "sealed_not_materialized",
    }


def _save_checkpoint(
    model: Any,
    environment: CurriculumResidualEnv,
    output: Path,
    protocol: RLProtocol,
    settings_path: Path,
    settings: dict[str, Any],
    episodes: list[dict[str, Any]],
) -> Path:
    step = int(model.num_timesteps)
    suffix = f"{step:06d}"
    model_path = output / f"model-step-{suffix}.zip"
    replay_path = output / f"replay-step-{suffix}.pkl"
    episodes_path = output / f"episodes-step-{suffix}.json"
    manifest_path = output / f"manifest-step-{suffix}.json"
    targets = (model_path, replay_path, episodes_path, manifest_path)
    if any(path.exists() for path in targets):
        raise FileExistsError(f"checkpoint step {step} already exists")
    model.save(model_path)
    if step:
        model.save_replay_buffer(replay_path)
    _write_json(episodes_path, {"episodes": episodes, "timestep": step})
    manifest = {
        "schema_version": 1,
        "protocol_sha256": protocol.sha256,
        "pilot_settings_sha256": _sha256(settings_path),
        "root_seed": settings["seed"],
        "completed_timesteps": step,
        "stage": environment.stage,
        "sampler_state": environment.sampler_state(),
        "model_file": model_path.name,
        "model_sha256": _sha256(model_path),
        "replay_file": replay_path.name if step else None,
        "replay_sha256": _sha256(replay_path) if step else None,
        "episodes_file": episodes_path.name,
    }
    _write_json(manifest_path, manifest)
    return manifest_path


def _load_latest(output: Path, protocol: RLProtocol, settings_path: Path) -> dict[str, Any]:
    manifests = sorted(output.glob("manifest-step-*.json"))
    if not manifests:
        raise FileNotFoundError("no completed Day 40 checkpoint to resume")
    latest = json.loads(manifests[-1].read_text(encoding="utf-8"))
    if latest["protocol_sha256"] != protocol.sha256:
        raise ValueError("checkpoint protocol digest changed")
    if latest["pilot_settings_sha256"] != _sha256(settings_path):
        raise ValueError("checkpoint pilot settings digest changed")
    for file_key, digest_key in (("model_file", "model_sha256"), ("replay_file", "replay_sha256")):
        name = latest[file_key]
        if name is not None and _sha256(output / name) != latest[digest_key]:
            raise ValueError(f"checkpoint {name} digest mismatch")
    return latest


def _save_learning_curve(output: Path, timestep: int) -> None:
    """Plot training episodes and separate development-group landing counts."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    episodes: list[dict[str, Any]] = []
    for path in sorted(output.glob("episodes-step-*.json")):
        if int(path.stem.rsplit("-", 1)[-1]) <= timestep:
            episodes.extend(json.loads(path.read_text(encoding="utf-8"))["episodes"])
    evaluations = []
    for path in sorted(output.glob("eval-step-*.json")):
        if int(path.stem.rsplit("-", 1)[-1]) <= timestep:
            evaluations.append(json.loads(path.read_text(encoding="utf-8")))

    figure, (training_axis, development_axis) = plt.subplots(2, 1, figsize=(9, 7))
    if episodes:
        window = min(25, len(episodes))
        steps = np.asarray([row["timestep"] for row in episodes])
        landed = np.asarray([row["outcome"] == "success" for row in episodes], dtype=float)
        rolling = np.convolve(landed, np.ones(window) / window, mode="valid")
        training_axis.plot(steps[window - 1 :], rolling, label=f"{window}-episode rolling success")
    training_axis.set(xlabel="Environment steps", ylabel="Training success fraction", ylim=(0, 1))
    training_axis.grid(alpha=0.25)
    if episodes:
        training_axis.legend()

    for group in (row["group"] for row in evaluations[0]["groups"]):
        steps = [entry["timestep"] for entry in evaluations]
        successes = [
            next(row for row in entry["groups"] if row["group"] == group)["policy_safe_landings"]
            for entry in evaluations
        ]
        development_axis.plot(steps, successes, marker="o", label=group)
    count = evaluations[0]["groups"][0]["episodes"]
    development_axis.set(
        xlabel="Environment steps",
        ylabel=f"Development landings / {count}",
        ylim=(0, count),
    )
    development_axis.grid(alpha=0.25)
    development_axis.legend()
    figure.tight_layout()
    figure.savefig(output / f"learning-curve-step-{timestep:06d}.png", dpi=140)
    plt.close(figure)


def run_training(
    protocol: RLProtocol,
    settings_path: Path,
    output: Path,
    *,
    until: int,
    resume: bool,
) -> dict[str, Any]:
    import torch
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback

    settings = load_pilot_settings(settings_path, protocol)
    if not 1 <= until <= settings["total_timesteps"]:
        raise ValueError("--until must be within the fixed training budget")
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=True)
    if not resume and any(output.iterdir()):
        raise FileExistsError(f"Day 40 output directory is not empty: {output}")
    cases_by_group = _development_cases(protocol, settings)
    baseline_path = output / "pid-development-baseline.json"
    if resume:
        latest = _load_latest(output, protocol, settings_path)
        baseline_data = json.loads(baseline_path.read_text(encoding="utf-8"))
        baseline = {row["case_id"]: row for row in baseline_data["cases"]}
    else:
        baseline_rows = [
            run_pid_case(protocol, case) for cases in cases_by_group.values() for case in cases
        ]
        baseline = {row["case_id"]: row for row in baseline_rows}
        _write_json(
            baseline_path,
            {"protocol_sha256": protocol.sha256, "split": "development", "cases": baseline_rows},
        )

    environment = CurriculumResidualEnv(protocol)

    class EpisodeCurve(BaseCallback):
        def __init__(self) -> None:
            super().__init__()
            self.episodes: list[dict[str, Any]] = []

        def _on_step(self) -> bool:
            for done, info in zip(self.locals["dones"], self.locals["infos"], strict=True):
                if done:
                    episode = info.get("episode", {})
                    self.episodes.append(
                        {
                            "timestep": int(self.num_timesteps),
                            "stage": info["curriculum_stage"],
                            "group": info["group"],
                            "case_id": info["case_id"],
                            "outcome": info["outcome"],
                            "return": float(episode.get("r", 0.0)),
                            "length": int(episode.get("l", 0)),
                            "interventions": info["interventions"],
                        }
                    )
            if self.num_timesteps % 500 == 0:
                print(f"SAC pilot: {self.num_timesteps} environment steps", flush=True)
            return True

    try:
        sac = settings["sac"]
        if resume:
            model = SAC.load(output / latest["model_file"], env=environment, device="cpu")
            if latest["replay_file"] is not None:
                model.load_replay_buffer(output / latest["replay_file"])
            environment.restore_sampler_state(latest["sampler_state"])
            if model.num_timesteps != latest["completed_timesteps"]:
                raise ValueError("checkpoint timestep mismatch")
        else:
            model = SAC(
                "MlpPolicy",
                environment,
                seed=settings["seed"],
                device=sac["device"],
                learning_rate=sac["learning_rate"],
                buffer_size=sac["buffer_size"],
                learning_starts=sac["learning_starts"],
                batch_size=sac["batch_size"],
                gamma=sac["gamma"],
                tau=sac["tau"],
                train_freq=sac["train_freq"],
                gradient_steps=sac["gradient_steps"],
                policy_kwargs={"net_arch": sac["net_arch"]},
                verbose=0,
            )
            initialize_zero_residual_actor(model)
            first_case = cases_by_group["nominal"][0]
            check_env = ResidualLandingEnv(protocol, first_case)
            try:
                initial_observation, _ = check_env.reset()
                initial_action, _ = model.predict(initial_observation, deterministic=True)
                if not np.array_equal(initial_action, np.zeros(2)):
                    raise RuntimeError("SAC actor is not zero-residual at initialization")
            finally:
                check_env.close()
            initial_eval = evaluate_model(model, protocol, cases_by_group, baseline, timestep=0)
            if any(
                entry["pid_safe_landings"] != entry["policy_safe_landings"]
                or entry["rescued_pid_failures"]
                or entry["new_failures_on_pid_success"]
                for entry in initial_eval["groups"]
            ):
                raise RuntimeError("zero-initialized SAC actor does not reproduce PID outcomes")
            _write_json(output / "eval-step-000000.json", initial_eval)
            _save_checkpoint(model, environment, output, protocol, settings_path, settings, [])
            _save_learning_curve(output, 0)

        for stage in settings["stages"]:
            completed = int(model.num_timesteps)
            if completed >= until:
                break
            boundary = stage["end_timestep"]
            if completed >= boundary:
                continue
            environment.set_stage(stage["name"])
            target = min(until, boundary)
            curve = EpisodeCurve()
            model.learn(target - completed, callback=curve, reset_num_timesteps=False)
            if model.num_timesteps != target:
                raise RuntimeError("SAC exceeded the fixed timestep budget")
            evaluation = evaluate_model(model, protocol, cases_by_group, baseline, timestep=target)
            _write_json(output / f"eval-step-{target:06d}.json", evaluation)
            manifest_path = _save_checkpoint(
                model, environment, output, protocol, settings_path, settings, curve.episodes
            )
            _save_learning_curve(output, target)
            print(
                f"saved {target} steps and {len(curve.episodes)} episodes to {manifest_path}",
                flush=True,
            )
        return {"completed_timesteps": int(model.num_timesteps), "output_dir": str(output)}
    finally:
        environment.close()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=root / "configs/rl-evaluation-v1.yaml")
    parser.add_argument("--settings", type=Path, default=root / "configs/sac-pilot-v1.yaml")
    parser.add_argument("--output-dir", type=Path, default=root / "artifacts/day40-sac-pilot")
    parser.add_argument("--until", type=int, default=6000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    result = run_training(
        protocol,
        args.settings.resolve(),
        args.output_dir.resolve(),
        until=args.until,
        resume=args.resume,
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
