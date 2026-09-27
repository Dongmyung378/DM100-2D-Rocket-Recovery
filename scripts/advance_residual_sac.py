"""Run multi-seed residual SAC experiments on development cases only."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from powered_landing_guidance.residual_landing import ResidualLandingEnv, ResidualSafetyFilter
from powered_landing_guidance.rl_protocol import load_protocol, run_pid_case, sample_cases
from powered_landing_guidance.rl_training import CURRICULUM_STAGES, CurriculumResidualEnv
from scripts.train_residual_sac import (
    _load_latest,
    _save_checkpoint,
    _save_learning_curve,
    _sha256,
    _write_json,
    evaluate_model,
    initialize_zero_residual_actor,
)

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ["nominal", "initial_ood", "wind", "model_mismatch", "sensor_delay", "combined"]
FILTER_FIELDS = {
    "throttle_limit",
    "gimbal_limit_deg",
    "fuel_reserve_kg",
    "attitude_limit_deg",
    "ground_margin_m",
}


def load_settings(path: Path, protocol: Any) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        settings = yaml.safe_load(stream)
    expected = {
        "schema_version",
        "protocol_sha256",
        "seed",
        "seed_count",
        "total_timesteps",
        "source",
        "stages",
        "evaluation",
        "variants",
        "sac",
    }
    if not isinstance(settings, dict) or set(settings) != expected:
        raise ValueError("invalid multi-seed SAC configuration")
    if settings["schema_version"] != 1 or settings["protocol_sha256"] != protocol.sha256:
        raise ValueError("protocol digest changed")
    if settings["seed"] != 378 or type(settings["seed_count"]) is not int:
        raise ValueError("root seed must be 378 and seed_count must be an integer")
    if settings["seed_count"] < 1:
        raise ValueError("seed_count must be positive")
    source = settings["source"]
    source_step = 0
    if source is not None:
        if not isinstance(source, dict) or set(source) != {"root_dir", "settings", "step"}:
            raise ValueError("invalid source checkpoint")
        source_step = source["step"]
        if type(source_step) is not int or source_step < 1:
            raise ValueError("source step must be positive")
    stages = settings["stages"]
    if not isinstance(stages, list) or not stages:
        raise ValueError("training stages are required")
    previous = source_step
    for stage in stages:
        if not isinstance(stage, dict) or set(stage) != {"name", "end_timestep"}:
            raise ValueError("invalid training stage")
        if stage["name"] not in CURRICULUM_STAGES:
            raise ValueError("unknown curriculum stage")
        end = stage["end_timestep"]
        if type(end) is not int or end <= previous:
            raise ValueError("stage boundaries must increase after the source step")
        previous = end
    if previous != settings["total_timesteps"]:
        raise ValueError("last stage must end at total_timesteps")
    evaluation = settings["evaluation"]
    if not isinstance(evaluation, dict) or set(evaluation) != {
        "split",
        "cases_per_group",
        "groups",
    }:
        raise ValueError("invalid evaluation settings")
    if evaluation["split"] != "development" or evaluation["groups"] != GROUPS:
        raise ValueError("only the six development groups may be evaluated")
    count = evaluation["cases_per_group"]
    maximum = protocol.settings["splits"]["development"]["episodes_per_group"]
    if type(count) is not int or not 1 <= count <= maximum:
        raise ValueError("invalid development case count")
    variants = settings["variants"]
    if not isinstance(variants, list) or not variants:
        raise ValueError("at least one safety-filter variant is required")
    names = []
    for variant in variants:
        if not isinstance(variant, dict) or set(variant) != {"name", "safety_filter"}:
            raise ValueError("invalid safety-filter variant")
        name = variant["name"]
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError("invalid variant name")
        if set(variant["safety_filter"]) != FILTER_FIELDS:
            raise ValueError("invalid safety-filter fields")
        make_filter(variant["safety_filter"])
        names.append(name)
    if len(set(names)) != len(names):
        raise ValueError("duplicate safety-filter variant")
    if source is not None:
        source_settings = yaml.safe_load((ROOT / source["settings"]).read_text(encoding="utf-8"))
        source_filter = source_settings["variants"][0]["safety_filter"]
        if variants[0]["safety_filter"] != source_filter:
            raise ValueError("control filter must match the source checkpoint")
        if len(variants) == 2:
            changed = [
                key
                for key in FILTER_FIELDS
                if variants[0]["safety_filter"][key] != variants[1]["safety_filter"][key]
            ]
            if len(changed) != 1:
                raise ValueError("paired variants must change exactly one filter parameter")
    sac = settings["sac"]
    expected_sac = {
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
    }
    if not isinstance(sac, dict) or set(sac) != expected_sac:
        raise ValueError("invalid SAC hyperparameters")
    if sac["device"] != "cpu" or sac["net_arch"] != [64, 64]:
        raise ValueError("this experiment uses CPU and a 64x64 network")
    for field in ("buffer_size", "learning_starts", "batch_size", "train_freq", "gradient_steps"):
        if type(sac[field]) is not int or sac[field] < 1:
            raise ValueError(f"invalid {field}")
    for field in ("learning_rate", "gamma", "tau"):
        if not np.isfinite(sac[field]) or sac[field] <= 0:
            raise ValueError(f"invalid {field}")
    if sac["gamma"] >= 1 or sac["tau"] > 1:
        raise ValueError("invalid gamma or tau")
    return settings


def make_filter(values: dict[str, float]) -> ResidualSafetyFilter:
    return ResidualSafetyFilter(
        throttle_limit=values["throttle_limit"],
        gimbal_limit_rad=np.deg2rad(values["gimbal_limit_deg"]),
        fuel_reserve_kg=values["fuel_reserve_kg"],
        attitude_limit_rad=np.deg2rad(values["attitude_limit_deg"]),
        ground_margin_m=values["ground_margin_m"],
    )


def derived_seeds(root_seed: int, count: int) -> list[int]:
    return [
        int(child.generate_state(1)[0]) for child in np.random.SeedSequence(root_seed).spawn(count)
    ]


def _case_baseline(
    output: Path, protocol: Any, cases_by_group: dict[str, list], *, resume: bool
) -> dict[str, dict[str, Any]]:
    path = output / "pid-development-baseline.json"
    if path.exists():
        if not resume:
            raise FileExistsError(path)
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["protocol_sha256"] != protocol.sha256:
            raise ValueError("PID baseline protocol changed")
        return {row["case_id"]: row for row in report["cases"]}
    rows = [run_pid_case(protocol, case) for cases in cases_by_group.values() for case in cases]
    _write_json(path, {"protocol_sha256": protocol.sha256, "cases": rows})
    return {row["case_id"]: row for row in rows}


def _source_checkpoint(source: dict[str, Any], index: int, protocol: Any) -> tuple[Path, dict]:
    source_dir = ROOT / source["root_dir"] / f"seed-{index}"
    settings_path = ROOT / source["settings"]
    manifest = _load_latest(source_dir, protocol, settings_path)
    if manifest["completed_timesteps"] != source["step"]:
        raise ValueError("source checkpoint does not match the frozen step")
    return source_dir, manifest


def _run_seed(
    protocol: Any,
    settings_path: Path,
    settings: dict[str, Any],
    output: Path,
    cases_by_group: dict[str, list],
    baseline: dict[str, dict[str, Any]],
    *,
    index: int,
    seed: int,
    variant: dict[str, Any],
    until: int,
    resume: bool,
) -> None:
    import torch
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback

    safety_filter = make_filter(variant["safety_filter"])
    environment = CurriculumResidualEnv(protocol, safety_filter=safety_filter)
    output.mkdir(parents=True, exist_ok=True)
    metadata = output / "run.json"
    if metadata.exists():
        if not resume:
            raise FileExistsError(f"run exists: {output}")
        record = json.loads(metadata.read_text(encoding="utf-8"))
        if record != {
            "config_sha256": _sha256(settings_path),
            "seed_index": index,
            "derived_seed": seed,
            "variant": variant["name"],
        }:
            raise ValueError("run metadata changed")
    elif any(output.iterdir()):
        raise FileExistsError(f"unrecognized output files: {output}")
    else:
        _write_json(
            metadata,
            {
                "config_sha256": _sha256(settings_path),
                "seed_index": index,
                "derived_seed": seed,
                "variant": variant["name"],
            },
        )

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
            return True

    try:
        torch.set_num_threads(1)
        if list(output.glob("manifest-step-*.json")):
            latest = _load_latest(output, protocol, settings_path)
            model = SAC.load(output / latest["model_file"], env=environment, device="cpu")
            if latest["replay_file"] is not None:
                model.load_replay_buffer(output / latest["replay_file"])
            environment.restore_sampler_state(latest["sampler_state"])
            if model.num_timesteps != latest["completed_timesteps"]:
                raise ValueError("checkpoint timestep mismatch")
        elif settings["source"] is not None:
            source_dir, source = _source_checkpoint(settings["source"], index, protocol)
            model = SAC.load(source_dir / source["model_file"], env=environment, device="cpu")
            model.load_replay_buffer(source_dir / source["replay_file"])
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            environment.restore_sampler_state(source["sampler_state"])
        else:
            sac = settings["sac"]
            model = SAC(
                "MlpPolicy",
                environment,
                seed=seed,
                device="cpu",
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
            check = ResidualLandingEnv(
                protocol, cases_by_group["nominal"][0], safety_filter=safety_filter
            )
            try:
                observation, _ = check.reset()
                initial_action, _ = model.predict(observation, deterministic=True)
                if not np.array_equal(initial_action, np.zeros(2)):
                    raise RuntimeError("initial SAC actor is not zero residual")
            finally:
                check.close()
        start = int(model.num_timesteps)
        expected_start = settings["source"]["step"] if settings["source"] else 0
        if not list(output.glob("manifest-step-*.json")) and start != expected_start:
            raise ValueError("unexpected initial model step")
        if not (output / f"eval-step-{start:06d}.json").exists():
            initial = evaluate_model(
                model,
                protocol,
                cases_by_group,
                baseline,
                timestep=start,
                safety_filter=safety_filter,
            )
            if settings["source"] is None and any(
                row["pid_safe_landings"] != row["policy_safe_landings"] for row in initial["groups"]
            ):
                raise RuntimeError("zero-residual actor did not reproduce PID")
            _write_json(output / f"eval-step-{start:06d}.json", initial)
            _save_checkpoint(model, environment, output, protocol, settings_path, settings, [])
            _save_learning_curve(output, start)
        for stage in settings["stages"]:
            completed = int(model.num_timesteps)
            if completed >= until:
                break
            if completed >= stage["end_timestep"]:
                continue
            environment.set_stage(stage["name"])
            target = min(until, stage["end_timestep"])
            curve = EpisodeCurve()
            model.learn(target - completed, callback=curve, reset_num_timesteps=False)
            if model.num_timesteps != target:
                raise RuntimeError("SAC exceeded the fixed environment-step budget")
            evaluation = evaluate_model(
                model,
                protocol,
                cases_by_group,
                baseline,
                timestep=target,
                safety_filter=safety_filter,
            )
            _write_json(output / f"eval-step-{target:06d}.json", evaluation)
            _save_checkpoint(
                model, environment, output, protocol, settings_path, settings, curve.episodes
            )
            _save_learning_curve(output, target)
            counts = ", ".join(
                f"{row['group']}={row['policy_safe_landings']}/{row['episodes']}"
                for row in evaluation["groups"]
            )
            print(f"{variant['name']} seed-{index} step {target}: {counts}", flush=True)
    finally:
        environment.close()


def _write_summary(output: Path, settings: dict[str, Any], seeds: list[int]) -> None:
    rows = []
    target = settings["total_timesteps"]
    for variant in settings["variants"]:
        for index, seed in enumerate(seeds):
            path = output / variant["name"] / f"seed-{index}" / f"eval-step-{target:06d}.json"
            if not path.exists():
                return
            evaluation = json.loads(path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "variant": variant["name"],
                    "seed_index": index,
                    "derived_seed": seed,
                    "groups": [
                        {
                            key: group[key]
                            for key in (
                                "group",
                                "episodes",
                                "pid_safe_landings",
                                "policy_safe_landings",
                                "rescued_pid_failures",
                                "new_failures_on_pid_success",
                                "severe_crashes",
                                "intervention_rate",
                                "intervention_reason_counts",
                                "intervention_axis_counts",
                            )
                        }
                        for group in evaluation["groups"]
                    ],
                }
            )
    path = output / "summary.json"
    if not path.exists():
        _write_json(
            path,
            {
                "root_seed": settings["seed"],
                "derived_seeds": seeds,
                "split": "development",
                "final_holdout": "sealed_not_materialized",
                "completed_timesteps_per_run": target,
                "runs": rows,
            },
        )


def run_experiment(
    protocol_path: Path,
    settings_path: Path,
    output: Path,
    *,
    until: int | None = None,
    resume: bool = False,
    seed_index: int | None = None,
) -> None:
    protocol = load_protocol(protocol_path)
    settings = load_settings(settings_path, protocol)
    source_step = settings["source"]["step"] if settings["source"] else 0
    target = settings["total_timesteps"] if until is None else until
    if type(target) is not int or not source_step < target <= settings["total_timesteps"]:
        raise ValueError("--until must be after the source and inside the fixed budget")
    if seed_index is not None and not 0 <= seed_index < settings["seed_count"]:
        raise ValueError("invalid seed index")
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError(f"experiment output exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    cases_by_group = {
        group: sample_cases(
            protocol, "development", group, count=settings["evaluation"]["cases_per_group"]
        )
        for group in settings["evaluation"]["groups"]
    }
    baseline = _case_baseline(output, protocol, cases_by_group, resume=resume)
    seeds = derived_seeds(settings["seed"], settings["seed_count"])
    indices = range(len(seeds)) if seed_index is None else [seed_index]
    for variant in settings["variants"]:
        for index in indices:
            _run_seed(
                protocol,
                settings_path,
                settings,
                output / variant["name"] / f"seed-{index}",
                cases_by_group,
                baseline,
                index=index,
                seed=seeds[index],
                variant=variant,
                until=target,
                resume=resume,
            )
    _write_summary(output, settings, seeds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/rl-evaluation-v1.yaml")
    parser.add_argument("--settings", type=Path, default=ROOT / "configs/sac-day41-v1.yaml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/day41-sac")
    parser.add_argument("--until", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed-index", type=int)
    args = parser.parse_args()
    run_experiment(
        args.protocol.resolve(),
        args.settings.resolve(),
        args.output_dir.resolve(),
        until=args.until,
        resume=args.resume,
        seed_index=args.seed_index,
    )


if __name__ == "__main__":
    main()
