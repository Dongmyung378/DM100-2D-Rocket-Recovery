"""Measure Day 38 fallback frequency on small, untrained residual probes."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from powered_landing_guidance.residual_landing import ResidualLandingEnv
from powered_landing_guidance.rl_protocol import GROUPS, RLProtocol, load_protocol, sample_cases


def run_probe(protocol: RLProtocol, cases_per_group: int, amplitude: float) -> dict:
    """Use training cases only; the random residual is not a trained policy."""
    if not 1 <= cases_per_group <= protocol.settings["splits"]["train"]["episodes_per_group"]:
        raise ValueError("cases_per_group exceeds the training split")
    if not np.isfinite(amplitude) or not 0.0 < amplitude <= 1.0:
        raise ValueError("amplitude must be in (0, 1]")
    rng = np.random.default_rng(protocol.settings["seed"])
    summaries = []
    for group in GROUPS:
        case_ids = []
        decisions = interventions = safe_landings = 0
        reasons: Counter[str] = Counter()
        times_us = []
        for case in sample_cases(protocol, "train", group, count=cases_per_group):
            case_ids.append(case.case_id)
            env = ResidualLandingEnv(protocol, case)
            try:
                env.reset()
                while True:
                    action = rng.uniform(-amplitude, amplitude, size=2)
                    _, _, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        break
                decisions += info["control_updates"]
                interventions += info["interventions"]
                safe_landings += int(info["outcome"] == "success")
                reasons.update(info["intervention_reason_counts"])
                times_us.extend(env.safety_times_us)
            finally:
                env.close()
        summaries.append(
            {
                "group": group,
                "case_ids": case_ids,
                "episodes": cases_per_group,
                "safe_landings_random_probe": safe_landings,
                "control_updates": decisions,
                "interventions": interventions,
                "intervention_rate": interventions / decisions,
                "intervention_reasons": dict(sorted(reasons.items())),
                "safety_mean_us": float(np.mean(times_us)),
                "safety_p95_us": float(np.percentile(times_us, 95)),
            }
        )
    return {
        "protocol_sha256": protocol.sha256,
        "root_seed": protocol.settings["seed"],
        "split": "train",
        "probe": "uniform_random_residual_not_trained_policy",
        "normalized_action_amplitude": amplitude,
        "cases_per_group": cases_per_group,
        "groups": summaries,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=root / "configs/rl-evaluation-v1.yaml")
    parser.add_argument("--output-dir", type=Path, default=root / "artifacts/day38-residual-check")
    parser.add_argument("--cases-per-group", type=int, default=3)
    parser.add_argument("--amplitude", type=float, default=0.25)
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    report = run_probe(protocol, args.cases_per_group, args.amplitude)
    output = args.output_dir.resolve()
    result_path = output / "random-residual-probe.json"
    if result_path.exists():
        raise FileExistsError(f"Day 38 output already exists: {result_path}")
    output.mkdir(parents=True, exist_ok=True)
    with result_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
    for entry in report["groups"]:
        print(
            f"{entry['group']}: intervention {entry['interventions']}/"
            f"{entry['control_updates']}, safety p95 {entry['safety_p95_us']:.1f} us"
        )
    print(f"saved Day 38 diagnostic report to {result_path}")


if __name__ == "__main__":
    main()
