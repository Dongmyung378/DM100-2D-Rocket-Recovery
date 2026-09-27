"""Supervised policy learning from the verified planar Teacher dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from powered_landing_guidance.offline_dataset import (
    compute_training_normalization_statistics,
    dataset_configuration_sha256,
)
from powered_landing_guidance.offline_dataset_generation import (
    packed_shard_sha256,
    validate_packed_trajectory_shard,
)


@dataclass(frozen=True)
class BCDataset:
    train_states: NDArray[np.float32]
    train_actions: NDArray[np.float32]
    validation_states: NDArray[np.float32]
    validation_actions: NDArray[np.float32]
    normalization: dict[str, Any]
    dataset_id: str
    dataset_configuration_sha256: str
    shard_sha256: dict[str, str]
    trajectory_counts: dict[str, int]


@dataclass(frozen=True)
class TrainingSettings:
    hidden_sizes: tuple[int, ...] = (64, 64)
    batch_size: int = 1024
    max_epochs: int = 80
    patience: int = 12
    min_delta: float = 1e-5
    learning_rate: float = 1e-3
    seed: int = 20260925
    overfit_examples: int = 16
    overfit_steps: int = 2000
    overfit_max_mse: float = 0.0025

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> TrainingSettings:
        allowed = set(cls.__dataclass_fields__)
        if set(values) != allowed:
            raise ValueError(f"training settings fields must be {sorted(allowed)}")
        hidden = values["hidden_sizes"]
        if (
            not isinstance(hidden, list)
            or not hidden
            or any(type(width) is not int or width <= 0 for width in hidden)
        ):
            raise ValueError("hidden_sizes must be a nonempty list of positive integers")
        integers = ("batch_size", "max_epochs", "patience", "overfit_examples", "overfit_steps")
        if any(type(values[name]) is not int or values[name] <= 0 for name in integers):
            raise ValueError("batch, epoch, patience, and overfit counts must be positive integers")
        if type(values["seed"]) is not int or values["seed"] < 0:
            raise ValueError("seed must be a nonnegative integer")
        floats = ("min_delta", "learning_rate", "overfit_max_mse")
        if any(
            isinstance(values[name], bool)
            or not isinstance(values[name], int | float)
            or not np.isfinite(values[name])
            or values[name] <= 0.0
            for name in floats
        ):
            raise ValueError("training rates and thresholds must be positive finite numbers")
        return cls(**{**values, "hidden_sizes": tuple(hidden)})

    def as_dict(self) -> dict[str, Any]:
        values = {name: getattr(self, name) for name in self.__dataclass_fields__}
        values["hidden_sizes"] = list(self.hidden_sizes)
        return values


def _load_verified_shard(
    directory: Path, manifest: dict[str, Any], split: str
) -> dict[str, NDArray[Any]]:
    record = manifest["shards"][split]
    expected_name = f"{split}-trajectories.npz"
    if record.get("file") != expected_name:
        raise ValueError(f"unexpected {split} shard filename")
    with np.load(directory / expected_name, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    if not validate_packed_trajectory_shard(arrays)["passed"]:
        raise ValueError(f"{split} shard failed schema validation")
    if packed_shard_sha256(arrays) != record.get("sha256"):
        raise ValueError(f"{split} shard digest mismatch")
    if len(arrays["trajectory_ids"]) != record.get("trajectory_count"):
        raise ValueError(f"{split} trajectory count mismatch")
    return arrays


def _aligned_pairs(arrays: dict[str, NDArray[Any]]) -> tuple[NDArray[Any], NDArray[Any]]:
    mask = np.ones(len(arrays["states"]), dtype=np.bool_)
    mask[arrays["state_offsets"][1:] - 1] = False
    states = arrays["states"][mask]
    actions = arrays["actions"]
    if len(states) != len(actions):
        raise ValueError("nonterminal states and actions are not aligned")
    return states, actions


def load_bc_dataset(directory: Path, config: dict[str, Any]) -> BCDataset:
    """Read only train and validation shards after checking their frozen provenance."""
    manifest = json.loads((directory / "dataset-manifest.json").read_text(encoding="utf-8"))
    normalization = json.loads((directory / "normalization.json").read_text(encoding="utf-8"))
    expected_digest = dataset_configuration_sha256(config)
    if (
        manifest.get("problem") != "day24-final-offline-dataset"
        or manifest.get("completion_gate", {}).get("passed") is not True
        or manifest.get("split_integrity", {}).get("passed") is not True
        or manifest.get("dataset_id") != config["offline_dataset_finalization"]["id"]
        or manifest.get("dataset_configuration_sha256") != expected_digest
        or manifest.get("teacher_provenance", {}).get("configuration_sha256")
        != config["teacher_protocol"]["configuration_sha256"]
    ):
        raise ValueError("final dataset manifest does not match the configured Teacher")
    if normalization != manifest.get("normalization") or normalization.get("fit_split") != "train":
        raise ValueError("normalization must match the final train-only manifest")
    if normalization.get("terminal_states_included") is not False:
        raise ValueError("normalization must exclude terminal states")
    for kind, width in (("state", 7), ("action", 2)):
        item = normalization[kind]
        if len(item["mean"]) != width or len(item["scale"]) != width:
            raise ValueError(f"invalid {kind} normalization dimensions")
        if (
            not np.all(np.isfinite(item["mean"]))
            or not np.all(np.isfinite(item["scale"]))
            or np.any(np.asarray(item["scale"]) <= 0.0)
        ):
            raise ValueError(f"invalid {kind} normalization values")
    train = _load_verified_shard(directory, manifest, "train")
    validation = _load_verified_shard(directory, manifest, "validation")
    if set(train["initial_condition_ids"].tolist()) & set(
        validation["initial_condition_ids"].tolist()
    ):
        raise ValueError("train and validation initial conditions overlap")
    train_states, train_actions = _aligned_pairs(train)
    validation_states, validation_actions = _aligned_pairs(validation)
    if normalization != compute_training_normalization_statistics(
        config, train_states, train_actions
    ):
        raise ValueError("normalization values do not match train transitions")
    if normalization["state"]["count"] != len(train_states) or normalization["action"][
        "count"
    ] != len(train_actions):
        raise ValueError("normalization row count does not match train data")
    state_mean = np.asarray(normalization["state"]["mean"], dtype=np.float64)
    state_scale = np.asarray(normalization["state"]["scale"], dtype=np.float64)
    action_mean = np.asarray(normalization["action"]["mean"], dtype=np.float64)
    action_scale = np.asarray(normalization["action"]["scale"], dtype=np.float64)

    def normalized(
        states: NDArray[Any], actions: NDArray[Any]
    ) -> tuple[NDArray[Any], NDArray[Any]]:
        return (
            np.asarray((states - state_mean) / state_scale, dtype=np.float32),
            np.asarray((actions - action_mean) / action_scale, dtype=np.float32),
        )

    train_x, train_y = normalized(train_states, train_actions)
    validation_x, validation_y = normalized(validation_states, validation_actions)
    return BCDataset(
        train_x,
        train_y,
        validation_x,
        validation_y,
        normalization,
        manifest["dataset_id"],
        expected_digest,
        {split: manifest["shards"][split]["sha256"] for split in ("train", "validation")},
        {
            split: len(shard["trajectory_ids"])
            for split, shard in (("train", train), ("validation", validation))
        },
    )


class BCPolicy:
    """Float32 ReLU MLP with a linear normalized-action head."""

    def __init__(self, hidden_sizes: tuple[int, ...], seed: int, activation: str = "relu") -> None:
        if activation not in {"relu", "tanh"}:
            raise ValueError("activation must be relu or tanh")
        self.activation = activation
        widths = (7, *hidden_sizes, 2)
        rng = np.random.default_rng(seed)
        self.weights = [
            np.asarray(
                rng.normal(
                    0.0, np.sqrt((2.0 if activation == "relu" else 1.0) / left), (left, right)
                ),
                dtype=np.float32,
            )
            for left, right in zip(widths[:-1], widths[1:], strict=True)
        ]
        self.biases = [np.zeros(right, dtype=np.float32) for right in widths[1:]]

    def copy(self) -> BCPolicy:
        result = BCPolicy(
            tuple(weight.shape[1] for weight in self.weights[:-1]), 0, self.activation
        )
        result.weights = [weight.copy() for weight in self.weights]
        result.biases = [bias.copy() for bias in self.biases]
        return result

    def predict_normalized(self, states: NDArray[Any]) -> NDArray[np.float32]:
        values = np.asarray(states, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 7:
            raise ValueError("states must have shape (batch, 7)")
        for weight, bias in zip(self.weights[:-1], self.biases[:-1], strict=True):
            pre = values @ weight + bias
            values = np.maximum(pre, 0.0) if self.activation == "relu" else np.tanh(pre)
        return values @ self.weights[-1] + self.biases[-1]

    def gradients(
        self,
        states: NDArray[np.float32],
        actions: NDArray[np.float32],
        sample_weights: NDArray[Any] | None = None,
        action_weights: tuple[float, float] = (1.0, 1.0),
    ) -> tuple[float, list[NDArray[np.float32]], list[NDArray[np.float32]]]:
        activations = [states]
        preactivations = []
        for weight, bias in zip(self.weights[:-1], self.biases[:-1], strict=True):
            pre = activations[-1] @ weight + bias
            preactivations.append(pre)
            activations.append(np.maximum(pre, 0.0) if self.activation == "relu" else np.tanh(pre))
        prediction = activations[-1] @ self.weights[-1] + self.biases[-1]
        error = prediction - actions
        row_weights = (
            np.ones((len(states), 1), dtype=np.float32)
            if sample_weights is None
            else np.asarray(sample_weights, dtype=np.float32).reshape(-1, 1)
        )
        column_weights = np.asarray(action_weights, dtype=np.float32).reshape(1, 2)
        if (
            row_weights.shape != (len(states), 1)
            or not np.all(np.isfinite(row_weights))
            or np.any(row_weights <= 0)
        ):
            raise ValueError("sample weights must be finite, positive, and aligned")
        if not np.all(np.isfinite(column_weights)) or np.any(column_weights <= 0):
            raise ValueError("action weights must be finite and positive")
        denominator = float(np.sum(row_weights) * np.sum(column_weights))
        loss = float(np.sum(row_weights * column_weights * error * error) / denominator)
        delta = error * row_weights * column_weights * np.float32(2.0 / denominator)
        weight_grads = [np.empty_like(weight) for weight in self.weights]
        bias_grads = [np.empty_like(bias) for bias in self.biases]
        for layer in range(len(self.weights) - 1, -1, -1):
            weight_grads[layer] = activations[layer].T @ delta
            bias_grads[layer] = np.sum(delta, axis=0)
            if layer:
                derivative = (
                    preactivations[layer - 1] > 0.0
                    if self.activation == "relu"
                    else 1.0 - activations[layer] ** 2
                )
                delta = (delta @ self.weights[layer].T) * derivative
        return loss, weight_grads, bias_grads

    def predict_action(
        self,
        state: NDArray[Any],
        normalization: dict[str, Any],
        throttle_bounds: tuple[float, float],
        gimbal_limit_rad: float,
    ) -> NDArray[Any]:
        """Return clipped physical actuator commands for one seven-state observation."""
        values = np.asarray(state, dtype=np.float64)
        if values.shape != (7,) or not np.all(np.isfinite(values)):
            raise ValueError("state must contain seven finite values")
        state_stats, action_stats = normalization["state"], normalization["action"]
        normalized = (values - np.asarray(state_stats["mean"])) / np.asarray(state_stats["scale"])
        prediction = self.predict_normalized(normalized.reshape(1, 7))[0].astype(np.float64)
        physical = prediction * np.asarray(action_stats["scale"]) + np.asarray(action_stats["mean"])
        return np.asarray(
            (
                np.clip(physical[0], *throttle_bounds),
                np.clip(physical[1], -gimbal_limit_rad, gimbal_limit_rad),
            ),
            dtype=np.float64,
        )


class Adam:
    def __init__(self, policy: BCPolicy, learning_rate: float) -> None:
        self.parameters = [*policy.weights, *policy.biases]
        self.first = [np.zeros_like(value) for value in self.parameters]
        self.second = [np.zeros_like(value) for value in self.parameters]
        self.learning_rate = learning_rate
        self.step_count = 0

    def step(self, weight_grads: list[NDArray[Any]], bias_grads: list[NDArray[Any]]) -> None:
        self.step_count += 1
        correction_first = 1.0 - 0.9**self.step_count
        correction_second = 1.0 - 0.999**self.step_count
        for value, gradient, first, second in zip(
            self.parameters, [*weight_grads, *bias_grads], self.first, self.second, strict=True
        ):
            first *= 0.9
            first += 0.1 * gradient
            second *= 0.999
            second += 0.001 * gradient * gradient
            value -= (
                self.learning_rate
                * (first / correction_first)
                / (np.sqrt(second / correction_second) + 1e-8)
            )


def action_error(
    policy: BCPolicy,
    states: NDArray[Any],
    actions: NDArray[Any],
    normalization: dict[str, Any],
    batch_size: int = 8192,
) -> dict[str, Any]:
    if not len(states):
        raise ValueError("cannot evaluate an empty split")
    sum_square = np.zeros(2, dtype=np.float64)
    sum_absolute = np.zeros(2, dtype=np.float64)
    for start in range(0, len(states), batch_size):
        difference = (
            policy.predict_normalized(states[start : start + batch_size])
            - actions[start : start + batch_size]
        )
        sum_square += np.sum(np.square(difference, dtype=np.float64), axis=0)
        sum_absolute += np.sum(np.abs(difference), axis=0)
    mse = sum_square / len(states)
    mae = sum_absolute / len(states)
    scale = np.asarray(normalization["action"]["scale"])
    return {
        "normalized_mse": float(np.mean(mse)),
        "normalized_rmse_by_action": np.sqrt(mse).tolist(),
        "throttle_mae": float(mae[0] * scale[0]),
        "gimbal_mae_deg": float(np.rad2deg(mae[1] * scale[1])),
        "samples": len(states),
    }


def overfit_check(dataset: BCDataset, settings: TrainingSettings) -> dict[str, Any]:
    if settings.overfit_examples > len(dataset.train_states):
        raise ValueError("overfit_examples exceeds train rows")
    rng = np.random.default_rng(settings.seed + 1)
    indices = rng.choice(len(dataset.train_states), settings.overfit_examples, replace=False)
    states, actions = dataset.train_states[indices], dataset.train_actions[indices]
    policy = BCPolicy(settings.hidden_sizes, settings.seed + 1)
    optimizer = Adam(policy, settings.learning_rate)
    for step in range(1, settings.overfit_steps + 1):
        _, weights, biases = policy.gradients(states, actions)
        optimizer.step(weights, biases)
        if step % 25 == 0 or step == settings.overfit_steps:
            error = action_error(policy, states, actions, dataset.normalization)
            if error["normalized_mse"] <= settings.overfit_max_mse:
                return {
                    "passed": True,
                    "steps": step,
                    "threshold": settings.overfit_max_mse,
                    "metrics": error,
                }
    return {
        "passed": False,
        "steps": settings.overfit_steps,
        "threshold": settings.overfit_max_mse,
        "metrics": error,
    }


def train_bc(
    dataset: BCDataset,
    settings: TrainingSettings,
    progress: Any = None,
    *,
    activation: str = "relu",
    sample_weights: NDArray[Any] | None = None,
    action_weights: tuple[float, float] = (1.0, 1.0),
    initial_policy: BCPolicy | None = None,
) -> tuple[BCPolicy, dict[str, Any]]:
    """Choose the lowest IID-validation MSE checkpoint with early stopping."""
    rng = np.random.default_rng(settings.seed)
    policy = (
        BCPolicy(settings.hidden_sizes, settings.seed, activation)
        if initial_policy is None
        else initial_policy.copy()
    )
    if tuple(weight.shape[1] for weight in policy.weights[:-1]) != settings.hidden_sizes:
        raise ValueError("initial policy architecture does not match training settings")
    if policy.activation != activation:
        raise ValueError("initial policy activation does not match")
    if sample_weights is not None and len(sample_weights) != len(dataset.train_states):
        raise ValueError("sample weights must align with training rows")
    optimizer = Adam(policy, settings.learning_rate)
    best = policy.copy()
    best_loss = float("inf")
    best_epoch = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, settings.max_epochs + 1):
        indices = rng.permutation(len(dataset.train_states))
        for start in range(0, len(indices), settings.batch_size):
            subset = indices[start : start + settings.batch_size]
            _, weights, biases = policy.gradients(
                dataset.train_states[subset],
                dataset.train_actions[subset],
                None if sample_weights is None else sample_weights[subset],
                action_weights,
            )
            optimizer.step(weights, biases)
        training = action_error(
            policy, dataset.train_states, dataset.train_actions, dataset.normalization
        )
        validation = action_error(
            policy, dataset.validation_states, dataset.validation_actions, dataset.normalization
        )
        if not np.isfinite(training["normalized_mse"]) or not np.isfinite(
            validation["normalized_mse"]
        ):
            raise FloatingPointError("nonfinite BC loss")
        history.append({"epoch": epoch, "train": training, "validation": validation})
        if validation["normalized_mse"] < best_loss - settings.min_delta:
            best_loss = validation["normalized_mse"]
            best_epoch = epoch
            best = policy.copy()
        if progress is not None:
            progress(epoch, training["normalized_mse"], validation["normalized_mse"], best_epoch)
        if epoch - best_epoch >= settings.patience:
            break
    return best, {
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "stopped_early": len(history) < settings.max_epochs,
        "history": history,
        "best_train": action_error(
            best, dataset.train_states, dataset.train_actions, dataset.normalization
        ),
        "best_validation": action_error(
            best, dataset.validation_states, dataset.validation_actions, dataset.normalization
        ),
    }


def save_checkpoint(path: Path, policy: BCPolicy, metadata: dict[str, Any]) -> None:
    """Write a safe NPZ checkpoint without pickle-dependent model objects."""
    arrays: dict[str, Any] = {
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False))
    }
    for index, (weight, bias) in enumerate(zip(policy.weights, policy.biases, strict=True)):
        arrays[f"weight_{index}"] = weight
        arrays[f"bias_{index}"] = bias
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def load_checkpoint(path: Path) -> tuple[BCPolicy, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        widths = tuple(metadata["training_settings"]["hidden_sizes"])
        policy = BCPolicy(widths, 0, metadata.get("activation", "relu"))
        expected = {"metadata_json"}
        for index in range(len(policy.weights)):
            for prefix, target in (("weight", policy.weights), ("bias", policy.biases)):
                name = f"{prefix}_{index}"
                expected.add(name)
                value = archive[name]
                if (
                    value.shape != target[index].shape
                    or value.dtype != np.dtype("float32")
                    or not np.all(np.isfinite(value))
                ):
                    raise ValueError(f"invalid checkpoint array {name}")
                target[index] = value.copy()
        if set(archive.files) != expected:
            raise ValueError("unexpected checkpoint fields")
    return policy, metadata
