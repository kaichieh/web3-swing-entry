"""
Train a NumPy logistic baseline for weekly crypto swing-entry classification.

Use `AR_SIDE=long` or `AR_SIDE=short` to choose the target.
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass

import numpy as np

import asset_config as ac
from prepare import (
    EXPERIMENTAL_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    LONG_TARGET_COLUMN,
    SHORT_TARGET_COLUMN,
    get_runtime_config,
    load_splits,
)

SEED = 42
LEARNING_RATE = 0.02
L2_REG = 1e-3
MAX_EPOCHS = 1200
PATIENCE = 120
POS_WEIGHT = 1.0
NEG_WEIGHT = 1.0
THRESHOLD_MIN = 0.30
THRESHOLD_MAX = 0.70
THRESHOLD_STEPS = 401
DEFAULT_INTERACTION_FEATURE_PAIRS = (("drawdown_7", "volume_vs_7"),)
DEFAULT_SIDE = "long"


@dataclass
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    balanced_accuracy: float
    positive_rate: float
    avg_realized_return: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def get_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value is not None else default


def get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


def get_env_csv(name: str, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return tuple(part.strip() for part in value.split(",") if part.strip())


def get_env_interaction_pairs(name: str, default: tuple[tuple[str, str], ...] = ()) -> tuple[tuple[str, str], ...]:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    pairs: list[tuple[str, str]] = list(default)
    for part in value.split(","):
        left, sep, right = part.strip().partition(":")
        if sep and left and right:
            candidate = (left.strip(), right.strip())
            if candidate not in pairs:
                pairs.append(candidate)
    return tuple(pairs)


def get_side() -> str:
    side = os.getenv("AR_SIDE", DEFAULT_SIDE).strip().lower()
    if side not in {"long", "short"}:
        raise ValueError("AR_SIDE must be 'long' or 'short'.")
    return side


def get_target_column(side: str) -> str:
    return LONG_TARGET_COLUMN if side == "long" else SHORT_TARGET_COLUMN


def get_realized_return_column(side: str) -> str:
    return "future_return_7" if side == "long" else "future_short_return_7"


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def standardize(train_x: np.ndarray, validation_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (train_x - mean) / std, (validation_x - mean) / std, (test_x - mean) / std


def add_bias(features: np.ndarray) -> np.ndarray:
    return np.concatenate([features, np.ones((features.shape[0], 1), dtype=features.dtype)], axis=1)


def add_interaction_terms(
    train_x: np.ndarray,
    validation_x: np.ndarray,
    test_x: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pairs = get_env_interaction_pairs("AR_EXTRA_INTERACTIONS", DEFAULT_INTERACTION_FEATURE_PAIRS)
    if not pairs:
        return train_x, validation_x, test_x
    feature_index = {name: idx for idx, name in enumerate(feature_names)}
    active_pairs = [(feature_index[left], feature_index[right]) for left, right in pairs if left in feature_index and right in feature_index]
    if not active_pairs:
        return train_x, validation_x, test_x

    def augment(features: np.ndarray) -> np.ndarray:
        extras = [features[:, i : i + 1] * features[:, j : j + 1] for i, j in active_pairs]
        return np.concatenate([features] + extras, axis=1)

    return augment(train_x), augment(validation_x), augment(test_x)


def assemble_feature_matrices(
    splits: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    feature_names = list(FEATURE_COLUMNS)
    for column in EXPERIMENTAL_FEATURE_COLUMNS:
        if column in splits["train"].frame.columns and column in get_env_csv("AR_EXTRA_BASE_FEATURES"):
            feature_names.append(column)
    drop_features = set(get_env_csv("AR_DROP_FEATURES"))
    feature_names = [name for name in feature_names if name not in drop_features]
    matrices = (
        splits["train"].frame[feature_names].to_numpy(dtype=np.float32),
        splits["validation"].frame[feature_names].to_numpy(dtype=np.float32),
        splits["test"].frame[feature_names].to_numpy(dtype=np.float32),
    )
    split_names = ("train", "validation", "test")
    issues: list[str] = []
    for split_name, matrix in zip(split_names, matrices):
        nan_counts = np.isnan(matrix).sum(axis=0)
        bad_columns = [(name, int(count)) for name, count in zip(feature_names, nan_counts) if count]
        if bad_columns:
            formatted = ", ".join(f"{name}={count}" for name, count in bad_columns)
            issues.append(f"{split_name}: {formatted}")
    if issues:
        raise ValueError("Selected features contain NaNs. " + " | ".join(issues))
    return (*matrices, feature_names)


def classification_stats(probabilities: np.ndarray, labels: np.ndarray, threshold: float) -> tuple[float, float, float, float, np.ndarray]:
    predictions = (probabilities >= threshold).astype(np.float32)
    tp = float(((predictions == 1) & (labels == 1)).sum())
    tn = float(((predictions == 0) & (labels == 0)).sum())
    fp = float(((predictions == 1) & (labels == 0)).sum())
    fn = float(((predictions == 0) & (labels == 1)).sum())
    return tp, tn, fp, fn, predictions


def select_threshold(probabilities: np.ndarray, labels: np.ndarray) -> float:
    threshold_min = get_env_float("AR_THRESHOLD_MIN", THRESHOLD_MIN)
    threshold_max = get_env_float("AR_THRESHOLD_MAX", THRESHOLD_MAX)
    threshold_steps = get_env_int("AR_THRESHOLD_STEPS", THRESHOLD_STEPS)
    best_threshold = 0.5
    best_f1 = -1.0
    best_bal_acc = -1.0
    for threshold in np.linspace(threshold_min, threshold_max, threshold_steps):
        tp, tn, fp, fn, _ = classification_stats(probabilities, labels, float(threshold))
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        specificity = tn / max(tn + fp, 1.0)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        bal_acc = 0.5 * (recall + specificity)
        if f1 > best_f1 or (abs(f1 - best_f1) < 1e-8 and bal_acc > best_bal_acc):
            best_threshold = float(threshold)
            best_f1 = f1
            best_bal_acc = bal_acc
    return best_threshold


def compute_metrics(logits: np.ndarray, labels: np.ndarray, realized_returns: np.ndarray, threshold: float) -> Metrics:
    probabilities = sigmoid(logits)
    tp, tn, fp, fn, predictions = classification_stats(probabilities, labels, threshold)
    accuracy = float((predictions == labels).mean())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    specificity = tn / max(tn + fp, 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    bal_acc = 0.5 * (recall + specificity)
    selected = realized_returns[predictions == 1]
    avg_realized_return = float(selected.mean()) if len(selected) else 0.0
    return Metrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        balanced_accuracy=bal_acc,
        positive_rate=float(predictions.mean()),
        avg_realized_return=avg_realized_return,
    )


def main() -> None:
    seed = get_env_int("AR_SEED", SEED)
    learning_rate = get_env_float("AR_LEARNING_RATE", LEARNING_RATE)
    l2_reg = get_env_float("AR_L2_REG", L2_REG)
    pos_weight = get_env_float("AR_POS_WEIGHT", POS_WEIGHT)
    neg_weight = get_env_float("AR_NEG_WEIGHT", NEG_WEIGHT)
    max_epochs = get_env_int("AR_MAX_EPOCHS", MAX_EPOCHS)
    patience_limit = get_env_int("AR_PATIENCE", PATIENCE)
    side = get_side()
    target_column = get_target_column(side)
    realized_return_column = get_realized_return_column(side)

    set_seed(seed)
    splits = load_splits(target_column=target_column)
    train_x, validation_x, test_x, feature_names = assemble_feature_matrices(splits)
    train_y = splits["train"].labels
    validation_y = splits["validation"].labels
    test_y = splits["test"].labels

    train_x, validation_x, test_x = standardize(train_x, validation_x, test_x)
    train_x, validation_x, test_x = add_interaction_terms(train_x, validation_x, test_x, feature_names)
    train_x = add_bias(train_x)
    validation_x = add_bias(validation_x)
    test_x = add_bias(test_x)

    weights = np.zeros(train_x.shape[1], dtype=np.float32)
    best_weights = weights.copy()
    best_validation_f1 = -math.inf
    best_threshold = 0.5
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        logits = train_x @ weights
        probs = sigmoid(logits)
        sample_weights = np.where(train_y == 1.0, pos_weight, neg_weight).astype(np.float32)
        gradient = train_x.T @ ((probs - train_y) * sample_weights) / train_x.shape[0]
        gradient[:-1] += l2_reg * weights[:-1]
        weights -= learning_rate * gradient

        validation_logits = validation_x @ weights
        threshold = select_threshold(sigmoid(validation_logits), validation_y)
        validation_metrics = compute_metrics(
            validation_logits,
            validation_y,
            splits["validation"].frame[realized_return_column].to_numpy(dtype=np.float32),
            threshold,
        )
        if validation_metrics.f1 > best_validation_f1:
            best_validation_f1 = validation_metrics.f1
            best_weights = weights.copy()
            best_threshold = threshold
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience_limit:
            break

    train_metrics = compute_metrics(
        train_x @ best_weights,
        train_y,
        splits["train"].frame[realized_return_column].to_numpy(dtype=np.float32),
        best_threshold,
    )
    validation_metrics = compute_metrics(
        validation_x @ best_weights,
        validation_y,
        splits["validation"].frame[realized_return_column].to_numpy(dtype=np.float32),
        best_threshold,
    )
    test_metrics = compute_metrics(
        test_x @ best_weights,
        test_y,
        splits["test"].frame[realized_return_column].to_numpy(dtype=np.float32),
        best_threshold,
    )

    config = get_runtime_config()
    symbol = ac.get_asset_symbol()
    print("---")
    print(f"task:                 {symbol}_{int(config['horizon_bars'])}bar_weekly_{side}")
    print(f"side:                 {side}")
    print(f"target_column:        {target_column}")
    print(f"model:                logistic_regression")
    print(f"features:             {len(feature_names)}")
    print(f"learning_rate:        {learning_rate}")
    print(f"l2_reg:               {l2_reg}")
    print(f"pos_weight:           {pos_weight}")
    print(f"neg_weight:           {neg_weight}")
    print(f"decision_threshold:   {best_threshold:.3f}")
    print(f"best_epoch:           {best_epoch}")
    print(f"train_accuracy:       {train_metrics.accuracy:.4f}")
    print(f"validation_accuracy:  {validation_metrics.accuracy:.4f}")
    print(f"validation_f1:        {validation_metrics.f1:.4f}")
    print(f"validation_bal_acc:   {validation_metrics.balanced_accuracy:.4f}")
    print(f"validation_precision: {validation_metrics.precision:.4f}")
    print(f"validation_recall:    {validation_metrics.recall:.4f}")
    print(f"test_accuracy:        {test_metrics.accuracy:.4f}")
    print(f"test_f1:              {test_metrics.f1:.4f}")
    print(f"test_bal_acc:         {test_metrics.balanced_accuracy:.4f}")
    print(f"test_precision:       {test_metrics.precision:.4f}")
    print(f"test_recall:          {test_metrics.recall:.4f}")
    print(f"test_positive_rate:   {test_metrics.positive_rate:.4f}")
    print(f"test_avg_return:      {test_metrics.avg_realized_return:.4%}")


if __name__ == "__main__":
    main()
