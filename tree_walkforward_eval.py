"""
Run a small decision-tree baseline under the same walk-forward protocol.

The goal is not to build a production tree learner, but to provide a simple
non-linear baseline that can be compared against the logistic model without
adding extra dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import asset_config as ac
from prepare import get_runtime_config, load_dataset_frame
from train import classification_stats, get_env_int, get_side, get_target_column, get_realized_return_column, set_seed
from walkforward_eval import (
    assemble_feature_names,
    build_window_slices,
    select_threshold_by_balanced_accuracy,
)

DEFAULT_MAX_DEPTH = 2
DEFAULT_MIN_LEAF = 40
DEFAULT_NUM_BINS = 9


@dataclass(frozen=True)
class TreeNode:
    probability: float
    feature_index: int = -1
    threshold: float = 0.0
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None or self.right is None or self.feature_index < 0


def weighted_gini(labels: np.ndarray, weights: np.ndarray) -> float:
    total_weight = float(weights.sum())
    if total_weight <= 0.0:
        return 0.0
    positive_weight = float(weights[labels == 1.0].sum())
    negative_weight = total_weight - positive_weight
    positive_rate = positive_weight / total_weight
    negative_rate = negative_weight / total_weight
    return 1.0 - positive_rate**2 - negative_rate**2


def candidate_thresholds(values: np.ndarray, num_bins: int) -> np.ndarray:
    if len(values) == 0:
        return np.array([], dtype=np.float32)
    quantiles = np.linspace(0.1, 0.9, num_bins, dtype=np.float32)
    candidates = np.quantile(values, quantiles)
    return np.unique(candidates.astype(np.float32))


def best_split(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    min_leaf: int,
    num_bins: int,
) -> tuple[int, float, float] | None:
    parent_impurity = weighted_gini(labels, weights)
    best_feature = -1
    best_threshold = 0.0
    best_gain = 0.0
    total_weight = float(weights.sum())

    for feature_index in range(features.shape[1]):
        feature_values = features[:, feature_index]
        for threshold in candidate_thresholds(feature_values, num_bins):
            left_mask = feature_values <= threshold
            right_mask = ~left_mask
            if int(left_mask.sum()) < min_leaf or int(right_mask.sum()) < min_leaf:
                continue
            left_weight = float(weights[left_mask].sum())
            right_weight = float(weights[right_mask].sum())
            if left_weight <= 0.0 or right_weight <= 0.0:
                continue
            gain = parent_impurity
            gain -= (left_weight / total_weight) * weighted_gini(labels[left_mask], weights[left_mask])
            gain -= (right_weight / total_weight) * weighted_gini(labels[right_mask], weights[right_mask])
            if gain > best_gain:
                best_feature = feature_index
                best_threshold = float(threshold)
                best_gain = gain

    if best_feature < 0:
        return None
    return best_feature, best_threshold, best_gain


def build_tree(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    depth: int,
    max_depth: int,
    min_leaf: int,
    num_bins: int,
) -> TreeNode:
    probability = float(weights[labels == 1.0].sum() / max(weights.sum(), 1e-8))
    if depth >= max_depth or len(features) < min_leaf * 2:
        return TreeNode(probability=probability)

    split = best_split(features, labels, weights, min_leaf=min_leaf, num_bins=num_bins)
    if split is None:
        return TreeNode(probability=probability)

    feature_index, threshold, _ = split
    left_mask = features[:, feature_index] <= threshold
    right_mask = ~left_mask
    left = build_tree(features[left_mask], labels[left_mask], weights[left_mask], depth + 1, max_depth, min_leaf, num_bins)
    right = build_tree(features[right_mask], labels[right_mask], weights[right_mask], depth + 1, max_depth, min_leaf, num_bins)
    return TreeNode(probability=probability, feature_index=feature_index, threshold=threshold, left=left, right=right)


def predict_probabilities(node: TreeNode, features: np.ndarray) -> np.ndarray:
    probabilities = np.zeros(features.shape[0], dtype=np.float32)
    for idx, row in enumerate(features):
        current = node
        while not current.is_leaf:
            current = current.left if row[current.feature_index] <= current.threshold else current.right
        probabilities[idx] = current.probability
    return probabilities


def compute_metrics_from_probabilities(
    probabilities: np.ndarray,
    labels: np.ndarray,
    realized_returns: np.ndarray,
    threshold: float,
) -> tuple[float, float, float, float, float]:
    tp, tn, fp, fn, predictions = classification_stats(probabilities, labels, threshold)
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    specificity = tn / max(tn + fp, 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    bal_acc = 0.5 * (recall + specificity)
    positive_rate = float(predictions.mean())
    avg_return = float(realized_returns[predictions == 1].mean()) if float(predictions.sum()) > 0 else 0.0
    return f1, bal_acc, positive_rate, precision, avg_return


def evaluate_walkforward(frame, side: str) -> dict[str, object]:
    feature_names = assemble_feature_names(frame.columns.tolist())
    max_depth = get_env_int("AR_TREE_MAX_DEPTH", DEFAULT_MAX_DEPTH)
    min_leaf = get_env_int("AR_TREE_MIN_LEAF", DEFAULT_MIN_LEAF)
    num_bins = get_env_int("AR_TREE_NUM_BINS", DEFAULT_NUM_BINS)
    windows = build_window_slices(len(frame))
    if not windows:
        raise RuntimeError("Not enough rows for tree walk-forward evaluation.")

    validation_f1s: list[float] = []
    validation_bals: list[float] = []
    test_f1s: list[float] = []
    test_bals: list[float] = []
    test_positive_rates: list[float] = []
    fold_lines: list[str] = []

    target_column = get_target_column(side)
    realized_return_column = get_realized_return_column(side)

    for fold_id, window in enumerate(windows, start=1):
        train_frame = frame.iloc[window.start:window.train_end].copy()
        validation_frame = frame.iloc[window.train_end:window.valid_end].copy()
        test_frame = frame.iloc[window.valid_end:window.test_end].copy()

        train_x = train_frame[feature_names].to_numpy(dtype=np.float32)
        validation_x = validation_frame[feature_names].to_numpy(dtype=np.float32)
        test_x = test_frame[feature_names].to_numpy(dtype=np.float32)
        train_y = train_frame[target_column].to_numpy(dtype=np.float32)
        validation_y = validation_frame[target_column].to_numpy(dtype=np.float32)
        test_y = test_frame[target_column].to_numpy(dtype=np.float32)
        train_weights = np.ones_like(train_y, dtype=np.float32)

        tree = build_tree(
            train_x,
            train_y,
            train_weights,
            depth=0,
            max_depth=max_depth,
            min_leaf=min_leaf,
            num_bins=num_bins,
        )
        validation_probs = predict_probabilities(tree, validation_x)
        test_probs = predict_probabilities(tree, test_x)
        threshold = select_threshold_by_balanced_accuracy(validation_probs, validation_y)
        validation_f1, validation_bal, _, _, _ = compute_metrics_from_probabilities(
            validation_probs,
            validation_y,
            validation_frame[realized_return_column].to_numpy(dtype=np.float32),
            threshold,
        )
        test_f1, test_bal, test_positive_rate, _, _ = compute_metrics_from_probabilities(
            test_probs,
            test_y,
            test_frame[realized_return_column].to_numpy(dtype=np.float32),
            threshold,
        )
        validation_f1s.append(validation_f1)
        validation_bals.append(validation_bal)
        test_f1s.append(test_f1)
        test_bals.append(test_bal)
        test_positive_rates.append(test_positive_rate)
        fold_lines.append(
            f"  fold={fold_id} "
            f"train={train_frame['date'].iloc[0].strftime('%Y-%m-%d')}->{train_frame['date'].iloc[-1].strftime('%Y-%m-%d')} "
            f"valid_end={validation_frame['date'].iloc[-1].strftime('%Y-%m-%d')} "
            f"test_end={test_frame['date'].iloc[-1].strftime('%Y-%m-%d')} "
            f"thr={threshold:.3f} "
            f"val_bal={validation_bal:.4f} "
            f"test_bal={test_bal:.4f} "
            f"test_pos={test_positive_rate:.4f}"
        )

    return {
        "folds": len(windows),
        "avg_validation_f1": float(np.mean(validation_f1s)),
        "avg_validation_bal": float(np.mean(validation_bals)),
        "avg_test_f1": float(np.mean(test_f1s)),
        "avg_test_bal": float(np.mean(test_bals)),
        "avg_test_pos_rate": float(np.mean(test_positive_rates)),
        "min_test_bal": float(np.min(test_bals)),
        "max_test_bal": float(np.max(test_bals)),
        "fold_lines": fold_lines,
    }


def main() -> None:
    seed = get_env_int("AR_SEED", 42)
    side = get_side()
    set_seed(seed)
    frame = load_dataset_frame()
    result = evaluate_walkforward(frame, side)
    config = get_runtime_config()
    symbol = ac.get_asset_symbol()

    print("---")
    print(f"task:                 {symbol}_{int(config['horizon_bars'])}bar_tree_walkforward_{side}")
    print(f"folds:                {result['folds']}")
    print(f"avg_validation_f1:    {result['avg_validation_f1']:.4f}")
    print(f"avg_validation_bal:   {result['avg_validation_bal']:.4f}")
    print(f"avg_test_f1:          {result['avg_test_f1']:.4f}")
    print(f"avg_test_bal:         {result['avg_test_bal']:.4f}")
    print(f"avg_test_pos_rate:    {result['avg_test_pos_rate']:.4f}")
    print(f"min_test_bal:         {result['min_test_bal']:.4f}")
    print(f"max_test_bal:         {result['max_test_bal']:.4f}")
    print("fold_details:")
    for line in result["fold_lines"]:
        print(line)


if __name__ == "__main__":
    main()
