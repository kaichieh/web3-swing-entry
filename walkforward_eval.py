"""
Run rolling walk-forward evaluation for the logistic swing-entry baseline.

This keeps the existing `train.py` baseline intact while measuring whether
performance is stable across multiple chronological windows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import asset_config as ac
from prepare import EXPERIMENTAL_FEATURE_COLUMNS, FEATURE_COLUMNS, get_runtime_config, load_dataset_frame
from train import (
    DEFAULT_SIDE,
    LEARNING_RATE,
    L2_REG,
    MAX_EPOCHS,
    NEG_WEIGHT,
    PATIENCE,
    POS_WEIGHT,
    THRESHOLD_MAX,
    THRESHOLD_MIN,
    THRESHOLD_STEPS,
    TrainedModel,
    add_bias,
    add_interaction_terms,
    classification_stats,
    compute_metrics,
    get_active_drop_features,
    get_active_extra_base_features,
    get_env_float,
    get_env_int,
    get_side,
    get_target_column,
    get_realized_return_column,
    set_seed,
    sigmoid,
    standardize,
)

DEFAULT_WF_TRAIN_FRACTION = 0.55
DEFAULT_WF_VALID_FRACTION = 0.15
DEFAULT_WF_TEST_FRACTION = 0.15
DEFAULT_WF_STEP_FRACTION = 0.05


@dataclass(frozen=True)
class WindowSlice:
    start: int
    train_end: int
    valid_end: int
    test_end: int


@dataclass(frozen=True)
class FoldResult:
    fold_id: int
    train_start_date: str
    train_end_date: str
    validation_end_date: str
    test_end_date: str
    threshold: float
    validation_bal_acc: float
    test_bal_acc: float
    validation_f1: float
    test_f1: float
    test_positive_rate: float


def assemble_feature_names(frame_columns: list[str], side: str) -> list[str]:
    feature_names = list(FEATURE_COLUMNS)
    extra_base_features = set(get_active_extra_base_features(side))
    for column in EXPERIMENTAL_FEATURE_COLUMNS:
        if column in frame_columns and column in extra_base_features:
            feature_names.append(column)
    drop_features = set(get_active_drop_features())
    feature_names = [name for name in feature_names if name not in drop_features]
    return feature_names


def select_threshold_by_balanced_accuracy(probabilities: np.ndarray, labels: np.ndarray) -> float:
    threshold_min = get_env_float("AR_THRESHOLD_MIN", THRESHOLD_MIN)
    threshold_max = get_env_float("AR_THRESHOLD_MAX", THRESHOLD_MAX)
    threshold_steps = get_env_int("AR_THRESHOLD_STEPS", THRESHOLD_STEPS)
    best_threshold = 0.5
    best_bal_acc = -1.0
    best_f1 = -1.0
    for threshold in np.linspace(threshold_min, threshold_max, threshold_steps):
        tp, tn, fp, fn, _ = classification_stats(probabilities, labels, float(threshold))
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        specificity = tn / max(tn + fp, 1.0)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        bal_acc = 0.5 * (recall + specificity)
        if bal_acc > best_bal_acc or (abs(bal_acc - best_bal_acc) < 1e-8 and f1 > best_f1):
            best_threshold = float(threshold)
            best_bal_acc = bal_acc
            best_f1 = f1
    return best_threshold


def build_window_slices(num_rows: int) -> list[WindowSlice]:
    train_size = max(240, int(num_rows * get_env_float("AR_WF_TRAIN_FRACTION", DEFAULT_WF_TRAIN_FRACTION)))
    valid_size = max(90, int(num_rows * get_env_float("AR_WF_VALID_FRACTION", DEFAULT_WF_VALID_FRACTION)))
    test_size = max(90, int(num_rows * get_env_float("AR_WF_TEST_FRACTION", DEFAULT_WF_TEST_FRACTION)))
    step_size = max(45, int(num_rows * get_env_float("AR_WF_STEP_FRACTION", DEFAULT_WF_STEP_FRACTION)))
    slices: list[WindowSlice] = []
    start = 0
    while start + train_size + valid_size + test_size <= num_rows:
        train_end = start + train_size
        valid_end = train_end + valid_size
        test_end = valid_end + test_size
        slices.append(WindowSlice(start=start, train_end=train_end, valid_end=valid_end, test_end=test_end))
        start += step_size
    return slices


def fit_window_model(
    train_x: np.ndarray,
    validation_x: np.ndarray,
    train_y: np.ndarray,
    validation_y: np.ndarray,
    feature_names: list[str],
    side: str,
) -> TrainedModel:
    learning_rate = get_env_float("AR_LEARNING_RATE", LEARNING_RATE)
    l2_reg = get_env_float("AR_L2_REG", L2_REG)
    pos_weight = get_env_float("AR_POS_WEIGHT", POS_WEIGHT)
    neg_weight = get_env_float("AR_NEG_WEIGHT", NEG_WEIGHT)
    max_epochs = get_env_int("AR_MAX_EPOCHS", MAX_EPOCHS)
    patience_limit = get_env_int("AR_PATIENCE", PATIENCE)

    train_x, validation_x, _ = standardize(train_x, validation_x, validation_x.copy())
    train_x, validation_x, _ = add_interaction_terms(train_x, validation_x, validation_x.copy(), feature_names)
    train_x = add_bias(train_x)
    validation_x = add_bias(validation_x)

    weights = np.zeros(train_x.shape[1], dtype=np.float32)
    best_weights = weights.copy()
    best_validation_bal_acc = -np.inf
    best_validation_f1 = -np.inf
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
        threshold = select_threshold_by_balanced_accuracy(sigmoid(validation_logits), validation_y)
        validation_metrics = compute_metrics(
            validation_logits,
            validation_y,
            np.zeros_like(validation_y, dtype=np.float32),
            threshold,
        )
        score_improved = (
            validation_metrics.balanced_accuracy > best_validation_bal_acc
            or (
                abs(validation_metrics.balanced_accuracy - best_validation_bal_acc) < 1e-8
                and validation_metrics.f1 > best_validation_f1
            )
        )
        if score_improved:
            best_validation_bal_acc = validation_metrics.balanced_accuracy
            best_validation_f1 = validation_metrics.f1
            best_weights = weights.copy()
            best_threshold = threshold
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience_limit:
            break

    return TrainedModel(
        side=side,
        target_column=get_target_column(side),
        realized_return_column=get_realized_return_column(side),
        feature_names=feature_names,
        weights=best_weights,
        threshold=best_threshold,
        best_epoch=best_epoch,
    )


def evaluate_fold(frame, window: WindowSlice, side: str, fold_id: int) -> FoldResult:
    feature_names = assemble_feature_names(frame.columns.tolist(), side)
    train_frame = frame.iloc[window.start:window.train_end].copy()
    validation_frame = frame.iloc[window.train_end:window.valid_end].copy()
    test_frame = frame.iloc[window.valid_end:window.test_end].copy()

    train_x = train_frame[feature_names].to_numpy(dtype=np.float32)
    validation_x = validation_frame[feature_names].to_numpy(dtype=np.float32)
    test_x = test_frame[feature_names].to_numpy(dtype=np.float32)
    train_y = train_frame[get_target_column(side)].to_numpy(dtype=np.float32)
    validation_y = validation_frame[get_target_column(side)].to_numpy(dtype=np.float32)
    test_y = test_frame[get_target_column(side)].to_numpy(dtype=np.float32)

    scaled_train_x, scaled_validation_x, scaled_test_x = standardize(train_x, validation_x, test_x)
    scaled_train_x, scaled_validation_x, scaled_test_x = add_interaction_terms(
        scaled_train_x,
        scaled_validation_x,
        scaled_test_x,
        feature_names,
    )
    model = fit_window_model(train_x, validation_x, train_y, validation_y, feature_names, side)
    scaled_train_x = add_bias(scaled_train_x)
    scaled_validation_x = add_bias(scaled_validation_x)
    scaled_test_x = add_bias(scaled_test_x)

    validation_metrics = compute_metrics(
        scaled_validation_x @ model.weights,
        validation_y,
        validation_frame[get_realized_return_column(side)].to_numpy(dtype=np.float32),
        model.threshold,
    )
    test_metrics = compute_metrics(
        scaled_test_x @ model.weights,
        test_y,
        test_frame[get_realized_return_column(side)].to_numpy(dtype=np.float32),
        model.threshold,
    )
    return FoldResult(
        fold_id=fold_id,
        train_start_date=train_frame["date"].iloc[0].strftime("%Y-%m-%d"),
        train_end_date=train_frame["date"].iloc[-1].strftime("%Y-%m-%d"),
        validation_end_date=validation_frame["date"].iloc[-1].strftime("%Y-%m-%d"),
        test_end_date=test_frame["date"].iloc[-1].strftime("%Y-%m-%d"),
        threshold=model.threshold,
        validation_bal_acc=validation_metrics.balanced_accuracy,
        test_bal_acc=test_metrics.balanced_accuracy,
        validation_f1=validation_metrics.f1,
        test_f1=test_metrics.f1,
        test_positive_rate=test_metrics.positive_rate,
    )


def main() -> None:
    seed = get_env_int("AR_SEED", 42)
    side = get_side() if DEFAULT_SIDE else "long"
    set_seed(seed)
    frame = load_dataset_frame()
    windows = build_window_slices(len(frame))
    if not windows:
        raise RuntimeError("Not enough rows for walk-forward evaluation.")

    results = [evaluate_fold(frame, window, side, idx + 1) for idx, window in enumerate(windows)]
    avg_validation_bal_acc = float(np.mean([item.validation_bal_acc for item in results]))
    avg_test_bal_acc = float(np.mean([item.test_bal_acc for item in results]))
    avg_validation_f1 = float(np.mean([item.validation_f1 for item in results]))
    avg_test_f1 = float(np.mean([item.test_f1 for item in results]))
    avg_test_positive_rate = float(np.mean([item.test_positive_rate for item in results]))
    min_test_bal_acc = float(np.min([item.test_bal_acc for item in results]))
    max_test_bal_acc = float(np.max([item.test_bal_acc for item in results]))

    config = get_runtime_config()
    symbol = ac.get_asset_symbol()
    print("---")
    print(f"task:                 {symbol}_{int(config['horizon_bars'])}bar_walkforward_{side}")
    print(f"folds:                {len(results)}")
    print(f"avg_validation_f1:    {avg_validation_f1:.4f}")
    print(f"avg_validation_bal:   {avg_validation_bal_acc:.4f}")
    print(f"avg_test_f1:          {avg_test_f1:.4f}")
    print(f"avg_test_bal:         {avg_test_bal_acc:.4f}")
    print(f"avg_test_pos_rate:    {avg_test_positive_rate:.4f}")
    print(f"min_test_bal:         {min_test_bal_acc:.4f}")
    print(f"max_test_bal:         {max_test_bal_acc:.4f}")
    print("fold_details:")
    for item in results:
        print(
            f"  fold={item.fold_id} "
            f"train={item.train_start_date}->{item.train_end_date} "
            f"valid_end={item.validation_end_date} "
            f"test_end={item.test_end_date} "
            f"thr={item.threshold:.3f} "
            f"val_bal={item.validation_bal_acc:.4f} "
            f"test_bal={item.test_bal_acc:.4f} "
            f"test_pos={item.test_positive_rate:.4f}"
        )


if __name__ == "__main__":
    main()
