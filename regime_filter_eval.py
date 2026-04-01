"""
Evaluate simple regime gates on top of the logistic walk-forward baseline.

The gates act as no-trade filters: a prediction only remains active when the
current row satisfies the selected regime condition.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import asset_config as ac
from prepare import get_runtime_config, load_dataset_frame
from train import (
    classification_stats,
    compute_metrics,
    get_env_int,
    get_side,
    get_target_column,
    get_realized_return_column,
    set_seed,
)
from walkforward_eval import assemble_feature_names, build_window_slices, fit_window_model


@dataclass(frozen=True)
class RegimeResult:
    name: str
    avg_validation_bal: float
    avg_test_bal: float
    avg_test_positive_rate: float
    min_test_bal: float
    max_test_bal: float


def compute_metrics_with_gate(
    logits: np.ndarray,
    labels: np.ndarray,
    realized_returns: np.ndarray,
    threshold: float,
    gate_mask: np.ndarray,
):
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
    predictions = ((probabilities >= threshold) & gate_mask).astype(np.float32)
    tp = float(((predictions == 1) & (labels == 1)).sum())
    tn = float(((predictions == 0) & (labels == 0)).sum())
    fp = float(((predictions == 1) & (labels == 0)).sum())
    fn = float(((predictions == 0) & (labels == 1)).sum())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    specificity = tn / max(tn + fp, 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    bal_acc = 0.5 * (recall + specificity)
    selected = realized_returns[predictions == 1]
    avg_realized_return = float(selected.mean()) if len(selected) else 0.0
    return {
        "f1": f1,
        "balanced_accuracy": bal_acc,
        "positive_rate": float(predictions.mean()),
        "avg_realized_return": avg_realized_return,
    }


def regime_candidates(train_frame, eval_frame, side: str) -> dict[str, np.ndarray]:
    vol_median = float(train_frame["volatility_7"].median())
    drawdown_q35 = float(train_frame["drawdown_7"].quantile(0.35))
    ret7_q65 = float(train_frame["ret_7"].quantile(0.65))
    ret7_q35 = float(train_frame["ret_7"].quantile(0.35))
    rsi_q40 = float(train_frame["rsi_7"].quantile(0.40))
    rsi_q60 = float(train_frame["rsi_7"].quantile(0.60))

    uptrend = (eval_frame["sma_gap_5"] > 0.0) & (eval_frame["sma_gap_10"] > 0.0)
    downtrend = (eval_frame["sma_gap_5"] < 0.0) & (eval_frame["sma_gap_10"] < 0.0)
    low_vol = eval_frame["volatility_7"] <= vol_median
    high_vol = eval_frame["volatility_7"] >= vol_median
    oversold = (eval_frame["drawdown_7"] <= drawdown_q35) & (eval_frame["rsi_7"] <= rsi_q40) & (eval_frame["ret_7"] <= ret7_q35)
    overbought = (eval_frame["ret_7"] >= ret7_q65) & (eval_frame["rsi_7"] >= rsi_q60)

    if side == "long":
        return {
            "no_filter": np.ones(len(eval_frame), dtype=bool),
            "uptrend_only": uptrend.to_numpy(dtype=bool),
            "uptrend_low_vol": (uptrend & low_vol).to_numpy(dtype=bool),
            "oversold_rebound": oversold.to_numpy(dtype=bool),
        }
    return {
        "no_filter": np.ones(len(eval_frame), dtype=bool),
        "downtrend_only": downtrend.to_numpy(dtype=bool),
        "downtrend_high_vol": (downtrend & high_vol).to_numpy(dtype=bool),
        "overbought_fade": overbought.to_numpy(dtype=bool),
    }


def evaluate_regimes(frame, side: str) -> list[RegimeResult]:
    feature_names = assemble_feature_names(frame.columns.tolist())
    windows = build_window_slices(len(frame))
    target_column = get_target_column(side)
    realized_return_column = get_realized_return_column(side)

    aggregates: dict[str, dict[str, list[float]]] = {}
    for window in windows:
        train_frame = frame.iloc[window.start:window.train_end].copy()
        validation_frame = frame.iloc[window.train_end:window.valid_end].copy()
        test_frame = frame.iloc[window.valid_end:window.test_end].copy()

        train_x = train_frame[feature_names].to_numpy(dtype=np.float32)
        validation_x = validation_frame[feature_names].to_numpy(dtype=np.float32)
        test_x = test_frame[feature_names].to_numpy(dtype=np.float32)
        train_y = train_frame[target_column].to_numpy(dtype=np.float32)
        validation_y = validation_frame[target_column].to_numpy(dtype=np.float32)
        test_y = test_frame[target_column].to_numpy(dtype=np.float32)

        model = fit_window_model(train_x, validation_x, train_y, validation_y, feature_names, side)

        mean = train_x.mean(axis=0, keepdims=True)
        std = train_x.std(axis=0, keepdims=True)
        std = np.where(std < 1e-6, 1.0, std)
        scaled_train = (train_x - mean) / std
        scaled_validation = (validation_x - mean) / std
        scaled_test = (test_x - mean) / std
        from train import add_interaction_terms, add_bias
        scaled_train, scaled_validation, scaled_test = add_interaction_terms(
            scaled_train,
            scaled_validation,
            scaled_test,
            feature_names,
        )
        scaled_validation = add_bias(scaled_validation)
        scaled_test = add_bias(scaled_test)

        validation_logits = scaled_validation @ model.weights
        test_logits = scaled_test @ model.weights
        validation_regimes = regime_candidates(train_frame, validation_frame, side)
        test_regimes = regime_candidates(train_frame, test_frame, side)

        for regime_name, validation_gate in validation_regimes.items():
            test_gate = test_regimes[regime_name]
            validation_metrics = compute_metrics_with_gate(
                validation_logits,
                validation_y,
                validation_frame[realized_return_column].to_numpy(dtype=np.float32),
                model.threshold,
                validation_gate,
            )
            test_metrics = compute_metrics_with_gate(
                test_logits,
                test_y,
                test_frame[realized_return_column].to_numpy(dtype=np.float32),
                model.threshold,
                test_gate,
            )
            bucket = aggregates.setdefault(
                regime_name,
                {"validation_bal": [], "test_bal": [], "test_positive_rate": []},
            )
            bucket["validation_bal"].append(float(validation_metrics["balanced_accuracy"]))
            bucket["test_bal"].append(float(test_metrics["balanced_accuracy"]))
            bucket["test_positive_rate"].append(float(test_metrics["positive_rate"]))

    results: list[RegimeResult] = []
    for name, values in aggregates.items():
        results.append(
            RegimeResult(
                name=name,
                avg_validation_bal=float(np.mean(values["validation_bal"])),
                avg_test_bal=float(np.mean(values["test_bal"])),
                avg_test_positive_rate=float(np.mean(values["test_positive_rate"])),
                min_test_bal=float(np.min(values["test_bal"])),
                max_test_bal=float(np.max(values["test_bal"])),
            )
        )
    results.sort(key=lambda item: item.avg_test_bal, reverse=True)
    return results


def main() -> None:
    seed = get_env_int("AR_SEED", 42)
    side = get_side()
    set_seed(seed)
    frame = load_dataset_frame()
    results = evaluate_regimes(frame, side)
    config = get_runtime_config()
    symbol = ac.get_asset_symbol()

    print("---")
    print(f"task:                 {symbol}_{int(config['horizon_bars'])}bar_regime_filter_{side}")
    for item in results:
        print(
            f"{item.name:20} "
            f"avg_validation_bal={item.avg_validation_bal:.4f} "
            f"avg_test_bal={item.avg_test_bal:.4f} "
            f"avg_test_pos_rate={item.avg_test_positive_rate:.4f} "
            f"min_test_bal={item.min_test_bal:.4f} "
            f"max_test_bal={item.max_test_bal:.4f}"
        )


if __name__ == "__main__":
    main()
