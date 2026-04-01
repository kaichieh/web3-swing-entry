"""
Score the latest available crypto bar and combine long/short models into one live decision.
"""

from __future__ import annotations

import json

import numpy as np

import asset_config as ac
import train as tr
from prepare import add_cross_asset_context_features, add_price_features, download_symbol_prices

NO_TRADE_MARGIN = 0.03
WEAK_SIGNAL_MARGIN = 0.02
STRONG_SIGNAL_MARGIN = 0.06


def score_row(feature_names: list[str], train_frame, latest_row) -> tuple[np.ndarray, dict[str, float]]:
    train_x = train_frame[feature_names].to_numpy(dtype=np.float32)
    latest_x = latest_row[feature_names].to_numpy(dtype=np.float32)
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    standardized_latest = (latest_x - mean) / std
    _, _, latest_augmented = tr.add_interaction_terms(train_x[:1], train_x[:1], standardized_latest, feature_names)
    latest_augmented = tr.add_bias(latest_augmented)
    raw_snapshot = {name: float(latest_row.iloc[0][name]) for name in feature_names}
    return latest_augmented, raw_snapshot


def classify_side(probability: float, threshold: float) -> str:
    gap = probability - threshold
    if gap >= STRONG_SIGNAL_MARGIN:
        return "strong"
    if gap >= WEAK_SIGNAL_MARGIN:
        return "weak"
    return "below_threshold"


def short_regime_gate(asset_key: str, snapshot: dict[str, float]) -> tuple[bool, str]:
    if asset_key != "eth":
        return True, "no_short_regime_gate_for_asset"
    sma_gap_5 = float(snapshot.get("sma_gap_5", 0.0))
    sma_gap_10 = float(snapshot.get("sma_gap_10", 0.0))
    is_downtrend = sma_gap_5 < 0.0 and sma_gap_10 < 0.0
    if is_downtrend:
        return True, "eth_short_downtrend_gate_passed"
    return False, "eth_short_blocked_outside_downtrend"


def choose_final_signal(
    asset_key: str,
    long_probability: float,
    long_threshold: float,
    short_probability: float,
    short_threshold: float,
    snapshot: dict[str, float],
) -> tuple[str, dict[str, float | str]]:
    long_gap = long_probability - long_threshold
    short_gap = short_probability - short_threshold
    margin = tr.get_env_float("AR_NO_TRADE_MARGIN", NO_TRADE_MARGIN)
    short_gate_passed, short_gate_reason = short_regime_gate(asset_key, snapshot)
    effective_short_gap = short_gap if short_gate_passed else min(short_gap, 0.0)

    if long_gap <= 0 and effective_short_gap <= 0:
        reason = "neither_side_clears_threshold"
        if short_gap > 0 and not short_gate_passed:
            reason = "short_blocked_by_regime_gate"
        return "no_trade", {
            "reason": reason,
            "decision_margin": round(margin, 4),
            "short_regime_gate": short_gate_reason,
        }
    if long_gap > 0 and effective_short_gap <= 0:
        reason = "only_long_clears_threshold"
        if short_gap > 0 and not short_gate_passed:
            reason = "long_wins_after_short_regime_block"
        return "long", {
            "reason": reason,
            "decision_margin": round(margin, 4),
            "short_regime_gate": short_gate_reason,
        }
    if effective_short_gap > 0 and long_gap <= 0:
        return "short", {
            "reason": "only_short_clears_threshold",
            "decision_margin": round(margin, 4),
            "short_regime_gate": short_gate_reason,
        }
    if abs(long_gap - effective_short_gap) < margin:
        return "no_trade", {
            "reason": "dual_signal_conflict_inside_margin",
            "decision_margin": round(margin, 4),
            "short_regime_gate": short_gate_reason,
        }
    if long_gap > effective_short_gap:
        return "long", {
            "reason": "long_gap_exceeds_short_gap",
            "decision_margin": round(margin, 4),
            "short_regime_gate": short_gate_reason,
        }
    return "short", {
        "reason": "short_gap_exceeds_long_gap",
        "decision_margin": round(margin, 4),
        "short_regime_gate": short_gate_reason,
    }


def build_model_rationale(snapshot: dict[str, float], side: str) -> list[str]:
    reasons: list[str] = []
    rsi_7 = float(snapshot.get("rsi_7", 50.0))
    drawdown_7 = float(snapshot.get("drawdown_7", 0.0))
    breakout_7 = float(snapshot.get("breakout_7", 0.0))
    volume_vs_7 = float(snapshot.get("volume_vs_7", 0.0))
    ret_7 = float(snapshot.get("ret_7", 0.0))

    if side == "long":
        if rsi_7 < 35:
            reasons.append("rsi_7 is in a softer zone for long mean reversion")
        if drawdown_7 <= -0.04:
            reasons.append("recent 7-bar drawdown is deep enough to support a rebound setup")
        if breakout_7 >= 1.0:
            reasons.append("price is also breaking above the recent 7-bar range")
        if volume_vs_7 >= 0.2:
            reasons.append("volume is running above its 7-bar average")
        if ret_7 <= -0.03:
            reasons.append("recent weekly return is weak enough to create a bounce candidate")
    else:
        if rsi_7 > 65:
            reasons.append("rsi_7 is stretched enough to support a short fade")
        if drawdown_7 >= -0.01:
            reasons.append("drawdown remains shallow, so downside room may still exist")
        if breakout_7 >= 1.0:
            reasons.append("price is extended above the recent 7-bar range")
        if volume_vs_7 >= 0.2:
            reasons.append("volume is elevated while the move is stretched")
        if ret_7 >= 0.04:
            reasons.append("recent weekly return is strong enough to consider a reversal short")

    if not reasons:
        reasons.append("feature snapshot is mixed, so the model is relying on the combined pattern")
    return reasons


def main() -> None:
    tr.set_seed(tr.get_env_int("AR_SEED", tr.SEED))
    asset_key = ac.get_asset_key()
    raw_prices = download_symbol_prices(asset_key)
    live_features = add_price_features(raw_prices)
    live_features = add_cross_asset_context_features(live_features, asset_key)

    long_model, long_state = tr.fit_model("long")
    short_model, short_state = tr.fit_model("short")
    long_feature_names = list(long_model.feature_names)
    short_feature_names = list(short_model.feature_names)
    required_feature_names = sorted(set(long_feature_names) | set(short_feature_names))

    latest_live = live_features.dropna(subset=required_feature_names).iloc[[-1]].copy()
    long_vector, latest_snapshot = score_row(long_feature_names, long_state["splits"]["train"].frame, latest_live)
    short_vector, _ = score_row(short_feature_names, short_state["splits"]["train"].frame, latest_live)

    long_probability = float(tr.sigmoid(long_vector @ long_model.weights)[0])
    short_probability = float(tr.sigmoid(short_vector @ short_model.weights)[0])
    final_signal, decision_info = choose_final_signal(
        asset_key,
        long_probability,
        float(long_model.threshold),
        short_probability,
        float(short_model.threshold),
        latest_snapshot,
    )

    output = {
        "asset_key": asset_key,
        "symbol": ac.get_asset_symbol(asset_key),
        "latest_raw_date": latest_live["date"].iloc[0].strftime("%Y-%m-%d"),
        "latest_close": round(float(latest_live["close"].iloc[0]), 2),
        "signal_summary": {
            "signal": final_signal,
            "holding_horizon_bars": int(ac.load_asset_config(asset_key)["horizon_bars"]),
            "decision_reason": str(decision_info["reason"]),
            "decision_margin": float(decision_info["decision_margin"]),
            "short_regime_gate": str(decision_info["short_regime_gate"]),
            "confidence_gap": round(abs((long_probability - long_model.threshold) - (short_probability - short_model.threshold)), 4),
        },
        "long_summary": {
            "predicted_probability": round(long_probability, 4),
            "decision_threshold": round(float(long_model.threshold), 4),
            "signal_strength": classify_side(long_probability, float(long_model.threshold)),
            "model_reasons": build_model_rationale(latest_snapshot, "long"),
        },
        "short_summary": {
            "predicted_probability": round(short_probability, 4),
            "decision_threshold": round(float(short_model.threshold), 4),
            "signal_strength": classify_side(short_probability, float(short_model.threshold)),
            "regime_gate_passed": bool(short_regime_gate(asset_key, latest_snapshot)[0]),
            "model_reasons": build_model_rationale(latest_snapshot, "short"),
        },
        "latest_feature_snapshot": {
            key: round(value, 4)
            for key, value in latest_snapshot.items()
            if key in {"ret_3", "ret_7", "drawdown_7", "volume_vs_7", "rsi_7", "volatility_7", "sma_gap_5", "sma_gap_10"}
        },
    }
    ac.get_latest_prediction_path(asset_key).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
