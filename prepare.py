"""
Prepare short-horizon crypto swing-entry data with mirrored long/short labels.

Default labels:
- long target: +upside barrier before downside barrier within 7 bars
- short target: -downside barrier before upside barrier within 7 bars
- neutral rows are dropped by default
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

import asset_config as ac

TRAIN_FRACTION = 0.70
VALID_FRACTION = 0.15

LONG_TARGET_COLUMN = "target_long_hit_up_first"
SHORT_TARGET_COLUMN = "target_short_hit_down_first"

FEATURE_COLUMNS = [
    "ret_1",
    "ret_2",
    "ret_3",
    "ret_5",
    "ret_7",
    "sma_gap_3",
    "sma_gap_5",
    "sma_gap_10",
    "volatility_3",
    "volatility_7",
    "range_pct",
    "volume_change_1",
    "volume_vs_7",
    "breakout_7",
    "drawdown_7",
    "rsi_7",
    "overnight_gap",
    "intraday_return",
    "upper_shadow",
    "lower_shadow",
]

EXPERIMENTAL_FEATURE_COLUMNS = [
    "ret_10",
    "ret_14",
    "sma_gap_14",
    "volatility_10",
    "funding_rate_1d",
    "funding_rate_3d_mean",
    "funding_rate_7d_mean",
    "funding_rate_z_7",
    "open_interest",
    "open_interest_value",
    "open_interest_change_1",
    "open_interest_change_3",
    "open_interest_z_7",
    "taker_buy_sell_ratio",
    "taker_buy_volume",
    "taker_sell_volume",
    "taker_flow_3d_mean",
    "range_z_7",
    "inside_bar",
    "outside_bar",
    "gap_up_flag",
    "gap_down_flag",
    "weekend_gap",
    "weekend_return",
    "weekend_range_pct",
    "up_day_ratio_7",
    "close_location_7",
    "atr_pct_7",
]


@dataclass(frozen=True)
class DatasetSplit:
    features: np.ndarray
    labels: np.ndarray
    frame: pd.DataFrame


def get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


def get_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value is not None else default


def get_env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value is not None and value.strip() else default


def get_runtime_config(asset_key: str | None = None) -> dict[str, object]:
    config = ac.load_asset_config(asset_key)
    return {
        "asset_key": str(config["asset_key"]),
        "symbol": str(config["symbol"]),
        "binance_symbol": str(config["binance_symbol"]),
        "binance_start_date": str(config["binance_start_date"]),
        "horizon_bars": get_env_int("AR_HORIZON_BARS", int(config["horizon_bars"])),
        "long_up_barrier": get_env_float("AR_LONG_UP_BARRIER", float(config["long_up_barrier"])),
        "long_down_barrier": get_env_float("AR_LONG_DOWN_BARRIER", float(config["long_down_barrier"])),
        "short_down_barrier": get_env_float("AR_SHORT_DOWN_BARRIER", float(config["short_down_barrier"])),
        "short_up_barrier": get_env_float("AR_SHORT_UP_BARRIER", float(config["short_up_barrier"])),
        "label_mode": get_env_str("AR_LABEL_MODE", str(config["label_mode"])),
    }


def ensure_cache_dir(asset_key: str | None = None) -> None:
    ac.get_cache_dir(asset_key).mkdir(parents=True, exist_ok=True)


def normalize_ohlcv_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized.columns = [column.lower() for column in normalized.columns]
    expected = ["date", "open", "high", "low", "close", "volume"]
    missing = [column for column in expected if column not in normalized.columns]
    if missing:
        raise RuntimeError(f"Downloaded dataset missing columns: {missing}")
    normalized = normalized[expected].copy()
    normalized["date"] = pd.to_datetime(normalized["date"])
    normalized = normalized.sort_values("date").drop_duplicates(subset="date", keep="last").reset_index(drop=True)
    return normalized


def normalize_spot_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized.columns = [column.lower() for column in normalized.columns]
    expected = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "trade_count",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ]
    missing = [column for column in expected if column not in normalized.columns]
    if missing:
        raise RuntimeError(f"Downloaded Binance spot dataset missing columns: {missing}")
    normalized = normalized[expected].copy()
    normalized["date"] = pd.to_datetime(normalized["date"])
    numeric_columns = [column for column in expected if column != "date"]
    for column in numeric_columns:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized = normalized.dropna(subset=["open", "high", "low", "close", "volume"])
    normalized = normalized.sort_values("date").drop_duplicates(subset="date", keep="last").reset_index(drop=True)
    return normalized


def binance_klines_url(symbol: str, start_time_ms: int | None = None, limit: int = 1000) -> str:
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1d&limit={limit}"
    if start_time_ms is not None:
        url += f"&startTime={start_time_ms}"
    return url


def binance_funding_url(symbol: str, start_time_ms: int | None = None, limit: int = 1000) -> str:
    url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&limit={limit}"
    if start_time_ms is not None:
        url += f"&startTime={start_time_ms}"
    return url


def binance_open_interest_hist_url(symbol: str, start_time_ms: int | None = None, limit: int = 500) -> str:
    url = f"https://fapi.binance.com/futures/data/openInterestHist?symbol={symbol}&period=1d&limit={limit}"
    if start_time_ms is not None:
        url += f"&startTime={start_time_ms}"
    return url


def binance_taker_flow_url(symbol: str, start_time_ms: int | None = None, limit: int = 500) -> str:
    url = f"https://fapi.binance.com/futures/data/takerlongshortRatio?symbol={symbol}&period=1d&limit={limit}"
    if start_time_ms is not None:
        url += f"&startTime={start_time_ms}"
    return url


def fetch_text(url: str, *, accept_json: bool = False) -> str:
    headers = {"User-Agent": "Mozilla/5.0"}
    if accept_json:
        headers["Accept"] = "application/json"
    request = Request(url, headers=headers)
    with urlopen(request, timeout=30) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset)


def fetch_json(url: str) -> object:
    return json.loads(fetch_text(url, accept_json=True))


def download_prices_from_binance(symbol: str, start_date: str) -> pd.DataFrame:
    start_time_ms = int(datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc).timestamp() * 1000)
    now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    rows: list[list[object]] = []

    while start_time_ms < now_ms:
        payload = fetch_json(binance_klines_url(symbol, start_time_ms=start_time_ms, limit=1000))
        if not isinstance(payload, list) or not payload:
            break
        rows.extend(payload)
        last_open_time = int(payload[-1][0])
        next_open_time = last_open_time + 24 * 60 * 60 * 1000
        if next_open_time <= start_time_ms:
            break
        start_time_ms = next_open_time
        if len(payload) < 1000:
            break

    if not rows:
        raise RuntimeError(f"Binance spot kline dataset for {symbol} is empty.")

    frame = pd.DataFrame(
        [
            {
                "date": pd.to_datetime(int(row[0]), unit="ms", utc=True).tz_localize(None),
                "open": row[1],
                "high": row[2],
                "low": row[3],
                "close": row[4],
                "volume": row[5],
                "quote_asset_volume": row[7],
                "trade_count": row[8],
                "taker_buy_base_volume": row[9],
                "taker_buy_quote_volume": row[10],
            }
            for row in rows
        ]
    )
    return normalize_spot_frame(frame)


def normalize_funding_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized.columns = [column.lower() for column in normalized.columns]
    expected = ["date", "funding_rate", "mark_price"]
    missing = [column for column in expected if column not in normalized.columns]
    if missing:
        raise RuntimeError(f"Downloaded Binance funding dataset missing columns: {missing}")
    normalized = normalized[expected].copy()
    normalized["date"] = pd.to_datetime(normalized["date"])
    normalized["funding_rate"] = pd.to_numeric(normalized["funding_rate"], errors="coerce")
    normalized["mark_price"] = pd.to_numeric(normalized["mark_price"], errors="coerce")
    normalized = normalized.dropna(subset=["date", "funding_rate"])
    normalized = normalized.sort_values("date").reset_index(drop=True)
    return normalized


def download_funding_from_binance(symbol: str, start_date: str) -> pd.DataFrame:
    start_time_ms = int(datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc).timestamp() * 1000)
    now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    rows: list[dict[str, object]] = []

    while start_time_ms < now_ms:
        payload = fetch_json(binance_funding_url(symbol, start_time_ms=start_time_ms, limit=1000))
        if not isinstance(payload, list) or not payload:
            break
        rows.extend(payload)
        last_funding_time = int(payload[-1]["fundingTime"])
        next_funding_time = last_funding_time + 1
        if next_funding_time <= start_time_ms:
            break
        start_time_ms = next_funding_time
        if len(payload) < 1000:
            break

    if not rows:
        raise RuntimeError(f"Binance funding-rate dataset for {symbol} is empty.")

    frame = pd.DataFrame(
        [
            {
                "date": pd.to_datetime(int(row["fundingTime"]), unit="ms", utc=True).tz_localize(None).normalize(),
                "funding_rate": row["fundingRate"],
                "mark_price": row.get("markPrice"),
            }
            for row in rows
        ]
    )
    return normalize_funding_frame(frame)


def normalize_open_interest_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized.columns = [column.lower() for column in normalized.columns]
    expected = ["date", "open_interest", "open_interest_value"]
    missing = [column for column in expected if column not in normalized.columns]
    if missing:
        raise RuntimeError(f"Downloaded Binance open-interest dataset missing columns: {missing}")
    normalized = normalized[expected].copy()
    normalized["date"] = pd.to_datetime(normalized["date"])
    normalized["open_interest"] = pd.to_numeric(normalized["open_interest"], errors="coerce")
    normalized["open_interest_value"] = pd.to_numeric(normalized["open_interest_value"], errors="coerce")
    normalized = normalized.dropna(subset=["date", "open_interest"])
    normalized = normalized.sort_values("date").drop_duplicates(subset="date", keep="last").reset_index(drop=True)
    return normalized


def download_open_interest_from_binance(symbol: str) -> pd.DataFrame:
    payload = fetch_json(binance_open_interest_hist_url(symbol, limit=500))
    if not isinstance(payload, list) or not payload:
        raise RuntimeError(f"Binance open-interest history dataset for {symbol} is empty.")
    frame = pd.DataFrame(
        [
            {
                "date": pd.to_datetime(int(row["timestamp"]), unit="ms", utc=True).tz_localize(None).normalize(),
                "open_interest": row["sumOpenInterest"],
                "open_interest_value": row["sumOpenInterestValue"],
            }
            for row in payload
        ]
    )
    return normalize_open_interest_frame(frame)


def normalize_taker_flow_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized.columns = [column.lower() for column in normalized.columns]
    expected = ["date", "taker_buy_sell_ratio", "taker_buy_volume", "taker_sell_volume"]
    missing = [column for column in expected if column not in normalized.columns]
    if missing:
        raise RuntimeError(f"Downloaded Binance taker-flow dataset missing columns: {missing}")
    normalized = normalized[expected].copy()
    normalized["date"] = pd.to_datetime(normalized["date"])
    for column in ["taker_buy_sell_ratio", "taker_buy_volume", "taker_sell_volume"]:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized = normalized.dropna(subset=["date", "taker_buy_sell_ratio"])
    normalized = normalized.sort_values("date").drop_duplicates(subset="date", keep="last").reset_index(drop=True)
    return normalized


def download_taker_flow_from_binance(symbol: str) -> pd.DataFrame:
    payload = fetch_json(binance_taker_flow_url(symbol, limit=500))
    if not isinstance(payload, list) or not payload:
        raise RuntimeError(f"Binance taker-flow dataset for {symbol} is empty.")
    frame = pd.DataFrame(
        [
            {
                "date": pd.to_datetime(int(row["timestamp"]), unit="ms", utc=True).tz_localize(None).normalize(),
                "taker_buy_sell_ratio": row["buySellRatio"],
                "taker_buy_volume": row["buyVol"],
                "taker_sell_volume": row["sellVol"],
            }
            for row in payload
        ]
    )
    return normalize_taker_flow_frame(frame)


def download_symbol_prices(asset_key: str | None = None) -> pd.DataFrame:
    key = asset_key or ac.get_asset_key()
    config = get_runtime_config(key)
    binance_symbol = str(config["binance_symbol"])
    start_date = str(config["binance_start_date"])
    cache_path = ac.get_spot_data_path(key)
    ensure_cache_dir(key)
    try:
        frame = download_prices_from_binance(binance_symbol, start_date)
    except (HTTPError, URLError, TimeoutError, ValueError, KeyError, IndexError, TypeError):
        if not cache_path.exists():
            raise
        frame = pd.read_csv(cache_path)
        if frame.empty:
            raise RuntimeError(f"Cached {binance_symbol} dataset is empty.")
        frame = normalize_spot_frame(frame)
    frame.to_csv(cache_path, index=False)
    return frame


def download_funding_rates(asset_key: str | None = None) -> pd.DataFrame:
    key = asset_key or ac.get_asset_key()
    config = get_runtime_config(key)
    binance_symbol = str(config["binance_symbol"])
    start_date = str(config["binance_start_date"])
    cache_path = ac.get_funding_data_path(key)
    ensure_cache_dir(key)
    try:
        frame = download_funding_from_binance(binance_symbol, start_date)
    except (HTTPError, URLError, TimeoutError, ValueError, KeyError, IndexError, TypeError):
        if not cache_path.exists():
            raise
        frame = pd.read_csv(cache_path)
        if frame.empty:
            raise RuntimeError(f"Cached funding-rate dataset for {binance_symbol} is empty.")
        frame = normalize_funding_frame(frame)
    frame.to_csv(cache_path, index=False)
    return frame


def download_open_interest(asset_key: str | None = None) -> pd.DataFrame:
    key = asset_key or ac.get_asset_key()
    config = get_runtime_config(key)
    binance_symbol = str(config["binance_symbol"])
    cache_path = ac.get_open_interest_data_path(key)
    ensure_cache_dir(key)
    try:
        frame = download_open_interest_from_binance(binance_symbol)
    except (HTTPError, URLError, TimeoutError, ValueError, KeyError, IndexError, TypeError):
        if not cache_path.exists():
            raise
        frame = pd.read_csv(cache_path)
        if frame.empty:
            raise RuntimeError(f"Cached open-interest dataset for {binance_symbol} is empty.")
        frame = normalize_open_interest_frame(frame)
    frame.to_csv(cache_path, index=False)
    return frame


def download_taker_flow(asset_key: str | None = None) -> pd.DataFrame:
    key = asset_key or ac.get_asset_key()
    config = get_runtime_config(key)
    binance_symbol = str(config["binance_symbol"])
    cache_path = ac.get_taker_flow_data_path(key)
    ensure_cache_dir(key)
    try:
        frame = download_taker_flow_from_binance(binance_symbol)
    except (HTTPError, URLError, TimeoutError, ValueError, KeyError, IndexError, TypeError):
        if not cache_path.exists():
            raise
        frame = pd.read_csv(cache_path)
        if frame.empty:
            raise RuntimeError(f"Cached taker-flow dataset for {binance_symbol} is empty.")
        frame = normalize_taker_flow_frame(frame)
    frame.to_csv(cache_path, index=False)
    return frame


def build_barrier_labels(
    df: pd.DataFrame,
    horizon_bars: int,
    up_barrier: float,
    down_barrier: float,
) -> tuple[np.ndarray, np.ndarray]:
    closes = df["close"].to_numpy(dtype=np.float64)
    highs = df["high"].to_numpy(dtype=np.float64)
    lows = df["low"].to_numpy(dtype=np.float64)
    labels = np.full(len(df), np.nan, dtype=np.float64)
    realized_returns = np.full(len(df), np.nan, dtype=np.float64)

    for idx in range(len(df)):
        entry = closes[idx]
        end = min(len(df), idx + horizon_bars + 1)
        if idx + 1 >= end:
            continue
        future_highs = highs[idx + 1 : end] / entry - 1.0
        future_lows = lows[idx + 1 : end] / entry - 1.0
        future_closes = closes[idx + 1 : end] / entry - 1.0
        realized_returns[idx] = future_closes[-1]

        hit_up = np.where(future_highs >= up_barrier)[0]
        hit_down = np.where(future_lows <= down_barrier)[0]
        up_idx = int(hit_up[0]) if hit_up.size else None
        down_idx = int(hit_down[0]) if hit_down.size else None

        if up_idx is None and down_idx is None:
            continue
        if up_idx is not None and down_idx is None:
            labels[idx] = 1.0
            continue
        if down_idx is not None and up_idx is None:
            labels[idx] = 0.0
            continue
        if up_idx < down_idx:
            labels[idx] = 1.0
        elif down_idx < up_idx:
            labels[idx] = 0.0
        else:
            labels[idx] = np.nan

    return labels, realized_returns


def add_weekend_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    weekday = out["date"].dt.dayofweek
    monday_mask = weekday == 0
    saturday_mask = weekday == 5
    sunday_mask = weekday == 6
    prev_close = out["close"].shift(1)

    out["weekend_gap"] = np.where(monday_mask, out["open"] / prev_close - 1.0, 0.0)
    out["weekend_return"] = np.where(sunday_mask, out["close"].pct_change(2), 0.0)
    out["weekend_range_pct"] = np.where(
        saturday_mask | sunday_mask,
        (out["high"] - out["low"]) / out["close"].replace(0, np.nan),
        0.0,
    )
    return out


def add_price_features(frame: pd.DataFrame) -> pd.DataFrame:
    df = normalize_ohlcv_frame(frame)

    close = df["close"]
    open_price = df["open"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"].replace(0, np.nan)

    df["ret_1"] = close.pct_change(1)
    df["ret_2"] = close.pct_change(2)
    df["ret_3"] = close.pct_change(3)
    df["ret_5"] = close.pct_change(5)
    df["ret_7"] = close.pct_change(7)
    df["ret_10"] = close.pct_change(10)
    df["ret_14"] = close.pct_change(14)

    df["sma_gap_3"] = close / close.rolling(3).mean() - 1.0
    df["sma_gap_5"] = close / close.rolling(5).mean() - 1.0
    df["sma_gap_10"] = close / close.rolling(10).mean() - 1.0
    df["sma_gap_14"] = close / close.rolling(14).mean() - 1.0

    df["volatility_3"] = df["ret_1"].rolling(3).std()
    df["volatility_7"] = df["ret_1"].rolling(7).std()
    df["volatility_10"] = df["ret_1"].rolling(10).std()
    df["range_pct"] = (high - low) / close
    df["volume_change_1"] = volume.pct_change(1)
    df["volume_vs_7"] = volume / volume.rolling(7).mean() - 1.0

    df["breakout_7"] = (close > close.shift(1).rolling(7).max()).astype(float)
    df["drawdown_7"] = (close - close.rolling(7).max()) / close.rolling(7).max()

    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(7).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(7).mean()
    rs = gain / (loss + 1e-10)
    df["rsi_7"] = 100 - (100 / (1 + rs))

    prev_close = close.shift(1)
    eps = 1e-6
    df["overnight_gap"] = open_price / prev_close - 1.0
    df["intraday_return"] = (close - open_price) / np.maximum(open_price.to_numpy(dtype=np.float64), eps)
    df["upper_shadow"] = (
        high.to_numpy(dtype=np.float64)
        - np.maximum(open_price.to_numpy(dtype=np.float64), close.to_numpy(dtype=np.float64))
    ) / np.maximum(close.to_numpy(dtype=np.float64), eps)
    df["lower_shadow"] = (
        np.minimum(open_price.to_numpy(dtype=np.float64), close.to_numpy(dtype=np.float64))
        - low.to_numpy(dtype=np.float64)
    ) / np.maximum(close.to_numpy(dtype=np.float64), eps)

    df["range_z_7"] = (df["range_pct"] - df["range_pct"].rolling(7).mean()) / (df["range_pct"].rolling(7).std() + 1e-10)
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_open = open_price.shift(1)
    prev_close = close.shift(1)
    prev_body_high = np.maximum(prev_open, prev_close)
    prev_body_low = np.minimum(prev_open, prev_close)
    df["inside_bar"] = ((high <= prev_high) & (low >= prev_low)).astype(float)
    df["outside_bar"] = ((high >= prev_high) & (low <= prev_low)).astype(float)
    df["gap_up_flag"] = (open_price > prev_body_high).astype(float)
    df["gap_down_flag"] = (open_price < prev_body_low).astype(float)

    rolling_high_7 = high.rolling(7).max()
    rolling_low_7 = low.rolling(7).min()
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["up_day_ratio_7"] = (df["ret_1"] > 0).astype(float).rolling(7).mean()
    df["close_location_7"] = (close - rolling_low_7) / (rolling_high_7 - rolling_low_7 + eps)
    df["atr_pct_7"] = true_range.rolling(7).mean() / close

    df = add_weekend_features(df)
    return df


def add_funding_features(frame: pd.DataFrame, funding_frame: pd.DataFrame | None) -> pd.DataFrame:
    df = frame.copy()
    if funding_frame is None or funding_frame.empty:
        df["funding_rate_1d"] = 0.0
        df["funding_rate_3d_mean"] = 0.0
        df["funding_rate_7d_mean"] = 0.0
        df["funding_rate_z_7"] = 0.0
        return df

    daily_funding = (
        funding_frame.groupby("date", as_index=False)
        .agg(
            funding_rate_1d=("funding_rate", "sum"),
            funding_rate_mean=("funding_rate", "mean"),
            funding_prints=("funding_rate", "count"),
            funding_mark_price=("mark_price", "mean"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )
    daily_funding["funding_rate_3d_mean"] = daily_funding["funding_rate_1d"].rolling(3).mean()
    daily_funding["funding_rate_7d_mean"] = daily_funding["funding_rate_1d"].rolling(7).mean()
    daily_funding["funding_rate_z_7"] = (
        daily_funding["funding_rate_1d"] - daily_funding["funding_rate_1d"].rolling(7).mean()
    ) / (daily_funding["funding_rate_1d"].rolling(7).std() + 1e-10)

    merged = df.merge(
        daily_funding[
            [
                "date",
                "funding_rate_1d",
                "funding_rate_3d_mean",
                "funding_rate_7d_mean",
                "funding_rate_z_7",
            ]
        ],
        on="date",
        how="left",
    )
    for column in ["funding_rate_1d", "funding_rate_3d_mean", "funding_rate_7d_mean", "funding_rate_z_7"]:
        merged[column] = merged[column].fillna(0.0)
    return merged


def add_open_interest_features(frame: pd.DataFrame, open_interest_frame: pd.DataFrame | None) -> pd.DataFrame:
    df = frame.copy()
    if open_interest_frame is None or open_interest_frame.empty:
        df["open_interest"] = 0.0
        df["open_interest_value"] = 0.0
        df["open_interest_change_1"] = 0.0
        df["open_interest_change_3"] = 0.0
        df["open_interest_z_7"] = 0.0
        return df

    oi = open_interest_frame.sort_values("date").reset_index(drop=True).copy()
    oi["open_interest_change_1"] = oi["open_interest"].pct_change(1)
    oi["open_interest_change_3"] = oi["open_interest"].pct_change(3)
    oi["open_interest_z_7"] = (
        oi["open_interest"] - oi["open_interest"].rolling(7).mean()
    ) / (oi["open_interest"].rolling(7).std() + 1e-10)

    merged = df.merge(
        oi[
            [
                "date",
                "open_interest",
                "open_interest_value",
                "open_interest_change_1",
                "open_interest_change_3",
                "open_interest_z_7",
            ]
        ],
        on="date",
        how="left",
    )
    for column in [
        "open_interest",
        "open_interest_value",
        "open_interest_change_1",
        "open_interest_change_3",
        "open_interest_z_7",
    ]:
        merged[column] = merged[column].fillna(0.0)
    return merged


def add_taker_flow_features(frame: pd.DataFrame, taker_flow_frame: pd.DataFrame | None) -> pd.DataFrame:
    df = frame.copy()
    if taker_flow_frame is None or taker_flow_frame.empty:
        df["taker_buy_sell_ratio"] = 0.0
        df["taker_buy_volume"] = 0.0
        df["taker_sell_volume"] = 0.0
        df["taker_flow_3d_mean"] = 0.0
        return df

    taker = taker_flow_frame.sort_values("date").reset_index(drop=True).copy()
    taker["taker_flow_3d_mean"] = taker["taker_buy_sell_ratio"].rolling(3).mean()

    merged = df.merge(
        taker[
            [
                "date",
                "taker_buy_sell_ratio",
                "taker_buy_volume",
                "taker_sell_volume",
                "taker_flow_3d_mean",
            ]
        ],
        on="date",
        how="left",
    )
    for column in ["taker_buy_sell_ratio", "taker_buy_volume", "taker_sell_volume", "taker_flow_3d_mean"]:
        merged[column] = merged[column].fillna(0.0)
    return merged


def add_features(frame: pd.DataFrame, asset_key: str | None = None) -> pd.DataFrame:
    config = get_runtime_config(asset_key)
    df = add_price_features(frame)
    funding_frame = download_funding_rates(asset_key)
    open_interest_frame = download_open_interest(asset_key)
    taker_flow_frame = download_taker_flow(asset_key)
    df = add_funding_features(df, funding_frame)
    df = add_open_interest_features(df, open_interest_frame)
    df = add_taker_flow_features(df, taker_flow_frame)

    long_labels, long_realized_returns = build_barrier_labels(
        df,
        int(config["horizon_bars"]),
        float(config["long_up_barrier"]),
        float(config["long_down_barrier"]),
    )
    short_labels, short_realized_returns = build_barrier_labels(
        df,
        int(config["horizon_bars"]),
        float(config["short_up_barrier"]),
        float(config["short_down_barrier"]),
    )

    label_mode = str(config["label_mode"])
    if label_mode == "keep-all-binary":
        long_labels = np.where(np.isnan(long_labels), 0.0, long_labels)
        short_labels = np.where(np.isnan(short_labels), 0.0, short_labels)

    df[LONG_TARGET_COLUMN] = long_labels
    df[SHORT_TARGET_COLUMN] = short_labels
    df["future_return_7"] = long_realized_returns
    df["future_short_return_7"] = -short_realized_returns

    needed = FEATURE_COLUMNS + EXPERIMENTAL_FEATURE_COLUMNS + ["future_return_7", "future_short_return_7"]
    if label_mode != "keep-all-binary":
        needed.extend([LONG_TARGET_COLUMN, SHORT_TARGET_COLUMN])
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=needed).reset_index(drop=True)
    return df


def split_indices(num_rows: int) -> tuple[int, int]:
    train_end = int(num_rows * TRAIN_FRACTION)
    valid_end = train_end + int(num_rows * VALID_FRACTION)
    if train_end <= 0 or valid_end >= num_rows:
        raise RuntimeError("Not enough rows to create chronological splits.")
    return train_end, valid_end


def save_processed_dataset(df: pd.DataFrame, asset_key: str | None = None) -> None:
    key = asset_key or ac.get_asset_key()
    config = get_runtime_config(key)
    train_end, valid_end = split_indices(len(df))
    ensure_cache_dir(key)
    processed_path = ac.get_processed_data_path(key)
    metadata_path = ac.get_metadata_path(key)
    df.to_csv(processed_path, index=False)
    metadata = {
        "asset_key": str(config["asset_key"]),
        "symbol": str(config["symbol"]),
        "binance_symbol": str(config["binance_symbol"]),
        "price_source": "binance_spot_klines",
        "funding_source": "binance_funding_rate_history",
        "open_interest_source": "binance_open_interest_hist_1d_last_month",
        "taker_flow_source": "binance_taker_buy_sell_1d_last_month",
        "horizon_bars": int(config["horizon_bars"]),
        "long_up_barrier": float(config["long_up_barrier"]),
        "long_down_barrier": float(config["long_down_barrier"]),
        "short_down_barrier": float(config["short_down_barrier"]),
        "short_up_barrier": float(config["short_up_barrier"]),
        "label_mode": str(config["label_mode"]),
        "feature_columns": FEATURE_COLUMNS,
        "experimental_feature_columns": EXPERIMENTAL_FEATURE_COLUMNS,
        "long_target_column": LONG_TARGET_COLUMN,
        "short_target_column": SHORT_TARGET_COLUMN,
        "train_rows": train_end,
        "validation_rows": valid_end - train_end,
        "test_rows": len(df) - valid_end,
        "long_positive_rate": float(df[LONG_TARGET_COLUMN].mean()),
        "short_positive_rate": float(df[SHORT_TARGET_COLUMN].mean()),
        "total_rows": len(df),
        "date_start": df["date"].iloc[0].strftime("%Y-%m-%d"),
        "date_end": df["date"].iloc[-1].strftime("%Y-%m-%d"),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_dataset_frame(asset_key: str | None = None) -> pd.DataFrame:
    path = ac.get_processed_data_path(asset_key)
    if not path.exists():
        raise FileNotFoundError(f"Processed dataset not found at {path}. Run prepare.py first.")
    return pd.read_csv(path, parse_dates=["date"])


def load_splits(target_column: str = LONG_TARGET_COLUMN, asset_key: str | None = None) -> dict[str, DatasetSplit]:
    df = load_dataset_frame(asset_key)
    train_end, valid_end = split_indices(len(df))
    splits = {
        "train": df.iloc[:train_end].copy(),
        "validation": df.iloc[train_end:valid_end].copy(),
        "test": df.iloc[valid_end:].copy(),
    }
    output: dict[str, DatasetSplit] = {}
    for name, frame in splits.items():
        output[name] = DatasetSplit(
            features=frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32),
            labels=frame[target_column].to_numpy(dtype=np.float32),
            frame=frame,
        )
    return output


def describe_dataset(df: pd.DataFrame) -> str:
    train_end, valid_end = split_indices(len(df))
    lines = [
        f"Rows: {len(df)}",
        f"Date range: {df['date'].iloc[0].date()} -> {df['date'].iloc[-1].date()}",
        f"Train/validation/test: {train_end}/{valid_end - train_end}/{len(df) - valid_end}",
        f"Long positive rate: {df[LONG_TARGET_COLUMN].mean():.3f}",
        f"Short positive rate: {df[SHORT_TARGET_COLUMN].mean():.3f}",
        f"Features: {', '.join(FEATURE_COLUMNS)}",
    ]
    return "\n".join(lines)


def main() -> None:
    key = ac.get_asset_key()
    config = get_runtime_config(key)
    symbol = str(config["symbol"])
    print(f"Downloading {symbol} daily prices from Binance spot...")
    raw = download_symbol_prices(key)
    processed = add_features(raw, key)
    save_processed_dataset(processed, key)
    print("Prepared dataset:")
    print(
        "Label config: "
        f"horizon={int(config['horizon_bars'])}, "
        f"long={float(config['long_up_barrier']):.2%}/{float(config['long_down_barrier']):.2%}, "
        f"short={float(config['short_down_barrier']):.2%}/{float(config['short_up_barrier']):.2%}, "
        f"mode={config['label_mode']}"
    )
    print(describe_dataset(processed))
    print(f"Processed data saved to: {ac.get_processed_data_path(key)}")


if __name__ == "__main__":
    main()
