from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AssetDefaults:
    key: str
    symbol: str
    horizon_bars: int
    long_up_barrier: float
    long_down_barrier: float
    short_down_barrier: float
    short_up_barrier: float
    label_mode: str


DEFAULT_ASSET_KEY = "btc"
ASSET_DEFAULTS: dict[str, AssetDefaults] = {
    "btc": AssetDefaults(
        key="btc",
        symbol="BTC-USD",
        horizon_bars=7,
        long_up_barrier=0.05,
        long_down_barrier=-0.025,
        short_down_barrier=-0.05,
        short_up_barrier=0.025,
        label_mode="drop-neutral",
    ),
    "eth": AssetDefaults(
        key="eth",
        symbol="ETH-USD",
        horizon_bars=7,
        long_up_barrier=0.06,
        long_down_barrier=-0.03,
        short_down_barrier=-0.06,
        short_up_barrier=0.03,
        label_mode="drop-neutral",
    ),
}

REPO_DIR = Path(__file__).resolve().parent
ASSETS_DIR = REPO_DIR / "assets"


def get_asset_key() -> str:
    candidate = os.getenv("AR_ASSET", DEFAULT_ASSET_KEY).strip().lower()
    if candidate not in ASSET_DEFAULTS:
        raise ValueError(f"Unsupported AR_ASSET '{candidate}'. Known assets: {', '.join(sorted(ASSET_DEFAULTS))}")
    return candidate


def get_asset_dir(asset_key: str | None = None) -> Path:
    return ASSETS_DIR / (asset_key or get_asset_key())


def load_asset_config(asset_key: str | None = None) -> dict[str, object]:
    key = asset_key or get_asset_key()
    defaults = ASSET_DEFAULTS[key]
    config: dict[str, object] = {
        "asset_key": defaults.key,
        "symbol": defaults.symbol,
        "horizon_bars": defaults.horizon_bars,
        "long_up_barrier": defaults.long_up_barrier,
        "long_down_barrier": defaults.long_down_barrier,
        "short_down_barrier": defaults.short_down_barrier,
        "short_up_barrier": defaults.short_up_barrier,
        "label_mode": defaults.label_mode,
    }
    config_path = get_asset_dir(key) / "config.json"
    if config_path.exists():
        config.update(json.loads(config_path.read_text(encoding="utf-8")))
    return config


def get_asset_symbol(asset_key: str | None = None) -> str:
    return str(load_asset_config(asset_key)["symbol"])


def get_cache_dir(asset_key: str | None = None) -> Path:
    key = asset_key or get_asset_key()
    return REPO_DIR / ".cache" / f"{key}-swing-entry"


def get_raw_data_path(asset_key: str | None = None) -> Path:
    key = asset_key or get_asset_key()
    return get_cache_dir(key) / f"{key}_daily.csv"


def get_processed_data_path(asset_key: str | None = None) -> Path:
    key = asset_key or get_asset_key()
    return get_cache_dir(key) / f"{key}_features.csv"


def get_metadata_path(asset_key: str | None = None) -> Path:
    return get_cache_dir(asset_key) / "metadata.json"


def get_latest_prediction_path(asset_key: str | None = None) -> Path:
    return get_cache_dir(asset_key) / "latest_prediction.json"


def get_chart_output_path(asset_key: str | None = None) -> Path:
    return get_cache_dir(asset_key) / "signal_chart.html"


def get_results_path(asset_key: str | None = None) -> Path:
    return get_asset_dir(asset_key) / "results.tsv"


def get_task_path(asset_key: str | None = None) -> Path:
    return get_asset_dir(asset_key) / "task.md"


def get_ideas_path(asset_key: str | None = None) -> Path:
    return get_asset_dir(asset_key) / "ideas.md"


def get_program_path(asset_key: str | None = None) -> Path:
    return get_asset_dir(asset_key) / "program.md"
