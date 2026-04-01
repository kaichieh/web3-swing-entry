# Binance Plan

This document defines the next data-source upgrade for `web3-swing-entry`.

The goal is to replace the current Yahoo-only OHLCV source with a Binance-first market data stack that is more useful for one-week long/short crypto trading.

## Why Upgrade

The current repo can already:

- build BTC and ETH weekly labels
- train long and short baselines
- score the latest bar
- render signal charts

But the current source is still weak for crypto-specific research because it mostly provides:

- daily OHLCV
- one generic volume field

That is enough for a baseline, but not enough to model:

- leverage crowding
- funding pressure
- taker aggressor flow
- futures-vs-index dislocation

## Data Layers

The Binance upgrade should be structured as four layers.

### 1. Spot Price

This should become the primary price source for BTC and ETH daily bars.

Suggested raw fields:

- `date`
- `symbol`
- `open`
- `high`
- `low`
- `close`
- `volume`
- `quote_asset_volume`
- `trade_count`
- `taker_buy_base_volume`
- `taker_buy_quote_volume`

Suggested cache path:

- `.cache/<asset>-swing-entry/<asset>_spot_daily.csv`

Primary use:

- replace Yahoo daily OHLCV with exchange-native price and volume structure

### 2. Futures Price

This layer adds derivatives pricing context.

Suggested raw fields:

- `date`
- `symbol`
- `mark_open`
- `mark_high`
- `mark_low`
- `mark_close`
- `index_open`
- `index_high`
- `index_low`
- `index_close`
- `basis`
- `basis_rate`

Suggested cache path:

- `.cache/<asset>-swing-entry/<asset>_futures_daily.csv`

Primary use:

- detect futures premium, discount, and distortion relative to spot/index

### 3. Positioning And Sentiment

This layer captures crowding and aggressor flow.

Suggested raw fields:

- `date`
- `symbol`
- `open_interest`
- `open_interest_value`
- `funding_rate`
- `top_trader_long_short_position_ratio`
- `top_trader_long_short_account_ratio`
- `global_long_short_account_ratio`
- `taker_buy_sell_ratio`
- `taker_buy_volume`
- `taker_sell_volume`

Suggested cache path:

- `.cache/<asset>-swing-entry/<asset>_sentiment_daily.csv`

Primary use:

- measure leverage buildup and crowd bias

### 4. Derived Features

This layer is what the model should directly consume.

Suggested new feature columns:

- `funding_rate_1d`
- `funding_rate_3d_mean`
- `funding_rate_7d_mean`
- `funding_rate_z_7`
- `open_interest_change_1`
- `open_interest_change_3`
- `open_interest_z_7`
- `basis_rate`
- `basis_rate_z_7`
- `taker_buy_sell_ratio`
- `taker_flow_3d_mean`
- `top_trader_pos_ratio`
- `top_trader_pos_ratio_change_3`
- `account_ratio_extreme_flag`

Suggested processed output:

- `.cache/<asset>-swing-entry/<asset>_features.csv`

## Minimum Viable Binance Upgrade

Do not add everything at once.

The first Binance version should only add:

1. Binance spot klines
2. funding rate
3. open interest
4. taker buy/sell volume

This is enough to materially improve the repo without turning `prepare.py` into a large refactor all at once.

## Proposed Implementation Order

### Phase 1: Replace Price Source

- add Binance spot daily download helper
- write `<asset>_spot_daily.csv`
- switch the main feature pipeline from Yahoo OHLCV to Binance spot OHLCV
- keep the same weekly targets so research stays comparable

### Phase 2: Add Derivatives Context

- add funding-rate fetch
- add open-interest fetch
- merge those into the daily feature frame
- write the merged raw sentiment cache

### Phase 3: Add Flow Signals

- add taker buy/sell volume
- derive `taker_buy_sell_ratio`
- test whether it improves long/short calibration

### Phase 4: Recalibrate The Weekly System

- rebuild BTC and ETH datasets
- retrain long and short baselines
- compare against the current Yahoo-based baseline in `results.tsv`

## Promotion Criteria For The Binance Upgrade

The Binance migration is worth keeping only if it improves at least one of:

- label usability
- long-side balanced accuracy
- short-side balanced accuracy
- signal density calibration
- latest-signal stability

and does not break the existing:

- `prepare.py`
- `train.py`
- `predict_latest.py`
- `chart_signals.py`

workflow.

## Notes

- Keep the repo daily-bar based for now.
- Do not add intraday Binance data until the daily Binance version is stable.
- Keep the first migration focused on data quality, not model complexity.
