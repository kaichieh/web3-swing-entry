# Task

## Current Objective

Build the first working version of a shared BTC/ETH weekly swing repo that can score:

- long
- short
- no_trade

using mirrored 7-bar barrier labels.

## Phase 1: Scaffold

- [x] Create shared repo docs and asset folders.
- [x] Define default BTC and ETH weekly barrier configs.
- [x] Add shared path/config helpers in `asset_config.py`.

## Phase 2: Data Pipeline

- [x] Implement `prepare.py` for crypto daily OHLCV download.
- [x] Add weekly features focused on 1 to 10 bar behavior.
- [x] Create mirrored long and short targets.
- [x] Save processed datasets and metadata per asset.

## Phase 3: Baseline Models

- [x] Implement `train.py` with `AR_SIDE=long|short`.
- [ ] Train one baseline model per side per asset.
- [ ] Report validation and test metrics separately for long and short.

## Phase 4: Live Decision

- [ ] Implement `predict_latest.py`.
- [ ] Combine long and short scores into one final signal.
- [ ] Return `no_trade` when neither side clears threshold with margin.

## Phase 5: Reporting

- [ ] Implement `chart_signals.py`.
- [ ] Add per-asset `results.tsv` logging format.
- [ ] Define a promotion gate for weekly dual-side models.

## Immediate Next Step

Run `prepare.py` for BTC and ETH, then train:

- `btc_features.csv`
- `eth_features.csv`
- metadata describing long/short label balance
