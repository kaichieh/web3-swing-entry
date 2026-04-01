# Web3 Swing Entry

This repo is a short-horizon crypto swing research workspace for:

- `BTC-USD`
- `ETH-USD`

It is designed for one-week trades and supports both:

- long entries
- short entries

## Core Goal

For each asset, the live pipeline should decide between:

- `long`
- `short`
- `no_trade`

The first version uses two mirrored binary targets instead of one three-class model:

- `long_target`: within the next 7 bars, price hits the upside barrier before the downside barrier
- `short_target`: within the next 7 bars, price hits the downside barrier before the upside barrier

This keeps the research loop simple while still enabling bidirectional signals.

## Default Research Shape

- horizon: `7` calendar bars
- BTC long target: `+5.0% before -2.5%`
- BTC short target: `-5.0% before +2.5%`
- ETH long target: `+6.0% before -3.0%`
- ETH short target: `-6.0% before +3.0%`
- label mode: `drop-neutral`

## Repo Layout

- `asset_config.py`: shared asset defaults and path helpers
- `prepare.py`: builds daily OHLCV features and long/short labels
- `train.py`: trains baseline weekly long/short models
- `predict_latest.py`: produces the current long/short/no-trade signal
- `chart_signals.py`: renders recent signal charts
- `research_batch.py`: runs formal research rounds
- `score_results.py`: refreshes headline scores and promotion gates
- `task.md`: current shared implementation and research queue
- `ideas.md`: parking lot for future experiments
- `binance_plan.md`: next-step schema and migration plan for Binance market data
- `assets/btc/`
- `assets/eth/`

Each asset folder keeps its own:

- `config.json`
- `results.tsv`
- `task.md`
- `ideas.md`
- `program.md`

## First Build Priorities

1. Build the shared multi-asset crypto data pipeline.
2. Produce weekly long/short labels for BTC and ETH.
3. Train a baseline classifier for each side.
4. Combine the two side scores into one live trading decision.
5. Validate that signal frequency and class balance are sane before any heavier backtesting.

## Next Data Upgrade

The current version uses Yahoo daily OHLCV as a baseline source.

The planned source upgrade is documented in [binance_plan.md](C:\Users\Jay\OneDrive\文件\codex\web3-swing-entry\binance_plan.md).

That migration is intended to add:

- Binance spot klines
- funding rate
- open interest
- taker buy/sell flow
