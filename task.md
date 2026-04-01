# Task

## Goal

Finish this task file by turning the repo into a working weekly BTC/ETH swing research system that can:

- prepare data
- train long and short baselines
- score the latest bar
- visualize signals
- record formal results

The repo should end with no major unchecked core-work items left in this file.

## Completed Foundation

- [x] Create shared repo docs and asset folders.
- [x] Define default BTC and ETH weekly barrier configs.
- [x] Add shared path/config helpers in `asset_config.py`.
- [x] Implement `prepare.py` for crypto daily OHLCV download.
- [x] Add weekly features focused on 1 to 10 bar behavior.
- [x] Create mirrored long and short targets.
- [x] Save processed datasets and metadata per asset.
- [x] Implement `train.py` with `AR_SIDE=long|short`.

## Core Remaining Work

- [x] Run `prepare.py` for BTC and confirm `btc_features.csv` and metadata are produced.
- [x] Run `prepare.py` for ETH and confirm `eth_features.csv` and metadata are produced.
- [x] Inspect BTC long/short label balance and record whether the weekly barrier setup is usable.
- [x] Inspect ETH long/short label balance and record whether the weekly barrier setup is usable.
- [x] Train BTC long baseline and capture validation/test metrics.
- [x] Train BTC short baseline and capture validation/test metrics.
- [x] Train ETH long baseline and capture validation/test metrics.
- [x] Train ETH short baseline and capture validation/test metrics.
- [x] Implement `predict_latest.py`.
- [x] Combine long and short model scores into one final `long` / `short` / `no_trade` decision.
- [x] Add a no-trade margin rule so weak conflicts do not force a position.
- [x] Implement `chart_signals.py`.
- [x] Make the chart clearly distinguish long, short, and no-trade states.
- [x] Write the first formal BTC experiment row to `assets/btc/results.tsv`.
- [x] Write the first formal ETH experiment row to `assets/eth/results.tsv`.
- [x] Define the first promotion gate for weekly dual-side models.

## Definition Of Done

This task file is considered complete when:

- BTC and ETH datasets can both be regenerated
- BTC and ETH long/short baselines have both been trained at least once
- live scoring exists through `predict_latest.py`
- charts exist through `chart_signals.py`
- each asset has at least one formal result row
- the promotion gate is written down
- the remaining work is iterative improvement rather than missing core workflow pieces

## Execution Order

1. Finish data generation and inspect label balance.
2. Finish the first full round of BTC and ETH long/short training.
3. Build live scoring in `predict_latest.py`.
4. Build visualization in `chart_signals.py`.
5. Log the first formal results and promotion gate.

## Current Focus

Work the file from top to bottom until only maintenance or future-research items remain.

## Recorded Findings

- BTC label balance is usable for a first pass, but the short target is materially more common than the long target: long positive rate `0.411`, short positive rate `0.653`.
- ETH shows the same skew: long positive rate `0.382`, short positive rate `0.643`.
- The first weekly logistic baseline is complete but not promotable because the short models collapse toward always-on predictions and balanced accuracy stays near random on several splits.

## Promotion Gate

The first weekly dual-side promotion gate is:

- `validation_long_bal_acc >= 0.53`
- `validation_short_bal_acc >= 0.53`
- `test_long_bal_acc >= 0.53`
- `test_short_bal_acc >= 0.53`
- long and short `test_positive_rate` must both stay between `0.15` and `0.85`
- the latest live signal must resolve to `long`, `short`, or `no_trade` without runtime errors

The current BTC and ETH baselines fail this gate and remain research-only.

## Next Cycle

The next implementation cycle is the Binance migration described in [binance_plan.md](C:\Users\Jay\OneDrive\文件\codex\web3-swing-entry\binance_plan.md).

Immediate next-step tasks for that cycle:

- [x] Add Binance spot daily download helpers.
- [x] Replace Yahoo daily OHLCV with Binance spot klines in `prepare.py`.
- [x] Add funding-rate raw cache and merge logic.
- [x] Add open-interest raw cache and merge logic.
- [x] Add taker buy/sell raw cache and merge logic.
- [x] Retrain BTC and ETH long/short baselines on the Binance-backed dataset.
- [x] Compare Binance-backed results against the current baseline in `results.tsv`.

## Binance Cycle Findings

- The Binance migration is fully wired into `prepare.py` with spot, funding, open interest, and taker-flow caches.
- BTC did not materially improve after adding Binance context features. Test balanced accuracy stayed close to random on both sides, and the short model still collapses to all-on predictions.
- ETH long validation improved with Binance context (`0.5680` validation balanced accuracy), but test balanced accuracy fell to `0.4891`, so the improvement did not hold out-of-sample.
- ETH short improved slightly relative to the pure baseline (`0.5093` test balanced accuracy vs `0.5000` before), but still remains below the promotion gate.
- Result: the data migration is operationally successful, but the current logistic setup is still research-only.

## Next Three Experiments

### Experiment 1: Short-Side Calibration Rescue

Problem:

- BTC short still collapses to effectively always-on predictions.
- ETH short improved a little, but still sits below the promotion gate.

What to test:

- raise `AR_THRESHOLD_MIN` and `AR_THRESHOLD_MAX` for short-side threshold search
- add short-side class weighting experiments
- force short-side signal density to stay below the promotion gate cap

Success condition:

- reduce short-side `test_positive_rate` below `0.85`
- improve short-side `test_bal_acc` without collapsing recall

Current finding:

- A moderate short-side calibration setting (`AR_POS_WEIGHT=0.9`, `AR_NEG_WEIGHT=1.1`, threshold search `0.60` to `0.95`) successfully reduced signal density for both BTC and ETH, but it did not improve balanced accuracy.
- BTC short `test_positive_rate` moved from `1.0000` to `0.4974`, but `test_bal_acc` stayed weak at `0.4952`.
- ETH short `test_positive_rate` moved from `0.9754` to `0.3563`, but `test_bal_acc` fell to `0.4718`.
- Conclusion: calibration can fix signal density, but the current short-side feature set still lacks enough discriminative power.

### Experiment 2: Binance Feature Selection Instead Of Full 33-Feature Stack

Problem:

- adding all Binance context fields at once did not create a stable improvement

What to test:

- funding-only extra features
- open-interest-only extra features
- taker-flow-only extra features
- one mixed compact set rather than all 13 extra fields

Success condition:

- any compact Binance subset that beats the current Binance full-stack result on test balanced accuracy

Current finding:

- BTC long did not show a meaningful difference between funding-only, compact-mix, and full-stack Binance features. The best test balanced accuracy remained `0.5178`, still below the promotion gate.
- ETH long validation improvement came almost entirely from the funding-only stack (`validation_bal_acc = 0.5680`), while open-interest-only, taker-only, and compact-mix all fell back to the weaker `0.5165` validation region.
- ETH long still failed out-of-sample even with funding-only features (`test_bal_acc = 0.4891`).
- Conclusion: funding is the only Binance context feature that clearly moves the long-side validation needle right now; OI and taker flow are not yet earning their complexity on the long side.

### Experiment 3: Barrier And Horizon Refit For Binance Data

Problem:

- the current weekly barriers may still be tuned for the old baseline assumptions instead of Binance-backed crypto structure

What to test:

- `7` vs `10` bar horizon
- tighter and wider BTC barriers
- tighter and wider ETH barriers

Success condition:

- produce a better class balance and more stable validation-to-test transfer

Current finding:

- BTC did not materially improve under any of the tested barrier or horizon refits. `10` bars produced the best BTC long test balanced accuracy at `0.5200`, but short still collapsed to all-on predictions with `1.0000` test positive rate and `0.5000` test balanced accuracy.
- BTC tighter (`4% / 2%`) and wider (`6% / 3%`) `7`-bar barriers changed label balance only modestly and did not create a stable long or short edge.
- ETH `10` bars also failed to improve transfer. Long test balanced accuracy reached only `0.5098`, while short stayed effectively always-on with `0.9696` test positive rate and `0.5004` test balanced accuracy.
- ETH tighter `7`-bar barriers (`5% / 2.5%`) improved long validation balanced accuracy to `0.5540`, but that did not hold on test (`0.5096`), so the gain looks like validation-only noise.
- ETH wider `7`-bar barriers (`7% / 3.5%`) produced the best Experiment 3 holdout result at `0.5259` long test balanced accuracy, but it still missed the promotion gate and long test positive rate remained too high at `0.9313`.
- Conclusion: target refitting alone is not enough. The current logistic setup still lacks robust out-of-sample discrimination, especially on the short side.

### Experiment 4: Walk-Forward Stability Check

Problem:

- single validation/test splits may be overstating fragile pockets of performance

What to test:

- add a rolling walk-forward evaluator that re-trains the logistic baseline across multiple chronological windows
- run it on the strongest BTC and ETH refit candidates from Experiment 3

Success condition:

- average walk-forward test balanced accuracy should stay meaningfully above `0.53`
- fold-to-fold test balanced accuracy should remain reasonably stable instead of collapsing back to random

Current finding:

- Added `walkforward_eval.py` to measure rolling-window stability without replacing the existing training pipeline.
- BTC `10`-bar same-barrier walk-forward results looked better on validation (`0.5403` long, `0.5480` short average balanced accuracy) than on holdout, but average test balanced accuracy still reverted to `0.5032` long and `0.5037` short.
- ETH `7`-bar wider-barrier walk-forward results showed the same pattern. Average validation balanced accuracy stayed elevated (`0.5673` long, `0.5458` short), while average test balanced accuracy fell back to `0.5062` long and `0.4969` short.
- The fold ranges were also unstable. BTC short varied from `0.4771` to `0.5401` test balanced accuracy, and ETH long varied from `0.4786` to `0.5352`.
- Conclusion: the better-looking single-split results from Experiment 3 are not stable enough under rolling evaluation. The repo now has a stronger verification tool, and the next worthwhile step is a small model-class comparison rather than more barrier tuning.

### Experiment 5: Small Tree Baseline Comparison

Problem:

- logistic may be too linear to capture the short-horizon crypto patterns in the current feature set

What to test:

- add a minimal dependency-free decision-tree walk-forward baseline
- compare it against the strongest BTC and ETH logistic candidates under the same rolling-window protocol

Success condition:

- beat the logistic walk-forward average test balanced accuracy on at least one asset-side pair
- improve prediction density without sacrificing holdout discrimination

Current finding:

- Added `tree_walkforward_eval.py` as a simple non-linear baseline with shallow numeric splits and the same rolling-window evaluation logic.
- BTC `10`-bar tree results did not improve holdout quality. Long average test balanced accuracy fell to `0.4945`, and short stayed effectively unusable at `0.4996` with `0.9765` average test positive rate.
- ETH `7`-bar wider-barrier tree results were slightly more realistic on signal density than the logistic version, but holdout balanced accuracy still stayed near random at `0.5099` long and `0.5005` short.
- Relative to logistic walk-forward, the tree baseline mainly changed signal density, not out-of-sample discrimination.
- Conclusion: the current feature stack appears to be the bigger bottleneck than model linearity. The next better experiment is regime-conditioned filtering or feature segmentation, not a deeper tree search.

### Experiment 6: Regime-Conditioned No-Trade Gates

Problem:

- even when average balanced accuracy stays near random, some market states may be obviously worse than others and should be blocked from live entries

What to test:

- apply simple regime gates on top of the logistic walk-forward baseline
- compare `no_filter` against trend and volatility filters, plus rebound/fade filters

Success condition:

- improve average walk-forward test balanced accuracy or materially reduce signal density without harming it

Current finding:

- Added `regime_filter_eval.py` to evaluate simple side-specific gates as post-model no-trade filters.
- BTC did not benefit from the tested long gates. `no_filter` remained the best long average test balanced accuracy at `0.5032`, while `uptrend_only`, `oversold_rebound`, and `uptrend_low_vol` all weakened it.
- BTC short also did not produce a usable gain. `downtrend_high_vol` nearly shut the strategy off (`0.0012` average test positive rate) while only reaching `0.5009` average test balanced accuracy.
- ETH long behaved similarly: `no_filter` stayed best at `0.5062`, and the rebound / trend gates mostly reduced participation without increasing holdout quality.
- ETH short showed the most useful behavior. `downtrend_only` improved average walk-forward test balanced accuracy from `0.4969` to `0.5032` while cutting average test positive rate from `0.3088` to `0.0799`.
- Conclusion: regime gates are better as safety rails than as alpha engines in the current system. The clearest near-term use is to suppress ETH short signals outside clear downtrends, not to claim a new promotable edge.

### Experiment 7: Live ETH Short Regime Gate

Problem:

- the research suggested ETH short signals are safer when they are limited to clear downtrends, but the live scorer was still free to emit ETH shorts outside that state

What to test:

- add an explicit `ETH short downtrend-only` live gate to `predict_latest.py`
- keep BTC and long-side behavior unchanged
- expose the gate status in the live output so blocked shorts are easy to inspect

Success condition:

- ETH short candidates outside downtrends should resolve to `no_trade` or let the long side win instead of emitting an unrestricted short

Current finding:

- Implemented a live ETH short regime gate based on `sma_gap_5 < 0` and `sma_gap_10 < 0`.
- The gate is explicit in the output through `signal_summary.short_regime_gate` and `short_summary.regime_gate_passed`.
- BTC remains unchanged and reports `no_short_regime_gate_for_asset`.
- On the current ETH live bar dated `2026-04-01`, the raw short score still cleared threshold, but the regime gate blocked it because both moving-average gaps were positive. The final signal resolved to `long` with decision reason `long_wins_after_short_regime_block`.
- Conclusion: the repo now uses the strongest regime-filter finding as a live safety rule instead of leaving it as research-only.
