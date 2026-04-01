# Ideas

## Near-Term

- Compare `7` vs `10` bar horizon.
- Compare tighter and looser BTC/ETH barrier pairs.
- Add `weekend_return` and `weekend_range_pct`.
- Test whether ETH needs slightly wider downside buffers than BTC.
- Compare threshold-based signal selection vs percentile-based selection.
- Start Phase 1 from `binance_plan.md` by replacing Yahoo OHLCV with Binance spot klines.
- Rescue short-side calibration so BTC/ETH short does not stay near always-on.
- Try compact Binance subsets instead of the full 33-feature stack.
- Refit barrier and horizon choices after the Binance migration.
- Add rolling walk-forward evaluation so single-split validation spikes do not mislead promotion decisions.
- Try recent-regime filters or volatility-state segmentation before adding more model complexity.
- If regime gates only improve signal density but not balanced accuracy, use them as safety rails in live scoring rather than as alpha sources.

## Mid-Term

- Add simple long/short trade summary tables.
- Compare independent thresholds with one shared confidence margin rule.
- Compare logistic against a small tree-based baseline after walk-forward evaluation is in place.
- If logistic and shallow-tree baselines both fail under walk-forward, test regime-conditioned feature sets rather than deeper models first.
- Compare side-specific live no-trade rules that block longs outside rebound regimes and shorts outside downtrend regimes.
- Add recent-regime summaries for trend, mean-reversion, and volatility expansion.
- Study whether neutral-heavy windows should be filtered instead of relabeled.
- Add Binance funding-rate and open-interest features after the spot migration is stable.

## Later

- Funding-rate features
- Open-interest features
- BTC dominance context
- Cross-asset relative-strength features
- Intraday versions after the daily baseline is stable
- Full Binance futures context including basis and top-trader ratios
