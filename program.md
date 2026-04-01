# Program

This repo should be run as a short-cycle research system, not as a one-off script dump.

## Working Loop

1. Read the active work in `task.md`.
2. Implement or run one coherent research step.
3. Record formal experiment output in each asset's `results.tsv`.
4. Update `task.md` with the best next step.
5. Move overflow ideas into `ideas.md`.

## What Counts As Done

A round is only complete when it includes:

- code or config changes
- measurable output
- a written result in `results.tsv`
- a narrowed next step in `task.md`

## Research Guardrails

- Keep the first baseline pure OHLCV.
- Optimize for stable weekly decision quality, not maximum in-sample return.
- Treat BTC and ETH as separate assets even when they share code.
- Prefer mirrored long/short targets over a premature three-class setup.
- Avoid adding external datasets until the weekly baseline is stable.

## Model Progression

Version order should be:

1. shared feature pipeline
2. logistic long baseline
3. logistic short baseline
4. live long/short arbitration rule
5. signal visualization and summary outputs
6. threshold and barrier sensitivity studies
7. simple trade-summary comparisons

## Promotion Mindset

Do not promote a live rule only because one side looks good.

A candidate should show:

- reasonable balance on both long and short validation
- non-trivial signal frequency
- usable behavior on both BTC and ETH
- no obvious collapse when thresholds move slightly
