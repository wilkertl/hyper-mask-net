# P3-bench scifact: teachers `20261112T223656Z-07608c76`

Generated from this run's `summary.json` and `searches.json`. Fit queries only; no tune or evaluation id is read.

Teachers: `contrastive`, `positive`, `rank-guided`. Queries: 565. Step cap: 8. Compute: 1,422.8 s on cuda:0.

## Nesting across budgets

Mean share of the smaller budget's mask inside the next larger one.

| teacher | 102→204 | 204→307 | 307→409 | 409→512 | 512→614 | 614→716 | 716→819 | 819→921 |
|---|---|---|---|---|---|---|---|---|
| contrastive | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| positive | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| rank-guided | 0.9970 | 0.9971 | 0.9976 | 0.9975 | 0.9970 | 0.9976 | 0.9969 | 0.9964 |

## Rank-guided search: objective gain against cost

Searches: 10,170 (both starts per query and budget); improved: 309; hit the cap: 4; shortlist proposals rejected by the exact check: 0.
Chosen start: contrastive 332, positive 4,753.

Objective gain: `searches.json` is not available in this root.

## Demotions (FR-007)

Documents that ranked above the best positive before an accepted search and not after it:

| unjudged documents | known negatives (judged 0) |
|---|---|
| 88 | 0 |
