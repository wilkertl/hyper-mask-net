# P3-bench nfcorpus: teachers `20261114T022810Z-24f79ec6`

Generated from this run's `summary.json` and `searches.json`. Fit queries only; no tune or evaluation id is read.

Teachers: `positive`, `contrastive`. Queries: 1810. Step cap: None. Compute: 1.2 s on cuda:0.

## Nesting across budgets

Mean share of the smaller budget's mask inside the next larger one.

| teacher | 102→204 | 204→307 | 307→409 | 409→512 | 512→614 | 614→716 | 716→819 | 819→921 |
|---|---|---|---|---|---|---|---|---|
| positive | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| contrastive | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

No rank-guided search ran (removed by the escalation record).
