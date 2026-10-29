# E2: do space descriptors predict selector shift? (gate G3)

Unit: the dimension; each environment is held out in turn. Target: the S1 selector's
selection frequency per dimension on validation queries (embeddings only, no judgments),
divided by k and averaged over the 10%-90% budgets, then over seeds. E2 scores no
retrieval, so it has no full or prefix rows.

- probe-descriptors `20261029T002359Z-c352e11b` (commit `e7627adfa001`)
- seeds 0, 1, 2; validation queries per environment: scifact 81, nfcorpus 259, msmarco 5000
- S1-S4 were shaped on these three environments and the targets come from selectors tuned on them, so E2's reading is exploratory; G3 is confirmed by rerunning this spec unchanged on new environments (#60).

## Shift check (research D0)

B1 predicts the held-out environment by the mean of the other environments' seed-averaged
profiles; the ceiling is the mean pairwise Spearman between seeds stepped up to the
three-seed average by Spearman-Brown. The gap is what a descriptor could explain. Rule
(fixed before any target): no shift when the gap is below 0.05 in at least
2 environments.

| held-out | B1 spearman | seed spearman | ceiling | gap | room (gap >= 0.05) |
|---|---|---|---|---|---|
| scifact | 0.4354 | 0.6022 | 0.8195 | +0.3841 | yes |
| nfcorpus | 0.5590 | 0.9221 | 0.9726 | +0.4136 | yes |
| msmarco | 0.5391 | 0.9950 | 0.9983 | +0.4592 | yes |

Reading: there is an environment-specific shift for descriptors to explain.

Descriptors were not evaluated in this run (`--targets-only`).
