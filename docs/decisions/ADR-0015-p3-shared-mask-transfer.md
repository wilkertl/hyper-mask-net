# ADR-0015: P3 follow-up measures one shared mask's cost and its transfer among positives

- **Status:** proposed
- **Date:** 2026-11-07
- **Issue:** none assigned; a ticket is still needed before this work gets a pull request

## Context

P3 shows that a query's relevant documents do not select identical dimensions. It does not say what
using one mask for all of them costs at every cardinality, or whether a mask built from some
relevant documents serves others of the same query. A future learnable target depends on both. This
stays descriptive: nothing is trained, ranked, or evaluated on unseen queries.

## Decision

- **Inputs.** Published F2 training queries and training judgments, multi-positive queries, D = 1024,
  stored documents unchanged, masked queries not re-normalized. Results live apart from P3's, in
  `evaluation/P3-shared-mask` and `reports/P3-shared-mask`.
- **Shared mask.** The top `k` of the weighted mean of a group's `c_ij = q_j * d_ij`, with the
  repository's gain weights. It exactly maximizes the weighted mean score at that `k`.
- **Regret.** A positive's own optimum minus its score under the mask at the same `k`, raw and divided
  by `sum|c_ij|`, which is never a signed score. A document without mass has undefined ratios.
- **Controls for count.** A group's worst depends on its size, so natural groups are stratified and
  masks are also built on drawn groups of fixed size. Cells of different construction sizes are also
  compared on common cohorts of the same queries, from stored per-query arrays.
- **Transfer.** A mask is built only from a split's construction documents and scored on the rest:
  `a` to `a` documents, or half of each query's positives. Splits are exhaustive when few exist, else
  seeded draws. The held-out documents do not reach the mean or the weights.
- **References, labeled as such.** The all-positive mask, which saw the held-out documents. A mask of
  unjudged corpus documents paired with the same query, for what sharing the query alone transfers;
  those documents are not negatives. An exact uniform-random expectation, with no structure.
- **Reporting.** Query-balanced first and pair-weighted apart, every `k`, eligible counts, pointwise
  exploratory intervals, repeat variation separated from query sampling. No threshold for a small
  loss.
- **Exact ties.** Masks order the gain-weighted sum with whole-number gains, not normalized weights:
  products of float16 sum exactly, so a tie in the mean stays a tie, on every device, and goes to the
  lower dimension. Normalized weights round, and a device could then break the tie and change scores.
- **Execution.** One query at a time on `run.device`, one ordering per document or mean. Orderings and
  checkpoint regrets are compared with the canonical P3 run when it is present.

## Evidence

- Report: [reports/P3-shared-mask](../../reports/P3-shared-mask/README.md), run
  `20261107T025534Z-d8bad5ff` on DataPolicy-4090 (RTX 4090); numbers are not typed here.

## Consequences

- Regret describes a compromise in similarity. It does not establish retrieval damage or rule out a
  better ranking mask, and overlap and regret can disagree.
- The mask is mean-optimal, not minimax. Nonzero mean regret shows no single mask reaches every
  positive's optimum at that `k`; it does not make the observed worst-positive regret irreducible.
- The worst positive's share of regret measures that positive alone, not the upper tail.
- Transfer here is among documents of a training query, not to unseen queries, and it may reflect
  relevant documents lying close to one another; nothing here attributes it to relevance.
