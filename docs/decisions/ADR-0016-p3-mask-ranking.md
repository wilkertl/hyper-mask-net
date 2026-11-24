# ADR-0016: P3 follow-up ranks the corpus under masks that maximize positive similarity

- **Status:** proposed
- **Date:** 2026-11-08
- **Issue:** none assigned; a ticket is still needed before this work gets a pull request

## Context

P3 and its shared-mask follow-up measure similarity between a query and its relevant documents. A
mask can raise that similarity while other documents rise further, so similarity alone cannot say
whether relevant documents move up the ranking. This experiment ranks the whole corpus under the
same masks. It is a privileged offline diagnostic on training judgments. It trains nothing, searches
for no ranking mask, and is not a retrieval method or an upper bound on one.

## Decision

Fixed before any outcome was seen.

- **Inputs.** Published F2 training queries and training judgments, reaching the code only through
  `load_training_judgments`. Multi-positive queries, as in the shared-mask run. Frozen Qwen, D = 1024,
  stored documents unchanged, masked queries not re-normalized. Results live in
  `evaluation/P3-mask-ranking` and `reports/P3-mask-ranking`.
- **Exact scoring.** Every document of the corpus is scored for every mask, in float64 and streamed
  in chunks. No FAISS, no candidate pool. Float16 values are multiples of 2^-24, so their products
  are multiples of 2^-48 and, with every partial sum below 2 in magnitude, float64 represents each
  score exactly in any order of summation. A score therefore does not depend on device or batch.
- **Ranks and the tie baseline.** Rank of a document is one plus the documents that strictly beat it,
  where ties go to the lower corpus row; that is the order of the retrieved lists. Full, prefix, and
  masked arms all go through the same scan and this one rule. A document with the query's own id is
  excluded, as `rank_documents` does. Only top-100 lists and rank counters are kept, no score matrix.
  The repository's `exact_search` is float32 and leaves exact ties to `topk`, so its order among
  documents with equal scores is arbitrary and can differ by device. On the pilot's 30 NFCorpus
  queries, one query with two identically scored documents (one relevant) differs in nDCG@10 by
  0.029; every other metric agrees to 3e-8. Numbers in this experiment are therefore not comparable
  to the last digit with E0 and other float32 baselines. Those results stand as published and are
  not recomputed.
- **Arms.** On the same queries, partitions, and budgets:
  - full query (`k = 1024`, the reference);
  - prefix-k, the protocol's mandatory context, not a claim;
  - shared mask of all known positives: privileged, it saw the documents it is scored on;
  - construction-only masks in transfer cells 1-1, 2-2, 5-5, 10-10, 20-20, scored on held-out
    positives of the same training query;
  - each positive's own mask: it maximizes that positive's similarity, a privileged diagnostic and
    never an upper bound on that positive's rank. Every positive is included because the cost is one
    rank per mask and no retrieved list.
  Masks are the top `k` of `shared_order`, with the same whole-number gains and tie rule as
  ADR-0015. Mean-similarity optimal is neither minimax nor ranking optimal.
- **Partitions.** Those of ADR-0015: same generator streams, every ordered split when there are at
  most 8 of them, else 8 seeded draws. The `half` cell is left out: with two positives it repeats
  1-1, and the 20-20 cell is the larger construction contrast on queries that can afford it.
- **Budgets.** Representation curves cover every integer `k`; ranking covers the repository's fixed
  grid, `floor(ratio * 1024)` for ratios 0.1 to 0.9 (102, 204, 307, 409, 512, 614, 716, 819, 921),
  with the full query at 1024. The pilot uses 102, 307, 512, 921. This grid is coverage of budgets
  for ranking, and it must not be read as all-k coverage.
- **Metrics.** nDCG@10 (linear gains), MRR@10, MRR@100, Recall@100 from `query_metrics` on the
  unchanged full training judgments. Construction positives stay relevant and stay in the ranking.
- **Held-out positives.** Separately, from exact ranks: reciprocal rank, Recall@10 and @100, log
  rank, and nDCG@10 restricted to held-out labels, in which a construction positive keeps its place
  with gain zero. That last metric is a restricted-label diagnostic. A secondary rank that leaves
  construction positives out is labeled as such and disambiguates crowding only.
- **Competitors.** Other documents, not confirmed irrelevant ones: judgments are incomplete, and an
  unjudged document that rises may be relevant.
- **Reporting.** Gains and losses per relevant document, per-query fraction improved or worsened, and
  the worst-served positive. Query-balanced first, document-weighted apart. The unit of inference
  is the query, with partitions averaged within it. Paired differences from the full query with a
  bootstrap over queries; construction sizes are compared on common eligible cohorts.
- **Hypotheses and families, frozen before the full run.**
  - Primary: on held-out positives of the same training query, a 1-1 construction-only mask changes
    the paired position against the full query (two-sided, no change as the null), in reciprocal
    rank, log rank, top 10, and top 100, for every environment and budget. One Holm family.
  - Secondary: shared and 1-1 masks against the full query on all-label nDCG@10, MRR@10, and
    Recall@100, for every environment and budget. One Holm family, which includes construction
    positives and so can reward a mask for lifting the documents that built it.
  - Reading rule: an all-label gain without a held-out gain is memorization of construction documents,
    not better retrieval of relevant documents. A table sets the two side by side, with the
    construction positives' own rank change.
  - Holm holds under the strong dependence across budgets; no adjustment is made across families.
    Everything else, including prefix-k, the shared mask's position changes, own masks, larger cells,
    and common cohorts, is descriptive with pointwise exploratory intervals.
- **Competing scores.** Corpus mean and spread, and the score at ranks 1, 10, and 100, are recorded
  for every mask, so the rise of positives can be set against the rise of others. A small seeded
  sample of queries also stores the top non-positive documents of a mask, scored under the full
  query. This is diagnosis, not negative mining; nothing is fitted to it.
- **Execution.** Each corpus chunk is scored against every mask once; device memory is bounded by
  chunk and mask batch size. Results are compared with the shared-mask run: same input hashes, and
  regret recomputed from these masks must match what that run stored. A pilot measures cost first;
  a stratified cap with stated weights is used if the eligible queries are too costly.

## Review decision on scope

Recorded on 2026-11-09 as the reviewer's decision after the pilot, not inferred from silence: all
eligible multi-positive queries of all three environments, every arm, the nine-budget grid (5,118
SciFact, 1,463,276 NFCorpus, and 148,607 MS MARCO masks). The run waits until no compute process is
on the shared GPU and it is idle across repeated checks, with a brief check of smaller batches first.

## Execution record

The full run completed once on the frozen snapshot. Its first report generation stopped on an
out-of-range index that later generations did not repeat, with the same code and inputs; the cause is
unresolved. Stored results were kept, checked on the CPU with bounded spot checks and invariants, and
reported once more. See `reports/P3-mask-ranking/RUNS.md`.

On MS MARCO, query and passage ids are numeric and collide, so the query-id exclusion above removed
one unrelated passage from each query's competition: the corpus was not ranked in full there. The
run is kept as written; `RUNS.md` records the exact correction of rank-based measures and its effect.

## Consequences

- Results on the positives that built a mask, and on training queries generally, are not a measure
  of generalization or of a learned selector.
- A rank increase of a held-out positive under a mask built from others of the same query says
  nothing about relevance causing it: relevant documents may lie close to each other.
- Float64 costs about 1.4 times float32 on the host's GPU, and buys device-independent ranks and exact
  ties; ties between duplicate documents are real, and float32 would break them differently per device.
