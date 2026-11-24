# ADR-0017: P3 compares exact positive masks with Learning-to-Select on the same training queries

- **Status:** proposed
- **Date:** 2026-11-09
- **Issue:** none assigned; a ticket is still needed before this work gets a pull request

## Context

ADR-0016 ranked the corpus under masks built from positives and compared them with the full query.
It left out single-positive queries and never set those masks beside a selector. The objective is
now: exact query-positive masks, their relationship, full-corpus rankings under the ideal positive
mask, a direct comparison with Learning-to-Select (L2S), and what that says about whether its signal
can be learned for unseen queries. This ADR fixes the protocol before any result of the comparison
exists. The completed run of ADR-0016 is immutable and is not relaunched.

## Planned files

Recorded before editing.

- New: `evaluation/oracle_comparison.py`, `cli/analyze_oracle_l2s.py`, `reports/p3_oracle_l2s.py`,
  `tests/unit/evaluation/test_oracle_comparison.py`, `tests/integration/test_p3_oracle_l2s.py`.
- Changed: `cli/__main__.py` and `cli/report.py` (registration), `reports/p3_shared.py` (decile grid),
  `docs/experiment_registry.md` (one row).
- Generated: `reports/P3-oracle-l2s/`, and `reports/P3-shared-mask/` regenerated from its stored run,
  after a copy of the previous report is archived with hashes.

## Decision

Fixed before any outcome was seen.

- **Reporting grid.** Every P3 stage reports `floor(ratio * 1024)` for ratios 0.1 to 1.0: 102, 204,
  307, 409, 512, 614, 716, 819, 921, 1024. No budget is a headline, checkpoint, or gate, and 307 is
  only the 30% point. Stored all-k outputs stay. Scientific runs, and experiments outside P3, are not
  edited. The P3 report's budget tables already hold the nine ratios below 1.0, with 1024 in its
  curve tables; the shared-mask report moves to the grid. ADR-0016's pilot subset is history.
- **Masks.** `c = q * d`, exact on float16 inputs.
  - Pair-optimal mask: top `k` of `c` for one positive.
  - Query positive oracle: one top `k` of the relevance-weighted mean of `c` over the query's
    positives, with the whole-number gains of `shared_order`, applied to every corpus document.
- **Cohorts.** Training queries with a positive in the corpus, reported in two separate strata and
  never pooled in one headline.
  - Single-positive queries: the pair-optimal and query oracle masks are identical, so one mask
    serves both, and that identity is checked. No within-query relationship or transfer exists here.
  - Multi-positive queries: everything else, including relationship and disjoint transfer.
- **Relationship (multi-positive).** On the grid, query-balanced: overlap of pair masks, signed
  consensus of positives on the oracle mask's coordinates, regret against each positive's own
  optimum under the oracle mask, and under a mask built from construction positives and scored on
  disjoint held-out positives (the 1-1 partitions of ADR-0015).
- **Three things that are kept apart.**
  - The all-positive oracle maximizes the mean similarity of the query's positives. It saw them. It
    is a privileged ceiling for that score, and nothing else.
  - Held-out transfer is a mask built from some positives and scored on others of the same query.
  - Neither is the global ranking optimum, and neither is an upper bound on any ranking metric: a
    mask that raises positives can raise competitors more.
- **Ranking arms.** Exact float64 scan of ADR-0016 with its lower-row tie rule, same corpus and
  documents for every arm of a query: full, prefix, all-positive oracle, held-out transfer (eligible
  queries), L2S. Each positive's pair-optimal mask appears only as a labeled target-specific
  secondary diagnostic; it never enters a primary ranking comparison.
- **Measures.** The repository's `query_metrics` on the unchanged training judgments: nDCG@10,
  MRR@10, MRR@100, Recall@100. From exact ranks: reciprocal rank, log rank, top 10 and 100 of
  positives. Paired differences with a bootstrap over queries, query-balanced, wins and losses per
  query. Score changes of positives and of the documents at ranks 1, 10, and 100, against the full
  query. Mask agreement with the oracle: Jaccard, and score regret to the ceiling. All intervals are
  pointwise and exploratory.
- **MS MARCO.** No document is excluded by id: every run ranks the corpus in full, and records how
  many ids collide. The exact correction of ADR-0016's run is the check that links the two.
- **L2S provenance gate.** Before any scan, for each environment: the checkpoint's recorded inputs
  are the F2 dataset, embeddings, and negatives this run reads (no F0 crossover); matching dataset
  fingerprints and model revision; payload hash; a clean training tree; one known seed; split query
  ids disjoint. A failed check stops the run. The checkpoint stores no query ids, so its training
  queries are established through the dataset fingerprint and the training code path, and the
  record says so.
- **What the L2S comparison can claim.** The oracle needs judgments, and only training judgments may
  build it, so the same-query comparison runs on queries L2S was trained on. Its numbers are
  **in-sample** and are labeled so. They say how the oracle and a fitted selector rank the selector's
  own training queries, never how L2S does on unseen queries. Report averages from other experiments
  are not compared.
- **Conventions.** Published L2S evaluation is float32 with ties left to `topk`. Here L2S masks use
  the same importance and a stable order, and the run measures the gap to `evaluate_methods` on the
  same queries.
- **Execution.** Paired pilot first, CPU on the host when its GPU is busy, starting with exact SciFact.
  Larger environments wait for the measured cost, under the bounds below.

## Bounded follow-up pilots

Recorded on 2026-11-10, after the SciFact pilot and before any NFCorpus or MS MARCO result of this
comparison existed. Same protocol, one run and one manifest per environment, each under its own root
and with its own report, so the SciFact run and report stay as they are.

- **Bounds.** NFCorpus at most 128 queries, MS MARCO at most 32, every decile, every document.
- **Sampling.** The two cohorts are never pooled, so a bound is split evenly between them (any
  surplus goes to the other cohort); proportional allocation would leave MS MARCO two multi-positive
  queries. Within a cohort, by seeded hash of the id. Multi-positive queries are stratified by
  positive count with the existing strata, and reported with weights `N_h / n_h`, resampled inside
  those strata. SciFact was uncapped, so this does not touch it.
- **Device.** The GPU only when it holds no compute process and at least 20 GB is free, at every one
  of five checks. Otherwise CPU, niced, 8 threads for NFCorpus. Exact float64 scores do not depend on
  the device.
- **MS MARCO on CPU.** First a probe of 4 queries, stopped at 300 s. The 32-query run goes ahead on
  the CPU only if that probe finishes and its scan rate projects the run, at a ceiling of 3,000 masks,
  to at most 900 s; it is stopped at 1,200 s. Otherwise the blocker is recorded and nothing else runs.
  Full MS MARCO is never launched from an extrapolation.
- **MS MARCO collisions.** Query ids equal to passage ids stay in the competition. For each such
  passage the run records its exact score under every mask, the check that a passage scoring above
  rank 100 is in the retrieved list, and which ranks it charged. The historical ranks, which excluded
  it, must equal the new ranks minus those charges (ADR-0016's correction rule).
- **Parity at 100%.** Every arm at `k = 1024`, including held-out and pair-optimal masks, must give
  the full query's ranks, and the retrieved lists of the list arms must be the same lists.
- **Duplicate launches.** Each root is claimed atomically once, and a claim that exists stops the
  launch.

## Reusing historical arms in a large run

Not implemented; a large run must satisfy all of this first. Arms of ADR-0016's run
`20261109T163931Z-9e9f5a0a` may stand in for new scans only when:

- the query, mask definition, partition, positive position, and budget are the same, which restricts
  reuse to multi-positive queries, to the arms full, prefix, all-positive (`shared`), 1-1, and
  pair-optimal (`own`), and to the nine budgets below 1024 plus the full query;
- the input hashes (documents, queries, training judgments, model revision) match the new run's;
- the bounded pilots above reported zero mismatches for that environment;
- on MS MARCO, ranks are corrected with the recorded rule, each passage's charge being computed from
  its exact score under that mask; top-100 lists are reused only if the passage is shown to be
  below rank 100 under that mask, and otherwise the mask is scanned again.

L2S arms, every single-positive query, and every `k = 1024` arm other than the full query have no
historical counterpart and are always scanned. The earlier float32 L2S evaluations use other queries
and conventions and are never reused here.

## Learnability, predefined and deferred

No selector is fitted now. A later fit needs its own design and evidence that warrants it. It will
use:

- query-disjoint train, tune, and evaluation cohorts assigned by a seeded hash inside the training
  split, fixed before fitting; targets come only from training judgments, with a train-only
  held-out fold for diagnostics if needed;
- query-only input at inference;
- a fixed global mask and an L2S trained on the same train cohort as controls;
- retrieval, mask agreement, and score agreement at every decile.

A fit to privileged positive scores alone cannot establish transfer. Claims about unseen queries wait
for that evaluation.

**Update 2026-11-12.** The researcher defined that evaluation as feature F12
([specs/013-p3-selector-benchmark](../../specs/013-p3-selector-benchmark/spec.md)). It departs from
the list above in three ways: the split pools F2 training and validation queries (not the training
split alone) into `p3-query-split-v1`; the L2S control is the authors' implementation retrained on
the fit split with their published parameters, not this ADR's F3 checkpoints; and no cohort is
stratified or weighted by positive count. The pilots under this ADR stay exploratory legacy evidence.

## Consequences

- The existing L2S checkpoints were tuned on validation nDCG@10 at k = 307, a dependency of those
  checkpoints that cannot change without retraining; they remain one seed per environment.
- Single-positive queries are most of SciFact and MS MARCO training, so the separate stratum matters
  and the multi-positive results of ADR-0016 do not describe them.
