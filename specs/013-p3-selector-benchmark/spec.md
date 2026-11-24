# Feature Specification: F12 P3 Selector Benchmark

**Feature Branch**: `agent/F12-p3-selector-benchmark` (not created; no ticket assigned)

**Created**: 2026-11-12

**Status**: Implemented and running (as of 2026-11-24). SciFact and NFCorpus are evaluated, and
MS MARCO is in progress; see [reports/P3-bench/RUNS.md](../../reports/P3-bench/RUNS.md). The
protocol is jointly proposed in
[ADR-0018](../../docs/decisions/ADR-0018-p3-selector-benchmark.md) (proposed until the cross-corpus
statistics are done).

> **Amendment 2026-11-24, documentation only.** Only this Status line changed. No requirement,
> scenario, or success criterion changed. The version fingerprinted at the protocol gate
> (`reports/P3-bench/freeze/protocol.json`, SHA-256 `b297c43d…`) is recoverable from code snapshot
> `6ce68a0` (`artifacts/p3-bench/code-snapshots/`).

**Input**: The researcher's decisions of 2026-11-12, after the exploratory P3 stages (ADR-0014 to
ADR-0017) and the three oracle-vs-L2S pilots. Those stages built masks from the very queries they
scored, and their L2S was in-sample, so none of them says how a selector does on unseen queries. This
feature defines the benchmark that can: a query-disjoint in-domain split, the frozen full vector and
the authors' Learning2Select as baselines, fit-query teacher masks, and query-only selectors
compared on held-out evaluation queries. The project's focus is the selector (ADR-0008).

[research.md](research.md) holds the background: B0 records the exploratory stages and their
provenance, kept separate from this benchmark; B1 the split; B3 the source checks. The rules are in
ADR-0018.

## User Scenarios & Testing *(mandatory)*

### User Story 0 - Legacy P3 evidence stays as recorded (Priority: P1)

The researcher can find every completed P3 stage, its run id, code snapshot, report, and ADR, marked
exploratory, and can see which of its methods carry into this benchmark.

**Independent Test**: research B0 lists every P3 report directory; the experiment registry has one
row per stage and one planned row for this benchmark; no legacy report is regenerated.

### User Story 1 - Valid baselines on one split (Priority: P1)

The researcher reads, per corpus, the frozen full vector and `l2s-official` evaluated on evaluation
queries at every decile, with the split fingerprint and the authors' commit on every table.
`l2s-official` is the authors' code, trained on fit only, with seed 42 in one process per corpus, the
published τ and epochs, and final-epoch weights.

**Acceptance Scenarios**:

1. **Given** the split manifest and its attached run manifest, **When** any stage loads queries,
   **Then** it checks the file hash and per-split fingerprints and refuses a mismatch.
2. **Given** the authors' `small_test` fixture, **When** the adapter and the authors' script run in
   separate processes with seed 42 on CPU, **Then** targets and final weights are identical.
3. **Given** any arm at k = 1024, **When** it is evaluated, **Then** its ranking equals the full
   query's for every evaluation query.
4. **Given** MS MARCO, **When** `l2s-official` is trained, **Then** the report flags that τ = 0.10 and
   50 epochs were published for the authors' 50k-passage subset and are applied unchanged.

### User Story 2 - Teacher masks on fit queries (Priority: P2)

On fit queries only, the researcher compares three teachers: positive masks; contrastive masks,
using the negatives `l2s-official` recorded; and rank-guided masks, from strict lexicographic
`(nDCG@10, GDR)` swap search started from both. For each, the researcher reads exact full-corpus
values, nesting across budgets, cost, and demotions of unjudged documents versus known negatives.

**Independent Test**: on a synthetic corpus small enough to enumerate every k-subset, the search
reaches the brute-force optimum of its objective. Swaps keep |mask| = k, and no accepted swap lowers
the objective. The final mask is never below either starting mask. No tune or evaluation id is read.

### User Story 3 - Query-only selectors against L2S (Priority: P2)

The researcher reads, per corpus, the three primary teacher arms against `l2s-official` on the
primary endpoint. Teacher arms share L2S's architecture, loss, recipe, and seed-42 RNG stream, and
differ from it only in their target. The researcher also reads them next to the controls: S1 from
scratch, S1 initialized from the existing official checkpoint, frozen full, global mask, and
`variance`.

**Independent Test**: fed `l2s-official`'s own targets, the teacher pipeline reproduces its weights
exactly. Replacing tune and evaluation labels leaves every teacher arm's weights unchanged.

### User Story 4 - Confirmation order (Priority: P3)

SciFact runs end to end first; NFCorpus and bounded MS MARCO follow under unchanged rules. If the
design changes after SciFact evaluation results are seen, SciFact becomes development data and the
same frozen candidate is tested on NFCorpus and MS MARCO.

### Edge Cases

- A query whose top-1000 pool, after positives are removed, has fewer than 64 documents. The
  authors' fallback draws from the whole corpus and may draw a positive. The adapter keeps that
  behavior and counts such queries.
- Exact score ties use one lower-row rule for every arm and inside the teacher objective. The
  authors' length-sorted order is used only inside the parity test.
- An MS MARCO passage whose id equals the query's id is handled by the ADR-0016 rule and reported in
  the checks.
- Duplicate document texts (NFCorpus) are tie-broken alike in every arm and counted.
- A rank-guided search with no improving swap returns its starting mask.
- An evaluation query with no positive judgment is not eligible. It is counted, not dropped silently.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Every stage MUST read queries only through `artifacts/p3-query-split-v1/manifest.json`
  with its existing ids. Before use, it MUST verify the file hash, seed `20261112`, the normalization
  (NFKC, casefold, whitespace), and the per-split fingerprints (research B1).
- **FR-002**: The official test split MUST NOT be read. TREC DL-19, DL-20, and DL-HARD are out of
  scope. Evaluation ids and labels MUST NOT reach negative mining, teacher generation, target
  generation, fitting, feature fitting, tuning, or caps. Tune labels MUST only enter ADR-0018's
  selection steps.
- **FR-003**: Queries MUST NOT be stratified, filtered, weighted, gated, or used to choose an endpoint
  by positive count. Each eligible evaluation query has weight 1, and all its judgments stay together.
- **FR-004**: `l2s-official` MUST be trained by the authors' implementation at commit
  `55f010f5f3e82a6d171ce0909793e72f9c066ec7` through an adapter, on fit queries, with the behavior of
  research B3, seed 42, one process per corpus, and final-epoch weights. It MUST record every
  sampled negative index, the device, the torch version, and the determinism flags. Exactly one
  `l2s-official` MUST be trained per corpus, on the same fit ids as the primary P3 arms (the bounded
  prefix if MS MARCO is bounded). No other L2S training (other seeds, a separate full-fit reference,
  temperature tuning) is part of F12.
- **FR-005**: Every arm MUST use exact float64 full-corpus scoring with frozen embeddings, fixed
  document vectors, no query renormalization, one tie rule, and the project deciles. 100% MUST equal
  the full vector; this is a parity check, not an outcome.
- **FR-006**: Teachers MUST follow ADR-0018 "Teachers":
  - the GDR formula as written there;
  - strict lexicographic acceptance;
  - both starting masks;
  - exact full-corpus acceptance and reported values;
  - caps fixed from a fit-only cost probe;
  - no optimality claim for real cases.
- **FR-007**: Reports MUST distinguish unjudged documents from known negatives.
- **FR-008**: The primary teacher arms MUST match `l2s-official` in architecture, loss, optimizer,
  schedule, epochs, checkpoint rule, seed, and RNG stream, and MUST differ only in the target
  `t_q = (1/9) Σ_k m_{q,k}/k`. The replay control check MUST pass before any teacher arm is trained.
- **FR-009**: Controls MUST follow ADR-0018 "Selectors":
  - S1-scratch and S1-official-init on the same eight cells (lr × `τ_s` × `β_end`), with ADR-0009's
    budget and checkpoint rules, step 0 eligible, the recorded L2S negatives, and the tune guard;
    S1-official-init reuses the existing `l2s-official` checkpoint;
  - frozen full;
  - the global mask by rank-guided selection frequency;
  - `variance`.

  Reports MUST give each arm's training compute and its number of selection candidates, and MUST
  state S1's larger selection budget as a limitation of its secondary comparison.
- **FR-010**: The primary endpoint, the one-sided test and intervals (`B = 100,000`, streams from
  `SeedSequence(20261112)`), the Holm family, the claim levels, the replication rule, and the
  confirmation scope MUST follow ADR-0018.
- **FR-010a**: Bounded work MUST follow ADR-0018's resource rule:
  - hash-ordered nested prefixes, with the same ids for every arm;
  - sizes chosen from fit-only probe measurements alone;
  - matched fit coverage for primary arms;
  - escalation instead of shrinking below the minimum coverage.

  The freeze gates (protocol, resource, selection, evaluation lock) MUST be passed in that order, and
  no evaluation label may be read before the fingerprinted evaluation-lock record.
- **FR-011**: Every artifact MUST carry a run manifest with the split fingerprint and, for L2S-based
  arms, the authors' commit. Every table MUST be generated from result files.
- **FR-012**: Offline cost (label generation, search, training) MUST be reported separately from
  online per-query cost.
- **FR-013**: Reports MUST describe results as a query-disjoint in-domain comparison against
  Learning2Select retrained under this protocol, not as a comparison with published numbers or a
  state-of-the-art claim.

### Key Entities

- **QuerySplit**: corpus; split (`fit`, `tune`, `evaluation`); sorted ids; fingerprint; seed;
  normalization; source F2 dataset run; verification record.
- **L2SNegatives**: per fit query, the sampled negative row indices and whether the fallback ran.
- **TeacherMask**: query id; budget; teacher (`positive`, `contrastive`,
  `rank-guided-from-positive`, `rank-guided-from-contrastive`, `rank-guided`); dimension indices;
  nDCG@10; GDR; accepted swaps; seconds.
- **BenchmarkArm**: name; training split; parameter count; seed; RNG replay record; checkpoint hash.
- **EvaluationRow**: corpus; arm; seed; query id; k; nDCG@10, MRR@10, Recall@100, nDCG@100.

## Success Criteria *(mandatory)*

- **SC-001**: The split manifest carries a run manifest with the researcher's 2026-11-12 verification
  record, and FR-001's check passes for all three corpora.
- **SC-002**: Three parity checks pass before any candidate is trained: `small_test` parity, 100%
  parity, and SciFact per-query parity against the authors' evaluator on their grid.
- **SC-003**: The replay control check passes, and the leakage test shows teacher-arm weights,
  masks, and targets unchanged when tune and evaluation labels are replaced.
- **SC-004**: The SciFact report gives, per teacher arm, the primary difference with its pointwise
  interval, its Holm decision, and its claim level, generated from result files, and it regenerates
  byte-identically.
- **SC-005**: NFCorpus and bounded MS MARCO reports exist under the same frozen rules, with
  realized sizes recorded at the resource gate. Any change after SciFact evaluation results are seen is
  recorded, and SciFact is labeled development data. Replication is stated only under ADR-0018's
  rule.

## Assumptions

- F2 embeddings, qrels, and corpora are the inputs; nothing is re-encoded.
- Compute is the existing shared RTX 4090 host; no paid resources.
- No ticket exists; the branch name above is a placeholder until one is assigned. A missing ticket
  does not block documentation work.
