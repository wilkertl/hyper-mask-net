# Feature Specification: F6 Geometry Probe (experiment P1)

**Feature Branch**: `agent/F6-geometry-probe`

**Created**: 2026-10-15

**Status**: Draft

**Input**: User description: "How can we improve retrieval gains from the geometry of the corpus in
the embedding space? Go ahead" with the two cheapest checks proposed in the conversation: a
label-free, query-aware variance baseline, and whether a corpus's effective dimensionality orders
its best budget. P1 decides whether S2 (a selector conditioned on corpus statistics) is worth
building. Rules below are fixed before any P1 number exists.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - A label-free geometric selector (Priority: P1)

The researcher scores, on every test set and TREC DL set, the importance
`q_d² · Var_corpus(x_d)` at every 10% budget, and compares it with L2S, the S1 retrieval selector,
global importance, prefix, random, and full 1024 from the S1 run, on the same queries.

**Why this priority**: it measures how much of what the selectors learn is corpus variance; if a
training-free score comes close, geometry carries most of the signal and S2 is promising.

**Independent Test**: on a synthetic corpus where one dimension carries a large constant offset
and another all the variance, the method keeps the varying dimension and its ranking equals the
full query's at k = 1 on that construction.

**Acceptance Scenarios**:

1. **Given** an environment's document embeddings, **When** the probe runs, **Then** per-dimension
   mean and variance are computed from documents only (no judgment of any split), streamed over
   the corpus.
2. **Given** the S1 run, **When** the report is generated, **Then** per environment it states the
   paired difference of the variance method against L2S, the S1 selector, and global importance
   in mean nDCG@10 over the 10%–90% budgets, with a 95% bootstrap interval and a p-value Holm-
   corrected over the three environments for each comparison.

### User Story 2 - Effective dimensionality against the best budget (Priority: P2)

The researcher sees, per environment, the corpus's participation ratio `(Σλ)² / Σλ²` over the
eigenvalues of its document covariance, the share of energy in the mean vector
`‖μ‖² / E‖x‖²`, and the budget at which L2S and the S1 selector reach their highest test nDCG@10.

**Why this priority**: E1 showed the best budget differs widely (SciFact ~40%, NFCorpus ~70%,
MS MARCO 100%); if effective dimensionality orders it, the budget can be predicted from the
corpus alone.

**Independent Test**: on independent columns with variances (4, 1, 1, 0) the participation ratio
is 36 / 18 = 2; the result does not depend on the streaming chunk size.

**Acceptance Scenarios**:

1. **Given** the three environments, **When** the report is generated, **Then** it lists the
   participation ratio, energy in the mean, and best budgets side by side and states that three
   environments are descriptive only (no test of the relation).

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Corpus statistics (mean, variance, covariance eigenvalues) MUST be computed from
  document embeddings only, streamed in chunks, and give the same result for any chunk size
  within tolerance.
- **FR-002**: The variance method MUST be `importance_d = q_d² · Var(x_d)`, top-k through
  `top_k_mask`, the masked query not re-normalized, documents unchanged.
- **FR-003**: The probe MUST score the variance method on the same query sets and budgets as the
  latest S1 run, so every comparison is paired per query against S1's result files.
- **FR-004**: The P1 report MUST be generated from result files only, with full 1024 and prefix
  rows, the protocol (in-domain), and regenerate byte-identically.
- **FR-005**: Every new function has a test; the participation ratio and the variance method have
  hand-computed tests.

## Success Criteria *(mandatory)*

- **SC-001** (reading 1): the report states, per environment, whether the variance method is
  significantly below, indistinguishable from, or above L2S over the 10%–90% budgets (Holm,
  alpha 0.05). "Promising for S2" means it recovers at least half of L2S's gain over global
  importance in at least 2 of 3 environments, i.e.
  `(variance − global) ≥ 0.5 · (L2S − global)` on the budget mean.
- **SC-002** (reading 2): the report states whether the ordering of participation ratios matches
  the ordering of L2S's best budgets across the three environments.
- **SC-003**: the probe completes within 30 minutes on DataPolicy-4090; the report regenerates
  byte-identically.

## Assumptions

- Inputs are F2's embeddings and the latest S1 `compare-selectors` run on DataPolicy-4090.
- Statistics are on the stored float16 document embeddings, accumulated in float64.
- P1 is a probe, not a gate: its readings inform whether to specify S2 and add environments.
