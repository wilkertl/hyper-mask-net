# Feature Specification: F7 Geometric Prior (experiment S2)

**Feature Branch**: `agent/F7-geometric-prior`

**Created**: 2026-10-15

**Status**: Draft

**Input**: User description: "Go ahead" on specifying S2 after P1 (`reports/P1/`): the label-free
score `q_d² · Var(x_d)` approximates the full query (best on MS MARCO, where it beats L2S and the
S1 selector), while learned selectors reshape the ranking toward relevance (best on SciFact). S2
trains S1's retrieval selector on top of the geometric score, so it starts as the geometric
method and learns only a correction. Rules below and in research.md are fixed before any S2 run.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - A selector with a geometric prior (Priority: P1)

The researcher trains, per environment, a selector whose score is
`log(q_d² · Var(x_d) + ε) + (W q + b)_d`, with the corpus variance computed from document
embeddings only, `W` and `b` the L2S-shaped linear map starting at zero, and the S1 retrieval
objective. At step 0 it is the P1 `variance` method.

**Why this priority**: it is the hypothesis: the prior carries the approximation that the learned
selectors lack, and training adds the reshaping that the prior lacks.

**Independent Test**: an untrained geometric selector ranks dimensions exactly as
`variance_importance`; on the planted synthetic set it recovers the planted dimensions like the
S1 selector; removing test judgments leaves checkpoints unchanged.

**Acceptance Scenarios**:

1. **Given** an environment, **When** the geometric grid trains, **Then** every cell starts from
   the prior with a zero linear map, uses S1's loss, relaxed mask, budgets, steps, and checkpoint
   rule, and is chosen on validation by S1's rule.
2. **Given** a saved geometric selector, **When** it is loaded, **Then** it reproduces its scores
   exactly (the variance is stored with the checkpoint) and is evaluated like any selector.

### User Story 2 - Experiment S2 (Priority: P1)

The researcher compares, on every test and TREC DL set at every 10% budget, the geometric
selector (3 seeds) with the S1 selector, L2S, the `variance` method, and the S1 baselines, and
reads a generated report with a drafted ADR.

**Acceptance Scenarios**:

1. **Given** the per-query results, **When** the report is generated, **Then** it states per
   environment the paired difference against the S1 selector and against `variance` (mean
   nDCG@10 over 10%–90%, and at k = 307), with bootstrap intervals and Holm over the three
   environments per comparison; seeds are averaged per query first.
2. **Given** an environment whose selected cell keeps its step-0 checkpoint for every seed,
   **When** the report is read, **Then** it says that there the geometric selector is the
   `variance` method itself.

## Requirements *(mandatory)*

- **FR-001**: The geometric selector MUST have the L2S linear map as its only learned
  parameters; the prior is fixed (constitution IV: no added learned capacity).
- **FR-002**: The corpus variance MUST come from document embeddings only (no judgment) and be
  stored in the checkpoint.
- **FR-003**: Training MUST reuse S1's objective, relaxed mask, examples, step count, checkpoint
  rule, validation scoring, and soft-mass guard unchanged.
- **FR-004**: The S1 code paths, checkpoints, and results MUST stay unchanged; the S1 report MUST
  regenerate byte-identically.
- **FR-005**: The S2 report MUST be generated from result files only, with full and prefix rows,
  protocol, byte-identical regeneration, and a drafted ADR-0010.

## Success Criteria *(mandatory)*

- **SC-001**: the geometric selector's mean nDCG@10 over 10%–90% is above the S1 selector's with
  Holm-corrected p ≤ 0.05 in at least 2 of 3 environments.
- **SC-002**: in no environment is it below the `variance` method on the same summary with
  Holm-corrected p ≤ 0.05 (it keeps the prior's strength).
- **SC-003**: each geometric grid finishes within 3 hours and `compare-selectors` within 30
  minutes on DataPolicy-4090 when the GPU is not shared.

## Assumptions

- Inputs: F2 artifacts, E1's seeded L2S selectors, S1's grids, seeded selectors, and run.
- `ε = 1e-12`; embeddings are L2-normalized, so `q_d² · Var(x_d)` is of order 1e-6.
- The S2 grid is S1's grid with the start fixed to the prior: 2 variants × 2 learning rates ×
  2 score temperatures × 2 final mask temperatures = 16 cells.
