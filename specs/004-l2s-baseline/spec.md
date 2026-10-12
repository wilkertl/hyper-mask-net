# Feature Specification: F3 Learning-to-Select Baseline and Gate G1

**Feature Branch**: `agent/F3-l2s-baseline`

**Created**: 2026-10-12

**Status**: Draft

**Input**: User description: "F3 l2s-baseline (roadmap feature F3, backlog M12.1, M3.1–M3.3, experiment E0, gate G1; research R4, R6, R7; ADR-0002 recipe; ADR-0003 sample-efficient selector candidates). On the F2 artifacts for SciFact, NFCorpus, and MS MARCO with the TREC DL 2019, DL 2020, and DL-HARD query sets: exact masked evaluation with per-query metrics and AP; oracle importance targets; the Learning-to-Select selector trained per environment with the official recipe; the full, prefix-truncation, random, global-importance, and oracle baselines; the ADR-0003 candidates chosen on validation only; the generated E0 report in the Learning-to-Select and DIME formats; and the G1 decision."

## User Scenarios & Testing *(mandatory)*

The users are the researcher, who decides gate G1 from the E0 report, and the agents implementing
F4 (selector shift), who need a validated per-environment selector and an evaluation pipeline they
can trust. The question G1 answers: is the mask pipeline correct, and does the baseline selector
learn its target, before anything is built on top of it?

### User Story 1 - Trustworthy exact evaluation (Priority: P1)

The researcher evaluates any dimension-selection method on an environment's test queries, or on an
evaluation query set, at any retained-dimension budget, with exact scoring over the full corpus,
and gets mean and per-query metrics saved as a result with a manifest. Selecting all 1,024
dimensions gives exactly the unmasked rankings.

**Why this priority**: every number in E0 and in later milestones comes through this pipeline; a
masking or metric error would invalidate all of them.

**Independent Test**: Evaluate the full model and a k = 1024 mask on SciFact and compare rankings;
compare metrics on a fixture with an independent reference implementation.

**Acceptance Scenarios**:

1. **Given** any method at k = 1024, **When** it is evaluated, **Then** its rankings equal the
   unmasked rankings for every query.
2. **Given** a fixture run and judgments, **When** metrics are computed, **Then** nDCG@10, MRR@10,
   MRR@100, Recall@100, and AP match an independent reference implementation.
3. **Given** a TREC DL query set with graded judgments, **When** binary metrics (AP, MRR, Recall)
   are computed, **Then** only relevance ≥ 2 counts as relevant, while nDCG uses the grades.
4. **Given** an evaluation, **When** it finishes, **Then** the result stores per-query values for
   every metric, the method, k, environment, query set, and protocol, with a manifest.

---

### User Story 2 - A baseline selector that learns its target (Priority: P1)

The researcher builds oracle targets from training labels and the F2 negative pools, trains the
Learning-to-Select selector per environment with the official recipe (temperature and epochs
chosen on validation), and saves it as a selector checkpoint with a manifest.

**Why this priority**: G1 asks whether the baseline learns; F4 compares selectors across
environments, so the selector must be reproducible and loadable.

**Independent Test**: Train on SciFact and check that validation KL ends below the uniform
predictor's, that a reloaded checkpoint gives identical outputs, and that no test label was used.

**Acceptance Scenarios**:

1. **Given** an environment's training labels and negative pools, **When** targets are built,
   **Then** each is a distribution over 1,024 dimensions that sums to 1 with no non-finite values,
   computed only from training and validation labels.
2. **Given** a trained selector, **When** its validation KL is compared with a uniform predictor's,
   **Then** it is lower.
3. **Given** a saved selector, **When** it is reloaded, **Then** it produces identical outputs.
4. **Given** test judgments deleted from the prepared environment, **When** selectors are trained
   again, **Then** they are identical.

---

### User Story 3 - Mandatory baselines and the ADR-0003 candidates (Priority: P2)

Every method needed for E0 is available through the same evaluation: full 1,024, prefix
truncation (the first k dimensions), seeded random k, global importance (one ranking of dimensions
shared by all queries of an environment, from training targets), the Learning-to-Select selector,
the ADR-0003 sample-efficient candidates, and oracle top-k (labeled non-deployable). For each
environment, one per-environment selector is chosen among the official recipe and the candidates
by validation nDCG@10 alone.

**Why this priority**: E0 needs the baselines; ADR-0003 requires the candidates before F4 retests
H1.

**Independent Test**: Run every method on SciFact at k = 307 and check each produces a valid mask,
that random is reproducible by seed, and that the chosen selector is the one with the best
validation score.

**Acceptance Scenarios**:

1. **Given** each method, **When** it selects k dimensions, **Then** it produces exactly k per
   query, and at k = 1024 equals full scoring.
2. **Given** the candidates of ADR-0003 — a wider temperature-by-epoch grid, validation-stopped
   training, and regularization toward a global selector — **When** they are trained, **Then**
   each is tuned and chosen on validation queries only, and the chosen selector per environment is
   recorded with every candidate's validation score.
3. **Given** the global selector used for regularization, **When** it is trained, **Then** it uses
   the training labels of all three environments and never a test label.

---

### User Story 4 - The E0 report and the G1 decision (Priority: P2)

The researcher runs one command per environment to evaluate every method at retained ratios
0.1–1.0, then one command to generate the E0 report: a Learning-to-Select-style table (SciFact,
NFCorpus, MS MARCO; nDCG@10, with the 30% row), a DIME-style table (TREC DL 2019, DL 2020,
DL-HARD; nDCG@10 and AP across ratios), paired significance against full and against the selector,
and the four G1 criteria of research R7. The report regenerates byte-identically from result files.

**Why this priority**: it is the milestone's output and the input to the G1 decision.

**Independent Test**: Regenerate the report twice and compare bytes; check every row names its
result file.

**Acceptance Scenarios**:

1. **Given** the result files, **When** the report is generated twice, **Then** the outputs are
   identical, and each table row names the result it came from.
2. **Given** the report, **When** the G1 criteria are read, **Then** each states its value and
   whether it is met: validation KL below uniform; k = 1024 identical to unmasked; oracle ≥
   selector ≥ random at k = 256 by nDCG@10; byte-identical regeneration.
3. **Given** every comparison table, **When** it is read, **Then** it includes full-1024 and
   prefix-k rows and states its protocol (in-domain) and scoring (exact).

### Edge Cases

- A query with no relevant document in a query set: excluded from that set's metrics, counted.
- k larger than 1024 or below 1: refused.
- Ties in masked scores: ranking is still deterministic for a fixed run.
- MS MARCO evaluation memory: the corpus is streamed in chunks; the score matrix never exists.
- A candidate's grid choice lies on the grid edge: recorded in the result, as R20 did for F0.
- The oracle uses evaluation labels by definition: it is labeled non-deployable and never feeds
  training or selection.

## Requirements *(mandatory)*

### Functional Requirements

**Evaluation (M12.1)**

- **FR-001**: Evaluation MUST rank the full corpus by exact inner product between masked queries
  and unmasked documents (the masked query is not re-normalized), streaming the corpus.
- **FR-002**: Metrics MUST include nDCG@10, MRR@10, MRR@100, Recall@100, and AP (to depth 1,000),
  with per-query values, and MUST match an independent reference implementation on fixtures.
- **FR-003**: Binary metrics MUST use the query set's relevance threshold: 2 for the TREC DL sets,
  1 otherwise; nDCG uses the graded judgments.
- **FR-004**: Each evaluation MUST be saved as F1's `EvaluationResult` (environment, protocol,
  split or query set, scoring "exact", method, k, mean and per-query metrics) with a manifest.

**Targets and selectors (M3.1, M3.2)**

- **FR-005**: Oracle targets MUST follow research R4 as updated by ADR-0002 (positives weighted by
  2^rel − 1, mean of the F2 sampled negatives, softmax at temperature τ) from training and
  validation labels only.
- **FR-006**: The Learning-to-Select selector MUST be trained per environment with the ADR-0002
  recipe, choosing τ and epochs by validation nDCG@10 at k = 307, and saved as a
  `SelectorCheckpoint` with its training manifest.
- **FR-007**: Training MUST be reproducible for a seed, run on the GPU when available, and never
  read test or evaluation-set judgments.

**Baselines and candidates (M3.3, ADR-0003)**

- **FR-008**: Full, prefix-k, seeded random-k, global importance, selector, and oracle top-k MUST
  be available as methods that produce importance scores for the shared top-k selection.
- **FR-009**: The ADR-0003 candidates MUST be trained per environment: (a) a wider τ × epochs
  grid, (b) validation-stopped training, (c) regularization toward a global selector trained on all
  three environments' training labels; each tuned on validation, and one selector chosen per
  environment by validation nDCG@10 at k = 307.

**Report and gate (E0, G1)**

- **FR-010**: One command per environment MUST evaluate every method at retained ratios 0.1, 0.2,
  …, 1.0 on the test queries and every evaluation query set.
- **FR-011**: One command MUST generate `reports/E0/` (Markdown and CSV) from result files only,
  with the Learning-to-Select-style and DIME-style tables, paired randomization tests
  (Holm-adjusted) with 95% bootstrap intervals, and the G1 criteria; regenerating it MUST give
  identical bytes.
- **FR-012**: All tests MUST run on CPU without network access.

### Key Entities

- **Target set**: oracle distributions for an environment's training and validation queries at one
  temperature.
- **Selector checkpoint**: a trained linear selector (kind linear, per_domain, or global) with its
  training manifest and the recipe and validation scores that chose it.
- **Evaluation result**: F1's `EvaluationResult` for one method, k, environment, and query set.
- **E0 report**: generated tables and the G1 criteria.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: k = 1024 reproduces unmasked rankings for 100% of queries in every environment and
  query set.
- **SC-002**: Metrics agree with the reference implementation to within 1e-6 on fixtures.
- **SC-003**: In every environment, the trained selector's validation KL is below the uniform
  predictor's.
- **SC-004**: In every environment, nDCG@10 at k = 256 orders oracle ≥ selector ≥ random.
- **SC-005**: Regenerating the E0 report from the same result files gives identical bytes.
- **SC-006**: 0 training or selection steps read a test or evaluation-set judgment, verified by
  tests.
- **SC-007**: The full E0 evaluation of all methods, budgets, and query sets for MS MARCO
  completes on the single RTX 4090 in under 3 hours.

## Assumptions

- Inputs are F2's published artifacts on DataPolicy-4090, and F3 runs there, so every number comes
  from one machine.
- The headline budget is 30% (k = 307) as in Learning to Select; G1's ordering check uses k = 256
  as research R7 states.
- Learning-to-Select reported numbers are cited from the paper, never copied into result files; E0
  compares trends, not exact values (backlog E0).
- Oracle targets are computed inside training from the published negatives rather than as a
  separate artifact; the checkpoint's manifest records τ and the inputs, so targets are
  reproducible.
- E0 is in-domain evaluation; no meta-level split is involved yet.
