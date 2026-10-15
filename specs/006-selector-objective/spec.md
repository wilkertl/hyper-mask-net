# Feature Specification: F5 Selector Objective and Experiment S1

**Feature Branch**: `agent/F5-selector-objective`

**Created**: 2026-10-14

**Status**: Draft

**Input**: User description: "F5 selector objective: train the per-environment L2S linear selector
with a retrieval (ranking) loss through a differentiable soft top-k mask instead of KL to the oracle
distribution; budget-aware; evaluated against L2S at every 10% budget." Context: ADR-0008 (proposed
with this spec) makes the per-environment selector the project's contribution and defers the
hypernetwork to an automation step; E0 and E1 (`reports/E0/`, `reports/E1/`) are the evidence.

## User Scenarios & Testing *(mandatory)*

The users are the researcher, who decides from the S1 report whether a retrieval-trained selector
replaces Learning-to-Select (L2S) as the project's selector, and the agents who build on that
selector afterwards. The question S1 answers: with the same linear selector, the same frozen
embeddings, and the same training labels, does training the selector for the retrieval it is used
for, rather than for matching the oracle's importance distribution, retrieve better at the budgets
that matter?

The evidence that motivates it (all from generated reports):

- At k = 256 the oracle mask reaches far higher nDCG@10 than the L2S selector in every environment
  (`reports/E0/tables/g1.csv`): better dimension choices exist that L2S does not find.
- The ADR-0003 recipe variants (wider grid, validation stopping, regularization toward the global
  selector) gained significantly over the official recipe only on SciFact: hyperparameters are
  exhausted.
- On MS MARCO, where training data is most abundant, the in-domain selector does not beat the
  global selector at most budgets and stays below full 1024 at every budget below 100%
  (`reports/E1/tables/progression.csv`): the limit is not data.
- The L2S loss scores the whole predicted distribution, while retrieval uses only its top-k
  ranking; on NFCorpus the chosen target is close to uniform (ADR-0006).

### User Story 1 - A selector trained for retrieval at a budget (Priority: P1)

The researcher trains, for one environment, a linear selector of the same shape as L2S whose loss
is the retrieval quality of the masked query: training queries scored against their positive and
hard-negative documents after keeping only the selector's top dimensions. The researcher picks the
hyperparameters on validation queries and gets a checkpoint with its manifest that the existing
evaluation can score like any other selector.

**Why this priority**: it is the whole hypothesis of F5; every other story measures or extends it.

**Independent Test**: on a small synthetic environment whose relevant documents differ from the
negatives only in a known set of dimensions, the trained selector ranks those dimensions first and
its masked retrieval beats an untrained selector; removing test judgments leaves the checkpoint
unchanged.

**Acceptance Scenarios**:

1. **Given** an environment's training queries, positives, and hard-negative pools, **When** the
   retrieval-trained selector is trained at budget k, **Then** the loss it minimizes depends on the
   query's scores against its positives and negatives after masking, and the masked query is not
   re-normalized.
2. **Given** a trained checkpoint, **When** it is evaluated, **Then** the evaluation uses the same
   hard top-k mask, exact scoring, and query sets as E1, so its numbers are comparable with the L2S
   selector's cell by cell.
3. **Given** test and evaluation-set judgments removed from the prepared environment, **When** the
   selector is trained again with the same seed, **Then** the checkpoint is identical.
4. **Given** a grid of candidate settings, **When** one is chosen, **Then** the choice uses
   validation queries and training labels only, by the rule fixed before any test number exists.

---

### User Story 2 - One selector for every budget (Priority: P1)

The researcher trains a variant that sees a different budget at each step, drawn from the 10%–90%
grid, so one checkpoint serves every budget, and compares it with the fixed-budget variant.

**Why this priority**: the best budget differs by environment (E1 progression: SciFact peaks near
40%, NFCorpus near 70%, MS MARCO only at 100%), so a selector tuned for one k is the wrong unit; a
budget-free selector is also what a later policy that chooses k needs.

**Independent Test**: on the synthetic environment, the multi-budget selector's top-k sets are
nested as k grows and its masked retrieval at each budget is within a stated tolerance of the
fixed-budget selector trained at that budget.

**Acceptance Scenarios**:

1. **Given** the 10%–90% budget grid, **When** the multi-budget variant trains, **Then** every
   budget of the grid is used during training and the budget seen at each step is recorded by the
   seed, so a rerun reproduces it.
2. **Given** a fixed-budget and a multi-budget checkpoint of one environment, **When** both are
   evaluated, **Then** each is scored at every 10% budget, not only at the budget it trained for.

---

### User Story 3 - Experiment S1 against L2S at every budget (Priority: P1)

The researcher trains the chosen retrieval-trained variant with three seeds per environment
(SciFact, NFCorpus, MS MARCO), evaluates it with the L2S selectors of E1, the global selector,
prefix, random, and full 1024, and reads a generated S1 report.

**Why this priority**: it produces the evidence for or against replacing L2S.

**Independent Test**: on synthetic result files with a planted improvement in one environment, the
report finds that environment and no other; regenerating it from the same files gives identical
bytes.

**Acceptance Scenarios**:

1. **Given** the three seeded checkpoints per environment and E1's seeded L2S selectors, **When** S1
   runs, **Then** every method is scored at every 10% budget on every test set and on MS MARCO's
   TREC DL 2019, DL 2020, and DL-HARD sets.
2. **Given** the per-query results, **When** the report is generated, **Then** it states, per
   environment, the difference against L2S in mean nDCG@10 over the 10%–90% budgets and at k = 307,
   each with a paired test, a 95% confidence interval, and a Holm-corrected p-value over the
   family of environments, seeds averaged per query before testing, as in E1.
3. **Given** the same report, **When** it is read, **Then** it shows the nDCG@10-by-budget curve of
   each method per test set (the E1 progression table extended with the new selector) and the
   difference against full 1024 at each budget.
4. **Given** the report, **When** the researcher reads its verdict section, **Then** each criterion
   states its value, its test, and whether it holds, with no number typed by hand, and an ADR is
   drafted from it.

---

### User Story 4 - Know where the gain comes from (Priority: P2)

The researcher sees, per environment, how the retrieval-trained selector's choices differ from
L2S's: their top-k overlap per budget, its agreement with the oracle mask on validation queries,
and the training and validation curves, so a gain or a loss can be explained rather than only
measured.

**Why this priority**: without it, a positive result cannot be attributed to the objective and a
negative one cannot inform the next attempt (a different capacity, or a budget policy).

**Independent Test**: on the synthetic environment the reported overlap between two identical
selectors is 1 at every budget and the overlap with a random selector is near k/1024.

**Acceptance Scenarios**:

1. **Given** both selectors of an environment, **When** the report is built, **Then** it shows
   their mean top-k overlap per budget on test queries, with the seed-to-seed overlap of each as a
   noise floor.
2. **Given** a retrieval-trained selector, **When** its fit is reported, **Then** it is stated as
   top-k agreement with the oracle mask on validation queries (the E1 rule), next to L2S's.

---

### Edge Cases

- A training query whose positives are all missing from the corpus embeddings: it is skipped and
  counted in the manifest, as in the oracle-pair construction.
- A hard-negative pool that repeats a document (pools are sampled with replacement): repeats are
  kept or collapsed by a rule fixed in research, the same for every environment.
- A tie at the k-th selector score: the hard mask breaks ties the way E0 and E1 do, so the
  comparison with L2S is not affected by tie handling.
- The soft mask's sharpness makes training unstable or the soft and hard masks disagree: the run
  records the agreement between the soft and hard top-k on validation queries, and a setting whose
  agreement falls below the rule fixed in research is not selectable.
- An environment where no retrieval-trained setting beats L2S on validation: the chosen setting
  still goes to S1, and the report states the validation result; nothing falls back silently to L2S.
- MS MARCO's training set is far larger than the others: training time per epoch is bounded by
  sampling a fixed number of steps, stated in research, so every environment gets a comparable
  budget of updates.
- k = 1024: every selector equals full 1024; the report takes the 100% column from full, as E1
  does.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The retrieval-trained selector MUST have the same form as the L2S selector (one
  1024 × 1024 linear map with bias on the frozen query embedding, producing one score per
  dimension); only its training objective and settings differ (constitution IV: no added
  capacity).
- **FR-002**: Training MUST minimize a loss on the retrieval scores of the masked training query
  against its positive and hard-negative documents, where the mask is a differentiable
  relaxation of keeping the top-k dimensions of the selector's scores; the masked query MUST NOT be
  re-normalized and document embeddings MUST NOT change.
- **FR-003**: Training MUST use only training-split judgments and the hard-negative pools F2
  already mined from training queries; validation judgments are used only to choose settings and
  checkpoints; test and evaluation-set judgments MUST NOT be read.
- **FR-004**: The system MUST offer a fixed-budget variant (trained at one k) and a multi-budget
  variant (a budget from the 10%–90% grid at each step).
- **FR-005**: The system MUST offer initialization from scratch and from the environment's L2S
  checkpoint (seed-matched), so the report can separate the effect of the objective from the
  effect of a better start.
- **FR-006**: Settings MUST be chosen per environment on validation nDCG@10 averaged over the
  10%–90% budgets, by one grid and rule shared by all environments and fixed in research before
  any test evaluation.
- **FR-007**: Each trained checkpoint MUST be saved with a run manifest and a record of its
  variant, initialization, settings, seed, inputs, and validation scores, and MUST load into the
  existing evaluation without changes to how other selectors are scored.
- **FR-008**: The chosen setting MUST be trained with seeds 0, 1, and 2 per environment, one run
  per seed.
- **FR-009**: S1 MUST score, on every test set and TREC DL set, at every 10% budget: the
  retrieval-trained selector (each seed), E1's L2S selector (each seed), the global selector,
  prefix, random, global importance, and full 1024, with exact scoring.
- **FR-010**: The S1 report MUST be generated from result files and contain: the nDCG@10-by-budget
  table and curves per test set; per environment, the paired difference against L2S in mean
  nDCG@10 over 10%–90% and at k = 307, with 95% bootstrap confidence intervals and Holm-corrected
  p-values; the difference against full 1024 per budget; AP on the TREC DL sets; top-k overlap
  with L2S per budget with the seed noise floor; oracle-mask agreement; training and validation
  curves.
- **FR-011**: The report MUST state the protocol of every table (in-domain), include full 1024 and
  prefix rows, and regenerate byte-identically from the same result files.
- **FR-012**: The report MUST end with a verdict section that evaluates the S1 criteria
  (SC-001, SC-002) and a drafted ADR recording the decision.
- **FR-013**: A test MUST show that the relaxed mask approaches the hard top-k mask as its
  sharpness grows, on a hand-computable example, and that at k = 1024 the masked query equals the
  full query.
- **FR-014**: A test MUST show that removing test and evaluation-set judgments from an environment
  leaves every trained checkpoint unchanged.
- **FR-015**: The L2S trainer, its checkpoints, and the E0 and E1 results MUST stay unchanged; the
  retrieval objective is added beside the KL objective, not in place of it.

### Key Entities

- **Retrieval-trained selector**: a linear selector checkpoint with its variant (fixed-budget at k,
  or multi-budget), initialization (scratch or L2S), settings, seed, and validation scores.
- **Training example**: a training query with its positive documents (graded judgments > 0) and
  its hard-negative pool, all from the training split.
- **S1 result**: per method, seed, test set, and budget, the per-query nDCG@10 (and AP on TREC DL),
  in the same record format as E0 and E1.
- **S1 comparison**: per environment and summary (mean over 10%–90%, or k = 307), the paired
  difference against L2S with its interval, p-value, and Holm-corrected p-value.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001** (S1 criterion 1): the retrieval-trained selector's mean nDCG@10 over the 10%–90%
  budgets is higher than L2S's with Holm-corrected p ≤ 0.05 in at least 2 of the 3 environments.
- **SC-002** (S1 criterion 2): in no environment is it lower than L2S, on the same summary, with
  Holm-corrected p ≤ 0.05.
- **SC-003**: Regenerating the S1 report from the same result files gives identical bytes.
- **SC-004**: 0 training steps read a test or evaluation-set judgment, verified by tests.
- **SC-005**: On a synthetic environment with a planted set of relevant dimensions, the
  retrieval-trained selector recovers at least 90% of that set at its size, and its masked
  retrieval beats the untrained selector.
- **SC-006**: Training one seed of the chosen setting finishes within 2 hours per environment on
  the single RTX 4090, and the full S1 evaluation within 30 minutes (E1's took 11.5 minutes).
- **SC-007**: Every number in the report traces to a result file with a run manifest; 0 numbers
  are typed by hand.

## Assumptions

- ADR-0007 (G2 passes) is accepted before F5 merges, and ADR-0008 (proposed with this spec)
  re-scopes the roadmap: the per-environment selector is the contribution, and the space
  descriptor and hypernetwork (Milestones 3 and 4) wait until S1 is decided.
- Inputs are F2's published artifacts and E1's seeded L2S selectors on DataPolicy-4090, and S1
  runs there, so every number comes from one machine.
- The environments are SciFact, NFCorpus, and MS MARCO, as in E0 and E1; more environments are a
  separate feature.
- The baseline is E1's `wide-grid` L2S selector with seeds 0–2, not retrained.
- The relaxed mask, the loss (e.g. softmax over the positive and negatives), the sharpness
  schedule, the budget sampling, and the grid are research decisions of the plan; this spec fixes
  only that the loss is computed on masked retrieval scores and that the evaluation mask is the
  hard top-k.
- The summary over budgets excludes 100%, where every selector equals full 1024.
- The criteria count environments, not TREC DL sets; TREC DL results are reported as context for
  MS MARCO.
- Higher capacity (low-rank, MLP), per-query budget policies, and soft re-weighting of dimensions
  instead of masking are out of scope; S1's result decides whether they are next.
