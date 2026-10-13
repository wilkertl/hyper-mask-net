# Feature Specification: F4 Selector Shift and Gate G2

**Feature Branch**: `agent/F4-selector-shift`

**Created**: 2026-10-12

**Status**: Draft

**Input**: User description: "F4 selector-shift (roadmap feature F4, task T034; backlog M12.3, M4.1–M4.3, gate G2 issue #20; experiment E1; research R8 and R9; ADR-0003 and ADR-0006). On the F2/F3 artifacts for SciFact, NFCorpus, and MS MARCO (full 8.8M-passage corpus; TREC DL 2019, DL 2020, DL-HARD as extra evaluation sets), retest H1 (selector shift) with the per-environment selector that F3 selected on validation (ADR-0003). M4.1 per-environment selectors, 3 seeds; M4.2 `hyperdime transfer-matrix` over every ordered pair at k = 307 with paired tests, bootstrap CIs, and a transferred fixed-subset baseline (ADR-0006); M4.3 selector behavior on a common unlabeled query battery against the seed noise floor; selector fit tested as a ranking against the oracle mask (ADR-0006); M12.3 paired tests and generated `reports/E1/`; G2 per research R8; MS MARCO deviation and NFCorpus caveats; a time-boxed read of the authors' MS MARCO setup before any run; exit with reports/E1/ and a drafted G2 ADR."

## Clarifications

### Session 2026-10-12

- Q: Which selector does F4 use per environment, given that F3 picked `global-regularized` for
  MS MARCO? → A: `wide-grid` in all three environments (FR-001). The global anchor could shrink the
  measured shift, and one recipe family keeps the comparison like for like; on MS MARCO it trails
  F3's pick by 0.0002 validation nDCG@10.
- Q: Is the transferred fixed subset (ADR-0006, Pasin et al.) a G2 criterion? → A: No. It is
  reported context; G2 keeps research R8's two criteria (FR-007, FR-015).

## User Scenarios & Testing *(mandatory)*

The users are the researcher, who decides gate G2 from the E1 report, and the agents who would
implement F5 (space descriptor) only if G2 passes. The question G2 answers is H1: do selectors
trained on different environments select different dimensions, beyond what retraining with a new
seed changes, and does that difference cost retrieval quality when a selector is moved to another
environment? If not, a hypernetwork that adapts the selector to an environment is not needed.

### User Story 1 - Per-environment selectors with a seed noise floor (Priority: P1)

The researcher trains, for SciFact, NFCorpus, and MS MARCO, the per-environment selector chosen in
F3 with three seeds each, and gets every checkpoint with its manifest and its in-domain test metrics
per seed.

**Why this priority**: every later measurement compares these selectors; without several seeds per
environment there is no null for "different because of the environment" versus "different because
of training noise".

**Independent Test**: Train SciFact with three seeds; check that each checkpoint reloads with
identical outputs, that a repeated run of one seed gives the same checkpoint, and that removing test
judgments leaves every checkpoint unchanged.

**Acceptance Scenarios**:

1. **Given** an environment and its F3 `wide-grid` cell, **When** selectors are trained for seeds
   0, 1, and 2, **Then** each uses that cell, differs only in its seed, and is saved with a
   manifest recording the recipe, cell, seed, and inputs.
2. **Given** a trained seed, **When** it is evaluated in-domain at k = 307, **Then** its test
   nDCG@10 (and AP on the TREC DL sets) is saved per query with a manifest.
3. **Given** test judgments removed from the prepared environment, **When** selectors are trained
   again, **Then** they are identical.
4. **Given** a selector, **When** its fit to its training target is tested, **Then** the result is
   stated as a ranking test (top-k agreement with the oracle mask against the global-importance
   baseline, under the rule fixed in F4 research), not as KL against a uniform predictor.

---

### User Story 2 - The transfer matrix (Priority: P1)

The researcher runs one command that evaluates every selector on every environment's test queries
at k = 307 and gets the A×B table: for each ordered pair (trained on A, tested on B), retrieval
metrics, mask similarity to B's in-domain selector, and the in-domain minus cross-domain difference
with a paired randomization p-value (Holm-adjusted over the ordered pairs) and a 95% bootstrap
interval. The same table carries the full-1024, MRL prefix, random, global-importance, and
transferred fixed-subset rows for each test environment.

**Why this priority**: G2 criterion 1 is read from this table.

**Independent Test**: On synthetic environments with a planted per-environment dimension subset,
the matrix shows in-domain beating cross-domain; with identical environments, no pair is
significant. Running the command twice gives identical output.

**Acceptance Scenarios**:

1. **Given** trained selectors for three environments, **When** the matrix is built, **Then** it
   has all 9 cells (3 in-domain, 6 cross-domain), each naming the selector checkpoints and the
   result files it came from.
2. **Given** the command, **When** it runs, **Then** no model is trained or updated, and test
   judgments are read only to score rankings.
3. **Given** environment A's fixed subset (the k dimensions its in-domain selector selects most
   often on A's training queries), **When** it is applied to every query of B, **Then** its row is
   reported next to the cross-domain selector A→B and the in-domain selector of B.
4. **Given** the MS MARCO test environment, **When** the matrix is built, **Then** the TREC DL
   2019, DL 2020, and DL-HARD sets are evaluated as extra query sets with nDCG@10 and AP, reported
   separately from the six ordered pairs that G2 counts.

---

### User Story 3 - Selector behavior against the seed noise floor (Priority: P1)

The researcher compares what the selectors select on one shared battery of unlabeled queries from
all three environments: JS and KL divergence of mean importance, overlap@k of the selected
dimensions, Spearman correlation of dimension rankings, and per-dimension selection frequency. Every
cross-environment value is shown next to the within-environment seed-to-seed value, and the
cross-versus-within difference is tested.

**Why this priority**: G2 criterion 2 is read from this comparison; weight-space distances are not
evidence (parameter symmetries), so behavior on shared inputs is the only accepted measure.

**Independent Test**: With three seeds of one synthetic environment and one planted different
environment, the cross-environment overlap is significantly below the seed-to-seed overlap; with
two copies of the same environment, it is not.

**Acceptance Scenarios**:

1. **Given** the battery and all selectors, **When** behavior is measured, **Then** every measure is
   computed for each within-environment seed pair and each cross-environment seed pair, on the same
   queries.
2. **Given** the measures, **When** the report shows them, **Then** each cross-environment value is
   reported relative to the within-environment noise floor, with its test result.
3. **Given** the analysis, **When** it runs, **Then** it reads query embeddings only, never a
   relevance judgment, and never compares weights.

---

### User Story 4 - The E1 report and the G2 draft (Priority: P2)

The researcher runs one command to generate `reports/E1/` from the result files: the transfer
matrix, the behavior comparison, per-seed in-domain metrics, the selector-fit tests, the payoff of
in-domain selectors over the global selector and over the transferred fixed subset, and both G2
criteria of research R8 with their values and whether each holds. A second command drafts the G2
ADR from the same result files for the researcher to decide.

**Why this priority**: it is the milestone's output and the input to the G2 decision.

**Independent Test**: Regenerate the report twice and compare bytes; check that every row names its
result files and that the G2 criteria are computed, not typed.

**Acceptance Scenarios**:

1. **Given** the result files, **When** the report is generated twice, **Then** the outputs are
   identical.
2. **Given** every comparison table, **When** it is read, **Then** it includes full-1024 and
   prefix-k rows and states its protocol (in-domain or transfer) and scoring (exact).
3. **Given** MS MARCO or NFCorpus rows, **When** they appear, **Then** they carry the ADR-0006
   notes: MS MARCO does not reproduce Learning-to-Select, on the full corpus (naming any concrete
   setup difference found in the authors' code); NFCorpus's selector does not fit its oracle
   distribution.
4. **Given** the report, **When** the G2 draft is produced, **Then** it states each criterion's
   value, the evidence location, and the pass/fail reading, and leaves the decision to the
   researcher.

### Edge Cases

- Ties at the k-th score: the fixed subset and other rank-valued methods are built without ties
  (rank scores, ties broken by lower dimension index), so their masks are deterministic; selector
  outputs are continuous, and a repeated run gives the same masks.
- Different test-set sizes (MS MARCO dev is far larger than SciFact or NFCorpus test): paired tests
  are per test environment, on that environment's queries, so sizes never mix within a test.
- An environment contributes more battery queries than the others: the battery samples the same
  number of queries from each environment, with a fixed seed.
- A seed fails to train or produces a non-finite output: the run stops; G2 is not computed from fewer
  than three seeds per environment.
- The global selector and the global-regularized recipe use other environments' training labels:
  this is recorded wherever such a selector appears; test labels are never involved.
- Fewer than three selectors per environment exist when the report runs: the report refuses rather
  than computing the noise floor from fewer seeds.

## Requirements *(mandatory)*

### Functional Requirements

**Selectors (M4.1)**

- **FR-001**: For each of SciFact, NFCorpus, and MS MARCO, the selector used in F4 MUST be F3's
  `wide-grid` candidate: the same recipe and grid in every environment, with the τ × epochs cell
  F3 chose on that environment's validation queries. It MUST be trained with seeds 0, 1, and 2 and
  saved as a `SelectorCheckpoint` with a manifest per seed. For MS MARCO this departs from F3's
  validation pick (`global-regularized`), which is anchored to a global selector trained on all
  three environments; the report states the departure and both validation scores.
- **FR-002**: Each seed MUST be evaluated in-domain at k = 307 on the environment's test queries
  (and, for MS MARCO, on TREC DL 2019, DL 2020, and DL-HARD), saving per-query metrics.
- **FR-003**: Training MUST NOT read test or evaluation-set judgments, and MUST be reproducible for
  a seed.
- **FR-004**: Each selector's fit to its target MUST be tested as a ranking: top-k agreement of the
  selector's mask with the oracle mask on validation queries, compared with the global-importance
  baseline's agreement, under a rule fixed in F4 research before any F4 run (ADR-0006).

**Transfer matrix (M4.2)**

- **FR-005**: One command MUST evaluate every (trained-on A, tested-on B) pair over the three
  environments at k = 307 with exact scoring, without training or updating any model, and write the
  matrix as a table plus one evaluation result per cell with a manifest.
- **FR-006**: For each test environment B and each ordered pair A→B with A ≠ B, the matrix MUST
  report in-domain minus cross-domain nDCG@10 per seed-averaged selector, a paired randomization
  p-value over B's test queries, Holm-adjusted over the six ordered pairs at α = 0.05, and a 95%
  paired bootstrap interval.
- **FR-007**: Each test environment MUST also report full-1024, MRL prefix-k, seeded random-k,
  global importance, and the fixed subset transferred from each other environment (that
  environment's in-domain selector's k most frequently selected dimensions on its own training
  queries, applied to every query of B). The transferred fixed subset is reported context for the
  G2 decision, with the in-domain selector's paired difference against it; it is not a G2
  criterion.
- **FR-008**: Mask similarity (overlap@k with B's in-domain selector on B's test queries) MUST be
  reported for every cell.

**Selector behavior (M4.3)**

- **FR-009**: A common battery MUST draw the same number of unlabeled test queries from each
  environment with a fixed seed; behavior is measured on query embeddings only.
- **FR-010**: On the battery, the analysis MUST compute, for every pair of selectors: JS and KL
  divergence of mean importance, mean per-query overlap@k, Spearman correlation of mean-importance
  dimension rankings, and the per-dimension selection frequency of each selector.
- **FR-011**: Within-environment seed pairs MUST form the noise floor; each cross-environment
  measure MUST be reported relative to it, and cross-environment overlap@k MUST be tested against
  within-environment overlap@k with a test fixed in F4 research before any run.
- **FR-012**: Analyses MUST NOT use distances between selector weights as evidence.

**Statistics and report (M12.3, E1, G2)**

- **FR-013**: Paired randomization tests, paired bootstrap intervals, and Holm correction MUST be
  promoted from the spike into the library with tests on synthetic per-query data with a known
  effect.
- **FR-014**: One command MUST generate `reports/E1/` (Markdown and CSV) from result files only:
  the transfer matrix, behavior against the noise floor, per-seed in-domain metrics, the
  selector-fit tests, the payoff over the global selector and the transferred fixed subset, and
  the G2 criteria; regenerating it MUST give identical bytes.
- **FR-015**: G2 criterion 1 MUST hold when cross-domain nDCG@10 is significantly below in-domain
  for at least 3 of the 6 ordered pairs (Holm, α = 0.05); criterion 2 when cross-environment
  overlap@k is significantly below within-environment overlap@k (research R8). The report states
  each value and whether it holds.
- **FR-016**: Every MS MARCO row MUST carry the ADR-0006 deviation note, and every NFCorpus row the
  weak-target caveat; before any F4 run, the authors' public code MUST be read (about one hour,
  not extended) for how it builds the MS MARCO corpus and queries, and any concrete difference
  named in the note.
- **FR-017**: One command MUST draft the G2 ADR and decision record from the result files for the
  researcher's review; the tag `g2-selector-shift` is applied to the merge of the accepted ADR.
- **FR-018**: All tests MUST run on CPU without network access.

### Key Entities

- **Seeded selector set**: the three seed checkpoints of one environment's `wide-grid` cell.
- **Transfer cell**: one (trained-on, tested-on) pair with metrics, mask similarity, the paired test
  against in-domain, and the result files it came from.
- **Query battery**: a seeded, environment-balanced sample of unlabeled test query embeddings.
- **Selector properties**: per-selector mean importance, selection frequency, and fit-to-target
  result on validation queries.
- **E1 report** and **G2 decision draft**: generated tables, both criteria, and the draft ADR.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 9 of 9 transfer cells and 3 of 3 seeds per environment are present, each traceable to
  its checkpoint and result files.
- **SC-002**: Regenerating the E1 report from the same result files gives identical bytes.
- **SC-003**: 0 training steps read a test or evaluation-set judgment, and 0 models change while the
  transfer matrix is built, verified by tests.
- **SC-004**: On synthetic data with a planted environment effect both G2 criteria hold, and on
  synthetic copies of one environment neither holds.
- **SC-005**: Every G2 criterion in the report states its value, its test, and whether it holds,
  with no number typed by hand.
- **SC-006**: Building the full transfer matrix, including MS MARCO's extra query sets, completes on
  the single RTX 4090 within 30 minutes (E0's comparable evaluation took 6.5 minutes).

## Assumptions

- Inputs are F2's published artifacts and F3's selections on DataPolicy-4090, and F4 runs there,
  so every number comes from one machine.
- The fixed budget is k = 307 (30%), the Learning-to-Select ratio and F3's selection budget.
- Seeds are 0, 1, and 2 (research R8 asks for three); for SciFact and NFCorpus, seed 0 reproduces
  F3's selected checkpoint.
- "Identical hyperparameters" (issue #17) means one recipe and one grid for all environments; the
  τ × epochs cell is chosen per environment on validation, as in F3.
- The battery uses test query embeddings without labels, a fixed number per environment set by the
  smallest test set; test labels are only read to score retrieval.
- Cross-domain cells use each source selector's three seeds; their per-query scores are averaged
  over seeds before the paired test, so each ordered pair is one test.
- The TREC DL sets are extra in-domain evaluation sets of MS MARCO; they are reported but not
  counted among the six ordered pairs of G2.
- Learning-to-Select and other papers' numbers are cited, never copied into result files.
