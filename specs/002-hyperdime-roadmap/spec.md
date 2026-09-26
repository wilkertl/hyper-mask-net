# Feature Specification: HyperDIME-Qwen Project Roadmap

**Feature Branch**: `docs/hyperdime-roadmap`

**Created**: 2026-09-25

**Status**: Draft

**Input**: User description: "HyperDIME-Qwen whole-project roadmap: an environment-conditioned hypernetwork that generates low-rank Learning-to-Select dimension selectors for frozen Qwen3-Embedding-0.6B (D=1024), following docs/plan/Qwen3_HyperDIME_Implementation_Plan.en.md end to end — M0–M3 reproduction (Gate G1), M4 selector shift (G2 Go/No-Go), M5 space descriptor and H2 probes (G3), M6–M9 space encoder, low-rank selector, hyper head, episodic meta-training, M10–M12 inference protocols P0/P1/P2, selection policies, exact evaluation, ablations, significance and efficiency (G4, G5). The plan decomposes the project into per-milestone spec-kit features; 001-m0-foundation is the first."

## Overview

This is an umbrella specification. It states what the whole project must demonstrate and in what
order, and it is realized through one spec-kit feature per milestone (`specs/001-m0-foundation`
is the first). Each milestone ends in a gate decision; a failed gate stops later milestones and is
recorded as a result, not worked around.

The project's claim is: *measurable changes in the retrieval environment of one frozen embedding
model explain changes in which of its 1024 dimensions each query should use, and those changes can
be converted, without retraining a selector in the new environment, into useful selector
parameters.* It is tested as three separately refutable hypotheses:

- **H1 — Selector shift**: the best dimension selector differs systematically across environments.
- **H2 — Geometry explains the shift**: statistics of the embedding space, computed without test
  labels, predict part of that difference.
- **H3 — Hypernetwork adaptation**: a network that maps those statistics to selector parameters
  beats a single global selector on environments it never trained on.

**Hypothesis first.** Before any foundation work, dataset expansion, or hypernetwork code, a small
spike (F0) tests H1 on five small public datasets with the code that already exists plus a thin
pipeline. Everything after it (User Stories 1–5, the full dataset pool) proceeds only if the spike
supports H1.

## User Scenarios & Testing *(mandatory)*

### User Story 0 - Quick test of selector shift (Priority: P0) — spike F0

The researcher embeds five small environments, trains one Learning-to-Select selector per
environment (three seeds each) and one global selector on all of them, and gets a transfer matrix
and mask-similarity analysis within days, not weeks. The result says whether H1 holds strongly enough
to justify the rest of the project.

**Why this priority**: H1 is the premise of everything else. If selectors do not shift, the
hypernetwork is unnecessary and the full roadmap should not be built.

**Independent Test**: Run the spike end to end with one command per step on one GPU and read
the verdict in the generated report.

**Acceptance Scenarios**:

1. **Given** SciFact, NFCorpus, FiQA-2018, ArguAna, and SciDocs, **When** per-environment and global
   selectors are trained, **Then** a 5 × 5 transfer matrix of nDCG@10 at a 30% retained-dimension
   ratio is produced, with full-1024, random, prefix-truncation, and global-selector rows.
2. **Given** three seeds per environment, **When** masks are compared on a common query battery,
   **Then** the report shows whether masks differ more across environments than across seeds.
3. **Given** the results, **When** the verdict is computed, **Then** it states H1 as supported or
   not, and the headroom (per-domain minus global nDCG@10) the hypernetwork could close.
4. **Given** no evaluation query, **When** selectors are trained, **Then** no evaluation label is
   used in training, negative mining, or targets.

The users are the researcher (who makes gate decisions and writes the paper), coding agents (who
implement tickets), and reviewers or readers (who must be able to reproduce every number).

### User Story 1 - Valid, reproducible baseline (Priority: P1) — Milestone 1, Gate G1

The researcher embeds three retrieval environments with the frozen model once, reproduces the
Learning-to-Select baseline on them, and compares it with full-dimension, prefix-truncation,
random, global-importance, and oracle selection, all with exact scoring. Every number traces to a
manifest.

**Why this priority**: Nothing later is interpretable without a correct mask pipeline and a
baseline that learns its target.

**Independent Test**: Run the reproduction on one environment from a clean clone and check that
the selector's loss converges, full-dimension selection equals unmasked retrieval, oracle selection
is the upper bound, and the report regenerates identically from result files.

**Acceptance Scenarios**:

1. **Given** cached embeddings for SciFact, NFCorpus, and the full MS MARCO passage corpus, **When**
   the baseline selector is trained, **Then** its training loss decreases and its held-out KL to the
   oracle is below that of a uniform predictor.
2. **Given** the MS MARCO selector, **When** it is evaluated on the TREC DL 2019, DL 2020, and
   DL-HARD query sets, **Then** the report lists nDCG@10 and AP next to the full-dimension run, in
   the same form as the DIME papers.
3. **Given** a selection budget of all 1024 dimensions, **When** retrieval runs, **Then** rankings
   equal unmasked exact retrieval.
4. **Given** the E0 report, **When** it is regenerated from result files, **Then** every table is
   identical and each row names its manifest.

---

### User Story 2 - Go/No-Go on selector shift (Priority: P2) — Milestone 2, Gate G2

The researcher trains one selector per environment, evaluates every selector on every environment,
compares their importance distributions and masks, and records whether selector shift is real and
matters for retrieval. If it is not, the project pivots to a universal selector and stops the
hypernetwork work.

**Why this priority**: The hypernetwork is only justified if H1 holds; this is the cheapest point
to find out.

**Independent Test**: Produce the transfer matrix and behavior analysis with one command and read
the recorded decision.

**Acceptance Scenarios**:

1. **Given** per-environment selectors, **When** the transfer matrix is built, **Then** it contains
   in-domain and cross-domain retrieval metrics plus mask-behavior measures for every ordered pair.
2. **Given** the matrix, **When** paired per-query tests with multiple-comparison correction are
   applied, **Then** the decision record states H1 as supported or not, with confidence intervals.
3. **Given** a No-Go, **When** the decision is merged, **Then** later milestones are closed as not
   planned and the evidence is kept.

---

### User Story 3 - Evidence that space statistics predict the shift (Priority: P3) — Milestone 3, Gate G3

The researcher computes per-dimension descriptors of each environment from allowed data only and
tests, with probes and permutation controls, whether they predict selector properties better than
a trivial baseline. The winning descriptor version is frozen.

**Why this priority**: Without H2 the hypernetwork has no signal to read.

**Independent Test**: Build descriptors for all environments, run the probes, and check the
permutation-test result and the frozen descriptor version.

**Acceptance Scenarios**:

1. **Given** an environment, **When** its descriptor is built twice with documents in different
   orders, **Then** the descriptors are identical; **and** permuting dimensions permutes its rows.
2. **Given** descriptors and per-environment selectors, **When** probes run, **Then** at least one
   feature set predicts a selector property above the trivial baseline under a permutation test,
   or the gate is recorded as failed.
3. **Given** protocol P0, **When** a descriptor is built, **Then** no query or label data is read.

---

### User Story 4 - Working hypernetwork prototype (Priority: P4) — Milestone 4

The researcher meta-trains the descriptor encoder and hypernetwork over environment episodes and
checks, on meta-validation environments only, that the generated selectors match or beat the
global selector before touching meta-test.

**Why this priority**: It is the project's contribution, but only worth building after G2 and G3.

**Independent Test**: Meta-train on the meta-train environments and compare the generated
selector with the global selector on meta-validation.

**Acceptance Scenarios**:

1. **Given** a zero hypernetwork output, **When** a selector is generated, **Then** it equals the
   global base selector exactly.
2. **Given** a config that lists a meta-test environment for training, **When** meta-training
   starts, **Then** it refuses to run.
3. **Given** a trained prototype, **When** evaluated on meta-validation, **Then** the result is
   recorded as match/beat/lose against the global selector, and meta-test stays unevaluated if it
   loses.

---

### User Story 5 - Paper-grade evaluation (Priority: P5) — Milestone 5, Gates G4 and G5

The researcher evaluates once on meta-test environments under protocols P0 (corpus only) and P1
(unlabeled calibration queries), plus P2 as a separate scenario, runs ablations and sensitivity
studies, measures offline and online cost separately, and generates every table from result files.

**Why this priority**: It turns the prototype into a publishable, defensible result.

**Independent Test**: Regenerate all tables E3–E6 from result files with one command.

**Acceptance Scenarios**:

1. **Given** meta-test environments, **When** the main comparison runs, **Then** every table
   includes full-dimension, prefix-truncation, per-domain, global, transferred, hypernetwork P0,
   hypernetwork P1, and oracle rows, each labeled with its protocol.
2. **Given** the efficiency study, **When** reported, **Then** encoding, selector, masking and
   scoring, and search times are separate, and offline adaptation cost is separate from per-query
   cost.
3. **Given** the ablation matrix, **When** one descriptor group or component is removed at a time,
   **Then** each ablation has its own result file and manifest.

### Edge Cases

- A gate fails: later milestones close as "not planned", an ADR records the evidence, and the
  repository is tagged at the decision.
- Too few truly distinct environments: extra environments from shards or subdomains are allowed
  but are reported as not equivalent to unseen domains.
- Prefix truncation alone matches query-aware selection: that is a reportable negative result, not
  a reason to drop the comparison (risk R5).
- The hypernetwork's gain matches a global selector trained with the same labels: the result is
  attributed to supervision, not to the hypernetwork (risk R6).
- Pseudo-feedback in P1 is circular: P0 remains the control and P1 is always labeled as such.
- A licensed collection is unavailable: Robust04 requires the TREC disks 4 and 5 license; without
  it, the environment is skipped, and every report says so, while meta-test still holds at least
  two public datasets.
- Two query sets share a corpus (FEVER and Climate-FEVER; MS MARCO and the TREC DL sets): corpus
  embeddings are computed once and shared, and the pair is never presented as two unseen domains.
- Documents longer than the encoder's input window (Robust04 news articles): they are truncated to
  a fixed, recorded length.

## Requirements *(mandatory)*

### Functional Requirements

**Spike F0 — Hypothesis test (runs first)**

- **FR-000**: The spike MUST test H1 on SciFact, NFCorpus, and FiQA-2018 (training qrels) and
  ArguAna and SciDocs (test-only: a fixed random half of the queries trains, the other half
  evaluates), with full corpora, the frozen model, and exact scoring.
- **FR-000a**: The spike MUST reuse the existing oracle, selector, loss, mask, retrieval, metric,
  split, and negative-mining code, and add only a dataset loader, an encoder with an embedding
  cache, a training loop, and the transfer-matrix analysis, each with unit tests.
- **FR-000b**: Every spike output MUST be written with a lightweight run manifest (git commit and
  dirty flag, config, seed, package versions), and its report MUST be generated from result files.
- **FR-000c**: The spike report MUST end with a recorded H1 verdict. Requirements FR-001 onward
  apply only if the verdict is "supported".

**Milestone 1 — Reproduction (M0–M3, E0)**

- **FR-001**: The project MUST provide versioned records, run manifests, deterministic seeding and
  hashing, and validated configs before any other module writes artifacts (feature 001).
- **FR-002**: The project MUST load SciFact, NFCorpus, and the full MS MARCO passage corpus into
  one normalized form, with query-level train/validation/test splits and guards that keep test
  labels out of every training path. The TREC DL 2019, DL 2020, and DL-HARD query sets MUST be
  loadable as extra evaluation sets over the MS MARCO corpus.
- **FR-003**: The project MUST embed all environments once with the frozen model at 1024
  dimensions and reuse the cached embeddings in every later milestone.
- **FR-004**: The project MUST compute oracle importance targets from positives and hard negatives
  (aggregating multiple positives), train the linear Learning-to-Select selector, and provide the
  full, prefix-truncation, random, global-importance, oracle, and per-domain baselines.
- **FR-005**: The project MUST evaluate with exact scoring and report nDCG@10, MRR@10, MRR@100, and
  Recall@100 with per-query values, plus average precision (AP) on the TREC collections, and
  produce the E0 report and the G1 decision.

**Milestone 2 — Scientific premise (M4)**

- **FR-006**: The project MUST train independent per-environment selectors and build an
  environment-by-environment transfer matrix with retrieval metrics and mask-behavior measures
  (divergence of importance distributions, overlap of selected dimensions, rank correlation).
- **FR-007**: The project MUST provide paired per-query significance tests with multiple-comparison
  correction and confidence intervals, and record the G2 decision in an ADR.

**Milestone 3 — Space representation (M5)**

- **FR-008**: The project MUST build per-dimension environment descriptors in five groups (document
  marginal, query marginal, interaction, contrast, redundancy), each restricted to the data its
  protocol allows, with normalization fitted on meta-train environments only.
- **FR-009**: The project MUST run probes with permutation tests relating descriptors to selector
  properties, record the G3 decision, and freeze a descriptor version.

**Milestone 4 — Prototype (M6–M9)**

- **FR-010**: The project MUST provide a descriptor encoder that preserves dimension identity, a
  selector parameterized as a global base plus a low-rank environment-specific change, and a
  hypernetwork that generates that change starting near zero.
- **FR-011**: The project MUST meta-train over environment episodes, refuse to load meta-test
  environments during training, and compare with the global selector on meta-validation before any
  meta-test evaluation.

**Milestone 5 — Paper-grade evaluation (M10–M12, E3–E6)**

- **FR-012**: The project MUST implement adaptation protocols P0, P1, and P2 as separate,
  labeled scenarios, with P0 and P1 unable to read labels.
- **FR-013**: The project MUST keep importance prediction separate from the number-of-dimensions
  policy, providing fixed top-k and fixed ratio; an adaptive-k policy inspired by RDIME is provided
  only after reproduction and is reported as a heuristic unless its assumptions hold.
- **FR-014**: The project MUST run the E3 main comparison, the E4 ablation matrix, E5 sensitivity
  studies, and E6 efficiency analysis, and record G4 and G5 decisions.
- **FR-015**: The project MUST generate every table and figure from result files with run
  manifests, with one command per experiment.

**Cross-cutting**

- **FR-016**: Every milestone MUST be delivered as its own spec-kit feature, planned only after the
  previous milestone's gate is decided.
- **FR-017**: Approximate search MAY be added only for the deployment analysis and never for
  scientific results.
- **FR-018**: The environment pool MUST cover every dataset used in the experiments of the
  reference papers on dimension importance (DIME, Getting off the DIME, CoDIME, Eclipse, Unveiling
  DIME, Statistical Foundations of DIME) and on Learning to Select:
  - the MS MARCO passage corpus with its dev queries and the TREC DL 2019, DL 2020, and DL-HARD
    query sets;
  - TREC Robust 2004;
  - the 13 publicly available BEIR tasks used in Unveiling DIME: ArguAna, Climate-FEVER, DBPedia,
    FEVER, FiQA-2018, HotpotQA, NFCorpus, NQ, Quora, SciDocs, SciFact, Touché-2020, TREC-COVID.

  The meta-train / meta-validation / meta-test assignment MUST be fixed and recorded in an ADR
  before Milestone 3 fits any descriptor normalization. Environments that share a corpus MUST NOT be
  split across meta-train and a held-out set unless the held-out one is reported as a
  query-shift-only environment. Meta-test MUST contain at least two whole datasets that are never
  used for gradients, normalization, or model selection.
- **FR-019**: Results on a dataset that a reference paper also reports MUST include that paper's
  metric and cutoff (nDCG@10 and AP for the TREC collections, nDCG@10 for BEIR), so the numbers can
  be compared.

### Key Entities

- **Environment**: a corpus, a query distribution, and an instruction, embedded by the frozen
  model; the unit of adaptation and of meta-level splits.
- **Selector**: maps a query embedding to an importance distribution over 1024 dimensions; global,
  per-environment, or generated.
- **Oracle target**: the importance distribution derived from a query's positives and negatives.
- **Space descriptor**: per-dimension statistics of an environment (1024 rows), built under a
  named protocol.
- **Gate decision**: the recorded outcome (pass/fail) of G1–G5 with its evidence and ADR.
- **Result file**: the output of an evaluation, with a manifest, from which tables are generated.
- **Milestone feature**: the spec-kit feature that delivers one milestone.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Every gate G1–G5 that is reached has a merged decision record and a repository tag,
  whether it passed or failed.
- **SC-002**: 100% of reported numbers regenerate from result files with manifests; 0 numbers are
  edited by hand.
- **SC-003**: 0 code paths used for training, adaptation, negative mining, descriptors, or
  normalization can read test labels or meta-test labels, verified by tests.
- **SC-004**: With all 1024 dimensions selected, retrieval matches unmasked retrieval on 100% of
  queries.
- **SC-005**: Each main-result claim (E3) is backed by a paired per-query test with multiple-
  comparison correction and a 95% confidence interval.
- **SC-006**: If H3 holds, the generated selector closes a measurable fraction of the gap between
  the global selector and the per-domain selector on meta-test environments, under P0 or P1.
- **SC-007**: Online per-query overhead of selection is reported separately and does not exceed the
  retrieval-time saving from using fewer dimensions at the reported k (G5).
- **SC-008**: A new reader can reproduce the E0 report from a clean clone by following the README
  and one command per step.
- **SC-009**: 100% of the datasets used in the reference papers' experiments have an environment in
  the pool (Robust04 subject to its license) and appear in at least one report with the reference
  paper's metric.

## Assumptions

- Qwen3-Embedding-0.6B stays the only encoder; other models are out of scope for this project.
- The spike's results are exploratory: they decide whether to invest, and are rerun under the
  full protocol (F1–F4) before any are reported in a paper.
- Gates G1 and G2 use SciFact, NFCorpus, and MS MARCO (with the TREC DL query sets); the other
  datasets are loaded and embedded before Milestone 3, so their cost is paid only if H1 holds.
- All datasets are public except TREC Robust 2004, whose documents (TREC disks 4 and 5) require a
  NIST license; the researcher supplies them locally. Downloads happen outside Git and are recorded
  in dataset manifests.
- Full corpora are embedded, not samples, so results are comparable with the reference papers.
- A single GPU workstation is available for embedding and training; CPU is enough for tests.
- Each milestone's detailed requirements come from its own spec, which may refine but not
  contradict this roadmap.
- The budget-aware policy (plan M11, Policy D) is out of scope.
- The inter-dimension Transformer encoder, contrast and redundancy features, and larger selector
  capacity are added only if an ablation shows the simpler version is limiting.
