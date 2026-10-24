# Feature Specification: F10 Descriptor Probe (experiment E2, gate G3)

**Feature Branch**: `agent/F10-descriptor-probe`

**Created**: 2026-10-24

**Status**: Draft

**Input**: ADR-0012 (S4, proposed) and the researcher's request to run E2 on the three current
environments. E1 (ADR-0007) showed the selector shifts between environments (H1); S1–S4 showed
the shift matters for retrieval. E2 tests H2: per-dimension statistics of an environment's
embedding space, computed without labels (protocols P0 and P1), predict how that environment's
specialized selector ranks the 1,024 dimensions, beyond what is shared by every environment.

With three environments, an environment-level probe has three points. The unit of analysis is
therefore the dimension: each environment gives 1,024 pairs (descriptor row `R_T[j]`, selector
property `y_T[j]`), and the test is leave-one-environment-out (fit on two environments' 2,048
rows, predict the held-out environment's 1,024). Rules below are fixed before any E2 run.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - A per-dimension space descriptor (Priority: P1)

The researcher builds, per environment, a descriptor `R_T` of shape `1024 × h` from document
embeddings (P0) and unlabeled training queries (P1), stored with feature names, groups, protocol,
and a run manifest, and reads which groups it contains.

- **A, document marginal (P0)**: the existing marginal statistics, plus skewness, excess
  kurtosis, and the share of documents beyond 3 standard deviations of the dimension's mean.
- **E, redundancy (P0)**: per dimension, the mean and maximum absolute correlation with the
  other dimensions, the share of its variance in the top-r principal components (r ∈ {8, 32,
  128}), and its top-r correlation energy.
- **B, query marginal (P1)**: group A's statistics over unlabeled training queries.
- **C, interaction (P1)**: mean, standard deviation, and quantiles of `q_j · d_j` over random
  query–document pairs, and separately over each query's top-10 documents by full-query exact
  retrieval.
- **D, pseudo-contrast (P1)**: mean over queries of `q_j · (mean of top-K documents − mean of a
  random background sample)`, top-K from full-query exact retrieval, for K ∈ {5, 10, 20}.

**Independent Test**: on hand-built matrices each statistic matches a hand-computed value;
permuting documents or queries leaves the descriptor unchanged; permuting dimensions permutes its
rows; streamed chunks give the same result as one pass; the P0 descriptor reads no queries and no
P0 or P1 descriptor reads qrels (enforced by code and tested).

### User Story 2 - Per-dimension selector targets (Priority: P1)

The researcher derives, per environment and seed, the property each probe must predict from
existing selector checkpoints, on validation queries without their labels:

- **Primary target**: the S1 retrieval selector's selection frequency per dimension (share of
  queries keeping the dimension in their top k), averaged over the nine 10%–90% budgets. S1 has
  no geometric prior, so the target is not built from the variance descriptor itself.
- **Secondary targets**: the same per budget; the S1 selector's mean score per dimension; the
  selection frequency of the linear part `W q + b` of the S4 budget selector.

**Independent Test**: on a synthetic selector the frequency matches a hand count; it sums to k
per budget; targets read no labels; averaging seeds averages their profiles.

### User Story 3 - Experiment E2 (Priority: P1)

The researcher runs, for each held-out environment, simple linear probes from each descriptor
set to the target, compares them with two baselines, and reads `reports/E2/` and a drafted
ADR-0013 with the G3 reading.

**Independent Test**: on synthetic environments where the target is a known linear function of
one descriptor column plus noise, the probe beats both baselines and the permutation test
rejects; when the target is unrelated to the descriptor, it does not; the held-out environment
never enters fitting, model choice, or normalization.

### Edge Cases

- A descriptor column constant within an environment (e.g. a dimension with zero variance in a
  small corpus) is kept, and normalization does not divide by zero.
- Ties in the target (many dimensions never selected at small budgets) use average ranks in
  Spearman.
- MS MARCO's 8.8M documents are streamed; random pairs and background samples are drawn with a
  fixed seed and capped sizes, so the descriptor is reproducible.
- A query whose top-K contains fewer than K documents (tiny test corpora) uses what it has.
- Environment-level targets (S4's learned `a_k`, E1 transfer degradation) have 3 points or 6
  pairs; they are tabulated next to environment-level descriptor summaries, never tested.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The descriptor uses document embeddings (P0) and the training split's query
  embeddings (P1, at most 10,000 queries chosen by a seeded hash of query ids); it never reads
  qrels, validation or test queries, or any label (constitution II). Each descriptor is stored as
  a `SpaceDescriptor` with its protocol and a run manifest.
- **FR-002**: Targets use the selected S1 and S4 cells' seeds 0, 1, 2 and the validation
  queries' embeddings only; validation and test judgments are never read.
- **FR-003**: Descriptor sets form a fixed ladder: (1) variance only, (2) A, (3) A + E,
  (4) A + E + tails, (5) + B + C, (6) + D. Sets 1–4 are P0, 5–6 are P1; every result is labeled
  with its protocol.
- **FR-004**: The probe is ridge regression on the target's ranks. Feature standardization and
  the ridge penalty (chosen from {0.1, 1, 10, 100, 1000} by the inner two-fold
  leave-one-environment-out on the training environments) use the training environments only.
- **FR-005**: Baselines, per held-out environment: (B1) dimension identity, the mean of the two
  training environments' target profiles, which is what a selector shared by all environments
  predicts; (B2) the held-out environment's own `variance`-method selection frequency, averaged
  over the nine budgets (label-free).
- **FR-006**: Controls: a permutation test that independently permutes descriptor rows within
  each environment and refits (1,000 permutations, seeded); a residual probe that predicts the
  held-out target's deviation from B1 from the descriptor's deviation from the training mean,
  reported as the direct test of "geometry explains the shift"; the seed noise ceiling, the mean
  pairwise Spearman between seeds' target profiles per environment; standardized probe
  coefficients per feature.
- **FR-007**: Environment-level context, descriptive only: S4's learned `a_k` per environment,
  E1's transfer degradation per pair, and environment summaries (participation ratio, mean
  energy, descriptor means).
- **FR-008**: Existing code paths, artifacts, and reports (E0, E1, P1, S1–S4) stay unchanged and
  regenerate byte-identically.
- **FR-009**: The E2 report is generated from result files only: protocol per row, the ladder
  per fold and seed, both baselines, permutation p-values, the residual probe, the noise ceiling,
  coefficients, the environment-level table, a note that these environments shaped S1–S4, and a
  drafted ADR-0013 with the G3 reading. It regenerates byte-identically.

### Key Entities

- **Space descriptor**: per environment and protocol, a `1024 × h` table with feature names and
  groups, from embeddings only.
- **Selector target profile**: per environment, seed, and target kind, a value per dimension.
- **Probe result**: per held-out environment, descriptor set, target, and seed: Spearman on the
  held-out environment, baselines, permutation p-value, chosen penalty, coefficients.

## Success Criteria *(mandatory)*

Spearman correlation between predicted and actual primary-target profiles on the held-out
environment's 1,024 dimensions, per fold; target averaged over seeds unless stated.

- **SC-001 (G3, provisional)**: at least one descriptor set has Spearman above both B1 and B2 on
  at least 2 of the 3 held-out environments, with permutation p ≤ 0.05 after Holm correction over
  the six descriptor sets within each fold, and stays above both baselines for each of the three
  seeds' targets in those environments.
- **SC-002**: the protocol of the best passing set is reported: G3 is read as holding under P0 if
  a P0 set (1–4) passes SC-001, otherwise under P1 if set 5 or 6 does.
- **SC-003**: building all three descriptors finishes within 3 hours, and the full probe
  (6 sets × 3 folds × targets × 1,000 permutations) within 1 hour, on DataPolicy-4090.

Secondary, reported but not decisive: the residual probe, per-budget and secondary targets,
the noise ceiling, and the environment-level table.

## Assumptions

- Stacked on `agent/F9-budget-prior` (S4, not yet merged); inputs are F2 artifacts (embeddings,
  splits), the selected S1 and S4 cells' seeds, and E1's transfer table, all on DataPolicy-4090.
- The three environments already shaped S1–S4, so E2's reading on them is exploratory; G3 is
  confirmed only by rerunning this spec unchanged on the environments of #60, outside this feature.
- Three environments cannot separate "descriptor predicts the shift" from "these three corpora
  happen to differ this way"; the dimension-level design increases sample size but the folds
  remain three. The report states this.
- Out of scope: new environments (#60), the space encoder and Hyper Head (gate G4), labeled
  contrast (protocol P2), and redundancy-aware selectors (an S-track experiment).
- The descriptor stays provisional (`schema_version` 0); freezing `SpaceDescriptorV1` (M5.5)
  follows a G3 reading on more environments.
