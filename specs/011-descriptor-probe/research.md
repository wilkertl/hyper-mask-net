# Research: F10 Descriptor Probe (experiment E2, gate G3)

Every decision is fixed before any E2 run. Code references are to `agent/F9-budget-prior`
(`5916e37`).

## D0. Shift check first

- **Decision**: `probe-descriptors --experiment E2 --targets-only` computes the targets (D6) and,
  per held-out environment, B1's Spearman with the seed-averaged primary target and the noise
  ceiling (D6). Gap = ceiling − B1. If the gap is below 0.05 in at least 2 of 3 environments, the
  run records `no_shift = true`; `report E2` and `decide G3` then read G3 as "not supported: no
  environment-specific shift beyond seed noise", and descriptors are not required. It reads no
  labels and no descriptor, and runs on existing checkpoints.
- **Rationale**: E1 measured shift on L2S selectors; S1's retrieval selectors may share more of
  their profile. If B1 already reaches the ceiling, no descriptor can beat it, and building
  groups A–E would spend the feature for nothing. The 0.05 threshold is fixed now, before any
  target is computed.
- **Alternatives**: no check (risks an uninterpretable failure); a significance test of the gap
  (three folds, one profile each; the ceiling already carries the noise).

## D1. Unit of analysis and folds

- **Decision**: one row per (environment, dimension), 3 × 1,024 rows. Three leave-one-environment-
  out folds: fit on two environments (2,048 rows), score on the held-out 1,024. Within a fold,
  the penalty is chosen by the inner two-fold leave-one-environment-out on the two training
  environments (fit on one, score on the other, mean Spearman).
- **Rationale**: H2 asks whether geometry explains the selector's *change* between environments.
  With three environments an environment-level regression has three points and no test; the
  descriptor is per dimension (`1024 × h`, plan M5) and so is the selector, so the dimension is
  the natural unit. Holding out a whole environment keeps the claim about a new environment, not
  about unseen dimensions of a seen one.
- **Alternatives**: random splits of dimensions (leaks environment identity, answers a different
  question); CCA between environment descriptor and selector distances (three points); waiting
  for #60 (the researcher chose to proceed on three environments).

## D2. A loader that cannot read labels

- **Decision**: a new `space.inputs.load_unlabeled(root, environment)` returns the document
  embeddings, the query embeddings, and query ids per split, from the dataset manifest,
  `queries.jsonl` (each `QueryRecord` carries its split), and the embeddings, and never opens a
  qrels file. `describe-space` uses only this loader. A test deletes every qrels file of a
  synthetic environment and checks the descriptor is unchanged; another replaces them with
  corrupt files and checks nothing reads them.
- **Rationale**: `training.inputs.load_inputs` builds oracle pairs and so reads training and
  validation judgments; constitution II requires P0/P1 code paths not to read qrels, enforced by
  code.
- **Alternatives**: `load_environment` with a flag (it opens training qrels by design; changing it
  touches F2's public API).

## D3. Group A: document marginals and tails (P0)

- **Decision**: per dimension, from document embeddings in `float64`:
  - streamed power sums `Σx, Σx², Σx³, Σx⁴`, `Σ|x|` give mean, population std, mean_abs, RMS,
    skewness `m3 / m2^{3/2}`, excess kurtosis `m4 / m2² − 3` (central moments from raw sums);
  - a second streamed pass gives the tail share, the fraction of documents with
    `|x − mean| > 3 · std`;
  - quantiles q05, q25, q50, q75, q95 (as `marginal_stats`, linear interpolation) on a fixed
    sample: all documents when there are at most 1,000,000, otherwise 1,000,000 rows chosen by a
    seeded hash of document ids.

  For corpora at most 1,000,000 documents the nine `MARGINAL_FEATURES` equal `marginal_stats`
  exactly (tested). A zero-variance dimension gets skewness, kurtosis, and tail share 0.
- **Rationale**: `marginal_stats` holds the whole matrix (36 GB for MS MARCO in float32); power
  sums stream; quantiles of a million rows have standard error far below the between-dimension
  spread. Tails test the rogue-dimension hypothesis (high variance from rare extreme documents).
- **Alternatives**: streaming quantile sketches (more code, approximate anyway); dropping
  quantiles (they are group A in the plan).

## D4. Group E: redundancy (P0)

- **Decision**: from the streamed covariance `C` (as `corpus_geometry`, which already accumulates
  the Gram matrix) and its correlation `R = C / (σ σᵀ)` (zero-variance rows set to 0):
  - `corr_mean_abs[j]` = mean over `i ≠ j` of `|R_ij|`; `corr_max_abs[j]` = max over `i ≠ j`;
  - `corr_top8[j]` = mean of the 8 largest `R_ij²`, `i ≠ j`;
  - `pc_share_r[j]` = `Σ_{i<r} λ_i V_ji² / C_jj` for r ∈ {8, 32, 128}, eigenpairs of `C`
    descending (`eigh`), the share of dimension j's variance inside the top-r principal
    subspace.

  The full `C` is stored as a diagnostic artifact (constitution: covariance only as diagnostic),
  not fed to the probe.
- **Rationale**: plan group E; a dimension that duplicates others or lives in a dominant
  direction may be dispensable once its partners are kept, independent of its variance.
- **Alternatives**: mutual information (needs density estimation); greedy mRMR scores (a selector,
  not a descriptor; out of scope).

## D5. Calibration queries and pairs (P1: groups B, C, D)

- **Decision**:
  - **Calibration set**: training-split queries, at most 10,000 by a seeded hash of query ids
    (all when fewer). Validation and test queries are never used by the descriptor, so the
    descriptor and the targets (D6) use disjoint queries.
  - **B**: D3's statistics over calibration query embeddings (no tail sample needed).
  - **Retrieval**: each calibration query's top-20 documents by exact full-query inner product
    (`exact_search_streamed`, chunks of 250,000).
  - **Background**: 100 documents per query by a draw seeded by the query id (with replacement
    across queries, without within a query, over the documents in id order so row order does
    not change it; all documents when fewer than 100).
  - **C**: over the products `q_j · d_j`, separately for (query, background document) pairs and
    (query, top-10 document) pairs: mean, std, q05, q50, q95 per dimension (10 columns).
  - **D**: `contrast_K[j]` = mean over queries of `q_j · (mean_{top-K} d_j − mean_{background}
    d_j)` for K ∈ {5, 10, 20}; with fewer than K retrieved documents, the available ones.
- **Rationale**: B, C, D are the plan's P1 groups; pseudo-relevance from the full query uses no
  label. Three K values show whether D is stable (risk R3, circularity).
- **Alternatives**: validation queries for calibration (would overlap the target queries);
  BM25 pseudo-relevance (another system; the full query is the reference the mask approximates).

## D6. Targets

- **Decision**: on the environment's validation-split query embeddings (at most 10,000 by seeded
  hash), without reading validation judgments:
  - **primary**: for each seed of the latest `selectors/retrieval-seeded` (S1), the selection
    frequency `selection_frequency(scores, k)` at each of the nine budgets, divided by k (so each
    budget's profile sums to 1), averaged over the budgets; then averaged
    over seeds 0–2. Per-seed profiles are kept for SC-001's seed condition and the noise ceiling;
  - **secondary**: the per-budget profiles; the S1 mean log-importance per dimension; the S1
    bias `b` (the `projection.bias` of each seed, averaged; query-independent, read from the
    checkpoint without queries); the selection frequency, averaged over budgets, of the
    correction `W q + b` alone of the latest `selectors/budget-seeded` (S4) seeds;
  - **noise ceiling**: per environment, the mean pairwise Spearman `ρ` between the three seeds'
    primary profiles, stepped up to the three-seed average by Spearman–Brown, `3ρ / (1 + 2ρ)`,
    because the target and B1 are three-seed averages.

  Seeded selectors are checked against their grid's selection and payload hashes as
  `compare-selectors` does.
- **Rationale**: S1 has no geometric prior, so the primary target is not built from `q² · Var`;
  frequency is what the mask does. The S4 correction is the part of the best selector the prior
  does not explain.
- **Alternatives**: E1's L2S selectors (KL-trained to the oracle, not the selector later work
  builds on; reported in E1 already); S4's full importance (contains `a_k · log(q² Var)`, so a
  variance descriptor would predict it by construction).

## D7. Direct probe (secondary)

- **Decision**: ridge regression from standardized features to the rank-transformed target
  (average ranks within each environment, scaled to [0, 1]). Standardization means and standard
  deviations are pooled over the fold's training environments (protocol rule 2); a feature with
  zero pooled std is dropped for that fold. Penalty from {0.1, 1, 10, 100, 1000} by D1's inner
  folds. Score: Spearman between predictions and the held-out target. Closed-form solution in
  `float64`; deterministic.
- **Descriptor ladder** (FR-003): (1) `variance` = `std²` only; (2) A without tails;
  (3) (2) + E; (4) (3) + tails; (5) (4) + B + C; (6) (5) + D. Sets 1–4 are P0, 5–6 P1.
- **Rationale**: the lowest-capacity probe (constitution IV); with ≤ 60 features and 2,048 rows,
  ridge is stable and its coefficients are readable. Rank targets match the Spearman score and
  remove the budget-dependent scale.
- **Alternatives**: gradient-boosted trees (more capacity, harder to read, more likely to memorize
  environment identity, risk R2); per-environment standardization of features (uses the held-out
  environment's own statistics; protocol rule 2 fixes normalization on training environments).

## D8. Baselines

- **Decision**: per fold, (B1) dimension identity: the mean of the two training environments'
  primary profiles at each dimension; (B2) the held-out environment's `variance`-method selection
  frequency on the same validation queries, averaged over budgets as in D6 (label-free; the
  scores `q² · Var` use that environment's corpus variance). For the per-seed condition, B1 uses
  the training environments' same-seed profiles.
- **Rationale**: B1 is what a selector shared by every environment predicts, so beating it means
  the descriptor carries environment-specific signal (H2 rather than H1's shared component). B2
  is the strongest label-free selector (P1, S2–S4) and is itself a geometric predictor; beating
  it means the probe adds to `q² · Var`.

## D9. Permutation test and Holm

- **Decision**: per fold, descriptor set, and probe form (shift, direct), 1,000 permutations
  (seed 0): independently permute the descriptor rows within each environment (training and
  held-out), redo the whole fit (penalty choice included), record the held-out Spearman.
  One-sided p = `(1 + #{null ≥ observed}) / 1001`. Holm over the descriptor sets evaluated
  within each fold, separately per probe form; SC-001 reads the shift form.
- **Rationale**: tests the descriptor–dimension pairing, the core of H2; refitting keeps the null
  honest. Within-fold Holm matches SC-001, which counts folds.

## D10. Shift probe (decisive)

- **Decision**: with ranks `r_E` (D7, scaled to [0, 1]) and standardized descriptors `R_E` for
  the fold's two training environments, and `r̄`, `R̄` their means:
  - fit ridge `Δr ≈ ΔR · β` on the training deviations `r_E − r̄`, `R_E − R̄` (2,048 rows; with two
    environments these are the contrast `(r_1 − r_2) / 2` and its negative, so the fit learns how
    the two training environments' descriptor differences track their target differences);
  - predict the held-out environment as `rank(B1) + (R_T − R̄) · β` (`rank(B1)` = ranks of `r̄`),
    scored by Spearman against its target, so it is compared on the same scale as B1 and B2;
  - penalty by generalized cross-validation on the 1,024 contrast rows (the inner two-fold of D1
    is not possible with deviations from two environments); no intercept (deviations are
    centred).

  Permutation (D9) permutes descriptor rows within each environment before the deviations are
  formed and refits, GCV included. With `β = 0` the prediction is B1, so the probe can only beat
  B1 through descriptor differences.
- **Rationale**: H2 claims geometry explains part of the shift, not the whole profile. Starting
  from the shared profile removes what dimension identity explains (risk R2) and asks only
  whether descriptor differences predict where the held-out selector departs from it. This is the
  decisive form (SC-001); the direct probe (D7) is reported as the stronger, secondary claim.
- **Alternatives**: B1 as a feature of the direct probe (mixes scales, and the probe can lean on
  identity alone); predicting the deviation and scoring Spearman of deviations only (not
  comparable with B1 and B2 on one scale; reported as a secondary column).

## D11. Environment-level context

- **Decision**: one descriptive table: per environment, participation ratio and mean energy (from
  the descriptor's covariance), mean of each descriptor column, the S4 learned `a_k` (mean over
  seeds, from the latest `evaluation/S4/all` run's sources and the seeded checkpoints), and E1's
  in-domain and transfer degradation (latest `evaluation/E1/all`). No test, no fit.

## D12. Commands and storage

- **Decision**:
  - `hyperdime describe-space data=<env>` writes `descriptors/<env>/<run_id>/` with
    `descriptor-P0.json` + `.npy` (groups A, E), `descriptor-P1.json` + `.npy` (A, B, C, D, E),
    `covariance.npy` (diagnostic), `inputs.json`, and a manifest.
  - `hyperdime probe-descriptors --experiment E2 [--envs ...] [--seeds 0 1 2]
    [--permutations 1000] [--targets-only]` (D0 with `--targets-only`) reads the latest descriptor per environment and the S1/S4 seeded
    selectors, writes `evaluation/E2/all/<run_id>/` with `targets-<env>.npy` (+ JSON index),
    `probe.jsonl`, `coefficients.json`, `context.json`, `inputs.json`.
  - `hyperdime report E2` writes `reports/E2/`; `hyperdime decide G3 [--adr ...]` writes
    `decisions/G3/<run_id>/gate_decision.json` and drafts ADR-0013 (as E1 → `decide G2`).
- **Rationale**: same pattern as P1 and S1–S4: artifacts only through `ArtifactWriter`, reports
  only from artifacts, decisions drafted, never decided.

## D13. Two stages

- **Decision**: stage 1 delivers D0, D2–D4 (P0, sets 1–4), D6–D12, the report, and the
  decision; stage 2 adds D5 (P1, sets 5–6). `describe-space --protocol P0` writes only
  `descriptor-P0`; `--protocol P1` (the default once stage 2 lands) writes both. The probe runs
  the sets its descriptors allow, Holm covers the sets evaluated, and the report lists them. A G3
  reading from stage 1 is a P0 reading (SC-002).
- **Rationale**: P0 is the strongest zero-shot claim and needs no retrieval; P1 carries the
  circularity risk (R3) and the MS MARCO retrieval pass. Stage 1 alone answers G3 under P0.

## D14. Cost

- **Decision**: `describe-space` on MS MARCO is two streamed passes over 8.8M × 1024 float16
  (as `probe-geometry`, minutes on the 4090), a 1M-row quantile sample (≈ 8 GB float64; done in
  column blocks of 128), and 10,000 queries' exact top-20 (one more streamed pass). The probe is
  CPU: 6 sets × 3 folds × 2 forms × (1 + 1,000 permutations) × 5 penalties of a closed-form
  ridge on ≤ 2,048 × 60, well under SC-003's hour.

## D15. Test-set reuse and the reading

- **Decision**: the report and ADR state that S1–S4 were shaped on these environments and that
  targets come from selectors tuned on them, so E2's reading is exploratory; G3 is confirmed by
  rerunning this spec unchanged on #60's environments. Test judgments are not read by any E2 step.
