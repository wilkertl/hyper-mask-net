---
description: "Task list for F10 descriptor probe (experiment E2, gate G3)"
---

# Tasks: F10 Descriptor Probe (experiment E2, gate G3)

**Input**: `specs/011-descriptor-probe/` (spec, plan, research D0–D15, data model,
contracts/cli.md, quickstart). **Tests**: required (constitution VI), CPU, offline,
hand-computable fixtures.

Format: `- [ ] T### [P?] [US?] description with file path`. `[P]` = different files, no pending
dependency. Stories: US0 shift check, US1 space descriptor, US2 selector targets, US3 experiment
E2. Two stages (research D13): **stage 1** = US2 → US0 (run the check) → US1 P0 → US3 (run, G3
under P0); **stage 2** = US1 P1 groups → rerun. Every E2 run is preceded by a commit of the code
it runs (pre-registration: the spec's criteria are already committed in `905ee1f`).

# Stage 1

## Phase 1: Setup

- [X] T001 Confirm the branch is `agent/F10-descriptor-probe` stacked on `agent/F9-budget-prior`, and record the files to change (plan "Source code") in the pull request description; regenerate `reports/E0 E1 P1 S1 S2 S3 S4` wherever artifacts exist and record `git diff --exit-code` on them as the FR-008 baseline (on this machine without artifacts, note that the check runs on DataPolicy-4090 in T022 and T037) *Done: branch `agent/F10-descriptor-probe` at F9's tip plus the F10 spec commits; no E0–S4 artifacts on this machine, so FR-008 is checked on DataPolicy-4090 in T033 and T037.*

## Phase 2: Foundational (blocks all stories)

- [X] T002 [P] Tests in tests/unit/evaluation/test_descriptor_probe.py for the rank helpers (research D7): `average_ranks` on `[3, 1, 1, 2]` gives `[1, 1/6, 1/6, 2/3]` (0-based average ranks `[3, 0.5, 0.5, 2]` divided by n − 1; ties share their average rank); `spearman_brown(0.5, 3) == 0.75` (`3ρ / (1 + 2ρ)`); Spearman of predictions reuses `evaluation.selector_shift.spearman` (tied average ranks) and is 1 for a monotone transform and −1 for its reverse
- [X] T003 Implement `average_ranks`, `spearman_brown`, and `LADDER` (data model "Descriptor ladder": six sets with names `variance`, `A`, `A+E`, `A+E+tails`, `+B+C`, `+D`, columns by feature name, protocol `P0` for 1–4 and `P1` for 5–6) in src/hyperdime/evaluation/descriptor_probe.py (depends on T002) *Done; `spearman` added (numpy, tied average ranks) so the probe works in float64 without torch.*

**Checkpoint**: rank utilities and the ladder definition exist; stories can start.

## Phase 3: User Story 2 – Per-dimension selector targets (P1)

**Goal**: target profiles from the S1 and S4 seeded selectors on validation queries, no labels.
**Independent test**: on a synthetic selector, frequencies match a hand count, each budget's
profile sums to 1 after dividing by k, no judgments are read, averaging seeds averages profiles.

- [X] T004 [P] [US2] Tests in tests/unit/evaluation/test_targets.py, D = 4, budgets {1, 2}, 3 queries: `target_profiles(selectors, queries, budgets)` returns `budget-<k>` profiles equal to hand-counted `selection_frequency / k` (each sums to 1), `mean` = their mean over budgets, `log_importance` = mean log-importance over queries, `bias` = the selector's `projection.bias`; for an S4 `BudgetGeometricSelector` the `s4_correction` profile uses `W q + b` alone (prior ignored); the `variance/mean` profile of `variance_importance` matches a hand count; seed averaging equals the mean of per-seed profiles; validation queries are chosen by a seeded hash capped at 10,000 and identical across calls
- [X] T005 [US2] Implement `target_profiles` and the validation-query sampler in src/hyperdime/evaluation/targets.py (research D6; keys per data model "Target profiles": `s1/seed-<n>/budget-<k>`, `s1/seed-<n>/mean`, `s1/mean`, `s1/seed-<n>/log_importance`, `s1/bias`, `s1/seed-<n>/bias`, `s4_correction/seed-<n>/mean`, `s4_correction/mean`, `variance/mean`) (depends on T004)
- [X] T006 [US2] Add a qrels-free validation-query loader for targets in src/hyperdime/evaluation/targets.py: query ids of the `validation` split from `queries.jsonl` and their embeddings, plus the corpus variance from document embeddings; reuse the S1 (`retrieval`) and S4 (`budget`) seeded-selector loading of `RetrievalSelectors` in src/hyperdime/cli/compare_selectors.py (grid-selection and payload-hash checks) without loading `SeededEnvironment` (which reads test qrels); test in tests/unit/evaluation/test_targets.py that deleting every qrels file of a synthetic environment leaves the profiles unchanged (depends on T005) *Done: `seeded_selectors` checks grid selection and payload hashes; queries come from `space.inputs.load_unlabeled` (T019–T020, done here as a shared prerequisite); the corpus variance for B2 is the S4 selectors' `variance` buffer (documents only). The qrels-deleted check is in tests/integration/test_f10_pipeline.py.*

**Checkpoint**: targets can be computed for any environment with S1 and S4 seeds.

## Phase 4: User Story 0 – Is there a shift to explain? (P1)

**Goal**: B1 versus the noise ceiling per held-out environment, from targets alone (D0).
**Independent test**: identical profiles plus seed noise → no room; an environment-specific
component → the gap; no labels, no descriptor.

- [X] T007 [P] [US0] Tests in tests/unit/evaluation/test_descriptor_probe.py for `shift_check(profiles)` on three synthetic environments × three seeds, D = 64: when every environment is one shared profile plus independent seed noise, B1's Spearman is within 0.05 of the stepped-up ceiling in all three folds and `no_shift` is true; when one environment adds its own component, its gap exceeds 0.05; when two do, `no_shift` is false (threshold "gap < 0.05 in at least 2 of 3 environments", fixed); B1 for a held-out environment is the mean of the other two's seed-averaged profiles *Done: a single odd environment also enters the other folds' B1, so it opens room in every fold; the test checks its gap is the largest.*
- [X] T008 [US0] Implement `shift_check` in src/hyperdime/evaluation/descriptor_probe.py (research D0, D6 ceiling `3ρ / (1 + 2ρ)` of the mean pairwise seed Spearman) (depends on T003, T007)
- [X] T009 [P] [US0] Tests in tests/integration/test_f10_pipeline.py on a synthetic three-environment artifact tree with S1 and S4 seeded selectors: `probe-descriptors --experiment E2 --targets-only` writes `evaluation/E2/all/<run_id>/` with `targets-<env>.npy` + `.json` (query count, selector run ids, payload hashes, ceilings), `check.json`, `inputs.json`, and a manifest; it reads no descriptor; it refuses (exit 1) with fewer than three environments or a missing seed *Done in tests/integration/test_f10_pipeline.py on the S4 toy stack with every qrels file deleted.*
- [X] T010 [US0] Implement src/hyperdime/cli/probe_descriptors.py with `--envs`, `--seeds 0 1 2`, `--permutations 1000`, `--targets-only` (contracts/cli.md), register it in src/hyperdime/cli/__main__.py (depends on T006, T008, T009)
- [X] T011 [P] [US0] Tests in tests/unit/reports/test_e2.py: `report E2` on a `--targets-only` run writes `reports/E2/README.md` and `tables/check.csv` (per held-out environment: B1 Spearman, ceiling, gap, and the `no_shift` reading) and states that descriptors were not evaluated; identical bytes on rerun
- [X] T012 [US0] Implement the check-only path of src/hyperdime/reports/e2.py and add `E2` to src/hyperdime/cli/report.py (depends on T010, T011)
- [X] T013 [US0] Run `ruff check . && ruff format --check . && mypy && pytest` and `PYTHONPATH=tools python -m scrap tests`; commit (`feat(evaluation): add E2 targets and the shift check`) before the run
- [x] T014 [US0] Run on DataPolicy-4090: `hyperdime probe-descriptors --experiment E2 --targets-only` then `hyperdime report E2`; record commit, run id, and time here; copy `reports/E2/` locally. **Stop rule**: if `no_shift` is true, skip to T030 (`decide G3` reads "not supported: no shift to explain") and do not start US1
  - Run 2026-10-28 on DataPolicy-4090 (ceia40901): commit `e7627ad`, probe-descriptors `20261029T002359Z-c352e11b`, 8 s; `report E2` 1 s. `no_shift` is false (room in all three folds; see `reports/E2/README.md`), so stage 1 continues with US1.

**Checkpoint**: the shift check is read; stage 1 continues only if there is room.

## Phase 5: User Story 1 – A per-dimension space descriptor, P0 (P1)

**Goal**: `describe-space --protocol P0`: groups A (with tails) and E, no queries, no qrels.
**Independent test**: hand-computed statistics, order invariance, dimension equivariance,
streamed = one pass, unchanged with qrels deleted.

- [ ] T015 [P] [US1] Tests in tests/unit/test_space_streaming.py on a 5 × 3 hand-built matrix: `document_moments` (streamed in chunks of 2) gives mean, population std, mean_abs, RMS, skewness `m3 / m2^{3/2}`, excess kurtosis `m4 / m2² − 3` equal to hand values; `tail_share` counts `|x − mean| > 3 · std` (a fixture with one 3σ outlier among 20 rows gives 0.05); chunked equals one pass; row permutation leaves results unchanged; a constant column gives std 0 and skew = kurt = tail3 = 0 without NaN; quantiles on ≤ 1,000,000 rows use all rows and the nine `MARGINAL_FEATURES` equal `space.marginal.marginal_stats` exactly; above the cap, rows are a seeded hash sample of document ids, identical across calls
- [ ] T016 [US1] Implement `document_moments`, `tail_share`, and the quantile sample in src/hyperdime/space/streaming.py (research D3; float64 power sums, quantiles in column blocks of 128) (depends on T015)
- [ ] T017 [P] [US1] Tests in tests/unit/test_space_redundancy.py on a 3 × 3 covariance with known correlations: `corr_mean_abs`, `corr_max_abs` (over `i ≠ j`), `corr_top8` (mean of the min(8, D−1) largest squared correlations), `pc_share_r` = `Σ_{i<r} λ_i V_ji² / C_jj` with r capped at D (a diagonal covariance gives share 1 for the top dimensions and 0 for the rest); a zero-variance dimension gets correlations 0 and finite PC share 0; permuting dimensions permutes the rows
- [ ] T018 [US1] Implement `redundancy_features(covariance)` in src/hyperdime/space/redundancy.py (research D4) and reuse the streamed Gram of `space.geometry.corpus_geometry` to produce the covariance (depends on T017)
- [X] T019 [P] [US1] Tests in tests/unit/test_space_inputs.py: `load_unlabeled(root, environment)` returns document embeddings, query embeddings, and query ids per split from the dataset manifest, `queries.jsonl`, and the embeddings; it gives identical output with every `qrels-*.jsonl` deleted and with each replaced by a corrupt file (proving none is opened); it refuses (`ContractError`) embeddings whose manifest dataset hash differs *Done early (needed by targets).*
- [X] T020 [US1] Implement `load_unlabeled` and `UnlabeledInputs` in src/hyperdime/space/inputs.py (research D2) (depends on T019)
- [ ] T021 [P] [US1] Tests in tests/unit/test_space_descriptor.py: `FEATURES` lists the data model's P0 columns in order (`doc_mean, doc_std, doc_mean_abs, doc_rms, doc_q05, doc_q25, doc_q50, doc_q75, doc_q95, doc_skew, doc_kurt, doc_tail3, red_corr_mean_abs, red_corr_max_abs, red_corr_top8, red_pc_share_8, red_pc_share_32, red_pc_share_128`, 18 columns) with groups; `build_descriptors(..., protocol="P0")` returns a `[D, 18]` float64 array and a `SpaceDescriptor` with `protocol="P0"`, `groups=["A", "E"]`, `num_queries=0` (the validator rejects queries under P0); document row order does not change it; permuting dimensions permutes rows
- [ ] T022 [US1] Implement src/hyperdime/space/descriptor.py (P0 path) and src/hyperdime/cli/describe_space.py (`data=<env> --protocol P0 --chunk-size N`, contracts/cli.md: `descriptors/<env>/<run_id>/` with `descriptor-P0.json` + `.npy`, `covariance.npy`, `inputs.json`, manifest; refuses mismatched embeddings with exit 1); register in src/hyperdime/cli/__main__.py; CLI test in tests/integration/test_f10_pipeline.py on a synthetic environment with qrels deleted (depends on T016, T018, T020, T021)

**Checkpoint**: P0 descriptors build for any environment without labels.

## Phase 6: User Story 3 – Experiment E2 (P1)

**Goal**: shift and direct probes over the ladder, baselines, permutation, report, decision.
**Independent test**: a planted linear signal is found and beats both baselines; an unrelated
target is not; the held-out environment never enters fitting.

- [ ] T023 [P] [US3] Tests in tests/unit/evaluation/test_descriptor_probe.py for the ridge (research D7, D10): `ridge_fit` on a 3-row, 1-feature example gives the hand-computed `β = Σxy / (Σx² + λ)` (no intercept on centred data); `gcv_penalty` picks the hand-computed minimum on a 4-row example from {0.1, 1, 10, 100, 1000}; standardization uses the training rows only and drops a zero-std feature; the inner two-fold choice (direct probe) fits on one training environment and scores on the other
- [ ] T024 [P] [US3] Tests in tests/unit/evaluation/test_descriptor_probe.py for `shift_probe` and `direct_probe` on three synthetic environments (D = 256) with a shared profile plus an environment term that is linear in one descriptor column plus noise: the shift probe's held-out Spearman exceeds B1 and B2 in every fold; with `β` forced to 0 the shift prediction ranks exactly as B1; with a target unrelated to the descriptor the shift probe does not beat B1 by more than noise; changing the held-out environment's target leaves the fitted coefficients unchanged (no leakage); `permutation_p` with 20 permutations equals `(1 + #{null ≥ observed}) / 21` on a fixed seed and refits the penalty inside each permutation; Holm is applied over the sets evaluated within a fold and form
- [ ] T025 [US3] Implement `ridge_fit`, `ridge_predict`, `gcv_penalty`, `shift_probe`, `direct_probe`, and `permutation_p` in src/hyperdime/evaluation/descriptor_probe.py (depends on T003, T023, T024)
- [ ] T026 [US3] Extend src/hyperdime/cli/probe_descriptors.py beyond `--targets-only`: read the latest descriptor per environment (refuse a missing one or one not made from the current embeddings), run the ladder sets the descriptors allow, both forms, baselines B1 and B2, 1,000 permutations for the primary target's seed-averaged rows, per-seed rows for SC-001's seed condition, secondary targets (per budget, `s1/bias`, `s4_correction`) without permutations; write `probe.jsonl` (data model "Probe result" fields), `coefficients.json`, and `context.json` (research D11: participation ratio and mean energy from `covariance.npy`, descriptor column means, S4 `a_k` mean over seeds from the seeded checkpoints, E1 degradation from the latest `evaluation/E1/all`); CLI test in tests/integration/test_f10_pipeline.py on the synthetic tree (depends on T022, T025)
- [ ] T027 [P] [US3] Tests in tests/unit/reports/test_e2.py for the full report: `criteria.csv` (per fold: best shift-probe set, its Spearman, B1, B2, Holm p, per-seed check, SC-001 met), SC-001 counts folds where a set passes "at least 2 of the 3", SC-002 names the protocol of the best passing set (P0 if any of sets 1–4 passes); `ladder.csv`, `seeds.csv`, `secondary.csv`, `ceiling.csv`, `coefficients.csv`, `context.csv`; the README lists the sets evaluated, states that E2 scores no retrieval (so no full or prefix rows), and that these environments shaped S1–S4; identical bytes on rerun
- [ ] T028 [US3] Implement the full report in src/hyperdime/reports/e2.py (depends on T026, T027)
- [ ] T029 [P] [US3] Tests in tests/unit/reports/test_e2.py for the G3 decision: `GateDecision(gate="G3")` with criteria SC-001 and SC-002 and evidence keys `check`, `ladder`, `ceiling`, `direct`, `test_set_reuse`; with `no_shift`, SC-001 reads "not supported: no shift to explain"; the drafted ADR-0013 has status "proposed" and an existing non-proposed ADR is never overwritten
- [ ] T030 [US3] Implement the G3 draft in src/hyperdime/reports/e2.py and add `G3` to src/hyperdime/cli/decide.py (`--adr docs/decisions/ADR-0013-g3-space-signal.md`) (depends on T028 or, after the stop rule in T014, on T012; T029)
- [ ] T031 [US3] Integration test tests/integration/test_f10_pipeline.py: synthetic three-environment tree with embeddings, S1 and S4 seeded selectors, and qrels files deleted → `probe-descriptors --targets-only` → `describe-space --protocol P0` per environment → `probe-descriptors` → `report E2` → `decide G3`; the planted signal passes SC-001 under P0; E0, E1, P1, S1–S4 report generators are untouched (depends on T030)
- [ ] T032 [US3] Run `ruff check . && ruff format --check . && mypy && pytest` and `PYTHONPATH=tools python -m scrap tests` (act on `HIGH` recommendations for changed test files or state why not); commit (`feat(space): add the P0 space descriptor and the E2 probe`) before any descriptor or probe run
- [ ] T033 [US3] Run on DataPolicy-4090 per quickstart: `describe-space data=<env> --protocol P0 --chunk-size 250000` for scifact, nfcorpus, msmarco, then `probe-descriptors --experiment E2`, `report E2`, `decide G3 --adr docs/decisions/ADR-0013-g3-space-signal.md`; regenerate E0, E1, P1, S1–S4 reports and check `git diff --exit-code` on them (FR-008); record commit, run ids, and times against SC-003 here
- [ ] T034 [US3] Copy `reports/E2/` and ADR-0013 locally; update docs/architecture.md (`hyperdime.space` descriptor modules, `evaluation.descriptor_probe`, `evaluation.targets`, the two commands), docs/experiment_registry.md (E2 row: stage-1 P0 reading, ADR-0013 proposed, exploratory), and README.md with generated numbers only; run `hyperdime report dimensions` if it changes; commit

**Checkpoint (end of stage 1)**: G3 has a P0 reading (or a "no shift" reading from T014).

# Stage 2

## Phase 7: User Story 1 – The P1 groups (P2)

**Goal**: groups B, C, D from unlabeled training queries; ladder sets 5–6.
**Independent test**: hand-computed products and contrast; query-order invariance; no
validation or test query and no qrels read.

- [ ] T035 [P] [US1] Tests in tests/unit/test_space_interaction.py on D = 3, 2 queries, 6 documents: the calibration set is training-split queries only, at most 10,000 by seeded hash (validation and test ids never appear); group B equals group A's 12 columns on query embeddings; with hand-picked top-K and background rows, `pair_rand_*` and `pair_top_*` (mean, std, q05, q50, q95 of `q_j · d_j` over background and top-10 pairs) and `contrast_k5/k10/k20` (`mean over queries of q_j · (mean top-K − mean background)`) equal hand values; a query with fewer than K retrieved documents uses those it has; background draws are without replacement within a query, seeded, identical across calls; query row order does not change the result
- [ ] T036 [US1] Implement `interaction_features(queries, documents, top, background)` and the calibration sampler in src/hyperdime/space/interaction.py (research D5; top-20 by `retrieval.exact.exact_search_streamed`, 100 background documents per query); extend src/hyperdime/space/descriptor.py and src/hyperdime/cli/describe_space.py with `--protocol P1` writing `descriptor-P1.json` + `.npy` (43 columns: 18 P0 + 12 B + 10 C + 3 D; groups `["A", "B", "C", "D", "E"]`, `num_queries` = calibration size) and make P1 the default; extend tests/unit/test_space_descriptor.py and tests/integration/test_f10_pipeline.py (qrels deleted, P0 columns of the P1 descriptor equal the P0 descriptor) (depends on T035)
- [ ] T037 Run checks and commit (`feat(space): add the P1 descriptor groups B, C, D`) before the run; on DataPolicy-4090 run `describe-space --protocol P1` for the three environments, `probe-descriptors --experiment E2`, `report E2`, `decide G3 --adr ...` (the ADR is redrafted only while "proposed"); check FR-008 and SC-003; record commit, run ids, and times here
- [ ] T038 Copy `reports/E2/` and ADR-0013; update docs/experiment_registry.md, docs/architecture.md, and README.md with generated numbers only; commit

## Phase 8: Polish

- [ ] T039 [P] Run `/speckit-analyze` consistency check across spec, plan, and tasks; fix drift in specs/011-descriptor-probe/
- [ ] T040 Final `ruff check . && ruff format --check . && mypy && pytest`, SCRAP, `git diff --stat`, and the pull request description (files changed, decisions D0–D15, risks, stage reached)

## Dependencies

- Setup → Foundational (T002–T003) → US2 (T004–T006) → US0 (T007–T014, run) → US1 P0
  (T015–T022) → US3 (T023–T034, run) → stage 2 US1 P1 (T035–T038) → Polish.
- T014's stop rule: with `no_shift`, go T029–T030 → T034 (decision and docs only) and skip
  T015–T028, T031–T033, and stage 2.
- US3's probe code (T023–T025) needs only T003 and can be written in parallel with US1.
- Each run (T014, T033, T037) follows a commit of the code it runs (T013, T032, T037).

## Parallel examples

- Foundational: T002 alone, then T003.
- US2: T004 → T005 → T006. US0: T007 ∥ T009 ∥ T011, then T008 → T010 → T012.
- US1 (P0): T015 ∥ T017 ∥ T019 ∥ T021, then T016, T018, T020 → T022.
- US3: T023 ∥ T024 ∥ T027 ∥ T029 (and in parallel with US1), then T025 → T026 → T028 → T030 → T031.
- Stage 2: T035 → T036.

## Implementation strategy

1. MVP = US2 + US0: targets and the shift check, run on existing checkpoints in about an hour,
   telling whether E2 can pass at all.
2. Stage 1 then adds the P0 descriptor and the probes: a complete G3 reading under P0, the
   strongest zero-shot protocol.
3. Stage 2 adds the P1 groups only if stage 1 leaves the question open or the researcher wants
   the P1 reading; the decision is redrafted while ADR-0013 is "proposed".
