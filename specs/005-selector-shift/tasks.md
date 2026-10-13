---

description: "Task list for F4 selector shift and gate G2"
---

# Tasks: F4 Selector Shift and Gate G2

**Input**: `specs/005-selector-shift/` (spec, plan, research, data-model, contracts). **Tests**:
required (constitution VI), CPU, offline, tests first in each story. Run tasks (T024+) execute on
DataPolicy-4090. Rules referenced as D1–D10 are in research.md.

## Phase 1: Foundational — records (blocks every story)

- [X] T001 [P] Tests in tests/unit/contracts/test_f4_records.py: JSON round-trip of `SelectorProperties`, `TransferCell`, and `GateDecision`; an `EvaluationResult` with `protocol="transfer"` validates; `TransferCell` rejects `overlap_at_k` outside [0, 1] and `p_value`/`p_holm` outside [0, 1]; `GateDecision.outcome` must be `"pass"` or `"fail"` and `gate` one of `"G1" … "G5"`; `SelectorProperties.k_values` entries outside [1, 1024] are rejected
- [X] T002 Add `"transfer"` to `Protocol` and the records `SelectorProperties`, `TransferCell`, `GateDecision` (with `Criterion`: `name`, `value: Finite`, `threshold: str`, `met: bool`) exactly as in data-model.md to src/hyperdime/contracts/schemas.py; export them from src/hyperdime/contracts/__init__.py

## Phase 2: User Story 1 — seeded selectors and their fit (P1)

**Goal**: three seeds of F3's `wide-grid` cell per environment, each with its own manifest; the
ranking-fit test of D2. **Independent test**: the toy pipeline trains three seeds, reloads them
identically, and gives identical checkpoints with test judgments deleted.

- [X] T003 [P] [US1] Tests in tests/unit/evaluation/test_fit.py: `fit_gain` by hand — 2 queries, D = 6, k = 2: selector masks {0,1},{2,3}, global {0,5}, oracle {0,1},{2,4} give per-query gains (1 − 0.5, 0.5 − 0) = (0.5, 0.5); seed averaging of selector overlaps is a per-query mean; the rule `fits = mean_gain > 0 and p_holm <= 0.05`
- [X] T004 [US1] Implement `fit_gain(selector_scores: Sequence[Tensor], global_scores: Tensor, oracle_scores: Tensor, k: int) -> Tensor` (per-query gain, D2) in src/hyperdime/evaluation/transfer.py, reusing `overlap_at_k` and `top_k_mask`
- [X] T005 [US1] Add `--recipe seeded` to src/hyperdime/cli/train_selector.py: read the latest `selectors/candidates/<env>` `selection.json`, refuse if its `inputs` differ from the current F2 inputs, take `family_best["wide-grid"]`'s `Cell`, train it once with `run.seed`, write `selectors/seeded/<env>/seed-<n>/<run_id>/` with `selector.safetensors`, `selector.json` (`kind="per_domain"`), and `selection.json` (cell, seed, validation nDCG@10, validation KL, source F3 run, inputs) (D1)
- [X] T006 [US1] Build a module-scoped three-environment toy stack in tests/integration/test_f4_pipeline.py (a second conftest.py would clash with tests/conftest.py's module name): train candidates, then `--recipe seeded` for seeds 0, 1, 2; each artifact's manifest records its seed; a reloaded seed gives identical outputs; deleting test judgments gives identical checkpoints; changed inputs make `--recipe seeded` exit 1

## Phase 3: User Story 2 — the transfer matrix (P1)

**Goal**: `hyperdime transfer-matrix` builds every cell, degradation test, and baseline row with no
training. **Independent test**: planted per-environment subsets make in-domain beat cross-domain;
identical environments give no significant pair; two runs give identical result files.

- [X] T007 [P] [US2] Tests in tests/unit/baselines/test_fixed_subset.py: `fixed_subset` from selection frequencies [0.9, 0.5, 0.1, 0.5] with k = 2 is {0, 1} (the tie between dimensions 1 and 3 goes to the lower index); `rank_importance` makes `top_k_mask` return exactly the subset for every query; frequencies are averaged over seeds before ranking
- [X] T008 [P] [US2] Tests in tests/unit/evaluation/test_transfer.py: `seed_average` of per-query metrics by hand (3 seeds × 2 queries); `degradation_tests` on synthetic per-query values with a known shift gives Δ equal to the shift, p ≤ 0.05, and an interval containing it, and with no shift p > 0.05; Holm applies over exactly the six ordered pairs and never over TREC sets; in-domain cells have `degradation = 0` and `p_value = None`; criterion 1 counts only pairs with Δ > 0 and `p_holm ≤ 0.05`, and holds at 3 of 6
- [X] T009 [US2] Implement `fixed_subset(frequencies: Sequence[Tensor], k: int) -> Tensor` and `rank_importance(subset: Tensor, queries: Tensor) -> Tensor` (constant `1024 − rank`) in src/hyperdime/baselines/methods.py (D6)
- [X] T010 [US2] Implement `seed_average`, `degradation_tests`, `criterion_1`, and `transfer_cells` (building `TransferCell` records: `degradation` = in-domain minus cell nDCG@10; `p_holm` `None` for TREC DL sets; `overlap_at_k` mean over the 9 seed pairs on the test queries) in src/hyperdime/evaluation/transfer.py (D4, D7)
- [X] T011 [US2] Add `hyperdime transfer-matrix` in src/hyperdime/cli/transfer_matrix.py and register it in src/hyperdime/cli/__main__.py: load seeded selectors read-only (refuse on a missing seed or a payload SHA-256 that differs from its record), F3's `selectors/global/all`, and F3's official τ for global importance; per test environment and query set one `evaluate_methods` pass at k = 307 with the D3 methods; write `results-<env>-<set>.jsonl` (`protocol` "transfer" when train ≠ test), `cells.jsonl`, `fixed_subsets.json`, `fit.json` (D2, Holm over environments), `inputs.json` under `evaluation/E1/all/<run_id>/`
- [X] T012 [US2] Extend tests/integration/test_f4_pipeline.py through `transfer-matrix`: 9 test cells; checkpoints' bytes unchanged after the run; a second run gives identical result files apart from manifests

## Phase 4: User Story 3 — behavior against the seed noise floor (P1)

**Goal**: battery measures for all 36 selector pairs and criterion 2's exact permutation test.
**Independent test**: three seeds of one synthetic environment plus a planted different one give
p ≤ 0.05; two copies of one environment do not.

- [X] T013 [P] [US3] Tests in tests/unit/evaluation/test_group_permutation.py: `group_permutation_test` enumerates exactly 1,680 assignments for 3 groups of 3; on an overlap matrix where within pairs are 1 and cross pairs 0, T = 1 and p = 6/1,680; on a constant matrix p = 1; the result does not depend on the order selectors are listed
- [X] T014 [P] [US3] Tests in tests/unit/evaluation/test_battery.py: `sample_battery` takes n = the smallest environment's count, all ids of that environment, is seeded and repeatable, sorts ids before sampling, and returns a SHA-256 of the sorted ids; `kl_divergence` by hand on [0.5, 0.5] vs [0.25, 0.75]; `pair_measures` returns overlap, JS, both KLs, and both Spearman values for one pair
- [X] T015 [US3] Implement `kl_divergence`, `pair_measures`, and `group_permutation_test(overlaps: Tensor, groups: Sequence[int]) -> tuple[float, float, int]` (T, p, assignments) in src/hyperdime/evaluation/selector_shift.py (D7)
- [X] T016 [US3] Implement `sample_battery` in src/hyperdime/evaluation/transfer.py (D5) and, in src/hyperdime/cli/transfer_matrix.py, the behavior step: run every seeded selector on the battery; write `properties-<env>-s<n>.json` (`SelectorProperties`, `mean_importance` "`(1024,)`, finite, sums to 1", `selection_frequency` "`(len(k_values), 1024)`, each row in [0, 1], sums to k") with `.npy` payloads, `battery.json`, and `behavior.json` (36 pairs, T, p, assignments, per-environment-pair overlaps); read embeddings only
- [X] T017 [US3] Planted-effect test in tests/unit/evaluation/test_group_permutation.py (synthetic selectors with planted environment preferences give criterion 2 true; one environment under three labels does not), and integration in tests/integration/test_f4_pipeline.py: the behavior step, fit, and fixed subsets read no test judgment (judgments moved to other documents, identical `behavior.json`, `fit.json`, `fixed_subsets.json`)

## Phase 5: User Story 4 — E1 report and the G2 draft (P2)

**Goal**: `reports/E1/` and a proposed G2 ADR from result files only. **Independent test**:
synthetic result files regenerate byte-identically and give the expected criteria.

- [ ] T018 [P] [US4] Tests in tests/unit/reports/test_e1.py: tables from synthetic result files (`transfer.csv`, `trec.csv`, `seeds.csv`, `behavior.csv`, `noise_floor.csv`, `fit.csv`, `fixed_subset.csv`, `payoff.csv`, `g2.csv`); full-1024 and prefix rows in every comparison table; protocol and scoring stated; MS MARCO rows carry the D9 note and NFCorpus rows the ADR-0006 caveat; the wide-grid departure is stated with both validation scores; identical bytes on regeneration; refuses with fewer than three seeds per environment
- [ ] T019 [P] [US4] Tests in tests/unit/reports/test_g2.py: `GateDecision` outcome "pass" only when both criteria are met; context entries (fit, fixed subset, global-selector payoff, notes) never change the outcome; the proposed ADR has status "proposed", names every evidence run, and an existing ADR whose status is not "proposed" is not overwritten
- [ ] T020 [US4] Implement src/hyperdime/reports/e1.py and `hyperdime report E1` in src/hyperdime/cli/report.py (payoff over the global selector: paired test, Holm over environments; fixed-subset comparisons: Holm over nine; D8)
- [ ] T021 [US4] Implement src/hyperdime/reports/g2.py and `hyperdime decide G2 [--adr]` in src/hyperdime/cli/decide.py, registered in src/hyperdime/cli/__main__.py; writes `decisions/G2/<run_id>/gate_decision.json` with a manifest and the proposed ADR
- [ ] T022 [US4] Extend tests/integration/test_f4_pipeline.py through `report E1` (generated twice, identical) and `decide G2`

## Phase 6: Polish

- [ ] T023 [P] Update docs/architecture.md (components: seeded selectors, transfer matrix, behavior, E1 report, decide) and README.md (F4 commands), and docs/experiment_registry.md (E1 `running` when runs start)
- [ ] T024 Run `ruff check . && ruff format --check . && mypy && pytest` and `PYTHONPATH=tools python -m scrap tests`; act on `HIGH` findings for new tests or explain them in the pull request

## Phase 7: Runs on DataPolicy-4090

- [ ] T025 Sync the branch to DataPolicy-4090; train `--recipe seeded` for seeds 0, 1, 2 in scifact, nfcorpus, msmarco; record wall times and whether seed 0 matches F3's `wide-grid` checkpoint
- [ ] T026 Run `hyperdime transfer-matrix`; record wall time against SC-006
- [ ] T027 Generate reports/E1 twice on the host (identical bytes), copy it back, and commit it on `exp/E1-selector-shift`
- [ ] T028 Run `hyperdime decide G2 --adr docs/decisions/ADR-0007-g2-selector-shift.md` and open it as proposed for the researcher's decision; on acceptance, tag the merge `g2-selector-shift` (issue #20)

## Dependencies

Phase 1 → US1 → US2 → US3 → US4 → Polish → Runs. US2's transfer cells need US1's seeded
selectors; US3 needs US1 only and can follow US1 in parallel with US2 (different functions, shared
files src/hyperdime/evaluation/transfer.py and src/hyperdime/cli/transfer_matrix.py, so commit in
order); US4 needs US2 and US3. Tests first within each story.

## Parallel examples

- Phase 1: T001 alongside reading data-model.md; T002 after.
- US2: T007 and T008 together; then T009 and T010 (different files) together; then T011.
- US3: T013 and T014 together; then T015.
- US4: T018 and T019 together.

## Implementation strategy

MVP is Phase 1 + US1 + US2: seeded selectors and the transfer matrix give G2 criterion 1. US3 adds
criterion 2; US4 turns both into the report and the proposed ADR. No run starts before T024 passes
on the branch, so every number comes from one committed version.
