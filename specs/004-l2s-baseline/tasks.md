---

description: "Task list for F3 Learning-to-Select baseline and gate G1"
---

# Tasks: F3 Learning-to-Select Baseline and Gate G1

**Input**: `specs/004-l2s-baseline/` (spec, plan, research). **Tests**: required, CPU, offline,
tests first in each story. Run tasks (T020+) execute on DataPolicy-4090.

## Phase 1: Setup

- [X] T001 Install the `science` extra in CI (.github/workflows/ci.yml) and README's test install, for `ir-measures`

## Phase 2: User Story 1 — trustworthy exact evaluation (P1)

- [X] T002 [P] [US1] Tests in tests/unit/evaluation/test_metrics_ap.py: AP by hand on a 5-document ranking; `relevance_threshold=2` makes grade-1 documents non-relevant for AP, MRR, and Recall but not for nDCG; defaults reproduce the previous values; agreement with `ir-measures` within 1e-6 on a random fixture (nDCG@10, RR@10, RR@100, R@100, AP)
- [X] T003 [P] [US1] Tests in tests/unit/evaluation/test_runs.py: `masked_rankings` with k = 1024 equals unmasked rankings; stacked evaluation of several (method, k) pairs equals evaluating each alone; queries without a relevant document are excluded and counted
- [X] T004 [US1] Add AP and `relevance_threshold` to `query_metrics` in src/hyperdime/evaluation/metrics.py
- [X] T005 [US1] Implement `masked_rankings` and `evaluate_methods` in src/hyperdime/evaluation/runs.py (one `exact_search_streamed` pass over all stacked masked queries; per-query metrics; `EvaluationResult` records)

## Phase 3: User Story 2 — a selector that learns its target (P1)

- [X] T006 [P] [US2] Tests in tests/unit/training/test_targets_and_trainer.py: `build_targets` rows sum to 1, are finite, and match `aggregate_targets` by hand on a tiny case; trainer defaults reproduce ADR-0002 training bit-for-bit; `device` works on CPU; anchor regularization at weight 0 equals no anchor; patience stops early; a saved selector reloads with identical outputs
- [X] T007 [US2] Implement `build_targets` in src/hyperdime/oracle/targets.py (positives and pools from a `DatasetBundle`, vectors gathered from the memory-mapped matrix)
- [X] T008 [US2] Extend `TrainConfig`/`train_selector` in src/hyperdime/training/selector_trainer.py with `device`, `anchor`/`l2_to_anchor`, `patience`, and `schedule`
- [X] T009 [US2] Implement recipes, grids, and validation scoring in src/hyperdime/training/selection.py, and safetensors save/load of selectors with `SelectorCheckpoint`

## Phase 4: User Story 3 — baselines and candidates (P2)

- [X] T010 [P] [US3] Tests in tests/unit/baselines/test_methods.py: each method gives exactly k per query; random is reproducible by seed; global importance is the mean training target; the oracle uses only the labels it is given
- [X] T011 [US3] Implement the methods of research D3 in src/hyperdime/baselines/methods.py
- [X] T012 [US3] Add `hyperdime train-selector --recipe official|candidates` and `hyperdime train-global` in src/hyperdime/cli/train_selector.py and src/hyperdime/cli/train_global.py; integration test in tests/integration/test_f3_pipeline.py on the toy environment (fake F2 artifacts), including the leakage check (test judgments deleted, identical selectors) and `selected` being the validation-best cell

## Phase 5: User Story 4 — E0 report and G1 (P2)

- [X] T013 [US4] Add `hyperdime evaluate` in src/hyperdime/cli/evaluate.py (every method at ratios 0.1–1.0 on test and each evaluation set; k = 1024 equality recorded in `checks.json`)
- [X] T014 [P] [US4] Tests in tests/unit/reports/test_e0.py: tables from synthetic result files, Holm-adjusted tests, G1 criteria values, identical bytes on regeneration
- [X] T015 [US4] Implement src/hyperdime/reports/e0.py and `hyperdime report E0` in src/hyperdime/cli/report.py
- [X] T016 [US4] Extend tests/integration/test_f3_pipeline.py through evaluate and report

## Phase 6: Polish

- [ ] T017 [P] Update docs/architecture.md and README.md with the F3 commands
- [ ] T018 Run ruff, format, mypy, pytest, and SCRAP; act on `HIGH` findings for new tests or explain them

## Phase 7: Runs on DataPolicy-4090

- [ ] T019 Sync the branch to DataPolicy-4090; train official selectors for scifact, nfcorpus, msmarco; train the global selector; train candidates
- [ ] T020 Evaluate every environment; record MS MARCO wall time (replace SC-007's ceiling)
- [ ] T021 Generate reports/E0 twice on the host (identical bytes), copy it back, and commit it on `exp/E0-baseline-reproduction`
- [ ] T022 Draft docs/decisions/ADR-0006-g1-baseline.md (status proposed) from the report for the researcher's decision

## Dependencies

Setup → US1 → US2 → US3 → US4 → Polish → Runs. Tests first within each story.
