# Tasks: F8 Learned Prior Weight (experiment S3)

**Input**: `specs/009-prior-weight/spec.md`, issue #59. **Tests**: required, CPU, offline.

- [X] T001 [P] Tests in tests/unit/test_geometric_selector.py: an untrained weighted `GeometricSelector` equals the `variance` method; its parameters are the linear map's plus `prior_weight`; `prior_weight = 0` gives the linear map alone; save and load of a weighted selector; an S2 (unweighted) checkpoint still loads without `prior_weight`
- [X] T002 Add `learn_weight` to `GeometricSelector` (scalar parameter `prior_weight`, starting at 1) in src/hyperdime/baselines/learning_to_select.py; `load_selector` sets it from the checkpoint
- [X] T003 [P] Tests in tests/unit/training: a `weighted` cell trains with the prior and a learned weight, records `prior_weight` in its history, and lowers it on a misleading prior; `retrieval_cells(inits=("weighted",))` has 16 cells; unweighted histories are unchanged
- [X] T004 Add the `weighted` init to src/hyperdime/training/retrieval_trainer.py and retrieval_selection.py, and the `weighted-grid | weighted-seeded` recipes to src/hyperdime/cli/train_selector.py
- [X] T005 Add `--experiment S3` to src/hyperdime/cli/compare_selectors.py (tracks per experiment: S1 retrieval; S2 + geometric; S3 + weighted), keeping S1 and S2 outputs unchanged
- [X] T006 [P] Tests in tests/unit/reports/test_s3.py with synthetic files: SC-001 against the S1 selector, S2 selector, and `variance`; SC-002 against S1; the prior-weight table; byte-identical regeneration; S2 report tests unchanged
- [X] T007 Parametrize src/hyperdime/reports/s2.py by candidate and references; add src/hyperdime/reports/s3.py, `report S3`, `decide S3` (`GateDecision(gate="S3")`, ADR-0011)
- [X] T008 Integration in tests/integration/test_f5_pipeline.py: weighted grid and seeds, `compare-selectors --experiment S3`, `report S3`, `decide S3`; S1 and S2 reports unchanged
- [ ] T009 Run on DataPolicy-4090; commit `reports/S3/`, ADR-0011, registry, README, architecture
- [ ] T010 Run ruff, format, mypy, pytest, and SCRAP
