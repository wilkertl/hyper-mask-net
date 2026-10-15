# Tasks: F7 Geometric Prior (experiment S2)

**Input**: `specs/008-geometric-prior/spec.md`. **Tests**: required (constitution VI), CPU, offline.

- [X] T001 [P] Tests in tests/unit/test_geometric_selector.py: an untrained `GeometricSelector` gives the same top-k as `variance_importance` at every k; its only parameters are the linear map's; save and load reproduce its scores; `load_selector` returns a `LinearSelector` for an L2S checkpoint and a `GeometricSelector` for a geometric one
- [X] T002 Implement `GeometricSelector(LinearSelector)` (buffer `variance`, zero-initialized projection, score `log(q² · variance + 1e-12) + projection(q)`, log-softmax) in src/hyperdime/baselines/learning_to_select.py; make `load_selector` in src/hyperdime/training/selection.py build it when the checkpoint has a `variance` tensor
- [X] T003 [P] Tests in tests/unit/training/test_retrieval_trainer.py and test_retrieval_selection.py: `train_retrieval_selector(..., prior=v)` starts from the prior (its step-0 validation equals `variance`'s) and recovers the planted dimensions; `retrieval_cells(inits=("prior",))` has 16 cells; a prior cell without a prior is refused
- [X] T004 Add `prior` to `train_retrieval_selector` and `train_retrieval_cells`, and `inits` to `retrieval_cells`, in src/hyperdime/training/retrieval_trainer.py and retrieval_selection.py
- [X] T005 Add `--recipe geometric-grid | geometric-seeded` to src/hyperdime/cli/train_selector.py (variance from `corpus_geometry` of the documents; selectors under `selectors/geometric-grid|geometric-seeded`)
- [X] T006 Add `--experiment S2` to src/hyperdime/cli/compare_selectors.py: also the geometric selectors (`geometric:<env>:s<n>`), their family bests, and `variance`; overlap and fit for the geometric selector; written under `evaluation/S2/all/`
- [X] T007 [P] Tests in tests/unit/reports/test_s2.py with synthetic files: SC-001 against the S1 selector, SC-002 against `variance`, Holm per comparison, the step-0 note, byte-identical regeneration; the S1 report tests still pass
- [X] T008 Implement src/hyperdime/reports/s2.py (reusing s1's helpers, parametrized by candidate) and `report S2`, `decide S2` (`GateDecision(gate="S2")`, ADR-0010)
- [X] T009 Integration in tests/integration/test_f5_pipeline.py: geometric grid and seeds, `compare-selectors --experiment S2`, `report S2`, `decide S2` on the toy stack
- [ ] T010 Run on DataPolicy-4090 (CPU threads capped per process); commit `reports/S2/`, ADR-0010, registry, README, architecture
- [ ] T011 Run ruff, format, mypy, pytest, and SCRAP
