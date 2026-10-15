# Tasks: F6 Geometry Probe (experiment P1)

**Input**: `specs/007-geometry-probe/spec.md`. **Tests**: required (constitution VI), CPU, offline.

- [X] T001 [P] Tests in tests/unit/test_space_geometry.py: `corpus_geometry` on independent columns with variances (4, 1, 1, 0) gives participation ratio 2 and those variances; any chunk size gives the same mean, variance, and eigenvalues; energy in the mean is `‖μ‖² / E‖x‖²` by hand
- [X] T002 [P] Tests in tests/unit/baselines/test_variance_importance.py: `variance_importance` equals `q² · var` by hand; on a corpus with a constant-offset dimension and a varying one, k = 1 keeps the varying one and ranks documents as the full query does
- [X] T003 Implement `CorpusGeometry` and `corpus_geometry(documents, chunk_size, device)` in src/hyperdime/space/geometry.py (streamed float64 sums, covariance eigenvalues, participation ratio, energy in the mean)
- [X] T004 Implement `variance_importance(queries, variance)` in src/hyperdime/baselines/methods.py
- [X] T005 Add `hyperdime probe-geometry` in src/hyperdime/cli/probe_geometry.py, registered in src/hyperdime/cli/__main__.py: per environment, `geometry-<env>.json` (statistics, eigenvalues) and `results-<env>-<set>.jsonl` for `variance` at every 10% budget on the S1 run's query sets, under `evaluation/P1/all/<run_id>/` with `inputs.json` naming the S1 run
- [X] T006 [P] Tests in tests/unit/reports/test_p1.py with synthetic P1 and S1 files: paired comparisons with Holm per comparison, the S2-promise reading, the dimensionality table, full and prefix rows, byte-identical regeneration
- [X] T007 Implement src/hyperdime/reports/p1.py and `hyperdime report P1` in src/hyperdime/cli/report.py
- [X] T008 Integration in tests/integration/test_f5_pipeline.py: `probe-geometry` and `report P1` on the S1 toy comparison
- [ ] T009 Run on DataPolicy-4090; commit `reports/P1/`; add P1 to docs/experiment_registry.md, the command to README.md and docs/architecture.md
- [ ] T010 Run ruff, format, mypy, pytest, and SCRAP
