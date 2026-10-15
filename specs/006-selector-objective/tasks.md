---

description: "Task list for F5 selector objective and experiment E2"
---

# Tasks: F5 Selector Objective and Experiment E2

**Input**: `specs/006-selector-objective/` (spec, plan, research, data-model, contracts). **Tests**:
required (constitution VI), CPU, offline, tests first in each story. Run tasks (T024+) execute on
DataPolicy-4090 (shared GPU: wait for ≥ 20 GB free). Rules referenced as D1–D9 are in research.md.

## Phase 1: Foundational — mask, loss, examples (blocks every story)

- [X] T001 [P] Tests in tests/unit/test_relaxed_mask.py: `soft_top_k(scores, k, beta)` on `s = [3, 1, 2, 0]`, k = 2: standardized `z`, threshold = midpoint of the 2nd and 3rd largest `z`, `mask > 0.5` exactly on {0, 2}; as `beta → 0` the mask equals `top_k_mask(s, 2)` within 1e-6; at k = 1024 (or k = D) the mask is all ones; the threshold carries no gradient (`torch.autograd.grad` of `mask.sum()` w.r.t. `s` equals the sigmoid term only); scaling `s` by 10 leaves the mask unchanged; `soft_mass_on_topk` by hand
- [X] T002 [P] Tests in tests/unit/training/test_retrieval_loss.py: `masked_retrieval_loss` by hand on D = 2, one query `q = [1, 0]`, mask `[1, 1]`, one positive `[1, 0]`, one negative `[0, 1]`, τ_s = 1: loss = `−log(e¹ / (e¹ + e⁰))`; gain weights `2**g − 1` normalized per query (grades {1, 2} → weights {1/4, 3/4}); in-batch positives of other queries enter the denominator; repeated negatives count twice; the masked query is not re-normalized (doubling the mask doubles the logits)
- [X] T003 [P] Tests in tests/unit/training/test_retrieval_examples.py: `retrieval_examples` gathers each unique document row once, keeps positives with grade > 0 and their weights, keeps the 64-pool repeats, skips and counts queries whose positives are all missing from the corpus; `load_examples` refuses any split but train and validation
- [X] T004 Implement `soft_top_k(scores: Tensor, k: int, beta: float) -> Tensor` and `soft_mass_on_topk(mask: Tensor, scores: Tensor, k: int) -> Tensor` in src/hyperdime/selection/relaxed.py (D3); export from src/hyperdime/selection/__init__.py
- [X] T005 Implement `masked_retrieval_loss(queries, mask, documents, positives, weights, negatives, score_temperature) -> Tensor` in src/hyperdime/training/losses.py (D2), beside `selector_kl_loss`
- [X] T006 Implement `RetrievalExamples` (fields per data-model.md: `queries` `[N, 1024]`, `documents` `[M, 1024]` unique rows, `positives` per query indices and weights "grade > 0; weights `2**g − 1`, normalized per query", `negatives` `[N, 64]` "the F2 pool, repeats kept", `skipped`) and `retrieval_examples(...)` in src/hyperdime/training/examples.py (D1), and extend `load_inputs` in src/hyperdime/training/inputs.py to expose training and validation examples without reading test judgments

## Phase 2: User Story 1 — a selector trained for retrieval at a budget (P1)

**Goal**: train the linear selector with the retrieval loss at k = 307, choose cells on
validation, save checkpoints. **Independent test**: on a synthetic environment with planted
relevant dimensions the selector recovers ≥ 90% of them and beats an untrained selector (SC-005);
checkpoints are unchanged when test judgments are removed (FR-014).

- [X] T007 [P] [US1] Tests in tests/unit/training/test_retrieval_trainer.py: on a planted synthetic set (D = 32, relevant dims {0..7}, positives differ from negatives only there) `train_retrieval_selector` at k = 8 puts ≥ 7 of the 8 planted dims in its top 8 for ≥ 90% of queries and lowers pool-rerank loss; same seed → identical state dict; `init="l2s"` starts from the given state; the kept checkpoint is the best validation pool-rerank nDCG@10 among those scored every `eval_every` steps; β follows the geometric schedule from 1 to `beta_end`
- [X] T008 [US1] Implement `RetrievalConfig` (`variant`, `init`, `lr`, `score_temperature`, `beta_end`, `steps = 20000`, `batch_size = 32`, `weight_decay = 1e-4`, `eval_every = 1000`, `seed`, `device`) and `train_retrieval_selector(train: RetrievalExamples, validation: RetrievalExamples, config, start: Mapping[str, Tensor] | None) -> tuple[LinearSelector, list[dict[str, float]]]` in src/hyperdime/training/retrieval_trainer.py: AdamW, cosine over steps, budget per batch (D4: k = 307 for `fixed-307`), history rows `step`, `train_loss`, `val_rerank_ndcg10`, `soft_mass`
- [X] T009 [US1] Add `RetrievalCell` (`variant` {`fixed-307`, `multi-budget`}, `init` {`scratch`, `l2s`}, `lr` {1e-4, 1e-3}, `score_temperature` {0.02, 0.05}, `beta_end` {0.1, 0.02}; `key` `<variant>-<init>-lr<lr>-ts<τ_s>-b<β_end>`), `retrieval_cells()` (32 cells in that product order), `score_on_validation_budgets` (full-corpus validation nDCG@10 at the nine 10%–90% budgets, all cells in one `evaluate_methods` pass, mean over budgets), the D3 guard (soft mass ≥ 0.9 to be selectable), `best_retrieval` (ties to the earlier cell), and `family_bests` in src/hyperdime/training/retrieval_selection.py (beside the L2S recipe in selection.py), with tests in tests/unit/training/test_retrieval_selection.py
- [X] T010 [US1] Add `--recipe retrieval-grid` to src/hyperdime/cli/train_selector.py: seed 0, for `init=l2s` read the latest `selectors/seeded/<env>/seed-0` read-only (exit 1 on a payload SHA-256 mismatch), train all cells, score, select; write `selectors/retrieval-grid/<env>/<run_id>/` with `<cell key>.safetensors`, `<cell key>.json` (`kind="per_domain"`), `history.json`, `selection.json` (`objective: "retrieval"`, inputs, per-cell scores per budget and mean, guard, selected cell, family bests, `skipped`)
- [X] T011 [US1] Integration in tests/integration/test_f5_pipeline.py (module-scoped toy stack reusing the F4 toy builders: three environments, seeded L2S selectors): `--recipe retrieval-grid` with `--steps 3` writes every file and scores every cell at the nine budgets; deleting test judgments gives identical checkpoints; an L2S start whose payload changed makes it exit 1

## Phase 3: User Story 2 — one selector for every budget (P1)

**Goal**: the multi-budget variant. **Independent test**: its seeded budget sequence repeats, it
sees every budget, and on the planted set its masked retrieval at each budget is within 0.05
pool-rerank nDCG@10 of the fixed-budget selector trained at that budget.

- [ ] T012 [P] [US2] Tests in tests/unit/training/test_retrieval_trainer.py: `budget_schedule(steps, seed)` draws from the nine `ratio_budgets()` values below 1024 only, uses all nine within 200 steps, and repeats for the same seed; the multi-budget selector on the planted set meets the 0.05 tolerance at k ∈ {D/4, D/2}
- [ ] T013 [US2] Implement `budget_schedule` and the `multi-budget` branch of `train_retrieval_selector` in src/hyperdime/training/retrieval_trainer.py (D4); record the schedule's seed in the history

## Phase 4: User Story 3 — experiment E2 against L2S at every budget (P1)

**Goal**: seeded retrieval selectors, `compare-selectors`, `report E2`, `decide E2`.
**Independent test**: synthetic result files with a planted improvement in one environment find it
there only; the report regenerates byte-identically.

- [ ] T014 [P] [US3] Tests in tests/unit/contracts/test_f5_records.py: `GateDecision(gate="E2")` round-trips; an unknown gate is still rejected
- [ ] T015 [US3] Add `"E2"` to `GateDecision.gate` in src/hyperdime/contracts/schemas.py (additive; D9)
- [ ] T016 [US3] Add `--recipe retrieval-seeded` to src/hyperdime/cli/train_selector.py: read the latest `selectors/retrieval-grid/<env>` selection, refuse (exit 1) if current inputs differ, retrain the selected cell with `run.seed` (l2s start from E1 seed `n`), write `selectors/retrieval-seeded/<env>/seed-<n>/<run_id>/` (D9)
- [ ] T017 [P] [US3] Tests in tests/unit/reports/test_e2.py with a synthetic builder tests/unit/reports/synthetic_e2.py: `budget_mean` per query over the nine budgets by hand; comparison against L2S averages seeds per query, then paired test, bootstrap CI, Holm over the three environments, separately for the budget mean and k = 307; a planted +0.05 in one environment gives criterion 1 false (1 of 3) and with two environments true; a planted −0.05 makes criterion 2 false; TREC DL rows never enter Holm; every table has full and prefix rows and states the protocol; regeneration is byte-identical
- [ ] T018 [US3] Add `hyperdime compare-selectors` in src/hyperdime/cli/compare_selectors.py and register it in src/hyperdime/cli/__main__.py: load retrieval-seeded, E1 seeded, grid family bests (seed 0), the global selector, and F3's global-importance τ read-only (exit 1 on a missing seed, a seeded cell that differs from the grid selection, or a hash mismatch); per environment and query set one `evaluate_methods` pass at every budget of `ratio_budgets()` below 1024 plus `full` at 1024, with method names `retrieval:<env>:s<n>`, `l2s:<env>:s<n>`, `family:<variant>-<init>:<env>`, `global-selector`, `prefix`, `random`, `global-importance`; write `evaluation/E2/all/<run_id>/results-<env>-<set>.jsonl` and `inputs.json` (D7)
- [ ] T019 [US3] Implement src/hyperdime/reports/e2.py (`load`, `budget_mean`, `comparison`, `criteria`, tables `progression.csv`, `comparison.csv`, `vs_full.csv`, `trec.csv`, `families.csv`, `grid.csv`, `e2.csv`, README) and `hyperdime report E2` in src/hyperdime/cli/report.py
- [ ] T020 [US3] Implement `decision` and `proposed_adr` for E2 (reusing the G2 pattern of src/hyperdime/reports/g2.py) in src/hyperdime/reports/e2.py and `hyperdime decide E2` in src/hyperdime/cli/decide.py: `decisions/E2/<run_id>/gate_decision.json` and `docs/decisions/ADR-0009-e2-selector-objective.md` (status "proposed"; an ADR not "proposed" is not overwritten)
- [ ] T021 [US3] Extend tests/integration/test_f5_pipeline.py through `retrieval-seeded` (seeds 0–2), `compare-selectors` (every method at every budget, `full` only at 1024; checkpoints unchanged; second run identical result files), `report E2`, and `decide E2`

## Phase 5: User Story 4 — where the gain comes from (P2)

**Goal**: overlap, fit, and training-curve diagnostics. **Independent test**: identical selectors
overlap 1 at every budget; a random selector's overlap is near k/1024.

- [ ] T022 [P] [US4] Tests in tests/unit/reports/test_e2.py: `overlap.json` builder gives 1.0 for identical selectors and ≈ k/D for random ones; the seed noise floor is the mean of the 3 seed pairs; fit reuses E1's `fit_gain` rule; `overlap.csv`, `fit.csv`, and the training-curve table appear in the report
- [ ] T023 [US4] In src/hyperdime/cli/compare_selectors.py write `overlap.json` (per environment and budget: retrieval↔L2S seed-matched overlap, both noise floors) and `fit.json` (oracle-mask agreement on validation for both selectors, F4 research D2); in src/hyperdime/reports/e2.py add `overlap.csv`, `fit.csv`, and the training-curve table from the grid and seeded `history.json` (D8)

## Phase 6: Runs (DataPolicy-4090)

- [ ] T024 Sync the branch to DataPolicy-4090 (git bundle) and run `train-selector --recipe retrieval-grid` for scifact, nfcorpus, msmarco; check every environment has at least one cell passing the D3 guard (else apply the straight-through fallback of D3 through an ADR before continuing)
- [ ] T025 Run `train-selector --recipe retrieval-seeded` for seeds 0, 1, 2 per environment; record each seed's training time against SC-006
- [ ] T026 Run `compare-selectors`, `report E2`, `decide E2`; record wall time against SC-006; copy `reports/E2/` and the proposed ADR-0009 into the branch

## Phase 7: Polish

- [ ] T027 [P] Document the retrieval objective, recipes, and commands in docs/architecture.md and README.md; add E2 to docs/experiment_registry.md
- [ ] T028 Run `ruff check . && ruff format --check . && mypy && pytest` and `PYTHONPATH=tools python -m scrap tests`; act on HIGH recommendations for changed test files

## Dependencies

Phase 1 → US1 → US2 (extends the trainer) → US3 (needs both variants; T014–T015 can start after
Phase 1) → US4 (extends `compare-selectors` and the report) → Runs → Polish. US3's report tests
(T017) only need synthetic files and can run in parallel with US1–US2.

## Parallel examples

- Phase 1: T001, T002, T003 together; then T004, T005, T006 (different files).
- US1: T007 while T009's selection helpers are written.
- US3: T014, T017 together; T018 and T019 after them.

## Implementation strategy

MVP is Phase 1 + US1: a fixed-budget retrieval-trained selector chosen on validation, which can
already be compared with L2S by hand on validation. US2 adds the budget-free variant, US3 the
evidence, US4 the explanation. Commit per task group with Conventional Commits scoped by package.
