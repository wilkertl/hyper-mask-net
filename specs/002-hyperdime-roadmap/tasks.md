---

description: "Task list for the HyperDIME-Qwen project roadmap"
---

# Tasks: HyperDIME-Qwen Project Roadmap

**Input**: Design documents from `specs/002-hyperdime-roadmap/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md

**Tests**: Required. The constitution (principle VI) and FR-000a require a test for every new behavior,
with hand-computable unit tests for critical math. Tests run on CPU without network access.

**Organization**: This is an umbrella feature. User Story 0 (the F0 spike) is broken into concrete
tasks, several already done on branch `agent/F0-h1-spike`. User Stories 1–5 are milestones, each
delivered by its own spec-kit feature, so their tasks are feature-level: specify, plan, generate
tasks, implement, run the experiment, record the gate. Their detailed tasks live in each feature's
own `tasks.md`. A story starts only when the previous gate passed.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: US0–US5 from spec.md
- `[X]` marks tasks already completed on `agent/F0-h1-spike`

## Path Conventions

Single project: `src/hyperdime/`, `tests/`, `configs/`, `reports/`, `docs/decisions/`, `specs/`.

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Get governance, planning docs, and tooling into version control.

- [X] T001 Fill the project constitution from AGENTS.md and the plan in .specify/memory/constitution.md (v1.0.0)
- [X] T002 Remove the Sync Impact Report HTML comment at the top of .specify/memory/constitution.md, then commit it with message `docs: ratify constitution v1.0.0`
- [X] T003 Commit specs/001-m0-foundation/ and specs/002-hyperdime-roadmap/ with message `docs(specs): add M0 foundation spec and project roadmap`
- [X] T004 [P] Exclude vendored `.specify` and `.agents` from ruff via `extend-exclude` in pyproject.toml
- [X] T005 [P] Set `[tool.mypy] python_version = "3.12"` in pyproject.toml, since numpy ≥ 2.5 stubs use 3.12 syntax (ruff keeps `target-version = "py311"`)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: The only prerequisite shared by every story: the package must be runnable as installed.

- [X] T006 Verify `pip install -e ".[embed,dev]"` in a fresh virtualenv makes `python -m hyperdime.spike --help` work without `PYTHONPATH`; if not, fix package discovery under `[tool.setuptools.packages.find]` in pyproject.toml

**Checkpoint**: `ruff check . && ruff format --check . && mypy && pytest` passes (60 tests).

---

## Phase 3: User Story 0 - Quick test of selector shift (Priority: P0) 🎯 MVP

**Goal**: An H1 verdict (research R0) from five small BEIR environments within days, before any
roadmap work.

**Independent Test**: `python -m hyperdime.spike all` with `V4_EMBEDDING_BASE_URL` set produces
`reports/F0/README.md` ending in a verdict under R0's three-part rule.

### Tests for User Story 0

- [X] T007 [P] [US0] Unit tests for BEIR parsing, oracle aggregation (hand-computed), trainer determinism, Holm (hand-computed), randomization test and bootstrap extremes, overlap/JS/Spearman, identical-ID exclusion, resumable cache, cache settings guard, leakage-safe splits, and corpus sampling in tests/unit/test_spike_components.py
- [X] T008 [P] [US0] vLLM client tests against a fake OpenAI-compatible server on loopback (endpoint rules, batching, truncation, index reordering, retries, malformed responses, missing model, API key) in tests/unit/test_vllm_encoder.py
- [X] T009 [P] [US0] Synthetic end-to-end run (cached embeddings → train → global → analyze → report, byte-identical rerun) in tests/integration/test_spike_pipeline.py

### Implementation for User Story 0

- [X] T010 [P] [US0] BEIR download and parsing (`download_beir`, `read_beir`, `judged_queries`) in src/hyperdime/data/loaders.py
- [X] T011 [P] [US0] Local frozen encoder `QwenEncoder` (length-sorted batches, last-token pooling, L2 norm, dtype) in src/hyperdime/embeddings/qwen.py
- [X] T012 [P] [US0] vLLM encoder `VllmEncoder` with the beir project's `V4_EMBEDDING_*` contract and client-side Qwen-token truncation in src/hyperdime/embeddings/remote.py
- [X] T013 [P] [US0] Resumable embedding cache with a settings guard (`embed_cached`, `load_cached`) in src/hyperdime/embeddings/cache.py
- [X] T014 [P] [US0] Oracle targets with relevance-weighted positives and mean hard negatives (research R4) in src/hyperdime/oracle/targets.py
- [X] T015 [P] [US0] Selector training with early stopping on validation KL (`train_selector`, `TrainConfig`) in src/hyperdime/training/selector_trainer.py
- [X] T016 [P] [US0] Paired randomization test, bootstrap CI, and Holm correction in src/hyperdime/evaluation/statistics.py
- [X] T017 [P] [US0] Overlap@k, JS divergence, Spearman, and mean importance in src/hyperdime/evaluation/selector_shift.py
- [X] T018 [P] [US0] Rankings with BEIR identical-ID exclusion (`rank_documents`) in src/hyperdime/retrieval/scoring.py
- [X] T019 [US0] Spike environments, leakage-safe splits (train qrels or 50/50 half split), and corpus sampling in src/hyperdime/spike/environments.py
- [X] T020 [US0] Spike steps `prepare`, `train`, `train_global`, `analyze` with R0's verdict rule in src/hyperdime/spike/pipeline.py, plus lightweight manifests in src/hyperdime/spike/manifest.py
- [X] T021 [US0] Report renderer in src/hyperdime/spike/report.py and CLI with `--backend auto|vllm|local` in src/hyperdime/spike/__main__.py
- [X] T022 [US0] Commit the spike on `agent/F0-h1-spike` with message `feat(spike): add F0 selector-shift experiment with vLLM embeddings`, after `ruff check . && ruff format --check . && mypy && pytest` passes
- [ ] T023 [US0] With the vLLM server up, export `V4_EMBEDDING_BASE_URL`, confirm `curl -s $V4_EMBEDDING_BASE_URL/models` lists `Qwen/Qwen3-Embedding-0.6B`, and run `python -m hyperdime.spike prepare` (writes artifacts/spike/<env>/ for scifact, nfcorpus, fiqa, arguana, scidocs)
- [ ] T024 [US0] Run `python -m hyperdime.spike train`, then `train-global`, then `analyze` (writes artifacts/spike/<env>/selectors/, artifacts/spike/global/, artifacts/spike/analysis/)
- [ ] T025 [US0] Run `python -m hyperdime.spike report` and commit the generated reports/F0/README.md on an `exp/F0-selector-shift` branch; never edit its numbers by hand
- [ ] T026 [US0] Record the H1 verdict, the three criteria values, the headroom per environment, and the decision (continue to US1, or stop / pivot to a universal selector) in docs/decisions/ADR-0002-h1-spike-verdict.md, starting from docs/decisions/ADR-0000-template.md
- [ ] T027 [P] [US0] Optional (research R0): a first H2 signal with group A document-marginal statistics (mean, std, mean_abs, RMS, q05–q95 per dimension) in src/hyperdime/space/marginal.py, a leave-one-environment-out probe predicting each dimension's mean importance against the "mean of other environments" baseline as a `probe` step in src/hyperdime/spike/pipeline.py, and a hand-computed test in tests/unit/test_space_marginal.py

**Checkpoint**: The ADR states whether H1 is supported. Stop here if it is not.

---

## Phase 4: User Story 1 - Valid, reproducible baseline (Priority: P1) — Milestone 1, Gate G1

**Goal**: Learning-to-Select reproduced on SciFact, NFCorpus, and full MS MARCO (with TREC DL
2019/2020/DL-HARD evaluation sets) under the full provenance protocol.

**Independent Test**: The E0 report regenerates byte-identically and shows the G1 criteria of
research R7.

- [ ] T028 [US1] Generate tasks for feature F1 by pointing .specify/feature.json at `specs/001-m0-foundation` and running `/speckit-tasks`, which writes specs/001-m0-foundation/tasks.md
- [ ] T029 [US1] Implement F1 (tickets M0.1–M0.4) with `/speckit-implement`: src/hyperdime/contracts/{schemas,validation,config,hashing,seeding,manifests}.py, `hyperdime diagnose`, uv.lock, .pre-commit-config.yaml; move the spike's manifests in src/hyperdime/spike/manifest.py onto `RunManifest`
- [ ] T030 [US1] Specify, plan, and generate tasks for feature F2 `data-embeddings` (M1.1–M1.4, M2.1–M2.2; research R2, R5) with `/speckit-specify`, `/speckit-plan`, `/speckit-tasks`, reusing src/hyperdime/data/loaders.py, src/hyperdime/embeddings/remote.py, and src/hyperdime/embeddings/cache.py
- [ ] T031 [US1] Implement F2, then embed SciFact, NFCorpus, and full MS MARCO through vLLM, with `ir_datasets` query sets `msmarco-passage/trec-dl-2019/judged`, `msmarco-passage/trec-dl-2020/judged`, and `dl-hard`
- [ ] T032 [US1] Specify, plan, and implement feature F3 `l2s-baseline` (M12.1 with AP, M3.1–M3.3; research R4, R6), reusing src/hyperdime/oracle/targets.py and src/hyperdime/training/selector_trainer.py
- [ ] T033 [US1] Run E0 on an `exp/E0-baseline-reproduction` branch, generate reports/E0/ (Learning-to-Select-style and DIME-style tables per contracts/reports-and-gates.md), and write docs/decisions/ADR-XXXX-g1-baseline.md; tag the merge `g1-baseline-valid`

**Checkpoint**: G1 decided.

---

## Phase 5: User Story 2 - Go/No-Go on selector shift (Priority: P2) — Milestone 2, Gate G2

**Goal**: The H1 decision under the full protocol (research R8).

**Independent Test**: `reports/E1/` regenerates from result files and its ADR states G2.

- [ ] T034 [US2] Specify, plan, and implement feature F4 `selector-shift` (M12.3, M4.1–M4.3), promoting src/hyperdime/evaluation/statistics.py and src/hyperdime/evaluation/selector_shift.py from the spike and adding `hyperdime transfer-matrix`
- [ ] T035 [US2] Run E1 (3 seeds per environment) on `exp/E1-selector-shift`, generate reports/E1/, and write docs/decisions/ADR-XXXX-g2-selector-shift.md; tag `g2-selector-shift`. On No-Go, close later milestone issues as not planned

**Checkpoint**: G2 decided. Stop on No-Go.

---

## Phase 6: User Story 3 - Space statistics predict the shift (Priority: P3) — Milestone 3, Gate G3

**Goal**: Frozen `SpaceDescriptorV1` and the H2 decision (research R10).

**Independent Test**: The descriptor invariance tests pass and `reports/E2/` states G3.

- [ ] T036 [US3] Write docs/decisions/ADR-XXXX-environment-pool.md (research R19 datasets and revisions; Robust04 availability) and docs/decisions/ADR-XXXX-meta-split.md (research R3 proposal, `corpus_group` rule from data-model.md) before any normalization is fitted
- [ ] T037 [US3] Specify, plan, and implement feature F5 `space-descriptor` (M5.1–M5.5, protocol-typed builders per contracts/module-interfaces.md), then embed the remaining reference datasets through vLLM
- [ ] T038 [US3] Run E2 probes on `exp/E2-space-signal`, generate reports/E2/, and write docs/decisions/ADR-XXXX-g3-space-signal.md; tag `g3-space-signal`

**Checkpoint**: G3 decided and descriptor v1 frozen.

---

## Phase 7: User Story 4 - Working hypernetwork prototype (Priority: P4) — Milestone 4

**Goal**: A meta-trained hypernetwork that matches or beats the global selector on meta-validation.

**Independent Test**: Zero-gate generation equals the base selector, and the meta-trainer refuses
meta-test environments.

- [ ] T039 [US4] Specify, plan, and implement feature F6 `hyperdime-prototype` (global selector R13, M7.1, M6.1, M8.1, M9.1; research R12, R14) in src/hyperdime/{hypernet,space,training}/
- [ ] T040 [US4] Run meta-validation, generate reports/meta-validation/, and write docs/decisions/ADR-XXXX-meta-validation.md (match / beat / lose); meta-test stays locked unless match or beat

**Checkpoint**: Meta-validation verdict recorded.

---

## Phase 8: User Story 5 - Paper-grade evaluation (Priority: P5) — Milestone 5, Gates G4 and G5

**Goal**: E3–E6 on meta-test under P0/P1/P2, with comparability tables (FR-019).

**Independent Test**: `hyperdime report` regenerates reports/E3–E6 byte-identically.

- [ ] T041 [US5] Specify, plan, and implement feature F7 `paper-evaluation` (M10.1, M11.1, M12.2; research R15, R16)
- [ ] T042 [US5] Run E3, E4, E5, and E6, generate reports/E3/ to reports/E6/, and write docs/decisions/ADR-XXXX-g4-hypernetwork.md and ADR-XXXX-g5-cost.md; tag `g4-hypernetwork` and `g5-cost`

**Checkpoint**: G4 and G5 decided.

---

## Phase 9: Polish & Cross-Cutting Concerns

- [X] T043 [P] Update the Components table and invariants in docs/architecture.md with the modules added by the spike (loaders, remote/cache embeddings, targets, trainer, statistics, selector_shift, scoring, spike)
- [X] T044 [P] Document the vLLM setup (`V4_EMBEDDING_*` variables, `ssh -L` tunnel, `curl $V4_EMBEDDING_BASE_URL/models`) and the spike command in README.md
- [ ] T045 Confirm the `checks` job in .github/workflows/ci.yml passes on the spike branch: it installs `.[dev]` without transformers and runs mypy (now `python_version = "3.12"`) on CI's Python 3.12; the 52 tests need neither network nor transformers

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: none
- **Foundational (Phase 2)**: after Setup; blocks US0's run tasks (T023–T026)
- **US0 (Phase 3)**: code done; the run (T023–T026) needs the vLLM server
- **US1–US5 (Phases 4–8)**: strictly sequential, each gated on the previous decision: ADR-0002 (H1
  supported) → G1 → G2 → G3 → meta-validation verdict → G4/G5
- **Polish (Phase 9)**: T043–T044 any time after T022; T045 before the first pull request

### User Story Dependencies

- **US0**: independent; the MVP
- **US1**: needs US0 verdict = supported
- **US2**: needs G1 = pass
- **US3**: needs G2 = pass
- **US4**: needs G3 = pass
- **US5**: needs meta-validation ≠ lose

### Within Each User Story

- Tests before or with implementation; each story ends in a generated report and an ADR
- Experiment runs on `exp/` branches commit only configs and generated reports

### Parallel Opportunities

- T004, T005 (done); T007–T018 (done, independent modules)
- T027 (optional H2 probe) in parallel with T023–T026
- T043, T044 in parallel with any story phase

---

## Parallel Example: User Story 0

```bash
# While embeddings run on the vLLM server (T023), work on independent files:
Task: "Optional group-A descriptor probe in src/hyperdime/space/marginal.py (T027)"
Task: "Update docs/architecture.md components table (T043)"
Task: "Document vLLM setup in README.md (T044)"
```

---

## Implementation Strategy

### MVP First (User Story 0 Only)

1. Finish Setup (T002–T003) and Foundational (T006)
2. Commit the spike (T022), run it on the vLLM server (T023–T025)
3. **STOP and decide**: record the H1 verdict (T026)
4. Only if supported: continue with US1

### Incremental Delivery

Each later story is one spec-kit feature ending in a tagged gate decision. A failed gate stops the
roadmap and is kept as evidence.

---

## Notes

- `[X]` tasks were implemented and verified with `ruff check . && ruff format --check . && mypy && pytest` (60 passed)
- No task may edit report numbers by hand; tables come from result files
- Commit after each task or logical group, with Conventional Commits scoped by package

---

## Phase 10: Convergence

- [X] T046 Extend tests/integration/test_spike_pipeline.py with a leakage test: run `pipeline.train` and `pipeline.train_global` twice, the second time with every `eval_labels` entry in each artifacts `splits.json` replaced by labels on other documents, and assert the saved selector weights are identical, per US0/AC4 and Constitution II (partial)
- [X] T047 Write a `manifest.json` (via `write_manifest` from src/hyperdime/spike/manifest.py, with the cache `settings`, row count, and SHA-256 of `embeddings.npy`) into each finished `corpus/` and `queries/` cache directory inside `prepare` in src/hyperdime/spike/pipeline.py, so no vector file exists without a manifest even if `prepare` stops before the environment manifest, per Constitution III and FR-000b (partial)
- [X] T048 Record the SHA-256 of each downloaded BEIR zip (before it is deleted) and a sorted-file tree SHA-256 of the extracted directory in `download_beir` in src/hyperdime/data/loaders.py (e.g. in artifacts/spike/beir/<name>.sha256.json), and include both as `dataset_hash` in the environment manifest written by `prepare` in src/hyperdime/spike/pipeline.py, with a hand-checkable test on a tiny zip in tests/unit/test_spike_components.py, per Constitution: Artifacts and Data Handling (partial)
- [X] T049 [P] Make the tokenizer injectable in `qwen_truncator` (src/hyperdime/embeddings/remote.py) and the model/tokenizer injectable in `QwenEncoder` (src/hyperdime/embeddings/qwen.py), then add stub-based unit tests for truncation to `max_length - 1` tokens and for input-order, L2-normalized output in tests/unit/test_vllm_encoder.py and a new tests/unit/test_qwen_encoder.py, per FR-000a (partial)
- [X] T050 [P] Update the F0 row of specs/002-hyperdime-roadmap/plan.md from "BEIR loader (HF)" to the UKP BEIR zip source actually used by src/hyperdime/data/loaders.py, stating why (train/dev/test qrels in one archive; the Hugging Face path stays the plan for F2 per research R2), per plan: F0 scope (contradicts)
