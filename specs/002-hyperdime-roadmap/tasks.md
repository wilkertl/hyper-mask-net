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
- Task IDs are stable: tasks added later (T053–T068, for FR-000d and research R20) take the next free
  IDs and are placed where they run, so IDs are not always in execution order

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
- [X] T053 Declare `numpy>=2` in `[project] dependencies` of pyproject.toml (imported by src/hyperdime/embeddings/remote.py, src/hyperdime/embeddings/cache.py, src/hyperdime/evaluation/statistics.py, src/hyperdime/spike/manifest.py; torch does not install it), then confirm that `pip install -e ".[dev]"` in a fresh Python 3.12 virtualenv runs `pytest` with no collection errors

**Checkpoint**: `ruff check . && ruff format --check . && mypy && pytest` passes (64 tests) in a fresh virtualenv.

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
- [X] T023 [US0] Embed the five environments: run `python -m hyperdime.spike prepare` (done on the DGX Spark cluster with `--backend local --device cuda` via scripts/slurm/f0-spike.sbatch, job f0-spike-407, commit `d6453f0`; writes artifacts/spike/<env>/ for scifact, nfcorpus, fiqa, arguana, scidocs)
- [X] T024 [US0] Run `python -m hyperdime.spike train`, then `train-global`, then `analyze` (same job; writes artifacts/spike/<env>/selectors/, artifacts/spike/global/, artifacts/spike/analysis/; every manifest `complete`, `git_dirty: false`)

### Diagnostics for User Story 0 (FR-000d, research R20)

The rules are fixed in research R20 (commit `a72a3c9`) and must not change once `diagnose` has run.
Every new function follows contracts/module-interfaces.md, section "F0 diagnostics"; the output
follows `DiagnosticsResult` in data-model.md; the CLI follows contracts/cli.md, section "Spike F0".

**Tests** (write first; they must fail before T058–T063 exist):

- [X] T054 [P] [US0] Hand-computed tests for `selection_frequency` (3 queries × 4 dimensions, k = 2: the result equals the counted masks and sums to k), `split_half_spearman` (two identical halves give 1.0; seeded and repeatable), and `profile_permutation_test` (identical score sets give p near 1, disjoint top-k supports give p ≤ 1/(n+1) + tolerance; seeded) in tests/unit/test_selector_shift_profiles.py
- [X] T055 [P] [US0] Hand-computed tests for `marginal_stats` in tests/unit/test_space_marginal.py: on a 4 × 3 tensor, check mean, std, mean_abs, RMS, and the q05/q25/q50/q75/q95 quantiles column by column; shuffling document rows leaves the output unchanged; permuting dimensions permutes output rows; output shape `(1024, 9)` on a random `(10, 1024)` input, with `MARGINAL_FEATURES` naming the 9 columns in order
- [X] T056 [P] [US0] Tests for `loeo_probe` and the D3 permutation test in tests/unit/test_spike_diagnostics.py: with 5 synthetic environments whose targets are a planted linear function of the features plus noise, the mean gain is > 0 and the exact permutation p is ≤ 0.05; with targets shuffled across environments, the gain is ≤ 0; multiplying the held-out environment's features by 1000 does not change the z-scoring statistics of that fold (normalization uses the 4 training environments only); the permutation test enumerates exactly 120 assignments
- [X] T057 [US0] Extend tests/integration/test_spike_pipeline.py: after the synthetic `analyze`, `diagnose` writes artifacts/spike/diagnostics/diagnostics.json with `oracle_shift`, `trainer`, `probe`, and `recommendation` keys and a `complete` manifest; replacing every `eval_labels` entry in each splits.json with labels on other documents leaves `oracle_shift` and `probe` unchanged; `diagnose` refuses to run when analysis/manifest.json is not `complete`; `report` output is byte-identical to the pre-diagnose output when diagnostics/ is absent, and gains a "Diagnostics (research R20)" section when it is complete; two `diagnose` runs give identical JSON

**Implementation**:

- [X] T058 [P] [US0] Add `selection_frequency(scores, k)`, `split_half_spearman(scores, k, draws, seed)`, and `profile_permutation_test(a, b, k, n, seed)` to src/hyperdime/evaluation/selector_shift.py, reusing `top_k_mask` from src/hyperdime/selection/topk.py and the existing `spearman`; statistic `1 − Spearman` of the two profiles, group sizes kept when shuffling, `p = (count + 1) / (n + 1)`
- [X] T027 [P] [US0] Descriptor group A, protocol P0: `marginal_stats(documents) -> Tensor (1024, 9)` and `MARGINAL_FEATURES = ("mean", "std", "mean_abs", "rms", "q05", "q25", "q50", "q75", "q95")` in a new src/hyperdime/space/marginal.py; documents only, no queries or labels (replaces the earlier optional `probe` step; the probe itself is T061)
- [X] T059 [US0] D1 in a new src/hyperdime/spike/diagnostics.py: `oracle_scores_for(data, qids)` returns `e_q ⊙ (p − n)` (reuse the positive weighting of `aggregate_targets` in src/hyperdime/oracle/targets.py and `oracle_scores` in src/hyperdime/oracle/importance.py) from `splits["train_labels"]` and the stored negatives only, for the train + validation queries; per environment the OracleProfile (`frequency` in [0, 1], "sums to k", k = 307) and `split_half_spearman` over 200 half-splits; for the 10 pairs, `rho_cross` over the same 200 random-half draws and `profile_permutation_test` with 1,000 permutations, Holm-adjusted; `o1 = (Holm-significant pairs ≥ 5) and (mean rho_cross < mean rho_within)`
- [X] T060 [US0] D2 in src/hyperdime/spike/diagnostics.py, per environment with τ and epochs read from <env>/selectors/temperature.json: (a) seed 0 on nested seeded subsets of 25/50/100% of the training queries, evaluation nDCG@10 at k = 307, 100%-vs-50% paired randomization test; (b) ensemble = mean of the three seeds' log-importances vs per-query mean of single seeds, paired test; (c) nDCG@10 on training queries (train labels) and evaluation queries, and overlap@k of selector mask vs oracle mask on each set; (d) Spearman of the selector's selection profile on train + validation queries vs the environment's OracleProfile; (e) grid-edge extension exactly as research R20 lists it, scored on validation nDCG@10 only, limited when the best extended cell beats the chosen one by > 0.005; Holm over environments for (a) and (b); `t = ` (a), (b), or (e) holds on ≥ 2 of 5 environments. Reuse `_fit`, `_evaluate`, `_load_selector`, and `_importance` from src/hyperdime/spike/pipeline.py rather than copying them
- [X] T061 [US0] D3 and the result in src/hyperdime/spike/diagnostics.py: `loeo_probe(features, targets, alpha=1.0)` with features from `marginal_stats` of each corpus, z-scored with statistics from the 4 training environments of each fold, ridge regression (closed form, α = 1 fixed) to the OracleProfile `frequency`, score = Spearman vs the held-out profile, baseline = mean profile of the other 4, gain = score − baseline; exact test over all 120 descriptor-to-environment assignments with mean gain as statistic; `s = mean_gain > 0 and p ≤ 0.05`; secondary target = trained selectors' mean importance, reported only; `recommendation` = `"stop_or_pivot"` if not O1, `"continue_strengthen_selector"` if O1 and T, else `"continue"`. `diagnose(envs, root, seeds, *, permutations=1000, half_splits=200)` calls `require_complete` on every input including analysis/, writes diagnostics/manifest.json with `status: "running"` before any output, then diagnostics.json, then the manifest with `complete`
- [X] T062 [US0] Add the `diagnose` step to src/hyperdime/spike/__main__.py with `--permutations` (default 1000) and `--half-splits` (default 200), honoring `--env`, `--seeds`, and `--root`; it is not part of `all`, and it prints the recommendation
- [X] T063 [US0] Render a "Diagnostics (research R20)" section in src/hyperdime/spike/report.py from diagnostics/diagnostics.json, only when its manifest is `complete`: a D1 table (per environment `rho_within`; per pair `rho_cross`, p Holm), the O1 value; a D2 table per environment (curve, ensemble test, fit gap, recovery, grid edge) and T; a D3 table per fold (score, baseline, gain), the permutation p out of 120, and S; then the recommendation, worded as in research R20's table

**Run and decide**:

- [X] T064 [US0] Add scripts/slurm/f0-diagnose.sh and scripts/slurm/f0-diagnose.sbatch, copied from scripts/slurm/f0-spike.sh and .sbatch, running `$PY -m hyperdime.spike diagnose && $PY -m hyperdime.spike report` on CPU (keeps `--gres=gpu:1` because the partition allocates whole GPU nodes; `--time=03:00:00`); the full caches and selector weights live only on the cluster (the local artifacts/spike/ copy has no `embeddings.npy` or `seed-*.pt`). List the job in scripts/slurm/README.md
- [X] T065 [US0] Run `diagnose` on the full F0 caches. Done locally on CPU instead of on the cluster (caches pulled from the DGX Spark over the VPN; every `embeddings.npy` matched its manifest's SHA-256; `analysis.json` byte-identical): 18.5 min at commit `7fdf1e5`, manifest `complete`, `git_dirty: false`
- [X] T025 [US0] Commit the generated reports/F0/README.md (with its Diagnostics section) on the `exp/F0-selector-shift` branch; never edit its numbers by hand
- [ ] T026 [US0] Accept or amend docs/decisions/ADR-0003-h1-spike-verdict.md (drafted as `proposed`: recommendation `continue_strengthen_selector`) (from docs/decisions/ADR-0000-template.md; ADR-0002 is the training recipe): the R0 verdict and its three criteria values, the per-environment headroom, the R20 values of O1, T, and S, the recommendation, and the decision (continue to US1 as planned, continue and strengthen the per-environment selector in F3, or stop / pivot to a universal selector). State that criteria 1 and 3 passed exactly at their thresholds and that ADR-0002 changed the recipe after a "not supported" run; tag the merge `f0-h1-verdict`

**Checkpoint**: ADR-0003 states whether H1 is supported and what R20 found. Stop here if it is not.

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
- [X] T045 Confirm the `checks` job in .github/workflows/ci.yml passes on the spike branch (open the pull request so it runs; so far it has run only on `master`): it installs `.[dev]` without transformers, which needs T053's numpy dependency, and runs mypy (`python_version = "3.12"`) on CI's Python 3.12; no test needs network or transformers
- [X] T066 [P] Add `hyperdime.space` (`marginal`) and the spike's `diagnose` step to the Components table in docs/architecture.md (the "F0 selector-shift spike" row lists `prepare|train|train-global|analyze|report|all`; add `diagnose`)
- [X] T067 [P] Update the test count in specs/002-hyperdime-roadmap/quickstart.md ("64 tests before R20") to the count `pytest` reports after T054–T057
- [X] T068 Decide on the untracked scripts/reports/build_project_report.py and reports/project/index.html: either commit them under a `chore(reports)` commit with the `gh` and logs/ inputs documented in the script docstring and kept out of any table that research R18 requires to come from result files alone, or delete them

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: none
- **Foundational (Phase 2)**: after Setup; T053 blocks the diagnostics tasks and the pull request (T045)
- **US0 (Phase 3)**: spike code done and run (T023–T024). Diagnostics order: tests T054–T057 →
  T058, T027 (parallel) → T059 → T060 → T061 → T062, T063 → T064 → T065 (cluster run) → T025 →
  T026
- **US1–US5 (Phases 4–8)**: strictly sequential, each gated on the previous decision: ADR-0002 (H1
  supported) → G1 → G2 → G3 → meta-validation verdict → G4/G5
- **Polish (Phase 9)**: T043–T044 any time after T022; T045 before the first pull request, after T053;
  T066–T067 after T063; T068 any time

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
- T054, T055, T056 (separate test files) together; then T058 and T027 (separate modules) together
- T066, T067, T068 in parallel with the cluster run (T065)
- T043, T044 in parallel with any story phase

---

## Parallel Example: User Story 0

```bash
# Diagnostics tests, three files at once:
Task: "Profile tests in tests/unit/test_selector_shift_profiles.py (T054)"
Task: "Group-A marginal tests in tests/unit/test_space_marginal.py (T055)"
Task: "LOEO probe tests in tests/unit/test_spike_diagnostics.py (T056)"
# Then the two independent modules:
Task: "Profile functions in src/hyperdime/evaluation/selector_shift.py (T058)"
Task: "marginal_stats in src/hyperdime/space/marginal.py (T027)"
```

---

## Implementation Strategy

### MVP First (User Story 0 Only)

1. Setup and Foundational done except T053 (numpy); do T053 first
2. Spike committed and run (T022–T024)
3. Diagnostics (T054–T063), cluster run (T064–T065), commit the report (T025)
4. **STOP and decide**: record the H1 verdict and R20's outcome in ADR-0003 (T026)
5. Only if ADR-0003 says continue: US1, with stronger per-environment selector candidates in F3
   when R20's recommendation is `continue_strengthen_selector`

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

---

## Phase 11: Convergence

- [X] T051 Make every spike output directory carry its manifest from the moment it has outputs: in src/hyperdime/spike/pipeline.py write the directory's `manifest.json` with `"status": "running"` before `prepare` writes `splits.json`/`negatives.json`, before `train`/`train_global` save any `seed-*.pt`, `history-seed-*.json`, or `temperature.json`, and before `analyze` writes `results.json`/`analysis.json`, then rewrite it with `"status": "complete"` at the end (add a `status` argument to `write_manifest` in src/hyperdime/spike/manifest.py); make `analyze` refuse inputs whose manifest is not complete, and cover the refusal in tests/integration/test_spike_pipeline.py, per Constitution III and FR-000b (partial)
- [X] T052 [P] Update the test count in specs/002-hyperdime-roadmap/quickstart.md ("52 tests" → the current count from `pytest`, 60 at the time of writing), per plan: quickstart validation guide (partial)
