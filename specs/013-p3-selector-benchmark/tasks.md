---
description: "Task list for F12 P3 selector benchmark (regenerated as built, 2026-11-24)"
---

# Tasks: F12 P3 Selector Benchmark

**Input**: `specs/013-p3-selector-benchmark/`: spec, plan (as built), research B0–B11, data-model,
contracts/cli.md, quickstart; and `docs/decisions/ADR-0018-p3-selector-benchmark.md`.

**Tests**: required (constitution VI and the spec's independent tests); CPU and offline.

**Status and IDs**:
- This file was regenerated on 2026-11-24 from the built code and
  [reports/P3-bench/RUNS.md](../../reports/P3-bench/RUNS.md).
- Task IDs are kept from the original list and the convergence pass, because plan.md, research.md
  and contracts/cli.md cite them. IDs are therefore not strictly sequential.
- The version fingerprinted at the protocol gate (SHA-256 `5b41c0e8…`) is recoverable from code
  snapshot `6ce68a0`.

**Marks**:
- `[X]` means done, with evidence in RUNS.md or the tests.
- A task that was done only in part is marked `[X]`, with the rest split out as "remainder → T0xx".

**Freeze gates (ADR-0018)**, each recorded with a SHA-256 fingerprint before the dependent stage:
1. protocol: T019, before the probe;
2. resource: T020, after the probe and before any tune metric, any teacher beyond the probe, or any
   evaluation label;
3. evaluation lock: T033a, before evaluation labels.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: can run in parallel (different files, no dependency on an incomplete task).
- **[Story]**: US0–US4 from spec.md.

---

## Phase 1: Setup (protocol documents)

**Purpose**: the frozen design every later stage relies on.

- [X] T002a Record the jointly proposed protocol in docs/decisions/ADR-0018-p3-selector-benchmark.md (status proposed)
- [X] T002b Close the remaining decisions in docs/decisions/ADR-0018-p3-selector-benchmark.md:
  - the S1 grid;
  - statistics;
  - the resource rule and freeze gates;
  - the replication rule.
  Review notes are in specs/013-p3-selector-benchmark/research.md B10.
- [X] T002c Narrow the scope in docs/decisions/ADR-0018-p3-selector-benchmark.md:
  - one official L2S per corpus on the primary arms' fit ids;
  - tuned L2S removed;
  - full-fit reference, other seeds and retraining deferred.
- [X] T019 Protocol gate: record and fingerprint ADR-0018 and the F12 docs with `p3-bench freeze-protocol` before any probe (reports/P3-bench/freeze/protocol.json, SHA-256 `00f68ec4…`)

---

## Phase 2: Foundational (blocking prerequisites)

**Purpose**: the split, the provenance and the shared evaluator. Every story depends on these.

- [X] T003a Researcher's independent check on the host, 2026-11-12, for all three corpora (research B1):
  - source query hashes match;
  - split counts match;
  - no normalized-text group crosses splits;
  - no kept query matches the primary official test's normalized text.
- [X] T003 Wrap the existing artifacts/p3-query-split-v1/manifest.json in provenance runs without changing any id: `p3-query-split-v1/provenance/<corpus>/` with `verification.json`. Done for SciFact `20261112T213306Z-725c0b8a`, NFCorpus `20261113T033930Z-90c4a87f` and MS MARCO `20261114T025422Z-8c5b643c`.
- [X] T004 [P] Tests in tests/unit/data/test_query_split.py: hash mismatch refused, overlapping splits refused, NFKC/casefold/whitespace normalization on hand cases. Remainder → T058.
- [X] T005 Implement `load_query_split(corpus, split)` and `verify_against_source` in src/hyperdime/data/query_split.py
- [X] T012 [P] Tests in tests/unit/evaluation/test_benchmark.py: k = 1024 equals full, floor deciles, lower-row ties, measures on a hand ranking, streamed scorer equal to the resident one
- [X] T014a [P] Tests in tests/unit/data/test_query_split.py: hash ordering `SHA-256("p3-bench-v1|20261112|<corpus>|<split>|<id>")` gives nested prefixes. Remainder (size-selection tests) → T051.
- [X] T043 CRITICAL: Record the code-snapshot commit in every P3-bench run manifest. Today the manifests say `git_commit: "unknown"`, and snapshots `6ce68a0`, `e50f364`, `3c1775a`, `e7827da`, `5347538` and `42bf508` appear only in the hand-written reports/P3-bench/RUNS.md.
  - In src/hyperdime/cli/p3_bench.py and src/hyperdime/cli/p3_bench_arms.py, accept the snapshot commit (for example from an environment variable set by the host runner) and pass it to `start_manifest`.
  - Add scripts/checks/p3_bench_snapshot_map.py, which writes reports/P3-bench/snapshots.json (run id → snapshot commit) from the bundles in artifacts/p3-bench/code-snapshots/ and the run manifests' timestamps.
  - Per Constitution III.
  Done: `HYPERDIME_CODE_COMMIT` fallback in `contracts/manifests.py` (test in tests/unit/contracts/test_manifests.py), plus `scripts/checks/p3_bench_snapshot_map.py` → reports/P3-bench/snapshots.json. Two runs are flagged `unverified` until the host's `5347538`/`42bf508` bundles are copied locally.
- [X] T058 [P] Add a test in tests/unit/data/test_query_split.py that `load_query_split` never returns or reads official test ids or test labels: a manifest whose `test` entries are poisoned still loads, and the poisoned values never appear. Per FR-002.
  Done: the loader test is in tests/unit/data/test_query_split.py, and the never-opened check covering every stage is in tests/integration/test_p3_bench.py.

**Checkpoint**: split and evaluator verified. US1 can start.

---

## Phase 3: User Story 0: legacy P3 evidence stays as recorded (Priority: P1)

**Goal**: every completed exploratory P3 stage can be found and is marked exploratory, separate
from this benchmark.

**Independent Test**: research B0 lists every P3 report directory; docs/experiment_registry.md has
one row per stage and a row for P3-bench; no legacy report is regenerated.

- [X] T001 [US0] Record the completed P3 stages, runs, snapshots, reports, ADRs, and what carries forward in specs/013-p3-selector-benchmark/research.md B0
- [X] T002 [US0] Link this feature from docs/experiment_registry.md and specs/002-hyperdime-roadmap/plan.md, and add the missing P3 score-matching row

**Checkpoint**: US0 is complete.

---

## Phase 4: User Story 1: valid baselines on one split (Priority: P1) 🎯 MVP

**Goal**: per corpus, the frozen full vector and one `l2s-official` (authors' code at `55f010f5`,
fit only, seed 42, published τ and epochs, final epoch), with the split fingerprint and the
authors' commit on every table.

**Independent Test**: the following, plus the refusals in contracts/cli.md:
- `small_test`-style parity is bit for bit (tests/unit/baselines/test_l2s_official.py);
- every arm at 1024 gives the full vector's ranks;
- the split check refuses a mismatch;
- MS MARCO reports carry the transfer flag.

### Tests for User Story 1

- [X] T010 [P] [US1] Tests in tests/unit/baselines/test_l2s_official.py. On a fixture with length ties, graded positives, a judged query without positives, and pool-fallback queries, the adapter and the authors' unmodified script, in separate seed-42 CPU processes, give bit-identical weights. Fallback is counted (4 of 23). Recorded pools give the same run without a second corpus copy.
- [X] T051 [P] [US1] Add tests for `large_resource_plan` and the cap rule of `cmd_freeze_resources` (src/hyperdime/cli/p3_bench.py) to tests/unit/evaluation/test_p3_bench_arms.py, using synthetic unit costs. Check:
  - sizes in ADR-0018's priority order: n_eval at its top value, then the largest n_rg that fits, then n_tune at its minimum;
  - N chosen larger first;
  - the smallest cap with ≥ 95% of searches stopping;
  - "escalate" below minimum coverage;
  - `realized_without_rank_guided` is recorded.
  Per FR-010a.

### Implementation for User Story 1

- [X] T011 [US1] Implement src/hyperdime/baselines/l2s_official.py:
  - commit check;
  - `authors_order`, which sorts by length;
  - a subprocess per corpus;
  - negative rebuild that must reproduce the authors' targets;
  - writes `state_dict.pt`, `targets.pt`, `negatives.npy`, `fallback.npy`, `valid.json` and `rng_after_labels.pt`.
- [X] T013 [US1] Official-evaluator parity as the `p3-bench official-parity` command in src/hyperdime/cli/p3_bench.py, instead of the planned scripts/checks/p3_bench_l2s_parity.py.
  - SciFact `20261112T214641Z-598e8e33`: equal per query.
  - NFCorpus `20261114T022735Z-b05d6c60`: parity at 1024 holds; 250 of 251 differences explained by scripts/checks/p3_bench_parity_explain.py.
  - Not feasible on MS MARCO.
- [X] T014 [US1] Fit-only cost probe per corpus (`p3-bench official --purpose probe`, then `p3-bench probe`). The realized sizes are in reports/P3-bench/RUNS.md. Grid deviations (NFCorpus and MS MARCO label sizes, MS MARCO search share 0.5) are recorded there.
- [X] T020 [US1] Resource gate per corpus with `p3-bench freeze-resources`:
  - SciFact passed;
  - NFCorpus escalated, and `p3-bench escalate --option remove-rank-guided` removed rank-guided and the global mask;
  - MS MARCO passed with n_eval 7,500, n_rg 4,000, n_tune 2,000, cap 8, shortlist 1,000.
  Records are in reports/P3-bench/freeze/.
- [X] T015 [US1] Train the single `l2s-official` per corpus on fit:
  - SciFact `20261112T214624Z-7ec05061`;
  - NFCorpus `20261114T022621Z-669b02f6`;
  - MS MARCO `20261114T175430Z-e875cb4f`, CPU, on the 4,000-id prefix.
- [X] T046 [US1] In the generated reports in src/hyperdime/cli/p3_bench_arms.py (`_report` for `evaluate`, `_family_report` for `family`), print under the title of every table:
  - the split fingerprint per split, from `load_query_split(...).fingerprints`;
  - the split manifest SHA-256;
  - for L2S-based arms, the authors' commit `L2S_COMMIT`.
  Store the same values in `summary.json`. Per FR-011.
- [X] T048 [US1] For MS MARCO, write `"transfer_flag"` in the `l2s-official` run summary (src/hyperdime/cli/p3_bench.py, `cmd_official`) and print it in the generated evaluation and family reports: "τ = 0.10 and 50 epochs were published for the authors' 50k-passage subset and are applied unchanged". Per US1/AC4.

**Checkpoint**: US1 is done when T046 and T048 are; the baselines themselves are complete.

---

## Phase 5: User Story 2: teacher masks on fit queries (Priority: P2)

**Goal**: positive, contrastive and rank-guided masks on fit queries only, with exact values,
nesting across budgets, cost, and demotions of unjudged documents versus known negatives.

**Independent Test**:
- on a synthetic corpus small enough to enumerate every subset, the search reaches the brute-force
  optimum (D ≤ 12, k ≤ 6);
- |mask| = k and no accepted swap lowers the objective;
- the result is never below either start;
- no tune or evaluation id is read.

### Tests for User Story 2

- [X] T021 [P] [US2] Tests in tests/unit/evaluation/test_teachers.py:
  - top-k ties go to the lower index;
  - positive and contrastive masks by hand;
  - GDR by hand;
  - brute-force optimum and never below the start;
  - the cap;
  - a shortlist covering the corpus equals the exact search;
  - a short shortlist reports exact values.
  Remainder → T055.
- [X] T055 [P] [US2] Add to tests/unit/evaluation/test_teachers.py:
  - the contrastive mask uses the recorded negatives' mean (`n̄` from `negatives.npy` rows);
  - when both searches reach the same objective, the result comes from the positive start;
  - a teachers run given a split whose tune and evaluation ids are poisoned produces identical masks.
  Per US2.
  Done: the strict-tie rule is in tests/unit/evaluation/test_teachers.py. Recorded negatives and the poisoned-split check are in tests/integration/test_p3_bench.py.

### Implementation for User Story 2

- [X] T022 [US2] Implement src/hyperdime/evaluation/teachers.py (`rank_guided_search`, `shortlist_search`, `gdr_weights`, `Objective`) and `p3-bench teachers` with resumable `Checkpoint` blocks. Runs:
  - SciFact `20261112T223656Z-07608c76`;
  - NFCorpus `20261114T022810Z-24f79ec6`, positive and contrastive only;
  - MS MARCO `20261114T183122Z-72b29180`, 4,000 queries in 2 segments.
- [X] T023 [US2] Write nesting, demotions (unjudged versus known negatives), searches improved or at the cap, and seconds into each teachers `summary.json`. Remainder (a generated report) → T052.
- [X] T052 [US2] Add src/hyperdime/reports/p3_bench.py with `teacher_report(summary: dict, corpus: str, run_id: str) -> str`, and a `p3-bench teacher-report --corpus --teachers-run` command that writes `README.md` next to the teachers `summary.json`. It reports:
  - nesting per consecutive budget pair;
  - objective gain against seconds;
  - searches improved and at the cap;
  - demotions split into unjudged documents and known negatives.
  Generate it from the files only. Per FR-007.

**Checkpoint**: US2 is done when T052 and T055 are.

---

## Phase 6: User Story 3: query-only selectors against L2S (Priority: P2)

**Goal**: the primary teacher arms against `l2s-official` on the primary endpoint. Arms differ only
in the target `t_q = (1/9) Σ_k m_{q,k}/k`. Controls are S1-scratch, S1-official-init, full, global
mask and `variance`.

**Independent Test**:
- fed `l2s-official`'s own targets, the teacher pipeline reproduces its weights exactly;
- replacing tune and evaluation labels leaves every teacher arm's weights unchanged;
- the report regenerates byte-identically from `summary.json`.

### Tests for User Story 3

- [X] T030 [P] [US3] Tests in tests/unit/evaluation/test_p3_bench_arms.py and tests/unit/evaluation/test_benchmark_stats.py cover:
  - the target `t_q` and nesting on hand cases;
  - global-mask ties to the lower dimension;
  - official and S1 checkpoints mapped to one linear layer;
  - escalation records;
  - the search lower bound;
  - family Holm and Bonferroni;
  - Holm monotonicity;
  - untrained arms skipped;
  - large and direct scoring equal;
  - checkpoint round trip, gaps and configurations;
  - the exact sign-flip path, the `(+1)/(B+1)` p-value, independent streams, bootstrap and claim levels.
  Remainder → T049, T050, T057.
- [X] T049 [P] [US3] Add a leakage test to tests/unit/evaluation/test_p3_bench_arms.py. Run teachers → `train-arm` on a tiny CPU fixture twice, the second time with every tune and evaluation qrel replaced by random labels. Teacher masks, `_target` outputs and teacher-arm `state_dict` bytes must be identical. Per SC-003 and Constitution II.
  Done in tests/integration/test_p3_bench.py rather than the unit file, because it needs the full F2-shaped fixture. The official reference is also compared byte for byte.
- [X] T050 [P] [US3] Add a CPU unit test of the replay control to tests/unit/baselines/test_l2s_official.py: on the T010 fixture, `train-arm --target official --seed 42` gives weights bit-identical to the adapter's `state_dict.pt` and `replay_equals_official: true`. Per FR-008.
  Done in tests/integration/test_p3_bench.py (same fixture as T049).
- [X] T053 [P] [US3] Add a test to tests/unit/evaluation/test_p3_bench_arms.py: `_report(summary, corpus, run_id)` run twice on a fixture `summary.json` round-tripped through JSON gives byte-identical text, and a SciFact-shaped summary yields every table. Per SC-004.
- [X] T057 [P] [US3] Cover the remaining S1 and statistics cases in tests/unit/evaluation/test_p3_bench_arms.py and tests/unit/evaluation/test_benchmark_stats.py, or cite the existing tests in tests/unit/training/ that already do (list them in the docstring):
  - cell order and ties;
  - step 0 eligible for selection;
  - the soft-mass tune guard ("no S1 cell passes the soft-mass guard" refusal);
  - bootstrap and sign-flip batches ≤ 10^7 elements;
  - teacher arms and `l2s-official` have equal parameter counts.
  Per T030.
  Done. `select_cell` and `S1_STEPS`, `S1_EVAL_EVERY`, `S1_GUARD` were extracted in `cli/p3_bench_arms.py`, with no change in behavior.

### Implementation for User Story 3

- [X] T031 [US3] SciFact: replay control `20261112T223653Z-bfc81ad7`, equal bit for bit; three teacher arms at seeds 42, 0, 1 and 2 (12 runs in reports/P3-bench/freeze/runs-scifact.json)
- [X] T032 [US3] SciFact controls:
  - S1-scratch `20261112T230604Z-220da458` and S1-official-init `20261112T231423Z-833ccb08` (8 cells × 21 checkpoints chosen on tune, seeds 0–2);
  - global mask;
  - `variance`;
  - `full`.
- [X] T033a [US3] SciFact evaluation lock: reports/P3-bench/freeze/evaluation-lock-scifact.json (SHA-256 `bc4aacca…`), 22 arm files and 121 evaluation ids
- [X] T033 [US3] SciFact evaluation `20261112T232241Z-fbc6f7b5` in one pass, with the README generated from `summary.json`. Remainder (missing report fields) → T044, T045, T047, T054.
- [X] T044 [US3] Add a cost section to `_report` in src/hyperdime/cli/p3_bench_arms.py, filled from the runs listed in the lock record. Write the same values into `summary.json` under `"cost"`. Per FR-012 and Constitution V.
  - Offline table: label generation (from the `l2s-official` summary `seconds`), teachers (`seconds`, `searches`), training per arm (each arm summary's `seconds`, S1 totals).
  - Online table: per-query selector forward time plus masked scoring, timed in `evaluate` and averaged over evaluation queries.
  Done in the new `p3-bench report` command (`reports/p3_bench.py`: `arm_costs`, `cost_lines`). Online scoring time comes from the fit-only probe; the selector forward pass is timed at report time on synthetic input.
- [X] T045 [US3] Add to `_report` one row per arm: training compute (seconds and device) and the number of selection candidates (1 for `l2s-official` and the teacher arms; `selection_candidates` from S1 `selection.json`). Add a fixed sentence stating S1's larger selection budget (8 cells × 21 checkpoints) as a limitation of the secondary comparison. Per FR-009.
- [X] T047 [US3] Add an `mrl-prefix` control arm to `cmd_evaluate` and `_evaluate_large` in src/hyperdime/cli/p3_bench_arms.py: at each decile k, keep dimensions `0 … k−1`. Include it in `per_query.csv`, `per_decile_ndcg10` and the secondary table. Per Constitution V ("every table includes full-1024 and MRL prefix-k comparisons").
  - It needs no training and no lock entry, because it is deterministic.
  - The lock record's `conventions` must name it, so already-locked runs remain valid: only new evaluations include it.
  - For SciFact and NFCorpus, which are already evaluated, adding the prefix arm would read their evaluation labels a second time. ADR-0018 evaluates each corpus once, so do not do that without a recorded researcher decision (an ADR-0018 amendment). Without one, report "prefix-k not evaluated (added after the lock)" for those two corpora.
  Done for future evaluations (MS MARCO onward). SciFact and NFCorpus reports state that prefix-k was not evaluated (`PREFIX_MISSING`). A `mrl-prefix` stream key was added in `benchmark_stats.ARMS`, with existing keys unchanged.
- [X] T054 [US3] Add the fixed sentence to `_report` and `_family_report`: "A query-disjoint in-domain comparison against Learning2Select retrained under this protocol; not a comparison with published numbers and not a state-of-the-art claim." Per FR-013.
- [X] T060 [US3] Once T044–T047, T052 and T054 are merged, regenerate the SciFact and NFCorpus READMEs from their existing `summary.json` and run files. Add a `--regenerate RUN` option to `p3-bench evaluate` that rewrites only `README.md` and never reads evaluation labels again. Copy the results to reports/P3-bench/runs/.
  Done as a separate `p3-bench report` command instead of `evaluate --regenerate`, so the evaluation run is never modified. Runs: SciFact `20261124T221548Z-94aa88a5`, NFCorpus `20261124T221551Z-f8313fcf`; teacher reports `20261124T221549Z-efd5834a`, `20261124T221552Z-4491ef19` (reports/P3-bench/RUNS.md).

**Checkpoint**: US3 is done for SciFact when T044–T060 are; the experiment itself is complete.

---

## Phase 7: User Story 4: confirmation order (Priority: P3)

**Goal**: NFCorpus and bounded MS MARCO under unchanged rules, then Holm across the family of three
corpora.

**Independent Test**:
- each corpus has its own evaluation-lock record written before its labels are read;
- `p3-bench family` produces Holm and Bonferroni at m = 8 from the `per_query.csv` files only.

- [X] T040 [US4] NFCorpus under the frozen rules: reference, replay control `20261114T022751Z-0f595cc9`, teachers, 8 teacher arms, S1 × 2, lock reports/P3-bench/freeze/evaluation-lock-nfcorpus.json (SHA-256 `e017ac55…`), evaluation `20261114T180408Z-f16e9038`, in two segments
- [ ] T041 [US4] MS MARCO at the realized sizes, all primary arms on the 4,000-id `n_rg` prefix. Run on the host with `msmarco_resume.sh`.
  - Done: reference, replay control `20261114T182941Z-ff228406`, teachers, 12 teacher arms.
  - Still to run: S1-scratch (started as `20261116T014506Z-b51f2627`), S1-official-init, reports/P3-bench/freeze/runs-msmarco.json, `evaluation-lock --family reports/P3-bench/freeze/family.json`, then `evaluate --max-hours 11` repeated until exit 0.
  - Copy the manifests and summaries to reports/P3-bench/runs/ and update reports/P3-bench/RUNS.md.
  - Do T044–T048 and T054 first if possible, so the MS MARCO report carries them from the start.
- [ ] T041a [US4] Run `p3-bench family --family reports/P3-bench/freeze/family.json --evaluations <{scifact, nfcorpus, msmarco: run}>`. Then write the replication statement under ADR-0018 into reports/P3-bench/RUNS.md: the same teacher arm superior in at least 2 eligible corpora, or in both NFCorpus and MS MARCO if SciFact is development data; otherwise corpus-specific evidence only.
- [ ] T042 [US4] Update docs/experiment_registry.md (the P3-bench row) from the result files. Label SciFact as development data if anything changed after its evaluation (no protocol change so far). Propose the follow-up specs: budget-conditioned arms, geometry, capacity, and the hypernetwork on held-out corpora.

**Checkpoint**: F12 is complete.

---

## Phase 8: Polish and cross-cutting concerns

- [X] T056 [P] Add tests/integration/test_p3_bench.py. On a tiny CPU fixture (about 40 documents, D = 16, 3 splits, a fake authors' clone such as the T010 fixture), run `split-provenance → freeze-protocol → official → probe → freeze-resources → official (reference) → teachers → train-arm → train-s1 → evaluation-lock → evaluate → family` through `hyperdime.cli.p3_bench.main`. Assert exit codes, written files, and parity at 1024.
  Done. Each official call resets the adapter's once-per-process guard, standing in for a fresh process. S1 runs a 30-step schedule with the guard off. Runtime is about 2 min.
- [X] T059 Update the Status line of specs/013-p3-selector-benchmark/spec.md: implemented and run for SciFact and NFCorpus, MS MARCO in progress. Add an amendment note like plan.md's, stating it is a documentation change with the frozen version in snapshot `6ce68a0`. Move docs/decisions/ADR-0018-p3-selector-benchmark.md from "proposed" to "accepted" once T041a is done.
- [ ] T061 Before committing, run `ruff check . && ruff format --check . && mypy && pytest` in the repository root (constitution workflow).
- [ ] T062 Commit the F12 code, tests, specs and reports/P3-bench on their own branch `agent/<ticket>-p3-selector-benchmark` from master, separate from the other P3 studies. They are uncommitted on `agent/unticketed-representation-analysis`. Fill every section of the PR template, and mark the PR `protocol-change: no`.

---

## Dependencies and execution order

```text
Setup (T002a–c, T019) → Foundational (T003a, T003, T004, T005, T012, T014a; T043, T058 open)
  → US0 (done)
  → US1 (T010, T011, T013, T014, T020, T015 done; T046, T048, T051 open)
      → US2 (T021, T022, T023 done; T052, T055 open)
          → US3 (T030–T033 done; T044–T047, T049, T050, T053, T054, T057 → T060)
              → US4 (T040 done; T041 → T041a → T042)
  → Polish (T056, T059, T061 → T062)
```

- **Before the MS MARCO evaluation (T041) and the family run (T041a)**: T043, T044, T045, T046,
  T047, T048 and T054, so that new reports carry every field the spec requires.
- **Tests that can be written at any time**: T049, T050, T051, T053, T055, T056, T057 and T058.
  They don't depend on the host runs.
- **Before T062**: T061.

## Parallel examples

```text
# US1 + foundational tests, together:
T051 (tests/unit/evaluation/test_p3_bench_arms.py — resource plan)
T058 (tests/unit/data/test_query_split.py)

# US3 tests, together:
T049 (leakage), T050 (replay unit test, tests/unit/baselines/test_l2s_official.py), T053 (regeneration)

# Report fields: same file (src/hyperdime/cli/p3_bench_arms.py), so in sequence:
T046 → T054 → T045 → T044 → T047
```

## Implementation strategy

1. **Correctness of what's already reported** (MVP for this round): T043, T046, T048, T054. These
   are small edits that make every existing and future report meet FR-011, FR-013 and
   Constitution III.
2. **Report completeness**: T044, T045, T047, T052, then T060 to regenerate SciFact and NFCorpus
   from their files.
3. **Test debt**: T049–T051, T053, T055–T058, all CPU and offline, in parallel.
4. **Finish the experiment**: T041 on the host, then T041a, T042 and T059.
5. **Ship**: T061, then T062.
