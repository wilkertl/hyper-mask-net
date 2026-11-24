# Implementation Plan: F12 P3 Selector Benchmark

**Branch**: `agent/F12-p3-selector-benchmark` (placeholder, no ticket; the code currently sits
uncommitted on `agent/unticketed-representation-analysis`) | **Date**: 2026-11-12, as built
2026-11-24 | **Spec**: [spec.md](spec.md) | **Research**: [research.md](research.md) |
**Protocol**: [ADR-0018](../../docs/decisions/ADR-0018-p3-selector-benchmark.md) (proposed) |
**Runs**: [reports/P3-bench/RUNS.md](../../reports/P3-bench/RUNS.md)

> **Amendment 2026-11-24, documentation only.** This plan was rewritten to describe the code that
> was actually built and run. No rule in ADR-0018 or the spec changed. The version fingerprinted at
> the protocol gate (`reports/P3-bench/freeze/protocol.json`, SHA-256 `ced73f2d…`) is recoverable
> from code snapshot `6ce68a0` (`artifacts/p3-bench/code-snapshots/`). Only the `probe` command
> re-checks those hashes, and every corpus has finished probing, so this edit doesn't block any
> remaining run.

## Summary

Turn the exploratory P3 findings into a query-disjoint in-domain benchmark.

- **Split**: one frozen split per corpus, `p3-query-split-v1`. F2 train and validation queries are
  pooled into fit, tune and evaluation, grouped by normalized text, with any overlap with the
  official test split removed. The ids are kept, and provenance runs wrap the manifest.
- **Baselines**: the frozen full vector, and exactly one `l2s-official` per corpus. That is the
  authors' Learning2Select at `55f010f5`, imported unmodified through an adapter. It is trained on
  the primary arms' fit ids with seed 42, the published τ and epochs, and final-epoch weights. Its
  sampled negatives and the RNG state after label generation are recorded.
- **Stage 2**: three teachers on fit queries: positive, contrastive (using the recorded
  negatives), and rank-guided. Rank-guided is a strict lexicographic `(nDCG@10, GDR)` swap search
  started from both of the first two. MS MARCO uses a shortlist search with exact acceptance.
- **Stage 3**: teacher arms identical to `l2s-official` except for their target
  `t_q = (1/9) Σ_k m_{q,k}/k`. They are compared with it on the primary endpoint, next to two S1
  controls (from scratch, and initialized from the official checkpoint), the frozen full vector, a
  global mask and `variance`. Holm correction is applied across all three corpora.
- **Order and status** (as of 2026-11-24):
  - SciFact and NFCorpus are evaluated. NFCorpus escalated at the resource gate, and the
    rank-guided arm and the global mask were removed from it.
  - MS MARCO passed its resource gate. Its reference, replay check, teachers and 12 teacher arms
    are done; S1, the evaluation lock, the evaluation and the family statistics are still running
    or pending.

## Technical Context

**Language/Version**: Python ≥ 3.11 (the host venv runs 3.12 with torch 2.11.0+cu128).

**Primary Dependencies**: torch, numpy, and the project's own modules:
- `contracts.manifests`: `ArtifactWriter`, `start_manifest`, `read_manifest`.
- `contracts.hashing`: `file_sha256`.
- `selection.topk.resolve_k`.
- `baselines.learning_to_select.BUDGETS`.
- `training.retrieval_trainer`, used for S1.
- `training.examples.retrieval_examples`.

The authors' clone at `55f010f5` is imported read-only and runs in a separate process per corpus.

The original plan expected to reuse `retrieval.exact`, `evaluation.representation_ranking`,
`evaluation.oracle_comparison` and `evaluation.statistics`. The built code has its own versions:
- `evaluation.benchmark` holds a chunked exact scorer that can stream from host memory.
- `evaluation.benchmark_stats` holds ADR-0018's statistics: per-procedure streams from
  `SeedSequence(20261112)` and batched sign-flip and bootstrap draws.

See [research B11](research.md#b11-as-built-implementation-decisions).

**Storage**: files under the artifacts root, and only versioned summaries in the repository.
- `artifacts/p3-query-split-v1/manifest.json`, plus provenance runs under
  `p3-query-split-v1/provenance/<corpus>/`.
- `artifacts/selectors/{l2s-official,p3-teacher-arms,p3-s1/<arm>}/<corpus>/<run>/`.
- `artifacts/evaluation/P3-bench/{probe-official,probe,official-parity,teachers,evaluation,family}/…`.
- `artifacts/scratch/p3-bench/<corpus>/<stage>-<config hash>/`: resumable query blocks.
- `reports/P3-bench/`:
  - `freeze/*.json`: the gate records;
  - `runs/…`: copies of manifests and summaries;
  - `RUNS.md`: the run record.

See [data-model.md](data-model.md).

**Testing**: pytest on CPU, offline. The current P3-bench unit tests pass: 48 passed, 1 GPU test
skipped. They use hand-computable fixtures and a synthetic `small_test`-style fixture to check
bit-for-bit parity with the authors' script.

**Target Platform**: Linux. Runs use the shared RTX 4090 hosts (`ceia40901`, and `ceia40902` for
the second-host check) under the project's free-memory rule: ≥ 20 GB free and no other compute
process, checked before each GPU launch.

**Project Type**: library plus CLI. The interface is the `p3-bench` subcommand of
`python -m hyperdime.cli`; see [contracts/cli.md](contracts/cli.md).

**Performance Goals**: fit within ADR-0018's per-corpus budgets (SciFact 8 h, NFCorpus 16 h,
MS MARCO 72 h), sized by the fit-only cost probe and a safety factor of 1.5.

**Constraints**:
- Scoring is exact float64 over the full corpus.
- Every job has a 12 h cap. Long stages stop at a query-block boundary after `--max-hours`, exit
  with code 75, and resume with the same command.
- MS MARCO's anonymous memory must stay within 64 GB. The corpus is memory-mapped copy-on-write.

**Scale/Scope**: realized query counts, from `freeze/resources-*.json`.

| Corpus | Documents | Fit | n_rg | Tune | Evaluation | Search |
|---|---|---|---|---|---|---|
| SciFact | 5,183 | 565 | 565 | 121 | 121 | exact, cap 8 |
| NFCorpus | 3,633 | 1,810 | removed | 388 | 388 | none (escalated) |
| MS MARCO | 8,841,823 | 4,000 | 4,000 | 2,000 | 7,500 | shortlist 1,000, cap 8 |

## Constitution Check

The check is re-evaluated against the built code. The ⚠️ rows point to open tasks in
[tasks.md](tasks.md).

| Principle | Status | Note |
|---|---|---|
| I. Frozen coordinates | ✅ | F2 float16 vectors only. The masked query is the full query with zeros outside its dimensions and is not renormalized. |
| II. Leakage-free | ✅ | Enforced in code: the test split is never loaded for scoring, evaluation labels are read only after the lock record's hashes are re-verified, and S1 selection reads tune labels only. Tested: replacing tune, evaluation and test labels leaves the reference, teachers and a teacher arm byte-identical, and no stage opens `qrels-test.jsonl` (tests/integration/test_p3_bench.py). |
| III. Provenance | ✅ with a recorded gap | New manifests record the code commit (`HYPERDIME_CODE_COMMIT` outside a git checkout). Past runs are mapped to snapshots in `reports/P3-bench/snapshots.json`; two runs stay `unverified` until the host's `5347538`/`42bf508` bundles are copied. |
| IV. Evidence-gated complexity | ✅ | Teacher arms add no capacity over L2S. Budget-conditioned variants are deferred. |
| V. Exact, generated measurement | ✅ with a recorded gap | Exact float64; tables generated from files; offline and online cost in `p3-bench report`. `mrl-prefix` is evaluated for MS MARCO onward; SciFact and NFCorpus lack it because adding it would read their evaluation labels a second time, and their reports say so. |
| VI. Tested math | ✅ | GDR, brute-force search, 100% parity, statistics, the resource plan, S1 selection, byte-identical regeneration, the replay control, and the end-to-end pipeline are tested. |
| VII. Replaceable modules | ✅ | Split, adapter, scorer, teachers, statistics and CLI are separate modules. |

**Gate result**: no violation without a justification or a remediation task. The ⚠️ rows don't
affect runs already made; they are gaps in reporting and tests.

## Project Structure

### Documentation (this feature)

```text
specs/013-p3-selector-benchmark/
├── spec.md          frozen at the protocol gate
├── plan.md          this file (as built)
├── research.md      B0–B10 frozen; B11 as-built decisions
├── data-model.md    artifacts, records, and entities as written by the code
├── quickstart.md    validation and run sequence
├── contracts/cli.md the p3-bench command contract
└── tasks.md         regenerated as built: 27 done, 23 open (T043–T062 and the US4 tail)
```

### Source Code (repository root)

```text
src/hyperdime/
  data/query_split.py           normalize, id_fingerprint, hash_order, QuerySplit,
                                load_query_split (hash/sort/disjoint/count checks),
                                verify_against_source
  baselines/l2s_official.py     L2S_COMMIT, PUBLISHED (τ, epochs), SEED; import_official
                                (checks the clone's commit), authors_order, run_official,
                                negative rebuild from recorded pools, subprocess entry point
  evaluation/benchmark.py       DECILES, AUTHORS_GRID, PRIMARY; positive_ranks (lower-row ties),
                                ideal_dcg, query_measures (nDCG@10, MRR@10, R@100, nDCG@100);
                                CorpusScorer (resident or streamed, chunked exact float64, top-N)
  evaluation/benchmark_stats.py stream(procedure, corpus, arm), sign_flip_p (exact or Monte
                                Carlo), bootstrap_means, intervals, holm, claim_level
  evaluation/teachers.py        top_k_dims, positive and contrastive masks, gdr_weights,
                                Objective, rank_guided_search, shortlist_search
  cli/p3_bench.py               split-provenance, freeze-protocol, official, probe (small and
                                large paths), freeze-resources (large_resource_plan),
                                official-parity
  cli/p3_bench_arms.py          teachers, train-arm, train-s1, evaluation-lock, evaluate,
                                escalate, family; Checkpoint (resumable blocks), _target,
                                _nesting, _statistics, _report, family_table
  reports/p3_bench.py           FRAMING, evaluation_provenance, arm_costs, cost_lines,
                                teacher_report, search_gains (used by report, teacher-report)
  contracts/manifests.py        HYPERDIME_CODE_COMMIT fallback for exported snapshots
  cli/__main__.py               registers p3-bench
scripts/checks/p3_bench_parity_explain.py   explains NFCorpus parity differences (diagnostic)
scripts/checks/p3_bench_snapshot_map.py      run id → code snapshot (reports/P3-bench/snapshots.json)
tests/unit/data/test_query_split.py
tests/unit/baselines/test_l2s_official.py
tests/unit/evaluation/test_benchmark.py
tests/unit/evaluation/test_benchmark_stats.py
tests/unit/evaluation/test_teachers.py
tests/unit/evaluation/test_p3_bench_arms.py
tests/unit/reports/test_p3_bench_report.py
tests/integration/test_p3_bench.py   every command on a tiny CPU corpus; replay, leakage, no test labels
```

**Structure Decision**: a single project.
- Evaluation and family tables are rendered in `cli/p3_bench_arms.py` (`_report`, `_family_report`).
- Cost, provenance and teacher reports are rendered in `reports/p3_bench.py`, used by the `report`
  and `teacher-report` commands.
- The planned `scripts/checks/p3_bench_l2s_parity.py` became the `official-parity` command.

## Stages and dependencies (as built)

| Stage | Command | Output | Gate to next | Status |
|---|---|---|---|---|
| 0 Split provenance | `split-provenance` | `verification.json` per corpus | SC-001 | done ×3 |
| 0b Protocol gate | `freeze-protocol` | `freeze/protocol.json` | before the probe | done |
| 1 Label probe | `official --purpose probe [--labels-only] --limit` | probe labels and timings | — | done ×3 |
| 1b Cost probe | `probe` | `probe.json` (fit only; re-checks the protocol hashes) | — | done ×3 |
| 1c Resource gate | `freeze-resources` (+ `escalate` on overrun) | `freeze/resources-<c>.json`, `escalation-<c>.json` | before any tune metric, any teacher beyond the probe, or any evaluation label | done ×3 (NFCorpus escalated) |
| 1d Reference | `official --purpose reference` | `l2s-official` weights, targets, negatives, fallback, RNG | SC-002 | done ×3 |
| 1e Parity | `official-parity` | per-query comparison on the authors' grid (diagnostic) | — | SciFact, NFCorpus; not feasible on MS MARCO |
| 2 Teachers | `teachers [--max-hours]` | masks, `searches.json`, nesting, demotions, `global_mask.npy` | brute-force tests | done ×3 |
| 3 Arms | `train-arm --target official\|<teacher> --seed` | the replay control (`official`), then teacher arms at seeds 42 and 0–2 | replay equals official | done ×3 |
| 3a S1 | `train-s1 --arm scratch\|official-init` | 8 cells × 21 checkpoints, chosen on tune; seeds 0–2 | — | SciFact, NFCorpus; MS MARCO running |
| 3b Evaluation lock | `evaluation-lock --runs --family` | `freeze/evaluation-lock-<c>.json` | before evaluation labels | SciFact, NFCorpus |
| 3c Evaluation | `evaluate [--max-hours]` | `summary.json`, `per_query.csv`, README | SC-003, SC-004 | SciFact, NFCorpus |
| 4 Family | `family` | Holm and Bonferroni at m = 8 over all three corpora; `family.json`, `family.csv` | SC-005 | pending MS MARCO |
| 4b Reports | `report`, `teacher-report` | cost, provenance and teacher reports from run files; no judgments | — | SciFact, NFCorpus; MS MARCO after its evaluation |
| Later | — | budget-conditioned arms, geometry, capacity, hypernetwork on held-out corpora | — | separate specs |

Full-vector and `l2s-official` evaluation rows were not read for any design choice; they appear
only with the candidates.

## Risks (updated)

- **Report fields before the final family report.** Some FR-009, FR-011, FR-012, FR-013 and US1/AC4
  fields are missing from the generated reports. The README is generated from `summary.json`, so
  adding those fields (T044–T048, T054) regenerates the existing reports without rerunning anything,
  provided the inputs are present in the run directories.
- **MS MARCO compute.** Projected at 62.2 h of the 69 h left after the probe. The S1 crash
  (streaming `rows_f64` given a GPU index) was fixed in local commit `42bf508` with a regression
  test; nothing partial was written.
- **Published MS MARCO τ and epochs** come from the authors' 50k-passage subset. They are applied
  unchanged, and the report must say so (T048).
- **Reference-side seed variability** isn't measured, because `l2s-official` has one seed.
- **Small SciFact evaluation (121 queries).** The protocol doesn't adapt to that after seeing
  results.

## Complexity Tracking

| Addition | Why needed | Simpler alternative rejected because |
|---|---|---|
| Rank-guided swap search | Positive-only masks are not ranking ceilings (ADR-0016, ADR-0017) | Positive or contrastive masks alone can't test that; bounded by caps and checked by brute force |
| `shortlist_search` (MS MARCO) | A full-corpus scan per swap doesn't fit the 72 h budget | A smaller n_rg loses fit coverage; acceptance stays exact, and a test shows a shortlist covering the corpus equals the exact search |
| Own `CorpusScorer` instead of `retrieval.exact` | Exact float64 ranks over 8.8M rows without a float64 corpus copy; streaming from host memory | `masked_scores` needs the full matrix in memory; equal bit for bit (tested) |
| `benchmark_stats` instead of `evaluation.statistics` | ADR-0018 needs per-procedure seeded streams, the exact path when 2^n ≤ B, and batched draws | The existing functions lack those contracts |
| Resumable `Checkpoint` blocks | MS MARCO teachers (~11 h) and evaluation (~24 h) exceed the 12 h job cap | Splitting by query ranges by hand risks gaps; blocks must tile without gaps and never resume another configuration (tested) |
| `escalate` and `family` commands | ADR-0018's escalation and multiplicity rules | A by-hand record isn't fingerprinted or reproducible |
