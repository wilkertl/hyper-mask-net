# Implementation Plan: HyperDIME-Qwen Project Roadmap

**Branch**: `docs/hyperdime-roadmap` | **Date**: 2026-09-25 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/002-hyperdime-roadmap/spec.md`

## Summary

**Hypothesis first**: a spike (F0, research R0) tests H1 on five small BEIR datasets (≈ 100k
documents) using the code already on `master` plus a thin loader, encoder, trainer, and
transfer-matrix analysis. It ends in an H1 verdict. Everything below is built only if the verdict is
"supported"; otherwise the project stops or pivots to a universal selector, having spent days rather
than months.

If H1 is supported, deliver the whole HyperDIME-Qwen study as seven gated features: F1 foundation (already planned in
`specs/001-m0-foundation`), F2 data and embeddings, F3 Learning-to-Select baseline (G1), F4 selector
shift (G2, Go/No-Go), F5 environment pool and space descriptor (G3), F6 hypernetwork prototype
(meta-validation), F7 paper-grade evaluation (G4, G5). Each feature is specified and planned with
spec-kit only after the previous gate is decided. The roadmap fixes the cross-cutting decisions
every feature relies on: an environment pool covering every dataset in the DIME-family and
Learning-to-Select experiments (MS MARCO with TREC DL'19/'20/DL-HARD, Robust04, 13 BEIR tasks —
research R19), with a proposed 7/3/4 meta-split plus a query-shift-only environment, the oracle
aggregation rule, quantitative gate criteria, the statistics toolkit, protocol enforcement by type,
the low-rank hypernetwork defaults, and generated reports ([research.md](research.md)).

## Technical Context

**Language/Version**: Python ≥ 3.11 (local 3.13, CI 3.12)

**Primary Dependencies**: torch, transformers (Qwen3-Embedding-0.6B, frozen), datasets (BEIR from
the Hugging Face Hub), ir_datasets (TREC DL query sets, DL-HARD, Robust04 over local licensed
files), numpy, scipy (statistics), scikit-learn (probes), pydantic, hydra-core,
omegaconf; faiss-cpu only for deployment analysis; all locked with `uv` (F1)

**Storage**: files under gitignored `artifacts/` (JSON/JSONL records, `.npy`/`.safetensors`
payloads, manifests); versioned `configs/`, `reports/`, `docs/decisions/`

**Testing**: pytest on CPU without network; unit tests with hand-computed values, property tests
(invariances, zero-delta equivalence, leakage guards), a 50-document / 10-query synthetic end-to-end
integration test (plan §8.3), regression fixtures with tolerances

**Target Platform**: one Linux GPU workstation for embedding and training; GitHub Actions CPU for CI

**Project Type**: Python research library + `hyperdime` CLI (single project, `src/` layout)

**Performance Goals**: embed ≈28.5M unique documents + ≈1.2M queries once (≈60 GB float16),
sharded and resumable, ≈9M of them before G2; online selector overhead reported per stage and kept
below the restricted-scoring saving (G5)

**Constraints**: frozen encoder, D = 1024; exact scoring for science; no test or meta-test labels
in any training, adaptation, negative-mining, descriptor, or normalization path; offline vs online
cost separated; no hand-edited numbers

**Scale/Scope**: 15 environments (13 BEIR + MS MARCO + Robust04) and 3 extra MS MARCO
evaluation query sets, ~15 new modules, ~15 CLI subcommands, 7 reports (E0–E6), 5 gates,
~30 backlog tickets

All unknowns are resolved in [research.md](research.md); the one spec clarification (environment
pool) was answered by the user.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Checked against [constitution v1.0.0](../../.specify/memory/constitution.md).

| Principle | Pre-research | Post-design | Evidence |
|---|---|---|---|
| I. Frozen Coordinate System | ✅ | ✅ | Embeddings computed once (F2) and reused; selection only masks queries; `LowRankSelector` acts on `e_q` only |
| II. Leakage-Free Protocols | ✅ | ✅ | Meta-train uses only datasets with *train* qrels (R3); `load_environment` hides test qrels by default; protocol-typed descriptor builders (R11); `DescriptorNormalizer.fit` and `meta_train` refuse non-meta-train input; per-domain ceiling on test-only datasets uses a disjoint query half and is never an input to the method |
| III. Contracts and Provenance First | ✅ | ✅ | F1 precedes everything; every CLI writes through `ArtifactWriter`; new entities in data-model carry schema versions |
| IV. Evidence-Gated Complexity | ✅ | ✅ | Features F4–F7 are specified only after the previous gate; quantitative gate criteria (R7, R8, R10); V2 encoder, contrast/redundancy groups, and adaptive-k are ablation-triggered; the 12 environments beyond SciFact, NFCorpus, and MS MARCO are embedded only after G2 |
| V. Exact, Generated Measurement | ✅ | ✅ | Exact scoring everywhere except E6 deployment tables; `hyperdime report` generates all tables deterministically; mandatory full-1024, MRL, and global-selector rows; randomization tests + Holm + bootstrap CIs (R9) |
| VI. Test-Verified Mathematics | ✅ | ✅ | Hand-computed tests for oracle aggregation, statistics, descriptor stats, low-rank reconstruction; property tests from plan §8.2 are acceptance criteria in the interface contract |
| VII. Independent, Replaceable Modules | ✅ | ✅ | `ImportanceModel` protocol unifies baselines and generated selectors; policies consume importance only; module boundaries fixed in [contracts/module-interfaces.md](contracts/module-interfaces.md) |
| Artifacts and Data Handling | ✅ | ✅ | Datasets pulled by `prepare-data` into `artifacts/`, pinned by revision; nothing large in Git |
| Workflow and Quality Gates | ✅ | ✅ | One `agent/` branch per ticket inside each feature; `exp/` branches commit only configs and generated reports; gate tags per `docs/workflow.md` |

**Result**: PASS, with one justified deviation for the F0 spike (see Complexity Tracking).

## Milestone Features and Sequencing

```text
F0 H1 spike ──H1 verdict──▶ (stop / pivot if not supported)
   │ supported
   ▼
F1 foundation ──▶ F2 data+embeddings ──▶ F3 L2S baseline ──G1──▶ F4 selector shift ──G2──▶
F5 env pool + descriptor ──G3──▶ F6 hypernet prototype ──meta-val──▶ F7 paper evaluation ──G4, G5
```

| Feature | Scope (backlog tickets) | Depends on | Exit | Key risks addressed |
|---|---|---|---|---|
| **F0** `h1-spike` | BEIR loader from the official UKP zips (train/dev/test qrels in one archive, SHA-256 recorded; the Hugging Face path stays the F2 plan, research R2), Qwen encoder via the `beir` project's vLLM server or local transformers + `.npy` cache, selector training loop, global selector, transfer matrix + mask analysis + stats, lightweight JSON manifests, generated report (R0) | code on `master` | **H1 verdict** | R1 small shift, found in days |
| **F1** `001-m0-foundation` | M0.1–M0.4 — **planned** | — | contracts, manifests, lock | pipeline differences mistaken for model differences |
| **F2** `data-embeddings` | M1.1 loaders (HF + `ir_datasets`, generic over `EnvironmentSpec`) for SciFact and NFCorpus, M1.2 full MS MARCO + TREC DL'19/'20/DL-HARD query sets (R2), M1.3 splits + leakage guards, M2.1 Qwen encoder, M2.2 sharded, resumable embedding cache, M1.4 negative pools (R5) | F1 | dataset manifests + cached embeddings for SciFact, NFCorpus, MS MARCO | test-label leakage; 8.8M-document embedding time |
| **F3** `l2s-baseline` | M12.1 exact evaluation pipeline (streamed over document shards; adds AP), M3.1 oracle targets with aggregation (R4), M3.2 selector trainer (L2S protocol, R6), M3.3 baselines, E0 report comparable with Learning to Select and the DIME papers' DL tables | F2 | **G1** (R7) | invalid baseline; R7 ANN effects |
| **F4** `selector-shift` | M12.3 statistics + generated tables (R9), M4.1 per-domain selectors × 3 seeds, M4.2 transfer matrix, M4.3 behavior analysis, G2 ADR | F3 | **G2** Go/No-Go (R8) | R1 small shift |
| **F5** `space-descriptor` | embeddings for the other 11 BEIR tasks and Robust04 (R19; Robust04 if licensed), environment-pool and meta-split ADRs (R3); M5.1–M5.3 groups A–E; M5.4 probes (R10); M5.5 freeze descriptor v1 | G2 = pass | **G3** | R2 dataset identity; R3 pseudo-feedback circularity |
| **F6** `hyperdime-prototype` | global selector (R13), M7.1 low-rank selector, M6.1 space encoder V1, M8.1 hyper head (R12), M9.1 episodic trainer (R14), M9.2 meta-validation | G3 = pass | match/beat global on meta-validation | R4 unstable parameters; R6 gain from supervision |
| **F7** `paper-evaluation` | M10.1 protocols P0/P1/P2, M11.1 policies (R15), E3 main result, E4 ablations (plan §7 matrix), E5 sensitivity, M12.2 efficiency + FAISS (R16) | F6 verdict ≠ lose | **G4**, **G5** | R5 MRL suffices; cost |

Each feature follows: `/speckit-specify` → `/speckit-clarify` (if needed) → `/speckit-plan` →
`/speckit-tasks` → implement on per-ticket `agent/` branches → `exp/` branch runs the experiment →
gate ADR, tag.

## Project Structure

### Documentation (this feature)

```text
specs/002-hyperdime-roadmap/
├── plan.md              # This file
├── research.md          # R1–R18 cross-cutting decisions
├── data-model.md        # entities added after M0
├── quickstart.md        # gate-by-gate validation
├── contracts/
│   ├── module-interfaces.md   # package boundaries between features
│   ├── cli.md                 # all hyperdime subcommands
│   └── reports-and-gates.md   # reports, ADRs, tags
└── checklists/requirements.md
```

### Source Code (repository root) — end state

```text
configs/{model,data,run,selector,space_rep,hyperhead,experiment}/
src/hyperdime/
├── contracts/     schemas, validation, config, hashing, seeding, manifests          (F1)
├── data/          loaders, splits*, sampling, negatives*                             (F2)
├── embeddings/    qwen*, cache, normalize                                            (F2)
├── oracle/        importance*, targets                                               (F3)
├── baselines/     learning_to_select*, global_selector, mrl_prefix, random_mask     (F3, F6)
├── retrieval/     exact*, scoring, faiss_backend                                    (F3, F7)
├── training/      losses*, selector_trainer, meta_trainer, episodes                 (F3, F6)
├── evaluation/    metrics*, statistics, selector_shift, cross_domain, ablations      (F3–F7)
├── space/         marginal, query_stats, interactions, contrast, redundancy,
│                  descriptor, encoder                                                (F5, F6)
├── hypernet/      hyperhead, modulation, generated_selector                          (F6)
├── selection/     topk*, masks, rdime                                                (F7)
└── cli/           __main__, diagnose, prepare_data, embed, ..., report, decide       (all)
tests/{unit,integration,regression,fixtures}/
experiments/{00_baseline_reproduction,01_selector_shift,02_space_signal,03_hypernetwork,04_ablations}/
reports/{E0..E6}/
docs/decisions/ADR-*.md
```

`*` = already implemented on `master`.

**Structure Decision**: Keep the plan §4 layout in the existing `src/hyperdime` package; each
feature fills only its packages, and cross-feature boundaries are those in
[contracts/module-interfaces.md](contracts/module-interfaces.md).

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| F0 writes lightweight JSON manifests instead of the F1 `RunManifest`/`ArtifactWriter`, and runs before "contracts before models" (principle III) | Tests the project's premise in days; F1–F7 are wasted effort if H1 fails | Building F1 first delays the premise test by weeks. Mitigation: the spike's manifests still record commit, config, seed, and versions; its report is generated, not hand-edited; its results are labeled exploratory and rerun under F1–F4 before publication; its loader, encoder, and trainer land in `src/` with tests and are upgraded to F1 contracts rather than rewritten |

## Risks

| Risk | Where it bites | Mitigation in this plan |
|---|---|---|
| R1 Selector shift too small | F4 | G2 with seed-variability null (R8); stop cleanly on No-Go |
| R2 Descriptor encodes dataset identity | F5, F7 | leave-one-environment-out probes with baseline; meta-test = unseen datasets |
| R3 Pseudo-feedback circularity | F5, F7 | P0 as control; P1 always labeled |
| R4 Unstable generated parameters | F6 | zero-gated low-rank deltas; `λ_delta` regularizer |
| R5 MRL prefix already suffices | F3, F7 | MRL row mandatory in every table |
| R6 Gain from supervision, not the hypernetwork | F6, F7 | label-matched global selector (R13) |
| R7 ANN masks the effect | F7 | exact scoring for science; FAISS only in E6 |
| Masked queries don't speed up dense scoring | F7 (G5) | measure restricted-coordinate scoring separately (R16) |
| Few meta-test environments (3 public + Robust04) | F7 (G4) | per-query paired tests within each; leave-one-dataset-out as optional robustness study |
| Embedding cost (≈28.5M documents, ≈60 GB) | F2, F5 | shared corpora embedded once; sharded, resumable jobs; ≈19M of it only after G2; 500k-cap fallback (R2) |
| Robust04 license unavailable | F5, F7 | local-path loader, explicit skip; meta-test still has 3 public datasets |
| Shared corpora look like separate domains | F5, F7 | `corpus_group` validation in `MetaSplit`; Climate-FEVER reported as query-shift only; DL sets are MS MARCO eval sets |
