---

description: "Task list for F2 data and embeddings"
---

# Tasks: F2 Data and Embeddings

**Input**: Design documents from `specs/003-data-embeddings/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md

**Tests**: Required (FR-007, FR-018, constitution VI); all on CPU without network, with fake
sources and a stub encoder. Tests first in each story.

**Organization**: US1 = M1.1–M1.3 (environments, records, splits, guards), US2 = M2.1–M2.2
(embedding), US3 = MS MARCO and the TREC DL sets at scale, US4 = M1.4 (negatives). Run tasks
(T030+) need the RTX 4090 workstation.

## Format: `[ID] [P?] [Story] Description`

---

## Phase 1: Setup

- [X] T001 Add `huggingface_hub>=0.30` and `pyarrow>=15` to the `data` extra in pyproject.toml (beside `datasets`, `ir-datasets`), relock with `uv lock`, and add configs/data/nfcorpus.yaml and configs/data/msmarco.yaml (environment, `source`, instruction, `validation_fraction: 0.1`) plus configs/experiment/prepare.yaml (`# @package _global_`, defaults model/data/run, `config_version: 1`)
- [X] T002 [P] Create tests/unit/data/ and tests/unit/embeddings/ and a fixture module tests/fixtures/beir_fake.py that writes a tiny BEIR-shaped source (corpus and queries parquet with `_id`, `title`, `text`; qrels TSVs with `query-id`, `corpus-id`, `score`) and a fake `ir_datasets` dataset object with `queries_iter()` and `qrels_iter()`

## Phase 2: Foundational

- [X] T003 Add `DatasetManifest` and `NegativePool` records (data-model.md, every field rule) to src/hyperdime/contracts/schemas.py and re-export them; tests in tests/unit/contracts/test_schemas.py extend `VALID` with both and add their field rules (`split` never `test`, `len(doc_ids) == pool_size`, `fingerprint` 64 hex)
- [X] T004 [P] Add `latest_artifact(root, kind, environment)` in src/hyperdime/artifacts.py returning the newest published `<root>/<kind>/<environment>/<run_id>/` with a complete manifest (run ids sort chronologically), raising `ContractError` when none exists; test in tests/unit/test_artifacts.py

## Phase 3: User Story 1 — environments in one leakage-safe form (P1) 🎯 MVP

**Independent Test**: prepare the fake SciFact-shaped source, check counts, splits, manifest, and that a training load never opens test judgments.

- [X] T005 [P] [US1] Tests in tests/unit/data/test_environments.py: the registry has scifact, nfcorpus, msmarco with 40-hex revisions; msmarco has the three evaluation query sets and `train_query_cap` 50,000; an unknown name raises `ContractError`; a `DataConfig` whose `source` differs from the registry is refused
- [X] T006 [P] [US1] Tests in tests/unit/data/test_prepare.py (fake sources): records and counts; document text joins title and text; validation is 10% of train by seed 0 and disjoint from train and test; preparing twice gives the same fingerprint; a judgment to a missing document is dropped and counted; a query left without a positive is excluded and counted; duplicate ids fail naming the id; `train_query_cap` samples deterministically; evaluation query sets get their own judgments file
- [X] T007 [P] [US1] Leakage tests in tests/unit/data/test_leakage.py: `load_environment(..., include_test_qrels=False)` returns no `test` or evaluation-set judgments and still works after `qrels-test.jsonl` and every `qrels-<eval>.jsonl` are deleted (so it never opens them); with `True` they are returned
- [X] T008 [US1] Implement `EvalQuerySet`, `EnvironmentSpec`, `ENVIRONMENTS`, and `environment_spec` in src/hyperdime/data/environments.py with the revisions of research D1 and the instructions (msmarco: "Given a web search query, retrieve relevant passages that answer the query")
- [X] T009 [US1] Implement src/hyperdime/data/sources.py: `SourceReader` protocol with `iter_corpus()`, `queries()`, `qrels(split)`, `eval_set(name)`; `HuggingFaceSource` (pinned `hf_hub_download` + `pyarrow` batch reading + TSV) and `IrDatasetsQuerySet`; both injectable for tests
- [X] T010 [US1] Implement `prepare_environment`, `load_environment`, `iter_documents`, and `DatasetBundle` in src/hyperdime/data/prepare.py per contracts/python-api.md, streaming documents to `documents.jsonl` and writing the per-split judgments files and `dataset.json` through the writer
- [X] T011 [US1] Add `hyperdime prepare-data` in src/hyperdime/cli/prepare_data.py and register it in src/hyperdime/cli/__main__.py; integration test in tests/integration/test_f2_pipeline.py with the fake source injected through a module-level factory

## Phase 4: User Story 2 — each environment embedded once (P1)

**Independent Test**: embed the fake environment with the stub encoder, reload, check rows, width, norm, order, reuse, and the settings guard.

- [X] T012 [P] [US2] Tests in tests/unit/embeddings/test_sharded.py: `embed_sharded` with a stub encoder writes ceil(N/shard) parts; an encoder that raises after two shards leaves two parts and a restart encodes only the rest; the concatenated result equals an uninterrupted run; a work directory with different settings is refused naming the key; texts are consumed as a stream (an iterator works)
- [X] T013 [P] [US2] Tests in tests/unit/embeddings/test_publish.py: `publish_embeddings` writes `<name>.npy` (float16) and `<name>.json` (`EmbeddingArtifact` with ids in order, dim 1024, payload SHA-256 matching the file); `load_embeddings` returns them memmapped; `check_settings` names differing keys; `encoder_settings` includes revision, max_length, dtype, backend, and the instruction only for queries
- [X] T014 [US2] Implement `embed_sharded`, `publish_embeddings` (memmap concatenation), `load_embeddings`, and `check_settings` in src/hyperdime/embeddings/cache.py, keeping `embed_cached`/`load_cached` for the spike
- [X] T015 [US2] Implement `encoder_settings` and `build_encoder` in src/hyperdime/embeddings/encoders.py over `QwenEncoder` (pinned revision, `max_length`, bfloat16 on GPU) and `VllmEncoder`
- [X] T016 [US2] Add `hyperdime embed` in src/hyperdime/cli/embed.py: latest dataset artifact, documents and every query set (train/validation/test queries in `queries`, each evaluation set separately), reuse of a published embedding with equal settings, refusal on different settings; integration test with a stub encoder factory in tests/integration/test_f2_pipeline.py
- [X] T017 [P] [US2] Add scripts/checks/f0_agreement.py: minimum and mean cosine between the F0 spike cache (`artifacts/spike/<env>/{corpus,queries}`) and an F2 embeddings artifact over shared ids; exit 1 below 0.999 (SC-004)

## Phase 5: User Story 3 — MS MARCO and the TREC DL sets (P2)

- [X] T018 [P] [US3] Tests: a fake MS MARCO-shaped source with a `dev` test split and three evaluation query sets prepares into `qrels-test.jsonl` plus `qrels-trec-dl-2019.jsonl`, `qrels-trec-dl-2020.jsonl`, `qrels-trec-dl-hard.jsonl` whose judgments all refer to corpus ids; `embed` writes one artifact per evaluation set and the documents once; documents are streamed (prepare and embed never build the full corpus list — checked by a generator source that fails if materialized twice)
- [X] T019 [US3] Make `prepare_environment` and `embed` stream documents end to end (parquet batches → `documents.jsonl` → shard texts) and support evaluation query sets through `IrDatasetsQuerySet`

## Phase 6: User Story 4 — hard-negative pools (P2)

- [X] T020 [P] [US4] Tests in tests/unit/retrieval/test_streamed.py: `exact_search_streamed` equals `exact_search` on a random 300 × 16 corpus for several chunk sizes, including a chunk smaller than `depth`
- [X] T021 [P] [US4] Tests in tests/unit/data/test_negative_pools.py: pools have 64 entries, none judged relevant or the query's own id, identical for the same seed, built for train and validation only; `build_negative_pools` never receives test judgments (a bundle loaded without them is enough); `sample_from_rankings` matches the spike's `sample_hard_negatives` on the same inputs
- [X] T022 [US4] Implement `exact_search_streamed` in src/hyperdime/retrieval/exact.py, split `sample_from_rankings` out of `sample_hard_negatives`, and add `build_negative_pools` in src/hyperdime/data/negatives.py
- [X] T023 [US4] Add `hyperdime mine-negatives` in src/hyperdime/cli/mine_negatives.py publishing `negatives.jsonl`; integration test in tests/integration/test_f2_pipeline.py including the leakage check (corrupted test judgments leave pools unchanged)

## Phase 7: Polish

- [X] T024 [P] Update docs/architecture.md components (data, embeddings, retrieval rows) and README.md (F2 commands)
- [X] T025 Run `ruff check . && ruff format --check . && mypy && pytest` and SCRAP; act on `HIGH` findings for new tests or explain them in the pull request

## Phase 8: Runs on the RTX 4090 workstation

- [X] T026 Sync the locked environment (`--extra cu128 --extra data --extra embed --extra dev`) on the workstation and run `hyperdime diagnose` — done on DataPolicy-4090 (`ceia40901`): uv 0.13, Python 3.12, torch 2.11.0+cu128; `diagnose` clean at `0f047ee` on `cuda:0`
- [X] T027 Prepare, embed, and mine negatives for SciFact and NFCorpus; check SC-001 counts in `dataset.json` — done: SciFact 5,183 docs / 728 train / 81 validation / 300 test; NFCorpus 3,633 / 2,331 / 259 / 323; 809 and 2,590 pools of 64; about 2 minutes in total
- [X] T028 Run scripts/checks/f0_agreement.py for SciFact against the F0 caches (SC-004) — done: SC-004 passes, minimum cosine 0.99972 (documents) and 0.99967 (queries)
- [X] T029 Prepare MS MARCO; check SC-001 counts; measure embedding throughput on one shard and replace SC-006's 12-hour ceiling in the spec with the measured estimate — done: MS MARCO 8,841,823 passages, 6,980 test, DL19/20/HARD 43/54/50, prepared in 85 s; 154 s per 65,536-passage shard, so about 6 hours in total
- [X] T030 Embed MS MARCO (resumable) and mine its negatives; record wall time and peak GPU memory in the pull request — done: 5.86 h wall time (16.8 GB GPU memory at steady state), 8,841,823 + 56,980 + 147 vectors, unit norm; 50,000 pools of 64 in 45 s

## Dependencies

- Setup → Foundational → US1 → US2 → US3 (extends US1 and US2) → US4 (needs US2's embeddings) → Polish → Runs
- Within each story, tests first

## Implementation Strategy

US1 + US2 on SciFact form the MVP; US3 scales them; US4 completes F3's inputs. One commit per story.
