# Implementation Plan: F2 Data and Embeddings

**Branch**: `agent/F2-data-embeddings` | **Date**: 2026-10-12 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/003-data-embeddings/spec.md`

## Summary

Prepare SciFact, NFCorpus, and full MS MARCO (with the TREC DL 2019, DL 2020, and DL-HARD query
sets) from pinned Hugging Face and `ir_datasets` sources into F1's shared records, with query
splits whose test judgments live in files training code never opens; embed each environment once
with the pinned frozen model in resumable shards that publish atomically through `ArtifactWriter`;
and mine ADR-0002 hard-negative pools for training and validation queries by streamed exact
search. Three `hyperdime` subcommands drive it. Covers backlog M1.1–M1.4 and M2.1–M2.2.

## Technical Context

**Language/Version**: Python ≥ 3.11 (CI 3.12)

**Primary Dependencies**: F1 contracts (pydantic, Hydra); torch, numpy; `huggingface_hub` and
`pyarrow` (download and read pinned parquet/TSV), `ir_datasets` (TREC query sets), all in the
locked `data` extra; `transformers` (`embed` extra) for the local encoder

**Storage**: published artifacts under `artifacts/{datasets,embeddings,negatives}/<env>/<run_id>/`
(JSON Lines records, float16 `.npy`); resumable shards under `artifacts/.work/`

**Testing**: pytest on CPU without network: fake parquet/TSV sources and fake `ir_datasets` objects
injected into the loaders, a deterministic stub encoder, small synthetic corpora

**Target Platform**: implementation and tests anywhere; real runs on the RTX 4090 workstation over
SSH (research D8)

**Project Type**: Python library + `hyperdime` CLI (single project)

**Performance Goals**: MS MARCO embedding within one 24 GB GPU, throughput measured before the full
run (SC-006); at most one shard lost per interruption (SC-005); negative mining for ~55,000 MS MARCO
queries in minutes on GPU

**Constraints**: no test judgment reachable from training paths; memory bounded by one shard of text
and one document chunk, never the whole corpus; every output published with a manifest

**Scale/Scope**: 3 environments, 3 evaluation query sets, 8,851,000 documents, ~70,000 embedded
queries; 3 CLI subcommands; ~6 modules touched or added

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Pre-research | Post-design | Evidence |
|---|---|---|---|
| I. Frozen Coordinate System | ✅ | ✅ | Pinned revision `97b0c614…` from `ModelConfig`; settings guard refuses mixing (D4); embedded once and reused |
| II. Leakage-Free Protocols | ✅ | ✅ | Test and evaluation-set judgments in separate files training code never opens (D2); negatives for train/validation only; tests corrupt test files and check training outputs are unchanged |
| III. Contracts and Provenance First | ✅ | ✅ | F1 records reused; new `DatasetManifest`, `NegativePool` records in `hyperdime.contracts`; every output through `ArtifactWriter` with `dataset_hash` |
| IV. Evidence-Gated Complexity | ✅ | ✅ | F2 is a G1 prerequisite; only the three G1 environments; other datasets wait for F5 |
| V. Exact, Generated Measurement | ✅ | ✅ | Exact streamed search for negatives (D6); counts in manifests are computed, not typed |
| VI. Test-Verified Mathematics | ✅ | ✅ | Hand-checkable tests for split sizes, streamed top-k equal to full top-k, resume equal to an uninterrupted run, pool rules |
| VII. Independent, Replaceable Modules | ✅ | ✅ | data, embeddings, retrieval, and CLI stay separate; the encoder is injected |
| Artifacts and Data Handling | ✅ | ✅ | Data and vectors only under gitignored `artifacts/`; sources pinned; SHA-256 everywhere |
| Workflow and Quality Gates | ✅ | ✅ | One branch per feature; one commit per ticket group; SCRAP in pre-commit |

**Result**: PASS, no deviations.

## Project Structure

### Documentation (this feature)

```text
specs/003-data-embeddings/
├── plan.md  research.md  data-model.md  quickstart.md  tasks.md
├── contracts/{python-api.md, cli.md}
└── checklists/requirements.md
```

### Source Code (repository root)

```text
configs/data/{scifact,nfcorpus,msmarco}.yaml      # DataConfig per environment
configs/experiment/prepare.yaml                   # model + data + run, for the three commands
src/hyperdime/contracts/schemas.py                # + DatasetManifest, NegativePool
src/hyperdime/data/environments.py                # EnvironmentSpec registry (new)
src/hyperdime/data/sources.py                     # pinned HF parquet/TSV and ir_datasets readers (new)
src/hyperdime/data/prepare.py                     # prepare_environment, load_environment (new)
src/hyperdime/data/negatives.py                   # + sample_from_rankings, build_negative_pools
src/hyperdime/embeddings/cache.py                 # + embed_sharded, publish/load embeddings
src/hyperdime/embeddings/encoders.py              # encoder settings + factory over qwen/remote (new)
src/hyperdime/retrieval/exact.py                  # + exact_search_streamed
src/hyperdime/cli/{prepare_data,embed,mine_negatives}.py, cli/__main__.py
src/hyperdime/artifacts.py                        # latest_artifact lookup (new)
tests/unit/data/, tests/unit/embeddings/, tests/integration/test_f2_pipeline.py
```

**Structure Decision**: Follow plan §4 and the roadmap's module contract: data code in
`hyperdime.data`, embeddings in `hyperdime.embeddings`, search in `hyperdime.retrieval`, one CLI
module per command. The spike's modules stay unchanged except where shared functions gain a
refactored core (`sample_hard_negatives`), which keeps their tests passing.

## Complexity Tracking

No constitution violations.

## Risks

- **GPU host access**: real runs need the RTX 4090 workstation over SSH; its address is needed
  before T-run tasks. Code and tests do not depend on it.
- **MS MARCO preparation memory**: the corpus is streamed from parquet in batches and written as
  JSON Lines; the run's host needs ~20 GB of disk for text and ~18 GB for vectors.
- **ir_datasets downloads** come from TREC, MS MARCO, and GitHub URLs at prepare time; their
  checksums are verified by `ir_datasets`, and the prepared records are fingerprinted.
- **Graded TREC judgments**: DL sets use relevance 0–3; F2 keeps grades as given. F3 must apply
  the TREC convention (relevance ≥ 2 counts as relevant for AP and MRR) when it evaluates.
