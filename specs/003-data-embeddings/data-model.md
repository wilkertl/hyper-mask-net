# Data Model: F2 Data and Embeddings

Phase 1 output for [plan.md](plan.md). Builds on F1's records
([001 data model](../001-m0-foundation/data-model.md)); new records are frozen pydantic models
with `extra="forbid"` and a `schema_version`, like F1's.

## EnvironmentSpec — `hyperdime.data.environments`, `spec_version: 1`

The code registry of environments; `DataConfig.environment` selects one, and `DataConfig.source`
must equal its `source`.

| Field | Type | Rules |
|---|---|---|
| `environment` | `Id` | `scifact`, `nfcorpus`, `msmarco` in F2 |
| `loader` | `Literal["beir_hf"]` | Hugging Face parquet + TSV (research D1) |
| `source` | `str` | e.g. `BeIR/scifact` |
| `source_revision` | 40-hex commit | pinned |
| `qrels_source` | `str` | e.g. `BeIR/scifact-qrels` |
| `qrels_revision` | 40-hex commit | pinned |
| `train_split` | `str \| None` | source split name for training judgments; `None` = no training judgments |
| `test_split` | `str` | `test`, or `dev` for MS MARCO |
| `train_query_cap` | `int \| None` | 50,000 for MS MARCO, else `None` |
| `corpus_group` | `Id` | equals `environment` in F2 (no shared corpora yet) |
| `eval_query_sets` | `list[EvalQuerySet]` | evaluation-only query sets over this corpus |
| `instruction` | `str` | Qwen query instruction |
| `license` | `Literal["public", "licensed"]` | `public` in F2 |

`EvalQuerySet`: `name` (`trec-dl-2019`, `trec-dl-2020`, `trec-dl-hard`), `loader:
Literal["ir_datasets"]`, `source` (the `ir_datasets` id).

`has_train_qrels` is derived: `train_split is not None`.

## DatasetManifest — `hyperdime.contracts.schemas`, `schema_version: 1`

Written as `dataset.json` in the prepared-environment artifact.

| Field | Type | Rules |
|---|---|---|
| `environment` | `Id` | |
| `spec` | `dict` | the `EnvironmentSpec` used, for self-contained provenance |
| `num_documents` | `int` | ≥ 1 |
| `num_queries` | `dict[str, int]` | per split and per evaluation query set |
| `num_qrels` | `dict[str, int]` | per judgments file |
| `dropped_qrels` | `int` | judgments whose document is not in the corpus |
| `excluded_queries` | `dict[str, int]` | per split: queries left with no positive judgment |
| `split_seed` | `int` | 0 |
| `validation_fraction` | `float` | 0.1 |
| `files` | `dict[str, Sha256]` | every record file in the artifact |
| `fingerprint` | `Sha256` | `config_hash` of `files`; the `dataset_hash` downstream manifests record |

## NegativePool — `hyperdime.contracts.schemas`, `schema_version: 1`

One per query, as JSON Lines (`negatives.jsonl`) in the negatives artifact.

| Field | Type | Rules |
|---|---|---|
| `query_id` | `Id` | |
| `environment` | `Id` | |
| `split` | `Literal["train", "validation"]` | never `test` |
| `doc_ids` | `list[Id]` | exactly `pool_size`; repeats allowed; none judged relevant; not the query's own id |
| `depth` | `int` | 1,000 |
| `pool_size` | `int` | 64 |
| `seed` | `int` | 0 |

## Artifacts (F1 layout, `artifacts/<kind>/<env>/<run_id>/`)

| Kind | Files | Inputs recorded in the manifest |
|---|---|---|
| `datasets/<env>` | `documents.jsonl`, `queries.jsonl`, `qrels-{train,validation,test}.jsonl`, `qrels-<eval set>.jsonl`, `dataset.json` | source and qrels revisions |
| `embeddings/<env>` | `documents.npy` + `documents.json`, `queries.npy` + `queries.json`, one `.npy` + `.json` pair per evaluation query set | `dataset_hash`, settings |
| `negatives/<env>` | `negatives.jsonl` | `dataset_hash`, embeddings run id |

Each `.json` next to a `.npy` is an `EmbeddingArtifact` whose `payload` points at it. `queries.npy`
holds the train, validation, and test queries of the environment.

Resumable work in progress lives in `artifacts/.work/embeddings/<env>/<role>/<settings-hash>/`
(`part-NNNNN.npy`, `settings.json`); readers never treat it as an artifact.

## Relationships

```text
EnvironmentSpec ─▶ datasets/<env> (DatasetManifest.fingerprint) ─▶ embeddings/<env> ─▶ negatives/<env>
DatasetManifest.fingerprint == RunManifest.dataset_hash of every embeddings and negatives run
```
