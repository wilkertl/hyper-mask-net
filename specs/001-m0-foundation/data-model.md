# Data Model: M0 Foundation

Phase 1 output for [plan.md](plan.md). All records are immutable, reject unknown fields, and carry
a `schema_version`. Decisions behind the conventions are in [research.md](research.md).

## Shared conventions

| Name | Definition | Rules |
|---|---|---|
| `Id` | non-empty string | stripped; no control characters |
| `Split` | `"train" \| "validation" \| "test"` | query-level split ([protocols.md](../../docs/protocols.md)) |
| `Protocol` | `"P0" \| "P1" \| "P2" \| "in-domain"` | adaptation protocol |
| `Sha256` | 64 lowercase hex characters | |
| `Dtype` | `"float16" \| "bfloat16" \| "float32"` | |
| `EMBEDDING_DIM` | `1024` | reused from `hyperdime.embeddings.qwen` |
| `PayloadRef` | `{path, sha256: Sha256, format: "npy" \| "safetensors"}` | `path` is relative to the artifact directory, POSIX, no `..` |

## Records (`hyperdime.contracts.schemas`)

### QueryRecord — `schema_version: 1`

| Field | Type | Rules |
|---|---|---|
| `query_id` | `Id` | unique within an environment |
| `text` | `str` | non-empty |
| `instruction` | `str \| None` | the Qwen task instruction; `None` means none |
| `environment` | `Id` | e.g. `scifact` |
| `split` | `Split` | |

### DocumentRecord — `schema_version: 1`

| Field | Type | Rules |
|---|---|---|
| `doc_id` | `Id` | unique within an environment |
| `title` | `str` | may be empty |
| `text` | `str` | non-empty after joining with title |
| `environment` | `Id` | |

### QrelsRecord — `schema_version: 1`

One judgment per record; a qrels file is JSON Lines of these.

| Field | Type | Rules |
|---|---|---|
| `query_id` | `Id` | |
| `doc_id` | `Id` | |
| `relevance` | `int` | ≥ 0; gains are linear (trec_eval) |
| `environment` | `Id` | |
| `split` | `Split` | test judgments are only readable by evaluation (enforced in M1.3) |

### EmbeddingArtifact — `schema_version: 1`

Describes one stored matrix; the matrix itself is a payload.

| Field | Type | Rules |
|---|---|---|
| `role` | `"queries" \| "documents"` | |
| `environment` | `Id` | |
| `ids` | `list[Id]` | row order; distinct; `len(ids) == num_rows` |
| `num_rows` | `int` | ≥ 1 |
| `dim` | `Literal[1024]` | |
| `dtype` | `Dtype` | |
| `normalized` | `bool` | L2-normalized rows |
| `pooling` | `Literal["last_token"]` | |
| `model_name` | `str` | `Qwen/Qwen3-Embedding-0.6B` |
| `model_revision` | `str` | pinned commit SHA |
| `instruction` | `str \| None` | queries only; must be `None` when `role == "documents"` |
| `payload` | `PayloadRef` | shape `(num_rows, 1024)` |

### SpaceDescriptor — `schema_version: 0` (provisional until M5.5)

| Field | Type | Rules |
|---|---|---|
| `environment` | `Id` | |
| `protocol` | `Protocol` | `in-domain` not allowed |
| `feature_names` | `list[str]` | distinct; length `h ≥ 1` |
| `groups` | `list["A" \| "B" \| "C" \| "D" \| "E"]` | must be allowed by `protocol` (P0: A, E only) |
| `num_documents` | `int` | ≥ 1; sample size used |
| `num_queries` | `int` | 0 under P0 |
| `payload` | `PayloadRef` | shape `(1024, h)` |

### SelectorCheckpoint — `schema_version: 0` (provisional until M7)

| Field | Type | Rules |
|---|---|---|
| `kind` | `"linear" \| "global" \| "per_domain" \| "generated"` | |
| `environment` | `Id \| None` | `None` for `global` |
| `dim` | `Literal[1024]` | |
| `rank` | `int \| None` | low-rank modulation rank; only for `generated` |
| `payload` | `PayloadRef` | `safetensors` |
| `training_manifest_sha256` | `Sha256` | fingerprint of the training run's manifest |

### EvaluationResult — `schema_version: 1`

| Field | Type | Rules |
|---|---|---|
| `environment` | `Id` | |
| `protocol` | `Protocol` | |
| `split` | `Split` | |
| `scoring` | `Literal["exact"]` | FAISS results are not scientific results |
| `selector` | `str` | e.g. `full`, `mrl_prefix`, `linear` |
| `k` | `int` | 1 ≤ k ≤ 1024 |
| `metrics` | `dict[str, float]` | keys like `ndcg@10`; finite |
| `per_query` | `dict[Id, dict[str, float]]` | same metric keys as `metrics` |
| `num_queries` | `int` | `== len(per_query)` |

## Provenance (`hyperdime.contracts.manifests`)

### RunManifest — `schema_version: 1`

| Field | Type | Rules |
|---|---|---|
| `run_id` | `Id` | `<UTC yyyymmddThhmmssZ>-<8 hex>` |
| `command` | `str` | e.g. `diagnose` |
| `git_commit` | `str` | 40-hex SHA or `"unknown"` |
| `git_dirty` | `bool \| None` | `None` when git is unavailable |
| `config_hash` | `Sha256` | canonical JSON of the validated config |
| `config` | `dict` | the validated config, for self-contained provenance |
| `dataset_hash` | `Sha256 \| Literal["none"]` | |
| `model_revision` | `str \| Literal["none"]` | |
| `seed` | `int` | ≥ 0 |
| `determinism` | `dict[str, str \| bool \| None]` | what the seed manager set, plus `PYTHONHASHSEED` |
| `device` | `str` | device actually used, e.g. `cpu`, `cuda:0` |
| `dtype` | `Dtype` | |
| `versions` | `dict[str, str]` | python, torch, numpy, hyperdime |
| `started_at` | `datetime` | timezone-aware UTC |
| `finished_at` | `datetime \| None` | ≥ `started_at`; set when the run completes |
| `artifacts` | `list[{name, sha256}]` | files published with this manifest |

### Lifecycle

```text
created (started_at, finished_at=None)
  → payloads staged in .<run_id>.tmp-<uuid>/
  → finalized (finished_at set, artifacts list filled)
  → manifest.json written last, directory renamed → published
```

A published artifact directory is immutable. A crash before the rename leaves only the hidden temp
directory; it is never read.

## Configuration (`hyperdime.contracts.config`)

`RunConfig` is composed by Hydra from `configs/` groups, then validated.

| Model | Fields | Rules |
|---|---|---|
| `ModelConfig` | `name`, `revision`, `embedding_dim`, `frozen`, `normalize`, `max_length`, `dtype` | `name` fixed; `embedding_dim: Literal[1024]`; `frozen: Literal[True]`; `revision` a 40-hex SHA |
| `DataConfig` | `environment`, `source`, `instruction`, `validation_fraction`, `sample` | `validation_fraction ∈ (0, 1)`; `sample` optional `{num_documents, num_queries, seed}` |
| `RunSettings` | `seed`, `device`, `dtype`, `output_dir` | `device ∈ {auto, cpu, cuda, cuda:N}` |
| `RunConfig` | `model`, `data`, `run`, `config_version` | `config_version: Literal[1]`; `extra="forbid"` at every level |

## Relationships

```text
RunConfig ──hash──▶ RunManifest ◀── every artifact directory
EmbeddingArtifact ─ids─▶ QueryRecord.query_id / DocumentRecord.doc_id
QrelsRecord ─query_id/doc_id─▶ QueryRecord / DocumentRecord
SelectorCheckpoint ─training_manifest_sha256─▶ RunManifest
SpaceDescriptor, EvaluationResult ─environment─▶ DataConfig.environment
```
