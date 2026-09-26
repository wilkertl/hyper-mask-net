# Data Model: HyperDIME-Qwen Project Roadmap

Phase 1 output for [plan.md](plan.md). Base records (`QueryRecord`, `DocumentRecord`,
`QrelsRecord`, `EmbeddingArtifact`, `SpaceDescriptor`, `SelectorCheckpoint`, `EvaluationResult`,
`RunManifest`, `RunConfig`) are defined in [001 data-model](../001-m0-foundation/data-model.md).
This file adds the entities later milestones need. Field lists are the target shape; each owning
feature finalizes them with a schema version.

## Environments and splits

### EnvironmentSpec — owner F2 (extended in F5)

| Field | Type | Rules |
|---|---|---|
| `environment` | `Id` | e.g. `scifact`, `nq`, `trec-covid`, `robust04` |
| `loader` | `"hf" \| "ir_datasets"` | research R2 |
| `source` | `str` | e.g. `BeIR/scifact`, `disks45/nocr/trec-robust-2004` |
| `source_revision` | `str` | pinned dataset revision or `ir_datasets` version |
| `qrels_source` | `str` | e.g. `BeIR/nq-train-qrels` for NQ training labels |
| `corpus_group` | `Id` | environments with the same corpus share it, e.g. `fever` for FEVER and Climate-FEVER; one embedding per group |
| `eval_query_sets` | `list[{name, source}]` | extra evaluation-only query sets over this corpus, e.g. `trec-dl-2019`, `trec-dl-2020`, `dl-hard` for MS MARCO |
| `instruction` | `str` | Qwen task instruction used for queries |
| `corpus_cap` | `int \| None` | `None` (full corpus) by default; the fallback cap is research R2 |
| `license` | `"public" \| "licensed"` | `licensed` for Robust04 |
| `local_path` | `Path \| None` | required when `license == "licensed"`; never downloaded |
| `has_train_qrels` | `bool` | true for MS MARCO, NQ, HotpotQA, FEVER, FiQA, SciFact, NFCorpus |
| `shard_of` | `Id \| None` | set for sub-environments; such rows are never "unseen domains" |

### MetaSplit — owner F5 (fixed by ADR)

| Field | Type | Rules |
|---|---|---|
| `meta_train` | `list[Id]` | every member has `has_train_qrels = true` |
| `meta_validation` | `list[Id]` | disjoint from the others |
| `meta_test` | `list[Id]` | ≥ 2 whole public datasets; no shards |
| `query_shift_only` | `list[Id]` | held-out environments whose `corpus_group` is in meta-train (Climate-FEVER); reported separately |
| `adr` | `str` | path to the ADR that fixed it |
| `frozen_at` | `datetime` | before any descriptor normalization is fitted |

Validation: no `corpus_group` appears in both `meta_train` and `meta_validation ∪ meta_test`;
the only exception is an environment listed in `query_shift_only`.

### DatasetManifest — owner F2

`RunManifest` plus: per-split query counts, qrels counts, corpus size before/after sampling,
sampling seed, `tree_sha256` of the normalized files. One per environment.

## Supervision artifacts

### NegativePool — owner F2

| Field | Type | Rules |
|---|---|---|
| `environment` | `Id` | |
| `split` | `Literal["train"]` | pools are never built for other splits |
| `depth` | `int` | 100 |
| `pool_size` | `int` | M; default 8 |
| `pools` | JSONL of `{query_id, doc_ids}` | no judged positive in `doc_ids` |

### TargetImportance — owner F3 (ticket M3.1)

| Field | Type | Rules |
|---|---|---|
| `environment` | `Id` | |
| `split` | `"train" \| "validation"` | |
| `temperature` | `float` | > 0 |
| `aggregation` | `{positives: "relevance_mean", negatives: "mean_top_m", m: int}` | research R4 |
| `ids` | `list[Id]` | row order |
| `payload` | `PayloadRef` | `(num_queries, 1024)`, rows sum to 1, finite |

## Selector analysis

### SelectorProperties — owner F4

| Field | Type | Rules |
|---|---|---|
| `selector` | `SelectorCheckpoint` ref | |
| `battery` | `str` | id of the common query battery |
| `mean_importance` | `PayloadRef` | `(1024,)`, sums to 1 |
| `selection_frequency` | `PayloadRef` | `(len(k_values), 1024)` |
| `k_values` | `list[int]` | |

### TransferCell — owner F4 (one row of the transfer matrix)

| Field | Type | Rules |
|---|---|---|
| `train_environment` | `Id` | |
| `test_environment` | `Id` | |
| `seed` | `int` | |
| `k` | `int` | |
| `metrics` | `dict[str, float]` | nDCG@10, MRR@10, MRR@100, Recall@100 |
| `js_divergence` | `float` | vs the in-domain selector on the battery |
| `overlap_at_k` | `float` | in [0, 1] |
| `spearman` | `float` | in [−1, 1] |
| `evaluation_result` | `run_id` | the `EvaluationResult` it summarizes |

### ProbeResult — owner F5

| Field | Type | Rules |
|---|---|---|
| `feature_set` | `list[str]` | descriptor groups used |
| `target` | `str` | e.g. `mean_importance` |
| `held_out_environment` | `Id` | leave-one-environment-out |
| `score` | `float` | e.g. R² or Spearman |
| `baseline_score` | `float` | trivial baseline (research R10) |
| `p_value` | `float` | permutation test |
| `num_permutations` | `int` | ≥ 1,000 |

## Prototype

### ModulationPackage — owner F6 (ticket M8.1)

| Field | Type | Rules |
|---|---|---|
| `environment` | `Id` | |
| `protocol` | `"P0" \| "P1" \| "P2"` | |
| `rank` | `int` | r |
| `payload` | `PayloadRef` | `safetensors` with `A (1024×r)`, `B (1024×r)`, `delta_b (1024)` |
| `base_selector` | `SelectorCheckpoint` ref | `W_0, b_0` |
| `descriptor` | `SpaceDescriptor` ref | input used to generate it |
| `hypernet` | `SelectorCheckpoint` ref | kind `generated`; the `G_ψ, H_φ` checkpoint |

Zero `A`, `B`, `delta_b` ⇒ generated selector ≡ base selector.

### EpisodeLog — owner F6

Per training step: environment, descriptor sample seed, batch size, KL, `||Δθ_T||`, gate value.
Written as JSONL next to the meta-training manifest.

## Evaluation and decisions

### EfficiencyRecord — owner F7

| Field | Type | Rules |
|---|---|---|
| `environment` | `Id` | |
| `stage` | `"encode" \| "selector" \| "mask" \| "score_dense" \| "score_restricted" \| "search" \| "adapt_offline"` | |
| `k` | `int \| None` | |
| `device` | `str` | |
| `num_queries` | `int` | ≥ 1,000 for online stages |
| `p50_ms`, `p95_ms`, `mean_ms` | `float` | |
| `theta_bytes` | `int \| None` | for `adapt_offline` |

### GateDecision — owner: each gate's feature

| Field | Type | Rules |
|---|---|---|
| `gate` | `"G1" … "G5"` | |
| `outcome` | `"pass" \| "fail"` | |
| `criteria` | `list[{name, value, threshold, met}]` | from research R7, R8, R10 and F7 |
| `evidence` | `list[run_id]` | result files it rests on |
| `adr` | `str` | `docs/decisions/ADR-XXXX-<gate>.md` |
| `tag` | `str` | e.g. `g2-selector-shift` |

## State transitions

```text
Milestone:  not started → specified → planned → in progress → gate decided
                                                           ├─ pass → next milestone may be specified
                                                           └─ fail → later milestones closed "not planned"

Evaluation on meta-test:  locked → (meta-validation verdict = match/beat) → unlocked → evaluated once → locked
```

## Relationships

```text
EnvironmentSpec ─┬─▶ DatasetManifest ─▶ EmbeddingArtifact ─▶ NegativePool ─▶ TargetImportance
                 └─▶ MetaSplit
TargetImportance ─▶ SelectorCheckpoint (linear/global/per_domain) ─▶ SelectorProperties ─▶ TransferCell
EmbeddingArtifact + MetaSplit ─▶ SpaceDescriptor ─▶ ProbeResult
SpaceDescriptor ─▶ hypernet ─▶ ModulationPackage ─▶ EvaluationResult / EfficiencyRecord
EvaluationResult, TransferCell, ProbeResult ─▶ GateDecision
```
