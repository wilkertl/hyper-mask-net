# Data Model: F4 Selector Shift and Gate G2

New records live in `hyperdime.contracts.schemas` beside F1's, with `schema_version` and JSON
round-trip tests. They refine the roadmap's planned records (owner F4,
`specs/002-hyperdime-roadmap/data-model.md`) where F4's research changed the design; the
differences are listed under each record.

## Protocol (changed)

`Protocol = Literal["P0", "P1", "P2", "in-domain", "transfer"]`. `"transfer"` marks an
`EvaluationResult` of a selector trained on another environment. Existing records keep validating.

## SelectorProperties (new)

| Field | Type | Rules |
|---|---|---|
| `schema_version` | `Literal[0]` | |
| `environment` | `Id` | the environment the selector was trained on |
| `seed` | `int ≥ 0` | |
| `selector_sha256` | `Sha256` | the checkpoint's payload hash |
| `battery_sha256` | `Sha256` | hash of the battery's sorted query ids (research D5) |
| `k_values` | `list[int]` | `[307]` in E1; each in [1, 1024] |
| `mean_importance` | `PayloadRef` (npy) | `(1024,)`, finite, sums to 1 |
| `selection_frequency` | `PayloadRef` (npy) | `(len(k_values), 1024)`, each row in [0, 1], sums to k |

Differences from the roadmap: the selector is referenced by payload hash and seed instead of an
embedded checkpoint, and the battery by hash.

## TransferCell (new)

One ordered (train, test) pair on one query set, seeds averaged (research D4).

| Field | Type | Rules |
|---|---|---|
| `schema_version` | `Literal[0]` | |
| `train_environment`, `test_environment` | `Id` | equal for in-domain cells |
| `query_set` | `str` | `"test"` or a TREC DL set name |
| `seeds` | `list[int]` | `[0, 1, 2]` |
| `k` | `int` | 307 |
| `metrics` | `dict[str, Finite]` | seed-averaged means: nDCG@10, MRR@10, MRR@100, Recall@100, AP |
| `in_domain_ndcg10` | `Finite` | the test environment's in-domain seed-averaged nDCG@10 |
| `degradation` | `Finite` | in-domain minus this cell, nDCG@10 (0 for in-domain cells) |
| `p_value`, `p_holm` | `float \| None` | in [0, 1]; both `None` for in-domain cells; `p_holm` also `None` for TREC DL sets, which are outside the six-pair family |
| `ci_low`, `ci_high` | `float \| None` | 95% bootstrap interval of `degradation` |
| `overlap_at_k` | `float` | in [0, 1]; mean over the 9 seed pairs on the test queries (D7) |
| `results` | `list[str]` | the `EvaluationResult` selector names this cell averages |

Differences from the roadmap: seeds are averaged into one cell (D4), so `seed` becomes `seeds`;
JS and Spearman move to the battery-level behavior result, where every selector pair is compared on
the same queries (D7).

## GateDecision (new)

| Field | Type | Rules |
|---|---|---|
| `schema_version` | `Literal[0]` | |
| `gate` | `Literal["G1", "G2", "G3", "G4", "G5"]` | |
| `outcome` | `Literal["pass", "fail"]` | computed: every criterion met → pass |
| `criteria` | `list[Criterion]` | `name`, `value: Finite`, `threshold: str`, `met: bool` |
| `context` | `dict[str, str]` | reported, not decisive: fit, fixed subset, global-selector payoff, notes |
| `evidence` | `list[str]` | run ids of the result files |
| `adr` | `str` | path of the proposed ADR |
| `tag` | `str` | `g2-selector-shift` |

The outcome is a draft: the researcher accepts or amends it in the ADR, as for G1.

## Behavior result (`behavior.json`, plain JSON with the run manifest)

`battery` (ids per environment, n, seed, SHA-256), `pairs` (36 rows: selector a, selector b,
within/cross, overlap@k, JS, KL a‖b, KL b‖a, Spearman of mean importance, Spearman of selection
frequency), `statistic` T, `assignments` 1,680, `p_value`, and per environment pair the cross and
within overlaps.

## Relationships

```text
F3 candidates selection ─▶ seeded SelectorCheckpoint ×3 per env ─┬▶ SelectorProperties (battery)
                                                                 ├▶ EvaluationResult per method ─▶ TransferCell
                                                                 └▶ fixed subset (training queries)
EvaluationResult, TransferCell, behavior.json, fit.json ─▶ reports/E1 ─▶ GateDecision ─▶ ADR (proposed)
```
