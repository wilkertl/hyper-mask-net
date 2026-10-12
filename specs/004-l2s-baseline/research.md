# Research: F3 Learning-to-Select Baseline and Gate G1

## D1. Evaluation in one corpus pass

- **Decision**: For one environment and query set, every (method, k) pair produces a masked copy of
  the query matrix; all copies are stacked and ranked in one `exact_search_streamed` pass over the
  memory-mapped F2 document matrix (depth 100; 1,000 for the TREC DL sets, for AP). Masked queries
  keep their unselected coordinates at zero and are not re-normalized (constitution I).
- **Rationale**: Reading 18 GB of MS MARCO vectors once instead of once per (method, k) makes the
  full E0 grid a few corpus passes. k = 1024 masks are checked against the unmasked rankings
  query by query (G1 criterion b).

## D2. Metrics

- **Decision**: `query_metrics` gains AP (over the ranking provided) and a `relevance_threshold`
  for binary metrics (AP, MRR, Recall): 2 for the TREC DL sets (TREC convention), 1 otherwise;
  nDCG keeps graded gains. The defaults reproduce today's values, so the spike is unchanged. Tests
  compare against `ir-measures` (locked `science` extra, now installed in CI) on fixtures.

## D3. Methods

| Method | Importance per query | Notes |
|---|---|---|
| `full` | none (k = 1024) | reference |
| `prefix` | 1024 − i for dimension i | MRL first-k (risk R5) |
| `random` | seeded uniform scores per query | seed 0 |
| `global-importance` | mean training oracle target of the environment | one ranking for all queries |
| `l2s` | official-recipe selector | ADR-0002 |
| `selected` | the validation-best of `l2s` and the ADR-0003 candidates | input to F4 |
| `oracle` | raw `e_q ⊙ (p − n)` from evaluation labels and negatives mined for those queries | non-deployable upper bound |

Top-k always goes through `selection.topk.top_k_mask` (constitution VII).

## D4. Training and the ADR-0003 candidates

- Targets: `aggregate_targets` (R4, ADR-0002) over training and validation queries with the F2
  pools; positive and negative document vectors are gathered from the memory-mapped matrix.
- Trainer: today's `train_selector` gains a `device`, an optional anchor (`W_0`, `b_0`) with
  weight `l2_to_anchor`, and an optional `patience` with a constant learning rate; defaults
  reproduce ADR-0002 exactly.
- Selection score: validation nDCG@10 at k = 307 with training labels, over the full corpus.
- Families, each tuned on validation:

  | Family | Grid |
  |---|---|
  | `official` | τ ∈ {0.005, 0.01, 0.02, 0.05, 0.1, 0.2} × epochs ∈ {10, 50, 200} (ADR-0002) |
  | `wide-grid` | τ ∈ {0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0} × epochs ∈ {3, 10, 50, 200, 500} |
  | `val-stopped` | official τ; constant learning rate, up to 500 epochs, stop when validation KL has not improved for 5 epochs |
  | `global-regularized` | official τ and epochs; `l2_to_anchor` ∈ {1e-4, 1e-3, 1e-2, 1e-1} toward the global selector |

- The global selector is one linear selector trained on the union of the three environments'
  training targets (each at its own official τ), validated on the union of their validation sets;
  it is never trained on a test label.
- `selected` = the cell with the best validation score over all families; ties go to the official
  recipe.

## D5. Commands and artifacts

| Command | Writes |
|---|---|
| `hyperdime train-selector data=<env> --recipe official` | `selectors/<env>/<run_id>/`: `official.safetensors`, `SelectorCheckpoint`, `selection.json` (every cell's validation score) |
| `hyperdime train-global` | `selectors/global/<run_id>/` |
| `hyperdime train-selector data=<env> --recipe candidates` | `selectors/<env>/<run_id>/` with each family's best and `selected` |
| `hyperdime evaluate data=<env>` | `evaluation/<env>/<run_id>/results.jsonl` (`EvaluationResult` per method, k, query set), `checks.json` |
| `hyperdime report E0` | `reports/E0/README.md`, `reports/E0/tables/*.csv` |

## D6. Report

- Learning-to-Select-style table: SciFact, NFCorpus, MS MARCO (dev), nDCG@10 at 30% and 100%.
- DIME-style table: DL 2019, DL 2020, DL-HARD, nDCG@10 and AP at ratios 0.1–1.0.
- Significance at 30%: paired randomization (10,000 sign flips) and 95% bootstrap CI against
  `full` and against `l2s`, Holm within each table.
- G1 (research R7): (a) validation KL < uniform for `l2s` in every environment; (b) k = 1024
  identical to unmasked for every method and query set; (c) nDCG@10 at k = 256: oracle ≥ l2s ≥
  random in every environment; (d) regenerating the report gives identical bytes (checked by
  generating twice).
