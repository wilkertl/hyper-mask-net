# Data Model: F9 Budget-Conditioned Prior Weight (experiment S4)

Existing records are reused: `SelectorCheckpoint`, `EvaluationResult`, `GateDecision`,
`Criterion`, run manifests. One additive contract change: `GateDecision.gate` adds `"S4"`.

## BudgetGeometricSelector (`baselines/learning_to_select.py`, research D1)

| Field | Kind | Shape | Start | Rules |
|---|---|---|---|---|
| `projection.weight`, `projection.bias` | parameters | `[1024, 1024]`, `[1024]` | 0 | the L2S linear map, shared by every budget |
| `prior_weights` | parameter | `[9]` | 1 | one per budget, projected to `≥ 0` after each step (D2) |
| `variance` | buffer | `[1024]` | corpus variance | fixed, from the environment's document embeddings |
| `budgets` | buffer (int64) | `[9]` | `BUDGETS` (102 … 921) | strictly increasing; `forward` refuses any other `k` |

`forward(queries, k)` → `[Q, 1024]` log-importance. Invariants: at the start it ranks as
`variance` for every k; with all weights equal to `a` it ranks as `GeometricSelector` with
`prior_weight = a` and the same map. Saved with `save_selector`; `load_selector` builds it when the
state has `budgets`, a `GeometricSelector` when it has `variance` only.

## RetrievalConfig (additions, `training/retrieval_trainer.py`)

| Field | Values | Default |
|---|---|---|
| `checkpoint_rule` | `rerank` (S1–S3), `validation` (D3) | `rerank` |
| `nonnegative_prior` | project prior weights to `≥ 0` after each step (D2) | `False` |

## RetrievalCell (additions, `training/retrieval_selection.py`)

| Field | Values |
|---|---|
| `init` | adds `budget` (S4) and `weighted-aligned` (control); both in `PRIOR_INITS` |
| `variant` | `multi-budget` only for the two new inits (D6) |

`retrieval_cells(inits=("budget",))` and `(("weighted-aligned",))` give 8 cells each.

## TrainedRetrieval (additions)

| Field | Meaning |
|---|---|
| `candidates` | `[(step, state)]`, step 0 and every `eval_every`; empty under `rerank` |
| `kept_step` | step of the kept state (from history under `rerank`, from the candidate pass under `validation`) |
| `candidate_scores` | `{step: mean nDCG@10 over budgets}` on the D4 subset |
| `soft_mass` | under `validation`: the kept state's soft mass at `β_end` (D5); under `rerank`: `history[-1]` as before |

## Grid `selection.json` (additions, both new tracks)

| Key | Content |
|---|---|
| `checkpoint_rule` | `"validation"` |
| `candidate_queries` | the D4 subset's query ids and its seed |
| `cells.<key>.candidate_scores` | per step, subset nDCG@10 mean over budgets, and the rerank score logged at that step |
| `cells.<key>.kept_step`, `cells.<key>.soft_mass` | as above |
| `cells.<key>.prior_weights` | kept state's weights (nine for `budget`, one for `weighted-aligned`) |
| `cells.<key>.projected_steps` | number of steps at which the D2 projection changed a weight |

Seeded `selection.json` carries the same per-cell entries for its one cell.

## `evaluation/S4/all/<run_id>/` (from `compare-selectors --experiment S4`)

| File | Content |
|---|---|
| `results.json` | `EvaluationResult`s per environment, query set, method, budget (test and TREC DL sets): `budget:<env>:s<n>`, `weighted-aligned:<env>:s<n>`, the S3/S2/S1/L2S seeds, `variance`, `full`, `prefix`, `random`, global methods |
| `diagnostics.json` | per environment, seed, budget: `prior_share`, overlap with `variance`, overlap with the correction alone, `a_k`; validation queries only (D5) |
| `ablation.json` | per environment, seed, budget: validation nDCG@10 of the kept state and of the same state with every `a_k = 0` (D5) |
| `overlap.json` | as S3, candidate `budget` |
| `sources.json` | every grid and seeded run read, with payload hashes |

## `reports/S4/` (from `report S4`)

`README.md` and `tables/`: `progression.csv`, `comparison.csv` (S4 against aligned S3, S3, S2,
`variance`, S1, L2S: mean over budgets and k = 307; nDCG@10, MRR@10, Recall@100; diff, CI, p,
Holm), `noninferiority.csv` (δ, one-sided p, Holm, met), `per_budget.csv` (S4 − aligned S3 per k),
`weights.csv`, `prior_share.csv`, `ablation.csv`, `kept_steps.csv`, `vs_full.csv`, `trec.csv`,
`s4.csv` (criteria). Identical bytes when run twice.

## GateDecision (`decisions/S4/<run_id>/gate_decision.json`)

`gate = "S4"`; criteria SC-001 (non-inferiority against aligned S3, S2, `variance`) and SC-002
(superiority against aligned S3, ≥ 1 of 3); context entries: weights, prior share, ablation, kept
steps, MS MARCO note, the test-set reuse note. Outcome is `pass` only when both criteria hold.
