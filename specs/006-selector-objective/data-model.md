# Data Model: F5 Selector Objective and Experiment E2

Existing records are reused: `SelectorCheckpoint`, `EvaluationResult`, `GateDecision`,
`Criterion`, run manifests. Only `GateDecision.gate` changes (adds `"E2"`).

## RetrievalExamples (in memory, `training/examples.py`)

| Field | Meaning | Rules |
|---|---|---|
| `queries` | `[N, 1024]` query embeddings | training (or validation) split only |
| `documents` | `[M, 1024]` unique gathered document rows | from the environment's embeddings |
| `positives` | per query, indices into `documents` and gain weights | grade > 0; weights `2**g − 1`, normalized per query |
| `negatives` | `[N, 64]` indices into `documents` | the F2 pool, repeats kept |
| `skipped` | count of queries without a positive in the corpus | recorded in `selection.json` |

Built from a `Supervision`-like loader that never receives test or evaluation-set judgments
(`load_inputs` already drops them).

## RetrievalCell

| Field | Values |
|---|---|
| `variant` | `fixed-307`, `multi-budget` |
| `init` | `scratch`, `l2s` |
| `lr` | `1e-4`, `1e-3` |
| `score_temperature` (τ_s) | `0.02`, `0.05` |
| `beta_end` | `0.1`, `0.02` |
| `steps` | `20000` (fixed) |

`key` = `<variant>-<init>-lr<lr>-ts<τ_s>-b<β_end>`; grid order is the product in the order above.

## Per-run outputs

`selectors/retrieval-grid/<env>/<run_id>/`

- `<cell key>/selector.safetensors`, `<cell key>/selector.json` (`SelectorCheckpoint`,
  `kind="per_domain"`).
- `history.json`: per cell, every 1,000 steps: training loss, validation pool-rerank nDCG@10
  (mean over budgets), soft-mask mass.
- `selection.json`: `objective: "retrieval"`, inputs (dataset, embeddings, negatives, E1 seeded
  run ids for the l2s start), per cell: best step, full-corpus validation nDCG@10 per budget and
  mean, guard value and pass/fail; the selected cell; the best cell per (variant, init) family;
  `skipped`.

`selectors/retrieval-seeded/<env>/seed-<n>/<run_id>/`: `selector.safetensors`, `selector.json`,
`selection.json` (cell, seed, history, validation scores, grid run id).

`evaluation/E2/all/<run_id>/`

- `results-<env>-<set>.jsonl`: `EvaluationResult` per method and budget (D7 method names).
- `overlap.json`: per environment and budget, retrieval↔L2S overlap and both noise floors.
- `fit.json`: per environment, oracle-mask agreement on validation for both selectors (E1 rule).
- `inputs.json`: every selector run id and payload SHA-256 read.

`decisions/E2/<run_id>/gate_decision.json`: `GateDecision(gate="E2", criteria=[SC-001, SC-002])`.

## State

`retrieval-grid` (seed 0, 32 cells, selection) → `retrieval-seeded` (selected cell, seeds 0–2) →
`compare-selectors` → `report E2` → `decide E2` (ADR-0009, proposed). `compare-selectors` refuses
when a seeded run's cell differs from the latest grid's selection or any payload hash mismatches.
