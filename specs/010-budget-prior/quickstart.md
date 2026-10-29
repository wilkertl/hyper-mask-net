# Quickstart: F9 Budget-Conditioned Prior Weight (experiment S4)

## Local checks (CPU, no network)

```bash
ruff check . && ruff format --check . && mypy && pytest
pytest tests/unit/test_geometric_selector.py tests/unit/training tests/unit/evaluation -q
pytest tests/integration/test_f5_pipeline.py -q        # synthetic end to end, S1–S4
PYTHONPATH=tools python -m scrap tests
```

Expected, each with a hand-computable fixture:

- an untrained `BudgetGeometricSelector` ranks as `variance` at all nine budgets; with equal
  weights `a` it ranks as `GeometricSelector(prior_weight=a)`; changing `a_k` changes only budget
  k's mask; an out-of-grid `k` raises; a weight pushed negative is projected to 0 (D1, D2);
- `prior_share` is 1 at the start and unchanged when `a`, `W`, `b` are scaled together (D5);
- `noninferiority_test` on a 4-query example gives the hand-counted p (D7);
- on a synthetic set where rerank and full-corpus nDCG@10 disagree, `checkpoint_rule="validation"`
  keeps the full-corpus best and can keep step 0; the guard reads the kept state (D3, D5);
- default configs leave S1–S3 histories, checkpoints, and reports byte-identical (FR-005);
- removing test judgments leaves the new grids' checkpoints identical (constitution II).

## Runs (DataPolicy-4090; wait for ≥ 20 GB free GPU memory, the GPU is shared)

```bash
for env in scifact nfcorpus msmarco; do
  for track in weighted-aligned budget; do
    hyperdime train-selector --recipe $track-grid data=$env
    for s in 0 1 2; do hyperdime train-selector --recipe $track-seeded data=$env run.seed=$s; done
  done
done
hyperdime compare-selectors --experiment S4 --chunk-size 250000
hyperdime report S4
hyperdime decide S4 --adr docs/decisions/ADR-0012-s4-budget-prior.md
hyperdime report S1 && hyperdime report S2 && hyperdime report S3 && git diff --exit-code reports/S1 reports/S2 reports/S3
```

Expected: `reports/S4/README.md` with the criteria table (non-inferiority at δ = 0.005 against the
aligned control, S2, and `variance`; superiority against the aligned control), per-budget
differences, learned `a_k`, the prior's share, the prior-removal ablation, kept steps, and the
note that these test sets informed the proposal; S1–S3 reports unchanged; grid times within
SC-003. On MS MARCO the budget selector is expected to stay at `variance` (research D9.2).
