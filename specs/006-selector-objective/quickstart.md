# Quickstart: F5 Selector Objective and Experiment E2

## Local checks (CPU, no network)

```bash
ruff check . && ruff format --check . && mypy && pytest
pytest tests/unit/selection/test_relaxed.py tests/unit/training -q   # mask and loss by hand
pytest tests/integration/test_f5_pipeline.py -q                      # synthetic end to end
PYTHONPATH=tools python -m scrap tests
```

Expected: the relaxed mask is exactly the hard top-k above 0.5 and converges to it as β → 0; on the
synthetic environment the selector recovers ≥ 90% of the planted dimensions (SC-005); removing
test judgments leaves checkpoints identical (FR-014); `report E2` is byte-identical on rerun
(SC-003); a planted improvement in one synthetic environment is found there and nowhere else.

## Runs (DataPolicy-4090; wait for ≥ 20 GB free GPU memory, the GPU is shared)

```bash
for env in scifact nfcorpus msmarco; do
  hyperdime train-selector --recipe retrieval-grid data=$env
  for s in 0 1 2; do hyperdime train-selector --recipe retrieval-seeded data=$env run.seed=$s; done
done
hyperdime compare-selectors
hyperdime report E2
hyperdime decide E2
```

Expected: `reports/E2/README.md` with the nDCG@10-by-budget table per test set, the comparison
against L2S per environment (mean over 10%–90% and k = 307, CI, Holm p), the family table, the
overlap and fit diagnostics, and both E2 criteria; `docs/decisions/ADR-0009-e2-selector-objective.md`
proposed. Record each seed's training time and `compare-selectors` wall time against SC-006.
