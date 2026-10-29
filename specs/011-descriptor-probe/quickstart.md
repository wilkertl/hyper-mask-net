# Quickstart: F10 Descriptor Probe (experiment E2, gate G3)

## Local checks (CPU, no network)

```bash
ruff check . && ruff format --check . && mypy && pytest
pytest tests/unit/test_space_*.py tests/unit/evaluation -q
pytest tests/integration/test_f10_pipeline.py -q      # synthetic environments end to end
PYTHONPATH=tools python -m scrap tests
```

Expected, each with a hand-computable fixture:

- moments, skewness, kurtosis, tail share, quantiles, correlations, PC share, products, and
  contrast match hand-computed values on small matrices; streamed equals one pass; on ≤ 1M
  documents the nine marginal columns equal `marginal_stats` (research D3–D5);
- descriptors are unchanged by document or query row order and permute with dimensions;
- `describe-space` gives the same descriptor with every qrels file deleted, and never opens a
  corrupted one (D2);
- average ranks with ties, ridge on a 3-row example, GCV and the inner penalty choice,
  Spearman–Brown, and the permutation p on a hand-counted null (D7, D9, D10);
- the shift check reports no room when environments share one profile plus seed noise, and the
  gap when one has its own component (D0); the shift probe with `β = 0` equals B1 (D10);
- on synthetic environments whose target is a linear function of one descriptor column plus
  noise plus a shared profile, the shift probe of set 2 beats B1 and B2 with small p; with an
  unrelated target it does not; the held-out environment's rows
  never enter standardization or penalty choice (D1, D7);
- E0, E1, P1, S1–S4 reports regenerate byte-identically (FR-008).

## Runs (DataPolicy-4090; the GPU is shared)

```bash
hyperdime probe-descriptors --experiment E2 --targets-only     # shift check first (D0)
hyperdime report E2                                            # read check.csv before going on
for env in scifact nfcorpus msmarco; do hyperdime describe-space data=$env --protocol P0 --chunk-size 250000; done
hyperdime probe-descriptors --experiment E2
hyperdime report E2
hyperdime decide G3 --adr docs/decisions/ADR-0013-g3-space-signal.md
for r in E0 E1 P1 S1 S2 S3 S4; do hyperdime report $r; done
git diff --exit-code reports/E0 reports/E1 reports/P1 reports/S1 reports/S2 reports/S3 reports/S4
```

Stage 2 reruns `describe-space --protocol P1` and the probe. Expected: `reports/E2/README.md`
with the shift check, the criteria table (per fold: the best set's shift-probe Spearman against
B1 and B2, its Holm p, the per-seed check), the ladder by protocol and probe form,
the noise ceilings, coefficients, the environment-level table, and the note that these
environments shaped S1–S4; other reports unchanged; descriptor and probe times within SC-003.
