# Quickstart: validating F4

Commands follow [contracts/cli.md](contracts/cli.md); rules follow [research.md](research.md).

## 1. Tests (any machine, CPU, no network)

```bash
uv sync --locked --extra cpu --extra dev
uv run pytest     # includes the F4 unit tests and the synthetic E1 pipeline
```

**Expected**: on synthetic environments with a planted per-environment subset, both G2 criteria
hold; on copies of one environment, neither holds (SC-004). Removing test judgments leaves the
seeded selectors identical, and `transfer-matrix` changes no checkpoint (SC-003).

## 2. Seeded selectors (DataPolicy-4090, after F3's artifacts)

```bash
uv sync --locked --extra cu128 --extra dev
for env in scifact nfcorpus msmarco; do
  for seed in 0 1 2; do
    uv run hyperdime train-selector --recipe seeded data=$env run.seed=$seed
  done
done
```

**Expected**: nine artifacts under `artifacts/selectors/seeded/`, each with a complete manifest
recording its seed; each `selection.json` names the F3 `wide-grid` cell it trained.

## 3. Transfer matrix and behavior

```bash
uv run hyperdime transfer-matrix
```

**Expected**: 9 test-query cells plus the TREC DL cells in `cells.jsonl` (SC-001), a 300-query-per-
environment battery (SciFact's 300 test queries set n), 36 behavior pairs, and `p_value` from
1,680 assignments. Completes within 30 minutes (SC-006).

## 4. Report and G2 draft

```bash
uv run hyperdime report E1
uv run hyperdime report E1 && git diff --exit-code reports/E1   # identical bytes (SC-002)
uv run hyperdime decide G2
```

**Expected**: `reports/E1/README.md` with both G2 criteria computed (SC-005), MS MARCO rows carrying
the research D9 note, NFCorpus rows its caveat; a proposed ADR for the researcher's decision.
