# Quickstart: validating F2

Commands follow [contracts/cli.md](contracts/cli.md).

## 1. Tests (any machine, CPU, no network)

```bash
uv sync --locked --extra cpu --extra dev --extra data
uv run pytest            # includes the F2 unit and integration tests with fake sources
```

## 2. SciFact and NFCorpus (RTX 4090 workstation)

```bash
uv sync --locked --extra cu128 --extra dev --extra data --extra embed
for env in scifact nfcorpus; do
  uv run hyperdime prepare-data data=$env
  uv run hyperdime embed data=$env
  uv run hyperdime mine-negatives data=$env
done
```

**Expected**: `dataset.json` shows SciFact 5,183 documents and 300 test queries, NFCorpus 3,633
and 323 (SC-001); each embeddings artifact has unit-norm 1024-wide rows; every pool has 64 entries.
A second `embed` prints the existing directory without recomputing.

SC-004 check against the F0 spike's SciFact vectors (needs the F0 caches):

```bash
uv run python scripts/checks/f0_agreement.py artifacts/spike/scifact "$(ls -d artifacts/embeddings/scifact/2* | tail -1)"
```

**Expected**: minimum cosine ≥ 0.999 for documents and queries.

## 3. MS MARCO and the TREC DL sets

```bash
uv run hyperdime prepare-data data=msmarco
uv run hyperdime embed data=msmarco --shard-size 65536   # resumable; rerun the same command after an interruption
uv run hyperdime mine-negatives data=msmarco
```

**Expected**: 8,841,823 documents, 6,980 test queries, and DL19/DL20/DL-HARD with 43/54/50 judged
queries (SC-001); the run stays within 24 GB of GPU memory; after a `Ctrl-C` and rerun, completed
shards are skipped (SC-005).
