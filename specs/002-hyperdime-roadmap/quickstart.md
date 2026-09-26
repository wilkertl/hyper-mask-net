# Quickstart: validating the roadmap, gate by gate

How to check that each milestone delivered what the roadmap promises. Commands follow
[contracts/cli.md](contracts/cli.md); criteria come from [research.md](research.md). Each milestone
feature's own `quickstart.md` has the detailed steps.

## Prerequisites

- Clean clone, `uv sync --locked --extra dev --extra cu128` (or `--extra cpu` for tests only).
- One GPU for embedding and training; network only for `prepare-data`.
- ≈ 20 GB free disk before G2 and ≈ 60 GB after (research R17).
- Optional: the licensed TREC disks 4 and 5 at a local path, for Robust04.
- Every step below writes artifacts with manifests under `artifacts/`; reports go to `reports/`.

## Spike F0 → H1 verdict (run first)

The model comes from the same vLLM server as the `beir` project, with the same environment
variables (see `src/hyperdime/embeddings/remote.py`):

```bash
export V4_EMBEDDING_BASE_URL=http://127.0.0.1:8000/v1   # e.g. the local end of an ssh -L tunnel
# export V4_EMBEDDING_API_KEY=...                        only if the server requires one
# export V4_EMBEDDING_TUNNEL_HOSTS=10.0.0.5              only for plain http to a non-loopback tunnel host
curl -s $V4_EMBEDDING_BASE_URL/models                    # must list Qwen/Qwen3-Embedding-0.6B

pip install -e ".[embed,dev]"                  # transformers is needed only for the tokenizer
pytest                                         # 52 tests, CPU only, no network, fake vLLM
python -m hyperdime.spike all                  # prepare → train → train-global → analyze → report
```

With `V4_EMBEDDING_BASE_URL` set, `prepare` embeds through vLLM (`--backend vllm`); without it,
it falls back to local transformers (`--backend local`, GPU if available). Texts are cut to
`--max-length` (512) Qwen tokens on the client, so results do not depend on the server's context
length. Only `prepare` needs the server; training and analysis run on CPU from the cached vectors.

`all` downloads the five BEIR datasets, embeds full corpora, trains 3 seeds per environment plus
the global selector, and writes `reports/F0/README.md`. Steps also run one at a time (`prepare`,
`train`, `train-global`, `analyze`, `report`), with `--env <name>` repeatable. Embedding resumes
from saved chunks after an interruption; reusing a cache built with a different backend, model,
`--max-length`, or `--corpus-cap` fails until that cache directory is deleted.

**Expected**: `reports/F0/README.md` shows the 5 × 5 transfer matrix, the baseline rows,
cross-environment vs cross-seed mask overlap, the per-domain vs global headroom, and the H1 verdict
under research R0's rule. **Continue below only if H1 is supported.**

## Milestone 1 → G1 (features F1–F3)

```bash
uv run pytest                                              # all unit + integration tests on CPU
uv run hyperdime diagnose                                  # F1: manifest written, exit 0
for env in scifact nfcorpus msmarco; do
  uv run hyperdime prepare-data data=$env
  uv run hyperdime embed data=$env
  uv run hyperdime mine-negatives data=$env
  uv run hyperdime build-targets data=$env
  uv run hyperdime train-selector data=$env selector=linear
done
uv run hyperdime evaluate experiment=E0                   # includes TREC DL'19, DL'20, DL-HARD over MS MARCO
uv run hyperdime report E0 && git diff --exit-code reports/E0   # regenerates byte-identically
uv run hyperdime decide G1
```

**Expected**: E0 shows, for each environment, validation KL below uniform, k=1024 identical to
unmasked, and oracle ≥ linear ≥ random at k=256. It has a Learning-to-Select-style table
(SciFact, NFCorpus, MS MARCO; 30% ratio row) and a DIME-style table (DL'19, DL'20, DL-HARD;
nDCG@10 and AP over retained-dimension ratios). The G1 ADR records pass/fail.

## Milestone 2 → G2 (F4)

```bash
uv run hyperdime train-selector experiment=E1 seeds=[0,1,2]   # per-domain selectors
uv run hyperdime transfer-matrix experiment=E1
uv run hyperdime report E1 && uv run hyperdime decide G2
```

**Expected**: E1 contains a 3×3 (train × test) matrix per seed with Holm-adjusted p-values and CIs,
plus seed-null mask-overlap comparisons. **Stop here on a No-Go.**

## Milestone 3 → G3 (F5)

```bash
for env in nq hotpotqa fever climate-fever fiqa dbpedia quora scidocs arguana touche2020 trec-covid; do
  uv run hyperdime prepare-data data=$env && uv run hyperdime embed data=$env   # climate-fever reuses fever's corpus embedding
done
uv run hyperdime prepare-data data=robust04 data.local_path=/path/to/disks45   # skipped with a message if absent
uv run hyperdime embed data=robust04
# merge ADR-XXXX-environment-pool.md and ADR-XXXX-meta-split.md before continuing
uv run hyperdime build-space protocol=P0 && uv run hyperdime build-space protocol=P1
uv run hyperdime probe experiment=E2
uv run hyperdime report E2 && uv run hyperdime decide G3
```

**Expected**: invariance tests pass; E2 reports leave-one-environment-out probe scores vs the
trivial baseline with permutation p-values; no meta-test environment appears in E2.

## Milestone 4 → meta-validation verdict (F6)

```bash
uv run hyperdime train-hypernet experiment=meta
uv run hyperdime train-hypernet experiment=meta data.meta_train=[trec-covid]   # must exit 2
uv run hyperdime evaluate experiment=meta-validation
```

**Expected**: zero-gate generation equals the global selector (unit test); the meta-validation
table says match / beat / lose vs the global selector. Meta-test stays locked unless match or beat.

## Milestone 5 → G4, G5 (F7)

```bash
uv run hyperdime adapt protocol=P0 && uv run hyperdime adapt protocol=P1
uv run hyperdime evaluate experiment=E3
uv run hyperdime evaluate experiment=E4 && uv run hyperdime evaluate experiment=E5
uv run hyperdime benchmark experiment=E6
for e in E3 E4 E5 E6; do uv run hyperdime report $e; done
uv run hyperdime decide G4 && uv run hyperdime decide G5
```

**Expected**: E3 has all eight method rows per meta-test environment, each labeled with its
protocol; Climate-FEVER appears in a separate query-shift-only table; every dataset from the
reference papers appears with that paper's metric (FR-019); E6 separates offline from online cost and dense from restricted-coordinate scoring;
`git diff --exit-code reports/` is clean after regenerating.
