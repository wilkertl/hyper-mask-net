# HyperDIME-Qwen

Environment-conditioned hypernetwork for query-aware dimension selection on a frozen
**Qwen3-Embedding-0.6B** (D = 1024).

A *Space Descriptor* summarizes how a retrieval environment — its corpus, query distribution, and
instruction — uses each of Qwen's 1024 coordinates. A *Space Encoder* and a *Hyper Head* turn that
description into a low-rank modification of a Learning-to-Select dimension selector, once per
environment. At query time the selector scores the dimensions, a selection policy keeps the top-k,
and exact retrieval runs over the unchanged document embeddings.

> **Status:** Milestone 1 (Reproduction), Phase A — not started. The repository was restarted on
> 2026-09-15 ([ADR-0001](docs/decisions/ADR-0001-restart-as-hyperdime.md)). Only the tested math
> utilities carried over from the previous prototype exist so far.

## Research questions

The hypernetwork is built only if each earlier link holds:

1. **H1 — Selector shift:** the optimal selector for the same Qwen model changes across environments.
2. **H2 — Geometry explains the shift:** statistics of the embedding space predict part of that change.
3. **H3 — Hypernetwork adaptation:** a Hyper Head maps environment statistics to selector parameters
   that generalize better than a fixed global selector.

Each hypothesis has a Go/No-Go gate; see [docs/architecture.md](docs/architecture.md#gates).

## Documentation

| Document | Purpose |
|---|---|
| [Implementation plan](docs/plan/Qwen3_HyperDIME_Implementation_Plan.en.md) | Full plan (English translation; Portuguese original and `.docx` alongside) |
| [Architecture](docs/architecture.md) | Components, module map, invariants, gates |
| [Protocols](docs/protocols.md) | Environments, splits, adaptation protocols P0–P2, leakage and reporting rules |
| [Workflow](docs/workflow.md) | Branches, issues, labels, milestones, pull requests, wiki |
| [Backlog](docs/backlog/issues.md) | Initial GitHub issues, published by `scripts/github/create_issues.py` |
| [Experiment registry](docs/experiment_registry.md) | Experiments E0–E6 and links to their reports |
| [Research references](docs/references.md) | Bibliography and links for the project's source papers |
| [Decisions](docs/decisions/) | Architecture decision records and gate decisions |
| [AGENTS.md](AGENTS.md) | Mandatory rules for coding agents |
| [Wiki](https://github.com/wilkertl/hyper-mask-net/wiki) | Navigable overview, generated from `docs/wiki/` |

## Layout

```text
src/hyperdime/
├── contracts/    M0      schemas, run manifests, validation
├── data/         M1      loaders, splits, hard-negative pools
├── embeddings/   M2      frozen Qwen encoding and caches
├── oracle/       M3      importance targets
├── baselines/    M3      Learning-to-Select, MRL prefix, random, global, oracle
├── evaluation/   M4/M12  metrics, selector-shift analysis, statistics
├── space/        M5/M6   space descriptor and encoder
├── hypernet/     M7/M8   low-rank selector and Hyper Head
├── training/     M3/M9   selector and episodic meta-training
├── selection/    M11     selection policies
├── retrieval/    M12     exact scoring (FAISS for deployment analysis only)
└── cli/                  entry points
configs/  experiments/  tests/  scripts/  docs/
```

## Development

Dependencies are locked in `uv.lock` ([uv](https://docs.astral.sh/uv/)). Pick the PyTorch build
with one extra, `cpu` or `cu128` (CUDA 12.8); they cannot be combined.

```bash
uv sync --locked --extra cpu --extra dev --extra data --extra science   # CPU; every test
uv sync --locked --extra cu128 --extra dev    # NVIDIA GPU
uv run pre-commit install                     # run the checks below before each commit
uv run hyperdime diagnose                     # smoke test: validates configs/, writes a manifest
```

`hyperdime diagnose` prints `artifacts/diagnose/<run_id>` and writes only its `manifest.json`;
it loads no model and needs no network. The checks CI runs, also run by pre-commit:

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest
PYTHONPATH=tools uv run python -m scrap tests   # test-code quality (ADR-0004)
```

Extras for later milestones are locked too: `embed` (transformers), `data` (datasets,
ir-datasets), `science` (scipy, scikit-learn, ir-measures), and `deploy` (faiss-cpu).
Datasets, embeddings, and checkpoints are never committed; they live under `artifacts/`.

## Data and embeddings (F2)

Each environment (`scifact`, `nfcorpus`, `msmarco`) is prepared, embedded, and given hard-negative
pools once, on one GPU machine ([F2 quickstart](specs/003-data-embeddings/quickstart.md)):

```bash
uv sync --locked --extra cu128 --extra data --extra embed --extra dev
uv run hyperdime prepare-data data=scifact     # pinned Hugging Face sources -> artifacts/datasets/
uv run hyperdime embed data=scifact            # resumable shards -> artifacts/embeddings/
uv run hyperdime mine-negatives data=scifact   # ADR-0002 pools -> artifacts/negatives/
```

Test judgments are written to their own files and are read only by evaluation. `embed` resumes
after an interruption and reuses an existing embedding made with the same settings.

## Baseline selector and gate G1 (F3)

```bash
for env in scifact nfcorpus msmarco; do uv run hyperdime train-selector data=$env --recipe official; done
uv run hyperdime train-global                     # anchor for the ADR-0003 regularized candidate
for env in scifact nfcorpus msmarco; do
  uv run hyperdime train-selector data=$env --recipe candidates
  uv run hyperdime evaluate data=$env             # every method at ratios 0.1-1.0, exact scoring
done
uv run hyperdime report E0                        # reports/E0 with the G1 criteria
```

Selectors are chosen on validation queries only; `evaluate` is the only step that reads test
judgments.

## Selector shift and gate G2 (F4)

```bash
for env in scifact nfcorpus msmarco; do
  for seed in 0 1 2; do uv run hyperdime train-selector --recipe seeded data=$env run.seed=$seed; done
done
uv run hyperdime transfer-matrix                  # every selector on every environment, k = 10%…100%
uv run hyperdime report E1                        # reports/E1 with both G2 criteria
uv run hyperdime decide G2                        # GateDecision and a proposed ADR-0007
```

`transfer-matrix` trains nothing; test judgments only score rankings, and the behavior battery
reads query embeddings only. Rules are fixed in `specs/005-selector-shift/research.md`.

## A selector trained for retrieval, experiment S1 (F5)

```bash
for env in scifact nfcorpus msmarco; do
  uv run hyperdime train-selector --recipe retrieval-grid data=$env     # 32 cells, seed 0
  for seed in 0 1 2; do uv run hyperdime train-selector --recipe retrieval-seeded data=$env run.seed=$seed; done
done
uv run hyperdime compare-selectors               # retrieval selector vs L2S, k = 10%…100%
uv run hyperdime report S1                       # reports/S1 with both S1 criteria
uv run hyperdime decide S1                       # GateDecision(gate="S1") and a proposed ADR-0009
```

The selector is the same linear map as Learning-to-Select, trained on masked retrieval through a
relaxed top-k mask; evaluation keeps the hard top-k mask. Rules are fixed in
`specs/006-selector-objective/research.md`; ADR-0008 explains why S1 comes before G3.

## Corpus geometry probe, experiment P1 (F6)

```bash
uv run hyperdime probe-geometry                  # corpus moments, covariance spectrum, q²·Var(x) at k = 10%…90%
uv run hyperdime report P1                       # reports/P1, paired with the latest S1 run
```

## A selector started from the geometric prior, experiment S2 (F7)

```bash
for env in scifact nfcorpus msmarco; do
  uv run hyperdime train-selector --recipe geometric-grid data=$env     # 16 cells from the prior
  for seed in 0 1 2; do uv run hyperdime train-selector --recipe geometric-seeded data=$env run.seed=$seed; done
done
uv run hyperdime compare-selectors --experiment S2   # geometric vs S1 selector, variance, L2S
uv run hyperdime report S2                           # reports/S2 with both S2 criteria
uv run hyperdime decide S2                           # GateDecision(gate="S2") and a proposed ADR-0010
```

The geometric selector scores `log(q_d² · Var(x_d) + ε) + (W q + b)_d`: the corpus variance is a
fixed prior from document embeddings, and `W, b` start at zero, so training learns a correction.

## A learned weight on the geometric prior, experiment S3 (F8)

```bash
for env in scifact nfcorpus msmarco; do
  uv run hyperdime train-selector --recipe weighted-grid data=$env      # 16 cells, a starts at 1
  for seed in 0 1 2; do uv run hyperdime train-selector --recipe weighted-seeded data=$env run.seed=$seed; done
done
uv run hyperdime compare-selectors --experiment S3   # weighted vs S1, S2 selectors, variance, L2S
uv run hyperdime report S3                           # reports/S3 with both S3 criteria
uv run hyperdime decide S3                           # GateDecision(gate="S3") and a proposed ADR-0011
```

The weighted selector scores `a · log(q_d² · Var(x_d) + ε) + (W q + b)_d`, with the scalar `a`
learned from 1, so training can turn the prior down where the ranking must be reshaped.

## Embedding service

Qwen3-Embedding-0.6B is served by the same vLLM server as the `beir` project, through its
OpenAI-compatible `POST {base_url}/embeddings`, with the same environment variables:

| Variable | Meaning |
|---|---|
| `V4_EMBEDDING_BASE_URL` | endpoint, e.g. `http://127.0.0.1:8000/v1` (the local end of `ssh -L 8000:localhost:8000 <host>`) |
| `V4_EMBEDDING_API_KEY` | bearer token, only if the server requires one; never passed on the command line |
| `V4_EMBEDDING_TUNNEL_HOSTS` | comma-separated non-loopback hosts reached over an encrypted tunnel, allowed over plain http |

The endpoint must be `https://`, loopback, or a declared tunnel host. Check it before a run:

```bash
curl -s $V4_EMBEDDING_BASE_URL/models    # must list Qwen/Qwen3-Embedding-0.6B
```

Texts are cut to 512 Qwen tokens on the client, which needs `transformers` for the tokenizer only
(the `embed` extra). Without the server, embedding falls back to local transformers.

## F0 spike: test H1 first

Before the roadmap is built, a spike tests selector shift (H1) on five small BEIR datasets
([research R0](specs/002-hyperdime-roadmap/research.md)):

```bash
uv sync --locked --extra cu128 --extra embed --extra dev   # or --extra cpu
uv run python -m hyperdime.spike all    # prepare → train → train-global → analyze → report
```

Only `prepare` uses the embedding service; the rest runs on CPU from cached vectors. The generated
report is `reports/F0/README.md`, ending in the H1 verdict.
