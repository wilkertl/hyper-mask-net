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

Install PyTorch for your platform first (CUDA or CPU build), then:

```bash
python -m pip install -e ".[dev]"
ruff check . && ruff format --check . && mypy && pytest
```

Datasets, embeddings, and checkpoints are never committed; they live under `artifacts/`.

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
(`pip install -e ".[embed]"`). Without the server, embedding falls back to local transformers.

## F0 spike: test H1 first

Before the roadmap is built, a spike tests selector shift (H1) on five small BEIR datasets
([research R0](specs/002-hyperdime-roadmap/research.md)):

```bash
python -m pip install -e ".[embed,dev]"
python -m hyperdime.spike all    # prepare → train → train-global → analyze → report
```

Only `prepare` uses the embedding service; the rest runs on CPU from cached vectors. The generated
report is `reports/F0/README.md`, ending in the H1 verdict.
