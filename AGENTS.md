# Agent rules

Mandatory for every coding agent, and for humans, working in this repository.

## Before editing

1. Read this file, [docs/architecture.md](docs/architecture.md), and [docs/protocols.md](docs/protocols.md).
2. Read the issue for your ticket and the contracts in `src/hyperdime/contracts/`.
3. List the files you intend to change in the issue or pull request before editing.
4. Work on `agent/<ticket>-<slug>`, one ticket per branch ([docs/workflow.md](docs/workflow.md)).

## Hard constraints

- Qwen3-Embedding-0.6B stays frozen; D = 1024.
- Never use test qrels, or any label from a meta-test environment, in training, adaptation,
  negative mining, descriptor construction, or normalization.
- Selection never modifies document embeddings or the index; the masked query is not re-normalized.
- Scientific results use exact scoring. FAISS is only for deployment analysis.
- Keep offline adaptation cost separate from online per-query cost.
- Do not change public APIs outside your ticket without justifying it in the pull request.
- Do not download or commit datasets, embeddings, or checkpoints. They live in `artifacts/`
  (gitignored); only manifests and configs are versioned.
- Every artifact is written together with a run manifest.
- Never edit experiment numbers by hand. Every table is generated from result files.

## Code

- Add a test for every new behavior. Critical math gets a unit test with a hand-computable result.
- Tests run on CPU without network access.
- Record non-trivial architectural decisions in `docs/decisions/ADR-XXXX-<slug>.md`
  (start from [ADR-0000-template.md](docs/decisions/ADR-0000-template.md)).

## Before committing

```bash
ruff check . && ruff format --check . && mypy && pytest
```

Use Conventional Commits scoped by package, e.g. `feat(oracle): aggregate multiple positives`.

## When finishing a ticket

Report the tests you ran, ruff and mypy results, the relevant smoke test, `git diff --stat`,
decisions and risks, and the proposed commit message. Fill in every section of the pull request
template.

## Ticket prompt template

```text
You are implementing ticket <ID> of the HyperDIME-Qwen project.

Before editing:
1. read AGENTS.md;
2. read docs/architecture.md;
3. read the contracts in src/hyperdime/contracts;
4. list the files you intend to change;
5. do not change APIs outside the ticket without justification.

Ticket goal:
<goal>

Required inputs/outputs:
<contract>

Acceptance criteria:
<definition of done>

Scientific constraints:
- Qwen3-Embedding-0.6B frozen;
- D=1024;
- do not use meta-test qrels in adaptation;
- separate offline and online cost;
- do not modify results manually.

When finished:
1. run the module's unit tests;
2. run ruff and mypy;
3. run the relevant smoke test;
4. show git diff --stat;
5. summarize decisions and risks;
6. propose the Conventional Commit message.
```
