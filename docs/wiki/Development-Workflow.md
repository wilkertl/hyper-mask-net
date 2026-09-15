# Development Workflow

Canonical version: [docs/workflow.md](https://github.com/wilkertl/hyper-mask-net/blob/master/docs/workflow.md).
Agent rules: [AGENTS.md](https://github.com/wilkertl/hyper-mask-net/blob/master/AGENTS.md).

## Branches

| Branch | Use |
|---|---|
| `master` | Protected, reproducible; pull requests only, CI required, squash merge |
| `agent/<ticket>-<slug>` | One ticket, one package, e.g. `agent/M3.1-oracle-importance` |
| `exp/<experiment>-<slug>` | Experiment configs and generated reports, e.g. `exp/E1-selector-shift` |
| `fix/<issue>-<slug>` | Bug fix linked to a bug issue |
| `docs/<slug>` | Documentation only |

Gate decisions are tagged on `master` (`g1-baseline-valid` … `g5-cost`), and milestones as
`milestone-<n>`.

## Working on a ticket

1. Pick an unblocked issue in the current milestone and assign yourself.
2. Create `agent/<ticket>-<slug>` from `master`.
3. Read AGENTS.md, docs/architecture.md, docs/protocols.md, and the contracts.
4. Implement with tests; run `ruff check . && ruff format --check . && mypy && pytest`.
5. Commit with Conventional Commits, e.g. `feat(oracle): aggregate multiple positives`.
6. Open a pull request with `Closes #<n>` and complete the seven checklist items.

## Labels

`type:` feature, infra, experiment, decision, bug, docs · `module:` M0–M12 · `phase:` A–F ·
`gate:` G1–G5 · `status:` blocked, needs-decision · `protocol-change` · `invalidates-results`

## Issue templates

- **Ticket:** implementation work with a contract and a definition of done.
- **Experiment:** question, gate, protocol, environments, metrics fixed before running.
- **Bug:** includes whether reported results are invalidated.
