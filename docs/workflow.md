# Development workflow

How branches, issues, labels, milestones, pull requests, tags, and the wiki are organized.

## Branches

| Branch | Purpose | Rules |
|---|---|---|
| `master` | Reproducible, protected state | Changes only through pull requests; CI must pass; squash merge |
| `agent/<ticket>-<slug>` | One backlog ticket | Branch from `master`; touches one package; e.g. `agent/M3.1-oracle-importance` |
| `exp/<experiment>-<slug>` | Run an experiment and add its generated report | Only `configs/`, `experiments/`, `reports/`; library changes go through an `agent/` branch; e.g. `exp/E1-selector-shift` |
| `fix/<issue>-<slug>` | Bug fix outside a ticket | Links the bug issue; e.g. `fix/42-qrel-leakage` |
| `docs/<slug>` | Documentation only | No code changes |

- Two open branches must not change the same package unless both issues state the contract between them.
- Rebase on `master` before opening a pull request, and delete the branch after merging.
- The plan calls the protected branch `main`; this repository keeps its existing default, `master`.

## Tags

Tag `master` when a gate decision is merged, whether the gate passed or failed, so that every
decision points at the code that produced its evidence: `g1-baseline-valid`, `g2-selector-shift`,
`g3-space-signal`, `g4-hypernetwork`, `g5-cost`. Tag milestone completion as `milestone-<n>`.

## Commits

Conventional Commits, scoped by package or area:

```text
feat(space): add per-dimension marginal descriptor
test(selection): verify masked dot-product equivalence
fix(data): prevent qrel leakage into meta-test descriptor
```

Each commit passes the tests of the modules it touches.

## Issues

- [docs/backlog/issues.md](backlog/issues.md) is the initial backlog. Once published, GitHub issues
  are the source of truth and the file is not kept in sync.
- Titles use `[<ticket>] <summary>`, e.g. `[M3.1] Oracle importance targets`.
- New issues use a template: **Ticket** (implementation), **Experiment** (runs and gate decisions),
  or **Bug**.
- Lifecycle: open → `status:blocked` while any issue in its *Blocked by* list is open → assigned
  with a branch → pull request with `Closes #<n>` → closed.
- Gate issues (`type:decision`) close only when their ADR is merged.

## Labels

| Label | Values | Meaning |
|---|---|---|
| `type:` | `feature`, `infra`, `experiment`, `decision`, `bug`, `docs` | Kind of work |
| `module:` | `M0` … `M12` | Plan module |
| `phase:` | `A` … `F` | Execution phase (plan §10) |
| `gate:` | `G1` … `G5` | Provides evidence for a Go/No-Go gate |
| `status:` | `blocked`, `needs-decision` | Workflow state |
| `protocol-change` | — | Pull request changes the scientific protocol |
| `invalidates-results` | — | Pull request invalidates previously reported results |

## Milestones

| Milestone | Scope | Exit criterion |
|---|---|---|
| Milestone 1 — Reproduction | M0–M3, exact evaluation, E0 | Gate G1 |
| Milestone 2 — Scientific premise | M4, significance tests | Gate G2: Go/No-Go for the hypernetwork |
| Milestone 3 — Space representation | M5 | Gate G3 and a frozen `SpaceDescriptorV1` |
| Milestone 4 — HyperDIME prototype | M6–M9 | Matches or beats the global selector on meta-validation |
| Milestone 5 — Paper-grade evaluation | M10–M12, E3–E6 | Gates G4 and G5 |

A failed gate closes the issues of later milestones as *not planned*, and the result is recorded as
evidence in an ADR.

## Pull requests

[.github/pull_request_template.md](../.github/pull_request_template.md) requires the seven items
from plan §9.3: contract changed, files changed, tests added, commands run, expected artifacts or
regressions, whether the scientific protocol changes, and whether earlier results are invalidated.

## Wiki

Wiki pages live in [docs/wiki/](wiki/) and are published by `scripts/github/publish_wiki.sh`, which
replaces the GitHub wiki's pages. Edit pages in the repository, never in the GitHub web editor.

## One-time GitHub setup

1. Install the GitHub CLI and authenticate with `gh auth login`.
2. Create labels, milestones, and issues:

   ```bash
   python3 scripts/github/create_issues.py --dry-run
   python3 scripts/github/create_issues.py
   ```

3. Enable the wiki (Settings → General → Features → Wikis), save any first page in the web UI so
   the wiki repository exists, then run `bash scripts/github/publish_wiki.sh`.
4. Protect `master` (Settings → Branches): require a pull request, require the `checks` status
   check, and require linear history.
