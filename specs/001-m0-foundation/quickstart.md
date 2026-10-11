# Quickstart: validating M0 Foundation

Runnable scenarios that show the feature works end to end. Interfaces are defined in
[contracts/](contracts/); records in [data-model.md](data-model.md).

## Prerequisites

- Linux or macOS, Python ≥ 3.11, `uv` installed, no network needed after `uv sync`.

## 1. Locked environment (User Story 4)

```bash
uv sync --locked --extra cpu --extra dev      # fails if uv.lock is stale
uv run pre-commit run --all-files             # ruff lint, ruff format, mypy
uv run pytest                                 # all tests, CPU only
```

**Expected**: every command exits 0; the 27 pre-existing unit tests still pass (SC-007); no file
under `.specify/` or `.agents/` is linted.

## 2. Diagnostic run (User Story 3)

```bash
uv run hyperdime diagnose
```

**Expected**: exit 0 in under 5 s; stdout prints `artifacts/diagnose/<run_id>`; that directory
contains only `manifest.json`, with `config_hash`, `git_commit`, `git_dirty`, `dataset_hash:
"none"`, `model_revision`, `seed`, `device`, `dtype`, `started_at`, `finished_at` all set.

Run it twice and compare:

```bash
uv run hyperdime diagnose > a && uv run hyperdime diagnose > b
jq -r .config_hash "$(cat a)/manifest.json" "$(cat b)/manifest.json" | uniq | wc -l   # → 1
```

## 3. Invalid configs fail explicitly (User Story 3)

```bash
uv run hyperdime diagnose model.embedding_dim=768          # constraint violated
uv run hyperdime diagnose +model.unknown_key=1             # unknown key
uv run hyperdime diagnose '~model.revision'                # missing field
```

**Expected**: each exits 2, prints a line naming the dotted field, and creates no new directory
under `artifacts/diagnose/`.

## 4. Contract tests (User Stories 1 and 2)

```bash
uv run pytest tests/unit/contracts -q
uv run pytest tests/integration/test_diagnose_cli.py -q
```

**Expected**: tests covering
- round-trip of all seven record types and `RunManifest` (SC-001);
- unsupported `schema_version`, unknown fields, width ≠ 1024, and NaN rejected with the field named
  (SC-004);
- key-order-independent config hashes and stable file/tree hashes (SC-002);
- `ArtifactWriter` refusing a missing manifest, and an exception mid-write leaving no visible
  artifact (SC-003);
- identical random draws after `seed_everything` with the same seed.

## 5. CI parity

Open a pull request from `agent/M0-foundation`. **Expected**: the `checks` job installs with
`uv sync --locked` and runs the same commands as step 1.
