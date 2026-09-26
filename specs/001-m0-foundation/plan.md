# Implementation Plan: M0 Foundation — Contracts and Reproducibility

**Branch**: `agent/M0-foundation` | **Date**: 2026-09-25 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/001-m0-foundation/spec.md`

## Summary

Give every later module (M1–M12) one shared set of versioned records, a run manifest that must
accompany every artifact, deterministic hashing and seeding, a validated config system, and a locked
environment. Records and configs are Pydantic v2 models; configs are composed from Hydra groups
through the compose API and then validated; artifacts are published atomically as directories whose
`manifest.json` is written last; dependencies are locked with `uv`. A `hyperdime diagnose` command
exercises the whole chain without loading the model. Covers backlog tickets M0.1–M0.4.

## Technical Context

**Language/Version**: Python ≥ 3.11 (declared); local venv 3.13.9, CI 3.12

**Primary Dependencies**: pydantic 2.x (records, validation), hydra-core + omegaconf (config
composition), numpy and torch (seeding, payloads); dev: pytest, ruff, mypy, pre-commit. The lock
also pins the later-milestone libraries (transformers, datasets, scipy, scikit-learn, ir-measures,
faiss-cpu) in optional extras.

**Storage**: files only — JSON/JSONL records, `.npy`/`.safetensors` payloads under gitignored
`artifacts/`; YAML configs under `configs/`

**Testing**: pytest on CPU, no network; unit tests in `tests/unit/contracts/`, one CLI integration
test in `tests/integration/`

**Target Platform**: Linux workstation (CPU, optional CUDA) and GitHub Actions `ubuntu-latest`

**Project Type**: Python library with a CLI (single project, `src/` layout)

**Performance Goals**: `hyperdime diagnose` < 5 s on a laptop CPU (SC-005); full test suite stays
well under a minute on CPU

**Constraints**: CPU-only tests without network; importing `hyperdime.contracts` or the CLI must
not import `transformers`, `datasets`, or `faiss`; no datasets, embeddings, or checkpoints in Git

**Scale/Scope**: 7 record types + `RunManifest` + 4 config models; ~6 new modules in
`hyperdime.contracts`, 1 CLI module, 4 config files; existing 27 tests unchanged

No `NEEDS CLARIFICATION` items remain; all choices are resolved in [research.md](research.md).

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Checked against [constitution v1.0.0](../../.specify/memory/constitution.md).

| Principle | Pre-research | Post-design | Evidence |
|---|---|---|---|
| I. Frozen Coordinate System | ✅ | ✅ | `ModelConfig` enforces `embedding_dim: Literal[1024]`, `frozen: Literal[True]`, pinned revision; `EmbeddingArtifact.dim: Literal[1024]` (R12, data-model) |
| II. Leakage-Free Protocols | ✅ | ✅ | No data or qrels are read in M0. `QrelsRecord.split` and `SpaceDescriptor.groups`/`protocol` rules give M1.3 and M5 the fields to enforce leakage guards; enforcement itself is out of scope and noted in spec assumptions |
| III. Contracts and Provenance First | ✅ | ✅ | This feature *is* the principle: versioned records, `RunManifest` with every required field, single seed manager, `ArtifactWriter` that cannot publish without a manifest (R8) |
| IV. Evidence-Gated Complexity | ✅ | ✅ | Foundation work precedes G1; `SpaceDescriptor` and `SelectorCheckpoint` are schema-only and marked provisional — no M5/M7 logic is written |
| V. Exact, Generated Measurement | ✅ | ✅ | `EvaluationResult.scoring: Literal["exact"]`; per-query metrics stored for paired tests; manifests make tables traceable to result files |
| VI. Test-Verified Mathematics | ✅ | ✅ | Hand-checkable hash tests (known SHA-256 of fixed bytes), round-trip and invalid-input tests for every record, seed-determinism test; all CPU, no network |
| VII. Independent, Replaceable Modules | ✅ | ✅ | Contracts live in their own package; CLI depends on contracts only; no pipeline coupling |
| Artifacts and Data Handling | ✅ | ✅ | Artifacts under gitignored `artifacts/`; each carries SHA-256 in its manifest; no DVC |
| Workflow and Quality Gates | ⚠️ | ⚠️ | One branch for four tickets — see Complexity Tracking. Pre-commit command, Conventional Commits, and PR template apply unchanged |

**Result**: PASS, with one justified deviation.

## Project Structure

### Documentation (this feature)

```text
specs/001-m0-foundation/
├── plan.md              # This file
├── research.md          # Phase 0: decisions R1–R12
├── data-model.md        # Phase 1: records, manifest, configs
├── quickstart.md        # Phase 1: validation scenarios
├── contracts/
│   ├── python-api.md    # hyperdime.contracts public API
│   ├── cli.md           # hyperdime diagnose
│   └── artifact-layout.md
├── checklists/
│   └── requirements.md  # spec quality checklist
└── tasks.md             # Phase 2 ($speckit-tasks — not created here)
```

### Source Code (repository root)

```text
pyproject.toml                      # deps, uv sources for cpu/cu128 torch, ruff extend-exclude,
                                    # console script `hyperdime`, pydantic mypy plugin
uv.lock                             # new, committed
.pre-commit-config.yaml             # new
.github/workflows/ci.yml            # switch to `uv sync --locked`
README.md                           # CPU and CUDA install instructions
docs/architecture.md                # schema summary; M0 row marked implemented

configs/
├── model/qwen3_0_6b.yaml
├── data/scifact.yaml
├── run/default.yaml
└── experiment/diagnose.yaml

src/hyperdime/
├── contracts/
│   ├── __init__.py                 # re-exports the public API
│   ├── schemas.py                  # Record, PayloadRef, 7 record types, dump/load helpers
│   ├── validation.py               # ContractError, ConfigError, validate_config
│   ├── config.py                   # ModelConfig, DataConfig, RunSettings, RunConfig, load_config
│   ├── hashing.py                  # canonical_json, config_hash, file_sha256, tree_sha256
│   ├── seeding.py                  # seed_everything
│   └── manifests.py                # RunManifest, start/finish_manifest, ArtifactWriter
└── cli/
    ├── __init__.py
    ├── __main__.py                 # python -m hyperdime.cli
    └── diagnose.py                 # argparse subcommand

tests/
├── unit/contracts/
│   ├── test_schemas.py
│   ├── test_config.py
│   ├── test_hashing.py
│   ├── test_seeding.py
│   └── test_manifests.py
├── integration/
│   └── test_diagnose_cli.py
└── fixtures/
    └── configs/                    # valid and invalid config trees for tests
```

**Structure Decision**: Keep the existing single-project `src/hyperdime` layout from plan §4.
`config.py` and `hashing.py`/`seeding.py` are added beside the plan's `schemas.py`,
`manifests.py`, and `validation.py` so each module has one responsibility; `validation.py` holds
the error types and top-level validation entry point. The plan's per-command `cli/*.py` files
become argparse subcommands under one `hyperdime` entry point. Existing modules are not modified.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| One branch (`agent/M0-foundation`) covers tickets M0.1–M0.4, while the workflow asks for one ticket per branch | The user chose M0 as a single spec-kit feature; the four tickets form a strict dependency chain (M0.2 → M0.3 → M0.4) inside one package and share one contract | Four branches would serialize review on shared files (`pyproject.toml`, `contracts/__init__.py`) with no parallelism gain. Mitigation: `tasks.md` groups work by ticket, and each ticket lands as its own Conventional Commit (`build(deps)`, `feat(contracts)`, `feat(contracts)`, `feat(cli)`) so history and issues map 1:1 |

## Risks

- **Hydra compose API and global state**: `initialize_config_dir` uses a global Hydra instance;
  `load_config` must wrap it in a context manager and clear it, or tests will interfere.
- **torch CPU/CUDA lock**: conflicting extras in `uv.lock` must be verified on CI; if resolution
  fails, fall back to a CPU-only lock and document the CUDA install as a post-sync step.
- **Provisional schemas**: `SpaceDescriptor` (v0) and `SelectorCheckpoint` (v0) will change at
  M5.5 and M7; the version field makes that an explicit, detectable break.
- **Python 3.13 locally vs 3.12 in CI**: the lock targets `>=3.11`; CI must keep running on the
  lock's lowest tested version so wheels exist for both.
