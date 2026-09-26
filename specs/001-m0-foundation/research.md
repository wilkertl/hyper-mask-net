# Research: M0 Foundation — Contracts and Reproducibility

Phase 0 output for [plan.md](plan.md). Each entry resolves an open choice from the Technical
Context.

## R1. Record schemas and validation

- **Decision**: Pydantic v2 models, `frozen=True`, `extra="forbid"`, one `schema_version` field
  typed as a `Literal` per model. JSON for single records, JSON Lines for collections.
- **Rationale**: Field-level error messages that name the model and field come for free (FR-003,
  SC-004); `model_dump_json` / `model_validate_json` give exact round-trips (FR-002); a `Literal`
  version makes unsupported versions a validation error (FR-004); `extra="forbid"` catches typos.
  Pydantic 2.12 is already in the venv, and it has a mypy plugin.
- **Alternatives considered**: stdlib `dataclasses` with hand-written validation (more code, weaker
  messages); `msgspec` (faster, but less familiar and weaker constraint messages); `attrs` +
  `cattrs` (two libraries for what one does).

## R2. Config system

- **Decision**: Hydra's **compose API** (`initialize_config_dir` + `compose`) for config groups and
  command-line overrides, converted with OmegaConf to a plain container, then validated into a
  Pydantic `RunConfig`. No `@hydra.main`.
- **Rationale**: The plan names Hydra/OmegaConf and the plan's `configs/` layout is Hydra groups
  (`model/`, `data/`, `experiment/`). The compose API gives group composition and overrides without
  Hydra changing the working directory or creating its own output folders, which would compete with
  the artifact writer. Validating with the same Pydantic machinery gives one error style for records
  and configs (FR-011), and `extra="forbid"` rejects unknown keys.
- **Alternatives considered**: `@hydra.main` (changes cwd, writes `outputs/`, harder to test);
  OmegaConf alone (would need a hand-written group/defaults resolver); plain YAML + Pydantic (no
  composition or overrides).

## R3. Lockfile and environment

- **Decision**: `uv` with a committed `uv.lock`. PyTorch CPU and CUDA builds are declared as two
  conflicting extras (`cpu`, `cu128`) mapped to the PyTorch wheel indexes through
  `[tool.uv.sources]`; the default dev and CI path is `uv sync --extra cpu --extra dev`.
- **Rationale**: uv is already installed and created `.venv`; `uv.lock` is a single cross-platform
  lock (FR-013); the CPU/CUDA split is the documented uv pattern for PyTorch. CI switches from
  `pip install` to `uv sync --locked`, which fails if the lock is stale (FR-014).
- **Alternatives considered**: `pip-tools` (one lock per platform and per torch variant); Poetry
  (weak support for alternative torch indexes); conda (heavy, and not what the repo uses).
- **Pins**: the M0 lock includes every dependency the backlog lists (torch, transformers, datasets,
  numpy, scipy, scikit-learn, ir-measures, hydra-core, omegaconf, faiss-cpu, pydantic, pytest,
  ruff, mypy, pre-commit), so later tickets add code, not dependencies. Heavy libraries unused by M0
  (transformers, datasets, faiss-cpu, ir-measures, scikit-learn, scipy) stay in optional extras so
  the base install for tests stays small.

## R4. Pre-commit and lint scope

- **Decision**: `.pre-commit-config.yaml` with local `language: system` hooks that call `ruff check`,
  `ruff format --check`, and `mypy` from the locked environment. `[tool.ruff] extend-exclude =
  [".specify", ".agents"]`.
- **Rationale**: System hooks use exactly the locked versions, so pre-commit and CI cannot drift
  (FR-014). The exclusion fixes the current `ruff` failures, all of which come from vendored
  spec-kit scripts (FR-015).
- **Alternatives considered**: the upstream `ruff-pre-commit` and `mirrors-mypy` repos (they pin
  their own versions and need stubs duplicated in the hook config).

## R5. Config fingerprint

- **Decision**: SHA-256 of the canonical JSON of the **validated** config: `json.dumps(obj,
  sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)` encoded as UTF-8.
- **Rationale**: Sorted keys give key-order independence (FR-007); hashing after validation means
  `0.1` and `1e-1` are already the same float, and defaults are filled in, so two configs that
  describe the same run hash the same. `allow_nan=False` makes non-finite values an error.
- **Alternatives considered**: hashing the YAML text (sensitive to formatting and key order);
  `hash()` (process-salted); pickle (not canonical across versions).

## R6. File and dataset fingerprints

- **Decision**: files → streamed SHA-256 in 1 MiB blocks. Directories/datasets → SHA-256 over the
  sorted list of `"<relative posix path>\t<file sha256>\n"` lines. No dataset → the literal
  `"none"`.
- **Rationale**: Stable across machines and independent of file enumeration order (SC-002). An
  explicit `"none"` satisfies the edge case that a missing dataset is never an empty field.
- **Alternatives considered**: hashing mtimes or sizes (not content-addressed); tar-then-hash
  (depends on tar metadata).

## R7. Seed manager

- **Decision**: `seed_everything(seed, deterministic=True)` seeds `random`, NumPy, and
  `torch` (CPU and all CUDA devices), calls `torch.use_deterministic_algorithms(True,
  warn_only=True)`, disables cuDNN benchmarking, and sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` if unset.
  It returns a record of what it set, which the manifest stores. `PYTHONHASHSEED` is recorded, not
  set, because it only takes effect at interpreter start.
- **Rationale**: One entry point for every randomness source (FR-008). `warn_only=True` avoids
  crashing on ops without deterministic kernels while still logging them.
- **Alternatives considered**: `lightning.seed_everything` (adds a large dependency for a few lines);
  a strict deterministic mode that raises (would block legitimate GPU runs; can be a flag later).

## R8. Artifact writer and atomicity

- **Decision**: Each artifact is a directory `artifacts/<kind>/<run_id>/` holding its payload files
  and a `manifest.json`. `ArtifactWriter` takes a mandatory `RunManifest`, writes everything into a
  sibling temporary directory `.<run_id>.tmp-<uuid>`, writes `manifest.json` last, fsyncs, and
  publishes with a single `os.rename` of the directory. A directory without `manifest.json` is
  invalid by definition, and readers refuse it.
- **Rationale**: A directory rename on the same filesystem is atomic on POSIX, so an interrupted
  write leaves only a hidden temp directory, never a visible artifact without its manifest (FR-009).
  Making the manifest a required constructor argument means the "no manifest" case cannot type-check,
  and the runtime check covers `None` passed dynamically.
- **Alternatives considered**: writing a manifest next to a single file (two renames, not atomic
  together); a central SQLite registry (more infrastructure than M0 needs).

## R9. Code provenance

- **Decision**: `git rev-parse HEAD` and `git status --porcelain --untracked-files=no` through
  `subprocess`; the manifest stores `git_commit` (or `"unknown"`) and `git_dirty` (true, false, or
  null when unknown).
- **Rationale**: Covers the dirty-tree and no-git edge cases without failing exploratory runs
  (spec assumption). Untracked files are excluded because `artifacts/` and scratch files would
  otherwise mark every run dirty.
- **Alternatives considered**: GitPython (a dependency for two commands); refusing dirty trees
  (rejected in the spec assumptions).

## R10. Numeric payloads and precision

- **Decision**: Records reference large payloads through a `PayloadRef(path, sha256, format)`
  where `format` is one of `npy` or `safetensors`; embedding matrices are `.npy` with dtype
  recorded in the record. Paths are relative to the artifact directory.
- **Rationale**: Keeps JSON records small and diffable (FR-005). `.npy` matches the plan's
  `documents.f16.npy`; relative paths keep artifacts relocatable.
- **Alternatives considered**: inlining base64 arrays (huge JSON, slow); Parquet (adds pyarrow for
  no M0 benefit).

## R11. Diagnostic command surface

- **Decision**: A console script `hyperdime` built on `argparse` with a `diagnose` subcommand:
  `hyperdime diagnose [--config-dir DIR] [--config-name NAME] [--output DIR] [overrides ...]`.
  Also runnable as `python -m hyperdime.cli`. Exit codes: 0 success, 2 invalid config, 1 other
  errors.
- **Rationale**: argparse is stdlib; subcommands leave room for the plan's `embed`, `build_space`,
  `train_selector`, and `evaluate` commands. The command never imports `transformers` or touches the
  network, satisfying FR-012 and SC-005.
- **Alternatives considered**: Typer/Click (an extra dependency for one subcommand); one script per
  command (the plan's `cli/*.py` files can still each register a subcommand).

## R12. Hard constraints encoded in config validation

- **Decision**: `ModelConfig` fixes `name = "Qwen/Qwen3-Embedding-0.6B"`, `embedding_dim:
  Literal[1024]`, `frozen: Literal[True]`, and requires a pinned `revision` (a commit SHA, not a
  branch name). `ScoringConfig.mode: Literal["exact"]` for scientific runs.
- **Rationale**: Principle I (frozen, D = 1024) and principle V (exact scoring) become validation
  errors rather than conventions (FR-011). A pinned revision is what makes `model_revision` in the
  manifest meaningful.
- **Alternatives considered**: warnings instead of errors (principles are MUST-level).
