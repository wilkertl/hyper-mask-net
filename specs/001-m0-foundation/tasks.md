---

description: "Task list for M0 Foundation — contracts and reproducibility (feature F1)"
---

# Tasks: M0 Foundation — Contracts and Reproducibility

**Input**: Design documents from `specs/001-m0-foundation/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md

**Tests**: Required. FR-016, SC-001–SC-004, and constitution principle VI require a test for every
new behavior and hand-checkable values for hashing; every test runs on CPU without network access.
Write each story's tests first and confirm they fail before implementing.

**Organization**: Tasks are grouped by user story (US1–US4 from spec.md). The plan maps them to
backlog tickets: US1 = M0.2 (records), US2 = M0.3 (manifests, hashing, seeds), US3 = M0.4 (configs,
`hyperdime diagnose`), US4 = M0.1 (lockfile, pre-commit, CI). Each ticket lands as its own
Conventional Commit (plan, Complexity Tracking).

**State of the repository at generation** (`master` at `5f49b89`, after the F0 spike and the SCRAP
tooling merged): `src/hyperdime/contracts/hashing.py` already has `file_sha256` and `tree_sha256`
(spike task T048), `src/hyperdime/cli/__init__.py` exists with a docstring only, ruff already
excludes `.specify` and `.agents`, mypy already targets Python 3.12, `numpy` is a declared
dependency, CI already runs SCRAP, and the suite has 118 tests. The spike writes lightweight JSON
manifests through `src/hyperdime/spike/manifest.py`; moving them onto `RunManifest` is task T033.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks)
- **[Story]**: US1–US4 from spec.md
- Paths are relative to the repository root (single project, `src/` layout)

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Dependencies and directories every story needs.

- [ ] T001 Add the runtime dependencies `pydantic>=2.7,<3`, `hydra-core>=1.3,<1.4`, and `omegaconf>=2.3,<2.4` to `[project] dependencies` in pyproject.toml, add `plugins = ["pydantic.mypy"]` under `[tool.mypy]`, and add the console script `[project.scripts] hyperdime = "hyperdime.cli.__main__:main"`; reinstall with `pip install -e ".[dev]"` and confirm `python -c "import pydantic, hydra, omegaconf"` works
- [ ] T002 [P] Create tests/unit/contracts/ (no `__init__.py`, matching tests/unit/) and tests/fixtures/configs/ with a `valid/` tree that mirrors configs/ (model/, data/, run/, experiment/) and an `invalid/` tree with one experiment file per failure: missing `model.revision`, unknown key `model.unknown_key`, `model.embedding_dim: 768`, `model.frozen: false`, and a `model.revision` that is a branch name (`main`) rather than a 40-hex SHA

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Error types and config fingerprints that US1–US3 all use.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [ ] T003 Create src/hyperdime/contracts/validation.py with `class ContractError(ValueError)` and `class ConfigError(ContractError)`, plus a helper `format_validation_error(model_name: str, error: pydantic.ValidationError, line: int | None = None) -> str` that renders one line per error as `<model>.<dotted.field>: <message>` (prefixed by `line <n>: ` for JSON Lines); `validate_config` is added in T023
- [ ] T004 [P] Write tests in tests/unit/contracts/test_hashing.py: `canonical_json({"b": 1, "a": [1.5, "x"]}) == '{"a":[1.5,"x"],"b":1}'`; `config_hash` equal for `{"a": 1, "b": {"c": 2}}` and `{"b": {"c": 2}, "a": 1}` and different when any value changes; `config_hash` of the literal `{"a": 1}` equals `hashlib.sha256(b'{"a":1}').hexdigest()` (hand-checkable); `float("nan")` and `float("inf")` raise `ContractError`; `NO_DATASET == "none"`; `config_hash` accepts a pydantic model and hashes its `model_dump(mode="json")`
- [ ] T005 Add `canonical_json(obj) -> str` (`json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)`, raising `ContractError` instead of `ValueError`), `config_hash(config: BaseModel | Mapping[str, Any]) -> str` (SHA-256 of the UTF-8 canonical JSON; models go through `model_dump(mode="json")`), and `NO_DATASET: Final = "none"` to src/hyperdime/contracts/hashing.py, keeping `file_sha256` and `tree_sha256` unchanged (research R5, R6)

**Checkpoint**: `pytest tests/unit/contracts/test_hashing.py` passes.

---

## Phase 3: User Story 1 - Shared, versioned records (Priority: P1) 🎯 MVP

**Goal**: Seven versioned record types that round-trip through JSON and JSON Lines and reject invalid input with the record type and field named (FR-001–FR-005).

**Independent Test**: Build one valid instance of each record type, save and reload it, and compare; then load deliberately invalid instances and check the error messages (spec US1).

### Tests for User Story 1

- [ ] T006 [P] [US1] Write tests/unit/contracts/test_schemas.py: for each of `QueryRecord`, `DocumentRecord`, `QrelsRecord`, `EmbeddingArtifact`, `SpaceDescriptor`, `SelectorCheckpoint`, `EvaluationResult`, a valid instance survives `dump_record`/`load_record` and `dump_records`/`load_records` unchanged, including `schema_version` (SC-001); a missing required field, a wrong type, and an unknown field each raise `ContractError` whose message contains the record type and the field name (SC-004); a JSON Lines file with a bad third line names `line 3`; an unsupported `schema_version` (2 for version-1 records, 1 for the version-0 `SpaceDescriptor` and `SelectorCheckpoint`) fails to load
- [ ] T007 [US1] Add field-rule tests to tests/unit/contracts/test_schemas.py, one per data-model.md rule: `EmbeddingArtifact` with `dim` 768 fails, `ids` with a duplicate fails, `len(ids) != num_rows` fails, `instruction` set when `role == "documents"` fails; `QrelsRecord.relevance` −1 fails; `PayloadRef.path` absolute, containing `..`, or with backslashes fails and `sha256` that is not 64 lowercase hex fails; `SpaceDescriptor` with `protocol: "in-domain"` fails, with `protocol: "P0"` and `groups` containing `"B"` fails ("P0: A, E only"), with `protocol: "P0"` and `num_queries` > 0 fails, with duplicate `feature_names` fails; `SelectorCheckpoint` with `kind: "global"` and an `environment` fails, with a non-`generated` kind and a `rank` fails, with `payload.format` other than `"safetensors"` fails; `EvaluationResult` with a NaN metric fails, with `k` 0 or 1025 fails, with `num_queries != len(per_query)` fails, with a `per_query` entry whose metric keys differ from `metrics` fails, with `scoring: "faiss"` fails; `Id` values that are empty or contain a control character fail

### Implementation for User Story 1

- [ ] T008 [US1] Create src/hyperdime/contracts/schemas.py with the shared types of data-model.md: `Id` (non-empty, stripped, no control characters), `Split = Literal["train", "validation", "test"]`, `Protocol = Literal["P0", "P1", "P2", "in-domain"]`, `Sha256` (64 lowercase hex), `Dtype = Literal["float16", "bfloat16", "float32"]`, `EMBEDDING_DIM` imported from `hyperdime.embeddings.qwen` (that module imports only torch at load time; keep it that way), `PayloadRef` (`path` relative, POSIX, no `..`; `format: Literal["npy", "safetensors"]`), and the base `Record(BaseModel)` with `model_config = ConfigDict(frozen=True, extra="forbid")` and `schema_version` typed as a `Literal` in each subclass (research R1)
- [ ] T009 [US1] Add the seven record classes to src/hyperdime/contracts/schemas.py with every field and rule from data-model.md: `QueryRecord` (v1), `DocumentRecord` (v1; "non-empty after joining with title"), `QrelsRecord` (v1; `relevance` ≥ 0), `EmbeddingArtifact` (v1; `dim: Literal[1024]`, `pooling: Literal["last_token"]`, "`ids` distinct; `len(ids) == num_rows`", "must be `None` when `role == "documents"`"), `SpaceDescriptor` (v0, provisional until M5.5; "`in-domain` not allowed", "P0: A, E only", "0 under P0"), `SelectorCheckpoint` (v0, provisional until M7; "`None` for `global`", "rank only for `generated`", `safetensors` payload, `training_manifest_sha256: Sha256`), `EvaluationResult` (v1; `scoring: Literal["exact"]`, "1 ≤ k ≤ 1024", finite metrics, "`num_queries == len(per_query)`", per-query keys equal `metrics` keys)
- [ ] T010 [US1] Add `dump_record`, `load_record`, `dump_records` (JSON Lines), and `load_records` to src/hyperdime/contracts/schemas.py, writing UTF-8 with a trailing newline and converting every `pydantic.ValidationError` and JSON decode error into `ContractError` through `format_validation_error` (with the line number for JSON Lines)
- [ ] T011 [US1] Re-export the public API of contracts/python-api.md from src/hyperdime/contracts/__init__.py (records, `PayloadRef`, dump/load helpers, `ContractError`, `ConfigError`, hashing functions, `NO_DATASET`), and add an `__all__`; confirm `python -c "import hyperdime.contracts, sys; assert 'transformers' not in sys.modules"`

**Checkpoint**: `pytest tests/unit/contracts/test_schemas.py` passes; US1 is usable by later modules.

---

## Phase 4: User Story 2 - Traceable artifacts through run manifests (Priority: P1)

**Goal**: `RunManifest`, one seed manager, and an `ArtifactWriter` that cannot publish an artifact without its manifest (FR-006–FR-009; the M0 gate).

**Independent Test**: Write an artifact through `ArtifactWriter` with and without a manifest; seed two runs identically and compare draws; create a manifest in a dirty and in a clean git tree (spec US2).

### Tests for User Story 2

- [ ] T012 [P] [US2] Write tests/unit/contracts/test_seeding.py: after `seed_everything(7)` twice, draws from `random.random()`, `numpy.random.rand(3)`, and `torch.rand(3)` are identical across the two seedings, and differ for seed 8; the returned dict records the seed, `deterministic`, `cudnn_benchmark: False`, `CUBLAS_WORKSPACE_CONFIG`, and `PYTHONHASHSEED` (recorded, not set); `deterministic=False` is reflected in the dict
- [ ] T013 [P] [US2] Write tests/unit/contracts/test_manifests.py for the manifest: `start_manifest` fills every FR-006 field; `run_id` matches `^\d{8}T\d{6}Z-[0-9a-f]{8}$`; `dataset_hash` defaults to `"none"`; in a temporary `git init` repository with one commit, `git_dirty` is `False`, and after editing a tracked file it is `True` (untracked files do not count, research R9); with `git` unavailable (monkeypatch `subprocess.run` to raise `FileNotFoundError`), `git_commit == "unknown"` and `git_dirty is None`; `finish_manifest` sets `finished_at >= started_at` and the artifact list; a manifest round-trips through `dump_record`/`load_record`
- [ ] T014 [US2] Add `ArtifactWriter` tests to tests/unit/contracts/test_manifests.py, one per guarantee in contracts/python-api.md: passing `None` as the manifest raises `ContractError`; nothing is visible under `root/kind/` before `commit()`; the published directory contains `manifest.json` whose `artifacts` list matches every payload file with its SHA-256 (check one by hand with `file_sha256`); an exception inside `with ArtifactWriter(...)` leaves no visible directory under `root/kind/` and no `.tmp-` directory; `commit()` with a staged name whose file was never written raises; committing a second writer with the same `run_id` raises and leaves the first artifact untouched; `read_manifest` on a directory without `manifest.json` raises `ContractError`

### Implementation for User Story 2

- [ ] T015 [P] [US2] Create src/hyperdime/contracts/seeding.py with `seed_everything(seed: int, *, deterministic: bool = True) -> dict[str, str | bool | None]`: seed `random`, NumPy, `torch.manual_seed`, and `torch.cuda.manual_seed_all`; when deterministic, call `torch.use_deterministic_algorithms(True, warn_only=True)`, set `torch.backends.cudnn.benchmark = False`, and set `CUBLAS_WORKSPACE_CONFIG=":4096:8"` only if unset; record `PYTHONHASHSEED` from the environment without setting it (research R7)
- [ ] T016 [US2] Create src/hyperdime/contracts/manifests.py with `RunManifest(Record)` (`schema_version: Literal[1]` and every field of data-model.md, "`finished_at` ≥ `started_at`", timezone-aware UTC datetimes, `git_dirty: bool | None`, `dataset_hash: Sha256 | Literal["none"]`, `artifacts: list[{name, sha256}]`), `start_manifest(command, config, *, dataset_hash=NO_DATASET, model_revision=None, device=None)` accepting `config: RunConfig | Mapping[str, Any]` (the mapping form lets the spike adopt it in T033; note this widening in the PR as a contract change), and `finish_manifest(manifest, artifacts)`; the git helper runs `git rev-parse HEAD` and `git status --porcelain --untracked-files=no` (research R9); `versions` records python, torch, numpy, and the installed `hyperdime-qwen` version
- [ ] T017 [US2] Add `ArtifactWriter(root, kind, manifest)` to src/hyperdime/contracts/manifests.py: stage files in `root/kind/.<run_id>.tmp-<uuid>/`, `path(name)` returns a staging path, `write_record(name, record)` uses `dump_record`, `commit()` hashes every staged file, finalizes the manifest with `finish_manifest`, writes `manifest.json` last, fsyncs the files and the staging directory, and publishes with one `os.rename` to `root/kind/<run_id>/`, refusing an existing target; `__exit__` removes the staging directory on error and commits otherwise (research R8, contracts/artifact-layout.md)
- [ ] T018 [US2] Add `read_manifest(artifact_dir) -> RunManifest` to src/hyperdime/contracts/manifests.py (raises `ContractError` without `manifest.json`), and re-export `RunManifest`, `start_manifest`, `finish_manifest`, `ArtifactWriter`, `read_manifest`, and `seed_everything` from src/hyperdime/contracts/__init__.py

**Checkpoint**: `pytest tests/unit/contracts` passes; the M0 gate (SC-003) holds.

---

## Phase 5: User Story 3 - Configured runs and a diagnostic command (Priority: P2)

**Goal**: Versioned Hydra configs validated into `RunConfig`, and `hyperdime diagnose`, which writes a manifest without loading the model (FR-010–FR-012).

**Independent Test**: Run `hyperdime diagnose` on the sample config and inspect its manifest; run it on each invalid fixture and check the exit code, the error line, and that nothing was written (spec US3).

### Tests for User Story 3

- [ ] T019 [P] [US3] Write tests/unit/contracts/test_config.py: `load_config(tests/fixtures/configs/valid, "experiment/diagnose")` returns a `RunConfig` with `model.embedding_dim == 1024` and `config_version == 1`; each `invalid/` fixture raises `ConfigError` whose message names the dotted field (`model.revision`, `model.unknown_key`, `model.embedding_dim`, `model.frozen`); `validation_fraction: 0.1` and `1e-1` give the same `config_hash`; overrides such as `run.seed=7` apply; `load_config` leaves `os.getcwd()` unchanged and can be called twice in one process (Hydra's global instance is cleared)
- [ ] T020 [P] [US3] Write tests/integration/test_diagnose_cli.py: `main(["diagnose", "--config-dir", <valid>, "--output", tmp_path])` exits 0, prints exactly one stdout line with the artifact directory, and that directory contains only `manifest.json` with `config_hash`, `git_commit`, `git_dirty`, `dataset_hash == "none"`, `model_revision`, `seed`, `device`, `dtype`, `started_at`, and `finished_at` set; two runs give the same `config_hash` and different `run_id`; each invalid fixture exits 2, writes `error: invalid config: <dotted.field>: ...` to stderr, and creates nothing under `tmp_path/diagnose/`; a subprocess running `python -m hyperdime.cli diagnose --help` and then checking `sys.modules` imports none of `transformers`, `datasets`, or `faiss`; a full diagnose run finishes in under 5 seconds (SC-005)

### Implementation for User Story 3

- [ ] T021 [P] [US3] Create configs/model/qwen3_0_6b.yaml (`name: Qwen/Qwen3-Embedding-0.6B`, `revision: 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` — the commit the F0 embeddings were made with, per artifacts/spike/*/manifest.json — `embedding_dim: 1024`, `frozen: true`, `normalize: true`, `max_length: 512`, `dtype: bfloat16`), configs/data/scifact.yaml (`environment: scifact`, `source`, the SciFact instruction from src/hyperdime/spike/environments.py, `validation_fraction: 0.1`), configs/run/default.yaml (`seed: 0`, `device: auto`, `dtype: float32`, `output_dir: artifacts`), and configs/experiment/diagnose.yaml (`defaults: [/model: qwen3_0_6b, /data: scifact, /run: default, _self_]`, `config_version: 1`); copy the same files into tests/fixtures/configs/valid/
- [ ] T022 [US3] Create src/hyperdime/contracts/config.py with `ModelConfig` (`name: Literal["Qwen/Qwen3-Embedding-0.6B"]`, `revision` a 40-hex SHA, `embedding_dim: Literal[1024]`, `frozen: Literal[True]`, `normalize: bool`, `max_length` > 0, `dtype: Dtype`), `DataConfig` ("`validation_fraction ∈ (0, 1)`", optional `sample: {num_documents, num_queries, seed}`), `RunSettings` ("`device ∈ {auto, cpu, cuda, cuda:N}`", `seed` ≥ 0, `dtype`, `output_dir`), and `RunConfig` (`model`, `data`, `run`, `config_version: Literal[1]`), all with `extra="forbid"` and `frozen=True` (research R12, data-model.md)
- [ ] T023 [US3] Add `validate_config(raw: Mapping[str, Any]) -> RunConfig` to src/hyperdime/contracts/validation.py, raising `ConfigError` with one `<dotted.field>: <message>` line per error, and `load_config(config_dir, config_name, overrides=())` to src/hyperdime/contracts/config.py using Hydra's `initialize_config_dir` + `compose` inside a context manager that clears `GlobalHydra` afterwards, never changing the working directory or writing files (research R2); re-export both from src/hyperdime/contracts/__init__.py
- [ ] T024 [US3] Create src/hyperdime/cli/__main__.py with `main(argv: Sequence[str] | None = None) -> int` building an argparse parser with subcommands (`python -m hyperdime.cli` calls it), and src/hyperdime/cli/diagnose.py registering `diagnose [--config-dir configs] [--config-name experiment/diagnose] [--output DIR] [OVERRIDE ...]`: load and validate the config, `seed_everything(run.seed)`, resolve `auto` to `cuda:0` when available else `cpu`, write `<output>/diagnose/<run_id>/manifest.json` through `ArtifactWriter` with `dataset_hash="none"` and `model_revision=model.revision`, print the directory on stdout and a summary on stderr; exit 2 on `ConfigError` (printing `error: invalid config: ...` lines), 1 on any other error (contracts/cli.md)

**Checkpoint**: `pytest tests/unit/contracts tests/integration/test_diagnose_cli.py` passes and `hyperdime diagnose` works from a fresh install.

---

## Phase 6: User Story 4 - Locked environment and local checks (Priority: P3)

**Goal**: A committed `uv.lock`, pre-commit checks that match CI, and CI installing from the lock (FR-013–FR-015).

**Independent Test**: In a fresh clone, follow the README, run the pre-commit checks on all files, and confirm CI passes from the same lock (spec US4).

- [ ] T025 [US4] Configure uv in pyproject.toml (research R3): optional extras `cpu = ["torch>=2.4"]` and `cu128 = ["torch>=2.4"]` declared as conflicting under `[tool.uv] conflicts`, mapped to `https://download.pytorch.org/whl/cpu` and `https://download.pytorch.org/whl/cu128` through `[tool.uv.sources]` and `[[tool.uv.index]]` (explicit indexes); later-milestone libraries (`transformers`, `datasets`, `scipy`, `scikit-learn`, `ir-measures`, `faiss-cpu`) in optional extras so the base install stays small; add `pre-commit` to the `dev` extra
- [ ] T026 [US4] Run `uv lock`, commit uv.lock, and verify `uv sync --locked --extra cpu --extra dev` then `uv run pytest` on Python 3.12; if the CPU/CUDA conflict does not resolve, fall back to a CPU-only lock and document `uv pip install torch --index-url .../cu128` as a post-sync step in README.md (plan, Risks)
- [ ] T027 [P] [US4] Create .pre-commit-config.yaml with local `language: system` hooks running `ruff check`, `ruff format --check`, `mypy`, and `PYTHONPATH=tools python -m scrap tests` (AGENTS.md "Before committing"), all from the locked environment (research R4); confirm `uv run pre-commit run --all-files` passes and lints nothing under `.specify/` or `.agents/`
- [ ] T028 [US4] Switch .github/workflows/ci.yml to `astral-sh/setup-uv` with `uv sync --locked --extra cpu --extra dev` in place of the two `pip install` steps, running ruff, format, mypy, pytest, and the existing SCRAP step through `uv run`; keep the SCRAP step's exit-status handling
- [ ] T029 [US4] Rewrite the install sections of README.md: CPU (`uv sync --locked --extra cpu --extra dev`) and CUDA (`--extra cu128`), `uv run hyperdime diagnose` as the first smoke test, and the `embed` extra for the spike's local encoder; remove the `pip install -e` instructions they replace
- [ ] T030 [US4] Verify SC-006 in a fresh clone under the scratch directory: follow only README.md, then run `uv run pre-commit run --all-files` and `uv run pytest`; record the commands and results in the pull request

**Checkpoint**: CI passes on the pull request using `uv sync --locked`.

---

## Phase 7: Polish & Cross-Cutting Concerns

- [ ] T031 [P] Add a "Contracts" section to docs/architecture.md summarizing the seven record types, `PayloadRef`, `RunManifest`, the artifact layout, and the `ArtifactWriter` guarantees, and mark the M0 row of the Components table implemented with its functions (FR-017)
- [ ] T032 [P] Update the counts that predate the F0 spike: SC-007 in specs/001-m0-foundation/spec.md and step 1 of specs/001-m0-foundation/quickstart.md say "the existing 27 unit tests"; replace with "every test that existed at `5f49b89` (118)"
- [ ] T033 Move the spike's manifests onto `RunManifest` (roadmap task T029, plan Complexity Tracking): `write_manifest` in src/hyperdime/spike/manifest.py builds a `RunManifest` through `start_manifest(command, config_mapping, ...)`, with `finished_at is None` meaning running; `require_complete` accepts both that form and the legacy `"status"` form so the existing F0 artifacts under artifacts/spike/ stay readable; the spike keeps its in-place, resumable output directories rather than `ArtifactWriter` — record that exception and its reason (resumable embedding caches) in docs/decisions/ADR-0005-spike-manifests-without-artifact-writer.md; extend tests/integration/test_spike_pipeline.py so a new synthetic run's manifests validate as `RunManifest` and a legacy-format manifest is still accepted
- [ ] T034 Run the scenarios of specs/001-m0-foundation/quickstart.md end to end and fix any drift between them and the implementation
- [ ] T035 Run `ruff check . && ruff format --check . && mypy && pytest` and `PYTHONPATH=tools python -m scrap tests`; act on SCRAP `HIGH` findings for the new test files or explain them in the pull request; fill in every section of the pull request template, including the `start_manifest` signature widening from T016

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: none
- **Foundational (Phase 2)**: after Setup; blocks every story
- **US1 (Phase 3)**: after Foundational
- **US2 (Phase 4)**: after US1 (`RunManifest` subclasses `Record`; `ArtifactWriter.write_record` uses `dump_record`)
- **US3 (Phase 5)**: after US2 (`hyperdime diagnose` writes through `ArtifactWriter`)
- **US4 (Phase 6)**: after Setup; independent of US1–US3 except T026 and T030, which run the full suite and should come last
- **Polish (Phase 7)**: T031–T032 any time after US3; T033 after US2; T034–T035 last

### User Story Dependencies

- **US1 (P1)**: independent; the MVP
- **US2 (P1)**: needs US1
- **US3 (P2)**: needs US1 and US2
- **US4 (P3)**: needs only Setup; can be done by a second agent in parallel with US1–US3

### Within Each User Story

- Tests first, and they must fail before implementation
- Shared types before records, records before helpers, helpers before re-exports
- One Conventional Commit per ticket: `feat(contracts): add versioned records` (US1), `feat(contracts): add run manifests, seeding, and the artifact writer` (US2), `feat(cli): add hyperdime diagnose and validated configs` (US3), `build(deps): lock dependencies with uv and add pre-commit` (US4)

### Parallel Opportunities

- T002 with T001; T004 with T003
- T006 then T007 (same file); T008–T010 in order (same file)
- T012, T013, T014 (test files) together; T015 in parallel with T016
- T019, T020, T021 together
- US4 (T025–T030) in parallel with US1–US3 on a separate worktree, merging after them
- T031, T032 together

---

## Parallel Example: User Story 2

```bash
# Tests, in parallel:
Task: "Seeding tests in tests/unit/contracts/test_seeding.py (T012)"
Task: "Manifest tests in tests/unit/contracts/test_manifests.py (T013, then T014 in the same file)"
# Then implementation, two files at once:
Task: "seed_everything in src/hyperdime/contracts/seeding.py (T015)"
Task: "RunManifest and start/finish_manifest in src/hyperdime/contracts/manifests.py (T016)"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Setup (T001–T002) and Foundational (T003–T005)
2. US1 (T006–T011): records usable by the F2 loaders
3. **STOP and validate**: `pytest tests/unit/contracts/test_schemas.py`

### Incremental Delivery

1. US1 → records
2. US2 → manifests and the M0 gate (SC-003)
3. US3 → configs and `hyperdime diagnose` (SC-005)
4. US4 → lockfile, pre-commit, CI on `uv` (SC-006)
5. Polish → docs, the spike's manifests on `RunManifest`, quickstart run

---

## Notes

- `[P]` tasks touch different files and depend on no incomplete task
- Tests run on CPU without network; the git-dirty test uses a temporary repository, not this one
- Commit after each ticket; never edit experiment numbers by hand
- F1 is unaffected by ADR-0003; its F3 changes begin when F3 is specified (roadmap task T032)
