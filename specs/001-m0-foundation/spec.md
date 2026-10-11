# Feature Specification: M0 Foundation — Contracts and Reproducibility

**Feature Branch**: `agent/M0-foundation`

**Created**: 2026-09-25

**Status**: Draft

**Input**: User description: "M0 foundation: contracts and reproducibility for HyperDIME-Qwen (backlog tickets M0.1–M0.4, plan section 5 "M0 — Bootstrap, contracts, and reproducibility"). Pinned dependencies with a lockfile and pre-commit; versioned, serializable schemas for QueryRecord, DocumentRecord, QrelsRecord, EmbeddingArtifact, SpaceDescriptor, SelectorCheckpoint, EvaluationResult; a RunManifest (git commit, config hash, dataset hash, model revision, seed, device, dtype, timestamps) with deterministic hashing and a single seed manager; a config system and a diagnostic CLI that loads a config and writes a manifest without training anything. Tests: serialization round-trip, deterministic hashes, invalid config fails explicitly. Gate: no later module may write an artifact without a manifest."

## User Scenarios & Testing *(mandatory)*

The users of this feature are the researcher and the coding agents who implement later modules
(M1–M12). They need one shared vocabulary of records, a guarantee that every artifact can be traced
to the exact code, config, data, and seed that produced it, and an environment that behaves the same
on every machine.

### User Story 1 - Shared, versioned records (Priority: P1)

A developer implementing a later module (for example the dataset loaders or the oracle) describes
its inputs and outputs using the project's shared record types instead of ad-hoc dictionaries. They
save a record to disk and load it back unchanged, and an invalid record is rejected with a message
that names the offending field.

**Why this priority**: Every later module depends on these records; the plan forbids data and
embedding modules from defining parallel schemas. Nothing else in the milestone can start cleanly
without them.

**Independent Test**: Create one valid instance of each of the seven record types, save and reload
each, and compare; then submit deliberately invalid instances and check the error messages.

**Acceptance Scenarios**:

1. **Given** a valid instance of any of the seven record types, **When** it is saved and reloaded,
   **Then** the reloaded record equals the original, including its schema version.
2. **Given** a record with a missing required field, a wrong type, or a vector whose width is not
   1024, **When** it is validated, **Then** validation fails with a message naming the record type
   and the field.
3. **Given** a saved record whose schema version is not supported by the current code, **When** it
   is loaded, **Then** loading fails explicitly rather than silently misreading it.

---

### User Story 2 - Traceable artifacts through run manifests (Priority: P1)

A developer runs any step that produces an artifact. The artifact is always written together with a
run manifest that records the code commit, config fingerprint, dataset fingerprint, model revision,
seed, device, numeric precision, and start/end times. A step that tries to write an artifact without
a manifest is stopped.

**Why this priority**: This is the M0 gate ("no later module may write an artifact without a
manifest") and constitution principle III. Without it, pipeline differences can be mistaken for
model differences.

**Independent Test**: Write a small artifact through the project's artifact writer with and without
a manifest; compute fingerprints of the same config with keys in different orders; seed two runs
identically and compare their random draws.

**Acceptance Scenarios**:

1. **Given** two configs with identical content but different key order, **When** both are
   fingerprinted, **Then** the fingerprints are identical; **and** changing any value changes the
   fingerprint.
2. **Given** the same file fingerprinted twice, on any machine, **When** the fingerprints are
   compared, **Then** they are identical.
3. **Given** an attempt to write an artifact without a manifest, **When** the write is requested,
   **Then** it is refused with an explicit error and no artifact file is left behind.
4. **Given** a seed, **When** two runs are initialized with it, **Then** their random draws from
   every randomness source used by the project are identical.
5. **Given** an uncommitted change in the working tree, **When** a manifest is created, **Then**
   the manifest records that the tree was dirty.

---

### User Story 3 - Configured runs and a diagnostic command (Priority: P2)

The researcher describes a run entirely through versioned config files (model, data, experiment
groups) and runs a diagnostic command that loads and validates the config and writes a manifest,
without loading the model or training anything. An invalid config fails before anything is written.

**Why this priority**: It proves the config, validation, and manifest pieces work together
end to end, and gives every later CLI a pattern to follow. It depends on Stories 1 and 2.

**Independent Test**: Run the diagnostic command on the sample config and inspect the manifest it
writes; run it on configs with a missing field, an unknown field, and a wrong dimension.

**Acceptance Scenarios**:

1. **Given** the sample config, **When** the diagnostic command runs, **Then** it exits
   successfully and writes a manifest containing every required field.
2. **Given** a config with a missing required field, an unknown key, or a value that violates a
   hard constraint (for example an embedding dimension other than 1024), **When** the diagnostic
   command runs, **Then** it exits with a non-zero status, names the problem, and writes nothing.
3. **Given** the same config run twice, **When** the two manifests are compared, **Then** the
   config fingerprints are identical and only the timestamps differ.

---

### User Story 4 - Locked environment and local checks (Priority: P3)

A new contributor or agent clones the repository, installs the locked dependency versions by
following the README, and runs the same lint, format, type, and test checks locally that CI runs,
before each commit.

**Why this priority**: It prevents version drift from changing results, but the other stories can
be built and tested in the current environment first.

**Independent Test**: In a fresh clone, follow the README install steps, run the pre-commit checks
on all files, and confirm CI passes on the same lockfile.

**Acceptance Scenarios**:

1. **Given** a fresh clone, **When** the README install steps are followed, **Then** the installed
   versions of all core dependencies match the committed lockfile.
2. **Given** the repository, **When** the pre-commit checks run on all files, **Then** they pass,
   and vendored tooling directories (spec-kit scaffolding) are excluded from lint and format.
3. **Given** a pull request, **When** CI runs, **Then** it installs from the same lockfile and runs
   the same checks.

### Edge Cases

- A config value that is a float with different textual representations (e.g. `0.1` vs `1e-1`)
  fingerprints identically once loaded.
- A manifest is created outside a git checkout or with git unavailable: the commit is recorded as
  unknown and flagged, never silently omitted.
- A run with no dataset (such as the diagnostic command) records the dataset fingerprint explicitly
  as "none" rather than leaving it empty.
- A write is interrupted: no partial artifact appears without its manifest.
- A record contains non-finite values (NaN or infinity) in a vector: validation rejects it.
- The requested device (e.g. a GPU) is not available: the manifest records the device actually
  used.

## Requirements *(mandatory)*

### Functional Requirements

**Records (M0.2)**

- **FR-001**: The system MUST provide seven record types — QueryRecord, DocumentRecord,
  QrelsRecord, EmbeddingArtifact, SpaceDescriptor, SelectorCheckpoint, EvaluationResult — each
  carrying an explicit schema version.
- **FR-002**: Every record type MUST save to and load from a human-readable, language-neutral
  format such that reloading yields a record equal to the original.
- **FR-003**: Validation MUST reject missing required fields, wrong types, non-finite numbers, and
  dimension mismatches (the embedding width is fixed at 1024), with a message naming the record
  type and field.
- **FR-004**: Loading a record with an unsupported schema version MUST fail explicitly.
- **FR-005**: Large numeric payloads (embedding matrices, checkpoints) MUST be referenced by path
  and fingerprint from their record, not embedded in it.

**Manifests, hashing, and seeds (M0.3)**

- **FR-006**: The system MUST provide a run manifest recording: code commit and dirty flag, config
  fingerprint, dataset fingerprint (or explicit "none"), model revision, seed, device, numeric
  precision, start and end timestamps in UTC, and the manifest's own schema version.
- **FR-007**: Config fingerprints MUST be independent of key order and MUST change when any value
  changes. File fingerprints MUST use SHA-256.
- **FR-008**: A single seed manager MUST seed every randomness source the project uses and set
  deterministic-execution options where available.
- **FR-009**: The system MUST provide one artifact writer that refuses to write any artifact
  without a manifest and never leaves an artifact on disk without its manifest.

**Configs and diagnostic command (M0.4)**

- **FR-010**: Runs MUST be described by versioned config files organized in groups (at least
  model and data), with a sample config for Qwen3-Embedding-0.6B and one dataset.
- **FR-011**: Config validation MUST reject missing required fields, unknown keys, and violations of
  hard constraints (embedding dimension 1024, frozen encoder) before any output is written.
- **FR-012**: A diagnostic command MUST load and validate a config, and write a manifest for it,
  without loading model weights, downloading data, or training anything.

**Tooling (M0.1)**

- **FR-013**: All runtime and development dependencies MUST be pinned in a committed lockfile, and
  the README MUST document installing it for CPU and for CUDA.
- **FR-014**: Local pre-commit checks MUST run lint, format, and type checks matching CI; CI MUST
  install from the lockfile.
- **FR-015**: Lint and format checks MUST exclude vendored tooling directories.

**Cross-cutting**

- **FR-016**: All tests for this feature MUST run on CPU without network access.
- **FR-017**: `docs/architecture.md` MUST gain a summary of the record types and the manifest,
  and the components table MUST mark the implemented M0 pieces.

### Key Entities

- **QueryRecord**: one query in an environment — identifier, text, instruction, split.
- **DocumentRecord**: one corpus document — identifier, title, text.
- **QrelsRecord**: a relevance judgment — query identifier, document identifier, graded relevance,
  split.
- **EmbeddingArtifact**: describes a stored embedding matrix — role (queries or documents), row
  count, width (1024), precision, normalization, model revision, ordered identifier list or
  identifier map, payload path and fingerprint.
- **SpaceDescriptor**: per-environment, per-dimension statistics — environment, protocol (P0/P1/P2),
  1024 × h values or a payload reference, feature names, descriptor version. Provisional until the
  descriptor is frozen in M5.5.
- **SelectorCheckpoint**: a trained selector — kind (linear, global, per-domain, generated),
  environment, parameter payload path and fingerprint, training manifest reference. Provisional
  until M7.
- **EvaluationResult**: metrics for one run — environment, protocol, split, scoring mode (exact),
  aggregate metrics, per-query metrics, manifest reference.
- **RunManifest**: the provenance record attached to every artifact (fields in FR-006).
- **RunConfig**: the validated description of a run, composed from config groups.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of the seven record types survive a save/reload cycle unchanged.
- **SC-002**: The same config fingerprints identically in 100% of trials, across key orderings and
  across at least two machines or processes.
- **SC-003**: 0 artifacts can be written without a manifest; every attempt in the test suite is
  refused.
- **SC-004**: Every invalid-input test case (records and configs) produces an error naming the
  offending field; none fails silently or with an unrelated error.
- **SC-005**: The diagnostic command produces a complete manifest for the sample config in under
  5 seconds on a laptop CPU, with no network access.
- **SC-006**: A fresh clone reaches a passing local check run (lint, format, types, tests) by
  following only the README.
- **SC-007**: The existing 27 unit tests keep passing.

## Assumptions

- Users are the researcher and coding agents; there is no end-user interface beyond the command
  line.
- SpaceDescriptor and SelectorCheckpoint are defined now in provisional form so that later tickets
  extend rather than invent them; their shapes may change at M5.5 and M7 with a schema-version bump.
- A dirty working tree is recorded in the manifest, not refused, so exploratory runs stay possible;
  experiment branches are expected to run from clean trees.
- Existing functions (`split_ids`, `exact_search`, metrics, and the others) keep their signatures;
  adopting the new records inside them is left to the tickets that own those modules.
- Data-version tooling such as DVC stays out of scope (constitution: not a bootstrap dependency).
- Python 3.11 or later, as already declared by the project.
