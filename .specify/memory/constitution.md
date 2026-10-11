# HyperDIME-Qwen Constitution

## Core Principles

### I. Frozen Coordinate System

- Qwen3-Embedding-0.6B MUST stay frozen; its weights are never trained, fine-tuned, or modified.
- The embedding dimension is fixed at D = 1024, with last-token pooling, the Qwen query
  instruction format, and L2 normalization in the main protocol.
- The same text, config, and model revision MUST produce the same vector within numerical
  tolerance; embeddings are computed once, cached, and reused.
- Selection is query-only: it MUST NOT modify document embeddings or the index, and the masked
  query MUST NOT be re-normalized.

Rationale: every claim about which dimensions matter assumes one fixed coordinate system.

### II. Leakage-Free Protocols (NON-NEGOTIABLE)

- Test qrels, and any label from a meta-test environment, MUST NOT reach training, adaptation,
  hard-negative mining, oracle targets, descriptor construction, or normalization statistics.
- Hard-negative pools come from training queries only.
- Code paths used by adaptation protocols P0 and P1 MUST NOT read qrels; the meta-trainer MUST
  refuse to load meta-test environments.
- Environment-level splits (meta-train/validation/test) and query-level splits
  (train/validation/test) are never mixed; meta-test is evaluated once, after the
  meta-validation gate.
- Leakage rules are enforced by code and covered by tests, not by convention alone.

Rationale: a single leak invalidates the zero-shot and few-query claims the project exists to test.

### III. Contracts and Provenance First

- Every module has serializable, versioned input/output contracts in `hyperdime.contracts` before
  its neural implementation; later modules MUST reuse them, not define parallel schemas.
- Every artifact MUST be written together with a run manifest recording git commit, config hash,
  dataset hash, model revision, seed, device, dtype, and timestamps.
- Seeds come from a single seed manager; hashes are deterministic.
- Public APIs outside the current ticket MUST NOT change without justification in the pull request.

Rationale: pipeline differences must never be mistaken for model differences.

### IV. Evidence-Gated Complexity

- Work proceeds through gates G1 (valid baseline) → G2 (selector shift, H1) → G3 (space signal,
  H2) → G4 (hypernetwork, H3) → G5 (acceptable cost); no code for a later stage is written before
  its enabling gate passes. In particular, no hypernetwork code before G2.
- Baselines come first: Learning-to-Select is reproduced before any new model is introduced.
- Start with the lowest-capacity variant (linear selector, per-dimension MLP encoder, rank-limited
  modulation); add capacity only when an ablation shows the simpler version is limiting.
- A failed gate is recorded as evidence in an ADR; it is never answered by adding complexity.

Rationale: the architecture is only justified if each premise it relies on has been shown to hold.

### V. Exact, Generated Measurement

- Scientific results use exact dot-product scoring; FAISS is only for deployment analysis.
- Offline adaptation cost is reported separately from online per-query cost.
- Every table includes full-1024 and MRL prefix-k comparisons, states its protocol (P0, P1, P2, or
  in-domain), and is generated from result files that have run manifests.
- Experiment numbers MUST NOT be edited by hand.
- Significance uses paired per-query tests, multiple-comparison correction for hypothesis
  families, and confidence intervals rather than p-values alone.

Rationale: results must be reproducible from artifacts, not from memory or manual transcription.

### VI. Test-Verified Mathematics

- Every new behavior has a test; every critical mathematical function (normalization, oracle
  `e_q ⊙ (p − n)`, KL target, masks, top-k, descriptor statistics, low-rank reconstruction) has a
  unit test with a hand-computable result.
- Required properties are tested: `k = 1024` equals full scoring; zero modulation equals the base
  selector; descriptors are invariant to document/query order and equivariant to dimension
  permutation.
- Tests run on CPU without network access, using small fixtures; large external-dataset metrics
  are never used as unit tests.

Rationale: a silent sign or indexing error in the selection math corrupts every downstream result.

### VII. Independent, Replaceable Modules

- Space statistics, space encoder, hyper head, target selector, selection policy, and retrieval
  are independent modules with explicit contracts; there is no end-to-end "magic pipeline".
- Importance prediction is separate from the policy that decides how many dimensions to keep.
- Training is separate from inference: labels may build targets during meta-training, but
  inference in a new environment uses only the signals its protocol allows.

Rationale: each component must be ablatable and swappable to attribute effects correctly.

## Artifacts and Data Handling

- Datasets, embeddings, and checkpoints MUST NOT be downloaded into or committed to Git; they live
  in `artifacts/` (gitignored) or external storage. Only manifests and configs are versioned.
- Each artifact carries a SHA-256 and provenance through its manifest.
- DVC or similar tooling is optional and MUST NOT become a critical dependency during bootstrap.

## Development Workflow and Quality Gates

- One ticket per branch, `agent/<ticket>-<slug>`, branched from `master`; experiment runs use
  `exp/<experiment>-<slug>` and touch only `configs/`, `experiments/`, and `reports/`.
- Before editing, read `AGENTS.md`, `docs/architecture.md`, `docs/protocols.md`, the ticket, and the
  relevant contracts, and list the files to be changed.
- Before committing, `ruff check . && ruff format --check . && mypy && pytest` MUST pass.
- Commits follow Conventional Commits scoped by package, e.g. `feat(oracle): ...`.
- Pull requests fill every section of the template: contract changed, files changed, tests added,
  commands run, expected artifacts or regressions, protocol changes, and invalidated results.
- Non-trivial architectural decisions are recorded in `docs/decisions/ADR-XXXX-<slug>.md`.

## Governance

- This constitution supersedes other practices where they conflict; `AGENTS.md`,
  `docs/architecture.md`, and `docs/protocols.md` elaborate it and MUST stay consistent with it.
- Amendments are made by pull request that updates this file, states the version bump and
  rationale, and updates any dependent documents in the same change. Changes to the scientific
  protocol also carry the `protocol-change` label.
- Versioning follows semantic versioning: MAJOR for removing or redefining a principle, MINOR for a
  new principle or materially expanded guidance, PATCH for clarifications.
- Every pull request review and every `/speckit-plan` Constitution Check verifies compliance;
  any deviation is justified in the plan's Complexity Tracking table or rejected.

**Version**: 1.0.0 | **Ratified**: 2026-09-25 | **Last Amended**: 2026-09-25
