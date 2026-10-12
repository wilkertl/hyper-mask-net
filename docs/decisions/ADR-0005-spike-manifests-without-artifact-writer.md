# ADR-0005: Spike outputs use RunManifest but not ArtifactWriter

- **Status:** accepted
- **Date:** 2026-10-12
- **Issue:** F1 task T033 (`specs/001-m0-foundation/tasks.md`)

## Context

F1 adds `RunManifest` and `ArtifactWriter`, which publishes each artifact as an immutable
`artifacts/<kind>/<run_id>/` directory in one atomic rename (FR-009). The F0 spike predates both:
each step writes into a fixed, in-place directory under `artifacts/spike/` with a lightweight JSON
manifest marked `running` before its outputs and `complete` after them. The roadmap plan (F0 row of
Complexity Tracking) promised the spike would be upgraded to F1's contracts rather than rewritten.

Two spike behaviors do not fit `ArtifactWriter`:

- `prepare` embeds corpora in resumable chunks inside the cache directory, so an interrupted
  multi-hour embedding job continues where it stopped; an atomic, write-once directory would
  restart from zero.
- Later steps read fixed paths (`artifacts/spike/<env>/selectors/`), and the F0 results, report,
  and ADR-0003 cite those paths.

## Decision

- Spike manifests are `RunManifest` records written through `start_manifest` and `finish_manifest`:
  an unset `finished_at` means running, and a completed manifest keeps the run's id and start time
  and lists the files in its directory with their SHA-256.
- The spike keeps its in-place directories and does not use `ArtifactWriter`. This is the only
  exception; every new module writes through `ArtifactWriter`.
- Manifests written before F1 (with a `status` field) remain valid inputs, so the F0 artifacts
  behind `reports/F0/README.md` and ADR-0003 are not invalidated.

## Evidence

`tests/integration/test_spike_pipeline.py`: a spike step's manifest validates as `RunManifest` and
lists its files; a completed manifest keeps the running one's run id; a pre-F1 manifest is still
accepted and still refused when not complete. Regenerating the F0 report from the existing
artifacts gives a byte-identical `reports/F0/README.md`.

## Consequences

- Spike provenance now has the same fields as every other artifact (config hash, versions,
  determinism, UTC start and end), and readers handle one manifest type.
- Spike outputs are mutable in place, unlike published artifacts; spike results stay exploratory,
  and anything reported is rerun under F2–F4 through `ArtifactWriter`.
- The spike's `seed` field is its base seed (0); per-step seed lists stay in the manifest's config.
