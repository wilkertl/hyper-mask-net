# Contract: F5 commands

Conventions of [the 001 CLI contract](../../001-m0-foundation/contracts/cli.md): config and
overrides, exit code 2 for invalid configs, artifacts only through `ArtifactWriter`, the artifact
directory printed on stdout.

## `hyperdime train-selector --recipe retrieval-grid data=<env>`

- Reads: F2 dataset, embeddings, and negatives of `<env>` (training and validation judgments
  only); for `init=l2s` cells, the latest E1 `selectors/seeded/<env>/seed-0` checkpoint, refusing
  (exit 1) on a payload hash mismatch.
- Trains the 32 cells of research D6 with seed 0, scores them on validation (full corpus, nine
  budgets), and selects one.
- Writes: `selectors/retrieval-grid/<env>/<run_id>/` (data model).

## `hyperdime train-selector --recipe retrieval-seeded data=<env> run.seed=<n>`

- Reads: the latest `selectors/retrieval-grid/<env>` selection and the inputs it names; refuses
  (exit 1) if the current F2 inputs differ; for an l2s cell, the E1 seeded checkpoint of seed `n`.
- Writes: `selectors/retrieval-seeded/<env>/seed-<n>/<run_id>/`.

## `hyperdime compare-selectors [--envs scifact nfcorpus msmarco] [--seeds 0 1 2] [--chunk-size N]`

- Reads: retrieval-seeded and E1 seeded selectors for every environment and seed, the grid's
  family bests (seed 0), F3's global selector and global-importance τ, F2 embeddings and datasets.
- Refuses (exit 1) on a missing seed, a seeded cell that differs from the grid's selection, or a
  payload hash mismatch.
- Writes: `evaluation/S1/all/<run_id>/` (data model). Test and TREC DL judgments score rankings
  only; validation judgments only for the fit diagnostic; nothing is trained.
- Deterministic: identical result files on the same inputs, apart from timestamps and run ids.

## `hyperdime report S1 [--artifacts artifacts] [--report-dir reports/S1]`

- Reads: the latest complete `evaluation/S1/all` run and the grid and seeded `selection.json`s.
- Writes: `reports/S1/README.md` and `reports/S1/tables/` (`progression.csv` per test set and
  method, `comparison.csv` (vs L2S: mean over budgets and k = 307, CI, p, Holm), `vs_full.csv`,
  `trec.csv`, `families.csv`, `overlap.csv`, `fit.csv`, `grid.csv`, `s1.csv` (criteria)); identical
  bytes when run twice.

## `hyperdime decide S1 [--adr docs/decisions/ADR-0009-s1-selector-objective.md]`

- Reads: the same result files as `report S1`.
- Writes: `decisions/S1/<run_id>/gate_decision.json` and the proposed ADR (every value generated,
  decision left to the researcher); refuses to overwrite an accepted ADR.
