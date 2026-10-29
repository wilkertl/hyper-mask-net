# Contract: F9 commands

Conventions of [the 001 CLI contract](../../001-m0-foundation/contracts/cli.md): config and
overrides, exit code 2 for invalid configs, artifacts only through `ArtifactWriter`, the artifact
directory printed on stdout. F8 commands and their outputs are unchanged.

## `hyperdime train-selector --recipe budget-grid | weighted-aligned-grid data=<env>`

- Reads: F2 dataset, embeddings, and negatives of `<env>` (training and validation judgments
  only); the corpus variance prior.
- Trains the 8 cells of research D6 with seed 0 under the aligned checkpoint rule (D3), scores
  every candidate on the D4 validation subset, the kept states on all validation queries, and
  selects one cell among those passing the guard (D5).
- Writes: `selectors/budget-grid/<env>/<run_id>/` or `selectors/weighted-aligned-grid/...`, with
  `selection.json` (data model), `history.json`, one checkpoint per cell's kept state.

## `hyperdime train-selector --recipe budget-seeded | weighted-aligned-seeded data=<env> run.seed=<n>`

- Reads: the latest grid of the same track; refuses (exit 1) if the current F2 inputs differ.
- Retrains the selected cell with seed `n` under the same rule and subset.
- Writes: `selectors/<track>-seeded/<env>/seed-<n>/<run_id>/`.

## `hyperdime compare-selectors --experiment S4 [--envs ...] [--seeds 0 1 2] [--chunk-size N]`

- Tracks scored beyond L2S and S1: `geometric`, `weighted`, `weighted-aligned`, `budget`
  (candidate last). Budget selectors are scored with per-budget importance in the same corpus pass.
- Additionally scores the validation set for the D5 ablation and writes `diagnostics.json` and
  `ablation.json`.
- Refuses (exit 1) on a missing seed, a seeded cell that differs from its grid's selection, or a
  payload hash mismatch. Writes `evaluation/S4/all/<run_id>/`. Test and TREC DL judgments only
  score rankings; nothing is trained.

## `hyperdime report S4 [--artifacts artifacts] [--report-dir reports/S4]`

- Reads the latest complete `evaluation/S4/all` run and the grid and seeded `selection.json`s of
  both tracks. Writes `reports/S4/README.md` and `tables/` (data model); identical bytes on rerun;
  refuses with fewer than three seeds per environment and track.

## `hyperdime decide S4 [--adr docs/decisions/ADR-0012-s4-budget-prior.md]`

- Writes `decisions/S4/<run_id>/gate_decision.json` with a manifest and, with `--adr`, the drafted
  ADR-0012 (status "proposed"); never overwrites an ADR whose status is not "proposed".

## Library (public additions)

- `baselines.learning_to_select.BudgetGeometricSelector(variance, budgets=BUDGETS)`.
- `baselines.methods.budget_importance(queries, model) -> dict[int, Tensor]`.
- `evaluation.statistics.noninferiority_test(a, b, margin, n_permutations=10_000, seed=0)`.
- `selection.diagnostics.prior_share(model, queries, k) -> float` (research D5).
- `RetrievalConfig.checkpoint_rule`, `RetrievalConfig.nonnegative_prior` (defaults keep F5–F8).
