# Contract: F4 commands

Conventions of [the 001 CLI contract](../../001-m0-foundation/contracts/cli.md): config and
overrides, exit code 2 for invalid configs, artifacts only through `ArtifactWriter`, the artifact
directory printed on stdout.

## `hyperdime train-selector --recipe seeded data=<env> run.seed=<n>`

- Reads: the latest `selectors/candidates/<env>` (its `wide-grid` best cell and inputs), and the
  F2 inputs it names; refuses (exit 1) if the inputs differ from the F3 run's.
- Writes: `selectors/seeded/<env>/seed-<n>/<run_id>/` (`selector.safetensors`, `selector.json`,
  `selection.json`).
- Labels: training and validation labels only (unchanged from F3).

## `hyperdime transfer-matrix [--envs scifact nfcorpus msmarco] [--seeds 0 1 2] [--chunk-size N]`

- Reads: the latest seeded selector per environment and seed, F3's `selectors/global/all`, F3's
  official selection (for the global-importance τ), F2 embeddings and datasets.
- Refuses (exit 1) when any environment has fewer than the requested seeds or a checkpoint's
  payload hash differs from its record.
- Writes: `evaluation/E1/all/<run_id>/` as listed in research D10.
- Labels: test and TREC DL judgments to score rankings only; training labels for the fit test (D2)
  and fixed subsets (D6); no model is trained or updated.
- Deterministic: two runs on the same inputs give identical result files apart from manifest
  timestamps and run ids.

## `hyperdime report E1 [--artifacts artifacts] [--report-dir reports/E1]`

- Reads: the latest complete `evaluation/E1/all` run and the seeded selectors' `selection.json`.
- Writes: `reports/E1/README.md` and `reports/E1/tables/` (`transfer.csv`, `trec.csv`,
  `methods.csv` (every method on every query set, FR-007), `seeds.csv`, `behavior.csv`,
  `noise_floor.csv`, `environment_pairs.csv` (cross against within overlap per environment pair,
  FR-011), `fit.csv`, `fixed_subset.csv`, `payoff.csv`, `g2.csv`); identical bytes when run
  twice.

## `hyperdime decide G2 [--adr docs/decisions/ADR-0007-g2-selector-shift.md]`

- Reads: the same result files as `report E1`.
- Writes: `decisions/G2/<run_id>/gate_decision.json` and the proposed ADR (status "proposed",
  every value generated, with the decision section left to the researcher). Refuses to overwrite an
  existing ADR whose status is not "proposed".
