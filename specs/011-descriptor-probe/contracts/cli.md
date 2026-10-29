# Contract: F10 commands

Conventions of [the 001 CLI contract](../../001-m0-foundation/contracts/cli.md): config and
overrides, exit code 2 for invalid configs, artifacts only through `ArtifactWriter`, the artifact
directory printed on stdout. Existing commands and their outputs are unchanged.

## `hyperdime describe-space data=<env> [--protocol P0|P1] [--chunk-size N] [--calibration 10000] [--background 100]`

- Reads, through `space.inputs.load_unlabeled` only: the latest F2 dataset manifest and
  `queries.jsonl` of `<env>` (query ids and splits), and its embeddings. Never opens a qrels file.
- Computes research D3–D4 (two streamed document passes, the quantile sample); with `--protocol
  P1` also D5 (the calibration queries' exact top-20 and background sample). `P0` until stage 2
  lands, then `P1` by default (D13).
- Writes `descriptors/<env>/<run_id>/`: `descriptor-P0.json` + `descriptor-P0.npy`,
  `descriptor-P1.json` + `descriptor-P1.npy` (P1 only; `SpaceDescriptor` records), `covariance.npy`
  (diagnostic), `inputs.json` (dataset and embeddings run ids, sample sizes, seeds).
- Refuses (exit 1) if the embeddings were not made from the dataset (manifest hash).

## `hyperdime probe-descriptors --experiment E2 [--envs ...] [--seeds 0 1 2] [--permutations 1000] [--targets-only]`

- Reads the latest descriptor of each environment, the validation-split query embeddings (no
  judgments), the latest S1 (`retrieval`) and S4 (`budget`) seeded selectors, checked against
  their grids' selections and payload hashes, and, for context only, the latest
  `evaluation/S4/all` and `evaluation/E1/all` runs.
- Computes targets (D6) and the shift check (D0); unless `--targets-only`, the ladder over the
  sets the descriptors allow, both probe forms, baselines, and permutation tests (D7–D10).
- Writes `evaluation/E2/all/<run_id>/`: `targets-<env>.npy` + `.json`, `check.json`, `probe.jsonl`,
  `coefficients.json`, `context.json`, `inputs.json`.
- Refuses (exit 1) on fewer than three environments or a missing seed; without
  `--targets-only`, on a missing descriptor or one not made from the current embeddings.

## `hyperdime report E2 [--artifacts artifacts] [--report-dir reports/E2]`

- Reads the latest complete `evaluation/E2/all` run (a `--targets-only` run gives a check-only
  report) and its descriptors. Writes `reports/E2/README.md` and `tables/` (data model);
  identical bytes on rerun.

## `hyperdime decide G3 [--adr docs/decisions/ADR-0013-g3-space-signal.md]`

- Writes `decisions/G3/<run_id>/gate_decision.json` with a manifest and, with `--adr`, the drafted
  ADR-0013 (status "proposed"); never overwrites an ADR whose status is not "proposed".

## Library (public additions)

- `space.inputs.load_unlabeled(root, environment) -> UnlabeledInputs`.
- `space.streaming.document_moments(documents, chunk_size, device) -> Moments`,
  `space.streaming.tail_share(documents, moments, chunk_size, device) -> ndarray`.
- `space.redundancy.redundancy_features(covariance) -> dict[str, ndarray]`.
- `space.interaction.interaction_features(queries, documents, top, background) -> dict[str, ndarray]`.
- `space.descriptor.build_descriptors(inputs, ...) -> tuple[ndarray, ndarray, list[str], list[str]]`,
  `space.descriptor.FEATURES` (names and groups in order).
- `evaluation.targets.target_profiles(selectors, queries, budgets) -> dict[str, ndarray]`.
- `evaluation.descriptor_probe.average_ranks`, `ridge_fit`, `ridge_predict`, `gcv_penalty`,
  `shift_probe`, `direct_probe`, `shift_check`, `spearman_brown`, `permutation_p`, `LADDER`.
