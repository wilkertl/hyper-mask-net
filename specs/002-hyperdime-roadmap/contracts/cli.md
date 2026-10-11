# Contract: `hyperdime` command line (whole project)

All subcommands share the conventions of [001 cli contract](../../001-m0-foundation/contracts/cli.md):
Hydra config and overrides, exit code 2 for invalid configs, artifacts only through
`ArtifactWriter`, artifact directory printed on stdout.

| Subcommand | Feature | Reads | Writes (`artifacts/<kind>/`) |
|---|---|---|---|
| `diagnose` | F1 | config | `diagnose/` manifest only |
| `prepare-data` | F2 | Hub datasets (network) | `datasets/<env>/` normalized records + DatasetManifest |
| `embed` | F2 | `datasets/<env>` | `embeddings/<env>/` queries and documents |
| `mine-negatives` | F2 | embeddings, train qrels | `negatives/<env>/` |
| `build-targets` | F3 | embeddings, negatives, train/validation qrels | `oracle/<env>/` TargetImportance |
| `train-selector` | F3, F4 | targets | `selectors/<kind>/<env>/` SelectorCheckpoint |
| `evaluate` | F3+ | embeddings, selector or baseline, test qrels | `evaluation/<experiment>/` EvaluationResult |
| `transfer-matrix` | F4 | per-domain selectors, test bundles | `evaluation/E1/` TransferCells (no training) |
| `build-space` | F5 | embeddings (+ calib queries per protocol) | `space/<protocol>/<env>/` SpaceDescriptor |
| `probe` | F5 | descriptors, SelectorProperties | `probes/` ProbeResults |
| `train-hypernet` | F6 | meta-train descriptors and targets | `hypernet/` checkpoint + EpisodeLog |
| `adapt` | F6, F7 | hypernet, descriptor of a new environment | `modulations/<protocol>/<env>/` ModulationPackage |
| `benchmark` | F7 | embeddings, selectors | `efficiency/` EfficiencyRecords |
| `report` | F3+ | result files + manifests | `reports/<experiment>/` tables (versioned) |
| `decide` | each gate | result files | draft ADR and GateDecision JSON for review |

Rules:

- Only `prepare-data` may use the network. For licensed collections (Robust04) it reads
  `data.local_path` instead and exits with a clear skip message, not an error, when it is unset.
- `embed` embeds each `corpus_group` once; environments sharing a corpus reuse the artifact.
  It writes shards and resumes from the last complete shard.
- `evaluate` also runs an environment's `eval_query_sets` (e.g. TREC DL'19/'20/DL-HARD for
  MS MARCO), each as its own `EvaluationResult`.
- `evaluate` is the only subcommand that loads test qrels; `train-*`, `mine-negatives`,
  `build-targets`, `build-space` (P0/P1), and `adapt` (P0/P1) must fail if a config points them at
  test labels.
- `train-hypernet` validates every environment against the frozen `MetaSplit` and exits 2 on any
  meta-test environment.
- `report` output is deterministic: running it twice on the same result files gives identical bytes.

## Spike F0: `python -m hyperdime.spike <step>`

The spike has its own small CLI (argparse, lightweight JSON manifests; plan Complexity Tracking).
F1 later moves its manifests onto `RunManifest`.

| Step | Reads (`artifacts/spike/`) | Writes | Labels it may read |
|---|---|---|---|
| `prepare` | BEIR zips (network) | `<env>/` splits, embeddings, negatives | training labels for negatives |
| `train` | `<env>/` | `<env>/selectors/` | training labels |
| `train-global` | `<env>/`, `<env>/selectors/temperature.json` | `global/` | training labels |
| `analyze` | all of the above | `analysis/` | evaluation labels, for scoring only |
| `diagnose` | all of the above, `analysis/` | `diagnostics/diagnostics.json` + manifest (research R20) | training labels for oracle profiles, models, and the probe; evaluation labels for scoring only |
| `report` | `analysis/`, `diagnostics/` if complete | `reports/F0/README.md` | none |
| `all` | | `prepare` → `train` → `train-global` → `analyze` → `report` | |

Rules:

- `diagnose` is not part of `all`, so the existing F0 run reproduces unchanged. It refuses to run
  unless `analyze` and every input it reads are `complete`. It is deterministic for fixed seeds.
- `diagnose` takes the same `--env`, `--seeds`, and `--root` options, plus `--permutations`
  (default 1,000) and `--half-splits` (default 200). It needs no GPU and no network.
- `report` adds a "Diagnostics (research R20)" section with D1, D2, D3, the values of rules O1, T,
  and S, and the recommendation. The section exists only when `diagnostics/` is complete, so a
  report regenerated before `diagnose` runs is byte-identical to the current one.
