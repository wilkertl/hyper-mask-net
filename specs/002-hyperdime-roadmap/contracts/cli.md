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
