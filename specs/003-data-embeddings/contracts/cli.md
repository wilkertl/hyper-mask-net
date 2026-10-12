# Contract: F2 commands

All follow `hyperdime diagnose` ([F1 cli contract](../../001-m0-foundation/contracts/cli.md)):
Hydra config (`--config-dir configs --config-name experiment/prepare`, overrides such as
`data=nfcorpus`), exit 2 with `error: invalid config: ...` and nothing written, exit 1 on any other
error with nothing visible, the published artifact directory on stdout.

| Command | Reads | Writes | Options |
|---|---|---|---|
| `hyperdime prepare-data` | pinned sources (network) | `artifacts/datasets/<env>/<run_id>/` | `--output` |
| `hyperdime embed` | latest `datasets/<env>` | `artifacts/embeddings/<env>/<run_id>/` | `--output`, `--backend local\|vllm`, `--base-url`, `--shard-size`, `--batch-size` |
| `hyperdime mine-negatives` | latest `datasets/<env>` (no test judgments), latest `embeddings/<env>` | `artifacts/negatives/<env>/<run_id>/` | `--output`, `--depth 1000`, `--count 64` |

- `embed` reuses a published embedding of the same environment and settings and prints its
  directory instead of recomputing; a published embedding with different settings is refused with
  the differing keys named (exit 1). Interrupted runs resume from `artifacts/.work/`.
- `mine-negatives` cannot be configured to read test judgments; it loads the environment with
  `include_test_qrels=False`.
- Only `prepare-data` uses the network.
