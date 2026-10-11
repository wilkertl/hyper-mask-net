# Contract: `hyperdime` command line

## `hyperdime diagnose`

Loads and validates a config and writes a manifest. It never loads model weights, downloads data,
or trains anything.

```text
hyperdime diagnose [--config-dir DIR] [--config-name NAME] [--output DIR] [OVERRIDE ...]
python -m hyperdime.cli diagnose ...
```

| Option | Default | Meaning |
|---|---|---|
| `--config-dir` | `configs` | Hydra config directory |
| `--config-name` | `experiment/diagnose` | primary config to compose |
| `--output` | value of `run.output_dir` (default `artifacts`) | artifact root |
| `OVERRIDE` | — | Hydra overrides, e.g. `run.seed=7 data=nfcorpus` |

### Behavior

1. Compose and validate the config.
2. Seed with `run.seed`; resolve `run.device` (`auto` → `cuda:0` if available, else `cpu`).
3. Write `<output>/diagnose/<run_id>/manifest.json` through `ArtifactWriter`, with
   `dataset_hash = "none"` and `model_revision = model.revision`.
4. Print the artifact directory on stdout (one line) and a human summary on stderr.

### Exit codes

| Code | Meaning | Output written |
|---|---|---|
| 0 | success | manifest directory |
| 2 | invalid config (missing field, unknown key, constraint violated) | nothing |
| 1 | any other error (I/O, unexpected) | nothing visible |

### Error format (stderr)

```text
error: invalid config: model.embedding_dim: Input should be 1024 (got 768)
```

One line per validation error, prefixed with the dotted field path.

### Performance

Completes in under 5 seconds on a laptop CPU (SC-005). Importing the CLI must not import
`transformers`, `datasets`, or `faiss`.
