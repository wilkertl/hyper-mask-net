# Contract: artifact and config layout on disk

## Artifacts (gitignored)

```text
artifacts/
└── <kind>/                         # e.g. diagnose, embeddings, oracle, selectors, evaluation
    ├── <run_id>/                   # published, immutable
    │   ├── manifest.json           # RunManifest, always present
    │   ├── *.json | *.jsonl        # records (schema_version inside each)
    │   └── *.npy | *.safetensors   # payloads referenced by PayloadRef
    └── .<run_id>.tmp-<uuid>/       # staging; ignored by readers
```

- A directory under `<kind>/` without `manifest.json` is invalid and is never read.
- `manifest.json` lists every other file in the directory with its SHA-256.
- `run_id` = `<UTC yyyymmddThhmmssZ>-<8 hex>`, so directories sort chronologically.

## Configs (versioned)

```text
configs/
├── model/qwen3_0_6b.yaml           # name, pinned revision, dim 1024, frozen, normalize
├── data/scifact.yaml               # environment, source, instruction, validation_fraction
├── run/default.yaml                # seed, device, dtype, output_dir
└── experiment/diagnose.yaml        # defaults: [model: qwen3_0_6b, data: scifact, run: default]
```

Later tickets add `data/{nfcorpus,msmarco}.yaml`, `space_rep/`, `selector/`, and `hyperhead/`
groups (plan §4). Every file carries `config_version: 1` at the composed root.
