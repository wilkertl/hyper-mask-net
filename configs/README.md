# Configs

Hydra/OmegaConf configuration groups, introduced by ticket M0.4.

| Group | Contents |
|---|---|
| `model/` | `qwen3_0_6b.yaml`: model revision, dimension, pooling, normalization |
| `data/` | One file per environment (`scifact`, `nfcorpus`, `msmarco`): instruction, sampling, splits |
| `space_rep/` | Descriptor feature groups |
| `selector/` | Learning-to-Select and low-rank selector settings |
| `hyperhead/` | Space Encoder and Hyper Head settings |
| `experiment/` | Compositions for experiments E0–E6 |

Configs are versioned, and their hash is recorded in every run manifest.
