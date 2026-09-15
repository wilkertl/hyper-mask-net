# Experiments

Each experiment gets a folder, created on its `exp/` branch:

| Folder | Experiment | Gate |
|---|---|---|
| `00_baseline_reproduction/` | E0 | G1 |
| `01_selector_shift/` | E1 | G2 |
| `02_space_signal/` | E2 | G3 |
| `03_hypernetwork/` | E3 | G4 |
| `04_ablations/` | E4, E5 | — |
| `05_efficiency/` | E6 | G5 |

A folder holds the config composition, the single command that reproduces the experiment, and the
generated report. Result files and manifests live in `artifacts/`; tables and figures are generated
from them into `reports/`. Register every experiment in
[docs/experiment_registry.md](../docs/experiment_registry.md).
