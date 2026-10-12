# Implementation Plan: F3 Learning-to-Select Baseline and Gate G1

**Branch**: `agent/F3-l2s-baseline` | **Date**: 2026-10-12 | **Spec**: [spec.md](spec.md)

## Summary

On F2's artifacts, add exact masked evaluation (all methods and budgets in one streamed corpus
pass, AP and TREC thresholds), oracle targets and the ADR-0002 selector with device support, the
mandatory baselines, the ADR-0003 candidates chosen on validation, and the generated E0 report with
the G1 criteria ([research.md](research.md)). Runs on DataPolicy-4090 where F2's artifacts live.

## Technical Context

**Language/Version**: Python ≥ 3.11 (CI 3.12) · **Dependencies**: F1 contracts, F2 artifacts,
torch, numpy; `ir-measures` (`science` extra) for metric tests · **Storage**: `artifacts/
{selectors,evaluation}/`, `reports/E0/` (versioned) · **Testing**: pytest on CPU, synthetic
environments and fixtures · **Platform**: DataPolicy-4090 for runs · **Performance**: MS MARCO E0
grid under 3 hours (SC-007) · **Constraints**: no test label in training or selection; exact
scoring; report generated only from result files.

## Constitution Check

| Principle | Status | Evidence |
|---|---|---|
| I | ✅ | masked queries not re-normalized; documents untouched |
| II | ✅ | targets and selection from training/validation labels; oracle labeled non-deployable, never feeds training |
| III | ✅ | `SelectorCheckpoint`, `EvaluationResult`, manifests with `dataset_hash` |
| IV | ✅ | G1 is the gate F3 decides; candidates limited to ADR-0003, no new capacity |
| V | ✅ | exact scoring; full and prefix rows in every table; paired tests with Holm and CIs; generated report |
| VI | ✅ | hand-checked AP and thresholds, reference metrics, k = 1024 equality, trainer defaults unchanged |
| VII | ✅ | importance methods separate from the top-k policy |

**Result**: PASS.

## Project Structure

```text
src/hyperdime/evaluation/metrics.py      # + AP, relevance_threshold
src/hyperdime/evaluation/runs.py         # masked_rankings, evaluate_methods (new)
src/hyperdime/baselines/methods.py       # Prefix, Random, GlobalImportance, Selector, Oracle (new)
src/hyperdime/oracle/targets.py          # + build_targets from F2 artifacts
src/hyperdime/training/selector_trainer.py  # + device, anchor, patience
src/hyperdime/training/selection.py      # recipes, grids, validation scoring (new)
src/hyperdime/reports/e0.py              # E0 report and G1 criteria (new)
src/hyperdime/cli/{train_selector,train_global,evaluate,report}.py
tests/unit/{evaluation,baselines,training,reports}/, tests/integration/test_f3_pipeline.py
```

## Risks

- Grid size on MS MARCO: about 90 trained cells for the candidates; each cell trains on GPU in
  about a minute and validation shares one corpus pass per family.
- `ir-measures` adds a compiled dependency to CI; it is already locked in the `science` extra.
