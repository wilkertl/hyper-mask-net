# Implementation Plan: F4 Selector Shift and Gate G2

**Branch**: `agent/F4-selector-shift` | **Date**: 2026-10-12 | **Spec**: [spec.md](spec.md)

## Summary

On F2's artifacts and F3's selections, train F3's `wide-grid` cell with three seeds per
environment, build the 3 × 3 transfer matrix at k = 307 in one corpus pass per query set, compare
selector behavior on a balanced unlabeled battery against the seed noise floor, test each
selector's fit as a ranking, and generate `reports/E1/` with both G2 criteria and a proposed G2
ADR. Every test rule is fixed in [research.md](research.md) before any run; the authors'-code check
of ADR-0006 is done (research D9). Runs on DataPolicy-4090 where F2 and F3 artifacts live.

## Technical Context

**Language/Version**: Python ≥ 3.11 (CI 3.12) · **Dependencies**: F1 contracts, F2 artifacts, F3
selections and `evaluate_methods`, torch, numpy; nothing new · **Storage**:
`artifacts/{selectors/seeded,evaluation/E1,decisions/G2}/`, `reports/E1/` (versioned) ·
**Testing**: pytest on CPU, synthetic environments with planted effects · **Platform**:
DataPolicy-4090 for runs · **Performance**: transfer matrix under 30 minutes (SC-006) ·
**Constraints**: no test label in training, fixed subsets, or fit tests; no training during the
transfer matrix; exact scoring; report generated only from result files.

## Constitution Check

| Principle | Status | Evidence |
|---|---|---|
| I | ✅ | masks through `masked_queries`; documents untouched; no re-normalization |
| II | ✅ | seeded training and fixed subsets read training labels; fit uses training-split validation labels; the battery reads embeddings only; test labels only score |
| III | ✅ | new records `TransferCell`, `SelectorProperties`, `GateDecision` and `Protocol "transfer"` in `hyperdime.contracts` (planned by the roadmap, owner F4); manifests per seed and per run |
| IV | ✅ | G2 is the gate F4 decides; no hypernetwork or descriptor code; the linear selector unchanged |
| V | ✅ | exact scoring; full and prefix rows in every table; paired tests, Holm, bootstrap CIs; generated report and decision |
| VI | ✅ | hand-computed tests for the grouping permutation test, seed averaging, fixed subset, fit gain; synthetic planted-effect test of both criteria |
| VII | ✅ | importance methods separate from `top_k_mask`; training (`train-selector`) separate from evaluation (`transfer-matrix`) |

**Result**: PASS (re-checked after Phase 1 design: unchanged).

## Project Structure

```text
src/hyperdime/contracts/schemas.py        # + Protocol "transfer", SelectorProperties, TransferCell, GateDecision
src/hyperdime/contracts/__init__.py       # exports
src/hyperdime/baselines/methods.py        # + fixed_subset, rank_importance
src/hyperdime/evaluation/selector_shift.py  # + kl_divergence, pair_measures, group_permutation_test
src/hyperdime/evaluation/transfer.py      # new: battery, seed averaging, degradation tests, fit gain, cells
src/hyperdime/cli/train_selector.py       # + --recipe seeded
src/hyperdime/cli/transfer_matrix.py      # new
src/hyperdime/cli/decide.py               # new
src/hyperdime/cli/report.py, cli/__main__.py  # + E1, register new commands
src/hyperdime/reports/e1.py               # new: E1 report and G2 criteria
src/hyperdime/reports/g2.py               # new: GateDecision and proposed ADR
tests/unit/{contracts,evaluation,baselines,reports}/, tests/integration/test_f4_pipeline.py
docs/architecture.md, README.md           # components and commands
```

## Risks

- Three seeds give 1,680 label assignments, so criterion 2's smallest p is 0.0036: enough for
  α = 0.05, but a single seed that trains badly can move T; the per-environment table shows each
  seed's overlaps.
- MS MARCO dominates runtime (about 7,000 dev queries × 16 variants over 8.8M passages, plus the
  TREC sets at depth 1,000); E0's larger grid took 6.5 minutes.
- Seed 0 retrained may differ in the last bits from F3's `wide-grid` checkpoint on GPU; recorded,
  not required (research D1).
