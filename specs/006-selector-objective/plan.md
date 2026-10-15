# Implementation Plan: F5 Selector Objective and Experiment E2

**Branch**: `agent/F5-selector-objective` | **Date**: 2026-10-14 | **Spec**: [spec.md](spec.md)

## Summary

Train the same 1024 × 1024 linear selector as Learning-to-Select (L2S) with a retrieval loss: a
multi-positive softmax over each training query's positives, F2 hard negatives, and in-batch
positives, scored with the query masked by a relaxed top-k of the selector's standardized scores
(research D2, D3). Two variants (fixed k = 307, multi-budget), two starts (scratch, L2S), one
32-cell grid chosen on validation nDCG@10 averaged over the 10%–90% budgets (D6); the chosen cell
is trained with seeds 0–2 and compared with E1's L2S seeds at every 10% budget in experiment E2,
which ends with a generated `reports/E2/` and a drafted ADR-0009 (D7–D9). Proposed with
ADR-0008. Runs on DataPolicy-4090, where the F2 artifacts and E1 selectors live.

## Technical Context

**Language/Version**: Python ≥ 3.11 (CI 3.12) · **Dependencies**: F1 contracts, F2 artifacts,
E1 seeded selectors, `evaluate_methods`, `evaluation.statistics`, torch, numpy; nothing new ·
**Storage**: `artifacts/{selectors/retrieval-grid,selectors/retrieval-seeded,evaluation/E2,
decisions/E2}/`, `reports/E2/` (versioned) · **Testing**: pytest on CPU; hand-computed tests for
the relaxed mask and the loss; a synthetic environment with planted relevant dimensions ·
**Platform**: DataPolicy-4090 (shared GPU: runs wait for free memory) · **Performance**: one
seed under 2 hours per environment, E2 evaluation under 30 minutes (SC-006) · **Constraints**:
training reads training judgments and F2 pools only; validation chooses cells and checkpoints;
evaluation with the hard top-k mask and exact scoring; L2S code and results unchanged (FR-015).

## Constitution Check

| Principle | Status | Evidence |
|---|---|---|
| I | ✅ | masked query `q ⊙ m`, never re-normalized; document embeddings only read; evaluation through `top_k_mask` |
| II | ✅ | examples from training judgments and F2 training pools (D1); validation only selects (D6); a test proves checkpoints do not change when test judgments are removed (FR-014) |
| III | ✅ | checkpoints are `SelectorCheckpoint(kind="per_domain")` with manifests; one additive contract change, `GateDecision.gate` gains `"E2"` (D9) |
| IV | ✅ | same linear selector, no added capacity; no stage beyond G2 (ADR-0008); L2S stays the reproduced baseline |
| V | ✅ | exact scoring; full and prefix rows in every table; paired tests, bootstrap CIs, Holm over environments; generated report and decision |
| VI | ✅ | hand-computed tests: relaxed mask (threshold, limit β → 0, k = 1024 all ones), loss on a 2-document example; synthetic recovery test (SC-005) |
| VII | ✅ | relaxed mask in `selection/`, loss in `training/losses.py`, trainer separate from evaluation (`compare-selectors`) and from the policy (budgets fixed by the evaluation) |

**Result**: PASS (re-checked after Phase 1 design: unchanged). Gate dependency: ADR-0007 (G2)
accepted before merge.

## Project Structure

```text
src/hyperdime/contracts/schemas.py           # GateDecision.gate + "E2"
src/hyperdime/selection/relaxed.py           # new: soft_top_k (D3), soft-mask mass guard
src/hyperdime/training/losses.py             # + masked_retrieval_loss (D2)
src/hyperdime/training/examples.py           # new: RetrievalExamples from judgments and pools (D1)
src/hyperdime/training/retrieval_trainer.py  # new: RetrievalConfig, train_retrieval_selector (D4–D6)
src/hyperdime/training/selection.py          # + retrieval grid cells, family bests, budget-mean validation
src/hyperdime/cli/train_selector.py          # + --recipe retrieval-grid | retrieval-seeded
src/hyperdime/cli/compare_selectors.py       # new: E2 evaluation (D7, D8)
src/hyperdime/cli/decide.py, cli/report.py, cli/__main__.py  # + E2
src/hyperdime/reports/e2.py                  # new: E2 report, criteria, ADR-0009 draft
tests/unit/{selection,training,reports}/, tests/integration/test_f5_pipeline.py
docs/architecture.md, README.md              # components and commands
```

## Risks

- The relaxed mask may be too soft to matter or too hard to train; the β grid and the D3 guard
  bound this, and the straight-through estimator is the fallback if every cell fails the guard.
- In-batch and hard negatives come from the unmasked ranking; documents the mask itself promotes
  are only seen at evaluation. If E2's gain is small on MS MARCO, mask-aware negative mining is
  the next step (D1 alternative).
- The l2s start may dominate the grid, which would make the result about fine-tuning; the family
  bests on test (D7) show this directly.
- The GPU is shared with another user's jobs: runs wait for ≥ 20 GB free and retry, so wall time
  is not a measure of cost; SC-006 is read from the run's own timer.
- MS MARCO's full-corpus validation pass over 32 cells is the largest cost (comparable to E1's
  145 variants, 11.5 minutes).
