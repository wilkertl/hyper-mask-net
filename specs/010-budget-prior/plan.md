# Implementation Plan: F9 Budget-Conditioned Prior Weight (experiment S4)

**Branch**: `agent/F9-budget-prior` | **Date**: 2026-10-16 | **Spec**: [spec.md](spec.md)

## Summary

Replace S3's single prior weight with nine nonnegative weights, one per 10%–90% budget, sharing
the L2S linear map: `s_d(q, k) = a_k · log(q_d² · Var(x_d) + ε) + (W q + b)_d`, starting at
`variance` (research D1, D2). Train it and an S3 control on the same 8-cell multi-budget grid,
both under a corrected checkpoint rule: candidates every 1,000 steps (step 0 included) scored by
exact full-corpus validation nDCG@10 with linear gains on a fixed subset, and the soft-mass guard
on the kept state (D3–D5). Experiment S4 compares the two (3 seeds each) with a predeclared
non-inferiority margin δ = 0.005 and superiority test, reports the weights, the scale-invariant
prior share, and a validation-only prior-removal ablation, and ends with `reports/S4/` and a
drafted ADR-0012 (D7). Stacked on F8; runs on DataPolicy-4090.

## Technical Context

**Language/Version**: Python ≥ 3.11 (CI 3.12) · **Dependencies**: F8's `GeometricSelector`,
retrieval trainer and grid, `evaluate_methods` (per-budget `Importance` already supported),
`evaluation.statistics`, `reports/s1–s3`; torch, numpy; nothing new · **Storage**:
`artifacts/{selectors/budget-grid,budget-seeded,weighted-aligned-grid,weighted-aligned-seeded,
evaluation/S4,decisions/S4}/`, `reports/S4/` (versioned) · **Testing**: pytest on CPU;
hand-computed tests for per-budget ranking, projection, prior share invariance, and the
non-inferiority p; synthetic disagreement between rerank and full-corpus checkpoints ·
**Platform**: DataPolicy-4090 (shared GPU) · **Performance**: each grid ≤ 3 h, `compare-selectors`
≤ 45 min (SC-003); the candidate pass adds ≈ 1.5M masked queries per MS MARCO grid (D4) ·
**Constraints**: training and checkpointing read training and validation judgments only; exact
scoring, hard top-k at evaluation; S1–S3 outputs byte-identical (FR-005).

No open clarifications: the margin, subset size, guard temperature, projection, and grid are fixed
in research D2–D7.

## Constitution Check

| Principle | Status | Evidence |
|---|---|---|
| I | ✅ | masked query `q ⊙ m`, not re-normalized; exactly k coordinates at evaluation; documents and index only read |
| II | ✅ | candidates and cells chosen on validation (D3, D4); diagnostics and ablation on validation (D5); a test shows the new grids' checkpoints do not change when test judgments are removed |
| III | ✅ | checkpoints are `SelectorCheckpoint` with manifests, self-describing through the `budgets` buffer; one additive change, `GateDecision.gate` adds `"S4"`; new public names listed in [contracts/cli.md](contracts/cli.md) |
| IV | ✅ | eight added scalars, motivated by S3's evidence (ADR-0011: below S2 at NFCorpus's small budgets, per-budget weight named as next); hard forward masks, mask-aware negatives, and adaptive budgets deferred (D9); no stage beyond G2 |
| V | ✅ | exact scoring; full and prefix rows; paired tests, bootstrap CIs, Holm over environments, predeclared δ; generated report and decision; test-set reuse stated (D8) |
| VI | ✅ | hand-computed tests: step-0 equality with `variance` per budget, equal-weights equality with S3, per-budget isolation, projection, prior-share invariance, non-inferiority p on 4 queries |
| VII | ✅ | the budget is an input to importance and is set by the evaluation, never chosen by the selector; checkpoint selection stays in `retrieval_selection`, diagnostics in `selection/diagnostics.py`, evaluation in `compare-selectors` |

**Result**: PASS (re-checked after Phase 1 design: unchanged). Dependency: F8 (ADR-0011 accepted)
merged before this branch.

## Project Structure

### Documentation (this feature)

```text
specs/010-budget-prior/
├── spec.md · plan.md · research.md · data-model.md · quickstart.md
├── contracts/cli.md
└── tasks.md             # /speckit-tasks
```

### Source code

```text
src/hyperdime/contracts/schemas.py             # GateDecision.gate + "S4"
src/hyperdime/baselines/learning_to_select.py  # + BudgetGeometricSelector (D1)
src/hyperdime/baselines/methods.py             # + budget_importance
src/hyperdime/selection/diagnostics.py         # new: prior_share, overlaps with variance / correction (D5)
src/hyperdime/evaluation/statistics.py         # + noninferiority_test (D7)
src/hyperdime/training/retrieval_trainer.py    # checkpoint_rule, nonnegative_prior, k-aware scoring, candidates
src/hyperdime/training/retrieval_selection.py  # budget / weighted-aligned inits, candidate pass, kept-state guard
src/hyperdime/training/selection.py            # load_selector recognizes budget checkpoints
src/hyperdime/cli/train_selector.py            # + budget-* and weighted-aligned-* recipes
src/hyperdime/cli/compare_selectors.py         # + --experiment S4, per-budget importance, diagnostics, ablation
src/hyperdime/cli/report.py, cli/decide.py     # + S4
src/hyperdime/reports/s4.py                    # new: S4 report, criteria, ADR-0012 draft
tests/unit/test_geometric_selector.py, tests/unit/{training,evaluation,selection,reports}/
tests/integration/test_f5_pipeline.py          # S4 end to end; S1–S3 unchanged
docs/architecture.md, docs/experiment_registry.md, README.md
```

**Structure Decision**: single package `hyperdime`, extending F8's modules in place; only
`selection/diagnostics.py` and `reports/s4.py` are new.

## Risks

- MS MARCO: with `W = b = 0`, no positive `a_k` changes the mask, so S4 likely equals `variance`
  there as S3 did; SC-002 needs only 1 of 3 environments for this reason, and D9.2 is the next step.
- The aligned rule may keep different steps than S3 did, so the control can differ from S3; the
  report shows both, and only the control enters the criteria.
- Subset-based candidate choice can still miss the full-validation best on MS MARCO; the cell
  choice uses all validation queries, and `selection.json` records both scores.
- Re-used test sets: S4's numbers are exploratory until the #60 rerun (D8).
- The GPU is shared: SC-003 is read from the run's own timer.

## Complexity Tracking

None.
