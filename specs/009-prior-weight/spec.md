# Feature Specification: F8 Learned Prior Weight (experiment S3)

**Feature Branch**: `agent/F8-prior-weight`

**Created**: 2026-10-16

**Status**: Draft

**Input**: Issue #59. S2 (ADR-0010) showed that the geometric prior helps where approximating the
full query matters (MS MARCO, every environment at small budgets) and hurts where the ranking must
be reshaped (SciFact at larger budgets). S3 lets training weigh the prior:
`importance_d = a · log(q_d² · Var(x_d) + ε) + (W q + b)_d`, with the scalar `a` learned from 1.
Rules below are fixed before any S3 run.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - A selector that learns how much to trust the prior (Priority: P1)

The researcher trains, per environment, S2's selector with one extra learned scalar `a` on the
prior, starting at 1 (so step 0 is the `variance` method, as in S2), with S2's objective, grid
rules, seeds, and evaluation, and sees the learned `a` per environment and seed.

**Independent Test**: an untrained weighted selector equals the `variance` method; `a` is its only
parameter beyond the L2S linear map; with `a` forced to 0 it is the linear selector alone; on a
synthetic set with a misleading prior, training lowers `a`.

### User Story 2 - Experiment S3 (Priority: P1)

The researcher compares the weighted selector (3 seeds) with the S1 selector, the S2 selector,
`variance`, and L2S at every 10% budget, and reads `reports/S3/` and a drafted ADR-0011.

## Requirements *(mandatory)*

- **FR-001**: The only added learned parameter is the scalar `a` (constitution IV); the prior and
  the linear map are as in S2.
- **FR-002**: Training reuses S2 unchanged except for `a`: objective, relaxed mask, 16-cell grid
  (2 variants × 2 learning rates × 2 score temperatures × 2 final mask temperatures), 20,000
  steps, checkpoint rule, validation selection, soft-mass guard.
- **FR-003**: S1 and S2 code paths, checkpoints, results, and reports stay unchanged; both
  reports regenerate byte-identically.
- **FR-004**: The S3 report is generated from result files only, with full and prefix rows,
  protocol, the learned `a` per environment and seed at the kept checkpoint, byte-identical
  regeneration, and a drafted ADR-0011.

## Success Criteria *(mandatory)*

Mean nDCG@10 over the 10%–90% budgets on the test sets, paired per query (seeds averaged),
Holm over the three environments for each reference:

- **SC-001**: in no environment is the weighted selector below the S1 selector, the S2 selector,
  or `variance` with Holm-corrected p ≤ 0.05 (it is never worse than the best of them).
- **SC-002**: it is above the S1 selector with Holm-corrected p ≤ 0.05 in at least 2 of 3
  environments.
- **SC-003**: each grid finishes within 3 hours and `compare-selectors` within 30 minutes on
  DataPolicy-4090 when the GPU is not shared.

## Assumptions

- Inputs: F2 artifacts, E1's L2S seeds, S1's and S2's grids, seeds, and runs (`master` at
  `d6b31d1`).
- Runs use S2's operating settings: environments in parallel, CPU threads capped per process,
  scoring chunks of 250,000 documents.
