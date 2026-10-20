# Feature Specification: F9 Budget-Conditioned Prior Weight (experiment S4)

**Feature Branch**: `agent/F9-budget-prior`

**Created**: 2026-10-16

**Status**: Draft

**Input**: ADR-0011 (S3 accepted) and the researcher's S4 proposal. S3 scores
`a · log(q_d² · Var(x_d) + ε) + (W q + b)_d` with one scalar `a` for every budget, so one dimension
ordering serves every k; on NFCorpus it is below the S2 selector at the smallest budgets, and
ADR-0011 names a per-budget weight as the next candidate. S4 gives each of the nine evaluated
budgets its own nonnegative weight `a_k` and shares `W, b`:
`s_d(q, k) = a_k · log(q_d² · Var(x_d) + ε) + (W q + b)_d`, mask `TopKMask(s, k)`.

Reading the S3 code also showed that the kept checkpoint is chosen by reranking each validation
query's positives and pool with gains `2^grade − 1`, while results use exact full-corpus nDCG@10
with linear gains, and that the soft-mass guard reads the final training state rather than the
kept one. S4 and an S3 control trained under the same corrected rule make budget conditioning the
only difference. Rules below are fixed before any S4 run.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - A selector with one prior weight per budget (Priority: P1)

The researcher trains, per environment, the S3 selector with the scalar replaced by nine
nonnegative weights (one per 10%–90% budget, each starting at 1, `W = b = 0`), so step 0 is the
`variance` method at every budget, and reads the learned `a_k` per environment, seed, and budget.

**Independent Test**: an untrained budget selector equals `variance` at every budget; its
parameters are the linear map plus nine weights; with all `a_k` equal it ranks as the S3 selector
with `a` equal to them; changing one `a_k` changes only budget k's mask; a weight driven below
zero is projected to 0; it refuses a budget it has no weight for.

### User Story 2 - Checkpoints chosen by the reported metric (Priority: P1)

The researcher trains with checkpoint candidates kept every 1,000 steps (step 0 included) and the
kept checkpoint chosen by exact full-corpus validation nDCG@10 (linear gains, hard top-k mask,
mean over the nine budgets), and the soft-mass guard computed on that kept checkpoint. S1–S3
keep their rule.

**Independent Test**: on a synthetic set where the rerank score and full-corpus nDCG@10 disagree,
the aligned rule keeps the full-corpus best; step 0 can be kept; the guard reads the kept state;
the default rule leaves S1–S3 histories and checkpoints unchanged.

### User Story 3 - Experiment S4 (Priority: P1)

The researcher compares the budget selector (3 seeds) with the aligned S3 control (3 seeds), and
as context S3, S2, `variance`, S1, and L2S, at every 10% budget; reads the learned weights, the
prior's share of the score, a validation-only prior-removal ablation, and the kept steps; and
reads `reports/S4/` and a drafted ADR-0012.

## Requirements *(mandatory)*

- **FR-001**: The only added learned parameters are eight weights (nine `a_k` instead of one `a`,
  constitution IV); the prior and the linear map are as in S3. The budget is an input to the
  importance; the budget itself is set by the evaluation, never by the selector (constitution VII).
- **FR-002**: Both tracks (budget and aligned control) share data, loss (multi-positive masked
  retrieval loss with gains `2^grade − 1`), relaxed mask, budgets drawn uniformly from the nine,
  AdamW with batch 32, 20,000 steps, cosine decay, weight decay 1e-4, the projection of prior
  weights to `≥ 0`, the 8-cell grid (learning rate {1e-4, 1e-3} × score temperature {0.02, 0.05}
  × final mask temperature {0.1, 0.02}), validation-only cell choice, and the aligned checkpoint
  rule. Seeds 0, 1, 2 retrain the selected cell.
- **FR-003**: Checkpoint candidates of a grid cell are scored on a fixed validation subset (at
  most 1,000 queries, chosen by a seeded hash of query ids; all queries when fewer); the cell
  choice uses all validation queries. Test and TREC DL judgments are never read by training,
  checkpointing, or cell choice.
- **FR-004**: Diagnostics on the kept checkpoint, validation queries only: per budget, the prior's
  share of the score `σ(a_k π) / (σ(a_k π) + σ(W q + b))` (per-query standard deviations, mean
  over queries; invariant to scaling `a`, `W`, `b` together); top-k overlap with `variance` and with
  the correction alone; nDCG@10 with every `a_k` set to 0 (prior removal).
- **FR-005**: S1, S2, and S3 code paths, checkpoints, results, and reports stay unchanged; their
  reports regenerate byte-identically.
- **FR-006**: The S4 report is generated from result files only, with full and prefix rows, the
  protocol, a note that these test sets informed the proposal, the margin `δ`, per-budget
  differences, secondary metrics, the diagnostics, byte-identical regeneration, and a drafted
  ADR-0012.

## Success Criteria *(mandatory)*

Mean nDCG@10 over the 10%–90% budgets on the test sets, paired per query (seeds averaged), 95%
paired bootstrap intervals, Holm over the three environments for each reference. Non-inferiority
margin `δ = 0.005` nDCG@10, fixed now.

- **SC-001 (non-inferiority)**: in every environment the budget selector is non-inferior to the
  aligned S3 control, the S2 selector, and `variance`: one-sided sign-flip test of
  `mean(S4 − ref) ≤ −δ` rejected at Holm-corrected p ≤ 0.025.
- **SC-002 (superiority)**: the budget selector is above the aligned S3 control with
  Holm-corrected p ≤ 0.05 in at least 1 of 3 environments.
- **SC-003**: each grid finishes within 3 hours and `compare-selectors` within 45 minutes on
  DataPolicy-4090 when the GPU is not shared.

Secondary, reported and Holm-corrected but not decisive: nDCG@10 at k = 307, MRR@10, Recall@100,
the per-budget differences, and the aligned control against S3.

## Assumptions

- Stacked on `agent/F8-prior-weight` (S3, not yet merged); inputs are F2 artifacts, E1's L2S
  seeds, and the S1–S3 grids, seeds, and runs.
- The existing test sets informed this proposal, so S4's reading on them is not confirmatory;
  confirmation is a rerun under this spec on the environments of #60, outside this feature.
- Out of scope, each its own later experiment if S4's reading calls for it: straight-through hard
  forward masks, negatives mined under the current hard masks (MS MARCO), linear gains in the
  loss, and adaptive budgets.
- Runs use S3's operating settings: environments in parallel, CPU threads capped per process,
  scoring chunks of 250,000 documents.
