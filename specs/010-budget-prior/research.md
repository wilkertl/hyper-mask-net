# Research: F9 Budget-Conditioned Prior Weight (experiment S4)

Every decision is fixed before any S4 run. Code references are to `agent/F8-prior-weight`
(`8cee80e`).

## D1. The budget selector

- **Decision**: `BudgetGeometricSelector(variance, budgets)` subclasses `GeometricSelector`; it
  holds `prior_weights`, a parameter of shape `[9]` starting at ones, and the budgets as a buffer
  so checkpoints describe themselves. `forward(queries, k)` returns
  `log_softmax(a_k · log(q² · Var + ε) + W q + b)`; it raises for a `k` outside its budgets.
  Importance for evaluation is a `Mapping[int, Tensor]`, which `evaluate_methods` already accepts
  (`Importance` in `evaluation/runs.py`), so the corpus pass and its cost are unchanged.
  `load_selector` recognizes the checkpoint by the `budgets` buffer.
- **Rationale**: eight added parameters is the smallest change that lets the ordering depend on k
  (constitution IV; ADR-0011 names it). Sharing `W, b` keeps the relevance correction learned from
  every budget's batches.
- **Alternatives**: a weight per budget on `W q + b` instead (equivalent up to the joint-scale
  invariance in D5, but starting it at 1 does not give `variance` at step 0); a budget embedding
  added to the scores (more parameters, not interpretable); a separate selector per budget (nine
  times the data split nine ways).

## D2. Nonnegative weights

- **Decision**: after each optimizer step, prior weights are clamped to `≥ 0` (projected gradient),
  for both the budget selector and the aligned S3 control.
- **Rationale**: keeps step 0 exactly at `variance` (a softplus parametrization would need an
  inverse-softplus start and changes the gradient scale) and keeps the control comparable. A
  negative weight would rank dimensions by *low* `q_d² Var(x_d)`, which no evidence supports.
  S3's weights never approached 0 from below (minimum 0.0005), so the projection is not expected
  to bind for the control; the report records how often it bound.

## D3. Aligned checkpoint rule

- **Finding**: `train_retrieval_selector` keeps the state with the highest
  `val_rerank_ndcg10` (each validation query's own positives plus its distinct pool, gains
  `2^grade − 1` from `RetrievalExamples.weights`), and only that state reaches the full-corpus
  validation in `score_on_validation_budgets`. Reported nDCG@10 uses linear gains
  (`evaluation/metrics.py`). A checkpoint better on the full corpus can be discarded, and on
  graded NFCorpus judgments the two gains disagree.
- **Decision**: `RetrievalConfig.checkpoint_rule ∈ {"rerank", "validation"}`, default `"rerank"`
  (S1–S3 unchanged). Under `"validation"` the trainer returns every evaluation-step state (step 0
  and every 1,000 steps: 21 states, 4 MB each on the CPU) as candidates, still logging the rerank
  score for comparison. `score_on_validation_budgets` then scores every candidate of every cell
  in one corpus pass on the D4 subset (exact, hard top-k, linear gains, mean over the nine
  budgets), keeps each cell's best (ties to the earlier step), and scores the kept states on the
  full validation set for the cell choice, as before.
- **Alternatives**: full validation for every candidate (21 × 8 × 9 variants of up to the full MS
  MARCO validation set per grid is too slow); rerank with linear gains (aligns gains but not the
  candidate set, which is the larger gap on MS MARCO).

## D4. Validation subset for candidates

- **Decision**: at most 1,000 validation queries, the smallest under `sha256(f"{seed}:{qid}")`
  with seed 0, identical for every cell, track, and seed; all queries when there are fewer
  (SciFact and NFCorpus). The subset's ids are written to `selection.json`.
- **Rationale**: deterministic, independent of row order, drawn from validation only. On MS MARCO
  the candidate pass is 8 × 21 × 9 × 1,000 ≈ 1.5M masked queries per grid, comparable to E1's
  145-variant pass (11.5 minutes for ≈ 1M).

## D5. The guard and the prior's share

- **Finding**: `TrainedRetrieval.soft_mass` reads `history[-1]`, the final state, not the kept
  one. And a small learned `a` does not by itself mean the prior was abandoned: scaling `a, W, b`
  by `c > 0` multiplies every score by `c`, which neither the hard top-k nor the standardized soft
  mask (`soft_top_k` standardizes per query) can see.
- **Decision (guard)**: under the aligned rule the guard is the soft mass of the kept state's
  scores at `β_end` (the temperature the schedule ends at), at most 1,024 validation queries,
  mean over budgets, threshold 0.9 as before. Step 0 is held to the same rule.
- **Decision (share)**: per budget, `share_k = mean_q σ_q(a_k π(q)) / (σ_q(a_k π(q)) +
  σ_q(W q + b))`, with `σ_q` the standard deviation over dimensions; invariant to the joint scale
  and 1 at step 0. Reported with top-k overlap against `variance` (`W = b = 0`) and against the
  correction alone (`a = 0`).
- **Decision (ablation)**: validation-only exact nDCG@10 of the kept states with every `a_k = 0`,
  scored in `compare-selectors`'s validation pass; context, not a criterion.
- **Alternatives**: the guard at the kept step's own β (fails every step-0 state at β = 1, though
  no training happened); the weight itself as the diagnostic (scale-confounded, above).

## D6. Grid and tracks

- **Decision**: two tracks with the same 8 cells (`multi-budget` only; lr {1e-4, 1e-3} × score
  temperature {0.02, 0.05} × β_end {0.1, 0.02}): `budget` (S4) and `weighted-aligned` (S3's
  selector, aligned rule, D2 projection). Inits `budget` and `weighted-aligned` join `PRIOR_INITS`.
  Recipes `budget-grid | budget-seeded | weighted-aligned-grid | weighted-aligned-seeded`.
- **Rationale**: a fixed-307 variant would train one of nine weights, so it is not S4. The control
  repeats S3 on the same reduced grid so the only difference is the eight weights; the original S3
  stays in the report as context.

## D7. Statistics

- **Decision**: add `noninferiority_test(a, b, margin)` to `evaluation/statistics.py`: one-sided
  sign-flip test of `mean(a − b) ≤ −margin` on `d + margin`, returning `(hits + 1) / (n + 1)`
  with the same resampling count and seed as `paired_randomization_test`. Holm over the three
  environments per reference. Superiority and secondary comparisons reuse the existing two-sided
  test and percentile bootstrap.
- **Margin**: `δ = 0.005` nDCG@10 on the mean over budgets, about 1.4% of NFCorpus's level and the
  size of S3's NFCorpus gap to S2 (−0.0051), which S3's "no significant degradation" gate let pass.
- **Alternatives**: a relative margin (environment levels differ from 0.37 to 0.83, but an absolute
  margin matches how differences are reported); equivalence testing (S4 is meant to be allowed to
  be better).

## D8. Pre-registration and fresh data

- **Decision**: spec, research, and the margin are committed before any S4 run; the report states
  that S1–S3's test results informed S4. Confirmation reruns this spec unchanged on #60's
  environments, outside F9.

## D9. What S4 does not test (ordered follow-ups)

1. Straight-through hard forward masks (exact k in the forward pass, sigmoid surrogate backward):
   targets the train/inference mismatch; biased gradient, to be validated empirically.
2. Masked-query negatives on MS MARCO: 64 negatives, 32 from the F2 pool and 32 mined under the
   current hard masks from training queries only, known positives excluded, mining reported as
   offline cost, against an equal-budget control. With `W = b = 0`, a positive `a_k` cannot change
   the mask, so S4 alone is expected to leave MS MARCO at `variance`.
3. Linear gains in the training loss, as an ablation on the aligned control.
4. Adaptive budgets (risk-based retained dimensionality), after fixed-budget selection.

## Literature, with limits

- Learning to Select (ACL 2026): query-only importance with no retrieval feedback at inference;
  S4 keeps that deployment model.
- ECLIPSE (arXiv 2412.14967): relevant and irrelevant evidence; refreshing negatives under the
  mask (D9.2) is an extension, not its result.
- When Reducing Representations Improves Performance (ECIR 2026): interactions among selected
  dimensions; motivates later subset-aware training, not an attainable gain here.
- Statistical Foundations of DIME (EACL 2026): adaptive dimensionality (D9.4); its assumptions do
  not guarantee nDCG on Qwen embeddings.
