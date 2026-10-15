# Research: F5 Selector Objective and Experiment S1

Every rule below is fixed before any S1 run, so no test result can shape it (spec FR-006).
Numbers cited as motivation come from generated tables in `reports/E0/` and `reports/E1/`.

## D1. Training examples

- **Decision**: A training example is one training query with its positive documents (training
  judgments with grade > 0, weighted by the ADR-0002 gain `2**grade − 1`) and its F2 hard-negative
  pool (64 documents sampled with replacement from its top-1,000 non-positives). Repeated
  negatives are kept, as in the L2S targets, so the two objectives see the same negatives with the
  same weights. Documents are gathered once per environment from the (memory-mapped) embedding
  matrix, as `oracle_pairs` does, but kept per document instead of being averaged into `p` and
  `n`. Queries whose positives are all missing from the corpus are skipped and counted.
  Validation examples are built the same way from validation judgments and pools.
- **Rationale**: same labels and negatives as L2S, so S1 isolates the objective (spec FR-003,
  FR-015). MS MARCO's training set is F2's 50,000-query cap, so all documents fit in GPU memory
  (at most 50,000 × (positives + 64) × 1,024 float16 ≈ 7 GB; gathered as unique rows, far less).
- **Alternatives considered**: mining new negatives with the masked query during training (an
  online, selector-dependent pool): stronger, but it changes the data as well as the objective,
  so a gain could not be attributed; left for later.

## D2. Loss

- **Decision**: For query `q`, selector scores `s = W q + b`, budget `k`, and relaxed mask
  `m = soft_top_k(s, k)` (D3), the masked query is `q ⊙ m` (not re-normalized), and each candidate
  document `d` gets the score `⟨q ⊙ m, d⟩ / τ_s`. The candidates are the query's positives, its 64
  hard negatives, and the positives of the other queries in the batch (in-batch negatives). The
  loss is the multi-positive softmax cross-entropy
  `−Σ_p w_p log softmax(scores)_p` with `w_p` the gain weights normalized to sum to 1, averaged
  over the batch.
- **Rationale**: the loss is the retrieval the selector is used for: it pushes positives above
  hard negatives under the mask at budget `k`, which KL to the oracle distribution does not
  (spec, motivation). In-batch positives add easy negatives that the full-corpus evaluation also
  contains; the hard negatives alone come only from the top of the unmasked ranking. Multi-positive
  softmax with gain weights keeps graded NFCorpus judgments meaningful, as in the L2S targets.
- **Amendment (2026-10-15, after the first SciFact grid, before any test evaluation)**: the
  in-batch negatives are the batch's *distinct* positive documents, each counted once, and each
  batch's positive lists are trimmed to its longest one. The first run padded every batch to the
  longest list in the environment (1,363 positives for one NFCorpus query) and compared every
  positive slot with every other, about 1.9 billion comparisons per step; NFCorpus's grid had not
  finished one environment in an hour. Counting a shared document once is also what "the
  positives of the other queries" means for a softmax over documents. The SciFact grid and seeds
  were rerun with the amended loss, so all S1 runs come from one commit.
- **Alternatives considered**: pairwise margin (hinge) loss: needs a margin per environment; a
  ListNet loss over the oracle ordering: still imitates the oracle; adding the KL loss as an
  auxiliary term: kept out so the objective is a clean swap (FR-015); can be an ablation if S1 is
  mixed.

## D3. Relaxed top-k mask

- **Decision**: Scores are standardized per query, `z = (s − mean(s)) / std(s)`. The threshold `t`
  is the midpoint between the k-th and (k+1)-th largest `z`, with no gradient through it. The mask
  is `m = sigmoid((z − t) / β)`, so `m > 0.5` holds exactly on the hard top-k and `m → hard top-k`
  as `β → 0`. `β` anneals geometrically from `β_0 = 1` to `β_end` over training. At k = 1024 the
  mask is all ones (no threshold). Evaluation always uses the hard top-k mask through
  `top_k_mask`, with its tie handling (`torch.topk`), unchanged from E0 and E1.
- **Guard**: on validation queries at the end of training, the soft-mask mass on the hard top-k,
  `Σ_{i∈topk} m_i / Σ_i m_i`, averaged over queries and budgets, must be ≥ 0.9; a cell below it is
  recorded but not selectable (spec edge case).
- **Rationale**: standardizing makes `β` independent of the selector's score scale, which the
  linear map can otherwise inflate to make any sigmoid hard and kill the gradient. A detached
  threshold keeps gradients on every dimension's own score, with the largest gradient near the
  boundary, where swaps change retrieval.
- **Alternatives considered**: straight-through estimator (hard mask forward, sigmoid backward):
  a candidate if the sigmoid relaxation fails the guard; sparsemax / entmax top-k or optimal
  transport top-k (Xie et al. 2020): exact but costlier and with more hyperparameters; Gumbel
  top-k sampling: adds variance with no evident benefit for a deterministic mask.

## D4. Budgets

- **Decision**: Two variants (FR-004). **fixed-307** trains at k = 307 (30%, the E0/E1 headline).
  **multi-budget** draws, for each batch, one k uniformly from the nine 10%–90% budgets
  (`ratio_budgets()` without 1024), from a generator seeded by `run.seed`, so a rerun sees the same
  sequence.
- **Rationale**: one batch, one k keeps the threshold computation vectorized; uniform sampling
  matches the evaluation summary, which weighs the nine budgets equally (D7).
- **Alternatives considered**: summing the loss over all nine budgets per batch: nine times the
  cost for the same expected gradient; conditioning the selector on k: adds an input, i.e.
  capacity (constitution IV).

## D5. Initialization

- **Decision**: **scratch** uses PyTorch's default `nn.Linear` initialization under `run.seed`
  (as L2S). **l2s** starts from the E1 `wide-grid` seeded checkpoint of the same seed (seed 0 for
  the grid, D6), read-only, refusing a payload whose SHA-256 differs from its record.
- **Rationale**: FR-005: if only the l2s start wins, the gain is the start plus fine-tuning, not
  the objective alone; the report shows both.

## D6. Optimization, grid, and selection

- **Decision**: AdamW, weight decay 1e-4, batch 32, cosine decay over a fixed number of steps,
  `S = 20,000` steps per run for every environment (about 14 epochs of SciFact-sized and 13 of
  MS MARCO's 50,000 queries; the same number of updates everywhere, spec edge case). Every 1,000
  steps the checkpoint is scored on validation by the **pool-rerank nDCG@10** of the hard mask
  (validation query against its positives and pool, averaged over the nine budgets); the best
  checkpoint is kept (as L2S keeps the lowest validation KL). The grid, shared by all
  environments, is
  `variant {fixed-307, multi-budget} × init {scratch, l2s} × lr {1e-4, 1e-3} × τ_s {0.02, 0.05} ×
  β_end {0.1, 0.02}` = 32 cells, trained with seed 0. Each cell is then scored on validation by
  **full-corpus nDCG@10** with the hard mask at the nine budgets, all cells in one corpus pass
  (`evaluate_methods`, as `score_on_validation` does), and the selected cell maximizes its mean
  over the nine budgets among cells that pass the D3 guard; ties go to the earlier cell in grid
  order. The best cell of each (variant, init) family is also recorded.
- **Rationale**: pool reranking is cheap enough to score 20 checkpoints per cell, and the final
  choice uses the same measurement as the evaluation (full corpus, hard mask). 32 cells × 20,000
  steps of batch 32 over ≤ 160 candidate documents per query is minutes per environment on the
  4090; the full-corpus validation pass for MS MARCO (about 5,000 validation queries × 32 cells ×
  9 budgets) is the largest cost, comparable to E1's 145 variants.
- **Alternatives considered**: selecting on k = 307 only, as F3 did: the spec's summary is the
  mean over budgets (FR-006); a larger grid: deferred until S1 shows which axis matters.

## D7. S1 evaluation

- **Decision**: `hyperdime compare-selectors` scores, per environment, its test set (and MS MARCO's
  TREC DL 2019, DL 2020, DL-HARD), in one corpus pass per query set, at every 10% budget: the
  retrieval-trained selector seeds 0–2 (`retrieval:<env>:s<n>`), E1's L2S seeds 0–2
  (`l2s:<env>:s<n>`), `global-selector`, `prefix`, `random`, `global-importance`, and `full` at
  1024 only. It also scores, with seed 0 only, the best cell of each (variant, init) family
  (`family:<variant>-<init>:<env>`) as context for FR-005. All in-domain; no transfer.
- **Summary and tests**: per query, seeds are averaged; the summary statistic is the per-query
  mean of nDCG@10 over the nine 10%–90% budgets (and, secondarily, nDCG@10 at k = 307). The paired
  difference against L2S per environment is tested with the paired randomization test and 95%
  bootstrap interval used in E1 (`evaluation.statistics`), and Holm-corrected over the three
  environments, separately for each summary. TREC DL sets are reported with AP and their own
  tests, not Holm-counted (spec Assumptions).
- **Criteria**: S1 passes when SC-001 (better than L2S, mean over budgets, Holm p ≤ 0.05, in ≥ 2 of
  3 environments) and SC-002 (worse in none, same test) both hold.

## D8. Diagnostics (US4)

- **Decision**: per environment and budget, the mean top-k Jaccard overlap on test query
  embeddings between the retrieval-trained selector and L2S (seed-matched pairs, mean over seeds),
  with each method's seed-to-seed overlap as the noise floor; the ranking fit of E1 (top-k overlap
  with the oracle mask on validation queries, gain over global importance, F4 research D2) for
  both; the training curves (training loss, pool-rerank validation nDCG@10, soft-mask mass) from
  each run's history.

## D9. Artifacts and records

- **Decision**: `train-selector --recipe retrieval-grid data=<env>` writes
  `selectors/retrieval-grid/<env>/<run_id>/` (every cell's checkpoint, `history.json`,
  `selection.json` with all validation scores, guard values, the selected cell, and family bests).
  `train-selector --recipe retrieval-seeded data=<env> run.seed=<n>` retrains the selected cell
  into `selectors/retrieval-seeded/<env>/seed-<n>/<run_id>/`; seed 0 is retrained, not copied
  (as F4 research D1). Checkpoints are `SelectorCheckpoint(kind="per_domain")`; the objective is
  recorded in `selection.json` (`"objective": "retrieval"`), so the contract is unchanged.
  `compare-selectors` writes `evaluation/S1/all/<run_id>/` (`results-<env>-<set>.jsonl`,
  `overlap.json`, `fit.json`, `inputs.json`). `decide S1` writes
  `decisions/S1/<run_id>/gate_decision.json` and drafts ADR-0009; `GateDecision.gate` gains the
  value `"S1"` (an additive contract change, justified in the pull request).
