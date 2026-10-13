# Research: F4 Selector Shift and Gate G2

Every rule below is fixed before any F4 run (ADR-0006). k = 307 (30%) throughout. Paired tests are
the two-sided sign-flip randomization test (10,000 flips, seed 0) and the 95% percentile paired
bootstrap (10,000 resamples, seed 0) of `hyperdime.evaluation.statistics`; Holm at α = 0.05 within
each family named below.

## D1. Selectors and seeds

- **Decision**: For each environment, read F3's latest `selectors/candidates/<env>` selection, take
  the `wide-grid` family's best cell (`family_best["wide-grid"]`, chosen on validation), and train
  it with seeds 0, 1, and 2: `hyperdime train-selector --recipe seeded data=<env> run.seed=<n>`,
  one run per seed, so each seed has its own manifest (`run.seed` recorded). Each run writes
  `selectors/seeded/<env>/seed-<n>/<run_id>/` with `selector.safetensors`, its
  `SelectorCheckpoint` (`kind = "per_domain"`), and `selection.json` (cell, seed, validation
  nDCG@10, validation KL, the F3 run it came from, inputs).
- **Rationale**: Clarification Q1 (spec). One run per seed reuses `run.seed` and the manifest
  machinery unchanged. Seed 0 is retrained rather than copied, so all three seeds come from the
  same code and commit; its agreement with F3's `wide-grid` checkpoint is recorded, not required
  (GPU kernels may differ in the last bits).
- **Alternatives considered**: one run writing three checkpoints (one manifest for three seeds,
  against FR-001); copying F3's checkpoint as seed 0 (mixes commits).

## D2. Selector fit as a ranking (replaces KL against uniform; ADR-0006)

- **Decision**: On environment e's validation queries, the oracle mask is the top-k of the raw
  oracle score `e_q ⊙ (p − n)`, using training-split labels and the F2 validation negatives (the
  same pairs F3 trains on). For each validation query,
  `gain(q) = mean over seeds of overlap@k(selector_s, oracle) − overlap@k(global-importance, oracle)`,
  where global importance is e's mean training target at F3's official τ (as in E0 and D3). **The selector
  fits its target as a ranking** when `mean gain > 0` and its Holm-adjusted p ≤ 0.05 (family: the
  three environments).
- **Rationale**: The selector is used as a ranking, so its fit is judged as one; global importance
  is the strongest query-independent ranking, so a selector that does not beat it has not learned
  per-query selection. Validation labels are training-split labels; test labels are not read.
- **Role**: reported in E1 and in the G2 draft as context for interpreting each environment's
  shift; not a G2 criterion (research R8 is unchanged).
- **Alternatives considered**: Spearman with the oracle score (dominated by the many unselected
  dimensions); KL against uniform (uninformative at high τ; ADR-0006).

## D3. Transfer matrix

- **Decision**: For each test environment B, one `evaluate_methods` pass over B's test queries
  (and, for MS MARCO, each TREC DL set at depth 1,000, relevance threshold 2) at k = 307, with
  methods: `selector:<A>:s<n>` for every source environment A and seed n (9), `fixed:<A>` for every
  A (3, D6), `full` (k = 1,024), `prefix`, `random` (seed 0), `global-importance` (B's own), and
  `global-selector` (F3's `selectors/global/all`). No model is trained or updated; checkpoints are
  loaded read-only and their SHA-256 checked against their records. Each method's
  `EvaluationResult` is written with `protocol = "transfer"` when A ≠ B, `"in-domain"` otherwise.
- **Rationale**: one corpus pass per query set, as E0 (MS MARCO dev: about 7,000 queries × 16
  variants, fewer than E0's 61).

## D4. One paired test per ordered pair

- **Decision**: For an ordered pair A → B, each query q of B gets `m_A(q)`, the nDCG@10 of A's
  three seeds averaged per query, and likewise `m_B(q)` for B's in-domain seeds. The degradation is
  `Δ(A→B) = mean_q [m_B(q) − m_A(q)]`, with its paired randomization p-value and bootstrap
  interval over B's queries. Holm family: the six ordered pairs. **G2 criterion 1 holds** when
  `Δ > 0` with Holm p ≤ 0.05 for at least 3 of the 6 pairs (research R8, "at least half").
- **Rationale**: Queries are the unit of the paired test; averaging seeds per query removes seed
  noise from the comparison while keeping one test per pair. The TREC DL sets are reported with
  the same statistics but are not among the six pairs.
- **Alternatives considered**: nine per-seed tests per pair (a family of 54, and seeds are not the
  question); an ensemble of seed log-importances (a model no one trained; F0 found ensembles do
  not match single seeds).

## D5. Query battery

- **Decision**: n = the smallest test query count among the three environments; from each
  environment, n test query ids sampled without replacement (`torch.Generator`, seed 0) from the
  sorted ids, all of them for the smallest environment. Only embeddings are read. The battery's
  ids and their SHA-256 are written with the behavior result.
- **Rationale**: Balanced sizes keep MS MARCO (about 7,000 dev queries) from dominating; test
  queries are used because validation sets are small (SciFact has about 80), and the battery reads
  no labels (constitution II restricts labels, not query text).

## D6. Fixed dimension subset (ADR-0006, Pasin et al.)

- **Decision**: For environment A, the selection frequency of each seed's selector on A's training
  queries, averaged over seeds; the subset is the 307 most frequent dimensions, ties broken by
  lower index. As a method it is a constant importance vector `1024 − rank`, so `top_k_mask`
  returns exactly that subset for every query. Reported, per test environment B: the in-domain
  selector against `fixed:A` for each A (including A = B), paired test, Holm over the nine
  comparisons. Not a G2 criterion (clarification Q2).
- **Rationale**: Training queries only, so the subset is a deployable baseline; rank-valued
  importance makes the mask independent of `topk`'s tie order.

## D7. Selector behavior and G2 criterion 2

- **Decision**: The nine selectors (three environments × three seeds) are run on the battery. For
  every unordered pair (36: 9 within an environment, 27 across), compute the battery-mean per-query
  overlap@k, JS divergence and both KL divergences of mean importance, Spearman of mean-importance
  ranks, and Spearman of selection frequencies. Statistic
  `T = mean overlap(within pairs) − mean overlap(cross pairs)`. Exact permutation test: all 1,680
  assignments of the nine selectors to three labelled groups of three; `p = #{T_perm ≥ T_obs} /
  1680` (smallest attainable 6/1,680 ≈ 0.0036, the relabelings of the observed grouping).
  **G2 criterion 2 holds** when `T > 0` and p ≤ 0.05. Per environment pair, the report gives the
  cross overlap against each environment's within overlap, descriptively (20 assignments per pair
  cannot reach 0.05).
- **Rationale**: Training randomness is the null (research R8), so selectors, not queries, are the
  units that are exchangeable under it; a per-query test would treat 900 battery queries as
  independent replications of three training runs.
- **Alternatives considered**: per-query paired test of within versus cross overlap (inflated
  significance); weight distances (excluded by the plan and FR-012).
- **Transfer-cell mask similarity** (FR-008): on B's test queries, the mean overlap@k between A's
  and B's selectors over the nine seed pairs.

## D8. Payoff and notes

- **Payoff over the global selector**: per environment, in-domain (seed-averaged) minus
  `global-selector`, paired test, Holm over the three environments; reported context.
- **Wide-grid departure on MS MARCO**: the report states F3's MS MARCO pick
  (`global-regularized`) and both validation scores.
- **NFCorpus caveat**: ADR-0006's note that its selector does not fit its oracle distribution, next
  to D2's ranking-fit result.

## D9. The authors' MS MARCO setup (ADR-0006 time-boxed check)

- **Read**: `github.com/wzy2001wzy/learning2select` at commit `d7d7254`, `train_and_eval.py`,
  `best_paths_hyperparams.json`, `beir/datasets/data_loader.py`, `readme.md`, on 2026-10-12;
  within the one-hour box.
- **Findings** (concrete differences from E0):
  1. The MS MARCO runs use a dataset folder named `msmarco_subset_50k` (also `_LLM` and
     `_LLM_hard` variants) that is not in the repository. Training uses its `train` split and
     evaluation its BEIR `test` split, over the corpus that folder contains. In BEIR's MS MARCO
     the `test` qrels are TREC DL 2019 (9,260 judgments), not dev, so the paper's MS MARCO numbers
     are on an unpublished subset folder whose test queries and corpus cannot be determined.
  2. nDCG@10 takes its ideal DCG from the top 1,000 retrieved documents, not from all judged
     documents, which can only raise scores when relevant documents are not retrieved.
  3. Budgets round up (30% of 1,024 = 308 dimensions); E0 rounds down (307).
  4. Training holds out a random 10% of training pairs for validation, as E0 does.
- **Context, not a finding**: E0's full-1024 nDCG@10 is 0.3800 on dev and 0.6654 on DL 2019; the
  paper's 0.562 matches neither, consistent with a different corpus.
- **Deviation note used in E1**: "E0 does not reproduce Learning-to-Select on MS MARCO (ADR-0006).
  E1 scores MS MARCO dev against the full 8.8M-passage corpus; the paper evaluates the `test`
  split of an unpublished `msmarco_subset_50k` folder, with an ideal DCG taken from the retrieved
  list."

## D10. Commands and records

| Command | Writes |
|---|---|
| `hyperdime train-selector --recipe seeded data=<env> run.seed=<n>` | `selectors/seeded/<env>/seed-<n>/<run_id>/` |
| `hyperdime transfer-matrix` | `evaluation/E1/all/<run_id>/`: `results-<env>-<set>.jsonl` (`EvaluationResult`), `cells.jsonl` (`TransferCell`), `properties-<env>-s<n>.json` (`SelectorProperties`) with `.npy` payloads, `battery.json`, `behavior.json` (36 pairs, T, p), `fixed_subsets.json`, `fit.json`, `inputs.json` |
| `hyperdime report E1` | `reports/E1/README.md`, `reports/E1/tables/*.csv` |
| `hyperdime decide G2` | `decisions/G2/<run_id>/gate_decision.json` (`GateDecision`) and a proposed ADR at `--adr` |

New records, planned in the roadmap data model with owner F4 (a contract change, justified in the
pull request): `TransferCell`, `SelectorProperties`, `GateDecision`; `Protocol` gains
`"transfer"`. Details in [data-model.md](data-model.md).
