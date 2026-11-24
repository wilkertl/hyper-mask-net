# Research: F12 P3 Selector Benchmark

**Feature**: [spec.md](spec.md) · **Date**: 2026-11-12

Sections are numbered B0–B9; the agreed protocol is
[ADR-0018](../../docs/decisions/ADR-0018-p3-selector-benchmark.md). "Verified" means checked on
2026-11-12 against a file in this repository or the authors' clone, or, where stated, by the
researcher on the host; nothing below is a measured result. Numbers from runs live in the
linked reports and are not repeated here.

## B0. Completed exploratory P3 stages are legacy evidence

All earlier P3 work used F2's query splits, training judgments only, and is exploratory. None of it is
re-scored or re-labelled by this feature, and none of it can be read as performance on unseen
queries: every mask was built from, or every selector was fitted to, the queries it was scored on.

| Stage | Question | Run(s) (host and device in each report) | Code snapshot | Report | ADR |
|---|---|---|---|---|---|
| P3 spike | validate the pipeline on F0 caches | `20261106T234600Z-0a96ea1c` (validation) | `2b7d031` dirty | [reports/P3-spike](../../reports/P3-spike/README.md) | 0014 |
| P3 canonical | per-dimension contributions of relevant pairs at every k | `20261106T231934Z-efae7f88` | `cb9d23b6e48f` (bundle in `artifacts/p3-spike/code-snapshots/`) | [reports/P3](../../reports/P3/README.md), RUNS.md | 0014 |
| Score-matching pilot | can positives be paired with score-matched unjudged documents? | `20261107T001827Z-099fb32c` | `8500602612f5` | [reports/P3-score-matching-pilot](../../reports/P3-score-matching-pilot/README.md) | 0014 |
| Shared mask | cost of one mask, transfer among positives | `20261107T025534Z-d8bad5ff` | `a36dacd1dc70` | [reports/P3-shared-mask](../../reports/P3-shared-mask/README.md) | 0015 |
| Mask ranking | full-corpus ranks under positive-built masks | `20261109T163931Z-9e9f5a0a` | `bd0fd382202d` | [reports/P3-mask-ranking](../../reports/P3-mask-ranking/README.md), RUNS.md, `integrity.json` | 0016 |
| Oracle vs L2S, provenance | F3 checkpoints usable for all three corpora | `20261109T235838Z-e9fd3107` | not recorded in the report | inside `reports/P3-oracle-l2s/RUNS.md` | 0017 |
| Oracle vs L2S, SciFact pilot (CPU, uncapped) | exact oracle against in-sample L2S | `20261109T235856Z-5d598106` | `cf2d71a2828b` | [reports/P3-oracle-l2s](../../reports/P3-oracle-l2s/README.md) | 0017 |
| Oracle vs L2S, NFCorpus bounded pilot | same, 128 queries | `20261112T160013Z-462aa6c4` | `d362793ae9be` | [reports/P3-oracle-l2s-nfcorpus-pilot](../../reports/P3-oracle-l2s-nfcorpus-pilot/RUNS.md) | 0017 |
| Oracle vs L2S, MS MARCO bounded pilot | same, 32 queries (probe `20261112T160140Z-00d34c7c`, not reported) | `20261112T160235Z-298d5679` | `d362793ae9be` | [reports/P3-oracle-l2s-msmarco-pilot](../../reports/P3-oracle-l2s-msmarco-pilot/RUNS.md) | 0017 |

Run copies and hashes of the two bounded pilots are in `artifacts/p3-spike/oracle-followup/runs`;
`~/p3-oracle-followup/launch.claim` on the host stays as their duplicate guard.

What carries forward is method, not numbers: exact float64 full-corpus ranking with one lower-row
tie rule (ADR-0016), the 100% parity check, the stored-run parity check, the id-collision rule for
MS MARCO, and the reading that a positive-built mask is a score ceiling, never a ranking ceiling
(ADR-0017). Two pilot design choices do **not** carry forward: stratifying or weighting queries by
positive count, and F3's L2S checkpoints (tuned on validation at k = 307, lowest validation KL).

## B1. Query split: `p3-query-split-v1`

**Decision**: one materialized split per corpus, read only from
`artifacts/p3-query-split-v1/manifest.json`.

Verified on 2026-11-12 from the manifest:

- protocol `P3 in-domain query partitions v1`, seed `20261112`, target ratios 0.70/0.15/0.15;
- method: pool F2 train and validation query ids; group by NFKC, casefold, and whitespace-normalized
  text; drop development groups matching an official test id or normalized test text; order groups
  by SHA-256 of seed, corpus, and minimum query id; assign whole groups at the cumulative 70% and
  85% boundaries;
- file SHA-256 `e4b1fa65c7f04a480d8cd9845e63e055ac6777c4c0ff95f1227e8f6551ff4896`.

| Corpus | F2 source run | pool (train + validation) | dropped (test overlap) | fit | tune | evaluation |
|---|---|---|---|---|---|---|
| SciFact | `20261012T034117Z-05f1e2b6` | 809 | 2 (`1291`, `871`) | 565 | 121 | 121 |
| NFCorpus | `20261012T034240Z-f6be20fc` | 2,590 | 4 (`PLAIN-24`, `-1070`, `-1126`, `-3470`) | 1,810 | 388 | 388 |
| MS MARCO | `20261012T034348Z-0403ce06` | 50,000 | 0 | 35,000 | 7,500 | 7,500 |

The three id lists per corpus are sorted, pairwise disjoint, sum to the pool minus the dropped ids,
and contain none of the dropped ids. Per-split fingerprints (first 16 hex of SHA-256 of the sorted
ids joined by newlines): SciFact `3ed985ebd1a59d67` / `7997975896395645` / `f58296404286e5e3`;
NFCorpus `d35ed33c39ce3932` / `5752f754df9cb6ed` / `b8ced26a39ab04fb`; MS MARCO `3e34a224d947b411` /
`b51331b44b1d8952` / `03ca696b03ae51ee` (fit / tune / evaluation).

**Verified independently by the researcher on 2026-11-12, on the host with F2 query text**, for all
three corpora: the source query hashes match the manifest's `source_queries_sha256`, the split counts
match, no group of identical normalized text crosses two splits, and no kept query matches the
normalized text of the primary official test split. The ids are kept as they are; nothing is
rematerialized.

**Still to do (T003)**: the manifest has no `RunManifest` (git commit, config hash, dataset hash).
T003 attaches one, wrapping the existing file with this verification record and its hash, before
any stage reads it. TREC DL-19, DL-20, and DL-HARD are outside this benchmark and were not checked.

Rules: the official test split stays reserved and is never read. Queries are never stratified,
filtered, weighted, gated, or used to pick the primary endpoint by positive count. Every evaluation
query with at least one positive judgment counts once; all judgments of a query stay with it.
Evaluation ids and labels never reach negative mining, fitting, target generation, feature fitting,
or tuning; tune labels never reach fitting except through a predeclared selection step.

F2 results are legacy evidence; F2 dataset, embedding, and negative-pool runs are provenance for the
vectors and qrels this feature reads. F2's own `validation` label is not used as a split here.

## B2. Retrieval baseline

The frozen full 1024-dimensional query against the same fixed document vectors, exact scoring over
the whole corpus. Every arm at 100% must reproduce its ranking (parity).

## B3. Selector to beat: authors' Learning2Select, retrained per corpus

Source: `/home/wilker/Code/learning2select` at `55f010f5f3e82a6d171ce0909793e72f9c066ec7`
(`train_and_eval.py`, `best_paths_hyperparams.json`). Verified behaviour:

| Item | Source behaviour | Where |
|---|---|---|
| Model | one `nn.Linear(1024, 1024)` with bias, `log_softmax` output | `DimensionPredictor` |
| Positives | every qrel with `rel > 0`, weights `2^rel − 1` normalized to sum 1; weighted positive mean | `generate_soft_oracle_labels_multilevel` |
| Negative pool | top `K = 1000` documents by **unmasked** score `d · q` over the whole corpus, **then** known positives removed (pool may be < 1000) | same |
| Negatives | `M = 64` drawn uniformly **with replacement** from the pool (`torch.randint`); if the pool has < 64, the whole pool plus uniform draws from the whole corpus (which can include positives); if empty, 64 uniform corpus draws | same |
| Target | `softmax((p̄ − n̄) ⊙ q / τ)`, q and d L2-renormalized inside label generation | same |
| Loss | `KLDivLoss(reduction="batchmean")` on log-probabilities | `train_predictor` |
| Optimizer | AdamW lr 1e-4, weight decay 1e-4, batch 32 shuffled, `CosineAnnealingLR(T_max=epochs)` stepped per epoch | same |
| Checkpoint | **final-epoch weights**; no validation set, no early stopping, no best-KL selection | `run_single_experiment` |
| Seed | default 42, set **once** by `set_random_seed` in `main` (line 636) **before** the loop over models and datasets, so a later dataset in one invocation starts from a consumed RNG state | `main` |
| RNG order per dataset | label draws (`torch.randint` on the training device), then `DataLoader` creation (no draw), model initialization (CPU generator), then one shuffle per epoch (CPU generator) | `run_single_experiment`, `train_predictor` |
| Corpus order | documents sorted by `len(title + text)` descending; this order defines row indices and so `topk` tie-breaking | `run_single_experiment` |
| Deciles | `ceil(D · p / 100)`: 103, 205, 308, 410, 512, 615, 717, 820, 922, 1024 for D = 1024 | `build_eval_schedule_by_percent` |
| Published Qwen3-0.6B | SciFact τ 0.05 / 200 epochs; NFCorpus τ 0.05 / 75; `msmarco_subset_50k` τ 0.10 / 50 | `best_paths_hyperparams.json` |

**Decision**: the reference arm `l2s-official` is retrained per corpus on **fit** queries with this
code path (imported or wrapped, not re-implemented), seed 42, the published τ and epochs, final-epoch
weights, **one process per corpus** so each corpus starts from the seeded state (recorded with device,
torch version, and determinism flags; CUDA and CPU draws differ, so the recorded negative indices,
not a re-draw, are the reference). MS MARCO's parameters come from the authors' 50k-passage subset and are **transferred as
fixed parameters to the full 8.8M corpus**; every table flags this. Published epochs were set for the
authors' full training sets, and every arm here trains on the smaller fit split; this limit is
recorded. Exactly one `l2s-official` is trained per corpus, at seed 42, on the same fit ids as the
primary P3 arms (the bounded prefix if MS MARCO is bounded). Other seeds, a separate full-fit
reference, and temperature sensitivity are deferred to a future study of training scale or
robustness.

**Discrepancies with the project's existing `l2s` (ADR-0002, F3)**, recorded here, not fixed by this
feature: F3 keeps the lowest-validation-KL state (the authors keep the final epoch; the docstring of
`training/selector_trainer.py` says otherwise and is stale); F3 tunes τ × epochs on validation
nDCG@10 at k = 307 (the authors use published fixed values); F3 draws from the "top-1000
non-positives" (the authors take top 1000 then remove positives, with a corpus-wide fallback). F3
checkpoints stay legacy evidence.

**Adapter and parity** (T010–T013): an adapter maps our F2 corpus, ids, qrels, and embeddings into the
authors' inputs, including their length-sorted corpus order, and records every sampled negative
index. Parity tests: (a) on the authors' `beir/datasets/small_test`, our adapter and their script give
identical targets and weights for the same seed on CPU; (b) at 100% the trained selector's ranking
equals the full query; (c) on SciFact, nDCG@10 from their `evaluate_predictor_strict` and from our
exact evaluator agree per query on their decile grid, up to documented ties. A resource probe
(peak RSS, GPU memory, time for label generation over 35,000 fit queries × 8.8M passages) runs on a
bounded slice before MS MARCO is scaled; label generation needs one full-corpus top-1000 per query.

## B4. Shared evaluation protocol

Every arm (full, fixed global mask, `l2s-official`, every candidate): exact float64 dot products over
the full corpus, frozen embeddings, fixed document vectors, masked query never renormalized, one
lower-row tie rule, the project deciles `k ∈ {102, 204, 307, 409, 512, 614, 716, 819, 921, 1024}`
(`resolve_k`, floor). 100% is full-vector parity. Metrics: nDCG@10 (primary), MRR@10, Recall@100,
nDCG@100 (secondary). Any bounded MS MARCO query sample is predeclared, seeded by query-id hash, and
independent of positive counts.

## B5. Stage 2: teachers on fit queries

Rules are in ADR-0018 ("Teachers"). Rationale for the choices:

- nDCG@10 is the objective because it is the endpoint. It is flat over most swaps, though: a single
  positive outside the top 10 scores 0 under every nearby mask, and one at rank 1 cannot improve. So
  ties are broken by the gain-weighted positive discounted rank (GDR), which moves with every rank,
  under strict lexicographic acceptance.
- Starting from both the positive and the contrastive mask, and accepting only strict improvements,
  means the rank-guided teacher never scores below either starting mask on its own fit objective.
- Cost: a swap changes scores by `−d_out·q_out + d_in·q_in`, so scoring one candidate swap is O(N)
  documents, not O(N·D). That makes exact swap scans plausible on SciFact and NFCorpus and points to
  a shortlist for MS MARCO. Neither is claimed until the fit-only cost probe has measured it.

## B6. Stage 3: selectors

Rules are in ADR-0018 ("Selectors"). Primary teacher arms are L2S with only the target replaced, so
a difference from `l2s-official` is attributable to the teacher. The target averages uniform
distributions over the nine budgets' masks: for nested teachers (positive, contrastive) it is a soft
rank; for the rank-guided teacher, whose masks need not nest, it is selection frequency across
budgets, with no forced nesting. Budget-conditioned variants would add capacity over L2S, so they are
deferred. S1 is the no-teacher control (direct masked retrieval). Since it needs tuning, both its
arms share one eight-cell grid, and S1-official-init reuses the existing `l2s-official` checkpoint.
The benchmark is about developing the project's own selectors, so S1's larger selection budget is
stated as a limitation of its secondary comparison rather than balanced with another L2S arm
(B10).

## B7. Statistics

Rules are in ADR-0018 ("Endpoint and statistics"). Helpers already exist in
`evaluation.statistics` (`paired_randomization_test`, `paired_bootstrap_ci`, `holm`). The threshold
0.005 applies to the point estimate for a *practical candidate*. Only a multiplicity-aware lower
bound at or above it supports an *established minimum gain*. No power claim is made without fit or
tune estimates.

## B8. Order of corpora and later tracks

SciFact first, then NFCorpus, then bounded MS MARCO (ADR-0018, "Confirmation scope"). Geometry
ablation, budget-conditioned selectors, and other capacity come after the first in-domain comparison.
Hypernetwork adaptation on held-out corpora is a later, separate feature.

## B9. Decisions

Every design decision is closed and recorded in ADR-0018:

- reference and teachers: fit-only `l2s-official` with the final epoch and the original recipe;
  project deciles, with the authors' grid only for parity; teacher objective `(nDCG@10, GDR)`;
  three teachers with L2S architecture and loss and a nine-budget averaged target; seed-42 matching,
  with teacher and S1 seeds 0–2 as secondary; exactly one `l2s-official` per corpus on the primary
  arms' fit ids;
- controls: frozen full; the S1 eight-cell grid shared by scratch and official initialization (the
  official checkpoint reused); global mask by rank-guided selection frequency; `variance`;
- statistics: one-sided sign-flip and bootstrap with `B = 100,000`, streams from `SeedSequence(20261112)`,
  Holm over the locked family, claim levels, and the replication rule;
- execution: the resource rule (hash-ordered nested prefixes, budgets, probe grid, size priority,
  minimum coverage, escalation, matched fit coverage) and the freeze gates; TREC out of scope; split
  ids preserved and wrapped.

What remains is not a decision but a measurement: the values the fit-only probe realizes (step caps,
shortlist, MS MARCO sizes). They are recorded at the resource gate.

## B10. Review notes on the closed defaults

- **S1 data.** S1's hard negatives are the recorded `l2s-official` negatives, so S1, the teachers,
  and L2S see the same fit data. Checkpoint scoring uses a tune pool of positives plus the top 1,000 by
  unmasked score, the authors' pool rule applied to tune queries, which tune labels may enter. ADR-0009
  used F2's mined pools, which do not exist for the old validation queries now in tune.
- **Selection asymmetry.** Each S1 arm has 8 cells × 21 checkpoints of selection, while `l2s-official`
  and the teacher arms have 1. This is reported as a limitation of the secondary S1 comparison and is
  not equalized.
- **Bounded MS MARCO teacher.** If `n_rg` is bounded, the single `l2s-official` is trained only on the
  same prefix, so the primary contrast stays matched and claims carry the coverage. Whether L2S gains
  from the full fit split is a training-scale question, deferred.
- **Reference-side seed variability** is not measured, since `l2s-official` has one seed. The seed
  spread reported covers teacher-side variability only.
- **Bonferroni bound.** With `m = 9`, the lower bound is the 0.0056 quantile, about 556 of the
  100,000 draws, which is stable enough.
- **Planning projections.** The ADR's cost projections come from about 18 ms per full-corpus MS MARCO
  mask in earlier runs. They guide the probe grid only; the size choice uses probe measurements alone.
- **Freeze records.** Each gate records its inputs and decisions with a SHA-256 fingerprint before
  the dependent stage.

## B11. As-built implementation decisions

Added 2026-11-24, after the code was built and run, so the plan can describe it. Sections B0–B10
are unchanged. None of these decisions changes a rule in ADR-0018; they are about how the rules were
implemented.

- **Exact scorer.**
  - Decision: `evaluation.benchmark.CorpusScorer` keeps the float16 corpus resident on the device
    or streams it from host memory, and widens one chunk at a time to float64.
  - Rationale: float16 products and their sums are exact in float64, so ranks equal the
    full-matrix path bit for bit (tested), without a float64 copy of 8.8M × 1024.
  - Alternative: reuse `retrieval.exact` / `representation_ranking`. Rejected because they need
    the matrix in memory.
- **Evaluator conventions.**
  - Decision: linear gain, with the nDCG ideal over all of the query's positives and lower-row
    ties. The authors' convention (`2^g − 1`, ideal over the retrieved list, float32 cosine)
    appears only in `official-parity`.
  - Rationale: the project metric applies to every arm; parity is a diagnostic. On NFCorpus,
    250 of the 251 per-query differences are explained by the ideal-set rule
    (`scripts/checks/p3_bench_parity_explain.py`).
- **Statistics module.**
  - Decision: `evaluation.benchmark_stats` provides per-procedure generators from
    `SeedSequence(20261112)`, keyed by procedure, corpus and arm. It also provides the exact
    sign-flip path when 2^n ≤ B, `(+1)/(B+1)` Monte Carlo p-values, batched bootstrap, Holm, and
    claim levels.
  - Alternative: `evaluation.statistics`, which lacks the stream and batching contracts.
- **Adapter isolation.**
  - Decision: `baselines.l2s_official` checks the clone's commit, runs the authors' code in a
    subprocess per corpus, and rebuilds every sampled negative from recorded pools. The run fails
    unless the rebuilt negatives reproduce the authors' targets bit for bit.
  - Rationale: FR-004's recording requirement, without modifying the authors' code.
- **MS MARCO search.**
  - Decision: `teachers.shortlist_search` swaps only within the top-N shortlist, then accepts a
    swap only after an exact full-corpus check. N = 1,000 and cap 8 were fixed at the resource gate.
  - Alternative: the exact search, projected at about 1.5× the cost at N = 10,000, and far more
    without a shortlist.
- **Large-corpus resource plan.**
  - Decision: `cli.p3_bench.large_resource_plan` sets sizes in priority order: n_eval at its top
    value, then the largest n_rg that fits, then n_tune. After that it picks N (larger first) and
    the smallest cap where ≥ 95% of searches stop. It also records the
    `realized_without_rank_guided` alternative for use after an escalation.
  - Open: this function has no tests yet (T051).
- **Resumable stages.**
  - Decision: `Checkpoint` stores completed 50-query blocks under
    `artifacts/scratch/p3-bench/<corpus>/<stage>-<config hash>`.
  - A stage stops at a block boundary after `--max-hours` with exit code 75, and resumes with the
    same command. It writes one artifact at the end.
  - It refuses gaps and other configurations.
  - Rationale: the 12 h job cap.
- **Escalation and family.**
  - Decision: `escalate` writes a fingerprinted record of the chosen ADR-0018 option. `family`
    applies Holm and Bonferroni at the locked family size (m = 8 after NFCorpus removed
    rank-guided).
- **Reports.**
  - Decision: report text is generated inside `cli.p3_bench_arms` (`_report`, `_family_report`)
    from `summary.json`.
  - The planned `reports/p3_bench.py` module was not built.
  - Fields FR-009, FR-011, FR-012, FR-013 and US1/AC4 require are still missing from those reports
    (T044–T048, T054).
- **Provenance gap.**
  - Runs used unreferenced code snapshots, kept as bundles in `artifacts/p3-bench/code-snapshots/`,
    so manifests record `git_commit: "unknown"`.
  - The snapshot ids appear only in RUNS.md (T043).
