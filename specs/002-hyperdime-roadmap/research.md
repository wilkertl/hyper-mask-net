# Research: HyperDIME-Qwen Project Roadmap

Phase 0 output for [plan.md](plan.md). Decisions that apply across milestones. Tooling decisions
for M0 (Pydantic records, Hydra compose, `uv` lock, atomic artifact writer, hashing, seeding) are
in [001 research](../001-m0-foundation/research.md) and are not repeated here. Each milestone
feature's own `research.md` may refine these decisions but must not contradict them without an ADR.

## R0. Hypothesis-first spike (F0)

- **Decision**: Test H1 before building anything else, with the smallest setup that can refute it.

  | Choice | Value |
  |---|---|
  | Environments | SciFact (5k docs), NFCorpus (3.6k), FiQA-2018 (57k) with train qrels; ArguAna (8.7k), SciDocs (25k) with a fixed 50/50 query split (train half / eval half) |
  | Size | ≈ 100k documents + ≈ 8k queries, full corpora, texts cut to 512 Qwen tokens; embedded through the `beir` project's vLLM server (same `V4_EMBEDDING_*` settings), < 1 GB of vectors. Fallbacks: local transformers (`--backend local`), and `--corpus-cap 10000` for small hardware (relative comparisons only) |
  | Targets | oracle as in R4 (relevance-weighted positives, top-8 hard negatives from depth 100, training queries only) |
  | Selectors | per-environment linear selector × 3 seeds; one global selector on the union (same labels) |
  | Training | R6 settings; 10% of training queries for early stopping; τ from {0.01, 0.05, 0.1} on validation |
  | Evaluation | exact scoring; nDCG@10 at 30% retained (k = 307), also k ∈ {128, 256, 512}; rows: full-1024, random-k, prefix-k, global, per-domain (in-domain and transferred), oracle |
  | Mask analysis | common battery = the eval queries of all five environments; overlap@k, JS divergence of mean importance, Spearman of dimension ranks; cross-environment vs cross-seed |
  | Statistics | paired randomization test + Holm, bootstrap 95% CI (R9) |

  **Verdict rule (proposed)** — H1 is *supported* when all three hold:
  1. transferred selectors lose significantly to the in-domain selector on ≥ half of the 20 ordered
     pairs;
  2. cross-environment mask overlap@k is significantly below cross-seed overlap;
  3. per-domain beats the global selector by a significant margin on ≥ 2 of 5 environments. This is
     the headroom a hypernetwork could close; without it H1 is true but not useful.

  Optional, same day: a first H2 signal with descriptor group A only (document marginals),
  leave-one-environment-out, predicting each dimension's mean importance (R10).
- **Rationale**: Every piece but the loader, encoder, trainer, and analysis already exists on
  `master`. Small corpora make the spike cost hours, not days. Criterion 3 answers the practical
  question — is there anything for the hypernetwork to gain — which R8 alone does not. ArguAna and
  SciDocs add two very different tasks (counter-argument retrieval, citation prediction) so that
  "shift" is not only within scientific QA.
- **Lightweight provenance**: a JSON manifest per output (git commit, dirty flag, config, seed,
  versions) instead of the full F1 contracts; enough to trace every number, cheap to write.
- **Alternatives considered**: the three Learning-to-Select datasets only (MS MARCO needs an 8.8M
  embedding; SciFact and NFCorpus alone are both scientific domains); building F1–F3 first (weeks
  before the premise is tested).

## R1. Decomposition into milestone features

- **Decision**: Six features beyond this roadmap, one per gate-bounded unit. Milestone 1 is split
  in three because it is the largest and its parts have different reviewers' concerns (tooling,
  data, science).

  | Feature | Backlog tickets | Ends with |
  |---|---|---|
  | **F0** `h1-spike` | loader, encoder + cache, trainer, transfer analysis (R0) | **H1 verdict**; F1–F7 start only if supported |
  | F1 `001-m0-foundation` | M0.1–M0.4 | contracts usable by all modules |
  | F2 `data-embeddings` | M1.1–M1.4, M2.1–M2.2 | cached embeddings + dataset manifests for SciFact, NFCorpus, MS MARCO (+ TREC DL query sets) |
  | F3 `l2s-baseline` | M12.1, M3.1–M3.3, E0 | **G1** |
  | F4 `selector-shift` | M12.3, M4.1–M4.3, G2 | **G2** (Go/No-Go) |
  | F5 `space-descriptor` | remaining reference datasets (R19), meta-split ADR, M5.1–M5.5 | **G3** |
  | F6 `hyperdime-prototype` | M7.1, M6.1, M8.1, M9.1–M9.2 | meta-validation verdict |
  | F7 `paper-evaluation` | M10.1, M11.1, E3, E4/E5, M12.2 | **G4**, **G5** |

- **Rationale**: Matches the plan's milestones and gates (FR-016); every feature ends in something
  a reviewer can check. Directory numbers are assigned by `/speckit-specify` when each feature is
  started.
- **Alternatives considered**: one feature per backlog ticket (≈30 specs, too much ceremony for
  2–3-file tickets); one feature per milestone (Milestone 1 too large for one review).

## R2. Dataset sources and corpora

- **Decision**:
  - The 13 BEIR tasks and MS MARCO come from the Hugging Face Hub (`BeIR/<name>` corpus/queries and
    `BeIR/<name>-qrels`, plus `BeIR/nq-train` for NQ training labels), pinned by dataset revision.
  - The TREC DL 2019, DL 2020, and DL-HARD query sets and qrels come from `ir_datasets`
    (`msmarco-passage/trec-dl-2019/judged`, `msmarco-passage/trec-dl-2020/judged`,
    `dl-hard`), mapped onto the same MS MARCO passage ids.
  - TREC Robust 2004 comes from `ir_datasets` (`disks45/nocr/trec-robust-2004`) and reads the
    licensed TREC disks 4 and 5 from a local path the researcher provides; it is never downloaded
    by the tool. If the path is missing, `prepare-data` skips it with an explicit message.
  - **Full corpora, no sampling.** Corpora are shared where datasets share them: FEVER and
    Climate-FEVER use one embedding of the same Wikipedia corpus; the TREC DL sets reuse the MS
    MARCO passage embeddings.
  - Documents are truncated to 512 tokens (title + text), recorded in `ModelConfig.max_length`;
    this matters only for Robust04 news articles and some DBPedia/TREC-COVID documents.
- **Rationale**: The point of these datasets is comparability with published DIME and
  Learning-to-Select numbers, and those use full corpora; sampling would change every ranking metric.
  The 512-token limit matches the encoders the DIME papers used and bounds embedding cost.
  Pinned revisions make `dataset_hash` meaningful.
- **Alternatives considered**: a 500k-document cap per corpus (the previous decision; about 10×
  cheaper but not comparable with the references, kept as a fallback if embedding time is
  prohibitive); the `beir` package's zip downloads (unpinned URLs); longer inputs for Robust04
  (≈ 4× cost for one environment).

## R3. Environment pool and meta-level splits (proposal; final in ADR before F5)

- **Decision (proposed)**: 15 environments (the TREC DL sets are query sets of MS MARCO, not
  separate environments; see R19).

  | Split | Environments | Label use |
  |---|---|---|
  | meta-train | MS MARCO, NQ (`nq-train` qrels), HotpotQA, FEVER, FiQA-2018, SciFact, NFCorpus | train-split qrels for oracle targets, selectors, negatives |
  | meta-validation | DBPedia (dev qrels), Quora (dev qrels), SciDocs | evaluation labels only, for model selection |
  | meta-test | TREC-COVID, ArguAna, Touché-2020, Robust04 (if licensed) | evaluation labels only, once, after meta-validation |
  | meta-test, query-shift only | Climate-FEVER | same corpus as FEVER; reported separately, never as an unseen domain |

  The TREC DL 2019/2020/DL-HARD sets are in-domain evaluation sets of the MS MARCO environment
  (E0, E1, and in-domain rows of E3). Meta-train environments may also be sharded into semantically
  coherent sub-environments for more episodes; shards are reported as not being unseen domains.
- **Rationale**: Every meta-train environment has a *training* qrels split, so the constitution's
  "never train on test qrels" holds. Every held-out environment needs only evaluation labels, which
  is all P0/P1 require. Meta-test holds three public datasets with tasks far from meta-train
  (biomedical, counter-argument, and argument retrieval) plus Robust04 when licensed, satisfying
  FR-018 even without Robust04. Climate-FEVER isolates a query-distribution shift over a seen corpus,
  which is a useful contrast rather than a leak, as long as it is labeled.
- **Per-domain upper bound on label-only-test environments**: the per-domain selector for a
  meta-validation or meta-test environment is trained on a fixed random half of its test queries and
  **all** methods are then evaluated on the other half. This selector is a reference ceiling, never
  an input to the hypernetwork.
- **Alternatives considered**: leave-one-dataset-out over all environments (much more
  meta-training compute; possible later as a robustness study); putting FEVER in a held-out split
  to make Climate-FEVER unseen (loses the largest fact-checking training set).

## R4. Oracle target with multiple positives

- **Decision**: `p` = relevance-weighted mean of the query's positive document embeddings;
  `n` = mean of its top-M hard negatives (M = 8 by default); `π_q = softmax(e_q ⊙ (p − n) / τ)`.
- **Rationale**: Matches the plan's "aggregated positive and negative" and reduces to the current
  `oracle_scores` for one positive and one negative, so the existing tests remain valid.
- **Alternatives considered**: averaging per-pair oracle distributions (not equal to the plan's
  formula; costlier); max-margin single pair (noisy).

## R5. Hard negatives

- **Decision**: For training queries only, full-dimension exact search with the frozen model,
  depth 100, remove all judged positives, keep the top M. Pools are stored as artifacts with
  manifests and never computed from test queries.
- **Rationale**: Same retriever as evaluation, no extra model; the existing `mine_hard_negatives`
  already implements the core. False negatives are a known risk and are covered by the M sweep in
  E5.
- **Alternatives considered**: BM25 negatives (second retriever); random negatives (too easy;
  weak oracle signal).

## R6. Baseline selector training

- **Decision**: Linear 1024→1024 + log-softmax, KL loss, AdamW (lr 1e-3, weight decay 0.01),
  batch 256, early stopping on validation-split KL with patience 5. τ chosen from
  {0.01, 0.05, 0.1} on validation. Evaluation budgets k ∈ {64, 128, 256, 512, 1024} plus ratios
  for curves. Following Learning to Select's protocol: BEIR in-domain train/test splits, 10% of each
  training set held out as validation, a 50,000 query–document-pair training subsample for MS
  MARCO, and a headline row at a 30% retained-dimension ratio.
- **Rationale**: Reproduces the Learning-to-Select configuration so E0 is comparable with its
  SciFact, NFCorpus, and MS MARCO tables; all choices are on validation queries only.
- **Alternatives considered**: tuning τ on test (forbidden); MLP selector (capacity added only if an
  ablation shows underfitting).

## R7. G1 pass criteria

- **Decision**: G1 passes when, on all three initial environments: (a) validation KL of the trained
  selector is below the uniform predictor's; (b) k = 1024 reproduces unmasked rankings exactly;
  (c) oracle top-k ≥ trained selector ≥ random-k in nDCG@10 at k = 256; (d) the E0 report
  regenerates byte-identically from result files.
- **Rationale**: Turns the plan's qualitative G1 ("converges, mask correct, ranking as expected")
  into checks; exact paper numbers are not required, as the plan states.

## R8. Selector-shift measurement and G2 criterion

- **Decision**: Per-environment selectors trained with 3 seeds each. Measures on a common query
  battery (held-out queries pooled across environments): Jensen-Shannon divergence of mean
  predicted importance, overlap@k of selected dimensions, Spearman correlation of dimension ranks;
  retrieval: in-domain minus cross-domain nDCG@10 for every ordered pair.
  **G2 passes** when both hold:
  1. cross-domain nDCG@10 is significantly below in-domain for at least half of the ordered pairs
     (paired test, Holm-corrected α = 0.05);
  2. across environments, mask overlap@k is significantly lower than across seeds within the same
     environment (seed variability is the null).
- **Rationale**: The plan warns that weight distances are not evidence (parameter symmetries);
  using seed-to-seed variation as the null separates real shift from training noise, and requiring a
  retrieval effect ensures the shift matters.
- **Alternatives considered**: weight-space distances (rejected by the plan); a single seed (no
  null for mask variability).

## R9. Statistics

- **Decision**: Paired two-sided randomization (sign-flip) tests on per-query metric differences,
  10,000 permutations; Holm–Bonferroni within each table's family; 95% paired bootstrap confidence
  intervals (10,000 resamples). Implemented once in `evaluation/statistics.py` and reused.
- **Rationale**: Standard in IR evaluation, distribution-free, and works with the per-query outputs
  `evaluate_run` already returns.
- **Alternatives considered**: paired t-test (normality doubtful for nDCG); Wilcoxon (tests medians,
  ties at 0 are common).

## R10. Space descriptor and G3 probes

- **Decision**: Groups A–E as in the plan, from a sample of 50k documents per environment and, for
  P1, 200 unlabeled calibration queries. Features are z-scored per feature with statistics fit on
  meta-train environments only. **G3 probe**: per-dimension regression (1024 samples per
  environment) predicting each dimension's mean predicted importance under the per-domain selector
  from its descriptor row, with leave-one-environment-out over meta-train ∪ meta-validation.
  Trivial baseline: the mean importance of that dimension in the other environments. Significance:
  permutation test that shuffles descriptor-environment assignment (1,000 permutations).
- **Rationale**: Environment-level probes have only ~7 samples; per-dimension probes give thousands
  while keeping the environment as the held-out unit, which directly tests "descriptor predicts
  selector" (H2). The baseline answers risk R2 (dataset identity vs geometry).
- **Alternatives considered**: CCA between descriptor distances and selector distances (too few
  environment pairs for stable estimates; kept as a secondary analysis).

## R11. Protocol enforcement

- **Decision**: Descriptor builders take typed inputs per protocol: the P0 builder's signature
  accepts only an `EmbeddingArtifact` for documents; P1 adds unlabeled query embeddings; only P2
  accepts labeled records. The meta-trainer validates its config against the meta-split ADR's
  registry and refuses meta-test environments. Tests assert that P0/P1 code paths never import or
  open qrels files.
- **Rationale**: Makes leakage a type or validation error instead of a convention (constitution
  II, SC-003).

## R12. Prototype architecture defaults

- **Decision**: Space encoder V1 = shared MLP over descriptor rows (hidden 128, output h_z = 64) +
  learned 1024 × 64 dimension-ID embedding. Selector `W_T = W_0 + A_T B_Tᵀ`, `b_T = b_0 + Δb_T`,
  rank r = 8 (ablated over {4, 8, 16, 32}). Hyper head: per-dimension linear heads on `Z_T` produce
  the rows of `A_T` and `B_T` and the entries of `Δb_T`, followed by a learnable scalar gate
  initialized to 0. `W_0, b_0` initialized from the global selector (R13).
- **Rationale**: Per-dimension heads map naturally from a 1024-row representation to 1024-row
  factors and keep parameter count independent of D²; the zero gate gives exact equality with the
  base selector at initialization (spec US4, scenario 1).
- **Alternatives considered**: diagonal/FiLM modulation (kept as an ablation, lower capacity);
  generating full W (infeasible, rejected by the plan).

## R13. Global selector control (risk R6)

- **Decision**: The global selector is a linear selector trained on the union of meta-train
  environments' training queries, with the same label budget the hypernetwork sees. It is both the
  initialization of `W_0, b_0` and the main comparison in every hypernetwork table.
- **Rationale**: Separates the benefit of supervision from the benefit of environment conditioning.

## R14. Episodic meta-training

- **Decision**: Each step samples one meta-train environment (uniform over environments, not
  queries), resamples its descriptor inputs from its training data, generates `θ_T`, and takes a
  batch of 256 training queries. Loss: KL only at first; `λ_delta ||Δθ_T||²` added if deltas grow
  unstable (risk R4). Model selection on meta-validation mean nDCG@10 at k = 256 under P1.
- **Rationale**: Environment-uniform sampling stops MS MARCO from dominating; descriptor
  resampling acts as augmentation and tests robustness to sampling.

## R15. Selection policies

- **Decision**: F7 provides fixed top-k and fixed ratio (already in `selection/topk.py`). The
  RDIME-inspired adaptive-k policy is implemented only after its criterion is reproduced on the
  baseline, and is reported as a heuristic if its assumptions do not hold for learned scores.
  Budget-aware policy: out of scope.

## R16. Efficiency accounting (G5)

- **Decision**: Report separately: Qwen query encoding, selector forward, mask, scoring, and search,
  each timed with device synchronization over ≥ 1,000 queries; offline adaptation time and
  `θ_T` size per environment. Scoring cost for masked queries is reported two ways: (a) exact dense
  scoring (no saving: zeroed coordinates still multiply), and (b) scoring restricted to the selected
  coordinates, with per-query gathers and FLOPs k/D. FAISS (flat and IVF) appears only in this
  deployment analysis.
- **Rationale**: A masked query does not make dense scoring cheaper by itself; G5 is only meaningful
  if the restricted-coordinate path is measured. Flagged as a project risk in the plan.

## R17. Compute and storage budget

- **Decision**: One GPU workstation. Unique corpora total ≈ 28.5M documents (MS MARCO 8.84M,
  FEVER/Climate-FEVER 5.42M shared, HotpotQA 5.23M, DBPedia 4.64M, NQ 2.68M, Robust04 0.53M,
  Quora 0.52M, Touché 0.38M, TREC-COVID 0.17M, the rest < 0.06M each) plus ≈ 1.2M queries:
  ≈ 60 GB at float16. Embedding runs once per corpus, resumable in shards, in order of need:
  F2 embeds SciFact, NFCorpus, MS MARCO (≈ 9M documents); F5 embeds the rest only if G2 passes.
  Exact search over 8.8M × 1024 runs in GPU batches, streaming document shards from disk.
  Selectors are ≈ 4 MB each; the hypernetwork is < 10M parameters.
- **Rationale**: Fits the single-workstation assumption; sharded, resumable embedding keeps a
  crash from costing a whole corpus; the G2 ordering spends most of the embedding budget only when
  the premise holds.

## R18. Reports and tables

- **Decision**: `reports/<experiment>/` holds generated Markdown and CSV tables and figures, built by
  `hyperdime report <experiment>` from `EvaluationResult` files and their manifests. Each table row
  links to its manifest `run_id`. `exp/` branches commit only the generated reports and configs.
- **Rationale**: Satisfies "no numbers by hand" (SC-002) and the `exp/` branch rule in the workflow.

## R19. Datasets from the reference papers

- **Decision**: The environment pool is the union of the datasets in the experiments of the
  reference papers in `Papers_2026` (FR-018). Read from each paper's experimental setup:

  | Dataset | Queries evaluated | DIME (SIGIR'24) | Getting off the DIME | CoDIME | Eclipse | Unveiling DIME | Stat. Found. DIME | Learning to Select | Role here |
  |---|---|---|---|---|---|---|---|---|---|
  | MS MARCO passage | dev / train | corpus | corpus | corpus | corpus | corpus | corpus | ✓ | meta-train env |
  | TREC DL 2019 | 43 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | | MS MARCO eval set |
  | TREC DL 2020 | 54 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | | MS MARCO eval set |
  | DL-HARD | 50 | ✓ | ✓ | | ✓ | ✓ | ✓ | | MS MARCO eval set |
  | TREC Robust 2004 | 249 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | | meta-test (licensed) |
  | SciFact | 300 test | | | | | ✓ | | ✓ | meta-train |
  | NFCorpus | 323 test | | | | | ✓ | | ✓ | meta-train |
  | ArguAna | 1,406 | | | | | ✓ | | | meta-test |
  | Climate-FEVER | 1,535 | | | | | ✓ | | | meta-test, query-shift only |
  | DBPedia | 400 test | | | | | ✓ | | | meta-validation |
  | FEVER | 6,666 test | | | | | ✓ | | | meta-train |
  | FiQA-2018 | 648 test | | | | | ✓ | | | meta-train |
  | HotpotQA | 7,405 test | | | | | ✓ | | | meta-train |
  | NQ | 3,452 test | | | | | ✓ | | | meta-train (`nq-train` labels) |
  | Quora | 10,000 test | | | | | ✓ | | | meta-validation |
  | SciDocs | 1,000 | | | | | ✓ | | | meta-validation |
  | Touché-2020 | 49 | | | | | ✓ | | | meta-test |
  | TREC-COVID | 50 | | | | | ✓ | | | meta-test |

  Unveiling DIME states that it excludes the four BEIR corpora that are not publicly available
  (BioASQ, Signal-1M, TREC-NEWS, Robust04); MS MARCO and CQADupStack also do not appear in its
  BEIR table, which leaves the 13 tasks above. CoDIME simulates clicks on
  DL'19, DL'20, and Robust04 relevance labels; that simulation is out of scope here. Two related
  papers in the folder (When Reducing Representations Improves; Load-sensitive Selective Pruning)
  use only MS MARCO, DL'19/'20, and Robust04, so they add nothing new.
- **Rationale**: Reporting on the same collections, with the same metrics (FR-019), lets every
  claim be placed next to the published DIME and Learning-to-Select numbers. The per-paper matrix
  also shows which comparisons are possible: DIME-family on TREC collections, Learning to Select on
  SciFact/NFCorpus/MS MARCO.
- **Alternatives considered**: only the BEIR subset (loses the DIME papers' main TREC
  collections); adding CQADupStack (not used by any reference).
