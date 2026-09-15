# Complete Implementation Plan — Hypernetwork for Adaptive Dimension Selection in Qwen3-Embedding-0.6B

**Goal:** implement and validate an architecture in which an *Embedding-Space Encoder* represents the retrieval environment produced by **Qwen3-Embedding-0.6B**, a **Hyper Head** converts that representation into the parameters of a *Dimension Selector*, and the selector produces per-dimension importance for each query — analogous to the predictor in *Learning to Select*, but adaptable to the environment/corpus.

**Fixed scope of the first version:** **Qwen3-Embedding-0.6B** only, maximum dimension **D = 1024**, frozen encoder, query-side selection, documents/index unmodified during selection.

---

## 1. Scientific basis and separation between evidence and proposal

### 1.1 What comes directly from the reference works

1. **Learning to Select (Wu et al., ACL 2026).** The method builds a target per-dimension importance distribution using supervised positives and hard negatives, and trains a query-only predictor to approximate it. At inference, the predictor receives only the query embedding, selects the Top-k dimensions, zeroes the rest, and keeps document embeddings and the ANN index unchanged. In the paper, the predictor is a single fully connected D→D layer followed by (log-)softmax. Qwen-Embedding-0.6B is evaluated with D=1024.
2. **Hypencoder (Killingback et al., SIGIR 2025).** This work uses a hypernetwork/hyperhead to turn the query representation into the weights and biases of a small query-specific network (*q-net*). What we reuse is the architectural pattern "context → parameter generator → target network", not the Hypencoder's scoring function.
3. **Statistical Foundations of DIME / RDIME (D'Erasmo et al., EACL 2026).** This work formalizes query-dependent selection and proposes a criterion for adaptively determining how many dimensions to retain per query, avoiding a single global k chosen by grid search. RDIME is an optional extension for the final selection step.
4. **Qwen3-Embedding-0.6B.** The model produces embeddings of up to 1024 dimensions, supports custom dimensionalities/MRL, and is instruction-aware. In this implementation, the full 1024 dimensionality will be preserved when generating embeddings, to allow arbitrary coordinate selection.

### 1.2 What is a hypothesis/proposal of this project

The central hypothesis is not established by the works above. It will be tested experimentally:

> **H1 — Selector shift:** the optimal dimension selector for the same Qwen3-Embedding-0.6B changes systematically across retrieval environments.
>
> **H2 — Geometry explains selector shift:** unsupervised/semi-supervised statistics of the space induced by Qwen contain enough information to predict part of that change.
>
> **H3 — Hypernetwork adaptation:** a Hyper Head can map the environment representation to the parameters of a query-aware selector and improve cross-domain generalization compared with a fixed global selector.

The causal chain to investigate is:

`distribution shift → embedding/retrieval geometry shift → optimal selector shift → hypernetwork adaptation`

The implementation must be built so that each link can be **refuted** separately. If H1 fails, the hypernetwork is unnecessary. If H1 holds and H2 fails, the space representation is insufficient. If H1/H2 hold and H3 fails, the problem lies in the parameterization/training of the Hyper Head.

---

## 2. Final conceptual architecture

### 2.1 Offline path, per environment

A retrieval environment is defined as:

`T = (corpus C, query distribution Q, instruction I, fixed Qwen3-Embedding-0.6B)`

Offline pipeline:

`C (+ optional Q_calib) → Qwen embeddings → Space Statistics R_T → Space Encoder Gψ → environment representation Z_T → Hyper Head Hφ → selector parameters θ_T`

The weights `θ_T` are persisted and reused for all queries of that environment.

### 2.2 Online path, per query

`query → Qwen3-Embedding-0.6B → e_q ∈ R^1024 → Dimension Selector f_{θ_T} → importance scores s_q ∈ R^1024 → selection policy → mask m_q → masked query → retrieval`

The minimal version keeps the *Learning to Select* strategy:

`e'_q = e_q ⊙ m_q`

`score(q,d) = (e'_q)^T e_d`

Document embeddings remain unchanged.

### 2.3 Recommended environment representation

Because the encoder is always the same, the identity of the 1024 coordinates is stable. The representation must therefore be:

- invariant to the order of the sampled documents/queries;
- **not invariant to the identity of the dimensions**;
- built as a per-dimension matrix:

`R_T ∈ R^(1024 × h_stats)`

Each row `r_j` represents dimension `j` and aggregates four families of signals:

1. **Marginal/document:** mean, standard deviation, quantiles, energy, and sparsity-like statistics of `d_j`.
2. **Query:** the same statistics for `q_j`, when calibration queries are available.
3. **Interaction:** statistics of `q_j d_j`, approximating the dimension's contribution to the inner product.
4. **Contrast/redundancy:** statistics of `q_j(p_j-n_j)` with pseudo/true positives and negatives, plus inter-dimensional correlation/redundancy information.

The first version should start simple and add groups through ablation.

---

# 3. Implementation principles

1. **Frozen and deterministic encoder.** The implementation must never change Qwen's weights in the initial phase.
2. **Contracts before models.** Every module must have serializable, versioned inputs/outputs before the neural implementation.
3. **Baselines first.** The Hyper Head only comes in after reproducing *Learning to Select* and the selector-shift experiment.
4. **Every hypothesis has a falsifiable test.** Metrics and acceptance criteria must be defined before the main training run.
5. **No "magic pipeline".** Space statistics, space encoder, hyperhead, target selector, and policy are independent, replaceable modules.
6. **Separate training from inference.** Labels may be used to create targets in meta-training; inference in a new environment must use only the signals allowed by the experimental protocol.
7. **Full reproducibility.** Configuration, seed, dataset version, checkpoint hash, model revision, and results must be recorded for every run.

---

# 4. Repository structure

```text
hyperdime-qwen/
├── pyproject.toml
├── README.md
├── AGENTS.md
├── .gitignore
├── .pre-commit-config.yaml
├── configs/
│   ├── model/qwen3_0_6b.yaml
│   ├── data/{scifact,nfcorpus,msmarco}.yaml
│   ├── space_rep/*.yaml
│   ├── selector/*.yaml
│   ├── hyperhead/*.yaml
│   └── experiment/*.yaml
├── src/hyperdime/
│   ├── contracts/
│   │   ├── schemas.py
│   │   ├── manifests.py
│   │   └── validation.py
│   ├── data/
│   │   ├── loaders.py
│   │   ├── splits.py
│   │   ├── sampling.py
│   │   └── negatives.py
│   ├── embeddings/
│   │   ├── qwen.py
│   │   ├── cache.py
│   │   └── normalize.py
│   ├── oracle/
│   │   ├── importance.py
│   │   └── targets.py
│   ├── baselines/
│   │   ├── learning_to_select.py
│   │   ├── global_selector.py
│   │   ├── mrl_prefix.py
│   │   └── random_mask.py
│   ├── space/
│   │   ├── marginal.py
│   │   ├── query_stats.py
│   │   ├── interactions.py
│   │   ├── contrast.py
│   │   ├── redundancy.py
│   │   ├── descriptor.py
│   │   └── encoder.py
│   ├── hypernet/
│   │   ├── hyperhead.py
│   │   ├── modulation.py
│   │   └── generated_selector.py
│   ├── selection/
│   │   ├── topk.py
│   │   ├── rdime.py
│   │   └── masks.py
│   ├── retrieval/
│   │   ├── exact.py
│   │   ├── faiss_backend.py
│   │   └── scoring.py
│   ├── training/
│   │   ├── selector_trainer.py
│   │   ├── meta_trainer.py
│   │   ├── episodes.py
│   │   └── losses.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── selector_shift.py
│   │   ├── cross_domain.py
│   │   ├── ablations.py
│   │   └── statistics.py
│   └── cli/
│       ├── embed.py
│       ├── build_space.py
│       ├── train_selector.py
│       ├── train_hypernet.py
│       └── evaluate.py
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── regression/
│   └── fixtures/
├── scripts/
│   ├── smoke_pipeline.sh
│   └── reproduce_paper.sh
├── experiments/
│   ├── 00_baseline_reproduction/
│   ├── 01_selector_shift/
│   ├── 02_space_signal/
│   ├── 03_hypernetwork/
│   └── 04_ablations/
├── artifacts/              # gitignored; manifests are versioned
├── reports/
│   ├── tables/
│   └── figures/
└── docs/
    ├── architecture.md
    ├── protocols.md
    ├── decisions/
    └── experiment_registry.md
```

---

# 5. Chained implementation modules

## M0 — Bootstrap, contracts, and reproducibility

**Scientific goal:** prevent pipeline differences from being mistaken for model differences.

**Implement:**
- `pyproject.toml` with pinned versions for `torch`, `transformers`, `datasets`, `faiss`, `numpy`, `scipy`, `scikit-learn`, `ir-measures`, `pytest`, `ruff`, `mypy`, `hydra-core`/`omegaconf`.
- a single seed manager;
- `RunManifest` containing `git_commit`, `config_hash`, `dataset_hash`, `model_revision`, `seed`, device, dtype, and timestamps;
- schemas for `QueryRecord`, `DocumentRecord`, `QrelsRecord`, `EmbeddingArtifact`, `SpaceDescriptor`, `SelectorCheckpoint`, `EvaluationResult`.

**Output:** a diagnostic CLI that loads a config and generates a manifest without training anything.

**Tests:** serialization round-trip; deterministic hashes; an invalid config fails explicitly.

**Gate:** no later module may write an artifact without a manifest.

---

## M1 — Data layer and splits

**Scientific goal:** ensure that selector shift is not caused by inconsistent splits or leakage.

**Implement:** normalized loaders for SciFact, NFCorpus, and MS MARCO; canonical IDs; qrels; train/validation/test; reproducible sampling of documents and queries; hard-negative pools kept separate from the evaluation set.

**Main contract:**

```python
DatasetBundle(
    corpus: dict[str, str],
    queries: dict[str, str],
    qrels: list[Qrel],
    split: str,
    instruction: str,
)
```

**Tests:** qrel IDs exist; no test label enters training; split hashes are stable; tiny fixtures run on CPU.

**Gate:** produce `dataset_manifest.json` for each environment.

---

## M2 — Frozen Qwen3-Embedding-0.6B

**Scientific goal:** completely fix the 1024-dimension coordinate system.

**Implement:** a single Qwen wrapper; dimension=1024; pooling as in the official implementation; configurable instruction for queries; documents with no transformation beyond what the model recommends; configurable L2 normalization, enabled in the main protocol.

**Artifacts:** `documents.f16.npy`, `queries.f16.npy`, ID→row maps, metadata with revision/hash.

**Invariants:** shape `(N,1024)`; norm ≈1 when normalized; the same text/config/revision produces the same vector within numerical tolerance.

**Tests:** shape; norm; batching; cache hit/miss; CPU/GPU comparison with tolerance; 10-document smoke test.

**Gate:** embeddings are frozen and reused in the following experiments.

---

## M3 — Importance oracle and reproduction of Learning to Select

**Scientific goal:** have a validated baseline before introducing the hypernetwork.

For each query `q`, form the aggregated positive embedding `p` and aggregated negative embedding `n`. Implement the per-dimension oracle score:

`r_q,j = e_q,j (p_j - n_j)`

Turn it into a target distribution with temperature:

`π_q = softmax(r_q / τ)`

Implement the baseline predictor:

`f_θ : R^1024 → R^1024`

Initially a single linear 1024→1024 layer followed by log-softmax, reproducing the configuration described in the paper.

Main loss:

`L_selector = KL(π_q || π̂_q)`

At inference: `Top-k(π̂_q)`, query-only mask, documents unchanged.

**Mandatory additional baselines:** full 1024; MRL prefix-k; random-k; global learned importance; oracle top-k (upper bound); per-domain predictor.

**Tests:** oracle checked by hand on a small tensor; exact Top-k; the mask does not alter documents; ranking equivalent to a score computed only over the selected coordinates; overfit on a synthetic dataset of 20 queries.

**Scientific gate:** reproduce the paper's central trend: the predictor learns the target and the mask pipeline works without reindexing documents. Do not require identical numbers before aligning datasets/hyperparameters.

---

## M4 — Experiment 0: prove or refute selector shift

**This module is mandatory before the Hyper Head.**

Train independent selectors:

`θ_SciFact*`, `θ_NFCorpus*`, `θ_MSMARCO*`

Build a transfer matrix:

`Train domain A → Test domain B`

Measure:
- nDCG@10, MRR@10/100, Recall@100;
- KL/Jensen-Shannon between the importance distributions produced for the same battery of queries;
- overlap@k between selected dimensions;
- Spearman correlation of dimension rankings;
- cross-domain vs in-domain degradation.

**Criterion for H1:** there must be consistent evidence that specialized selectors behave differently and that at least part of those differences matters for retrieval. Distance between weights alone is not sufficient evidence, because of parameterization symmetries.

**Decision:**
- if there is no functional selector shift, stop the Hyper Head work and pursue a universal selector;
- if there is, proceed to M5.

---

## M5 — Space Descriptor v1: observable per-dimension statistics

**Scientific goal:** test H2 without training a hypernetwork yet.

Build `R_T ∈ R^(1024×h)` with versioned features.

### Group A — Document marginal
Per dimension `j`: mean, std, mean_abs, RMS, robust min/max, q05/q10/q25/q50/q75/q90/q95, optional skewness, optional kurtosis.

### Group B — Query marginal
The same features using `Q_calib`, without labels.

### Group C — Interaction
For controlled samples of query-document pairs: mean/std/quantiles of `q_j*d_j`; split random pairs and top-retrieved pairs into separate channels.

### Group D — Contrast
When allowed by the protocol:
- supervised in meta-training: `Δ_q,j = q_j(p_j-n_j)`;
- unsupervised in a new domain: pseudo-positive top-K and random/lower-ranked negatives.

### Group E — Redundancy
Avoid feeding the full covariance to the network initially. Use per dimension: mean absolute correlation, max absolute correlation, top-r correlation energy, and participation in principal components. Keep the full covariance only as a diagnostic artifact.

**Normalization:** statistics must be normalized across environments using parameters computed **only on the meta-train environments**.

**Scientific test of H2 — before the Hyper Head:** evaluate whether distance/similarity between `R_T` relates to the functional distance between the selectors from M4. Use simple regressions/CCA/probes and permutation tests. If no signal exists, iterate on the descriptor.

**Gate:** at least one feature set must predict a property of the selector above a trivial baseline — e.g. mean per-dimension importance, or degradation when transferring a selector between environments.

---

## M6 — Space Encoder

**Goal:** turn `R_T` into a learned representation while preserving dimension identity.

### Recommended V1 — per-dimension MLP + dimension ID embedding

For each dimension:

`t_j = MLP_stats(r_j) + p_j`

where `p_j` is a learned embedding of the identity `j ∈ [1,1024]`.

Output: `Z_T ∈ R^(1024×h_z)`.

### V2 — contextualization across dimensions

Add 1–3 Transformer blocks over the 1024 dimension tokens to model redundancy/interaction. Do not implement V2 before an ablation shows a limitation of V1.

**Tests:** permuting documents does not change `R_T`; permuting dimensions changes the representation; gradients flow; fixed output shape; controlled memory.

---

## M7 — Parameterizable Target Dimension Selector

The target network must stay small so that the Hyper Head has a tractable problem.

### Target baseline

`f_θ(e_q) = W e_q + b`, with `W ∈ R^(1024×1024)`.

Generating all of W would require >1M parameters per environment. Therefore, the first hypernetwork **must not generate W directly**.

### Recommended parameterization: residual low-rank modulation

Keep a global base selector:

`W_T = W_0 + A_T B_T^T`

with rank `r << 1024`, e.g. `r ∈ {4,8,16,32}`.

Bias:

`b_T = b_0 + Δb_T`

The Hyper Head generates only `A_T`, `B_T`, and optionally `Δb_T`.

An even simpler alternative for a first proof:

`W_T = diag(γ_T) W_0 + diag(β_T)` or FiLM modulation of a hidden state.

**Principle:** first prove adaptation with low capacity; increase capacity only if underfitting is demonstrated.

**Tests:** with delta=0, output identical to the global selector; shapes; effective rank; deterministic checkpoint.

---

## M8 — Hyper Head

**Input:** `Z_T` from the Space Encoder.

**Output:** parameters/modulations of the target selector.

### Recommended V1

1. per-dimension contextualization over `Z_T`;
2. optional global pooling `g_T` for environment context;
3. separate heads to generate the low-rank factors;
4. explicit scale control so that `ΔW` starts close to zero.

Scheme:

`Z_T → HyperHead_A → A_T`

`Z_T → HyperHead_B → B_T`

`Pool(Z_T) → HyperHead_b → Δb_T`

Recommended initialization: the final stage that generates deltas starts near zero so that `θ_T ≈ θ_0` initially, preventing the first step from destroying the base selector.

**Do not literally copy the Hypencoder's hyperhead.** The Hypencoder conditions the network on query tokens and produces one q-net per query. Here the context is an environment descriptor, and the network produced/modulated is persisted per environment.

---

## M9 — Episodic/meta-learning training

The training unit is an **environment episode**, not an isolated query.

For each episode:

1. choose environment `T_i`;
2. load `R_Ti`, computed only with allowed data;
3. generate `θ_Ti = Hφ(Gψ(R_Ti))`;
4. sample a minibatch of queries from the environment;
5. obtain the oracle `π_q`;
6. compute `π̂_q = f_{θ_Ti}(e_q)`;
7. compute the loss;
8. backpropagate through generated selector → Hyper Head → Space Encoder.

Initial loss:

`L = KL(π_q || π̂_q)`

Add only later:

`L_total = L_KL + λ_rank L_rank + λ_delta ||Δθ_T||²`

where `L_rank` can be a surrogate over the positive-negative difference after masking.

### Meta-learning splits

Environments must be separated, not just queries:
- **meta-train environments**: used to learn Gψ/Hφ/θ0;
- **meta-validation environments**: architecture/hyperparameter selection;
- **meta-test environments**: never used for gradients or for descriptor normalization.

With few datasets, create additional environments by domain/subdomain, instruction, or semantically coherent shard, but document that shards of the same dataset are not equivalent to a truly unseen domain.

**Gate:** the hypernetwork must beat or match the global selector on meta-validation before meta-test is evaluated.

---

## M10 — Inference protocols

Implement three explicit levels; never mix them in a single table without labeling.

### P0 — Corpus-only zero-query

Input during adaptation: document embeddings only. Uses Groups A/E of the descriptor. This is the strongest zero-shot protocol, but has less signal.

### P1 — Unlabeled calibration queries

Input: corpus + a small set of queries without qrels. Enables Groups B/C and pseudo-feedback. Probably the most realistic main protocol.

### P2 — Few-label calibration

Allows a small labeled set for the descriptor/adaptation. Must be treated as a separate scenario, not as zero-shot.

Online, once `θ_T` has been generated, each query needs only:

`encode query → selector forward → mask → retrieval`

Record Qwen, selector, mask/scoring, and search latency separately.

---

## M11 — Selection Policy

The architecture must separate **importance prediction** from the **number-of-dimensions policy**.

### Policy A — Fixed Top-k
Mandatory for direct comparison with *Learning to Select*.

### Policy B — Fixed ratio
`k = floor(ρD)` for cost-quality curves.

### Policy C — RDIME-inspired adaptive k
Implement after reproduction of the criterion is validated; use the importance score produced by the new selector as input only if the theoretical formulation remains valid. If RDIME's statistical assumption is not directly compatible with learned scores, report it as an RDIME-inspired heuristic, not as canonical RDIME.

### Policy D — Budget-aware
Optional: choose k subject to a latency budget, keeping this project's contribution separate from the load-sensitive strategy.

---

## M12 — Retrieval and evaluation

Start with **exact dot product** to eliminate ANN effects. Only then add FAISS.

Primary metrics: nDCG@10, MRR@10/100, Recall@100.

Selection metrics: number of retained dimensions; overlap@k; entropy of `π̂_q`; KL to the oracle; distribution of used dimensions; per-dimension frequency.

Efficiency metrics: selector time; total query time; memory; FLOPs/estimate; offline adaptation cost; size of `θ_T`.

Generalization metrics: in-domain; selector transfer A→B; hypernetwork zero-shot/few-query; gap to the oracle/per-domain selector.

**Significance:** paired per-query tests, with multiple-comparison correction when there are families of hypotheses. Report confidence intervals, not just p-values.

---

# 6. Scientific experimental plan

## E0 — Reproduction sanity
Goal: validate embeddings, oracle, and baseline selector.

## E1 — Selector shift
Question: does a selector trained on A degrade on B? Do the masks/importance distributions change systematically?

## E2 — Space signal
Question: do environment descriptors predict properties of the specialized selector?

## E3 — Hypernetwork main result
Compare:
- full 1024;
- MRL prefix;
- per-domain Learning-to-Select;
- global multi-domain selector;
- selector trained on source and transferred;
- HyperDIME P0;
- HyperDIME P1;
- oracle/per-domain upper bound.

## E4 — Ablations
Remove one group at a time: document stats; query stats; interaction; contrast; redundancy; dimension-ID embeddings; inter-dimension Transformer; low-rank rank; number of calibration queries.

## E5 — Sensitivity
Seeds, corpus sample size, Q_calib sample size, temperature τ, hard-negative K/M, rank r, selection k.

## E6 — Efficiency
Separate offline adaptation cost from online per-query cost.

---

# 7. Minimal ablation matrix

| Experiment | Space input | Hyper Head | Selector | Purpose |
|---|---|---|---|---|
| B0 | none | no | fixed global | universal baseline |
| B1 | none | no | per-domain | specialization upper bound |
| H0 | document stats | yes | low-rank | test corpus-only |
| H1 | + query stats | yes | low-rank | effect of the query distribution |
| H2 | + interactions | yes | low-rank | contribution of the scoring mechanism |
| H3 | + contrast | yes | low-rank | signal close to the oracle |
| H4 | H3 + redundancy | yes | low-rank | complementarity between dimensions |
| H5 | H4 | yes | richer target | need for greater capacity |

(Note: the experiment IDs H0–H5 in this table are unrelated to hypotheses H1–H3 in §1.2.)

---

# 8. Testing strategy

## 8.1 Unit tests
Every critical mathematical function must have a test on small tensors with a hand-computable result: L2 normalization, oracle `q*(p-n)`, KL target, masks, Top-k, descriptor stats, correlation summaries, low-rank reconstruction.

## 8.2 Property tests
- permuting documents does not change the descriptor;
- swapping a dimension must swap the corresponding channel, not make it disappear;
- `k=1024` is equivalent to full-dimensional scoring;
- zero modulation is equivalent to the base selector;
- no meta-test function may access qrels under protocol P0/P1.

## 8.3 Integration tests
Synthetic end-to-end pipeline with 50 docs/10 queries on CPU: mocked embedding fixture → oracle → train selector → build descriptor → hyperhead → inference → metric.

## 8.4 Regression tests
Save results from deterministic fixtures and check tolerances. Do not use large metrics from external datasets as unit tests.

## 8.5 Scientific tests
Scripts that automatically generate tables E1–E6 from manifests, with no manual editing of numbers.

---

# 9. Repository control for working with agents in the IDE

## 9.1 Branching

- `main`: only reproducible, protected states.
- each agent task: `agent/<ticket>-<module>-<description>`.
- never allow two agents to change the same module without an explicit contract.

Examples:

`agent/M3-oracle-targets`

`agent/M5-document-stats`

`agent/M8-lowrank-hyperhead`

## 9.2 Commits
Use Conventional Commits:

`feat(space): add per-dimension marginal descriptor`

`test(selector): verify masked dot-product equivalence`

`fix(data): prevent qrel leakage into meta-test descriptor`

Each commit must pass the tests related to its module.

## 9.3 Pull requests
Every PR must state:
1. contract changed;
2. files changed;
3. tests added;
4. commands run;
5. expected artifacts/regressions;
6. whether it changes the scientific protocol;
7. whether it invalidates previous results.

## 9.4 AGENTS.md
Create a file with mandatory rules for agents:
- read `docs/architecture.md` and the module contract before editing;
- do not change public APIs outside the ticket;
- never use test qrels in adaptation;
- do not download/commit datasets or checkpoints into Git;
- add a test for every new behavior;
- run `ruff`, `mypy`, and a `pytest` subset before committing;
- record non-trivial architectural decisions in `docs/decisions/ADR-XXXX.md`;
- do not manually "fix" experiment numbers;
- every table must be derived from result files.

## 9.5 Large artifacts
Git must version only manifests/configs. Embeddings/checkpoints live in `artifacts/` or external storage. Each artifact gets a SHA256 and provenance. If needed, adopt DVC after the first reproduction; do not make DVC a critical dependency during bootstrap.

---

# 10. Execution sequence for agents

## Phase A — Foundation
**A1:** M0 contracts/reproducibility.
**A2:** M1 datasets.
**A3:** M2 Qwen embeddings/cache.

Constraint: A2/A3 may not define parallel schemas; they use M0.

## Phase B — Scientific baseline
**B1:** M3 oracle.
**B2:** M3 predictor.
**B3:** exact retrieval + metrics.
**B4:** reproduction and report.

Constraint: no hypernetwork before B4.

## Phase C — Evidence for the hypothesis
**C1:** M4 per-domain selectors.
**C2:** cross-domain transfer matrix.
**C3:** selector behavior analysis.
**C4:** Go/No-Go decision for H1.

## Phase D — Space representation
**D1:** marginal stats.
**D2:** query/interaction stats.
**D3:** contrast/redundancy.
**D4:** H2 probes and permutation tests.
**D5:** freeze `SpaceDescriptorV1`.

## Phase E — Hypernetwork
**E1:** low-rank target selector.
**E2:** space encoder V1.
**E3:** hyperhead V1.
**E4:** episodic trainer.
**E5:** meta-validation.

## Phase F — Final evaluation
**F1:** P0.
**F2:** P1.
**F3:** ablations.
**F4:** efficiency.
**F5:** significance + tables.

---

# 11. Agent-ready tickets

### Ticket M3.1 — Oracle importance
**Inputs:** query embeddings, relevant doc IDs, hard-negative IDs.
**Output:** `TargetImportance[query_id,1024]`.
**DoD:** unit tests against manual computation; distribution sums to 1; no NaN; CLI generates artifact + manifest.

### Ticket M4.2 — Selector transfer matrix
**Inputs:** per-domain checkpoints + test bundles.
**Output:** A×B CSV/Parquet with metrics and mask similarity.
**DoD:** no training runs during evaluation; table reproducible with one command.

### Ticket M5.1 — Document descriptor
**Inputs:** N×1024 matrix.
**Output:** 1024×h and a schema of feature names.
**DoD:** invariance to row permutation; tolerance to sampling; no qrels used.

### Ticket M7.1 — Low-rank selector
**Inputs:** base W0,b0 + A,B,Δb.
**Output:** 1024 logits.
**DoD:** zero delta == base; autograd validated; configurable rank.

### Ticket M8.1 — Hyperhead
**Inputs:** `SpaceDescriptorV1`/`Z_T`.
**Output:** modulation package.
**DoD:** generated parameters respect shapes; near-zero initialization; gradient test.

### Ticket M9.1 — Episodic trainer
**Inputs:** list of meta-train environments.
**Output:** `Gψ,Hφ,θ0` checkpoint, logs, and manifest.
**DoD:** batches alternate environments; meta-test is impossible to load in the trainer, enforced by config validation.

---

# 12. Prompt template for agents in the IDE

```text
You are implementing ticket <ID> of the HyperDIME-Qwen project.

Before editing:
1. read AGENTS.md;
2. read docs/architecture.md;
3. read the contract in src/hyperdime/contracts;
4. list the files you intend to change;
5. do not change APIs outside the ticket without justification.

Ticket goal:
<goal>

Required inputs/outputs:
<contract>

Acceptance criteria:
<DoD>

Scientific constraints:
- Qwen3-Embedding-0.6B frozen;
- D=1024;
- do not use meta-test qrels in adaptation;
- separate offline and online cost;
- do not modify results manually.

When finished:
1. run the module's unit tests;
2. run ruff/mypy;
3. run the relevant smoke test;
4. show git diff --stat;
5. summarize decisions and risks;
6. propose the Conventional Commit message.
```

---

# 13. Scientific Go/No-Go criteria

### Gate G1 — Valid baseline
The Learning-to-Select baseline converges, the mask is mathematically correct, and exact retrieval reproduces the expected ranking.

### Gate G2 — H1 supported
There is functional selector shift between environments, observed both in retrieval metrics and in mask behavior.

### Gate G3 — H2 supported
The environment descriptor predicts selector/transfer properties above baseline, robustly across seeds/sampling.

### Gate G4 — H3 supported
The hypernetwork reduces the gap between the global selector and the per-domain selector on environments not used for meta-training.

### Gate G5 — Acceptable cost
Offline adaptation is amortizable, and the selector's online overhead does not eliminate the advantage of dimensionality reduction.

If any gate fails, record the result as evidence and do not move forward by indiscriminately increasing complexity.

---

# 14. Main risks and mitigation

**R1 — Small selector shift.** Mitigation: prove it first in M4; if small, abandon the hypernetwork.

**R2 — Descriptor learns dataset identity, not useful geometry.** Mitigation: meta-test on truly separate environments; permutation/probe controls; reduce explicit semantic features.

**R3 — Pseudo-feedback creates circularity.** Mitigation: P0 corpus-only as a control; P1 clearly labeled; never use qrels in the meta-test descriptor.

**R4 — Hyper Head generates unstable parameters.** Mitigation: residual low-rank, near-zero deltas, clipping, regularization of Δθ.

**R5 — MRL already solves the problem.** Mitigation: always compare against Qwen prefix truncation; the contribution is only valid if a query-aware subset adds value beyond MRL.

**R6 — Gain comes from supervising the selector, not from the hypernetwork.** Mitigation: compare with a global multi-domain selector with the same amount of labels and similar capacity.

**R7 — ANN masks the effect.** Mitigation: primary scientific results on exact scoring; ANN only in the deployment analysis.

---

# 15. Minimal first version to implement

Do not start with the 1024 dimension-token Transformer. The minimal version must contain:

1. Qwen3-Embedding-0.6B → normalized 1024 embeddings;
2. reproduced linear Learning-to-Select;
3. independent selectors on ≥3 environments;
4. proof of selector shift;
5. `1024 × h` descriptor with document/query marginals + simple interaction;
6. `MLP_stats(r_j)+dimension_id_embedding` as the Space Encoder;
7. linear base selector + low-rank modulation;
8. simple Hyper Head generating low-rank deltas per environment;
9. episodic training;
10. meta-test evaluation P0/P1;
11. only then contrast, redundancy, Transformer, and adaptive-k.

This order reduces the risk of building a complex architecture before proving there is signal for it to exploit.

---

# 16. Deliverables per milestone

**Milestone 1 — Reproduction:** M0–M3 code, tests, baseline report.

**Milestone 2 — Scientific premise:** M4 + selector-shift report + Go/No-Go decision.

**Milestone 3 — Space representation:** M5 + H2 probes + frozen `SpaceDescriptorV1`.

**Milestone 4 — HyperDIME prototype:** M6–M9 + meta-train checkpoint + meta-validation evaluation.

**Milestone 5 — Paper-grade evaluation:** M10–M12 + ablations + significance + efficiency + automatically generated tables.

---

# 17. Core references

- Wu, Z.; Zhang, R.; Nie, Z. **Learning to Select: Query-Aware Adaptive Dimension Selection for Dense Retrieval.** ACL 2026.
- Killingback, J.; Zeng, H.; Zamani, H. **Hypencoder: Hypernetworks for Information Retrieval.** SIGIR 2025.
- Eichholtz, A. et al. **Hypencoder Revisited: Reproducibility and Analysis of Non-Linear Scoring for First-Stage Retrieval.** SIGIR 2026.
- D'Erasmo, G. et al. **Statistical Foundations of DIME: Risk Estimation for Practical Index Selection.** EACL 2026.
- Zhang, Y. et al. **Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models.** 2025.
- Kusupati, A. et al. **Matryoshka Representation Learning.** NeurIPS 2022.

---

# 18. Expected architectural result

In the end, the system must explicitly realize three levels of adaptation:

`Qwen3-Embedding-0.6B (fixed coordinate system)`

`↓`

`Space Descriptor + Space Encoder (environment representation)`

`↓`

`Hyper Head (environment → selector parameters)`

`↓`

`Dimension Selector (query → dimension importance)`

`↓`

`Selection Policy (importance → subset/k)`

`↓`

`Query-only masked dense retrieval`

Scientific success will not be "the hypernetwork trained", but demonstrating that **measurable changes in the retrieval environment of the same Qwen explain changes in the optimal selector and can be converted — without retraining a full selector in the new domain — into useful parameters for query-aware dimension selection**.
