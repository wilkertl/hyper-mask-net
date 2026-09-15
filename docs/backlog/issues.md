# Backlog

Initial GitHub backlog derived from the implementation plan (§10 execution sequence, §11 tickets,
§16 milestones). Publish it with `python3 scripts/github/create_issues.py`. After publishing, GitHub
issues are the source of truth and this file is not kept in sync.

Format read by the script: each milestone header is followed by a `description` comment, and each
issue header (`[TICKET] Title`) is followed by a comment with `labels` and `depends`. Dependencies
must appear earlier in the file; the script turns them into *Blocked by* links.

# Milestone 1 — Reproduction
<!-- description: Contracts, data, frozen Qwen embeddings, oracle targets, the Learning-to-Select baseline and exact evaluation (plan M0-M3, Gate G1). -->

## [M0.1] Tooling: pinned dependencies, lockfile, pre-commit
<!-- labels: type:infra, module:M0, phase:A; depends: -->
**Goal:** every machine installs the same versions and runs the same checks.

**Current state:** `pyproject.toml` has lower bounds only; CI runs ruff, mypy and pytest on CPU.

**Scope**
- Pin `torch`, `transformers`, `datasets`, `numpy`, `scipy`, `scikit-learn`, `ir-measures`, `hydra-core`/`omegaconf`, `faiss-cpu`, `pytest`, `ruff`, `mypy` with a committed lockfile.
- Document CUDA and CPU PyTorch installation.
- Add `.pre-commit-config.yaml` running ruff (lint and format) and mypy.
- CI installs from the lockfile.

**Definition of Done**
- [ ] A fresh clone following the README installs the locked versions.
- [ ] `pre-commit run --all-files` passes.
- [ ] CI is green.

**Plan:** §5 M0, §9.5

## [M0.2] Contract schemas and validation
<!-- labels: type:feature, module:M0, phase:A; depends: -->
**Goal:** serializable, versioned records that every later module uses instead of ad-hoc dictionaries.

**Scope**
- `contracts/schemas.py`: `QueryRecord`, `DocumentRecord`, `QrelsRecord`, `EmbeddingArtifact`, `SpaceDescriptor`, `SelectorCheckpoint`, `EvaluationResult`, each with a `schema_version`.
- `contracts/validation.py`: explicit errors for invalid records and configs.

**Definition of Done**
- [ ] Round-trip serialization test for every schema.
- [ ] Invalid inputs fail with explicit messages.
- [ ] Schema summary added to `docs/architecture.md`.

**Plan:** §5 M0; §10 (A2 and A3 must not define parallel schemas)

## [M0.3] Run manifests, seed manager and deterministic hashing
<!-- labels: type:feature, module:M0, phase:A; depends: M0.2 -->
**Goal:** pipeline differences can never be mistaken for model differences.

**Scope**
- `contracts/manifests.py`: `RunManifest` with `git_commit`, `config_hash`, `dataset_hash`, `model_revision`, `seed`, device, dtype, and timestamps.
- One seed manager for Python, NumPy, PyTorch and CUDA determinism flags.
- SHA256 hashing of configs (key-order independent) and files.
- An artifact writer that refuses to write without a manifest.

**Definition of Done**
- [ ] The same config always hashes identically, regardless of key order.
- [ ] Writing an artifact without a manifest raises.
- [ ] Tests run on CPU.

**Plan:** §3 principle 7, §5 M0 (gate: no artifact without a manifest)

## [M0.4] Config system and diagnostic CLI
<!-- labels: type:feature, module:M0, phase:A; depends: M0.3 -->
**Goal:** every run is fully described by a versioned config.

**Scope**
- `configs/` groups from plan §4 using Hydra/OmegaConf, starting with `model/qwen3_0_6b.yaml` and one data config.
- `hyperdime.cli.diagnose`: loads and validates a config and writes a manifest without training anything.

**Definition of Done**
- [ ] The CLI writes a manifest for the sample config.
- [ ] An invalid config fails explicitly.
- [ ] Integration test for the CLI.

**Plan:** §5 M0 output

## [M1.1] DatasetBundle and BEIR loaders for SciFact and NFCorpus
<!-- labels: type:feature, module:M1, phase:A; depends: M0.2 -->
**Goal:** normalized access to environments with canonical IDs.

**Scope**
- `DatasetBundle(corpus, queries, qrels, split, instruction)` in `data/loaders.py`.
- Loaders for SciFact and NFCorpus. Document text is title and body joined by a newline, with no empty leading line.
- Per-dataset query instruction from config.
- Tiny CPU fixtures in `tests/fixtures/`.

**Definition of Done**
- [ ] Every qrel ID exists in the corpus and queries.
- [ ] Unit tests run offline on fixtures; no downloads in tests.

**Plan:** §5 M1

## [M1.2] MS MARCO loader with reproducible corpus sampling
<!-- labels: type:feature, module:M1, phase:A; depends: M1.1 -->
**Goal:** make MS MARCO usable as a third environment within the compute budget.

**Scope**
- MS MARCO loader producing a `DatasetBundle`.
- Seeded corpus sampling that keeps every judged passage of the sampled queries plus random distractors.
- Sample IDs and hash recorded in the dataset manifest.
- ADR recording the chosen sample size and embedding compute budget.

**Definition of Done**
- [ ] The same seed yields the same sample hash.
- [ ] Every sampled query keeps all of its judged passages.
- [ ] ADR merged.

**Plan:** §5 M1, §14

## [M1.3] Splits, leakage guards and dataset manifests
<!-- labels: type:feature, module:M1, phase:A; depends: M1.1, M0.3 -->
**Goal:** selector shift cannot be caused by inconsistent splits or leakage.

**Current state:** `hyperdime.data.splits.split_ids` provides a deterministic query-level split.

**Scope**
- Train, validation and test per environment; BEIR test qrels are never re-split.
- Guards that raise when test labels reach training code.
- `dataset_manifest.json` per environment with split hashes.

**Definition of Done**
- [ ] Split hashes are stable across runs.
- [ ] A test proves no test label enters training.
- [ ] One manifest per environment.

**Plan:** §5 M1, docs/protocols.md

## [M2.1] Frozen Qwen3-Embedding-0.6B encoder wrapper
<!-- labels: type:feature, module:M2, phase:A; depends: M0.2 -->
**Goal:** completely fix the 1024-dimensional coordinate system.

**Current state:** `hyperdime.embeddings.qwen.format_query` and `last_token_pool` (padding-side agnostic) exist.

**Scope**
- Single wrapper in `embeddings/qwen.py`: pinned model revision, D=1024, official pooling, query instruction, documents without formatting.
- Configurable L2 normalization, enabled in the main protocol.
- Eval mode, no gradients, explicit dtype handling.

**Definition of Done**
- [ ] Output shape `(N, 1024)` and norms close to 1 when normalized.
- [ ] Same text, config and revision give the same vector within tolerance.
- [ ] Batch-size invariance test.
- [ ] CPU/GPU tolerance test, marked to skip in CI.
- [ ] Encoder parameters never require gradients.

**Plan:** §5 M2

## [M2.2] Embedding cache artifacts and embed CLI
<!-- labels: type:feature, module:M2, phase:A; depends: M2.1, M1.3, M0.4 -->
**Goal:** embeddings are computed once and reused by every later experiment.

**Scope**
- `embeddings/cache.py`: `documents.f16.npy`, `queries.f16.npy`, ID-to-row maps, metadata with model revision and config hash.
- Cache hit and miss handling.
- `cli/embed.py` writing artifacts with manifests.

**Definition of Done**
- [ ] A cache hit skips encoding.
- [ ] Ten-document smoke test.
- [ ] Reloaded arrays are identical to the written ones.

**Plan:** §5 M2

## [M1.4] Hard-negative pools
<!-- labels: type:feature, module:M1, phase:A; depends: M1.3, M2.2 -->
**Goal:** hard negatives for oracle targets, kept apart from evaluation data.

**Current state:** `hyperdime.data.negatives.mine_hard_negatives` mines the top non-positive documents with exact search.

**Scope**
- Pools per training query (retrieve depth K, keep M), mined from training queries only.
- Optional skip of the top ranks to reduce false negatives.
- Pools stored as an artifact with a manifest.

**Definition of Done**
- [ ] Pools never contain judged positives.
- [ ] Pool construction cannot read validation or test qrels.
- [ ] Deterministic given the embeddings.

**Plan:** §5 M1

## [M12.1] Exact retrieval evaluation pipeline
<!-- labels: type:feature, module:M12, phase:B; depends: M2.2, M0.3 -->
**Goal:** trustworthy retrieval numbers without ANN effects.

**Current state:** `hyperdime.retrieval.exact.exact_search` and `hyperdime.evaluation.metrics.evaluate_run` exist.

**Scope**
- Pipeline: query embeddings and optional masks → exact top-N → nDCG@10, MRR@10/100, Recall@100 → `EvaluationResult` with per-query values and a manifest.
- Cross-check metrics against `ir-measures` on fixtures.

**Definition of Done**
- [ ] Metrics match `ir-measures` on fixtures.
- [ ] Per-query values are saved.
- [ ] A run with k=1024 equals the unmasked run.

**Plan:** §5 M12, §10 B3

## [M3.1] Oracle importance targets
<!-- labels: type:feature, module:M3, phase:B; depends: M1.4, M2.2 -->
**Goal:** per-query target distributions over dimensions.

**Current state:** `hyperdime.oracle.importance` computes the oracle for one positive and one negative per query.

**Scope**
- Aggregate several positives into `p` and hard negatives into `n` per query.
- Temperature τ from config.
- Output `TargetImportance[query_id, 1024]` via a CLI, with a manifest.

**Definition of Done**
- [ ] Unit tests against manual computation.
- [ ] Every distribution sums to 1, with no NaN.
- [ ] The CLI generates the artifact and manifest.

**Plan:** §5 M3, §11 ticket M3.1

## [M3.2] Learning-to-Select selector trainer
<!-- labels: type:feature, module:M3, phase:B; depends: M3.1 -->
**Goal:** reproduce the paper's query-only predictor.

**Current state:** `hyperdime.baselines.learning_to_select.LinearSelector` and `hyperdime.training.losses.selector_kl_loss` exist.

**Scope**
- `training/selector_trainer.py`: AdamW, early stopping on validation KL, `SelectorCheckpoint` with manifest.
- `cli/train_selector.py`.

**Definition of Done**
- [ ] The trainer overfits a synthetic 20-query set.
- [ ] Validation KL decreases on SciFact.
- [ ] A reloaded checkpoint gives identical outputs.

**Plan:** §5 M3, §10 B2

## [M3.3] Baselines: MRL prefix, random, global importance, oracle top-k
<!-- labels: type:feature, module:M3, phase:B; depends: M3.1, M12.1 -->
**Goal:** every result can be compared against the mandatory baselines.

**Scope**
- `baselines/mrl_prefix.py`: first-k coordinates.
- `baselines/random_mask.py`: seeded random-k.
- `baselines/global_selector.py`: one importance vector shared by all queries of an environment.
- Oracle top-k upper bound, labeled as non-deployable.
- Full 1024 is k=1024 of any policy.

**Definition of Done**
- [ ] Every baseline produces masks through the selection policy.
- [ ] With k=1024, every baseline equals full scoring.
- [ ] The random baseline is reproducible for a seed.

**Plan:** §5 M3 (mandatory additional baselines), risk R5

## [E0] Reproduction report and Gate G1
<!-- labels: type:experiment, gate:G1, phase:B; depends: M3.2, M3.3, M12.1 -->
**Goal:** validate embeddings, oracle and baseline selector before any hypernetwork work.

**Scope**
- In-domain on SciFact and NFCorpus: full, MRL prefix, random, global, Learning-to-Select and oracle top-k at retained ratios 0.1 to 1.0.
- Report in `experiments/00_baseline_reproduction/`, generated from result files.
- Update `docs/experiment_registry.md`.

**Definition of Done**
- [ ] Tables generated by one command.
- [ ] The trend is compared with the paper; identical numbers are not required.
- [ ] G1 decision recorded in an ADR and tagged `g1-baseline-valid`.

**Plan:** §6 E0, §13 G1

# Milestone 2 — Scientific premise
<!-- description: Test H1 (selector shift) across SciFact, NFCorpus and MS MARCO and decide Go/No-Go for the hypernetwork (plan M4, Gate G2). -->

## [M12.3] Paired significance tests and generated tables
<!-- labels: type:feature, module:M12, phase:C; depends: M12.1 -->
**Goal:** statistical support for gate decisions.

**Note:** the plan schedules this in phase F. It is pulled forward because Gate G2 needs it.

**Scope**
- `evaluation/statistics.py`: paired per-query tests (bootstrap or permutation), confidence intervals, Holm correction for hypothesis families.
- Table generation from `EvaluationResult` files into `reports/tables/`.

**Definition of Done**
- [ ] Tests on synthetic per-query data with a known effect.
- [ ] Generated tables contain no hand-entered numbers.

**Plan:** §5 M12 significance, §8.5

## [M4.1] Per-domain selectors
<!-- labels: type:experiment, module:M4, phase:C; depends: E0, M1.2 -->
**Goal:** specialized selectors for each environment.

**Scope**
- Train selectors for SciFact, NFCorpus and MS MARCO with identical hyperparameters and at least three seeds.

**Definition of Done**
- [ ] Checkpoints with manifests for every environment and seed.
- [ ] In-domain metrics per seed.

**Plan:** §5 M4, §10 C1

## [M4.2] Selector transfer matrix
<!-- labels: type:experiment, module:M4, phase:C; depends: M4.1, M12.3 -->
**Goal:** measure how selectors degrade across environments.

**Scope**
- A×B table (CSV or Parquet) with retrieval metrics and mask similarity for every train/test environment pair.
- Cross-domain versus in-domain degradation with confidence intervals.

**Definition of Done**
- [ ] No training runs during evaluation.
- [ ] The table is reproducible with one command.

**Plan:** §5 M4, §11 ticket M4.2

## [M4.3] Selector behavior analysis
<!-- labels: type:experiment, module:M4, phase:C; depends: M4.1 -->
**Goal:** show whether specialized selectors behave differently, beyond seed noise.

**Scope**
- `evaluation/selector_shift.py`: KL and Jensen-Shannon divergence between importance distributions on a shared query battery, overlap@k, Spearman correlation of dimension rankings, per-dimension selection frequency.
- Seed-to-seed variation within an environment as the noise floor.

**Definition of Done**
- [ ] Differences reported relative to the seed noise floor.
- [ ] Weight-space distances are not used as evidence.

**Plan:** §5 M4

## [G2] Go/No-Go decision for H1 (selector shift)
<!-- labels: type:decision, gate:G2, status:needs-decision, phase:C; depends: M4.2, M4.3 -->
**Goal:** decide whether the hypernetwork is needed.

**Scope**
- ADR presenting the evidence: behavior differences above the noise floor, and retrieval degradation under transfer.

**Definition of Done**
- [ ] ADR merged and `master` tagged `g2-selector-shift`.
- [ ] If No-Go: close Milestone 3 and 4 issues as not planned and open an issue for a universal selector.

**Plan:** §5 M4 decision, §13 G2, risk R1

# Milestone 3 — Space representation
<!-- description: Build SpaceDescriptorV1 and test H2 with probes before any hypernetwork is trained (plan M5, Gate G3). -->

## [M5.1] Document marginal descriptor (Group A)
<!-- labels: type:feature, module:M5, phase:D; depends: G2 -->
**Goal:** per-dimension corpus statistics with preserved dimension identity.

**Scope**
- N×1024 embeddings → 1024×h matrix plus a feature-name schema.
- Features: mean, std, mean_abs, RMS, robust min/max, quantiles q05, q10, q25, q50, q75, q90, q95, optional skewness and kurtosis.

**Definition of Done**
- [ ] Invariant to document row permutation.
- [ ] Stable under corpus subsampling (tolerance test).
- [ ] Permuting dimensions permutes descriptor rows.
- [ ] No qrels used.

**Plan:** §5 M5 Group A, §11 ticket M5.1

## [M5.2] Query marginal and interaction statistics (Groups B and C)
<!-- labels: type:feature, module:M5, phase:D; depends: M5.1 -->
**Goal:** add signal from the query distribution and the scoring mechanism.

**Scope**
- Group B: the Group A features on unlabeled calibration queries.
- Group C: statistics of `q_j * d_j`, with separate channels for random pairs and top-retrieved pairs.
- Features tagged with the lowest protocol that may use them.

**Definition of Done**
- [ ] No labels read.
- [ ] Same permutation and stability tests as M5.1.

**Plan:** §5 M5 Groups B and C

## [M5.3] Contrast and redundancy statistics and normalization (Groups D and E)
<!-- labels: type:feature, module:M5, phase:D; depends: M5.2 -->
**Goal:** complete the descriptor and normalize it across environments without leakage.

**Scope**
- Group D: supervised `q_j (p_j - n_j)` for meta-training, and a pseudo-feedback variant for new environments.
- Group E: mean and max absolute correlation, top-r correlation energy, principal-component participation. The full covariance is saved only as a diagnostic artifact.
- Cross-environment normalization fit on meta-train environments only.

**Definition of Done**
- [ ] Normalization refuses environments outside meta-train.
- [ ] A test proves P0 and P1 descriptor builders cannot read qrels.

**Plan:** §5 M5 Groups D and E, docs/protocols.md

## [M5.4] H2 probes and permutation tests (Gate G3)
<!-- labels: type:experiment, module:M5, gate:G3, phase:D; depends: M5.3, M4.3 -->
**Goal:** test whether descriptors predict selector properties, before any hypernetwork exists.

**Scope**
- Relate descriptor similarity to functional selector distance from M4 with simple regressions, CCA or probes.
- Permutation tests and trivial baselines; robustness across seeds and sample sizes.

**Definition of Done**
- [ ] Report generated from result files.
- [ ] G3 ADR merged and `master` tagged `g3-space-signal`.
- [ ] If no feature group beats the trivial baseline, an issue is opened to iterate on the descriptor.

**Plan:** §5 M5 H2 test, §13 G3, risk R2

## [M5.5] Freeze SpaceDescriptorV1
<!-- labels: type:feature, module:M5, phase:D; depends: M5.4 -->
**Goal:** a stable input contract for the Space Encoder.

**Scope**
- Freeze the feature list and schema version.
- Regression fixture for the descriptor.

**Definition of Done**
- [ ] Schema version bump policy documented.
- [ ] Regression test on the fixture.

**Plan:** §10 D5

# Milestone 4 — HyperDIME prototype
<!-- description: Low-rank selector, Space Encoder, Hyper Head and episodic meta-training, evaluated on meta-validation (plan M6-M9). -->

## [M7.1] Low-rank modulated selector
<!-- labels: type:feature, module:M7, phase:E; depends: M5.5 -->
**Goal:** a target selector small enough for the Hyper Head to generate.

**Scope**
- `W_T = W_0 + A_T B_T^T` and `b_T = b_0 + Δb_T`, with rank r in {4, 8, 16, 32}.
- Optional diagonal or FiLM modulation variant for a first proof.

**Definition of Done**
- [ ] Zero delta equals the base selector.
- [ ] Autograd validated.
- [ ] Configurable rank; deterministic checkpoint.

**Plan:** §5 M7, §11 ticket M7.1

## [M6.1] Space Encoder V1
<!-- labels: type:feature, module:M6, phase:E; depends: M5.5 -->
**Goal:** a learned environment representation that preserves dimension identity.

**Scope**
- `t_j = MLP_stats(r_j) + p_j`, with a learned dimension-ID embedding `p_j`, giving `Z_T` of shape 1024×h_z.
- No inter-dimension Transformer until an ablation shows V1 is limiting.

**Definition of Done**
- [ ] Permuting dimensions changes the output.
- [ ] Gradients flow to all parameters.
- [ ] Fixed output shape; memory use measured.

**Plan:** §5 M6

## [M8.1] Hyper Head V1
<!-- labels: type:feature, module:M8, phase:E; depends: M6.1, M7.1 -->
**Goal:** map the environment representation to selector modulations.

**Scope**
- `Z_T → A_T`, `Z_T → B_T`, `Pool(Z_T) → Δb_T` with separate heads.
- Near-zero initialization of the final stage and explicit scale control, so that θ_T starts close to θ_0.
- Not the Hypencoder per-query hyperhead.

**Definition of Done**
- [ ] Generated parameters have the expected shapes.
- [ ] At initialization, the generated selector is close to the base selector.
- [ ] Gradient test.

**Plan:** §5 M8, §11 ticket M8.1, risk R4

## [M9.1] Episodic meta-trainer
<!-- labels: type:feature, module:M9, phase:E; depends: M8.1 -->
**Goal:** train Space Encoder, Hyper Head and base selector across environments.

**Scope**
- Episodes: pick an environment, generate θ_T from its descriptor, sample a query batch, KL loss to oracle targets, backpropagate through the generated selector to the Hyper Head and Space Encoder.
- Rank and delta regularizers behind config flags, off by default.
- Environment-level meta-train, meta-validation and meta-test split in config.

**Definition of Done**
- [ ] Batches alternate environments.
- [ ] Loading a meta-test environment in the trainer raises through config validation.
- [ ] Checkpoint of G_ψ, H_φ and θ_0 with manifest.
- [ ] End-to-end CPU integration test with 50 documents and 10 queries.

**Plan:** §5 M9, §8.3, §11 ticket M9.1

## [M9.2] Meta-validation: HyperDIME versus global selector
<!-- labels: type:experiment, module:M9, phase:E; depends: M9.1 -->
**Goal:** decide whether the prototype earns a meta-test evaluation.

**Scope**
- Compare against a global multi-domain selector trained with the same labels and similar capacity.

**Definition of Done**
- [ ] Generated report with confidence intervals.
- [ ] HyperDIME matches or beats the global selector on meta-validation, or an ADR records that it does not.

**Plan:** §5 M9 gate, risk R6

# Milestone 5 — Paper-grade evaluation
<!-- description: Inference protocols, selection policies, the main meta-test result, ablations, sensitivity and efficiency (plan M10-M12, Gates G4 and G5). -->

## [M10.1] Inference protocols P0, P1 and P2
<!-- labels: type:feature, module:M10, phase:F; depends: M9.2 -->
**Goal:** adaptation that can only use the inputs its protocol allows.

**Scope**
- Adaptation entry points per protocol, as defined in docs/protocols.md.
- θ_T persisted per environment.
- Result files carry their protocol label.

**Definition of Done**
- [ ] Property test: no P0 or P1 path can access qrels.
- [ ] Table generation refuses results without a protocol label.

**Plan:** §5 M10

## [M11.1] Selection policies
<!-- labels: type:feature, module:M11, phase:F; depends: M9.2 -->
**Goal:** separate importance prediction from how many dimensions to keep.

**Current state:** fixed top-k and fixed ratio exist in `hyperdime.selection.topk`.

**Scope**
- Policy interface.
- RDIME-inspired adaptive k, reported as a heuristic unless the RDIME assumptions hold for learned scores.
- Optional budget-aware policy.

**Definition of Done**
- [ ] The fixed-k policy matches `top_k_mask`.
- [ ] Adaptive policy validated on fixtures.
- [ ] RDIME compatibility documented.

**Plan:** §5 M11

## [E3] Main result on meta-test environments (Gate G4)
<!-- labels: type:experiment, gate:G4, phase:F; depends: M10.1, M11.1 -->
**Goal:** test H3 on environments never used for meta-training.

**Scope**
- Compare full 1024, MRL prefix, per-domain Learning-to-Select, global multi-domain selector, source-trained transferred selector, HyperDIME P0, HyperDIME P1, and the oracle or per-domain upper bound.
- P0 and P1 reported in separate, labeled tables.

**Definition of Done**
- [ ] Generated tables with confidence intervals.
- [ ] G4 ADR merged and `master` tagged `g4-hypernetwork`.

**Plan:** §6 E3, §13 G4

## [E4/E5] Ablations and sensitivity
<!-- labels: type:experiment, phase:F; depends: E3 -->
**Goal:** explain which choices matter and how robust the result is.

**Scope**
- Ablation matrix from plan §7: remove one descriptor group at a time, dimension-ID embeddings, inter-dimension Transformer, rank, number of calibration queries.
- Sensitivity: seeds, corpus and calibration sample sizes, τ, hard-negative K and M, rank r, selection k.

**Definition of Done**
- [ ] All tables generated from result files.
- [ ] Registry updated.

**Plan:** §6 E4 and E5, §7

## [M12.2] Efficiency and deployment analysis (Gate G5)
<!-- labels: type:experiment, module:M12, gate:G5, phase:F; depends: M10.1 -->
**Goal:** show whether the method is worth its cost.

**Scope**
- Offline adaptation cost per environment.
- Online latency split into Qwen encoding, selector, masking and scoring, and search, with warm-up and repeated passes.
- Memory, estimated FLOPs and size of θ_T.
- FAISS backend used only for deployment analysis.

**Definition of Done**
- [ ] Generated efficiency report.
- [ ] G5 ADR merged and `master` tagged `g5-cost`.

**Plan:** §5 M12, §6 E6, §13 G5, risk R7
