# HyperDIME: a corpus-conditioned selector for unseen corpora

Consolidated proposal from the multi-agent discussion. Status: **proposed**, not ratified.
Nothing here has been run. Every threshold is prospective and must be registered in the ADRs of §10
before any new outcome is opened.

- **Discussion:** `round0-seed.md`, `round1-*`, `round2-*`, `round3-synthesis.md` (Astra),
  `round4-opus-e.md`, `round4-opus-m.md`.
- **Independent council:** `council-astra.md`, `council-opus.md`, `council-neon.md` (missing: backend unreachable).
- **Base text:** the round-3 synthesis, amended. Amendment labels refer to the round-4 files:
  B1–B3 and N1–N8 are Opus-E's, A1–A11 and S1–S6 are Opus-M's, C-labels are council items.
  Section §9 lists every decision the orchestrator made where reviewers disagreed.

## 1. Decision summary

1. **Question.** Can an unseen corpus's unlabeled statistics predict a correction to a frozen,
   corpus-balanced selector that improves masked exact retrieval? The question is not whether a
   large hypernetwork can be trained. With the families available, only a very small generated
   output is identifiable.
2. **Comparator (G+V).** A corpus-balanced global S4 selector, trained without the recipient's
   family, using the recipient's own document variance in the geometric prior and shared per-budget
   prior weights. The legacy E1 "global selector" is not a valid comparator: it is a plain union,
   93.6% MS MARCO, that includes the recipient's own training queries (`train_global.py:54-71`).
3. **First generated output (HyperPrior).** An offset and a budget slope on the prior weights,
   predicted from p ≤ 3 corpus-level features, where p is tied to the number of training families.
   The offset-only model is a nested ablation chosen inside each fold.
4. **Ladder.** G+V → HyperPrior → tied per-dimension bias. Rank-four is **deferred** (S1): its own
   gate needs ≥ 15 training families and the pool is about 4–6.
5. **First step.** An oracle ladder on SciFact, NFCorpus and MS MARCO using
   `train`/`validation` splits only, runnable once the multi-corpus S4 trainer and the ladder driver
   exist (§10). It measures how much headroom survives a fair G+V, and which
   output type could capture it.
6. **Expected outcome.** An estimation-level study (A7). A confirmatory superiority claim needs a
   development family SD ≤ ~0.007–0.009, which current evidence does not suggest.

## 2. Evidence base (verified)

Results on the three development corpora are exploratory: S1–S4, E2 and P3 were shaped on them.

| Observation | Value and source | Consequence |
|---|---|---|
| Legacy shift payoff at k = 307 | In-domain minus union-global: SciFact +0.1116, NFCorpus +0.0171, MS MARCO −0.0019 (`reports/E1/tables/payoff.csv`) | The observed specialist-minus-union contrast at k = 307. Changing recipient inclusion, family weights, loss and prior can move it in either direction. Held-family headroom against G+V is unmeasured. |
| Legacy union composition | 45,000 of 48,059 training queries are MS MARCO (93.6%). L2S-KL, no variance prior. Trains on the recipient's training queries and selects its epoch count by retrieval on every corpus's project `validation` split (`src/hyperdime/cli/train_global.py:54-85`; counts in `reports/P3/README.md:44-48`) | Legacy → G+V changes objective, prior, balancing and recipient exclusion at once. Step one isolates them (N3). |
| Prior share at k = 307 (S4, budget track) | MS MARCO 0.9512, NFCorpus 0.4281, SciFact 0.3580. NFCorpus falls from 0.5008 at k = 102 to 0 at k = 716 (`reports/S4/tables/prior_share.csv`) | Differences are about 2.7×, not orders of magnitude, and the **budget profile** differs. That motivates offset plus slope. |
| MS MARCO prior weights stay near initialization | S3 kept weight 1.0000 in every seed (ADR-0011). S4 kept steps 1000–10000 by full-corpus validation, but the rerank-best step is 0 in all six MS MARCO rows, and kept weights stay at 0.64–1.00 across both tracks (budget track 0.6687–0.9979); S3 retained 1.0000 but ended near 0.016 in all three MS MARCO fits (`reports/S3/tables/weights.csv:2-4`) (`reports/S4/tables/kept_steps.csv:2-7`, `reports/S4/tables/weights.csv:1-7`) | The cross-corpus spread in prior weight is partly an optimizer and initialization artifact, so it is weak evidence of a corpus property (council-opus). |
| Prior utility sits at small budgets | Removing the prior: MS MARCO −0.20 at k = 102, \|Δ\| ≤ 0.0016 at k ≥ 716; NFCorpus −0.059 at 102, \|Δ\| ≤ 0.0017 from k = 409 (ADR-0012 ablation) | High-budget weights are weakly identified. Fit to the F2 oracle in utility space, not to nine weights (B2). |
| Per-budget weights help | S4 vs aligned S3: NFCorpus +0.0111, SciFact +0.0094, MS MARCO +0.0001 (ADR-0012) | One shared weight is too restrictive. Around the mean curve, an offset explains 90–96% of prior-share deviation, offset + slope 98–100% (round4-opus-e §3.1). |
| Per-dimension geometry vs selector shift (G3) | 0 of 18 shift cells pass. SciFact variance shift 0.4376 vs B1 0.4354; the direct profile probe reaches 0.54/0.68/0.79 (ADR-0013, `reports/E2`) | Per-dimension corrections lack support. G3 with 2 training environments is also weak, so a fresh rerun is required before any coordinate rung. |
| What masks must capture | Positive-only masks rank below L2S; the contrastive teacher `(p̄ − n̄) ⊙ q` is closest (P3-bench, P3-mask-ranking) | Corpus dependence enters through competing documents. PRF-DIME and Eclipse are mandatory label-free rivals. |
| Selector objective is saturated | No target, loss, per-budget, capacity, direct-ranking or init effect once lr is tuned equally (ADR-0020, post-hoc A1) | The remaining lever is cross-corpus conditioning or data, not a new per-corpus objective. |

## 3. Architecture

Let q be the normalized frozen Qwen3-Embedding-0.6B query vector (D = 1024) and kᵢ = ⌊iD/10⌋ for
i = 1…9. From the unchanged document matrix of corpus t compute the population variance vₜ ∈ R^D,
and define the prior pₜ(q) = log(q² ⊙ vₜ + 10⁻¹²). Every rung uses the S4 logits

  **sₜᵢ(q) = W q + b + aₜᵢ · pₜ(q) + cₜ**

with absent terms zero. The mask is the top kᵢ logits (ties by coordinate id). The score is the exact
inner product (q ⊙ mask)ᵀd over the whole unchanged corpus, in float64, with no renormalization of the
masked query and fixed document-id tie order. At k = 1024 every arm must reproduce full-vector
rankings exactly.

| Rung | Output and parameters | Learned parameters added | Cached per corpus |
|---|---|---:|---:|
| **G+V** | W ∈ R^{D×D}, b ∈ R^D, a⁰ ∈ R⁹ ≥ 0; aₜ = a⁰ | 1,049,609 | 0 |
| **HyperPrior** | tᵢ = (i − 5)/4; aₜᵢ = max(0, a⁰ᵢ + r_W·(u + tᵢ v)), with u = u₀ + c_uᵀφₜ and v = v₀ + c_vᵀφₜ. φₜ ∈ R^p is standardized, with p = min(3, ⌊(K_train − 1)/3⌋) fixed at Gate A (A4). Offset-only ablation: v ≡ v₀ = 0 | 2p + 2 | 2 scalars |
| **Tied bias** (conditional) | cₜ = Xₜ w, where Xₜ ∈ R^{D×h} are standardized per-dimension departure rows (P0: h = 18), folded into b offline | h | D |
| ~~Rank four~~ | **Deferred** (S1). Reopen only with ≥ 15 audited training families under a new ADR. | — | — |

- **Feature order** (registered, A4): log participation ratio log[(tr Σ)²/tr(Σ²)], then log N, then the
  anisotropy λ_max(Σ)/tr Σ, where Σ is the document covariance. A geometry feature comes first, so that
  HyperPrior at p = 1 is not identical to the size-only rival, which runs through the identical
  Stage B pipeline with φ = log N. Features are computed on a fixed-size seeded document subsample
  (the pool's smallest N), so the participation ratio does not inherit finite-N bias. Under P1, the
  mean top-one full-vector cosine on calibration queries takes the **first** slot; at p < 3 the P1
  track may instead be the query-centred prior alone, chosen at Gate A. Each
  feature's **predicted sign** for u and v is
  registered at Gate A (council-opus).
- **Scale** (N1): r_W is the median, over source training queries, of SD(Wq + b)/SD(pₜ) for the fold's
  frozen base, with family-balanced source weights and a registered exclusion rule for zero-spread
  queries. It is reported per fold and makes u and v comparable across fold bases (council-astra).
  If r_W is zero or undefined the rung closes for that fold. G+V itself keeps the unscaled prior.
- **Initialization:** u₀ = v₀ = 0 and c_u = c_v = w = 0, so before fitting every rung equals G+V exactly. The projection
  admits an exact zero weight.
- **Online cost:** identical to G+V for HyperPrior and for the folded bias. Offline cost: streamed
  moments O(ND) and covariance O(ND²), reported separately (G5).

## 4. Training

1. **Stage A, the base.** For each outer leave-one-family-out fold, train the S4 selector on the source
   families only. Sample a family uniformly, then a query within it, then a budget. Use same-corpus
   batches, each source's own variance, the S4 relaxed top-k multi-positive masked retrieval loss and
   temperature schedule, and select checkpoints by hard-mask exact validation retrieval on source
   families. Freeze the base. (Drop the family- vs corpus-uniform sensitivity unless a multi-corpus
   family exists, S5.)
   **Equal effort (ADR-0020, council-r2-opus):** every trained arm (G+V, every base arm, the
   specialist, the Fb and Xw oracles) selects its learning rate on the same grid
   {3e-5, 1e-4, 3e-4, 1e-3} using fit and checkpoint partitions only, and trains three seeds averaged
   per query. Resource plan: the learning rate is chosen once per outer fold and reused for inner
   refits; inner refits use one seed; MS MARCO checkpoint selection is capped at about 2,000 queries.
   Step one alone needs 25+ base fits, so this plan is frozen with the step-one resource plan.
2. **Stage B, the map.** Inside each outer pool, rotate an inner pseudo-target family. Refit Stage A
   without it, compute its descriptor from permitted inputs, and fit **both** attainable oracles on
   identical fit queries: F1 with v fixed at 0, and F2 with u and v free (an F2 optimum's u generally
   differs from the F1 optimum). On a flat utility surface, take the point nearest the origin within
   0.001 nDCG@10 of the maximum. Express the targets in r_W units. Fit each restricted map with its own
   matched intercept(s) by equal-family ridge regression, using one preregistered λ grid (or a fixed λ)
   for every p, identical bounds, seeds and regret partitions (B2, A4, N2). The inner pseudo-target's
   labels never enter target construction, base selection, scaling or λ selection for other folds.
   Sensitivity analysis: fit directly by inner held-family retrieval utility.
3. **Offset vs offset + slope** is chosen inside each outer fold. Offset-only is kept when the slope
   adds < 0.005 inner held-family gain. That choice is part of the tested procedure (A2).
4. **Tied bias** (only if unlocked): episodic masked retrieval loss on w with the prior frozen, a penalty
   on generated-logit RMS, and stopping chosen inside source folds.
5. **Deployment to a recipient:** refit the base on all source families, compute φ and X from the
   recipient's documents only, and apply the map with no gradients and no recipient labels.

## 5. Gates

Δ is the paired difference in the family endpoint of §6. A failed screen keeps the last accepted rung.

**Step one: development oracle ladder on SciFact, NFCorpus and MS MARCO** (after the multi-corpus trainer and driver of §10).

- **Data (B1, council-astra).** Fix the score partition and resource cap before any fit. Use only the
  project's `train` and `validation` splits. `validation` is a seeded 10%
  of official train queries (`src/hyperdime/data/prepare.py:107-111`). Fit and checkpoint partitions
  are hashed from `train`; the untouched score partition is `validation`, plus a hash-held part of
  `train` if needed, stratified by positive count and identical across arms. **Never read `test`.** For
  MS MARCO that split is the official dev set (`reports/E1/README.md:25`). Score IDs are excluded from
  the training and checkpointing of **every** arm, including a rerun of the recipient-inclusive legacy
  union and any upstream teacher or temperature selection; the stock union consumed `validation`
  (`train_global.py:54-85`). Archive ID disjointness per arm. `validation` is held out for this
  experiment but was used for development before, so step one stays exploratory.
- **Base arms (N3, in this order):** legacy union (rerun clean); LOCO union with the same objective;
  LOCO balanced with the same objective; balanced S4 objective **without** prior; S4 objective with
  pooled variance; S4 objective with target variance (= G+V).
  Same update count and prior where an arm isolates one factor.
- **References:** a same-recipe in-domain specialist (headroom H = specialist − G+V), plus the parameter
  -free rivals full-1024, MRL prefix, `variance`, PRF-DIME and Eclipse, and the **query-centred P1
  prior** log((q − μ_Q)² ⊙ vₜ + 10⁻¹²) (council-opus).
- **Oracles** (W, b frozen; fitted on recipient fit labels, scored on the untouched partition):
  - F1, one offset;
  - F2, offset + slope;
  - F9, nine independent nonnegative weights;
  - Fb, F9 plus one free D-vector bias;
  - attainable Xw, the tied bias from document rows.

  Report Rⱼ = mean(Fⱼ − G+V)/H, computed on equal-family mean gains, only where H ≥ 0.010 with a
  lower bound > 0; otherwise report absolute gains (one rule throughout). Report Fb − F9 to isolate the
  free bias. Rerun at equal fit-label counts.
- **Compute (N7).** Freeze a resource plan first, capping the MS MARCO untouched partition at about
  2,000 queries. If MS MARCO does not fit, run SciFact and NFCorpus first and record MS MARCO as
  *pending*, not failed. SciFact's untouched partition is about 80–135 queries (N8); its R thresholds
  are resolvable only because H is likely ≈ 0.1.
- **Descriptive:** MS MARCO topical shards, specialist − G+V per shard. This separates query-
  distribution shift from corpus shift.
- **No result of step one unlocks a confirmatory claim.** Opus-E's predictions (round4-opus-e §5,
  E1–E12) are registered. The central ones:
  - SciFact H between +0.08 and +0.14;
  - R_F9 < 0.25;
  - R_Fb ≥ 0.50;
  - attainable Xw < 0.25.

  R_F9 ≥ 0.50 would overturn them.

| Gate | Data | Criterion (registered) | Action |
|---|---|---|---|
| **A. Inventory** | before any new outcome | Audit split provenance and family membership. Overlap ≥ 1% of the smaller collection merges families; Wikipedia derivatives and StackExchange count as one family each. Record task type (claim, argument, duplicate question, QA, …), language, judgment density (from published provenance, or after scoring, with a missing category) and Qwen3 training
exposure as **known / unknown** (Qwen3's data list ends in "etc."). Hash the family graph, roles,
encoder revision, instructions, split IDs, seeds, score caps and the allowed reads of every step
(fitting, mining, checkpoints, normalization, P1, scoring). Require ≥ 7 label-bearing families: ≥ 5 meta-train after reserving 2 for meta-validation, whose training splits are excluded from Stages A/B, mining and descriptor moments (A10). Fix p and the feature signs. Publish the expected K_train and K_test after the overlap audit and licence check. A candidate whose inclusion would merge existing families (e.g. BioASQ/PubMed with NFCorpus or SciFact; SCIDOCS with SciFact; unverified, to be checked here) is excluded, not merged. ≥ 2 meta-train families must be not-known-exposed. | Fewer families: pilot estimates only. |
| **B. Payoff census** | meta-train, family-held-out base | Mean H ≥ 0.010, ≥ 2 families with H_f ≥ 0.005, and every delete-one mean ≥ 0.005. Report the target- vs pooled-variance ablation. | **Proceed only on a pass** (A3). On a failure keep G+V and stop escalation in this run, recording the failure as ***concentrated*** if mean H ≥ 0.010 but a delete-one mean is < 0.005; *precise* if the family-level one-sided 95% t upper bound of mean H is < 0.010; *imprecise* otherwise. Because LOFO folds share fitted components, that bound is descriptive, not a calibrated test. Reopening needs new families and a new ADR. |
| **C. Attainability** | meta-train | Attainable oracle increment ≥ 0.005 over the frozen lower rung with identical prior coefficients, on ≥ 2 families with H_f ≥ 0.010 (R_j reported per family). For the bias, also free-bias incremental recovery (Fb − F9) ≥ 0.25 of H where H is measurable. Stability: fit each half's adjustment on that half and score it on the other, in both directions; held-utility regret ≤ 0.002 with a positive gain in both halves (ICC dropped, S2). | Below threshold, close that rung. |
| **D. HyperPrior** | meta-train outer folds | One procedure, with the offset/slope choice inside folds (A2). Mean gain ≥ 0.005 over G+V; ≥ 40% of the attainable F2 (or F1) oracle gain; no family whose query-level one-sided 95% upper bound of Δ_f vs G+V is below −0.005 (registered operating characteristic: about 6–8% false kill for a concentrated +0.02–0.03 method, about 82% detection of a −0.020 family at K = 5; council-r2-opus); **positive paired mean gains over the matched global (§6 comparator 2), size-only and donor controls**, with the gain over the matched global also positive on at least half the families with H_f ≥ 0.005 (a tie with size-only selects the simpler size rule). Sign-flip and descriptor-permutation p-values are **reported, never binding, in this run**, because outer folds share K − 2 training families; report them with their resolution (1/2^K for signs, 1/K! or coarser for descriptor permutations within registered exchangeability blocks, with every supervised step refitted, or state that the shuffle is uninformative). | Pass: a prior checkpoint is licensed. Fail: keep G+V. A failure here never licenses a larger head. |
| **E. G3 rerun** | three preassigned fresh meta-train families | ADR-0013 replayed unchanged; fewer than three fresh families means *pending* (with the historical three among five meta-train families, only two are fresh). A P0 bias needs a P0 pass. The tied-bias retrieval diagnostic (statistic, folds, threshold) is registered before E runs, and runs only if C licensed the bias (N5). | A G3 failure closes the bias rung in this run. A discordant positive bias diagnostic motivates a new ADR on unused families, with no escalation here. |
| **F. Tied bias** | meta-train outer folds | C and E pass; ≥ 0.005 incremental held-family gain over the locked lower rung; positive gains over a matched fit on a fixed nonzero source template and over the donor; the same no-loss rule as D. Same descriptive p-value rule as D. | Fail: keep the lower rung. |
| **G. Meta-validation** | 2 untouched families, one pass | The no-loss rule of D, applied to each family. The mean increment is reported without a threshold (A6). | Fail: stop before test unblinding. |
| **H. Locked meta-test, G4/G5** | the audited meta-test families (8–12 are needed for §6's confirmatory branch; the nominated pool supports about 4–7, so estimation is the expected case) | §6. G5 separately: a measured online saving and a break-even volume below the registered workload, against PRF-DIME, Eclipse and offline pseudo-query tuning. | Report at the registered claim level. |

## 6. Evaluation

- **Pool.** SciFact, NFCorpus and MS MARCO are development data only, permanently.
  - Candidate label-bearing additions (subject to audit): FiQA, the NQ/HotpotQA/FEVER Wikipedia family,
    Quora, LoTTE dev, BioASQ.
  - Candidate evaluation-only families: ArguAna/Touché, SCIDOCS, TREC-COVID, Robust04/TREC-NEWS,
    Signal-1M, CQADupStack.
  - These are nominations, not counts.
  - Official test qrels are used only in the single locked evaluation.
  - No reference is ever fitted on any label of a meta-test corpus.
  - P2 uses a separate family pool.
- **Estimand.** For each query and seed, the paired difference in exact linear-gain nDCG@10, averaged
  over the nine budgets, then over seeds, queries, corpora within a family, and families with equal
  weight. Secondary outcomes: k = 307 (E1 continuity), per-budget curves, MRR@10/100, Recall@100, prior
  shares and mask overlap.
- **Protocol.** P0 is primary, because every recipient can supply its documents. P1 is compared on
  identical eligible corpora. P1 calibration queries are exactly 500 IDs drawn from an **independently
  supplied unlabeled query roster**, fixed at Gate A and never scored; at least 200 others are
  reserved for evaluation, and P0 and P1 are scored on the same frozen complement. The project's test
  query list cannot serve as that roster, because it keeps only queries with judged positives
  (`src/hyperdime/data/prepare.py:95-111`); a collection whose roster can only be recovered from qrels
  is P1-ineligible (A8 as amended by council-astra, to be ratified by the protocol ADR). P1 deployment
  is charged its calibration encoding and search cost. If P1 has K ≈ 0 at meta-test, P1 evidence is meta-train only and exploratory (N5).
- **Comparators, in a fixed testing order:**
  1. **G+V**, primary.
  2. **Matched global:** the same clipped offset + slope family with (u₀, v₀) fitted to the
     family-balanced mean oracle and c = 0. It isolates the descriptor slopes (A5, council-astra
     "refitted global constant").
  3. **Donor:** the mean outcome over all different-family source descriptors, each keeping the
     recipient's variance.

  Testing stops at the first failure. Rivals reported with Holm adjustment within their own family:
  - size-only (log N) and nearest-corpus;
  - full 1024, MRL prefix, `variance`;
  - balanced global without variance;
  - pooled-variance G+V;
  - PRF-DIME and Eclipse (charged their retrieval passes);
  - offline pseudo-query tuning (charged to adaptation);
  - the query-centred P1 prior.

  The feature-augmented global is a parity unit test, not a comparator (S3).
- **Statistics.** Exact family sign-flip, one-sided α = 0.05, with location intervals obtained by test
  inversion. Also report:
  - per-family effects;
  - **m of K positive families**;
  - delete-one means;
  - a paired hierarchical bootstrap;
  - a REML random-effects **prediction interval for a new family**, marked unstable at small K.

  Query-level tests inside corpora are diagnostics, not degrees of freedom.
- **Power (A1, A7, council-opus).**
  - **Exact sign-flip power for +0.010 at family SD 0.010:** 0.79, 0.90 and 0.94 at K = 8, 10 and 12.
    These are simulation estimates under normal family effects (`round4-opus-m.md:63-85`). A +0.005
    effect needs about 25 families by the one-sided normal approximation, before finite-sample
    penalties.
  - **Concentrated effects:** with means of 0.00875–0.010 carried by 20–25% of the families, power falls
    to 0.20–0.47 (`council-opus.md:98-108`), so this scenario is simulated with its own between-family
    variance and registered.
  - **Size limits:** null rejection ≤ 0.055 is required only under **symmetric** nulls (normal and
    symmetric heavy-tailed). Left-skewed nulls (size 0.084–0.137 at K = 8–12) and a one-family stress
    test are simulated and reported, but are not pass/fail criteria.
  - **Primary-test requirement:** ≥ 80% power at +0.010 under the conservative development SD **and**,
    whenever the meta-train LOFO effects are concentrated (registered rule: the top family carries
    > 50% of the summed positive LOFO gain), under the registered concentrated scenario (25% of
    families × 0.035), using ≥ 10,000 draws. Otherwise the meta-test is estimation-level.
  - **Predicted-mover readout** (key secondary): before unblinding, families whose predicted offset
    |ĉ_uᵀφ̃| exceeds the smallest offset whose oracle gain was ≥ 0.005 on meta-train are labelled
    movers. Movers' effects and non-movers' harm are reported separately.
  - **Conservative SD:** defined before outcomes, with its uncertainty and a sensitivity analysis for
    correlated development folds. The 0.007–0.009 threshold in §1 is a planning example that depends on
    development K and the upper-limit convention.
  - **Attribution power:** report separately the probability that Gate G and the primary test pass,
    and that all three ordered contrasts pass, under a stated matched-global and donor alternative
    (two contrasts already had joint success ≈ 0.40 at K = 10 with G+V +0.010 and donor +0.005,
    `round4-opus-m.md:114-120`). Descriptor attribution is confirmatory only if that joint power is
    prospectively adequate **and** the whole sequence passes; otherwise it is exploratory.
  - **Additional report:** power at the locked candidate's own leave-one-family-out estimate.
- **Claim level (B3 + A11, wording fixed in advance).**
  - **Below 8 families, or if the power simulation fails:** estimates, m of K and the prediction
    interval only.
  - **At 8–14 families:** a passing test "supports a positive mean contrast over G+V on the audited
    panel, under the symmetric-family assumption". Extending it to the eligibility frame requires a
    documented sampling argument registered at Gate A. It is always stated with m of K and the
    new-family prediction interval.
  - **Never used below 15 families:** "generalizes to new corpora", a minimum-effect claim,
    non-inferiority, or "safe to deploy".
  - **At ≥ 15 families:** transport also requires a sampling frame registered at Gate A.
- **Harm guard.** Material harm is Δ_f < −0.010 against G+V, tested per family with query resampling
  inside each corpus and Holm adjustment across families. Significant harm blocks any deployment
  recommendation. A deployment recommendation needs the harm guard to pass as well as the primary test.
  Wherever a family's simultaneous interval crosses −0.010, its harm is reported as unresolved.
- **Confounds, reported in every table:**
  - task type;
  - Qwen3 exposure (known/unknown);
  - log corpus size;
  - judgment density;
  - distance to the nearest source family.

  A gain that only the size-only rival or a single task type explains is reported as such.
  Pre-registered sensitivity: the meta-test effect on families whose task type is absent from
  meta-train (e.g. argument retrieval), against a 1-NN-task rival (the matched-global map refitted on
  meta-train families sharing the recipient's task type).

## 7. Kill criteria and expected outcomes

| Result | Decision |
|---|---|
| Gate B fails precisely (upper bound of H < 0.010) | Finish with G+V. The hypernetwork program ends in this run. |
| Gate B fails by concentration (most headroom in one family) | Publish per-family headroom and the headroom family's oracle ladder; ship G+V. A corpus map cannot be learned from one family. This is the most likely failure. |
| Oracles help but the cross-fitted map, matched global or donor test fails | Publish the attainable oracle gain and the prediction failure separately. |
| R_F9 < 0.25 while R_Fb ≥ 0.50 on SciFact | The gain lives in per-dimension corrections. HyperPrior is at best marginal; the bias rung depends on attainable Xw and G3. |
| Attainable Xw increment < 0.005 over the frozen prior rung, although Fb − F9 recovers ≥ 0.25 of H | A useful bias exists, but document-row features cannot express it. Close the bias rung at Gate C. |
| R_F9 ≥ 0.50 | Overturns the F9 prediction. HyperPrior advances only if its attainable F1/F2 oracle and its held-family predictor pass Gates C and D (nine free weights can fit a pattern no offset or slope captures). |
| Locked estimate with one-sided upper 95% bound < +0.005 | Retire the practical-gain claim on this panel. |
| Any meta-test outcome | Cannot trigger a second selection, re-lock or re-scoring on the same families. |
| Losing to a cheaper rival in G5 | The deployment case ends, even with a mean gain over G+V. |

The prior across all agents: some SciFact headroom survives a fair G+V, NFCorpus and MS MARCO
headroom mostly disappears, and HyperPrior helps a subset of families modestly. The likely deliverable
is an honest estimation study, possibly with the conclusion that a fair corpus-balanced selector plus
the target variance prior is the best label-free option at this scale.

## 8. Threats to validity

- **Task type and family identity.** SciFact's gap may be a claim-query effect, and FEVER is also a
  claim task. Report by task type; never treat task type as a model input in this run.
- **Pretraining exposure.** The corpora with known Qwen3 exposure (MS MARCO, NQ, HotpotQA, …) are also
  the label-bearing ones, so selection into meta-train is confounded with exposure. Record it, and
  report the effect separately for known and unknown exposure.
- **Optimizer artifacts.** Prior weights can stay at initialization, and many coefficients can give the
  same hard mask. Report kept steps, rerank-best steps, zero frequency and F2 vs F9.
- **Thin supervision.** SciFact partitions are small. Report common-label-count sensitivity and
  absolute scores when H is small or negative.
- **Pool selection.** Label-bearing families may differ systematically from evaluation-only ones.
  Transport beyond the eligibility frame needs an argument about sampling.
- **Leakage.** Every descriptor path is qrels-free by construction and by test. Only the evaluator
  opens final qrels after the lock. Family-aware negative mining.

## 9. Disagreements and how they were resolved

| Item | Positions | Decision | Residual dissent |
|---|---|---|---|
| First output | Astra council: one offset shared across budgets. Opus council: one offset on the global per-budget weights, a single feature. Opus-E and Opus-M (round 4): offset + slope, with offset-only as a nested ablation. | Offset + slope with the choice made inside each fold; p tied to K_train; geometry feature first. The slope is kept because the in-fold choice falls back to offset-only at no cost. The S4 per-budget gains support a⁰ ∈ R⁹, already in G+V, and the share decomposition uses three in-domain bases, so neither establishes a per-corpus slope (council-r2-opus, council-r2-astra). | Council-opus preferred log N as the single first feature. It was overruled because at p = 1 that model would be identical to the size-only rival. Council-r2-opus accepts this, provided the size-only rival runs through the identical Stage B pipeline and features use a fixed-size document sample (both now in §3). |
| Failed G3 | All: the prior rung survives. Opus-E had wanted a bias override; now withdrawn. | A G3 failure closes bias in this run. A registered diagnostic is reported. | None. |
| Test-only labels | Final scoring only. | As stated. P1 calibration uses unscored test queries (A8), pending ADR ratification. | None. |
| Claim at 8–12 families | Opus-E: the frame-level estimand, estimation by default. Opus-M: confirmatory, scoped and symmetry-qualified. Council: non-inferiority first (opus), estimation default (astra); round 2 (astra): panel scope unless a sampling design supports the frame. | The wording in §6: a symmetry-qualified panel contrast, extended to the frame only with a registered sampling argument. Non-inferiority is not claimed: it is nearly automatic for a model initialized at and shrunk towards G+V, and a mean NI test has inflated size under the left-skewed nulls that matter for safety. The shifted-test lower location bound is reported as an estimate; safety is the harm guard's job. | None: council-opus accepts the omission with this reasoning (council-r2-opus §2). |
| Protocol order | Opus-E and Opus-M both withdrew "P1 first". | P0 primary. | None. |
| Development gates at small K | Synthesis: binding sign-flip p ≤ 0.05. Opus-M: min p = 1/32 at K = 5. Council-opus: all-positive gates kill concentrated effects. Council-astra (round 2): LOFO folds share K − 2 training families at any K, so their sign flips are never calibrated. | Development p-values are **descriptive at every K**; point criteria, the no-loss rule and positive gains over matched constant, size-only and donor decide. Type I error is controlled only at the single locked meta-test. (This replaces the orchestrator's earlier "binding at K_train ≥ 8".) | None: council-r2-opus independently asked for p-values never binding in this run. |
| Rank four | Synthesis: conditional rung. Opus-M: cut. | Deferred (S1). | None. |

## 10. Implementation and records

- **ADRs to write before any new outcome:**
  1. **Protocol ADR.** Family ownership, the test-only ruling, P1 calibration (A8), the freeze
     sequence. It records two departures from `docs/protocols.md`: P0, not P1, is primary, and
     architecture is chosen by meta-train leave-one-family-out, with meta-validation only a harm
     screen (A9).
  2. **Design ADR.** The S4-compatible residuals, r_W units, exact-zero handling, the scope of G3, and
     the deferral of rank four.
  3. **Evaluation spec for G4/G5.** Thresholds, the power simulation, claim templates, the workload.
- **Reuse:** `baselines/learning_to_select.py`, `training/{losses,retrieval_trainer,examples}.py`,
  `selection/{relaxed,topk}.py`, `space/{descriptor,geometry,streaming}.py`,
  `evaluation/{descriptor_probe,statistics,benchmark_stats}.py`, `retrieval/exact.py`.
- **New work:**
  - a **multi-corpus S4 trainer** with LOCO and balanced sampling (G+V does not exist yet;
    council-opus);
  - the oracle-ladder driver;
  - family manifests and read allowlists;
  - the HyperPrior head in `hypernet/`;
  - family-level sign-flip, power simulation and harm reports.
- **Acceptance tests:**
  - zero-residual parity at every budget;
  - reachable zero weights;
  - feature/bias parity;
  - fold-local normalization;
  - denied qrels access on descriptor paths;
  - equal-family aggregation;
  - exact parity at 1024.

## Scores

| Reviewer | Plan reviewed | Score |
|---|---|---|
| Council · GPT-6.2-Astra | round-2 consensus | 7 / 10 |
| Council · Claude Opus 5.5 | round-2 consensus | 6.5 / 10 (about 7.5 for round 3 with its fixes) |
| Council · Gemini 4 Neon | round-2 consensus | **missing**: two attempts failed with 502 Bad Gateway from the Antigravity backend; no output, no substitute |
| Council R2 · GPT-6.2-Astra | PROPOSAL.md (first merge) | 8 / 10; its six must-fix changes are applied |
| Council R2 · Claude Opus 5.5 | PROPOSAL.md (first merge) | 8.0 / 10; about 9 with items 1–7, about 9.5 with all 13; all 13 are applied |
| Opus-E (round 4) | round-3 synthesis | sign with amendments (B1–B3 blocking, all adopted) |
| Opus-M (round 4) | round-3 synthesis | sign with amendments (A1–A4 blocking, all adopted) |
