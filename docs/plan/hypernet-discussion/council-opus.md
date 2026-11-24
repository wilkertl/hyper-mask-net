# Council review — Claude Opus 5.5

Independent council member. Input: `COUNCIL-BRIEF.md`, `BRIEF.md`, `round0-seed.md`,
`round1-{astra,opus-e,opus-m}.md`, `round2-{astra,opus-e,opus-m}.md`. I did not read other
`council-*.md` files. `round3-synthesis.md` was only skimmed, for §6. The repository was read-only
except for this file. I made no runs, downloads or commits.

Skills applied:
- scientific-critical-thinking: data kept apart from interpretation, rival explanations;
- peer-review: location, observation, consequence, action;
- experimental-design: unit, blocking, leakage;
- statistical-power: SESOI and gate operating characteristics.

Tags:
- **[V]**: I read it in the file myself this session (path:line).
- **[C]**: my own computation. The script is `/tmp/council-opus-scripts/power.py`, run with numpy
  and not committed.
- **[I]**: my interpretation.

---

## 0. Bottom line

1. The round-2 consensus is well grounded on the leakage rulings, G+V and the ladder. It is weak in
   three places.
   - **Its power analysis contradicts its own expected outcome.**
     - Every participant predicts the gains will be concentrated in a few SciFact-like families.
     - The "8–12 families can confirm +0.010" figure assumes homogeneous effects (SD 0.010).
     - With the same mean concentrated in 20–25% of families, the exact sign-flip has power
       **0.20–0.47** at K = 8–12 **[C]**.
   - **The meta-train screens are unanimity tests.**
     - With about 5 families, a sign-flip at p ≤ 0.05 needs every family to be positive.
     - One MS MARCO-like family at −0.002 fails the gate even when the mean headroom is +0.029
       (p = 0.0625) **[C]**.
   - **The "prior share varies 2.7×" evidence is weaker than the consensus says.** On MS MARCO,
     S4's rerank-best step is 0 in every seed. The 0.95 value is mostly what an optimizer at lr 1e-4
     left untouched, not a measured corpus property.
2. **Two confounds were missed by all three agents.**
   - **Task type** (claim verification, argument, duplicate-question, QA) is crossed with document
     source. "Family" blocks only by source. SciFact's gap is plausibly a *claim-query* effect, and
     FEVER and Climate-FEVER are claim tasks on Wikipedia.
   - **Qwen3 exposure is confounded with label availability.** The exposed corpora are the
     label-bearing ones. The meta-train set will therefore over-represent low-headroom,
     encoder-saturated corpora relative to meta-test.
3. **A missing comparator.** P1 has no closed-form analogue of what `v_T` is to P0. I add one
   zero-parameter P1 arm, a query-centred variance prior. The symmetry argument that made G+V
   mandatory makes it mandatory too.
4. **Score: 6.5/10.** The must-fix list is in §5. The plan I would sign is in §4. It is the same
   ladder with these changes:
   - futility-only screens on meta-train;
   - a one-feature rung 1;
   - gates designed for heterogeneous effects;
   - a two-axis family definition;
   - a first step that runs on the dev three with one new trainer.

---

## 1. Verification of load-bearing claims

I checked 10 claims: 9 are confirmed (2 of them with a caveat) and 1 is confirmed with a material
qualification (#4).

| # | Claim (round-2 consensus) | Verdict | Evidence |
|---|---|---|---|
| 1 | The E1 global selector is a plain union with no corpus weights, and it includes the target's own training queries | **Confirmed** | `src/hyperdime/cli/train_global.py:30,36` (default envs = all three); `:55-60` (`torch.cat` over all envs); `:83-84` (epoch chosen by the unweighted env-mean of validation nDCG) [V] |
| 2 | 45,000 / 48,059 = 93.6% MS MARCO | **Confirmed** | `reports/P3/README.md:46-48` (train queries 728 / 2,331 / 45,000) [V]; ratio [C] |
| 3 | Prior share at k = 307: 0.95 / 0.43 / 0.36; NFCorpus 0 from k = 716 and SciFact 0 from k = 819 | **Confirmed** | `reports/S4/tables/prior_share.csv:13,31,49,35,54` [V] |
| 4 | "Prior weight varies about 2.7× on a scale-free share" read as a corpus property | **Confirmed numerically; qualified** | `prior_share` is a ratio of the across-dimension spreads of the two terms. It is 1 when W is still at its zero init (`src/hyperdime/selection/diagnostics.py:23-26`, `src/hyperdime/baselines/learning_to_select.py:47-48`) [V]. MS MARCO's S4 cell chose lr 1e-4 (`docs/decisions/ADR-0012-s4-budget-prior.md:36`). Its **rerank-best step is 0 in all three seeds**, with subset nDCG flat at 0.3548–0.3549 (`reports/S4/tables/kept_steps.csv:5-7`) [V]. MS MARCO's 0.95 therefore mostly records "training never helped", and its magnitude is set by the init and the lr. It is not an estimated optimum. |
| 5 | MS MARCO's S3 weight never left init | **Confirmed** | `docs/decisions/ADR-0011-s3-prior-weight.md:33,39` [V] |
| 6 | E1 in-domain minus global at k = 307: +0.1116 / +0.0171 / −0.0019 | **Confirmed** | `reports/E1/tables/payoff.csv:2-4` [V] |
| 7 | A reference selector trained on a meta-test environment's training labels is inadmissible | **Confirmed** (the literal rule) | `AGENTS.md:15-16` "or any label from a meta-test environment, in training…" [V]. `docs/protocols.md:22` [V] |
| 8 | Qwen3 SFT data include MS MARCO, NQ and HotpotQA | **Confirmed, with a caveat the consensus missed** | Qwen3 Embedding, p. 13, Table 6: "MS MARCO, NQ, HotpotQA, NLI, Dureader, … CodeSearchNet .etc", about 7M labeled pairs plus about 12M synthetic in SFT and about 150M synthetic in pre-training [V]. The "**.etc**" means exposure cannot be enumerated. Tagging only three corpora as exposed under-counts. |
| 9 | E2: the direct probe's apparent signal is mostly variance. B2 = 0.5381 / 0.6246 / 0.6687, and the SciFact direct-variance cell is −0.0015 vs B2 | **Confirmed** | `reports/E2/tables/ladder.csv:3,10,17,31` [V] |
| 10 | The MS MARCO F12 held-out result is not yet available | **Confirmed** | `reports/P3-bench/RUNS.md:153` ("running"); the evaluation dir holds only `nfcorpus/` and `scifact/` [V] |
| 11 | 8–12 meta-test families power about +0.010 | **Confirmed only under homogeneity** | At mean 0.010 and SD 0.010, the exact sign-flip power is 0.80 / 0.90 / 0.95 at K = 8 / 10 / 12 [C], which reproduces Opus-M. Under a concentrated effect it is 0.20–0.47 (§2, C1) [C]. |

Two facts that bear on "runnable now" (§4.5):
- `GeometricSelector` holds **one** corpus's `variance` buffer
  (`src/hyperdime/baselines/learning_to_select.py:46`) [V].
- `train-global` trains the L2S KL objective via `train_selector`
  (`src/hyperdime/cli/train_global.py:71`), not the S4 retrieval loss [V].

So a corpus-balanced G+V needs a new multi-environment S4 trainer that swaps `v_T` per episode. It
is small, but it is not "existing code", as round 2 implied.

---

## 2. Critique of the round-2 consensus

### Critical (they change what is built or what is concluded)

**C1. The power plan assumes homogeneous effects, contrary to everyone's stated prior.**
- *Observation.*
  - Opus-M's sign-flip table uses a family SD of 0.010 around a mean of 0.010
    (`round2-opus-m.md:481-488`).
  - All three agents expect concentrated effects: "R1 passes on one or two SciFact-like families"
    (`round2-opus-m.md:604`). Opus-E's prior is the same (`round1-opus-e.md:482-485`).
- *Evidence [C], exact one-sided sign-flip at α = 0.05, 1,500 replicates.* Two scenarios have a
  mean family gain of about 0.009–0.010:
  - 25% of families gain 0.035, the rest 0 ± 0.004;
  - 20% gain 0.05, the rest 0 ± 0.004.

  | K | homogeneous (0.010, SD 0.010) | 25% × 0.035 | 20% × 0.05 | null (type I) |
  |---|---|---|---|---|
  | 6 | 0.65 | 0.17 | 0.14 | 0.05 |
  | 8 | 0.80 | 0.27 | 0.20 | 0.05 |
  | 10 | 0.90 | 0.37 | 0.27 | 0.04 |
  | 12 | 0.95 | 0.47 | 0.36 | 0.04 |

- *Consequence.* The confirmatory test is most likely to be run in exactly the world where it has
  about 30% power. A "fail" would then be uninformative, yet the kill rule would retire the method.
- *Action.*
  - Register the mixture scenario as the **primary planning scenario**.
  - Make the confirmatory primary a test that matches a concentrated alternative. Two options:
    - the mean contrast restricted to families the *label-free* model predicts will move (see §4.4);
    - a two-part claim: a harm guard on all families, plus superiority on predicted-movers.
  - Otherwise pre-declare estimation-only.

**C2. The meta-train screens are unanimity tests at K ≈ 5, so they are passable only by
homogeneous effects.**
- *Observation.* R1 unlocks on "family-mean H ≥ 0.010 with exact one-sided sign-flip p ≤ 0.05
  over families" **and** "leave-one-family-out mean ≥ 0.005 for every left-out family"
  (`round2-opus-m.md:517`). R2 needs the same sign-flip (`:518`).
- *Evidence [C].*
  - With K = 5 the attainable p values are multiples of 1/32. Families
    {0.10, 0.02, 0.015, 0.012, −0.002} give p = 0.0625: **fail**, at mean 0.029.
  - SciFact-like plus four nulls gives p = 0.44.
  - MS MARCO is a meta-train family, and E1 puts its headroom at about −0.002.
- *Consequence.* The headroom gate is decided by whether the dev family MS MARCO lands at +0.001 or
  −0.001, not by whether learnable headroom exists. This is the "unpassable gate" pattern
  (statistical-power skill).
- *Action.* At meta-train, use **futility-only** rules (stop on evidence of absence), plus a
  **learnability** condition: at least 2 families with headroom, so LOFO can learn a direction.
  Never require significance over 5 units to proceed. See §4.3.

**C3. Rung 1's capacity does not match K_train, and its motivating evidence is partly an optimizer
artifact.**
- *Observation.* Rung 1 fits 3 slopes (Opus-M) or 3 + 3 budget slopes (Opus-E,
  `round2-opus-e.md:296`) by LOFO ridge on K_train ≈ 5–7 families. Its rationale is the "2.7×"
  prior share.
- *Evidence.*
  - Verification #4: the MS MARCO end of that 2.7× is a step-0 / lr-1e-4 artifact.
  - The remaining contrast is two small corpora (SciFact, NFCorpus) whose shares decay with k.
- *Consequence.*
  - A ridge fit with p = 3–6 on 4–6 training points per fold is close to interpolation. LOFO will
    usually lose to predict-the-mean, so rung 1 is expected to fail by construction.
  - A failure would then wrongly be read as "no corpus channel".
- *Action.* Use one pre-registered, signed feature (§4.1, R1), with a free global offset in the
  comparator so that only the slope is credited. Report the F9 oracle shape descriptively.

**C4. Task type is an uncontrolled second clustering axis (groupthink).**
- *Observation.* Families are defined by document source and near-duplicate overlap
  (`round2-opus-m.md:515,579`). Query task is ignored.
- *Evidence.*
  - SciFact's gain survives against a global selector that beats full on SciFact
    (E1-d, `round1-opus-e.md:81`). That points to a query-form correction.
  - L2S pseudo-queries recover about 3% of the supervised gain on SciFact and about 73% on
    MS MARCO (`round2-opus-e.md:207-213`, L2S Table 5 p. 8).
  - FEVER and Climate-FEVER are claim-verification tasks.
  - The resulting groups share a task across different sources: ArguAna and Touché (argument),
    Quora and CQADupStack (duplicate questions), MS MARCO / NQ / HotpotQA / FiQA (QA).
- *Consequences.*
  1. A balanced G+V that includes FEVER may close SciFact's gap with no corpus conditioning.
  2. A "generalizes to unseen families" claim can be carried by task-type matching across
     families. That is interpolation in task space, not corpus geometry.
- *Action.*
  - Record task type for every corpus. Stratify the split by source family × task type.
  - Pre-register a sensitivity analysis: the meta-test effect on families whose task type is absent
    from meta-train.
  - Add "the nearest meta-train *task*" to the 1-NN control.

### Important

**I1. Qwen3 exposure is confounded with label availability and probably with headroom.**
- The label-bearing pool (MS MARCO, NQ, HotpotQA, likely FEVER and others under ".etc") is the
  exposed pool (verification #8). Meta-test is test-only and plausibly less exposed.
- An encoder fine-tuned on a corpus's query distribution leaves little query-side correction to
  learn. That fits MS MARCO's near-zero headroom.
- So meta-train teaches "deviate little from G+V", and meta-test may need larger deviations. This
  is a systematic shift in exactly the learned quantity. `log N` is also confounded with exposure,
  because the big web/Wikipedia corpora are the exposed ones.
- Action:
  - Treat exposure as *known-exposed / unknown*, never "unexposed".
  - Report headroom by stratum in the census.
  - Require at least 2 meta-train families that are not known-exposed (SciFact, NFCorpus and
    FiQA are candidates).

**I2. Missing P1 analytic comparator.**
- Round 2 made G+V mandatory so the hypernetwork is not credited for what `v_T` gives for free.
- Under P1 the same logic requires a closed-form use of `Q_calib`. A query-centred prior does this:

  `log((q − μ_Q)² ⊙ v_T + ε)`, where `μ_Q` is the calibration-query mean.

  It removes the instruction and query-form direction, which rival R1 predicts is SciFact's gain.
- It has zero learned parameters, is P1-legal and costs one mean.
- Without it, a P1 win cannot be attributed to the learned map.

**I3. The ICC gate (H3) is uninformative at K ≈ 5.**
- A between-family ICC estimated from 5 groups has a confidence interval spanning most of [0, 1].
- Replace it with Astra's **split-half retrieval transfer**: fit the oracle on half A of a family's
  training queries, score on half B, and require it to retain ≥ 50% of the oracle gain. This is
  per family, so it does not depend on K.

**I4. The oracle-recovery ratio R_j is a one-family statistic.**
- If headroom sits on one family, R_j = ΣΔ / ΣH is that family's ratio.
- Action: report R_j per family, and license a rung only on families with headroom (§4.3, G2).

**I5. Meta-validation spends test-only families on a near-powerless harm check.**
- Two of the ~8–12 test-only families go to meta-val (`round2-opus-m.md:521,579`). That costs
  about 20% of meta-test units for a check with almost no power.
- Action: use LOFO on meta-train as the decision procedure. Make meta-val **one** label-bearing
  family (non-test labels), and keep every test-only family for meta-test. This needs a
  protocols.md note, because meta-val is still a distinct set and AGENTS.md is untouched.

**I6. G+V's own variance is unmeasured.**
- The headroom H = specialist − G+V depends on a G+V fitted LOFO with K−1 families.
- Its seed and fold variability adds to the variance of H and to the treatment contrast.
- Action: 3 seeds of G+V per fold, averaged per query, with the matched "continued" arm (§4.2).

### Minor

- **M1. The "first step" is not code-free.** A multi-env S4 trainer with a per-episode `v_T` swap is
  needed (§1, closing facts). Say so, and size it.
- **M2. The dev-three test splits are reused** (`ADR-0012:test_set_reuse`). Score the first step
  on the `p3-query-split-v1` evaluation ids for SciFact and NFCorpus (F12 split). MS MARCO waits
  for F12.
- **M3. Opus-E's bounded λ reparameterization** (`round2-opus-e.md:292-296`) changes the S4 model
  form after S4 was accepted. Astra's additive projected offset `max(0, a⁰_k + t)` nests G+V
  exactly, handles zeros and keeps the accepted S4 form. Prefer it.
- **M4. "Correct-minus-donor > 0" as a fixed-sequence step at K = 8.** The donor pool is the 4–6
  meta-train families, so donor draws are highly correlated. Average over all donors per family
  before the family-level test.
- **M5. Equal-decile averaging weights the low-k budgets heavily**, because differences are
  largest there. Keep the decile mean as primary for continuity with ADR-0018, and report the
  per-budget curve.
- **M6. Corpus-size confound in noise.** Per-family SE depends on the number of evaluation queries,
  not on N. Require ≥ 150 evaluation queries per primary family (as Opus-M), and weight families
  equally.

---

## 3. The five open disagreements: decisions

**(1) What is the first generated output?**
- **Decision.** One corpus-conditioned scalar: an additive offset to the global per-budget prior
  weights, driven by **one** pre-registered label-free feature. The comparator absorbs a free
  global offset, so only the corpus-dependent slope is credited.
- The F9 per-budget oracle and the Fb bias oracle are measured alongside it. They decide whether a
  larger rung is ever licensed.
- **Reason.**
  - It is the only output whose number of parameters (1) fits K_train ≈ 5–7.
  - It nests G+V exactly and keeps the accepted S4 form (`max(0, a⁰_k + t)` reaches zero and can
    leave it again).
  - The evidence for per-budget *shape* differences comes from two small corpora plus one
    optimizer artifact (verification #4), too little to justify 9 corpus-dependent outputs.
  - Opus-E's prediction that the prior offset recovers little on SciFact is plausible. That is a
    reason to *measure* the bias oracle early, not to *build* the bias rung first.

**(2) What does a failed G3 rerun kill?**
- **Decision.** It kills every rung that maps per-dimension descriptor rows to outputs: tied bias,
  rank-4 and any latent-to-per-dimension head. It does not kill rung 1, which reads no descriptor
  rows.
- A G3 pass is **necessary, not sufficient**. The per-dimension rung must also pass its own LOFO
  retrieval test.
- **Reason.**
  - ADR-0013 prescribes the rerun unchanged, so honoring it avoids a forking path.
  - Opus-E's "retrieval test can rescue a G3 failure" turns a registered gate into an advisory one.
    It would rest on an underpowered K ≈ 5 retrieval screen, which is the more permissive of the two
    tests.
  - A G3 failure says nothing about an environment-level scalar, so it cannot bind rung 1
    (Opus-E's category-error point).

**(3) Policy on corpora whose only labels are test labels**
- **Decision.** Default: **no re-split**. Test-only corpora are evaluation-only, and P2 needs its
  own pool.
- An ADR to re-split at most 2 test-only families into meta-train may be requested **only** if both
  hold:
  - G2 licenses a per-dimension rung;
  - the G3 rerun passes.

  Any re-split corpus becomes permanently meta-train-only.
- **Reason.**
  - The leading rung has one corpus-conditioned parameter and does not need more training families.
  - The confirmatory test is bound by meta-test families. Each re-split moves one unit across one
    for one, costing about 10% of meta-test units (`round2-opus-m.md:586-589`).
  - Spend them only when a rung needs them.

**(4) Confirmatory or exploratory at 8–12 families?**
- **Decision.** Confirmatory **non-inferiority at δ = 0.010**, then superiority, in a fixed
  sequence.
  - Superiority is labeled "powered" only if a pre-unblinding simulation shows ≥ 80% power. The
    simulation uses the **mixture** scenario and the upper 80% limit of the LOFO heterogeneity.
  - Otherwise superiority is reported as estimation: the per-family effects and the REML prediction
    interval.
- **Reason.**
  - Superiority at a concentrated effect has 20–47% power at K = 8–12 (C1). Calling it
    confirmatory invites misreading a fail as a refutation.
  - Non-inferiority at 0.010 is feasible. If true Δ ≈ 0 and the family SD is about 0.008, d ≈ 1.25
    and K ≈ 6–7 suffice [C, normal approximation].
  - The margin is meaningful: it is below the 0.012–0.030 cost of naive transfer between corpora
    (E1-b, `round1-opus-e.md:79`).
  - Non-inferiority at 0.005 needs about 25 families and stays descriptive.

**(5) Primary endpoint and primary comparator**
- **Endpoint.** Per-query nDCG@10 averaged over the nine deciles 102…921, with seeds averaged per
  query. Take the corpus mean, then the **equal-weight family mean**. k = 307 is secondary.
- **Comparator.**
  - Rung 1 is compared with **G+V⁺**: G+V plus a refit global offset `c`, the matched-capacity
    global.
  - Plain G+V is reported too, and it is the reference for the non-inferiority step.
  - Every higher rung is compared with the accepted rung below it.
  - Any rung reading descriptor rows also needs correct > donor, averaged over donors.
- **Reason.**
  - The decile mean is the ADR-0018/F12 convention, and it scores all nine outputs.
  - G+V⁺ separates "corpus information" from "one more fitted number". Astra's matched-global
    concern is met without the degenerate template arm (`round2-opus-m.md:111-118`).

---

## 4. The proposal I would sign

### 4.1 Architecture rungs

**Notation.**
- `q ∈ R^1024` is the frozen, unit-norm Qwen3 query embedding. Corpus T has document variance
  `v_T ∈ R^1024`.
- `k ∈ K = {102, 204, …, 921}`. `π_T(q) = log(q ⊙ q ⊙ v_T + 1e−12)`.
- Common to every rung:
  `m = TopK(s(q), k)`, and `score(q, d) = Σ_j m_j q_j d_j`.
  Documents are untouched, the masked query is not re-normalized, and scoring is exact float64
  with the lower-row tie rule.

**R0 · G+V (base, frozen).**
`s(q) = a⁰_k · π_T(q) + W₀ q + b₀`

- `W₀` is 1024×1024, `b₀` is 1024, and `a⁰ ∈ R⁹≥0`. Each episode uses the corpus's own `v_T`.
- Trained once with the S4 recipe (ADR-0012), frozen unchanged:
  - masked retrieval loss through the soft top-k, one budget per batch;
  - training negatives only;
  - **family-uniform, then corpus-uniform** sampling;
  - 3 seeds.
- **G+V⁺** (matched comparator): `a_k = max(0, a⁰_k + c)`, with one global `c` chosen on LOFO
  training-family validation.

**R1 · prior calibrator (the first corpus-conditioned output; 1 learned corpus parameter).**
`a_k(T) = max(0, a⁰_k + c + β · φ̃_T)`

- P0: `φ_T = log N_T`. This is pre-registered, with directional hypothesis β > 0: larger corpora
  keep more prior. The basis is AdaWidth Prop. 1, pp. 6–7, and the S4 direction, flagged as weak.
- P1: `φ_T` = mean RDIME `ε̂²` over `Q_calib` (Statistical Foundations, Cor. 1, p. 3).
- `φ̃` is standardized on training families only. β is fitted by a 1-D grid that maximizes the
  equal-family validation decile-mean on training families, with no log-link and no zero problem.
- Log participation ratio is a registered **secondary** feature, never added to the primary fit.

**R2 · tied coordinate bias (18 parameters; locked behind G2 + G3).**
`s(q) += X_T w`

- `X_T = clip((F_T − M_{−T}) / σ, −5, 5)` is 1024 × 18: the P0 descriptor groups A+E, centred by
  the leave-it-out mean on training families.
- `w ∈ R^18` is zero-initialized and trained with the R0 loss, with R0 and R1 frozen.
- Matched control: a free global `Δb ∈ R^1024` trained for the same steps.
- Donor control: `X` from another meta-train family, with `v_T` kept.

**R3 · rank-4 query interaction `(X_T H)(B q)`: closed.** It reopens only if all of these hold:
- the G3 rerun passes;
- R2 is accepted;
- the incremental Fr4 oracle over Fb is ≥ 0.005 on ≥ 2 headroom families;
- ≥ 10 audited label-bearing families.

### 4.2 Training rules common to all rungs

- S4 recipe frozen before any new-corpus number. The same lr grid for every arm and its matched
  control (C11/A1 showed lr decides contrasts).
- **LOFO means everything is refit without the held-out family:** base, standardization, `c`, β, w
  and grids.
- Calibration queries are hash-selected and disjoint from evaluation ids in meta-train as well as
  meta-test.

### 4.3 Gates (thresholds fingerprinted in an ADR before the first number)

| Gate | Data | Pass / unlock | Kill |
|---|---|---|---|
| **G0 audit + split freeze** | none read | Families = document source. Each corpus tagged with **task type** and **Qwen3 exposure** (known/unknown). ≥ 150 evaluation queries per primary unit. Split randomized within source × task × label-availability strata. ≥ 5 label-bearing meta-train families (dev three included); ≥ 8 meta-test families. | < 5 meta-train families: development screens are exploratory only. < 8 meta-test families: no confirmatory test |
| **G1 headroom census** (LOFO) | meta-train non-test labels | ≥ 2 families with `H_f = S4_f − G+V_{−f} ≥ 0.010` and query-level 95% lower bound > 0. Report by exposure and task stratum. | < 2 such families: **ship G+V** (ADR + published census). No significance over families is required either way. |
| **G2 oracle ladder + reliability** | meta-train | Rung j is licensed if, on ≥ 2 headroom families, its oracle (F1 offset; F9; Fb; Fr4) beats the accepted lower rung by ≥ 0.005 with a query-level lower bound > 0, **and** split-half transfer of the oracle keeps ≥ 50% of the gain. | F1 and Fb both unlicensed: the gap needs per-corpus W (labels) → P2 on its own pool; ship G+V |
| **G3 rung-1 predictability** (LOFO) | meta-train | Mean LOFO gain vs G+V⁺ ≥ 0.005 (point); sign(β̂φ̃_f) = sign(t*_f) on all but at most one headroom family; refitted permutation null over all K! assignments, p ≤ 0.05; no family < −0.01 | Fail → G+V⁺ is the deliverable; record it |
| **G3b rung 2** | meta-train | E2 rerun unchanged (ADR-0013) passes; Fb licensed in G2; LOFO incremental gain over R1 ≥ 0.005; donor-averaged correct − donor > 0 on all but at most one headroom family | Any part fails → per-dimension generation recorded as unsupported |
| **G4 meta-val** | 1 label-bearing family (non-test labels) | No harm: Δ ≥ −0.01 vs G+V | Harm → no meta-test; at most one recorded re-lock |
| **Lock** | — | Checkpoint hashes, split manifest, evaluation ids, analysis code; power simulation (§4.4) | — |

### 4.4 Evaluation (meta-test, once, scoring-only process)

- **Unit.** The source family, weighted equally. Corpus-level analysis is a sensitivity, and so is
  "families whose task type is absent from meta-train".
- **Endpoint.** As in disagreement (5).
- **Comparators.** All arms use the same queries, the nine deciles plus 1024 parity, and the exact
  evaluator:
  - full 1024 and MRL prefix-k (mandatory);
  - `variance`;
  - G+V (primary reference) and G+V⁺ (matched);
  - legacy union (continuity only);
  - donor- and shuffle-descriptor arms;
  - 1-NN family and 1-NN task;
  - **P1 query-centred prior** `log((q − μ_Q)² ⊙ v_T + ε)` + `W₀q + b₀` (P1 tables);
  - PRF-DIME and Eclipse (online cost reported separately);
  - per-corpus pseudo-query self-tuning of `c` (offline cost reported separately).
  - No in-domain, oracle or P2 arm touches meta-val or meta-test labels.
- **Primary chain.** Fixed sequence, one-sided α = 0.05, exact sign-flip over all 2^K family sign
  vectors:
  1. Locked rung **non-inferior** to G+V at δ = 0.010.
  2. Locked rung **superior** to G+V. This is "powered" only if the lock-time simulation gives
     ≥ 80% power under the mixture scenario, using the upper 80% limit of the LOFO heterogeneity;
     otherwise it is labeled estimation.
  3. Superior to G+V⁺.
  4. If R2 is the locked rung, superior to the donor average.
- **Key secondary (pre-registered, estimation).**
  - The **enrichment readout**: before unblinding, the locked model's label-free `t̂_T` marks
    "predicted movers" (`|β̂φ̃_T|` above the smallest offset whose oracle gain was ≥ 0.005 on
    meta-train).
  - Report the movers' family effects and the non-movers' harm separately. This is the honest
    readout of a concentrated effect.
- **Always reported.**
  - REML μ, τ and the 95% prediction interval for a new family;
  - delete-one-family signs;
  - per-corpus harm guard (Δ < −0.01, Holm);
  - strata by exposure and by task type;
  - Δ against distance to the nearest meta-train family and task;
  - the per-budget curve;
  - offline and online cost split.
- **Futility.** If the upper 95% bound of the family-mean gain vs G+V is below +0.005, retire the
  practical-gain claim at any K.

### 4.5 First concrete step, runnable now on the dev three (exploratory, no binding gate)

1. **Code (one ticket).** A multi-environment S4 trainer:
   - `BudgetGeometricSelector` with the `variance` buffer swapped per episode;
   - corpus-uniform sampling;
   - `masked_retrieval_loss`.

   Plus a `t`-offset grid evaluator. Everything else exists: the exact scorer, the F12 splits and
   S4.
2. **Runs.** For each held-out `c ∈ {SciFact, NFCorpus}`, score on c's F12 (`p3-query-split-v1`)
   evaluation ids. MS MARCO is held out only after F12 finishes there. Arms, 3 seeds:
   - G+V_{−c}, trained balanced on the other two;
   - the union with and without c, using the *same* S4 loss, which separates "balancing" from
     "LOCO";
   - G+V_{−c} with `v_pool` (H1b);
   - c's own in-domain S4 (legal, because dev is meta-train);
   - the oracle F1 offset and F9 grids, fitted on c's tune ids;
   - oracle Fb (1024 biases, W frozen, trained on c's fit ids);
   - the P1 query-centred prior, with μ_Q from c's *unlabeled* training queries;
   - full, prefix and `variance`.
3. **Registered predictions (before running).**
   - SciFact headroom over G+V_{−SciFact} ≥ 0.08;
   - R_F1 < 0.25 and R_Fb ≥ 0.5 on SciFact (Opus-E);
   - NFCorpus headroom < 0.01.
4. **How the results steer the program.** None of these readings is binding.
   - **The query-centred P1 prior or F1 recovers ≥ 50% of SciFact's headroom.** Rival R1 gains
     support and a cheap closed form exists. Prioritize the P1 track and the task-type
     stratification.
   - **Fb ≥ 0.5 while F1 < 0.25.** The value lies in per-dimension corrections, and the G3 rerun
     becomes the program's critical path.
   - **Both fail.** The SciFact gap needs labels. Recommend shipping G+V after the G1 census on new
     corpora, and budget embeddings accordingly.

---

## 5. Score and must-fix items

**Score of the round-2 consensus plan: 6.5 / 10.**
- Strong on unit, leakage, comparators and honesty about power.
- Weak where it matters for a decision: power under the expected (concentrated) effect, gates that
  can be passed at K ≈ 5, the confounds, and capacity at rung 1.

**Must-fix items to reach ≥ 9** (ordered by impact):
1. **Concentration-aware power.** Register the mixture scenario as primary for planning. Make
   non-inferiority at δ = 0.010 the confirmatory first step, with superiority powered only if
   simulation allows. Add the label-free enrichment readout (C1, §4.4).
2. **Replace the significance-over-families unlock rules** at meta-train (R1/R2 sign-flip at
   K ≈ 5) with futility-only kills plus "≥ 2 headroom families" learnability (C2, §4.3).
3. **Rung 1 gets one pre-registered signed feature** and a G+V⁺ comparator that absorbs the global
   offset. Drop budget slopes and p = 3 ridge until K_train ≥ 10 (C3).
4. **Task type as a second blocking and stratification axis.** Add the task-absent sensitivity
   analysis and the 1-NN-task control (C4).
5. **Exposure as known/unknown**, tied to label availability. Require ≥ 2 not-known-exposed
   meta-train families and stratified census reporting (I1).
6. **Add the zero-parameter P1 query-centred prior** as the P1 analytic comparator (I2).
7. **Replace the ICC gate with split-half retrieval transfer.** Report R_j per headroom family
   (I3, I4).
8. **Restate the prior-share evidence** with the MS MARCO step-0 caveat, and size the first step
   honestly: one new trainer, F12 splits (verification #4, M1, M2).
9. **Keep test-only families for meta-test.** Make meta-val a single label-bearing family (I5).

---

## 6. Round 3: what changes my judgment

I skimmed `round3-synthesis.md`. My review above is based on rounds 0–2.
- **It adds things I asked for.** It requires asymmetric and null scenarios in the power
  simulation, a conservative heterogeneity limit, and estimation by default at 8–14 families
  (`round3-synthesis.md:102-104,115`). It also asks for kept-step and zero-frequency reporting
  (`:131`) and an F1/F2 oracle ahead of F9 (`:65`). These partly address C1 and the
  verification #4 caveat.
- **It does not address everything.**
  - It does not make the meta-train screens futility-only.
  - It does not add task type or the P1 analytic comparator.
  - It still frames the primary as superiority. Under my simulation, that test is underpowered in
    the expected world.
- **Net effect.** If those items land, I would score round 3 about 7.5.
