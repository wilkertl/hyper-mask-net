# Round 4 — Opus-M (methods lead / red team): sign-off or dissent

I re-read `BRIEF.md`, `round0-seed.md`, all round-1 and round-2 files, and `round3-synthesis.md`.
I did not open any `council-*.md` file. Skills applied: experimental-design (unit, leakage, gate
structure), statistical-power (operating characteristics of each gate and of the final test), and
scientific-critical-thinking (whether each claim follows from the design).

Tags. **[mine]** marks a number I computed this round in `/tmp/hm-r4/power.py` and
`/tmp/hm-r4/skew.py` with the project venv. These scripts are not committed. Simulations use 3,000–4,000
draws, so Monte Carlo SE ≤ 0.009 on any probability. Sign-flip tests enumerate all 2^K vectors.
**[assumed]** marks a planning input that nobody has measured. `§N` refers to `round3-synthesis.md`.

---

## 1. Verdict

**SIGN WITH AMENDMENTS.** Four of the amendments block sign-off (§4, A1–A4).

The synthesis gets the important structure right:
- the unit is the family;
- the base is frozen and corpus-balanced, and G+V is the comparator;
- every development measurement that influences a decision comes from meta-train;
- no meta-test label reaches a reference, a census, a miner or meta-validation;
- there is a single locked candidate and a fixed-sequence primary chain;
- the design is powered at +0.010, and the downgrade rules are declared in advance.

The four blocking defects are narrow, and each is a rule that either cannot pass or has an
undefined outcome:
- **A1.** The §6 null-size criterion either cannot pass or depends on which skew scenario is chosen.
- **A2.** Gate D's Holm step cannot pass at the stated floor of five families.
- **A3.** Gate B has a failure branch with no action.
- **A4.** The HyperPrior fit has more free parameters than the training families can identify at the
  floor.

None of these changes the architecture.

---

## 2. Methods audit of the synthesis

### 2.1 Gates in §5

Columns: **Defined?** asks whether the input, statistic, threshold and action are all specified.
**Can fail?** asks whether a plausible state of the world fails the gate, and whether a plausible
state passes it. **Leak-free?** is judged against AGENTS.md and `docs/protocols.md`. **Powered?** gives
the operating characteristic at the anticipated K.

| Gate | Defined? | Can fail / can pass? | Leak-free? | Powered? |
|---|---|---|---|---|
| **Step one** (dev three, oracle ladder) | Yes. Fit, checkpoint and score partitions are hash-split within non-test IDs, with identical partitions across arms. "No numerical result here unlocks confirmation" is correct. | Yes on both sides. R is defined only when H ≥ 0.010 with a positive lower bound, which avoids unstable ratios. | Yes. The historical corpora are permanently meta-train (D9). Their test splits have already been seen and are not used here. | Not a gate (n = 3). Note that SciFact's fit partition is about ⅓ of 728 queries, so the specialist is weaker than E1's in-domain selector and H will be **understated** relative to E1. |
| **A. Inventory** | Mostly. "≥ 5 training families **and** two eligible validation families" means ≥ 7 label-bearing families in total, because meta-validation needs non-test labels (§6). The text should say so. | Yes. Under the round-2 pool estimates (3 historical plus about 3–5 new label-bearing families) it may well fail **[assumed]**. | Yes. Nothing is read. | n/a |
| **B. Payoff census** | **No (A3).** The pass rule is three point conditions. "UB < 0.010" ends the program. "Other failures leave headroom unresolved" has **no action**. | Yes. | Yes, provided the base is family-held-out, the specialist is fit on recipient fit labels, and scoring is on the untouched partition. | Point rules at K = 5–8 form a lenient screen. That is acceptable, because D and F carry inference. |
| **C. Attainability and stability** | The attainability part is defined. The ICC part is not: the units are unclear (families? families × budgets? the two numbers (u, v)?). | The attainable increment ≥ 0.005 can fail and can pass. ICC ≥ 0.70 at K = 5–8 families is a coin flip near a true ICC of 0.7, because the interval spans roughly 0.2–0.95 **[assumed]**. | Yes. | The ICC is not powered. It duplicates what D measures directly (see §5, S2). |
| **D. HyperPrior** | Nearly. "Holm-adjust the two family sign-flip screens" is ambiguous. I read it as offset-only versus offset+slope. | **At K = 5 it cannot pass (A2).** The smallest exact one-sided sign-flip p is 1/32 = 0.031, which exceeds the Holm threshold of 0.025. At K = 6, Holm requires all six family effects to be positive. | Yes, if the permutation null refits standardization, λ and the offset/slope choice (the synthesis says so). | P(pass), true gain 0.010, SD 0.010 **[mine]**: <br>K = 5: 0.43 without Holm, 0.00 with Holm. <br>K = 6: 0.64 / 0.34. <br>K = 8: 0.81 / 0.65. <br>Null pass (no Holm): K = 5 0.03, K = 6 0.05, K = 8 0.05. <br>The conjunction with the ≥ 0.005 point rule, ≥ 40% of oracle, and beating NN and size-only lowers each figure further. |
| **E. G3 rerun** | Yes. ADR-0013 is replayed unchanged on a preassigned fresh triplet. | Yes, and its power is unknown. I accept that: the rule binds the protocol, not the truth. | Yes. Meta-train only, using in-domain S1 targets of meta-train families. | Unknown. Report the triplet's query counts beside the result. |
| **F. Bias, then rank** | Bias: yes. Rank needs ≥ 20 environments in ≥ 15 families. | Bias: yes. **Rank cannot pass under any audited pool anyone has proposed**, so it changes no decision in this run (§5, S1). | Yes. | Bias: as for D, without the Holm problem. |
| **G. Meta-validation** | Yes. | Yes. P(pass) with K = 2, family SD 0.010 and within-family SE 0.005 **[mine]**: 0.47 at true 0, 0.70 at +0.005, 0.87 at +0.010. The harm clause alone gives 0.66 / 0.83 / 0.93. | Yes, provided "untouched" also excludes the meta-val families' **training** splits from Stage A/B and from mining. It should say so. | Not a power gate, as the synthesis states. Its 13% false-kill rate at +0.010 must enter the joint power calculation (A6). |
| **H. Meta-test and G5** | Yes. | Yes. | Yes. The evaluator alone opens qrels after the lock. | See §2.2. |

### 2.2 Evaluation in §6: rechecking the power statements

**Statement 1.** "+0.005 at family SD 0.010 needs roughly 25 independent families by the one-sided
normal approximation."
- n = ((z₀.₉₅ + z₀.₈₀)·σ/δ)² = ((1.645 + 0.842) × 2)² = 4.974² = **24.7** **[mine]**. Correct.
- With a one-sample t, K = 27 gives 0.81 power **[mine]**. The exact sign-flip needs slightly more.
- "Before finite-sample penalties" is accurate.

**Statement 2.** "At 8–12 families design for a true +0.010 against zero."
Exact one-sided sign-flip at α = 0.05, power at a true family-mean gain of +0.010 **[mine]**:

| Family SD | K = 6 | K = 8 | K = 10 | K = 12 |
|---|---|---|---|---|
| 0.008 | 0.81 | 0.93 | 0.97 | 0.99 |
| 0.010 | 0.63 | 0.79 | 0.90 | 0.94 |
| 0.0146 | 0.41 | 0.53 | 0.64 | 0.71 |
| 0.015 | 0.39 | 0.48 | 0.62 | 0.70 |
| 0.010, true gain +0.005 | — | 0.33 | 0.43 | 0.51 |

The statement is correct at SD ≤ 0.010. **But §6's own conservative rule makes 80% hard to reach:**
- §6 plans with the upper one-sided 80% confidence limit of the development SD. That limit is
  √((K−1)/χ²₀.₂₀,K−1) times the point estimate: **1.56× at K_dev = 5, 1.46× at 6, 1.35× at 8,
  1.29× at 10** **[mine]**.
- 80% power at +0.010 needs a planning SD of at most about 0.0101 at K = 8, 0.0116 at K = 10 and
  0.0130 at K = 12 (t approximation, which matches the exact table above).
- So the **development point SD must be at most roughly 0.007 (K = 8) to 0.009 (K = 12)**.
- The only heterogeneity measured so far is E1's in-domain-minus-union headroom: +0.112, +0.017,
  −0.002. Its SD is about 0.06. That is not the candidate's effect, but it shows that one family
  can carry most of a corpus-level gain.

**My expectation: the registered simulation will most likely downgrade the meta-test to estimation.**
The synthesis permits this, and §1 should say it is the likely outcome rather than a corner case
(A7).

**Statement 3.** "Require null rejection ≤ 0.055 … including asymmetric and null scenarios."
- **This criterion cannot be met for left-skewed nulls by any test of the mean at K ≤ 12.**
- Left skew means mean-zero family effects where most families gain slightly and a few lose a lot.
- Size at nominal α = 0.05, family SD 0.010 **[mine]**:

| Null scenario | K = 8 | K = 10 | K = 12 |
|---|---|---|---|
| Normal | 0.053 | 0.054 | 0.053 |
| Left skew, skewness 1 (gamma) | 0.086 | 0.085 | 0.084 |
| Left skew, skewness 2 (exponential) | 0.134 | 0.137 | 0.123 |
| Right skew, skewness 2 | 0.010 | 0.013 | 0.017 |

- The studentized sign-flip is identical to the mean statistic, because Σx² is invariant under
  sign flips. The one-sample t has the same size to within 0.006.
- **Consequence.** The criterion either always fails, which forces estimation regardless of the
  data, or passes or fails depending on which asymmetric scenarios are registered. The second is a
  forking path.
- **The left-skew case is the deployment-relevant failure mode:** a small gain on most families and
  a material loss on a few. The harm guard, not the mean test, has to catch it. Hence A1.

**Statement 4 (joint attribution).** "Do not infer joint power from one comparison." This is correct
and matters.
- Fixed sequence G+V → donor, both contrasts with family SD 0.010 and correlation 0.5 **[mine]**:
  - if the donor contrast is also +0.010, joint power is 0.69 / 0.83 / 0.91 at K = 8 / 10 / 12;
  - if the donor recovers half the gain (contrast +0.005), joint power is **0.34 / 0.40 / 0.48**.
- A donor that receives a mean-shifted prior will often recover part of the gain. Attribution is
  therefore likely underpowered even when superiority over G+V is not.

**Harm guard.** It is well defined, it can fail, it is leak-free, and it is honest that it cannot
certify safety. Its precision depends on query count:
- With per-query SD s of the decile-mean paired difference **[assumed: 0.05–0.10]** and a
  Bonferroni two-sided 95% interval over K = 10, the half-width is 0.008–0.016 at 300 queries and
  0.020–0.040 at 50 queries **[mine]**.
- "Every simultaneous lower bound above −0.010" therefore needs a point gain of about −0.002 to
  +0.006 on a 300-query family, and +0.010 to +0.030 on TREC-COVID- or Touché-sized families.
- Panel-wide exclusion of harm will essentially never be established. The text already treats it
  as unresolved risk, which is the right reading.

**Estimand, pairing, multiplicity.** Sound:
- per query and seed, then decile mean, then seeds and queries, corpora within family, families;
- exact float64 scoring with fixed tie order and parity at 1024;
- fixed-sequence primary chain, Holm within prespecified secondary families, Holm jointly across
  budgets;
- query-level tests are diagnostics only.

**Leakage in §6.** None found. Two clarifications are needed:
- P1 needs non-evaluation calibration queries, and test-only collections have none unless the
  test-query set is hash-split into an unscored calibration part and a scored part. §6 should say
  which applies (A8).
- The meta-validation families' training splits must be excluded from meta-train (§2.1, row G).

---

## 3. The four OPEN-for-sign-off items

### 3.1 Budget-profile parameterization (HyperPrior)

**Position: SIGN the offset-plus-slope form aₜᵢ = max(0, a⁰ᵢ + u + tᵢv), tᵢ = (i−5)/4, with the
offset-only model nested inside it, on four conditions.** A4 is blocking.

Why the form is acceptable:
- The S4 evidence is a budget *shape* difference: flat at about 0.95 on MS MARCO, and decaying to
  exactly 0 by k = 716–819 on SciFact and NFCorpus.
- A clipped line in weight space with W, b frozen can express both shapes. It keeps exact zeros
  reachable and reproduces G+V at u = v = 0.
- F2 measures this family's attainable ceiling, and F9 measures what it misses. That makes the
  restriction testable rather than assumed.

The four conditions:
1. **The capacity must match K_train (A4, blocking).**
   - Stage B fits C from cross-fitted pseudo-targets. With K_train = 5, an outer fold has 4 source
     families, so the map is fitted on **4 (φ, target) pairs**.
   - C[3,2] has 6 coefficients, and λ is also selected inside those 4 points. That is an
     interpolation, not a regression. Even with 3 outputs fixed to offset-only it fits 3
     coefficients to 4 points.
   - Rule: fix the number of descriptor features at Gate A from the audited count, as
     p = min(3, ⌊(K_train − 1)/3⌋). That gives p = 1 at K_train = 5–6, p = 2 at 7–9, and p = 3 at
     ≥ 10.
   - Add features in a preregistered order: log participation ratio, then log N, then anisotropy.
   - Geometry goes first so that the p = 1 model is not identical to the size-only rival, which
     would make D's "beats size-only" comparison vacuous.
   - Fix λ by rule when p = 1 (for example, the LOO-optimal λ over the whole outer pool), because
     inner selection on 3–4 points is noise.
2. **Fit to the attainable F2 oracle, not to projected F9 weights.**
   - §4 step 2 says "the attainable oracle's projected nine weights". It should name F2 (F1 for the
     offset-only model).
   - Fitting a two-number family to F9 targets puts weight on budget interactions that the family
     cannot express, and on flat regions where F9 weights are unidentified.
3. **Give HyperPrior and its matched global a shared intercept.**
   - The standardized φ has mean zero, and C has no intercept. The family-balanced mean oracle
     offset (u₀, v₀) is therefore left at the pooled a⁰.
   - The §6 matched global ("refit its global budget weights") can capture that mean offset, so it
     can beat HyperPrior even when φ is informative. Or, if a⁰ is already near-optimal, the matched
     global ≈ G+V and ordered test 2 becomes a repeat of test 1.
   - Define the HyperPrior matched global as the same clipped F2 form with (u₀, v₀) fitted to the
     family-balanced mean of the oracle targets and C = 0. Define HyperPrior as that model plus Cᵀφ.
   - Test 2 then isolates exactly the descriptor slopes (A5).
4. **Choose offset-only versus offset-plus-slope inside each outer fold** by the stated "< 0.005
   increment" rule, and test only the output of that procedure (A2). The permutation null already
   repeats the choice inside each shuffle.

### 3.2 What a failed G3 rerun kills

**Position: SIGN the compromise. G3 failure closes bias and rank in this run, and the tied-bias
retrieval diagnostic is still reported, but it cannot escalate.** Three conditions:
1. **Register the diagnostic in full before G3 runs:** the Xw map, the LOFO retrieval endpoint, the
   donor and shuffle contrasts, and thresholds identical to Gate F. A diagnostic specified after a
   G3 failure is a forking path.
2. **Run it only if Gate C licensed the bias**, meaning an attainable Xw increment ≥ 0.005 over the
   accepted lower rung. If the oracle says there is nothing to predict, a predictability diagnostic
   has no decision value.
3. **"Independent families" for any follow-up ADR** means families not used anywhere in this run:
   not meta-train, not meta-val, not meta-test. A follow-up that reuses this run's meta-train
   families would be a second look at the same data.

Reasons:
- G3 is the only registered test of whether descriptor rows predict coordinate departures. The
  round-1 defence of its failure was power, and a rerun on fresh families removes that defence.
- Opus-E's point stands that mask-frequency agreement is a weak proxy for retrieval (E1-c). That is
  why the diagnostic is worth *reporting*.
- But allowing a positive diagnostic to override a failed registered gate within the same run gives
  coordinate models two chances to pass. That inflates the false-escalation rate of the bias rung,
  roughly doubling it when the two tests are weakly correlated.

### 3.3 Claim wording at 8–12 families

**Position: SIGN the panel-scoped confirmatory claim, conditional on the registered simulation
passing. Otherwise the result is estimation.** The wording should be fixed in the evaluation spec:

> "On the K audited evaluation families listed in [manifest hash], the locked P0 arm improved the
> equal-family mean of decile-averaged nDCG@10 over G+V by X (95% sign-flip-inverted interval [L, U];
> exact one-sided family sign-flip p = …, under the assumption that family effects are symmetric).
> m of K families had a positive point estimate. The 95% REML prediction interval for a new family
> is [a, b] (unstable at this K). This study does not estimate the effect on corpora outside this
> panel."

Prohibited at 8–14 families:
- "generalizes to unseen corpora";
- "the hypernetwork transfers";
- any statement about a minimum effect of 0.005;
- any non-inferiority statement;
- "safe to deploy".

Why a panel claim follows from this test:
- For a *fixed* panel, the null is that the panel mean is zero, and family effects may be
  heterogeneous.
- The family sign-flip reference distribution includes between-family spread, so it is
  **conservative** for that null.
- A pass therefore supports the panel statement under symmetry, and the claim needs nothing more.
- Transport beyond the panel needs a sampling argument that the label-driven allocation (§9,
  selection into the pool) does not supply. That is why ≥ 15 families alone should not unlock it.
- The ≥ 15 rule should also require a sampling frame registered at Gate A.
- "Replicated in m of K" is a description, never a significance claim.

### 3.4 Protocol order (P0 vs P1) and ordered attribution tests

**Position: SIGN P0 primary. I change my round-2 position ("P1 first if eligible").**

Reasons:
1. **P1 eligibility is thin on meta-test.**
   - The synthesis requires 500 *non-evaluation* calibration queries per corpus.
   - Test-only collections have no non-evaluation queries, except by hash-splitting the test-query
     set into an unscored calibration part. That is impossible for 49–50-query sets (Touché,
     TREC-COVID) and costly elsewhere.
   - P1's eligible K would be well below P0's. A P1-primary test would be the less powered one.
2. **The protocol must be fixed before outcomes.** P0 is the only protocol every recipient
   satisfies, so choosing it requires no data-dependent eligibility rule.
3. **P0 is the broader deployment claim.** As Astra argues, a P1-only gain supports adaptation to a
   query population, which is a narrower claim.
4. **P1's value is preserved** as a secondary paired P0-versus-P1 contrast on the P1-eligible subset,
   reported as estimation and stated with its reduced K.

**Ordered attribution: SIGN the fixed sequence G+V → matched global → donor, stopping at the first
failure, with A5.**
- Ordering by expected power, highest first, is correct for a fixed sequence.
- Without A5, the second step is either redundant (matched global ≈ G+V) or unfair to HyperPrior
  (the matched global captures an intercept that HyperPrior cannot).
- I accept that attribution will usually be underpowered. Joint power is 0.34–0.48 at K = 8–12 if
  the donor recovers half the gain (§2.2). The registration should report that number rather than
  imply that attribution is the expected outcome.
- Incremental claims of a locked bias arm over HyperPrior are secondary (Holm), not part of the
  chain.

---

## 4. Amendments (most important first)

Quoted "old" text is from `round3-synthesis.md`.

**A1 (blocking). §6 "Statistics and power", null-size criterion.**
- Old: "Require null rejection ≤0.055 and ≥80% power for the primary test at +0.010 under the
  conservative development SD."
- New: "Require null rejection ≤0.055 under the registered **symmetric** null scenarios (normal and
  symmetric heavy-tailed family effects), and ≥80% power for the primary test at +0.010 under the
  conservative development SD. Simulate and report the size under registered **asymmetric** nulls:
  left skew with skewness 1 and 2, and a one-family-carries-the-effect mixture. These are not
  pass/fail criteria, because no test of the mean holds size under left skew at K ≤ 12. The claim
  states the symmetry assumption. Any deployment recommendation additionally requires that the
  family harm guard finds no significant harm."
- Why: as written, the criterion either cannot be met (left-skew size is 0.085–0.137 at K = 8–12,
  §2.2) or is met by picking mild scenarios.

**A2 (blocking). §5 Gate D, the Holm step.**
- Old: "Keep the offset-only model if the slope adds <0.005 held-family gain. Apply the same
  qualifying tests to the retained model (use F1 for its recovery). Repeat the offset/slope selection
  inside each shuffle and Holm-adjust the two family sign-flip screens."
- New: "Choose offset-only versus offset-plus-slope inside each outer fold, keeping offset-only if
  the slope adds <0.005 inner held-family gain. Apply the qualifying tests once, to the held-family
  output of this whole procedure, using F1 or F2 recovery according to each fold's choice. Repeat
  the choice inside each shuffle. No multiplicity adjustment is needed, because one procedure is
  tested. Register the operating characteristic: at K_train = 5 the exact sign-flip passes only when
  all five families are positive (minimum p = 1/32)."
- Why: with Holm, the gate cannot pass at the K = 5 floor (1/32 > 0.025), and it costs about half
  its power at K = 6 (0.34 versus 0.64).

**A3 (blocking). §5 Gate B, the action column.**
- Old: "An upper one-sided 95% bound below 0.010 ends the large-gain program. Other failures leave
  headroom unresolved."
- New: "Proceed to C only on a pass. On any failure, retain G+V and stop escalation in this run.
  Record the failure as *conclusive* if the upper one-sided 95% bound is below 0.010, and as
  *imprecise* otherwise. Reopening after an imprecise failure requires additional families and a new
  ADR."
- Why: a gate whose failure branch has no action invites a post-hoc choice.

**A4 (blocking). §3 HyperPrior row and §4 step 2, capacity against K_train.**
- Old (§3): "Standardized φₜ[3]. Generate u,v = Cᵀφₜ, C[3,2]."
- New (§3): "Standardized φₜ[p] with p = min(3, ⌊(K_train−1)/3⌋), fixed at Gate A from the audited
  count. Features enter in the preregistered order: log participation ratio, then log N, then
  anisotropy (for P1, top-one cosine replaces anisotropy). u = u₀ + c_uᵀφₜ and v = v₀ + c_vᵀφₜ,
  where (u₀, v₀) is the shared intercept of A5."
- Old (§4 step 2): "Fit C by equal-corpus least squares to the attainable oracle's projected nine
  weights plus an L2 penalty."
- New (§4 step 2): "Fit (c_u, c_v) by equal-corpus penalized least squares to each pseudo-target
  family's attainable F2 oracle (F1 for offset-only), through the same projected forward equation.
  When p = 1, fix λ by leave-one-family-out over the outer pool instead of an inner search."
- Why: at K_train = 5, each outer fold fits 6 coefficients plus λ to 4 pseudo-target pairs (§3.1).

**A5. §6 comparator contracts, "Matched global" row (HyperPrior case).**
- Old: "Give G+V the same additional optimization budget. Refit its global budget weights."
- New: "For HyperPrior: the same clipped offset-plus-slope family with (u₀, v₀) fitted, by the same
  procedure and data, to the family-balanced mean of the source oracle targets, and C = 0. HyperPrior
  is this model plus the descriptor slopes. For bias and rank: [existing template text unchanged]."
- Why: this makes ordered test 2 isolate the descriptor slopes exactly. Otherwise it is either
  redundant with test 1 or biased against HyperPrior (§3.1, condition 3).

**A6. §5 Gate G, and the power simulation in §6.**
- Old (G): "require positive mean increment over its predecessor and no family point loss >0.010
  versus G+V."
- New (G): "require no family point loss >0.010 versus G+V. Report the mean increment over the
  predecessor without a threshold."
- Add to §6: "Report the probability of passing Gate G and the primary test jointly."
- Why: with K = 2, the mean clause removes half of null methods, but the primary test already
  controls type I. It also kills 13% of +0.010 methods, against 7% for the harm clause alone (§2.1).

**A7. §1, last line.**
- Add: "Under the conservative-SD rule, 80% power at +0.010 needs a development point SD of at most
  about 0.007 (K = 8) to 0.009 (K = 12). The expected outcome is therefore an estimation-level
  meta-test. Resources and claims are planned for that case."
- Also in §6: report power at the locked candidate's outer-fold LOFO estimate as well as at +0.010.
  If the former is below 0.5, say that the test is low-powered for the plausible effect.

**A8. §6 "Pool and split", P1 calibration on test-only collections.**
- Add: "For a test-only collection, P1 calibration queries may be a hash-selected subset of its test
  queries fixed at Gate A. Those queries are never scored. Both P0 and P1 are scored only on the
  complement. Collections with fewer than 500 + 200 test queries are P1-ineligible."
- Why: without this sentence, P1's meta-test eligibility is undefined. AGENTS.md bans labels in
  descriptors, not unlabeled query text, and `protocols.md` leakage rule 1 is satisfied because no
  P1 path reads qrels.

**A9. §10, "one protocol ADR".**
- Add: "The ADR records two departures from `docs/protocols.md`. (i) P0, not P1, is the primary
  protocol (protocols.md: 'P1 is expected to be the main protocol'). (ii) Architecture and
  hyperparameters are selected by meta-train LOFO, and meta-validation is a harm screen only
  (protocols.md: 'Architecture and hyperparameters are chosen on meta-validation')."
- Why: both departures are justified (§3.4; meta-validation has 2 families). Both must still be
  recorded, or the plan silently contradicts the project protocol.

**A10. §5 Gate A.**
- Old: "Require ≥5 training families for formal development screens and two eligible validation
  families."
- New: "Require ≥7 label-bearing families: ≥5 for meta-train after reserving two for
  meta-validation. Exclude the meta-validation families' training splits from Stage A/B, mining and
  descriptor moments."

**A11. §6 "Claim level".** Adopt the wording in §3.3.
- Add: "Transport at ≥15 families also requires a sampling frame registered at Gate A."

---

## 5. Simplification: complexity that changes no decision

**S1. Cut the rank-four rung from this run.**
- What to cut: the Fr4 oracle in step one and Gate B, the source-basis and fold-basis variants, the
  §3 rank row, rank costs, the rank acceptance tests, and Gate F's rank clause.
- Why: Gate F requires ≥ 20 environments in ≥ 15 independent training families, against a pool that
  everyone estimates at about 4–6 training families after reserving meta-validation. No Fr4
  measurement can therefore change a decision.
- Replace all of it with one line: "Rank four is deferred; reopen only with ≥ 15 audited training
  families under a new ADR."
- Opus-E's registered prediction concerns R_F9 and R_Fb, so nothing in that prediction is lost.

**S2. Drop the ICC in Gate C. Keep held-half utility regret as the stability criterion.**
- At K = 5–8 the ICC is imprecise, its units are undefined, and on flat surfaces it measures
  coefficient instability that does not matter for retrieval.
- Gate D tests held-family utility directly, so a target too unstable to learn already fails there.
- For coordinate outputs, keep the centered-logit agreement on probe queries as a descriptive
  check.

**S3. Remove "Feature-augmented global" from the comparator table.**
- §6 itself says it "is algebraically the cached linear bias". It is a parity unit test, not a
  comparator.
- Move it to the acceptance tests in §10, which already list "feature/bias parity".

**S4. Defer the P1 departure rows (h = 43) and the P1 bias contract** until the bias rung unlocks
*and* at least 5 meta-train families are P1-eligible. HyperPrior's P1 variant needs only the single
top-one-cosine swap.

**S5. Drop the family-uniform versus corpus-uniform sampler sensitivity in §4 step 1**, unless the
audit finds a multi-corpus family. If every family has one corpus, the two samplers are identical.

**S6. Drop the positive-mean clause of Gate G (A6).**

Kept despite their cost, because each can change a decision:
- PRF-DIME and Eclipse, and offline pseudo-query tuning (they decide G5 and the amortization claim);
- nearest-corpus and size-only rivals (they decide attribution);
- the MS MARCO topical shards (cheap, and they separate the corpus mechanism from the query
  mechanism);
- the three-seed average.

---

## Sources consulted this round

- Round files: `BRIEF.md`, `round0-seed.md`, `round1-{astra,opus-e,opus-m}.md`,
  `round2-{astra,opus-e,opus-m}.md`, `round3-synthesis.md`. No `council-*.md` file was opened.
- Repository: `docs/protocols.md:20-44` (split roles, P0/P1/P2, leakage rules) and `AGENTS.md:15`
  (hard constraint).
- Computations: `/tmp/hm-r4/power.py` and `/tmp/hm-r4/skew.py` (exact sign-flip power and size,
  SD confidence-limit multipliers, gate and meta-validation operating characteristics, joint
  attribution power, harm-guard half-widths). Project venv, not committed.
- Papers: none newly opened. No paper claim in this file depends on a new reading.
