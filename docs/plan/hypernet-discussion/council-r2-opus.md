# Council round 2 — Claude Opus 5.5

Reviewer: council member Claude Opus 5.5 (round-1 file `council-opus.md`, score 6.5/10).
Document reviewed: `PROPOSAL.md` (merged round-3 synthesis + round-4 sign-offs + council reviews).
Also read: `council-opus.md`, `council-astra.md`, `round3-synthesis.md` §5–§7, `round4-opus-e.md`
§3.1/§4/§5, `round4-opus-m.md` §2/§4, `COUNCIL-BRIEF.md`, `docs/protocols.md`, and the repo
files cited below. Read-only except this file; no runs, downloads or commits.

Tags: **[V]** read in the file this session (path:line); **[C]** my own computation (numpy via the
project venv, scripts in `/tmp/council-r2-opus/`, not committed); **[I]** interpretation.

---

## 0. Bottom line

- **Score: 8.0 / 10** (round-2 consensus 6.5; I had predicted about 7.5 for round 3).
- The merge fixed most of what I asked for: the meta-train screens are no longer unanimity tests,
  p is tied to K_train (so p = 1 at the expected K_train = 5–6), a matched global isolates the
  descriptor slopes, the P1 query-centred prior is in, the exposure tags are known/unknown, the ICC
  has been replaced by split-half regret, the meta-validation families are label-bearing, and the
  new multi-corpus trainer is named as new work.
- **Gate B now matches my futility/learnability intent**, and my simulation says its delete-one
  rule tracks "≥ 2 real headroom families" better than my own round-1 wording did (§4).
- I **accept both overrulings** (log N first; non-inferiority first), but the stated reason for
  the second is technically wrong and should be replaced (§2).
- I **accept the orchestrator's K_train ≥ 8 rule in principle**. It has three defects that matter:
  1. the permutation test's resolution is misstated;
  2. with p-values off, Gate D has no check against the matched global, so an intercept-only gain
     can pass development and then fail at meta-test comparator 2;
  3. the point no-loss rule that now decides Gate D kills 15–30% of good concentrated methods (§3).
- **Largest new problem: feasibility of the meta-test count.** Gate H plans 8–12 meta-test
  families, but §6 nominates six evaluation-only families, three of them licence-gated. The
  overlap rule may also merge the development families (BioASQ ⊃ PubMed) (§5, P1).

---

## 1. Status of my round-1 must-fix items

| # | Round-1 must-fix | Status | Where in PROPOSAL.md | What remains |
|---|---|---|---|---|
| 1 | Concentration-aware power; non-inferiority (NI) first; enrichment readout | **Partly** | §6 Power l.202-213 (concentrated scenario simulated and registered, power 0.20–0.47); §1.6 and A7 (estimation by default); §9 l.275 (NI overruled) | The **primary-test requirement (l.210-211) still uses only the homogeneous scenario**. The concentrated scenario is "simulated and registered" with no stated consequence. The label-free "predicted movers" enrichment readout is absent. |
| 2 | Futility-only meta-train screens with a "≥ 2 headroom families" learnability rule | **Largely resolved** | Gate B l.148 (point rules, no significance over families); Gates D and F l.150-152 and §9 l.277 (p-values non-binding at K_train < 8) | Gate B's failure classification ignores the concentration branch, and §7 has no row for it. The point no-loss rule in D has an unregistered 15–30% false-kill rate. There is a discontinuity at K_train = 8 (§3, §4). |
| 3 | One pre-registered signed feature; a matched comparator that absorbs the global offset | **Resolved in substance** | §3 l.67 (p = min(3, ⌊(K_train−1)/3⌋), so p = 1 at K_train = 5–6); l.70-75 (feature order, predicted signs at Gate A); §6 l.178-180 (matched global, A5) | Offset + slope was kept over offset-only. I accept that because of the in-fold fallback, but §9 justifies it with the wrong evidence (§5, P6). |
| 4 | Task type as a second axis; a task-absent sensitivity analysis; a 1-NN-task control | **Partly** | Gate A l.147 (recorded); §6 l.226-233 (reported); §8 l.254-255 | No **task-absent sensitivity** and no **1-NN-task rival**. Stratifying the meta-train/meta-test split by task is impossible, because label availability decides the split. That makes the sensitivity analysis the only remaining defence. |
| 5 | Exposure as known/unknown; ≥ 2 not-known-exposed meta-train families; stratified census | **Resolved** | Gate A l.147; §6 l.228; §8 l.256-258 | The ≥ 2 requirement is met automatically by SciFact and NFCorpus. Make it explicit in Gate A (one clause). |
| 6 | Zero-parameter P1 query-centred prior | **Resolved** | §5 l.119-120; §6 l.190 | Nit: the formula lacks the ε = 10⁻¹² used everywhere else. |
| 7 | ICC replaced by split-half transfer; R_j per headroom family | **Mostly** | Gate C l.149 (S2); step one l.128 | Gate C's "attainable oracle increment ≥ 0.005" has no stated aggregation (family mean or per headroom family). A family mean dilutes one-family headroom as K grows. |
| 8 | Restate the prior-share evidence (MS MARCO step-0 caveat); size the first step honestly | **Resolved** | §2 l.44; §10 l.294-295 | Wording nit on "barely leave initialization" (§5, P8). |
| 9 | Keep test-only families for meta-test; label-bearing meta-validation | **Resolved** | Gate A l.147 (A10); §6 l.163-166 | Two meta-validation families instead of my one. At K_train = 5–6 this does not change p, so I accept it. |
| — | (I6/M4) G+V seeds; donor averaged over all donors | **Not resolved** | §4 gives no seed count; §6 l.181 uses one hash-assigned donor | See must-fix items 9 and 10. |

---

## 2. The two overruled items: accept or dissent

**(a) log N as the single first feature: I accept the overruling.**
- The reason holds once it is stated precisely. At p = 1, a log-N HyperPrior is the size-only
  rival, so Gate D's "at least the utility of the size-only rival" (l.150) could never
  discriminate it.
- My own evidence for log N was weak. Its directional basis was the MS MARCO end of the prior
  share, which I showed in round 1 to be a step-0 / lr-1e-4 artifact. `kept_steps.csv:2-7` [V]
  still shows rerank-best step 0 in all six MS MARCO rows.
- Two conditions for the acceptance (must-fix item 8):
  1. **The participation ratio must not be a size proxy by construction.** The sample
     tr(S²) is biased upward by about (tr Σ)²/N, so the plug-in PR ≈ 1/(1/PR + 1/N) [C, Gaussian
     approximation]. The bias is small when PR ≪ N, but NFCorpus (N = 3,633) and SciFact
     (N = 5,183, `reports/P3/README.md:46-48` [V]) are exactly where it is largest. Compute the
     features on a fixed-size seeded document subsample (for example the smallest N in the pool),
     or use an unbiased tr(Σ²) estimator.
  2. **The size-only rival must go through the identical Stage B pipeline**: same
     offset/slope family, same in-fold choice, same ridge rule. §6 l.184 does not say so.
     round3-synthesis did.
- §9's wording "so it could never beat it" should read "it would be identical to the size-only
  rival, so Gate D could not separate geometry from size".

**(b) Non-inferiority at δ = 0.010 as the first confirmatory step: I accept the outcome, not the
stated reason.**
- §9 l.275 says NI was dropped because "the sign-flip test of superiority does not support it".
  That is not correct. A shifted sign-flip tests H₀: μ ≤ −0.010 with the same machinery, and §6
  l.193-194 already inverts shifted tests for location intervals. The NI bound is therefore
  available as an estimate at no cost.
- The reasons that do hold, and that now persuade me:
  1. **NI is nearly automatic for a nested model.** HyperPrior is initialized at G+V (c = 0,
     l.77-78), and ridge shrinkage pulls it towards G+V. A map that learned nothing passes NI at
     0.010. The test would certify the shrinkage, not the method.
  2. **NI is a safety-flavoured claim, and the test's size is worst exactly under the
     safety-relevant null.** Under left-skewed nulls the rejection rate is 0.084–0.137 at K = 8–12
     (`round4-opus-m.md` §2.2). The per-family harm guard (§6 l.223-225) is the right instrument
     for safety. A mean NI test is not.
- My round-1 concern, that a "fail" in the concentrated world would be misread as refutation, is
  now handled by estimation by default (§1.6, A7). It still needs the primary-test requirement to
  name the concentrated scenario (must-fix item 3).
- Replace the §9 reason accordingly (must-fix item 12).

---

## 3. The orchestrator decision: development p-values bind only when K_train ≥ 8 (§9 l.277, Gate D l.150)

**The principle is right, and I accept it.**
- The locked meta-test is single-shot, with one candidate and a fixed comparator sequence. Its
  type I error does not depend on whether the development screens used p-values. The screens are
  resource-allocation decisions.
- At K_train = 5 the exact sign-flip needs all five families positive (minimum p = 1/32), which
  is the unanimity test I objected to in round 1 (C2).
- Turning those p-values into reports is the futility-first design I asked for.

**Defects to fix.**

1. **The permutation test's resolution is misstated.** l.150 says "min p = 1/2^K" for both
   tests. That holds for the sign-flip. The descriptor-permutation test permutes K descriptors
   over K families, so its resolution is 1/K!: 1/120 at K = 5 and 1/720 at K = 6.
   - The full refit is cheap here. The family oracles (u\*, v\*) and the bases do not depend on
     φ, so each permutation reruns only the Stage B ridge and the in-fold choice.
   - Its power at K = 5 is still low. A one-sided p ≤ 0.05 needs a near-perfect rank association
     (Spearman ρ ≥ 0.9 at n = 5) [C].
   - So keeping it non-binding is defensible, but the stated reason is wrong. Give each test its
     own resolution.
2. **Gate D lost its only attribution check.**
   - With the p-values non-binding, D compares HyperPrior with G+V, the oracle, size-only and
     nearest-corpus (l.150), but **not with the matched global** (§6 l.178-180).
   - HyperPrior contains the shared intercept (u₀, v₀). If the frozen base's a⁰ is miscalibrated
     for held-out families in general, the intercept alone can deliver ≥ 0.005 over G+V and pass
     D. The candidate then fails comparator 2 at meta-test, after spending the only meta-test.
   - That is a leaky gate in the resource sense, and matched-global predictions are cheap.
   - Add a point criterion: mean gain over the matched global > 0 and positive on at least half
     the families with H_f ≥ 0.005.
3. **The point no-loss rule now decides Gate D, and its operating characteristic is unregistered.**
   Simulation [C] (`/tmp/council-r2-opus/sim2.py`, 200,000 draws):
   - Setup: query SEs of 0.008/0.005/0.002/0.006… (SciFact-, NFCorpus- and MS MARCO-sized
     untouched partitions); non-mover true effects N(−0.001, 0.003).
   - "No family's point estimate < −0.010" **kills a genuinely good concentrated method 23% (K = 5)
     to 30% (K = 6)** of the time with one mover at +0.03, and 15–23% with two movers at +0.02.
   - Alternatives on the same scenarios:

   | Rule (K = 5) | false kill, 1 mover +0.03 | false kill, 2 movers +0.02 | detects one family at −0.020 | detects one family at −0.015 |
   |---|---|---|---|---|
   | point Δ_f < −0.010 (current) | 0.23 | 0.15 | 0.96 | 0.83 |
   | query-level one-sided 95% upper bound < −0.005 | 0.08 | 0.06 | 0.82 | 0.54 |
   | point Δ_f < −0.015 | 0.05 | 0.03 | 0.80 | 0.52 |

   - Gate G and the meta-test harm guard sit downstream, so the development screen can afford the
     SE-aware rule. Either way, register the operating characteristic.
4. **A discontinuity at K_train = 8.** At exactly 8 the gate reverts to a sign-flip whose power
   under the concentrated scenario is 0.20–0.27 (my round-1 table).
   - The expected pool cannot reach K_train = 8 anyway: at most 8 label-bearing families minus 2
     for meta-validation (§5, P1).
   - Simpler and consistent: **development p-values are never binding in this run**. They are
     reported with their resolution. Type I error is controlled only at the locked meta-test, or
     not at all if K_test < 8 (estimation).
5. **The single-shot property is asserted but not written down.**
   - round3-synthesis §8 had "Meta-test outcomes cannot trigger a second selection on the same
     families". That sentence did not survive the merge.
   - It is the property the orchestrator's argument depends on. Restore it in §7.

---

## 4. Does Gate B now match my futility recommendation?

Gate B (l.148): mean H ≥ 0.010, **and** ≥ 2 families with H_f ≥ 0.005, **and** every delete-one
mean ≥ 0.005. There is no significance over families.

**Simulation [C]** (`/tmp/council-r2-opus/sim.py`, 200,000 draws):
- The development three are fixed at Opus-E's registered point predictions (H = 0.10 / 0.006 /
  0.003, `round4-opus-e.md` §5 E1–E3), with query SEs of 0.008 / 0.005 / 0.002.
- Each new family has headroom U(0.015, 0.05) with probability π, and N(0.002, 0.004) otherwise
  (SE 0.005).
- The "ideal" pass rate is the probability that at least one new family has real headroom, so
  that two headroom families exist and LOFO can learn: 1 − (1 − π)^(K−3).

| K | π | ideal (≥ 2 real headroom families) | **PROPOSAL rule** | without delete-one | delete-one > 0 | my round-1 rule (≥ 2 with H_f ≥ 0.010 and LB > 0) |
|---|---|---|---|---|---|---|
| 5 | 0 | 0 | 0.25 | 0.84 | 0.80 | 0.37 |
| 5 | 0.25 | 0.44 | 0.56 | 0.91 | 0.88 | 0.64 |
| 5 | 0.5 | 0.75 | 0.79 | 0.96 | 0.95 | 0.84 |
| 6 | 0.25 | 0.58 | 0.63 | 0.95 | 0.93 | 0.76 |
| 8 | 0 | 0 | 0.15 | 0.94 | 0.87 | 0.55 |
| 8 | 0.25 | 0.76 | 0.71 | 0.99 | 0.97 | 0.89 |

**Reading [I].**
- **Yes, it now matches the futility/learnability intent, and better than my own round-1 rule.**
- The delete-one ≥ 0.005 clause does the work. It asks whether headroom exists beyond the single
  largest family, which is the learnability condition.
- The "≥ 2 families with H_f ≥ 0.005" clause is nearly redundant: NFCorpus at a true +0.006
  satisfies it by noise. My round-1 "≥ 2 with LB > 0" rule passes too often in the one-family
  world (0.37–0.55 at π = 0), for the same reason.
- I withdraw my wording in favour of PROPOSAL's.

**Remaining defects in Gate B.**
1. **The failure classification ignores the most likely failure.**
   - Everyone's prior is that SciFact carries the headroom. Then mean H ≫ 0.010 (about 0.02–0.03)
     while delete-one fails.
   - The action column (l.148) classifies failures only by the upper bound of mean H. This failure
     would be recorded as "imprecise", which misstates it.
   - The bound's method is also unspecified (sign-flip inversion or t).
   - Add a third class, **"concentrated: headroom in one family; a map cannot be learned"**, and a
     matching §7 row. That row should publish per-family headroom and the F-ladder for the
     headroom family as the deliverable.
2. The bound for "conclusive" needs a named method: family-level one-sided t, with sign-flip
   inversion reported.

---

## 5. New problems introduced or left by the merge

**P1. The meta-test target of 8–12 families is not supported by the pool as nominated (Gate H
l.154 vs §6 l.161-162).**
- §6 lists six evaluation-only candidates: ArguAna/Touché, SCIDOCS, TREC-COVID,
  Robust04/TREC-NEWS, Signal-1M and CQADupStack. Robust04, TREC-NEWS and Signal-1M need licences
  [I, BEIR distribution].
- The overlap rule (≥ 1% of the smaller collection, connected components, Gate A l.147) can only
  lower the count.
- **It may also collapse meta-train** [I, to be verified at Gate A]:
  - NFCorpus documents are PubMed articles, and SciFact abstracts are well-cited biomedical
    papers, so BioASQ's PubMed corpus very likely overlaps each by ≥ 1%.
  - Adding BioASQ as a label-bearing family would then merge NFCorpus, SciFact and BioASQ into
    **one** family, reducing K_train rather than raising it.
  - SCIDOCS (Semantic Scholar) against SciFact (S2ORC) needs the same check.
- Consequences:
  - The realistic outcome is K_train ≈ 4–6 and K_test ≈ 4–7. The 8-family confirmatory branch and
    the K_train ≥ 8 p-value rule are then both dormant.
  - §1.6 already expects an estimation study. Gate H and §6 should still say plainly that K_test
    < 8 is the expected case.
  - Gate A should state that a candidate whose inclusion merges existing families is excluded,
    not merged.

**P2. No equal learning-rate grid across arms (§4 l.85-90, §5 l.115-120).**
- ADR-0020 found that the learning rate decides contrasts once other effects are null.
  It tunes lr per arm on an equal grid {3e-5, 1e-4, 3e-4, 1e-3}
  (`docs/decisions/ADR-0020-p3-objective-study.md:40,85-88` [V]).
- The S4 cells chose lr 1e-4 for MS MARCO and 1e-3 for the other two
  (`docs/decisions/ADR-0012-s4-budget-prior.md:36` [V]).
- The multi-corpus G+V needs one lr, and the specialist reference uses the in-domain one. An
  unequal grid moves H, and therefore Gate B, directly.
- PROPOSAL never states the lr rule. My round-1 §4.2 rule was dropped.

**P3. The "untouched" step-one score partition has already been used for recipe selection
(§5 l.110-114).**
- The S4 cells (lr, temperature, b) were selected on each development corpus's `validation`
  split (ADR-0012:36, `reports/S4/README.md:9` [V]). The specialist's recipe is therefore tuned on
  the same queries that score H.
- The bias is probably small, and the step is exploratory.
- Required changes:
  - call it "untouched by step-one fitting; reused for S4 recipe selection";
  - select every step-one arm's lr on the fit/checkpoint partitions only (P2).

**P4. Stage A/B compute is not planned (only step one has N7, l.130-133).**
- Stage B needs K_train outer folds × (1 + (K_train − 1) inner refits) base fits. At K_train = 5
  that is 25 multi-corpus S4 fits, times seeds and the lr grid. Each includes MS MARCO checkpoint
  selection by exact hard-mask retrieval over 8.8M passages (§4 l.88-89).
- That can be 300 trainings before any HyperPrior is fitted.
- The required resource plan should register:
  - lr chosen once per outer fold and reused for its inner refits;
  - one seed for inner refits and three for outer bases;
  - the MS MARCO checkpoint-selection queries capped (about 2,000, as in step one).

**P5. The P1 feature is dormant at the expected K (§3 l.70-75).**
- Under P1 the calibration-query feature replaces **anisotropy**, which enters only at p = 3,
  that is at K_train ≥ 10.
- At the expected K_train = 5–6 (p = 1), the P1 HyperPrior is therefore identical to the P0 one.
  The "P1 is compared on identical eligible corpora" (§6 l.171-172) is then a comparison of the
  P1 prior comparator only.
- Fix: either put the P1 feature first under P1, or state that at p < 3 the P1 track is just
  the query-centred prior.

**P6. §9 row 1 cites the wrong evidence for the slope (l.272).**
- "The S4 per-budget gains (+0.011, +0.009)" compare nine per-budget weights with **one
  budget-free weight** (ADR-0012:26 [V]; `round4-opus-e.md` §3.1 states the caveat itself).
  That supports a⁰ ∈ R⁹, which G+V already has. It says nothing about a per-corpus slope.
- The 90–96% vs 98–100% decomposition comes from n = 3 in-domain bases, each with its own W, in
  prior-share units. Opus-E: "says nothing about the optimum under a shared frozen W".
- The decision survives because the in-fold choice falls back to offset-only. The justification
  should say exactly that.

**P7. Gate C has no stated aggregation (l.149).**
- "Attainable oracle increment ≥ 0.005" is undefined as to whether it means the family mean or
  per family.
- As a family mean, one-family headroom of 0.03 passes at K = 5 (0.006) and fails at K = 7
  (0.004). The gate's verdict then depends on K, not on attainability.
- Specify "on ≥ 2 families with H_f ≥ 0.010", matching Gate B's learnability logic.

**P8. Evidence-table wording (§2 l.44).**
- "MS MARCO weights barely leave initialization" is contradicted by the same row's numbers.
  - The S3 final weight is 0.0161–0.0162. Only the *kept* checkpoint is step 0 (weight 1.0000),
    `reports/S3/tables/weights.csv:2-4` [V]; Astra flagged this.
  - The S4 kept weights are 0.64–1.00 across both tracks (`reports/S4/tables/weights.csv:2-7` [V];
    0.6384 in aligned seed 0; the budget track alone is 0.67–1.00).
- Retitle the row "On MS MARCO no training step improves the selection metric" and correct the
  range.

**P9. Smaller items.**
- §6 l.207-209: "left-skewed and one-family nulls (size 0.085–0.137)". The range is Opus-M's
  **left-skew** result (0.084–0.137, `round4-opus-m.md` §2.2). The one-family null was proposed
  for simulation but not computed. Say so.
- §3 l.67: at K_train ≤ 3 (the step-one pilot), p = 0, so HyperPrior reduces to the matched
  global. State this so the pilot is not read as a HyperPrior test.
- §3 l.70: write log[(tr Σ)²/tr(Σ²)] with explicit brackets.
- §6 l.181: the single hash-assigned donor. With ≤ 5 meta-train descriptors, average over all
  source-family donors (Astra and my round-1 M4). It is free for a scalar map and removes donor
  luck.
- §4: state the seed count (three per arm, averaged per query). It was in round 3 and dropped.

**Checked and correct [V].**
- `train_global.py:54-71`: concatenation and `train_selector` (L2S-KL).
- `prepare.py:107-111` and `configs/data/*.yaml:4`: `validation_fraction: 0.1`.
- `reports/E1/tables/payoff.csv:2-4`: +0.1116 / +0.0171 / −0.0019.
- `prior_share.csv`: 0.9512 / 0.4281 / 0.3580; NFCorpus 0.5008 → 0 at 716.
- `kept_steps.csv:2-7`: rerank-best step 0 in all six MS MARCO rows.
- `ablation.csv`:
  - MS MARCO −0.2007 at k = 102 and |Δ| ≤ 0.0016 at k ≥ 716;
  - NFCorpus −0.0589 at 102 and |Δ| ≤ 0.0017 from k = 409.
- ADR-0012:26: +0.0111 / +0.0094 / +0.0001.
- Power figures 0.79 / 0.90 / 0.94 (`round4-opus-m.md` §2.2).
- 0.20–0.47 (my round 1).
- A7's SD ≤ 0.007–0.009.

---

## 6. Score and the must-fix items to reach ≥ 9

**Score: 8.0 / 10.**
- The design is now sound in its logic: the comparator, the nesting, the leakage rules, the
  futility-style screens and honest claim levels.
- It falls short of 9 for three reasons:
  - one leaky development gate (D lacks the matched global);
  - unregistered operating characteristics for the rules that now decide the screens;
  - a pool-feasibility gap that makes Gate H's 8–12 an unsupported plan.
- Everything below is a text change. None needs a new run.

Must-fix items (old → new, most important first):

1. **Gate D, l.150 (attribution at development).**
   - Old: "…at least the utility of the size-only and nearest-corpus rivals; no family loses > 0.010."
   - New: "…at least the utility of the size-only and nearest-corpus rivals; **mean gain over the
     matched global (§6 comparator 2) > 0, positive on at least half the families with
     H_f ≥ 0.005**; no family with material loss (item 4)."
2. **Gate D, l.150, and §9 l.277 (p-value rule).**
   - Old: "Family sign-flip and descriptor-permutation p (full refit) are binding only when
     K_train ≥ 8; below that they are reported with their resolution (min p = 1/2^K)."
   - New: "Family sign-flip and descriptor-permutation p (full refit of Stage B) are **reported,
     never binding, in this run**, with their resolution (sign-flip 1/2^K; permutation 1/K!).
     Type I error is controlled only at the locked meta-test."
   - Add to §7: "**Meta-test outcomes cannot trigger a second selection, re-lock or re-scoring on
     the same families.**"
3. **§6 Power, l.210-211 (primary requirement).**
   - Old: "≥ 80% power at +0.010 under the conservative development SD, using ≥ 10,000 draws."
   - New: "≥ 80% power at +0.010 under the conservative development SD **and under the
     registered concentrated scenario (25% of families × 0.035) whenever the meta-train LOFO
     effects are concentrated (registered rule: the top family carries > 50% of the summed
     positive LOFO gain)**, using ≥ 10,000 draws. Otherwise the meta-test is estimation-level."
   - Add a key secondary: "**Predicted-mover readout:** before unblinding, families with
     |ĉ_uᵀφ̃| above the smallest offset whose oracle gain was ≥ 0.005 on meta-train are labelled
     movers. Report movers' effects and non-movers' harm separately."
4. **Gates D, F and G (no-loss rule).**
   - Old: "no family loses > 0.010".
   - New: "no family whose query-level one-sided 95% upper bound of Δ_f is below −0.005
     (operating characteristic registered: about 6–8% false kill for a concentrated +0.02–0.03
     method, about 82% detection of a −0.020 family at K = 5)".
   - Alternatively keep the point rule, but register its 15–30% false-kill rate.
5. **Gate B, l.148, and §7 l.239 (failure classes).**
   - Old (action): "…recording the failure as *conclusive* if the upper one-sided 95% bound is
     < 0.010, *imprecise* otherwise."
   - New: "…recording the failure as ***concentrated*** if mean H ≥ 0.010 but a delete-one mean is
     < 0.005; *conclusive* if the family-level one-sided 95% t upper bound of mean H is < 0.010;
     *imprecise* otherwise."
   - Add a §7 row: "Gate B fails by concentration | Publish per-family headroom and the
     headroom family's F-ladder; ship G+V; a corpus map is not learnable from one family."
6. **Gate A l.147, Gate H l.154 and §6 l.158-166 (pool feasibility).**
   - Add to Gate A: "Publish the expected K_train and K_test after the overlap audit and licence
     check. A candidate whose inclusion merges existing families (e.g. BioASQ's PubMed with
     NFCorpus/SciFact; SCIDOCS with SciFact) is excluded, not merged. ≥ 2 meta-train families
     must be not-known-exposed."
   - Gate H, old: "8–12 meta-test families". New: "the audited meta-test families (8–12 needed
     for §6's confirmatory branch; the nominated pool supports about 4–7, so estimation is the
     expected case)".
7. **§4 step 1 and §5 base arms (equal lr grid and compute).**
   - Add: "Every trained arm (G+V and all base arms, the specialist, oracles Fb/Xw) selects lr on
     the same grid {3e-5, 1e-4, 3e-4, 1e-3} using fit/checkpoint partitions only (ADR-0020). Three
     seeds per arm, averaged per query. Stage A/B resource plan: lr chosen once per outer fold
     and reused for inner refits; inner refits use one seed; MS MARCO checkpoint selection is
     capped at about 2,000 queries."
   - Step one, add: "`validation` is untouched by step-one fitting, but S1–S4 recipes were
     selected on it."
8. **§3 feature order, l.70-75.**
   - Add: "Features are computed on a fixed-size seeded document subsample (the pool's smallest
     N), so that the participation ratio does not inherit finite-N bias. The size-only rival uses
     the identical Stage B pipeline with φ = log N. Under P1 the calibration-query feature takes
     the **first** slot (or: at p < 3 the P1 track is the query-centred prior only)."
9. **§6 comparators, l.181 (donor).**
   - Old: "a hash-assigned different-family descriptor".
   - New: "the mean outcome over all different-family source descriptors, each keeping the
     recipient's variance".
10. **§6 Confounds, l.226-233 (task).**
    - Add: "Pre-registered sensitivity analysis: the meta-test effect on families whose task type
      is absent from meta-train (e.g. argument retrieval). Rival: 1-NN-task, the matched-global
      map refitted on the meta-train families sharing the recipient's task type."
11. **Gate C, l.149 (aggregation).**
    - Old: "Attainable oracle increment ≥ 0.005 over the accepted rung."
    - New: "Attainable oracle increment ≥ 0.005 over the accepted rung **on ≥ 2 families with
      H_f ≥ 0.010** (R_j reported per family)."
12. **§9 text corrections, l.272 and l.275.**
    - Row 1, old reason: "The S4 per-budget gains (+0.011, +0.009) and the 90–96% vs 98–100% fit
      show a single offset loses information". New: "The slope is kept because the in-fold choice
      falls back to offset-only at no cost. The S4 per-budget gains support a⁰ ∈ R⁹ (already in
      G+V), and the share decomposition uses three in-domain bases, so neither establishes a
      per-corpus slope."
    - Row 1 dissent: "could never beat it" becomes "would be identical to it".
    - NI row, old reason: "the sign-flip test of superiority does not support it". New: "NI is
      nearly automatic for a model initialized at and shrunk towards G+V, and a mean NI test has
      inflated size under the left-skewed nulls that matter for safety. The shifted-test lower
      location bound is reported as an estimate. Safety is the harm guard's job."
13. **§2 l.44, §6 l.207-209, l.190 (accuracy nits).**
    - Retitle the row and correct the S4 weight range to 0.64–1.00 (P8).
    - "left-skew nulls (0.084–0.137); the one-family null is to be simulated".
    - Add +10⁻¹² to the P1 prior formula.

**With items 1–7 I would score about 9. With all 13, about 9.5.** The remaining ceiling is not
in the text but in the world: a likely K_test < 8 caps the study at estimation, and the plan now
says that honestly.

---

## 7. Verification log

- [V] `PROPOSAL.md` l.1-317 in full; `council-opus.md` in full; `council-astra.md` in full;
  `round3-synthesis.md` l.55-148; `round4-opus-e.md` l.78-152, 237-370; `round4-opus-m.md`
  l.39-146, 279-383.
- [V] `reports/S3/tables/weights.csv:2-4`; `reports/S4/tables/{kept_steps,weights,prior_share,ablation}.csv`;
  `docs/decisions/ADR-0011-s3-prior-weight.md:25-45`; `ADR-0012-s4-budget-prior.md:26-36`;
  `ADR-0020-p3-objective-study.md:40,85-88`; `reports/S4/README.md:9,31-33`;
  `reports/P3/README.md:44-64`; `src/hyperdime/cli/train_global.py:45-90`;
  `src/hyperdime/data/prepare.py:100-115`; `configs/data/*.yaml`; `docs/protocols.md:1-60`.
- [C] `/tmp/council-r2-opus/sim.py` (Gate B operating characteristics) and `sim2.py` (no-loss
  rule variants), 200,000 draws each, numpy from the project `.venv`.
- [I] BioASQ/PubMed and SCIDOCS/S2ORC overlap risks, and the licence status of Robust04,
  TREC-NEWS and Signal-1M, come from dataset provenance knowledge. They were not verified in the
  repository and are to be checked at Gate A.
