# Round 4 · Opus-E (evidence lead) · Sign-off on `round3-synthesis.md`

Author: Opus-E (Claude Opus 5.5). Before writing I re-read `BRIEF.md`, `round0-seed.md`, all
round-1 and round-2 files and `round3-synthesis.md`. I did not open any `council-*.md` file. I
edited no other file. I trained nothing, downloaded nothing and committed nothing.

**Tags.**
- **[V]** I read the value in the named file or PDF page during this round.
- **[C]** I recomputed the value. The scripts are in `/tmp/oe4/` and are not committed.
- **[I]** My interpretation.

PDF pages count the cover page. Text was extracted with pypdf into `/tmp`.

---

## 1. Verdict

**SIGN WITH AMENDMENTS.**

Every number and citation in the synthesis that I checked is correct (§2). The staged design
follows from the evidence:
- G+V is the comparator;
- HyperPrior comes first;
- the oracle ladder runs before any generator is built;
- family is the unit of inference;
- estimation is the default.

I block on three amendments (§4, B1–B3):
- **B1.** One step-one wording opens a leakage path on MS MARCO.
- **B2.** The fitting target for the HyperPrior map would make the primary arm learn from
  coordinates of the weight curve that retrieval does not identify.
- **B3.** The claim wording at 8–14 families mixes two estimands.

Each is a text change. None changes the program.

---

## 2. Factual audit

Locations refer to `round3-synthesis.md` (`§n` and line number).

| # | Synthesis claim (location) | Check | Verdict |
|---|---|---|---|
| A1 | E1 in-domain minus union-global at k=307: SciFact +0.1116, NFCorpus +0.0171, MS MARCO −0.0019 (§2, l.21) | `reports/E1/tables/payoff.csv:2-4` **[V]** | Correct |
| A2 | 728 + 2,331 + 45,000 training queries; MS MARCO 45,000/48,059 = 93.6% (§2, l.22) | `reports/P3/README.md` coverage table **[V]**; 0.9363 **[C]** | Correct |
| A3 | Default trainer concatenates every environment's training tensors (§2, l.22) | `src/hyperdime/cli/train_global.py:54-60` **[V]** | Correct, but incomplete. The legacy union is an **L2S-KL** selector (`targets_at`, `train_selector`) with no variance prior (`:56-71`) **[V]**. Legacy union → LOCO balanced G+V therefore changes four things at once: objective, prior, balancing and recipient exclusion. See amendment N3. |
| A4 | S4 budget-track prior share at 307: MS MARCO 0.9512, NFCorpus 0.4281, SciFact 0.3580; NFCorpus 0.5008 at 102 → 0 at 716 (§2, l.23) | `reports/S4/tables/prior_share.csv` **[V]**. The shares are computed on validation queries (`reports/S4/README.md:234`) **[V]** | Correct |
| A5 | "Report S4's mean per-query ratio SD(a p)/(SD(a p)+SD(linear correction))" (§3, l.45) | `reports/S4/README.md:236`: `σ(a_k π)/(σ(a_k π)+σ(Wq+b))` per query **[V]** | Correct |
| A6 | S3: all three MS MARCO retained weights equal the initial 1.0000 (§2, l.24) | `docs/decisions/ADR-0011-s3-prior-weight.md:31,33`, "every seed keeps its step-0 checkpoint" **[V]** | Correct |
| A7 | E2: none of 18 shift cells passed; SciFact variance shift 0.4376, B1 0.4354, B2 0.5381; direct variance probe 0.5366 (§2, l.25) | `reports/E2/tables/criteria.csv:2-19`; `ladder.csv:2-4,10` **[V]** | Correct |
| A8 | NFCorpus one-positive transfer regret/mass 0.2397 vs privileged 0.1289 at 307 (§2, l.26) | `reports/P3-shared-mask/README.md:25,330` **[V]** | Correct |
| A9 | SciFact R0 +0.0190 over untuned REF; +0.0054 [−0.0062, +0.0170] over tuned KG; Holm p 1.000 (§2, l.27) | `reports/P3-objective/README.md` ranking table and C8 row **[V]** | Correct |
| A10 | Search-Adaptor lists 14 BEIR datasets in Table 7, p. 12 (§2, l.28) | `/home/wilker/Papers/Search-Adapter.pdf` p. 12, Table 7 (NFCorpus … MSMarco, 14 rows) **[V]** | Correct |
| A11 | G+V learned count 1,049,609; HyperPrior 6 (ablation 3); bias h = 18/43; rank-four 4h+4,096 = 4,168/4,268; cached cumulative 9 / 1,033 / 5,129; 8,192 MACs per query (§3 table, §3 Costs) | 1024² + 1024 + 9; 4·18 + 4096; 4·43 + 4096; 9 + 1024 + 4096; Bq + A(Bq) = 2 × 4,096 **[C]** | Correct |
| A12 | kᵢ = floor(iD/10) | 102, 204, 307, 409, 512, 614, 716, 819, 921 **[C]** | Correct |
| A13 | ε = 10⁻¹²; population variance; participation ratio tr(Σ)²/tr(Σ²) (§3) | `baselines/learning_to_select.py:38,58`; `space/geometry.py:19,23-24` **[V]** | Correct |
| A14 | Stage A initializes W, b to zero and a⁰ to ones "as in S4" (§3, l.45) | `learning_to_select.py:47-48,83`; `reports/S4/README.md:4-6` ("at step 0 it is the `variance` method at every budget") **[V]** | Correct |
| A15 | P0 h = 18, P1 h = 43 (§3) | `space/descriptor.py:37-50` **[V]** (widths verified in round 2) | Correct |
| A16 | Step one: "keeping **official validation** for the untouched score where available" (§5, l.63) | The project has no official validation split. `validation` is a 10% seeded split of train for all three corpora (`configs/data/{scifact,nfcorpus,msmarco}.yaml`, `validation_fraction: 0.1`; `src/hyperdime/data/prepare.py:107-111`) **[V]**. MS MARCO's *official* dev set is the project's **test** split (`reports/E1/README.md:25`, "E1 scores MS MARCO dev") **[V]** | **Wrong as worded; leakage hazard.** Fix in B1 |
| A17 | Gate E: SC-001 requires ≥2 of 3 held-out environments above B1 and B2 for every seed, with Holm-adjusted permutation p ≤ 0.05 (§5, l.73) | `docs/decisions/ADR-0013-g3-space-signal.md:20-21`; `criteria.csv` columns **[V]** | Correct |
| A18 | ≈25 families for +0.005 at SD 0.010, 80% power, one-sided normal approximation (§6, l.102) | ((1.645 + 0.842) × 2)² = 24.7 **[C]** | Correct |
| A19 | "At 8–12 families design for a true +0.010" (§6, l.102) | Exact one-sided family sign-flip at α = 0.05, normal family effects, 2,000–4,000 replicates **[C]**. Power at +0.010: K = 8 gives 0.93 / 0.80 / 0.52 at SD 0.008 / 0.010 / 0.015; K = 12 gives 0.99 / 0.94 / 0.71. Null size 0.04–0.05 | Correct only for SD ≤ 0.010. With about 5 development families, the conservative SD (upper 80% limit) is about 1.56× the point estimate (χ², 4 df) **[C]**. A point SD of 0.010 therefore becomes 0.0156, and 8–12 families then has 52–71% power. Add these numbers (N4). |
| A20 | PRF-DIME: query times top-document centroid (DIME §3, eq. 5, p. 5); Eclipse bottom-of-list pseudo-negatives (§3, eq. 1, p. 3); L2S pseudo-queries (§6.5, Table 5, p. 8) (§9, l.132) | pypdf: DIME "(5)" and "centroid" on p. 5; Eclipse "(1)" on p. 3; L2S "Table 5" and "6.5" on p. 8 **[V]** | Correct |
| A21 | Opus-E predicts SciFact R_F9 < 0.25 and R_Fb ≥ 0.50 (§5, l.65) | `round2-opus-e.md` §4 D1 and §5.2 S0 **[V]** | Correct. Re-registered with more predictions in §5. |
| A22 | Characterizations of my positions in §7 (G3 override; test-only labels moved into training; replication counts and PI at 8–12; P1 first if eligible) | `round2-opus-e.md` D2, D3, D4, §5.3 **[V]** | Correct. On D4 I also had an exact family sign-flip as the primary analysis, so my position and Opus-M's were closer than the table suggests. |
| A23 | "No reported project comparison estimates the incremental rank-four effect" (§8, l.120) | No `reports/` table measures it. ADR-0001's query-conditioned negative was not re-read | Correct as far as I checked |
| A24 | Stage B: "Fit C by **equal-corpus** least squares" (§4, l.56) | Internal check: §6 uses equal-family weights everywhere | **Inconsistent.** Fix in N2. |
| A25 | Published artifacts, a one-command oracle ladder and balanced checkpoints are "unverified" (§10, l.138); no MS MARCO F12 in `reports/` | `reports/P3-bench/runs/evaluation/P3-bench/evaluation/` has only `nfcorpus/` and `scifact/`; `RUNS.md:153` says the MS MARCO stage is "running" **[V]** | Correct. Compute for MS MARCO in step one is unplanned (N7). |

**Summary.** I checked 25 items. 22 are correct. One is correct but incomplete (A3). One is an
internal inconsistency (A24). One is a wording that opens a leakage path (A16). There are no
wrong numbers.

---

## 3. The four OPEN-for-sign-off items

### 3.1 Budget-profile parameterization: aₜᵢ = max(0, a⁰ᵢ + u + tᵢv)

**Position.**
- **Sign offset-plus-slope as the primary HyperPrior.** Keep the single offset as the nested
  ablation and as the model retained under the registered "<0.005 slope gain" rule.
- **A single shared offset is defensible as an ablation, not as the default.**
- **Offset-plus-slope is sufficient for this run**, on the evidence below. Sufficiency is then
  measured by F9 − F2 and is not extended within this run.

**Evidence that budget shape varies by corpus.**
1. **Per-budget weights beat one weight in-domain on two of three corpora.** S4 against aligned S3:
   - NFCorpus +0.0111 [+0.0082, +0.0141];
   - SciFact +0.0094 [+0.0049, +0.0141];
   - MS MARCO +0.0001.

   Source: `reports/S4/README.md`, SC-001 table **[V]**. With a single weight, the prior *hurts*
   SciFact at high budgets: setting it to zero gains +0.0066 at k = 819 and +0.0200 at k = 921
   (`reports/S4/README.md`, "prior removed" table) **[V]**.

   Caveat: this compares one budget-free weight with nine per-budget weights. It is not a direct
   test of a corpus offset added to a shared per-budget curve.
2. **The scale-free prior-share curves differ in shape**, not only in level:
   - MS MARCO is flat at about 0.95;
   - NFCorpus falls from 0.50 to 0 at k = 716;
   - SciFact falls from 0.47 to 0 at k = 819.

   Source: `prior_share.csv` **[V]**.

**Evidence that two numbers are enough.**
1. **Curve decomposition [C].** I took the three seed-mean S4 curves and fitted each corpus's
   deviation from the three-corpus mean curve.
   - In prior-share units, one offset explains 94% / 90% / 96% of the mean-square deviation
     (MS MARCO / NFCorpus / SciFact). Offset plus a linear slope in tᵢ explains 100% / 98% / 99%.
   - Linear-plus-clip fits of each share curve leave an RMSE of 0.001 / 0.035 / 0.018.
   - Adding a quadratic term reduces NFCorpus to 0.006. NFCorpus's concave-then-zero shape is the
     only clear miss.
   - In raw weight units the offset alone explains ≥ 99%. That figure is dominated by MS MARCO's
     level, and raw units are not comparable across these in-domain W's.

   This is shape evidence from n = 3 corpora, each with its own W. It says nothing about the
   optimum under a shared frozen W.
2. **Retrieval utility is concentrated at small budgets.** Removing the prior entirely changes
   validation nDCG@10 as follows (`reports/S4/README.md`, prior-removed table) **[V]**:
   - MS MARCO: −0.20 at k = 102, −0.039 at 307, between −0.0013 and +0.0016 at k ≥ 716;
   - NFCorpus: −0.059 at 102 and ≤ 0.0017 in absolute value from k = 409.

   The NFCorpus curvature that a line misses sits at k = 409–614, where the prior barely moves
   retrieval [I]. A third basis function would therefore buy little utility per parameter. With
   K_train ≈ 5 families, a third output would add three map coefficients to the current six.
3. **The clip makes the offset more expressive than it looks [I].** The balanced base curve a⁰ is
   likely to decay with k, because two of the three development corpora decay. In that case a
   negative u moves the budget at which the weight reaches zero. That is a shape change, and a
   pure log-additive offset (Opus-M round 1) could not produce it. The additive projected form is
   the better choice.

**Conditions for my signature on 3.1** (B2, N1):
1. **Fit target.** Fit C to the F2 oracle's own (u\*, v\*) per family, or directly by inner-fold
   retrieval utility. Do not fit it by least squares to F9's nine projected weights.
   - Item 2 above shows that F9's weights at utility-flat budgets are unidentified: on MS MARCO
     at k ≥ 716, removing the prior entirely moves nDCG@10 by ≤ 0.0016.
   - Least squares on those coordinates would let noise steer the map.
   - Where the F2 optimum is flat, take the point closest to (0, 0), that is closest to G+V,
     within 0.001 of the maximum.
2. **Units.** Express u and v in units of each fold base's scale ratio:
   r_W = median over source training queries of SD(Wq + b)/SD(p).
   - Stage B learns C on inner bases and applies it to the outer refit (§4).
   - A raw additive offset means different things on bases with different ‖W‖.
   - S4's in-domain raw weights span 0.003–0.98 across corpora (`weights.csv`) **[V]**, which
     shows how much the scale moves.
   - The synthesis currently only *reports* this transfer error (§4, l.57). Normalizing removes
     most of it at zero cost.
3. **Escalation beyond two numbers.** If F9 − F2 ≥ 0.005 with a positive lower bound on a
   meta-train family, record a limitation. Do not add a third output in this run.

### 3.2 What a failed G3 rerun kills

**Position: accept the compromise. I withdraw the override I proposed in round 2 (D2).**

A failed fresh G3 closes bias and rank in this run. The prespecified tied-bias retrieval
diagnostic is still reported. A discordant positive result leads to a new ADR on new families.
HyperPrior never depends on G3.

**Reasons.**
1. **The override is now redundant.** Step one already measures what I wanted it to measure: the
   *attainable* Xw oracle, a retrieval-scored test of whether descriptor rows can express a
   useful bias (§5, l.65). Gate C requires that attainability before any bias generator is built.
   My concern was that E2 scores mask agreement and not retrieval (E1-c: near-chance overlap of
   0.33 costs only 0.012–0.030). The concern is met *upstream* of G3, not by overriding it.
2. **With K_train ≈ 5 families my override test would be underpowered.** Letting an underpowered
   retrieval test override a registered gate after seeing it fail is a forking path.
3. **"Fewer than three fresh families means pending"** (Gate E) already prevents a pool-size
   failure from being read as a negative.

**Amendment (N5).** Register the diagnostic's statistic, folds and threshold before Gate E
runs. Report it next to G3 in the same table, whatever the G3 result.

### 3.3 Claim wording at 8–12 families

**Position: sign estimation as the default and the simulation-gated upgrade. Change the wording
of the upgrade (B3).**

**The defect.** The synthesis allows a qualified result at 8–14 families to "confirm the
registered mean contrast **for the audited evaluation panel**" (§6, l.104). That wording does not
match the test.
- The primary test is a family sign-flip. It treats family effects as sign-symmetric draws, so
  its inferential target is the **centre of the distribution the families come from**, not the
  fixed mean of these K families.
- A genuine fixed-panel statement would use only within-family query and seed uncertainty. That
  is a much easier test, and it is not the one registered.

**Proposed wording**, to use when the simulation passes:

> "Across K audited families, treated as exchangeable members of the registered eligibility
> frame, the locked P0 arm changed equal-family mean decile nDCG@10 relative to G+V by Δ̄
> (one-sided 95% sign-flip bound L; p). m of K families had a positive point effect. The 95%
> prediction interval for a new family from this frame is [a, b]."

- m/K and the prediction interval must appear in the same sentence as Δ̄, including in any
  abstract.
- The words "generalizes to new corpora" are forbidden below 15 families.

**Why estimation will be the usual outcome** (my simulation, A19).
- Power at +0.010 with 8–12 families is 0.80–0.94 at SD 0.010, but only 0.52–0.71 at SD 0.015.
- A conservative SD taken from about 5 development families is about 1.56× the point estimate.
- So the registered ≥ 80% criterion will usually fail. Plan the write-up as an estimation study.
  Expect the upgraded claim only if development heterogeneity is small (point SD ≲ 0.007).

### 3.4 Protocol order (P0 vs P1) and the ordered attribution tests

**Position: sign P0 as primary, and the fixed sequence G+V → matched global → donor. I withdraw
my round-2 "P1 first if eligible".**

**Reasons for P0 first.**
1. **Eligibility, not preference, decides this.** P1 needs 500 distinct non-evaluation queries
   per corpus (§3, l.47). The candidate meta-test families are evaluation-only sources whose
   only queries are test queries (§6, l.80-82).
   - P1 is therefore ineligible on almost every meta-test family. Its comparison with P0 will
     have K near zero there.
   - Making P1 primary "if eligible" would select the panel by protocol. That changes the
     estimand and shrinks K. This is the synthesis's own point (§7).
2. **Rival R1 is still tested, but in exploratory form.** R1 says the shift is driven by the
   query distribution, not by the corpus. The test is P1 against P0 on meta-train LOFO, with
   matched sample sizes (§9).
   - A flat P0 result together with a positive P1 result narrows the claim (§8). I sign that
     rule.
   - State in §6 that any P1 meta-test evidence covers a reduced, possibly empty panel (N5).

**Reasons for the ordered attribution tests.**
- A fixed sequence with stopping at the first failure controls familywise error with no α
  splitting. Passing all three is the only route to a descriptor-specific claim, which is the
  right standard.
- **A consequence to state explicitly [I].** For HyperPrior, the matched global is G+V with
  refitted global budget weights. That is nearly G+V itself, so steps 1 and 2 are strongly
  correlated, and step 3 (donor φ, three scalars) carries the attribution.
- The joint simulation must therefore model correlations between arms, as §6 already requires.
  It should not count three tests' worth of evidence.

---

## 4. Amendments (most important first)

### Blocking

**B1 · §5 step one, l.63: scoring partition (leakage).**
- *Old:* "Draw all fit and tuning examples from official train queries, keeping official
  validation for the untouched score where available. When only a train split is eligible,
  preassign its three non-overlapping partitions by hash."
- *New:* "Draw fit, checkpoint-choice and untouched-score partitions only from each corpus's
  project `train` and `validation` splits. These are the F2 records: `validation` is a 10% seeded
  split of the official train queries (`prepare.py:107-111`). Score the untouched partition on
  `validation`, plus a hash-held part of `train` if more queries are needed. Never read the
  project `test` split. For MS MARCO the `test` split is the official dev set (`reports/E1/README.md:25`)."
- *Why:* no project corpus has an "official validation" split. On MS MARCO the natural reading of
  the phrase is the dev set, which is this project's test qrels. Reading those qrels to make
  design decisions violates AGENTS.md:15 even on development corpora.

**B2 · §4 Stage B, l.56: target for the HyperPrior map.**
- *Old:* "Fit C by equal-corpus least squares to the attainable oracle's projected nine weights
  plus an L2 penalty."
- *New:* "For each inner pseudo-target family, take the attainable F2 oracle (u\*, v\*). Where
  the utility surface is flat, use the point nearest (0, 0) within 0.001 nDCG@10 of the maximum.
  Express (u\*, v\*) in units of that fold base's scale ratio r_W (see N1). Fit C by
  equal-family ridge regression of (u\*, v\*) on standardized φ. As a sensitivity analysis, fit C
  directly by inner held-family retrieval utility. Do the same for the offset-only ablation with
  the F1 oracle."
- *Why:* see §3.1. The nine F9 weights are not identified at utility-flat budgets, so least
  squares on them fits noise. F2 is the oracle for the model actually being fitted.

**B3 · §6 Claim level, l.104: scope of the 8–14-family claim.**
- *Old:* "At 8–14, a qualified result can confirm the registered mean contrast for the audited
  evaluation panel."
- *New:* "At 8–14, a qualified result supports the registered mean contrast for the eligibility
  frame from which the audited families were drawn, as the exact family sign-flip tests it. It
  must be stated together with m of K positive families and the 95% prediction interval for a new
  family, using the template in round4-opus-e §3.3. 'Generalizes to new corpora' is not used
  below 15 families."
- *Why:* the test's estimand is the centre of a family distribution, not a fixed-panel mean
  (§3.3).

### Non-blocking

**N1 · §3, l.41 and l.45: units of u and v.**
Add: "u and v are in units of r_W, where r_W is the median over source training queries of
SD(Wq + b)/SD(p) for the fold's frozen base. Report r_W per fold." This removes most of the
inner-to-outer base transfer error that §4 l.57 currently only reports.

**N2 · §4, l.56: weighting.**
Change "equal-corpus" to "equal-family", to match §6.

**N3 · §2, row 2 (l.22): composition of the legacy union.**
Add: "The legacy union is an L2S-KL selector with no variance prior (`train_global.py:56-71`).
Legacy → LOCO-balanced G+V therefore changes objective, prior, balancing and recipient exclusion
at once. Step one isolates them in this order:
1. legacy;
2. LOCO union, same objective;
3. LOCO balanced, same objective;
4. S4 objective with pooled variance;
5. S4 objective with target variance."

**N4 · §6, l.102: power.**
Add the exact sign-flip numbers from A19. Add the consequence that a conservative SD from about
5 development families (about 1.56× the point estimate) will usually put 8–12 families below 80%.
"Plan the report as an estimation study."

**N5 · Gate E (l.73) and §6, l.84: diagnostics and P1 eligibility.**
- Register the tied-bias retrieval diagnostic (statistic, folds, threshold) before Gate E runs.
- State that evaluation-only families have no non-evaluation query pool, so the P1-vs-P0
  comparison at meta-test may have K ≈ 0. P1 evidence then comes from meta-train LOFO and is
  exploratory.

**N6 · §2: add an evidence row.**
- *Observation:* removing the prior moves S4 validation nDCG@10 by −0.20 at k = 102 and by at
  most 0.0016 in absolute value at k ≥ 716 on MS MARCO; on NFCorpus by −0.059 at 102 and by at
  most 0.0017 in absolute value from k = 409. Source: `reports/S4/README.md`, prior-removed table.
- *Consequence:* the prior's utility sits at small k. High-budget weights are weakly identified.
  This supports the offset-plus-slope family and fitting in utility space.

**N7 · §5 step one: MS MARCO compute.**
- MS MARCO F12 is still unreported (`RUNS.md:153`).
- Before the run, freeze a resource plan (ADR-0018 pattern) with a capped untouched partition
  (for example 2,000 queries) and the number of arms × seeds × budgets.
- If it does not fit, run step one on SciFact and NFCorpus first. Record MS MARCO as pending, not
  failed.

**N8 · §5 step one: small SciFact partitions.**
- SciFact has 809 non-test queries (728 train + 81 validation). The untouched partition will
  have about 80–135 queries. F12's 121-query paired intervals were about ±0.013 (round 1, F12).
- State that the R thresholds (0.25 and 0.50) are resolvable on SciFact only because H is
  likely ≈ 0.1. On NFCorpus and MS MARCO, absolute gains are the reported quantity.

---

## 5. My prediction for step one on the three development corpora (recorded before any run)

**Scope and conventions.**
- The step is the synthesis's runnable development experiment (§5, l.63-65), after B1. It
  covers LOCO bases, the specialist, the oracle ladder and the variance ablation.
- **Endpoint:** decile-mean exact nDCG@10 on the untouched validation-derived partition. Every Δ
  is paired against LOCO-balanced S4 G+V with target variance unless stated otherwise.
- H = specialist − G+V.
- **Anchors:** the ranges below come from test-split magnitudes in `reports/overview/README.md`,
  `reports/S4/README.md` and `reports/E1/`. Partitions differ, so read them as ±0.01 on SciFact.
- **Confidence:** the percentage is my subjective probability that the stated range holds.
- **Falsifier:** the result that would make me record the prediction as wrong.

| # | Quantity | Prediction | Confidence | Falsifier |
|---|---|---|---|---|
| E1 | SciFact H | +0.08 to +0.14 | 80% | H < +0.05 (LOCO balancing removes the SciFact gap: Opus-M's I1 right) |
| E2 | NFCorpus H | −0.005 to +0.015, point +0.006. **Does not** meet "H ≥ 0.010 with lower bound > 0", so ratios are undefined | 70% | H ≥ 0.010 with lower bound > 0 |
| E3 | MS MARCO H | −0.003 to +0.010, point +0.003 | 70% | H > +0.010 with lower bound > 0 |
| E4 | SciFact, prior-only oracles | R_F1 ≤ R_F2 ≤ R_F9 < 0.25 (registered in round 2) | 70% | **R_F9 ≥ 0.50 overturns it.** HyperPrior alone would then be the right program (my round 2 §5.4) |
| E5 | SciFact, free bias | R_Fb ≥ 0.50 (registered in round 2) | 55% | R_Fb < 0.25 |
| E6 | SciFact, **attainable** tied bias Xw (P0, h = 18, rows centred on the other two corpora) | Recovers < 0.25 of H, although Fb recovers ≥ 0.50. A useful bias exists, but document-row features cannot express it. The bias rung would then close at Gate C before G3 | 65% | Attainable Xw ≥ 0.50 of H |
| E7 | SciFact, rank four over bias with the fixed SVD basis | R_Fr4 − R_Fb < 0.15 | 70% | R_Fr4 − R_Fb ≥ 0.25 |
| E8 | Sufficiency of two numbers (all three corpora) | F9 − F2 ≤ 0.003 and F2 − F1 ≤ 0.005 | 70% | F9 − F2 ≥ 0.005 with lower bound > 0 on any corpus |
| E9 | Target vs pooled variance inside G+V | MS MARCO +0.002 to +0.010; SciFact within ±0.003; NFCorpus −0.002 to +0.006 | 55% | Target worse than pooled by > 0.003 on any corpus |
| E10 | Balancing alone (LOCO union vs LOCO balanced, same objective, prior and updates) | Within ±0.005 on each held corpus | 65% | Difference > 0.005 in absolute value with lower bound excluding 0 |
| E11 | LOCO-balanced G+V vs legacy union (which saw the recipient's labels) | NFCorpus: G+V higher by +0.005 to +0.020, because the legacy union sits below `variance` and full there (0.3462 vs 0.3587 and 0.3619). SciFact within ±0.02. MS MARCO within ±0.01 | 60% | G+V below legacy union by > 0.02 on SciFact, or by > 0.005 on NFCorpus |
| E12 | MS MARCO topical shards (descriptive) | Every shard's specialist − G+V gap < 0.02, far below SciFact's H. This is rival R1: shift follows the query distribution more than the corpus | 60% | Any shard gap ≥ 0.05 |

**What the joint outcome would mean [I].**
- **E1 + E4 + E5 + E6 all hold.** Headroom is concentrated on SciFact. Prior prediction cannot
  reach it, and document-row features cannot express it. The likely deliverable is G+V plus a
  HyperPrior that helps only marginally.
- **E4 fails.** HyperPrior is the program, and I was wrong about where the SciFact gain lives.
- **E6 fails.** The tied bias earns its Gate C pass and becomes the real candidate, subject to
  G3.

---

## 6. Verification log (this round)

**Read [V]:**
- `reports/E1/tables/payoff.csv`;
- `reports/E1/README.md:25`;
- `reports/S4/tables/{prior_share,weights}.csv`;
- `reports/S4/README.md:1-40, 230-345`;
- `docs/decisions/ADR-0011-s3-prior-weight.md:27-39`;
- `reports/E2/tables/{criteria,ladder}.csv`;
- `reports/P3/README.md:40-70`;
- `reports/P3-shared-mask/README.md:21-25, 104, 330`;
- `reports/P3-objective/README.md:1-40`;
- `reports/overview/README.md:73-107`;
- `src/hyperdime/cli/train_global.py:50-90`;
- `src/hyperdime/baselines/learning_to_select.py:30-83`;
- `src/hyperdime/space/geometry.py:1-61`;
- `src/hyperdime/space/descriptor.py:35-52`;
- `src/hyperdime/data/prepare.py:30-128`;
- `configs/data/{scifact,nfcorpus,msmarco}.yaml`;
- `AGENTS.md:15`;
- `docs/decisions/ADR-0013-g3-space-signal.md:20-21`;
- `reports/P3-bench/RUNS.md:149-153`;
- PDFs, located with pypdf: DIME p. 5 (eq. 5); Eclipse p. 3 (eq. 1); Search-Adaptor p. 12
  (Table 7, 14 BEIR rows); Learning to Select p. 8 (§6.5, Table 5).

**Recomputed [C]** (`/tmp/oe4/fit.py`, `/tmp/oe4/pw.py`):
- the parameter counts, budgets and the 93.6% share;
- the normal-approximation K ≈ 24.7;
- the offset and offset-plus-slope decomposition of the S4 curves;
- exact family sign-flip power for K = 6–12, SD 0.008–0.015, μ = 0, 0.005, 0.010;
- the χ² upper-80% SD factor of about 1.56 at 4 df.

**Not verified:**
- ADR-0001's outcome;
- the eligibility and family counts of candidate new corpora (as the synthesis says);
- whether published artifacts suffice to run step one without re-embedding.
