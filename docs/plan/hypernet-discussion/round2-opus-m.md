# Round 2 — Opus-M (methods lead / red team): cross-critique, protocol rulings, revised program

I read `BRIEF.md`, `round0-seed.md`, `round1-astra.md`, `round1-opus-e.md`, and my own
`round1-opus-m.md`. Skills used: peer-review (each proposal gets strengths, then errors and
unsupported claims with locations, then risks), experimental-design (unit, blocking, leakage), and
statistical-power (gate operating characteristics, sample size from a SESOI).

Tags. **[data]** = a number I read in a repository file in this round, with its path. **[mine]** =
a computation I ran in `/tmp/hm-r2/gates.py` with the project venv. I did not commit it. **[unverified]**
= not checked against a primary source. Line references `file:N` point to the round files in this
folder unless a path is given.

---

## 0. Verified facts this round

| # | Fact | Source |
|---|---|---|
| V1 | The E1 global selector concatenates every environment's training queries and targets (`torch.cat`), with no corpus weights. The epoch count, however, is chosen by the **unweighted mean over environments** of validation nDCG@10. | [data] `src/hyperdime/cli/train_global.py:54-60` (concatenation), `:83-84` (environment-mean selection) |
| V2 | Training-query counts: SciFact 728, NFCorpus 2,331, MS MARCO 45,000. MS MARCO therefore supplies 45,000 / 48,059 = **93.6%** of the global selector's training queries. | [data] `specs/003-data-embeddings/tasks.md:76`; `docs/experiment_registry.md` row P3 ("F2's 45,000 training queries"); [mine] the ratio |
| V3 | S3's MS MARCO prior weight of 1.0000 in every seed is the **step-0 checkpoint**: "no training step improved validation". | [data] `docs/decisions/ADR-0011-s3-prior-weight.md:31,33` |
| V4 | S4's identified prior share at k = 307 is MS MARCO 0.9512, NFCorpus 0.4281, SciFact 0.3580. NFCorpus and SciFact reach 0 by k = 716 and 819; MS MARCO stays at about 0.95 for every k. | [data] `docs/decisions/ADR-0012-s4-budget-prior.md:34` |
| V5 | Astra's E2 figures match the tables, e.g. MS MARCO set 1 shift 0.5657 against B1 0.5391 and B2 0.6687. | [data] `reports/E2/tables/criteria.csv:4` |
| V6 | `protocols.md` says: "BEIR test qrels are used only for final evaluation." It also says meta-test "is evaluated once, after the meta-validation gate". | [data] `docs/protocols.md:21-22` |

What V1–V4 change:
- My round-1 "~94%" stands as 93.6%. I1 (the E1 headroom is inflated by a query-weighted global
  selector) is **still a hypothesis**. The environment-mean epoch choice moderates it, and nobody
  has trained a corpus-balanced global selector.
- **All four proposals overstate how much the prior weight varies across environments.** That
  includes mine (`round1-opus-m.md:73-74`), Opus-E's (`round1-opus-e.md:86`, "3 orders of magnitude"),
  and the seed's (`round0-seed.md:15`).
  - The raw `a` and `a_k` are not scale-identified: top-k is invariant to rescaling the whole score.
    Opus-E saw this (`round1-opus-e.md:498-500`) but its evidence table did not apply it.
  - MS MARCO's S3 value was never estimated (V3).
  - The identified quantity, prior share, varies about **2.7× at k = 307** (0.95 / 0.43 / 0.36), plus
    a budget **shape** difference (flat versus decaying to 0) (V4).
  - That is still a real environment-level signal. It is smaller than the round-1 framing implied,
    and it is carried by n = 3 environments.

---

## 1. Per-proposal review

### 1.1 Round 0 seed (capacity ladder H-0..H-3)

**Strongest ideas.**
- Grow capacity only when each rung wins on unseen corpora (`round0-seed.md:7-9`).
- **H-2 is a comparator, not a rung** (`:35`). A per-dimension gain that is a function of descriptor
  rows is equivalent to a global selector given those rows as features. This is the cleanest
  statement in any proposal of why the matched global control must see the same inputs.
- Flags that MS MARCO's α "never moved off init" (`:16-17`), confirmed by V3.
- Pseudo-corpora must be checked to "reproduce the real spread of α" (`:38-39`). That is the right
  validity test for any augmentation.

**Errors and unsupported claims.**
- *The capacity does not match the corpora.*
  - `U_T, V_T` are generated per dimension from `(row_j, Z_T)` (`:33`). Generating both factors per
    environment lets the input subspace rotate per corpus. This is Astra's objection
    (`round1-astra.md:95`), and I agree with it.
  - The descriptor includes a **dimension-ID embedding** (`:31`). With a handful of training families,
    an ID embedding plus `Z_T` reproduces E2's B1 shared profile and then memorizes each family's
    deviation. That is plan risk R2.
- *The success rule cannot be met.*
  - "Non-inferior to global on every held-out corpus (δ = 0.005)" (`:50-51`) is a conjunction of K
    per-corpus tests.
  - With true Δ = 0 and a per-corpus SE of 0.004, the probability of passing at K = 5 is **0.005**.
    At SE = 0.008 it is 0.000 [mine]. The rule fails a perfectly harmless method.
  - Replace it with a corpus-level harm guard: a per-corpus test of Δ < −0.01, Holm over corpora.
- *Leakage.*
  - "Recover stated share of headroom" (`:51`) and "in-domain S4 as ceiling" (`:50`) need an
    in-domain selector trained on held-out corpora's labels. §2 rules this inadmissible.
- *Feasibility.*
  - "~10 train + ≥ 5 held-out" (`:48`) gives a minimum detectable standardized effect of about 1.4
    at K = 5 (one-sided α = 0.05). Only very large gains would be detectable.

**Risk it underrates.** Pseudo-corpora (clusters, subsamples, mixtures; `:38`) add episodes, not
independent environments. Unless they are labeled as shards, they inflate the apparent K_train.

### 1.2 Astra (P0 rank-4 correction + bias on a frozen pooled S4, 4,186 parameters)

**Strongest ideas.**
- **G3 replication before building** (`round1-astra.md:60,99`). The E2 shift failure in all 18
  cells is the only direct prior evidence against descriptor-driven per-dimension corrections. Astra
  is right to make it bind, and right that "predicting which coordinates are generally useful is
  easier than predicting which ones change" (`:67`).
- **Shared input factor B, tied row map H.** The corpus-conditioned part is `h(r+1)` = **90 numbers**
  (P0) (`:84-93`). The correction cannot rotate the query subspace per corpus (`:95`). Its count is
  modest, and its identification relies on 1,024 coordinate rows per corpus being exchangeable
  observations, which is exactly what E2 tested.
- **X = 0 recovers the base exactly, even after training** (`:91`). This is a strong
  interpretability property.
- **Calibration queries are hash-selected from a non-evaluation pool before any split**, and
  descriptor support is kept apart from supervised queries (`:77,100`).
- **A donor-descriptor control that leaves `v_T` untouched** (`:107`). It isolates the generated
  channel from the analytic one.
- **No in-domain reference on meta-test** (`:109`). Astra reads AGENTS.md correctly here, and I was
  wrong in round 1 (§3).
- An honest power statement: ≈ 32 families for SESOI 0.005 at SD 0.010, two-sided (`:111`), with a
  pre-declared pilot fallback.

**Errors and unsupported claims.**
1. **The treatment omits the best-documented environment-level channel.** `a` is "nine nonnegative
   shared budget weights" (`:91`), so they are global.
   - The rank-4 term `A_T(Bq)` is linear in q, and the bias `c_T` is constant in q. Neither can
     reproduce a change in the weight on `log(q² v)`.
   - So the design cannot express the one quantity that does vary across environments (V4).
   - The `a_k-only` control (`:107`) is the right object, but it should be the **base rung** that
     the rank-4 model must beat. As written, a win against pooled S4 could come entirely from what
     the bias and rank-4 term emulate of a prior-share change.
2. **The "matched global" control is either degenerate or a different model** (`:107`).
   - "Feed a fixed coordinate template … the column-standardized M". Standardizing M by the
     model's own normalization gives X = 0. Then `[A|c] = 0`, H and B receive **zero gradient**, and
     the arm is the frozen base with no optimization to match.
   - If the template is a fixed non-zero matrix, the arm is a global bias plus a global rank-4 ΔW: a
     low-rank continuation of W₀.
   - Name it that way ("global rank-4 + bias continuation, same steps"). It is the right control, but
     only once it is stated as a non-degenerate model.
3. **The primary estimand and model selection are at k = 307 only** (`:101,109`). The project's
   frozen convention for P3/F12 (ADR-0018) is the decile mean 102..921, and 307 has no special
   status. Selecting and testing at one budget also wastes the per-budget structure (V4) that
   distinguishes environments. Use the decile mean for both, with 307 descriptive.
4. **The equivalence rule cannot be met** (`:113`). "Paired 90% interval wholly inside ±0.002" at the
   family level:
   - with SD 0.010 and K = 10, the SE is 0.0032 and the 90% half-width about 0.0058. Equivalence
     cannot be declared at any realistic K (it needs K ≳ 70).
   - The descriptor-swap and bias-only rejection rules will therefore always return "unresolved".
     Astra admits this (`:113`, last sentences), but then they are not falsification criteria.
   - Make them superiority contrasts (correct minus donor > 0) and accept that "no mechanism
     evidence" is the honest outcome when they fail.
5. **The power figure contradicts the corpus pool.**
   - 32 meta-test families, plus meta-validation, plus training families, exceeds every census in this
     discussion: about 10–14 test-only families and 5–6 label-bearing ones (`round1-opus-m.md:197-206`,
     [unverified]).
   - So Astra's program is a pilot by its own rule (`:111`). That should be said up front, and the
     SESOI should be revisited (§5).
6. **The meta-validation gate is a coin flip at the SESOI.** "Mean gain ≥ 0.005" (`:113`): with
   SD 0.010 over 4 meta-val families, P(pass) is 0.16 if the true gain is 0, 0.50 at 0.005, and 0.84
   at 0.010 [mine].

**Capacity versus corpora.** 90 conditioned numbers plus a 4,096-entry shared B is acceptable *if*
the coordinate rows carry exchangeable signal. The binding risk is not the count but the premise, and
the G3 gate addresses it. Astra's design is the best-specified per-dimension rung of the four.

**Leakage.** I found none in the treatment. Statistics are fit on meta-train only (`:79`), negatives
come from training queries (`:100`), there are no meta-test labels, and the donor is drawn from
meta-train.

### 1.3 Opus-E ("HyperDIME-R": 8-dim latent → a_k, tied Δb, rank-8 modulation)

**Strongest ideas.**
- **Rivals R1–R4** (`round1-opus-e.md:123-135`) and the **corpus-only shard contrast M2**
  (`:347-350,466-467`): the same query distribution over different corpora. This is the only proposal
  that separates corpus shift from query-distribution shift. E1-d (`:81`), where the global selector
  beats full on SciFact, is good evidence that part of SciFact's gap is query-side.
- **The scale non-identification of `a_k`** (`:498-500`). The target should be prior share, or `a_k`
  with W₀ frozen.
- **Conditioning the meta-test on ≥ 80% estimated power** (`:411-415`).
- **A pseudo-L2S self-training rival** (C8, `:429`). This is the same idea as my C10: the right
  comparator for "do we need amortization at all".
- **An auditable first rung, HyperDIME-0** (`:453-454`), which converges with HyperPrior.
- **A measurable bottleneck**, ‖µ_T‖² (`:364,463`).

**Errors and unsupported claims.**
1. **Test qrels in training.** §3.4 meta-trains on "BEIR test-only datasets split query-wise 50/50 by
   sorted id, as Search-Adaptor did" (`:342-343`). That is training on BEIR test qrels, which
   `protocols.md:22` and the AGENTS.md hard constraint both forbid. It is admissible only through an
   ADR amending both, and then those corpora are permanently meta-train-only (§2.3).
2. **Census leakage.** Stage 0 runs "on every environment of the pool" (`:439`), including those that
   will become meta-test. It trains in-domain S4 on their labels and makes a stop/go decision (G0) on
   the result. See ruling §2.2.
3. **Meta-test labels in arms.** C9 (P2 fine-tune on 50 labeled training queries of the target) and
   C10 (in-domain reference) both train on labels of meta-test environments (`:430-431`). See ruling
   §2.1.
4. **The training weights do not match the estimand.**
   - Sampling is "∝ sqrt(n_train_T)" (`:359`), which gives MS MARCO about 7.9× SciFact's weight
     (√45,000 / √728). Evaluation, by contrast, weights environments equally (`:397-399`).
   - The base learns a query-mix-weighted compromise, the same defect V1 identifies in E1's global
     selector, only milder.
   - Train with uniform weights over families, then over environments within a family. Keep sqrt
     weighting as an ablation.
5. **The jointly trained W₀ breaks the matched control.** Stage B "freeze nothing" (`:366,369-370`)
   updates W₀, U, and V together with the hypernetwork. The z = 0 control is the Stage-A artifact.
   - The primary contrast therefore compares HyperDIME-R(W₀′) with control(W₀). It confounds corpus
     conditioning with a different, longer-trained base.
   - Either freeze W₀, as Astra and I do, or add a "Stage A continued for the same steps" control.
     Freezing is cleaner and also makes `a_k` identified.
6. **The latent is too wide for the number of families.**
   - An 8-dim `z_T` (`:307`) with K_train ≈ 5–6 label-bearing families under current rules can give
     each training family its own code.
   - ψ([r_j, z_T]) (`:310`) then becomes a family-specific per-dimension map. That is precisely the
     memorization path R2 describes, and leave-family-out evaluation detects it only after it has
     cost a meta-val read.
   - The KL term bounds ‖µ‖ but not the number of distinguishable codes.
   - Rule: latent width ≤ ⌊K_train-families / 3⌋ until LOFO shows underfitting.
7. **HyperDIME-0 has too many inputs.** Up to 8 environment scalars (`:302-304,453`) for ~6–10
   training corpora means ridge with p ≈ K. Leave-one-out is then unstable, and the permutation null
   becomes wide but easy to beat by chance with tuned λ. Pre-register p ≤ 3.
8. **The P1 descriptor in meta-training uses the supervised queries.** "P1 docs + unlabeled
   train-split queries" (`:360`) lets the descriptor encode the exact queries whose labels drive the
   loss. At meta-test, `Q_calib` is disjoint from the evaluation queries. Use a disjoint calibration
   subset in meta-train too, as Astra does (`round1-astra.md:100`).
9. **The gate counts and kill rules confound power with effect.**
   - G0 counts environments whose query-level CI excludes zero (`:443`). MAIR tasks with about 80
     queries can rarely show that, so G0 partly measures query count.
   - K2's "mean Δ < 0.005 kills" (`:476`) and K3's "within 0.002" (`:477`) are point-estimate rules.
     A method with true gain 0.005 is killed half the time, and "within 0.002" is unattainable as an
     equivalence statement (see Astra item 4).
   - G4-val's "≥ 0.003" (`:455`) passes a null method with probability 0.27 at 4 meta-val
     environments [mine].
10. **Stage 1 is "advisory"** (`:448-451`). A rerun of G3 with adequate n removes the only excuse for
    its round-1 failure, which was power. If the rerun fails, the per-dimension heads have no support,
    so it should bind for Δb and γ, though not for the `a_k` head.
11. **MAIR's labels** (`:344-346`): whether MAIR's ~80 labeled queries per task are test labels is
    [unverified]. If they are, item 1 applies. Six domains also means at most six families, however
    many tasks there are.

**Capacity versus corpora.** "≥ 20 real environments" (`:339`) is reachable only with item 1
(needs an ADR) or MAIR (item 11). Under current rules, K_train is about 8–12 corpora in about 5–6
families, since the Wikipedia cluster counts once. The model has ~1.07M jointly trained shared
parameters, an 8-dim latent, and a z-conditioned per-dimension MLP. That is mismatched to K_train
unless W₀ is frozen and the latent narrowed.

**Power.** The arithmetic is right: n_E = 12, SD 0.010 gives ~50%; SD 0.006 gives ~85%. But an SD
estimated from 8 LOFO units has an 80% CI of about 0.76×–1.57× the point estimate [mine]. Plan with
the **upper** limit, or the power claim is optimistic in about half of cases.

### 1.4 My own round-1 proposal (HyperPrior, gates H0–H5)

Reviewed in §3.

---

## 2. Protocol rulings on Opus-E's open questions

Governing text:
- **AGENTS.md, hard constraints:** "Never use test qrels, or any label from a meta-test environment,
  in training, adaptation, negative mining, descriptor construction, or normalization."
- **`docs/protocols.md:21-22`:** meta-test "is evaluated once, after the meta-validation gate"; "BEIR
  test qrels are used only for final evaluation".
- **`docs/protocols.md:36-42`:** leakage rules 1–5. Rule 3: "The meta-trainer refuses to load
  meta-test environments."
- The project instructions take precedence over defaults. Where AGENTS.md and protocols.md differ,
  the stricter text governs until an ADR reconciles them.

### 2.1 Is a reference selector trained on a meta-test environment's *training* labels admissible?

**Ruling: no, under the current rules.** This covers Opus-E's C10 and C9 (P2), my round-1 C11/C12
and leakage rule 8 (`round1-opus-m.md:222-225,271-273`), and the seed's "in-domain S4 as ceiling".

Reasons:
1. **Literal text.** The constraint says "*any* label from a meta-test environment", not "test
   qrels". It lists "training" without exception for reference arms. A meta-test environment's train
   qrels are such a label, and fitting S4 on them is training.
2. **Purpose.**
   - The rule protects the claim that every number on meta-test environments was produced without
     their labels. An in-domain reference does not feed the treatment, but it creates a
     **meta-test-label-dependent quantity** inside the confirmatory report: the gap-closure
     denominator.
   - Once computed, that quantity invites post-lock decisions: dropping environments whose
     denominator is near zero, or reframing the claim around "fraction recovered".
   - ADR-0018's freeze discipline exists to prevent exactly this.
3. **P2 is a separate scenario, but not on these environments.** `protocols.md:26-32` defines P2
   (few-label calibration) as a scenario that is never reported as zero-shot. It does not override
   AGENTS.md's ban on adaptation with meta-test labels. A P2 evaluation therefore needs **its own
   environment set**, distinct from the P0/P1 meta-test, or an ADR amending AGENTS.md.
4. **What is lost is small.** The primary endpoint is the contrast with G+V, which needs no
   in-domain arm. Gap closure is secondary.

**What is admissible instead.**
- Gap closure and the oracle capacity ladder on **meta-train environments under leave-family-out**.
  For each held-out family, the hypernetwork is refit without it, and that family's own in-domain S4
  is the denominator. Meta-train labels may be read.
- The same on **meta-validation**, whose purpose under protocols.md is decisions.
- If the user wants a meta-test denominator anyway, an ADR may amend AGENTS.md to allow a **sealed**
  post-lock reference arm. It would be computed after the meta-test lock, by a scoring-only process,
  with outputs written to a separate result family. No decision, exclusion, or primary or secondary
  hypothesis may read it, and it would be reported descriptively. I do not recommend this: it buys a
  ratio we can already estimate under LOFO.

### 2.2 Should the payoff-census gate use only meta-train environments?

**Ruling: yes, meta-train only, after the split manifest is frozen. It is never run on meta-test,
and preferably not on meta-validation either.**

Reasons:
1. The census trains in-domain S4 on each environment's training labels. On a meta-test environment
   that breaches §2.1.
2. A census is a **decision** (stop/go, and which output head to build). Making it on meta-test
   outcomes violates "evaluated once, after the meta-validation gate". It also turns meta-test into
   development data, a forking path (round-1 T1). Running a census on the whole pool and splitting
   afterwards is worse still: the split could, deliberately or not, sort high-gap corpora to one side.
3. On meta-validation it is permitted by the letter of protocols.md. But meta-val will have only 2–4
   families, and they should be spent on one go/no-go and one choice among ≤ 2 variants. A census
   there would consume it.

**Operational consequences.**
- **Order matters.** (i) Audit the pool (families, overlaps, licences, encoder exposure). (ii) Freeze
  and fingerprint the split manifest, randomized within strata of family × label availability. (iii)
  Only then train anything on new corpora.
- The census gives **weak evidence** with K_train ≈ 6–8. Opus-E's "≥ 30% of environments" rule at
  K = 8 needs 3 corpora. If the true share of payoff-bearing environments is 0.2, P(pass) = 0.20. At
  0.4 it is 0.68 [mine].
  - So G0 is a screen against a flat landscape, not an estimate of it.
  - Define "payoff-bearing" by the **point gap ≥ 0.005 with the corpus's query count ≥ 150**, not by
    a CI excluding zero, so that query count does not masquerade as payoff (§1.3, item 9).
- **Selection bias.** The census describes the meta-train population, which under current rules is
  the label-bearing corpora. Meta-test will be test-only BEIR-style sets, a somewhat different
  population (§5.4). The census licenses building; it does not predict the meta-test effect.

### 2.3 Related rulings Opus-E did not ask for but its proposal needs

- **Re-splitting test-only datasets' test qrels into train/test halves** (Search-Adaptor style,
  `round1-opus-e.md:342-343`): **inadmissible without an ADR** amending AGENTS.md and
  `protocols.md:22`.
  - If the user approves, each such corpus becomes **meta-train-only forever**. Its re-split
    "test" half may be used for meta-train LOFO validation, never for any meta-val or meta-test
    claim.
  - It is zero-sum: every corpus re-split for training is a family removed from the meta-test pool
    (§5.4).
- **Hierarchical model** (Opus-E Q3): the primary test stays a family-level (or environment-level)
  exact sign-flip. A mixed model with queries nested in environments, and the environment as a
  random effect, estimates the same fixed effect. With balanced arms per query, it cannot add
  effective units beyond the number of environments. Report it as a sensitivity analysis and use it
  for the prediction interval (REML τ), as in round 1 §8.
- **"Environment" versus "corpus" claim** (Opus-E Q6): state the claim as **environment** from the
  start, as `protocols.md:8` defines it (corpus, query distribution, instruction). The **P0 versus
  P1** contrast and M2's same-query-distribution shards are the pre-registered tests of the corpus
  mechanism. A K4 outcome then narrows the mechanism without renaming the claim.

---

## 3. My proposal: what I concede, what I keep

### 3.1 Concessions

1. **Withdrawn: the in-domain reference on labeled meta-test corpora** (`round1-opus-m.md:222-225,
   271-273`, and arms C11/C12). Astra was right (`round1-astra.md:109`). Gap closure moves to
   meta-train LOFO and meta-val (§2.1).
   - The 50/50 randomization of labeled corpora between meta-train and meta-test existed only to make
     C11 computable, so it is withdrawn too.
   - Label-bearing corpora now go to meta-train (§5.4).
2. **The prior-weight evidence was overstated** (V3, V4). My D4 used raw `a_k`, and my I2 said it
   "spans more than an order of magnitude". The identified spread is about 2.7× in prior share at
   k = 307, plus a budget-shape difference. HyperPrior survives because it freezes W₀, which makes
   `a_k` identified relative to W₀. But its gate targets should be stated in prior share as well as
   `a_k`.
3. **I mischaracterized tied low-rank designs.** I wrote that rank-r ΔW needs K_train ≥ 15 because
   "a generator of 2·1024·r factors has no environment-level data" (`round1-opus-m.md:566-568`).
   - That is true of the plan's free `A_T, B_T` and the seed's `U_T, V_T`.
   - It is false of Astra's design: shared B, with A tied to descriptor rows through 90 numbers.
   - The binding condition for such designs is the **validity of the tied map**, which is what G3
     tests, not a corpus count.
   - Revised Level-3 condition: G3 rerun passes **and** oracle Fr4 − Fb ≥ 0.25 of headroom, with
     selection by LOFO. The K_train ≥ 15 rule is dropped.
4. **My gates used point thresholds too.** H4's "mean Δ^LOCO ≥ 0.005 and corpus-level p ≤ 0.05" at
   K_train = 8 and SD 0.010 has these pass probabilities [mine]:

   | True mean gain | 0 | 0.005 | 0.010 | 0.015 |
   |---|---|---|---|---|
   | P(pass), K = 8 | 0.04 | 0.32 | 0.81 | 0.98 |

   It is a screen for gains of about 0.010 or more, not for the 0.005 SESOI. The gate is not
   miscalibrated (its type I error is 4%), but its stated purpose was. It is now labeled as a
   ≥ 0.010 screen.
5. **My meta-test requirement cannot be met jointly with training needs.** "K_test ≥ 20 corpora in
   ≥ 10 families" (`round1-opus-m.md:158`) leaves too few families for meta-train once families, not
   corpora, are the unit. I revise the SESOI and the unit in §5.4.
6. **p ≤ 3 features is a capacity rule, not evidence that my three are the right ones.** I keep
   log N (AdaWidth, `round1-opus-e.md:219-224`) and log participation ratio. For P1 I would take
   Opus-E's mean RDIME ε̂² over `Q_calib` (`round1-opus-e.md:303-304`), which has a stated
   derivation, over my top-1 cosine, which has none. I drop anisotropy unless the audit shows log PR
   and log N are nearly collinear across the pool.

### 3.2 Is HyperPrior too small to be called a hypernetwork?

**Yes, as a name. No, as a rung.**
- Formally it is a map from an unlabeled corpus to parameters of the selector (Ha et al.'s definition
  of a network generating another's weights, cited in `round1-astra.md:73`). In practice it is **a
  ridge regression from three corpus statistics to nine prior weights**.
- Calling it "the HyperDIME hypernetwork" would oversell it. I now call it **rung 1: a
  corpus-conditioned prior calibrator**. The word "hypernetwork" is reserved for rung 3 (§5), which
  generates per-dimension or low-rank corrections from descriptor rows.
- Its value is methodological. It is the strongest label-free corpus channel the evidence supports
  (V4), and it must be beaten before any per-dimension generation is credited. In that role it is a
  **comparator** for every larger design, in the same sense as the seed's H-2.

### 3.3 Does it leave real headroom on the table if the gates pass?

**Probably yes, on SciFact-like environments, and the ladder is what reveals this.**
- **Why there is headroom HyperPrior cannot reach.**
  - The prior weight only trades the analytic channel against a fixed W₀. It cannot change **which**
    dimensions W₀ prefers.
  - E1's cross-environment overlap is 0.33 against a random-mask expectation of 0.30 and a seed
    floor of 0.74 (`reports/E1/tables/noise_floor.csv`; Opus-E's arithmetic, `round1-opus-e.md:80`).
  - SciFact's identified prior share at 307 is only 0.36 (V4), so most of SciFact's in-domain
    advantage lives in W, not in the prior.
  - On a SciFact-like corpus, the oracle F9 will likely recover only a small share of the gap. If so,
    HyperPrior is the wrong final rung.
- **Why this is not a hidden cost.** Gate H2, the oracle capacity ladder on meta-train LOFO, measures
  F9, Fb, and Fr4 before any generator is trained. If F9 < 0.5 of headroom while Fb or Fr4 ≥ 0.5,
  the ladder **requires** escalation, provided G3 passes.
  - What HyperPrior would leave behind is the share between F9's and Fr4's oracle recovery, and that
    is measured.
  - If G3 fails, that share was out of reach for any descriptor-row generator on present evidence,
    and HyperPrior is not what forfeits it.
- **The real risk is the opposite one.** If gates H1 and H2 pass mainly because of one SciFact-like
  family, every rung's "gain" becomes a one-family result. That is why the merged program (§5)
  requires a leave-one-family-out sign check at every gate.

### 3.4 What I keep

- **G+V as the primary comparator.** All three proposals converge on it:
  - Astra's "locked strongest global comparator" retains the target's true variance prior
    (`round1-astra.md:107`).
  - Opus-E's "global-geometric, z = 0" is G+V (`round1-opus-e.md:295-296,424`).
  - Remaining amendments: train it **corpus-balanced** (family-uniform), not query-weighted (V1/V2)
    or sqrt-weighted. Lock it as the better of the Stage-A artifact and "Stage A continued for the
    treatment's step count", chosen on meta-train LOFO.
- **H1b**, target `v_T` against pooled `v`. It is the cheapest test of whether any corpus channel
  exists.
- **Shuffle and 1-NN nulls, the log-N-only control, and C10 self-training** as a rival to
  amortization.
- **Family as the unit**, exact sign-flip, REML prediction interval, harm guard, fixed-sequence
  primary chain.

---

## 4. The five most important unresolved disagreements

### D1. What is generated first, and what each rung must beat

- **Positions.** Seed, Opus-E, and I generate `a_k` first. Astra keeps `a_k` global and generates a
  rank-4 correction plus a bias (`round1-astra.md:91`).
- **Mine.**
  - The order is `a_k` (rung 1), then a per-dimension bias, then tied rank-r (rung 3).
  - **Each rung's primary comparator is the rung below it, not G+V.** Astra's model must carry a
    generated `a_k` head, or be compared with the `a_k`-only model. Otherwise its gain cannot be
    attributed to query-dependent correction.
  - Astra's own `a_k-only` and `bias-only` controls (`:107`) become the rung-1 and rung-2 references.

### D2. Does the G3 rerun bind, and for what?

- **Positions.** Astra: it gates the whole program (`round1-astra.md:99`). Opus-E: advisory
  (`round1-opus-e.md:451`). Me (round 1): it gates Level 2 only.
- **Mine.** It is **binding for every rung that maps descriptor rows to per-dimension outputs**
  (Opus-E's Δb and γ, Astra's A_T and c_T). It does not bind rung 1, which uses no descriptor rows.
  - The only defence of the round-1 failure was low power. A rerun on more families removes it.
  - Run E2 unchanged under its registered SC-001/002 (ADR-0013), on new meta-train corpora only.

### D3. Labels from meta-test environments

- **Positions.** Astra: none. Opus-E: in-domain C10, P2 C9, a census over the whole pool, and test-qrel
  re-splits for meta-train. My round 1: an in-domain reference after the lock.
- **Mine.** The rulings in §2: none, unless an ADR amends AGENTS.md. Gap closure moves to LOFO. P2
  needs its own environment set. A re-split corpus is meta-train-only forever.
- I expect this to be settled by the user, since it touches a hard constraint. Until then the merged
  program assumes the strict reading.

### D4. Joint or frozen base, and how training is weighted

- **Positions.** Opus-E trains W₀ jointly with the hypernetwork, with sqrt(n) sampling
  (`round1-opus-e.md:359,366`). Astra freezes a family-uniform pooled base (`round1-astra.md:91,100`).
  I freeze a corpus-balanced base.
- **Mine.** **Freeze W₀** for every confirmatory treatment, with family-uniform then
  environment-uniform sampling.
  - Joint training confounds conditioning with a different base and leaves `a_k` unidentified (§1.3,
    item 5).
  - G+V is locked as the better, by meta-train LOFO, of the Stage-A artifact and "Stage A continued
    for the treatment's step count". This is Astra's matched-optimization point, without the
    degenerate template arm.

### D5. Confirmatory scale: SESOI, unit, and the role of meta-validation

- **Positions.** All three proposals aim at SESOI 0.005. Astra needs ≈ 32 families, Opus-E needs
  n_E with ≥ 80% power, and I asked for ≥ 20 corpora in ≥ 10 families. None of these fits a pool of
  about 14–18 families split between training and testing (§5.4).
- **Mine.**
  - The unit is the **family** (Wikipedia and StackExchange clusters count once).
  - The confirmatory primary is **superiority powered at a true family-mean gain of 0.010**. At an
    SD of 0.010, an exact one-sided sign-flip at α = 0.05 has these powers [mine]:

    | K families | 6 | 8 | 10 | 12 |
    |---|---|---|---|---|
    | Power at gain 0.010 | 0.66 | 0.81 | 0.90 | 0.95 |
    | Power at gain 0.005 | 0.27 | 0.34 | 0.43 | 0.50 |

    At SD 0.015, power at gain 0.010 and K = 10 falls to 0.63.
  - Non-inferiority at 0.005 (≈ 25+ families) and the 0.005 superiority are reported as
    **estimation**: a REML prediction interval, not a test.
  - Meta-validation, at 2 families, cannot be a power gate. Astra's and Opus-E's meta-val thresholds
    pass null methods 16–27% of the time (§1.2, item 6; §1.3, item 9). So the **go/no-go comes from
    meta-train LOFO** (more units). Meta-val keeps two jobs: a harm check, and choosing among ≤ 2
    variants.

---

## 5. The merged gated program I now support

**Conventions.**
- Every threshold is fingerprinted in an ADR before its first number exists (ADR-0018 pattern).
- Exact float64 full-corpus scoring, lower-row tie rule, documents untouched, masked query not
  re-normalized.
- Endpoint: per-query nDCG@10 averaged over the deciles 102..921, then the corpus mean, then the
  family mean.
- "LOFO" means every quantity for family f, including the G+V base, standardization, ridge λ and
  hyperparameters, is fit without f.
- Training sampling is family-uniform, then environment-uniform.
- Seeds 0–2 are averaged per query before testing.

### 5.1 Rungs, unlock thresholds, and kill criteria

| Rung | Content | Data | Unlock next rung if | Kill (then) |
|---|---|---|---|---|
| **R0 Protocol and pool** | Pool audit: families by source and ≥ 1% near-duplicate overlap, licences, Qwen3 SFT exposure. Resource gate (ADR-0018 style; small and medium corpora first). S4 recipe frozen unchanged (ADR-0012). Split manifest randomized within family × label-availability strata, then fingerprinted. User decision on the re-split ADR (§2.3); default no. | none read | Manifest frozen with ≥ 5 meta-train families and ≥ 8 meta-test families | < 5 meta-train families: stop; the program is a dev-only study. < 8 meta-test families: run the program, but **pre-declare R5 as estimation-only** |
| **R0′ Dev-only, exploratory** (in parallel; no gate) | On SciFact/NFCorpus/MS MARCO with existing artifacts: legacy global (C4) against corpus-balanced global (C4b) against G+V (C5). Finish F12 MS MARCO. | dev | — | — (labeled exploratory, D9; tests I1 cheaply) |
| **R1 Meta-train census** | H0 recipe replication (S4 against S3 and `variance`, δ = 0.005). H1 headroom `H_f` = in-domain S4 − G+V. H1b `v_T` against `v_pool`. H2 oracle ladder F9 / Fb / Fr4 as share of headroom, with prior share reported next to `a_k*`. H3 ICC of the chosen family's targets (split halves × 3 seeds). Opus-E's census count. M2 MS MARCO shards (descriptive). | meta-train train/val labels; LOFO | Family-mean H ≥ 0.010 with exact one-sided sign-flip p ≤ 0.05 over families; leave-one-family-out mean ≥ 0.005 for every left-out family; ≥ 30% of corpora with point H ≥ 0.005 (only corpora with ≥ 150 evaluation queries); ICC ≥ 0.7 for the chosen output family | Upper 95% bound of family-mean H < 0.010: **ship G+V**, ADR. ICC < 0.5: stop. H1b fails: soft kill (continuing needs an ADR). F9, Fb and Fr4 all < 0.25: the gap needs labels (P2, separate environment set) |
| **R2 Rung 1: prior calibrator** (HyperPrior = HyperDIME-0) | Ridge `log a_k*` (and prior share) on p ≤ 3 pre-registered φ: P0 {log N, log PR}, P1 {+ mean RDIME ε̂²}. W₀ frozen. | meta-train; LOFO | **All of:** LOFO mean gain vs G+V ≥ 0.005 (point); sign-flip p ≤ 0.05 over families; above the 95th percentile of the shuffle null (refit per permutation); ≥ 1-NN (C6); ≥ log-N-only (C6b); ≥ 0.4 × oracle F9 gain | Fails the shuffle null or p > 0.05: no corpus-conditioned calibrator. Deliver G+V, or C10 if it wins |
| **R2-G3 (parallel)** | E2 rerun unchanged, SC-001/002 (ADR-0013), on new meta-train corpora | meta-train | Passes as registered | Fails: **every per-dimension rung is closed** |
| **R3 Rung 2/3: hypernetwork proper** | Only if R2-G3 passes **and** oracle Fb or Fr4 ≥ F9 + 0.25 of headroom. At most 2 candidates, chosen by LOFO: (a) Astra P0/P1 rank-4 + bias **plus** the rung-1 `a_k` head, W₀ frozen; (b) Opus-E tied Δb map, latent width ≤ ⌊K_fam/3⌋, W₀ frozen, disjoint calibration queries | meta-train; LOFO | LOFO mean gain **vs rung 1** (vs G+V if rung 1 failed but G3 passed) ≥ 0.005 (point), sign-flip p ≤ 0.05; correct-minus-donor-descriptor > 0, p ≤ 0.05 | Fails: rung 1 (or G+V) is final. Per-dimension generation recorded as unsupported |
| **R4 Meta-val** (2 families) | Harm check; choose among ≤ 2 locked variants (P0/P1, or rung 1/3). **Not a power gate.** | meta-val labels | No family with point Δ < −0.01 vs G+V; then lock and fingerprint (checkpoints, manifest, evaluation ids, analysis code) | Any family Δ < −0.01: do not open meta-test; diagnose on meta-train only; at most one re-lock, recorded |
| **R5 Meta-test, once** | Scoring-only process against the lock | meta-test test qrels, evaluation only | — | §5.3 |

Why is R1's headroom gate "≥ 0.010 with a p-value" when R2's is "≥ 0.005 point plus p"?
- At R1, the quantity is a ceiling. It must be comfortably above the SESOI, or no learned rung can
  clear R5's 0.010-powered test.
- At R2/R3, the gain is a LOFO prediction. A point ≥ 0.005 plus significance is the screen: at
  K = 8 it passes 0.81 of true-0.010 methods and 0.04 of nulls [mine, §3.1 item 4].

### 5.2 Comparators (same evaluation queries, deciles plus 1024 parity, exact evaluator)

| ID | Arm | Role |
|---|---|---|
| C0 / C1 | Full 1024 / MRL prefix-k | Mandatory (protocols, R5) |
| C3 | `variance` on the target | The analytic corpus channel alone |
| C4 | Legacy query-weighted global L2S | Continuity with E1 only |
| C4b | Corpus-balanced global, no prior | Supervision without corpus information |
| **C5 = G+V** | C4b + target `v_T` prior with global `α_k`; locked as best of Stage-A or continued for equal steps | **Primary comparator for rung 1**; Opus-E's z = 0 and Astra's locked global are this arm |
| C5p | G+V with pooled `v` | H1b |
| R1-ref | Rung 1 (prior calibrator) | **Primary comparator for rung 3** (D1) |
| C6 / C6b | 1-NN family's parameters / log-N-only ridge | Library lookup; size confound |
| C7 | Donor (hash-assigned meta-train family) or shuffled descriptor, `v_T` kept | "Uses corpus information" (superiority, not equivalence) |
| Ablations | bias-only, `a_k`-only, fixed-budget rank-4 (Astra) | Mechanism attribution within rung 3, descriptive |
| C9 | PRF-DIME, Eclipse | Label-free per-query rivals; online cost separate |
| C10 | Per-corpus self-training from pseudo-queries or Eclipse pseudo-labels (my C10 = Opus-E C8), hyperparameters fixed on meta-train | Rival to amortization (H5): if C10 ≥ treatment, the claim becomes cost |
| — | **No** in-domain or P2 arm on meta-val/meta-test environments | §2.1 |

### 5.3 Meta-test analysis (pre-registered)

- **Primary chain**, fixed-sequence, each step one-sided at α = 0.05, family means, exact sign-flip
  (all 2^K vectors):
  1. Locked treatment > C5, powered for a true 0.010 (§4, D5).
  2. Locked treatment > C7, the donor control: it uses corpus information.
  3. The other protocol (P0 if P1 is primary, or vice versa) > C5.
  4. Practical: point ≥ 0.005 **and** a lower 95% bound > 0.
- **Always reported, not tested:** REML μ, τ, and the **95% prediction interval for a new family**;
  non-inferiority at δ = 0.005 read off it (descriptive); per-family effects; leave-one-family-out
  signs; the slope of Δ on distance to the nearest meta-train family; strata by Qwen3 SFT exposure.
- **Harm guard:** per corpus, one-sided test of Δ < −0.01, Holm over corpora. "Safe to deploy"
  requires none.
- **Secondary families (Holm within each):** treatment against {C3, C4b, C6, C6b, C9, C10}; P0
  against P1; per-corpus query-level replication counts; gap closure **on meta-train LOFO and
  meta-val only**.
- **Kill at meta-test:** step 1 fails, or a leave-one-family-out sign flips. Then there is no
  generalization claim, G+V (or rung 1, if it passed step 1 as the locked arm) is the deliverable,
  and the result is recorded as evidence.
- **Downgrade, pre-declared:** if the meta-test has < 8 families, or if the upper 80% limit of the
  LOFO between-family SD exceeds 0.012, R5 is run as estimation only (prediction interval and
  per-family effects; no superiority claim).

### 5.4 Corpus budget

All family counts are **[unverified]** until R0's audit.

| Pool | Content (candidates) | Families | Role |
|---|---|---|---|
| Dev | SciFact, NFCorpus, MS MARCO | 3 (sci-claims, bio-nutrition, web) | Meta-train only; never meta-val or meta-test (D9) |
| Label-bearing new | NQ / HotpotQA / FEVER (one Wikipedia family), FiQA, Quora (dev), LoTTE dev, BioASQ (licence) | ≈ 3–5 | **All to meta-train.** It is the only source of training labels, and K_train binds learning |
| Test-only | ArguAna + Touché (argument), TREC-COVID, SCIDOCS, Robust04 / TREC-NEWS (news), Signal-1M (social), BRIGHT coding and math domains, CQADupStack / LoTTE test (if StackExchange is not in meta-train), others the audit finds | ≈ 8–12 after overlap exclusions (Climate-FEVER and DBPedia fall in the Wikipedia family; SCIDOCS against SciFact and TREC-COVID against NFCorpus need overlap checks) | Randomized by seed: **2 families to meta-val, the rest (target ≥ 10, minimum 8) to meta-test** |
| Augmentation | MS MARCO topical shards; pseudo-corpora | 0 independent | Meta-train episodes and M2 only; labeled as shards; never counted in K |
| Optional expansion (needs an ADR and an audit) | Multilingual retrieval sets with training labels (Qwen3 is multilingual) | ? | Could raise K_train. Language shift is a different environment axis; report it as a separate stratum |

**Consequences.**
1. Meta-test is drawn from test-only collections, while meta-train holds the label-bearing ones. That
   is a systematic population difference. It matches deployment (new, unlabeled corpora), but it
   must be stated as the scope of the claim.
2. Re-splitting test-only sets for training (§2.3) moves families out of meta-test one for one.
   With about 8–12 test-only families, every re-split family costs about 10% of meta-test power.
   I recommend against it unless R1 shows K_train < 5 families.
3. Embedding cost is dominated by the 2.7–5.4M-document Wikipedia corpora. One Wikipedia corpus in
   meta-train (NQ, 2.7M) suffices for the family. FEVER, Climate-FEVER, and DBPedia add no independent
   unit.

### 5.5 What this program buys, stated plainly

- With about 5–7 meta-train families, R1 and R2 can detect **large** corpus-conditioned effects of
  about 0.010 or more, and they can stop the program cheaply when the landscape is flat. R2 needs no
  hypernetwork code.
- With 8–12 meta-test families, R5 can confirm a **family-mean gain of about 0.010**. It cannot
  confirm 0.005, and it cannot confirm non-inferiority. Those are reported as estimates.
- A "hypernetwork proper" (R3) is built only if two independent pieces of evidence agree:
  - the oracle says per-dimension corrections carry headroom beyond the prior weight;
  - the G3 rerun says descriptor rows predict them.
- My prior, as opinion: R1 passes on one or two SciFact-like families. Rung 1 captures a real but
  small share. G3 decides whether rung 3 is ever trained. P1 beats P0, which is Opus-E's R1 rival.
  Every one of these outcomes is identifiable under the program above.

## Sources consulted this round

- Round files: `BRIEF.md`, `round0-seed.md`, `round1-astra.md`, `round1-opus-e.md`,
  `round1-opus-m.md`.
- Repository: AGENTS.md; `docs/protocols.md`; `specs/004-l2s-baseline/research.md:51`;
  `src/hyperdime/cli/train_global.py`; `specs/003-data-embeddings/tasks.md:76`;
  `docs/experiment_registry.md` (P3 row); ADR-0011 and ADR-0012 (evidence sections);
  `reports/E2/tables/criteria.csv`.
- Computations: `/tmp/hm-r2/gates.py` and `/tmp/hm-r2/k.py` (gate operating characteristics,
  sign-flip power, SD-estimate intervals), project venv, not committed.
- Papers: none newly opened this round. Paper claims are taken from the round-1 files with their
  page citations.
