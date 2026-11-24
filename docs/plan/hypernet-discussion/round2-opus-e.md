# Round 2 — Opus-E (evidence lead): cross-critique, concessions, merged design

Author: Opus-E (Claude Opus 5.5). I read `BRIEF.md`, `round0-seed.md`, `round1-astra.md`,
`round1-opus-m.md` and my own `round1-opus-e.md`. The repository is read-only except this file.
I trained nothing, downloaded nothing and committed nothing.

**Method.** I used the scientific-critical-thinking workflow: fix the unit, extract the evidence
before judging, and keep data apart from interpretation. From the peer-review skill I used the
comment structure: location, observation, evidence, why it matters, and requested action. Its
confidentiality intake gate is for unpublished third-party manuscripts. It does not apply to
internal design notes the user owns, so I did not run it.

**Tags.**
- **[V]** I opened the file or PDF page and read the number myself during this round.
- **[C]** I recomputed the number. Scripts are in `/tmp/oe2-scripts/` and are not committed.
- **[R]** Restated from a narrative, not cross-checked.
- **[I]** My interpretation.

PDF page numbers count the cover page. Paper text was extracted with pypdf to `/tmp/oe2/`.

---

## 0. Bottom line

1. **The orchestrator's fact holds, and it is sharper than "about 94%".**
   - `src/hyperdime/cli/train_global.py:54-60` concatenates every environment's training rows with
     no reweighting. Only the choice of epoch averages validation over environments (`:80-81`).
   - The F2 training-query counts are 45,000 MS MARCO, 728 SciFact and 2,331 NFCorpus
     (`reports/P3/README.md:46-48, 60-64`) **[V]**. That makes MS MARCO **93.6%** of training rows
     **[C]**. This assumes one target row per training query, which `targets_at(*train)` implies
     **[R]**.
   - It is also **not leave-one-corpus-out**: the target's own training queries are in the union.
2. **Re-assessed, the headroom picture changes per environment, not wholesale** [I]:
   - **MS MARCO.** E1's −0.002 compares an in-domain selector with what is essentially another
     MS MARCO selector. It says nothing about transfer.
   - **NFCorpus.** The +0.017 is measured against a global selector that is *below full and below
     `variance`* (0.3462 vs 0.3619 and 0.3587, `reports/overview/README.md`) **[V]**. So the
     headroom over a variance-containing comparator is smaller.
   - **SciFact.** The gap probably survives. Both donor environments' in-domain selectors lose
     0.161 and 0.188 there (E1-b), more than the union global loses (0.11). A *leave-one-corpus-out*
     corpus-balanced global would see no SciFact labels at all.
   - So Opus-M's I1 is probably wrong about SciFact. It is right about NFCorpus and MS MARCO.
3. **The three of us (and the seed) made one shared error.** We called the S3/S4 prior weight
   "the environment-level parameter that varies by orders of magnitude". Raw `a_k` is not
   scale-identified. The scale-free S4 `prior_share` at k = 307 is 0.95 (MS MARCO), 0.43 (NFCorpus)
   and 0.36 (SciFact) (`reports/S4/tables/prior_share.csv`) **[V]**. What really differs is the
   *shape over budgets*:
   - flat at about 0.95 on MS MARCO;
   - falling to exactly 0 by k = 716–819 on SciFact and NFCorpus.

   That breaks the shared-β form of HyperPrior (§1.3).
4. **A structural reason to expect HyperPrior to recover little where the headroom is** [I]:
   - A per-budget prior weight can only trade off the leave-one-corpus-out global correction
     against `variance`.
   - On SciFact both are about 0.12–0.14 below in-domain (global 0.7105, `variance` 0.6964,
     weighted 0.8340) **[V]**.
   - It is still the right *first rung*, because it is cheap and needs no new code. The oracle
     ladder (Opus-M's H2) must include the per-dimension bias Fb and rank-4 from day one, and it,
     not a prior, should pick the output family.
5. **Factual audit.**
   - **Opus-M.**
     - The power arithmetic reproduces exactly **[C]**.
     - The BEIR document counts and label availability agree with Search-Adaptor Table 7
       (p. 12) on all 14 overlapping corpora **[V]**.
     - The paper quotes check out: Qwen3 Table 6 p. 13; Leblanc §4.3 p. 8; Unveiling DIME §5.3;
       L2S Table 5 p. 8.
   - **Astra.** Most numbers check out. One is misread: 0.3321 is a SciFact random-mask value. The
     NFCorpus one-positive transfer regret is 0.2397. One is selectively reported: R0's +0.0190 is
     measured against the *untuned* REF, and against tuned L2S the contrast is +0.0054 and not
     significant. The literature citations are all to first pages.
6. **The binding constraint is the number of independent families, not the architecture.**
   Astra asks for 32 families and Opus-M for 20 corpora in at least 10 families. The public pool
   gives roughly 8–12 families for meta-train and meta-test together (§1.2). The merged design
   (§5) is therefore an estimation study with gated screens and pre-declared downgrades. It is not
   a confirmatory trial powered at 80%.

---

## 1. Review of each proposal

### 1.1 Round 0 seed (`round0-seed.md`)

**Strongest ideas.**
- **The capacity ladder H-0 → H-3, where each rung must win on unseen corpora** (`:34-35`). All
  three round-1 proposals converged on it independently: Astra's ablations, Opus-M's H2 ladder, and
  my HyperDIME-0 → HyperDIME-R.
- **The "feature-augmented global selector must be a comparator" remark** (`:35`). It is the seed
  of Astra's matched-global control.
- **Gate 3, LOCO ridge from about 5 descriptors against predict-the-mean with a permutation test**
  (`:44`). This is Opus-M's H4 almost verbatim.
- **Pseudo-corpora "checked to reproduce the real spread of α"** (`:39`). This is a useful
  validity check for augmentation, and none of the round-1 files kept it.

**Errors and unsupported claims.**
- **`:15` "S3 prior weight ≈1.0 MS MARCO, ≈0.002–0.02 NFCorpus/SciFact".** The magnitude is in
  raw units (see §0.3). The seed does flag the MS MARCO weight as an initialization artifact
  (`:17`), which is correct. The scale-free reading is `prior_share` 0.95 / 0.43 / 0.36.
- **`:18-20` "predicting per-dimension selection rate beats B1 (0.54/0.68/0.79 vs 0.44/0.56/0.54)".**
  These are the *direct* probe's variance-only cells: 0.5366 / 0.6815 / 0.7864 in
  `reports/E2/tables/ladder.csv` **[V]**. The seed omits B2, the environment's own `variance`
  profile, which is 0.5381 / 0.6246 / 0.6687. On SciFact the direct probe does *not* beat B2
  (−0.0015). Most of that "signal" is variance, as Astra says (`round1-astra.md:67`).
- **The generator emits `U_T`, `V_T` per corpus from a table that includes a dimension-ID
  embedding** (`:29-33`). With fewer than 10 training families this is the memorization path
  (plan R2). Opus-M (`:558-559`) and I (round 1 §3.2) both excluded ID embeddings.
- **The model uses `α_T(k)·(q⊙q⊙v_T)` (linear), not S4's `log(q²v + ε)`** (`:28`). These are
  different priors. The log form is the one validated (S2–S4).

**Underrated risk.** "In-domain S4 as ceiling" (`:49`) trains on a meta-test environment's labels.
See §3.1.

### 1.2 Astra (`round1-astra.md`)

**Strongest ideas.**
1. **Departure standardization `X = (F_T − M)/s`, with the absolute `v_T` kept separate**
   (`:79`). This is the right lesson from E2. The generator should see how a coordinate departs
   from its usual statistics, because that is what would predict a *change*. The direct probe's
   success comes from the usual part, which `b_0` and the prior already carry.
2. **The matched-global control** (`:107`). It uses the identical generator, fed a fixed meta-train
   template, and gets the same second-stage optimization from the same base. A continued-training
   S4 arm is added too. This is the only control proposed in round 1 that separates "extra
   optimization or parameters" from "corpus information" *at matched capacity*. My `z = 0` control
   had fewer free parameters than the treatment and so did not.
3. **The foreign-descriptor donor control** (`:107`). It swaps X only and leaves `v_T` untouched.
   It isolates the learned channel from the analytic one. It is cleaner than my random-environment
   C6, which also swapped the scalars.
4. **Mechanism kill by an equivalence band** (`:113`): "paired 90% interval wholly inside ±0.002".
   This is the correct way to *reject* a mechanism, rather than reading a non-significant
   difference as no difference.
5. **Exact accounting is correct** **[C]**:
   - `D² + D + 9 = 1,049,609`;
   - `rD + h(r+1) = 4,096 + 90 = 4,186` for P0 (h = 18) and `4,096 + 215 = 4,311` for P1 (h = 43);
   - `2Dr = 8,192` extra MACs per query.
   - The descriptor widths 18 and 43 match `src/hyperdime/space/descriptor.py:38-50` and the E2
     context table **[V]**.
6. **The cost/break-even definition and the G5 proposal** (`:103`). These are the most concrete
   in the discussion.
7. **No in-domain denominator at confirmation** (`:109`). This avoids the admissibility problem
   altogether (§3.1).

**Errors and unsupported claims.**

| Loc. | Claim | Check | Consequence |
|---|---|---|---|
| `:69` | "transferring one positive's mask to another at 307 incurred 0.3321 normalized regret versus 0.1292 with all positives visible" (NFCorpus) | `reports/P3-shared-mask/README.md:25`: NFCorpus one-positive transfer at 307 = **0.2397**; privileged all-positive mask on the same queries = 0.1289 (0.1292 is the multi-positive shared-mask row, `:104`). **0.3321** occurs only as SciFact's *uniform random mask* regret at k = 716 (`:16`, `:88`) **[V]** | Misread. The qualitative point survives: transfer costs about 1.9×, not 2.6×. |
| `:69` | "direct ranking R0 gained 0.0190 versus REF on SciFact" | Correct number (`reports/P3-objective/README.md:12`) **[V]**, but REF is L2S at untuned lr 1e-4. Against **tuned** L2S (C8, R0 − KG): SciFact +0.0054 [−0.0062, +0.0170], NFCorpus −0.0036 [−0.0109, +0.0038], Holm p 1.000 (`:33`) **[V]** | It supports "R0 is a safe objective", not "R0 helps". P3-objective's own conclusion is that only lr tuning (C11) is supported. |
| `:67` | E2 failed in all 18 cells; SciFact all-P1 0.4623 vs identity 0.4354 vs variance 0.5381; MS MARCO direct 0.7864, shift 0.5657, B2 0.6687 | All match `reports/E2/tables/criteria.csv` and `ladder.csv` **[V]** | Correct. |
| `:68` | S1 better than L2S in 2/3; S4 prior share at 307 is 0.9512 / 0.4281 / 0.3580 | `reports/S1/README.md:15`; `reports/S4/tables/prior_share.csv` **[V]** | Correct. Astra is the only participant who reported `prior_share`, not raw `a_k`. |
| `:71-73` | Every paper is cited at "p. 1" | Each claim is a one-line characterization that a title page can support. None extracts the facts that bear on this design. The most relevant one is Leblanc §4.3, p. 8 **[V]**: with **10** meta-train tasks of "subtle" difference, their hypernetworks perform "similar to those of the strawman", "a single predictor … yielding competitive performances". | It is the closest published analogue to our regime, and it predicts that the matched-global control wins. Astra's design never engages with it. |
| `:111` | 32 families needed (SD 0.010, two-sided) | Arithmetic is right: 31.4 **[C]**. Not feasible: Search-Adaptor Table 7 (p. 12) lists 14 BEIR corpora **[V]**, and they fall into about 8–10 families (Wikipedia: NQ, HotpotQA, FEVER, Climate-FEVER, DBPedia) | The pilot fallback (`:111`) will be the actual outcome. It should be planned as such (§4, D4). |

**Underrated risks.**
- **The rank-4 query interaction `A_T(Bq)` is the least evidence-backed component, yet it is
  Astra's headline.** No project result shows that a corpus changes *how queries use* coordinates
  as opposed to *which coordinates are generally useful*:
  - E2 tested per-dimension frequency, not query × corpus interaction.
  - ADR-0001's query-conditioned design "did not beat the linear selector" [R].
  - Hypencoder Revisited reports uneven out-of-domain transfer (§7) (my round 1 §2.1 T2).

  Astra itself names the per-budget prior as "a credible simpler alternative" (`:95`). The order
  of evidence argues for bias-only and `a_k` as primary, with rank-4 as an escalation that the
  oracle ladder must license (D1).
- **Making the G3 replication a program-wide kill** (`:99`, "a repeated failure stops the generator
  program"). E2's target is the S1 selection-frequency profile, a per-dimension quantity. It does
  not test environment-level outputs such as `a_k` or the mixing weight. Elsewhere (`:95`) Astra
  says the budget-weight arm needs separate comparison. Gating it on E2 is a category error (D2).
- **Train/test mismatch from in-sample centering.** `M_ij` is the mean of the meta-train
  descriptors, so each meta-train environment's X is shrunk toward 0 by its own share of M:
  - training X has variance about `(K−1)/K · σ²`;
  - a new environment's X has variance about `(K+1)/K · σ²`;
  - at K = 7 families the ratio is 1.33 **[C]**.

  The generator therefore sees larger departures at test time than it ever saw in training. A fix
  is to centre each training environment by the leave-it-out mean.
- **A k = 307 primary endpoint** (`:109`). The leading candidate output is per-budget, ADR-0018/F12
  uses the decile mean, and both other participants use the decile mean. A single-budget primary
  would score a per-budget head on one of its nine outputs (D5).

### 1.3 Opus-M (`round1-opus-m.md`)

**Strongest ideas.**
1. **G+V as primary comparator** (`:16-18`, `:401`). It is corpus-balanced and contains the target
   corpus's `v_T` with global per-budget α. I adopt it. My C3 was the same model trained with
   `sqrt(n_train)` sampling, which is about **74%** MS MARCO by expected episode share **[C]**. That
   is not corpus-balanced, so I concede the point (§2).
2. **Gate H1b, `v_T` against `v_pool`** (`:310-319`). It is the cheapest test of whether *any*
   target-corpus statistic matters. Nobody else proposed it. If it fails, the prior on a learned
   corpus channel should drop sharply.
3. **The H2 oracle ladder with a ratio-of-sums statistic and the label-quantity subsampling
   control** (`:321-341`). It tells us which outputs are worth predicting before predicting any.
4. **H3 reliability via an ICC** (`:343-349`). This is the right question, because E2's seed
   ceilings show the targets are noisy: SciFact seed Spearman 0.60, `reports/E2/tables/check.csv`
   **[V]**.
5. **H4 nulls** (`:358-367`): shuffle, 1-NN and `log N`-only. With a 3-feature ridge they are
   computable by full enumeration.
6. **Qwen3 exposure tagging** (`:268-270`). This is verified: Table 6, p. 13 lists "MS MARCO, NQ,
   HotpotQA, NLI, …" in SFT **[V]**. Along with the instruction freeze and the generic-instruction
   control (`:262-267`), it is the most complete treatment of rival R1 in round 1.
7. **The harm guard and the prediction interval as the reported quantity** (`:439-441`,
   `:461-463`).

**Factual checks.**
- **D2, the training counts.** 728 and 2,331 are not merely derived; they are the F2 counts in
  `reports/P3/README.md:62-64` **[V]**.
- **D3** matches `reports/overview/README.md` **[V]**.
- **D6** matches `reports/E1/tables/noise_floor.csv` (0.7445 / 0.3295) **[V]**.
- **D12**, L2S Table 5, p. 8 **[V]**. GritLM rows (Full / Unsup / Supervised):
  - SciFact 0.786 / 0.790 / 0.902;
  - NFCorpus 0.439 / 0.441 / 0.459;
  - MS MARCO 0.536 / 0.602 / 0.626.

  So pseudo-queries recover about 73% of the supervised gain on MS MARCO, about 3% on SciFact and
  about 10% on NFCorpus **[C]**. This is direct evidence for rival R1, and a warning that C10
  (pseudo-query tuning) will fail exactly where the headroom is.
- **D13**, Unveiling DIME §5.3, p. 8, "Different tasks favor different levels of pruning", and
  Table 3, p. 10 **[V]**.
- **D16**, Leblanc p. 8. The quote is exact **[V]**. Two corrections:
  - The 10-task regime is the pixel-swap experiment, not "90–300 tasks". It is the one *closest to
    ours*, and there the method tied the strawman.
  - Binary MNIST has 90 tasks, of which 34 are meta-test.
- **The BEIR table** (`:175-195`), checked against Search-Adaptor Table 7 (p. 12) **[V]**:
  - Every listed document count matches: 3.6K, 5K, 8.67K, 25K, 57K, 171K, 382K, 523K, 2.68M,
    4.63M, 5.23M, 5.42M, 8.84M.
  - The "has non-test labels" column is consistent with Table 7's train/test pair ratios.
    Search-Adaptor's train ≈ test pairs (re-split) for ArguAna (703/703), SciDocs, TREC-COVID,
    Touché, NQ (2,097/2,104), Climate-FEVER.
    Its native train pairs are much larger than test for NFCorpus, SciFact, FiQA, HotpotQA, FEVER
    and MS MARCO.
    Quora (7,626/15,675) and DBPedia (5,673/43,515) match "dev only".
  - NQ: Search-Adaptor re-split BEIR-NQ test rather than use nq-train. That is consistent with
    Opus-M's note that nq-train has a different corpus build.
  - Not verifiable from local sources (consistent with my recollection, still **unverified**):
    BioASQ, LoTTE, CQADupStack, BRIGHT, TREC-COVID's 50 and Touché's 49 queries.
  - Omitted:
    - MAIR, with 126 tasks, about 80 labeled queries each, in 6 domains (Align then Train §5,
      p. 6; my round 1). It is a breadth source for meta-train and P2, not for per-environment
      oracles.
    - MIRACL, which is multilingual and out of scope; omitting it is correct.
- **Power** (`:117-164`), re-run independently with SciPy **[C]**:
  - closed form: K = 27/19/15/12/8/5 for d = 0.5/0.6/0.7/0.8/1.0/1.5 at one-sided α = 0.05;
  - at one-sided α = 0.025: 34/19/10/6;
  - the MDE row: 2.30, 1.19, 0.98, 0.85, 0.77, 0.68, 0.61, 0.52, 0.46;
  - D1: mean 0.0423, SD 0.0608, d = 0.695;
  - simulation, f = 0.5: 0.45/0.63/0.76/0.85/0.92/0.98, versus Opus-M's 0.43/0.64/0.76/0.84/0.94/0.98;
  - non-inferiority: d = 0.530 → K = 24.

  **The arithmetic is right.** My objections concern the inputs and the unit:
  1. **The planning gaps are against the legacy, query-weighted, non-LOCO global** that Opus-M
     says inflates headroom. Against G+V, the NFCorpus and MS MARCO gaps are plausibly ≈ 0 (§0.2).
     The "30% of corpora have large gaps" mixture rests on 1 of 3 environments. The exact 95%
     Clopper–Pearson interval for that proportion is 0.008–0.906 **[C]**.
  2. **The recovery f is not design-neutral.** For HyperPrior, f on SciFact-like corpora is
     bounded by how far mixing a LOCO global with `variance` can go, which is far less than 0.5
     (§0.4).
  3. **There is a unit inconsistency.** §8 (`:435`) makes *family means* the primary analysis. §3
     sizes the study in *corpora* (K_test ≥ 20 spanning ≥ 10 families), and §4 (`:202`) says the
     meta-test pool covers "only about 10–14 families". With family units the effective K is
     ≤ 10–14 before removing the meta-train families. At K = 7–8, simulated power at f = 0.5 is
     0.53–0.63, and at f = 0.25 it is 0.35–0.43 **[C]**.

**Errors and unsupported claims.**
- **I1** (`:64-69`): "A corpus-balanced global selector could close much of the SciFact gap with no
  corpus information at all."
  - Gate H1 itself specifies that G+V be **fit LOCO** (`:298`). Under LOCO, SciFact's labels are
    absent.
  - The two remaining donors transfer to SciFact *worse* than the union global did: in-domain
    selectors from MS MARCO and NFCorpus lose 0.161 and 0.188, against the union global's loss
    of 0.1116 (in-domain 0.8353 vs global 0.7238 at k = 307) **[V]**.
  - The likely result is that SciFact's headroom stays at about 0.1 [I]. The 93.6% fact mostly
    affects the NFCorpus and MS MARCO rows.
- **D4/I2** "spans more than an order of magnitude" (`:73-74`). This uses raw `a_k` (§0.3). Under
  HyperPrior's *shared, frozen* `W_0`, raw `a_k*` is identified. Its cross-corpus spread under a
  shared `W_0`, however, has never been measured. All existing `a_k` values come from different
  in-domain `W`s. H2/F9 is the first measurement.
- **The functional form `a_k(T) = softplus(α_k + βᵀφ̃)` with β shared across budgets** (`:502`).
  When the softplus argument is small, this is log-additive: the corpus shifts `log a_k` by the
  same amount at every k. The S4 weights (`reports/S4/tables/weights.csv`, seed-mean) **[V]** show
  corpus-dependent *shapes*:
  - The SciFact / MS MARCO ratio is about 0.133/0.88 ≈ 0.15 at k = 102, about 0.016/0.83 ≈ 0.02 at
    k = 716, and 0 at k = 819 and 921.
  - NFCorpus is exactly 0 for k ≥ 716.

  A shared β cannot fit this. Opus-M lists "shared vs per-budget β" only as an ablation (`:415`).
- **Stage B targets `log a_k*` on a grid `{0} ∪ logspace(−3, 0, 13)`** (`:533-535`). There are two
  defects:
  1. `a* = 0` is a likely optimum: S4 reaches exactly 0 at high k. Its log is −∞, so ridge on
     `log a*` is undefined. A censored model or a bounded parameterization is needed.
  2. The grid caps at 1. MS MARCO's S4 `a_k` runs to 0.998 (seed 1, k = 102), and S3's MS MARCO
     weight never left its initial value of 1.0 (ADR-0011 [R]; seed `:17`). Under a corpus-balanced
     `W_0` with a different norm, the optimum may exceed 1.

  **Fix.** Parameterize a bounded mixing weight `λ_k ∈ [0, 1]` on standardized terms,
  `z = λ_k·std(prior) + (1−λ_k)·std(Wq + b)`. This is the quantity behind `prior_share`, and it
  makes both endpoints (`variance` alone, correction alone) reachable. Then regress `logit λ*`
  with a ridge that clips to [0.01, 0.99]. Allow one budget slope per feature:
  `logit λ_k = α_k + βᵀφ̃ + γᵀφ̃·k̃`, which is 9 + 3 + 3 = 15 parameters.
- **Leakage rule 4** (`:255-259`, "test-only corpora are never trained on") is a correct literal
  reading of AGENTS.md. Its consequence goes unpriced. Excluding the dev three, the labeled
  non-test pool is FiQA, HotpotQA, FEVER, NQ-train, Quora-dev, DBPedia-dev (67 queries), BioASQ
  (licence) and LoTTE-dev. That is about 4 new *families*: Wikipedia, StackExchange/finance, Quora
  and biomedical. BioASQ joins NFCorpus's family. After a 50/50 family-blocked split, meta-train
  gets about 2 new families, so K_train ≈ 5 families. The gates H1–H4 then have an MDE of
  d ≈ 1.2–1.4 **[C]**. They can only detect huge effects. See D3.
- **H1's "corpus-level p ≤ 0.05, which needs K ≥ 5 new labeled corpora"** (`:302`). This counts
  corpora, while §5 clusters by family. Under family units, H1 may be *unpassable by construction*.
  That makes it a hard stop decided by pool size, not by the data.

**Underrated risk.** HyperPrior can pass H4 and still be useless. It could predict `λ_k` well
(MS MARCO-like corpora want λ ≈ 1, small corpora want it to fall with k) while the retrieval gain
over G+V stays ≈ 0. G+V's global α already sits near the mean, and the gap lives where neither
term is good. Opus-M's H4 condition (1) requires ≥ 0.4 × the F9 oracle gain *and* ≥ 0.005
absolute. The absolute floor is what protects against this, so it must be kept.

---

## 2. My round-1 proposal: what I concede, what I keep

### 2.1 Concessions

| # | Round-1 position (`round1-opus-e.md`) | Concession | Why |
|---|---|---|---|
| C1 | `:86-87`, `:109-110`, `:266-268`: the prior/correction balance "spans 3 orders of magnitude" | **Retracted.** | Raw `a` is not scale-identified, as my own `:498-500` noted. Scale-free `prior_share` at k = 307 is 0.95 / 0.43 / 0.36 **[V]**. The environment-level difference is real but about 2.7×, and it lies mostly in the *budget shape*. |
| C2 | E1-a "Certainty High within env" (`:78`) | **Relabelled.** The comparator is a query-weighted (93.6% MS MARCO), non-LOCO union. MS MARCO's −0.0019 is *not evidence about transfer*. NFCorpus's +0.0171 is measured against a global below `variance`, so it overstates headroom over G+V. SciFact's gap probably survives LOCO (§1.3). E1-b, E1-c and E1-e compare in-domain selectors and are unaffected. P1's "recovers 271% / 88% / 11% of L2S's gain over global importance" uses `global-importance`, the per-environment mean oracle target (`reports/overview/README.md`), not the union selector, so it is unaffected. | Code check, `train_global.py:54-60` **[V]**. |
| C3 | C3 "global-geometric" trained with `T ∝ sqrt(n_train)` (`:359`) | **Replaced by G+V, corpus-balanced** (Opus-M C5), plus Astra's matched-capacity global. | `sqrt` sampling puts about 74% of episodes on MS MARCO **[C]**. My `z = 0` control also had fewer parameters than the treatment. |
| C4 | Gate G0 on "every environment of the pool" (`:440`) | **Wrong.** The payoff census reads labels and drives a go/no-go decision, so it must use meta-train environments only (§3.2). | Opus-M leakage rule 5 (`:260-262`). |
| C5 | G0's "≥ 30% of environments with gap ≥ 0.005" rule | **Replaced by Opus-M's H1**: kill on mean < 0.005 or upper bound < 0.010. | A count-of-environments rule at K ≈ 5 is noisy, and it stops on absence of evidence rather than evidence of absence. |
| C6 | 8-number latent `z_T` with MLP encoder and heads, about 5k parameters, as the main design (`:298-311`) | **Demoted to Level 3.** The first predictor is a ridge on ≤ 3 statistics (Opus-M H4). End-to-end episodic training needs the escalation condition (Opus-M `:368-371`). | With K_train ≈ 5–7 families, an 8-dimensional latent cannot be identified. Leblanc's 10-task regime tied the strawman (p. 8) **[V]**. |
| C7 | C10, in-domain S4 on the meta-test target as a reference | **Not in the confirmatory plan** (§3.1). | AGENTS.md wording. |
| C8 | MAIR weighting `∝ min(1, n_train/200)` (`:345-346`) | **Withdrawn as ad hoc.** MAIR stays a source of P2 few-label evaluation and of descriptor diversity for HyperPrior's φ standardization, *if* family-blocked. It does not supply oracle targets: with about 80 queries per task, H3 reliability will fail. | ICC logic of Opus-M H3. |
| C9 | E2 "direct probe beats B2 … not SciFact" (`:88`) | **Minor correction.** On SciFact the direct probe beats B2 for sets 2 and 3 (+0.0160, +0.0093) and not for the others. | `reports/E2/tables/ladder.csv` **[V]**. |
| C10 | Search-Adaptor 50/50 re-split of test-only BEIR sets for training (`:342-343`) | **Requires an ADR**, as Opus-M says. I still advocate it, meta-train only (D3). | AGENTS.md "Never use test qrels" is literal. |

### 2.2 What I keep, and why

- **`variance` and G+V as baselines that a hypernetwork must beat.** This is now consensus.
- **Rival R1 (query distribution or instruction, not corpus)** as a pre-registered alternative,
  together with the **MS MARCO topic-shard test** (M2, my `:347-350`, `:465-467`). It holds the
  query distribution fixed and varies the corpus. No other proposal does that. Opus-M's
  generic-instruction control and P0 vs P1 attack R1 from the query side, and the shard test
  attacks it from the corpus side. It is mechanistic only and never meta-test. New evidence for R1
  this round: L2S pseudo-queries recover about 73% of the gain on MS MARCO but about 3% on SciFact
  (Table 5, p. 8) **[V]**.
- **The nearest-environment (1-NN) control.** Opus-M has it as C6.
- **Competitor-aware supervision** (P3-mask-ranking, F12 teacher arms). This is uncontested.
- **No dimension-ID embedding** in the generator. This is shared with Opus-M. Astra's design has
  none either; the seed's does.

---

## 3. Answers to my own open questions (Opus-M's lens, my answers)

### 3.1 Is a reference selector trained on a meta-test environment's training labels admissible?

**Under the current AGENTS.md, no.** The text reads: "Never use test qrels, **or any label from a
meta-test environment**, in **training**, adaptation, negative mining, descriptor construction, or
normalization." Training a reference S4 on a meta-test corpus's *training* qrels is training with
a label from a meta-test environment. Firewalling its outputs does not change the wording. Opus-M's
rule 8 (`:271-273`) and the seed (`:49`) both assume it is allowed. It is not, unless an ADR
amends the rule.

**Should an ADR allow it?** Only narrowly, and never in the primary chain. There are four reasons
[I]:
1. **Selection bias.** Only corpora with native training splits can have the reference. Those are
   the large QA-like ones: FiQA, HotpotQA, FEVER, NQ and Quora-dev. Gap closure computed on that
   subset says little about test-only corpora.
2. **Burn-in.** The reference is a strong, privileged view of meta-test, and seeing it shapes the
   next iteration. That is acceptable only because meta-test is single-use anyway (Opus-M rule 5).
3. **The denominator is unstable.** The in-domain recipe is noisy (P3-objective: lr alone moves
   SciFact by 0.0136), and on NFCorpus held-out in-domain is not distinguishable from full (F12,
   S1 vs `l2s-official` +0.004 with CI including 0) **[V from round 1]**.
4. **It is not needed.** Astra's estimand (`:109`) gives absolute gain against G+V. Gap closure
   can be reported on **meta-train environments via LOCO**: train the hypernet without corpus c,
   and compare with c's own in-domain S4, which is legal because c is meta-train.

**My position.** The pre-registered plan uses LOCO gap closure on meta-train only. If the team
wants meta-test gap closure, it needs an ADR, written before the lock, that:
- defines a "reference arm";
- allows only post-lock, scoring-only training on meta-test *training* splits;
- excludes the reference from every gate and decision;
- labels the result "post-lock reference, labeled subset, descriptive".

### 3.2 Should the payoff-census gate use only meta-train environments?

**Yes.** There are three reasons:
1. **Opus-M rule 5.** No decision reads meta-test. A go/no-go is a decision.
2. **Conditioning.** If the census reads meta-test labels and passes, the meta-test estimate is
   conditional on a favourable look. That is optional stopping at the corpus level, and it biases
   μ_Δ upward.
3. **Splits must be frozen first.** Corpora must be assigned to meta-train or meta-test *before*
   any census number exists. Otherwise the census tempts a reassignment of the large-gap corpora.

**Corollary.** With K_train ≈ 5–7 families the census is a *screen*, with an MDE of d ≈ 1.0–1.4
**[C]**. Its rules must be asymmetric:
- **Stop** only on *evidence of absence*: the upper bound of headroom over G+V is below 0.010
  (Opus-M H1 kill).
- **Never stop** on a non-significant pass test.
- Otherwise proceed, with the realized headroom estimate written into the power plan.

A census on meta-val is acceptable only if meta-val is then burned for that purpose and is not
reused for model choice.

---

## 4. The most important unresolved disagreements

### D1 — What should the first generated output be, and who decides?

| Position | Holder |
|---|---|
| Rank-4 query-interaction correction plus bias (4,186 parameters); `a_k`-only and bias-only as ablations | Astra |
| Nine prior weights from ≤ 3 statistics (12 parameters); bias at Level 2 only if H2 picks Fb and G3 passes; rank-4 "not proposed" | Opus-M |
| Bottlenecked `a_T` + Δb + rank-r (round 1) | me (now conceded, C6) |

**My position.** Let the *oracle ladder* decide, and expect it to prefer Fb over F9.
- On SciFact, where most known headroom is, the prior weight only moves the score between two
  components that are both about 0.13 below in-domain (global 0.7105, `variance` 0.6964, in-domain
  weighted 0.8340) **[V]**.
- Mixtures can beat both components: NFCorpus `geometric` 0.3725 is above L2S 0.3622 and
  `variance` 0.3587 **[V]**. But that was with an *in-domain* correction.
- Prediction [I], to be registered: R_F9 < 0.25 and R_Fb ≥ 0.5 on SciFact-like corpora.

So H2 must include F1, F9, Fb and Fr4 from the first run. HyperPrior stays the first predictor
built, because it is cheap and code-free. Rank-4 needs `R_Fr4 − R_Fb ≥ 0.25` (Opus-M `:566`).
No project result supports a query × corpus interaction (§1.2).

**Evidence that settles it.** The H2 ladder on the dev three is runnable *now*, with existing code
and a LOCO `W_0`. It is exploratory, but it gives the first measurement of F9 against Fb.

### D2 — What does a failed G3/E2 replication kill?

Astra says the whole generator program (`:99`). Opus-M says Level 2 only (`:380-384`).

**My position.** Level 2 only, and E2 should not be the *only* gate for Level 2:
- E2 scores Spearman agreement with S1 *selection frequency*, a per-dimension mask-membership
  statistic.
- E1-c shows that masks with near-chance cross-environment overlap (0.33 against chance 0.30)
  lose only 0.012–0.030 on MS MARCO and NFCorpus **[V]**. Mask agreement is a weak proxy for
  retrieval.
- Rule: keep the E2 replication as ADR-0013 prescribes, but let a failure kill Level 2 only if a
  **retrieval-scored LOCO test** of the tied bias map also fails. That test predicts Δb from
  descriptor rows, plugs it into G+V and compares Δ against G+V with the shuffle null.
- Environment-level outputs never depend on E2.

### D3 — May test-only corpora supply meta-train labels, via an ADR?

Opus-M says no (`:255-259`). Seed and Astra are silent. I say yes, meta-train only.

The evidence is the pool arithmetic of §1.3. Without them, K_train ≈ 5 families and every gate is
a screen for d ≥ 1.2. With them, ArguAna, SciDocs, TREC-COVID, Touché, Climate-FEVER (FEVER
family), CQADupStack and LoTTE-test become *candidates* for meta-train.

The trade-off is zero-sum: every family moved to meta-train leaves meta-test. My proposed rule:
- Fix the family list.
- Randomize families to the two sides, blocked by label regime.
- A test-only corpus that lands in meta-train uses its only qrels as training labels.
- It is permanently ineligible for any evaluation claim, recorded in the split manifest.

This respects the purpose of the rule (no evaluated quantity ever saw its own labels) and needs an
ADR. If the ADR is refused, the plan must say in advance that H1–H4 can detect only very large
effects.

### D4 — Can a confirmatory corpus-level claim be made at all?

Astra wants 32 families, Opus-M 20 corpora in at least 10 families, and I wanted 12 environments.

**Data.**
- 14 BEIR corpora (Search-Adaptor Table 7, p. 12) fall into roughly 8–10 families. LoTTE,
  CQADupStack and part of BRIGHT add mainly StackExchange, which is one family.
- At K = 7–8 family units, simulated power for the f = 0.5 scenario is 0.53–0.63, and 0.35–0.43
  for f = 0.25 **[C]**.
- Those scenarios themselves rest on optimistic planning gaps (§1.3).

**My position.** Pre-register an *estimation study with a decision rule*, not an 80%-powered test:
- The primary analysis is the exact sign-flip over **families** (with K ≥ 6, the minimum p is
  ≤ 0.016).
- The headline is the REML **prediction interval for a new family** (Opus-M `:439-441`).
- Claim levels are fixed by realized K, following Opus-M's downgrade rule (`:160-164`):
  - K_fam ≥ 15: population claim allowed;
  - 8–14: "replicated in m of K families" plus an interval;
  - < 8: pilot.
- Astra's retirement rule (`:113`, "upper 95% bound below +0.005 → retire") applies at every K,
  because it is a futility rule and so stays valid when power is low.

### D5 — Primary endpoint and primary comparator

There are two disagreements here:
- **Endpoint.** Astra uses k = 307 (`:109`); Opus-M and I use the decile mean.
- **Comparator.** Astra uses "the strongest global comparator locked by validation", which
  includes a matched-capacity global. Opus-M uses G+V.

**My position.**
- **Endpoint: the decile mean**, as in ADR-0018/F12, with k = 307 secondary. Per-budget outputs
  are the leading candidate, and a single-budget endpoint makes eight of nine outputs unscored.
- **Comparators: two necessary contrasts.**
  1. Against G+V, corpus-balanced and fit on meta-train. This is the analytic corpus channel.
  2. Against the **matched-capacity global** for the arm under test (Astra's control). For
     HyperPrior that is G+V with its α refit. For a bias or rank-4 arm it is the same generator fed
     the fixed meta-train template, with the same second-stage steps.

  Success requires both, plus the donor or shuffle contrast (Astra and Opus-M agree on this). It
  does not matter which global wins validation: whichever is stronger is the comparator.

---

## 5. The merged design I would now support

**Name.** *Ladder-gated HyperDIME.* It combines:
- Opus-M's gates and statistics;
- Astra's departure features, matched-global and donor controls, the exact contract, and cost;
- the seed's ladder;
- my R1 tests and corrected evidence base.

### 5.1 Model family (one score, nested rungs; all zero-initialized to G+V)

```
z_{T,k}(q) = λ_k(T)·std_j[ log(q_j² v_{T,j} + ε) ]            # analytic corpus channel (P0, exact v_T)
           + (1 − λ_k(T))·std_j[ (W_0 q + b_0)_j + Δb_{T,j} + (A_T B q)_j ]
m = TopK(z, k);   score(q, d) = Σ_j m_j q_j d_j                # index untouched, no renormalization
```

`std_j` standardizes across the 1024 coordinates, which makes λ the scale-free mixing weight
behind `prior_share`. Both endpoints are reachable: `variance` alone at λ = 1 and the correction
alone at λ = 0.

**The rungs.** Each is built only if the previous rung's gate passes and H2 licenses it.

| Rung | Generated per corpus | Generator | Learned env-level params | Licensed by |
|---|---|---|---|---|
| R0 = G+V | nothing; `λ_k = σ(α_k)` | — | 9 (α) | baseline |
| R1 HyperPrior | `λ_k(T) = σ(α_k + βᵀφ̃ + γᵀφ̃·k̃)` | ridge, LOCO λ | 15 (p = 3) | H4 on F9 |
| R2 tied bias | `Δb_{T,j} = hᵀ x_{T,j}` (linear first), then a 16-unit MLP | shared over j; Astra's departure rows `x = (F − M_{−T})/s` | h = 18 (P0) / 43 (P1) | H2 picks Fb; retrieval-LOCO bias test (D2) |
| R3 rank-4 | `A_T = X_T H`, shared B (Astra) | as Astra | +4,096 + 4h | `R_Fr4 − R_Fb ≥ 0.25` and K_train ≥ 15 families |

- φ is pre-specified now, following Opus-M `:518-523`:
  - P0: `log N`, log participation ratio, anisotropy;
  - P1: replace anisotropy with the mean top-1 query–corpus cosine.
- `M_{−T}` is the leave-it-out mean for training environments (§1.2).
- `W_0, b_0` are corpus-balanced, trained once (Stage A, S4 recipe, the corpus drawn uniformly
  within a uniformly drawn family), then frozen.

### 5.2 Stages and gates

All gate thresholds go in an ADR with a fingerprint before the first number. Gates use meta-train
only.

| Stage | What | Gate / kill | Data |
|---|---|---|---|
| **S0 (now, exploratory)** | On the dev three, with existing code: (a) LOCO corpus-balanced G+V vs the legacy union, which tests I1; (b) H1b analogue, `v_T` vs `v_pool`; (c) oracle ladder F1/F9/Fb/Fr4 with LOCO `W_0`; (d) MS MARCO topic-shard in-domain − G+V gaps (R1 vs R4) | None binding: dev results are exploratory (D9). Pre-register the predictions *R_F9 < 0.25, R_Fb ≥ 0.5 on SciFact; shard gaps ≈ 0* | SciFact, NFCorpus, MS MARCO; F2 artifacts |
| **S1** | Data audit; family list; ADRs (test-only meta-train labels D3; instruction freeze; reference-arm rule §3.1); family-blocked randomized split; resource gate (ADR-0018 pattern) | Split manifest fingerprinted before any new-corpus number | none read |
| **S2** | On meta-train: H0 recipe freeze; **H1** headroom over G+V; **H1b**; **H2** ladder with the n = 500 label-quantity control; **H3** ICC; E2 replication (ADR-0013) | H1 kill: mean < 0.005 or UB < 0.010 → ship G+V. H2: output family = smallest with R ≥ 0.5; R_Fr4 < 0.25 → P2. H3: ICC < 0.5 → stop | meta-train |
| **S3** | **H4** LOCO for the chosen family, P0 and P1 separately; nulls: full-enumeration shuffle, Astra donor, 1-NN, `log N` only | Pass all of: mean Δ ≥ 0.005, ≥ 0.4 × oracle, shuffle p ≤ 0.05, ≥ 1-NN. Otherwise stop, or try C10 | meta-train |
| **S4** | End-to-end (episodic) fine-tuning only under Opus-M's escalation rule; matched-global control trained with identical steps | Meta-val, once: Δ vs both comparators ≥ 0.003, donor recovers < ½ | meta-val (burned) |
| **S5** | Locked meta-test, once, scoring-only process | §5.3 | meta-test |

### 5.3 Confirmation (S5)

- **Unit:** the family. Corpus-level results are a sensitivity analysis.
- **Primary estimand:** the family-mean decile-mean nDCG@10 difference, with exact float64
  scoring and Astra's tie and parity contract (`:102`).
- **Primary chain** (fixed sequence, one-sided α = 0.05):
  1. non-inferiority vs G+V (δ = 0.005);
  2. superiority vs G+V;
  3. superiority vs matched-capacity global;
  4. correct-minus-donor > 0.

  P1 before P0 if P1 is eligible (Opus-M).
- **Reported regardless:** the REML prediction interval for a new family; the harm guard
  (Δ < −0.01 per corpus, Holm); leave-one-family-out; results by Qwen3-SFT exposure stratum.
- **Futility:** UB < +0.005 retires the rung (Astra). A donor within ±0.002 (90% CI) rejects the
  corpus-conditioning mechanism. If R1 or R2 is equivalent to R0 within ±0.002, keep the
  simpler one.
- **Claim level by realized K_fam:** see D4.
- **Rivals, secondary, with costs reported separately:**
  - PRF-DIME and Eclipse (per-query online);
  - C10 pseudo-query tuning (per-corpus offline; L2S Table 5 predicts it wins on web-like corpora
    and fails on specialized ones);
  - P2 50-query fine-tune, never called zero-shot;
  - MRL prefix and full 1024 (mandatory).
- **Cost:** Astra's G5 and break-even definition. HyperPrior adds zero online cost over S4. R3 adds
  8,192 MACs per query.

### 5.4 What would change my mind

- If S0(c) shows **R_F9 ≥ 0.5 on SciFact with a LOCO `W_0`**, my §0.4 prediction is wrong, and
  HyperPrior alone is the right program.
- If S0(d) shows MS MARCO shard gaps comparable to the cross-dataset gaps, R1 weakens and P0
  becomes more credible.
- If the D3 ADR is refused and K_train ≤ 5 families, I would recommend stopping at S2. The output
  would then be a G+V ship decision plus a published headroom census. Building a hypernetwork
  whose gates cannot detect less than d ≈ 1.2 is not worth it.

---

## 6. Verification log (this round)

**Read and confirmed [V]:**
- `specs/004-l2s-baseline/research.md:51`;
- `src/hyperdime/cli/train_global.py:30-90`;
- `reports/P3/README.md:46-48, 60-64` (training counts);
- `reports/E2/tables/{criteria,ladder,context,check}.csv`;
- `reports/S4/tables/{prior_share,weights}.csv`;
- `reports/S1/README.md:15`;
- `reports/P3-shared-mask/README.md:16-29, 80-110`;
- `reports/P3-objective/README.md:1-35`;
- `reports/overview/README.md:80-110`;
- `reports/E1/tables/noise_floor.csv`;
- `reports/P3-bench/RUNS.md:133-153` (MS MARCO F12 still running; not in `reports/`);
- `src/hyperdime/space/descriptor.py:37-50`;
- `src/hyperdime/contracts/schemas.py:171-187` (`kind="generated"` exists);
- Search-Adaptor p. 12, Table 7;
- Qwen3 Embedding p. 13, Table 6;
- Leblanc et al. p. 8, §4.3–4.4;
- Unveiling DIME p. 8, §5.3;
- Learning to Select p. 8, §6.5 and Table 5.

**Recomputed [C]:**
- 93.6% MS MARCO share;
- the `sqrt(n)` share of 73.8%;
- Astra's parameter counts and MACs;
- Opus-M's closed-form K, the MDE table, the simulation (4,000 replicates), non-inferiority K;
- the MDE at K = 4–7 (1.65 / 1.36 / 1.19 / 1.07);
- the Clopper–Pearson interval for 1/3 (0.008–0.906);
- the centering variance ratio;
- L2S pseudo-query recovery shares.

**Not verified:**
- BioASQ, LoTTE, CQADupStack and BRIGHT split facts, and TREC-COVID/Touché query counts;
- ADR-0001's outcome;
- S3 weights (ADR-0011);
- that `targets_at` emits exactly one row per training query.
