# Round 1 — Opus-M (methods lead / red team): an experimental program a corpus→selector hypernetwork must pass

Independent proposal. I have not read `round0-seed.md` or any other round file.

**Bottom line.**
1. The experimental unit for "generalizes to unseen corpora" is the **corpus**, or the corpus
   family when corpora share documents or sources. Queries are nested subsamples. Three corpora
   cannot support a corpus-level claim, and neither can six. By my arithmetic (§3), a corpus-level
   superiority claim needs **K_test ≈ 15–20 effectively independent meta-test corpora**. A
   corpus-level non-inferiority claim at δ = 0.005 needs **≈ 25**.
2. Most of what currently looks like corpus information is already available without a
   hypernetwork:
   - the label-free prior `q_d²·Var_T(x_d)` is a closed-form corpus→selector map;
   - the environment-level parameter that actually varies is the prior weight (S4 mean `a_k`:
     MS MARCO 0.835, SciFact 0.056, NFCorpus 0.025).
   So the comparator a hypernetwork must beat is **"global corpus-balanced selector + the target
   corpus's variance prior with per-budget global weights"** (called **G+V** below), not the legacy
   query-weighted global L2S.
3. The design I would bet on is **HyperPrior**, the simplest one that could pass. It keeps one
   global linear correction `W_0` and generates only the nine per-budget prior weights
   `a_k(T) = softplus(α_k + βᵀφ(T))` from **p ≤ 3 pre-specified label-free corpus statistics**:
   12 environment-level parameters in total. The first version is fit by leave-one-corpus-out
   ridge regression, so it is auditable. Its first test is a gate, not a training run.
4. Five gates come before any hypernetwork code. Each has a pre-registered threshold and a kill
   criterion:
   - headroom over G+V on new labeled corpora;
   - whether the target corpus's statistic matters at all;
   - an oracle capacity ladder (`a` → `a_k` → `Δb` → rank-4 `ΔW`);
   - reliability of the corpus-level targets;
   - LOCO label-free predictability against shuffle and nearest-corpus nulls.

Throughout, **[data]** marks a number I read from a file in this repository, with its path.
**[unverified]** marks a fact from memory or one I did not check against a primary source.
**[mine]** marks a computation I ran in `/tmp` and did not commit.

---

## 1. What the evidence allows (data, then interpretation)

### 1.1 Data

| # | Fact | Source |
|---|---|---|
| D1 | In-domain L2S minus the global selector, k = 307, test: SciFact +0.1116 [CI +0.078, +0.146], NFCorpus +0.0171 [+0.003, +0.032], MS MARCO −0.0019 [−0.005, +0.001] | [data] `reports/E1/tables/payoff.csv` |
| D2 | The global selector is "one linear selector trained on the union of the three environments' training targets". The union is **not corpus-balanced**. MS MARCO has 45,000 F2 training queries, against roughly 728 for SciFact and roughly 2,331 for NFCorpus (these two are 809−81 and 2,590−259, derived from BEIR sizes and E2's validation counts) | [data] `specs/004-l2s-baseline/research.md:51`; registry row P3; [unverified] the BEIR train sizes |
| D3 | Mean 10–90% nDCG@10 on the test split. SciFact: weighted 0.834, L2S 0.817, global 0.711, variance 0.696, full 0.700. NFCorpus: geometric 0.373, full 0.362, global 0.346. MS MARCO: variance 0.376, full 0.380, global 0.363 | [data] `reports/overview/README.md` |
| D4 | S4 mean per-budget prior weight: SciFact 0.0555, NFCorpus 0.0250, MS MARCO 0.8352. Participation ratio 98.8, 90.9, 223.9 | [data] `reports/E2/README.md`, environment-level context table |
| D5 | E2 / G3. The shift probe beats the shared profile B1 by at most +0.034 Spearman, never beats the variance baseline B2, and passes 0 of 3 held-out environments. The direct probe's gains over B1 come largely from variance | [data] `reports/E2/README.md`; ADR-0013 |
| D6 | Cross-environment overlap@k is 0.33, against a within-environment seed floor of 0.74 | [data] `reports/E1/tables/noise_floor.csv` |
| D7 | P3-bench, query-disjoint. Teacher-mask arms fall below `l2s-official` (NFCorpus: positive −0.043, contrastive −0.007). S1 arms sit slightly above, with intervals that include zero on NFCorpus | [data] `reports/P3-bench/RUNS.md` lines 75–133 |
| D8 | P3-objective. The only supported effect is L2S learning-rate tuning (C11: SciFact +0.0136, Holm p 0.036). No target, loss, per-budget, capacity, or initialization effect is identified | [data] `reports/P3-objective/README.md` |
| D9 | S1–S4 / E2 / P3 were shaped on these three corpora, so readings on them are exploratory | ADR-0012, ADR-0013 `test_set_reuse`; BRIEF |
| D10 | Qwen3-Embedding's supervised fine-tuning data include MS MARCO, NQ, and HotpotQA ("…etc.") | Qwen3 Embedding, Table 6, p. 13 |
| D11 | L2S authors: "predictors trained on one dataset do not consistently generalize well across domains" | Learning to Select, Limitations, p. 9 (proceedings p. 18674) |
| D12 | L2S trained on LLM-generated pseudo-queries "recover a substantial portion" of the supervised gain on MS MARCO, but give "only marginal improvements" on SciFact and NFCorpus | Learning to Select §6.5, p. 8 (Table 5) |
| D13 | PRF-DIME's best pruning rate varies across 13 public BEIR tasks | Unveiling DIME §5.3, p. 8; Table 3, p. 10 |
| D14 | One dimension subset learned on DL'19 transfers to DL'20 and Robust04 with changes "in the third decimal place". These are TREC collections over related text | When Reducing Representations Improves §4.4, pp. 11–12 |
| D15 | The number of dimensions a query needs "grows logarithmically with corpus size" | AdaWidth, abstract p. 1; §1, p. 2 |
| D16 | Dataset→predictor hypernetworks in the PAC-Bayes / sample-compression line meta-train on 90–300 tasks. "When tasks don't vary much … a single predictor (the strawman) yields competitive performances" | Leblanc et al., pp. 7–8 (§4.2–4.4) |
| D17 | Search-Adaptor splits test-only BEIR sets 50/50 by sorted query id to obtain training labels. Its Table 7 lists 14 BEIR sets with train/test pair counts | Search-Adaptor, p. 5; Table 7, p. 12 |

### 1.2 Interpretation (each item is my reading, open to challenge)

- **I1. The headroom estimate from E1 is inflated and fragile.**
  - D1 is driven by one corpus: SciFact's +0.112 against roughly 0 on MS MARCO.
  - D2 means the "global" comparator was about 94% MS MARCO by query count, so it is close to an
    MS MARCO selector evaluated on SciFact.
  - A corpus-balanced global selector could close much of the SciFact gap with no corpus
    information at all. Nobody has measured this. It is the first thing to measure (Gate H1).
- **I2. The best-documented corpus→selector map is the analytic prior, not a learned one.**
  - D3, D4, and D5 together say this.
  - `Var_T(x_d)` is already a label-free, per-corpus, per-dimension statistic.
  - The second best-documented signal is one environment-level scalar per budget, the prior weight,
    which spans more than an order of magnitude across the three corpora.
  - Per-dimension descriptors did not predict departures from the shared profile (D5).
- **I3. A hypernetwork trained on K_train ≈ 6–8 labeled corpora sees K_train environment-level
  data points.**
  - Any 1024 × h descriptor identifies its corpus uniquely, so a flexible map can memorize
    per-corpus outputs (plan risk R2).
  - D16 shows that working dataset→model hypernetworks use two orders of magnitude more tasks.
  - Capacity must therefore be matched to K_train, not to D.
- **I4. Per-corpus selectors improve little over L2S once tuning is matched (D7, D8).**
  - The in-domain reference a hypernetwork imitates is a weak, noisy target. Its recipe must be
    frozen before new corpora are opened.

---

## 2. Claim, estimand, unit

- **Claim to test (H3 of the plan, sharpened).**
  - Given only an unseen corpus (P0), or the corpus plus unlabeled calibration queries (P1),
    generated selector parameters retrieve better than the same selector without the generated
    part.
  - "The same selector" means G+V: identical global weights `W_0, b_0` and α, and the same
    target-corpus prior `v_T`.
  - The claim covers the population of retrieval corpora the meta-test set samples.
- **Treatment:** corpus-conditioned generation of selector parameters. It is applied once per
  corpus.
- **Experimental unit:** the corpus. Where corpora share documents or a source, the unit is the
  family cluster (§5).
- **Observational unit:** the query, nested in its corpus. Query-level paired tests support only
  corpus-specific statements. Counting queries across corpora as independent replicates of a
  corpus-level treatment is **pseudoreplication** (experimental-design skill, "The mistakes that
  ruin studies", item 1).
- **Design:** a randomized complete block design within each corpus, where every arm is scored on
  the identical evaluation queries (the query is the block). Across corpora, a two-stage analysis.
- **Estimand:** μ_Δ = E_corpus[Δ_c], where:
  - `Δ_c = mean_q [ȳ_q(H) − ȳ_q(G+V)]`;
  - `ȳ_q` is query q's mean nDCG@10 over the nine deciles 102…921, the project's headline;
  - every corpus gets equal weight, unweighted by query count. Otherwise MS MARCO-sized corpora
    dominate and the estimand stops being about corpora.

---

## 3. Power: how many corpora

All numbers in this section are **[mine]**: `/tmp/hm-scripts/power.py`, run with SciPy from the
project venv. I followed the statistical-power skill: the effect size comes from a smallest effect
of interest and planning data, and is varied as a sensitivity range.

**Planning values.**
- The only corpus-level effect data available are D1's three gaps (+0.1116, +0.0171, −0.0019):
  mean 0.042, SD 0.061, standardized d ≈ 0.70.
- If the hypernetwork recovers a fraction f of each corpus's gap, both mean and SD scale with f,
  so d stays ≈ 0.70 when measurement noise is negligible. With realistic noise d is lower.
- This is a three-point estimate dominated by SciFact, so it is very uncertain (cf. I1).

**Closed form** (one-sample t on Δ_c, 80% power):

| d | 0.5 | 0.6 | 0.7 | 0.8 | 1.0 | 1.5 |
|---|---|---|---|---|---|---|
| K, one-sided α = 0.05 | 27 | 19 | 15 | 12 | 8 | 5 |
| K, one-sided α = 0.025 | 34 | — | 19 | — | 10 | 6 |

The minimum detectable d for a given K (one-sided α = 0.05, 80%):

| K | 3 | 6 | 8 | 10 | 12 | 15 | 18 | 24 | 30 |
|---|---|---|---|---|---|---|---|---|---|
| MDE d | 2.30 | 1.19 | 0.98 | 0.85 | 0.77 | 0.68 | 0.61 | 0.52 | 0.46 |

**Exact sign-flip floor.** With K corpora the smallest attainable p is 2^−K. K ≥ 5 is needed for
p ≤ 0.05, K ≥ 6 for 0.025, and K ≥ 7 for 0.0125. At K = 3 no corpus-level test can ever reject.

**Simulation.** The gap distribution is heavy-tailed, to mimic D1: 30% of corpora with gap
N(0.08, 0.03) and 70% with N(0.01, 0.01). Recovery is f. Between-corpus heterogeneity τ = 0.005
and per-corpus query-sampling SE = 0.008 are added. 2,000 replicates; values are power for the
t-test / Wilcoxon signed-rank:

| K | 6 | 8 | 10 | 12 | 15 | 20 |
|---|---|---|---|---|---|---|
| f = 0.50 (mean Δ ≈ 0.016) | 0.43 / 0.46 | 0.64 / 0.59 | 0.76 / 0.70 | 0.84 / 0.80 | 0.94 / 0.89 | 0.98 / 0.96 |
| f = 0.25 (mean Δ ≈ 0.008) | 0.31 / 0.29 | 0.41 / 0.35 | 0.52 / 0.45 | 0.60 / 0.55 | 0.71 / 0.65 | 0.83 / 0.78 |

**Non-inferiority at δ = 0.005.** Take the true μ_Δ = 0 and SD(Δ_c) ≈ √(τ² + SE²) ≈ 0.0095. Then
d = 0.005 / 0.0095 ≈ 0.53, which needs **K ≈ 25**. If instead μ_Δ = +0.005, d ≈ 1.05 and K ≈ 8.

**Requirement (pre-registered).**
- K_test ≥ 20 meta-test corpora, spanning ≥ 10 families, each with ≥ 150 evaluation queries after
  calibration queries are removed.
- If fewer are available, the confirmatory corpus-level claim is **downgraded in advance** to
  per-corpus replication counts (descriptive). No population claim is made.
- Gates on meta-train corpora have K_train ≈ 6–8. They can detect only large effects
  (MDE d ≈ 1.0 at K = 8), so they are **screens for the presence of a large signal**, not
  confirmations.

---

## 4. Corpora: which have training labels

Everything in this table is **[unverified]** unless marked otherwise. It is BEIR and LoTTE split
structure as I remember it. A data-audit ticket must confirm splits, sizes, licences, and overlaps
before the split manifest is frozen. Search-Adaptor's Table 7 (p. 12) gives train/test pair counts
but splits test-only sets itself (p. 5).

| Corpus | Docs | Non-test labels (usable for meta-training) | Test qrels | Family | In Qwen3 SFT (D10) |
|---|---|---|---|---|---|
| MS MARCO | 8.8M | train | dev/DL | web | yes |
| SciFact | 5K | train | test | scientific claims | ? |
| NFCorpus | 3.6K | train/dev | test | biomedical | ? |
| FiQA-2018 | 57K | train/dev | test | finance QA (StackExchange+Reddit) | ? |
| HotpotQA | 5.2M | train/dev | test | Wikipedia | yes |
| FEVER | 5.4M | train/dev | test | Wikipedia | ? |
| NQ | 2.7M | nq-train (different corpus build) | test | Wikipedia | yes |
| Quora | 523K | dev | test | duplicate questions | ? |
| DBPedia-entity | 4.6M | dev (≈67 q) | test | Wikipedia/entities | ? |
| BioASQ | ~15M | train (licence) | test | biomedical | ? |
| LoTTE dev (5 topics) | various | dev qrels | — | StackExchange | ? |
| TREC-COVID | 171K | — | test (50 q) | biomedical | ? |
| ArguAna | 8.7K | — | test | argument | ? |
| Touché-2020 | 382K | — | test (49 q) | argument | ? |
| SCIDOCS | 25K | — | test | scientific citation | ? |
| Climate-FEVER | 5.4M (= FEVER corpus) | — | test | Wikipedia | ? |
| CQADupStack (12 forums) | 457K total | — | test | StackExchange | ? |
| LoTTE test (5 topics × search/forum) | various | — | test | StackExchange | ? |
| BRIGHT (12 domains) | various | — | test | reasoning; 7 StackExchange-derived | ? |

**Consequences.**
- About 8–10 corpora outside the dev three carry non-test labels, and they come from about 5
  families. That is the meta-train pool, and it will stay small.
- The meta-test pool can be large (about 30–40 corpora) because meta-test needs only test qrels,
  but those corpora fall into only about 10–14 families.
- The Wikipedia cluster (HotpotQA, FEVER, Climate-FEVER, NQ, DBPedia) and the StackExchange
  cluster (FiQA, CQADupStack, LoTTE, part of BRIGHT) are not independent units.
- Embedding cost matters. Five corpora of 2.7–5.4M documents roughly double the MS MARCO
  embedding work. A resource gate in the style of ADR-0018 should price this, and small and medium
  corpora come first.

---

## 5. Splits and leakage

### 5.1 Splits, by corpus (frozen before any new-corpus number exists)

| Role | Corpora | Rule |
|---|---|---|
| **Dev** | SciFact, NFCorpus, MS MARCO | May be meta-train. **Never** meta-val or meta-test; never pooled into a confirmatory estimate (D9) |
| **Meta-train** | Dev + about half of the new labeled corpora | Gradients, ridge fits, descriptor standardization, and gate computations (LOCO inside this set) |
| **Meta-val** | About 4 test-only corpora from 4 distinct families | Used **once**: go/no-go plus a choice among ≤ 3 pre-specified variants. Burned afterwards |
| **Meta-test** | The other half of the new labeled corpora + the remaining test-only corpora | Opened once, after an evaluation lock in the style of ADR-0018 |

- **Randomized assignment, blocked by family.** Use stratified randomization with a recorded
  seed. The strata are (family × label availability). Labeled new corpora split 50/50 between
  meta-train and meta-test. The labeled meta-test corpora are what make the gap-closure secondary
  endpoint computable, because an in-domain selector can be trained on them **as a reference arm
  only, after the lock**.
- **Cluster rule.** Corpora whose documents overlap go to the same side of every split.
  - Overlap is measured on normalized-text SHA-256 plus MinHash near-duplicates.
  - The threshold is ≥ 1% of the smaller corpus.
  - Query-text overlap counts too: LoTTE against CQADupStack, and BRIGHT StackExchange against
    CQADupStack.
  - If FEVER is meta-train, Climate-FEVER is ineligible as an "unseen corpus". Its P0 descriptor is
    FEVER's.
- **Shards are not corpora.** MS MARCO topic shards, CQADupStack forums used as extra training
  episodes, and TREC-DL query sets are labeled as such (protocols.md). For corpus-level inference,
  CQADupStack's 12 forums count as one family, analyzed with cluster means.
- **Query splits inside a meta-test corpus (P1).**
  - Calibration queries are `n_cal = min(256, ⌊0.3·n_q⌋)` hashed ids, with identical normalized
    text grouped, as in `p3-query-split-v1`.
  - They are disjoint from evaluation ids, and their qrels are never opened.
  - Corpora with `n_q < 150` are P0-only in the primary analysis. TREC-COVID and Touché are below
    this; they appear in a secondary table.

### 5.2 Leakage rules (extending protocols.md §Leakage)

1. **Descriptors** (`v_T`, φ(T), descriptor rows) come from documents (P0) or documents plus
   calibration queries (P1). Loading goes through `space.inputs.load_unlabeled`, which opens no
   judgments file. This must be enforced in code and tested.
2. **Standardization** statistics for φ and the descriptor rows are fit on meta-train corpora only.
   They are written as a hashed artifact **before** any meta-val or meta-test descriptor is
   computed.
3. **Pseudo-labels** (PRF top-m, inverse-cloze or title pseudo-queries, LLM pseudo-queries) for a
   meta-val or meta-test corpus are built from that corpus and its calibration queries only.
   - Their hyperparameters (m, negative depth, generator) are fixed on meta-train corpora.
   - Pseudo-label quality may be measured against qrels **only on meta-train corpora**.
4. **Test-only corpora are never trained on.** Their only labels are test qrels, and AGENTS.md
   forbids test qrels in training.
   - The Search-Adaptor trick (D17) of re-splitting test qrels would require an ADR amending
     AGENTS.md. Any corpus so used would be permanently ineligible for evaluation claims.
   - I recommend not doing it.
5. **No decision reads meta-test.** Architecture, feature list, margins, k grid, recipes, and
   instructions are all fixed before it is opened. The meta-test evaluation runs once, against a
   fingerprinted lock that lists checkpoint hashes, the split manifest, and evaluation ids.
6. **Instructions are part of the environment.** The Qwen instruction text per corpus is frozen
   from a published source (MTEB/Qwen task instructions) before any result. It is never tuned.
   - P1 descriptors partly encode the instruction through the query embeddings. The report states
     this.
   - A secondary control uses one generic instruction for all corpora.
7. **Encoder exposure is tagged.** Corpora in Qwen3's supervised fine-tuning data (D10) are a
   stratification variable, reported in every table. This is not leakage, since the encoder is
   frozen and shared by every arm, but it limits scope.
8. **The in-domain reference on labeled meta-test corpora** is trained from their *training*
   labels after the lock. It is a reference arm only. Its outputs never feed the hypernetwork,
   standardization, or any choice.

---

## 6. Gates before any hypernetwork code

Common conventions:
- Every gate uses meta-train corpora only. LOCO means every quantity for corpus c is fit without c.
- The evaluator is the exact float64 full-corpus one with the lower-row tie rule (ADR-0016,
  ADR-0018), on query-disjoint evaluation splits.
- Each gate's thresholds sit in an ADR with a SHA-256 fingerprint before its first number exists.
- "Corpus-level p" is the exact one-sided sign-flip over corpora.

### Gate H0: recipe freeze and replication on new labeled corpora (#60)

- **Run:** the S4 recipe, unchanged (ADR-0012), and the corpus-balanced G+V comparator, on each new
  meta-train corpus.
- **Record:** in-domain minus full, minus variance, and minus G+V.
- **Threshold:** none for passing. It fixes the in-domain recipe that the gates and the reference
  arm use.
- **Kill:** if S4 fails non-inferiority to S3 or `variance` (δ = 0.005) on a majority of new
  corpora, the in-domain reference is replaced by S3 per ADR-0011, recorded, and frozen before H1.

### Gate H1: headroom over the right comparator

- **Measure:** `H_c = ȳ(in-domain S4) − ȳ(G+V)` on each new meta-train corpus, where G+V is
  corpus-balanced and fit LOCO.
- **Pass:** all of the following.
  - mean_c H_c ≥ 0.010;
  - corpus-level p ≤ 0.05, which needs K ≥ 5 new labeled corpora;
  - pointwise H_c ≥ 0.005 in at least half the corpora.
- **Kill:** mean H_c < 0.005, or the one-sided 95% upper bound < 0.010. In that case nothing is
  left for a hypernetwork to recover. Ship G+V as the corpus-generalizing selector and record an
  ADR.
- **Side output:** on the dev corpora, report G+V against the legacy query-weighted global
  selector. This tests I1 directly, but it is exploratory.

### Gate H1b: does the target corpus's statistic matter at all?

- **Measure:** G+V with `v_T`, the target corpus's variance, against G+V with `v_pool`, the variance
  pooled over the other meta-train corpora. Everything else is identical.
- **Pass:** mean_c[ȳ(v_T) − ȳ(v_pool)] > 0 with corpus-level p ≤ 0.05.
- **Kill (soft):** if the target statistic is no better than the pooled one, then even the analytic
  corpus channel carries nothing measurable. The prior probability that a learned channel will is
  then low. Continue only with an explicit justification in an ADR.
- **Why this gate:** it is the cheapest test that separates "uses corpus information" from "has a
  variance-shaped prior".

### Gate H2: oracle capacity ladder (privileged; sets the hypernetwork's output family)

- **Setup:** freeze `W_0, b_0` from G+V fit LOCO. Fit only the family's parameters on corpus c's
  training labels, choose by c's validation split, and score on c's evaluation split.

| Family | Free parameters per corpus | Name |
|---|---|---|
| one prior weight `a` | 1 | F1 |
| per-budget weights `a_k` | 9 | F9 |
| F9 + bias shift `Δb ∈ R^1024` | 1,033 | Fb |
| Fb + rank-4 `ΔW = A Bᵀ` | 9,225 | Fr4 |
| full in-domain S4 | ≈1.05M | ID |

- **Statistic:** `R_j = Σ_c [ȳ_j − ȳ(G+V)] / Σ_c H_c`, a ratio of sums, which is stable when some
  H_c ≈ 0.
- **Rule:** the hypernetwork's output family is the smallest j with R_j ≥ 0.5.
- **Kill:** R_Fr4 < 0.25. A residual hypernetwork over `W_0` cannot reach the gap even with oracle
  labels. The gap needs a per-corpus `W` (labels), and the scenario becomes P2.
- **Control for label quantity (D2, T5):** rerun F9 and ID with every corpus's training set
  subsampled to a common n = 500 queries. If headroom depends on n, "selector shift" is partly
  label-quantity shift. Report it.

### Gate H3: reliability of the corpus-level targets

- **Measure:** estimate the chosen family's parameters (for F9, `log a_k*`) on two disjoint halves
  of each corpus's training queries, plus three seeds.
- **Statistic:** ICC = between-corpus variance / total variance.
- **Pass:** ICC ≥ 0.7. **Kill:** ICC < 0.5, because a predictor would be fitting noise.
- **Note:** this mirrors E2's seed ceilings (D5), but at the level of the generated parameters.

### Gate H4: label-free predictability, LOCO (this *is* the simplest hypernetwork)

- **Predictor:**
  - inputs: the family's parameters for held-out corpus c, predicted from φ(T) (§10, p ≤ 3
    features fixed now) by ridge regression, with λ chosen by inner LOCO;
  - output: plugged into `W_0, b_0`, then scored on c's evaluation split;
  - result: `Δ^LOCO_c = ȳ(pred) − ȳ(G+V)`.
- **Pass: all four.**
  1. mean Δ^LOCO ≥ 0.005 and ≥ 0.4 × the oracle gain of the chosen family (H2);
  2. corpus-level p ≤ 0.05;
  3. above the 95th percentile of a **shuffle null**: φ reassigned among meta-train corpora,
     every permutation when K! is small, otherwise 10,000, with the fit redone each time;
  4. not below a **1-NN control**: copy the parameters of the nearest meta-train corpus in
     standardized φ space.
- **Kill:** if (1)–(3) fail, label-free statistics do not carry the information at this
  granularity, so no hypernetwork. The fallback is per-corpus self-supervised tuning (C10).
- **Run separately for P0 and P1.** P1 may pass where P0 fails.
- **When to escalate:** writing episodic or end-to-end hypernetwork code (§10, Levels 2–3) is
  justified only if H4 passes **and** the ridge predictor is visibly limiting:
  - LOCO prediction error of `log a_k*` > 2× the H3 within-corpus error; and
  - an oracle family above F9 (Fb or Fr4) had R ≥ 0.5 more than F9.

### Gate H5: competitor parity (reframes the claim; does not kill)

- **Compare** H4's predictor with C10, the per-corpus self-supervised tuning of `a_k` from
  pseudo-queries generated on the target corpus (§7).
- **Reading:** if C10 ≥ H4 at acceptable offline cost, a hypernetwork adds only amortization. Its
  claim is then cost (E6 / G5), and it must be priced against C10's offline cost.

### Gate G3-rerun (gates Level 2 only)

Rerun E2 unchanged on the new meta-train corpora, as ADR-0013 itself prescribes. If G3 fails again,
per-dimension Δb generated from descriptor rows (Level 2) is killed. HyperPrior (Level 1) does not
depend on it.

---

## 7. Comparators and ablations

Every arm is scored on the same evaluation queries, at the same nine deciles plus 1024 as a parity
check, with the exact evaluator.

| ID | Arm | Labels on target | What it separates |
|---|---|---|---|
| C0 | Full 1024 | none | mandatory (protocols) |
| C1 | Prefix-k (MRL) | none | mandatory (R5) |
| C2 | Random-k | none | floor, context |
| C3 | `variance` on the target corpus | none | the analytic corpus channel alone |
| C4 | Legacy global L2S, query-weighted union | none | continuity with E1; **not** the primary comparator |
| C4b | Corpus-balanced global selector, no prior, S4 objective | none | supervision without any corpus information (R6) |
| **C5** | **G+V**: C4b + target `v_T` prior with nine global `α_k` | none | **primary comparator**. Matched capacity and inputs except the generated term `βᵀφ(T)` |
| C5p | G+V with pooled `v_pool` | none | target-corpus statistic against any variance prior (H1b) |
| C6 | 1-NN corpus: `a_k*` of the nearest meta-train corpus in φ space | none | **dataset identification / library lookup** |
| C6b | Ridge on `log N_T` only | none | the corpus-size confound (D15) |
| C7 | Shuffled-φ HyperPrior, distribution over permutations | none | **"uses corpus information" vs "extra parameters and features"** |
| C8 | HyperPrior with φ computed on a random 10% document sample | none | descriptor sampling stability (E5) |
| C9a / C9b | PRF-DIME / Eclipse at the same deciles | none (an extra retrieval pass online) | label-free **per-query test-time** adaptation; online cost reported separately |
| C10 | Per-corpus self-supervised `a_k` (and optionally S4) tuning from pseudo-queries: inverse-cloze / title spans; LLM queries optional, as in L2S §6.5 | none (offline training) | label-free **per-corpus offline** adaptation, the direct rival to amortization |
| C11 | In-domain S4 (labeled meta-test corpora only) | train labels, reference | upper reference, gap closure |
| C12 | Oracle `a_k*` tuned on target train labels | train labels, privileged | ceiling of the F9 family |
| H-P0 / H-P1 | HyperPrior | none | treatment |

**Ablations (secondary, descriptive, Holm within family):**
- remove each φ feature in turn;
- shared β against per-budget β;
- P0 against P1;
- `v_T` from 10%, 30%, and 100% of documents;
- n_cal ∈ {32, 128, 256};
- generic against task instruction;
- three seeds of `W_0`.

**Dataset-identification diagnostics (pre-registered, secondary):**
- the slope of Δ_c on standardized distance to the nearest meta-train corpus in φ space. A negative
  slope with no gain far away means interpolation, not generalization;
- results reported separately for meta-test corpora inside a meta-train family and for those in
  unseen families (leave-cluster-out).

---

## 8. Statistics

- **Primary endpoint.** μ_Δ = the unweighted mean over meta-test units of Δ_c (H-P1 against C5).
  - ȳ_q is the mean nDCG@10 over the nine deciles. Training seeds 0–2 are averaged per query
    before testing, as in the project convention.
  - Units are corpora. Where a family contributes several corpora, the primary analysis uses family
    means, and corpus-level analysis is a sensitivity.
- **Primary test.** An exact one-sided sign-flip over units: all 2^K vectors when 2^K ≤ 10^5,
  otherwise B = 10^5. The interval inverts the sign-flip test. A t-interval is a sensitivity.
- **Always reported, not tested:** a random-effects meta-analysis (REML) of per-corpus Δ_c with
  query-bootstrap SEs, giving μ, τ, and the **95% prediction interval for a new corpus**. That
  interval is the practically relevant quantity.
- **Margins.**
  - Non-inferiority δ_NI = 0.005 nDCG@10, consistent with S4 / ADR-0012.
  - Superiority SESOI = 0.005, so "established minimum gain" means a lower bound ≥ 0.005, as in
    ADR-0018.
- **Multiplicity of the primary chain.** Fixed-sequence gatekeeping, each step one-sided at
  α = 0.05, which controls the familywise error rate. A step is tested only if the previous one
  passed:
  1. H-P1 non-inferior to C5 (μ_Δ > −0.005);
  2. H-P1 superior to C5 (μ_Δ > 0);
  3. H-P0 non-inferior to C5;
  4. H-P0 superior to C5;
  5. H-P1 against C7: uses corpus information (superiority);
  6. H-P1 established minimum gain (LCB ≥ 0.005).
- **Secondary families (Holm within each, α = 0.05):**
  - H-P1 against {C4b, C6, C6b, C9a, C9b, C10};
  - per-corpus paired query-level tests of H-P1 against C5, B = 10^5 sign flips, giving a
    replication count;
  - gap closure `GC = Σ_c Δ_c / Σ_c (C11 − C5)` on labeled meta-test corpora, with a corpus
    bootstrap interval.
- **Harm guard (pre-registered, part of the success definition).** Per corpus, a one-sided test of
  Δ_c < −0.01, Holm over corpora. Claiming "safe to deploy" requires that no corpus shows
  significant harm. Each harmed corpus is listed.
- **Descriptive only:** per-decile curves, MRR@10, Recall@100, overlap@k against in-domain, and
  `a_k(T)` against `a_k*`.
- **Kill at meta-test.** If step 1 fails, the hypernetwork does not generalize. G+V is the
  deliverable, and the result is recorded as evidence (plan §13, "record the result as evidence").

---

## 9. Threats to validity

| # | Threat | Consequence | Remedy in this design |
|---|---|---|---|
| T1 | Forking paths: the prior, per-budget weights, k grid, lr grids, selection rules, and descriptor groups were all chosen after results on SciFact/NFCorpus/MS MARCO | Every effect measured on the dev three is optimistically biased | Dev corpora never in meta-val/test. The recipe is frozen (H0) before new corpora. Confirmation only on meta-test |
| T2 | A single influential corpus (SciFact's gap is 5–50× the others) | Mean effects driven by one unit; d ≈ 0.7 is a three-point guess | Equal weights, sign-flip and Wilcoxon, leave-one-unit-out sensitivity of every conclusion, prediction interval |
| T3 | The legacy global selector is query-weighted, about 94% MS MARCO | Headroom (D1) overstated | Corpus-balanced C4b/C5 as comparators; H1 measures the true headroom |
| T4 | Encoder exposure (MS MARCO, NQ, HotpotQA in Qwen3 SFT) | Headroom may vanish on encoder-seen corpora; effects not exchangeable across strata | Tag and stratify; report by stratum; no pooling claim across strata without a homogeneity check |
| T5 | Label-regime confound: about 0.7k binary vs about 2.3k graded vs 45k sparse training queries | "Selector shift" may be label-quantity or label-density shift | The H2 subsampling control at a common n |
| T6 | Corpus size (D15), participation ratio, and anisotropy co-vary (D4: MS MARCO is largest in N and PR) | A descriptor "works" only through log N | C6b, and log N kept as an explicit, separable feature |
| T7 | Evaluator heterogeneity (float32 F3 vs float64 P3; MS MARCO full corpus vs authors' subset; ideal-DCG conventions, `reports/P3-bench/RUNS.md` parity notes) | Cross-experiment numbers not comparable to the last digit | One exact evaluator for every arm and corpus; no recomputation of legacy results |
| T8 | Test-set reuse on the dev three across E1, S1–S4, E2 | Those test splits are no longer held out | Dev numbers labeled exploratory; `p3-query-split-v1` style splits for new corpora |
| T9 | The in-domain target is weak and noisy (D7, D8) | Gap closure is relative to a moving target; it can flatter or punish the hypernetwork | Recipe frozen at H0; the primary endpoint compares against C5, not against in-domain |
| T10 | Pseudoreplication | Query-level p-values mistaken for corpus-level evidence | Two-stage analysis; corpus or family as the unit; query tests labeled corpus-specific |
| T11 | Descriptor = dataset identity (plan R2, I3) | Memorization of meta-train corpora that looks like generalization in-family | p ≤ 3 features, LOCO, leave-family-out reporting, C6/C7 nulls, distance-slope diagnostic |
| T12 | Instruction confound (the environment includes I) | P1 gains may come from instruction-induced query geometry | Instructions frozen from a published source; generic-instruction control |
| T13 | Family non-independence (Wikipedia, StackExchange clusters) | Inflated effective K | Family-level units in the primary test; overlap audit |
| T14 | Small-query corpora (TREC-COVID 50, Touché 49) | Large per-corpus SE inflates SD(Δ_c) and kills power | ≥ 150 evaluation queries for primary units; small corpora secondary |
| T15 | The power assumptions (d, τ, SE) come from three corpora | The study may still be underpowered | Report MDE at the realized K; downgrade rules fixed in advance (§3) |

---

## 10. The hypernetwork I would bet on: HyperPrior (the simplest that could pass)

### 10.1 Level 1, HyperPrior-9: the primary candidate

**Score** for query q in corpus T at decile k:

```
z_{q,d}(T,k) = a_k(T) · log(q_d² · v_{T,d} + ε) + (W_0 q + b_0)_d
m_q          = top-k(z_q),     e'_q = e_q ⊙ m_q   (not re-normalized),   score = e'_qᵀ e_doc
a_k(T)       = softplus(α_k + βᵀ φ̃(T)),   φ̃ = (φ − μ_train) / σ_train
```

| Tensor | Shape | Source | Count |
|---|---|---|---|
| `W_0, b_0` | 1024×1024, 1024 | global, corpus-balanced, meta-train labels | 1,049,600 (shared, not generated) |
| `v_T` | 1024 | target corpus document variance (P0, streamed: `space.geometry`) | analytic, not learned |
| `α` | 9 | global | 9 |
| `β` | p ≤ 3 | global; the **only** corpus-conditioned learned parameters | ≤ 3 |
| `φ(T)` | p | label-free statistics (below) | — |

So there are 12 environment-level learned parameters. That matches K_train ≈ 6–8 corpora × 9
budgets of targets with ridge shrinkage. With β = 0 the selector **is** C5, which satisfies
architecture.md's invariant that zero modulation gives θ_0.

**Pre-specified φ.** These are fixed now, before any new-corpus data, to limit forking:
- **P0** (documents only):
  - φ1 = log N_T;
  - φ2 = log participation ratio of the corpus covariance (`space.geometry`);
  - φ3 = mean pairwise cosine of 10,000 seeded document pairs (anisotropy).
- **P1** (documents and calibration queries): φ1, φ2, and φ3' = mean top-1 cosine between
  calibration queries and the corpus, a measure of query–corpus match strength.
- No feature may be added after meta-train data are seen without going through an amended,
  fingerprinted ADR, and the meta-train gates are then re-run.

**Training (two-stage; auditable; no episodic code).**
1. **Stage A.** Fit `W_0, b_0, α` (with β = 0) with the S4 recipe (ADR-0012), using:
   - masked retrieval CE through `soft_top_k`, one decile drawn uniformly per batch;
   - **corpus-balanced episodes**: corpus drawn uniformly, then queries within it;
   - the corpus's own `v_T` in each episode.
   This artifact is C5.
2. **Stage B.** For each meta-train corpus, get `a_k*(T)`: a grid over `a ∈ {0} ∪ logspace(−3, 0,
   13)`, choosing per decile by validation nDCG@10 with `W_0` frozen. Exact evaluation makes this
   cheap. Then fit `log a_k* ≈ α_k + βᵀφ̃(T)` by ridge, with λ chosen by LOCO. This is Gate H4.
3. **Stage C (optional; only if H4 passes and the escalation condition holds).** Fine-tune β
   end-to-end through the Stage-A loss with `W_0` frozen:
   - corpus-balanced episodes;
   - weight decay on β;
   - early stopping by LOCO validation.
   Compare with Stage B on meta-val, once.

**Cost.**
- Online: identical to S4 — one 1024×1024 matvec, 1024 logs, top-k. No added latency.
- Offline per corpus: one streamed pass for `v_T` and the covariance (O(N·D²) for participation
  ratio, already implemented). For P1, an exact top-1 scan for n_cal queries.
- Each is reported separately (E6).

### 10.2 Level 2 (only if H2 picks Fb **and** the G3 rerun passes): a per-dimension bias generator

```
Δb_{T,d} = g_ω(r̃_{T,d}),   g_ω: R^h → R,  MLP h→16→1, last layer zero-initialized
z_{q,d} += Δb_{T,d}
```

- `r̃_{T,d}` is the standardized P0/P1 descriptor row (h = 18 or 43 columns, `space.descriptor`).
  The parameter count is h·16 + 16 + 17, about 330 for h = 18.
- It has **no dimension-ID embedding**. The shared per-dimension profile already lives in `b_0`, and
  an ID embedding would memorize E2's B1.
- Zero initialization keeps θ_T = θ_0 at step 0. Principled initialization matters because generated
  weights at the wrong scale destabilize the main network (Chang et al., abstract, p. 1).
- Training uses Stage-A episodes with `W_0` frozen, plus a penalty λ‖Δb‖².

### 10.3 Level 3: low-rank ΔW (not proposed)

This is the plan's M7 / M8 design. It is admissible only if H2 shows R_Fr4 − R_Fb ≥ 0.25, and only
with K_train ≥ 15. With fewer corpora a generator of 2·1024·r factors has no environment-level data
to learn from (I3, D16).

### 10.4 Why this design and not a larger one

- It puts the corpus dependence where the evidence puts it: the analytic prior plus a
  budget-dependent mixing weight (D3, D4, I2).
- It makes the treatment contrast a single, interpretable term `βᵀφ(T)`. That is what lets C5 and
  C7 separate "corpus information" from "extra parameters".
- It can fail cleanly at Gate H4, using code that already exists (`masked_retrieval_loss`,
  `soft_top_k`, `space.geometry`, the exact evaluator). No hypernetwork module has to be written to
  learn that the idea is dead.

---

## 11. What would kill the idea (summary)

| Stage | Kill condition | Then |
|---|---|---|
| H1 | Headroom over corpus-balanced G+V: mean < 0.005, or upper bound < 0.010 | Ship G+V; ADR |
| H1b | Target `v_T` no better than pooled `v` | Strong prior against any corpus channel; continuing needs an ADR |
| H2 | Oracle Fr4 recovers < 25% of headroom | The gap needs labels; move to P2 few-label adaptation |
| H3 | ICC of the target parameters < 0.5 | Targets are noise; stop |
| H4 | LOCO ridge fails magnitude, corpus-level p, or the shuffle null | No hypernetwork; C10 or G+V |
| Meta-val | H-P1 not non-inferior to C5 on meta-val | Do not open meta-test; ADR |
| Meta-test | Step 1 (non-inferiority) fails, or any corpus shows significant harm > 0.01 | No generalization or safety claim; record as evidence |

---

## 12. Open questions for round 2

1. Is G+V, with the target `v_T` and global per-budget α, accepted as the primary comparator? My
   claim is that any other choice credits the hypernetwork with corpus information the closed-form
   prior already supplies.
2. Should the confirmatory unit be the family rather than the corpus? This costs about half the
   effective K and buys validity against T13.
3. Is C10 (per-corpus pseudo-query tuning) in scope as a rival? L2S §6.5 suggests it works on web
   data and not on specialized corpora (D12), which is exactly where the headroom is.
4. Can enough meta-test corpora be embedded within compute? A resource gate in the style of
   ADR-0018 is needed, and the downgrade rule of §3 applies if K_test < 20.
5. Does anyone have evidence that the per-dimension Δb channel carries signal that E2 missed, and
   that would justify Level 2 before the G3 rerun?

## Sources consulted

- Repository files: AGENTS.md; docs/architecture.md; protocols.md; the implementation plan
  (§§1–2, M5–M12, E0–E6, §§13–18); experiment_registry.md; references.md; ADR-0007 to ADR-0020;
  reports overview, E1 (tables), E2, P3-bench/RUNS.md, P3-objective; specs/004 research D4.
- Papers: extracted to `/tmp/hm-papers` with pypdf from the offline uv cache. Pages are cited
  inline. I opened Learning to Select, When Reducing Representations Improves, Statistical
  Foundations of DIME, Eclipse, Unveiling DIME, AdaWidth, Search-Adaptor, Qwen3 Embedding, Leblanc
  et al., and Principled Weight Initialization, plus abstracts of Align then Train and Hypencoder
  Revisited.
- Skills applied: experimental-design (unit, blocking, pseudoreplication, randomization by
  stratum), statistical-power (SESOI, sensitivity range, simulation), and
  scientific-critical-thinking (data vs interpretation, threats, traceable critique).
