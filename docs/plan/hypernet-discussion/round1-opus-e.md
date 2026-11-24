# Round 1 — Opus-E (evidence lead): independent proposal

Author: Opus-E (Claude Opus 5.5). Written without reading any other `round*-*.md` file, including
`round0-seed.md`. Read-only on the repository except this file. No training, download, or commit.

Method: the scientific-critical-thinking appraisal workflow (fix the unit, extract before judging,
grade certainty per finding), the hypothesis-generation rule (observations, hypotheses, rivals and
predictions kept apart), and the literature-review rule (cite the location of every claim taken from
a source; say "not located" rather than "does not exist").

Notation: `D = 1024`; `k ∈ {102, 204, …, 921}` (the deciles); "env" = environment
`T = (corpus, query distribution, instruction, frozen Qwen3)`. **[V]** = I opened the file and read the
number myself. **[R]** = number restated from an ADR or a report narrative that I did not cross-check
against a table. Unmarked numbers in §1 are [V].

---

## 0. Bottom line (for readers who stop here)

1. The project has **three environments**, and its "corpus shift" evidence is overwhelmingly carried by
   **one** of them (SciFact). On MS MARCO the global selector is already as good as in-domain, and on
   NFCorpus the in-domain gain is ≈ 0.017 nDCG@10 at k = 307 (E1) and statistically indistinguishable
   from the full vector on the query-disjoint benchmark (F12). A hypernetwork's maximum payoff on
   these environments is therefore small except on SciFact.
2. A label-free, corpus-conditioned selector **already exists**: `variance`, `q_d² · Var_C(x_d)`. It
   matches every learned selector on MS MARCO and recovers most of the gap on NFCorpus, none of it on
   SciFact. Any hypernetwork must beat `variance` *and* the global selector; "beats the global selector"
   alone is not a result.
3. No result in the project shows that corpus geometry predicts the environment-specific part of a
   selector (G3 failed on every set), but the G3 test had almost no power (two training environments per
   fold). Absence of evidence, not evidence of absence.
4. Within an environment the selector family is **saturated against objective choices** (P3-objective):
   target encoding, loss, per-budget supervision, budget input, direct ranking and initialization do not
   separate from a learning-rate-tuned L2S. A hypernetwork cannot be justified by "better selectors";
   only by "a good selector without labels on a new corpus".
5. The literature supports the *form* (low-rank / residual hypernetwork, set-encoder over per-dimension
   statistics, near-zero init, sample-compression style bounds) but offers **no precedent** that a
   corpus descriptor predicts a dimension selector. The closest label-free signals (DIME pseudo-relevance,
   Eclipse pseudo-irrelevance) are *query-time* and need a first-stage retrieval.
6. My proposal (§3): a deliberately small **environment-conditioned residual on top of `variance` and a
   global selector**, whose outputs are (i) a per-budget mixing weight `a_k` (the S4 quantity, 9 numbers),
   (ii) a per-dimension bias `Δb ∈ R^1024` from a map shared across dimensions, and (iii) an optional
   rank-r modulation, all decoded from an 8-number bottleneck `z_T` (1,041 generated numbers per
   environment). It is trained episodically on **≥ 20 real environments (BEIR datasets with query splits,
   MAIR tasks; never shards of one dataset as "unseen")**, held out by domain family, and evaluated against
   `variance`, the identical model with `z_T = 0` (global-geometric), a nearest-environment selector, a
   descriptor-shuffled control, a self-trained pseudo-L2S, and a few-label (P2) fine-tune. A payoff census
   comes first: if in-domain selectors beat global-geometric by ≥ 0.005 in fewer than 30% of environments,
   no hypernetwork is built. It is killed if, on held-out environments, it does not beat global-geometric
   by ≥ 0.005 mean nDCG@10 over the 9 deciles with a significant environment-level test, or if the
   descriptor-shuffled control recovers half of its gain (§4.5).

---

## 1. Project evidence: what the results establish about corpus shift

### 1.1 Experimental unit, and why it matters

Every "corpus shift" claim is a claim about **environments**. The project has **n = 3** (SciFact,
NFCorpus, MS MARCO). Query-level tests (thousands of queries) give tight intervals *within* an
environment, but any claim of the form "selectors depend on the corpus in a way a descriptor can
predict" has an effective sample size of 3, and S1–S4/E2/P3 were all shaped on these 3
(`docs/decisions/ADR-0012`, `ADR-0013` "test_set_reuse"). Environments also differ in three
confounded ways at once: corpus, query distribution (SciFact queries are claims; NFCorpus queries are
short health topics with up to 1,363 positives; MS MARCO queries are web questions with ~1 positive),
and judgment density. **No experiment in the project varies the corpus while holding the query
distribution fixed**, so "corpus shift" and "query-distribution shift" are not separated anywhere.

### 1.2 Evidence table

Certainty uses a GRADE-like scale adapted to this setting: **High** (pre-registered or gate criterion,
replicated in ≥ 2 environments, effect well outside noise), **Moderate** (one clean test or replicated but
post hoc), **Low** (exploratory, single environment, or confounded), **Very low** (underpowered or
in-sample). The "Bears on hypernet" column is my interpretation, separated from the data columns.

| ID | Source (file) | Data (verified unless [R]) | Certainty | Bears on the hypernetwork |
|---|---|---|---|---|
| **E1-a** | `reports/E1/tables/payoff.csv` | In-domain minus global selector, nDCG@10 at k = 307: SciFact **+0.1116** [+0.078, +0.146], Holm p 0.0003; NFCorpus **+0.0171** [+0.003, +0.032], Holm p 0.035; MS MARCO **−0.0019** [−0.005, +0.001], p 0.24. (BRIEF's +0.112/+0.017/−0.002 match.) | High within env; **Low** as a general "corpus" claim (n = 3, one env carries it) | The gap a hypernet can close is ~0.11 on one env, ~0.02 on one, 0 on one. |
| **E1-b** | `reports/E1/README.md`, transfer table | Degradation when moving a selector: into SciFact +0.161 (from MS MARCO), +0.188 (from NFCorpus); into NFCorpus +0.015 (p 0.06, n.s.), +0.030; into MS MARCO +0.012, +0.015. 5/6 Holm-significant. | High (gate G2) | Transfer *cost* is asymmetric: large only when the target is SciFact. |
| **E1-c** | `reports/E1/README.md`, behavior | Top-307 overlap across envs **0.3295** vs within-env seed floor **0.7445**; uniform-random expectation is k/D = **0.2998** (my computation). Spearman of importance 0.18 across vs 0.61 within. | High that masks differ | Cross-env masks are **near chance overlap**, yet on MS MARCO/NFCorpus they lose only 0.012–0.030. Many masks are near-equivalent there: mask identity ≠ retrieval value. A hypernet supervised on mask agreement would chase noise. |
| **E1-d** | `reports/E1/README.md`, by budget | SciFact in-domain selector beats **full** by +0.135 at k = 307 (0.8353 vs 0.7000) and still +0.075 at k = 921 (0.7749). The *global* selector also beats full on SciFact (0.7238 at 307). | High (in-domain) | The SciFact effect is query-side **re-weighting/denoising**, not compression. It looks like an instruction/query-distribution correction (claims vs abstracts) at least as much as a corpus property. |
| **E1-e** | `reports/E1/tables/fit.csv` | In-domain selector over global importance: SciFact +0.140 (81 val queries), NFCorpus +0.028, MS MARCO +0.032. | Moderate | — |
| **P1** | `reports/P1/README.md` | `variance` (label-free, document variance only) mean nDCG@10 over 10–90%: MS MARCO 0.3760 vs L2S 0.3607 (**+0.015**), vs full 0.3800; NFCorpus 0.3587 vs L2S 0.3622 (−0.003, n.s.); SciFact 0.6964 vs L2S 0.8166 (**−0.120**). Recovers 271% / 88% / 11% of L2S's gain over global importance. Participation ratio does **not** order the best budget (SC-002 fails). | High within env | A label-free corpus statistic is already a strong P0 "hypernetwork" on 2/3 envs. It is the baseline to beat. |
| **S1** | `docs/decisions/ADR-0009`, `reports/S1` | Retrieval-trained linear selector beats L2S on MS MARCO +0.0073 and SciFact +0.0104 (Holm p 0.057); NFCorpus = L2S by construction (step 0 kept). [R for p-values] | Moderate (test sets later reused) | Objective change gives ~0.01. |
| **S2** | `ADR-0010` | Selector initialized from the prior: vs S1 +0.010 MS MARCO, +0.010 NFCorpus, **−0.031 SciFact**; MS MARCO selected cell = `variance` itself. [R] | Moderate | The prior helps where approximating full matters, hurts where re-ranking matters. |
| **S3** | `ADR-0011` | One learned prior weight: MS MARCO **1.0** (all seeds; = `variance`), NFCorpus **0.001–0.011**, SciFact **0.018–0.020**. [R] | Moderate | The optimal mix of "geometry" vs "learned correction" is a strongly **environment-level** quantity, spanning 3 orders of magnitude across 3 envs. |
| **S4** | `reports/S4/README.md`, `ADR-0012` | Per-budget prior weights; non-inferior to every comparator at δ = 0.005; vs aligned S3 +0.0111 NFCorpus, +0.0094 SciFact, +0.0001 MS MARCO. Mean `a_k`: MS MARCO 0.835, SciFact 0.056, NFCorpus 0.025 (`reports/E2/README.md`, context). `a_k` falls with k (NFCorpus 0.065 → 0 by k = 716). S4 vs `variance`: MS MARCO +0.0001, NFCorpus +0.0185, SciFact +0.1472. | Moderate (proposed ADR, exploratory) | **The 9-vector `a_k` is the most compact per-environment parameter the project has identified.** It is the natural first output of a hypernet. |
| **G3 / E2** | `reports/E2/README.md` | Target = S1 selection frequency. B1 (mean of other envs' profiles) Spearman 0.44/0.56/0.54; B2 (held-out env's own `variance` frequency) 0.54/0.62/0.67. **Shift probe never beats B2 on any set or env**; best gain over B1 is +0.034 (MS MARCO, set 5). Direct probe beats B2 on NFCorpus (+0.06 to +0.07) and MS MARCO (+0.08 to +0.12), not SciFact. Seed ceiling for SciFact only 0.82 (seed Spearman 0.60). | **Very low** power | Each fold fits the shift correction on the deviations of **two** envs from their mean, which are one profile and its negative: 2 × 1024 rows, but a single environment contrast, so the fit cannot tell environment-general from pair-specific structure. The failure says little. What *does* hold: descriptors predict the variance-like part of the profile, not the residual. |
| **P3** | `reports/P3/README.md` §1–3 | Top-50 contributions `q_j d_j` sit at the 0.95 percentile of both \|q\| and \|d\| (SciFact 0.949, NFCorpus 0.914, MS MARCO 0.953). `log|c|` variance ≈ 0.40 query + 0.40 doc + 0.19 covariance (NFCorpus 0.07). Relevant-minus-reference frequency excess is split-half stable at **0/1023 k on SciFact and NFCorpus**, 1016/1023 on MS MARCO (45k queries). | Moderate (descriptive, in-sample) | At corpus level, "which dimensions matter for relevance" is mostly **magnitude**, and the relevance-specific residual is only detectable with tens of thousands of queries. A corpus descriptor has little residual signal to learn from small labeled envs. |
| **P3-agg** | `reports/P3/README.md` §3 | Averaging positives cancels 0.050 (SciFact) / 0.263 (NFCorpus) of contribution mass; on NFCorpus the mean opposes ≥ 1 positive on 860 dims. | Moderate | Multi-positive envs blur oracle targets; target choice matters more there. |
| **P3-shared** | `reports/P3-shared-mask/README.md` | Shared-mask regret/mass ≈ 0.028 (SciFact), 0.13 (NFCorpus), 0.033 (MS MARCO) at k ≈ 400; one positive's mask on another: 0.10 / 0.24. | Moderate | — |
| **P3-rank** | `reports/P3-mask-ranking/README.md`, primary family | A mask built from one positive changes held-out positives' reciprocal rank vs full: SciFact **−0.093** at k = 307 (Holm p 0.012), MS MARCO **−0.048** (Holm 0.011), NFCorpus −0.002 (n.s.), +0.006 at 921. | High (pre-declared family, 3 envs, consistent sign at small k) | **Maximizing positive similarity does not improve ranking; competitors rise more.** A hypernet target must be competitor-aware (contrastive or ranking), not positive-only. |
| **P3-oracle** | `reports/P3-oracle-l2s/README.md` | SciFact single-positive, k = 307: privileged oracle 0.989, L2S **in-sample 0.972**, prefix 0.670. | Very low for generalization (in-sample by design) | The linear selector can memorize its training queries; its held-out score is ~0.80 (F12). Generalization, not capacity, is the bottleneck. |
| **F12** | `reports/P3-bench/runs/…/scifact/…/README.md`, `…/nfcorpus/…/README.md` | Query-disjoint, primary endpoint D vs `l2s-official`. SciFact (121 eval q): positive −0.1045, contrastive −0.0369, rank-guided −0.0967 (all intervals < 0); S1-scratch +0.0098 [−0.002, +0.023]; S1-official-init +0.0159 [+0.004, +0.029]; `variance` −0.0708; full −0.0681; global mask −0.149. NFCorpus (388 eval q): positive −0.043; contrastive −0.0069; S1 +0.004/+0.005 (CIs include 0); `variance` −0.0049 and full −0.0028 (CIs include 0). **MS MARCO: not in `reports/`** (RUNS.md shows it running), so unknown. | High for SciFact/NFCorpus (frozen protocol); MS MARCO missing | On held-out queries, **L2S beats full by ≈ 0.068 on SciFact and by ≈ 0.003 (n.s.) on NFCorpus.** On NFCorpus, even the in-domain selector is not distinguishable from doing nothing. |
| **P3-target** | `ADR-0019`, `reports/P3-target` | Mask-average target (E1) < L2S (E0) on both corpora; per-budget BCE (E2) recovers it; budget input recovers nothing (E3 ≈ E3c ≈ E1). [R] | Low (development data; later explained by lr) | — |
| **P3-objective** | `reports/P3-objective/README.md`, `post-hoc-A1/README.md` | Only C11 (lr tuning of L2S) supported: +0.0136 SciFact, +0.0043 NFCorpus. Post hoc A1: no C1–C10 contrast supported; SciFact KS deficit −0.033 → −0.007. Direct ranking R0 vs tuned L2S: +0.005 SciFact, −0.004 NFCorpus (mixed). | Moderate (pre-registered, but development data, 2 envs) | Within an env the linear selector is **saturated w.r.t. objective**; optimization scale (lr) matters as much as any objective. Expect the same sensitivity, amplified, in a hypernet. |
| **ADR-0001** | `docs/architecture.md`, "Design choices" | A query-conditioned (Hypencoder-like) design "was tried and did not beat the linear selector". [R; ADR-0001 itself not re-read] | Low | Do not spend capacity on per-query generated networks. |

### 1.3 What the project evidence establishes, and what it does not

**Established (High/Moderate):**
- Selectors trained on different environments pick different dimensions (E1-c), and moving one costs
  retrieval (E1-b). Both hold on n = 3.
- The *size* of the payoff is wildly heterogeneous: ~0.11 (SciFact), ~0.02 (NFCorpus), ~0 (MS MARCO) at
  k = 307 (E1-a), and on held-out queries the in-domain selector's advantage over full is ≈ 0.068 on
  SciFact and ≈ 0.003 n.s. on NFCorpus (F12).
- A label-free corpus statistic (`variance`) is as good as any learned selector on MS MARCO and close
  on NFCorpus (P1, S4).
- The optimal weight between geometry and learned correction is an environment-level quantity spanning
  three orders of magnitude (S3, S4).
- Positive-only targets do not improve ranking (P3-rank, F12 teacher arms).
- Objective choices do not separate within an environment once the learning rate is tuned
  (P3-objective).

**Not established:**
- That the shift is caused by the **corpus** rather than the query distribution or instruction. SciFact's
  large gain also appears with the *global* selector beating full (E1-d), which points at a query-side
  correction, and SciFact is the env whose queries differ most in form from its documents.
- That any descriptor predicts the shift (G3: Very low power; failed).
- That a fourth environment would look like any of the three. With n = 3, the between-environment
  variance of the payoff is unestimated.

**Rival explanations of the E1 shift that a hypernetwork design must survive** (hypothesis-generation
step 5; all `candidate`):
- R1 *Query-distribution/instruction mismatch*: the selector learns to suppress dimensions that encode
  query form (claims, questions) rather than corpus topic. Prediction: P1 (with `Q_calib`) beats P0
  sharply; corpus-only descriptors fail.
- R2 *Label-density artifact*: NFCorpus's many graded positives and MS MARCO's ~1 positive produce
  different oracle targets regardless of geometry. Prediction: the shift shrinks when targets are
  equalized (e.g. top-1 positive only).
- R3 *Flat landscape*: on most corpora many masks are near-equivalent (E1-c), so "shift" is seed/selection
  noise except where one direction dominates. Prediction: on new environments, `variance` ≈ in-domain
  in most, with rare large-gain outliers.
- R4 *Genuine corpus geometry* (the project's H2): prediction: descriptor-shuffled controls lose to real
  descriptors on held-out environments.

---

## 2. Literature: what the supplied papers say that bears on a corpus → selector hypernetwork

Scope and boundary (literature-review rule): only the PDFs in `/home/wilker/Papers/` (personal
document excluded), text extracted to `/tmp`, read on 2026-11-24. No external search was run, so "not
located" below means "not in these papers", not "does not exist". Page numbers are PDF pages (`p.`);
for the one PDF whose text extraction lost page breaks (Hypencoder Revisited) I cite sections.

### 2.1 Synthesis by theme

**T1. Dataset-conditioned hypernetworks: the form is well supported, the data regime is not.**
- Leblanc et al. (ICML 2025) is the only paper here that maps *a dataset* to a predictor's parameters.
  Encoder–decoder hypernetwork with an explicit bottleneck: a permutation-invariant DeepSet encodes the
  task's sample, a decoder emits the downstream weights (§3, eq. 3, p.4; DeepSet Def. 4.1, p.7). Its
  PAC-Bayes variant puts a Gaussian posterior `N(µ, I)` on the latent and the bound's complexity term is
  `‖µ‖²` (eq. 4, p.5), so the size of environment-specific information is *measurable*.
- Three facts limit transfer to HyperDIME. (i) The certificates bound the generated predictor's risk
  using the **new task's labeled sample** (`L̂_{S'}` in the bounds of §3.1–3.2, pp.5–6): under P0/P1 there
  is no such sample, so no certificate exists; only P2 could carry one. (ii) Their meta-training sets are
  300 tasks (moons, §4.2 p.7), 90/150 tasks (MNIST/CIFAR, p.8), and, in the regime closest to ours,
  10 meta-train tasks (pixel swap, p.8). (iii) With 30 training tasks on label-shuffled MNIST, "severe
  overfitting on meta-train tasks" despite an extensive grid (Appendix, p.22). Baxter's framework,
  which they adopt, assumes tasks are i.i.d. draws from an environment distribution (p.2); BEIR-style
  corpora chosen by hand are not.
- Ha et al. (HyperNetworks) show a large weight block can be generated from a tiny embedding: a 12,544-
  weight kernel from a 4-number embedding through a 4,240-parameter hypernetwork with near-equal MNIST
  accuracy (§4.1, p.6). The idea transfers as "generate a few coefficients in a shared basis", not as
  "generate a 1024×1024 matrix".
- Chang et al. (ICLR 2020): standard Glorot/He initialization applied to a hypernetwork produces mainnet
  weights at the wrong scale; Adam hides much of it until the loss diverges (§1, p.1); hyperfan-in/out
  formulas fix the variance (§4, Table 1, p.6). The project already found that **optimization scale**
  decides P3-objective's contrasts (C11; A1), so hypernetwork scale control is not a detail.
- Shen et al. (survey) list hypernetworks as a control function `h(s_desc)` that maps a task/domain
  description to all or part of the learner's weights (§4.1.3, p.11), with recommendation examples
  (Hamur, multi-domain adapters). It establishes the vocabulary, not evidence for retrieval.

**T2. Per-query generated networks in retrieval (Hypencoder): strong in-domain, uneven out-of-domain,
expensive, not our design.**
- Killingback et al. generate a per-query q-net from contextualized query token embeddings; the hyperhead
  "had stable training" across variants (§3.3, p.4). The encoder is trained end to end, not frozen.
- Eichholtz et al. (Revisited, §7 Conclusion): in-domain claims reproduce; out-of-domain "the advantage
  does not extend uniformly to the broader BEIR benchmark"; gains with other encoders "depend on the
  encoder and fine-tuning strategy" (Abstract); training costs six days on two A100s and was not
  re-run from scratch (§7). Combined with ADR-0001 (query-conditioned design did not beat the linear
  selector [R]), the literature gives no reason to put capacity into per-query generation.

**T3. Label-free dimension-importance signals exist, but they are query-time and need a first-stage
retrieval.**
- DIME (Faggioli et al., SIGIR 2024): magnitude DIME `u(i) = |q_i|` (eq. 4) and PRF DIME
  `u(i) = q_i · p_i`, p = centroid of the top-k_f retrieved documents (eq. 5), both §3, p.5; the oracle
  DIME uses Pearson correlation with judgments (eq. 3, p.5).
- Eclipse (D'Erasmo et al., ICTIR 2025): `u(i) = α q_i p*_i − β q_i p̄_i`, p̄ = centroid of the *bottom* k⁻
  documents of the same retrieved list (eq. 1, §3, p.3). This is the label-free counterpart of
  L2S's target `softmax((p̄ − n̄) ⊙ q / τ)` (ADR-0020 "Facts"): Eclipse replaces judged positives and mined
  negatives with pseudo-relevant and pseudo-irrelevant documents.
- Statistical Foundations of DIME (D'Erasmo et al., EACL 2026): under `q = θ + ε z`, keep dimension i iff
  `θ_i² > ε²` (Theorem 1, pp.2–3); Kernel DIME `u_q = q ⊙ Σ w_i d_i` (Def. 1, eq. 3, p.3) is unbiased for θ²
  with uniform weights (Theorem 2, p.3); RDIME sets `ε̂² = (1/p) Σ (q_i² − u_i)` and keeps `u_i > ε̂²`
  (Corollary 1, p.3): a per-query budget with no grid search.
- Unveiling DIME (Campagnano et al., SIGIR 2025): DIME as denoising (§3); across 13 BEIR tasks,
  "different tasks favor different levels of pruning" and strong baselines (Quora, NQ) barely move
  (§5, p.8). This is independent support for an **environment-dependent optimal budget/strength**,
  which is what S3/S4's `a_k` captures in the project.
- **Not located in any paper:** a corpus-level (query-independent) descriptor that predicts a dimension
  selector or its strength. All label-free signals above are computed per query at search time.

**T4. Learned selectors and subsets: per-dataset, and they do not transfer.**
- Learning to Select (Wu et al., ACL 2026) trains per dataset "without cross-domain transfer" (§4, p.4)
  and states as a limitation that "predictors trained on one dataset do not consistently generalize well
  across domains" (Limitations, p.9). Its LLM pseudo-query variant (doc → LLM → query, the source doc as
  positive) gives "non-trivial gains" on MS MARCO but "only marginal improvements" on SciFact and NFCorpus
  (Table 5 and text, p.8): **synthetic queries help when the target query distribution is generic**. This
  is the same pattern as the project's E1/P1 (MS MARCO easy, SciFact query-specific) and supports rival
  R1 (§1.3).
- Pasin et al. (When Reducing Representations Improves): query-wise genetic-algorithm subsets reach up to
  0.981 nDCG@10 in an oracle setting (Abstract, p.1), matching the project's in-sample oracle (0.989,
  P3-oracle). A single subset learned on DL-19 transfers to DL-20/Robust04 with changes "in the third
  decimal place" (Table 3 and §4.4, p.11): that is **non-inferiority of a mild (~12%) reduction**, not
  transfer of a gain (DL-20 0.682 vs 0.684 full for TAS-B; Robust04 0.439 vs 0.447).

**T5. Corpus-level determinants of how many dimensions a query needs (AdaWidth).**
- Yang & Zhao (AdaWidth, 2026): the width a query needs is set by the competing documents at the cutoff,
  an order statistic (§4.1, eqs. 13–16, p.6); under exponential overtaking, `q_τ[M_first] ≲ a + b log N +
  c log k` with predicted signs b > 0, c < 0 (Proposition 1 and eqs. 21–23, pp.6–7), validated on
  subsampled corpora (§5.1, p.7). Their adapter rotates documents too (index changes), which HyperDIME
  forbids; the **scaling law is the transferable part**: corpus size and competitor density are principled
  P0 descriptors for the budget/strength outputs. Note their baselines include L2S retrained per encoder
  and dataset, with large reported margins (p.7); I did not verify those numbers.

**T6. Supervised per-dataset adapters: the payoff of adaptation is heterogeneous across corpora.**
- Search-Adaptor (Yoon et al., ACL 2024): residual adapter `ê = e + f(e)` shared by queries and corpus
  (§4.1, p.4), ranking loss + recovery (L1 to the original embedding) + prediction regularizers (§4.2–4.4,
  pp.4–5). Over 14 BEIR datasets, average +0.0475 (Google API) and +0.0349 (OpenAI) nDCG@10, with
  SciFact +0.0856 and ArguAna +0.1739 (§5.2, p.5). For test-only datasets they split queries 50/50 by sorted
  id (§5.1, p.5); training pairs range 0.7K–532K (§5.1; Appendix A.1, p.12). Separate query/corpus
  adapters were worse than a shared one (Table 4, p.8). Two transferable points: (a) a **14-dataset pool with
  query splits** is an established way to obtain many labeled environments; (b) a **recovery-to-base
  regularizer** is the adapter analogue of a hypernetwork's `‖Δθ_T‖` shrinkage.
- Align then Train (Maekawa et al., 2026): query-side adapters "without re-indexing the corpus" on MAIR,
  **126 retrieval tasks across 6 domains with about 80 labeled queries per task** (§5 Datasets, p.6; 10,038
  queries total). A large pool of environments, but with few labels each: suitable for meta-train breadth
  and for P2 few-label evaluation, weak as a source of per-environment oracle selectors.
- SPIN (Jiao et al.) selects salient internal neurons by linear probing for classification
  (Abstract, p.1); it is a different task and space and does not bear on corpus conditioning.

### 2.2 What the literature implies for the design (interpretation)

| # | Implication | Source |
|---|---|---|
| L1 | Generate **few coefficients in a shared learned basis** (bottleneck), not a full or per-env low-rank W. | Ha §4.1 p.6; Leblanc §3.1 p.5, §5 p.9 ("parameters varying across tasks admit a compact representation") |
| L2 | Expect meta-overfitting below a few tens of environments; size outputs to the number of environments. | Leblanc pp.8, 22 |
| L3 | Make zero modulation the exact base selector and control output scale; tune the hypernet's learning rate on the same grid as the comparators. | Chang §1, §4; project C11/A1 |
| L4 | Label-free per-query signals (PRF/Eclipse) are strong **competitors** and natural **P1 descriptor inputs** when aggregated over unlabeled calibration queries. | DIME eq. 5 p.5; Eclipse eq. 1 p.3 |
| L5 | Corpus size and competition density should enter as descriptors for the strength/budget outputs. | AdaWidth Prop. 1, pp.6–7 |
| L6 | The optimal strength/budget varies by dataset under label-free estimators: an environment-level scalar head is justified. | Unveiling DIME §5 p.8; project S3/S4 |
| L7 | The query distribution, not only the corpus, determines whether synthetic/label-free supervision works: P0 may fail where P1 succeeds. | L2S Table 5 p.8 |
| L8 | Build the environment pool from BEIR-style datasets with query-split train/test, plus MAIR for breadth. | Search-Adaptor §5.1 p.5; Align then Train §5 p.6 |
| L9 | No paper provides a certificate for unlabeled adaptation; P2 is the only protocol that can carry a PAC-Bayes/sample-compression bound. | Leblanc §3 pp.5–6 |

---

## 3. Proposal: a geometry-anchored, bottlenecked residual hypernetwork ("HyperDIME-R")

### 3.1 Design rules derived from §1–§2 (each rule cites its evidence)

1. **Anchor on what already works without labels.** The base score contains the `variance` prior with the
   new corpus's own document variance (P1, S2–S4). A hypernetwork that must rediscover geometry would spend
   capacity on what is free.
2. **Predict environment-level quantities first.** The one parameter that clearly varies across
   environments is the prior-vs-correction balance (`a`, S3: 1.0 / 0.002–0.011 / 0.018–0.020; `a_k`, S4).
   It is 9 numbers.
3. **Per-dimension outputs only through a map shared across dimensions.** With few environments, a
   dimension-ID embedding plus per-environment outputs can memorize environments (plan risk R2). A map
   `r_{T,j} → Δb_{T,j}` tied across dimensions gets 1024 rows per environment. E2's *direct* probe (exactly
   this kind of map, linear) beat the held-out environment's own `variance` profile on NFCorpus and MS MARCO
   (§1.2 G3/E2): the only positive descriptor signal the project has.
4. **Bottleneck the environment information** and measure it (Leblanc L1, L2): an 8-number latent `z_T`.
5. **Competitor-aware supervision, not positive-only** (P3-rank, F12 teachers).
6. **Objective is not where gains are** (P3-objective): use the S4 training recipe; tune the learning rate
   on an equal grid for every arm.
7. **Index untouched, query not re-normalized, exact scoring** (hard constraints); AdaWidth's and
   Search-Adaptor's document-side transforms are therefore out of scope.

### 3.2 Model

Selector for environment T at budget k ∈ K = {102, …, 921} (scores before top-k):

```
s_{T,k}(q) = a_{T,k} · log(q² ⊙ v_T + ε)                          # geometric prior, v_T = Var_C(x) per dim (P0, exact)
           + W_0 q + b_0                                           # global correction, meta-trained, 1,049,600 params
           + U diag(γ_T) Vᵀ q                                      # environment modulation in a shared rank-r basis
           + Δb_T                                                  # environment per-dimension bias
mask_{T,k}(q) = top-k(s_{T,k}(q));  score(q, d) = ⟨q ⊙ mask, d⟩    # documents/index untouched, no renormalization
```

Generated per environment, `θ_T = (a_T ∈ R^9_{>0}, γ_T ∈ R^r, Δb_T ∈ R^1024)`: 1,041 numbers at r = 8.
Shared meta-parameters: `W_0, b_0` (1,049,600), `U, V ∈ R^{1024×r}` (16,384 at r = 8), plus the hypernetwork.
With `z_T = 0` the model is a **global-geometric selector** (global S4 with each environment's own `v_T`):
the mandatory capacity- and label-matched control (protocols.md, risk R6).

**Hypernetwork** (≈ 5k parameters; numbers are targets for the pilot, not tuned values):

```
Descriptor   R_T ∈ R^{1024×h}     per-dimension rows (P0: groups A+E, h≈24; P1: + B, C, D-pseudo, h≈40)
             g_T ∈ R^{m}          environment scalars (P0: log N, participation ratio, mean energy, log mean
                                  nearest-neighbour similarity of a document sample; P1: + mean RDIME ε̂² over
                                  Q_calib, mean score gap rank1−rank100 over Q_calib), m ≤ 8
Normalize    per feature, statistics from meta-train environments only (leakage rule 2)
Encoder      e_j = φ(r_{T,j})  (MLP h→32→32, shared over j);   ē = mean_j e_j  (and std_j e_j)
             µ_T = E([ē, std, g_T]) ∈ R^8   (MLP →32→8);   z_T = µ_T + σ·ξ, ξ~N(0,I) in training only
Heads        log a_T   = log a_0 + A z_T            (A ∈ R^{9×8}, zero-init; a_0 = global's learned a_k)
             γ_T       = C z_T                      (C ∈ R^{r×8}, zero-init)
             Δb_{T,j}  = ψ([r_{T,j}, z_T])          (MLP (h+8)→32→1, shared over j, last layer zero-init)
```

- All last layers are zero-initialized, so at step 0 every generated selector equals the global-geometric
  selector exactly (architecture.md invariant "zero modulation = base"; Chang L3). Hidden layers use
  standard fan-in init; hyperfan scaling matters little when the output layer starts at zero, but the
  learning rate of the hypernet is tuned on the same grid as the controls.
- **No dimension-ID embedding in v1.** It is added only if a pilot shows the tied map underfits on
  meta-train *and* meta-validation improves with it.
- The Group D "pseudo" channel (P1 only): for each unlabeled calibration query, Eclipse's
  `q_j (p̄⁺_j − p̄⁻_j)` with p̄⁺ the top-5 and p̄⁻ ranks 90–100 of the unmasked exact list (DIME eq. 5, Eclipse
  eq. 1), averaged per dimension over `Q_calib`. No judgment is read.

**Online cost** per query: as S4 plus `2·1024·r` FLOPs for the rank-r term (≈ 16k at r = 8) and one
elementwise log: one 1024×1024 mat-vec (≈ 2.1 MFLOP) dominates. **Offline cost** per new environment: one
streamed pass over the corpus for `v_T` and group A/E statistics (O(N·D)); for P1, one exact search per
calibration query; then one forward pass of the hypernetwork. Reported separately (constraint).

### 3.3 Why these outputs and not `A_T B_Tᵀ` from the plan

The plan (M7/M8) generates `A_T, B_T ∈ R^{1024×r}` per environment: 16k–65k free numbers per environment
from a descriptor, learnable only with many environments (Leblanc L2: overfitting at 30 tasks). Here the
only per-environment free quantities are 8 latent numbers; everything else is shared. This is the
"compact representation of what varies across tasks" that Leblanc et al. recommend (§5, p.9) and Ha et
al. exploit (§4.1, p.6). If the pilot shows the 8-dim bottleneck saturates (meta-validation keeps
improving with its width), widen it before adding per-environment matrices.

### 3.4 Training

**Environment pool (the binding constraint).** Target ≥ 20 real environments, grouped into domain
families so that held-out folds remove a whole family:
- BEIR datasets with training qrels (MS MARCO, NQ, HotpotQA, FiQA, FEVER, NFCorpus, SciFact, …) and BEIR
  test-only datasets split query-wise 50/50 by sorted id, as Search-Adaptor did (§5.1, p.5) (ArguAna,
  SciDocs, TREC-COVID, Touché, Climate-FEVER, DBPedia, …).
- MAIR tasks for breadth (126 tasks, ~80 labeled queries each; Align then Train §5, p.6), used for meta-train
  and for P2 few-label evaluation; their small label counts make their in-domain selectors noisy, so they
  are weighted down in the episodic loss (weight ∝ min(1, n_train/200)).
- **Corpus-only shifts** to separate corpus from query distribution (rival R1): cluster MS MARCO's corpus
  into ~8 topical shards (k-means on document embeddings, seed fixed), keep the MS MARCO training queries
  whose positives fall in a shard. Same query distribution, different corpus. Used for meta-train augmentation
  and for the mechanistic test M2 (§4.4); never as meta-test (protocols.md: shards are not unseen domains).
- SciFact, NFCorpus and MS MARCO are meta-train only: S1–S4/E2/P3 were shaped on them.

Downloads and encoding of new corpora are future work; nothing is fetched in this round.

**Episodic loop** (plan M9, with the S4 loss):

```
for step in 1..S:
    T ~ meta-train environments (probability ∝ sqrt(n_train_T), so MS MARCO does not dominate)
    R_T, g_T = descriptor(T, protocol)          # cached; P0 docs only, P1 docs + unlabeled train-split queries
    θ_T = H(R_T, g_T)                            # one forward pass; z_T noisy (σ = 1) during training
    B  = 32 training queries of T (fit split), positives, 64 hard negatives mined from T's train split only
    L  = (1/9) Σ_k masked_retrieval_loss(s_{T,k}, B; relaxed top-k at k, β annealed 1→0.1, τ_s = 0.05)
       + λ_z ‖µ_T‖² / 2                          # PAC-Bayes KL to N(0, I) (Leblanc eq. 4)
       + λ_Δ (‖Δb_T‖² / 1024 + ‖γ_T‖²)           # recovery-to-base shrinkage (Search-Adaptor's L_Rec analogue)
    update H, W_0, b_0, U, V
```

- Stage A (warm start): train the global-geometric selector (`z = 0`) on all meta-train environments with
  the same loss; this is also the control arm. Stage B: freeze nothing, train the hypernet from Stage A.
- Model selection: on meta-validation environments only, by mean nDCG@10 over the 9 deciles on each
  environment's tune queries, environment-averaged. Grid: lr ∈ {1e-4, 3e-4, 1e-3, 3e-3} (same grid for the
  global-geometric control), λ_z ∈ {0, 1e-3, 1e-2}, λ_Δ ∈ {0, 1e-3}, r ∈ {0, 8}. Seeds 0–2 for the selected
  cell and for the control.
- The masked retrieval loss is the S1/S4/R0 objective (`masked_retrieval_loss`), chosen because per-budget
  `a_k` needs a budget-specific relaxation (a KL to one averaged target cannot identify per-budget weights,
  ADR-0019/0020 CE linearity); P3-objective found it statistically indistinguishable from tuned L2S.

**Two-stage diagnostic (not the method).** Fit in-domain S4 selectors per meta-train environment; regress
their `a_k` and frequency profiles on descriptors (leave-family-out). This is G3 rerun with adequate n and
tells whether the hypernet's targets are predictable before end-to-end training.

---

## 4. Evaluation, gates, and the results that would falsify it

### 4.1 Hypotheses (all `candidate`; kept apart from predictions)

- **H3′ (primary, causal within the design):** on environments unseen in meta-training, conditioning the
  selector on `z_T` from P0 (resp. P1) descriptors raises mean nDCG@10 over the 9 deciles relative to the
  identical model with `z_T = 0`, trained on the same labels with the same effort.
- **H3′-a (strength):** most of that gain comes from `a_T` (9 numbers), not from `Δb_T` or `γ_T`.
- Rivals R1–R4 of §1.3 are tested by the mechanistic contrasts in §4.4.

### 4.2 Experimental unit, endpoint, statistics

- **Unit of inference: the environment.** Queries are nested within environments; the primary test treats
  each meta-test environment as one observation, equally weighted (no query weighting, so MS MARCO-sized
  environments do not dominate).
- **Per-environment effect:** `Δ_T = mean_q [ȳ_q(arm) − ȳ_q(control)]`, `ȳ_q` = query q's nDCG@10 averaged
  over k ∈ {102, …, 921} (the F12 endpoint, so numbers are comparable with ADR-0018), exact float64
  scoring, full corpus, lower-row tie rule, parity at k = 1024.
- **Primary test:** one-sided paired sign-flip over environments on `mean_T Δ_T`, exact over all `2^{n_E}`
  sign vectors when `n_E ≤ 16`. Primary contrast: HyperDIME-R (P0) vs global-geometric (`z = 0`); then P1 vs
  global-geometric; Holm over the two.
- **Claim levels** (mirroring ADR-0018): *superior* = Holm p < 0.05; *practical* = superior and
  `mean Δ ≥ 0.005`; *robust* = practical and leave-one-environment-out jackknife never changes the sign.
- **Secondary:** per-environment query-level sign-flip with Holm across environments; gap closure
  `G_T = (arm − control)/(in-domain − control)` reported only where the denominator's query-level CI excludes
  zero (otherwise the ratio is unstable); per-decile curves; MRR@10, Recall@100.
- **Power (to be computed, not assumed).** If the between-environment SD of `Δ_T` is 0.010 and the true mean
  is 0.005, n_E = 12 gives t ≈ 1.7 and roughly 50% power one-sided at α = 0.05; at SD 0.006 it is ≈ 85%
  (normal approximation, my arithmetic). The SD is unknown: estimate it from leave-family-out runs on
  meta-train (Stage 2) **before** fixing n_E, and do not run meta-test with < 80% estimated power for a
  0.005 mean effect.

### 4.3 Comparators (every table labels the protocol: P0, P1, P2, in-domain)

| ID | Arm | Protocol | Why |
|---|---|---|---|
| C0 | full 1024 | — | mandatory |
| C1 | prefix-k (MRL) | — | mandatory (risk R5) |
| C2 | `variance` (`q² · Var_C`) | P0 | the existing label-free corpus-conditioned selector (§1, P1/S4) |
| C3 | **global-geometric**, `z = 0` | P0 | primary control: same architecture, labels, effort (risk R6) |
| C4 | global L2S without prior | — | E1's global selector, for continuity |
| C5 | nearest environment: in-domain S4 of the meta-train environment closest in standardized `g_T` | P0/P1 | a non-parametric "hypernetwork"; if it ties, the learned map adds nothing |
| C6 | HyperDIME-R with descriptors of a random other meta-train environment (20 draws) | — | conditioning-information control: separates corpus information from extra capacity/regularization |
| C7 | PRF-DIME and Eclipse at query time | P1-like | label-free per-query competitors; higher online cost, reported separately |
| C8 | pseudo-L2S: an L2S trained on the target environment's unlabeled calibration queries with Eclipse pseudo-positives/negatives | P1 | self-training alternative that needs no hypernetwork |
| C9 | P2: global-geometric and HyperDIME-R each fine-tuned on 50 labeled training queries of the target | P2 | few-label scenario, never called zero-shot |
| C10 | in-domain S4 on the target's own training labels | reference | the gap's upper end; computed after the hypernet is frozen, never feeds it |

C10 uses training labels of a meta-test environment to train a *reference arm*. That is not adaptation of
the hypernetwork, but AGENTS.md says "any label from a meta-test environment" must stay out of training;
Opus-M should rule on whether C10 is admissible or must use a disjoint environment-level split.

### 4.4 Staged plan with gates (cheapest and most informative first)

**Stage 0 — payoff census (no hypernetwork code).** On every environment of the pool: in-domain S4,
global-geometric, `variance`, full, prefix, query-disjoint (`p3-query-split-v1`-style split per environment).
Also two **oracle-output ceilings**: global-geometric with `a_k` tuned per environment on its tune labels
(9 numbers), and with `Δb_T` fitted per environment (1024 numbers, W frozen).
- *Gate G0:* in-domain − global-geometric ≥ 0.005 with a query-level 95% CI above zero in ≥ 30% of
  environments. If it fails, **stop**: the landscape is flat (R3) and a hypernetwork has nothing to close.
- The oracle ceilings say which outputs are worth predicting. If tuned `a_k` alone closes ≥ 70% of the
  gap, build only the `a_T` head (HyperDIME-0 below).

**Stage 1 — predictability (G3 rerun with adequate n).** Leave-family-out regression of Stage-0 targets
(per-environment `a_k`; per-dimension in-domain-minus-global frequency profile) on P0 and P1 descriptors,
against B1/B2 of E2 and a permutation test (environments' descriptors permuted). Advisory: a failure is
recorded, but end-to-end training may still find signal.

**Stage 2 — HyperDIME-0, then HyperDIME-R, on meta-validation.** HyperDIME-0 = only the `a_T` head from
`g_T` (≤ 8 scalars → 9 numbers; ridge or a 2-layer MLP with ≈ 200 parameters). HyperDIME-R adds `Δb_T`, `γ_T`.
- *Gate G4-val:* environment-mean Δ vs C3 ≥ 0.003 on meta-validation **and** C6 (shuffled descriptors)
  recovers less than half of that gain. Otherwise stop before meta-test (plan M9 rule).
- Estimate the between-environment SD of Δ here for the power calculation.

**Stage 3 — meta-test, once,** with the protocol, environment list, checkpoints and comparator set
fingerprinted before any meta-test label is read (ADR-0018 freeze-gate pattern).

**Mechanistic contrasts (descriptive unless pre-registered):**
- M1 (bottleneck use): `‖µ_T‖²` vs the C10 − C3 gap across meta-test environments; H3′ predicts a positive
  rank correlation.
- M2 (corpus vs query distribution, R1 vs R4): on MS MARCO corpus shards (same query distribution), R4
  predicts in-domain − global gaps comparable to those between real datasets; R1 predicts ≈ 0.
  Complement: P0 vs P1 on meta-test (R1 predicts P0 ≈ C3 < P1).
- M3 (label density, R2): the C10 − C3 gap vs positives per query and judgment depth.

### 4.5 Falsification: results that would kill or narrow the idea

| ID | Result | Consequence |
|---|---|---|
| K0 | G0 fails (gap < 0.005 in > 70% of environments) | No hypernetwork; the global-geometric selector (or `variance`) is the answer. |
| K1 | Meta-validation Δ vs C3 < 0.003, or C6 ≈ HyperDIME-R | Stop before meta-test; conditioning carries no usable corpus information. |
| K2 | Meta-test primary sign-flip p ≥ 0.05, or mean Δ < 0.005 | H3′ refuted for this design and pool. |
| K3 | C5 (nearest environment) within 0.002 of HyperDIME-R | The learned map adds nothing over retrieval of a stored selector. |
| K4 | P0 ≈ C3 while P1 > C3, and M2 shows ≈ 0 shard gaps | The "corpus shift" framing is wrong; the shift is query-distribution/instruction. The method is re-scoped to P1 and the claim renamed. |
| K5 | Jackknife over meta-test environments flips the sign | Gain is driven by one or two environments; no general claim. |
| K6 | HyperDIME-R (P1) below C8 (pseudo-L2S) by > 0.002 | A hypernetwork is unnecessary in practice: per-environment self-training is better. |

**My prior, labeled as opinion:** the most likely outcome on current evidence is that G0 passes on a
minority of environments, that HyperDIME-0's `a_T` captures most of what is predictable from P0 (strength
of the geometric prior vs correction, driven by corpus size/effective dimensionality), and that the large
SciFact-like gains are query-distribution effects visible only under P1 (K4 partially true). The design is
built so that each of these outcomes is identifiable rather than absorbed into a single headline number.

---

## 5. Risks, open questions for the other participants, and verification log

### 5.1 Risks specific to this design

- **Environment pool quality.** MAIR tasks carry ~80 labeled queries; their in-domain references (C10) and
  Stage-0 gaps will be noisy. Gate G0 counts only environments whose query-level CI excludes zero.
- **Near-duplicate environments across folds** (e.g. FEVER / Climate-FEVER, NQ / HotpotQA share Wikipedia).
  Folds must hold out whole domain families, or meta-test inflates.
- **`a_k` is not scale-identified.** Top-k is invariant to a positive rescaling of the whole score, so `a_k`
  trades off against `‖W_0 q + b_0‖`. Report S4's `prior_share` (ADR-0012) next to `a_k`; regress on it in
  Stage 1, not on raw `a_k`.
- **MS MARCO in the pool.** With 8.8M documents it dominates compute and, if sampled by size, the loss.
  Sampling ∝ sqrt(n_train) and environment-equal evaluation weights address the second; the first is a
  resource-rule question (ADR-0018 pattern).
- **F12 on MS MARCO is not reported yet** (`reports/P3-bench/RUNS.md`, "running"). If L2S does not beat full
  on held-out MS MARCO queries, two of three current environments show no in-domain payoff over full on
  held-out queries, which strengthens K0 risk.

### 5.2 Questions for Astra (architecture) and Opus-M (methods)

1. Astra: is an 8-number bottleneck plus a tied per-dimension map enough capacity, or do you see evidence
   for per-environment low-rank matrices that §1 missed? Which result in `reports/` would justify it?
2. Astra: should `W_0` be frozen after Stage A (hypernet learns only residuals) to reduce meta-overfitting,
   at the cost of joint optimization?
3. Opus-M: is the environment the right primary unit given that n_E will be ~10–20, or should the primary
   test be a hierarchical model (queries within environments) with the environment as random effect?
4. Opus-M: admissibility of C10 (in-domain reference trained on a meta-test environment's training labels).
5. Opus-M: should Gate G0 be a hard stop, given it is evaluated on environments that will later be
   meta-test (it reads their labels via in-domain training)? A clean alternative is to run G0 only on
   meta-train + meta-validation environments.
6. Both: K4 would re-scope the project from "corpus" to "environment incl. query distribution". Is that an
   acceptable outcome to pre-register, or should the claim be stated as "environment" from the start?

### 5.3 Verification log

- Verified against files in this round [V]: every E1 number (`reports/E1/README.md`, `tables/payoff.csv`,
  `tables/fit.csv`); P1 (`reports/P1/README.md`); S4 non-inferiority and comparison rows
  (`reports/S4/README.md`); E2/G3 tables (`reports/E2/README.md`); P3 §1–3 figures (`reports/P3/README.md`);
  shared-mask regrets (`reports/P3-shared-mask/README.md`); mask-ranking primary-family rows
  (`reports/P3-mask-ranking/README.md`); oracle vs L2S SciFact rows (`reports/P3-oracle-l2s/README.md`);
  F12 SciFact and NFCorpus endpoint and per-decile tables (`reports/P3-bench/runs/evaluation/P3-bench/
  evaluation/{scifact,nfcorpus}/*/README.md`); P3-objective primary and A1 tables
  (`reports/P3-objective/README.md`, `post-hoc-A1/README.md`).
- Restated, not cross-checked against a table [R]: S1–S3 effect sizes and p-values (from ADR-0009–0011),
  P3-target stage summaries (ADR-0019), ADR-0001's outcome (via `docs/architecture.md`).
- My own arithmetic: random-overlap expectation k/D = 0.2998; the power figures in §4.2 (normal
  approximation).
- Literature numbers are as printed in the PDFs; I did not re-derive them (e.g. Search-Adaptor's per-dataset
  gains, AdaWidth's margins over L2S, Pasin et al.'s Table 3).
