# ADR-0020: P3 objective study, target versus loss versus direct ranking (exploratory)

- **Status:** accepted (protocol recorded before any run; amendment A1 recorded post hoc before
  it ran; evidence added after the runs)
- **Date:** 2026-11-15
- **Issue:** none assigned (no ticket; administrative)

## Context

ADR-0019's pilot found that a selector trained on the averaged contrastive-mask target (E1) ranks
below Learning2Select (L2S, E0), and that a per-budget membership loss on the same masks (E2)
recovers the gap. Its reading overreached in two ways, corrected here and in ADR-0019:

- Decoding the target back to top-k preserved the positive and contrastive masks exactly, but not
  the rank-guided ones (SciFact fit nDCG@10, mean over nine budgets: teacher 0.9897, decoded
  0.9793). "Decoding loses nothing" holds only for the nested teachers.
- E1 → E2 changed the loss (KL → relaxed-membership BCE) and the way the masks are presented to it
  (one averaged distribution → nine binary label vectors) at once. The pilot cannot say whether
  the gain is due to the loss or to the target, and "the average alone causes the loss" is not
  established.

This study separates target from loss, per-budget supervision from aggregation, budget-input
capacity from per-budget information, direct ranking training from distillation, and
initialization from objective. It is separate from the frozen F12 family (ADR-0018). That
family's protocol, results and MS MARCO stage are unchanged, and none of its evaluation data is
read.

### Facts the design relies on (checked on the F12 fit artifacts before writing this ADR)

- One source, `s = (p̄ − n̄) ⊙ q̂`, produces both encodings. L2S's target is
  `t_off = softmax(s/τ)` (stored `targets.pt`). The contrastive mask `m_k` is `top-k(s)`.
  `top-k(t_off) = m_k` on SciFact (0 differing coordinates at k = 102 and 921). On NFCorpus 10
  and 4 coordinates differ, out of 1,810 queries × k; these are float32 ties.
- The contrastive masks are exactly nested: 0 violations on both corpora. Hence the averaged
  target `t̄ = (1/9) Σ_k m_k/k` is a step function of the rank level of `s`, and every `m_k` is
  recovered exactly from `t̄` by thresholding. **`t̄` and `{m_k}` carry identical information**,
  presented in two domains.
- `t_off` is flat: its minimum is 8.2e-4 (SciFact) and 8.0e-4 (NFCorpus), against the uniform
  9.8e-4. `t̄` is zero outside the top 921 coordinates and peaked inside them. The two KL losses
  therefore have very different scales, so the learning rate is tuned per arm on equal grids.
- KL(t ‖ p) and CE(t, p) differ by the entropy of `t`, which does not depend on the student. They
  are the same loss, so a KL-versus-CE "ablation" is not a loss contrast and is not run. CE is
  linear in its target, so nine per-budget CE terms without a budget input equal one CE on `t̄`
  (ADR-0019). Per-budget supervision under KL therefore needs a budget input, and its control
  needs the same budget input.

## Decision

### Data use

- Corpora: SciFact and NFCorpus (pilot). Both already informed the redesign (ADR-0019), so
  **everything here is exploratory/development**. MS MARCO is not used.
- Training: the F12 fit queries valid for `l2s-official` (565 SciFact, 1,810 NFCorpus), their
  fit labels, and the recorded `l2s-official` negatives (`negatives_f2_rows.npy`). Targets and
  masks come from the F12 `l2s-official` and teacher runs; the study adds no new mining.
- Selection: tune-A, the first half of the tune queries in ADR-0018 hash order (as ADR-0019).
- Reporting: tune-B, the second half. **Tune-B was already observed in ADR-0019** (E0–E4), so it
  is not an independent confirmation. No result here is confirmatory, and none is promised on data
  already seen.
- Never read: F12 evaluation ids/labels/results, official test labels (`qrels-test`), corpus text.
  Nothing here feeds F12.

### Model, input and online cost (identical for every arm unless stated)

`z = W q + b`, `W ∈ R^{1024×1024}`, 1,049,600 parameters. Training uses the raw query; inference
uses the L2-normalized query, then a binary top-k mask on the query only, with no
renormalization. Documents and the index are untouched, and scoring is exact over the full
corpus in float64. Online cost per query: one 1024 × 1024 matrix-vector product (≈ 2.1 MFLOP),
then top-k, then masked scoring. The budget-input arms (KBk, KBa) add `c_k ∈ R^{1024}` for 9
budgets: 9,216 parameters and one vector addition. They are compared only with each other. No arm
uses one model per budget.

### Common optimization (equal effort)

ADR-0019's trainer (the authors' `train_predictor` loop):

- AdamW, weight decay 1e-4, batch 32, DataLoader shuffling, cosine schedule per epoch.
- Published epochs: SciFact 200, NFCorpus 75. The final epoch is kept, with no early stopping on
  tune.
- Every arm sees every fit query once per epoch, with the same optimizer steps. Every arm whose
  loss involves budgets uses all nine budgets 102…921 in every batch. The ranking arms therefore
  use all nine per batch, not one random budget as in F12's S1.
- Relaxation temperature β: annealed geometrically from 1 to 0.1 over all steps, wherever a relaxed
  top-k appears.
- The only tuned hyperparameter is the learning rate, on the same grid for every trained arm:
  **lr ∈ {3e-5, 1e-4, 3e-4, 1e-3}**. It is selected at seed 42 by tune-A mean nDCG@10 over the
  nine budgets, with ties going to the smaller lr. Seeds 0 and 1 are then trained at the selected
  lr. Seeds are paired across arms: the same seed gives the same initialization and the same batch
  order.
- Ranking score temperature: fixed at 0.05, the `RetrievalConfig` default, set before any of these
  data. It is not tuned, so the ranking arm's grid has the same size as every other arm's.
- Parity: the trainer with the KL loss on `t_off`, lr 1e-4, seed 42 must reproduce the F12
  `l2s-official` checkpoint. Bit equality is recorded. On the new device, a non-bitwise result is
  reported with its largest difference and does not stop the study.

### Arms

| Arm | Loss | Target / encoding | Domain and scale of the target | Init |
|---|---|---|---|---|
| REF | — | F12 `l2s-official`, unchanged (seed 42) | — | — |
| KG | KL(t ‖ softmax z) | graded `t_off` | distribution on D, sum 1, flat (min ≈ 0.8/D) | seeded |
| KS | KL(t ‖ softmax z) | rank-step `t̄` | distribution on D, sum 1, zero outside top-921 | seeded |
| MS | relaxed membership BCE | binary `m_k`, 9 budgets | `{0,1}^D` per k, sum k | seeded |
| MG | relaxed membership BCE | graded `y_k = σ((ŝ − θ_k(ŝ))/β)`, ŝ = standardized `log t_off` | `[0,1]^D` per k, sum ≈ k | seeded |
| KBk | mean_k KL(m_k/k ‖ softmax(z + c_k)) | per-budget `m_k/k` | 9 distributions, uniform on top-k | seeded, c_k = 0 |
| KBa | mean_k KL(t̄ ‖ softmax(z + c_k)) | `t̄` for every k | as KS | seeded, c_k = 0 |
| R0 | masked retrieval CE, mean over 9 budgets | relevance labels + recorded negatives + in-batch positives | per query: multi-positive softmax over candidates, score temperature 0.05 | seeded |
| RI | as R0 | as R0 | as R0 | `l2s-official` |
| KGI | as KG | as KG | as KG | `l2s-official` |

Notes:

- KG at lr 1e-4 is exactly the L2S recipe. Its seed 42 must equal REF; seeds 0 and 1 give REF's
  seed spread. KG with the lr grid is the **comparably tuned L2S control**. REF stays intact.
- Relaxed membership BCE is ADR-0019's E2 loss:
  - The student is ẑ, the standardized z. Its relaxed membership is `σ((ẑ − θ_k(ẑ))/β)`, where
    θ_k lies midway between the k-th and (k+1)-th of ẑ and carries no gradient.
  - The loss is the mean over k of `BCE(relaxed_k, label_k)`, averaged over dimensions and the
    batch.
  - Labels are per-dimension Bernoulli memberships in [0, 1]. **No normalized probability
    distribution is ever fed to BCE.**
- MG's label is the same relaxation applied to the teacher's own score at the same step's β.
  Standardizing `log t_off` gives exactly the standardized `s`, so τ drops out. As β → 0 the label
  becomes `m_k`. MG therefore keeps graded information (margins in standard deviations around
  each threshold) inside the membership domain.
  - This is a target transformation, recorded as a factor: graded information enters KG as
    softmax(s/τ) and MG as relaxed rank margins.
  - That is the closest graded analogue available in the membership domain. The limitation is
    declared rather than forcing an identical "graded" in both domains.
- KS and MS are the information-equivalent pair: nested masks ⇔ `t̄`. Their difference is the
  loss (and the domain the loss requires), not the information.
- R0 and RI:
  - Ranking is trained directly through a differentiable surrogate: softmax cross-entropy of the
    masked query's scores over the query's positives (gain-weighted), its recorded `l2s-official`
    negatives, and the other batch queries' positives.
  - The mask is the relaxed top-k (`soft_top_k`), so nDCG is not optimized directly. The function
    is `masked_retrieval_loss`, as in S1.
  - The information is the same labels and the same negative pool from which L2S builds p̄ and n̄.
    In-batch positives are extra fit-only information and are declared.
  - Unjudged documents are never used as negatives beyond the recorded L2S pool, which F12 drew as
    in that protocol. Judged negatives stay distinct from unjudged documents, because only the
    recorded pool is used.
  - Target and loss cannot be separated for the ranking objective. R0 − KG is a package contrast:
    direct ranking versus distillation from the same labels.
- RI and KGI both start at `l2s-official` and get the same additional epochs, so both receive twice
  the scratch arms' exposure. They are compared with each other (objective given init) and with
  R0 (init given objective), and never with scratch arms as if exposure were equal.

### Contrasts (declared before any result)

The primary quantity is the tune-B paired difference in mean nDCG@10 over the nine budgets.
Per query, each arm's value is first averaged over the three paired seeds.

| ID | Contrast | Holds fixed | Varies |
|---|---|---|---|
| C1 | KS − KG | KL loss, architecture, effort | target: rank-step vs graded |
| C2 | MS − MG | membership loss, architecture, effort | target: binary vs graded (relaxed) |
| C3 | MS − KS | information (nested ⇔ `t̄`), architecture | loss: membership vs KL |
| C4 | MG − KG | graded information, architecture | loss: membership vs KL (plus the declared transformation) |
| C6 | KBk − KBa | KL, budget offsets (same capacity) | per-budget targets vs aggregated |
| C7 | KBa − KS | target `t̄`, KL | +9,216 budget-offset parameters |
| C8 | R0 − KG | labels, negatives, architecture, effort | direct ranking objective vs tuned distillation |
| C9 | RI − KGI | L2S initialization, extra effort | objective: ranking vs KL on `t_off` |
| C10 | RI − R0 | ranking objective | initialization: L2S vs seeded |
| C11 | KG − REF | KL on `t_off` | lr tuning on the equal grid |

C5 = C2 − C1, the target × loss interaction, is reported without a test. The ranking table lists
every arm minus REF descriptively.

Uncertainty and multiplicity:

- Paired bootstrap over tune-B queries, B = 10,000.
- Pointwise 95% intervals, plus Bonferroni intervals at α = 0.05/10 for the ten contrasts of each
  corpus.
- Two-sided sign-flip p-values (B = 10,000) with Holm over the ten contrasts of each corpus.
- Seed spread of a contrast: the range over seeds s ∈ {42, 0, 1} of the same-seed difference.

**Reading rule (exploratory):**

- **Supported:** the sign agrees on both corpora, |D| exceeds the seed spread on both, and the
  Bonferroni interval excludes zero on at least one corpus.
- **Mixed:** the signs disagree.
- **Not separated:** anything else.

No contrast is confirmatory. A supported contrast names a hypothesis for a separate protocol on
untouched data.

Secondary (descriptive):

- Per-budget nDCG@10 curves; MRR@10 and Recall@100 averaged over the nine budgets.
- Budget 1024 as parity: every arm keeps the full query, so its metrics must equal the full-query
  score exactly.
- Diagnostics, not win criteria, on fit queries (in sample):
  - Jaccard of the student top-k with `m_k`;
  - KL(`t_off` ‖ softmax z);
  - nDCG@10 regret against the decoded contrastive mask.

Cost:

- Offline: training seconds per run, number of runs, and the size of the selection grid. These
  are reported per arm, together with the device.
- Online: parameters and FLOPs per query, as above.

### Budget

- Per corpus: 9 trained arms × 4 lr at seed 42, plus 9 × 2 extra seeds, giving 54 trainings.
  Each takes seconds to minutes on one GPU.
- Exact tune scoring: 64 checkpoints × tune queries × 10 budgets.
- Device: the second RTX 4090, `ceia40902` (host alias `diario-gpu`). The first host's GPU is in
  use by another user's job and gets nothing new. Host, GPU UUID and device are recorded in every
  manifest's run record.

### Files

- `src/hyperdime/evaluation/objective_study.py`: graded membership label, membership loss with a
  label tensor, ranking loss over the nine budgets, arm registry, and the contrast statistics.
- `src/hyperdime/cli/p3_objective.py`: `hyperdime p3-objective train|score|select|report`.
- `tests/unit/evaluation/test_objective_study.py`: hand-computable math, nested ⇔ `t̄`
  invertibility, KL–CE constant, the label limit β → 0, the ranking loss at one budget equal to
  `masked_retrieval_loss`, and the contrast statistics.
- `reports/P3-objective/`: generated README, RUNS.md, freeze records. A registry row.

## Consequences

- The F12 family and ADR-0018 documents are not touched.
- ADR-0019's evidence wording is corrected (rank-guided decoding; E2 confounds target and loss).
- Every result is development evidence on corpora already used for design.

## Amendment A1 (post hoc; recorded after the primary results, before running it)

The primary selection put KS, KBk and KBa (both corpora), R0 (both) and MS (SciFact) at lr 1e-3,
the top of the grid. KS's tune-A score was still rising steeply there (SciFact 0.8145 → 0.8299
from 3e-4 to 1e-3; NFCorpus 0.3793 → 0.3879). Its primary deficit may therefore partly reflect
under-optimization rather than the target.

- Sensitivity: every trained arm gets lr ∈ {3e-3, 1e-2} at seed 42. That extends all grids
  equally to six points.
- Selection: on tune-A only, by the same rule as the primary run.
- Arms whose selected lr changes get seeds 0 and 1 at the new lr. The ten contrasts are
  recomputed on tune-B.
- These results are labeled **post hoc**. They never replace the pre-registered table, and they
  are read only as a check on whether the primary pattern survives a wider grid.
- Report rendering fix (no number changes): the contrast "meaning" strings no longer contain the
  table separator `|`.

## Evidence

All results are exploratory, from development data, and labeled as such. Run IDs, coverage, cost
and the full reading are in [reports/P3-objective/RUNS.md](../../reports/P3-objective/RUNS.md); the
tables are generated in [README.md](../../reports/P3-objective/README.md) (pre-registered grid) and
[post-hoc-A1/README.md](../../reports/P3-objective/post-hoc-A1/README.md).

- Checks passed:
  - the trainer reproduces `l2s-official` bit for bit on the second GPU (both corpora);
  - every arm equals REF at k = 1024;
  - the masks are nested.
- Pre-registered grid:
  - Only C11 (tuning L2S's learning rate) is supported: +0.014 on SciFact, +0.004 on NFCorpus.
  - C1 (target under KL) is mixed: a large SciFact deficit for KS, none on NFCorpus.
  - Every other contrast is mixed or not separated.
- The primary selection put the KL-on-`t̄` arms at the top of the grid. Under A1 (both grids
  extended equally, post hoc):
  - The SciFact KS deficit mostly vanishes (C1 −0.033 → −0.007; C3 +0.031 → +0.005).
  - No C1–C10 contrast is supported on either corpus.
  - The KL-on-`t̄` arms still select A1's top rate.
- Reading:
  - No target effect and no loss effect is identified once each arm gets its learning rate.
  - Neither per-budget supervision, budget-input capacity, direct ranking training, nor
    initialization separates from the tuned L2S recipe.
  - The gap ADR-0019 attributed to the averaged target is, on this evidence, mostly an
    optimization-scale effect of KL on a step target.
  - The official L2S at its published rate is under-tuned on these splits.
- Nothing is confirmatory. A confirmation would need untouched queries and a bracketing grid fixed
  in advance.
