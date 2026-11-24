# ADR-0019: P3 target encoding, diagnostic and controlled ablation (exploratory)

- **Status:** accepted (protocol recorded before any diagnostic or pilot run; evidence added after both stages)
- **Date:** 2026-11-14
- **Issue:** none assigned (no ticket; administrative)

## Context

F12 (ADR-0018) found on SciFact that selectors trained on P3 teacher masks, encoded as
`t_q = (1/9) Σ_k m_{q,k}/k`, score below the official Learning2Select (L2S) on held-out queries.
Those results were seen, so anything designed from them is development work. The question here is
mechanistic and is studied separately from the frozen F12 family, which is completed first and is
not reused as tune data.

A difference in held-out nDCG@10 cannot separate three explanations:

- (a) the teacher masks rank the corpus poorly, even on the queries they were built from;
- (b) information is lost when masks become a training distribution (averaging over budgets keeps
  only membership frequency; it drops the graded contrast of L2S's softmax target);
- (c) the linear student cannot reproduce the target, or reproduces it without generalizing.

Two facts frame the design. First, the contrastive mask and the L2S target come from the same
quantity, `(p̄ − n̄) ⊙ q`: the mask is its top-k, the target its softmax at temperature τ (on
L2-normalized vectors). Comparing the two encodings of that one source isolates the encoding.
Second, cross-entropy is linear in its target, so `Σ_k CE(m_k/k, p) = 9 · CE(t_q, p)`: a per-budget
cross-entropy without a budget input trains exactly the averaged target. Avoiding the average needs
either a loss that is not linear in the target or a budget input.

## Decision

### Data use (both stages)

- Corpora: SciFact (development: its F12 evaluation was seen) and NFCorpus. MS MARCO is excluded
  from the pilot to keep its F12 resources intact.
- Training: fit queries and fit labels only, with the recorded `l2s-official` negatives.
- Diagnostics: fit queries (privileged, in-sample) and tune queries (students out of sample).
- Pilot selection and reporting: tune queries split in ADR-0018 hash order into **tune-A** (first
  half, any selection such as λ) and **tune-B** (second half, the reported comparison). Every arm is
  reported on the same tune-B ids.
- Never read: F12 evaluation labels or results beyond what is already published in the F12 run
  record, and the official test labels. Nothing here feeds mining, targets, normalization, or
  selection in F12. Every result is labeled exploratory/development. A confirmation would need
  untouched data and a separate protocol; no holdout is carved out of queries already used.

### Stage D: mechanistic diagnostic (reuses F12 artifacts; no training)

Per corpus, per budget `k ∈ {102, …, 921}`, per query, all exact full-corpus float64 with the
lower-row rule (ADR-0018 evaluator), for each teacher available (positive, contrastive,
rank-guided on SciFact only) and the official target:

| Quantity | Definition | Explains |
|---|---|---|
| teacher nDCG@10 | nDCG@10 under the teacher's own mask `m_k` (fit, privileged) | (a) |
| decoded nDCG@10 | nDCG@10 under top-k of the training target `t_q` (mask average or L2S softmax) | (b) |
| decode agreement | Jaccard(top-k(`t_q`), `m_k`); for L2S, Jaccard(top-k(`t_off`), contrastive `m_k`) | (b) |
| student nDCG@10 | nDCG@10 under top-k of the trained arm's score on fit (in-sample) and tune | (c) |
| student agreement | Jaccard(student top-k, top-k(`t_q`)) | (c) |
| regret | decoded nDCG − student nDCG (fit); teacher nDCG − decoded nDCG | (b), (c) |
| margin | best positive masked score minus best non-positive masked score, per mask | all |
| divergence | KL(`t_q` ‖ softmax(student)) on fit | (c) |
| seed stability | mean pairwise Jaccard of student top-k over seeds 42, 0, 1, 2 | (c) |
| nesting | share of the k-mask inside the next larger mask (teachers) | (b) |
| search effect | share of rank-guided searches that changed the mask, Jaccard to the start, Δ(nDCG@10, GDR) | (a) |

Tune rows use the positive teacher only (tune positives are allowed; contrastive needs the L2S
negatives, which exist only for fit queries). Single- versus multi-positive queries are a
secondary breakdown; the primary aggregate weights every query equally, as in F12. A local search
result is never called optimal. Differences between quantities describe where performance is lost;
they do not identify a cause without the controlled ablation below.

### Stage A: controlled ablation of the target encoding (pilot)

Hypothesis: with the same information source (`(p̄ − n̄) ⊙ q`), the same linear capacity, and the
same optimization budget, the binary mask-average encoding loses ranking-relevant information that
the graded L2S target keeps; a per-budget encoding recovers part of it only if it avoids the
average; a budget input helps only beyond what its extra parameters give.

Common recipe (L2S's): `z = W q + b` on the raw query in training and the L2-normalized query at
inference, as F12; AdamW lr 1e-4, weight decay 1e-4, batch 32, cosine schedule per epoch, published
epochs (SciFact 200, NFCorpus 75), final epoch, seeds 42, 0, 1. One trainer serves every arm; with
the KL loss it must reproduce `l2s-official` bit for bit at seed 42 (checked first). Masks `m_k` are
the F12 contrastive masks (same source as the L2S target). At inference every arm uses only `q`
(and k when the arm is budget-conditioned).

| Arm | Loss (per query, mean over the batch) | Extra parameters |
|---|---|---|
| E0 official | KL(`t_off` ‖ softmax(z)) | none |
| E1 mask average | KL(`t̄`‖ softmax(z)), `t̄ = (1/9) Σ_k m_k/k` | none |
| E2 per-budget membership | `(1/9) Σ_k BCE(σ((ẑ − θ_k)/β), m_k)`, ẑ = z standardized per query, θ_k midway between the k-th and (k+1)-th of ẑ (no gradient), β annealed geometrically 1 → 0.1 over training (ADR-0009's relaxation, no tuning) | none |
| E3 budget input | `(1/9) Σ_k CE(m_k/k, softmax(z + c_k))`; inference at k uses `z + c_k` | `c_k ∈ R^1024`, 9 vectors, zero-initialized |
| E3c capacity control | as E3, but every budget's target is `t̄` | same as E3 |
| E4 hybrid | KL(`t_off` ‖ softmax(z)) + λ · KL(`t̄` ‖ softmax(z)), λ ∈ {0.1, 0.3, 1.0} chosen on tune-A | none |

E0 and E1 at seed 42 equal the F12 `l2s-official` and contrastive-arm checkpoints; they are reused
after the trainer check. Budget: at most 3 seeds × (E0, E1, E2, E3, E3c) + 3 λ × seed 42 + 2 extra
seeds of the selected λ, per corpus: under 25 trainings of seconds to minutes each on one GPU or
CPU, plus exact tune scoring.

Pre-registered reading (exploratory, no confirmatory claim):

- Primary pilot quantity: tune-B mean nDCG@10 over the nine budgets, per arm, paired against E0,
  with a pointwise 95% paired bootstrap interval (B = 10,000) and the seed spread (range over 3).
- E1 < E0 by more than the seed spread on both corpora supports a loss from the mask encoding under
  matched information and capacity.
- E2 or E3 recovering part of E0 − E1 supports per-budget information; E3 − E3c separates it from
  the budget input's capacity.
- E4 is a candidate only if it exceeds E0 by at least 0.005 on tune-B in both corpora, with the
  interval above zero in at least one, and the gap larger than the seed spread. Even then it is a
  development result to be tested under a separate protocol on untouched data.
- Diagnostics on positive disagreement or coordinate magnitude may motivate one later auxiliary
  ablation; none is added now.

### Planned files

- `src/hyperdime/evaluation/target_study.py`: Jaccard, decode, margin, KL, encodings, losses, trainer.
- `src/hyperdime/cli/p3_target.py`: `hyperdime p3-target diagnose|ablate|report`.
- `tests/unit/evaluation/test_target_study.py`: hand-computable math, the trainer's bit-for-bit
  check against the authors' `train_predictor` (when their clone is present), the CE linearity.
- `reports/P3-target/` (generated README, tables, RUNS.md), a registry row, this ADR.

## Evidence

The F12 SciFact run record motivates the question
([reports/P3-bench/RUNS.md](../../reports/P3-bench/RUNS.md)). All results below are exploratory
and come from development data; the run IDs and the reading against each criterion are in
[reports/P3-target/RUNS.md](../../reports/P3-target/RUNS.md), and the tables in the generated
[README](../../reports/P3-target/README.md).

- Stage D (SciFact, NFCorpus):
  - The teacher masks rank their own fit queries well.
  - The top-k of the L2S target equals the contrastive mask.
  - Decoding the mask average back to top-k preserves the positive and contrastive masks exactly
    (both nested). It does not preserve the rank-guided masks (SciFact fit nDCG@10, mean over
    nine budgets: teacher 0.9897, decoded 0.9793), whose nesting is below 1.
  - The gap opens at the student: the mask-trained student falls below the L2S student on fit,
    and again on tune.
- Stage A, against the pre-registered reading:
  - **Met:** E1 < E0 beyond the seed spread on both corpora.
  - **E2 recovers the gap.** Its difference to E0 is small and positive, and most of its
    intervals include zero.
  - **E3 ≈ E3c ≈ E1:** the budget input recovers nothing.
  - **E4 is not a candidate.**
- Reading (corrected 2026-11-15): E2 changes the loss (KL → relaxed-membership BCE) and the
  presentation of the masks (one averaged distribution → nine label vectors) at once. The pilot
  therefore does not identify whether the target or the loss matters. It does not show that the
  averaging alone causes the gap. ADR-0020 separates the two. With each arm's learning rate tuned on equal
  grids (ADR-0020, amendment A1, post hoc), most of the E1 − E0 gap disappears. Every arm here
  used lr 1e-4, which suits the graded target but not the step target under KL.
- No candidate is named for confirmation.

## Consequences

- The F12 frozen family is unaffected; its documents are not edited.
- SciFact is development data for any design that follows.
- A positive pilot result does not confirm anything; it names one candidate for a separate,
  pre-registered test on data not used here.
