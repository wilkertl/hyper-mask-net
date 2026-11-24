# ADR-0018: P3 selector benchmark protocol (F12)

- **Status:** proposed. Drafted jointly with the researcher on 2026-11-12, and all design decisions
  are closed. The values realized by the fit-only probe are recorded under "Freeze gates" before any
  held-out outcome is opened.
- **Date:** 2026-11-12
- **Issue:** none assigned

## Context

The exploratory P3 stages (ADR-0014 to ADR-0017) built masks from the very queries they scored, and
their L2S comparisons were in-sample, so they say nothing about unseen queries. Feature F12
([specs/013-p3-selector-benchmark](../../specs/013-p3-selector-benchmark/spec.md)) defines a
query-disjoint in-domain benchmark for developing the project's own selectors. This ADR records the
protocol the researcher and the agent agreed on, so that it is fixed before any evaluation query is
scored. The benchmark keeps exactly one official Learning2Select reference per corpus; every other L2S
training is deferred (see "Reference"). Background and source checks are in
the feature's [research.md](../../specs/013-p3-selector-benchmark/research.md).

## Decision

### Split and scope

- Queries come only from `artifacts/p3-query-split-v1/manifest.json` (seed `20261112`; NFKC,
  casefold, and whitespace normalization; groups of identical normalized text kept whole; F2 train
  and validation ids pooled; official test overlap removed; 70/15/15 fit/tune/evaluation). Its ids
  are kept as they are. It is not rematerialized; a run manifest with the verification record is
  attached to it (research B1).
- The official test split is reserved and never read. TREC DL-19, DL-20, and DL-HARD are outside
  this benchmark; they are not overlap-checked against the split and are not used.
- Queries are never stratified, filtered, weighted, gated, or used to choose an endpoint by positive
  count. Each evaluation query with at least one positive judgment counts once; all judgments of a
  query stay with it.
- Evaluation ids and labels never reach negative mining, mask generation, target generation,
  fitting, feature fitting, tuning, or caps. Tune labels enter only the predeclared selection steps
  below.

### Evaluation, shared by every arm

Exact float64 scoring over the full corpus, frozen Qwen3-Embedding-0.6B vectors, fixed document
vectors, no query renormalization, one lower-row tie rule, project deciles
`k ∈ {102, 204, 307, 409, 512, 614, 716, 819, 921, 1024}`. At 100% every arm must give the full
query's ranking exactly; this is a parity check, not an outcome. The authors' rounded-up grid
(103, 205, …) is used only in the parity test against their evaluator.

### Reference: `l2s-official`

The authors' Learning2Select at commit `55f010f5f3e82a6d171ce0909793e72f9c066ec7`, called through an
adapter, trained on fit queries only. It keeps the original recipe (unmasked top 1000 before
removing positives, 64 negatives with replacement and the authors' fallback, graded target, KL,
AdamW 1e-4 with weight decay 1e-4, batch 32, cosine schedule), the published temperature and epochs
(SciFact 0.05/200, NFCorpus 0.05/75, MS MARCO 0.10/50), final-epoch weights, and seed 42. The authors'
script seeds once, before its loop over datasets, so each corpus runs in its own process. MS MARCO's
parameters were published for the authors' 50k-passage subset. They are applied unchanged to the full
corpus, and every table says so. Published epochs were set for the authors' full training sets. Here
every arm trains on the smaller fit split; this limit is recorded.

**Exactly one `l2s-official` per corpus** is trained in this benchmark. It uses the same fit query
ids as the primary P3 arms: the whole fit split on SciFact and NFCorpus, and, if MS MARCO's fit
coverage is bounded (resource rule), the same bounded prefix. No full-fit reference beside a bounded
one, no other seeds, and no temperature or other retraining sensitivity are part of this benchmark.
They are deferred to a future study of training scale or robustness. S1-official-init reuses this
checkpoint and does not train another L2S.

### Teachers (stage 2, fit queries only)

For each fit query and each of the nine budgets 102 to 921:

1. **positive**: top-k of `q ⊙ p̄`, with `p̄` the gain-weighted mean of the positives
   (weights `2^g − 1`);
2. **contrastive**: top-k of `q ⊙ (p̄ − n̄)`, with `n̄` the mean of the 64 negatives that
   `l2s-official` drew for that query (recorded indices, not re-drawn);
3. **rank-guided**: bounded local search started separately from mask 1 and from mask 2. A move
   swaps one selected dimension for one unselected one, so k stays fixed. A swap is accepted only if
   it is strictly better under the lexicographic objective `(nDCG@10, GDR)`. Each query's
   rank-guided mask is the better of the two search results under the same objective; exact ties go
   to the search started from the positive mask. Both searches are reported.

**Gain-weighted positive discounted rank (GDR)**, higher is better:
`GDR(q, m) = Σ_{p ∈ P(q)} w_p / log2(1 + r_p(m))`, with `w_p = (2^{g_p} − 1) / Σ_{p'} (2^{g_{p'}} − 1)`
and `r_p(m)` the 1-based full-corpus rank of positive p under mask m (lower-row tie rule). nDCG@10
uses the same fit judgments.

- Reported values and final masks are always exact over the full corpus. On SciFact and NFCorpus,
  swaps are scored exactly against the full corpus. On larger corpora, a candidate shortlist may
  propose swaps, but each swap is accepted only after an exact full-corpus check, and the final mask
  is verified exactly. The search can never end below its starting mask, and this is checked.
- Step, time, and shortlist caps are fixed per corpus by the resource rule below, from a cost probe
  on fit queries only, before any teacher is generated for the full fit split. They are never tuned on
  teacher quality or on tune or evaluation data. No feasibility for a corpus is claimed before its
  probe.
- Brute force on enumerable synthetic cases checks the search. No real-case mask is called optimal.
- Reports separate gains from demoting unjudged documents and from demoting known negatives.
  Unjudged documents are never called negatives.

### Selectors (stage 3)

- **Primary teacher arms** (three: positive, contrastive, rank-guided) use exactly the architecture,
  loss, optimizer, schedule, epochs, final-epoch checkpoint, and seed of `l2s-official`. Only the
  target changes: `t_q = (1/9) Σ_{k ∈ {102,…,921}} m_{q,k} / k`, the average over the nine budgets of
  the uniform distribution on each budget's mask. Teacher arms are not tuned. Budget-conditioned
  variants are deferred to a later secondary.
- **Seed and RNG matching.** `l2s-official` and each teacher arm run at seed 42 in their own
  process, on the same ordered set of fit queries, so that weight initialization and shuffle order
  match where feasible. In the authors' code the order is seed, label draws, model initialization,
  then each epoch's shuffle. On CUDA the label draws use the CUDA generator, so the CPU stream at
  initialization is untouched and teacher arms match by construction. Otherwise the CPU (and CUDA)
  RNG state right after the official label generation is saved and replayed in each teacher arm
  before initialization. The official state is never altered. A control check runs first: the
  teacher pipeline fed `l2s-official`'s own targets must reproduce its weights exactly. Device,
  torch version, and determinism flags are recorded.
- **Direct-retrieval controls (S1).** ADR-0009's objective in multi-budget mode:
  - soft top-k membership `sigmoid((z − t)/β)`, with β annealed geometrically from 1 to `β_end`;
  - gain-weighted multi-positive cross-entropy over the positives, 64 hard negatives, and the
    batch's distinct positives, at score temperature `τ_s`;
  - one budget drawn uniformly per batch from the nine.

  Two arms: **S1-scratch** (default `nn.Linear` initialization) and **S1-official-init** (starting
  from the existing `l2s-official` checkpoint of the same corpus and fit ids, reused, not
  retrained). Both use the same eight cells, `lr ∈ {1e-4, 1e-3}` ×
  `τ_s ∈ {0.02, 0.05}` × `β_end ∈ {0.02, 0.1}`, trained at seed 42 with ADR-0009 D6's budget:
  AdamW, weight decay 1e-4, batch 32, cosine decay over 20,000 steps.
  - The hard negatives are the recorded `l2s-official` negatives of each fit query, the same data
    the teachers use.
  - Checkpoints at step 0 and every 1,000 steps are scored on tune queries by pool-rerank nDCG@10
    over the nine budgets. The pool is the query's positives plus its top 1,000 by unmasked score.
    The best checkpoint is kept, and step 0 is eligible. For S1-official-init, step 0 is
    `l2s-official` itself.
  - ADR-0009's soft-mass guard (≥ 0.9) is computed on tune queries.
  - Cells that pass the guard are ranked by full-corpus tune mean nDCG@10 over the nine deciles.
    Ties go to the earlier cell in the order lr, then `τ_s`, then `β_end`, each ascending.
  - The selected cell of each arm is retrained at seeds 0–2 as a secondary check. S1-official-init
    starts every seed from the same seed-42 `l2s-official` checkpoint.
  - **Limitation.** Each S1 arm is selected from 8 cells × 21 checkpoints on tune, while
    `l2s-official` and the teacher arms are not tuned. Reports give each arm's training compute and
    number of selection candidates, and S1 comparisons, which are secondary, are read with this
    limitation stated. No L2S arm is added to offset it.
- **Other controls**: frozen full; **global mask** (per budget, the k dimensions selected most often
  by rank-guided masks across fit queries, ties to the lower dimension index); `variance`
  (`q_d² · Var(x_d)` with the variance over corpus documents; label-free, a cheap separate control).
- Seeds 0, 1, 2 for the three teacher arms and for each S1 arm's selected cell are a secondary
  robustness check. `l2s-official` is trained once, at seed 42; it has no additional seeds.

### Endpoint and statistics

- **Primary endpoint**, per corpus:
  - `D = mean_q d_q` over eligible evaluation queries, each weighted equally;
  - `d_q = ȳ_q(teacher arm) − ȳ_q(l2s-official)`, where `ȳ_q` is query q's mean nDCG@10 over the nine
    deciles 102 to 921, at seed 42.

  Per-decile results, other metrics, and mask or score agreement are secondary.
- **Test** (one-sided; H1: `E[d] > 0`): paired sign-flip randomization with `B = 100,000` flips.
  - `p = (1 + #{b : T*_b ≥ T_obs}) / (B + 1)`, where `T = mean d_q` and ties count as ≥.
  - If `2^n ≤ 100,000`, the test is exact over all `2^n` sign vectors instead, with
    `p = #{T* ≥ T_obs} / 2^n`.
- **Intervals**: paired percentile bootstrap that resamples queries with replacement, `B = 100,000`.
  - A pointwise two-sided 95% interval (2.5th and 97.5th percentiles), labeled pointwise.
  - A one-sided Bonferroni lower bound: the `α/m` quantile, with `α = 0.05` and `m` the size of the
    locked family.
- **Random streams**: NumPy `PCG64` from `SeedSequence(20261112)`, with separate child streams for
  bootstrap and sign flips. Each child is keyed by (procedure, corpus, arm) through a fixed code table
  stored with the results. Draws are made in batches of at most 10^7 elements, with no n × B matrix.
- **Uncertainty** is reported separately for queries and for training seeds. Across queries: the
  test and intervals above, at seed 42. Across training seeds: the spread (range and SD) of
  `D_s = mean_q [ȳ_q(teacher arm, seed s) − ȳ_q(l2s-official, seed 42)]` for `s ∈ {0, 1, 2}`, with no
  test. This shows teacher-side training variability only; reference-side seed variability is left to
  the deferred robustness study.
- **Multiplicity**: Holm at α = 0.05 over the one-sided p-values of the locked family. The family is
  every primary teacher arm × every eligible corpus:
  - if SciFact counts, 3 × 3;
  - if SciFact becomes development, 3 × 2, re-locked before NFCorpus or MS MARCO evaluation is
    opened.

  A corpus or arm removed under the resource escalation rule leaves the family before any
  evaluation is opened.
- **Claim levels**:
  - *superior*: `D > 0` with Holm-adjusted p < 0.05;
  - *practical candidate*: superior, and `D ≥ 0.005` (absolute nDCG@10);
  - *established minimum gain*: Bonferroni lower bound ≥ 0.005. Only this level supports a claim of
    a minimum gain.
- No power or precision is claimed unless it comes from estimates on fit or tune data.

### Resource rule (decided now; values realized after the fit-only probe)

- **Query order.** Within each corpus and split, ids are ordered by
  `SHA-256("p3-bench-v1|20261112|<corpus>|<split>|<id>")` ascending. This is uniform and independent
  of positives. Any bound n takes the first n ids, so smaller bounds are prefixes of larger ones, and
  every arm uses the same ids.
- **Machine.** The existing shared RTX 4090 host only. Jobs start only under the project rule (at
  least 20 GB of GPU memory free over repeated idle checks), one F12 job at a time, and every job is
  resumable. Each job is capped at 20 GB of GPU memory, 64 GB of RAM, and 12 h wall time.
- **Budgets** (GPU wall time, everything included):

  | Corpus | Probe | All of F12 |
  |---|---|---|
  | SciFact | ≤ 1 h | ≤ 8 h |
  | NFCorpus | ≤ 1 h | ≤ 16 h |
  | MS MARCO | ≤ 3 h | ≤ 72 h |

- **Probe** (fit queries and fit labels only; no tune or evaluation id is read). Per corpus:
  1. `l2s-official` label generation on the first n fit ids, with n ∈ {256, 1,024, 4,096} capped at
     the fit size: seconds per query, peak RSS, peak GPU memory.
  2. Exact full-corpus masked scoring of fit queries in batches at the ten deciles: seconds per
     query-mask. This times the evaluator. Tune and evaluation costs are projected from this number
     alone.
  3. Rank-guided search on the first n fit ids, n ∈ {16, 64, 256}, all nine budgets, both starts,
     step caps s ∈ {8, 32, 128}. On MS MARCO also shortlist N ∈ {1,000, 10,000}. Recorded: steps
     until no improving swap, seconds per step, and the rejection rate of exact acceptance.
  4. Training time for one L2S-recipe arm and for 1,000 S1 steps with checkpoint scoring.
- **Size selection.** Projected cost = probe unit costs × sizes + fixed costs, times a safety factor
  of 1.5.
  - SciFact and NFCorpus use every query. The step cap is the smallest s at which, on the probe, at
    least 95% of searches stop without hitting it, provided the budget allows; otherwise the largest
    s that fits.
  - On MS MARCO, sizes are chosen in this priority order, each the largest that keeps the projection
    within budget:
    1. evaluation `n_eval ∈ {7,500, 4,000, 2,000}`;
    2. rank-guided fit `n_rg ∈ {35,000, 16,000, 8,000, 4,000}`;
    3. tune `n_tune ∈ {7,500, 4,000, 2,000}` (S1 selection);
    4. then s and N by the same 95% rule.
  - The minimum interpretable coverage is `n_eval = 2,000`, `n_rg = 4,000`, and `n_tune = 2,000`.
- **Matched fit coverage.** If `n_rg` is smaller than the fit split, everything trained for that
  corpus uses the same first `n_rg` fit ids: the single `l2s-official` (trained only on that prefix,
  with the unchanged recipe), the three teacher arms, the global mask, and both S1 arms. Claims for
  that corpus state "fit coverage `n_rg`". No full-fit model is trained beside it; a full-fit
  comparison belongs to the deferred training-scale study.
- **Escalation.** If even the minimum coverage is projected over budget, nothing on that corpus is
  evaluated. The probe result is recorded, and the researcher chooses one of three options before any
  evaluation:
  - a larger budget;
  - removing the rank-guided arm for that corpus from the locked family;
  - reporting the corpus as not run.

  The search keeps exact full-corpus acceptance and exact final values under every cap.
- **Planning evidence, not probe results** (from RTX 4090 runs; the probe replaces it):
  - MS MARCO full-corpus scoring at about 18 ms per mask: 148,607 masks in 2,727 s, peak RSS about
    27 GB (`reports/P3-mask-ranking/README.md`, cost table).
  - About 19 ms per mask in the MS MARCO oracle pilot
    (`reports/P3-oracle-l2s-msmarco-pilot/RUNS.md`).
  - Projections at that rate:
    - full MS MARCO evaluation at seed 42 (8 masked arms × 9 deciles plus the full query, 73 masks
      per query × 7,500 queries) ≈ 2.7 h;
    - secondary seeds 0–2 (3 teacher arms and 2 S1 arms × 9 deciles) ≈ 5 h;
    - S1 tune scoring at `n_tune = 7,500` (2 arms × 8 cells × 9 deciles) ≈ 5.4 h;
    - L2S label generation, one process per corpus ≈ 10 min;
    - rank-guided search: up to 9 budgets × 2 starts × s exact acceptance scans per query, so at
      s = 32 at most about 10 s per query. That is ≈ 100 h for all 35,000 fit queries and ≈ 11 h for
      4,000.

    A bounded `n_rg` on MS MARCO is therefore likely. SciFact (5,183 documents) and NFCorpus (3,633)
    are small enough that their full-corpus scans cost little.

### Freeze gates

1. **Protocol gate.** This ADR, with all rules above, is recorded with a SHA-256 fingerprint before the probe. The probe may
   read fit queries and fit labels only.
2. **Resource gate.** The probe's measured unit costs and the realized `n_eval`, `n_rg`, `n_tune`, s,
   and N, with the id prefixes they imply, are written into `reports/P3-bench/RUNS.md` and into the
   table below, and fingerprinted. This happens before any teacher is generated beyond the probe, before
   any tune metric is computed, and before any evaluation label is read.
3. **Selection.** S1 selection reads tune labels only.
4. **Evaluation lock.** Before a corpus's evaluation labels are read, a fingerprinted record lists every trained
   arm's checkpoint hash, the selected cells, and the evaluation id list. The evaluator then scores
   every arm in one pass, once per corpus.

Tune metrics are selection data. A held-out outcome is any quantity computed with evaluation labels.

| Corpus | n_eval | n_tune | n_rg | step cap s | shortlist N | probe run |
|---|---|---|---|---|---|---|
| SciFact | all (121) | all (121) | all (565) | after probe | none | — |
| NFCorpus | all (388) | all (388) | all (1,810) | after probe | none | — |
| MS MARCO | after probe | after probe | after probe | after probe | after probe | — |

### Confirmation scope and replication

- SciFact runs first. It counts toward confirmation only if this protocol, with SciFact's realized
  values, was recorded and fingerprinted before its evaluation was opened and nothing changed afterwards.
- Any design change after SciFact evaluation results are seen makes SciFact development data. The
  same frozen candidate is then tested independently on NFCorpus and MS MARCO.
- **Replication**: the same prespecified teacher arm, on the same primary endpoint, is *superior*
  under the locked Holm family in at least two eligible corpora. If SciFact is development, it must
  be superior in both NFCorpus and MS MARCO. Otherwise only corpus-specific evidence is claimed.
  Practical-gain levels are reported per corpus in every case.
- Wording: results are a query-disjoint in-domain comparison against Learning2Select retrained under
  this protocol. They are not a comparison with published numbers and not a state-of-the-art claim.
  No hypernetwork work is part of F12.

## Evidence

Protocol only; no F12 result exists. Source checks of the authors' code and of the split manifest are
in research B1 and B3. The cost planning evidence is cited above from existing run records.

## Consequences

- F3 checkpoints and every earlier P3 result stay legacy evidence and are not re-scored.
- SciFact's 121 evaluation queries may not resolve a 0.005 difference. The protocol accepts that,
  rather than adapting after looking.
- Matched seeds make teacher arms and `l2s-official` differ only in their targets. This holds only if
  the replay control check passes.
- If MS MARCO's `n_rg` is bounded, its single `l2s-official` is trained on the same prefix, and MS
  MARCO claims are scoped to that fit coverage.
- S1's larger selection budget limits how its secondary comparison can be read. It is reported, not
  offset by another L2S arm.
- Deferred to a future study of training scale or robustness: a full-fit L2S reference beside a
  bounded one, additional L2S seeds, tuned-temperature L2S, and any other L2S retraining.
