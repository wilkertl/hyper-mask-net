# P3-objective run record (ADR-0020; exploratory)

Protocol: [ADR-0020](../../docs/decisions/ADR-0020-p3-objective-study.md). Its SHA-256 and the code
snapshot were recorded in [freeze/protocol.json](freeze/protocol.json) before any training or score.
Tables: the generated [README.md](README.md) (pre-registered grid) and
[post-hoc-A1/README.md](post-hoc-A1/README.md) (amendment A1, post hoc). No number below was
typed by hand; each is copied from those generated files.

## Compute

- Host: `ceia40902` (alias `diario-gpu`). GPU: NVIDIA GeForce RTX 4090,
  UUID `GPU-7ba81dde-05d0-306c-2880-e259c0add8b2`, device `cuda:0`.
  - Before launch: 0 compute processes, 9 MiB used.
  - The UUID is recorded in every training manifest (`config.compute`).
- The first host (`ceia40901`, GPU-8a7933bc…) was not used. Another user's job held its GPU.
- Environment: `uv sync --frozen --extra cu128` into the project folder (torch 2.11.0+cu128). The
  authors' clone `learning2select` at 55f010f5 needed `tqdm==4.70.1`, the version on the first
  host.
- Inputs were copied from `ceia40901:~/p3-bench/artifacts` and verified by SHA-256 on arrival (44
  files):
  - the query split;
  - the F12 `l2s-official` runs (SciFact 20261112T214624Z-7ec05061, NFCorpus
    20261114T022621Z-669b02f6);
  - the teacher runs (SciFact 20261112T223656Z-07608c76, NFCorpus 20261114T022810Z-24f79ec6);
  - `qrels-validation`.
- The embeddings, `qrels-train` and `documents.json` already on the second host match the first
  host's hashes. No corpus text and no `qrels-test` were copied. Nothing was downloaded again.
- Code snapshots (unreferenced commits; bundles in `artifacts/p3-spike/code-snapshots/`):
  - primary `5963306c4c81`;
  - A1 `4f62b6540828`, which adds the post-hoc rates, `--extended`, `--label` and the label text.
    The training math is unchanged.
- Drivers: [freeze/driver.py](freeze/driver.py) and [freeze/driver_a1.py](freeze/driver_a1.py),
  with log [freeze/driver.log](freeze/driver.log).
  - Both hold a claim directory and refuse a busy GPU.
  - They are resumable through `ledger-<corpus>.json`. A1 reused the primary ledger, so the primary
    runs were not retrained.
- Smoke runs (scratch root `~/…/smoke`, not results, never read for metrics): 4 trainings and 1
  score run on SciFact. The GPU-index regression check reproduced the MS MARCO S1 crash on the
  unfixed snapshot.

## Checks

- Trainer parity: KG at lr 1e-4, seed 42 equals the F12 `l2s-official` checkpoint bit for bit on
  this GPU, on both corpora (`max_abs_difference` 0.0).
- Parity budget k = 1024: every arm equals REF exactly (largest gap 0.00e+00, both corpora).
- The contrastive masks are nested (0 violations), so KS and MS carry the same information.
- Local regeneration of the primary README from the copied run records reproduces
  `results.json` exactly. Only the contrast labels changed (they had contained the table
  separator).

## Runs

| corpus | primary score (final) | primary grid score | A1 score (final) | A1 grid score |
|---|---|---|---|---|
| SciFact | 20261116T015507Z-58357188 | 20261116T015039Z-39baa1ea | 20261116T022417Z-dff04909 | 20261116T022317Z-09123756 |
| NFCorpus | 20261116T021352Z-b2b6ee6f | 20261116T020733Z-d62bf1c8 | 20261116T023356Z-f9cbb66f | 20261116T023042Z-9fad02a4 |

Training run IDs: `freeze/trainings-<corpus>.json` (primary) and `freeze/trainings-a1-<corpus>.json`
(all, A1 included). Selections: `freeze/selection-<corpus>.json` and `freeze/selection-a1-<corpus>.json`.
Run manifests and summaries are in `runs/`; checkpoints stay on the host.

## Coverage

- Fit (training): 565 SciFact and 1,810 NFCorpus queries.
- Tune-A (selection) and tune-B (report): 61 tune-B queries on SciFact, 194 on NFCorpus.
- Seeds: 42, 0 and 1, paired across arms.
- Grid sizes: 4 learning rates per trained arm in the primary run; 6 per arm after A1, equal for
  every arm.

## Cost

- Offline training, GPU seconds on the RTX 4090:
  - SciFact: 56 primary trainings, 700 s; plus 26 A1 trainings, 276 s.
  - NFCorpus: 56 primary trainings, 1,010 s; plus 26 A1 trainings, 504 s.
  - The ranking arms are the costly ones: about 40 s (SciFact) and 65 s (NFCorpus) per training,
    against 2–8 s for the others.
- Scoring: 3–59 s per score run.
- Online, per query: identical for all arms. That is one 1024 × 1024 map (1,049,600 parameters,
  ≈ 2.1 MFLOP), top-k, and masked exact scoring. KBk and KBa add 9,216 parameters and one vector
  addition.

## Reading (exploratory; ADR-0020 rule)

Pre-registered grid ([README.md](README.md), tune-B, paired over 3 seeds):

- **C11, tuning L2S's learning rate: supported.**
  - KG (lr 3e-4 on both corpora) − REF (published lr 1e-4) is +0.0136 on SciFact (Bonferroni
    interval above 0) and +0.0043 on NFCorpus.
  - The official reference is under-tuned on these splits. Part of any "beats L2S" result,
    including F12's S1, can be a tuning effect.
- **C1, target under KL (KS − KG): mixed.**
  - SciFact −0.0329 (Bonferroni interval below 0); NFCorpus +0.0013.
- **C3, loss under identical information (MS − KS): not separated.**
  - SciFact +0.0306 (the pointwise interval excludes 0, the Bonferroni interval does not);
    NFCorpus +0.0005.
- **C2, target under membership, and C4, loss under graded targets:** about 0 on both corpora
  (|D| ≤ 0.006; mixed signs).
- **C5, interaction:**
  - SciFact: +0.0365. Only the KL + rank-step cell is low; changing either the target or the loss
    restores it.
  - NFCorpus: −0.0026.
- **C6, per-budget vs aggregated with offsets: mixed and small. C7, offset capacity: about 0.**
- **C8, direct ranking vs tuned L2S: mixed** (+0.0054, −0.0036).
- **C9, objective given L2S init: not separated** (−0.0078, −0.0035).
- **C10, init given ranking: not separated** (−0.0146, −0.0013).

Why A1 was needed: KS, KBk and KBa selected lr 1e-3, the top of the primary grid, on both corpora,
with tune-A still rising (SciFact KS 0.8145 → 0.8299).

Post hoc A1 ([post-hoc-A1/README.md](post-hoc-A1/README.md); recorded in ADR-0020 before running;
every arm's grid extended equally):

- Most of the SciFact KS deficit was a learning-rate effect:
  - KS moves to lr 1e-2, and tune-B rises from 0.7597 to 0.7852.
  - C1 shrinks from −0.0329 to −0.0074 (interval includes 0).
  - C3 shrinks from +0.0306 to +0.0051.
  - C5 falls from +0.0365 to +0.0041.
- On NFCorpus, the KL-on-`t̄` arms (KS, KBk, KBa) become the best arms (≈ +0.010 over REF,
  intervals above 0). No C1–C10 contrast is supported in either corpus. C11 stays supported.
- KS, KBk, KBa (both corpora) and MG (SciFact) still select the new top rate 1e-2, with tune-A
  rising more slowly (SciFact KS 0.8382 → 0.8422). Their values remain possibly under-optimized.

What can and cannot be said:

- Target effect: none survives once each loss gets its learning rate.
  - The rank-step target `t̄` under KL needs roughly 10–100× the learning rate of the graded
    `t_off`.
  - At the pre-registered grid, this scale difference looks like a target effect on SciFact.
- Loss effect: none is identified. Membership BCE and KL reach the same tune-B level under the
  same information (C3) and under graded targets (C4).
- Per-budget supervision and budget-input capacity: no effect (C6, C7).
- Direct ranking training (S1's loss on the same labels and negatives): not separated from tuned
  distillation (C8). F12's S1 advantage over the untuned official L2S may therefore partly be a
  tuning effect like C11; F12 itself is not re-read here.
- Initialization: starting from L2S does not help the ranking objective (C10). From the same L2S
  start, the ranking objective adds nothing over continued KL training (C9).
- Correction of ADR-0019's reading: E1 < E0 and E2's recovery were measured at lr 1e-4 for every
  arm. Under equal-size grids, most of that gap is optimization scale, not the target or the loss.
- Diagnostics (fit, in sample; not win criteria):
  - MS fits the contrastive masks almost exactly (SciFact Jaccard 0.987).
  - KS, at its A1 rate, reaches Jaccard 0.858 with KL(`t_off` ‖ p) 0.33 and still ranks like KG
    on tune.
  - Mask agreement and KL to the L2S target do not predict tune ranking.

## Limits

- Two corpora, both already used for design (development data). Tune-B was already seen in
  ADR-0019 and is small (61 SciFact queries). Three seeds.
- A1 is post hoc, and its edge selections show that even six points do not bracket the optimum
  for the KL-on-`t̄` arms.
- MG's graded labels are a declared transformation (relaxed rank margins), not `t_off` itself.
- The ranking arms use in-batch positives (extra fit-only information). Their score temperature
  0.05 is fixed, not tuned.
- The init arms (RI, KGI) receive twice the exposure of scratch arms by design. They are compared
  only with each other and with R0.
- Nothing here is confirmatory. A test of "no target/loss effect under tuned learning rates" would
  need untouched queries and a bracketing grid fixed in advance.
