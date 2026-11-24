# P3-bench (F12): run record

Protocol: [ADR-0018](../../docs/decisions/ADR-0018-p3-selector-benchmark.md), specs/013. The files
under `runs/` are copies of the host artifacts' manifests and summaries; the values live there and
in the freeze records, and nothing here was edited by hand. No tune or evaluation judgment has been
read, and no held-out outcome exists.

## Where and with what

- Host `ceia40901` (RTX 4090), root `~/p3-bench/artifacts` (symlinks to the F2 `datasets/` and
  `embeddings/`, read only), runner `~/p3-bench/run.sh`.
- Code: unreferenced snapshot `6ce68a0bd2c5e26b74600084fe09f57057cea04f` (bundle in
  `artifacts/p3-bench/code-snapshots/`), on the host venv with torch 2.11.0+cu128.
- The authors' clone at `55f010f5f3e82a6d171ce0909793e72f9c066ec7` in `~/p3-bench/learning2select`.
- GPU rule: before each launch, five checks with at least 20 GB free and no compute process (all
  showed 24,080 MiB free, 0 processes). Duplicate guards are claim directories in
  `~/p3-bench/claims/` (`probe-official-scifact`, `probe-scifact`, `reference-scifact`), kept.

## Gate sequence

| Step | Run | Record |
|---|---|---|
| Split provenance (ids unchanged; every check passed) | `20261112T213306Z-725c0b8a` | `runs/p3-query-split-v1/provenance/…/verification.json` |
| Protocol freeze, before the probe | — | `freeze/protocol.json` (SHA-256 `00f68ec4…`); the probe re-verified its file hashes |
| Probe: official labels, 256 fit ids (labels only) | `20261112T213354Z-0cb0ba5b` | `runs/evaluation/P3-bench/probe-official/` |
| Probe: official labels + training, all 565 fit ids | `20261112T213355Z-56dd9646` | same |
| Probe: scoring, rank-guided search, S1 steps (fit only) | `20261112T213411Z-417072aa` | `runs/evaluation/P3-bench/probe/…/probe.json` |
| Resource freeze, before any tune metric or evaluation label | — | `freeze/resources-scifact.json` (SHA-256 `57cdeec7…`) |
| Single official reference, seed 42, fit ids, final epoch | `20261112T214624Z-7ec05061` | `runs/selectors/l2s-official/…`; `state_dict.pt` SHA-256 `85ca3d3b…` (host) |
| Official-evaluator parity on fit queries | `20261112T214641Z-598e8e33` | `runs/evaluation/P3-bench/official-parity/…/summary.json` |

## What was checked

- **Official recipe parity** (CPU, `tests/unit/baselines/test_l2s_official.py`, locally and on the
  host): the authors' unmodified script and our adapter, in separate seed-42 processes, give
  bit-identical final weights on a fixture with length ties, graded positives, a judged query
  without positives, and pool-fallback queries.
- **Recorded negatives**: in every official run the rebuilt negatives reproduce the authors' target
  of every query bit for bit (the run fails otherwise). SciFact fit has no fallback query.
- **Determinism on the host GPU**: the reference's weights, targets, and negatives are
  byte-identical to the probe's independent full-fit process.
- **Common evaluator vs the authors' evaluator** (fit queries, in-sample, a diagnostic only): equal
  per query at every decile of the authors' grid (103 … 1024), and every arm at 1024 gives the full
  vector's ranks. The two conventions differ (the authors use float32 cosine, `2^g − 1`, and an
  ideal over the retrieved list; the common evaluator uses exact float64, linear gain, and an ideal
  over all positives). They agree here because SciFact's grades are binary and the positives fall
  inside the authors' retrieved list.

## Resource decision (from `freeze/resources-scifact.json`)

Every fit, tune, and evaluation query (565 / 121 / 121), no shortlist, step cap 8: every probe
search stopped before 8 steps, so the 95% rule chose the smallest cap. The projection is within the
8 h SciFact budget.

## Stage 3 (after the resource gate)

Code snapshot `e50f3647765bf6adbef6a479329ac7028cbbdb92`; chain `~/p3-bench/scifact_stage3.sh`.

| Step | Run | Record |
|---|---|---|
| Replay control (official targets, seed 42) | `20261112T223653Z-bfc81ad7` | weights equal `l2s-official` bit for bit |
| Teachers on all 565 fit queries, cap 8 | `20261112T223656Z-07608c76` | `runs/evaluation/P3-bench/teachers/…/summary.json` |
| Teacher arms, 3 teachers × seeds 42, 0, 1, 2 | 12 runs, `freeze/runs-scifact.json` | `runs/selectors/p3-teacher-arms/` |
| S1-scratch, 8 cells, tune selection, seeds 0–2 | `20261112T230604Z-220da458` | `runs/selectors/p3-s1/scifact/scratch/…/selection.json` |
| S1-official-init, same | `20261112T231423Z-833ccb08` | `runs/selectors/p3-s1/scifact/official-init/…/selection.json` |
| Evaluation lock: 22 arm checkpoints, 121 evaluation ids | — | `freeze/evaluation-lock-scifact.json` (SHA-256 `bc4aacca…`) |
| Evaluation, one pass, every arm | `20261112T232241Z-fbc6f7b5` | `runs/evaluation/P3-bench/evaluation/scifact/…/README.md` (generated), `summary.json`, `per_query.csv` |

The evaluation re-verified every checkpoint hash against the lock record before it read evaluation
judgments, and every arm at 1024 reproduced the full vector's ranks. The protocol and gates were
unchanged before evaluation, so SciFact counts toward the locked family under ADR-0018.

## Reading (SciFact; values in the generated README)

- All three primary teacher arms score below `l2s-official` on the primary endpoint, with pointwise
  intervals entirely below zero and one-sided p near 1. However Holm is completed, none can be
  *superior*. Seeds 0–2 give the same picture (teacher-side spread only).
- Order among teachers: contrastive is closest to L2S; positive and rank-guided are far below it.
  The rank-guided search changed few masks (309 of 10,170 searches improved), so it stays close to
  positive. Demotions it made were all of unjudged documents (no judged negatives exist in SciFact
  fit).
- Secondary: both S1 arms score above `l2s-official` (the official-init interval excludes zero,
  scratch does not). S1 was selected from 8 cells × 21 checkpoints on tune, while L2S and the teacher
  arms were not tuned, so that comparison carries the limitation ADR-0018 states. The global mask,
  `variance`, and the full vector are below L2S.
- Likely reason (an interpretation, not a tested claim): a hard-mask target averaged over budgets
  keeps only membership and drops the graded contrast L2S's soft target carries, and masks built from
  the query's own positives say nothing about competing documents.

## NFCorpus

Gate 1 ran on snapshot `e50f3647…` (SciFact's); later steps on `3c1775ac94a2…` (bundle in
`artifacts/p3-bench/code-snapshots/`), which adds only the escalation path, optional controls, and
the family command. The five protocol files are unchanged (the probe re-verified their hashes).

| Step | Run | Record |
|---|---|---|
| Split provenance (every check passed) | `20261113T033930Z-90c4a87f` | `runs/p3-query-split-v1/provenance/nfcorpus/` |
| Probe: official labels, 256 fit ids (labels only) | `20261113T033947Z-24a35c3f` | `runs/evaluation/P3-bench/probe-official/nfcorpus/` |
| Probe: official labels + training, all 1,810 fit ids | `20261113T034004Z-1297a89f` | same |
| Probe: scoring, rank-guided search, S1 steps | `20261113T034023Z-340e4903` | `runs/evaluation/P3-bench/probe/nfcorpus/…/probe.json` |
| Resource freeze: **escalate** (rank-guided over budget at every cap) | — | `freeze/resources-nfcorpus.json` (SHA-256 `0086064c…`) |
| Escalation choice: **remove the rank-guided arm** (and the global-mask control, which is defined from rank-guided masks) | — | `freeze/escalation-nfcorpus.json` (SHA-256 `2becfab7…`) |
| Single official reference, seed 42, all fit ids | `20261114T022621Z-669b02f6` | weights, targets, negatives byte-identical to the probe's full fit |
| Official-evaluator parity (diagnostic) | `20261114T022735Z-b05d6c60` | full-vector parity at 1,024 holds; per-query values differ (below) |
| Replay control (official targets, seed 42) | `20261114T022751Z-0f595cc9` | weights equal `l2s-official` bit for bit |
| Teachers (positive, contrastive) on 1,810 fit queries | `20261114T022810Z-24f79ec6` | `runs/evaluation/P3-bench/teachers/nfcorpus/…/summary.json` |
| Teacher arms, 2 teachers × seeds 42, 0, 1, 2; S1-scratch; S1-official-init | 8 + 2 runs | `freeze/runs-nfcorpus.json` |
| Family members file, m = 8 | — | `freeze/family.json` |
| Evaluation lock: 17 arm files, 388 evaluation ids, family m = 8 | — | `freeze/evaluation-lock-nfcorpus.json` (SHA-256 `e017ac55…`) |
| Evaluation (GPU, 2 job segments; resumed after the first 50-query block) | `20261114T180408Z-f16e9038` | `runs/evaluation/P3-bench/evaluation/nfcorpus/…/README.md` |

**Escalation.** The probe searched one query within its time share; one exact swap scan took about
1.6 s on NFCorpus (many positives per query). The escalation record's lower bound assumes every
search stops after a single scan at the fastest measured scan: even that exceeds the 16 h budget, so
no cap could fit regardless of the probe's small sample. A larger budget is outside the compute
authorized for F12, and "not run" would discard two arms that fit. The rank-guided arm therefore
leaves the NFCorpus family before any NFCorpus evaluation label is read (ADR-0018, "Escalation"
and "Multiplicity").

**Parity.** NFCorpus fit judgments are binary, like SciFact's, but many queries have positives
outside the authors' top-1,000 retrieved list, where the authors' nDCG@10 takes its ideal; the common
evaluator takes it over all positives. `scripts/checks/p3_bench_parity_explain.py` (output
`{"both": 250, "predicted_only": 314, "actual_only": 1, "neither": 1245, "graded_queries": 0}` at
k = 1,024) shows that this rule predicts 250 of the 251 queries that differ at the full vector; the
predicted-only cases agree despite the different ideals (consistent with no positive in the top
10, which was not checked separately), and one difference is left unexplained (float32 cosine against exact float64 is the remaining convention difference).
Parity at 1,024 holds. The parity record is a diagnostic, not a gate.

**Reading (NFCorpus; values in the generated README).** Both teacher arms fall below
`l2s-official` (positive D −0.043, contrastive D −0.007; both pointwise intervals are below zero,
and both one-sided p values are near 1). The seed spread of D is under 0.001. Neither arm reaches
any claim level; the family's Holm step is still pending MS MARCO. The S1 controls sit slightly
above the reference, with intervals that include zero.

## MS MARCO

Snapshot `e7827da6a5dd…` adds the large-corpus path (chunked exact scorer, CPU label generation
with recorded pools, shortlist search, ADR-0018 MS MARCO size rule); its exactness is unit-tested
against the direct path. The authors' evaluator cannot run within the job caps on MS MARCO (it
materializes an N × k column copy of the normalized corpus per decile), so official-evaluator parity
is not run there; it is a diagnostic only.

| Step | Run | Record |
|---|---|---|
| Split provenance (every check passed) | `20261114T025422Z-8c5b643c` | `runs/p3-query-split-v1/provenance/msmarco/` |
| Probe: official labels, 256 fit ids (labels only, CPU) | `20261114T025444Z-9c593387` | `runs/evaluation/P3-bench/probe-official/msmarco/` |
| Probe: official labels + training, 1,024 fit ids (CPU) | `20261114T030153Z-88d3b82e` | same |
| Probe: scoring, shortlist search (N 1,000 and 10,000; caps 8, 32, 128) on 173 queries, S1 steps (GPU) | `20261114T031155Z-0a85cbfd` | `runs/evaluation/P3-bench/probe/msmarco/…/probe.json` |
| Resource freeze: **pass**, n_eval 7,500, n_rg = n_fit 4,000, n_tune 2,000, cap 8, shortlist 1,000, projected 62.2 h of the 69 h left after the probe | — | `freeze/resources-msmarco.json` (SHA-256 `daf4d4be…`) |
| Single official reference, seed 42, 4,000 fit ids (CPU) | `20261114T175430Z-e875cb4f` | first 1,024 targets and negatives equal the 1,024 probe's |
| Replay control (official targets, seed 42, CPU RNG replayed after the CPU labels) | `20261114T182941Z-ff228406` | `replay_equals_official: true` |
| Teachers (GPU, resumable), 12 teacher arms (CPU), S1 × 2, evaluation lock (family m = 8), evaluation | running (`msmarco_stage.sh`) | — |

**Resource decision.** `large_resource_plan` fixes the sizes first, in priority order, priced at
the cheapest measured search (cap 8, N = 1,000):

1. n_eval keeps its top value, 7,500, with n_rg and n_tune at their minimums.
2. n_rg is the largest value that fits next to it, 4,000; 8,000 does not fit.
3. n_tune falls to its minimum, 2,000.

Then N and the cap are chosen, larger N first. No cap fits at N = 10,000, whose searches cost about
1.5× more. At N = 1,000, cap 8 fits, and 98.7% of searches stop under it (the rule needs 95%).

The protocol's "remove rank-guided" option was recorded but not chosen. Its sizes
(`realized_without_rank_guided`: n_rg 35,000, n_tune 4,000, 59.1 h) apply only after an
escalation, and none happened.

Unit costs, before the 1.5 safety factor:

| Step | Cost |
|---|---|
| Labels | 0.53 s/query |
| Exact masked scoring | 0.056 s/query·mask |
| Top-1,000 pass | 0.20 s |
| Shortlist search at cap 8 | 10.2 s/query |

**Deviations at gate 1.**

- **Search share 0.5.** The probe used a search share of 0.5 instead of the CLI default of 0.7 to
  stay near its 3 h budget. The share only shortens how long the probe searches; the 95% rule and
  the projection use the measured searches (173 queries).
- **Label grid.** The label probe ran the 256 and 1,024 sizes of the grid; 4,096 was skipped
  because 1,024 labels already give the per-query label cost the projection needs.
- **Parity.** Official-evaluator parity is not run on MS MARCO (see above).
- **NFCorpus label grid (same kind of deviation).** NFCorpus gate 1 ran 256 and all 1,810 fit ids
  instead of 1,024.

**Memory.** The labels and the reference map an authors-order float32 copy of the corpus as
copy-on-write (`mmap_mode="c"`). The kernel counts those pages in VmRSS. The 256-label probe
reports `peak_rss_mib` 89,543. On the running reference (4,000 ids), the split is:

| Measure | Size |
|---|---|
| VmRSS | 70.5 GiB |
| RssAnon (anonymous) | 36.4 GiB |
| RssFile (file-backed, reclaimable page cache) | 34.0 GiB |

The 64 GB cap is read against anonymous memory, the part the job owns, and that part stays within
it.

**Job cap.** Teachers (about 11.4 h before the safety factor) and evaluation (about 24 h) do not fit
one 12 h job. Snapshot `5347538ab910…` makes both stages resumable without changing their outputs:
- query blocks of 50 are checkpointed under `artifacts/scratch/p3-bench/<corpus>/<stage>-<config
  hash>`;
- a job stops at a block boundary after `--max-hours` (11) with exit code 75, and the same command
  resumes it;
- one artifact is written at the end;
- its summary carries the summed compute seconds and `job_segments`.

The NFCorpus evaluation ran in two segments (the first limited to one block) as a live check of
the resume path.

**Lost time (operational, not compute).** The first reference launch (01:42 host time) died with
the agent session that started it: it was not detached, and no output was written. It was
relaunched detached at 14:54. The idle hours consumed none of the compute budget.

**Completed stages.**

- Teachers 20261114T183122Z-72b29180: 4,000 queries, 39,069 compute seconds in 2 segments.
- 12 teacher arms (CPU), recorded in the host's `logs/stage-msmarco.out`.

**S1 crash and resume (code fix, no protocol change).**

- The crash: S1 scratch failed at its first tune scoring (2026-11-15 02:30 host time).
  - `CorpusScorer.rows_f64` in streaming mode (`resident=False`, used only by MS MARCO S1)
    received a row index on the GPU and called `.numpy()` on it.
  - No S1 artifact was written, so nothing partial exists.
- The fix: one line, `index.cpu().numpy()`, with a GPU regression test. It is applied to the host
  snapshot as local commit 42bf508 on top of 5347538.
  - The fixed path returns exactly the same rows. Teachers, arms and the reference used the
    resident path and are unaffected.
  - The unfixed snapshot reproduced the error on the second host.
- The resume: `msmarco_resume.sh` reuses the reference, the teachers and the 12 arms. It then runs
  S1 scratch and official-init, the runs record, the lock (m = 8) and the evaluation (resumable).
  - Its GPU steps wait for the project rule (≥ 20 GB free, no compute process) and never preempt
    other users' jobs.
  - It started 2026-11-15 22:46 host time. The GPU freed soon after.
  - S1 scratch: 20261116T014506Z-b51f2627.

## Regenerated reports and code provenance (2026-11-24, local, no judgments read)

These were generated locally from the files under `runs/`. No host job ran, and no evaluation label
was read. They add the fields the spec requires that the evaluation READMEs lacked. The evaluation
runs themselves are unchanged.

| Corpus | Report run (`p3-bench report`) | Teacher report run (`p3-bench teacher-report`) |
|---|---|---|
| SciFact | `20261124T221548Z-94aa88a5` | `20261124T221549Z-efd5834a` |
| NFCorpus | `20261124T221551Z-f8313fcf` | `20261124T221552Z-4491ef19` |

- **Report runs** (`runs/evaluation/P3-bench/report/<corpus>/`):
  - the evaluation tables;
  - the scope sentence (FR-013);
  - the split fingerprints and the authors' commit (FR-011);
  - offline cost per arm with selection candidates and the S1 limitation (FR-009, FR-012);
  - online cost per query, from the probe's scoring time plus a selector forward pass timed at
    report time on the CPU.

  SciFact and NFCorpus have no MRL prefix-k rows. The `mrl-prefix` control was added after their
  evaluation locks, and adding it would read their evaluation labels a second time. The reports
  say so.
- **Teacher report runs** (`runs/evaluation/P3-bench/teacher-report/<corpus>/`): nesting across
  budgets, search counts, and demotions of unjudged documents versus known negatives (FR-007). The
  objective-gain table needs `searches.json`, which is on the host only. Rerun `teacher-report`
  there to add it.
- **Code provenance** ([snapshots.json](snapshots.json), from
  `scripts/checks/p3_bench_snapshot_map.py`): each of the 43 local run manifests is mapped to the
  newest bundled snapshot committed before it started.
  - The mapping agrees with the snapshot this record names for every run, except the two runs that
    started after the newest local bundle (`e7827da`). These are flagged `unverified`: the MS MARCO
    split provenance, and the NFCorpus evaluation, which ran on the resume snapshot `5347538`.
  - The bundles for `5347538` and `42bf508` are on the host only.
  - New runs record their snapshot directly: set `HYPERDIME_CODE_COMMIT=<40-hex commit>` in the
    host runner. `start_manifest` uses it when the code isn't a git checkout.

## Next

Family members (written before any NFCorpus or MS MARCO evaluation label was read):
`freeze/family.json`, m = 8 (SciFact 3, NFCorpus 2 after the escalation, MS MARCO 3). After the
MS MARCO evaluation, `p3-bench family` computes Holm and the Bonferroni bounds at m = 8 for all
three corpora.

For MS MARCO:
- After its evaluation, run `p3-bench report` and `p3-bench teacher-report` on the host. The host
  snapshot predates the report fields, so its evaluation README won't carry them.
- Copy the results here.
