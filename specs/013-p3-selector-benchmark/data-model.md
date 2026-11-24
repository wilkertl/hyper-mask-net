# Data Model: F12 P3 Selector Benchmark (as built)

Every run directory is written by `ArtifactWriter` and contains a `manifest.json`. The manifest
records the run id, command, config and config hash, dataset hash, model revision, seed,
determinism flags, device, dtype, versions and timestamps. Paths below are relative to the
artifacts root (`--root`, default `artifacts/`). `reports/P3-bench/runs/` mirrors the small files.

## Entities

### QuerySplit (`data/query_split.py`)

| Field | Meaning |
|---|---|
| corpus | `scifact`, `nfcorpus`, `msmarco` |
| ids | per split (`fit`, `tune`, `evaluation`), sorted tuples |
| fingerprints | `id_fingerprint`, which doesn't depend on order |
| excluded_test_overlap_ids | development ids removed for matching official test text |
| seed, normalization | `20261112`; NFKC, casefold, whitespace |

The loader refuses a changed file hash, unsorted ids, overlapping splits, and counts that differ
from the manifest. `hash_order(corpus, split, ids)` gives the nested prefixes,
`SHA-256("p3-bench-v1|20261112|<corpus>|<split>|<id>")`.

### L2SNegatives and the official reference (`selectors/l2s-official/<corpus>/<run>/`)

| File | Content |
|---|---|
| `state_dict.pt` | final-epoch weights; its SHA-256 goes into the evaluation lock |
| `targets.pt` | the authors' soft targets per fit query |
| `negatives.npy` | sampled negative row indices per fit query (64 per query) |
| `fallback.npy` | whether the whole-corpus fallback ran, per query |
| `valid.json` | fit queries that have a positive |
| `rng_after_labels.pt` | RNG state after label generation, replayed by the teacher arms |
| `summary.json` | n_fit, fallback count, loss, seconds, peak memory, device, source runs |

### TeacherMask (`evaluation/P3-bench/teachers/<corpus>/<run>/`)

- Masks per query × budget × teacher: `positive`, `contrastive`, and `rank-guided` with its two
  starts.
- `searches.json` records accepted swaps and seconds per search.
- `global_mask.npy` holds the global-mask control, by selection frequency with ties to the lower
  dimension.
- `summary.json` holds: `nesting`, the share of the smaller mask inside the next larger one;
  `demotions`, unjudged versus known negatives; `searches`, `searches_improved`,
  `searches_hit_cap`, `step_cap` and `seconds`.

Validation:
- |mask| = k;
- acceptance is strictly lexicographic on `(nDCG@10, GDR)`;
- the result is never below either start;
- a tie between the searches goes to the positive start.

### BenchmarkArm

| Arm | Directory | Key outputs |
|---|---|---|
| teacher arm | `selectors/p3-teacher-arms/<corpus>/<run>/` | weights, `summary.json` (`final_train_loss`, `seconds`, `replay_equals_official` for target `official`) |
| S1 | `selectors/p3-s1/<corpus>/{scratch,official-init}/<run>/` | `selection.json` (`cells`, `selected`, `selection_candidates` = 8 × 21, `tune_queries`), weights per seed |
| controls | — | `full`, `variance` (computed at evaluation), `global-mask` (from the teachers run) |

The target is `t_q = (1/9) Σ_k m_{q,k}/k` (`_target`). The architecture is the authors' linear
layer, and `_linear` maps official and S1 checkpoints to the same layer.

### EvaluationRow (`evaluation/P3-bench/evaluation/<corpus>/<run>/per_query.csv`)

Columns: `query, k, arm, ndcg@10, mrr@10, recall@100, ndcg@100`. The run's `summary.json` has
`evaluation_queries`, `family_size`, `full_vector_parity_at_1024`, `per_decile_ndcg10`, `primary`
(D, p, intervals, seed D), `secondary` and `holm`. `README.md` is generated from `summary.json`.

Primary endpoint: per query, the mean of nDCG@10 over `PRIMARY`, the nine deciles 102 … 921. Every
eligible evaluation query has weight 1. D is the teacher arm minus `l2s-official`, at seed 42.

### Family (`evaluation/P3-bench/family/<run>/`)

`family.json` and `family.csv` hold Holm and Bonferroni at m = `len(members)` over every corpus's
evaluation run.

### Report runs (no judgments read)

| Directory | Files | Built from |
|---|---|---|
| `evaluation/P3-bench/report/<corpus>/<run>/` | `README.md`, `costs.json` (offline per arm: training seconds, prerequisite seconds, wall, device, selection candidates; online per query: masked scoring from the probe, selector forward timed at report time), `provenance.json` | the evaluation's `summary.json` and manifest, the lock record's runs, the resource record's probe |
| `evaluation/P3-bench/teacher-report/<corpus>/<run>/` | `README.md`, `gains.json` (chosen searches, improved, mean and max Δ nDCG@10, mean Δ GDR, search seconds) | the teachers' `summary.json` and `searches.json` |

`summary.json` of evaluations made from 2026-11-24 also holds `provenance`: split manifest
SHA-256, per-split fingerprints, `l2s_commit` and `transfer_flag`. Each manifest's `git_commit` is
the checkout's HEAD or, outside a checkout, `HYPERDIME_CODE_COMMIT`.

`reports/P3-bench/snapshots.json` (from `scripts/checks/p3_bench_snapshot_map.py`) maps each run
id to `snapshot`, `method` (`recorded` or `inferred`) and, where it applies, `unverified`.

## Gate records (`reports/P3-bench/freeze/`, all fingerprinted)

| Record | Kind | Key fields | Written before |
|---|---|---|---|
| `protocol.json` | protocol | SHA-256 of ADR-0018 and spec/plan/research/tasks, L2S commit, split manifest hash | the probe (which re-checks it) |
| `resources-<c>.json` | resources | probe run and hashes, unit costs, `projected_hours_by_cap`, `stop_share_by_cap`, `realized` {n_fit, n_rg, n_tune, n_eval, shortlist, step_cap}, budget, safety factor | teachers beyond the probe, tune metrics, evaluation labels |
| `escalation-<c>.json` | escalation | option, reason, lower-bound hours, removed arms and controls, resource-record hash | anything evaluated on that corpus |
| `runs-<c>.json` | — | run id per stage and arm (official, teachers, `arm:<teacher>@<seed>`, `s1:<arm>`) | the lock |
| `family.json` | — | members `corpus:arm`, note | any NFCorpus or MS MARCO evaluation label |
| `evaluation-lock-<c>.json` | lock | `arm_sha256` per arm@seed, `runs`, `evaluation_ids` and their hash, conventions, resource-record hash | evaluation labels (`evaluate` re-checks every hash) |

## State transitions per corpus

```text
split verified → protocol frozen → probed → resources frozen ─┬─ pass ─────────────┐
                                                             └─ escalate → choice ─┤
reference trained → (parity) → replay control → teachers → arms, S1 → runs record ─┘
→ evaluation lock → evaluated (once) → family (after all corpora)
```

Resumable stages (teachers, evaluate) write blocks to
`scratch/p3-bench/<corpus>/<stage>-<config hash>/` and one artifact at the end.
