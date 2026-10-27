# Data Model: F10 Descriptor Probe (experiment E2, gate G3)

Existing records are reused: `SpaceDescriptor` (schema 0, provisional), `PayloadRef`,
`GateDecision` (gate `"G3"` already allowed), `Criterion`, run manifests. No contract change.

## Descriptor features (`space/descriptor.py`, research D3–D5)

Column order is fixed; names are the `SpaceDescriptor.feature_names`. All values `float64`, one
row per dimension (`[1024, h]`).

| Group | Prefix | Columns | Source | Protocol |
|---|---|---|---|---|
| A | `doc_` | `mean, std, mean_abs, rms, q05, q25, q50, q75, q95` | documents (quantiles on ≤ 1M-row sample) | P0 |
| A (tails) | `doc_` | `skew, kurt, tail3` | documents, streamed | P0 |
| E | `red_` | `corr_mean_abs, corr_max_abs, corr_top8, pc_share_8, pc_share_32, pc_share_128` | streamed covariance | P0 |
| B | `qry_` | A's 12 columns | calibration queries | P1 |
| C | `pair_rand_`, `pair_top_` | `mean, std, q05, q50, q95` each | `q_j · d_j` over background / top-10 pairs | P1 |
| D | `contrast_` | `k5, k10, k20` | `q_j · (mean top-K − mean background)` | P1 |

`descriptor-P0` holds A (with tails) and E (18 columns, `num_queries = 0`); `descriptor-P1` holds
all groups (43 columns: 18 P0, 12 B, 10 C, 3 D; `num_queries` = calibration size). Invariants (tested): unchanged under
document or query row permutation; rows permute with dimensions; streamed equals one-pass; a
zero-variance dimension gives finite values (skew, kurt, tail3, correlations = 0).

## Descriptor ladder (`evaluation/descriptor_probe.py`, D7)

| Set | Name | Columns | Protocol |
|---|---|---|---|
| 1 | `variance` | `doc_std` squared | P0 |
| 2 | `A` | group A without tails | P0 |
| 3 | `A+E` | 2 + E | P0 |
| 4 | `A+E+tails` | 3 + tails | P0 |
| 5 | `+B+C` | 4 + B + C | P1 |
| 6 | `+D` | 5 + D | P1 |

## Target profiles (`evaluation/targets.py`, D6)

`targets-<env>.npy`, float64, indexed by `targets-<env>.json`:

| Key | Shape | Meaning |
|---|---|---|
| `s1/seed-<n>/budget-<k>` | `[1024]` | S1 seed n selection frequency at budget k, divided by k |
| `s1/seed-<n>/mean` | `[1024]` | mean over the nine budgets (primary, per seed) |
| `s1/mean` | `[1024]` | mean over seeds of `s1/seed-<n>/mean` (primary) |
| `s1/seed-<n>/log_importance` | `[1024]` | mean log-importance over queries |
| `s1/bias`, `s1/seed-<n>/bias` | `[1024]` | `projection.bias` of the S1 selector (secondary) |
| `s4_correction/seed-<n>/mean`, `s4_correction/mean` | `[1024]` | same as primary for `W q + b` of the S4 budget selector |
| `variance/mean` | `[1024]` | `variance`-method frequency, averaged over budgets (baseline B2) |

The JSON index records the query count, the selector run ids and payload hashes, and the noise
ceiling per target (mean pairwise Spearman `ρ` between seeds and its three-seed step-up
`3ρ / (1 + 2ρ)`).

## Shift check (`check.json`, research D0)

Per held-out environment: B1's Spearman with the seed-averaged primary target, the noise ceiling,
the gap, and `no_shift` (gap < 0.05 in at least 2 of 3 environments). Written by every
`probe-descriptors` run; a `--targets-only` run writes only targets, `check.json`, and
`inputs.json`.

## Probe result (`probe.jsonl`, one line each)

| Field | Type | Meaning |
|---|---|---|
| `heldout` | env | held-out environment |
| `target` | `s1` · `s4_correction` · `s1/budget-<k>` | target profile |
| `seed` | `mean` · `0` · `1` · `2` | seed-averaged or one seed's target |
| `set` | 1–6 or `B1` · `B2` | descriptor set or baseline |
| `protocol` | `P0` · `P1` · `—` | of the set |
| `form` | `shift` · `direct` | D10 (decisive) or D7 |
| `spearman` | float | on the held-out 1,024 dimensions |
| `penalty` | float · null | chosen by GCV (shift) or inner folds (direct) |
| `p`, `p_holm` | float · null | permutation p (D9), Holm over the sets evaluated within the fold and form (seed `mean`, primary target only) |
| `n_permutations` | int | 1,000 for primary rows, 0 otherwise |

`coefficients.json`: per fold and set, standardized coefficients by feature name (seed `mean`,
primary target). `context.json`: research D11's environment-level table inputs.

## Report tables (`reports/E2/tables/`)

`check.csv` (D0), `criteria.csv` (SC-001/SC-002 per fold and set, shift form), `ladder.csv`
(Spearman per fold, set, form),
`seeds.csv` (per-seed direct Spearman with both baselines), `secondary.csv` (per-budget and S4
targets), `ceiling.csv`, `coefficients.csv`, `context.csv`. `README.md` and the tables regenerate
byte-identically.

## GateDecision G3

`gate = "G3"`, criteria SC-001 (value: number of folds where a set passes; threshold ≥ 2 of 3)
and SC-002 (protocol of the best passing set), evidence keys as in S4 (`check`, `ladder`,
`ceiling`, `direct`, `test_set_reuse`); with `no_shift`, SC-001 reads "not
supported: no shift to explain"; drafted ADR-0013 status "proposed", never overwriting a
non-proposed ADR.
