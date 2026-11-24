# P3-oracle-l2s, MS MARCO bounded pilot: run record and reading

`README.md` and `tables/` are generated from the run below and are unedited; hashes are in
`report-files.sha256`. Values live in the tables; none are typed here. Protocol: ADR-0017,
"Bounded follow-up pilots". The SciFact report in `reports/P3-oracle-l2s` was not regenerated (its
hashes were checked before and after this launch).

## Runs

- Same launch, snapshot, and claim as the NFCorpus pilot (`d362793ae9be`, clean), on the RTX 4090
  after five more GPU checks with no compute process and 24,080 MiB free.
- Probe `20261112T160140Z-00d34c7c`: 4 queries (2 single, 2 multi), 304 masks, 53.1 s, within its
  300 s limit. Projected 32-query time at a ceiling of 3,000 masks: 211 s, within the 900 s bound,
  so the pilot went ahead. The probe's run stays under `~/p3-oracle-followup/msmarco-probe` and is
  not reported.
- Pilot `20261112T160235Z-298d5679`: 32 training queries, 16 single-positive and 16 multi-positive
  (stratified by positive count, weights `N_h / n_h`), 2,222 masks, every one of the 8,841,823
  passages scored for every mask, every decile 102 to 1024. Training judgments only.
  `run-manifest.json` is its manifest.
- Cost: 74.6 s inside the run (76.3 s wall clock); scan 41.6 s (about 19 ms per mask), float32
  check 9.9 s, read 3.4 s; peak RSS 28.1 GB (the float16 corpus), peak GPU allocation 9.1 GB
  (14.4 GB reserved).

## What was checked (`checks.csv`)

- L2S provenance: all 16 checks passed.
- Parity at 100%: 0 rank and 0 list mismatches at k = 1024 across all arms.
- Id collisions retained: each of the 32 queries has a passage with its id; none was excluded, and
  all 2,222 masks carried it. Its exact score was recorded under every mask, and it charged no rank
  to any tracked positive in this sample, so new rank minus charges equals the stored rank of
  ADR-0016's run trivially here: 1,411 shared masks compared, 0 mismatches. The correction rule was
  exercised but not stressed by a non-zero charge.
- The float32 `evaluate_methods` gives the same nDCG@10 on these queries.

## Reading

- 16 queries per stratum: intervals are wide and exploratory.
- L2S is **in-sample** but, with 45,000 training queries, gains far less over the full query than on
  SciFact or NFCorpus, and the privileged oracle stays well above it at every decile below 1024.
- No larger MS MARCO run follows from this. A full run would need the reuse conditions of ADR-0017
  and its own approval; any time estimate from this pilot is an extrapolation.
