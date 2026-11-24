# P3-oracle-l2s, NFCorpus bounded pilot: run record and reading

`README.md` and `tables/` are generated from the run below and are unedited; hashes are in
`report-files.sha256`. Values live in the tables; none are typed here. Protocol: ADR-0017,
"Bounded follow-up pilots". The SciFact report in `reports/P3-oracle-l2s` is a separate run and was
not regenerated (its hashes were checked before and after this launch).

## Run

- Scientific run `20261112T160013Z-462aa6c4` on `ceia40901`, snapshot
  `d362793ae9be9c3d876c7ffd11ebc18a0993b69d` (clean), one launch of the guarded launcher
  (`artifacts/p3-spike/oracle-followup/followup.sh`, claim `~/p3-oracle-followup/launch.claim`).
  `run-manifest.json` is that run's manifest.
- Device: RTX 4090 (`cuda:0`). All five GPU checks showed no compute process and 24,080 MiB free.
- Bound 128 training queries, split evenly: 64 single-positive, 64 multi-positive (stratified by
  positive count, weights `N_h / n_h`). Every corpus document (3,633) scored for every mask, every
  decile 102 to 1024. Training judgments only.
- Cost: 3.6 s inside the run (7.0 s wall clock with start-up), peak RSS 2.1 GB, peak GPU allocation
  0.7 GB.

## What was checked (`checks.csv`)

- L2S provenance: all 16 checks passed (F2 dataset, embeddings and negatives, seed 0).
- Parity at 100%: every arm at k = 1024 gives the full query's ranks and lists (0 mismatches).
- Stored-run parity with ADR-0016's run on 20,521 shared masks: 0 mismatches.
- NFCorpus has no query id equal to a document id; nothing excluded by id.
- The repository's float32 `evaluate_methods` gives the same nDCG@10 on these queries.
- 5,659 tracked documents tie exactly with another document: NFCorpus has duplicate texts. The
  lower-row rule decides them in every arm alike.

## Reading

- L2S is **in-sample**. On single-positive queries it reaches almost the oracle's ranking, which is
  what a linear map of 1024 x 1024 can do on queries it was fitted to. It says nothing about unseen
  queries.
- On multi-positive queries L2S ranks above the all-positive oracle at the lower deciles. The oracle
  maximizes the positives' mean similarity, not a ranking; L2S was trained against hard negatives.
  This is the case ADR-0017 warns about: the oracle is not a ranking ceiling.
- A bounded sample: the intervals cover sampling queries within each stratum and are exploratory.
