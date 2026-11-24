# P3-oracle-l2s: run record and reading

`README.md` and `tables/` are generated from the run named in the README and are unedited. Values
live in the tables; none are typed here. Protocol: ADR-0017.

## Run

- Scientific run `20261109T235856Z-5d598106` on `ceia40901`, CPU (the shared GPU was held by another
  job), snapshot `cf2d71a2828bbd5ccc93aac02588277608182d90`, SciFact only, every training query with
  a positive, uncapped. `run-manifest.json` is that run's manifest. Its working root
  `~/p3-oracle-pilot` links to the F2 artifacts and to the mask-ranking run, and nothing was written
  into them. The report was generated from snapshot `46e84e141db9b8f3640836343ec4a5e9dd64ca39`,
  which differs only in report code.
- The provenance audit passed for SciFact, NFCorpus, and MS MARCO (run
  `20261109T235838Z-e9fd3107`, `evaluation/P3-oracle-l2s/provenance` in that root), so a checkpoint
  is ready for each. Only SciFact has been compared so far.

## What was checked

- Ranks of the same masks agree exactly with the stored mask-ranking run of ADR-0016 on the masks
  they share, and every arm at `k = 1024` is the full query.
- The all-positive oracle is never exceeded on its relevance-weighted mean score by prefix-k or L2S.
- For every single-positive query the pair-optimal mask is the oracle's mask.
- The repository's float32 `evaluate_methods` gives the same nDCG@10 as this run's exact scan.
- SciFact has no id collisions. `checks.csv` holds the counts.

## Reading

- L2S is **in-sample**. These are its training queries and positives, so its gains here cannot be
  read as performance on unseen queries. Its strong in-sample agreement with the oracle's ranking
  benefit does not show that its signal transfers. A linear map of this size can fit a training set
  of 728 queries.
- The all-positive oracle is a score ceiling only. Its rank metrics in the tables are not a ceiling,
  and where it falls below the full query or L2S the paired tables say so.
- `held minus L2S` compares a mask that saw one other positive with a selector fitted to all of
  them. It is not a race between equals. The informative contrast on held-out positives is held-out
  transfer against the full query and against the privileged oracle.

## Limits

- One environment, one L2S seed per environment (seed 0). L2S was selected at k = 307 on validation
  when it was trained; that dependency of the checkpoint is not changed here.
- Query-disjoint claims need the separate evaluation that ADR-0017 predefines; no selector was
  fitted.
