# P3-mask-ranking: run record and reading

`README.md`, `tables/`, and `figures/` are generated from the run and are unedited; their hashes are
in `report-files.sha256`. This file records how they came to be and how to read them. Result values
live in the tables, and none are typed here.

## Run

- Scientific run `20261109T163931Z-9e9f5a0a` on `ceia40901`, RTX 4090, clean snapshot
  `bd0fd382202d454c2e262fc19859074b16a43c0a`. It exited 0. Hashes of its files are in
  `run-files.sha256`; the run was not modified after it was written, and a later hash check agreed.
- Before launch, the watcher's smaller-batch probe passed its memory limit and matched the pilot.
- The second host (`ceia40902`) never launched: its setup stood down when this host took the claim.

## Report failure, cause unresolved

The first report generation, right after the run, ended with `index 4294967913 is out of bounds for
axis 0 with size 2162` (`original-report-failure.txt`; no traceback was captured, only the error
line). The same code and the same run files then produced an identical report in every later
generation, including five runs of the exact command, and the resampling matrices are in range with
the same hashes each time. That establishes non-reproduction and nothing more. The value is 2^32 +
617, but no evidence in hand identifies a cause: this account cannot read kernel logs, and the host
exposes no memory-error counters. No report code was changed. The final report was generated once
more from the unchanged snapshot, and the rest of this record does not depend on the failed attempt.

## Checks of the stored run

`integrity.json` (`scripts/checks/p3_ranking_integrity.py`, CPU only, GPU hidden; manifest in
`integrity-manifest.json`) reports no failed check on any corpus.

- **Every stored row:** selection identical to the census of eligible training queries, none in
  another split, no capping, unit weights; input hashes; masks per query and arm, partitions
  included; no duplicate mask; rank bounds; scores on the exact 2^-48 grid; better score implies
  better rank within each mask; pairing of the arrays the report reads.
- **Bounded spot checks:** a seeded sample per environment, reached through every stratum and every
  arm the corpus allows and all nine budgets plus the full query. Masks were rebuilt from inputs,
  kept exactly `k` coordinates, and at `k = 1024` were the query bit for bit. Partitions index only
  the query's training positives, are disjoint, and a mask does not change when documents outside
  its construction set are blanked. A separate NumPy rescan of the whole corpus reproduced ranks,
  ties, top-100 metrics, and the scores at ranks 1, 10, and 100 exactly. Per-query arrays were
  rebuilt from stored ranks and matched.
- **Paired primary comparisons:** recomputed from per-query arrays apart from the report's table
  code, in the query-balanced unit, and all rows agree with `tables/claims.csv`. The first attempt
  flagged one cell, which was `0.0` against `-0.0000` in print; the comparison now reads numbers.

The spot checks are a sample and do not prove every other rank. A full rescan was not run, since
nothing above gave a reason for one.

## Reading the result

Privileged diagnostic on training judgments. Compare the primary family with `tables/memorization.csv`.

- **SciFact and MS MARCO:** the large all-label gains of a 1-1 mask come from the document that
  built it. The held-out positive of the same query does not benefit in reciprocal rank, and more
  queries worsen than improve. This is the memorization pattern ADR-0016 defined.
- **NFCorpus:** a partial exception. Held-out positives gain in log rank and in the top 10 and 100,
  while reciprocal rank barely moves. Its queries have many positives, often on the same topic.
- **Competition:** `tables/competition.csv` shows why similarity is not ranking. Masks that raise a
  positive's margin often lower or barely move its score, and the rank-10 score falls as well.
- **Size of the construction set:** larger cells exist mostly on NFCorpus and are descriptive.

Limits of the design:

- With 108 tests and 10,000 permutations, Holm cannot report an adjusted p below 0.0108, so several
  rows show exactly that floor.
- Only multi-positive training queries are ranked, a minority of MS MARCO's training queries. SciFact
  has few queries, so its intervals are wide.
- Judgments are incomplete: a competitor that rises may be an unjudged relevant document.
- MS MARCO was not ranked over literally every document; see the next section.
- Reciprocal rank and top-k disagree on NFCorpus, so no single measure summarizes it.

## MS MARCO id collision

The historical run stands as written and its generated report is unedited, including the sentence
that every corpus document was scored. On MS MARCO, query and passage ids are both numeric, so
every ranked query's id equals some passage's id, and the BEIR-style rule (`rank_documents`, meant
for corpora whose queries also appear as documents) removed one unrelated passage from each query's
competition. The passage was still scored into the corpus statistics, and it was never a training
positive. SciFact and NFCorpus have no such collision. Applying the same rule to every arm does not
by itself cancel a bias, so the effect was measured.

`exclusion-sensitivity/` (`scripts/checks/p3_ranking_exclusion_sensitivity.py`, CPU only, run
`20261109T180212Z-0514a02d`) scores the excluded passage under every stored mask and, from the
stored rank and exact score of each tracked positive and the lower-row tie rule, derives the rank
that positive would have had. With no correction the same code reproduces every stored per-query
array exactly. `exposure.csv` counts, per arm and budget, the evaluated positives whose rank moved,
those that left the top 10 or 100, and the queries involved; `primary-comparison.csv` sets the
stored primary family beside the corrected one, on the same resamples and permutations, with Holm
over the full family.

The conclusions do not change: no row changes the sign of a difference, the side of zero of its
interval, or its Holm reading at 0.05. Only rank-based measures are corrected. `sensitivity.json`
lists them; nDCG, MRR, and Recall from top lists are not recomputed. In no list-bearing mask did the
passage reach the stored score at rank 100, so it could not have entered those lists at all.

The same rule is the default of `rank_documents` outside this experiment, and no other result in
this repository was examined here.

## Recommendation

Do not take positive-similarity-maximizing masks as a selector target on this evidence. Where
relevant documents are few, what they gain is not transferred to the other relevant documents of
the same query. Any next step should be a separate design discussion about a ranking-aware objective
and a cross-query evaluation. No selector was trained here.
