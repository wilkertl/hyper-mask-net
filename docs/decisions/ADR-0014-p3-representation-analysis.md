# ADR-0014: P3 is a descriptive analysis of the stored retrieval space

- **Status:** proposed
- **Date:** 2026-11-05
- **Issue:** none assigned; a ticket is still needed before this work gets a pull request

## Context

Before proposing selector architectures, descriptors, objectives, or retrieval methods, we want to
know what dimension-level contributions `c_j = q_j * d_j` of known relevant pairs look like. Earlier
work averages positives first (`oracle.targets.aggregate_pairs`), which hides disagreement among
them. This experiment must stay descriptive: nothing here may select, rank, or benchmark.

## Decision

- **Space.** Only the stored final embeddings. The pipeline persists neither token embeddings nor
  intermediate encoder states, so no correspondence with their coordinates is assumed.
- **Labels.** Training judgments only. `analyze_representations.load_training_judgments` opens
  `qrels-train.jsonl` by name and refuses a record of any other split; embeddings come through
  `space.inputs.load_unlabeled`, which opens no judgments file. `training.inputs.load_inputs` and
  `data.prepare.load_environment` are not used, because both read validation judgments.
- **What counts as a negative.** Judged nonrelevant (grade 0) documents get the same statistics in a
  separate table with their coverage. Unjudged documents appear only as the labeled reference below
  and are never called negative; the corpus is never ranked.
- **Individual pairs first.** Each positive stays a row. Aggregation is analyzed last, as the object
  of study, with the repository's oracle weights `2**grade - 1`.
- **Three claims kept apart.** Maximizing similarity to one known relevant document and preserving
  the original similarity are measured. Improving a ranking is not, because no document's score is
  compared with a competitor's.
- **Every cardinality.** The restriction to 307 was a reporting choice plus per-budget mask
  construction, not a scientific limit. One ordering per pair now gives every `k` from 1 to 1024: a
  dimension is in two top-`k` masks exactly when `k > max(rank_a, rank_b)`, so a histogram of those
  maxima gives exact intersections, and a histogram of ranks gives frequency, with no mask stored.
  `--budgets` now names summary checkpoints only, and no checkpoint is singled out.
- **Reference control for the shared query.** For each query, documents drawn uniformly from its
  corpus, excluding its judged documents, in the number of its positives, over
  `--reference-samples` seeded repeats, each query with its own generator so its draw does not
  depend on which other queries ran. They are unjudged, not negatives. Reference rows are stored;
  reference contributions are recomputable from them.
- **Magnitude control.** Selection frequency of relevant against reference documents, the excess
  and its split-half stability, and the same analysis on `c_j / (mean|q_j| * mean|d_j|)`, a scale
  from unlabeled vectors only (all analyzed queries and a seeded corpus sample).
- **Count control.** Jaccard is compared within strata of positive count. Sign-conflict coverage can
  only grow with group size, and the cancelled share can move either way, so both are compared at
  fixed group sizes on both groups, with equal weights because a reference document has no grade.
- **Bounded pairs.** Queries over `--max-query-pairs` document pairs use a seeded sample of pairs,
  the same bound for both groups. The report counts how many queries that affects.
- **Balancing.** Corpus summaries average within a query and then across queries. Pair-weighted
  values are computed separately and labeled. Split halves assign whole queries.
- **Numerics.** Stored float16 vectors are widened to float64, so the decomposition identity is
  checked without float32 rounding. Ratios divide by a mass, never by a signed score, and an
  undefined ratio is NaN, dropped from summaries and `null` in JSON.
- **Ties.** Descending contribution, then lower dimension index (stable sort).
- **"Stable".** Lowest split-half Spearman of the query-balanced frequency profile at least
  `reports.p3_controls.STABLE_SPEARMAN`. It is a labeling rule for these data, with a shuffled-dimension
  reference beside it, and it says nothing about unseen queries.
- **Identifier.** P3 follows the registry's P-rows; P2 is taken by the F11 covariance probe's spec.
- **Figures.** Static SVG from `reports.p3`, because plotting libraries are not project
  dependencies. Each figure's data is in the run's `summary.json`.
- **No resumption.** Measured cost does not warrant it; `--max-queries` bounds a run, and the cost
  table records runtime and storage.
- **Spike inputs.** `scripts/checks/p3_spike_inputs.py` lets a run use local F0 caches for
  validation when the published artifacts are out of reach. It re-derives F2's training split from
  `train.tsv`, writes only training judgments, withholds query text, and never opens `test.tsv` or
  the spike's `splits.json`.
- **Sources stay apart.** `analyze-representations` refuses non-F2 inputs without
  `--allow-spike-inputs` and refuses a run mixing the two. `report P3` writes such a run to
  `reports/P3-spike`, refuses to write it to `reports/P3`, and marks it as validation only. The
  first spike report is frozen in `reports/P3-spike/initial-run` with its provenance.
- **Input verification.** Before analysis, a run hashes both payloads and checks D, split counts,
  disjoint splits, and unit norms, stopping on a failure and recording the rest in `inputs.json`.
  `scripts/checks/p3_input_differences.py` records how F2 and spike inputs differ.

- **Devices.** Contributions, orderings, curves, reference controls, and rank histograms run in
  batches of whole queries on `run.device` (`--batch-pairs`), in float64, with a stable sort that
  treats both signs of zero as equal. The corpus stays memory-mapped and only gathered rows move.
  A relevant pair's ordering is computed once and read back by every later step. Draws are planned
  per query on the CPU, so neither the batch size nor the device changes them. The per-pair
  statistics and corpus summaries still run on the CPU; the report states device and phase times,
  so a CPU run cannot read as a GPU one. `scripts/checks/p3_device_parity.py` compares two runs:
  orderings and reference draws must match exactly, and other arrays within a tolerance of `1e-6`,
  because float64 sums and float32 casts may differ in their last bits between devices or runs.

## Evidence

- Canonical report: [reports/P3](../../reports/P3/README.md), run `20261106T231934Z-efae7f88` on DataPolicy-4090 (RTX 4090). All P3 runs and their purposes: [RUNS.md](../../reports/P3/RUNS.md).
- Coverage pilot of a score-matched reference control, with no curves:
  [reports/P3-score-matching-pilot](../../reports/P3-score-matching-pilot/README.md). It limits what
  such a control could describe, so no matched curve has been computed.
- Validation on spike-derived inputs: [reports/P3-spike](../../reports/P3-spike/README.md). No number
  is typed here.
- Run manifests: `artifacts/evaluation/P3/all/<run_id>/manifest.json` on DataPolicy-4090, and
  `artifacts/p3-spike/evaluation/P3/all/<run_id>/manifest.json` locally (gitignored).

## Consequences

- The report can motivate hypotheses but cannot support a method: its patterns are in-sample.
- A difference from reference documents is not a relevance effect by itself. Relevant documents
  also lie closer to the query and to one another, and this control does not match on
  that. Matching on score within a bounded sample needs no ranking of the corpus; it has not been run. Reference documents may include unjudged relevant ones.
- Canonical results exist for all three environments on the published F2 artifacts. MS MARCO has
  the 45,000 training queries of F2's `train_query_cap`, not every query of its source. Numbers are
  cited from the report, which reads the run; spike-derived numbers stay validation only.
- A run maps files, so its peak RSS can far exceed its allocations and its device memory: on MS
  MARCO most of it is reclaimable pages of the embeddings and staged arrays
  (`scripts/checks/p3_memory_profile.py`, `reports/P3/memory-profile.json`).
- No public API changes outside the new command, its report, and their registrations. Earlier
  results are not affected.
