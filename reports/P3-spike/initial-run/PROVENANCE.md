# Initial spike-derived P3 report

Frozen output of the first P3 implementation, kept so these numbers stay identifiable. Later P3 code
cannot regenerate it, because this run has no all-cardinality curves.

- Run: `artifacts/p3-spike/evaluation/P3/all/20261105T222415Z-de526aa7` (gitignored), commit
  `2b7d031` with a dirty working tree.
- Inputs: F0 spike caches converted by `scripts/checks/p3_spike_inputs.py`; SciFact and NFCorpus,
  not MS MARCO. Not the published F2 artifacts.
- Code and report at that moment: `artifacts/p3-spike/initial-run-snapshot/code-and-report.tar.gz`,
  with the run's file hashes in `run-hashes.txt` beside it.
- Superseded for reading by [reports/P3-spike](../README.md), whose controls change how its
  within-query agreement should be read.

## Later correction

The frozen report's limitation "Sign conflicts and cancellation can only grow as positives are added"
is wrong for cancellation. Sign-conflict coverage is monotone under set expansion; the normalized
cancelled share is not: `(1, -1), (-1, 1)` cancels all mass, and adding `(1, -1)` cancels 2/3. Later
reports and a regression test carry the corrected statement. The frozen text stays as it was.
