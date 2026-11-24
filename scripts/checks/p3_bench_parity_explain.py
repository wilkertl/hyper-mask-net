"""Explain F12 official-evaluator parity differences on a corpus (diagnostic, fit queries only).

The authors' nDCG@10 takes its ideal over the top-1,000 retrieved list; the common evaluator takes
it over all positives. When fewer than min(10, P) positives fall inside the top 1,000, the two
ideals differ. This counts, at k = 1,024 (the full vector, where both rankings coincide), how many
per-query differences that rule predicts. Usage (host, from ~/p3-bench):

    python -I scripts/checks/p3_bench_parity_explain.py artifacts nfcorpus <parity run> <n_fit>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from hyperdime.cli.p3_bench import _positives, load_fit_inputs
from hyperdime.evaluation.benchmark import masked_scores, positive_ranks


def main(root: str, corpus: str, run: str, n_fit: str) -> None:
    inputs = load_fit_inputs(Path(root), corpus, int(n_fit))
    parity = Path(root) / "evaluation" / "P3-bench" / "official-parity" / corpus / run
    rows = json.loads((parity / "per_query.json").read_text())
    docs = torch.from_numpy(np.asarray(inputs.documents).astype(np.float64))
    qindex = {q: i for i, q in enumerate(inputs.query_ids)}
    doc_rows = {d: i for i, d in enumerate(inputs.doc_ids)}
    counts = {"both": 0, "predicted_only": 0, "actual_only": 0, "neither": 0, "graded_queries": 0}
    for row in rows:
        positives, grades = _positives(inputs, row["query"], doc_rows)
        counts["graded_queries"] += int(max(grades) > 1)
        q = torch.from_numpy(inputs.queries[qindex[row["query"]]].astype(np.float64))
        ranks = positive_ranks(masked_scores(docs, q, None), torch.tensor(positives))
        predicted = min(10, int((ranks <= 1000).sum())) < min(10, len(positives))
        actual = abs(row["official:1024"] - row["authors:1024"]) > 1e-9
        names = {(True, True): "both", (True, False): "predicted_only"}
        names.update({(False, True): "actual_only", (False, False): "neither"})
        key = names[(predicted, actual)]
        counts[key] += 1
    print(json.dumps({"corpus": corpus, "parity_run": run, **counts}))


if __name__ == "__main__":
    main(*sys.argv[1:5])
