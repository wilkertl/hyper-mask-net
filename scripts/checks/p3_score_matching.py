"""Coverage-only pilot of a score-matched P3 reference control, and its report.

    python scripts/checks/p3_score_matching.py <root> --device cuda
    python scripts/checks/p3_score_matching.py <root> --report-to reports/P3-score-matching-pilot

The first writes ``<root>/evaluation/P3-matching-pilot/all/<run_id>``. Training judgments only: they
reach this script through ``load_training_judgments``. Every judged document of a query is excluded
from its candidates. No mask, ordering, or curve is computed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from hyperdime.artifacts import latest_artifact
from hyperdime.cli.analyze_representations import (
    CANONICAL,
    ENVIRONMENTS,
    load_training_judgments,
    verify_inputs,
)
from hyperdime.contracts.hashing import NO_DATASET, file_sha256
from hyperdime.contracts.manifests import ArtifactWriter, resolve_device, start_manifest
from hyperdime.contracts.schemas import DatasetManifest, EmbeddingArtifact, load_record
from hyperdime.contracts.seeding import seed_everything
from hyperdime.contracts.validation import ContractError
from hyperdime.data.prepare import MANIFEST, qrels_file
from hyperdime.evaluation.representation_matching import _Env, analyze_matching, select_stratified
from hyperdime.reports.p3_matching import write_report
from hyperdime.space.inputs import load_unlabeled

COMPLETE_EXPERIMENT = "evaluation/P3-matching-pilot"


def _environment(writer: ArtifactWriter, root: Path, env: str, args: argparse.Namespace) -> Any:
    inputs = load_unlabeled(root, env)
    dataset = load_record(DatasetManifest, inputs.dataset_dir / MANIFEST)
    documents = load_record(EmbeddingArtifact, inputs.embeddings_dir / "documents.json")
    queries = load_record(EmbeddingArtifact, inputs.embeddings_dir / "queries.json")
    provenance = dataset.spec.get("provenance", CANONICAL)
    if provenance != CANONICAL:
        raise ContractError(f"{inputs.dataset_dir} is not a published F2 artifact: {provenance}")
    verification = verify_inputs(inputs, documents, queries, dataset, args.seed)
    judgments = load_training_judgments(inputs.dataset_dir, inputs.splits.get("train", []))
    rows = {doc: row for row, doc in enumerate(inputs.doc_ids)}
    positive: dict[str, np.ndarray] = {}
    judged: dict[str, np.ndarray] = {}
    for qid, labels in judgments.items():
        found = sorted(rows[d] for d, g in labels.items() if d in rows and g > 0)
        if len(found) > 1:
            positive[qid] = np.array(found, dtype=np.int64)
            judged[qid] = np.array(sorted(rows[d] for d in labels if d in rows), dtype=np.int64)
    counts = {qid: found.size for qid, found in positive.items()}
    ids, strata, weights, table = select_stratified(counts, args.seed, args.max_queries)
    summary = analyze_matching(
        _Env(
            ids,
            lambda qid: np.asarray(inputs.query_vectors[inputs.query_rows[qid]]),
            inputs.documents,
            positive,
            judged,
        ),
        seed=args.seed,
        repeats=args.repeats,
        sample_size=args.sample_size,
        caliper=args.caliper,
        max_query_pairs=args.max_query_pairs,
        weights=weights,
        strata=strata,
        strata_table=table,
        device=args.resolved,
        batch_queries=args.batch_queries,
        path=lambda name: writer.path(f"{env}/{name}"),
    )
    summary["cost"]["artifact_bytes"] = {
        path.name: path.stat().st_size
        for path in writer.path(f"{env}/matched-rows.npz").parent.iterdir()
    }
    return summary, {
        "provenance": provenance,
        "dataset": inputs.dataset_dir.name,
        "embeddings": inputs.embeddings_dir.name,
        "qrels_train_sha256": file_sha256(inputs.dataset_dir / qrels_file("train")),
        "documents_sha256": documents.payload.sha256,
        "queries_sha256": queries.payload.sha256,
        "model_revision": documents.model_revision,
        "train_query_cap": verification["train_query_cap"],
        "grades": sorted({g for labels in judgments.values() for g in labels.values()}),
    }


def _dump(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_clean(item) for item in value]
    if isinstance(value, float | np.floating):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path)
    parser.add_argument("--envs", nargs="+", default=list(ENVIRONMENTS))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-queries", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--sample-size", type=int, default=200_000)
    parser.add_argument("--caliper", type=float, default=0.02)
    parser.add_argument("--max-query-pairs", type=int, default=2000)
    parser.add_argument("--batch-queries", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report-to", type=Path)
    args = parser.parse_args(argv)
    if args.report_to:
        run = latest_artifact(args.root, COMPLETE_EXPERIMENT, "all")
        write_report(run, args.report_to)
        print(args.report_to)
        return 0
    args.resolved = resolve_device(args.device)
    settings = {
        key: getattr(args, key)
        for key in ("envs", "max_queries", "repeats", "sample_size", "caliper", "max_query_pairs",
                    "batch_queries", "seed")
    }  # fmt: skip
    manifest = start_manifest(
        "p3-score-matching",
        settings,
        seed=args.seed,
        dataset_hash=NO_DATASET,
        device=args.resolved,
        determinism=seed_everything(args.seed),
    )
    sources: dict[str, Any] = {"settings": settings, "device": args.resolved}
    with ArtifactWriter(args.root, f"{COMPLETE_EXPERIMENT}/all", manifest) as writer:
        for env in args.envs:
            summary, source = _environment(writer, args.root, env, args)
            _dump(writer.path(f"{env}/summary.json"), _clean(summary))
            sources[env] = source
        _dump(writer.path("inputs.json"), _clean(sources))
        print(writer.commit())
    return 0


if __name__ == "__main__":
    sys.exit(main())
