"""``hyperdime analyze-mask-rankings``: rank the corpus under masks built from relevant documents.

A privileged offline diagnostic on published F2 artifacts (ADR-0016). Labels reach it only through
``load_training_judgments``, as in P3, and embeddings through ``load_unlabeled``. Multi-positive
training queries only. Results go to ``evaluation/P3-mask-ranking``, apart from earlier P3 runs.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from hyperdime.artifacts import latest_artifact
from hyperdime.cli.analyze_representations import (
    CANONICAL,
    ENVIRONMENTS,
    _clean,
    load_training_judgments,
    verify_inputs,
)
from hyperdime.cli.common import add_config_arguments, run_with_config
from hyperdime.contracts.config import RunConfig
from hyperdime.contracts.hashing import NO_DATASET, file_sha256
from hyperdime.contracts.manifests import ArtifactWriter, resolve_device, start_manifest
from hyperdime.contracts.schemas import DatasetManifest, EmbeddingArtifact, load_record
from hyperdime.contracts.seeding import seed_everything
from hyperdime.contracts.validation import ContractError
from hyperdime.data.prepare import MANIFEST, qrels_file
from hyperdime.evaluation.representation_engine import Clock, device_memory, reset_device_memory
from hyperdime.evaluation.representation_matching import select_stratified
from hyperdime.evaluation.representation_ranking import (
    BATCH_MASKS,
    CHUNK_DOCS,
    analyze_rankings,
)
from hyperdime.evaluation.representation_shared import TRANSFER_CELLS
from hyperdime.evaluation.runs import ratio_budgets
from hyperdime.space.inputs import load_unlabeled

COMMAND = "analyze-mask-rankings"
EXPERIMENT = "evaluation/P3-mask-ranking"
SHARED_EXPERIMENT = "evaluation/P3-shared-mask"
DEFAULT_BUDGETS = tuple(k for k in ratio_budgets() if k < 1024)
CELL_MINIMUM = {name: a + h for name, a, h in TRANSFER_CELLS if a is not None and h is not None}


def register(commands: Any) -> None:
    parser = commands.add_parser(
        COMMAND, help="P3 follow-up: full-corpus rankings under masks built from relevant documents"
    )
    add_config_arguments(parser)
    parser.add_argument("--envs", nargs="+", default=list(ENVIRONMENTS))
    parser.add_argument("--budgets", nargs="+", type=int, default=list(DEFAULT_BUDGETS))
    parser.add_argument(
        "--max-queries",
        type=int,
        help="bound on multi-positive queries per environment, stratified by positive count",
    )
    parser.add_argument("--chunk-docs", type=int, default=CHUNK_DOCS)
    parser.add_argument("--batch-masks", type=int, default=BATCH_MASKS)
    parser.add_argument("--no-own-masks", action="store_true")
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    return run_with_config(args, lambda config, output: _build(config, output, args))


def _shared_run(root: Path) -> Path | None:
    try:
        return latest_artifact(root, SHARED_EXPERIMENT, "all")
    except ContractError:
        return None


def _stored_regrets(
    shared: Path | None, env: str, ids: list[str], hashes: dict[str, str], counts: dict[str, int]
) -> tuple[Any, dict[str, Any]]:
    """Reader of the regrets the shared-mask run stored, once its inputs are shown to be these."""
    if shared is None or not (shared / env / "selected-queries.json").exists():
        return None, {"shared_mask_run": None}
    inputs = json.loads((shared / "inputs.json").read_text(encoding="utf-8"))[env]
    differing = [name for name, value in hashes.items() if inputs.get(name) != value]
    if differing:
        raise ContractError(f"shared-mask run {shared.name} used other inputs: {differing}")
    stored_ids = json.loads((shared / env / "selected-queries.json").read_text(encoding="utf-8"))
    tables: dict[str, Any] = {}
    rows: dict[str, dict[str, int]] = {"shared": {qid: i for i, qid in enumerate(stored_ids)}}
    for name, minimum in CELL_MINIMUM.items():
        rows[name] = {qid: i for i, qid in enumerate(q for q in stored_ids if counts[q] >= minimum)}

    def read(index: int, name: str) -> np.ndarray | None:
        qid = ids[index]
        if qid not in rows[name]:
            return None
        if name not in tables:
            file = (
                "shared-mean_norm.npy" if name == "shared" else f"transfer{name}-held_mean_norm.npy"
            )
            tables[name] = np.load(shared / env / file, mmap_mode="r")
        return np.asarray(tables[name][rows[name][qid]], dtype=np.float64)

    return read, {"shared_mask_run": shared.name}


def _environment(
    env: str,
    config: RunConfig,
    root: Path,
    writer: ArtifactWriter,
    device: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.perf_counter()
    clock = Clock(device)
    reset_device_memory(device)
    inputs = load_unlabeled(root, env)
    documents = load_record(EmbeddingArtifact, inputs.embeddings_dir / "documents.json")
    queries = load_record(EmbeddingArtifact, inputs.embeddings_dir / "queries.json")
    for record in (documents, queries):
        if record.model_revision != config.model.revision:
            raise ContractError(f"{inputs.embeddings_dir} holds revision {record.model_revision}")
    dataset = load_record(DatasetManifest, inputs.dataset_dir / MANIFEST)
    provenance = dataset.spec.get("provenance", CANONICAL)
    if provenance != CANONICAL:
        raise ContractError(f"{inputs.dataset_dir} is not a published F2 artifact: {provenance}")
    seed = config.run.seed
    verification = verify_inputs(inputs, documents, queries, dataset, seed)
    train_ids = inputs.splits.get("train", [])
    judgments = load_training_judgments(inputs.dataset_dir, train_ids)
    doc_rows = {doc: row for row, doc in enumerate(inputs.doc_ids)}
    positive: dict[str, np.ndarray] = {}
    grades: dict[str, np.ndarray] = {}
    for qid in train_ids:
        labels = judgments.get(qid, {})
        found = sorted((doc_rows[d], g) for d, g in labels.items() if d in doc_rows and g > 0)
        if found:
            positive[qid] = np.array([row for row, _ in found], dtype=np.int64)
            grades[qid] = np.array([g for _, g in found], dtype=np.int64)
    multi = {qid: rows.size for qid, rows in positive.items() if rows.size > 1}
    ids, strata, weights, table = select_stratified(multi, seed, args.max_queries or len(multi))
    hashes = {
        "qrels_train_sha256": file_sha256(inputs.dataset_dir / qrels_file("train")),
        "documents_sha256": documents.payload.sha256,
        "queries_sha256": queries.payload.sha256,
    }
    stored, source = _stored_regrets(_shared_run(root), env, ids, hashes, multi)
    summary = analyze_rankings(
        ids,
        lambda qid: np.asarray(inputs.query_vectors[inputs.query_rows[qid]]),
        inputs.documents,
        inputs.doc_ids,
        positive,
        grades,
        judgments,
        seed=seed,
        budgets=args.budgets,
        device=device,
        clock=clock,
        path=lambda name: writer.path(f"{env}/{name}"),
        own=not args.no_own_masks,
        chunk_docs=args.chunk_docs,
        batch_masks=args.batch_masks,
        stored_regret=stored,
    )
    directory = writer.path(f"{env}/selected-queries.json")
    directory.write_text(json.dumps(ids), encoding="utf-8")
    np.savez(writer.path(f"{env}/selection.npz"), strata=strata, weights=weights)
    summary["coverage"].update(
        {
            "training_queries": len(train_ids),
            "queries_with_positive": len(positive),
            "eligible_multi_positive": len(multi),
            "selection": table,
            "capped": len(ids) < len(multi),
            "selection_weights_all_one": bool(np.allclose(weights, 1.0)),
        }
    )
    sizes = {p.name: p.stat().st_size for p in directory.parent.iterdir() if p.is_file()}
    summary["cost"].update(
        {
            "total_seconds": time.perf_counter() - started,
            "device": device,
            "device_memory": device_memory(device),
            "phase_seconds": clock.seconds,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "artifact_bytes": sum(sizes.values()),
            "chunk_docs": args.chunk_docs,
            "batch_masks": args.batch_masks,
        }
    )
    (directory.parent / "summary.json").write_text(
        json.dumps(_clean(summary), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"{env}: {summary['cost']['total_seconds']:.0f}s", file=sys.stderr, flush=True)
    return {
        "provenance": provenance,
        "dataset": inputs.dataset_dir.name,
        "embeddings": inputs.embeddings_dir.name,
        **hashes,
        **source,
        "model_revision": documents.model_revision,
        "train_query_cap": verification["train_query_cap"],
        "grades": sorted({int(g) for found in grades.values() for g in found}),
    }


def _build(config: RunConfig, output: Path, args: argparse.Namespace) -> Path:
    if args.max_queries is not None and args.max_queries < 2:
        raise ValueError("--max-queries must be at least 2")
    if any(not 1 <= k < 1024 for k in args.budgets):
        raise ValueError("--budgets are retained dimensions below 1024; the full query is implicit")
    device = resolve_device(config.run.device)
    manifest = start_manifest(
        COMMAND,
        config,
        dataset_hash=NO_DATASET,
        device=device,
        determinism=seed_everything(config.run.seed),
    )
    sources: dict[str, Any] = {
        "experiment": "P3-mask-ranking",
        "environments": args.envs,
        "budgets": args.budgets,
        "max_queries": args.max_queries,
        "own_masks": not args.no_own_masks,
        "device": device,
        "seed": config.run.seed,
    }
    with ArtifactWriter(output, f"{EXPERIMENT}/all", manifest) as writer:
        for env in args.envs:
            sources[env] = _environment(env, config, output, writer, device, args)
        writer.path("inputs.json").write_text(
            json.dumps(_clean(sources), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return writer.commit()
