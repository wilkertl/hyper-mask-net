"""``hyperdime analyze-shared-masks``: what one shared mask costs a query's relevant documents.

A follow-up to P3 on published F2 artifacts (ADR-0015). Labels reach it only through
``load_training_judgments``, as in P3, and embeddings through ``load_unlabeled``. Multi-positive
training queries only. Results go to ``evaluation/P3-shared-mask``, apart from P3's own runs.
"""

from __future__ import annotations

import argparse
import json
import resource
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
from hyperdime.contracts.manifests import (
    ArtifactWriter,
    read_manifest,
    resolve_device,
    start_manifest,
)
from hyperdime.contracts.schemas import DatasetManifest, EmbeddingArtifact, load_record
from hyperdime.contracts.seeding import seed_everything
from hyperdime.contracts.validation import ContractError
from hyperdime.data.prepare import MANIFEST, qrels_file
from hyperdime.evaluation.representation_engine import (
    Clock,
    device_memory,
    reset_device_memory,
)
from hyperdime.evaluation.representation_matching import select_stratified
from hyperdime.evaluation.representation_shared import analyze_shared_masks
from hyperdime.space.inputs import load_unlabeled

COMMAND = "analyze-shared-masks"
EXPERIMENT = "evaluation/P3-shared-mask"


def register(commands: Any) -> None:
    parser = commands.add_parser(
        COMMAND, help="P3 follow-up: cost and transfer of one shared mask among a query's positives"
    )
    add_config_arguments(parser)
    parser.add_argument("--envs", nargs="+", default=list(ENVIRONMENTS))
    parser.add_argument(
        "--max-queries",
        type=int,
        help="pilot bound on multi-positive queries, stratified by positive count",
    )
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    return run_with_config(args, lambda config, output: _build(config, output, args))


def _canonical(root: Path) -> Path | None:
    """The latest published P3 run, if any: orderings and earlier summaries to check against."""
    try:
        return latest_artifact(root, "evaluation/P3", "all")
    except ContractError:
        return None


def _cross_check(
    run_dir: Path,
    ids: list[str],
    all_queries: list[str],
    budgets_run: Path,
    env: str,
    equal_weights: bool,
) -> dict[str, Any]:
    """This run's regret at P3's checkpoints against what P3 recorded for the same queries. P3's
    mean is grade-weighted and this one is not, so they compare only when all grades are equal."""
    source = budgets_run / env
    with np.load(source / "query-stats.npz") as found:
        mean, worst = found["aggregate_mean_regret"], found["aggregate_worst_regret"]
    budgets = json.loads((source / "summary.json").read_text(encoding="utf-8"))["budgets"]
    at = np.array([k - 1 for k in budgets])
    rows = {qid: i for i, qid in enumerate(all_queries)}
    picks = np.array([rows[qid] for qid in ids])
    here_mean = np.load(run_dir / "shared-mean_norm.npy")[:, at]
    here_worst = np.load(run_dir / "shared-max_norm.npy")[:, at]
    return {
        "source_run": budgets_run.name,
        "checkpoints": budgets,
        "largest_mean_regret_difference": float(np.nanmax(np.abs(here_mean - mean[picks])))
        if equal_weights
        else None,
        "largest_worst_regret_difference": float(np.nanmax(np.abs(here_worst - worst[picks]))),
    }


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
    judged: dict[str, set[int]] = {}
    for qid in train_ids:
        labels = judgments.get(qid, {})
        found = sorted((doc_rows[d], g) for d, g in labels.items() if d in doc_rows and g > 0)
        if found:
            positive[qid] = np.array([row for row, _ in found], dtype=np.int64)
            grades[qid] = np.array([g for _, g in found], dtype=np.int64)
            judged[qid] = {doc_rows[d] for d in labels if d in doc_rows}
    multi = {qid: rows.size for qid, rows in positive.items() if rows.size > 1}
    ids, _, weights, table = select_stratified(multi, seed, args.max_queries or len(multi))
    canonical = _canonical(root)
    all_queries = sorted(positive)
    order_lookup = None
    if canonical is not None and (canonical / env / "pair-order.npy").exists():
        stored = np.load(canonical / env / "pair-order.npy", mmap_mode="r")
        offsets = np.concatenate([[0], np.cumsum([positive[q].size for q in all_queries])])
        where = {qid: i for i, qid in enumerate(all_queries)}

        def order_lookup(qid: str) -> np.ndarray:
            at = where[qid]
            return np.asarray(stored[offsets[at] : offsets[at + 1]])

    load_seconds = time.perf_counter() - started
    summary = analyze_shared_masks(
        ids,
        lambda qid: np.asarray(inputs.query_vectors[inputs.query_rows[qid]]),
        inputs.documents,
        {qid: positive[qid] for qid in ids},
        {qid: grades[qid] for qid in ids},
        {qid: judged[qid] for qid in ids},
        seed=seed,
        device=device,
        clock=clock,
        path=lambda name: writer.path(f"{env}/{name}"),
        canonical_order=order_lookup,
    )
    staged = writer.path(f"{env}/selected-queries.json")
    staged.write_text(json.dumps(ids), encoding="utf-8")
    context: dict[str, Any] = {}
    if canonical is not None and (canonical / env / "control-stats.npz").exists():
        summary["checks"]["canonical"] = _cross_check(
            staged.parent,
            ids,
            all_queries,
            canonical,
            env,
            len({int(g) for qid in ids for g in grades[qid]}) == 1,
        )
        with np.load(canonical / env / "control-stats.npz") as found:
            cosine = found["document_cosine_positive"]
        rows = {qid: i for i, qid in enumerate(all_queries)}
        picks = np.array([rows[qid] for qid in ids])
        np.save(writer.path(f"{env}/positive-cosine.npy"), cosine[picks])
        context["positive_cosine_source"] = canonical.name
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
    sizes = {p.name: p.stat().st_size for p in staged.parent.iterdir() if p.is_file()}
    summary["cost"].update(
        {
            "load_seconds": load_seconds,
            "total_seconds": time.perf_counter() - started,
            "device": device,
            "device_memory": device_memory(device),
            "phase_seconds": clock.seconds,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "artifact_bytes": sum(sizes.values()),
        }
    )
    (staged.parent / "summary.json").write_text(
        json.dumps(_clean(summary), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "provenance": provenance,
        "dataset": inputs.dataset_dir.name,
        "embeddings": inputs.embeddings_dir.name,
        "qrels_train_sha256": file_sha256(inputs.dataset_dir / qrels_file("train")),
        "documents_sha256": documents.payload.sha256,
        "queries_sha256": queries.payload.sha256,
        "model_revision": documents.model_revision,
        "train_query_cap": verification["train_query_cap"],
        "grades": sorted({int(g) for found in grades.values() for g in found}),
        "canonical_p3_run": canonical.name if canonical else None,
        "embedding_commit": read_manifest(inputs.embeddings_dir).git_commit,
        **context,
    }


def _build(config: RunConfig, output: Path, args: argparse.Namespace) -> Path:
    if args.max_queries is not None and args.max_queries < 2:
        raise ValueError("--max-queries must be at least 2")
    device = resolve_device(config.run.device)
    manifest = start_manifest(
        COMMAND,
        config,
        dataset_hash=NO_DATASET,
        device=device,
        determinism=seed_everything(config.run.seed),
    )
    sources: dict[str, Any] = {
        "experiment": "P3-shared-mask",
        "environments": args.envs,
        "max_queries": args.max_queries,
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
