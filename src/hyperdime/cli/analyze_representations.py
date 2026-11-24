"""``hyperdime analyze-representations``: experiment P3, per-dimension contributions of known pairs.

Descriptive, in the stored retrieval space, on training queries and training judgments only
(ADR-0014). Labels reach this command through ``load_training_judgments`` and nothing else: it opens
``qrels-train.jsonl`` by name and refuses a record of any other split, and the embeddings come from
``space.inputs.load_unlabeled``, which opens no judgments file.
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np

from hyperdime.cli.common import add_config_arguments, run_with_config
from hyperdime.cli.transfer_matrix import BUDGETS
from hyperdime.contracts.config import RunConfig
from hyperdime.contracts.hashing import NO_DATASET, file_sha256
from hyperdime.contracts.manifests import (
    ArtifactWriter,
    read_manifest,
    resolve_device,
    start_manifest,
)
from hyperdime.contracts.schemas import (
    DatasetManifest,
    EmbeddingArtifact,
    QrelsRecord,
    load_record,
    load_records,
)
from hyperdime.contracts.seeding import seed_everything
from hyperdime.contracts.validation import ContractError
from hyperdime.data.prepare import MANIFEST, qrels_file
from hyperdime.evaluation.representation import analyze_environment, select_queries
from hyperdime.evaluation.representation_controls import analyze_controls
from hyperdime.evaluation.representation_engine import (
    Clock,
    device_memory,
    reset_device_memory,
)
from hyperdime.space.inputs import UnlabeledInputs, load_unlabeled

COMMAND = "analyze-representations"
ENVIRONMENTS = ("scifact", "nfcorpus", "msmarco")
CANONICAL = "published F2 artifacts"
NORM_TOLERANCE = 0.01
NORM_SAMPLE = 10_000


def register(commands: Any) -> None:
    parser = commands.add_parser(
        COMMAND, help="experiment P3: per-dimension contributions of known relevant pairs"
    )
    add_config_arguments(parser)
    parser.add_argument("--envs", nargs="+", default=list(ENVIRONMENTS))
    parser.add_argument(
        "--max-queries", type=int, help="seeded bound on training queries per environment"
    )
    parser.add_argument(
        "--budgets",
        nargs="+",
        type=int,
        default=list(BUDGETS),
        help="checkpoints for summary tables; curves always cover every k from 1 to 1024",
    )
    parser.add_argument("--stability-draws", type=int, default=20)
    parser.add_argument("--across-pairs", type=int, default=2000)
    parser.add_argument(
        "--reference-samples",
        type=int,
        default=5,
        help="repeated samples of corpus-reference documents per query",
    )
    parser.add_argument(
        "--max-query-pairs",
        type=int,
        default=2000,
        help="document pairs per query above which agreement uses a seeded sample of pairs",
    )
    parser.add_argument(
        "--batch-pairs",
        type=int,
        default=8192,
        help="target pairs per batch on the device (run.device); whole queries, so one large "
        "query may exceed it. Changes speed and memory, not results",
    )
    parser.add_argument(
        "--allow-spike-inputs",
        action="store_true",
        help="accept inputs converted from F0 spike caches; such a run is validation only",
    )
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    return run_with_config(args, lambda config, output: _build(config, output, args))


def load_training_judgments(dataset_dir: Path, train_ids: list[str]) -> dict[str, dict[str, int]]:
    """Graded judgments of the training split; the only place this experiment reads a label."""
    allowed = set(train_ids)
    found: dict[str, dict[str, int]] = {}
    for record in load_records(QrelsRecord, dataset_dir / qrels_file("train")):
        if record.split != "train" or record.query_id not in allowed:
            raise ContractError(
                f"{dataset_dir}: judgment of query {record.query_id!r} in split "
                f"{record.split!r} found among the training judgments"
            )
        found.setdefault(record.query_id, {})[record.doc_id] = record.relevance
    return found


def _build(config: RunConfig, output: Path, args: argparse.Namespace) -> Path:
    if not all(1 <= k <= 1024 for k in args.budgets):
        raise ValueError("budgets must be in [1, 1024]")
    if args.max_queries is not None and args.max_queries < 2:
        raise ValueError("--max-queries must be at least 2 for split-half stability")
    if args.reference_samples < 1 or args.max_query_pairs < 1 or args.batch_pairs < 1:
        raise ValueError(
            "--reference-samples, --max-query-pairs, and --batch-pairs must be at least 1"
        )
    budgets = sorted(set(args.budgets))
    seed = config.run.seed
    device = resolve_device(config.run.device)
    manifest = start_manifest(
        COMMAND,
        config,
        dataset_hash=NO_DATASET,
        device=device,
        determinism=seed_everything(seed),
    )
    sources: dict[str, Any] = {
        "experiment": "P3",
        "environments": args.envs,
        "settings": {
            "max_queries": args.max_queries,
            "budgets": budgets,
            "stability_draws": args.stability_draws,
            "across_pairs": args.across_pairs,
            "reference_samples": args.reference_samples,
            "max_query_pairs": args.max_query_pairs,
            "allow_spike_inputs": args.allow_spike_inputs,
            "device": device,
            "batch_pairs": args.batch_pairs,
            "seed": seed,
        },
    }
    with ArtifactWriter(output, "evaluation/P3/all", manifest) as writer:
        for env in args.envs:
            sources[env] = _environment(env, config, output, writer, budgets, args, device)
        kinds = {sources[env]["provenance"] == CANONICAL for env in args.envs}
        if len(kinds) > 1:
            raise ContractError("one run must not mix published F2 inputs with other inputs")
        _write_json(writer, "inputs.json", sources)
        return writer.commit()


def _environment(
    env: str,
    config: RunConfig,
    root: Path,
    writer: ArtifactWriter,
    budgets: list[int],
    args: argparse.Namespace,
    device: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    clock = Clock(device)
    reset_device_memory(device)
    inputs: UnlabeledInputs = load_unlabeled(root, env)
    documents_record = load_record(EmbeddingArtifact, inputs.embeddings_dir / "documents.json")
    queries_record = load_record(EmbeddingArtifact, inputs.embeddings_dir / "queries.json")
    for record in (documents_record, queries_record):
        if record.model_revision != config.model.revision:
            raise ContractError(
                f"{inputs.embeddings_dir} holds revision {record.model_revision}, "
                f"not the configured {config.model.revision}"
            )
    dataset = load_record(DatasetManifest, inputs.dataset_dir / MANIFEST)
    provenance = dataset.spec.get("provenance", CANONICAL)
    if provenance != CANONICAL and not args.allow_spike_inputs:
        raise ContractError(
            f"{inputs.dataset_dir} is not a published F2 artifact ({provenance}); pass "
            "--allow-spike-inputs to run it as validation, or use the F2 artifacts"
        )
    verification = verify_inputs(inputs, documents_record, queries_record, dataset, config.run.seed)
    train_ids = inputs.splits.get("train", [])
    judgments = load_training_judgments(inputs.dataset_dir, train_ids)
    # F2 drops a query without a positive; keep the guard for any other input.
    eligible = [qid for qid in train_ids if any(g > 0 for g in judgments.get(qid, {}).values())]
    selected = select_queries(eligible, config.run.seed, args.max_queries)
    doc_rows = {doc_id: row for row, doc_id in enumerate(inputs.doc_ids)}
    load_seconds = time.perf_counter() - started
    summary = analyze_environment(
        selected,
        lambda qid: np.asarray(inputs.query_vectors[inputs.query_rows[qid]]),
        inputs.documents,
        doc_rows,
        {qid: judgments[qid] for qid in selected},
        budgets,
        seed=config.run.seed,
        stability_draws=args.stability_draws,
        path=lambda name: writer.path(f"{env}/{name}"),
        device=device,
        batch_pairs=args.batch_pairs,
        clock=clock,
    )
    curves = analyze_controls(
        selected,
        lambda qid: np.asarray(inputs.query_vectors[inputs.query_rows[qid]]),
        inputs.documents,
        doc_rows,
        {qid: judgments[qid] for qid in selected},
        budgets,
        np.load(writer.path(f"{env}/pair-order.npy"), mmap_mode="r"),
        seed=config.run.seed,
        reference_samples=args.reference_samples,
        max_query_pairs=args.max_query_pairs,
        stability_draws=args.stability_draws,
        across_pairs=args.across_pairs,
        path=lambda name: writer.path(f"{env}/{name}"),
        device=device,
        batch_pairs=args.batch_pairs,
        clock=clock,
    )
    _write_json(writer, f"{env}/curves.json", _clean(curves))
    staged = sorted(p for p in writer.path(f"{env}/pairs.jsonl").parent.iterdir() if p.is_file())
    summary["coverage"]["training_queries"] = len(train_ids)
    summary["coverage"]["training_queries_without_positive"] = len(train_ids) - len(eligible)
    summary["cost"].update({
        "load_seconds": load_seconds,
        "total_seconds": time.perf_counter() - started,
        "device": device,
        "batch_pairs": args.batch_pairs,
        "device_memory": device_memory(device),
        "phase_seconds": clock.seconds,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "artifact_bytes": {path.name: path.stat().st_size for path in staged},
    })  # fmt: skip
    _write_json(writer, f"{env}/summary.json", _clean(summary))
    return {
        "provenance": provenance,
        "verification": verification,
        "dataset": inputs.dataset_dir.name,
        "dataset_fingerprint": dataset.fingerprint,
        "split_seed": dataset.split_seed,
        "validation_fraction": dataset.validation_fraction,
        "qrels_train_sha256": file_sha256(inputs.dataset_dir / qrels_file("train")),
        "embeddings": inputs.embeddings_dir.name,
        "documents_sha256": documents_record.payload.sha256,
        "queries_sha256": queries_record.payload.sha256,
        "dtype": documents_record.dtype,
        "normalized": documents_record.normalized,
        "pooling": documents_record.pooling,
        "model_name": documents_record.model_name,
        "model_revision": documents_record.model_revision,
    }


def verify_inputs(
    inputs: UnlabeledInputs,
    documents: EmbeddingArtifact,
    queries: EmbeddingArtifact,
    dataset: DatasetManifest,
    seed: int,
) -> dict[str, Any]:
    """Check what the analysis relies on, and record it; opens no judgments file.

    Hashes both payloads (``load_embeddings`` checks only the shape), the 1024 width, the split
    bookkeeping against the dataset manifest, and unit norms on every query and a seeded sample of
    documents. A failed check stops the run.
    """
    for name, record in (("documents", documents), ("queries", queries)):
        found = file_sha256(inputs.embeddings_dir / record.payload.path)
        if found != record.payload.sha256:
            raise ContractError(f"{inputs.embeddings_dir}: {name} payload hash does not match")
        if not record.normalized:
            raise ContractError(f"{inputs.embeddings_dir}: {name} are not marked normalized")
    if inputs.documents.shape[1] != 1024 or inputs.query_vectors.shape[1] != 1024:
        raise ContractError(f"{inputs.embeddings_dir}: expected D = 1024")
    seen: set[str] = set()
    for split, ids in inputs.splits.items():
        missing = [qid for qid in ids if qid not in inputs.query_rows]
        if missing:
            raise ContractError(f"{split} query {missing[0]} has no embedding")
        if seen & set(ids):
            raise ContractError(f"split {split} shares a query with another split")
        seen |= set(ids)
        expected = dataset.num_queries.get(split)
        if expected is not None and expected != len(ids):
            raise ContractError(f"split {split}: {len(ids)} queries, manifest says {expected}")
    rng = np.random.default_rng([seed, 7])
    picks = np.sort(rng.permutation(documents.num_rows)[: min(NORM_SAMPLE, documents.num_rows)])
    train_rows = sorted(inputs.query_rows[qid] for qid in inputs.splits.get("train", []))
    norms = {
        "documents": np.linalg.norm(np.asarray(inputs.documents[picks], np.float64), axis=1),
        "train_queries": np.linalg.norm(
            np.asarray(inputs.query_vectors[train_rows], np.float64), axis=1
        ),
    }
    for name, values in norms.items():
        if values.size and float(np.abs(values - 1.0).max()) > NORM_TOLERANCE:
            raise ContractError(f"{name}: a norm is further than {NORM_TOLERANCE} from 1")
    manifest = read_manifest(inputs.embeddings_dir)
    settings = manifest.config.get("embedding") or manifest.config.get("model") or {}
    return {
        "payload_hashes_match": True,
        "dim": 1024,
        "documents": documents.num_rows,
        "queries_per_split": {split: len(ids) for split, ids in inputs.splits.items()},
        "train_query_cap": dataset.spec.get("train_query_cap"),
        "embeddings_made_from_this_dataset": True,  # load_unlabeled refuses anything else
        "norm_sampled_documents": int(picks.size),
        "document_norm_range": [float(norms["documents"].min()), float(norms["documents"].max())],
        "train_query_norm_range": [
            float(norms["train_queries"].min()),
            float(norms["train_queries"].max()),
        ],
        "instruction": queries.instruction,
        "encoding_settings": settings,
        "embedding_commit": manifest.git_commit,
        "embedding_git_dirty": manifest.git_dirty,
    }


def _clean(value: Any) -> Any:
    """JSON without NaN: an undefined summary value is ``null``."""
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


def _write_json(writer: ArtifactWriter, name: str, payload: Any) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    writer.path(name).write_text(text, encoding="utf-8")
