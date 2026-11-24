"""``hyperdime analyze-oracle-l2s``: exact positive masks beside L2S on the same queries.

A privileged, paired, in-sample diagnostic on published F2 artifacts (ADR-0017). Labels reach it
only through ``load_training_judgments``. The official L2S checkpoint must pass its provenance
checks first; ``--provenance-only`` stops there and scans nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hyperdime.artifacts import latest_artifact
from hyperdime.baselines.methods import selector_importance
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
from hyperdime.evaluation.oracle_comparison import (
    ARMS,
    Comparison,
    boundary_ties,
    compare_masks,
    save,
)
from hyperdime.evaluation.representation_engine import Clock, device_memory
from hyperdime.evaluation.representation_matching import select_stratified
from hyperdime.evaluation.representation_ranking import BATCH_MASKS, CHUNK_DOCS
from hyperdime.evaluation.representation_ranking import ROWS as STORED_ROWS
from hyperdime.evaluation.runs import QuerySet, evaluate_methods, ratio_budgets
from hyperdime.space.inputs import UnlabeledInputs, load_unlabeled
from hyperdime.training.selection import load_selector

COMMAND = "analyze-oracle-l2s"
EXPERIMENT = "evaluation/P3-oracle-l2s"
OFFICIAL = "selectors/official"
STORED_RUN = "evaluation/P3-mask-ranking"
BUDGETS = tuple(ratio_budgets())
STORED_ARMS = {
    "full": "full", "prefix": "prefix", "oracle": "shared", "held": "1-1", "own": "own",
}  # fmt: skip


def register(commands: Any) -> None:
    parser = commands.add_parser(
        COMMAND, help="P3: exact positive masks beside L2S on the same training queries"
    )
    add_config_arguments(parser)
    parser.add_argument("--envs", nargs="+", default=list(ENVIRONMENTS))
    parser.add_argument(
        "--max-queries", type=int, help="bound on queries per environment, seeded by query id"
    )
    parser.add_argument("--chunk-docs", type=int, default=CHUNK_DOCS)
    parser.add_argument("--batch-masks", type=int, default=BATCH_MASKS)
    parser.add_argument("--provenance-only", action="store_true")
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    return run_with_config(args, lambda config, output: _build(config, output, args))


def verify_l2s(
    root: Path,
    env: str,
    inputs: UnlabeledInputs,
    dataset: DatasetManifest,
    model_revision: str,
) -> dict[str, Any]:
    """What ties the checkpoint to this run's inputs; every name in ``checks`` must be true."""
    official = latest_artifact(root, OFFICIAL, env)
    manifest = read_manifest(official)
    selection = json.loads((official / "selection.json").read_text(encoding="utf-8"))
    record, _ = load_selector(official, "official")
    negatives = latest_artifact(root, "negatives", env)
    train = inputs.splits.get("train", [])
    other = [q for name, ids in inputs.splits.items() if name != "train" for q in ids]
    recorded = selection["inputs"]
    config = manifest.config
    checks = {
        "one published checkpoint": sum(
            1 for path in (root / OFFICIAL / env).iterdir() if path.is_dir()
        )
        == 1,
        "dataset is the F2 dataset read here": recorded["dataset"] == inputs.dataset_dir.name,
        "embeddings are the F2 embeddings read here": (
            recorded["embeddings"] == inputs.embeddings_dir.name
        ),
        "negatives are the latest F2 negatives": recorded["negatives"] == negatives.name,
        "checkpoint fingerprint is the dataset's": manifest.dataset_hash == dataset.fingerprint,
        "embeddings fingerprint is the dataset's": (
            read_manifest(inputs.embeddings_dir).dataset_hash == dataset.fingerprint
        ),
        "negatives fingerprint is the dataset's": (
            read_manifest(negatives).dataset_hash == dataset.fingerprint
        ),
        "published F2 dataset": dataset.spec.get("provenance", CANONICAL) == CANONICAL,
        "model revision matches": config["model"]["revision"] == model_revision,
        "environment matches": record.environment == env and config["data"]["environment"] == env,
        "payload hash matches": (
            file_sha256(official / record.payload.path) == record.payload.sha256
        ),
        "trained from a clean tree": manifest.git_dirty is False,
        "trained by train-selector": manifest.command == "train-selector",
        "official recipe": selection["recipe"] == "official",
        "split ids are disjoint": not set(train) & set(other),
        "per-domain, 1024 wide": record.kind == "per_domain" and record.dim == 1024,
    }
    return {
        "checks": checks,
        "failed": sorted(name for name, passed in checks.items() if not passed),
        "checkpoint": official.name,
        "training_git_commit": manifest.git_commit,
        "training_seed": manifest.seed,
        "chosen_cell": selection["chosen"],
        "selection_rule": "validation nDCG@10 at k = 307 (the checkpoint's recorded recipe)",
        "training_queries": len(train),
        "training_query_ids_sha256": hashlib.sha256("\n".join(sorted(train)).encode()).hexdigest(),
        "training_queries_source": (
            "the checkpoint stores no ids: its training queries are this dataset's train split, "
            "tied by the dataset fingerprint and the train-selector code path"
        ),
        "payload_sha256": record.payload.sha256,
        "inputs": recorded,
        "sample": "in-sample: L2S was fitted on these training queries",
    }


def select(
    counts: dict[str, int], seed: int, bound: int | None
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Query ids, cohort (1 for a single positive, 2 for several), sampling weight, and sampling
    stratum. A bound is split evenly between the cohorts, which are never pooled, with any surplus
    going to the other cohort; multi-positive queries are drawn by positive-count stratum."""
    singles = sorted(
        (q for q, c in counts.items() if c == 1),
        key=lambda q: hashlib.sha256(f"{seed}:{q}".encode()).hexdigest(),
    )
    multi = {q: c for q, c in counts.items() if c > 1}
    limit = len(counts) if bound is None else bound
    single_take = min(len(singles), limit // 2)
    multi_take = min(len(multi), limit - single_take)
    single_take = min(len(singles), limit - multi_take)
    chosen_single = singles[:single_take]
    ids = list(chosen_single)
    cohort = [1] * len(ids)
    sampled = [0] * len(ids)
    weights = [len(singles) / len(ids)] * len(ids) if ids else []
    if multi_take:
        multi_ids, multi_strata, multi_weights, _ = select_stratified(multi, seed, multi_take)
        ids.extend(multi_ids)
        cohort.extend([2] * len(multi_ids))
        sampled.extend(int(s) + 1 for s in multi_strata)
        weights.extend(float(w) for w in multi_weights)
    order = np.argsort(ids)
    return (
        [ids[at] for at in order],
        np.array(cohort, dtype=np.int8)[order],
        np.array(weights, dtype=np.float64)[order],
        np.array(sampled, dtype=np.int8)[order],
    )


def stored_parity(
    root: Path, env: str, ids: list[str], comparison: Comparison, budgets: tuple[int, ...]
) -> dict[str, Any]:
    """Ranks of the same masks in ADR-0016's run, compared exactly. That run excluded a document
    with the query's id, so a rank it charged here is taken off first."""
    try:
        run = latest_artifact(root, STORED_RUN, "all")
    except ContractError:
        return {"compared": 0, "reason": "no stored mask-ranking run under this output"}
    directory = run / env
    if not directory.is_dir():
        return {"compared": 0, "reason": "stored run has no such environment"}
    stored_ids = json.loads((directory / "selected-queries.json").read_text(encoding="utf-8"))
    query_of = {qid: at for at, qid in enumerate(stored_ids)}
    inputs = json.loads((run / "inputs.json").read_text(encoding="utf-8"))
    slot_of = {k: slot for slot, k in enumerate(inputs["budgets"])}
    with np.load(directory / "masks.npz") as masks, np.load(directory / "tracked.npz") as tracked:
        lookup = {
            (int(q), int(row), int(partition), int(positive), int(budget)): at
            for at, (q, row, partition, positive, budget) in enumerate(
                zip(
                    masks["query"],
                    masks["row"],
                    masks["partition"],
                    masks["positive"],
                    masks["budget"],
                    strict=True,
                )
            )
        }
        offsets, ranks = tracked["offsets"], tracked["rank"]
    here = comparison.masks
    here_offsets, here_ranks = comparison.tracked["offsets"], comparison.tracked["rank"]
    charged = comparison.tracked["collision_charged"]
    compared = mismatched = 0
    for mask in range(here["query"].size):
        arm = ARMS[int(here["arm"][mask])]
        if arm not in STORED_ARMS:
            continue
        qid = ids[int(here["query"][mask])]
        k = budgets[int(here["budget"][mask])]
        if qid not in query_of:
            continue
        slot = -1 if arm == "full" else slot_of.get(k)
        if slot is None:
            continue
        key = (
            query_of[qid],
            STORED_ROWS.index(STORED_ARMS[arm]),
            int(here["partition"][mask]),
            int(here["positive"][mask]),
            slot,
        )
        at = lookup.get(key)
        if at is None:
            continue
        compared += 1
        old = ranks[offsets[at] : offsets[at + 1]]
        span = slice(here_offsets[mask], here_offsets[mask + 1])
        mismatched += int(
            not np.array_equal(old, here_ranks[span] - charged[span].astype(np.int64))
        )
    return {"compared": compared, "mismatched": mismatched, "run": run.name}


def float32_gap(
    ids: list[str],
    vectors: torch.Tensor,
    judgments: dict[str, dict[str, int]],
    importance: torch.Tensor,
    inputs: UnlabeledInputs,
    comparison: Comparison,
    budgets: tuple[int, ...],
    device: str,
) -> dict[str, Any]:
    """The repository's float32 evaluation of L2S against this run's exact numbers."""
    evaluation = evaluate_methods(
        QuerySet("train", ids, vectors, judgments),
        {"l2s": importance},
        list(budgets),
        inputs.documents,
        inputs.doc_ids,
        environment=inputs.environment,
        device=device,
    )
    arrays = comparison.arrays
    arm = ARMS.index("l2s")
    largest = 0.0
    different = 0
    for result in evaluation.results:
        slot = budgets.index(result.k)
        for at, qid in enumerate(ids):
            metrics = (result.per_query or {}).get(qid)
            if metrics is None:
                continue
            gap = abs(float(metrics["ndcg@10"]) - float(arrays["ndcg@10"][arm, slot, at]))
            largest = max(largest, gap)
            different += int(gap > 1e-6)
    return {"largest_ndcg10_gap": largest, "query_budget_pairs_above_1e-6": different}


def _environment(
    env: str,
    config: RunConfig,
    root: Path,
    writer: ArtifactWriter | None,
    device: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.perf_counter()
    inputs = load_unlabeled(root, env)
    documents = load_record(EmbeddingArtifact, inputs.embeddings_dir / "documents.json")
    queries = load_record(EmbeddingArtifact, inputs.embeddings_dir / "queries.json")
    dataset = load_record(DatasetManifest, inputs.dataset_dir / MANIFEST)
    seed = config.run.seed
    verify_inputs(inputs, documents, queries, dataset, seed)
    provenance = verify_l2s(root, env, inputs, dataset, documents.model_revision)
    if provenance["failed"] or writer is None:
        return {"provenance": provenance}
    train_ids = inputs.splits["train"]
    judgments = load_training_judgments(inputs.dataset_dir, train_ids)
    doc_rows = {doc: row for row, doc in enumerate(inputs.doc_ids)}
    positive: dict[str, np.ndarray] = {}
    grades: dict[str, np.ndarray] = {}
    for qid in train_ids:
        found = sorted(
            (doc_rows[d], g) for d, g in judgments.get(qid, {}).items() if d in doc_rows and g > 0
        )
        if found:
            positive[qid] = np.array([row for row, _ in found], dtype=np.int64)
            grades[qid] = np.array([g for _, g in found], dtype=np.int64)
    ids, strata, weights, sampled = select(
        {q: rows.size for q, rows in positive.items()}, seed, args.max_queries
    )
    vectors = inputs.queries(ids)
    _, model = load_selector(latest_artifact(root, OFFICIAL, env), "official")
    importance = selector_importance(vectors, model).detach()
    clock = Clock(device)
    comparison = compare_masks(
        ids,
        lambda qid: np.asarray(inputs.query_vectors[inputs.query_rows[qid]]),
        inputs.documents,
        inputs.doc_ids,
        positive,
        grades,
        judgments,
        importance.numpy(),
        seed=seed,
        budgets=BUDGETS,
        device=device,
        clock=clock,
        chunk_docs=args.chunk_docs,
        batch_masks=args.batch_masks,
    )
    save(writer.path(f"{env}/per-query.npz"), comparison.arrays)
    save(writer.path(f"{env}/relationship.npz"), comparison.relation)
    save(writer.path(f"{env}/tracked.npz"), comparison.tracked)
    save(writer.path(f"{env}/masks.npz"), comparison.masks)
    np.savez(writer.path(f"{env}/selection.npz"), strata=strata, weights=weights, sampled=sampled)
    writer.path(f"{env}/selected-queries.json").write_text(json.dumps(ids), encoding="utf-8")
    collisions = sum(1 for qid in ids if qid in doc_rows)
    with clock.span("float32"):
        float32 = float32_gap(
            ids, vectors, {q: judgments[q] for q in ids}, importance, inputs, comparison, BUDGETS,
            device,
        )  # fmt: skip
    checks = {
        **comparison.checks,
        "l2s_boundary_ties": boundary_ties(importance, BUDGETS),
        "stored_run_parity": stored_parity(root, env, ids, comparison, BUDGETS),
        "float32_evaluation_gap": float32,
        "ids_colliding_with_a_document": collisions,
    }  # fmt: skip
    counts = np.array([positive[qid].size for qid in ids], dtype=np.int64)
    summary = {
        "coverage": {
            "training_queries": len(train_ids),
            "queries_with_positive": len(positive),
            "queries": len(ids),
            "single_positive": int((counts == 1).sum()),
            "multi_positive": int((counts > 1).sum()),
            "positives": int(counts.sum()),
            "documents": int(inputs.documents.shape[0]),
            "budgets": list(BUDGETS),
            "capped": args.max_queries is not None and len(ids) < len(positive),
            "weights_all_one": bool(np.allclose(weights, 1.0)),
        },
        "checks": checks,
        "provenance": provenance,
        "cost": {
            "total_seconds": time.perf_counter() - started,
            "device": device,
            "device_memory": device_memory(device),
            "phase_seconds": clock.seconds,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        },
    }
    writer.path(f"{env}/summary.json").write_text(
        json.dumps(_clean(summary), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"{env}: {summary['cost']['total_seconds']:.1f}s", file=sys.stderr, flush=True)
    return {
        "provenance": provenance,
        "dataset": inputs.dataset_dir.name,
        "embeddings": inputs.embeddings_dir.name,
        "qrels_train_sha256": file_sha256(inputs.dataset_dir / qrels_file("train")),
        "documents_sha256": documents.payload.sha256,
        "queries_sha256": queries.payload.sha256,
        "model_revision": documents.model_revision,
    }


def _build(config: RunConfig, output: Path, args: argparse.Namespace) -> Path:
    device = resolve_device(config.run.device)
    manifest = start_manifest(
        COMMAND,
        config,
        dataset_hash=NO_DATASET,
        device=device,
        determinism=seed_everything(config.run.seed),
    )
    sources: dict[str, Any] = {
        "experiment": "P3-oracle-l2s",
        "environments": args.envs,
        "budgets": list(BUDGETS),
        "max_queries": args.max_queries,
        "device": device,
        "seed": config.run.seed,
        "provenance_only": args.provenance_only,
    }
    kind = f"{EXPERIMENT}/provenance" if args.provenance_only else f"{EXPERIMENT}/all"
    with ArtifactWriter(output, kind, manifest) as writer:
        failed: list[str] = []
        for env in args.envs:
            sources[env] = _environment(
                env, config, output, None if args.provenance_only else writer, device, args
            )
            failed.extend(f"{env}: {name}" for name in sources[env]["provenance"]["failed"])
        writer.path("inputs.json").write_text(
            json.dumps(_clean(sources), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if failed and not args.provenance_only:
            raise ContractError(f"L2S provenance failed: {failed}")
        return writer.commit()
