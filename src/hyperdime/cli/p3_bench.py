"""``hyperdime p3-bench``: the F12 benchmark's initial path (ADR-0018, specs/013).

Subcommands, in gate order:

- ``split-provenance``: verify ``p3-query-split-v1`` against the F2 queries and attach a run
  manifest (ids unchanged).
- ``freeze-protocol``: SHA-256 record of ADR-0018 and the F12 docs; required before the probe.
- ``official``: one isolated authors' Learning2Select process on fit queries (labels, optional
  training). Used by the probe (``--purpose probe``) and for the single reference.
- ``probe``: fit-only cost probe (exact scoring, rank-guided search, S1 steps).
- ``freeze-resources``: apply ADR-0018's resource rule to the probe and record the realized values.
- ``official-parity``: the authors' evaluator against the common exact evaluator on fit queries.

Corpora above ``LARGE_CORPUS`` documents (MS MARCO) never hold a float64 copy of the corpus: exact
ranks come from :class:`~hyperdime.evaluation.benchmark.CorpusScorer`, the authors' label
generation runs on the CPU over a memory-mapped float32 copy in their corpus order with its
candidate pools recorded, and the rank-guided search proposes swaps from a shortlist and accepts
each only after an exact full-corpus check (ADR-0018). Their sizes follow ADR-0018's MS MARCO
priority order, and the resource record also holds the sizes that apply if an escalation removes
the rank-guided arm, so that choice is fixed before any teacher is built.

Only fit queries and fit judgments are loaded; tune and evaluation judgments are discarded at read
time and test judgments are never opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hyperdime.baselines.l2s_official import (
    DEFAULT_CLONE,
    L2S_COMMIT,
    PUBLISHED,
    PUBLISHED_SOURCE,
    SEED,
    authors_order,
    import_official,
    run_official,
    transfer_flag,
)
from hyperdime.contracts.hashing import file_sha256
from hyperdime.contracts.manifests import ArtifactWriter, read_manifest, start_manifest
from hyperdime.contracts.validation import ContractError
from hyperdime.data.query_split import (
    MANIFEST_SHA256,
    SPLIT_DIR,
    hash_order,
    load_query_split,
    verify_against_source,
)
from hyperdime.evaluation.benchmark import (
    AUTHORS_GRID,
    DECILES,
    PRIMARY,
    CorpusScorer,
    masked_query,
    masked_scores,
    positive_ranks,
    query_measures,
)
from hyperdime.evaluation.teachers import (
    contrastive_mask,
    make_objective,
    positive_mask,
    positive_mean,
    rank_guided_search,
    shortlist_search,
)

COMMAND = "p3-bench"
MODEL_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
PROTOCOL_FILES = (
    "docs/decisions/ADR-0018-p3-selector-benchmark.md",
    "specs/013-p3-selector-benchmark/spec.md",
    "specs/013-p3-selector-benchmark/plan.md",
    "specs/013-p3-selector-benchmark/research.md",
    "specs/013-p3-selector-benchmark/tasks.md",
)
PROBE_LABEL_SIZES = (256, 1024, 4096)
PROBE_SEARCH_SIZES = (16, 64, 256)
PROBE_CAPS = (8, 32, 128)
BUDGET_HOURS = {"scifact": (1.0, 8.0), "nfcorpus": (1.0, 16.0), "msmarco": (3.0, 72.0)}
SAFETY = 1.5
STOP_SHARE = 0.95
LARGE_CORPUS = 1_000_000
# ADR-0018 MS MARCO priority order; shortlists are tried larger first (better proposals).
MS_N_EVAL = (7_500, 4_000, 2_000)
MS_N_RG = (35_000, 16_000, 8_000, 4_000)
MS_N_TUNE = (7_500, 4_000, 2_000)
MS_SHORTLISTS = (10_000, 1_000)
SCORER_CHUNK = 1 << 16


def is_large(n_documents: int) -> bool:
    return n_documents > LARGE_CORPUS


# ----------------------------------------------------------------------------- inputs


@dataclass
class FitInputs:
    corpus: str
    query_ids: list[str]  # fit ids in ADR-0018 hash order (possibly a prefix)
    queries: np.ndarray  # [Q, D] float16
    doc_ids: list[str]
    documents: np.ndarray  # [N, D] float16, F2 row order
    lengths: list[int]
    qrels: dict[str, dict[str, int]]
    dataset_run: str
    embedding_run: str


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle]


def _embedding_dir(root: Path, corpus: str) -> Path:
    runs = sorted(
        p
        for p in (root / "embeddings" / corpus).iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )
    if len(runs) != 1:
        raise ContractError(
            f"expected one embedding run for {corpus}, found {[p.name for p in runs]}"
        )
    return runs[0]


def load_fit_inputs(root: Path, corpus: str, limit: int | None = None) -> FitInputs:
    split = load_query_split(root, corpus)
    order = hash_order(corpus, "fit", split.ids["fit"])
    if limit is not None:
        order = order[:limit]
    wanted = set(order)
    dataset = root / "datasets" / corpus / split.source_dataset_run
    emb = _embedding_dir(root, corpus)
    doc_ids = json.loads((emb / "documents.json").read_text())["ids"]
    query_rows = {
        qid: i for i, qid in enumerate(json.loads((emb / "queries.json").read_text())["ids"])
    }
    qrels: dict[str, dict[str, int]] = {}
    for name in ("train", "validation"):  # never qrels-test
        for record in _jsonl(dataset / f"qrels-{name}.jsonl"):
            if record["query_id"] in wanted:
                qrels.setdefault(record["query_id"], {})[record["doc_id"]] = int(
                    record["relevance"]
                )
    lengths_by_id = {}
    file_order = []
    for record in _jsonl(dataset / "documents.jsonl"):
        lengths_by_id[record["doc_id"]] = len(record["title"] + record["text"])
        file_order.append(record["doc_id"])
    if file_order != doc_ids:
        raise ContractError("documents.jsonl order differs from the embedding rows")
    queries = np.load(emb / "queries.npy", mmap_mode="r")
    return FitInputs(
        corpus=corpus,
        query_ids=order,
        queries=np.asarray(queries[[query_rows[q] for q in order]]),
        doc_ids=doc_ids,
        documents=np.load(emb / "documents.npy", mmap_mode="r"),
        lengths=[lengths_by_id[d] for d in doc_ids],
        qrels=qrels,
        dataset_run=split.source_dataset_run,
        embedding_run=emb.name,
    )


def _peaks(device: str) -> dict[str, float]:
    peaks = {"peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
    if device.startswith("cuda"):
        peaks["peak_gpu_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
        peaks["peak_gpu_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
    return peaks


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _require_record(path: Path, kind: str) -> dict[str, Any]:
    if not path.is_file():
        raise ContractError(f"missing {kind} freeze record {path}")
    record = json.loads(path.read_text())
    if record.get("kind") != kind:
        raise ContractError(f"{path} is not a {kind} freeze record")
    return dict(record)


def _check_protocol(record_path: Path, repo: Path) -> dict[str, Any]:
    record = _require_record(record_path, "protocol")
    for name, digest in record["files"].items():
        if file_sha256(repo / name) != digest:
            raise ContractError(f"{name} changed since the protocol freeze")
    return record


# ----------------------------------------------------------------------------- split provenance


def cmd_split_provenance(args: argparse.Namespace) -> int:
    split = load_query_split(args.root, args.corpus)
    queries_path = args.root / "datasets" / args.corpus / split.source_dataset_run / "queries.jsonl"
    report = verify_against_source(split, _jsonl(queries_path), file_sha256(queries_path))
    report["manifest_sha256"] = split.manifest_sha256
    report["source_dataset_run"] = split.source_dataset_run
    report["researcher_verification"] = "independent host check by the researcher, 2026-11-12"
    manifest = start_manifest(
        f"{COMMAND} split-provenance",
        {"corpus": args.corpus, "split_manifest_sha256": split.manifest_sha256},
        dataset_hash=split.manifest_sha256,
        model_revision=MODEL_REVISION,
        seed=20261112,
        device="cpu",
    )
    with ArtifactWriter(args.root, f"{SPLIT_DIR}/provenance/{args.corpus}", manifest) as writer:
        _write_json(writer.path("verification.json"), report)
    print(
        json.dumps(
            {"run_id": manifest.run_id, "passed": report["passed"], "checks": report["checks"]}
        )
    )
    return 0 if report["passed"] else 1


# ----------------------------------------------------------------------------- protocol freeze


def cmd_freeze_protocol(args: argparse.Namespace) -> int:
    out = args.out
    if out.exists():
        raise ContractError(f"{out} exists; freeze records are never overwritten")
    record = {
        "kind": "protocol",
        "created_at": datetime.now(UTC).isoformat(),
        "files": {name: file_sha256(args.repo / name) for name in PROTOCOL_FILES},
        "l2s_commit": L2S_COMMIT,
        "split_manifest_sha256": MANIFEST_SHA256,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_json(out, record)
    print(json.dumps({"record": str(out), "sha256": file_sha256(out)}))
    return 0


# ----------------------------------------------------------------------------- official run


def cmd_official(args: argparse.Namespace) -> int:
    if args.purpose == "reference":
        resources = _require_record(args.resource_record, "resources")
        limit = resources["realized"][args.corpus]["n_fit"]
    else:
        limit = args.limit
    inputs = load_fit_inputs(args.root, args.corpus, limit)
    temperature, epochs = PUBLISHED[args.corpus]
    order = authors_order(inputs.lengths)
    device = args.device
    torch.use_deterministic_algorithms(False)
    manifest = start_manifest(
        f"{COMMAND} official --purpose {args.purpose}",
        {
            "corpus": args.corpus,
            "purpose": args.purpose,
            "n_fit": len(inputs.query_ids),
            "temperature": temperature,
            "epochs": epochs,
            "published_source": PUBLISHED_SOURCE.get(args.corpus, args.corpus),
            "l2s_commit": L2S_COMMIT,
            "train": not args.labels_only,
            "fit_ids_sha256": hashlib.sha256("\n".join(inputs.query_ids).encode()).hexdigest(),
        },
        dataset_hash=file_sha256(args.root / "p3-query-split-v1" / "manifest.json"),
        model_revision=MODEL_REVISION,
        seed=SEED,
        device=device,
        determinism={
            "torch": torch.__version__,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        },
    )
    module = import_official(args.clone)
    corpus_ids = [inputs.doc_ids[row] for row in order]
    large = is_large(len(order))
    if large and device != "cpu":
        raise ContractError("on a large corpus the authors' label generation runs on the CPU")
    documents = _authors_documents(args.root, inputs, order)
    queries = torch.from_numpy(inputs.queries.astype(np.float32))
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    began = time.perf_counter()
    result = run_official(
        module,
        inputs.query_ids,
        queries,
        corpus_ids,
        documents,
        inputs.qrels,
        temperature=temperature,
        epochs=epochs,
        device=device,
        train=not args.labels_only,
        record_pools=large,
    )
    seconds = time.perf_counter() - began
    negatives_f2 = np.asarray(order)[result.negatives]
    summary = {
        "n_fit": len(inputs.query_ids),
        "n_valid": len(result.valid_query_ids),
        "fallback_queries": int(result.fallback.sum()),
        "seconds": {**result.seconds, "total": seconds},
        "final_train_loss": result.final_train_loss,
        "device": device,
        "dataset_run": inputs.dataset_run,
        "embedding_run": inputs.embedding_run,
        "recorded_pools": large,
        **_peaks(device),
    }
    if args.corpus in PUBLISHED_SOURCE:  # US1/AC4: parameters published for another corpus
        summary["transfer_flag"] = transfer_flag(args.corpus)
    kind = (
        f"selectors/l2s-official/{args.corpus}"
        if args.purpose == "reference"
        else f"evaluation/P3-bench/probe-official/{args.corpus}"
    )
    with ArtifactWriter(args.root, kind, manifest) as writer:
        _write_json(writer.path("summary.json"), summary)
        _write_json(writer.path("valid_query_ids.json"), result.valid_query_ids)
        np.save(writer.path("negatives_f2_rows.npy"), negatives_f2)
        np.save(writer.path("fallback.npy"), result.fallback)
        torch.save(result.targets, writer.path("targets.pt"))
        torch.save(result.rng_after_labels, writer.path("rng_after_labels.pt"))
        if result.state_dict:
            torch.save(result.state_dict, writer.path("state_dict.pt"))
    print(json.dumps({"run_id": manifest.run_id, **summary}))
    return 0


def _authors_documents(root: Path, inputs: FitInputs, order: list[int]) -> torch.Tensor:
    """Float32 documents in the authors' order. A large corpus is written once, chunk by chunk,
    to a scratch file and memory-mapped copy-on-write, so the authors' normalized copy is the only
    full anonymous float32 matrix in the process."""
    if not is_large(len(order)):
        return torch.from_numpy(np.asarray(inputs.documents, dtype=np.float32)[order])
    scratch = root / "scratch" / "p3-bench" / inputs.corpus
    path = scratch / f"authors-order-f32-{inputs.embedding_run}.npy"
    if not path.exists():
        scratch.mkdir(parents=True, exist_ok=True)
        partial = scratch / f"{path.stem}.partial.npy"
        shape = (len(order), inputs.documents.shape[1])
        out = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32, shape=shape)
        index = np.asarray(order)
        for start in range(0, len(order), 1 << 16):
            rows = index[start : start + (1 << 16)]
            block = np.asarray(inputs.documents[np.sort(rows)], dtype=np.float32)
            out[start : start + len(rows)] = block[np.argsort(np.argsort(rows))]
        out.flush()
        del out
        partial.rename(path)
    return torch.from_numpy(np.load(path, mmap_mode="c"))


# ----------------------------------------------------------------------------- probe


def _positives(
    inputs: FitInputs, qid: str, doc_rows: dict[str, int]
) -> tuple[list[int], list[float]]:
    rows, grades = [], []
    for doc_id, rel in inputs.qrels.get(qid, {}).items():
        if rel > 0 and doc_id in doc_rows:
            rows.append(doc_rows[doc_id])
            grades.append(float(rel))
    return rows, grades


def cmd_probe(args: argparse.Namespace) -> int:
    protocol = _check_protocol(args.protocol_record, args.repo)
    corpus = args.corpus
    probe_hours, _ = BUDGET_HOURS[corpus]
    deadline = time.perf_counter() + probe_hours * 3600 * args.search_share
    labels_dir = args.root / "evaluation" / "P3-bench" / "probe-official" / corpus / args.labels_run
    labels_manifest = read_manifest(labels_dir)
    inputs = load_fit_inputs(args.root, corpus, labels_manifest.config["n_fit"])
    valid = json.loads((labels_dir / "valid_query_ids.json").read_text())
    negatives = np.load(labels_dir / "negatives_f2_rows.npy")
    neg_by_query = dict(zip(valid, negatives, strict=True))
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    if is_large(len(inputs.doc_ids)):
        return _probe_large(args, protocol, inputs, labels_manifest, neg_by_query, deadline)
    doc_rows = {d: i for i, d in enumerate(inputs.doc_ids)}
    docs16 = np.asarray(inputs.documents)
    docs = torch.from_numpy(docs16.astype(np.float64)).to(device)
    query_index = {q: i for i, q in enumerate(inputs.query_ids)}
    manifest = start_manifest(
        f"{COMMAND} probe",
        {
            "corpus": corpus,
            "labels_run": args.labels_run,
            "caps": PROBE_CAPS,
            "sizes": PROBE_SEARCH_SIZES,
        },
        dataset_hash=file_sha256(args.root / "p3-query-split-v1" / "manifest.json"),
        model_revision=MODEL_REVISION,
        seed=20261112,
        device=str(device),
    )
    report: dict[str, Any] = {"protocol_record_files": protocol["files"], "device": str(device)}

    # (2) exact full-corpus scoring of fit queries at the ten deciles (prefix masks; timing only)
    timing_ids = [q for q in inputs.query_ids if q in neg_by_query][:256]
    tick = time.perf_counter()
    count = 0
    for qid in timing_ids:
        rows, grades = _positives(inputs, qid, doc_rows)
        q = torch.from_numpy(inputs.queries[query_index[qid]].astype(np.float64)).to(device)
        pos = torch.tensor(rows, device=device)
        for k in DECILES:
            query_measures(
                positive_ranks(masked_scores(docs, q, torch.arange(k, device=device)), pos).cpu(),
                torch.tensor(grades, dtype=torch.float64),
            )
            count += 1
    if device.type == "cuda":
        torch.cuda.synchronize()
    report["scoring"] = {"query_masks": count, "seconds": time.perf_counter() - tick}
    report["scoring"]["seconds_per_query_mask"] = report["scoring"]["seconds"] / count

    # (3) rank-guided search: nested prefixes, nine budgets, both starts, cap max(PROBE_CAPS)
    searches = []
    reached = 0
    stopped_early = False
    for qid in [q for q in inputs.query_ids if q in neg_by_query][: max(PROBE_SEARCH_SIZES)]:
        if time.perf_counter() > deadline:
            stopped_early = True
            break
        rows, grades = _positives(inputs, qid, doc_rows)
        q16 = inputs.queries[query_index[qid]]
        q64 = q16.astype(np.float64)
        p_bar = positive_mean(docs16[rows].astype(np.float64), np.asarray(grades))
        n_bar = docs16[neg_by_query[qid]].astype(np.float64).mean(0)
        objective = make_objective(torch.tensor(grades), device)
        q_dev = torch.from_numpy(q64).to(device)
        pos = torch.tensor(rows, device=device)
        for k in PRIMARY:
            for start_name, start in (
                ("positive", positive_mask(q64, p_bar, k)),
                ("contrastive", contrastive_mask(q64, p_bar, n_bar, k)),
            ):
                result = rank_guided_search(docs, q_dev, pos, objective, start, cap=max(PROBE_CAPS))
                searches.append(
                    {
                        "query": reached,
                        "k": k,
                        "start": start_name,
                        "steps": result.steps,
                        "hit_cap": result.hit_cap,
                        "seconds": result.seconds,
                        "step_seconds": result.step_seconds,
                        "start_value": [result.start_ndcg, result.start_gdr],
                        "end_value": [result.ndcg, result.gdr],
                    }
                )
        reached += 1
    report["search"] = {"queries": reached, "stopped_early": stopped_early, "searches": searches}

    # (4) S1: 1,000 steps with checkpoint scoring (stand-in validation: fit queries; probe only)
    report["s1"] = _time_s1(inputs, neg_by_query, doc_rows, device)
    report.update(_peaks(str(device)))
    with ArtifactWriter(args.root, f"evaluation/P3-bench/probe/{corpus}", manifest) as writer:
        _write_json(writer.path("probe.json"), report)
    print(
        json.dumps(
            {
                "run_id": manifest.run_id,
                "queries_searched": reached,
                "scoring_s_per_mask": report["scoring"]["seconds_per_query_mask"],
                "s1_seconds": report["s1"]["seconds"],
                **_peaks(str(device)),
            }
        )
    )
    return 0


def _probe_large(
    args: argparse.Namespace,
    protocol: dict[str, Any],
    inputs: FitInputs,
    labels_manifest: Any,
    neg_by_query: dict[str, np.ndarray],
    deadline: float,
) -> int:
    """The probe on a large corpus: batched exact scoring through the chunked scorer, one exact
    top-1,000 pass (tune pools), shortlist rank-guided search for each shortlist size, S1 steps."""
    corpus = args.corpus
    device = torch.device(args.device)
    doc_rows = {d: i for i, d in enumerate(inputs.doc_ids)}
    docs16 = inputs.documents
    scorer = CorpusScorer(docs16, device, chunk_rows=SCORER_CHUNK)
    query_index = {q: i for i, q in enumerate(inputs.query_ids)}
    labels_dir = args.root / "evaluation" / "P3-bench" / "probe-official" / corpus / args.labels_run
    manifest = start_manifest(
        f"{COMMAND} probe",
        {
            "corpus": corpus,
            "labels_run": args.labels_run,
            "caps": PROBE_CAPS,
            "sizes": PROBE_SEARCH_SIZES,
            "shortlists": MS_SHORTLISTS,
            "scorer_chunk_rows": SCORER_CHUNK,
        },
        dataset_hash=file_sha256(args.root / "p3-query-split-v1" / "manifest.json"),
        model_revision=MODEL_REVISION,
        seed=20261112,
        device=str(device),
    )
    report: dict[str, Any] = {
        "protocol_record_files": protocol["files"],
        "device": str(device),
        "labels_run_summary": json.loads((labels_dir / "summary.json").read_text()),
    }
    ids = [q for q in inputs.query_ids if q in neg_by_query]

    # (2) exact scoring, the ten decile prefix masks of a query in one batch; one top-1,000 pass
    timing_ids = ids[:64]
    tick = time.perf_counter()
    for qid in timing_ids:
        rows, _ = _positives(inputs, qid, doc_rows)
        q = torch.from_numpy(inputs.queries[query_index[qid]].astype(np.float64)).to(device)
        batch = torch.stack([masked_query(q, torch.arange(k, device=device)) for k in DECILES])
        scorer.positive_ranks(batch, [torch.tensor(rows)] * len(DECILES))
    _sync(device)
    seconds = time.perf_counter() - tick
    count = len(timing_ids) * len(DECILES)
    tick = time.perf_counter()
    for qid in timing_ids[:8]:
        rows, _ = _positives(inputs, qid, doc_rows)
        q = torch.from_numpy(inputs.queries[query_index[qid]].astype(np.float64)).to(device)
        scorer.top(q, 1000, exclude=rows)
    _sync(device)
    report["scoring"] = {
        "query_masks": count,
        "seconds": seconds,
        "seconds_per_query_mask": seconds / count,
        "seconds_per_top_pass": (time.perf_counter() - tick) / min(8, len(timing_ids)),
    }

    # (3) shortlist rank-guided search: nine budgets, both starts, each shortlist, cap max(caps)
    searches: list[dict[str, Any]] = []
    reached = 0
    stopped_early = False
    for qid in ids[: max(PROBE_SEARCH_SIZES)]:
        rows, grades = _positives(inputs, qid, doc_rows)
        q16 = inputs.queries[query_index[qid]]
        q64 = q16.astype(np.float64)
        p_bar = positive_mean(np.asarray(docs16[rows], dtype=np.float64), np.asarray(grades))
        n_bar = np.asarray(docs16[neg_by_query[qid]], dtype=np.float64).mean(0)
        objective = make_objective(torch.tensor(grades), device)
        q_dev = torch.from_numpy(q64).to(device)
        pos = torch.tensor(rows, device=device)
        found: list[dict[str, Any]] = []
        for shortlist in MS_SHORTLISTS:
            for k in PRIMARY:
                for start_name, start in (
                    ("positive", positive_mask(q64, p_bar, k)),
                    ("contrastive", contrastive_mask(q64, p_bar, n_bar, k)),
                ):
                    if time.perf_counter() > deadline:
                        stopped_early = True
                        break
                    result = shortlist_search(
                        scorer,
                        q_dev,
                        pos,
                        objective,
                        start,
                        cap=max(PROBE_CAPS),
                        shortlist=shortlist,
                    )
                    found.append(
                        {
                            "query": reached,
                            "shortlist": shortlist,
                            "k": k,
                            "start": start_name,
                            "steps": result.steps,
                            "hit_cap": result.hit_cap,
                            "rejected": result.rejected,
                            "seconds": result.seconds,
                            "step_seconds": result.step_seconds,
                            "start_value": [result.start_ndcg, result.start_gdr],
                            "end_value": [result.ndcg, result.gdr],
                        }
                    )
        if stopped_early:  # a partly searched query is discarded
            break
        searches.extend(found)
        reached += 1
    report["search"] = {"queries": reached, "stopped_early": stopped_early, "searches": searches}
    report["s1"] = _time_s1(inputs, neg_by_query, doc_rows, device)
    report.update(_peaks(str(device)))
    with ArtifactWriter(args.root, f"evaluation/P3-bench/probe/{corpus}", manifest) as writer:
        _write_json(writer.path("probe.json"), report)
    print(
        json.dumps(
            {
                "run_id": manifest.run_id,
                "queries_searched": reached,
                "scoring_s_per_mask": report["scoring"]["seconds_per_query_mask"],
                "top_pass_s": report["scoring"]["seconds_per_top_pass"],
                "s1_seconds": report["s1"]["seconds"],
                **_peaks(str(device)),
            }
        )
    )
    return 0


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _time_s1(
    inputs: FitInputs,
    neg_by_query: dict[str, np.ndarray],
    doc_rows: dict[str, int],
    device: torch.device,
) -> dict[str, Any]:
    from hyperdime.baselines.learning_to_select import BUDGETS
    from hyperdime.training.examples import retrieval_examples
    from hyperdime.training.retrieval_trainer import RetrievalConfig, train_retrieval_candidates

    ids = [q for q in inputs.query_ids if q in neg_by_query]
    pools = {q: [inputs.doc_ids[r] for r in neg_by_query[q]] for q in ids}
    rows = {q: i for i, q in enumerate(inputs.query_ids)}
    train = retrieval_examples(
        ids, inputs.queries, rows, np.asarray(inputs.documents), doc_rows, inputs.qrels, pools
    )
    stand_in = retrieval_examples(
        ids[:64], inputs.queries, rows, np.asarray(inputs.documents), doc_rows, inputs.qrels, pools
    )
    config = RetrievalConfig(
        budgets=BUDGETS,
        steps=1000,
        eval_every=1000,
        lr=1e-3,
        score_temperature=0.05,
        beta_end=0.1,
        seed=SEED,
        device=str(device),
    )
    tick = time.perf_counter()
    train_retrieval_candidates(train, stand_in, config)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return {
        "steps": 1000,
        "checkpoint_scorings": 2,
        "seconds": time.perf_counter() - tick,
        "note": "validation stand-in = 64 fit queries (probe only; no tune labels)",
    }


# ----------------------------------------------------------------------------- resource freeze


def _search_seconds_at(searches: list[dict[str, Any]], cap: int) -> float:
    total = 0.0
    for search in searches:
        steps = search["step_seconds"]
        overhead = search["seconds"] - sum(steps)
        total += overhead + sum(steps if search["steps"] < cap else steps[:cap])
    return total


def cmd_freeze_resources(args: argparse.Namespace) -> int:
    out = args.out
    if out.exists():
        raise ContractError(f"{out} exists; freeze records are never overwritten")
    corpus = args.corpus
    probe_dir = args.root / "evaluation" / "P3-bench" / "probe" / corpus / args.probe_run
    probe = json.loads((probe_dir / "probe.json").read_text())
    split = load_query_split(args.root, corpus)
    n_fit, n_tune, n_eval = (len(split.ids[s]) for s in ("fit", "tune", "evaluation"))
    if "labels_run_summary" in probe:  # large-corpus probe
        record = large_resource_plan(probe, n_fit, n_tune, n_eval, BUDGET_HOURS[corpus])
        record.update(
            {
                "kind": "resources",
                "created_at": datetime.now(UTC).isoformat(),
                "corpus": corpus,
                "probe_run": args.probe_run,
                "probe_manifest_sha256": file_sha256(probe_dir / "manifest.json"),
                "probe_json_sha256": file_sha256(probe_dir / "probe.json"),
            }
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        _write_json(out, record)
        realized = record["realized"][corpus]
        shown = {"record": str(out), "sha256": file_sha256(out), "rule": record["rule"]}
        shown["without_rank_guided"] = record["realized_without_rank_guided"]
        print(json.dumps({**shown, **realized}))
        return 0 if realized["step_cap"] is not None else 3
    searches = probe["search"]["searches"]
    queries = probe["search"]["queries"]
    _, total_hours = BUDGET_HOURS[corpus]
    per_mask = probe["scoring"]["seconds_per_query_mask"]
    s1_per_step = probe["s1"]["seconds"] / probe["s1"]["steps"]
    fixed = {
        "evaluation": n_eval * (8 * 9 + 1) * per_mask,
        "secondary_seeds": n_eval * 5 * 3 * 9 * per_mask,
        "s1_training": 2 * 8 * 20 * 1000 * s1_per_step,
        "s1_tune_scoring": n_tune * 2 * 8 * 9 * per_mask,
    }
    shares = {cap: float(np.mean([s["steps"] < cap for s in searches])) for cap in PROBE_CAPS}
    projections = {}
    for cap in PROBE_CAPS:
        teacher = _search_seconds_at(searches, cap) / max(queries, 1) * n_fit
        projections[cap] = SAFETY * (teacher + sum(fixed.values())) / 3600
    budget_ok = [cap for cap in PROBE_CAPS if projections[cap] <= total_hours]
    meeting = [cap for cap in budget_ok if shares[cap] >= STOP_SHARE]
    step_cap: int | None
    if meeting:
        step_cap, rule = min(meeting), "smallest cap with >= 95% of searches stopping before it"
    elif budget_ok:
        step_cap, rule = max(budget_ok), "no cap reached 95%; largest cap within budget"
    else:
        step_cap, rule = None, "escalate: minimum coverage over budget"
    record = {
        "kind": "resources",
        "created_at": datetime.now(UTC).isoformat(),
        "corpus": corpus,
        "probe_run": args.probe_run,
        "probe_manifest_sha256": file_sha256(probe_dir / "manifest.json"),
        "probe_json_sha256": file_sha256(probe_dir / "probe.json"),
        "probe_queries_searched": queries,
        "stop_share_by_cap": shares,
        "projected_hours_by_cap": projections,
        "fixed_seconds": fixed,
        "budget_hours": total_hours,
        "safety_factor": SAFETY,
        "rule": rule,
        "realized": {
            corpus: {
                "n_fit": n_fit,
                "n_rg": n_fit,
                "n_tune": n_tune,
                "n_eval": n_eval,
                "step_cap": step_cap,
                "shortlist": None,
            }
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_json(out, record)
    print(
        json.dumps(
            {
                "record": str(out),
                "sha256": file_sha256(out),
                "step_cap": step_cap,
                "rule": rule,
                "projected_hours": projections,
                "stop_share": shares,
            }
        )
    )
    return 0 if step_cap is not None else 3


def large_resource_plan(
    probe: dict[str, Any], n_fit: int, n_tune: int, n_eval: int, hours: tuple[float, float]
) -> dict[str, Any]:
    """ADR-0018's MS MARCO size rule on the large-corpus probe.

    Projected seconds (times the 1.5 safety factor) of everything after the probe, against the
    corpus budget minus the probe budget: official labels and training on ``n_rg`` fit ids,
    13 more L2S-recipe trainings (replay control and 3 teachers x 4 seeds), the teacher search,
    S1 (2 arms x (8 cells + 3 seeds) x 20,000 steps), tune pools (one exact top-1,000 pass per tune
    query) and tune scoring (2 x 8 cells x 9 deciles per query), evaluation (8 x 9 + 1 masks per
    query) and secondary seeds (5 x 3 x 9). Sizes are chosen in the priority order n_eval, n_rg,
    n_tune, each the largest that fits with the later ones at their minimum and the cheapest search
    setting; then the shortlist (larger first) and the step cap by the 95% rule. If even the
    minimum coverage does not fit, the cap is None (escalation), and the sizes that apply if the
    rank-guided arm is removed are computed by the same order without search and with 10 trainings.
    """
    probe_hours, total_hours = hours
    budget = total_hours - probe_hours
    labels = probe["labels_run_summary"]
    label_s = labels["seconds"]["labels"] / labels["n_fit"]
    train_s = labels["seconds"]["train"] / max(labels["n_valid"], 1)
    per_mask = probe["scoring"]["seconds_per_query_mask"]
    top_pass = probe["scoring"]["seconds_per_top_pass"]
    s1_step = probe["s1"]["seconds"] / probe["s1"]["steps"]
    searches = probe["search"]["searches"]
    queries = max(probe["search"]["queries"], 1)

    def search_s(shortlist: int, cap: int) -> float:
        chosen = [x for x in searches if x["shortlist"] == shortlist]
        return float(_search_seconds_at(chosen, cap) / queries)

    def share(shortlist: int, cap: int) -> float:
        chosen = [x for x in searches if x["shortlist"] == shortlist]
        return float(np.mean([x["steps"] < cap for x in chosen])) if chosen else 0.0

    def hours_of(ne: int, nr: int, nt: int, search: float | None) -> float:
        trainings = 14 if search is not None else 10
        seconds = (
            nr * label_s
            + trainings * nr * train_s
            + (nr * search if search is not None else 0.0)
            + 2 * (8 + 3) * 20_000 * s1_step
            + nt * (top_pass + 2 * 8 * 9 * per_mask)
            + ne * (8 * 9 + 1 + 5 * 3 * 9) * per_mask
        )
        return float(SAFETY * seconds / 3600)

    def choose(search: float | None) -> tuple[int, int, int] | None:
        ne_min, nr_min, nt_min = MS_N_EVAL[-1], MS_N_RG[-1], MS_N_TUNE[-1]
        if hours_of(ne_min, min(nr_min, n_fit), nt_min, search) > budget:
            return None
        ne = next(
            v for v in MS_N_EVAL if hours_of(min(v, n_eval), nr_min, nt_min, search) <= budget
        )
        nr = next(v for v in MS_N_RG if hours_of(ne, min(v, n_fit), nt_min, search) <= budget)
        nt = next(v for v in MS_N_TUNE if hours_of(ne, nr, min(v, n_tune), search) <= budget)
        return min(ne, n_eval), min(nr, n_fit), min(nt, n_tune)

    cheapest = min(search_s(n, min(PROBE_CAPS)) for n in MS_SHORTLISTS) if searches else None
    sizes = choose(cheapest) if cheapest is not None else None
    step_cap, shortlist, rule = None, None, "escalate: minimum coverage over budget"
    if sizes is not None:
        ne, nr, nt = sizes
        for n in MS_SHORTLISTS:
            fits = [c for c in PROBE_CAPS if hours_of(ne, nr, nt, search_s(n, c)) <= budget]
            if not fits:
                continue
            meeting = [c for c in fits if share(n, c) >= STOP_SHARE]
            shortlist = n
            if meeting:
                step_cap, rule = min(meeting), "smallest cap with >= 95% of searches stopping"
            else:
                step_cap, rule = max(fits), "no cap reached 95%; largest cap within budget"
            break
    without = choose(None)
    realized: dict[str, int | None] = {
        "n_fit": n_fit,
        "n_rg": n_fit,
        "n_tune": n_tune,
        "n_eval": n_eval,
    }
    if sizes is not None and step_cap is not None:
        realized = {"n_fit": sizes[1], "n_rg": sizes[1], "n_tune": sizes[2], "n_eval": sizes[0]}
    elif without is not None:
        realized = {
            "n_fit": without[1],
            "n_rg": without[1],
            "n_tune": without[2],
            "n_eval": without[0],
        }
    realized.update({"step_cap": step_cap, "shortlist": shortlist})
    no_rg = (
        None
        if without is None
        else {"n_fit": without[1], "n_rg": without[1], "n_tune": without[2], "n_eval": without[0],
              "step_cap": None, "shortlist": None,
              "projected_hours": hours_of(without[0], without[1], without[2], None)}
    )  # fmt: skip
    return {
        "budget_hours": total_hours,
        "budget_after_probe_hours": budget,
        "safety_factor": SAFETY,
        "unit_seconds": {
            "label_per_query": label_s,
            "train_per_query": train_s,
            "scoring_per_query_mask": per_mask,
            "top_pass": top_pass,
            "s1_per_step": s1_step,
        },
        "search_seconds_per_query": {
            str(n): {str(c): search_s(n, c) for c in PROBE_CAPS} for n in MS_SHORTLISTS
        },
        "stop_share": {str(n): {str(c): share(n, c) for c in PROBE_CAPS} for n in MS_SHORTLISTS},
        "probe_queries_searched": probe["search"]["queries"],
        "rule": rule,
        "realized": {"msmarco": realized},
        "realized_without_rank_guided": no_rg,
        "projected_hours": (
            hours_of(sizes[0], sizes[1], sizes[2], search_s(shortlist, step_cap))
            if sizes is not None and step_cap is not None and shortlist is not None
            else None
        ),  # fmt: skip
    }


# ----------------------------------------------------------------------------- official parity


def cmd_official_parity(args: argparse.Namespace) -> int:
    """Authors' evaluator vs the common exact evaluator, per fit query, on both grids (T013)."""
    _require_record(args.resource_record, "resources")
    corpus = args.corpus
    run_dir = args.root / "selectors" / "l2s-official" / corpus / args.official_run
    manifest = read_manifest(run_dir)
    inputs = load_fit_inputs(args.root, corpus, manifest.config["n_fit"])
    module = import_official(args.clone)
    order = authors_order(inputs.lengths)
    corpus_ids = [inputs.doc_ids[r] for r in order]
    docs_authors = torch.from_numpy(np.asarray(inputs.documents, dtype=np.float32)[order])
    queries32 = torch.from_numpy(inputs.queries.astype(np.float32))
    predictor = module.DimensionPredictor(input_dim=queries32.shape[1])
    predictor.load_state_dict(torch.load(run_dir / "state_dict.pt"))
    device = args.device
    predictor.to(device).eval()
    with torch.no_grad():
        q_norm = module.l2_normalize_rows(queries32.to(device))
        log_probs = predictor(q_norm).cpu()
    doc_rows = {d: i for i, d in enumerate(inputs.doc_ids)}
    docs = torch.from_numpy(np.asarray(inputs.documents).astype(np.float64)).to(device)
    full_ranks_equal = True
    rows_out = []
    for qi, qid in enumerate(inputs.query_ids):
        rows, grades = _positives(inputs, qid, doc_rows)
        if not rows:
            continue
        authors = module.evaluate_predictor_strict(
            predictor,
            [qid],
            corpus_ids,
            queries32[qi : qi + 1],
            docs_authors,
            {qid: inputs.qrels[qid]},
            device=device,
        )["predictor_by_dim"]
        q = torch.from_numpy(inputs.queries[qi].astype(np.float64)).to(device)
        pos = torch.tensor(rows, device=device)
        gains = torch.tensor(grades, dtype=torch.float64)
        full = positive_ranks(masked_scores(docs, q, None), pos).cpu()
        row: dict[str, Any] = {"query": qid}
        for grid_name, grid in (("authors", AUTHORS_GRID), ("project", DECILES)):
            for k in grid:
                dims = torch.topk(log_probs[qi], k).indices.to(device)
                ranks = positive_ranks(masked_scores(docs, q, dims), pos).cpu()
                row[f"{grid_name}:{k}"] = query_measures(ranks, gains)["ndcg@10"]
                if k == 1024:
                    full_ranks_equal &= bool(torch.equal(ranks, full))
        for k in AUTHORS_GRID:
            row[f"official:{k}"] = authors[str(k)]
        rows_out.append(row)
    summary: dict[str, Any] = {
        "queries": len(rows_out),
        "full_vector_parity_at_1024": full_ranks_equal,
        "deciles": {},
    }
    for k in AUTHORS_GRID:
        official = np.array([r[f"official:{k}"] for r in rows_out])
        ours = np.array([r[f"authors:{k}"] for r in rows_out])
        summary["deciles"][str(k)] = {
            "official_mean": float(official.mean()),
            "common_mean": float(ours.mean()),
            "queries_differing": int((np.abs(official - ours) > 1e-9).sum()),
            "max_abs_difference": float(np.abs(official - ours).max()),
        }
    summary["project_grid_means"] = {
        str(k): float(np.mean([r[f"project:{k}"] for r in rows_out])) for k in DECILES
    }
    summary["conventions"] = {
        "official": (
            "float32 cosine on renormalized rows, torch.topk ties in length-sorted order, "
            "top-2000 then top-1000 list, gain 2^g-1, ideal over the retrieved list"
        ),
        "common": (
            "float64 exact dot products of stored vectors, lower-row ties in F2 order, "
            "linear gain, ideal over all positives"
        ),
    }
    mf = start_manifest(
        f"{COMMAND} official-parity",
        {"corpus": corpus, "official_run": args.official_run},
        dataset_hash=file_sha256(args.root / "p3-query-split-v1" / "manifest.json"),
        model_revision=MODEL_REVISION,
        seed=SEED,
        device=device,
    )
    with ArtifactWriter(args.root, f"evaluation/P3-bench/official-parity/{corpus}", mf) as writer:
        _write_json(writer.path("summary.json"), summary)
        _write_json(writer.path("per_query.json"), rows_out)
    print(
        json.dumps(
            {
                "run_id": mf.run_id,
                "full_vector_parity_at_1024": full_ranks_equal,
                "deciles": summary["deciles"],
            }
        )
    )
    return 0


# ----------------------------------------------------------------------------- registration


def register(commands: Any) -> None:
    parser = commands.add_parser(COMMAND, help="F12 P3 selector benchmark (ADR-0018)")
    sub = parser.add_subparsers(dest="p3_bench_command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--root", type=Path, default=Path("artifacts"))
        p.add_argument("--corpus", required=True, choices=sorted(PUBLISHED))

    p = sub.add_parser("split-provenance")
    common(p)
    p.set_defaults(handler=cmd_split_provenance)

    p = sub.add_parser("freeze-protocol")
    p.add_argument("--repo", type=Path, default=Path("."))
    p.add_argument("--out", type=Path, default=Path("reports/P3-bench/freeze/protocol.json"))
    p.set_defaults(handler=cmd_freeze_protocol)

    p = sub.add_parser("official")
    common(p)
    p.add_argument("--purpose", choices=("probe", "reference"), required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--labels-only", action="store_true")
    p.add_argument("--resource-record", type=Path)
    p.add_argument("--device", default="cpu")
    p.add_argument("--clone", type=Path, default=DEFAULT_CLONE)
    p.set_defaults(handler=cmd_official)

    p = sub.add_parser("probe")
    common(p)
    p.add_argument("--labels-run", required=True)
    p.add_argument("--protocol-record", type=Path, required=True)
    p.add_argument("--repo", type=Path, default=Path("."))
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--search-share", type=float, default=0.7, help="share of the probe budget for search"
    )
    p.set_defaults(handler=cmd_probe)

    p = sub.add_parser("freeze-resources")
    common(p)
    p.add_argument("--probe-run", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(handler=cmd_freeze_resources)

    p = sub.add_parser("official-parity")
    common(p)
    p.add_argument("--official-run", required=True)
    p.add_argument("--resource-record", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--clone", type=Path, default=DEFAULT_CLONE)
    p.set_defaults(handler=cmd_official_parity)

    from hyperdime.cli import p3_bench_arms

    p3_bench_arms.register(sub)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - convenience
    import argparse as _argparse

    parser = _argparse.ArgumentParser()
    register(parser.add_subparsers(dest="command", required=True))
    args = parser.parse_args(argv)
    return int(args.handler(args))


__all__ = ["register", "load_fit_inputs"]
