"""``hyperdime p3-bench`` stages after the resource gate (ADR-0018, specs/013).

- ``teachers``: positive, contrastive, and rank-guided masks on fit queries (cap from the resource
  record), nesting and demotion counts, nine-budget targets, and the global mask. When the resource
  gate escalated and the escalation record removes the rank-guided arm (ADR-0018, "Escalation"),
  only the positive and contrastive masks are built; the global mask, defined from rank-guided
  masks, is then undefined and left out.
- ``train-arm``: one isolated process training an L2S-recipe arm on a target (``official`` at seed
  42 is the replay control and must reproduce ``l2s-official`` exactly).
- ``train-s1``: the eight S1 cells for one arm (scratch or official-init), selection on tune, and
  the selected cell at seeds 0-2.
- ``evaluation-lock``: fingerprint every arm and the evaluation ids before evaluation labels.
- ``evaluate``: one pass over evaluation queries for every arm, with ADR-0018's statistics.
- ``escalate``: the researcher's escalation choice for a corpus whose resource gate escalated.
- ``family``: Holm over the locked family and Bonferroni bounds at its final size, from the stored
  per-query results of every evaluated corpus.

Selection input for every linear arm is the L2-normalized query, as in the authors' evaluator;
top-k uses ``torch.topk`` on the evaluation device.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from hyperdime.baselines.l2s_official import DEFAULT_CLONE, PUBLISHED, SEED, import_official
from hyperdime.cli.p3_bench import (
    MODEL_REVISION,
    SCORER_CHUNK,
    FitInputs,
    _jsonl,
    _positives,
    _require_record,
    _write_json,
    is_large,
    load_fit_inputs,
)
from hyperdime.contracts.hashing import file_sha256
from hyperdime.contracts.manifests import ArtifactWriter, read_manifest, start_manifest
from hyperdime.contracts.validation import ContractError
from hyperdime.data.query_split import hash_order, id_fingerprint, load_query_split
from hyperdime.evaluation.benchmark import (
    DECILES,
    PRIMARY,
    CorpusScorer,
    masked_query,
    masked_scores,
    positive_ranks,
    query_measures,
)
from hyperdime.evaluation.benchmark_stats import (
    bootstrap_means,
    claim_level,
    holm,
    intervals,
    sign_flip_p,
    stream,
)
from hyperdime.evaluation.teachers import (
    better,
    contrastive_mask,
    make_objective,
    positive_mask,
    positive_mean,
    rank_guided_search,
    shortlist_search,
    top_k_dims,
)
from hyperdime.reports.p3_bench import (
    FRAMING,
    arm_costs,
    cost_lines,
    evaluation_provenance,
    provenance_lines,
    search_gains,
    selector_forward_seconds,
    teacher_report,
)

TEACHERS = ("positive", "contrastive", "rank-guided")
S1_CELLS: tuple[dict[str, float], ...] = tuple(
    {"lr": lr, "score_temperature": tau, "beta_end": beta}
    for lr in (1e-4, 1e-3)
    for tau in (0.02, 0.05)
    for beta in (0.02, 0.1)
)
SECONDARY_SEEDS = (0, 1, 2)
S1_STEPS = 20_000  # ADR-0009 budget; 21 checkpoints including step 0
S1_EVAL_EVERY = 1_000
S1_GUARD = 0.9  # final soft mass a cell needs to be eligible
FAMILY_SIZE = 9  # 3 teacher arms x 3 corpora (ADR-0018)
SPLIT_FILE = "p3-query-split-v1/manifest.json"


def _official_dir(root: Path, corpus: str, run: str) -> Path:
    return root / "selectors" / "l2s-official" / corpus / run


def _manifest_kwargs(root: Path, device: str, seed: int) -> dict[str, Any]:
    return {
        "dataset_hash": file_sha256(root / SPLIT_FILE),
        "model_revision": MODEL_REVISION,
        "seed": seed,
        "device": device,
    }


def _normalized(queries: np.ndarray, device: torch.device) -> Tensor:
    q = torch.from_numpy(np.asarray(queries, dtype=np.float32)).to(device)
    normalized: Tensor = q / (q.norm(dim=1, keepdim=True) + 1e-12)
    return normalized


# ----------------------------------------------------------------------------- resumable stages

INCOMPLETE = 75  # exit code: the stage stopped at its wall-time limit; rerun the same command
BLOCK_QUERIES = 50


class Checkpoint:
    """Completed query blocks of one long stage, so that a stage longer than the 12 h job cap
    (ADR-0018) runs as several resumed jobs and still writes one artifact.

    Blocks live under ``scratch/p3-bench/<corpus>/<stage>-<key>``, ``key`` hashing the stage's
    full configuration, so a changed input never resumes from stale blocks. A block is complete
    once its JSON is renamed into place (arrays first). Queries are processed in order and every
    per-query result is deterministic, so resumed and uninterrupted runs give equal outputs.
    """

    def __init__(
        self, root: Path, stage: str, corpus: str, config: dict[str, Any], max_hours: float | None
    ) -> None:
        key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
        self.directory = root / "scratch" / "p3-bench" / corpus / f"{stage}-{key}"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_seconds = None if max_hours is None else max_hours * 3600
        self.began = time.perf_counter()
        self.segment = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    def blocks(self) -> list[tuple[int, int, dict[str, np.ndarray], dict[str, Any]]]:
        """Completed blocks in order; they must tile ``[0, end)`` without gaps."""
        out: list[tuple[int, int, dict[str, np.ndarray], dict[str, Any]]] = []
        for meta_path in sorted(self.directory.glob("block-*.json")):
            start, stop = (int(x) for x in meta_path.stem.split("-")[1:3])
            if start != (out[-1][1] if out else 0):
                raise ContractError(f"checkpoint {self.directory} has a gap before {start}")
            arrays_path = meta_path.with_suffix(".npz")
            arrays = dict(np.load(arrays_path)) if arrays_path.exists() else {}
            out.append((start, stop, arrays, json.loads(meta_path.read_text())))
        return out

    def save(
        self, start: int, stop: int, arrays: dict[str, np.ndarray], meta: dict[str, Any]
    ) -> None:
        stem = self.directory / f"block-{start:07d}-{stop:07d}"
        if arrays:
            np.savez(stem.with_suffix(".tmp.npz"), **arrays)  # type: ignore[arg-type]
            stem.with_suffix(".tmp.npz").rename(stem.with_suffix(".npz"))
        tmp = stem.with_suffix(".tmp")
        _write_json(tmp, {**meta, "segment": self.segment})
        tmp.rename(stem.with_suffix(".json"))

    def expired(self) -> bool:
        return self.max_seconds is not None and time.perf_counter() - self.began >= self.max_seconds

    def stop(self, done: int, total: int) -> int:
        status = {"incomplete": True, "done": done, "of": total, "rerun_exit_code": INCOMPLETE}
        print(json.dumps({**status, "checkpoint": str(self.directory)}))
        return INCOMPLETE


# ----------------------------------------------------------------------------- teachers


def cmd_teachers(args: argparse.Namespace) -> int:
    resources = _require_record(args.resource_record, "resources")
    realized = resources["realized"][args.corpus]
    cap = realized["step_cap"]
    teachers = _teacher_names(args.resource_record, args.escalation_record, args.corpus)
    official = _official_dir(args.root, args.corpus, args.official_run)
    inputs = load_fit_inputs(args.root, args.corpus, realized["n_rg"])
    valid = json.loads((official / "valid_query_ids.json").read_text())
    negatives = np.load(official / "negatives_f2_rows.npy")
    device = torch.device(args.device)
    large = is_large(len(inputs.doc_ids))
    docs16 = inputs.documents if large else np.asarray(inputs.documents)
    scorer = CorpusScorer(docs16, device, chunk_rows=SCORER_CHUNK) if large else None
    docs = None if large else torch.from_numpy(docs16.astype(np.float64)).to(device)
    doc_rows = {d: i for i, d in enumerate(inputs.doc_ids)}
    query_index = {q: i for i, q in enumerate(inputs.query_ids)}
    n, dim = len(valid), docs16.shape[1]
    shortlist = realized.get("shortlist")
    masks = {name: np.zeros((n, len(PRIMARY), dim), dtype=bool) for name in teachers}
    searches: list[dict[str, Any]] = []
    demotions = {"unjudged": 0, "known_negative": 0}
    config = {
        "corpus": args.corpus,
        "official_run": args.official_run,
        "step_cap": cap,
        "teachers": list(teachers),
        "resource_record_sha256": file_sha256(args.resource_record),
        "escalation_record_sha256": (
            file_sha256(args.escalation_record) if args.escalation_record else None
        ),
    }
    manifest = start_manifest(
        "p3-bench teachers", config, **_manifest_kwargs(args.root, str(device), 20261112)
    )
    checkpoint = Checkpoint(
        args.root, "teachers", args.corpus, {**config, "device": str(device)}, args.max_hours
    )
    prior_seconds, segments, first = 0.0, set(), 0
    for start, stop, arrays, meta in checkpoint.blocks():
        for name in teachers:
            masks[name][start:stop] = arrays[name]
        searches.extend(meta["searches"])
        for key in demotions:
            demotions[key] += meta["demotions"][key]
        prior_seconds += meta["seconds"]
        segments.add(meta["segment"])
        first = stop
    segments.add(checkpoint.segment)
    block_start, block_searches = first, len(searches)
    block_demotions = dict(demotions)
    began = time.perf_counter()
    for qi in range(first, n):
        qid = valid[qi]
        rows, grades = _positives(inputs, qid, doc_rows)
        judged_zero = {
            doc_rows[d] for d, r in inputs.qrels.get(qid, {}).items() if r <= 0 and d in doc_rows
        }
        q64 = inputs.queries[query_index[qid]].astype(np.float64)
        p_bar = positive_mean(np.asarray(docs16[rows], dtype=np.float64), np.asarray(grades))
        n_bar = np.asarray(docs16[negatives[qi]], dtype=np.float64).mean(0)
        objective = make_objective(torch.tensor(grades), device)
        q_dev = torch.from_numpy(q64).to(device)
        pos = torch.tensor(rows, device=device)

        def search(
            dims: np.ndarray, q_dev: Tensor = q_dev, pos: Tensor = pos, objective: Any = objective
        ) -> Any:
            if scorer is not None:
                return shortlist_search(scorer, q_dev, pos, objective, dims, cap, shortlist)
            assert docs is not None
            return rank_guided_search(docs, q_dev, pos, objective, dims, cap=cap)

        for ki, k in enumerate(PRIMARY):
            starts = {
                "positive": positive_mask(q64, p_bar, k),
                "contrastive": contrastive_mask(q64, p_bar, n_bar, k),
            }
            masks["positive"][qi, ki, starts["positive"]] = True
            masks["contrastive"][qi, ki, starts["contrastive"]] = True
            if "rank-guided" not in teachers:
                continue
            results = {name: search(dims) for name, dims in starts.items()}
            a, b = results["positive"], results["contrastive"]
            chosen_name = "contrastive" if better((b.ndcg, b.gdr), (a.ndcg, a.gdr)) else "positive"
            chosen = results[chosen_name]
            masks["rank-guided"][qi, ki, chosen.dims] = True
            if docs is not None and not np.array_equal(np.sort(chosen.dims), starts[chosen_name]):
                before = masked_scores(
                    docs, q_dev, torch.as_tensor(starts[chosen_name], device=device)
                )
                after = masked_scores(docs, q_dev, torch.as_tensor(chosen.dims, device=device))
                demoted = (
                    (_above_best(before, pos) & ~_above_best(after, pos))
                    .nonzero()
                    .squeeze(1)
                    .tolist()
                )
                for row in demoted:
                    demotions["known_negative" if row in judged_zero else "unjudged"] += 1
            for name, result in results.items():
                searches.append(
                    {
                        "query": qi,
                        "k": k,
                        "start": name,
                        "steps": result.steps,
                        "hit_cap": result.hit_cap,
                        "seconds": result.seconds,
                        "start_value": [result.start_ndcg, result.start_gdr],
                        "end_value": [result.ndcg, result.gdr],
                        "chosen": name == chosen_name,
                        "rejected": getattr(result, "rejected", 0),
                    }
                )
        if qi + 1 - block_start == BLOCK_QUERIES or qi + 1 == n:
            now = time.perf_counter()
            checkpoint.save(
                block_start,
                qi + 1,
                {name: masks[name][block_start : qi + 1] for name in teachers},
                {
                    "searches": searches[block_searches:],
                    "demotions": {k: demotions[k] - block_demotions[k] for k in demotions},
                    "seconds": now - began,
                },
            )
            prior_seconds += now - began
            began, block_start, block_searches = now, qi + 1, len(searches)
            block_demotions = dict(demotions)
            if qi + 1 < n and checkpoint.expired():
                return checkpoint.stop(qi + 1, n)
    seconds = prior_seconds
    targets = {name: _target(masks[name]) for name in teachers}
    global_mask = None
    if "rank-guided" in teachers:
        counts = masks["rank-guided"].sum(0)  # [9, D]
        global_mask = np.zeros((len(PRIMARY), dim), dtype=bool)
        for ki, k in enumerate(PRIMARY):
            global_mask[ki, top_k_dims(counts[ki].astype(np.float64), k)] = True
    summary = {
        "teachers": list(teachers),
        "queries": n,
        "step_cap": cap,
        "seconds": seconds,
        "searches": len(searches),
        "searches_improved": sum(s["end_value"] != s["start_value"] for s in searches),
        "searches_hit_cap": sum(s["hit_cap"] for s in searches),
        "rank_guided_from": {
            name: sum(s["chosen"] and s["start"] == name for s in searches)
            for name in ("positive", "contrastive")
        },
        "nesting": {name: _nesting(masks[name]) for name in teachers},
        "demotions": demotions if docs is not None else "not computed on a large corpus",
        "searches_rejected": sum(x["rejected"] for x in searches),
        "device": str(device),
        "job_segments": len(segments),
    }
    with ArtifactWriter(
        args.root, f"evaluation/P3-bench/teachers/{args.corpus}", manifest
    ) as writer:
        for name in teachers:
            np.save(writer.path(f"masks_{name}.npy"), masks[name])
            torch.save(targets[name], writer.path(f"targets_{name}.pt"))
        if global_mask is not None:
            np.save(writer.path("global_mask.npy"), global_mask)
        _write_json(writer.path("valid_query_ids.json"), valid)
        _write_json(writer.path("searches.json"), searches)
        _write_json(writer.path("summary.json"), summary)
    print(
        json.dumps(
            {"run_id": manifest.run_id, **{k: v for k, v in summary.items() if k != "nesting"}}
        )
    )
    return 0


def _teacher_names(
    resource_record: Path, escalation_record: Path | None, corpus: str
) -> tuple[str, ...]:
    """Teachers built for ``corpus``: all three, or, after an escalation that removed the
    rank-guided arm, positive and contrastive only. Any other escalated state refuses to run."""
    cap = _require_record(resource_record, "resources")["realized"][corpus]["step_cap"]
    if cap is not None:
        if escalation_record is not None:
            raise ContractError("an escalation record applies only to an escalated resource gate")
        return TEACHERS
    if escalation_record is None:
        raise ContractError("the resource gate escalated; an escalation record is required")
    escalation = _require_record(escalation_record, "escalation")
    if escalation["corpus"] != corpus or escalation["resource_record_sha256"] != file_sha256(
        resource_record
    ):
        raise ContractError("the escalation record belongs to another resource record")
    if escalation["option"] != "remove-rank-guided":
        raise ContractError(f"escalation option {escalation['option']!r} trains no teacher")
    return tuple(t for t in TEACHERS if t != "rank-guided")


def _above_best(scores: Tensor, positives: Tensor) -> Tensor:
    best = positives[torch.argmin(positive_ranks(scores, positives))]
    rows = torch.arange(scores.shape[0], device=scores.device)
    own = scores[best]
    above = (scores > own) | ((scores == own) & (rows < best))
    above[positives] = False
    return above


def _target(masks: np.ndarray) -> Tensor:
    """``t_q = (1/9) sum_k m_{q,k} / k``: average of the uniform distributions on each mask."""
    sizes = masks.sum(2, keepdims=True).astype(np.float64)
    return torch.from_numpy((masks / sizes).mean(1).astype(np.float32))


def _nesting(masks: np.ndarray) -> list[float]:
    """Per consecutive budget pair, the mean share of the smaller mask inside the larger one."""
    shares = []
    for ki in range(masks.shape[1] - 1):
        inside = (masks[:, ki] & masks[:, ki + 1]).sum(1) / masks[:, ki].sum(1)
        shares.append(float(inside.mean()))
    return shares


# ----------------------------------------------------------------------------- teacher arms


def cmd_train_arm(args: argparse.Namespace) -> int:
    _require_record(args.resource_record, "resources")
    official = _official_dir(args.root, args.corpus, args.official_run)
    official_manifest = read_manifest(official)
    inputs = load_fit_inputs(args.root, args.corpus, official_manifest.config["n_fit"])
    valid = json.loads((official / "valid_query_ids.json").read_text())
    if args.target == "official":
        targets = torch.load(official / "targets.pt")
        source = official
    else:
        source = (
            args.root / "evaluation" / "P3-bench" / "teachers" / args.corpus / args.teachers_run
        )
        if json.loads((source / "valid_query_ids.json").read_text()) != valid:
            raise ContractError("teacher queries differ from the official reference's")
        targets = torch.load(source / f"targets_{args.target}.pt")
    target_file = "targets.pt" if args.target == "official" else f"targets_{args.target}.pt"
    _, epochs = PUBLISHED[args.corpus]
    module = import_official(args.clone)
    query_index = {q: i for i, q in enumerate(inputs.query_ids)}
    x_train = torch.from_numpy(inputs.queries[[query_index[q] for q in valid]].astype(np.float32))
    device = args.device
    manifest = start_manifest(
        "p3-bench train-arm",
        {
            "corpus": args.corpus,
            "target": args.target,
            "seed": args.seed,
            "epochs": epochs,
            "official_run": args.official_run,
            "teachers_run": args.teachers_run,
            "targets_sha256": file_sha256(source / target_file),
        },
        **_manifest_kwargs(args.root, device, args.seed),
    )
    module.set_random_seed(args.seed)
    replayed = official_manifest.device == "cpu" and args.seed == SEED
    if replayed:  # ADR-0018 "Seed and RNG matching": CPU label draws consumed the CPU stream
        torch.set_rng_state(torch.load(official / "rng_after_labels.pt")["cpu"])
    tick = time.perf_counter()
    dataset = module.PredictorDataset(x_train, targets)
    loader = module.DataLoader(dataset, batch_size=32, shuffle=True)
    predictor = module.DimensionPredictor(input_dim=x_train.shape[1]).to(device)
    predictor, loss = module.train_predictor(
        model=predictor, train_loader=loader, epochs=epochs, device=device, lr=1e-4
    )
    state = {k: v.detach().cpu().clone() for k, v in predictor.state_dict().items()}
    summary: dict[str, Any] = {
        "seconds": time.perf_counter() - tick,
        "final_train_loss": float(loss),
        "rng_replayed_after_cpu_labels": replayed,
    }
    if args.target == "official":
        reference = torch.load(official / "state_dict.pt")
        summary["replay_equals_official"] = all(
            torch.equal(state[k], reference[k]) for k in reference
        )
    with ArtifactWriter(args.root, f"selectors/p3-teacher-arms/{args.corpus}", manifest) as writer:
        torch.save(state, writer.path("state_dict.pt"))
        _write_json(writer.path("summary.json"), summary)
    print(
        json.dumps({"run_id": manifest.run_id, "target": args.target, "seed": args.seed, **summary})
    )
    if args.target == "official" and args.seed == SEED and not summary["replay_equals_official"]:
        return 4
    return 0


# ----------------------------------------------------------------------------- S1


def _split_qrels(root: Path, corpus: str, ids: set[str]) -> dict[str, dict[str, int]]:
    split = load_query_split(root, corpus)
    dataset = root / "datasets" / corpus / split.source_dataset_run
    qrels: dict[str, dict[str, int]] = {}
    for name in ("train", "validation"):  # never qrels-test
        for record in _jsonl(dataset / f"qrels-{name}.jsonl"):
            if record["query_id"] in ids:
                qrels.setdefault(record["query_id"], {})[record["doc_id"]] = int(
                    record["relevance"]
                )
    return qrels


def _query_vectors(root: Path, corpus: str, ids: list[str], inputs: FitInputs) -> np.ndarray:
    emb = next(
        p
        for p in (root / "embeddings" / corpus).iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )
    rows = {q: i for i, q in enumerate(json.loads((emb / "queries.json").read_text())["ids"])}
    return np.asarray(np.load(emb / "queries.npy", mmap_mode="r")[[rows[q] for q in ids]])


def _top_pool(docs: Tensor, query: np.ndarray, positives: set[int], size: int = 1000) -> list[int]:
    scores = docs @ torch.from_numpy(query.astype(np.float64)).to(docs.device)
    order = torch.argsort(-scores, stable=True)[: size + len(positives)].tolist()
    return [r for r in order if r not in positives][:size]


MASK_BATCH = 512


def _tune_score_large(
    model: torch.nn.Module,
    queries: np.ndarray,
    positives: list[tuple[list[int], list[float]]],
    scorer: CorpusScorer,
) -> float:
    """:func:`_tune_score` through the chunked scorer, many masks per corpus pass (exact)."""
    device = scorer.device
    model.eval().to(device)
    with torch.no_grad():
        importance = model(_normalized(queries, device))
    jobs = [(qi, k) for qi in range(len(positives)) for k in PRIMARY]
    values = []
    for start in range(0, len(jobs), MASK_BATCH):
        chunk = jobs[start : start + MASK_BATCH]
        batch = torch.stack(
            [
                masked_query(
                    torch.from_numpy(queries[qi].astype(np.float64)).to(device),
                    torch.topk(importance[qi], k).indices,
                )
                for qi, k in chunk
            ]
        )
        ranks = scorer.positive_ranks(batch, [torch.tensor(positives[qi][0]) for qi, _ in chunk])
        for (qi, _), r in zip(chunk, ranks, strict=True):
            gains = torch.tensor(positives[qi][1], dtype=torch.float64)
            values.append(query_measures(r, gains)["ndcg@10"])
    return float(np.mean(values))


def _tune_score(
    model: torch.nn.Module,
    queries: np.ndarray,
    positives: list[tuple[list[int], list[float]]],
    docs: Tensor,
    device: torch.device,
) -> float:
    model.eval().to(device)
    with torch.no_grad():
        importance = model(_normalized(queries, device))
    values = []
    for qi, (rows, grades) in enumerate(positives):
        q = torch.from_numpy(queries[qi].astype(np.float64)).to(device)
        pos = torch.tensor(rows, device=device)
        gains = torch.tensor(grades, dtype=torch.float64)
        for k in PRIMARY:
            dims = torch.topk(importance[qi], k).indices
            values.append(
                query_measures(positive_ranks(masked_scores(docs, q, dims), pos).cpu(), gains)[
                    "ndcg@10"
                ]
            )
    return float(np.mean(values))


def select_cell(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """ADR-0018 S1 selection: among cells passing the soft-mass guard, the highest tune mean
    nDCG@10; ties go to the lower cell index."""
    eligible = [c for c in cells if c["passes_guard"]]
    if not eligible:
        raise ContractError("no S1 cell passes the soft-mass guard")
    return max(eligible, key=lambda c: (c["tune_mean_ndcg10"], -c["cell"]))


def cmd_train_s1(args: argparse.Namespace) -> int:
    from hyperdime.baselines.learning_to_select import BUDGETS
    from hyperdime.training.examples import retrieval_examples
    from hyperdime.training.retrieval_trainer import RetrievalConfig, train_retrieval_candidates

    resources = _require_record(args.resource_record, "resources")
    realized = resources["realized"][args.corpus]
    official = _official_dir(args.root, args.corpus, args.official_run)
    inputs = load_fit_inputs(args.root, args.corpus, realized["n_fit"])
    valid = json.loads((official / "valid_query_ids.json").read_text())
    negatives = np.load(official / "negatives_f2_rows.npy")
    device = torch.device(args.device)
    large = is_large(len(inputs.doc_ids))
    docs16 = inputs.documents if large else np.asarray(inputs.documents)
    # large corpora stream the float16 rows from host memory, leaving the GPU to training
    scorer = (
        CorpusScorer(docs16, device, chunk_rows=SCORER_CHUNK, resident=False) if large else None
    )
    docs = None if large else torch.from_numpy(docs16.astype(np.float64)).to(device)
    doc_rows = {d: i for i, d in enumerate(inputs.doc_ids)}
    rows_fit = {q: i for i, q in enumerate(inputs.query_ids)}
    pools = {q: [inputs.doc_ids[r] for r in negatives[i]] for i, q in enumerate(valid)}
    train = retrieval_examples(
        valid, inputs.queries, rows_fit, docs16, doc_rows, inputs.qrels, pools
    )
    split = load_query_split(args.root, args.corpus)
    tune_ids = hash_order(args.corpus, "tune", split.ids["tune"])[: realized["n_tune"]]
    tune_qrels = _split_qrels(args.root, args.corpus, set(tune_ids))
    tune_ids = [q for q in tune_ids if any(r > 0 for r in tune_qrels.get(q, {}).values())]
    tune_vectors = _query_vectors(args.root, args.corpus, tune_ids, inputs)
    tune_pos = []
    tune_pools = {}
    for i, qid in enumerate(tune_ids):
        rows = [doc_rows[d] for d, r in tune_qrels[qid].items() if r > 0 and d in doc_rows]
        tune_pos.append((rows, [float(tune_qrels[qid][inputs.doc_ids[r]]) for r in rows]))
        if docs is not None:
            pool = _top_pool(docs, tune_vectors[i], set(rows))
            tune_pools[qid] = [inputs.doc_ids[r] for r in pool]
    if scorer is not None:
        for first in range(0, len(tune_ids), 256):
            span = range(first, min(first + 256, len(tune_ids)))
            vectors = torch.from_numpy(tune_vectors[list(span)].astype(np.float64))
            tops = scorer.top_many(vectors, 1000, [tune_pos[i][0] for i in span])
            for i, top in zip(span, tops, strict=True):
                tune_pools[tune_ids[i]] = [inputs.doc_ids[r] for r in top]
    tune_rows = {q: i for i, q in enumerate(tune_ids)}
    tune = retrieval_examples(
        tune_ids, tune_vectors, tune_rows, docs16, doc_rows, tune_qrels, tune_pools
    )
    start = None
    if args.arm == "official-init":
        state = torch.load(official / "state_dict.pt")
        start = {"projection.weight": state["net.0.weight"], "projection.bias": state["net.0.bias"]}
    manifest = start_manifest(
        "p3-bench train-s1",
        {
            "corpus": args.corpus,
            "arm": args.arm,
            "official_run": args.official_run,
            "cells": list(S1_CELLS),
            "tune_ids_sha256": id_fingerprint(tune_ids),
            "resource_record_sha256": file_sha256(args.resource_record),
        },
        **_manifest_kwargs(args.root, str(device), SEED),
    )

    def train_cell(
        cell: dict[str, float], seed: int
    ) -> tuple[torch.nn.Module, list[dict[str, float]], float]:
        config = RetrievalConfig(
            budgets=BUDGETS,
            eval_budgets=BUDGETS,
            steps=S1_STEPS,
            eval_every=S1_EVAL_EVERY,
            seed=seed,
            device=str(device),
            lr=cell["lr"],
            score_temperature=cell["score_temperature"],
            beta_end=cell["beta_end"],
        )
        tick = time.perf_counter()
        model, history, _ = train_retrieval_candidates(train, tune, config, start)
        return model, history, time.perf_counter() - tick

    cells = []
    states = {}
    for index, cell in enumerate(S1_CELLS):
        model, history, seconds = train_cell(cell, SEED)
        best = max(history, key=lambda row: row["val_rerank_ndcg10"])
        guard = history[-1]["soft_mass"]
        score = (
            _tune_score(model, tune_vectors, tune_pos, docs, device)
            if docs is not None
            else _tune_score_large(model, tune_vectors, tune_pos, scorer)  # type: ignore[arg-type]
        )
        cells.append(
            {
                "cell": index,
                **cell,
                "checkpoint_step": best["step"],
                "rerank": best["val_rerank_ndcg10"],
                "soft_mass_final": guard,
                "passes_guard": guard >= S1_GUARD,
                "tune_mean_ndcg10": score,
                "seconds": seconds,
            }
        )
        states[index] = {k: v.cpu() for k, v in model.state_dict().items()}
    selected = select_cell(cells)
    chosen = int(selected["cell"])
    seeds = {}
    for seed in SECONDARY_SEEDS:
        model, history, seconds = train_cell(S1_CELLS[chosen], seed)
        seeds[seed] = {k: v.cpu() for k, v in model.state_dict().items()}
    selection = {
        "cells": cells,
        "selected": chosen,
        "selection_candidates": len(cells) * (S1_STEPS // S1_EVAL_EVERY + 1),
        "tune_queries": len(tune_ids),
    }
    with ArtifactWriter(args.root, f"selectors/p3-s1/{args.corpus}/{args.arm}", manifest) as writer:
        for index, state in states.items():
            torch.save(state, writer.path(f"cell-{index}.pt"))
        torch.save(states[chosen], writer.path("selected-seed42.pt"))
        for seed, state in seeds.items():
            torch.save(state, writer.path(f"selected-seed{seed}.pt"))
        _write_json(writer.path("selection.json"), selection)
    print(json.dumps({"run_id": manifest.run_id, "arm": args.arm, "selected": selected}))
    return 0


# ----------------------------------------------------------------------------- evaluation lock


def _arm_files(root: Path, corpus: str, runs: dict[str, str]) -> dict[str, Path]:
    files = {"l2s-official@42": _official_dir(root, corpus, runs["official"]) / "state_dict.pt"}
    for key, run in runs.items():
        if key.startswith("arm:"):
            files[key[4:]] = root / "selectors" / "p3-teacher-arms" / corpus / run / "state_dict.pt"
    teachers = root / "evaluation" / "P3-bench" / "teachers" / corpus / runs["teachers"]
    if (teachers / "global_mask.npy").exists():  # undefined when rank-guided was removed
        files["global-mask"] = teachers / "global_mask.npy"
    for arm in ("scratch", "official-init"):
        base = root / "selectors" / "p3-s1" / corpus / arm / runs[f"s1:{arm}"]
        for seed in (SEED, *SECONDARY_SEEDS):
            files[f"s1-{arm}@{seed}"] = base / f"selected-seed{seed}.pt"
    return files


def cmd_evaluation_lock(args: argparse.Namespace) -> int:
    if args.out.exists():
        raise ContractError(f"{args.out} exists; freeze records are never overwritten")
    runs = json.loads(args.runs.read_text())
    files = _arm_files(args.root, args.corpus, runs)
    family = _family_members(args.family) if args.family else None
    if family is not None:
        present = {t for t in TEACHERS if f"{t}@{SEED}" in files}
        mine = {arm for corpus, arm in family if corpus == args.corpus}
        if mine != present:
            raise ContractError(f"family arms {sorted(mine)} differ from trained {sorted(present)}")
    split = load_query_split(args.root, args.corpus)
    realized = _require_record(args.resource_record, "resources")["realized"][args.corpus]
    evaluation_ids = hash_order(args.corpus, "evaluation", split.ids["evaluation"])[
        : realized["n_eval"]
    ]
    record = {
        "kind": "evaluation-lock",
        "created_at": datetime.now(UTC).isoformat(),
        "corpus": args.corpus,
        "runs": runs,
        "arm_sha256": {name: file_sha256(path) for name, path in sorted(files.items())},
        "evaluation_ids": evaluation_ids,
        "evaluation_ids_sha256": id_fingerprint(evaluation_ids),
        "resource_record_sha256": file_sha256(args.resource_record),
        "family": [f"{c}:{a}" for c, a in family] if family is not None else None,
        "conventions": (
            "linear arms select top-k of the selector on the L2-normalized query (torch.topk on "
            "the evaluation device); exact float64 scoring; lower-row ties; linear-gain nDCG; "
            "untrained controls full, variance, and mrl-prefix (the first k dimensions)"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.out, record)
    print(
        json.dumps(
            {
                "record": str(args.out),
                "sha256": file_sha256(args.out),
                "arms": len(files),
                "evaluation_queries": len(evaluation_ids),
            }
        )
    )
    return 0


# ----------------------------------------------------------------------------- evaluation


def _linear(state: dict[str, Tensor]) -> torch.nn.Module:
    prefix = "net.0." if "net.0.weight" in state else "projection."
    weight, bias = state[prefix + "weight"], state[prefix + "bias"]
    layer = torch.nn.Linear(weight.shape[1], weight.shape[0])
    layer.load_state_dict({"weight": weight, "bias": bias})
    return layer


def cmd_evaluate(args: argparse.Namespace) -> int:
    lock = _require_record(args.lock_record, "evaluation-lock")
    corpus = args.corpus
    files = _arm_files(args.root, corpus, lock["runs"])
    for name, path in files.items():
        if file_sha256(path) != lock["arm_sha256"][name]:
            raise ContractError(f"{name} changed since the evaluation lock")
    device = torch.device(args.device)
    inputs = load_fit_inputs(args.root, corpus, 1)  # documents and ids only
    eval_ids = lock["evaluation_ids"]
    qrels = _split_qrels(args.root, corpus, set(eval_ids))  # evaluation labels: read only here
    doc_rows = {d: i for i, d in enumerate(inputs.doc_ids)}
    eval_ids = [
        q for q in eval_ids if any(r > 0 and d in doc_rows for d, r in qrels.get(q, {}).items())
    ]
    vectors = _query_vectors(args.root, corpus, eval_ids, inputs)
    large = is_large(len(inputs.doc_ids))
    docs16 = inputs.documents if large else np.asarray(inputs.documents)
    scorer = CorpusScorer(docs16, device, chunk_rows=SCORER_CHUNK) if large else None
    docs = None if large else torch.from_numpy(docs16.astype(np.float64)).to(device)
    variance = scorer.variance() if scorer is not None else docs.var(0, unbiased=False)  # type: ignore[union-attr]
    normalized = _normalized(vectors, device)
    global_mask = np.load(files["global-mask"]) if "global-mask" in files else None
    importances: dict[str, Tensor] = {}
    with torch.no_grad():
        for name, path in files.items():
            if path.suffix == ".pt":
                importances[name] = _linear(torch.load(path)).to(device)(normalized)
    checkpoint = Checkpoint(
        args.root,
        "evaluate",
        corpus,
        {"lock_record_sha256": file_sha256(args.lock_record), "device": str(device)},
        args.max_hours,
    )
    rows_out: list[dict[str, Any]] = []
    first, block_start = 0, 0
    segments = {checkpoint.segment}
    for _, stop, _, meta in checkpoint.blocks():
        rows_out.extend(meta["rows"])
        segments.add(meta["segment"])
        first = block_start = stop
    block_rows = len(rows_out)
    for qi in range(first, len(eval_ids)):
        qid = eval_ids[qi]
        if qi > first and (qi - block_start == BLOCK_QUERIES):
            checkpoint.save(block_start, qi, {}, {"rows": rows_out[block_rows:]})
            block_start, block_rows = qi, len(rows_out)
            if checkpoint.expired():
                return checkpoint.stop(qi, len(eval_ids))
        rows = [doc_rows[d] for d, r in qrels[qid].items() if r > 0 and d in doc_rows]
        gains = torch.tensor(
            [float(qrels[qid][inputs.doc_ids[r]]) for r in rows], dtype=torch.float64
        )
        q = torch.from_numpy(vectors[qi].astype(np.float64)).to(device)
        pos = torch.tensor(rows, device=device)
        var_importance = q * q * variance
        if scorer is not None:
            rows_out.extend(_evaluate_large(scorer, qid, q, pos, gains, importances, qi,
                                            var_importance, global_mask))  # fmt: skip
            continue
        assert docs is not None
        full = positive_ranks(masked_scores(docs, q, None), pos).cpu()
        full_measures = query_measures(full, gains)
        for k in DECILES:
            arms: dict[str, Tensor | None] = {
                name: torch.topk(imp[qi], k).indices for name, imp in importances.items()
            }
            arms["variance"] = torch.topk(var_importance, k).indices
            arms["mrl-prefix"] = torch.arange(k, device=device)
            if global_mask is not None:
                arms["global-mask"] = (
                    torch.arange(1024, device=device)
                    if k == 1024
                    else torch.from_numpy(np.nonzero(global_mask[PRIMARY.index(k)])[0]).to(device)
                )
            rows_out.append({"query": qid, "k": k, "arm": "full", **full_measures})
            for name, dims in arms.items():
                ranks = positive_ranks(masked_scores(docs, q, dims), pos).cpu()
                if k == 1024 and not torch.equal(ranks, full):
                    raise ContractError(f"{name} at 1024 differs from the full vector for {qid}")
                rows_out.append({"query": qid, "k": k, "arm": name, **query_measures(ranks, gains)})
    if block_start < len(eval_ids):
        checkpoint.save(block_start, len(eval_ids), {}, {"rows": rows_out[block_rows:]})
    family_size = len(lock["family"]) if lock.get("family") else FAMILY_SIZE
    summary = _statistics(rows_out, corpus, eval_ids, family_size)
    summary["evaluation_queries"] = len(eval_ids)
    summary["full_vector_parity_at_1024"] = True
    summary["job_segments"] = len(segments)
    summary["provenance"] = evaluation_provenance(args.root, corpus)
    manifest = start_manifest(
        "p3-bench evaluate",
        {"corpus": corpus, "lock_record_sha256": file_sha256(args.lock_record)},
        **_manifest_kwargs(args.root, str(device), 20261112),
    )
    with ArtifactWriter(args.root, f"evaluation/P3-bench/evaluation/{corpus}", manifest) as writer:
        with writer.path("per_query.csv").open("w", newline="") as handle:
            out = csv.DictWriter(handle, fieldnames=list(rows_out[0]))
            out.writeheader()
            out.writerows(rows_out)
        _write_json(writer.path("summary.json"), summary)
        writer.path("README.md").write_text(_report(summary, corpus, manifest.run_id))
    print(json.dumps({"run_id": manifest.run_id, "primary": summary["primary"]}))
    return 0


def _evaluate_large(
    scorer: CorpusScorer,
    qid: str,
    q: Tensor,
    pos: Tensor,
    gains: Tensor,
    importances: dict[str, Tensor],
    qi: int,
    var_importance: Tensor,
    global_mask: np.ndarray | None,
) -> list[dict[str, Any]]:
    """One evaluation query on a large corpus: every arm and decile, batched corpus passes; the
    same rows, order, and 1,024 parity check as the direct path."""
    device = scorer.device
    jobs: list[tuple[str, int, Tensor | None]] = [("full", 0, None)]
    for k in DECILES:
        arms: dict[str, Tensor | None] = {
            name: torch.topk(imp[qi], k).indices for name, imp in importances.items()
        }
        arms["variance"] = torch.topk(var_importance, k).indices
        arms["mrl-prefix"] = torch.arange(k, device=device)
        if global_mask is not None:
            arms["global-mask"] = (
                torch.arange(1024, device=device)
                if k == 1024
                else torch.from_numpy(np.nonzero(global_mask[PRIMARY.index(k)])[0]).to(device)
            )
        jobs.extend((name, k, dims) for name, dims in arms.items())
    ranks: list[Tensor] = []
    for start in range(0, len(jobs), MASK_BATCH):
        chunk = jobs[start : start + MASK_BATCH]
        batch = torch.stack([masked_query(q, dims) for _, _, dims in chunk])
        ranks.extend(scorer.positive_ranks(batch, [pos] * len(chunk)))
    full = ranks[0]
    full_measures = query_measures(full, gains)
    out: list[dict[str, Any]] = []
    for (name, k, _), r in zip(jobs[1:], ranks[1:], strict=True):
        if k == 1024 and not torch.equal(r, full):
            raise ContractError(f"{name} at 1024 differs from the full vector for {qid}")
        if not out or out[-1]["k"] != k:
            out.append({"query": qid, "k": k, "arm": "full", **full_measures})
        out.append({"query": qid, "k": k, "arm": name, **query_measures(r, gains)})
    return out


def _per_query_mean(
    rows: list[dict[str, Any]], arm: str, ids: list[str], ks: tuple[int, ...]
) -> np.ndarray:
    table = {(r["query"], r["k"]): r["ndcg@10"] for r in rows if r["arm"] == arm}
    return np.array([np.mean([table[(q, k)] for k in ks]) for q in ids])


def _statistics(
    rows: list[dict[str, Any]], corpus: str, ids: list[str], family_size: int = FAMILY_SIZE
) -> dict[str, Any]:
    reference = _per_query_mean(rows, "l2s-official@42", ids, PRIMARY)
    present = {r["arm"] for r in rows}
    primary = {}
    for teacher in [t for t in TEACHERS if f"{t}@{SEED}" in present]:
        diffs = _per_query_mean(rows, f"{teacher}@{SEED}", ids, PRIMARY) - reference
        p = sign_flip_p(diffs, stream("signflip", corpus, teacher))
        bounds = intervals(
            bootstrap_means(diffs, stream("bootstrap", corpus, teacher)), family_size
        )
        spread = [
            float((_per_query_mean(rows, f"{teacher}@{s}", ids, PRIMARY) - reference).mean())
            for s in SECONDARY_SEEDS
        ]
        primary[teacher] = {
            "D": float(diffs.mean()),
            "p_one_sided": p,
            **bounds,
            "claim_level": claim_level(float(diffs.mean()), None, bounds["bonferroni_lower"]),
            "seed_D": spread,
            "seed_D_range": float(max(spread) - min(spread)),
            "seed_D_sd": float(np.std(spread, ddof=1)),
        }
    secondary = {}
    for arm, name in (
        ("s1-scratch@42", "s1-scratch"),
        ("s1-official-init@42", "s1-official-init"),
        ("global-mask", "global-mask"),
        ("variance", "variance"),
        ("mrl-prefix", "mrl-prefix"),
        ("full", "full"),
    ):
        if arm not in present:
            continue
        diffs = _per_query_mean(rows, arm, ids, PRIMARY) - reference
        bounds = intervals(bootstrap_means(diffs, stream("bootstrap", corpus, name)), family_size)
        secondary[name] = {
            "D": float(diffs.mean()),
            "pointwise_low": bounds["pointwise_low"],
            "pointwise_high": bounds["pointwise_high"],
        }
    arms = sorted({r["arm"] for r in rows})
    deciles = {
        arm: {
            str(k): float(np.mean([r["ndcg@10"] for r in rows if r["arm"] == arm and r["k"] == k]))
            for k in DECILES
        }
        for arm in arms
    }
    return {
        "primary": primary,
        "secondary": secondary,
        "per_decile_ndcg10": deciles,
        "family_size": family_size,
        "holm": "pending: computed over the locked family by `p3-bench family`",
    }


def _report(summary: dict[str, Any], corpus: str, run_id: str) -> str:
    lines = [
        f"# P3-bench {corpus}: evaluation `{run_id}`",
        "",
        "Generated from `summary.json` of this run. Primary: teacher arm minus `l2s-official`",
        "(seed 42), per-query mean nDCG@10 over deciles 102-921, every evaluation query weighted",
        "equally.",
        f"Holm and claim levels wait for the locked family ({summary['family_size']} members).",
        "",
        FRAMING,
        "",
        *provenance_lines(summary.get("provenance")),
        "| arm | D | one-sided p (unadjusted) | pointwise 95% | Bonferroni lower (a/m) "
        "| seed D (0,1,2) |",
        "|---|---|---|---|---|---|",
    ]
    for name, row in summary["primary"].items():
        seeds = ", ".join(f"{v:+.4f}" for v in row["seed_D"])
        lines.append(
            f"| {name} | {row['D']:+.4f} | {row['p_one_sided']:.4f} | "
            f"[{row['pointwise_low']:+.4f}, {row['pointwise_high']:+.4f}] | "
            f"{row['bonferroni_lower']:+.4f} | {seeds} |"
        )
    lines += [
        "",
        "Secondary (minus `l2s-official`, pointwise 95%):",
        "",
        "| arm | D | pointwise 95% |",
        "|---|---|---|",
    ]
    for name, row in summary["secondary"].items():
        lines.append(
            f"| {name} | {row['D']:+.4f} | "
            f"[{row['pointwise_low']:+.4f}, {row['pointwise_high']:+.4f}] |"
        )
    arms = list(summary["per_decile_ndcg10"])
    lines += [
        "",
        "Mean nDCG@10 per decile:",
        "",
        "| arm | " + " | ".join(map(str, DECILES)) + " |",
        "|---|" + "---|" * len(DECILES),
    ]
    for arm in arms:
        values = summary["per_decile_ndcg10"][arm]
        lines.append(f"| {arm} | " + " | ".join(f"{values[str(k)]:.4f}" for k in DECILES) + " |")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- escalation


ESCALATION_OPTIONS = ("remove-rank-guided", "not-run", "larger-budget")


def search_lower_bound_hours(
    searches: list[dict[str, Any]], searches_per_query: int, n_queries: int, safety: float
) -> float:
    """Least projected rank-guided cost: every search needs at least one full swap scan (the scan
    that finds no improving swap), timed at the fastest scan the probe measured."""
    fastest = min(t for search in searches for t in search["step_seconds"])
    return float(safety * fastest * searches_per_query * n_queries / 3600)


def cmd_escalate(args: argparse.Namespace) -> int:
    if args.out.exists():
        raise ContractError(f"{args.out} exists; freeze records are never overwritten")
    resources = _require_record(args.resource_record, "resources")
    if resources["corpus"] != args.corpus or resources["realized"][args.corpus]["step_cap"]:
        raise ContractError("escalation applies only to an escalated resource gate of this corpus")
    probe = json.loads(
        (args.root / "evaluation" / "P3-bench" / "probe" / args.corpus / resources["probe_run"])
        .joinpath("probe.json")
        .read_text()
    )
    realized = resources["realized"][args.corpus]
    searches_per_query = 2 * len(PRIMARY)
    lower = search_lower_bound_hours(
        probe["search"]["searches"],
        searches_per_query,
        realized["n_rg"],
        resources["safety_factor"],
    )
    fixed = resources["safety_factor"] * sum(resources["fixed_seconds"].values()) / 3600
    removed = ["rank-guided"] if args.option == "remove-rank-guided" else []
    record = {
        "kind": "escalation",
        "created_at": datetime.now(UTC).isoformat(),
        "corpus": args.corpus,
        "resource_record_sha256": file_sha256(args.resource_record),
        "option": args.option,
        "removed_primary_arms": removed,
        "removed_controls": ["global-mask"] if removed else [],
        "budget_hours": resources["budget_hours"],
        "rank_guided_lower_bound_hours": lower,
        "fixed_hours_with_safety": fixed,
        "reason": args.reason,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.out, record)
    print(json.dumps({"record": str(args.out), "sha256": file_sha256(args.out), **record}))
    return 0


# ----------------------------------------------------------------------------- family


def _family_members(path: Path) -> list[tuple[str, str]]:
    members = [tuple(m.split(":", 1)) for m in json.loads(path.read_text())["members"]]
    for corpus, arm in members:
        if corpus not in PUBLISHED or arm not in TEACHERS:
            raise ContractError(f"unknown family member {corpus}:{arm}")
    if len(set(members)) != len(members):
        raise ContractError("duplicate family member")
    return [(c, a) for c, a in members]


def family_table(
    per_query: dict[str, list[dict[str, Any]]], members: list[tuple[str, str]]
) -> list[dict[str, Any]]:
    """Per member: D, one-sided p, Holm-adjusted p over the members, pointwise interval, and the
    Bonferroni lower bound at the family's size. Streams are the per-corpus ones, so p and the
    pointwise interval equal each corpus's evaluation summary."""
    rows: list[dict[str, Any]] = []
    for corpus, arm in members:
        table = per_query[corpus]
        ids = list(dict.fromkeys(r["query"] for r in table))
        diffs = _per_query_mean(table, f"{arm}@{SEED}", ids, PRIMARY) - _per_query_mean(
            table, "l2s-official@42", ids, PRIMARY
        )
        bounds = intervals(bootstrap_means(diffs, stream("bootstrap", corpus, arm)), len(members))
        rows.append(
            {
                "corpus": corpus,
                "arm": arm,
                "queries": len(ids),
                "D": float(diffs.mean()),
                "p_one_sided": sign_flip_p(diffs, stream("signflip", corpus, arm)),
                **bounds,
            }
        )
    adjusted = holm({f"{r['corpus']}:{r['arm']}": r["p_one_sided"] for r in rows})
    for row in rows:
        row["holm_p"] = adjusted[f"{row['corpus']}:{row['arm']}"]
        row["claim_level"] = claim_level(row["D"], row["holm_p"], row["bonferroni_lower"])
    return rows


def _read_per_query(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        return [
            {**r, "k": int(r["k"]), "ndcg@10": float(r["ndcg@10"])} for r in csv.DictReader(handle)
        ]


def _family_report(rows: list[dict[str, Any]], sources: dict[str, Any], run_id: str) -> str:
    lines = [
        f"# P3-bench locked family: `{run_id}`",
        "",
        "Generated from the evaluation runs' `per_query.csv`. D: teacher arm minus `l2s-official`",
        "(seed 42), per-query mean nDCG@10 over deciles 102-921. Holm over the family's one-sided",
        f"sign-flip p-values; Bonferroni lower bound at alpha/m with m = {len(rows)}.",
        "",
        FRAMING,
        "",
        "| corpus | arm | queries | D | p (one-sided) | Holm p | pointwise 95% | Bonferroni lower "
        "| claim level |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['corpus']} | {r['arm']} | {r['queries']} | {r['D']:+.4f} | "
            f"{r['p_one_sided']:.4f} | {r['holm_p']:.4f} | "
            f"[{r['pointwise_low']:+.4f}, {r['pointwise_high']:+.4f}] | "
            f"{r['bonferroni_lower']:+.4f} | {r['claim_level']} |"
        )
    lines += ["", "Evaluation runs:", ""]
    lines += [f"- {c}: `{v['run']}` (per_query.csv SHA-256 `{v['per_query_sha256'][:12]}…`)"
              for c, v in sources.items()]  # fmt: skip
    for corpus, value in sources.items():
        if value.get("provenance"):
            lines += ["", f"{corpus}:", "", *provenance_lines(value["provenance"])]
    return "\n".join(lines) + "\n"


def cmd_family(args: argparse.Namespace) -> int:
    members = _family_members(args.family)
    runs = json.loads(args.evaluations.read_text())  # {corpus: evaluation run id}
    if {c for c, _ in members} - set(runs):
        raise ContractError("a family corpus has no evaluation run")
    per_query, sources = {}, {}
    for corpus, run in runs.items():
        directory = args.root / "evaluation" / "P3-bench" / "evaluation" / corpus / run
        per_query[corpus] = _read_per_query(directory / "per_query.csv")
        sources[corpus] = {
            "run": run,
            "per_query_sha256": file_sha256(directory / "per_query.csv"),
            "provenance": evaluation_provenance(args.root, corpus),
        }
    rows = family_table(per_query, members)
    manifest = start_manifest(
        "p3-bench family",
        {"members": [f"{c}:{a}" for c, a in members], "evaluations": sources},
        **_manifest_kwargs(args.root, "cpu", 20261112),
    )
    with ArtifactWriter(args.root, "evaluation/P3-bench/family", manifest) as writer:
        _write_json(writer.path("family.json"), {"members": len(members), "rows": rows})
        with writer.path("family.csv").open("w", newline="") as handle:
            out = csv.DictWriter(handle, fieldnames=list(rows[0]))
            out.writeheader()
            out.writerows(rows)
        writer.path("README.md").write_text(_family_report(rows, sources, manifest.run_id))
    print(json.dumps({"run_id": manifest.run_id, "rows": rows}))
    return 0


# ----------------------------------------------------------------------------- regenerated reports


def cmd_report(args: argparse.Namespace) -> int:
    """Regenerate an evaluation's report with every field the spec requires, from run files only.

    Writes a new run under ``evaluation/P3-bench/report/<corpus>``; the evaluation run itself is
    never modified, and no judgment is read (T060)."""
    corpus = args.corpus
    eval_dir = args.root / "evaluation" / "P3-bench" / "evaluation" / corpus / args.evaluation_run
    summary = json.loads((eval_dir / "summary.json").read_text())
    lock = _require_record(args.lock_record, "evaluation-lock")
    if read_manifest(eval_dir).config.get("lock_record_sha256") != file_sha256(args.lock_record):
        raise ContractError("the evaluation run was not made under this lock record")
    resources = _require_record(args.resource_record, "resources")
    probe_file = (
        args.root / "evaluation" / "P3-bench" / "probe" / corpus / resources["probe_run"]
    ) / "probe.json"
    probe = json.loads(probe_file.read_text()) if probe_file.exists() else None
    provenance = summary.get("provenance") or evaluation_provenance(args.root, corpus)
    forward = selector_forward_seconds(args.device)
    costs = arm_costs(
        args.root, corpus, lock["runs"], list(summary["per_decile_ndcg10"]), probe, forward,
        args.device,
    )  # fmt: skip
    manifest = start_manifest(
        "p3-bench report",
        {
            "corpus": corpus,
            "evaluation_run": args.evaluation_run,
            "lock_record_sha256": file_sha256(args.lock_record),
            "summary_sha256": file_sha256(eval_dir / "summary.json"),
        },
        **_manifest_kwargs(args.root, args.device, 20261112),
    )
    body = _report({**summary, "provenance": provenance}, corpus, args.evaluation_run)
    note = (
        f"Regenerated by report run `{manifest.run_id}` from evaluation run "
        f"`{args.evaluation_run}` and the runs its lock record names; no judgment was read."
    )
    text = body + "\n" + note + "\n\n" + "\n".join(cost_lines(costs))
    with ArtifactWriter(args.root, f"evaluation/P3-bench/report/{corpus}", manifest) as writer:
        _write_json(writer.path("costs.json"), costs)
        _write_json(writer.path("provenance.json"), provenance)
        writer.path("README.md").write_text(text)
    print(json.dumps({"run_id": manifest.run_id, "evaluation_run": args.evaluation_run}))
    return 0


def cmd_teacher_report(args: argparse.Namespace) -> int:
    """The fit-query teacher report (US2, FR-007), from a teachers run's files only."""
    run_dir = args.root / "evaluation" / "P3-bench" / "teachers" / args.corpus / args.teachers_run
    summary = json.loads((run_dir / "summary.json").read_text())
    searches_file = run_dir / "searches.json"
    gains = search_gains(json.loads(searches_file.read_text())) if searches_file.exists() else None
    manifest = start_manifest(
        "p3-bench teacher-report",
        {
            "corpus": args.corpus,
            "teachers_run": args.teachers_run,
            "summary_sha256": file_sha256(run_dir / "summary.json"),
        },
        **_manifest_kwargs(args.root, "cpu", 20261112),
    )
    with ArtifactWriter(
        args.root, f"evaluation/P3-bench/teacher-report/{args.corpus}", manifest
    ) as writer:
        _write_json(writer.path("gains.json"), gains)
        writer.path("README.md").write_text(
            teacher_report(summary, args.corpus, args.teachers_run, gains)
        )
    print(json.dumps({"run_id": manifest.run_id, "teachers_run": args.teachers_run}))
    return 0


# ----------------------------------------------------------------------------- registration


def register(sub: Any) -> None:
    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--root", type=Path, default=Path("artifacts"))
        p.add_argument("--corpus", required=True, choices=sorted(PUBLISHED))
        p.add_argument("--device", default="cpu")
        p.add_argument("--resource-record", type=Path)

    p = sub.add_parser("teachers")
    common(p)
    p.add_argument("--official-run", required=True)
    p.add_argument("--escalation-record", type=Path)
    p.add_argument("--max-hours", type=float, help="stop at a block boundary (exit 75); rerun")
    p.set_defaults(handler=cmd_teachers)

    p = sub.add_parser("train-arm")
    common(p)
    p.add_argument("--official-run", required=True)
    p.add_argument("--teachers-run")
    p.add_argument("--target", required=True, choices=("official", *TEACHERS))
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--clone", type=Path, default=DEFAULT_CLONE)
    p.set_defaults(handler=cmd_train_arm)

    p = sub.add_parser("train-s1")
    common(p)
    p.add_argument("--official-run", required=True)
    p.add_argument("--arm", required=True, choices=("scratch", "official-init"))
    p.set_defaults(handler=cmd_train_s1)

    p = sub.add_parser("evaluation-lock")
    common(p)
    p.add_argument("--runs", type=Path, required=True, help="JSON naming every arm's run")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--family", type=Path, help='JSON {"members": ["corpus:arm", ...]}')
    p.set_defaults(handler=cmd_evaluation_lock)

    p = sub.add_parser("evaluate")
    common(p)
    p.add_argument("--lock-record", type=Path, required=True)
    p.add_argument("--max-hours", type=float, help="stop at a block boundary (exit 75); rerun")
    p.set_defaults(handler=cmd_evaluate)

    p = sub.add_parser("escalate")
    common(p)
    p.add_argument("--option", required=True, choices=ESCALATION_OPTIONS)
    p.add_argument("--reason", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(handler=cmd_escalate)

    p = sub.add_parser("report", help="regenerate an evaluation report from run files (T060)")
    common(p)
    p.add_argument("--evaluation-run", required=True)
    p.add_argument("--lock-record", type=Path, required=True)
    p.set_defaults(handler=cmd_report)

    p = sub.add_parser("teacher-report", help="fit-query teacher report (FR-007)")
    p.add_argument("--root", type=Path, default=Path("artifacts"))
    p.add_argument("--corpus", required=True, choices=sorted(PUBLISHED))
    p.add_argument("--teachers-run", required=True)
    p.set_defaults(handler=cmd_teacher_report)

    p = sub.add_parser("family")
    p.add_argument("--root", type=Path, default=Path("artifacts"))
    p.add_argument("--family", type=Path, required=True)
    p.add_argument("--evaluations", type=Path, required=True, help="JSON {corpus: run id}")
    p.set_defaults(handler=cmd_family)


__all__ = ["register"]
