"""Spike steps: prepare, train, train-global, and analyze (research R0)."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import Tensor

from hyperdime.baselines.learning_to_select import LinearSelector
from hyperdime.contracts.hashing import file_sha256
from hyperdime.data.loaders import beir_fingerprint, download_beir, read_beir
from hyperdime.data.negatives import mine_hard_negatives
from hyperdime.embeddings.cache import embed_cached, load_cached
from hyperdime.evaluation.metrics import query_metrics
from hyperdime.evaluation.selector_shift import (
    js_divergence,
    mean_importance,
    overlap_at_k,
    spearman,
)
from hyperdime.evaluation.statistics import holm, paired_bootstrap_ci, paired_randomization_test
from hyperdime.oracle.targets import aggregate_targets
from hyperdime.retrieval.scoring import rank_documents
from hyperdime.selection.topk import apply_mask, resolve_k, top_k_mask
from hyperdime.spike.environments import ENVIRONMENTS, SPLIT_SEED, make_splits, sample_corpus
from hyperdime.spike.manifest import read_json, write_json, write_manifest
from hyperdime.training.selector_trainer import TrainConfig, train_selector

DIM = 1024
HEADLINE_K = resolve_k(0.3, DIM)
K_VALUES = (128, 256, HEADLINE_K, 512)
TEMPERATURES = (0.01, 0.05, 0.1)
NEGATIVE_DEPTH = 100
NEGATIVE_POOL = 8
RANKING_DEPTH = 100
METRICS = ("ndcg@10", "recall@100")
ALPHA = 0.05

PerQuery = dict[str, dict[str, float]]


class Encoder(Protocol):
    """Local ``QwenEncoder`` or ``VllmEncoder``; ``settings`` identify the vectors it makes."""

    @property
    def settings(self) -> dict[str, Any]: ...

    def encode(self, texts: Sequence[str], progress: str = "") -> Tensor: ...

    def encode_queries(self, texts: Sequence[str], instruction: str | None) -> Tensor: ...


@dataclass
class EnvData:
    doc_ids: list[str]
    docs: Tensor
    doc_index: dict[str, int]
    query_index: dict[str, int]
    queries: Tensor
    splits: dict[str, Any]
    negatives: dict[str, list[str]]

    def query_rows(self, qids: Sequence[str]) -> Tensor:
        return self.queries[[self.query_index[qid] for qid in qids]]


def load_env(root: Path, env: str) -> EnvData:
    doc_ids, docs = load_cached(root / env / "corpus")
    query_ids, queries = load_cached(root / env / "queries")
    return EnvData(
        doc_ids=doc_ids,
        docs=docs,
        doc_index={doc_id: row for row, doc_id in enumerate(doc_ids)},
        query_index={qid: row for row, qid in enumerate(query_ids)},
        queries=queries,
        splits=read_json(root / env / "splits.json"),
        negatives=read_json(root / env / "negatives.json"),
    )


def prepare(env: str, root: Path, encoder: Encoder, corpus_cap: int) -> None:
    """Download, split, embed corpus and queries, and mine negatives for training queries."""
    spec = ENVIRONMENTS[env]
    dataset = read_beir(download_beir(env, root / "beir"))
    splits = make_splits(spec, dataset)
    write_json(root / env / "splits.json", asdict(splits))
    settings = {**encoder.settings, "corpus_cap": corpus_cap}
    sampled = sample_corpus(dataset, splits, corpus_cap)
    doc_ids = sorted(sampled, key=lambda doc_id: (len(dataset.corpus[doc_id]), doc_id))
    doc_ids, docs = embed_cached(
        doc_ids,
        [dataset.corpus[doc_id] for doc_id in doc_ids],
        lambda texts: encoder.encode(texts, progress=f"{env} corpus"),
        root / env / "corpus",
        settings=settings,
    )
    _cache_manifest(root / env / "corpus", settings, len(doc_ids))
    qids = splits.train + splits.validation + splits.evaluation
    qids, queries = embed_cached(
        qids,
        [dataset.queries[qid] for qid in qids],
        lambda texts: encoder.encode_queries(texts, spec.instruction),
        root / env / "queries",
        settings={**settings, "instruction": spec.instruction},
    )
    _cache_manifest(
        root / env / "queries", {**settings, "instruction": spec.instruction}, len(qids)
    )
    index = {qid: row for row, qid in enumerate(qids)}
    labelled = splits.train + splits.validation
    negatives = _mine(
        queries[[index[qid] for qid in labelled]], docs, doc_ids, labelled, splits.train_labels
    )
    write_json(root / env / "negatives.json", negatives)
    write_manifest(
        root / env,
        "prepare",
        {
            "env": env,
            "dataset_hash": beir_fingerprint(root / "beir", env),
            "encoder": encoder.settings,
            "instruction": spec.instruction,
            "split_seed": SPLIT_SEED,
            "negative_depth": NEGATIVE_DEPTH,
            "negative_pool": NEGATIVE_POOL,
            "counts": {
                name: len(getattr(splits, name)) for name in ("train", "validation", "evaluation")
            },
            "num_documents": len(doc_ids),
            "corpus_size": len(dataset.corpus),
            "corpus_cap": corpus_cap,
        },
    )


def _cache_manifest(directory: Path, settings: Mapping[str, Any], rows: int) -> None:
    """Give a finished embedding cache its own manifest, so no vectors exist without one."""
    write_manifest(
        directory,
        "embed",
        {
            "settings": dict(settings),
            "rows": rows,
            "embeddings_sha256": file_sha256(directory / "embeddings.npy"),
            "ids_sha256": file_sha256(directory / "ids.json"),
        },
    )


def train(env: str, root: Path, seeds: Sequence[int]) -> None:
    """Pick the temperature on validation with seed 0, then train one selector per seed."""
    data = load_env(root, env)
    split = data.splits
    labels = split["train_labels"]
    validation_scores: dict[str, float] = {}
    for temperature in TEMPERATURES:
        model, _ = _fit(data, split, temperature, seed=0)
        scores = _evaluate(
            _importance(model, data.query_rows(split["validation"])),
            HEADLINE_K,
            data,
            split["validation"],
            labels,
        )
        validation_scores[str(temperature)] = _mean(scores, "ndcg@10")
    temperature = max(TEMPERATURES, key=lambda t: validation_scores[str(t)])
    directory = root / env / "selectors"
    for seed in seeds:
        model, history = _fit(data, split, temperature, seed)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), directory / f"seed-{seed}.pt")
        write_json(directory / f"history-seed-{seed}.json", history)
    write_json(
        directory / "temperature.json",
        {"temperature": temperature, "validation_ndcg@10": validation_scores},
    )
    write_manifest(
        directory,
        "train",
        {
            "env": env,
            "seeds": list(seeds),
            "temperature": temperature,
            "train_config": asdict(TrainConfig()),
        },
    )


def train_global(envs: Sequence[str], root: Path, seeds: Sequence[int]) -> None:
    """One selector on the union of every environment's training labels (risk R6 control)."""
    parts: dict[str, list[Tensor]] = {"tq": [], "tt": [], "vq": [], "vt": []}
    for env in envs:
        data = load_env(root, env)
        split = data.splits
        temperature = read_json(root / env / "selectors" / "temperature.json")["temperature"]
        for name, key in (("train", "t"), ("validation", "v")):
            parts[f"{key}q"].append(data.query_rows(split[name]))
            parts[f"{key}t"].append(
                _targets(data, split[name], split["train_labels"], data.negatives, temperature)
            )
    directory = root / "global"
    directory.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        model, history = train_selector(
            torch.cat(parts["tq"]),
            torch.cat(parts["tt"]),
            torch.cat(parts["vq"]),
            torch.cat(parts["vt"]),
            TrainConfig(seed=seed),
        )
        torch.save(model.state_dict(), directory / f"seed-{seed}.pt")
        write_json(directory / f"history-seed-{seed}.json", history)
    write_manifest(
        directory,
        "train-global",
        {"envs": list(envs), "seeds": list(seeds), "train_config": asdict(TrainConfig())},
    )


def analyze(envs: Sequence[str], root: Path, seeds: Sequence[int]) -> dict[str, Any]:
    """Transfer matrix, baselines, mask behavior, and the H1 verdict."""
    selectors = {
        env: [_load_selector(root / env / "selectors" / f"seed-{s}.pt") for s in seeds]
        for env in envs
    }
    global_models = [_load_selector(root / "global" / f"seed-{s}.pt") for s in seeds]
    results: dict[str, dict[str, dict[str, PerQuery]]] = {}
    battery_parts: list[Tensor] = []
    for target in envs:
        data = load_env(root, target)
        qids, labels = data.splits["evaluation"], data.splits["eval_labels"]
        queries = data.query_rows(qids)
        battery_parts.append(queries)
        prefix = torch.arange(DIM, 0, -1, dtype=torch.float32).expand(len(qids), DIM)
        randoms = [
            torch.rand(len(qids), DIM, generator=torch.Generator().manual_seed(s)) for s in seeds
        ]
        temperature = read_json(root / target / "selectors" / "temperature.json")["temperature"]
        # Ceiling only: oracle targets use evaluation labels and never feed any trained model.
        eval_negatives = _mine(queries, data.docs, data.doc_ids, qids, labels)
        oracle = _targets(data, qids, labels, eval_negatives, temperature)
        systems: dict[str, list[Tensor]] = {
            "prefix": [prefix],
            "random": randoms,
            "oracle": [oracle],
            "global": [_importance(m, queries) for m in global_models],
        }
        for source in envs:
            systems[f"selector:{source}"] = [_importance(m, queries) for m in selectors[source]]
        per_env: dict[str, dict[str, PerQuery]] = {
            "full": {str(DIM): _run([None], DIM, data, qids, labels)}
        }
        for name, importances in systems.items():
            per_env[name] = {str(k): _run(importances, k, data, qids, labels) for k in K_VALUES}
        results[target] = per_env
    battery = torch.cat(battery_parts)
    analysis: dict[str, Any] = {
        "config": {
            "envs": list(envs),
            "seeds": list(seeds),
            "headline_k": HEADLINE_K,
            "k_values": list(K_VALUES),
            "alpha": ALPHA,
            "battery_size": battery.shape[0],
            "temperatures": {
                env: read_json(root / env / "selectors" / "temperature.json") for env in envs
            },
            "counts": {
                env: read_json(root / env / "manifest.json")["config"]["counts"] for env in envs
            },
        },
        "summary": {
            env: {
                name: {k: {m: _mean(pq, m) for m in METRICS} for k, pq in by_k.items()}
                for name, by_k in systems_.items()
            }
            for env, systems_ in results.items()
        },
        "mask": _mask_behavior(selectors, battery),
    }
    analysis["criteria"] = _criteria(results, analysis["mask"], envs)
    write_json(root / "analysis" / "results.json", results)
    write_json(root / "analysis" / "analysis.json", analysis)
    write_manifest(root / "analysis", "analyze", analysis["config"])
    return analysis


def _fit(
    data: EnvData, split: Mapping[str, Any], temperature: float, seed: int
) -> tuple[LinearSelector, list[dict[str, float]]]:
    labels = split["train_labels"]
    return train_selector(
        data.query_rows(split["train"]),
        _targets(data, split["train"], labels, data.negatives, temperature),
        data.query_rows(split["validation"]),
        _targets(data, split["validation"], labels, data.negatives, temperature),
        TrainConfig(seed=seed),
    )


def _targets(
    data: EnvData,
    qids: Sequence[str],
    labels: Mapping[str, Mapping[str, int]],
    negatives: Mapping[str, Sequence[str]],
    temperature: float,
) -> Tensor:
    positives = [
        {data.doc_index[d]: g for d, g in labels[q].items() if g > 0 and d in data.doc_index}
        for q in qids
    ]
    pools = [[data.doc_index[d] for d in negatives[q]] for q in qids]
    return aggregate_targets(data.query_rows(qids), data.docs, positives, pools, temperature)


def _mine(
    queries: Tensor,
    docs: Tensor,
    doc_ids: Sequence[str],
    qids: Sequence[str],
    labels: Mapping[str, Mapping[str, int]],
) -> dict[str, list[str]]:
    """Top non-positive documents per query; the query's own ID is excluded too (ArguAna)."""
    excluded = [{d for d, g in labels[q].items() if g > 0} | {q} for q in qids]
    depth = min(len(doc_ids), max(NEGATIVE_DEPTH, max(map(len, excluded)) + NEGATIVE_POOL))
    pools = mine_hard_negatives(queries, docs, doc_ids, excluded, NEGATIVE_POOL, depth)
    return dict(zip(qids, pools, strict=True))


def _importance(model: LinearSelector, queries: Tensor) -> Tensor:
    with torch.no_grad():
        log_importance: Tensor = model(queries)
    return log_importance


def _evaluate(
    importance: Tensor | None,
    k: int,
    data: EnvData,
    qids: Sequence[str],
    labels: Mapping[str, Mapping[str, int]],
) -> PerQuery:
    queries = data.query_rows(qids)
    if importance is not None:
        queries = apply_mask(queries, top_k_mask(importance, k))
    rankings = rank_documents(queries, data.docs, qids, data.doc_ids, RANKING_DEPTH)
    per_query: PerQuery = {}
    for qid in qids:
        values = query_metrics(rankings[qid], labels[qid], (10, 100))
        per_query[qid] = {m: values[m] for m in METRICS}
    return per_query


def _run(
    importances: Sequence[Tensor | None],
    k: int,
    data: EnvData,
    qids: Sequence[str],
    labels: Mapping[str, Mapping[str, int]],
) -> PerQuery:
    """Per-query metrics averaged over runs (seeds) of one system."""
    return _average([_evaluate(imp, k, data, qids, labels) for imp in importances])


def _average(runs: Sequence[PerQuery]) -> PerQuery:
    return {
        qid: {m: sum(run[qid][m] for run in runs) / len(runs) for m in METRICS} for qid in runs[0]
    }


def _mean(per_query: PerQuery, metric: str) -> float:
    return sum(values[metric] for values in per_query.values()) / len(per_query)


def _load_selector(path: Path) -> LinearSelector:
    model = LinearSelector(DIM)
    model.load_state_dict(torch.load(path, weights_only=True))
    return model.eval()


def _mask_behavior(
    selectors: Mapping[str, Sequence[LinearSelector]], battery: Tensor
) -> dict[str, Any]:
    """Cross-seed vs cross-environment mask overlap on a label-free query battery."""
    log_importance = {
        env: [_importance(m, battery) for m in models] for env, models in selectors.items()
    }
    masks = {env: [top_k_mask(li, HEADLINE_K) for li in lis] for env, lis in log_importance.items()}
    envs = list(selectors)
    cross_seed = torch.stack(
        [overlap_at_k(a, b) for env in envs for a, b in itertools.combinations(masks[env], 2)]
    ).mean(dim=0)
    cross_env = torch.stack(
        [
            overlap_at_k(a, b)
            for x, y in itertools.combinations(envs, 2)
            for a in masks[x]
            for b in masks[y]
        ]
    ).mean(dim=0)
    means = {
        env: torch.stack([mean_importance(li) for li in lis]) for env, lis in log_importance.items()
    }
    centroid = {env: m.mean(dim=0) for env, m in means.items()}
    return {
        "overlap_cross_seed": float(cross_seed.mean()),
        "overlap_cross_env": float(cross_env.mean()),
        "overlap_p": paired_randomization_test(cross_seed.tolist(), cross_env.tolist()),
        "overlap_diff_ci": paired_bootstrap_ci(cross_seed.tolist(), cross_env.tolist()),
        "js_env": {x: {y: js_divergence(centroid[x], centroid[y]) for y in envs} for x in envs},
        "js_seed": {env: _mean_pairwise(means[env], js_divergence) for env in envs},
        "spearman_env": {x: {y: spearman(centroid[x], centroid[y]) for y in envs} for x in envs},
        "spearman_seed": {env: _mean_pairwise(means[env], spearman) for env in envs},
    }


def _mean_pairwise(rows: Tensor, fn: Callable[[Tensor, Tensor], float]) -> float:
    values = [fn(a, b) for a, b in itertools.combinations(rows, 2)]
    return sum(values) / len(values)


def _paired(results: Mapping[str, Any], env: str, better: str, worse: str) -> dict[str, Any]:
    a = results[env][better][str(HEADLINE_K)]
    b = results[env][worse][str(HEADLINE_K)]
    qids = sorted(a)
    xs, ys = [a[q]["ndcg@10"] for q in qids], [b[q]["ndcg@10"] for q in qids]
    return {
        "diff": sum(xs) / len(xs) - sum(ys) / len(ys),
        "p": paired_randomization_test(xs, ys),
        "ci": paired_bootstrap_ci(xs, ys),
    }


def _criteria(
    results: Mapping[str, Any], mask: Mapping[str, Any], envs: Sequence[str]
) -> dict[str, Any]:
    """Research R0's three-part rule for H1 at the headline k (nDCG@10)."""
    transfer = {
        f"{src}->{dst}": _paired(results, dst, f"selector:{dst}", f"selector:{src}")
        for dst in envs
        for src in envs
        if src != dst
    }
    _adjust(transfer)
    transfer_hits = [name for name, t in transfer.items() if t["p_holm"] < ALPHA and t["diff"] > 0]
    headroom = {env: _paired(results, env, f"selector:{env}", "global") for env in envs}
    _adjust(headroom)
    headroom_hits = [env for env, t in headroom.items() if t["p_holm"] < ALPHA and t["diff"] > 0]
    c1 = len(transfer_hits) >= len(transfer) / 2
    c2 = mask["overlap_p"] < ALPHA and mask["overlap_cross_seed"] > mask["overlap_cross_env"]
    c3 = len(headroom_hits) >= 2
    return {
        "transfer": transfer,
        "transfer_significant": transfer_hits,
        "headroom": headroom,
        "headroom_significant": headroom_hits,
        "c1_transfer_loss": c1,
        "c2_masks_shift": c2,
        "c3_headroom_over_global": c3,
        "verdict": "supported" if c1 and c2 and c3 else "not supported",
    }


def _adjust(tests: dict[str, dict[str, Any]]) -> None:
    adjusted = holm({name: t["p"] for name, t in tests.items()})
    for name, t in tests.items():
        t["p_holm"] = adjusted[name]
