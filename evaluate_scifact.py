"""Evaluate an unmasked, random-mask, and learned-mask retriever on BEIR SciFact."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import faiss
import numpy as np
import torch
from beir.datasets.data_loader import GenericDataLoader
from transformers import AutoTokenizer

from inference import load_predictor
from models import HyperMaskConfig, select_top_k
from prepare_embeddings import format_query


def parse_k_values(value: str) -> list[float]:
    values = [float(item) for item in value.split(",")]
    if not values or any(not 0 < item <= 1 for item in values):
        raise ValueError("k-values must be a comma-separated list of ratios in (0, 1].")
    return values


def encode_queries(
    model: object,
    tokenizer: object,
    queries: dict[str, str],
    instruction: str,
    batch_size: int,
    device: torch.device,
) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    """Encode all test queries and predict their learned dimension importance."""
    query_ids = sorted(queries)
    embeddings: list[torch.Tensor] = []
    importances: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(query_ids), batch_size):
            texts = [format_query(queries[id_], instruction) for id_ in query_ids[start : start + batch_size]]
            encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")  # type: ignore[operator]
            encoded = {name: value.to(device) for name, value in encoded.items()}
            embedding, importance = model.predict_embedding_and_importance(**encoded)  # type: ignore[attr-defined]
            embeddings.append(embedding.cpu())
            importances.append(importance.cpu())
    return query_ids, torch.cat(embeddings), torch.cat(importances)


def metric_summary(
    ranked_ids: np.ndarray, query_ids: list[str], qrels: dict[str, dict[str, int]], cutoff: int
) -> dict[str, float]:
    """Compute standard graded nDCG, binary recall, and reciprocal-rank averages."""
    ndcgs: list[float] = []
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    for query_id, ranking in zip(query_ids, ranked_ids, strict=True):
        labels = {document_id: score for document_id, score in qrels[query_id].items() if score > 0}
        gains = [labels.get(document_id, 0) for document_id in ranking[:cutoff]]
        dcg = sum((2**gain - 1) / math.log2(rank + 2) for rank, gain in enumerate(gains))
        ideal = sorted(labels.values(), reverse=True)[:cutoff]
        idcg = sum((2**gain - 1) / math.log2(rank + 2) for rank, gain in enumerate(ideal))
        ndcgs.append(dcg / idcg if idcg else 0.0)
        recalls.append(sum(document_id in labels for document_id in ranking[:cutoff]) / len(labels))
        first_relevant_rank = next(
            (rank for rank, document_id in enumerate(ranking[:cutoff], start=1) if document_id in labels), None
        )
        reciprocal_ranks.append(1 / first_relevant_rank if first_relevant_rank else 0.0)
    return {
        f"ndcg@{cutoff}": float(np.mean(ndcgs)),
        f"recall@{cutoff}": float(np.mean(recalls)),
        f"mrr@{cutoff}": float(np.mean(reciprocal_ranks)),
    }


def retrieve(index: faiss.Index, embeddings: torch.Tensor, document_ids: list[str], cutoff: int) -> np.ndarray:
    """Search the fixed document index and convert integer results to BEIR IDs."""
    _, indices = index.search(embeddings.float().numpy(), cutoff)
    return np.asarray([[document_ids[index_] for index_ in row] for row in indices])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--data-dir", default="data/scifact")
    parser.add_argument("--query-instruction", default="Given a scientific claim, retrieve relevant scientific abstracts.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--k-values", default="0.1,0.2,0.3,0.5")
    parser.add_argument("--cutoff", type=int, default=100)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output", help="Optional JSON result path.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.cutoff < 10:
        raise ValueError("batch-size must be positive and cutoff must be at least 10.")
    data_dir = Path(args.data_dir)
    document_embeddings = np.load(data_dir / "document_embeddings.npy")
    document_ids = json.loads((data_dir / "document_ids.json").read_text())
    if len(document_ids) != len(document_embeddings):
        raise ValueError("document_ids.json and document_embeddings.npy have different lengths.")
    _, queries, qrels = GenericDataLoader(data_folder=str(data_dir / "raw" / "scifact")).load(split="test")
    device = torch.device(args.device)
    model, metadata = load_predictor(args.checkpoint_dir, device)
    config = HyperMaskConfig(**metadata["model_config"])
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=config.trust_remote_code)
    query_ids, embeddings, importance = encode_queries(
        model, tokenizer, queries, args.query_instruction, args.batch_size, device
    )
    index = faiss.IndexFlatIP(config.hidden_dim)
    index.add(np.ascontiguousarray(document_embeddings.astype(np.float32, copy=False)))
    cutoff = min(args.cutoff, len(document_ids))
    results: dict[str, dict[str, float]] = {
        "unmasked": metric_summary(retrieve(index, embeddings, document_ids, cutoff), query_ids, qrels, cutoff)
    }
    generator = torch.Generator().manual_seed(args.seed)
    for k in parse_k_values(args.k_values):
        count = round(k * config.hidden_dim)
        learned = select_top_k(importance, embeddings, k)
        random_indices = torch.rand((len(query_ids), config.hidden_dim), generator=generator).topk(count, dim=-1).indices
        random_mask = torch.zeros_like(embeddings).scatter_(1, random_indices, 1)
        results[f"learned_k={k:g}"] = metric_summary(
            retrieve(index, learned, document_ids, cutoff), query_ids, qrels, cutoff
        )
        results[f"random_k={k:g}"] = metric_summary(
            retrieve(index, embeddings * random_mask, document_ids, cutoff), query_ids, qrels, cutoff
        )
    payload = {"dataset": "beir/scifact", "test_queries": len(query_ids), "cutoff": cutoff, "metrics": results}
    print(json.dumps(payload, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
