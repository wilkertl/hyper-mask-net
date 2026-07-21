"""Download BEIR SciFact and create train/validation relevance triples.

The BEIR ``train`` qrels are split by query for model training and validation.
Its ``test`` qrels remain untouched for retrieval evaluation.  Negatives are
mined from the SciFact corpus with the frozen Qwen encoder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss
import numpy as np
import torch
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from transformers import AutoModel, AutoTokenizer

from prepare_embeddings import encode_texts, format_query

SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"


def document_text(document: dict[str, str]) -> str:
    """Combine BEIR title and body without adding an empty leading line."""
    return "\n".join(part for part in (document.get("title", ""), document["text"]) if part)


def encode_in_batches(
    model: torch.nn.Module,
    tokenizer: object,
    texts: list[str],
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> np.ndarray:
    """Return normalized float32 embeddings in the input order."""
    parts = [
        encode_texts(model, tokenizer, texts[start : start + batch_size], max_length, device)
        for start in range(0, len(texts), batch_size)
    ]
    return torch.cat(parts).numpy().astype(np.float32, copy=False)


def split_query_ids(query_ids: list[str], validation_fraction: float, seed: int) -> tuple[list[str], list[str]]:
    """Split distinct BEIR training query IDs deterministically."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation-fraction must be in (0, 1).")
    ordered = np.array(sorted(query_ids))
    if len(ordered) < 2:
        raise ValueError("SciFact train split needs at least two labeled queries.")
    rng = np.random.default_rng(seed)
    rng.shuffle(ordered)
    validation_count = min(max(1, round(len(ordered) * validation_fraction)), len(ordered) - 1)
    return sorted(ordered[validation_count:].tolist()), sorted(ordered[:validation_count].tolist())


def build_triples(
    query_ids: list[str],
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    corpus: dict[str, dict[str, str]],
    index: faiss.Index,
    document_ids: list[str],
    query_embeddings: np.ndarray,
    negative_depth: int,
) -> list[dict[str, str]]:
    """Pair each query with a judged positive and a retrieved hard negative."""
    triples: list[dict[str, str]] = []
    for query_id, embedding in zip(query_ids, query_embeddings, strict=True):
        positives = sorted(document_id for document_id, score in qrels[query_id].items() if score > 0)
        if not positives:
            continue
        _, candidate_indices = index.search(embedding[None], negative_depth)
        positive_set = set(positives)
        negative_id = next(
            (document_ids[index_] for index_ in candidate_indices[0] if document_ids[index_] not in positive_set),
            None,
        )
        if negative_id is None:
            raise ValueError(f"No hard negative found for SciFact query {query_id}.")
        triples.append(
            {
                "query": queries[query_id],
                "positive": document_text(corpus[positives[0]]),
                "negative": document_text(corpus[negative_id]),
            }
        )
    if not triples:
        raise ValueError("No triples were created from the selected SciFact split.")
    return triples


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/scifact")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--query-instruction", default="Given a scientific claim, retrieve relevant scientific abstracts.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--negative-depth", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.negative_depth < 2:
        raise ValueError("batch-size must be positive and negative-depth must be at least 2.")
    output_dir = Path(args.output_dir)
    raw_dir = Path(util.download_and_unzip(SCIFACT_URL, str(output_dir / "raw")))
    corpus, queries, qrels = GenericDataLoader(data_folder=str(raw_dir)).load(split="train")
    query_ids = sorted(query_id for query_id, labels in qrels.items() if any(score > 0 for score in labels.values()))
    train_ids, validation_ids = split_query_ids(query_ids, args.validation_fraction, args.seed)

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True).to(device).eval()
    document_ids = sorted(corpus)
    with torch.inference_mode():
        document_embeddings = encode_in_batches(
            model, tokenizer, [document_text(corpus[id_]) for id_ in document_ids],
            args.batch_size, args.max_length, device,
        )
        train_embeddings = encode_in_batches(
            model, tokenizer, [format_query(queries[id_], args.query_instruction) for id_ in train_ids],
            args.batch_size, args.max_length, device,
        )
        validation_embeddings = encode_in_batches(
            model, tokenizer, [format_query(queries[id_], args.query_instruction) for id_ in validation_ids],
            args.batch_size, args.max_length, device,
        )
    index = faiss.IndexFlatIP(document_embeddings.shape[1])
    index.add(document_embeddings)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "document_embeddings.npy", document_embeddings)
    (output_dir / "document_ids.json").write_text(json.dumps(document_ids) + "\n")
    write_jsonl(
        output_dir / "train.jsonl",
        build_triples(train_ids, queries, qrels, corpus, index, document_ids, train_embeddings, args.negative_depth),
    )
    write_jsonl(
        output_dir / "validation.jsonl",
        build_triples(
            validation_ids, queries, qrels, corpus, index, document_ids, validation_embeddings, args.negative_depth
        ),
    )
    (output_dir / "split.json").write_text(
        json.dumps({"seed": args.seed, "train_query_ids": train_ids, "validation_query_ids": validation_ids}, indent=2)
        + "\n"
    )
    print(f"Wrote {len(train_ids)} train and {len(validation_ids)} validation SciFact triples to {output_dir}")


if __name__ == "__main__":
    main()
