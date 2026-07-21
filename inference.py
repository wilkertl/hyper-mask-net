"""Mask a raw query and run exact FAISS inner-product retrieval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss
import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from models import HyperMaskConfig, HyperMaskNet, LinearMaskPredictor, select_top_k
from prepare_embeddings import format_query


def load_predictor(checkpoint_dir: str | Path, device: torch.device) -> tuple[object, dict]:
    """Load a saved predictor and its frozen Qwen encoder."""
    checkpoint = Path(checkpoint_dir)
    metadata = json.loads((checkpoint / "config.json").read_text())
    config = HyperMaskConfig(**metadata["model_config"])
    weights = load_file(checkpoint / "model.safetensors", device=str(device))
    if metadata["architecture"] == "hyper":
        model = HyperMaskNet(config).to(device).eval()
        model.hyperhead.load_state_dict(weights)
    elif metadata["architecture"] == "linear":
        model = HyperMaskNet(config).to(device).eval()
        model.hyperhead = LinearMaskPredictor(
            config.hidden_dim, config.prediction_temperature
        ).to(device).eval()
        model.hyperhead.load_state_dict(weights)
    else:
        raise ValueError(f"Unknown architecture: {metadata['architecture']}")
    return model, metadata


def load_documents(path: str | None, dimension: int, count: int) -> np.ndarray:
    """Load float32 document vectors or create a deterministic dummy collection."""
    if path:
        vectors = np.load(path)
        if vectors.ndim != 2 or vectors.shape[1] != dimension:
            raise ValueError(f"Document vectors must have shape [N, {dimension}].")
        return np.ascontiguousarray(vectors.astype(np.float32, copy=False))
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((count, dimension), dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-12)
    return vectors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--query-instruction")
    parser.add_argument("--document-embeddings", help="Float32 .npy matrix [N, D].")
    parser.add_argument("--k", type=float, default=0.3, help="Top-k ratio (<=1) or count (>1).")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--dummy-documents", type=int, default=1000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_n < 1:
        raise ValueError("top-n must be positive.")
    device = torch.device(args.device)
    model, metadata = load_predictor(args.checkpoint_dir, device)
    config = HyperMaskConfig(**metadata["model_config"])
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    encoded = tokenizer(
        [format_query(args.query, args.query_instruction)],
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        query_embedding, importance = model.predict_embedding_and_importance(**encoded)  # type: ignore[attr-defined]
        value: int | float = int(args.k) if args.k > 1 else args.k
        masked_query = select_top_k(importance, query_embedding, value)
    documents = load_documents(args.document_embeddings, config.hidden_dim, args.dummy_documents)
    index = faiss.IndexFlatIP(config.hidden_dim)
    index.add(documents)
    scores, indices = index.search(
        masked_query.detach().float().cpu().numpy(), min(args.top_n, len(documents))
    )
    print(
        json.dumps(
            {
                "query": args.query,
                "selected_dimensions": int((masked_query != 0).sum().item()),
                "results": [
                    {"index": int(index_), "score": float(score)}
                    for index_, score in zip(indices[0], scores[0], strict=True)
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
